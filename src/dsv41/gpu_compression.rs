//! Isolated projected compression and source-BF16 local index scores.
//! Dense projections, cacheQAT, externalallreduce/candidates/topk remain separate.
use super::{
    compression::{
        self, CompressionCursor, CompressionPlan, IndexNetworkPolicy, IndexShape, Ratio,
    },
    gpu_numerics::{DeviceOutput, GpuNumerics},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use uuid::Uuid;
pub const PTX: &str = include_str!("../ptx/dsv41_compression.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_compression.cu");
const MODULE: &str = "dsv41_compression_v1";
const NAMES: &[&str] = &[
    "dsv41_compression_identity",
    "dsv41_compression_reset",
    "dsv41_compression_copy_bf16",
    "dsv41_compression_prefill2",
    "dsv41_compression_append2",
    "dsv41_index_scores_source",
];
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
struct Validity {
    lanes: BTreeMap<usize, (LaneKey, bool)>,
}
#[derive(Clone)]
struct LaneToken {
    dev: Arc<CudaDevice>,
    registry: Arc<Mutex<Validity>>,
    key: LaneKey,
    status: Arc<CudaSlice<u32>>,
}
impl LaneToken {
    fn valid(&self) -> Result<()> {
        let r = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("compression registry poisoned"))?;
        ensure!(
            r.lanes.get(&self.key.slot) == Some(&(self.key, false)),
            "stale/failed compression output generation"
        );
        Ok(())
    }
    fn poison(&self) {
        if let Ok(mut r) = self.registry.lock() {
            if let Some((key, failed)) = r.lanes.get_mut(&self.key.slot) {
                if *key == self.key {
                    *failed = true;
                }
            }
        }
    }
    fn check(&self) -> Result<()> {
        self.valid()?;
        self.dev.synchronize()?;
        let status = self.dev.dtoh_sync_copy(self.status.as_ref())?[0];
        self.valid()?;
        if status != 0 {
            self.poison();
            anyhow::bail!("compression GPU status{status} (1=input,2=overflow)");
        }
        Ok(())
    }
}
pub struct CompressionKernels {
    dev: Arc<CudaDevice>,
    stream_id: usize,
    boot: Uuid,
    source_id: u64,
}
impl CompressionKernels {
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !s.stream.is_null() && s.stream as usize == self.stream_id,
            "the exact compression owner blockingstream is required"
        );
        let mut flags = 0;
        let rc = unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "valid compression blockingstream required"
        );
        Ok(())
    }
    fn input<T>(&self, x: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &x.device()),
            "compression CUDAdevice/context mismatch"
        );
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream, boot: Uuid) -> Result<Arc<Self>> {
        let source_id = u64::from_str_radix(
            env!("DSV41_COMPRESSION_SOURCE_ID").trim_start_matches("0x"),
            16,
        )?;
        let this = Self {
            dev: dev.clone(),
            stream_id: stream.stream as usize,
            boot,
            source_id,
        };
        this.stream(stream)?;
        if dev.get_func(MODULE, "dsv41_compression_identity").is_none() {
            dev.load_ptx(Ptx::from_src(PTX), MODULE, NAMES)?;
        }
        let mut stamp = dev.alloc_zeros::<u64>(2)?;
        dev.memset_zeros(&mut stamp)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_compression_identity")
                .context("compressionidentity")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (ptr(&stamp),),
                )?;
        }
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&stamp)? == [source_id, 1],
            "compression source/ABI mismatch"
        );
        Ok(Arc::new(this))
    }
    pub fn source_id(&self) -> u64 {
        self.source_id
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    pub fn cuda_source_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(CUDA_SOURCE))
    }
    pub fn index_scores<'a>(
        &self,
        s: &CudaStream,
        q: &'a CudaSlice<bf16>,
        key: &'a CudaSlice<bf16>,
        weights: &'a CudaSlice<bf16>,
        batch: IndexBatchIdentity,
        network: IndexNetworkPolicy,
    ) -> Result<IndexScoresGpu<'a>> {
        self.stream(s)?;
        for input in [q, key, weights] {
            self.input(input)?;
        }
        let shape = batch.shape;
        shape.validate()?;
        ensure!(
            batch.lanes.len() == shape.batch
                && batch.start_positions.len() == shape.batch
                && batch
                    .lanes
                    .iter()
                    .all(|l| l.boot == self.boot && l.generation > 0 && l.slot < 16)
                && batch
                    .lanes
                    .iter()
                    .map(|l| l.slot)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == shape.batch
                && batch.start_positions.iter().all(|&p| p
                    .checked_add(shape.queries)
                    .is_some_and(|n| n <= compression::MAX_CONTEXT)),
            "index lane/position identity invalid"
        );
        let rows = shape.batch * shape.queries;
        let scores = shape.scores();
        ensure!(
            q.len() == rows * 4 * 128
                && key.len() == shape.batch * shape.keys * 128
                && weights.len() == rows * 4,
            "index suppliedBF16 tensor shape mismatch"
        );
        let dot = self.dev.alloc_zeros::<bf16>(scores * 4)?;
        let relu = self.dev.alloc_zeros::<bf16>(scores * 4)?;
        let product = self.dev.alloc_zeros::<bf16>(scores * 4)?;
        let scaled = self.dev.alloc_zeros::<bf16>(rows * 4)?;
        let local = self.dev.alloc_zeros::<bf16>(scores)?;
        let promoted = self.dev.alloc_zeros::<f32>(scores)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_index_scores_source")
                .context("indexscorekernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig {
                        grid_dim: (u32::try_from(scores)?, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        ptr(q),
                        ptr(key),
                        ptr(weights),
                        ptr(&dot),
                        ptr(&relu),
                        ptr(&product),
                        ptr(&scaled),
                        ptr(&local),
                        ptr(&promoted),
                        ptr(&status),
                        shape.queries as u32,
                        shape.keys as u32,
                    ),
                )?;
        }
        Ok(IndexScoresGpu {
            _inputs: (q, key, weights),
            dev: self.dev.clone(),
            batch,
            network,
            dot,
            relu,
            product,
            scaled,
            local,
            promoted,
            status,
        })
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct IndexBatchIdentity {
    pub shape: IndexShape,
    pub lanes: Vec<LaneKey>,
    pub start_positions: Vec<usize>,
}
pub struct IndexScoresGpu<'a> {
    _inputs: (
        &'a CudaSlice<bf16>,
        &'a CudaSlice<bf16>,
        &'a CudaSlice<bf16>,
    ),
    dev: Arc<CudaDevice>,
    pub batch: IndexBatchIdentity,
    pub network: IndexNetworkPolicy,
    dot: CudaSlice<bf16>,
    relu: CudaSlice<bf16>,
    product: CudaSlice<bf16>,
    scaled: CudaSlice<bf16>,
    local: CudaSlice<bf16>,
    promoted: CudaSlice<f32>,
    status: CudaSlice<u32>,
}
impl IndexScoresGpu<'_> {
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(status == 0, "indexscore status{status}");
        Ok(())
    }
    pub fn download_checked(&self) -> Result<compression::IndexScores> {
        self.check()?;
        Ok(compression::IndexScores {
            scaled_weights: self.dev.dtoh_sync_copy(&self.scaled)?,
            dot: self.dev.dtoh_sync_copy(&self.dot)?,
            relu: self.dev.dtoh_sync_copy(&self.relu)?,
            product: self.dev.dtoh_sync_copy(&self.product)?,
            local_bf16: self.dev.dtoh_sync_copy(&self.local)?,
            promoted_f32: self.dev.dtoh_sync_copy(&self.promoted)?,
        })
    }
    /// Result is pending. Caller must check status before publishing/using it.
    pub unsafe fn source_bf16_unchecked(&mut self) -> Result<&mut CudaSlice<bf16>> {
        ensure!(
            self.network == IndexNetworkPolicy::SourceBf16LocalAndNetwork,
            "BF16network policy required"
        );
        Ok(&mut self.local)
    }
    /// Widened sourceBF16local values; this explicitly changes NETWORK rounding.
    /// No local BF16dot/product/sum boundary was removed.
    pub unsafe fn alternative_f32_unchecked(&mut self) -> Result<&mut CudaSlice<f32>> {
        ensure!(
            self.network == IndexNetworkPolicy::SourceBf16LocalPromotedF32NetworkAlternative,
            "F32network alternative policy required"
        );
        Ok(&mut self.promoted)
    }
}
impl Drop for IndexScoresGpu<'_> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
pub struct GpuCompressor {
    kernels: Arc<CompressionKernels>,
    cursor: CompressionCursor,
    ratio: Ratio,
    dim: usize,
    max_lanes: usize,
    kv: CudaSlice<f32>,
    score: CudaSlice<f32>,
    statuses: Vec<Arc<CudaSlice<u32>>>,
    registry: Arc<Mutex<Validity>>,
}
impl GpuCompressor {
    pub fn new(
        kernels: Arc<CompressionKernels>,
        s: &CudaStream,
        ratio: Ratio,
        dim: usize,
        max_lanes: usize,
    ) -> Result<Self> {
        kernels.stream(s)?;
        ensure!(
            (1..=512).contains(&dim) && (1..=16).contains(&max_lanes),
            "compressor shape bound"
        );
        let cursor = CompressionCursor::new(kernels.boot, max_lanes, ratio)?;
        let kv = kernels.dev.alloc_zeros::<f32>(max_lanes * 2 * dim)?;
        let score = kernels.dev.alloc_zeros::<f32>(max_lanes * 2 * dim)?;
        let mut statuses = Vec::new();
        for _ in 0..max_lanes {
            let mut status = kernels.dev.alloc_zeros::<u32>(1)?;
            kernels.dev.memset_zeros(&mut status)?;
            statuses.push(Arc::new(status));
        }
        Ok(Self {
            kernels,
            cursor,
            ratio,
            dim,
            max_lanes,
            kv,
            score,
            statuses,
            registry: Arc::new(Mutex::new(Validity {
                lanes: BTreeMap::new(),
            })),
        })
    }
    fn token(&self, key: LaneKey) -> Result<LaneToken> {
        ensure!(key.slot < self.max_lanes, "compression slot bound");
        let token = LaneToken {
            dev: self.kernels.dev.clone(),
            registry: self.registry.clone(),
            key,
            status: self.statuses[key.slot].clone(),
        };
        token.valid()?;
        Ok(token)
    }
    pub fn reset_lane(&mut self, s: &CudaStream, key: LaneKey) -> Result<()> {
        self.kernels.stream(s)?;
        self.cursor.reset(key)?;
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("compression registry poisoned"))?
            .lanes
            .insert(key.slot, (key, false));
        let result = unsafe {
            self.kernels
                .dev
                .get_func(MODULE, "dsv41_compression_reset")
                .context("compressionreset")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems((2 * self.dim) as u32),
                    (
                        ptr(&self.kv),
                        ptr(&self.score),
                        ptr(self.statuses[key.slot].as_ref()),
                        self.dim as u32,
                        key.slot as u32,
                    ),
                )
        };
        if result.is_err() {
            self.token(key)?.poison();
            self.cursor.poison(key)?;
        }
        result.map_err(Into::into)
    }
    fn output<'a>(
        &self,
        plan: CompressionPlan,
        token: LaneToken,
        inputs: ProjectionInputs<'a>,
    ) -> Result<CompressionOutputGpu<'a>> {
        let count = (plan.output_rows * self.dim).max(1);
        let values = self.kernels.dev.alloc_zeros::<bf16>(count)?;
        let pool = if self.ratio == Ratio::Two {
            Some(self.kernels.dev.alloc_zeros::<f32>(count)?)
        } else {
            None
        };
        Ok(CompressionOutputGpu {
            _inputs: inputs,
            kernels: self.kernels.clone(),
            plan,
            dim: self.dim,
            values,
            pool,
            token,
        })
    }
    pub fn push_ratio1<'a>(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        tokens: usize,
        projected: &'a CudaSlice<bf16>,
    ) -> Result<CompressionOutputGpu<'a>> {
        self.kernels.stream(s)?;
        self.kernels.input(projected)?;
        let token = self.token(key)?;
        let plan = self.cursor.plan(key, start, tokens)?;
        ensure!(
            self.ratio == Ratio::One && projected.len() == tokens * self.dim,
            "ratio1 suppliedBF16projection shape"
        );
        let out = self.output(
            plan.clone(),
            token.clone(),
            ProjectionInputs::RatioOne(projected),
        )?;
        let result = unsafe {
            self.kernels
                .dev
                .get_func(MODULE, "dsv41_compression_copy_bf16")
                .context("ratio1copy")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems(projected.len() as u32),
                    (
                        ptr(projected),
                        ptr(&out.values),
                        ptr(token.status.as_ref()),
                        projected.len() as u64,
                    ),
                )
        };
        if let Err(e) = result {
            token.poison();
            self.cursor.poison(key)?;
            return Err(e.into());
        }
        self.cursor.commit(&plan)?;
        Ok(out)
    }
    pub fn push_ratio2<'a>(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        tokens: usize,
        kv: &'a CudaSlice<f32>,
        score: &'a CudaSlice<f32>,
    ) -> Result<CompressionOutputGpu<'a>> {
        self.kernels.stream(s)?;
        self.kernels.input(kv)?;
        self.kernels.input(score)?;
        let token = self.token(key)?;
        let plan = self.cursor.plan(key, start, tokens)?;
        ensure!(
            self.ratio == Ratio::Two && kv.len() == tokens * self.dim && score.len() == kv.len(),
            "ratio2 suppliedF32projection shape"
        );
        let out = self.output(
            plan.clone(),
            token.clone(),
            ProjectionInputs::RatioTwo(kv, score),
        )?;
        let pool = out.pool.as_ref().unwrap();
        let result = if start == 0 {
            unsafe {
                self.kernels
                    .dev
                    .get_func(MODULE, "dsv41_compression_prefill2")
                    .context("prefillpool")?
                    .launch_on_stream(
                        s,
                        LaunchConfig::for_num_elems(
                            (plan.output_rows * self.dim).max(self.dim) as u32
                        ),
                        (
                            ptr(kv),
                            ptr(score),
                            ptr(&self.kv),
                            ptr(&self.score),
                            ptr(pool),
                            ptr(&out.values),
                            ptr(token.status.as_ref()),
                            tokens as u32,
                            self.dim as u32,
                            key.slot as u32,
                        ),
                    )
            }
        } else {
            unsafe {
                self.kernels
                    .dev
                    .get_func(MODULE, "dsv41_compression_append2")
                    .context("appendpool")?
                    .launch_on_stream(
                        s,
                        LaunchConfig::for_num_elems(self.dim as u32),
                        (
                            ptr(kv),
                            ptr(score),
                            ptr(&self.kv),
                            ptr(&self.score),
                            ptr(pool),
                            ptr(&out.values),
                            ptr(token.status.as_ref()),
                            self.dim as u32,
                            key.slot as u32,
                            (start % 2) as u32,
                        ),
                    )
            }
        };
        if let Err(e) = result {
            token.poison();
            self.cursor.poison(key)?;
            return Err(e.into());
        }
        self.cursor.commit(&plan)?;
        Ok(out)
    }
    pub fn download_tail_checked(&self, key: LaneKey) -> Result<(Vec<f32>, Vec<f32>)> {
        let token = self.token(key)?;
        token.check()?;
        let kv = self.kernels.dev.dtoh_sync_copy(&self.kv)?;
        let score = self.kernels.dev.dtoh_sync_copy(&self.score)?;
        let a = key.slot * 2 * self.dim;
        Ok((
            kv[a..a + 2 * self.dim].to_vec(),
            score[a..a + 2 * self.dim].to_vec(),
        ))
    }
}
impl Drop for GpuCompressor {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}
enum ProjectionInputs<'a> {
    RatioOne(&'a CudaSlice<bf16>),
    RatioTwo(&'a CudaSlice<f32>, &'a CudaSlice<f32>),
}
pub struct CompressionOutputGpu<'a> {
    _inputs: ProjectionInputs<'a>,
    kernels: Arc<CompressionKernels>,
    pub plan: CompressionPlan,
    dim: usize,
    values: CudaSlice<bf16>,
    pool: Option<CudaSlice<f32>>,
    token: LaneToken,
}
impl<'a> CompressionOutputGpu<'a> {
    pub fn check(&self) -> Result<()> {
        self.token.check()
    }
    pub fn status_code(&self) -> Result<u32> {
        self.kernels.dev.synchronize()?;
        Ok(self
            .kernels
            .dev
            .dtoh_sync_copy(self.token.status.as_ref())?[0])
    }
    pub fn download_checked(&self) -> Result<(Vec<bf16>, Option<Vec<f32>>)> {
        self.check()?;
        let n = self.plan.output_rows * self.dim;
        let b = self.kernels.dev.dtoh_sync_copy(&self.values)?[..n].to_vec();
        let f = match &self.pool {
            Some(p) => Some(self.kernels.dev.dtoh_sync_copy(p)?[..n].to_vec()),
            None => None,
        };
        Ok((b, f))
    }
    /// Same ordered stream; caller checks this output and returned normalization
    /// before visible publication. Empty output has no RMS call, as the source.
    pub fn normalize(
        self,
        s: &CudaStream,
        numerics: &GpuNumerics,
        weight_f32: &'a CudaSlice<f32>,
    ) -> Result<NormalizedCompressionGpu<'a>> {
        self.kernels.stream(s)?;
        self.token.valid()?;
        if self.plan.output_rows == 0 {
            return Ok(NormalizedCompressionGpu::Empty(self));
        }
        let normalized =
            numerics.rms_norm_1017(s, &self.values, weight_f32, self.plan.output_rows, self.dim)?;
        Ok(NormalizedCompressionGpu::Latent(UnrotatedLatentGpu {
            prepared: self,
            normalized,
            _norm_weight: weight_f32,
        }))
    }
}
impl Drop for CompressionOutputGpu<'_> {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}
/// The normalized latent owner is immutable. WK can read it independently of
/// outer compressed-KV RoPE, which receives its own distinct owner.
pub enum NormalizedCompressionGpu<'a> {
    /// Retains the incomplete-tail operation/status until the parent checks it.
    Empty(CompressionOutputGpu<'a>),
    Latent(UnrotatedLatentGpu<'a>),
}
impl<'a> NormalizedCompressionGpu<'a> {
    pub fn check(&self) -> Result<()> {
        match self {
            Self::Empty(v) => v.check(),
            Self::Latent(v) => v.check(),
        }
    }
    pub fn latent(&self) -> Option<&UnrotatedLatentGpu<'a>> {
        match self {
            Self::Latent(v) => Some(v),
            Self::Empty(_) => None,
        }
    }
}
pub struct UnrotatedLatentGpu<'a> {
    prepared: CompressionOutputGpu<'a>,
    normalized: DeviceOutput<bf16>,
    _norm_weight: &'a CudaSlice<f32>,
}
impl UnrotatedLatentGpu<'_> {
    pub fn plan(&self) -> &CompressionPlan {
        &self.prepared.plan
    }
    pub fn check(&self) -> Result<()> {
        self.prepared.check()?;
        self.normalized.check()
    }
    pub fn download_checked(&self) -> Result<Vec<bf16>> {
        self.check()?;
        self.normalized.download_checked()
    }
    /// Caller orders WK and later checks complete statuses before publication.
    pub unsafe fn unrotated_values_unchecked(&self) -> &CudaSlice<bf16> {
        self.normalized.device_values_unchecked()
    }
    /// Queues a distinct BF16 copy; preserves this immutable WK source even if
    /// the returned copy is rotated/quantized. Caller later checks self.status.
    pub unsafe fn enqueue_rotary_copy_unchecked(&self, s: &CudaStream) -> Result<CudaSlice<bf16>> {
        self.prepared.kernels.stream(s)?;
        self.prepared.token.valid()?;
        let count = self.prepared.plan.output_rows * self.prepared.dim;
        let output = self.prepared.kernels.dev.alloc_zeros::<bf16>(count)?;
        self.prepared
            .kernels
            .dev
            .get_func(MODULE, "dsv41_compression_copy_bf16")
            .context("rotarycopy")?
            .launch_on_stream(
                s,
                LaunchConfig::for_num_elems(count as u32),
                (
                    ptr(self.normalized.device_values_unchecked()),
                    ptr(&output),
                    ptr(self.prepared.token.status.as_ref()),
                    count as u64,
                ),
            )?;
        Ok(output)
    }
}
