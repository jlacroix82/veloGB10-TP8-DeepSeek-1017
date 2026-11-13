//! PRIVATE additive C1 chunk-attention composition; no old engine changes.
//! Context<=8192, contiguous native chunks1..128 including positive starts.
//! This is a checked correctness control, not asynchronous/optimized prefill,
//! learned-stage equivalence, quality or serving qualification.
use super::{
    cache_quant::CacheFormat,
    chunk_plan::{self, ChunkTicket, LayerChunk, INDEX_SOURCES, KV_SOURCES},
    collectives::{RoutedLayout, RowKind},
    config::Dsv41Config,
    gpu_attention::AttentionKernels,
    gpu_cache_quant::GpuCacheQuant,
    gpu_chunk_compressor::{ChunkCompressionKernels, ChunkLatentReceipt, GpuChunkCompressor},
    gpu_chunk_store::{GpuChunkStore, OwnedChunkSnapshot},
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseShape, DequantCast, GpuDense},
    gpu_gemm::{GemmShape, GpuGemm},
    gpu_index_block::{
        GpuIndexBlock, GpuIndexKeyStore, GpuIndexOutput, IndexBlockKernels, IndexOutputIdentity,
        PublishedCandidates,
    },
    gpu_numerics::GpuNumerics,
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_weights::{LoadScope, LoadedWeights},
    position::{FrequencyBackend, RopeProfile},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
const HIDDEN: usize = 5120;
const Q_RANK: usize = 1280;
const HEADS: usize = 8;
const DIM: usize = 512;
const O_RANK: usize = 1024;
pub const STEP_ARRAY_BOUND_BYTES: usize = 128 * 1024 * 1024;
pub const LOCAL_ARRAY_BOUND_BYTES: usize = 32 * 1024 * 1024;
/// All explicit local buffers retained by execute_inner, with conservative
/// overlap of each producer and child outputs. CUDA contexts/loadedweights and
/// caller-retained completed history are outside this per-operation policy.
pub fn local_array_bytes(tokens: usize, new_rows: usize) -> Result<usize> {
    ensure!(
        (1..=128).contains(&tokens) && new_rows <= tokens,
        "local attention allocation domain"
    );
    let quant = |width| tokens * width + tokens * (width / 32) + 4;
    let bf16 = |width| tokens * width * 2 + 4;
    let plain = |width| tokens * width * 2;
    Ok(
        quant(HIDDEN)+bf16(Q_RANK)+bf16(Q_RANK)+quant(Q_RANK)+bf16(HEADS*DIM)+plain(HEADS*DIM)
        +bf16(DIM)+bf16(DIM)+plain(DIM)+quant(DIM)+bf16(DIM)+plain(HEADS*DIM)+quant(HEADS*DIM)
        +bf16(HEADS*DIM)+bf16(O_RANK)+quant(O_RANK)+bf16(HIDDEN)+tokens*HIDDEN*4+plain(HIDDEN)+4
        +(Q_RANK+DIM)*4 // persistent norm promotion arrays counted conservatively
        +new_rows*DIM*2 // normalized source output (also bounded by child)
        +new_rows*DIM*2 // distinct outer rotary copy
        +new_rows*DIM/2+new_rows*(DIM/16)+new_rows*DIM*2+4,
    )
}
struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: Option<T>,
}
impl<T> Queued<T> {
    fn new(dev: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: dev.clone(),
            value: Some(value),
        }
    }
    fn into_checked(mut self) -> Result<T> {
        self.dev.synchronize()?;
        Ok(self.value.take().unwrap())
    }
}
impl<T> std::ops::Deref for Queued<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().unwrap()
    }
}
impl<T> std::ops::DerefMut for Queued<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}
impl<T> Drop for Queued<T> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}

pub struct ChunkAttentionKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    gemm: GpuGemm,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
    attention: AttentionKernels,
    cache: GpuCacheQuant,
    index: Arc<IndexBlockKernels>,
    compression: Arc<ChunkCompressionKernels>,
}
impl ChunkAttentionKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(
            !stream.stream.is_null(),
            "chunk attention needs owned blocking stream"
        );
        Ok(Arc::new(Self {
            dev: dev.clone(),
            gemm: GpuGemm::load(dev, &stream)?,
            dense: GpuDense::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            attention: AttentionKernels::load(dev, &stream)?,
            cache: GpuCacheQuant::load(dev, &stream)?,
            index: IndexBlockKernels::load(dev, stream.clone(), boot)?,
            compression: ChunkCompressionKernels::load(dev, stream.clone(), boot)?,
            stream,
            boot,
        }))
    }
    fn input<T>(&self, v: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&v.device(), &self.dev),
            "chunk attention context differs"
        );
        Ok(())
    }
    fn copy<T: DeviceRepr>(&self, src: &CudaSlice<T>, dst: &mut CudaSlice<T>) -> Result<()> {
        self.input(src)?;
        self.input(dst)?;
        ensure!(
            !src.is_empty() && src.len() == dst.len(),
            "chunk attention DtoD extent"
        );
        self.dev.bind_to_thread()?;
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *dst.device_ptr(),
                *src.device_ptr(),
                src.len() * std::mem::size_of::<T>(),
                self.stream.stream,
            )
        };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "chunk attention DtoD failed {rc:?}"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ChunkAttentionOperationIds {
    pub index_bf16: Option<u64>,
    pub wo_b_f32: u64,
}
/// Matrix rows are temporal prefill positions even for a positive-start chunk.
/// They must never be mislabeled as independent decode lanes whenT>1.
pub fn row_kind(start: usize, tokens: usize) -> Result<RowKind> {
    chunk_plan::window_chunk(start, tokens)?;
    Ok(if start == 0 || tokens > 1 {
        RowKind::ContiguousPrefillRows
    } else {
        RowKind::IndependentDecodeLanes
    })
}
fn rope_profile(ratio: usize) -> RopeProfile {
    if ratio == 0 {
        RopeProfile::Plain
    } else {
        RopeProfile::Yarn
    }
}

/// Shared across all40 layer objects for one C1 session. Index results/candidates
/// are immutable source publications, never inferred from a consumer's layer ID.
pub struct ChunkIndexExchange {
    kernels: Arc<ChunkAttentionKernels>,
    rank: usize,
    manifest: String,
    completion: [u8; 32],
    key_store: GpuIndexKeyStore,
    lane: Option<LaneKey>,
    ticket: Option<ChunkTicket>,
    latest: Option<Arc<GpuIndexOutput>>,
    candidates: Option<Arc<PublishedCandidates>>,
    backend: Option<FrequencyBackend>,
    last_epoch: u64,
    failed: bool,
}
impl ChunkIndexExchange {
    pub fn new(
        k: Arc<ChunkAttentionKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        comm: &GpuCollectives,
    ) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            comm.rank() == m.rank && comm.compute_stream().stream == k.stream.stream,
            "chunk exchange rank/stream"
        );
        let key_store = GpuIndexKeyStore::new(k.index.clone(), c, m)?;
        Ok(Self {
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            completion: comm.completion_identity(),
            last_epoch: comm.expected_epoch(),
            kernels: k,
            key_store,
            lane: None,
            ticket: None,
            latest: None,
            candidates: None,
            backend: None,
            failed: false,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.kernels.boot
                && key.slot == 0
                && key.generation > 0
                && self.lane.is_none_or(|old| key.generation > old.generation),
            "chunk exchange reset requires newer C1 generation"
        );
        self.key_store.reset_lane(key)?;
        self.lane = Some(key);
        self.ticket = None;
        self.latest = None;
        self.candidates = None;
        self.backend = None;
        self.failed = false;
        Ok(())
    }
    fn require(
        &mut self,
        p: &LayerChunk,
        backend: FrequencyBackend,
        comm: &GpuCollectives,
    ) -> Result<()> {
        ensure!(
            !self.failed
                && self.lane == Some(p.ticket.lane)
                && comm.rank() == self.rank
                && comm.completion_identity() == self.completion
                && comm.compute_stream().stream == self.kernels.stream.stream
                && comm.expected_epoch() >= self.last_epoch
                && self.backend.is_none_or(|old| old == backend),
            "chunk exchange generation/backend/owner/epoch differs"
        );
        match &self.ticket {
            Some(old) if *old == p.ticket => (),
            Some(old) => {
                ensure!(
                    p.layer == 0 && p.ticket.version > old.version && p.ticket.start == old.end,
                    "new chunk index publication starts at layer0/contiguous newer ticket only"
                );
                self.ticket = Some(p.ticket.clone());
                self.latest = None;
                self.candidates = None;
            }
            None => {
                ensure!(
                    p.layer == 0 && p.ticket.start == 0,
                    "first chunk exchange must start layer0/prefix0"
                );
                self.ticket = Some(p.ticket.clone());
            }
        }
        self.backend = Some(backend);
        Ok(())
    }
    fn publish(&mut self, p: &LayerChunk, value: GpuIndexOutput) -> Result<Arc<GpuIndexOutput>> {
        value.indices_for_plan(p, self.rank, &self.manifest, self.completion)?;
        let receipt = value.receipt()?;
        ensure!(
            p.publishes_index_selection && receipt.index_source == p.layer,
            "only exact index source publishes selection"
        );
        if p.layer == 20 {
            let candidate = value
                .candidates
                .as_ref()
                .context("ratio1 source20 must publish candidate receipt")?;
            let cr = candidate.receipt()?;
            ensure!(
                cr.index_source == 20 && cr.ticket == p.ticket && cr.all_checks_passed,
                "candidate source/ticket differs"
            );
            self.candidates = Some(candidate.clone());
        }
        let value = Arc::new(value);
        self.latest = Some(value.clone());
        Ok(value)
    }
    fn selection(&self, p: &LayerChunk) -> Result<Arc<GpuIndexOutput>> {
        ensure!(
            !self.failed && self.ticket.as_ref() == Some(&p.ticket),
            "index exchange has stale/failed chunk"
        );
        let output = self
            .latest
            .as_ref()
            .context("index source did not publish this chunk")?;
        output.indices_for_plan(p, self.rank, &self.manifest, self.completion)?;
        Ok(output.clone())
    }
    fn poison(&mut self, key: LaneKey) {
        self.failed = true;
        let _ = self.key_store.invalidate_lane(key);
    }
}
#[derive(Default)]
struct Validity {
    lane: Option<LaneKey>,
    version: Option<u64>,
    failed: bool,
}
#[derive(Clone)]
struct OutputToken {
    registry: Arc<Mutex<Validity>>,
    ticket: ChunkTicket,
}
impl OutputToken {
    fn check(&self) -> Result<()> {
        let v = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk attention registry poisoned"))?;
        ensure!(
            !v.failed && v.lane == Some(self.ticket.lane) && v.version == Some(self.ticket.version),
            "chunk attention output stale/failed generation/version"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct ChunkAttentionReceipt {
    pub rank: usize,
    pub manifest_sha256: String,
    pub layer: usize,
    pub ticket: ChunkTicket,
    pub ratio: usize,
    pub frequency_backend: FrequencyBackend,
    pub frequency_table_sha256: String,
    pub operation_ids: ChunkAttentionOperationIds,
    pub epoch_before: u64,
    pub epoch_after_index: u64,
    pub epoch_after_attention: u64,
    pub completion_owner: [u8; 32],
    pub row_kind: RowKind,
    pub index_receipt: Option<IndexOutputIdentity>,
    pub compressor_receipt: Option<ChunkLatentReceipt>,
    pub window_rows: usize,
    pub compressed_rows: usize,
    pub combined_rows: usize,
    pub topk: usize,
    pub source_decode_window_order: bool,
    pub WO_B_local_dtype: &'static str,
    pub WO_B_network_dtype: &'static str,
    pub all_checks_passed: bool,
    pub source_order_preserved: bool,
    pub upstream_x_norm_checked_by_caller_required: bool,
    pub old_T1_padding_arithmetic_equivalence_qualified: bool,
    pub full_vs_chunk_GPU_arithmetic_equivalence_qualified: bool,
    pub learned_model_or_quality_qualified: bool,
    pub optimized_prefill: bool,
    pub local_explicit_array_bound_bytes: usize,
    pub combined_component_array_bound_bytes: usize,
}
pub struct CheckedChunkAttention {
    kernels: Arc<ChunkAttentionKernels>,
    values: CudaSlice<bf16>,
    snapshot: Arc<OwnedChunkSnapshot>,
    receipt: ChunkAttentionReceipt,
    token: OutputToken,
}
impl CheckedChunkAttention {
    pub fn check(&self) -> Result<()> {
        self.token.check()?;
        self.snapshot.check()
    }
    pub fn receipt(&self) -> Result<&ChunkAttentionReceipt> {
        self.check()?;
        Ok(&self.receipt)
    }
    pub fn device_values(&self) -> Result<&CudaSlice<bf16>> {
        self.check()?;
        Ok(&self.values)
    }
    pub fn snapshot(&self) -> Result<&Arc<OwnedChunkSnapshot>> {
        self.check()?;
        Ok(&self.snapshot)
    }
    /// Explicit untimed diagnostic only; production composition uses device data.
    pub fn download_checked(&self) -> Result<Vec<bf16>> {
        self.check()?;
        Ok(self.kernels.dev.dtoh_sync_copy(&self.values)?)
    }
}

pub struct GpuAttentionChunk {
    kernels: Arc<ChunkAttentionKernels>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    manifest: String,
    layer: usize,
    q_norm_f32: CudaSlice<f32>,
    kv_norm_f32: CudaSlice<f32>,
    compressor: Option<GpuChunkCompressor>,
    index: Option<GpuIndexBlock>,
    registry: Arc<Mutex<Validity>>,
    prefix: usize,
    last_version: u64,
    completion: Option<[u8; 32]>,
    backend: Option<FrequencyBackend>,
}
impl GpuAttentionChunk {
    pub fn new(
        k: Arc<ChunkAttentionKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        layer: usize,
        w: Arc<LoadedWeights>,
    ) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            layer < 40
                && w.scope() == &LoadScope::Layer(layer)
                && w.manifest_sha256() == m.manifest_sha256,
            "actual chunk-attention layer scope/manifest"
        );
        let p = format!("layers.{layer}.attn");
        for (name, outputs, inner) in [
            ("wq_a", Q_RANK, HIDDEN),
            ("wq_b", HEADS * DIM, Q_RANK),
            ("wkv", DIM, HIDDEN),
            ("wo_b", HIDDEN, O_RANK),
        ] {
            ensure!(
                w.native_u8(&format!("{p}.{name}.weight"))?.len() == outputs * inner
                    && w.native_u8(&format!("{p}.{name}.scale"))?.len()
                        == outputs / 32 * (inner / 32),
                "native chunk attention FP8 matrix shape"
            );
        }
        ensure!(
            w.bf16(&format!("{p}.wo_a.weight"))?.len() == O_RANK * HEADS * DIM
                && w.f32(&format!("{p}.attn_sink"))?.len() == HEADS,
            "native WO-A/sink shape"
        );
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        let mut qnorm = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(Q_RANK)?);
        let mut kvnorm = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(DIM)?);
        k.position.cast_bf16_f32(
            &k.stream,
            w.bf16(&format!("{p}.q_norm.weight"))?,
            &mut qnorm,
            &mut status,
        )?;
        k.position.cast_bf16_f32(
            &k.stream,
            w.bf16(&format!("{p}.kv_norm.weight"))?,
            &mut kvnorm,
            &mut status,
        )?;
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "chunk-attention norm initialization"
        );
        let compressor = if KV_SOURCES.contains(&layer) {
            Some(GpuChunkCompressor::new(
                k.compression.clone(),
                c,
                m,
                layer,
                w.clone(),
            )?)
        } else {
            None
        };
        let index = if INDEX_SOURCES.contains(&layer) {
            Some(GpuIndexBlock::new(k.index.clone(), c, m, layer, w.clone())?)
        } else {
            None
        };
        Ok(Self {
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            layer,
            q_norm_f32: qnorm.into_checked()?,
            kv_norm_f32: kvnorm.into_checked()?,
            kernels: k,
            weights: w,
            compressor,
            index,
            registry: Arc::new(Mutex::new(Validity::default())),
            prefix: 0,
            last_version: 0,
            completion: None,
            backend: None,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.kernels.boot && key.slot == 0 && key.generation > 0,
            "chunk-attention C1slot0 generation"
        );
        {
            let v = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk attention registry poisoned"))?;
            ensure!(
                v.lane.is_none_or(|old| key.generation > old.generation),
                "chunk-attention generation must advance"
            );
        }
        self.kernels.dev.synchronize()?;
        if let Some(c) = &mut self.compressor {
            c.reset_lane(key)?;
        }
        if let Some(i) = &mut self.index {
            i.reset_lane(key)?;
        }
        *self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk attention registry poisoned"))? = Validity {
            lane: Some(key),
            version: None,
            failed: false,
        };
        self.prefix = 0;
        self.completion = None;
        self.backend = None;
        Ok(())
    }
    fn poison(&mut self, key: LaneKey) {
        if let Ok(mut v) = self.registry.lock() {
            v.failed = true;
        }
        if let Some(c) = &mut self.compressor {
            let _ = c.invalidate_lane(key);
        }
        if let Some(i) = &mut self.index {
            let _ = i.invalidate_lane(key);
        }
    }
    /// Caller supplies checked HC-normalized x. This method handles no HC, FFN,
    /// head or token sampling, and does not commit the store's layer transaction.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_checked(
        &mut self,
        ticket: &ChunkTicket,
        input: &CudaSlice<bf16>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        ids: ChunkAttentionOperationIds,
        exchange: &mut ChunkIndexExchange,
        store: &mut GpuChunkStore,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedChunkAttention> {
        let result = self.execute_inner(
            ticket,
            input,
            query_freq,
            compressed_freq,
            ids,
            exchange,
            store,
            comm,
        );
        if result.is_err() {
            // Proposed owner_r2 API: terminate this shared session BEFORE our
            // explicit cleanup fences. Inner qualified primitive Drop fences
            // still follow their documented synchronous-control contracts.
            comm.abort_session();
            self.poison(ticket.lane);
            exchange.poison(ticket.lane);
            let _ = store.abort(ticket);
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        ticket: &ChunkTicket,
        input: &CudaSlice<bf16>,
        qfreq: &GpuFrequencyRows,
        kfreq: Option<&GpuFrequencyRows>,
        ids: ChunkAttentionOperationIds,
        exchange: &mut ChunkIndexExchange,
        store: &mut GpuChunkStore,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedChunkAttention> {
        let owner = comm.completion_identity();
        // Required even for uncompressed layers0/1, which carry no IndexOutput
        // receipt that could otherwise expose a substituted cache session.
        store.check_owner(self.rank, &self.manifest, owner, &self.kernels.stream)?;
        let plan = store.layer(ticket, self.layer)?;
        let t = ticket.tokens;
        let profile = rope_profile(plan.ratio);
        let backend = qfreq.backend();
        let positions = (ticket.start..ticket.end)
            .map(|p| p as u32)
            .collect::<Vec<_>>();
        ensure!(
            ticket.start == self.prefix
                && ticket.version > self.last_version
                && input.len() == t * HIDDEN
                && self.completion.is_none_or(|old| old == owner)
                && self.backend.is_none_or(|old| old == backend)
                && comm.rank() == self.rank
                && comm.compute_stream().stream == self.kernels.stream.stream,
            "chunk-attention input/prefix/session/backend differs"
        );
        ensure!(
            qfreq.complete_table_verified()
                && qfreq.profile() == profile
                && qfreq.positions() == positions
                && qfreq.table_sha() == backend.table_sha(profile),
            "chunk-attention exact query frequency rows"
        );
        ensure!(
            ids.index_bf16.is_some() == plan.index_scoring_required
                && ids.index_bf16.is_none_or(|id| id != ids.wo_b_f32),
            "explicit index/attention operation posture differs"
        );
        ensure!(
            Arc::ptr_eq(&exchange.kernels, &self.kernels)
                && exchange.rank == self.rank
                && exchange.manifest == self.manifest,
            "chunk-attention exchange owner differs"
        );
        let comp = plan.compressed.as_ref();
        let new_rows = comp.map_or(0, |v| v.new_rows);
        if plan.owns_kv_and_index_keys && new_rows > 0 {
            let f =
                kfreq.context("completed source groups require exact compression frequencies")?;
            ensure!(
                f.complete_table_verified()
                    && f.profile() == profile
                    && f.backend() == backend
                    && f.table_sha() == backend.table_sha(profile)
                    && f.positions()
                        == comp
                            .unwrap()
                            .groups
                            .iter()
                            .map(|g| g.rope_position as u32)
                            .collect::<Vec<_>>(),
                "chunk-attention compressed group frequency rows differ"
            );
        } else {
            ensure!(kfreq.is_none(), "unused compressed frequency rows");
        }
        self.kernels.input(input)?;
        // Conservative explicit temporary accounting excludes loaded weights and
        // persistent stores, includes the three child component maxima plus
        // retained local Q/output stages. It is a bound, not a measured peak.
        let local_bytes = local_array_bytes(t, new_rows)?;
        ensure!(
            local_bytes <= LOCAL_ARRAY_BOUND_BYTES
                && LOCAL_ARRAY_BOUND_BYTES
                    + super::gpu_index_block::STEP_ARRAY_BOUND_BYTES
                    + super::gpu_chunk_compressor::STEP_ARRAY_BOUND_BYTES
                    + super::gpu_chunk_store::SNAPSHOT_ARRAY_BOUND
                    <= STEP_ARRAY_BOUND_BYTES,
            "declared chunk temporary policy"
        );
        exchange.require(&plan, backend, comm)?;
        {
            let mut v = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk attention registry poisoned"))?;
            ensure!(
                !v.failed && v.lane == Some(ticket.lane),
                "chunk-attention failed/unbound generation"
            );
            v.version = Some(ticket.version);
        }
        self.last_version = ticket.version;
        self.completion = Some(owner);
        self.backend = Some(backend);
        let token = OutputToken {
            registry: self.registry.clone(),
            ticket: ticket.clone(),
        };
        let epoch_before = comm.expected_epoch();
        let k = self.kernels.clone();
        let s = &k.stream;
        let p = format!("layers.{}.attn", self.layer);
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        let xq = Queued::new(
            &k.dev,
            k.numerics.quantize_activations_bf16(s, input, t, HIDDEN)?,
        );
        let qa = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                xq.device_values_unchecked(),
                xq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wq_a.weight"))?,
                self.weights.native_u8(&format!("{p}.wq_a.scale"))?,
                GemmShape {
                    rows: t,
                    outputs: Q_RANK,
                    inner: HIDDEN,
                },
            )?
        });
        let qr = Queued::new(&k.dev, unsafe {
            k.numerics.rms_norm_1017(
                s,
                qa.device_values_unchecked(),
                &self.q_norm_f32,
                t,
                Q_RANK,
            )?
        });
        let qrq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, qr.device_values_unchecked(), t, Q_RANK)?
        });
        let qb = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                qrq.device_values_unchecked(),
                qrq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wq_b.weight"))?,
                self.weights.native_u8(&format!("{p}.wq_b.scale"))?,
                GemmShape {
                    rows: t,
                    outputs: HEADS * DIM,
                    inner: Q_RANK,
                },
            )?
        });
        let mut q = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * HEADS * DIM)?);
        unsafe {
            k.copy(qb.device_values_unchecked(), &mut q)?;
        }
        k.position.rotate(
            s,
            &mut q,
            qfreq,
            profile,
            &positions,
            &mut status,
            1,
            t,
            HEADS,
            DIM,
            false,
        )?;
        let wk = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                xq.device_values_unchecked(),
                xq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wkv.weight"))?,
                self.weights.native_u8(&format!("{p}.wkv.scale"))?,
                GemmShape {
                    rows: t,
                    outputs: DIM,
                    inner: HIDDEN,
                },
            )?
        });
        let knorm = Queued::new(&k.dev, unsafe {
            k.numerics
                .rms_norm_1017(s, wk.device_values_unchecked(), &self.kv_norm_f32, t, DIM)?
        });
        let mut window = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * DIM)?);
        unsafe {
            k.copy(knorm.device_values_unchecked(), &mut window)?;
        }
        k.position.rotate(
            s,
            &mut window,
            qfreq,
            profile,
            &positions,
            &mut status,
            1,
            t,
            1,
            DIM,
            false,
        )?;
        let wq = Queued::new(
            &k.dev,
            k.numerics.quantize_activations_bf16(s, &window, t, DIM)?,
        );
        let wqat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                wq.device_values_unchecked(),
                wq.device_scales_unchecked(),
                t,
                DIM,
                DequantCast::OfficialBf16Round,
            )?
        });
        xq.check()?;
        qa.check()?;
        qr.check()?;
        qrq.check()?;
        qb.check()?;
        wk.check()?;
        knorm.check()?;
        wq.check()?;
        wqat.check()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "query/window RoPE status"
        );
        let latent = match &mut self.compressor {
            Some(c) => Some(c.execute_checked(&plan, input)?),
            None => None,
        };
        let compressor_receipt = latent.as_ref().map(|l| l.receipt().cloned()).transpose()?;
        let selection = if plan.ratio == 0 {
            None
        } else if let Some(index) = &mut self.index {
            let candidate = exchange.candidates.clone();
            let unrotated = latent
                .as_ref()
                .map(|l| l.values_for_plan(&plan, self.rank, &self.manifest))
                .transpose()?
                .flatten();
            // This call completes WK and preserves its unrotated source BEFORE
            // outer KV rotation below. No caller can obtain a mutable source view.
            let output = index.execute_checked(
                &plan,
                input,
                unsafe { qr.device_values_unchecked() },
                unrotated,
                qfreq,
                kfreq,
                &mut exchange.key_store,
                candidate.as_deref(),
                ids.index_bf16,
                comm,
            )?;
            Some(exchange.publish(&plan, output)?)
        } else {
            Some(exchange.selection(&plan)?)
        };
        let epoch_after_index = comm.expected_epoch();
        let index_receipt = selection
            .as_ref()
            .map(|v| v.receipt().cloned())
            .transpose()?;
        let mut compressed_rotated = None;
        let mut compressed_qat = None;
        if let Some(l) = &latent {
            if let Some(v) = l.rotary_copy_for_plan(&plan, self.rank, &self.manifest)? {
                let mut rotated = Queued::new(&k.dev, v);
                let f = kfreq.context("completed outer KV group frequency")?;
                let group_positions = comp
                    .unwrap()
                    .groups
                    .iter()
                    .map(|g| g.rope_position as u32)
                    .collect::<Vec<_>>();
                k.position.rotate(
                    s,
                    &mut rotated,
                    f,
                    profile,
                    &group_positions,
                    &mut status,
                    1,
                    new_rows,
                    1,
                    DIM,
                    false,
                )?;
                let qat = Queued::new(
                    &k.dev,
                    k.cache.quantize(
                        s,
                        &rotated,
                        new_rows,
                        DIM,
                        CacheFormat::CompressedKvE4m3Group16,
                    )?,
                );
                qat.native().check()?;
                compressed_rotated = Some(rotated);
                compressed_qat = Some(qat);
            }
        }
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "compressed KV RoPE status"
        );
        let snapshot = Arc::new(
            store.prepare_checked(
                ticket,
                self.layer,
                unsafe { wqat.device_values_unchecked() },
                compressed_qat
                    .as_ref()
                    .map(|v| unsafe { v.device_qat_unchecked() }),
                selection.clone(),
            )?,
        );
        ensure!(
            snapshot.plan() == &plan
                && snapshot.kv_rows()
                    == plan.window.snapshot_rows + comp.map_or(0, |v| v.stored_rows)
                && snapshot.topk() == plan.attention_index_capacity,
            "chunk snapshot plan/geometry differs"
        );
        let mut o = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * HEADS * DIM)?);
        unsafe {
            k.attention.sparse(
                s,
                &q,
                snapshot.device_kv_unchecked(),
                self.weights.f32(&format!("{p}.attn_sink"))?,
                snapshot.device_indices_unchecked(),
                &mut o,
                &mut status,
                1,
                t,
                HEADS,
                snapshot.kv_rows(),
                snapshot.topk(),
            )?;
        }
        k.position.rotate(
            s,
            &mut o,
            qfreq,
            profile,
            &positions,
            &mut status,
            1,
            t,
            HEADS,
            DIM,
            true,
        )?;
        let oq = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, &o, t, HEADS * DIM)?,
        );
        let oqat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                oq.device_values_unchecked(),
                oq.device_scales_unchecked(),
                t,
                HEADS * DIM,
                DequantCast::OfficialBf16Round,
            )?
        });
        let oa = Queued::new(&k.dev, unsafe {
            k.dense.bf16_linear_bf16(
                s,
                oqat.device_values_unchecked(),
                self.weights.bf16(&format!("{p}.wo_a.weight"))?,
                DenseShape {
                    rows: t,
                    outputs: O_RANK,
                    inner: HEADS * DIM,
                },
            )?
        });
        let oaq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, oa.device_values_unchecked(), t, O_RANK)?
        });
        let ob = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                oaq.device_values_unchecked(),
                oaq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wo_b.weight"))?,
                self.weights.native_u8(&format!("{p}.wo_b.scale"))?,
                GemmShape {
                    rows: t,
                    outputs: HIDDEN,
                    inner: O_RANK,
                },
            )?
        });
        let mut reduce = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(t * HIDDEN)?);
        unsafe {
            k.position
                .cast_bf16_f32(s, ob.device_values_unchecked(), &mut reduce, &mut status)?;
        }
        oq.check()?;
        oqat.check()?;
        oa.check()?;
        oaq.check()?;
        ob.check()?;
        snapshot.check()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "sparse/inverseRoPE/output projection status"
        );
        let kind = row_kind(ticket.start, t)?;
        let layout = RoutedLayout::new(t, kind)?;
        comm.enqueue_reduce(ids.wo_b_f32, layout, &mut reduce)?;
        comm.synchronize()?;
        let mut output = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * HIDDEN)?);
        k.position
            .cast_f32_bf16(s, &reduce, &mut output, &mut status)?;
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "source BF16 row-parallel final cast"
        );
        token.check()?;
        snapshot.check()?;
        exchange.last_epoch = comm.expected_epoch();
        self.prefix = ticket.end;
        let receipt = ChunkAttentionReceipt {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            layer: self.layer,
            ticket: ticket.clone(),
            ratio: plan.ratio,
            frequency_backend: backend,
            frequency_table_sha256: qfreq.table_sha().into(),
            operation_ids: ids,
            epoch_before,
            epoch_after_index,
            epoch_after_attention: comm.expected_epoch(),
            completion_owner: owner,
            row_kind: kind,
            index_receipt,
            compressor_receipt,
            window_rows: plan.window.snapshot_rows,
            compressed_rows: comp.map_or(0, |v| v.stored_rows),
            combined_rows: snapshot.kv_rows(),
            topk: snapshot.topk(),
            source_decode_window_order: snapshot.source_decode_window_order(),
            WO_B_local_dtype: "complete BF16 Linear then exact F32 promotion",
            WO_B_network_dtype: "F32 recursive doubling then one BF16 cast",
            all_checks_passed: true,
            source_order_preserved: true,
            upstream_x_norm_checked_by_caller_required: true,
            old_T1_padding_arithmetic_equivalence_qualified: false,
            full_vs_chunk_GPU_arithmetic_equivalence_qualified: false,
            learned_model_or_quality_qualified: false,
            optimized_prefill: false,
            local_explicit_array_bound_bytes: local_bytes,
            combined_component_array_bound_bytes: STEP_ARRAY_BOUND_BYTES,
        };
        let values = output.into_checked()?;
        // Snapshot is intentionally NOT committed here. Full HC/FFN caller must
        // finish its checked layer before advancing ChunkCursor to the next one.
        Ok(CheckedChunkAttention {
            kernels: k,
            values,
            snapshot,
            receipt,
            token,
        })
    }
}
impl Drop for GpuAttentionChunk {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention_state;
    use super::*;
    #[test]
    fn temporal_chunks_keep_prefill_kind_including_initial_one_token() {
        assert_eq!(row_kind(0, 1).unwrap(), RowKind::ContiguousPrefillRows);
        assert_eq!(row_kind(0, 128).unwrap(), RowKind::ContiguousPrefillRows);
        assert_eq!(row_kind(127, 128).unwrap(), RowKind::ContiguousPrefillRows);
        assert_eq!(row_kind(128, 1).unwrap(), RowKind::IndependentDecodeLanes);
        assert!(row_kind(8192, 1).is_err());
        assert!(row_kind(0, 129).is_err());
    }
    #[test]
    fn old_one_token_padding_is_not_claimed_bitwise_equal_to_chronological_chunk_padding() {
        let mut different_padding = 0;
        for p in 1..128 {
            let old = attention_state::window_plan(128, p, 1).unwrap();
            let new = chunk_plan::window_chunk(p, 1).unwrap();
            let old_globals = old.query_indices[0]
                .iter()
                .filter(|&&slot| slot >= 0)
                .map(|&slot| p - (p - slot as usize) % 128)
                .collect::<Vec<_>>();
            let new_globals = new.query_indices[0]
                .iter()
                .filter(|&&slot| slot >= 0)
                .map(|&slot| new.snapshot_global_base + slot as usize)
                .collect::<Vec<_>>();
            assert_eq!(old_globals, new_globals);
            if old.query_indices[0] != new.query_indices[0] {
                different_padding += 1;
            }
        }
        assert_eq!(different_padding, 126);
        // Equality of reachable key order proves semantic indexing only. GPU
        // sparse-reduction padding/shape/whole-prefill equivalence is unrun.
    }
    #[test]
    fn retained_local_array_accounting_covers_maximum_source_chunk_without_fit_claim() {
        let bytes = local_array_bytes(128, 128).unwrap();
        assert!(bytes < LOCAL_ARRAY_BOUND_BYTES);
        assert!(local_array_bytes(129, 1).is_err());
        assert!(local_array_bytes(1, 2).is_err());
        for t in [1, 2, 16, 75, 128] {
            assert!(local_array_bytes(t, t).unwrap() >= local_array_bytes(t, 0).unwrap());
        }
        assert_eq!(
            LOCAL_ARRAY_BOUND_BYTES
                + super::super::gpu_index_block::STEP_ARRAY_BOUND_BYTES
                + super::super::gpu_chunk_compressor::STEP_ARRAY_BOUND_BYTES
                + super::super::gpu_chunk_store::SNAPSHOT_ARRAY_BOUND,
            STEP_ARRAY_BOUND_BYTES
        );
    }
}
