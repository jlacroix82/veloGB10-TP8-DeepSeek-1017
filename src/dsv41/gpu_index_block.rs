//! PRIVATE DRAFT: bounded C1<=8192 source index composition. No model/serving
//! capability is enabled by this file. Uses native checkpoint matrices, source
//! BF16 local/network stages, and an exact all-selected shortcut only<=512keys.
//! First implementation fences complete primitive producers before publication;
//! it is deliberately not an asynchronous/graph/performance qualification.
use super::{
    bf16_collectives::MatrixBf16Index,
    cache_quant::CacheFormat,
    chunk_plan::{self, ChunkTicket, LayerChunk, KV_SOURCES, MAX_CONTEXT},
    compression::{IndexNetworkPolicy, IndexShape},
    config::Dsv41Config,
    gpu_bf16_collectives::Bf16Workspace,
    gpu_cache_quant::GpuCacheQuant,
    gpu_collectives::GpuCollectives,
    gpu_compression::{CompressionKernels, IndexBatchIdentity},
    gpu_dense::{DenseShape, GpuDense},
    gpu_gemm::{GemmShape, GpuGemm},
    gpu_index_select::IndexSelection,
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
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

pub const STEP_ARRAY_BOUND_BYTES: usize = 64 * 1024 * 1024;
const KEY_DIM: usize = 128;
const LOCAL_HEADS: usize = 4;
/// Every queued temporary fences on an error as well as successful return.
/// This component synchronizes the BF16 collective before subsequent fallible
/// selection work; no pending RDMA producer is freed by these local fences.
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

pub struct IndexBlockKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    gemm: GpuGemm,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
    cache: GpuCacheQuant,
    compression: Arc<CompressionKernels>,
    selection: IndexSelection,
}
impl IndexBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(
            !stream.stream.is_null(),
            "index component requires owned blocking stream"
        );
        Ok(Arc::new(Self {
            dev: dev.clone(),
            gemm: GpuGemm::load(dev, &stream)?,
            dense: GpuDense::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            cache: GpuCacheQuant::load(dev, &stream)?,
            compression: CompressionKernels::load(dev, &stream, boot)?,
            selection: IndexSelection::load(dev, &stream)?,
            stream,
            boot,
        }))
    }
    fn input<T>(&self, v: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&v.device(), &self.dev),
            "index CUDA context differs"
        );
        Ok(())
    }
    fn copy_range<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        src_first: usize,
        dst: &mut CudaSlice<T>,
        dst_first: usize,
        count: usize,
    ) -> Result<()> {
        self.input(src)?;
        self.input(dst)?;
        ensure!(
            count > 0
                && src_first.checked_add(count).is_some_and(|n| n <= src.len())
                && dst_first.checked_add(count).is_some_and(|n| n <= dst.len()),
            "index DtoD range"
        );
        self.dev.bind_to_thread()?;
        let width = std::mem::size_of::<T>();
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *dst.device_ptr() + (dst_first * width) as u64,
                *src.device_ptr() + (src_first * width) as u64,
                count * width,
                self.stream.stream,
            )
        };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "index DtoD failure {rc:?}"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Publication {
    ticket: ChunkTicket,
    owner: usize,
    rows: usize,
}
struct KeyLane {
    lane: LaneKey,
    failed: bool,
    ends: [usize; 4],
    pubs: [Option<Publication>; 4],
    buffers: Vec<CudaSlice<bf16>>,
}
/// C1 only, four owned native BF16 QAT index caches. Initial<=8K allocation is
/// exactly5MiB. This owner must be invalidated/reset after ANY outer model error.
pub struct GpuIndexKeyStore {
    kernels: Arc<IndexBlockKernels>,
    rank: usize,
    manifest: String,
    frequency_backend: Option<FrequencyBackend>,
    lane: Option<KeyLane>,
}
impl GpuIndexKeyStore {
    pub fn new(k: Arc<IndexBlockKernels>, c: &Dsv41Config, m: &RankManifest) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(m.rank < 8, "index rank");
        Ok(Self {
            kernels: k,
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            frequency_backend: None,
            lane: None,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.kernels.boot && key.slot == 0 && key.generation > 0,
            "first index cache is C1 slot0 only"
        );
        ensure!(
            self.lane
                .as_ref()
                .is_none_or(|p| key.generation > p.lane.generation),
            "index reset generation must advance"
        );
        self.kernels.dev.synchronize()?;
        let mut buffers = Queued::new(&self.kernels.dev, Vec::new());
        for owner in KV_SOURCES {
            let ratio = if owner < 20 { 2 } else { 1 };
            buffers.push(
                self.kernels
                    .dev
                    .alloc_zeros::<bf16>(MAX_CONTEXT / ratio * KEY_DIM)?,
            );
        }
        self.kernels.dev.synchronize()?;
        self.lane = Some(KeyLane {
            lane: key,
            failed: false,
            ends: [0; 4],
            pubs: std::array::from_fn(|_| None),
            buffers: buffers.into_checked()?,
        });
        self.frequency_backend = None;
        Ok(())
    }
    pub fn invalidate_lane(&mut self, key: LaneKey) -> Result<()> {
        let lane = self.lane.as_mut().context("index cache lane unbound")?;
        ensure!(lane.lane == key, "index cache invalidation generation");
        lane.failed = true;
        Ok(())
    }
    fn slot(owner: usize) -> Result<usize> {
        KV_SOURCES
            .iter()
            .position(|&p| p == owner)
            .context("invalid index-key owner")
    }
    fn publish(&mut self, plan: &LayerChunk, keys: Option<&CudaSlice<bf16>>) -> Result<()> {
        ensure!(
            plan.owns_kv_and_index_keys,
            "non-owner index-key publication"
        );
        let owner = plan.layer;
        let slot = Self::slot(owner)?;
        let comp = plan.compressed.as_ref().context("compressed owner plan")?;
        let lane = self.lane.as_mut().context("index cache lane unbound")?;
        ensure!(
            lane.lane == plan.ticket.lane && !lane.failed && lane.ends[slot] == plan.ticket.start,
            "index owner stale/noncontiguous/failed prefix"
        );
        ensure!(
            lane.pubs[slot]
                .as_ref()
                .is_none_or(|p| p.ticket.version < plan.ticket.version),
            "duplicate/stale index owner version"
        );
        ensure!(
            if comp.new_rows == 0 {
                keys.is_none()
            } else {
                keys.is_some_and(|v| v.len() == comp.new_rows * KEY_DIM)
            },
            "index owner exact new QAT row extent"
        );
        if let Some(keys) = keys {
            self.kernels.copy_range(
                keys,
                0,
                &mut lane.buffers[slot],
                plan.ticket.start / plan.ratio * KEY_DIM,
                keys.len(),
            )?;
        }
        self.kernels.dev.synchronize()?;
        lane.ends[slot] = plan.ticket.end;
        lane.pubs[slot] = Some(Publication {
            ticket: plan.ticket.clone(),
            owner,
            rows: comp.stored_rows,
        });
        Ok(())
    }
    fn snapshot(&self, plan: &LayerChunk) -> Result<CudaSlice<bf16>> {
        let owner = plan.kv_source.context("index-key source missing")?;
        let slot = Self::slot(owner)?;
        let lane = self.lane.as_ref().context("index cache lane unbound")?;
        let rows = plan
            .compressed
            .as_ref()
            .context("index compressed plan missing")?
            .stored_rows;
        ensure!(
            lane.lane == plan.ticket.lane
                && !lane.failed
                && rows > 0
                && lane.pubs[slot]
                    == Some(Publication {
                        ticket: plan.ticket.clone(),
                        owner,
                        rows
                    }),
            "index key source did not publish this exact lane/version/chunk"
        );
        let mut output = Queued::new(
            &self.kernels.dev,
            self.kernels.dev.alloc_zeros::<bf16>(rows * KEY_DIM)?,
        );
        self.kernels
            .copy_range(&lane.buffers[slot], 0, &mut output, 0, rows * KEY_DIM)?;
        self.kernels.dev.synchronize()?;
        output.into_checked()
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct IndexOutputIdentity {
    pub rank: usize,
    pub manifest_sha256: String,
    pub layer: usize,
    pub index_source: usize,
    pub ticket: ChunkTicket,
    pub rows: usize,
    pub keys: usize,
    pub topk: usize,
    pub offset: usize,
    pub local_dtype: &'static str,
    pub network_dtype: &'static str,
    pub network_operation_id: Option<u64>,
    pub network_executed: bool,
    pub completion_owner: [u8; 32],
    pub complete_epoch: u64,
    pub exact_all_selected_shortcut: bool,
    pub NCCL_bit_equivalence_qualified: bool,
    pub complete_model_or_quality_qualified: bool,
    pub all_checks_passed: bool,
    pub frequency_backend: FrequencyBackend,
    pub yarn_table_sha256: String,
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
            .map_err(|_| anyhow::anyhow!("index validity poisoned"))?;
        ensure!(
            !v.failed && v.lane == Some(self.ticket.lane) && v.version == Some(self.ticket.version),
            "index output stale/failed generation/version"
        );
        Ok(())
    }
}
pub struct PublishedCandidates {
    mask: CudaSlice<u8>,
    identity: IndexOutputIdentity,
    token: OutputToken,
}
impl PublishedCandidates {
    pub fn receipt(&self) -> Result<&IndexOutputIdentity> {
        self.token.check()?;
        Ok(&self.identity)
    }
    fn for_plan(
        &self,
        plan: &LayerChunk,
        rank: usize,
        manifest: &str,
        owner: [u8; 32],
    ) -> Result<&CudaSlice<u8>> {
        self.token.check()?;
        ensure!(
            self.identity.layer == 20
                && self.identity.ticket == plan.ticket
                && self.identity.rank == rank
                && self.identity.manifest_sha256 == manifest
                && self.identity.completion_owner == owner
                && self.identity.keys
                    == plan
                        .compressed
                        .as_ref()
                        .context("candidate consumer compressed plan")?
                        .stored_rows,
            "candidate source identity/owner/geometry differs"
        );
        Ok(&self.mask)
    }
}
pub struct IndexOutput {
    identity: IndexOutputIdentity,
    indices: Option<CudaSlice<i32>>,
    pub candidates: Option<Arc<PublishedCandidates>>,
    token: OutputToken,
}
pub type GpuIndexOutput = IndexOutput;
/// Pure metadata bridge used by the cache assembler as well as GPU owners.
/// Receipt validation proves association, not the numerical data by itself.
pub fn validate_index_receipt(
    p: &LayerChunk,
    v: &IndexOutputIdentity,
    rank: usize,
    manifest: &str,
    owner: [u8; 32],
) -> Result<()> {
    let comp = p
        .compressed
        .as_ref()
        .context("index consumer compressed plan")?;
    ensure!(
        rank < 8
            && Some(v.index_source) == p.index_source
            && v.layer == v.index_source
            && v.ticket == p.ticket
            && v.rank == rank
            && v.manifest_sha256 == manifest
            && v.completion_owner == owner
            && v.rows == p.ticket.tokens
            && v.keys == comp.stored_rows
            && v.topk == comp.stored_rows.min(512)
            && v.offset == 0
            && v.all_checks_passed
            && !v.complete_model_or_quality_qualified
            && !v.NCCL_bit_equivalence_qualified
            && v.exact_all_selected_shortcut == (comp.stored_rows <= 512)
            && v.network_executed == (comp.stored_rows > 512)
            && v.network_operation_id.is_some() == v.network_executed
            && v.local_dtype == "BF16 dot/ReLU/scaled-weight/product/headsum"
            && v.network_dtype == "BF16 RNE each serial XOR round"
            && v.yarn_table_sha256 == v.frequency_backend.table_sha(RopeProfile::Yarn),
        "index selection source/lane/version/session/extent/offset/precision differs"
    );
    Ok(())
}
impl IndexOutput {
    pub fn receipt(&self) -> Result<&IndexOutputIdentity> {
        self.check()?;
        Ok(&self.identity)
    }
    pub fn check(&self) -> Result<()> {
        self.token.check()
    }
    /// Fully checked immutable device indices. Caller retains this owner and
    /// binds ticket/offset when composing a cache snapshot; this is not a model.
    pub fn device_indices(&self) -> Result<Option<&CudaSlice<i32>>> {
        self.check()?;
        Ok(self.indices.as_ref())
    }
    /// The long-context cache assembler is the sole owner of adding its current
    /// window snapshot offset. Non-index layers reuse their exact source result.
    pub fn indices_for_plan(
        &self,
        plan: &LayerChunk,
        rank: usize,
        manifest: &str,
        completion_owner: [u8; 32],
    ) -> Result<Option<&CudaSlice<i32>>> {
        self.check()?;
        validate_index_receipt(plan, &self.identity, rank, manifest, completion_owner)?;
        Ok(self.indices.as_ref())
    }
}
pub struct GpuIndexBlock {
    kernels: Arc<IndexBlockKernels>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    manifest: String,
    layer: usize,
    key_norm_f32: Option<CudaSlice<f32>>,
    workspace: Bf16Workspace,
    validity: Arc<Mutex<Validity>>,
    prefix: usize,
    last_version: u64,
    completion: Option<[u8; 32]>,
    frequency_backend: Option<FrequencyBackend>,
}
impl GpuIndexBlock {
    pub fn new(
        k: Arc<IndexBlockKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        layer: usize,
        w: Arc<LoadedWeights>,
    ) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            c.index_source_layers.contains(&layer)
                && w.scope() == &LoadScope::Layer(layer)
                && w.manifest_sha256() == m.manifest_sha256,
            "actual index weight scope/manifest/layer"
        );
        let p = format!("layers.{layer}.attn.indexer");
        ensure!(
            w.native_u8(&format!("{p}.wq_b.weight"))?.len() == 512 * 1280
                && w.native_u8(&format!("{p}.wq_b.scale"))?.len() == 16 * 40
                && w.bf16(&format!("{p}.weights_proj.weight"))?.len() == 4 * 5120,
            "native local index matrix shape"
        );
        let key_norm_f32 = if KV_SOURCES.contains(&layer) {
            ensure!(
                w.bf16(&format!("{p}.wk.weight"))?.len() == 128 * 512,
                "native indexWK is128x512 latent projection"
            );
            let mut output = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(128)?);
            let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
            k.position.cast_bf16_f32(
                &k.stream,
                w.bf16(&format!("{p}.k_norm.weight"))?,
                &mut output,
                &mut status,
            )?;
            k.dev.synchronize()?;
            ensure!(
                k.dev.dtoh_sync_copy(&status)? == [0],
                "index RMS weight promotion"
            );
            Some(output.into_checked()?)
        } else {
            None
        };
        let workspace = Bf16Workspace::new(&k.dev, &k.stream, 128 * MAX_CONTEXT)?;
        Ok(Self {
            kernels: k,
            weights: w,
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            layer,
            key_norm_f32,
            workspace,
            validity: Arc::new(Mutex::new(Validity::default())),
            prefix: 0,
            last_version: 0,
            completion: None,
            frequency_backend: None,
        })
    }
    pub fn reset_lane(&mut self, lane: LaneKey) -> Result<()> {
        ensure!(
            lane.boot == self.kernels.boot && lane.slot == 0 && lane.generation > 0,
            "first index block C1 lane"
        );
        self.kernels.dev.synchronize()?;
        let mut v = self
            .validity
            .lock()
            .map_err(|_| anyhow::anyhow!("index registry poisoned"))?;
        ensure!(
            v.lane.is_none_or(|old| lane.generation > old.generation),
            "index generation must advance"
        );
        // A failed BF16 workspace is deliberately sticky. Replace it before a
        // new generation becomes visible; never silently clear an error flag.
        self.workspace =
            Bf16Workspace::new(&self.kernels.dev, &self.kernels.stream, 128 * MAX_CONTEXT)?;
        *v = Validity {
            lane: Some(lane),
            version: None,
            failed: false,
        };
        self.prefix = 0;
        self.completion = None;
        self.frequency_backend = None;
        Ok(())
    }
    pub fn invalidate_lane(&mut self, lane: LaneKey) -> Result<()> {
        let mut v = self
            .validity
            .lock()
            .map_err(|_| anyhow::anyhow!("index registry poisoned"))?;
        ensure!(v.lane == Some(lane), "index invalidation lane");
        v.failed = true;
        Ok(())
    }
    /// Supplied latent is the complete checked normalized UNROTATED BF16
    /// compressor output for exactly plan.compressed.groups. Its producer must
    /// remain alive throughout this call. This method does not certify how an
    /// arbitrary supplied tensor was computed; full attention composition is a
    /// separate gate. x/qr upstream statuses must likewise be checked by caller.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_checked(
        &mut self,
        plan: &LayerChunk,
        x: &CudaSlice<bf16>,
        qr: &CudaSlice<bf16>,
        unrotated_latent: Option<&CudaSlice<bf16>>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        store: &mut GpuIndexKeyStore,
        candidate: Option<&PublishedCandidates>,
        operation_id: Option<u64>,
        comm: &mut GpuCollectives,
    ) -> Result<IndexOutput> {
        let result = self.execute_inner(
            plan,
            x,
            qr,
            unrotated_latent,
            query_freq,
            compressed_freq,
            store,
            candidate,
            operation_id,
            comm,
        );
        if result.is_err() {
            comm.abort_session();
            // Crossed/forged attempts also revoke prior publications.
            if let Ok(mut state) = self.validity.lock() {
                state.failed = true;
            }
            if let Some(lane) = store.lane.as_mut() {
                lane.failed = true;
            }
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        p: &LayerChunk,
        x: &CudaSlice<bf16>,
        qr: &CudaSlice<bf16>,
        latent: Option<&CudaSlice<bf16>>,
        qfreq: &GpuFrequencyRows,
        kfreq: Option<&GpuFrequencyRows>,
        store: &mut GpuIndexKeyStore,
        candidate: Option<&PublishedCandidates>,
        id: Option<u64>,
        comm: &mut GpuCollectives,
    ) -> Result<IndexOutput> {
        let comp = p
            .compressed
            .as_ref()
            .context("index requires compressed plan")?;
        ensure!(
            p.layer == self.layer
                && p.publishes_index_selection
                && p.ticket.start == self.prefix
                && p.ticket.version > self.last_version
                && p.ticket.start.checked_add(p.ticket.tokens) == Some(p.ticket.end)
                && comp == &chunk_plan::compressed_chunk(p.ticket.start, p.ticket.tokens, p.ratio)?
                && p.window == chunk_plan::window_chunk(p.ticket.start, p.ticket.tokens)?
                && p.index_scoring_required == (comp.stored_rows > 512)
                && p.owns_kv_and_index_keys == KV_SOURCES.contains(&p.layer)
                && p.publishes_candidates == (p.layer == 20)
                && p.candidate_source == (p.layer > 20).then_some(20)
                && p.attention_index_capacity
                    == p.window.query_capacity + comp.stored_rows.min(512)
                && p.full_candidate_selection_exact == (comp.stored_rows.div_ceil(8) <= 2048)
                && !p.numerical_or_model_equivalence_qualified,
            "index chunk/owner/visibility plan differs"
        );
        let expected_source = KV_SOURCES.iter().copied().filter(|&s| s <= p.layer).last();
        ensure!(
            p.kv_source == expected_source
                && p.index_source == Some(p.layer)
                && p.ratio == if p.layer < 20 { 2 } else { 1 },
            "index source/ratio identity"
        );
        let owner = comm.completion_identity();
        ensure!(
            store.rank == self.rank
                && store.manifest == self.manifest
                && Arc::ptr_eq(&store.kernels, &self.kernels)
                && comm.rank() == self.rank
                && comm.compute_stream().stream == self.kernels.stream.stream
                && self.completion.is_none_or(|old| old == owner),
            "index rank/context/stream/session owner differs"
        );
        self.kernels.input(x)?;
        self.kernels.input(qr)?;
        let t = p.ticket.tokens;
        let n = comp.stored_rows;
        ensure!(
            x.len() == t * 5120 && qr.len() == t * 1280,
            "index source BF16 input dimensions"
        );
        ensure!(
            qfreq.profile() == RopeProfile::Yarn
                && qfreq.table_sha() == qfreq.backend().table_sha(RopeProfile::Yarn)
                && self.frequency_backend.is_none_or(|b| b == qfreq.backend())
                && store.frequency_backend.is_none_or(|b| b == qfreq.backend())
                && qfreq.complete_table_verified()
                && qfreq.positions()
                    == (p.ticket.start..p.ticket.end)
                        .map(|v| v as u32)
                        .collect::<Vec<_>>(),
            "index verified query frequency positions"
        );
        ensure!(
            candidate.is_some() == (p.layer > 20) && id.is_some() == p.index_scoring_required,
            "index candidate/BF16 operation posture"
        );
        let scratch_estimate = t * n * 34 + 4 * t * 512 + 4 * MAX_CONTEXT * 128 + 1024 * 1024;
        ensure!(
            scratch_estimate <= STEP_ARRAY_BOUND_BYTES,
            "index temporary array budget"
        );
        {
            let mut v = self
                .validity
                .lock()
                .map_err(|_| anyhow::anyhow!("index registry poisoned"))?;
            ensure!(
                !v.failed && v.lane == Some(p.ticket.lane),
                "index failed/unbound generation"
            );
            v.version = Some(p.ticket.version);
        }
        self.last_version = p.ticket.version;
        self.completion = Some(owner);
        self.frequency_backend = Some(qfreq.backend());
        store.frequency_backend = Some(qfreq.backend());
        let token = OutputToken {
            registry: self.validity.clone(),
            ticket: p.ticket.clone(),
        };
        let k = self.kernels.clone();
        let s = &k.stream;
        let prefix = format!("layers.{}.attn.indexer", self.layer);
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        if p.owns_kv_and_index_keys && comp.new_rows > 0 {
            let latent = latent.context("owner requires normalized unrotated latent")?;
            k.input(latent)?;
            ensure!(
                latent.len() == comp.new_rows * 512,
                "index latent512 extent"
            );
            let f = kfreq.context("completed key groups require exact frequency rows")?;
            let positions = comp
                .groups
                .iter()
                .map(|g| g.rope_position as u32)
                .collect::<Vec<_>>();
            ensure!(
                f.complete_table_verified()
                    && f.profile() == RopeProfile::Yarn
                    && f.backend() == qfreq.backend()
                    && f.positions() == positions,
                "index key frequency backend/positions"
            );
            let projected = Queued::new(
                &k.dev,
                k.dense.bf16_linear_bf16(
                    s,
                    latent,
                    self.weights.bf16(&format!("{prefix}.wk.weight"))?,
                    DenseShape {
                        rows: comp.new_rows,
                        outputs: 128,
                        inner: 512,
                    },
                )?,
            );
            let normalized = Queued::new(&k.dev, unsafe {
                k.numerics.rms_norm_1017(
                    s,
                    projected.device_values_unchecked(),
                    self.key_norm_f32.as_ref().context("owner key norm")?,
                    comp.new_rows,
                    128,
                )?
            });
            let mut rotated = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(comp.new_rows * 128)?);
            unsafe {
                k.copy_range(
                    normalized.device_values_unchecked(),
                    0,
                    &mut rotated,
                    0,
                    comp.new_rows * 128,
                )?;
            }
            k.position.rotate(
                s,
                &mut rotated,
                f,
                RopeProfile::Yarn,
                &positions,
                &mut status,
                1,
                comp.new_rows,
                1,
                128,
                false,
            )?;
            let qat = Queued::new(
                &k.dev,
                k.cache.quantize(
                    s,
                    &rotated,
                    comp.new_rows,
                    128,
                    CacheFormat::IndexUe8m0Group32,
                )?,
            );
            projected.check()?;
            normalized.check()?;
            qat.native().check()?;
            ensure!(
                k.dev.dtoh_sync_copy(&status)? == [0],
                "index key rotation status"
            );
            store.publish(p, Some(unsafe { qat.device_qat_unchecked() }))?;
        } else {
            ensure!(
                latent.is_none() && kfreq.is_none(),
                "unused index latent/frequency input"
            );
            if p.owns_kv_and_index_keys {
                store.publish(p, None)?;
            }
        }
        let identity = IndexOutputIdentity {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            layer: self.layer,
            index_source: self.layer,
            ticket: p.ticket.clone(),
            rows: t,
            keys: n,
            topk: n.min(512),
            offset: 0,
            local_dtype: "BF16 dot/ReLU/scaled-weight/product/headsum",
            network_dtype: "BF16 RNE each serial XOR round",
            network_operation_id: id,
            network_executed: p.index_scoring_required,
            completion_owner: owner,
            complete_epoch: comm.expected_epoch(),
            exact_all_selected_shortcut: !p.index_scoring_required,
            NCCL_bit_equivalence_qualified: false,
            complete_model_or_quality_qualified: false,
            all_checks_passed: true,
            frequency_backend: qfreq.backend(),
            yarn_table_sha256: qfreq.table_sha().into(),
        };
        if n == 0 {
            self.prefix = p.ticket.end;
            return Ok(IndexOutput {
                identity,
                indices: None,
                candidates: None,
                token,
            });
        }
        let visible = Queued::new(
            &k.dev,
            k.dev.htod_sync_copy(
                &comp
                    .visible_rows_per_query
                    .iter()
                    .map(|&v| v as u32)
                    .collect::<Vec<_>>(),
            )?,
        );
        let keys = Queued::new(&k.dev, store.snapshot(p)?);
        let mut mask = Queued::new(&k.dev, k.dev.alloc_zeros::<u8>(t * n)?);
        let mut indices = Queued::new(&k.dev, k.dev.alloc_zeros::<i32>(t * n.min(512))?);
        let mut final_identity = identity;
        if p.index_scoring_required {
            let qrq = Queued::new(
                &k.dev,
                k.numerics.quantize_activations_bf16(s, qr, t, 1280)?,
            );
            let projected = Queued::new(&k.dev, unsafe {
                k.gemm.fp8_complete_bf16(
                    s,
                    qrq.device_values_unchecked(),
                    qrq.device_scales_unchecked(),
                    self.weights.native_u8(&format!("{prefix}.wq_b.weight"))?,
                    self.weights.native_u8(&format!("{prefix}.wq_b.scale"))?,
                    GemmShape {
                        rows: t,
                        outputs: 512,
                        inner: 1280,
                    },
                )?
            });
            let mut rotated = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * 512)?);
            unsafe {
                k.copy_range(
                    projected.device_values_unchecked(),
                    0,
                    &mut rotated,
                    0,
                    t * 512,
                )?;
            }
            k.position.rotate(
                s,
                &mut rotated,
                qfreq,
                RopeProfile::Yarn,
                qfreq.positions(),
                &mut status,
                1,
                t,
                4,
                128,
                false,
            )?;
            let q = Queued::new(
                &k.dev,
                k.cache
                    .quantize(s, &rotated, t * 4, 128, CacheFormat::IndexUe8m0Group32)?,
            );
            let weights = Queued::new(
                &k.dev,
                k.dense.bf16_linear_bf16(
                    s,
                    x,
                    self.weights
                        .bf16(&format!("{prefix}.weights_proj.weight"))?,
                    DenseShape {
                        rows: t,
                        outputs: 4,
                        inner: 5120,
                    },
                )?,
            );
            qrq.check()?;
            projected.check()?;
            q.native().check()?;
            weights.check()?;
            ensure!(
                k.dev.dtoh_sync_copy(&status)? == [0],
                "index query rotation status"
            );
            let mut scores = k.compression.index_scores(
                s,
                unsafe { q.device_qat_unchecked() },
                &keys,
                unsafe { weights.device_values_unchecked() },
                IndexBatchIdentity {
                    shape: IndexShape {
                        batch: 1,
                        queries: t,
                        keys: n,
                    },
                    lanes: vec![p.ticket.lane],
                    start_positions: vec![p.ticket.start],
                },
                IndexNetworkPolicy::SourceBf16LocalAndNetwork,
            )?;
            scores.check()?;
            {
                let mut reduced = comm.enqueue_bf16_matrix(
                    id.unwrap(),
                    MatrixBf16Index::new(t, n)?,
                    unsafe { scores.source_bf16_unchecked()? },
                    &mut self.workspace,
                )?;
                reduced.check(comm)?;
                let source = unsafe { reduced.device_values_unchecked() };
                Self::select(
                    &k,
                    self.rank,
                    &self.manifest,
                    p,
                    source,
                    &visible,
                    candidate,
                    owner,
                    &mut mask,
                    &mut indices,
                    &mut status,
                )?;
            }
            scores.check()?;
            final_identity.complete_epoch = comm.expected_epoch();
        } else {
            // Scores cannot change the selected reachable set in this domain.
            // Actual source keys were still appended above for later>512steps.
            let zero_scores = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(t * n)?);
            Self::select(
                &k,
                self.rank,
                &self.manifest,
                p,
                &zero_scores,
                &visible,
                candidate,
                owner,
                &mut mask,
                &mut indices,
                &mut status,
            )?;
        }
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "index selection sticky status"
        );
        let mut output = Queued::new(&k.dev, k.dev.alloc_zeros::<i32>(t * n.min(512))?);
        k.copy_range(&indices, 0, &mut output, 0, t * n.min(512))?;
        let published = if p.layer == 20 {
            let mut owned = Queued::new(&k.dev, k.dev.alloc_zeros::<u8>(t * n)?);
            k.copy_range(&mask, 0, &mut owned, 0, t * n)?;
            Some(Arc::new(PublishedCandidates {
                mask: owned.into_checked()?,
                identity: final_identity.clone(),
                token: token.clone(),
            }))
        } else {
            None
        };
        k.dev.synchronize()?;
        token.check()?;
        self.prefix = p.ticket.end;
        Ok(IndexOutput {
            identity: final_identity,
            indices: Some(output.into_checked()?),
            candidates: published,
            token,
        })
    }
    fn select(
        k: &Arc<IndexBlockKernels>,
        rank: usize,
        manifest: &str,
        p: &LayerChunk,
        scores: &CudaSlice<bf16>,
        visible: &CudaSlice<u32>,
        candidate: Option<&PublishedCandidates>,
        owner: [u8; 32],
        mask: &mut CudaSlice<u8>,
        indices: &mut CudaSlice<i32>,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let t = p.ticket.tokens;
        let n = p.compressed.as_ref().unwrap().stored_rows;
        if p.layer == 20 {
            k.selection
                .candidates(&k.stream, scores, visible, mask, status, t, n)?
                .check()?;
        }
        let selected = if p.layer == 20 {
            Some(&*mask)
        } else {
            candidate
                .map(|c| c.for_plan(p, rank, manifest, owner))
                .transpose()?
        };
        k.selection
            .positions(
                &k.stream, scores, visible, selected, indices, status, t, n, 0,
            )?
            .check()
    }
}
impl Drop for GpuIndexBlock {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}
impl Drop for GpuIndexKeyStore {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(layer: usize) -> LayerChunk {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let cfg = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let boot = Uuid::from_u128(17);
        let key = LaneKey {
            boot,
            slot: 0,
            generation: 1,
        };
        let mut cursor = chunk_plan::ChunkCursor::new(&cfg, boot, 1).unwrap();
        cursor.reset(key).unwrap();
        let mut result = None;
        for start in (0..1280).step_by(128) {
            let ticket = cursor.begin(key, start, 128).unwrap();
            for l in 0..40 {
                let p = cursor.layer(&ticket, l).unwrap();
                if start == 1152 && l == layer {
                    result = Some(p.clone());
                }
                cursor.commit_layer(&p).unwrap();
            }
            cursor.commit(&ticket).unwrap();
        }
        result.unwrap()
    }
    fn receipt(p: &LayerChunk) -> IndexOutputIdentity {
        let n = p.compressed.as_ref().unwrap().stored_rows;
        let source = p.index_source.unwrap();
        IndexOutputIdentity {
            rank: 3,
            manifest_sha256: "a".repeat(64),
            layer: source,
            index_source: source,
            ticket: p.ticket.clone(),
            rows: p.ticket.tokens,
            keys: n,
            topk: n.min(512),
            offset: 0,
            local_dtype: "BF16 dot/ReLU/scaled-weight/product/headsum",
            network_dtype: "BF16 RNE each serial XOR round",
            network_operation_id: (n > 512).then_some(27),
            network_executed: n > 512,
            completion_owner: [5; 32],
            complete_epoch: 789,
            exact_all_selected_shortcut: n <= 512,
            NCCL_bit_equivalence_qualified: false,
            complete_model_or_quality_qualified: false,
            all_checks_passed: true,
            frequency_backend: FrequencyBackend::CudaTorch,
            yarn_table_sha256: FrequencyBackend::CudaTorch
                .table_sha(RopeProfile::Yarn)
                .into(),
        }
    }
    #[test]
    fn metadata_bridge_reuses_source_not_consumer_layer_and_rejects_crossed_outputs() {
        for layer in [
            2, 3, 7, 8, 13, 14, 19, 20, 23, 24, 27, 28, 31, 32, 35, 36, 39,
        ] {
            let p = plan(layer);
            let v = receipt(&p);
            validate_index_receipt(&p, &v, 3, &"a".repeat(64), [5; 32]).unwrap();
            for control in 0..10 {
                let mut bad = v.clone();
                match control {
                    0 => bad.index_source = (bad.index_source + 4) % 40,
                    1 => bad.ticket.lane.generation += 1,
                    2 => bad.ticket.version += 1,
                    3 => bad.ticket.start += 1,
                    4 => bad.keys += 1,
                    5 => bad.offset = 255,
                    6 => bad.network_dtype = "F32 alternative",
                    7 => bad.all_checks_passed = false,
                    8 => bad.completion_owner = [6; 32],
                    _ => bad.network_operation_id = None,
                }
                assert!(validate_index_receipt(&p, &bad, 3, &"a".repeat(64), [5; 32]).is_err());
            }
        }
    }
    #[test]
    fn failed_or_newer_attempt_permanently_revokes_previous_publication() {
        let p = plan(20);
        let registry = Arc::new(Mutex::new(Validity {
            lane: Some(p.ticket.lane),
            version: Some(p.ticket.version),
            failed: false,
        }));
        let token = OutputToken {
            registry: registry.clone(),
            ticket: p.ticket.clone(),
        };
        token.check().unwrap();
        registry.lock().unwrap().failed = true;
        assert!(token.check().is_err());
        let mut next = p.ticket.clone();
        next.version += 1;
        next.lane.generation += 1;
        *registry.lock().unwrap() = Validity {
            lane: Some(next.lane),
            version: Some(next.version),
            failed: false,
        };
        assert!(token.check().is_err());
        OutputToken {
            registry,
            ticket: next,
        }
        .check()
        .unwrap();
    }
}
