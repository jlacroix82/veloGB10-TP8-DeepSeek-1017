//! PRIVATE packed native1017 block composition with true per-lane attention. Delayed mHC and source
//! MoE precision/order are unchanged. No head/HTTP/reset/prefix commit is hidden.
//! The exact cache snapshot commits only after this whole layer has checked.
use super::{
    batch_plan::{BatchTicket, PackedKind},
    chunk_plan,
    collectives::RoutedLayout,
    config::Dsv41Config,
    gpu_batch_attention_block::{
        BatchAttentionBlockKernels, GpuBatchAttentionBlock, PackedAttentionReceipt,
    },
    gpu_batch_store::{GpuBatchStore, PackedKvSnapshot},
    gpu_collectives::GpuCollectives,
    gpu_experts::{ExpertShape, ExpertWorkspace},
    gpu_hc::HcKernels,
    gpu_hc_graph::HcGraphs,
    gpu_moe_block::{MoeBlock, MoeBlockKernels},
    gpu_numerics::GpuNumerics,
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_weights::{LoadScope, LoadedWeights},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use half::bf16;
use serde::Serialize;
use std::sync::{Arc, Mutex, MutexGuard};
const DIM: usize = 5120;
const COPIES: usize = 4;
const HC_WIDTHS: usize = 16;
const HC_BUFFERS: usize = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecodeHcScratchMode {
    Off,
    Reuse,
}
impl DecodeHcScratchMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "reuse" => Ok(Self::Reuse),
            _ => anyhow::bail!("unsupported packed HC scratch policy {value}"),
        }
    }
}

fn reusable_width(kind: PackedKind, width: usize) -> bool {
    kind == PackedKind::DecodeLanes && (1..=HC_WIDTHS).contains(&width)
}
const fn scratch_bytes(t: usize) -> usize {
    4 + t * ((24 + 4 + 4 + 16 + 24 + 4 + 16) * 4 + (DIM + COPIES * DIM + DIM) * 2)
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeHcScratchReceipt {
    pub width: usize,
    pub borrowed_buffers: usize,
    pub retained_bytes_for_width: usize,
    pub zeroing_preserved: bool,
}

struct HcScratch {
    status: CudaSlice<u32>,
    attn_mixes: CudaSlice<f32>,
    attn_pre: CudaSlice<f32>,
    attn_post: CudaSlice<f32>,
    attn_comb: CudaSlice<f32>,
    collapsed_attn: CudaSlice<bf16>,
    after_attn: CudaSlice<bf16>,
    ffn_mixes: CudaSlice<f32>,
    ffn_post: CudaSlice<f32>,
    ffn_comb: CudaSlice<f32>,
    collapsed_ffn: CudaSlice<bf16>,
}
impl HcScratch {
    fn new(dev: &Arc<CudaDevice>, t: usize) -> Result<Self> {
        // Initialize storage on the existing allocator stream at model load.
        // Borrowers also clear each buffer at its original use boundary.
        Ok(Self {
            status: dev.alloc_zeros(1)?,
            attn_mixes: dev.alloc_zeros(t * 24)?,
            attn_pre: dev.alloc_zeros(t * 4)?,
            attn_post: dev.alloc_zeros(t * 4)?,
            attn_comb: dev.alloc_zeros(t * 16)?,
            collapsed_attn: dev.alloc_zeros(t * DIM)?,
            after_attn: dev.alloc_zeros(t * COPIES * DIM)?,
            ffn_mixes: dev.alloc_zeros(t * 24)?,
            ffn_post: dev.alloc_zeros(t * 4)?,
            ffn_comb: dev.alloc_zeros(t * 16)?,
            collapsed_ffn: dev.alloc_zeros(t * DIM)?,
        })
    }
}

#[derive(Default)]
struct ScratchOwner {
    owner: Option<[u8; 32]>,
    poisoned: bool,
}
impl ScratchOwner {
    fn bind(&mut self, owner: [u8; 32]) -> Result<()> {
        ensure!(
            owner != [0; 32] && self.owner.is_none() && !self.poisoned,
            "HC scratch binds one model transport owner"
        );
        self.owner = Some(owner);
        Ok(())
    }
    fn check(&self, owner: [u8; 32]) -> Result<()> {
        ensure!(
            self.owner == Some(owner) && !self.poisoned,
            "HC scratch owner mismatch or poisoned model"
        );
        Ok(())
    }
}
struct HcPool {
    binding: ScratchOwner,
    graphs: Option<HcGraphs>,
    widths: Vec<HcScratch>,
}
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct BatchBlockOperationIds {
    pub attention_f32: u64,
    pub routed_experts_f32: u64,
}
impl BatchBlockOperationIds {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.attention_f32 < self.routed_experts_f32,
            "packed collective IDs must be unique and ordered attention->MoE"
        );
        Ok(())
    }
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

enum BlockTemp<'a, T> {
    Owned(Queued<CudaSlice<T>>),
    Reused(&'a mut CudaSlice<T>),
}
impl<'a, T: DeviceRepr + ValidAsZeroBits> BlockTemp<'a, T> {
    fn init(
        dev: &Arc<CudaDevice>,
        existing: Option<&'a mut CudaSlice<T>>,
        len: usize,
    ) -> Result<Self> {
        if let Some(value) = existing {
            ensure!(
                value.len() == len && Arc::ptr_eq(&value.device(), dev),
                "HC scratch device/shape mismatch"
            );
            dev.memset_zeros(value)?;
            Ok(Self::Reused(value))
        } else {
            Ok(Self::Owned(Queued::new(dev, dev.alloc_zeros(len)?)))
        }
    }
}
impl<T> std::ops::Deref for BlockTemp<'_, T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(v) => v,
            Self::Reused(v) => v,
        }
    }
}
impl<T> std::ops::DerefMut for BlockTemp<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(v) => v,
            Self::Reused(v) => v,
        }
    }
}

pub struct BatchBlockKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    hc: HcKernels,
    numerics: GpuNumerics,
    position: PositionKernels,
    decode_hc: Option<Mutex<HcPool>>,
}
impl BatchBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>) -> Result<Arc<Self>> {
        Self::load_with_hc_scratch(dev, stream, DecodeHcScratchMode::Off)
    }
    pub fn load_with_hc_scratch(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        mode: DecodeHcScratchMode,
    ) -> Result<Arc<Self>> {
        Self::load_with_hc_graph(dev, stream, mode, false)
    }
    pub fn load_with_hc_graph(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        mode: DecodeHcScratchMode,
        graph: bool,
    ) -> Result<Arc<Self>> {
        ensure!(
            !graph || mode == DecodeHcScratchMode::Reuse,
            "HC graph requires persistent scratch"
        );
        let decode_hc = if mode == DecodeHcScratchMode::Reuse {
            let widths = (1..=HC_WIDTHS)
                .map(|t| HcScratch::new(dev, t))
                .collect::<Result<Vec<_>>>()?;
            dev.synchronize()?;
            Some(Mutex::new(HcPool {
                binding: ScratchOwner::default(),
                graphs: graph.then(HcGraphs::default),
                widths,
            }))
        } else {
            None
        };
        let me = Self {
            dev: dev.clone(),
            hc: HcKernels::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            decode_hc,
            stream,
        };
        me.check_stream()?;
        Ok(Arc::new(me))
    }
    pub fn bind_scratch(&self, owner: [u8; 32]) -> Result<()> {
        if let Some(pool) = &self.decode_hc {
            pool.lock()
                .map_err(|_| anyhow::anyhow!("HC scratch lock poisoned"))?
                .binding
                .bind(owner)?;
        }
        Ok(())
    }
    pub fn scratch_bytes(&self) -> usize {
        self.decode_hc
            .as_ref()
            .map_or(0, |_| (1..=HC_WIDTHS).map(scratch_bytes).sum())
    }
    pub fn poison_scratch(&self) {
        if let Some(pool) = &self.decode_hc {
            // A poisoned lock already refuses use; still poison its owner.
            pool.lock()
                .unwrap_or_else(|e| e.into_inner())
                .binding
                .poisoned = true;
        }
    }
    fn checkout_scratch(
        &self,
        owner: [u8; 32],
        kind: PackedKind,
        width: usize,
    ) -> Result<Option<MutexGuard<'_, HcPool>>> {
        if let Some(pool) = self
            .decode_hc
            .as_ref()
            .filter(|_| reusable_width(kind, width))
        {
            let lease = pool
                .lock()
                .map_err(|_| anyhow::anyhow!("HC scratch lock poisoned"))?;
            lease.binding.check(owner)?;
            Ok(Some(lease))
        } else {
            Ok(None)
        }
    }
    fn check_stream(&self) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            !self.stream.stream.is_null()
                && unsafe { sys::cuStreamGetFlags(self.stream.stream, &mut flags) }
                    == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "owned blocking chunk block stream required"
        );
        Ok(())
    }
}
impl Drop for BatchBlockKernels {
    fn drop(&mut self) {
        if self.decode_hc.is_some() {
            let _ = self.dev.synchronize();
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct BatchBlockReceipt {
    pub revision: String,
    pub manifest_sha256: String,
    pub rank: usize,
    pub layer: usize,
    pub ticket: BatchTicket,
    pub operation_ids: BatchBlockOperationIds,
    pub completion_owner: [u8; 32],
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub attention: PackedAttentionReceipt,
    pub incoming_FFN_pre_used_for_attention: bool,
    pub attention_pre_used_for_FFN: bool,
    pub returned_pre_is_FFN_pre: bool,
    pub engram_before_block_required: bool,
    pub engram_execution_certified_by_block: bool,
    pub exact_cache_layer_committed: bool,
    pub packed_prefix_or_head_committed: bool,
    pub expert_workspace_bytes: usize,
    pub all_checks_passed: bool,
    pub full_model_or_quality_qualified: bool,
    pub optimized_execution: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_hc_scratch: Option<DecodeHcScratchReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_hc_graph: Option<DecodeHcGraphReceipt>,
}
#[derive(Debug, Clone, Serialize)]
pub struct DecodeHcGraphReceipt {
    pub width: usize,
    pub kernel_nodes: usize,
    pub captured_this_layer: bool,
}
/// Fully owned/fenced outputs. The caller feeds next_pre to the NEXT layer,
/// and initializes fresh identity premix for a new model forward/chunk.
pub struct CheckedBatchBlock {
    pub h: CudaSlice<bf16>,
    pub next_pre: CudaSlice<f32>,
    pub receipt: BatchBlockReceipt,
    pub snapshot: Arc<PackedKvSnapshot>,
}
pub struct GpuBlockBatch {
    kernels: Arc<BatchBlockKernels>,
    weights: Arc<LoadedWeights>,
    attention: GpuBatchAttentionBlock,
    moe: MoeBlock,
    workspace: Option<ExpertWorkspace>,
    attn_norm_f32: CudaSlice<f32>,
    ffn_norm_f32: CudaSlice<f32>,
    manifest: String,
    revision: String,
    rank: usize,
    layer: usize,
    engram_required: bool,
    failed: bool,
    completion: Option<[u8; 32]>,
}
impl GpuBlockBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: Arc<BatchBlockKernels>,
        attention_kernels: Arc<BatchAttentionBlockKernels>,
        moe_kernels: Arc<MoeBlockKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        layer: usize,
        w: Arc<LoadedWeights>,
        completion_owner: [u8; 32],
        max_lanes: usize,
    ) -> Result<Self> {
        k.check_stream()?;
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            layer < 40
                && w.scope() == &LoadScope::Layer(layer)
                && w.manifest_sha256() == m.manifest_sha256,
            "actual chunk block layer scope/manifest"
        );
        let prefix = format!("layers.{layer}");
        for stage in ["attn", "ffn"] {
            ensure!(
                w.f32(&format!("{prefix}.hc_{stage}_fn"))?.len() == 24 * COPIES * DIM
                    && w.f32(&format!("{prefix}.hc_{stage}_base"))?.len() == 24
                    && w.f32(&format!("{prefix}.hc_{stage}_scale"))?.len() == 3,
                "source HC matrix geometry"
            );
        }
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        let mut attn = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(DIM)?);
        let mut ffn = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(DIM)?);
        k.position.cast_bf16_f32(
            &k.stream,
            w.bf16(&format!("{prefix}.attn_norm.weight"))?,
            &mut attn,
            &mut status,
        )?;
        k.position.cast_bf16_f32(
            &k.stream,
            w.bf16(&format!("{prefix}.ffn_norm.weight"))?,
            &mut ffn,
            &mut status,
        )?;
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "chunk block norm initialization"
        );
        let attention = GpuBatchAttentionBlock::new(
            attention_kernels,
            c,
            m,
            completion_owner,
            layer,
            w.clone(),
            max_lanes,
        )?;
        let moe = MoeBlock::new(moe_kernels, w.clone(), m, layer)?;
        Ok(Self {
            kernels: k,
            weights: w,
            attention,
            moe,
            workspace: None,
            attn_norm_f32: attn.into_checked()?,
            ffn_norm_f32: ffn.into_checked()?,
            manifest: m.manifest_sha256.clone(),
            revision: m.model_revision.clone(),
            rank: m.rank,
            layer,
            engram_required: c.engram_layer_ids.contains(&layer),
            failed: false,
            completion: Some(completion_owner),
        })
    }
    /// Explicit caller action only. Shared packed store resets once outside40
    /// blocks; this resets only this block's generation-bound attention state.
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        self.kernels.check_stream()?;
        self.attention.reset_lane(key)?;
        self.failed = false;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn execute_checked(
        &mut self,
        ticket: &BatchTicket,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        store: &mut GpuBatchStore,
        comm: &mut GpuCollectives,
        ids: BatchBlockOperationIds,
    ) -> Result<CheckedBatchBlock> {
        let result = self.execute_inner(
            ticket,
            h,
            incoming_pre,
            query_freq,
            compressed_freq,
            store,
            comm,
            ids,
        );
        if result.is_err() {
            comm.abort_session();
            self.kernels.poison_scratch();
            self.failed = true;
            let _ = store.abort(ticket);
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        ticket: &BatchTicket,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        store: &mut GpuBatchStore,
        comm: &mut GpuCollectives,
        ids: BatchBlockOperationIds,
    ) -> Result<CheckedBatchBlock> {
        self.kernels.check_stream()?;
        ids.validate()?;
        ensure!(
            !self.failed,
            "failed chunk block requires explicit fresh generation"
        );
        let owner = comm.completion_identity();
        let k = self.kernels.clone();
        let s = &k.stream;
        let t = ticket.total_rows();
        store.check_owner(self.rank, &self.manifest, owner, s)?;
        let plan = store.layer(ticket, self.layer)?;
        ensure!(
            comm.rank() == self.rank
                && comm.compute_stream().stream == s.stream
                && self.completion.is_none_or(|old| old == owner)
                && h.len() == t * COPIES * DIM
                && incoming_pre.len() == t * COPIES
                && Arc::ptr_eq(&h.device(), &k.dev)
                && Arc::ptr_eq(&incoming_pre.device(), &k.dev),
            "chunk block residual/premix/rank/stream/completion binding"
        );
        ensure!(
            plan.short_selection_exact(),
            "first packed block refuses learned index selection above512 keys"
        );
        self.completion = Some(owner);
        let epoch_before = comm.expected_epoch();
        let prefix = format!("layers.{}", self.layer);
        // Error-site abort precedes this wrapper's explicit fence. Internal
        // qualified primitive Drop fences retain their synchronous-control scope.
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{comm.abort_session();let _=k.dev.synchronize();return Err(e.into());}}};}
        let mut lease = k.checkout_scratch(owner, ticket.kind(), t)?;
        let used_scratch = lease.is_some();
        // These are all block-local; none become the next layer's inputs or
        // point into the KV snapshot. Keep the lease through checked commit.
        let (scratch_slot, graph_slot) = match lease.as_mut() {
            Some(p) => {
                let HcPool { widths, graphs, .. } = &mut **p;
                (Some(&mut widths[t - 1]), graphs.as_mut())
            }
            None => (None, None),
        };
        let (
            s_status,
            s_attn_mixes,
            s_attn_pre,
            s_attn_post,
            s_attn_comb,
            s_collapsed_attn,
            s_after_attn,
            s_ffn_mixes,
            s_ffn_post,
            s_ffn_comb,
            s_collapsed_ffn,
        ) = match scratch_slot {
            Some(HcScratch {
                status,
                attn_mixes,
                attn_pre,
                attn_post,
                attn_comb,
                collapsed_attn,
                after_attn,
                ffn_mixes,
                ffn_post,
                ffn_comb,
                collapsed_ffn,
            }) => (
                Some(status),
                Some(attn_mixes),
                Some(attn_pre),
                Some(attn_post),
                Some(attn_comb),
                Some(collapsed_attn),
                Some(after_attn),
                Some(ffn_mixes),
                Some(ffn_post),
                Some(ffn_comb),
                Some(collapsed_ffn),
            ),
            None => (
                None, None, None, None, None, None, None, None, None, None, None,
            ),
        };
        if self.workspace.as_ref().is_none_or(|w| w.rows() != t) {
            self.workspace = Some(queued!(self.moe.workspace(t)));
        }
        let workspace_bytes = ExpertShape::native1017().workspace_bytes(t)?;
        let mut status = queued!(BlockTemp::init(&k.dev, s_status, 1));
        let mut attn_mixes = queued!(BlockTemp::init(&k.dev, s_attn_mixes, t * 24));
        let mut attn_pre = queued!(BlockTemp::init(&k.dev, s_attn_pre, t * 4));
        let mut attn_post = queued!(BlockTemp::init(&k.dev, s_attn_post, t * 4));
        let mut attn_comb = queued!(BlockTemp::init(&k.dev, s_attn_comb, t * 16));
        // NEW attention premix is computed from h, but attention consumes the
        // incoming FFN premix, exactly the delayed source schedule.
        let mut collapsed_attn = queued!(BlockTemp::init(&k.dev, s_collapsed_attn, t * DIM));
        let weight = queued!(self.weights.f32(&format!("{prefix}.hc_attn_fn")));
        let scale = queued!(self.weights.f32(&format!("{prefix}.hc_attn_scale")));
        let base = queued!(self.weights.f32(&format!("{prefix}.hc_attn_base")));
        ensure!(
            weight.len() == 24 * COPIES * DIM
                && scale.len() == 3
                && base.len() == 24
                && [weight, scale, base]
                    .iter()
                    .all(|v| Arc::ptr_eq(&v.device(), &k.dev)),
            "HC triplet weight shape/context"
        );
        let checked = match k.hc.check_attention_input(
            s,
            h,
            weight,
            &mut attn_mixes,
            &mut status,
            scale,
            base,
            &mut attn_pre,
            &mut attn_post,
            &mut attn_comb,
            incoming_pre,
            &mut collapsed_attn,
            t,
            DIM,
        ) {
            Ok(checked) => checked,
            Err(error) => {
                comm.abort_session();
                let _ = k.dev.synchronize();
                return Err(error);
            }
        };
        let graph_receipt = if let Some(g) = graph_slot {
            let args = checked.arguments();
            let captured = queued!(g.enqueue(t, s, args, || checked.enqueue()));
            Some(DecodeHcGraphReceipt {
                width: t,
                kernel_nodes: 3,
                captured_this_layer: captured,
            })
        } else {
            queued!(checked.enqueue());
            None
        };
        let normalized_attn = Queued::new(
            &k.dev,
            queued!(k
                .numerics
                .rms_norm_1017(s, &collapsed_attn, &self.attn_norm_f32, t, DIM)),
        );
        // New attention requires its supplied normalized input producer checked.
        queued!(normalized_attn.check());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&*status)) == [0],
            "incoming attention HC/norm status"
        );
        let attention = queued!(self.attention.execute_checked(
            ticket,
            unsafe { normalized_attn.device_values_unchecked() },
            query_freq,
            compressed_freq,
            ids.attention_f32,
            store,
            comm
        ));
        let attention_receipt = queued!(attention.receipt().cloned());
        let snapshot = queued!(attention.snapshot().cloned());
        let mut after_attn = queued!(BlockTemp::init(&k.dev, s_after_attn, t * COPIES * DIM));
        queued!(k.hc.expand(
            s,
            attention.values()?,
            h,
            &attn_post,
            &attn_comb,
            &mut after_attn,
            &mut status,
            t,
            DIM
        ));
        let mut ffn_mixes = queued!(BlockTemp::init(&k.dev, s_ffn_mixes, t * 24));
        let mut next_pre = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        let mut ffn_post = queued!(BlockTemp::init(&k.dev, s_ffn_post, t * 4));
        let mut ffn_comb = queued!(BlockTemp::init(&k.dev, s_ffn_comb, t * 16));
        queued!(k.hc.project(
            s,
            &after_attn,
            self.weights.f32(&format!("{prefix}.hc_ffn_fn"))?,
            &mut ffn_mixes,
            &mut status,
            t,
            DIM
        ));
        queued!(k.hc.split(
            s,
            &ffn_mixes,
            self.weights.f32(&format!("{prefix}.hc_ffn_scale"))?,
            self.weights.f32(&format!("{prefix}.hc_ffn_base"))?,
            &mut next_pre,
            &mut ffn_post,
            &mut ffn_comb,
            &mut status,
            t
        ));
        let mut collapsed_ffn = queued!(BlockTemp::init(&k.dev, s_collapsed_ffn, t * DIM));
        // Attention-pre collapses THIS FFN. Fresh FFN-pre belongs to NEXT layer.
        queued!(k.hc.collapse(
            s,
            &after_attn,
            &attn_pre,
            &mut collapsed_ffn,
            &mut status,
            t,
            DIM
        ));
        let normalized_ffn = Queued::new(
            &k.dev,
            queued!(k
                .numerics
                .rms_norm_1017(s, &collapsed_ffn, &self.ffn_norm_f32, t, DIM)),
        );
        queued!(normalized_ffn.check());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&*status)) == [0],
            "post-attention/FFN HC/norm status"
        );
        let prepared =
            queued!(self
                .moe
                .prepare(s, unsafe { normalized_ffn.device_values_unchecked() }, t));
        // Borrowed outputs use direct matches: no escaping mutable reference is
        // returned from a closure, and errors abort/fence before local teardown.
        let routed = match self.moe.enqueue_routed(
            &prepared,
            self.workspace.as_mut().expect("workspace allocated"),
        ) {
            Ok(v) => v,
            Err(e) => {
                comm.abort_session();
                let _ = k.dev.synchronize();
                return Err(e);
            }
        };
        let mut reduced = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * DIM)));
        let source = unsafe { routed.device_values_unchecked() };
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *reduced.device_ptr(),
                *source.device_ptr(),
                source.len() * 4,
                s.stream,
            )
        };
        queued!(if rc == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(anyhow::anyhow!("routed source copy failed {rc:?}"))
        });
        queued!(comm.enqueue_reduce(
            ids.routed_experts_f32,
            RoutedLayout::new(t, ticket.kind().collective_rows())?,
            &mut reduced
        ));
        let mut ffn_output = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(t * DIM)));
        let mut combination_status = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<u32>(1)));
        let combined = match self.moe.finish_checked(
            &prepared,
            &reduced,
            comm,
            &mut ffn_output,
            &mut combination_status,
        ) {
            Ok(v) => v,
            Err(e) => {
                comm.abort_session();
                let _ = k.dev.synchronize();
                return Err(e);
            }
        };
        let mut final_h = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(t * COPIES * DIM)));
        queued!(k.hc.expand(
            s,
            unsafe { combined.device_values_unchecked() },
            &after_attn,
            &ffn_post,
            &ffn_comb,
            &mut final_h,
            &mut status,
            t,
            DIM
        ));
        queued!(comm.synchronize());
        queued!(normalized_attn.check());
        queued!(normalized_ffn.check());
        queued!(attention.check());
        queued!(routed.check());
        queued!(combined.check());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&*status)) == [0],
            "complete HC/MoE/chunk block numeric status"
        );
        let h = queued!(final_h.into_checked());
        let next_pre = queued!(next_pre.into_checked());
        // Commit exact cache only after ALL attention/HC/MoE outputs and producer
        // statuses are checked/fenced. This is NOT the final head/prefix commit.
        queued!(store.commit_layer(ticket, &snapshot));
        Ok(CheckedBatchBlock {
            h,
            next_pre,
            snapshot,
            receipt: BatchBlockReceipt {
                revision: self.revision.clone(),
                manifest_sha256: self.manifest.clone(),
                rank: self.rank,
                layer: self.layer,
                ticket: ticket.clone(),
                operation_ids: ids,
                completion_owner: owner,
                epoch_before,
                epoch_after: comm.expected_epoch(),
                attention: attention_receipt,
                incoming_FFN_pre_used_for_attention: true,
                attention_pre_used_for_FFN: true,
                returned_pre_is_FFN_pre: true,
                engram_before_block_required: self.engram_required,
                engram_execution_certified_by_block: false,
                exact_cache_layer_committed: true,
                packed_prefix_or_head_committed: false,
                expert_workspace_bytes: workspace_bytes,
                all_checks_passed: true,
                full_model_or_quality_qualified: false,
                optimized_execution: false,
                decode_hc_scratch: used_scratch.then_some(DecodeHcScratchReceipt {
                    width: t,
                    borrowed_buffers: HC_BUFFERS,
                    retained_bytes_for_width: scratch_bytes(t),
                    zeroing_preserved: true,
                }),
                decode_hc_graph: graph_receipt,
            },
        })
    }
}
impl Drop for GpuBlockBatch {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    #[test]
    fn failed_owner_cannot_reenter_or_rebind_the_pool() {
        let mut p = ScratchOwner::default();
        assert!(p.check([1; 32]).is_err());
        p.bind([1; 32]).unwrap();
        assert!(p.check([2; 32]).is_err());
        p.check([1; 32]).unwrap();
        p.poisoned = true;
        assert!(p.check([1; 32]).is_err());
        assert!(p.bind([2; 32]).is_err());
    }

    #[test]
    fn pool_admits_decode_width_changes_but_never_prefill_or_oversize() {
        for width in [1, 8, 16, 1, 8] {
            assert!(reusable_width(PackedKind::DecodeLanes, width));
            assert!(!reusable_width(PackedKind::PrefillSegments, width));
        }
        assert!(!reusable_width(PackedKind::DecodeLanes, 0));
        assert!(!reusable_width(PackedKind::DecodeLanes, 17));
    }
}
