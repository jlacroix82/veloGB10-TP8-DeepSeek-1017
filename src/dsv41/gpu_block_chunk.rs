//! PRIVATE additive8192-context C1 block composition. Delayed mHC and source
//! MoE precision/order are unchanged. No head/HTTP/reset/prefix commit is hidden.
//! The exact cache snapshot commits only after this whole layer has checked.
use super::{
    chunk_plan::{self, ChunkTicket},
    collectives::RoutedLayout,
    config::Dsv41Config,
    gpu_attention_chunk::{
        row_kind, ChunkAttentionKernels, ChunkAttentionOperationIds, ChunkAttentionReceipt,
        ChunkIndexExchange, GpuAttentionChunk,
    },
    gpu_chunk_store::{GpuChunkStore, OwnedChunkSnapshot},
    gpu_collectives::GpuCollectives,
    gpu_experts::{ExpertShape, ExpertWorkspace},
    gpu_hc::HcKernels,
    gpu_moe_block::{MoeBlock, MoeBlockKernels},
    gpu_numerics::GpuNumerics,
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_weights::{LoadScope, LoadedWeights},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::sync::Arc;
const DIM: usize = 5120;
const COPIES: usize = 4;
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct ChunkBlockOperationIds {
    pub index_bf16: Option<u64>,
    pub attention_f32: u64,
    pub routed_experts_f32: u64,
}
impl ChunkBlockOperationIds {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.attention_f32 < self.routed_experts_f32
                && self.index_bf16.is_none_or(|id| id < self.attention_f32),
            "source collective IDs must be unique and ordered index->attention->MoE"
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
pub struct ChunkBlockKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    hc: HcKernels,
    numerics: GpuNumerics,
    position: PositionKernels,
}
impl ChunkBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>) -> Result<Arc<Self>> {
        let me = Self {
            dev: dev.clone(),
            hc: HcKernels::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            stream,
        };
        me.check_stream()?;
        Ok(Arc::new(me))
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
#[derive(Debug, Clone, Serialize)]
pub struct ChunkBlockReceipt {
    pub revision: String,
    pub manifest_sha256: String,
    pub rank: usize,
    pub layer: usize,
    pub ticket: ChunkTicket,
    pub operation_ids: ChunkBlockOperationIds,
    pub completion_owner: [u8; 32],
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub attention: ChunkAttentionReceipt,
    pub incoming_FFN_pre_used_for_attention: bool,
    pub attention_pre_used_for_FFN: bool,
    pub returned_pre_is_FFN_pre: bool,
    pub engram_before_block_required: bool,
    pub engram_execution_certified_by_block: bool,
    pub exact_cache_layer_committed: bool,
    pub chunk_prefix_or_head_committed: bool,
    pub expert_workspace_bytes: usize,
    pub all_checks_passed: bool,
    pub full_model_or_quality_qualified: bool,
    pub optimized_execution: bool,
}
/// Fully owned/fenced outputs. The caller feeds next_pre to the NEXT layer,
/// and initializes fresh identity premix for a new model forward/chunk.
pub struct CheckedChunkBlock {
    pub h: CudaSlice<bf16>,
    pub next_pre: CudaSlice<f32>,
    pub receipt: ChunkBlockReceipt,
    pub snapshot: Arc<OwnedChunkSnapshot>,
}
pub struct GpuBlockChunk {
    kernels: Arc<ChunkBlockKernels>,
    weights: Arc<LoadedWeights>,
    attention: GpuAttentionChunk,
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
impl GpuBlockChunk {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: Arc<ChunkBlockKernels>,
        attention_kernels: Arc<ChunkAttentionKernels>,
        moe_kernels: Arc<MoeBlockKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        layer: usize,
        w: Arc<LoadedWeights>,
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
        let attention = GpuAttentionChunk::new(attention_kernels, c, m, layer, w.clone())?;
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
            completion: None,
        })
    }
    /// Explicit caller action only. Shared store/exchange reset once outside40
    /// blocks; this resets only this block's generation-bound attention state.
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        self.kernels.check_stream()?;
        self.attention.reset_lane(key)?;
        self.failed = false;
        self.completion = None;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn execute_checked(
        &mut self,
        ticket: &ChunkTicket,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        exchange: &mut ChunkIndexExchange,
        store: &mut GpuChunkStore,
        comm: &mut GpuCollectives,
        ids: ChunkBlockOperationIds,
    ) -> Result<CheckedChunkBlock> {
        let result = self.execute_inner(
            ticket,
            h,
            incoming_pre,
            query_freq,
            compressed_freq,
            exchange,
            store,
            comm,
            ids,
        );
        if result.is_err() {
            comm.abort_session();
            self.failed = true;
            let _ = store.abort(ticket);
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        ticket: &ChunkTicket,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        exchange: &mut ChunkIndexExchange,
        store: &mut GpuChunkStore,
        comm: &mut GpuCollectives,
        ids: ChunkBlockOperationIds,
    ) -> Result<CheckedChunkBlock> {
        self.kernels.check_stream()?;
        ids.validate()?;
        ensure!(
            !self.failed,
            "failed chunk block requires explicit fresh generation"
        );
        let owner = comm.completion_identity();
        let k = self.kernels.clone();
        let s = &k.stream;
        let t = ticket.tokens;
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
            ids.index_bf16.is_some() == plan.index_scoring_required,
            "planned index-source operation posture"
        );
        self.completion = Some(owner);
        let epoch_before = comm.expected_epoch();
        let prefix = format!("layers.{}", self.layer);
        // Error-site abort precedes this wrapper's explicit fence. Internal
        // qualified primitive Drop fences retain their synchronous-control scope.
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{comm.abort_session();let _=k.dev.synchronize();return Err(e.into());}}};}
        if self.workspace.as_ref().is_none_or(|w| w.rows() != t) {
            self.workspace = Some(queued!(self.moe.workspace(t)));
        }
        let workspace_bytes = ExpertShape::native1017().workspace_bytes(t)?;
        let mut status = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<u32>(1)));
        let mut attn_mixes = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 24)));
        let mut attn_pre = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        let mut attn_post = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        let mut attn_comb = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 16)));
        // NEW attention premix is computed from h, but attention consumes the
        // incoming FFN premix, exactly the delayed source schedule.
        queued!(k.hc.project(
            s,
            h,
            self.weights.f32(&format!("{prefix}.hc_attn_fn"))?,
            &mut attn_mixes,
            &mut status,
            t,
            DIM
        ));
        queued!(k.hc.split(
            s,
            &attn_mixes,
            self.weights.f32(&format!("{prefix}.hc_attn_scale"))?,
            self.weights.f32(&format!("{prefix}.hc_attn_base"))?,
            &mut attn_pre,
            &mut attn_post,
            &mut attn_comb,
            &mut status,
            t
        ));
        let mut collapsed_attn = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(t * DIM)));
        queued!(k
            .hc
            .collapse(s, h, incoming_pre, &mut collapsed_attn, &mut status, t, DIM));
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
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "incoming attention HC/norm status"
        );
        let attention = queued!(self.attention.execute_checked(
            ticket,
            unsafe { normalized_attn.device_values_unchecked() },
            query_freq,
            compressed_freq,
            ChunkAttentionOperationIds {
                index_bf16: ids.index_bf16,
                wo_b_f32: ids.attention_f32
            },
            exchange,
            store,
            comm
        ));
        let attention_receipt = queued!(attention.receipt().cloned());
        let snapshot = queued!(attention.snapshot().cloned());
        let mut after_attn =
            Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(t * COPIES * DIM)));
        queued!(k.hc.expand(
            s,
            attention.device_values()?,
            h,
            &attn_post,
            &attn_comb,
            &mut after_attn,
            &mut status,
            t,
            DIM
        ));
        let mut ffn_mixes = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 24)));
        let mut next_pre = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        let mut ffn_post = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        let mut ffn_comb = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 16)));
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
        let mut collapsed_ffn = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(t * DIM)));
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
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
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
            RoutedLayout::new(t, row_kind(ticket.start, t)?)?,
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
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "complete HC/MoE/chunk block numeric status"
        );
        let h = queued!(final_h.into_checked());
        let next_pre = queued!(next_pre.into_checked());
        // Commit exact cache only after ALL attention/HC/MoE outputs and producer
        // statuses are checked/fenced. This is NOT the final head/prefix commit.
        queued!(store.commit_layer(ticket, &snapshot));
        Ok(CheckedChunkBlock {
            h,
            next_pre,
            snapshot,
            receipt: ChunkBlockReceipt {
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
                chunk_prefix_or_head_committed: false,
                expert_workspace_bytes: workspace_bytes,
                all_checks_passed: true,
                full_model_or_quality_qualified: false,
                optimized_execution: false,
            },
        })
    }
}
impl Drop for GpuBlockChunk {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_collective_ids_are_explicit_and_cannot_alias() {
        ChunkBlockOperationIds {
            index_bf16: Some(3),
            attention_f32: 4,
            routed_experts_f32: 5,
        }
        .validate()
        .unwrap();
        ChunkBlockOperationIds {
            index_bf16: None,
            attention_f32: 0,
            routed_experts_f32: 1,
        }
        .validate()
        .unwrap();
        for ids in [
            ChunkBlockOperationIds {
                index_bf16: Some(4),
                attention_f32: 4,
                routed_experts_f32: 5,
            },
            ChunkBlockOperationIds {
                index_bf16: Some(5),
                attention_f32: 4,
                routed_experts_f32: 6,
            },
            ChunkBlockOperationIds {
                index_bf16: None,
                attention_f32: 4,
                routed_experts_f32: 4,
            },
            ChunkBlockOperationIds {
                index_bf16: None,
                attention_f32: 5,
                routed_experts_f32: 4,
            },
        ] {
            assert!(ids.validate().is_err());
        }
    }
    #[test]
    fn delayed_HC_premixes_and_returned_FFN_pre_are_distinct_at_chunk_widths() {
        for rows in [1, 2, 16, 75, 128] {
            let dim = 8;
            let h = (0..rows)
                .flat_map(|r| {
                    [1., 10., 100., 1000.]
                        .into_iter()
                        .flat_map(move |v| vec![bf16::from_f32(v + r as f32); dim])
                })
                .collect::<Vec<_>>();
            let incoming = (0..rows).flat_map(|_| [1., 0., 0., 0.]).collect::<Vec<_>>();
            let attn = (0..rows).flat_map(|_| [0., 1., 0., 0.]).collect::<Vec<_>>();
            let ffn = (0..rows).flat_map(|_| [0., 0., 1., 0.]).collect::<Vec<_>>();
            let a = super::super::hc::collapse(&h, &incoming, rows, dim).unwrap();
            let wrong = super::super::hc::collapse(&h, &attn, rows, dim).unwrap();
            assert_ne!(a, wrong);
            let f = super::super::hc::collapse(&h, &attn, rows, dim).unwrap();
            let next = super::super::hc::collapse(&h, &ffn, rows, dim).unwrap();
            assert_ne!(f, next);
            for row in 0..rows {
                assert_eq!(a[row * dim].to_f32(), 1. + row as f32);
                assert_eq!(f[row * dim].to_f32(), 10. + row as f32);
                assert_eq!(next[row * dim].to_f32(), 100. + row as f32);
            }
        }
    }
}
