//! Fenced first native1017 C1 block composition.
//! HC delayed premix, attention, FFN and source BF16/F32 boundaries are explicit.
//! Engram augmentation is supplied by the caller BEFORE this block. This module
//! does not enable serving, hide the512context bound, or claim tuned throughput.
use super::{
    collectives::{RoutedLayout, RowKind},
    config::Dsv41Config,
    gpu_attention_block::{AttentionBlockKernels, GpuAttentionBlock, MAX_NATIVE_CONTEXT},
    gpu_collectives::GpuCollectives,
    gpu_experts::ExpertWorkspace,
    gpu_hc::HcKernels,
    gpu_kv_store::GpuKvStore,
    gpu_moe_block::{MoeBlock, MoeBlockKernels},
    gpu_numerics::GpuNumerics,
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_weights::LoadedWeights,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;
const DIM: usize = 5120;
const COPIES: usize = 4;
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct BlockOperationIds {
    pub attention: u64,
    pub routed_experts: u64,
}
impl BlockOperationIds {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.attention != self.routed_experts,
            "attention and routed-expert collective IDs must differ"
        );
        Ok(())
    }
}
pub struct BlockKernels {
    dev: Arc<CudaDevice>,
    stream_id: usize,
    hc: HcKernels,
    numerics: GpuNumerics,
    position: PositionKernels,
}
impl BlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, s: &CudaStream) -> Result<Arc<Self>> {
        let me = Self {
            dev: dev.clone(),
            stream_id: s.stream as usize,
            hc: HcKernels::load(dev, s)?,
            numerics: GpuNumerics::load(dev, s)?,
            position: PositionKernels::load(dev, s)?,
        };
        me.stream(s)?;
        Ok(Arc::new(me))
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            !s.stream.is_null() && s.stream as usize == self.stream_id,
            "exact block owner stream required"
        );
        ensure!(
            unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) } == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "blocking block stream required"
        );
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct BlockOutputIdentity {
    pub revision: String,
    pub manifest_sha256: String,
    pub rank: usize,
    pub layer: usize,
    pub lane: LaneKey,
    pub start: usize,
    pub tokens: usize,
    pub operation_ids: BlockOperationIds,
    pub incoming_pre_used_for_attention: bool,
    pub attention_pre_used_for_ffn: bool,
    pub returned_pre_is_FFN_pre: bool,
    pub engram_before_block_required: bool,
    pub maximum_native_context: usize,
    pub all_checks_passed: bool,
    pub whole_model_qualified: bool,
}
pub struct GpuBlock {
    kernels: Arc<BlockKernels>,
    weights: Arc<LoadedWeights>,
    attention: GpuAttentionBlock,
    moe: MoeBlock,
    workspace: Option<ExpertWorkspace>,
    attn_norm_f32: CudaSlice<f32>,
    ffn_norm_f32: CudaSlice<f32>,
    manifest_sha256: String,
    revision: String,
    rank: usize,
    layer: usize,
    engram_required: bool,
    failed: bool,
}
/// Outputs are owned and fenced. The caller can feed them directly to the next
/// block; every producer from this block has completed before this is returned.
pub struct BlockOutput {
    pub h: CudaSlice<bf16>,
    pub next_pre: CudaSlice<f32>,
    pub identity: BlockOutputIdentity,
}
impl GpuBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernels: Arc<BlockKernels>,
        attention_kernels: Arc<AttentionBlockKernels>,
        moe_kernels: Arc<MoeBlockKernels>,
        s: &CudaStream,
        config: &Dsv41Config,
        manifest: &RankManifest,
        layer: usize,
        weights: Arc<LoadedWeights>,
    ) -> Result<Self> {
        kernels.stream(s)?;
        config.validate()?;
        manifest.verify_identity()?;
        let prefix = format!("layers.{layer}");
        for stage in ["attn", "ffn"] {
            ensure!(
                weights.f32(&format!("{prefix}.hc_{stage}_fn"))?.len() == 24 * 4 * DIM
                    && weights.f32(&format!("{prefix}.hc_{stage}_base"))?.len() == 24
                    && weights.f32(&format!("{prefix}.hc_{stage}_scale"))?.len() == 3,
                "block HC geometry"
            );
        }
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=kernels.dev.synchronize();return Err(e.into());}}};}
        let mut status = queued!(kernels.dev.alloc_zeros::<u32>(1));
        let mut attn_norm_f32 = queued!(kernels.dev.alloc_zeros::<f32>(DIM));
        let mut ffn_norm_f32 = queued!(kernels.dev.alloc_zeros::<f32>(DIM));
        queued!(kernels.position.cast_bf16_f32(
            s,
            weights.bf16(&format!("{prefix}.attn_norm.weight"))?,
            &mut attn_norm_f32,
            &mut status
        ));
        queued!(kernels.position.cast_bf16_f32(
            s,
            weights.bf16(&format!("{prefix}.ffn_norm.weight"))?,
            &mut ffn_norm_f32,
            &mut status
        ));
        queued!(kernels.dev.synchronize());
        ensure!(
            queued!(kernels.dev.dtoh_sync_copy(&status)) == [0],
            "block norm-weight initialization failed"
        );
        let attention = queued!(GpuAttentionBlock::new(
            attention_kernels,
            s,
            config,
            manifest,
            layer,
            weights.clone()
        ));
        let moe = queued!(MoeBlock::new(moe_kernels, weights.clone(), manifest, layer));
        Ok(Self {
            kernels,
            weights,
            attention,
            moe,
            workspace: None,
            attn_norm_f32,
            ffn_norm_f32,
            manifest_sha256: manifest.manifest_sha256.clone(),
            revision: manifest.model_revision.clone(),
            rank: manifest.rank,
            layer,
            engram_required: config.engram_layer_ids.contains(&layer),
            failed: false,
        })
    }
    /// Reset the shared KVstore once outside all40 blocks, then reset each block.
    pub fn reset_lane(&mut self, s: &CudaStream, key: LaneKey) -> Result<()> {
        self.kernels.stream(s)?;
        self.attention.reset_lane(s, key)?;
        self.failed = false;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        tokens: usize,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        kv_store: &mut GpuKvStore,
        collectives: &mut GpuCollectives,
        operation_ids: BlockOperationIds,
    ) -> Result<BlockOutput> {
        let dev = self.kernels.dev.clone();
        match self.execute_inner(
            s,
            key,
            start,
            tokens,
            h,
            incoming_pre,
            query_freq,
            compressed_freq,
            kv_store,
            collectives,
            operation_ids,
        ) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.failed = true;
                let _ = dev.synchronize();
                Err(e)
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        tokens: usize,
        h: &CudaSlice<bf16>,
        incoming_pre: &CudaSlice<f32>,
        query_freq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        kv_store: &mut GpuKvStore,
        collectives: &mut GpuCollectives,
        operation_ids: BlockOperationIds,
    ) -> Result<BlockOutput> {
        self.kernels.stream(s)?;
        operation_ids.validate()?;
        ensure!(!self.failed, "failed block requires fresh lane reset");
        self.attention.plan(key, start, tokens)?;
        ensure!(
            h.len() == tokens * COPIES * DIM
                && incoming_pre.len() == tokens * COPIES
                && Arc::ptr_eq(&h.device(), &self.kernels.dev)
                && Arc::ptr_eq(&incoming_pre.device(), &self.kernels.dev),
            "block residual/premix shape/context"
        );
        ensure!(
            collectives.rank() == self.rank && collectives.compute_stream().stream == s.stream,
            "block collective rank/stream mismatch"
        );
        let k = self.kernels.clone();
        let prefix = format!("layers.{}", self.layer);
        // Fence every fallible queued expression BEFORE local producers can drop.
        // This catches nested tensor lookups as well as CUDA launch failures.
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();return Err(e.into());}}};}
        if self.workspace.as_ref().is_none_or(|w| w.rows() != tokens) {
            self.workspace = Some(queued!(self.moe.workspace(tokens)));
        }
        let mut status = queued!(k.dev.alloc_zeros::<u32>(1));
        queued!(k.dev.memset_zeros(&mut status));
        let mut attn_mixes = queued!(k.dev.alloc_zeros::<f32>(tokens * 24));
        let mut attn_pre = queued!(k.dev.alloc_zeros::<f32>(tokens * 4));
        let mut attn_post = queued!(k.dev.alloc_zeros::<f32>(tokens * 4));
        let mut attn_comb = queued!(k.dev.alloc_zeros::<f32>(tokens * 16));
        // Source ordering: derive the NEXT premix from the uncollapsed incoming HC
        // residual, while THIS attention input consumes the previous FFN's premix.
        queued!(k.hc.project(
            s,
            h,
            self.weights.f32(&format!("{prefix}.hc_attn_fn"))?,
            &mut attn_mixes,
            &mut status,
            tokens,
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
            tokens
        ));
        let mut collapsed_attention = queued!(k.dev.alloc_zeros::<bf16>(tokens * DIM));
        queued!(k.hc.collapse(
            s,
            h,
            incoming_pre,
            &mut collapsed_attention,
            &mut status,
            tokens,
            DIM
        ));
        let normalized_attention = queued!(k.numerics.rms_norm_1017(
            s,
            &collapsed_attention,
            &self.attn_norm_f32,
            tokens,
            DIM
        ));
        let mut attention = match self.attention.enqueue(
            s,
            key,
            start,
            tokens,
            unsafe { normalized_attention.device_values_unchecked() },
            query_freq,
            compressed_freq,
            kv_store,
        ) {
            Ok(value) => value,
            Err(error) => {
                let _ = k.dev.synchronize();
                return Err(error);
            }
        };
        queued!(attention.enqueue_reduction(s, collectives, operation_ids.attention));
        let mut after_attention = queued!(k.dev.alloc_zeros::<bf16>(tokens * COPIES * DIM));
        queued!(k.hc.expand(
            s,
            unsafe { attention.device_output_unchecked()? },
            h,
            &attn_post,
            &attn_comb,
            &mut after_attention,
            &mut status,
            tokens,
            DIM
        ));
        let mut ffn_mixes = queued!(k.dev.alloc_zeros::<f32>(tokens * 24));
        let mut next_pre = queued!(k.dev.alloc_zeros::<f32>(tokens * 4));
        let mut ffn_post = queued!(k.dev.alloc_zeros::<f32>(tokens * 4));
        let mut ffn_comb = queued!(k.dev.alloc_zeros::<f32>(tokens * 16));
        queued!(k.hc.project(
            s,
            &after_attention,
            self.weights.f32(&format!("{prefix}.hc_ffn_fn"))?,
            &mut ffn_mixes,
            &mut status,
            tokens,
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
            tokens
        ));
        let mut collapsed_ffn = queued!(k.dev.alloc_zeros::<bf16>(tokens * DIM));
        // Delayed premix: attention's coefficients collapse FFN input; freshly
        // computed FFN coefficients are returned for the NEXT layer.
        queued!(k.hc.collapse(
            s,
            &after_attention,
            &attn_pre,
            &mut collapsed_ffn,
            &mut status,
            tokens,
            DIM
        ));
        let normalized_ffn =
            queued!(k
                .numerics
                .rms_norm_1017(s, &collapsed_ffn, &self.ffn_norm_f32, tokens, DIM));
        let prepared = queued!(self.moe.prepare(
            s,
            unsafe { normalized_ffn.device_values_unchecked() },
            tokens
        ));
        let routed = match self.moe.enqueue_routed(
            &prepared,
            self.workspace.as_mut().expect("workspace allocated above"),
        ) {
            Ok(value) => value,
            Err(error) => {
                let _ = k.dev.synchronize();
                return Err(error);
            }
        };
        let mut reduced = queued!(k.dev.alloc_zeros::<f32>(tokens * DIM));
        let source = unsafe { routed.device_values_unchecked() };
        let bytes = queued!(source
            .len()
            .checked_mul(4)
            .context("routed copy byte overflow"));
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(*reduced.device_ptr(), *source.device_ptr(), bytes, s.stream)
        };
        queued!(if rc == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(anyhow::anyhow!("routed partial DtoD copy failed {rc:?}"))
        });
        let kind = if start == 0 {
            RowKind::ContiguousPrefillRows
        } else {
            RowKind::IndependentDecodeLanes
        };
        queued!(collectives.enqueue_reduce(
            operation_ids.routed_experts,
            RoutedLayout::new(tokens, kind)?,
            &mut reduced
        ));
        let mut ffn_output = queued!(k.dev.alloc_zeros::<bf16>(tokens * DIM));
        let mut combination_status = queued!(k.dev.alloc_zeros::<u32>(1));
        let combined = match self.moe.finish_checked(
            &prepared,
            &reduced,
            collectives,
            &mut ffn_output,
            &mut combination_status,
        ) {
            Ok(value) => value,
            Err(error) => {
                let _ = k.dev.synchronize();
                return Err(error);
            }
        };
        let mut final_h = queued!(k.dev.alloc_zeros::<bf16>(tokens * COPIES * DIM));
        queued!(k.hc.expand(
            s,
            unsafe { combined.device_values_unchecked() },
            &after_attention,
            &ffn_post,
            &ffn_comb,
            &mut final_h,
            &mut status,
            tokens,
            DIM
        ));
        // A correctness-first fence retains every producer/borrow until all device
        // users and collective epochs complete. No per-head tensors are read back.
        queued!(collectives.synchronize());
        queued!(normalized_attention.check());
        queued!(normalized_ffn.check());
        queued!(attention.check(collectives));
        queued!(routed.check());
        queued!(combined.check());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "HC/block numeric status failed"
        );
        Ok(BlockOutput {
            h: final_h,
            next_pre,
            identity: BlockOutputIdentity {
                revision: self.revision.clone(),
                manifest_sha256: self.manifest_sha256.clone(),
                rank: self.rank,
                layer: self.layer,
                lane: key,
                start,
                tokens,
                operation_ids,
                incoming_pre_used_for_attention: true,
                attention_pre_used_for_ffn: true,
                returned_pre_is_FFN_pre: true,
                engram_before_block_required: self.engram_required,
                maximum_native_context: MAX_NATIVE_CONTEXT,
                all_checks_passed: true,
                whole_model_qualified: false,
            },
        })
    }
}
impl Drop for GpuBlock {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collective_operations_cannot_alias() {
        assert!(BlockOperationIds {
            attention: 4,
            routed_experts: 4
        }
        .validate()
        .is_err());
        BlockOperationIds {
            attention: 4,
            routed_experts: 5,
        }
        .validate()
        .unwrap();
    }
    #[test]
    fn delayed_premix_oracle_rejects_immediate_self_premix() {
        let dim = 8;
        let h = [1f32, 10., 100., 1000.]
            .into_iter()
            .flat_map(|x| vec![bf16::from_f32(x); dim])
            .collect::<Vec<_>>();
        let incoming = [1., 0., 0., 0.];
        let attention_pre = [0., 1., 0., 0.];
        let next_ffn_pre = [0., 0., 1., 0.];
        let attention_input = super::super::hc::collapse(&h, &incoming, 1, dim).unwrap();
        let wrong_attention = super::super::hc::collapse(&h, &attention_pre, 1, dim).unwrap();
        assert_eq!(attention_input, vec![bf16::ONE; dim]);
        assert_ne!(attention_input, wrong_attention);
        let identity = (0..16)
            .map(|i| if i / 4 == i % 4 { 1. } else { 0. })
            .collect::<Vec<_>>();
        let post = [1., 0., 0., 0.];
        let attention_value = vec![bf16::from_f32(2.); dim];
        let after =
            super::super::hc::expand(&attention_value, &h, &post, &identity, 1, dim).unwrap();
        let ffn_input = super::super::hc::collapse(&after, &attention_pre, 1, dim).unwrap();
        let wrong_ffn = super::super::hc::collapse(&after, &next_ffn_pre, 1, dim).unwrap();
        assert_eq!(ffn_input, vec![bf16::from_f32(10.); dim]);
        assert_ne!(ffn_input, wrong_ffn);
        let next_input = super::super::hc::collapse(&after, &next_ffn_pre, 1, dim).unwrap();
        assert_eq!(next_input, vec![bf16::from_f32(100.); dim]);
    }
}
