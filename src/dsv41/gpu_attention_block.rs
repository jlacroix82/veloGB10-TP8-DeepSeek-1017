//! Owned, bounded C1 official1017 attention composition.
//!
//! The exact short-context control is <=512 native tokens. Every compressed key
//! fits index_topk512, and all candidate blocks fit2048; index scoring cannot
//! change the selected set in this domain. Longer contexts fail closed.
//! Input is the block's already-normalized BF16 attention input (mHC/Engram
//! composition is external). The source WO-B schedule rounds each rank's Linear
//! result to BF16 BEFORE widening for its F32 allreduce.
use super::{
    attention_state::{self, LayerSharing, SharingPlan},
    cache_quant::CacheFormat,
    collectives::{RoutedLayout, RowKind},
    compression::Ratio,
    config::Dsv41Config,
    gpu_attention::AttentionKernels,
    gpu_cache_quant::{CacheQuantGpu, GpuCacheQuant},
    gpu_collectives::GpuCollectives,
    gpu_compression::{CompressionKernels, GpuCompressor, NormalizedCompressionGpu},
    gpu_dense::{DenseOutput, DenseShape, DequantCast, GpuDense},
    gpu_gemm::{GemmOutput, GemmShape, GpuGemm},
    gpu_numerics::{DeviceOutput, GpuNumerics, QuantizedGpu},
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_weights::{LoadScope, LoadedWeights},
    position::RopeProfile,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

/// Every temporary queued producer survives normal completion and synchronizes
/// before freeing on an early error. This is lifetime safety, not a GPU readback.
struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: T,
}
impl<T> Queued<T> {
    fn new(dev: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: dev.clone(),
            value,
        }
    }
}
impl<T> std::ops::Deref for Queued<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl<T> std::ops::DerefMut for Queued<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
impl<T> Drop for Queued<T> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}

pub const MAX_NATIVE_CONTEXT: usize = 512;
pub const MAX_PREFILL: usize = 128;
pub const HIDDEN: usize = 5120;
pub const LOCAL_HEADS: usize = 8;
pub const HEAD_DIM: usize = 512;
pub const Q_RANK: usize = 1280;
pub const O_RANK: usize = 1024;
/// Static synthetic allocation policy for one pending C1 step, excluding loaded
/// weights, shared KV store and CUDA context/module allocations. Not measured.
pub const STEP_ARRAY_BOUND_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct AttentionStep {
    pub lane: LaneKey,
    pub layer: usize,
    pub start: usize,
    pub tokens: usize,
    pub ratio: usize,
    pub compressed_owner: Option<usize>,
    pub owns_compressor: bool,
    pub compressed_rows: usize,
    pub new_compressed_rows: usize,
    pub compressed_positions: Vec<usize>,
    pub window_rows: usize,
    pub combined_rows: usize,
    pub topk: usize,
}
impl AttentionStep {
    pub fn new(sharing: &LayerSharing, lane: LaneKey, start: usize, tokens: usize) -> Result<Self> {
        ensure!(
            lane.generation > 0 && lane.slot < 16,
            "invalid C1 attention lane"
        );
        ensure!(
            (1..=MAX_PREFILL).contains(&tokens) && (start == 0 || tokens == 1),
            "initial prefill1..128 then one-token decode required"
        );
        let end = start
            .checked_add(tokens)
            .context("attention position overflow")?;
        ensure!(
            end <= MAX_NATIVE_CONTEXT,
            "attention block exact full-selection control is bounded to512 native tokens"
        );
        ensure!(
            matches!(sharing.ratio, 0 | 1 | 2) && sharing.layer < 40,
            "unsupported attention layer/ratio"
        );
        let comp = if sharing.ratio == 0 {
            0
        } else {
            end / sharing.ratio
        };
        let progress = if sharing.ratio == 0 {
            None
        } else {
            Some(attention_state::compression_progress(
                start,
                tokens,
                sharing.ratio,
            )?)
        };
        let window = attention_state::window_plan(128, start, tokens)?;
        let topk = window.query_indices[0].len() + comp;
        ensure!(
            comp <= 512 && comp.div_ceil(8) <= 2048 && topk <= 640,
            "bounded full-selection proof failed"
        );
        Ok(Self {
            lane,
            layer: sharing.layer,
            start,
            tokens,
            ratio: sharing.ratio,
            compressed_owner: sharing.kv_source,
            owns_compressor: sharing.owns_kv_and_index_keys,
            compressed_rows: comp,
            new_compressed_rows: progress.as_ref().map_or(0, |x| x.append_rows),
            compressed_positions: progress.map_or_else(Vec::new, |x| x.new_latent_rope_positions),
            window_rows: window.kv_rows,
            combined_rows: window.kv_rows + comp,
            topk,
        })
    }
    pub fn query_positions(&self) -> Vec<u32> {
        (self.start..self.start + self.tokens)
            .map(|x| x as u32)
            .collect()
    }
    pub fn profile(&self) -> RopeProfile {
        if self.ratio == 0 {
            RopeProfile::Plain
        } else {
            RopeProfile::Yarn
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct BlockIdentity {
    pub revision: String,
    pub manifest_sha256: String,
    pub rank: usize,
    pub layer: usize,
    pub maximum_native_context: usize,
    pub step_array_bound_bytes: usize,
    pub skips_index_scoring_exactly_in_bounded_domain: bool,
    pub source_WO_B_local_output: String,
    pub source_WO_B_network: String,
    pub whole_model_qualified: bool,
    pub woa_reconstruction_policy: String,
}
#[derive(Debug, Serialize)]
pub struct DiagnosticTensor {
    pub name: String,
    pub dtype: String,
    pub elements: usize,
    pub native_little_endian_sha256: String,
    /// Private offline oracle payload only; ordinary JSON identity reports omit bytes.
    #[serde(skip)]
    pub bytes: Vec<u8>,
}
impl DiagnosticTensor {
    fn new(name: impl Into<String>, dtype: &str, elements: usize, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            elements,
            native_little_endian_sha256: format!("{:x}", Sha256::digest(&bytes)),
            bytes,
        }
    }
    fn bf16(name: impl Into<String>, values: Vec<bf16>) -> Self {
        Self::new(
            name,
            "BF16",
            values.len(),
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        )
    }
    fn f32(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self::new(
            name,
            "F32",
            values.len(),
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        )
    }
}
pub struct AttentionBlockKernels {
    dev: Arc<CudaDevice>,
    stream_id: usize,
    boot: Uuid,
    gemm: GpuGemm,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
    attention: AttentionKernels,
    cache: GpuCacheQuant,
    compression: Arc<CompressionKernels>,
}
impl AttentionBlockKernels {
    pub fn component_identities(&self) -> serde_json::Value {
        let sha = |s: &str| format!("{:x}", Sha256::digest(s.as_bytes()));
        serde_json::json!({
            "GEMM":{"source_id":format!("{:016x}",self.gemm.source_id()),"ptx_sha256":self.gemm.ptx_sha256()},
            "dense":{"source_id":format!("{:016x}",self.dense.source_id()),"ptx_sha256":self.dense.ptx_sha256()},
            "numerics":{"source_id":format!("{:016x}",self.numerics.source_id()),"ptx_sha256":self.numerics.ptx_sha256()},
            "compression":{"source_id":format!("{:016x}",self.compression.source_id()),"ptx_sha256":self.compression.ptx_sha256()},
            "cache_QAT":{"source_id":format!("{:016x}",self.cache.source_id()),"ptx_sha256":self.cache.ptx_sha256()},
            "attention":{"source_id":env!("DSV41_ATTENTION_SOURCE_ID"),"ptx_sha256":sha(super::gpu_attention::PTX)},
            "position":{"source_id":env!("DSV41_POSITION_SOURCE_ID"),"ptx_sha256":sha(super::gpu_position::PTX)},
            "CPU_reference_frequency_plain_sha256":RopeProfile::Plain.table_sha(),"CPU_reference_frequency_yarn_sha256":RopeProfile::Yarn.table_sha(),
            "qualified_whole_attention_block":false
        })
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(
            !stream.stream.is_null(),
            "explicit blocking attention stream required"
        );
        let me = Self {
            dev: dev.clone(),
            stream_id: stream.stream as usize,
            boot,
            gemm: GpuGemm::load(dev, stream)?,
            dense: GpuDense::load(dev, stream)?,
            numerics: GpuNumerics::load(dev, stream)?,
            position: PositionKernels::load(dev, stream)?,
            attention: AttentionKernels::load(dev, stream)?,
            cache: GpuCacheQuant::load(dev, stream)?,
            compression: CompressionKernels::load(dev, stream, boot)?,
        };
        me.stream(stream)?;
        Ok(Arc::new(me))
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            s.stream as usize == self.stream_id && !s.stream.is_null(),
            "exact attention owner stream required"
        );
        let rc = unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "blocking attention owner stream required"
        );
        Ok(())
    }
    fn input<T>(&self, v: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &v.device()),
            "attention device/context mismatch"
        );
        Ok(())
    }
    fn copy<T: DeviceRepr>(
        &self,
        s: &CudaStream,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(src)?;
        self.input(dst)?;
        ensure!(
            src.len() == dst.len() && !src.is_empty(),
            "attention copy extent"
        );
        let bytes = src
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .context("copy overflow")?;
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(*dst.device_ptr(), *src.device_ptr(), bytes, s.stream)
        };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "attention DtoD copy failed {rc:?}"
        );
        Ok(())
    }
}
#[derive(Clone)]
struct Progress {
    key: LaneKey,
    position: usize,
    initialized: bool,
    failed: bool,
}
struct CompletionToken {
    state: Arc<Mutex<Option<Progress>>>,
    key: LaneKey,
}
impl CompletionToken {
    fn valid(&self) -> Result<()> {
        let s = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("attention progress poisoned"))?;
        ensure!(
            s.as_ref().is_some_and(|v| v.key == self.key && !v.failed),
            "stale/failed attention generation"
        );
        Ok(())
    }
    fn poison(&self) {
        if let Ok(mut s) = self.state.lock() {
            if let Some(v) = s.as_mut() {
                if v.key == self.key {
                    v.failed = true;
                }
            }
        }
    }
}
pub struct GpuAttentionBlock {
    kernels: Arc<AttentionBlockKernels>,
    weights: Arc<LoadedWeights>,
    sharing: LayerSharing,
    identity: BlockIdentity,
    state: Arc<Mutex<Option<Progress>>>,
    init_status: Queued<CudaSlice<u32>>,
    q_norm_f32: Queued<CudaSlice<f32>>,
    kv_norm_f32: Queued<CudaSlice<f32>>,
    compressor_norm_f32: Option<Queued<CudaSlice<f32>>>,
    compressor: Option<GpuCompressor>,
    // Persistent per-step buffers permit pending compression/cache owners to borrow
    // stable inputs while all temporary producer owners stay alive in the returned operation.
    projected_comp_bf16: Option<CudaSlice<bf16>>,
    projected_comp_kv_f32: Option<CudaSlice<f32>>,
    projected_comp_score_f32: Option<CudaSlice<f32>>,
}
impl GpuAttentionBlock {
    pub fn new(
        kernels: Arc<AttentionBlockKernels>,
        s: &CudaStream,
        config: &Dsv41Config,
        manifest: &RankManifest,
        layer: usize,
        weights: Arc<LoadedWeights>,
    ) -> Result<Self> {
        kernels.stream(s)?;
        config.validate()?;
        manifest.verify_identity()?;
        ensure!(
            weights.scope() == &LoadScope::Layer(layer)
                && weights.manifest_sha256() == manifest.manifest_sha256,
            "attention weights scope/manifest differs"
        );
        let sharing = SharingPlan::from_config(config)?
            .layers
            .get(layer)
            .context("attention layer missing")?
            .clone();
        let prefix = format!("layers.{layer}.attn");
        // Validate all required native projection pairs and source-cast WO-A before allocation.
        for name in ["wq_a", "wq_b", "wkv", "wo_b"] {
            weights.native_u8(&format!("{prefix}.{name}.weight"))?;
            weights.native_u8(&format!("{prefix}.{name}.scale"))?;
        }
        ensure!(
            weights.bf16(&format!("{prefix}.wo_a.weight"))?.len()
                == O_RANK * LOCAL_HEADS * HEAD_DIM
                && weights.f32(&format!("{prefix}.attn_sink"))?.len() == LOCAL_HEADS,
            "attention local head/output geometry differs"
        );
        let mut init_status = Queued::new(&kernels.dev, kernels.dev.alloc_zeros::<u32>(1)?);
        kernels.dev.memset_zeros(&mut *init_status)?;
        let mut q_norm_f32 = Queued::new(&kernels.dev, kernels.dev.alloc_zeros::<f32>(Q_RANK)?);
        let mut kv_norm_f32 = Queued::new(&kernels.dev, kernels.dev.alloc_zeros::<f32>(HEAD_DIM)?);
        kernels.position.cast_bf16_f32(
            s,
            weights.bf16(&format!("{prefix}.q_norm.weight"))?,
            &mut q_norm_f32,
            &mut init_status,
        )?;
        kernels.position.cast_bf16_f32(
            s,
            weights.bf16(&format!("{prefix}.kv_norm.weight"))?,
            &mut kv_norm_f32,
            &mut init_status,
        )?;
        let (compressor, compressor_norm_f32) = if sharing.owns_kv_and_index_keys {
            weights.bf16(&format!("{prefix}.compressor.wkv.weight"))?;
            if sharing.ratio == 2 {
                weights.bf16(&format!("{prefix}.compressor.wgate.weight"))?;
            }
            let mut w = Queued::new(&kernels.dev, kernels.dev.alloc_zeros::<f32>(HEAD_DIM)?);
            kernels.position.cast_bf16_f32(
                s,
                weights.bf16(&format!("{prefix}.compressor.norm.weight"))?,
                &mut w,
                &mut init_status,
            )?;
            (
                Some(GpuCompressor::new(
                    kernels.compression.clone(),
                    s,
                    if sharing.ratio == 1 {
                        Ratio::One
                    } else {
                        Ratio::Two
                    },
                    HEAD_DIM,
                    16,
                )?),
                Some(w),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            identity: BlockIdentity {
                revision: manifest.model_revision.clone(),
                manifest_sha256: manifest.manifest_sha256.clone(),
                rank: manifest.rank,
                layer,
                maximum_native_context: MAX_NATIVE_CONTEXT,
                step_array_bound_bytes: STEP_ARRAY_BOUND_BYTES,
                skips_index_scoring_exactly_in_bounded_domain: true,
                source_WO_B_local_output: "complete BF16 Linear then F32 promotion".into(),
                source_WO_B_network:
                    "F32 allreduce then BF16 output; no FP32 local-projection substitution".into(),
                whole_model_qualified: false,
                woa_reconstruction_policy: format!("{:?}", manifest.woa_policy),
            },
            kernels,
            weights,
            sharing,
            state: Arc::new(Mutex::new(None)),
            init_status,
            q_norm_f32,
            kv_norm_f32,
            compressor_norm_f32,
            compressor,
            projected_comp_bf16: None,
            projected_comp_kv_f32: None,
            projected_comp_score_f32: None,
        })
    }
    pub fn identity(&self) -> &BlockIdentity {
        &self.identity
    }
    pub fn reset_lane(&mut self, s: &CudaStream, lane: LaneKey) -> Result<()> {
        self.kernels.stream(s)?;
        ensure!(
            lane.boot == self.kernels.boot && lane.generation > 0 && lane.slot < 16,
            "attention lane identity"
        );
        {
            let state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("attention progress poisoned"))?;
            ensure!(
                state
                    .as_ref()
                    .is_none_or(|v| v.key.slot != lane.slot || lane.generation > v.key.generation),
                "attention reset must advance slot generation"
            );
        }
        if let Some(c) = &mut self.compressor {
            c.reset_lane(s, lane)?;
        }
        *self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("attention progress poisoned"))? = Some(Progress {
            key: lane,
            position: 0,
            initialized: false,
            failed: false,
        });
        Ok(())
    }
    pub fn plan(&self, lane: LaneKey, start: usize, tokens: usize) -> Result<AttentionStep> {
        let s = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("attention progress poisoned"))?;
        let p = s.as_ref().context("attention lane unbound")?;
        ensure!(
            p.key == lane && !p.failed && start == p.position && (!p.initialized || start > 0),
            "stale/failed/noncontiguous attention step"
        );
        AttentionStep::new(&self.sharing, lane, start, tokens)
    }
}
impl Drop for GpuAttentionBlock {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

enum StageOwner {
    Quant(Queued<QuantizedGpu>),
    GemmBf16(Queued<GemmOutput<bf16>>),
    DenseBf16(Queued<DenseOutput<bf16>>),
    DenseF32(Queued<DenseOutput<f32>>),
    Norm(Queued<DeviceOutput<bf16>>),
    Cache(Queued<CacheQuantGpu>),
    Bf16(Queued<CudaSlice<bf16>>),
    F32(Queued<CudaSlice<f32>>),
}
impl StageOwner {
    fn check(&self) -> Result<()> {
        match self {
            Self::Quant(v) => v.check(),
            Self::GemmBf16(v) => v.check(),
            Self::DenseBf16(v) => v.check(),
            Self::DenseF32(v) => v.check(),
            Self::Norm(v) => v.check(),
            Self::Cache(v) => v.native().check(),
            Self::Bf16(_) | Self::F32(_) => Ok(()),
        }
    }
}
fn replace_buffer<T: DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &Arc<CudaDevice>,
    slot: &mut Option<CudaSlice<T>>,
    len: usize,
) -> Result<()> {
    if slot.as_ref().is_none_or(|x| x.len() != len) {
        *slot = Some(dev.alloc_zeros::<T>(len)?);
    }
    Ok(())
}
impl GpuAttentionBlock {
    /// Enqueues every local attention stage. KVstore mutates only this lane/layer's
    /// checked cache ownership; all temporary source owners survive in the pending
    /// result. Complete the mandatory external F32 WO-B reduction before publishing.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue<'a>(
        &'a mut self,
        s: &CudaStream,
        lane: LaneKey,
        start: usize,
        tokens: usize,
        input: &'a CudaSlice<bf16>,
        query_freq: &'a GpuFrequencyRows,
        compressed_freq: Option<&'a GpuFrequencyRows>,
        kv_store: &mut super::gpu_kv_store::GpuKvStore,
    ) -> Result<PendingAttention<'a>> {
        let token = CompletionToken {
            state: self.state.clone(),
            key: lane,
        };
        let dev = self.kernels.dev.clone();
        match self.enqueue_inner(
            s,
            lane,
            start,
            tokens,
            input,
            query_freq,
            compressed_freq,
            kv_store,
        ) {
            Ok(pending) => Ok(pending),
            Err(error) => {
                token.poison();
                let _ = dev.synchronize();
                Err(error)
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_inner<'a>(
        &'a mut self,
        s: &CudaStream,
        lane: LaneKey,
        start: usize,
        tokens: usize,
        input: &'a CudaSlice<bf16>,
        query_freq: &'a GpuFrequencyRows,
        compressed_freq: Option<&'a GpuFrequencyRows>,
        kv_store: &mut super::gpu_kv_store::GpuKvStore,
    ) -> Result<PendingAttention<'a>> {
        self.kernels.stream(s)?;
        self.kernels.input(input)?;
        let plan = self.plan(lane, start, tokens)?;
        ensure!(
            input.len() == tokens * HIDDEN,
            "attention BF16 input extent"
        );
        ensure!(
            query_freq.profile() == plan.profile()
                && query_freq.positions() == plan.query_positions()
                && query_freq.complete_table_verified(),
            "complete verified query frequency table and exact positions required"
        );
        if plan.owns_compressor && plan.new_compressed_rows > 0 {
            let f =
                compressed_freq.context("completed compressor requires selected frequency rows")?;
            ensure!(
                f.profile() == plan.profile()
                    && f.positions()
                        == plan
                            .compressed_positions
                            .iter()
                            .map(|x| *x as u32)
                            .collect::<Vec<_>>()
                    && f.complete_table_verified(),
                "compressed frequency rows/profile/complete source table differ"
            );
        } else {
            ensure!(
                compressed_freq.is_none(),
                "unused compressed frequency rows not accepted"
            );
        }
        let k = self.kernels.clone();
        let weights = self.weights.clone();
        let prefix = format!("layers.{}.attn", plan.layer);
        let token = CompletionToken {
            state: self.state.clone(),
            key: lane,
        };
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        k.dev.memset_zeros(&mut *status)?;
        let mut stages = Vec::new();
        let mut stage_labels = Vec::new();
        let xq = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, input, tokens, HIDDEN)?,
        );
        let qa = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                xq.device_values_unchecked(),
                xq.device_scales_unchecked(),
                weights.native_u8(&format!("{prefix}.wq_a.weight"))?,
                weights.native_u8(&format!("{prefix}.wq_a.scale"))?,
                GemmShape {
                    rows: tokens,
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
                tokens,
                Q_RANK,
            )?
        });
        let qrq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, qr.device_values_unchecked(), tokens, Q_RANK)?
        });
        let qb = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                qrq.device_values_unchecked(),
                qrq.device_scales_unchecked(),
                weights.native_u8(&format!("{prefix}.wq_b.weight"))?,
                weights.native_u8(&format!("{prefix}.wq_b.scale"))?,
                GemmShape {
                    rows: tokens,
                    outputs: LOCAL_HEADS * HEAD_DIM,
                    inner: Q_RANK,
                },
            )?
        });
        let mut q = Queued::new(
            &k.dev,
            k.dev.alloc_zeros::<bf16>(tokens * LOCAL_HEADS * HEAD_DIM)?,
        );
        unsafe {
            k.copy(s, qb.device_values_unchecked(), &mut q)?;
        }
        k.position.rotate(
            s,
            &mut q,
            query_freq,
            plan.profile(),
            &plan.query_positions(),
            &mut status,
            1,
            tokens,
            LOCAL_HEADS,
            HEAD_DIM,
            false,
        )?;
        let kv = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                xq.device_values_unchecked(),
                xq.device_scales_unchecked(),
                weights.native_u8(&format!("{prefix}.wkv.weight"))?,
                weights.native_u8(&format!("{prefix}.wkv.scale"))?,
                GemmShape {
                    rows: tokens,
                    outputs: HEAD_DIM,
                    inner: HIDDEN,
                },
            )?
        });
        let kv_norm = Queued::new(&k.dev, unsafe {
            k.numerics.rms_norm_1017(
                s,
                kv.device_values_unchecked(),
                &self.kv_norm_f32,
                tokens,
                HEAD_DIM,
            )?
        });
        let mut window = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(tokens * HEAD_DIM)?);
        unsafe {
            k.copy(s, kv_norm.device_values_unchecked(), &mut window)?;
        }
        k.position.rotate(
            s,
            &mut window,
            query_freq,
            plan.profile(),
            &plan.query_positions(),
            &mut status,
            1,
            tokens,
            1,
            HEAD_DIM,
            false,
        )?;
        let win_quant = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, &window, tokens, HEAD_DIM)?,
        );
        let win_qat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                win_quant.device_values_unchecked(),
                win_quant.device_scales_unchecked(),
                tokens,
                HEAD_DIM,
                DequantCast::OfficialBf16Round,
            )?
        });
        // Own all upstream producers before compression's borrowed stable buffers.
        stages.extend([
            StageOwner::Quant(xq),
            StageOwner::GemmBf16(qa),
            StageOwner::Norm(qr),
            StageOwner::Quant(qrq),
            StageOwner::GemmBf16(qb),
            StageOwner::GemmBf16(kv),
            StageOwner::Norm(kv_norm),
            StageOwner::Bf16(window),
            StageOwner::Quant(win_quant),
        ]);
        stage_labels.extend([
            "input_FP8_Q32",
            "query_A_BF16",
            "query_RMS_BF16",
            "query_RMS_FP8_Q32",
            "query_B_BF16",
            "window_projection_BF16",
            "window_RMS_BF16",
            "window_rotated_BF16",
            "window_FP8_Q32",
        ]);
        let mut compressed_normalized = None;
        if plan.owns_compressor {
            if plan.ratio == 1 {
                replace_buffer(&k.dev, &mut self.projected_comp_bf16, tokens * HEAD_DIM)?;
                let projection = Queued::new(
                    &k.dev,
                    k.dense.bf16_linear_bf16(
                        s,
                        input,
                        weights.bf16(&format!("{prefix}.compressor.wkv.weight"))?,
                        DenseShape {
                            rows: tokens,
                            outputs: HEAD_DIM,
                            inner: HIDDEN,
                        },
                    )?,
                );
                unsafe {
                    k.copy(
                        s,
                        projection.device_values_unchecked(),
                        self.projected_comp_bf16.as_mut().unwrap(),
                    )?;
                }
                stages.push(StageOwner::DenseBf16(projection));
                stage_labels.push("compressor_projection_BF16");
                let prepared = self
                    .compressor
                    .as_mut()
                    .context("missing ratio1 compressor")?
                    .push_ratio1(
                        s,
                        lane,
                        start,
                        tokens,
                        self.projected_comp_bf16.as_ref().unwrap(),
                    )?;
                compressed_normalized = Some(
                    prepared.normalize(
                        s,
                        &k.numerics,
                        self.compressor_norm_f32
                            .as_ref()
                            .context("missing compressor norm")?,
                    )?,
                );
            } else {
                replace_buffer(&k.dev, &mut self.projected_comp_kv_f32, tokens * HEAD_DIM)?;
                replace_buffer(
                    &k.dev,
                    &mut self.projected_comp_score_f32,
                    tokens * HEAD_DIM,
                )?;
                let mut xf = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(tokens * HIDDEN)?);
                k.position.cast_bf16_f32(s, input, &mut xf, &mut status)?;
                let projection = Queued::new(
                    &k.dev,
                    k.dense.f32_bf16_linear_f32(
                        s,
                        &xf,
                        weights.bf16(&format!("{prefix}.compressor.wkv.weight"))?,
                        DenseShape {
                            rows: tokens,
                            outputs: HEAD_DIM,
                            inner: HIDDEN,
                        },
                    )?,
                );
                let gate = Queued::new(
                    &k.dev,
                    k.dense.f32_bf16_linear_f32(
                        s,
                        &xf,
                        weights.bf16(&format!("{prefix}.compressor.wgate.weight"))?,
                        DenseShape {
                            rows: tokens,
                            outputs: HEAD_DIM,
                            inner: HIDDEN,
                        },
                    )?,
                );
                unsafe {
                    k.copy(
                        s,
                        projection.device_values_unchecked(),
                        self.projected_comp_kv_f32.as_mut().unwrap(),
                    )?;
                    k.copy(
                        s,
                        gate.device_values_unchecked(),
                        self.projected_comp_score_f32.as_mut().unwrap(),
                    )?;
                }
                stages.extend([
                    StageOwner::F32(xf),
                    StageOwner::DenseF32(projection),
                    StageOwner::DenseF32(gate),
                ]);
                stage_labels.extend([
                    "compressor_input_F32",
                    "compressor_projection_F32",
                    "compressor_gate_F32",
                ]);
                let prepared = self
                    .compressor
                    .as_mut()
                    .context("missing ratio2 compressor")?
                    .push_ratio2(
                        s,
                        lane,
                        start,
                        tokens,
                        self.projected_comp_kv_f32.as_ref().unwrap(),
                        self.projected_comp_score_f32.as_ref().unwrap(),
                    )?;
                compressed_normalized = Some(
                    prepared.normalize(
                        s,
                        &k.numerics,
                        self.compressor_norm_f32
                            .as_ref()
                            .context("missing compressor norm")?,
                    )?,
                );
            }
        }
        // The normalized latent remains immutable. Under this exact full-selection
        // bound WK/index scores are mathematically irrelevant, but outer KV rotation
        // still receives a separate copy as in the general architecture.
        let mut compressed_rotated = None;
        let mut compressed_qat = None;
        if let Some(latent) = compressed_normalized.as_ref().and_then(|n| n.latent()) {
            ensure!(
                latent.plan().positions == plan.compressed_positions,
                "compressor/store position mismatch"
            );
            let mut rotated =
                Queued::new(&k.dev, unsafe { latent.enqueue_rotary_copy_unchecked(s)? });
            let freq = compressed_freq.context("latent lacks frequency rows")?;
            k.position.rotate(
                s,
                &mut rotated,
                freq,
                plan.profile(),
                &plan
                    .compressed_positions
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                &mut status,
                1,
                plan.new_compressed_rows,
                1,
                HEAD_DIM,
                false,
            )?;
            compressed_qat = Some(Queued::new(
                &k.dev,
                k.cache.quantize(
                    s,
                    &rotated,
                    plan.new_compressed_rows,
                    HEAD_DIM,
                    CacheFormat::CompressedKvE4m3Group16,
                )?,
            ));
            compressed_rotated = Some(rotated);
        }
        let snapshot = unsafe {
            kv_store.prepare_async(
                s,
                plan.layer,
                lane,
                start,
                tokens,
                win_qat.device_values_unchecked(),
                compressed_qat.as_ref().map(|v| v.device_qat_unchecked()),
            )?
        };
        ensure!(
            snapshot.kv_rows() == plan.combined_rows
                && snapshot.topk() == plan.topk
                && snapshot.tokens() == tokens
                && snapshot.layer() == plan.layer
                && snapshot.key() == lane
                && snapshot.start() == start
                && snapshot.compressed_owner() == plan.compressed_owner,
            "KV snapshot identity/geometry mismatch"
        );
        let mut output = Queued::new(
            &k.dev,
            k.dev.alloc_zeros::<bf16>(tokens * LOCAL_HEADS * HEAD_DIM)?,
        );
        unsafe {
            k.attention.sparse(
                s,
                &q,
                snapshot.device_kv_unchecked(),
                weights.f32(&format!("{prefix}.attn_sink"))?,
                snapshot.device_indices_unchecked(),
                &mut output,
                &mut status,
                1,
                tokens,
                LOCAL_HEADS,
                plan.combined_rows,
                plan.topk,
            )?;
        }
        k.position.rotate(
            s,
            &mut output,
            query_freq,
            plan.profile(),
            &plan.query_positions(),
            &mut status,
            1,
            tokens,
            LOCAL_HEADS,
            HEAD_DIM,
            true,
        )?;
        let oq = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, &output, tokens, LOCAL_HEADS * HEAD_DIM)?,
        );
        let oqat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                oq.device_values_unchecked(),
                oq.device_scales_unchecked(),
                tokens,
                LOCAL_HEADS * HEAD_DIM,
                DequantCast::OfficialBf16Round,
            )?
        });
        let oa = Queued::new(&k.dev, unsafe {
            k.dense.bf16_linear_bf16(
                s,
                oqat.device_values_unchecked(),
                weights.bf16(&format!("{prefix}.wo_a.weight"))?,
                DenseShape {
                    rows: tokens,
                    outputs: O_RANK,
                    inner: LOCAL_HEADS * HEAD_DIM,
                },
            )?
        });
        let oaq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, oa.device_values_unchecked(), tokens, O_RANK)?
        });
        // Source RowParallelLinear: complete local BF16 output BEFORE F32 allreduce.
        let ob = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                oaq.device_values_unchecked(),
                oaq.device_scales_unchecked(),
                weights.native_u8(&format!("{prefix}.wo_b.weight"))?,
                weights.native_u8(&format!("{prefix}.wo_b.scale"))?,
                GemmShape {
                    rows: tokens,
                    outputs: HIDDEN,
                    inner: O_RANK,
                },
            )?
        });
        let mut reduction = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(tokens * HIDDEN)?);
        unsafe {
            k.position.cast_bf16_f32(
                s,
                ob.device_values_unchecked(),
                &mut reduction,
                &mut status,
            )?;
        }
        stages.extend([
            StageOwner::Bf16(q),
            StageOwner::DenseBf16(win_qat),
            StageOwner::Bf16(output),
            StageOwner::Quant(oq),
            StageOwner::DenseBf16(oqat),
            StageOwner::DenseBf16(oa),
            StageOwner::Quant(oaq),
            StageOwner::GemmBf16(ob),
        ]);
        stage_labels.extend([
            "query_rotated_BF16",
            "window_QAT_BF16",
            "attention_inverse_rotated_BF16",
            "attention_FP8_Q32",
            "attention_QAT_BF16",
            "WO_A_BF16",
            "WO_A_FP8_Q32",
            "WO_B_local_BF16",
        ]);
        if let Some(v) = compressed_rotated {
            stages.push(StageOwner::Bf16(v));
            stage_labels.push("compressed_rotated_BF16");
        }
        if let Some(v) = compressed_qat {
            stages.push(StageOwner::Cache(v));
            stage_labels.push("compressed_FP4_group16_QAT");
        }
        let final_output = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(tokens * HIDDEN)?);
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("attention progress poisoned"))?;
            let p = state.as_mut().context("attention lane missing")?;
            p.position = start + tokens;
            p.initialized = true;
        }
        Ok(PendingAttention {
            kernels: k,
            weights,
            identity: self.identity.clone(),
            plan,
            token,
            stages,
            stage_labels,
            compressed_normalized,
            snapshot,
            status,
            reduction,
            final_output,
            reduction_enqueued: false,
            collective_identity: None,
            checked: false,
            init_status: &self.init_status,
            _input: input,
            _query_freq: query_freq,
            _compressed_freq: compressed_freq,
            _exclusive_block: PhantomData,
        })
    }
}
pub struct PendingAttention<'a> {
    kernels: Arc<AttentionBlockKernels>,
    weights: Arc<LoadedWeights>,
    pub identity: BlockIdentity,
    pub plan: AttentionStep,
    token: CompletionToken,
    stages: Vec<StageOwner>,
    stage_labels: Vec<&'static str>,
    compressed_normalized: Option<NormalizedCompressionGpu<'a>>,
    snapshot: super::gpu_kv_store::OwnedKvSnapshot,
    status: Queued<CudaSlice<u32>>,
    reduction: Queued<CudaSlice<f32>>,
    final_output: Queued<CudaSlice<bf16>>,
    reduction_enqueued: bool,
    collective_identity: Option<[u8; 32]>,
    checked: bool,
    init_status: &'a CudaSlice<u32>,
    _input: &'a CudaSlice<bf16>,
    _query_freq: &'a GpuFrequencyRows,
    _compressed_freq: Option<&'a GpuFrequencyRows>,
    _exclusive_block: PhantomData<&'a mut GpuAttentionBlock>,
}
impl PendingAttention<'_> {
    /// Uses the existing fixed-program epoch/session. This does not create a link,
    /// reset epochs, choose an operation ID or replace the caller's recovery guard.
    pub fn enqueue_reduction(
        &mut self,
        s: &CudaStream,
        collectives: &mut GpuCollectives,
        operation_id: u64,
    ) -> Result<()> {
        self.token.valid()?;
        self.kernels.stream(s)?;
        ensure!(
            !self.reduction_enqueued
                && collectives.rank() == self.identity.rank
                && collectives.compute_stream().stream == s.stream,
            "wrong/repeated WO-B reduction session"
        );
        let kind = if self.plan.start == 0 {
            RowKind::ContiguousPrefillRows
        } else {
            RowKind::IndependentDecodeLanes
        };
        let layout = RoutedLayout::new(self.plan.tokens, kind)?;
        let completion_identity = collectives.completion_identity();
        let result = (|| {
            collectives.enqueue_reduce(operation_id, layout, &mut self.reduction)?;
            self.kernels.position.cast_f32_bf16(
                s,
                &self.reduction,
                &mut self.final_output,
                &mut self.status,
            )?;
            Ok(())
        })();
        if result.is_err() {
            self.token.poison();
        } else {
            self.reduction_enqueued = true;
            self.collective_identity = Some(completion_identity);
        }
        result
    }
    /// Mandatory before safe publication. Checks whole component statuses (no
    /// per-head payload readbacks); the current diagnostic path is not a speed claim.
    pub fn check(&mut self, collectives: &mut GpuCollectives) -> Result<()> {
        let result = (|| {
            self.token.valid()?;
            ensure!(
                self.reduction_enqueued,
                "WO-B F32 allreduce was not enqueued"
            );
            ensure!(
                collectives.rank() == self.identity.rank
                    && collectives.compute_stream().stream as usize == self.kernels.stream_id
                    && self.collective_identity == Some(collectives.completion_identity()),
                "wrong collective completion session"
            );
            collectives.synchronize()?;
            ensure!(
                self.kernels.dev.dtoh_sync_copy(self.init_status)? == [0],
                "attention norm initialization failed"
            );
            for stage in &self.stages {
                stage.check()?;
            }
            ensure!(
                self.stage_labels.len() == self.stages.len(),
                "attention diagnostic stage labels differ"
            );
            if let Some(c) = &self.compressed_normalized {
                c.check()?;
            }
            self.snapshot.check()?;
            ensure!(
                self.kernels.dev.dtoh_sync_copy(&self.status)? == [0],
                "attention position/sparse/cast status failed"
            );
            self.token.valid()?;
            Ok(())
        })();
        if result.is_err() {
            self.token.poison();
        } else {
            self.checked = true;
        }
        result
    }
    pub fn download_checked(&mut self, collectives: &mut GpuCollectives) -> Result<Vec<bf16>> {
        self.check(collectives)?;
        Ok(self.kernels.dev.dtoh_sync_copy(&self.final_output)?)
    }
    /// Explicitly untimed private oracle readbacks, after successful collective
    /// and stage checks. Production consumption uses the device-only accessor.
    pub fn diagnostic_tensors(&self) -> Result<Vec<DiagnosticTensor>> {
        self.token.valid()?;
        ensure!(
            self.checked,
            "diagnostics require a checked complete attention block"
        );
        let mut result = Vec::new();
        for (label, stage) in self.stage_labels.iter().zip(&self.stages) {
            match stage {
                StageOwner::Quant(v) => {
                    let q = v.download_checked()?;
                    result.push(DiagnosticTensor::new(
                        format!("{label}.values"),
                        "F8_E4M3",
                        q.e4m3_values.len(),
                        q.e4m3_values,
                    ));
                    result.push(DiagnosticTensor::new(
                        format!("{label}.scales"),
                        "UE8M0",
                        q.ue8m0_scales.len(),
                        q.ue8m0_scales,
                    ));
                }
                StageOwner::GemmBf16(v) => {
                    result.push(DiagnosticTensor::bf16(*label, v.download_checked()?))
                }
                StageOwner::DenseBf16(v) => {
                    result.push(DiagnosticTensor::bf16(*label, v.download_checked()?))
                }
                StageOwner::DenseF32(v) => {
                    result.push(DiagnosticTensor::f32(*label, v.download_checked()?))
                }
                StageOwner::Norm(v) => {
                    result.push(DiagnosticTensor::bf16(*label, v.download_checked()?))
                }
                StageOwner::Bf16(v) => result.push(DiagnosticTensor::bf16(
                    *label,
                    self.kernels.dev.dtoh_sync_copy(v)?,
                )),
                StageOwner::F32(v) => result.push(DiagnosticTensor::f32(
                    *label,
                    self.kernels.dev.dtoh_sync_copy(v)?,
                )),
                StageOwner::Cache(v) => {
                    let c = v.download_checked()?;
                    result.push(DiagnosticTensor::new(
                        format!("{label}.packed"),
                        "FP4_E2M1x2",
                        c.packed.len(),
                        c.packed,
                    ));
                    result.push(DiagnosticTensor::new(
                        format!("{label}.scales"),
                        "F8_E4M3",
                        c.scales.len(),
                        c.scales,
                    ));
                    result.push(DiagnosticTensor::bf16(format!("{label}.QAT"), c.qat));
                }
            }
        }
        if let Some(latent) = self.compressed_normalized.as_ref().and_then(|v| v.latent()) {
            result.push(DiagnosticTensor::bf16(
                "compressor_normalized_unrotated_BF16",
                latent.download_checked()?,
            ));
        }
        result.push(DiagnosticTensor::bf16("sparse_combined_KV_BF16", unsafe {
            self.kernels
                .dev
                .dtoh_sync_copy(self.snapshot.device_kv_unchecked())?
        }));
        let ids = unsafe {
            self.kernels
                .dev
                .dtoh_sync_copy(self.snapshot.device_indices_unchecked())?
        };
        result.push(DiagnosticTensor::new(
            "sparse_causal_indices",
            "I32",
            ids.len(),
            ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        result.push(DiagnosticTensor::f32(
            "WO_B_reduced_F32",
            self.kernels.dev.dtoh_sync_copy(&self.reduction)?,
        ));
        result.push(DiagnosticTensor::bf16(
            "attention_complete_BF16",
            self.kernels.dev.dtoh_sync_copy(&self.final_output)?,
        ));
        Ok(result)
    }
    /// For a same-stream downstream mHC stage. Retain this pending owner and check
    /// all statuses before final publication; source reduction must already be queued.
    pub unsafe fn device_output_unchecked(&self) -> Result<&CudaSlice<bf16>> {
        self.token.valid()?;
        ensure!(self.reduction_enqueued, "attention output before reduction");
        Ok(&self.final_output)
    }
}
impl Drop for PendingAttention<'_> {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
        if !self.checked {
            self.token.poison();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sharing(layer: usize, ratio: usize, source: Option<usize>) -> LayerSharing {
        LayerSharing {
            layer,
            ratio,
            kv_source: source,
            index_source: source,
            owns_kv_and_index_keys: source == Some(layer),
            owns_topk: source == Some(layer),
            produces_candidates: layer == 20,
            consumes_candidates: layer > 20,
        }
    }
    #[test]
    fn bounded_full_selection_is_exact_for_every_ratio_and_boundary() {
        let lane = LaneKey {
            boot: Uuid::nil(),
            slot: 7,
            generation: 1,
        };
        for ratio in [0, 1, 2] {
            for (start, tokens) in [
                (0, 1),
                (0, 75),
                (0, 128),
                (127, 1),
                (128, 1),
                (510, 1),
                (511, 1),
            ] {
                let s = sharing(
                    if ratio == 0 { 0 } else { 20 },
                    ratio,
                    if ratio == 0 { None } else { Some(20) },
                );
                let p = AttentionStep::new(&s, lane, start, tokens).unwrap();
                assert!(p.topk <= 640 && p.compressed_rows <= 512);
                assert_eq!(p.combined_rows, p.window_rows + p.compressed_rows);
                if ratio > 0 {
                    assert_eq!(p.compressed_rows, (start + tokens) / ratio);
                    assert_eq!(
                        p.new_compressed_rows,
                        (start + tokens) / ratio - start / ratio
                    );
                }
            }
        }
        assert!(AttentionStep::new(&sharing(20, 1, Some(20)), lane, 512, 1).is_err());
        assert!(AttentionStep::new(&sharing(20, 1, Some(20)), lane, 0, 129).is_err());
        assert!(AttentionStep::new(&sharing(20, 2, Some(20)), lane, 4, 2).is_err());
    }
    #[test]
    fn consumer_ownership_positions_and_rope_profiles_do_not_alias() {
        let lane = LaneKey {
            boot: Uuid::nil(),
            slot: 0,
            generation: 1,
        };
        let owner = AttentionStep::new(&sharing(2, 2, Some(2)), lane, 0, 75).unwrap();
        assert!(owner.owns_compressor);
        assert_eq!(owner.profile(), RopeProfile::Yarn);
        assert_eq!(
            owner.compressed_positions,
            (0..74).step_by(2).collect::<Vec<_>>()
        );
        let consumer = AttentionStep::new(&sharing(3, 2, Some(2)), lane, 75, 1).unwrap();
        assert!(!consumer.owns_compressor);
        assert_eq!(consumer.compressed_owner, Some(2));
        assert_eq!(consumer.compressed_positions, [74]);
        let plain = AttentionStep::new(&sharing(0, 0, None), lane, 0, 75).unwrap();
        assert_eq!(plain.profile(), RopeProfile::Plain);
        assert!(plain.compressed_positions.is_empty());
        assert_eq!(plain.query_positions(), (0..75).collect::<Vec<_>>());
    }
    #[test]
    fn official_all40_layers_keep_every_reachable_compressed_key_at512() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let config = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let sharing = SharingPlan::from_config(&config).unwrap();
        let lane = LaneKey {
            boot: Uuid::nil(),
            slot: 0,
            generation: 1,
        };
        assert_eq!(sharing.layers.len(), 40);
        let mut owners = Vec::new();
        for layer in &sharing.layers {
            let p = AttentionStep::new(layer, lane, 511, 1).unwrap();
            if p.owns_compressor {
                owners.push(layer.layer);
            }
            if layer.layer < 2 {
                assert_eq!(p.compressed_rows, 0);
                assert_eq!(p.topk, 128);
            } else {
                assert!(p.compressed_rows <= config.index_topk);
                assert!(
                    p.compressed_rows.div_ceil(config.candidate_block_size)
                        <= config.candidate_topk_blocks
                );
                assert_eq!(p.topk, 128 + p.compressed_rows);
            }
        }
        assert_eq!(owners, [2, 8, 14, 20]);
    }
}
