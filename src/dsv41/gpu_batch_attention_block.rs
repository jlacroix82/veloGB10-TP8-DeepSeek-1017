//! PRIVATE native1017 packed attention composition, <=128 flattened rows.
//! Each dense projection is one shared matrix call; sparse attention is one
//! packed launch, with native per-member logical tiles. C1 controls stay intact.
//! The first learned path requires <=512 stored compressed keys in EVERY lane;
//! selecting all causal keys is then exact and no index-score reduction is used.
//! No HC/MoE/head or cache transaction commit occurs in this component.
use super::{
    batch_compression_plan,
    batch_plan::{BatchLayerPlan, BatchTicket},
    cache_quant::CacheFormat,
    chunk_plan::KV_SOURCES,
    collectives::RoutedLayout,
    config::Dsv41Config,
    gpu_batch_attention::PackedAttentionKernels,
    gpu_batch_compressor::{BatchCompressionKernels, GpuPackedCompressor, PackedLatentReceipt},
    gpu_batch_store::{GpuBatchStore, PackedKvSnapshot},
    gpu_cache_quant::GpuCacheQuant,
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseShape, DequantCast, GpuDense},
    gpu_gemm::{GemmGroupWorkspace, GemmShape, GpuGemm},
    gpu_numerics::GpuNumerics,
    gpu_position::{GpuFrequencyRows, PositionKernels},
    gpu_status_cut::{ProjectionStatusBackend, ProjectionStatusCut, ProjectionStatusSnapshot},
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
const QR: usize = 1280;
const HEADS: usize = 8;
const D: usize = 512;
const ORANK: usize = 1024;
pub const STEP_ARRAY_BOUND: usize = 128 * 1024 * 1024;

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
    fn checked_take(mut self) -> Result<T> {
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

pub struct BatchAttentionBlockKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    gemm: GpuGemm,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
    cache: GpuCacheQuant,
    sparse: Arc<PackedAttentionKernels>,
    compression: Arc<BatchCompressionKernels>,
    projection_status_cut: Option<ProjectionStatusCut>,
}
impl BatchAttentionBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        Self::load_with_gemm_workspace(dev, stream, boot, None)
    }
    pub fn load_with_gemm_workspace(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        workspace: Option<Arc<GemmGroupWorkspace>>,
    ) -> Result<Arc<Self>> {
        Self::load_with_status_backend(
            dev,
            stream,
            boot,
            workspace,
            ProjectionStatusBackend::Original,
        )
    }

    pub(crate) fn load_with_status_backend(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        workspace: Option<Arc<GemmGroupWorkspace>>,
        status_backend: ProjectionStatusBackend,
    ) -> Result<Arc<Self>> {
        ensure!(
            !boot.is_nil() && !stream.stream.is_null(),
            "packed block needs actual boot/owned stream"
        );
        Ok(Arc::new(Self {
            dev: dev.clone(),
            projection_status_cut: match status_backend {
                ProjectionStatusBackend::Original => None,
                ProjectionStatusBackend::PinnedSingleFence => {
                    Some(ProjectionStatusCut::new(dev, stream.clone(), boot)?)
                }
            },
            gemm: GpuGemm::load_with_workspace(dev, &stream, workspace)?,
            dense: GpuDense::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            cache: GpuCacheQuant::load(dev, &stream)?,
            sparse: PackedAttentionKernels::load(dev, stream.clone(), boot)?,
            compression: BatchCompressionKernels::load(dev, stream.clone(), boot)?,
            stream,
            boot,
        }))
    }
    pub(crate) fn bind_projection_status_owner(&self, owner: [u8; 32]) -> Result<()> {
        if let Some(cut) = &self.projection_status_cut {
            cut.bind_owner(owner)?;
        }
        Ok(())
    }
    pub(crate) fn projection_status_snapshot(&self) -> Option<ProjectionStatusSnapshot> {
        self.projection_status_cut
            .as_ref()
            .map(ProjectionStatusCut::snapshot)
    }
    fn copy<T: DeviceRepr>(&self, a: &CudaSlice<T>, b: &mut CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&a.device(), &self.dev)
                && Arc::ptr_eq(&b.device(), &self.dev)
                && !a.is_empty()
                && a.len() == b.len(),
            "packed block D2D context/extent differs"
        );
        self.dev.bind_to_thread()?;
        let r = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *b.device_ptr(),
                *a.device_ptr(),
                a.len() * std::mem::size_of::<T>(),
                self.stream.stream,
            )
        };
        ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "packed block D2D failed {r:?}"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Default)]
struct State {
    key: Option<LaneKey>,
    prefix: usize,
    version: u64,
    failed: bool,
}
#[derive(Default)]
struct Registry {
    lanes: Vec<State>,
}
fn reserve_progress(registry: &Arc<Mutex<Registry>>, ticket: &BatchTicket) -> Result<()> {
    let mut r = registry
        .lock()
        .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
    // Validate EVERY participant before invalidating any old publication. A bad
    // second lane must not partially reserve the first lane's progress.
    for s in ticket.segments() {
        let v = r
            .lanes
            .get(s.ticket.lane.slot)
            .context("packed slot exceeds owner")?;
        ensure!(
            v.key == Some(s.ticket.lane)
                && !v.failed
                && v.prefix == s.ticket.start
                && v.version < ticket.version(),
            "packed attention prefix/generation/version differs"
        );
    }
    for s in ticket.segments() {
        r.lanes[s.ticket.lane.slot].version = ticket.version();
    }
    Ok(())
}
fn poison(reg: &Arc<Mutex<Registry>>, ticket: &BatchTicket) {
    if let Ok(mut r) = reg.lock() {
        for s in ticket.segments() {
            if let Some(v) = r.lanes.get_mut(s.ticket.lane.slot) {
                if v.key == Some(s.ticket.lane) {
                    v.failed = true;
                }
            }
        }
    }
}
#[derive(Clone)]
struct Token {
    registry: Arc<Mutex<Registry>>,
    ticket: BatchTicket,
}
impl Token {
    fn check(&self) -> Result<()> {
        let r = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("packed block registry poisoned"))?;
        ensure!(
            self.ticket
                .segments()
                .iter()
                .all(|s| r.lanes.get(s.ticket.lane.slot).is_some_and(|v| v.key
                    == Some(s.ticket.lane)
                    && !v.failed
                    && v.prefix == s.ticket.end
                    && v.version == self.ticket.version())),
            "packed attention output stale/failed"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PackedAttentionReceipt {
    pub rank: usize,
    pub manifest_sha256: String,
    pub layer: usize,
    pub ticket: BatchTicket,
    pub completion_owner: [u8; 32],
    pub frequency_backend: FrequencyBackend,
    pub frequency_table_sha256: String,
    pub operation_id: u64,
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub compressor: Option<PackedLatentReceipt>,
    pub query_rows: usize,
    pub packed_lanes: usize,
    pub single_shared_projection_per_matrix: bool,
    pub one_packed_sparse_launch: bool,
    pub all_compressed_keys_selected_exactly: bool,
    pub skipped_index_scores_only_because_all_keys_fit: bool,
    pub index_keys_before_outer_KV_RoPE: bool,
    pub source_logical_tiles_preserved: bool,
    pub WO_B_local_dtype: &'static str,
    pub WO_B_network_dtype: &'static str,
    pub all_checks_passed: bool,
    pub caller_checked_HC_norm_required: bool,
    pub learned_model_qualified: bool,
    pub step_array_bound: usize,
}
pub struct CheckedPackedAttention {
    k: Arc<BatchAttentionBlockKernels>,
    values: CudaSlice<bf16>,
    snapshot: Arc<PackedKvSnapshot>,
    receipt: PackedAttentionReceipt,
    token: Token,
}
impl CheckedPackedAttention {
    pub fn check(&self) -> Result<()> {
        self.token.check()?;
        self.snapshot.check()
    }
    pub fn values(&self) -> Result<&CudaSlice<bf16>> {
        self.check()?;
        Ok(&self.values)
    }
    pub fn snapshot(&self) -> Result<&Arc<PackedKvSnapshot>> {
        self.check()?;
        Ok(&self.snapshot)
    }
    pub fn receipt(&self) -> Result<&PackedAttentionReceipt> {
        self.check()?;
        Ok(&self.receipt)
    }
    pub fn download_checked(&self) -> Result<Vec<bf16>> {
        self.check()?;
        Ok(self.k.dev.dtoh_sync_copy(&self.values)?)
    }
}

pub struct GpuBatchAttentionBlock {
    k: Arc<BatchAttentionBlockKernels>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    manifest: String,
    owner: [u8; 32],
    layer: usize,
    max_lanes: usize,
    q_norm: CudaSlice<f32>,
    kv_norm: CudaSlice<f32>,
    key_norm: Option<CudaSlice<f32>>,
    compressor: Option<GpuPackedCompressor>,
    registry: Arc<Mutex<Registry>>,
    backend: Option<FrequencyBackend>,
}
impl GpuBatchAttentionBlock {
    pub fn new(
        k: Arc<BatchAttentionBlockKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        owner: [u8; 32],
        layer: usize,
        w: Arc<LoadedWeights>,
        max_lanes: usize,
    ) -> Result<Self> {
        super::chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            layer < 40
                && (1..=16).contains(&max_lanes)
                && owner.iter().any(|v| *v != 0)
                && w.scope() == &LoadScope::Layer(layer)
                && w.manifest_sha256() == m.manifest_sha256,
            "actual packed attention layer/session/manifest required"
        );
        let p = format!("layers.{layer}.attn");
        for (name, n, inner) in [
            ("wq_a", QR, HIDDEN),
            ("wq_b", HEADS * D, QR),
            ("wkv", D, HIDDEN),
            ("wo_b", HIDDEN, ORANK),
        ] {
            ensure!(
                w.native_u8(&format!("{p}.{name}.weight"))?.len() == n * inner
                    && w.native_u8(&format!("{p}.{name}.scale"))?.len() == n / 32 * (inner / 32),
                "actual native packed attention matrix geometry"
            );
        }
        ensure!(
            w.bf16(&format!("{p}.wo_a.weight"))?.len() == ORANK * HEADS * D
                && w.f32(&format!("{p}.attn_sink"))?.len() == HEADS,
            "native WO-A/sink geometry"
        );
        let mut qnorm = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(QR)?);
        let mut kvnorm = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(D)?);
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
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
        let keynorm = if KV_SOURCES.contains(&layer) {
            ensure!(
                w.bf16(&format!("{p}.indexer.wk.weight"))?.len() == 128 * D,
                "actual native WK geometry"
            );
            let mut kn = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(128)?);
            k.position.cast_bf16_f32(
                &k.stream,
                w.bf16(&format!("{p}.indexer.k_norm.weight"))?,
                &mut kn,
                &mut status,
            )?;
            Some(kn)
        } else {
            None
        };
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "packed attention norm conversion status"
        );
        let compressor = if KV_SOURCES.contains(&layer) {
            Some(GpuPackedCompressor::new(
                k.compression.clone(),
                c,
                m,
                owner,
                layer,
                w.clone(),
                max_lanes,
            )?)
        } else {
            None
        };
        Ok(Self {
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            owner,
            layer,
            max_lanes,
            q_norm: qnorm.checked_take()?,
            kv_norm: kvnorm.checked_take()?,
            key_norm: keynorm.map(Queued::checked_take).transpose()?,
            k,
            weights: w,
            compressor,
            registry: Arc::new(Mutex::new(Registry {
                lanes: vec![State::default(); max_lanes],
            })),
            backend: None,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.k.boot && key.slot < self.max_lanes && key.generation > 0,
            "packed attention reset lane differs"
        );
        {
            let r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
            ensure!(
                r.lanes[key.slot]
                    .key
                    .is_none_or(|old| key.generation > old.generation),
                "packed attention reset must advance generation"
            );
        }
        self.k.dev.synchronize()?;
        if let Some(c) = &mut self.compressor {
            c.reset_lane(key)?;
        }
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?
            .lanes[key.slot] = State {
            key: Some(key),
            ..State::default()
        };
        Ok(())
    }
    pub fn invalidate(&mut self, ticket: &BatchTicket) {
        poison(&self.registry, ticket);
        if let Some(c) = &mut self.compressor {
            c.invalidate(ticket);
        }
    }
    /// Input is the caller's checked HC-normalized flattened member rows.
    /// Cache layer commit belongs to the complete HC/attention/MoE caller.
    pub fn execute_checked(
        &mut self,
        ticket: &BatchTicket,
        input: &CudaSlice<bf16>,
        qfreq: &GpuFrequencyRows,
        compressed_freq: Option<&GpuFrequencyRows>,
        operation_id: u64,
        store: &mut GpuBatchStore,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedPackedAttention> {
        let result = self.execute_inner(
            ticket,
            input,
            qfreq,
            compressed_freq,
            operation_id,
            store,
            comm,
        );
        if result.is_err() {
            comm.abort_session();
            self.invalidate(ticket);
            let _ = store.abort(ticket);
            let _ = self.k.dev.synchronize();
        }
        result
    }
    fn execute_inner(
        &mut self,
        ticket: &BatchTicket,
        input: &CudaSlice<bf16>,
        qfreq: &GpuFrequencyRows,
        kfreq: Option<&GpuFrequencyRows>,
        operation_id: u64,
        store: &mut GpuBatchStore,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedPackedAttention> {
        store.check_owner(self.rank, &self.manifest, self.owner, &self.k.stream)?;
        let plan = store.layer(ticket, self.layer)?;
        let rows = ticket.total_rows();
        let ratio = plan.lanes.first().context("empty packed plan")?.ratio;
        let profile = if ratio == 0 {
            RopeProfile::Plain
        } else {
            RopeProfile::Yarn
        };
        let positions = ticket
            .segments()
            .iter()
            .flat_map(|s| s.ticket.start..s.ticket.end)
            .map(|p| p as u32)
            .collect::<Vec<_>>();
        ensure!(
            ticket.boot() == self.k.boot
                && plan.short_selection_exact()
                && rows > 0
                && rows <= 128
                && input.len() == rows * HIDDEN
                && Arc::ptr_eq(&input.device(), &self.k.dev)
                && comm.rank() == self.rank
                && comm.completion_identity() == self.owner
                && comm.compute_stream().stream == self.k.stream.stream,
            "packed attention context/geometry/owner or unimplemented learned selection"
        );
        let backend = qfreq.backend();
        ensure!(
            self.backend.is_none_or(|b| b == backend)
                && qfreq.complete_table_verified()
                && qfreq.profile() == profile
                && qfreq.positions() == positions
                && qfreq.table_sha() == backend.table_sha(profile),
            "native packed query frequency identity differs"
        );
        let compression_plan = if KV_SOURCES.contains(&self.layer) {
            Some(batch_compression_plan::build(&plan)?)
        } else {
            None
        };
        let new_rows = compression_plan.as_ref().map_or(0, |c| c.output_rows);
        if new_rows > 0 {
            let f = kfreq.context("packed source completed groups need frequencies")?;
            ensure!(
                f.complete_table_verified()
                    && f.backend() == backend
                    && f.profile() == RopeProfile::Yarn
                    && f.table_sha() == backend.table_sha(RopeProfile::Yarn)
                    && f.positions() == compression_plan.as_ref().unwrap().rope_positions,
                "packed compressed frequency rows differ"
            );
        } else {
            ensure!(kfreq.is_none(), "unused packed compressed frequencies");
        }
        reserve_progress(&self.registry, ticket)?;
        self.backend = Some(backend);
        let k = self.k.clone();
        let s = &k.stream;
        let p = format!("layers.{}.attn", self.layer);
        let epoch_before = comm.expected_epoch();
        // Every queued temporary fences before early-error teardown. The shared
        // session is synchronously controlled; outer error abort precedes its
        // explicit fence, while old inner Drop fences retain their stated scope.
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        let xq = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, input, rows, HIDDEN)?,
        );
        let qa = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                xq.device_values_unchecked(),
                xq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wq_a.weight"))?,
                self.weights.native_u8(&format!("{p}.wq_a.scale"))?,
                GemmShape {
                    rows,
                    outputs: QR,
                    inner: HIDDEN,
                },
            )?
        });
        let qr = Queued::new(&k.dev, unsafe {
            k.numerics
                .rms_norm_1017(s, qa.device_values_unchecked(), &self.q_norm, rows, QR)?
        });
        let qrq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, qr.device_values_unchecked(), rows, QR)?
        });
        let qb = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                qrq.device_values_unchecked(),
                qrq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wq_b.weight"))?,
                self.weights.native_u8(&format!("{p}.wq_b.scale"))?,
                GemmShape {
                    rows,
                    outputs: HEADS * D,
                    inner: QR,
                },
            )?
        });
        let mut q = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(rows * HEADS * D)?);
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
            rows,
            HEADS,
            D,
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
                    rows,
                    outputs: D,
                    inner: HIDDEN,
                },
            )?
        });
        let kn = Queued::new(&k.dev, unsafe {
            k.numerics
                .rms_norm_1017(s, wk.device_values_unchecked(), &self.kv_norm, rows, D)?
        });
        let mut window = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(rows * D)?);
        unsafe {
            k.copy(kn.device_values_unchecked(), &mut window)?;
        }
        k.position.rotate(
            s,
            &mut window,
            qfreq,
            profile,
            &positions,
            &mut status,
            1,
            rows,
            1,
            D,
            false,
        )?;
        let wq = Queued::new(
            &k.dev,
            k.numerics.quantize_activations_bf16(s, &window, rows, D)?,
        );
        let wqat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                wq.device_values_unchecked(),
                wq.device_scales_unchecked(),
                rows,
                D,
                DequantCast::OfficialBf16Round,
            )?
        });
        if let Some(cut) = &k.projection_status_cut {
            cut.check(
                self.owner,
                ticket.boot(),
                s,
                [
                    xq.status_device_word(),
                    qa.status_device_word(),
                    qr.status_device_word(),
                    qrq.status_device_word(),
                    qb.status_device_word(),
                    wk.status_device_word(),
                    kn.status_device_word(),
                    wq.status_device_word(),
                    wqat.status_device_word(),
                    &status,
                ],
            )?;
        } else {
            for check in [
                xq.check(),
                qa.check(),
                qr.check(),
                qrq.check(),
                qb.check(),
                wk.check(),
                kn.check(),
                wq.check(),
                wqat.check(),
            ] {
                check?;
            }
            ensure!(
                k.dev.dtoh_sync_copy(&status)? == [0],
                "packed query/window rotation failed"
            );
        }
        let latent = match &mut self.compressor {
            Some(c) => {
                Some(c.execute_checked(self.rank, &self.manifest, self.owner, &plan, input)?)
            }
            None => None,
        };
        let compressor = latent.as_ref().map(|v| v.receipt().cloned()).transpose()?;
        // Native WK owns only source2/8/14/20 and MUST see normalized UNROTATED
        // latent. Short-context source topk selects every stored key exactly;
        // projected Q/index-score math has no effect and is not approximated.
        let mut key_qat = None;
        if KV_SOURCES.contains(&self.layer) {
            if new_rows > 0 {
                let values = latent
                    .as_ref()
                    .context("source latent absent")?
                    .values_for_plan(self.rank, &self.manifest, self.owner, &plan)?
                    .context("completed source latent absent")?;
                let projected = Queued::new(
                    &k.dev,
                    k.dense.bf16_linear_bf16(
                        s,
                        values,
                        self.weights.bf16(&format!("{p}.indexer.wk.weight"))?,
                        DenseShape {
                            rows: new_rows,
                            outputs: 128,
                            inner: D,
                        },
                    )?,
                );
                let normalized = Queued::new(&k.dev, unsafe {
                    k.numerics.rms_norm_1017(
                        s,
                        projected.device_values_unchecked(),
                        self.key_norm.as_ref().context("source key norm")?,
                        new_rows,
                        128,
                    )?
                });
                let mut rotated = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(new_rows * 128)?);
                unsafe {
                    k.copy(normalized.device_values_unchecked(), &mut rotated)?;
                }
                k.position.rotate(
                    s,
                    &mut rotated,
                    kfreq.unwrap(),
                    RopeProfile::Yarn,
                    &compression_plan.as_ref().unwrap().rope_positions,
                    &mut status,
                    1,
                    new_rows,
                    1,
                    128,
                    false,
                )?;
                let qat = Queued::new(
                    &k.dev,
                    k.cache
                        .quantize(s, &rotated, new_rows, 128, CacheFormat::IndexUe8m0Group32)?,
                );
                projected.check()?;
                normalized.check()?;
                qat.native().check()?;
                ensure!(
                    k.dev.dtoh_sync_copy(&status)? == [0],
                    "packed WK rotation failed"
                );
                store.publish_index_keys_checked(
                    ticket,
                    self.layer,
                    Some(unsafe { qat.device_qat_unchecked() }),
                )?;
                key_qat = Some(qat);
            } else {
                store.publish_index_keys_checked(ticket, self.layer, None)?;
            }
        }
        let mut compressed_rotated = None;
        let mut compressed_qat = None;
        if let Some(latent) = &latent {
            if let Some(copy) =
                latent.rotary_copy_for_plan(self.rank, &self.manifest, self.owner, &plan)?
            {
                let mut rotated = Queued::new(&k.dev, copy);
                k.position.rotate(
                    s,
                    &mut rotated,
                    kfreq.unwrap(),
                    profile,
                    &compression_plan.as_ref().unwrap().rope_positions,
                    &mut status,
                    1,
                    new_rows,
                    1,
                    D,
                    false,
                )?;
                let qat = Queued::new(
                    &k.dev,
                    k.cache.quantize(
                        s,
                        &rotated,
                        new_rows,
                        D,
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
            "packed compressed rotation failed"
        );
        let snapshot = Arc::new(
            store.prepare_short_checked(
                ticket,
                self.layer,
                unsafe { wqat.device_values_unchecked() },
                compressed_qat
                    .as_ref()
                    .map(|q| unsafe { q.device_qat_unchecked() }),
            )?,
        );
        ensure!(
            snapshot.plan() == &plan,
            "packed snapshot differs from attention plan"
        );
        let sparse = Queued::new(
            &k.dev,
            k.sparse
                .enqueue(
                    self.rank,
                    &self.manifest,
                    self.owner,
                    &snapshot,
                    &q,
                    self.weights.f32(&format!("{p}.attn_sink"))?,
                )?
                .finish()?,
        );
        let mut o = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(rows * HEADS * D)?);
        k.copy(sparse.values(), &mut o)?;
        k.position.rotate(
            s,
            &mut o,
            qfreq,
            profile,
            &positions,
            &mut status,
            1,
            rows,
            HEADS,
            D,
            true,
        )?;
        let oq = Queued::new(
            &k.dev,
            k.numerics
                .quantize_activations_bf16(s, &o, rows, HEADS * D)?,
        );
        let oqat = Queued::new(&k.dev, unsafe {
            k.dense.quantized_activations_to_bf16(
                s,
                oq.device_values_unchecked(),
                oq.device_scales_unchecked(),
                rows,
                HEADS * D,
                DequantCast::OfficialBf16Round,
            )?
        });
        let oa = Queued::new(&k.dev, unsafe {
            k.dense.bf16_linear_bf16(
                s,
                oqat.device_values_unchecked(),
                self.weights.bf16(&format!("{p}.wo_a.weight"))?,
                DenseShape {
                    rows,
                    outputs: ORANK,
                    inner: HEADS * D,
                },
            )?
        });
        let oaq = Queued::new(&k.dev, unsafe {
            k.numerics
                .quantize_activations_bf16(s, oa.device_values_unchecked(), rows, ORANK)?
        });
        let ob = Queued::new(&k.dev, unsafe {
            k.gemm.fp8_complete_bf16(
                s,
                oaq.device_values_unchecked(),
                oaq.device_scales_unchecked(),
                self.weights.native_u8(&format!("{p}.wo_b.weight"))?,
                self.weights.native_u8(&format!("{p}.wo_b.scale"))?,
                GemmShape {
                    rows,
                    outputs: HIDDEN,
                    inner: ORANK,
                },
            )?
        });
        let mut reduce = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(rows * HIDDEN)?);
        unsafe {
            k.position
                .cast_bf16_f32(s, ob.device_values_unchecked(), &mut reduce, &mut status)?;
        }
        for check in [
            oq.check(),
            oqat.check(),
            oa.check(),
            oaq.check(),
            ob.check(),
            snapshot.check(),
        ] {
            check?;
        }
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "packed sparse/output projection status failed"
        );
        comm.enqueue_reduce(
            operation_id,
            RoutedLayout::new(rows, ticket.kind().collective_rows())?,
            &mut reduce,
        )?;
        comm.synchronize()?;
        let mut output = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(rows * HIDDEN)?);
        k.position
            .cast_f32_bf16(s, &reduce, &mut output, &mut status)?;
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0] && comm.completion_identity() == self.owner,
            "packed complete projection/source BF16 cast failed"
        );
        snapshot.check()?;
        {
            let mut r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
            for segment in ticket.segments() {
                let v = &mut r.lanes[segment.ticket.lane.slot];
                v.prefix = segment.ticket.end;
            }
        }
        let receipt = PackedAttentionReceipt {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            layer: self.layer,
            ticket: ticket.clone(),
            completion_owner: self.owner,
            frequency_backend: backend,
            frequency_table_sha256: qfreq.table_sha().into(),
            operation_id,
            epoch_before,
            epoch_after: comm.expected_epoch(),
            compressor,
            query_rows: rows,
            packed_lanes: ticket.lanes(),
            single_shared_projection_per_matrix: true,
            one_packed_sparse_launch: true,
            all_compressed_keys_selected_exactly: true,
            skipped_index_scores_only_because_all_keys_fit: true,
            index_keys_before_outer_KV_RoPE: true,
            source_logical_tiles_preserved: true,
            WO_B_local_dtype: "BF16 complete local Linear",
            WO_B_network_dtype: "source BF16 local widened to qualified F32 network candidate",
            all_checks_passed: true,
            caller_checked_HC_norm_required: true,
            learned_model_qualified: false,
            step_array_bound: STEP_ARRAY_BOUND,
        };
        let _retain = (
            &key_qat,
            &compressed_rotated,
            &compressed_qat,
            &latent,
            &qr,
            &qa,
            &qb,
            &kn,
            &wk,
            &xq,
            &wqat,
        );
        Ok(CheckedPackedAttention {
            k: self.k.clone(),
            values: output.checked_take()?,
            snapshot,
            receipt,
            token: Token {
                registry: self.registry.clone(),
                ticket: ticket.clone(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::batch_plan::{BatchCursor, LaneSegment, PackedKind};
    fn ticket() -> BatchTicket {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let c = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let boot = Uuid::from_u128(0x164100ba);
        let mut cursor = BatchCursor::new(&c, boot, 16).unwrap();
        let keys = [7, 2].map(|slot| LaneKey {
            boot,
            slot,
            generation: 1,
        });
        for key in keys {
            cursor.reset_lane(key).unwrap();
        }
        cursor
            .begin(
                PackedKind::PrefillSegments,
                keys.into_iter()
                    .map(|key| LaneSegment {
                        key,
                        start: 0,
                        tokens: 3,
                    })
                    .collect(),
            )
            .unwrap()
    }
    fn registry(t: &BatchTicket) -> Arc<Mutex<Registry>> {
        let mut lanes = vec![State::default(); 16];
        for s in t.segments() {
            lanes[s.ticket.lane.slot] = State {
                key: Some(s.ticket.lane),
                ..State::default()
            };
        }
        Arc::new(Mutex::new(Registry { lanes }))
    }
    #[test]
    fn bad_second_lane_cannot_partially_reserve_first_lane_or_publish_output() {
        let t = ticket();
        let r = registry(&t);
        let second = t.segments()[1].ticket.lane.slot;
        r.lock().unwrap().lanes[second].prefix = 1;
        assert!(reserve_progress(&r, &t).is_err());
        assert_eq!(
            r.lock().unwrap().lanes[t.segments()[0].ticket.lane.slot].version,
            0
        );
        assert!(Token {
            registry: r.clone(),
            ticket: t.clone()
        }
        .check()
        .is_err());
        r.lock().unwrap().lanes[second].prefix = 0;
        reserve_progress(&r, &t).unwrap();
        assert!(reserve_progress(&r, &t).is_err());
        assert!(Token {
            registry: r.clone(),
            ticket: t.clone()
        }
        .check()
        .is_err());
        for s in t.segments() {
            r.lock().unwrap().lanes[s.ticket.lane.slot].prefix = s.ticket.end;
        }
        Token {
            registry: r,
            ticket: t,
        }
        .check()
        .unwrap();
    }
    #[test]
    fn participant_poison_does_not_touch_other_generation_or_lane() {
        let t = ticket();
        let r = registry(&t);
        let first = t.segments()[0].ticket.lane;
        r.lock().unwrap().lanes[first.slot].key = Some(LaneKey {
            generation: 2,
            ..first
        });
        let untouched = t.segments()[1].ticket.lane.slot;
        poison(&r, &t);
        assert!(!r.lock().unwrap().lanes[first.slot].failed);
        assert!(r.lock().unwrap().lanes[untouched].failed);
        assert!(!r.lock().unwrap().lanes[0].failed);
        assert!(reserve_progress(&r, &t).is_err());
    }
}
