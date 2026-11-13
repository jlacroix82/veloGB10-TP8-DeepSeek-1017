//! PRIVATE packed native compressor: shared Dense projections, groupwise F32
//! pooling, BF16-before-RMS, per-lane F32 odd tail, unrotated immutable output.
//! Separate pool/tail launches prevent an old-tail read/new-tail write race.
use super::{
    batch_compression_plan::{self, PackedCompressionPlan},
    batch_plan::{BatchLayerPlan, BatchTicket},
    chunk_plan::KV_SOURCES,
    config::Dsv41Config,
    gpu_dense::{DenseShape, GpuDense},
    gpu_module::Module,
    gpu_numerics::{DeviceOutput, GpuNumerics},
    gpu_position::PositionKernels,
    gpu_weights::{LoadScope, LoadedWeights},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};
use uuid::Uuid;
const D: usize = 512;
const HIDDEN: usize = 5120;
pub const STEP_ARRAY_BOUND: usize = 16 * 1024 * 1024;
pub const PTX: &str = include_str!("../ptx/dsv41_batch_compression.ptx");
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
pub struct BatchCompressionKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    module: Module,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
}
impl BatchCompressionKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(!boot.is_nil(), "packed compressor boot required");
        let module = Module::load(
            dev,
            &stream,
            PTX,
            &[
                "dsv41_batch_compression_reset",
                "dsv41_batch_compression_pool2",
                "dsv41_batch_compression_tail2",
            ],
            "dsv41_batch_compression_identity",
            u64::from_str_radix(
                env!("DSV41_BATCH_COMPRESSION_SOURCE_ID").trim_start_matches("0x"),
                16,
            )?,
            1,
        )?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            dense: GpuDense::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            module,
            stream,
            boot,
        }))
    }
    fn launch(&self, name: &str, elements: usize, args: &mut [*mut c_void]) -> Result<()> {
        ensure!(
            elements > 0 && elements <= 128 * D,
            "bounded nonempty packed compressor launch"
        );
        unsafe {
            self.module.launch(
                name,
                &self.stream,
                (elements.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                args,
            )
        }
    }
}
#[derive(Clone, Debug, Default)]
struct State {
    key: Option<LaneKey>,
    prefix: usize,
    version: u64,
    tail: Option<usize>,
    failed: bool,
}
#[derive(Default)]
struct Registry {
    lanes: Vec<State>,
}
fn poison(registry: &Arc<Mutex<Registry>>, ticket: &BatchTicket) {
    if let Ok(mut r) = registry.lock() {
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
            .map_err(|_| anyhow::anyhow!("packed compressor registry poisoned"))?;
        ensure!(
            self.ticket
                .segments()
                .iter()
                .all(|s| r.lanes.get(s.ticket.lane.slot).is_some_and(|v| v.key
                    == Some(s.ticket.lane)
                    && !v.failed
                    && v.prefix == s.ticket.end
                    && v.version == self.ticket.version())),
            "stale/failed packed latent producer"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct PackedLatentReceipt {
    pub rank: usize,
    pub manifest_sha256: String,
    pub completion_owner: [u8; 32],
    pub source_layer: usize,
    pub plan: PackedCompressionPlan,
    pub output_rows: usize,
    pub columns: usize,
    pub projection_dtype: &'static str,
    pub pool_tail_dtype: &'static str,
    pub RMS_input_dtype: &'static str,
    pub RMS_epsilon: f64,
    pub output_unrotated: bool,
    pub shared_packed_projection: bool,
    pub old_tail_read_completed_before_tail_write: bool,
    pub all_checks_passed: bool,
    pub full_model_qualified: bool,
}
pub struct PackedLatentOutput {
    k: Arc<BatchCompressionKernels>,
    values: Option<DeviceOutput<bf16>>,
    receipt: PackedLatentReceipt,
    token: Token,
}
impl PackedLatentOutput {
    pub fn receipt(&self) -> Result<&PackedLatentReceipt> {
        self.token.check()?;
        Ok(&self.receipt)
    }
    pub fn values_for_plan(
        &self,
        rank: usize,
        manifest: &str,
        owner: [u8; 32],
        plan: &BatchLayerPlan,
    ) -> Result<Option<&CudaSlice<bf16>>> {
        self.token.check()?;
        ensure!(
            self.receipt.rank == rank
                && self.receipt.manifest_sha256 == manifest
                && self.receipt.completion_owner == owner
                && self.receipt.plan == batch_compression_plan::build(plan)?,
            "packed latent rank/source/session/ticket differs"
        );
        if let Some(v) = &self.values {
            v.check()?;
            Ok(Some(unsafe { v.device_values_unchecked() }))
        } else {
            Ok(None)
        }
    }
    /// A copy protects the source's normalized unrotated WK input from outer KV
    /// RoPE/cacheQAT mutation. The copy is fenced before its independent publication.
    pub fn rotary_copy_for_plan(
        &self,
        rank: usize,
        manifest: &str,
        owner: [u8; 32],
        plan: &BatchLayerPlan,
    ) -> Result<Option<CudaSlice<bf16>>> {
        let Some(values) = self.values_for_plan(rank, manifest, owner, plan)? else {
            return Ok(None);
        };
        let mut copy = self.k.dev.alloc_zeros::<bf16>(values.len())?;
        let result = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *copy.device_ptr(),
                *values.device_ptr(),
                values.len() * 2,
                self.k.stream.stream,
            )
        };
        self.k.dev.synchronize()?;
        ensure!(
            result == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "packed latent immutable rotary copy failed"
        );
        self.token.check()?;
        Ok(Some(copy))
    }
}
pub struct GpuPackedCompressor {
    k: Arc<BatchCompressionKernels>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    manifest: String,
    owner: [u8; 32],
    layer: usize,
    ratio: usize,
    max_lanes: usize,
    norm: CudaSlice<f32>,
    tail_kv: CudaSlice<f32>,
    tail_score: CudaSlice<f32>,
    registry: Arc<Mutex<Registry>>,
}
impl GpuPackedCompressor {
    pub fn new(
        k: Arc<BatchCompressionKernels>,
        c: &Dsv41Config,
        manifest: &RankManifest,
        owner: [u8; 32],
        layer: usize,
        weights: Arc<LoadedWeights>,
        max_lanes: usize,
    ) -> Result<Self> {
        super::chunk_plan::validate_config(c)?;
        manifest.verify_identity()?;
        ensure!(
            KV_SOURCES.contains(&layer)
                && (1..=16).contains(&max_lanes)
                && owner.iter().any(|b| *b != 0)
                && weights.scope() == &LoadScope::Layer(layer)
                && weights.manifest_sha256() == manifest.manifest_sha256,
            "actual packed compressor source/session required"
        );
        let prefix = format!("layers.{layer}.attn.compressor");
        let ratio = if layer < 20 { 2 } else { 1 };
        ensure!(
            weights.bf16(&format!("{prefix}.wkv.weight"))?.len() == D * HIDDEN
                && weights.bf16(&format!("{prefix}.norm.weight"))?.len() == D,
            "native packed compressor geometry"
        );
        if ratio == 2 {
            ensure!(
                weights.bf16(&format!("{prefix}.wgate.weight"))?.len() == D * HIDDEN,
                "native ratio2 gate geometry"
            );
        }
        let mut norm = k.dev.alloc_zeros::<f32>(D)?;
        let mut status = k.dev.alloc_zeros::<u32>(1)?;
        let conversion = k.position.cast_bf16_f32(
            &k.stream,
            weights.bf16(&format!("{prefix}.norm.weight"))?,
            &mut norm,
            &mut status,
        );
        k.dev.synchronize()?;
        conversion?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "packed compressor norm conversion failed"
        );
        let tail_kv = k.dev.alloc_zeros::<f32>(max_lanes * D)?;
        let tail_score = k.dev.alloc_zeros::<f32>(max_lanes * D)?;
        Ok(Self {
            k,
            weights,
            rank: manifest.rank,
            manifest: manifest.manifest_sha256.clone(),
            owner,
            layer,
            ratio,
            max_lanes,
            norm,
            tail_kv,
            tail_score,
            registry: Arc::new(Mutex::new(Registry {
                lanes: vec![State::default(); max_lanes],
            })),
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.k.boot && key.slot < self.max_lanes && key.generation > 0,
            "packed compressor lane key differs"
        );
        {
            let mut r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed compressor registry poisoned"))?;
            let old = &r.lanes[key.slot];
            ensure!(
                old.key.is_none_or(|v| key.generation > v.generation),
                "compressor reset must advance generation"
            );
            r.lanes[key.slot] = State {
                key: Some(key),
                failed: true,
                ..State::default()
            };
        }
        let mut status = Queued::new(&self.k.dev, self.k.dev.alloc_zeros::<u32>(1)?);
        self.k.dev.memset_zeros(&mut *status)?;
        let (mut a, mut b, mut st, mut slot, mut lanes) = (
            *self.tail_kv.device_ptr(),
            *self.tail_score.device_ptr(),
            *status.device_ptr(),
            key.slot as u64,
            self.max_lanes as u64,
        );
        let mut args = [
            &mut a as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut lanes as *mut _ as *mut c_void,
        ];
        self.k
            .launch("dsv41_batch_compression_reset", D, &mut args)?;
        self.k.dev.synchronize()?;
        ensure!(
            self.k.dev.dtoh_sync_copy(&status)? == [0],
            "packed compressor reset status failed"
        );
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("packed compressor registry poisoned"))?
            .lanes[key.slot]
            .failed = false;
        Ok(())
    }
    pub fn invalidate(&mut self, ticket: &BatchTicket) {
        poison(&self.registry, ticket);
    }
    pub fn execute_checked(
        &mut self,
        rank: usize,
        manifest: &str,
        owner: [u8; 32],
        plan: &BatchLayerPlan,
        x: &CudaSlice<bf16>,
    ) -> Result<PackedLatentOutput> {
        ensure!(
            rank == self.rank
                && manifest == self.manifest
                && owner == self.owner
                && plan.layer == self.layer
                && plan.ticket.boot() == self.k.boot,
            "packed compressor owner/source differs"
        );
        let geometry = batch_compression_plan::build(plan)?;
        ensure!(
            geometry.ratio == self.ratio && x.len() == geometry.input_rows * HIDDEN,
            "packed compressor input rows differ"
        );
        self.k.module.check_buffer(x)?;
        self.k.module.check_stream(&self.k.stream)?;
        {
            let r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed compressor registry poisoned"))?;
            for m in &geometry.members {
                let s = r
                    .lanes
                    .get(m.key.slot)
                    .context("compressor slot exceeds owner")?;
                ensure!(
                    s.key == Some(m.key)
                        && !s.failed
                        && s.prefix == m.start
                        && s.version < plan.ticket.version()
                        && s.tail == m.required_old_tail,
                    "packed compressor generation/prefix/F32 tail differs"
                );
            }
        }
        let result = self.execute_inner(plan, &geometry, x);
        match result {
            Ok(values) => {
                {
                    let mut r = self
                        .registry
                        .lock()
                        .map_err(|_| anyhow::anyhow!("packed compressor registry poisoned"))?;
                    for m in &geometry.members {
                        let s = &mut r.lanes[m.key.slot];
                        s.prefix = m.end;
                        s.version = plan.ticket.version();
                        s.tail = m.retained_new_tail;
                    }
                }
                let receipt = PackedLatentReceipt {
                    rank: self.rank,
                    manifest_sha256: self.manifest.clone(),
                    completion_owner: self.owner,
                    source_layer: self.layer,
                    plan: geometry.clone(),
                    output_rows: geometry.output_rows,
                    columns: D,
                    projection_dtype: if self.ratio == 2 {
                        "source F32 input/weights/accumulator"
                    } else {
                        "source BF16 Linear complete output"
                    },
                    pool_tail_dtype: if self.ratio == 2 { "F32" } else { "none" },
                    RMS_input_dtype: "BF16 after complete source pool/projection",
                    RMS_epsilon: 1e-20,
                    output_unrotated: true,
                    shared_packed_projection: true,
                    old_tail_read_completed_before_tail_write: true,
                    all_checks_passed: true,
                    full_model_qualified: false,
                };
                Ok(PackedLatentOutput {
                    k: self.k.clone(),
                    values,
                    receipt,
                    token: Token {
                        registry: self.registry.clone(),
                        ticket: plan.ticket.clone(),
                    },
                })
            }
            Err(e) => {
                poison(&self.registry, &plan.ticket);
                let _ = self.k.dev.synchronize();
                Err(e)
            }
        }
    }
    fn execute_inner(
        &mut self,
        plan: &BatchLayerPlan,
        g: &PackedCompressionPlan,
        x: &CudaSlice<bf16>,
    ) -> Result<Option<DeviceOutput<bf16>>> {
        let k = &self.k;
        let s = &k.stream;
        let prefix = format!("layers.{}.attn.compressor", self.layer);
        let shape = DenseShape {
            rows: g.input_rows,
            outputs: D,
            inner: HIDDEN,
        };
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();return Err(e.into());}}};}
        // Maximum raw projection/cast/group storage is below16MiB at128 rows. This
        // bound excludes already resident source weights and tiny persistent tails.
        ensure!(
            g.input_rows * (HIDDEN * 4 + D * 8)
                + g.output_rows * D * 4
                + g.groups.len() * 24
                + g.tails.len() * 16
                < STEP_ARRAY_BOUND,
            "packed compressor explicit step bound"
        );
        if self.ratio == 1 {
            let projected = Queued::new(
                &k.dev,
                queued!(k.dense.bf16_linear_bf16(
                    s,
                    x,
                    queued!(self.weights.bf16(&format!("{prefix}.wkv.weight"))),
                    shape
                )),
            );
            let normalized = queued!(k.numerics.rms_norm_1017(
                s,
                unsafe { projected.device_values_unchecked() },
                &self.norm,
                g.output_rows,
                D
            ));
            queued!(projected.check());
            queued!(normalized.check());
            return Ok(Some(normalized));
        }
        let mut cast = Queued::new(
            &k.dev,
            queued!(k.dev.alloc_zeros::<f32>(g.input_rows * HIDDEN)),
        );
        let mut status = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<u32>(1)));
        queued!(k.dev.memset_zeros(&mut *status));
        queued!(k.position.cast_bf16_f32(s, x, &mut cast, &mut status));
        let kv = Queued::new(
            &k.dev,
            queued!(k.dense.f32_bf16_linear_f32(
                s,
                &cast,
                queued!(self.weights.bf16(&format!("{prefix}.wkv.weight"))),
                shape
            )),
        );
        let gate = Queued::new(
            &k.dev,
            queued!(k.dense.f32_bf16_linear_f32(
                s,
                &cast,
                queued!(self.weights.bf16(&format!("{prefix}.wgate.weight"))),
                shape
            )),
        );
        let groups = if g.groups.is_empty() {
            None
        } else {
            Some(Queued::new(
                &k.dev,
                queued!(k.dev.htod_sync_copy(
                    &g.groups
                        .iter()
                        .flat_map(|x| x.iter().copied())
                        .collect::<Vec<_>>()
                )),
            ))
        };
        let tails = if g.tails.is_empty() {
            None
        } else {
            Some(Queued::new(
                &k.dev,
                queued!(k.dev.htod_sync_copy(
                    &g.tails
                        .iter()
                        .flat_map(|x| x.iter().copied())
                        .collect::<Vec<_>>()
                )),
            ))
        };
        let mut norm_input = if g.output_rows == 0 {
            None
        } else {
            Some(Queued::new(
                &k.dev,
                queued!(k.dev.alloc_zeros::<bf16>(g.output_rows * D)),
            ))
        };
        let (mut kp, mut gp, mut tk, mut ts, mut ep, mut rows, mut lanes) = (
            unsafe { *kv.device_values_unchecked().device_ptr() },
            unsafe { *gate.device_values_unchecked().device_ptr() },
            *self.tail_kv.device_ptr(),
            *self.tail_score.device_ptr(),
            *status.device_ptr(),
            g.input_rows as u64,
            self.max_lanes as u64,
        );
        if let (Some(descriptors), Some(out)) = (&groups, &mut norm_input) {
            let (mut dp, mut op, mut count) = (
                *descriptors.device_ptr(),
                *out.device_ptr(),
                g.output_rows as u64,
            );
            let mut args = [
                &mut kp as *mut _ as *mut c_void,
                &mut gp as *mut _ as *mut c_void,
                &mut tk as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut dp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut count as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                "dsv41_batch_compression_pool2",
                g.output_rows * D,
                &mut args
            ));
        }
        if let Some(descriptors) = &tails {
            let (mut dp, mut count) = (*descriptors.device_ptr(), g.tails.len() as u64);
            let mut args = [
                &mut kp as *mut _ as *mut c_void,
                &mut gp as *mut _ as *mut c_void,
                &mut tk as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut dp as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut count as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                "dsv41_batch_compression_tail2",
                g.tails.len() * D,
                &mut args
            ));
        }
        let normalized = if let Some(input) = &norm_input {
            Some(queued!(k.numerics.rms_norm_1017(
                s,
                input,
                &self.norm,
                g.output_rows,
                D
            )))
        } else {
            None
        };
        queued!(kv.check());
        queued!(gate.check());
        if let Some(output) = &normalized {
            queued!(output.check());
        }
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "packed pool/tail/cast status failed"
        );
        let _ = plan;
        Ok(normalized)
    }
}
