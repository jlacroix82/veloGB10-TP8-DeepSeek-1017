//! Native1017 text-only FFN composition. The caller supplies the F32 TP8
//! reduction between local routed experts and the final shared-expert add.
//! This conservative first executor retains every queued input and checks all
//! component statuses. It does not claim allocation-free serving or a model.
use super::{
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseOutput, DenseShape, GpuDense},
    gpu_experts::{
        ExpertBank, ExpertShape, ExpertWorkspace, GpuExperts, PendingCombined, PendingExperts,
    },
    gpu_gemm::{GemmGroupWorkspace, GemmOutput, GemmShape, GpuGemm},
    gpu_moe::MoeKernels,
    gpu_numerics::{GpuNumerics, QuantizedGpu},
    gpu_position::PositionKernels,
    gpu_weights::{LoadScope, LoadedWeights},
    weights::RankManifest,
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use half::bf16;
use std::sync::Arc;

const DIM: usize = 5120;
const INTER: usize = 2304;

/// Shared kernels are loaded once per rank, rather than once per model layer.
pub struct MoeBlockKernels {
    dev: Arc<CudaDevice>,
    dense: GpuDense,
    gemm: GpuGemm,
    numerics: GpuNumerics,
    position: PositionKernels,
    route: MoeKernels,
    experts: GpuExperts,
}
impl MoeBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Arc<Self>> {
        Self::load_with_gemm_workspace(dev, stream, None)
    }
    pub fn load_with_gemm_workspace(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        workspace: Option<Arc<GemmGroupWorkspace>>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            dev: dev.clone(),
            dense: GpuDense::load(dev, stream)?,
            gemm: GpuGemm::load_with_workspace(dev, stream, workspace)?,
            numerics: GpuNumerics::load(dev, stream)?,
            position: PositionKernels::load(dev, stream)?,
            route: MoeKernels::load(dev, stream)?,
            experts: GpuExperts::load(dev, stream)?,
        }))
    }
}

pub struct MoeBlock {
    kernels: Arc<MoeBlockKernels>,
    weights: Arc<LoadedWeights>,
    bank: ExpertBank,
    layer: usize,
}
impl MoeBlock {
    pub fn new(
        kernels: Arc<MoeBlockKernels>,
        weights: Arc<LoadedWeights>,
        manifest: &RankManifest,
        layer: usize,
    ) -> Result<Self> {
        ensure!(
            manifest.rank < 8 && layer < 40 && *weights.scope() == LoadScope::Layer(layer),
            "FFN rank/layer scope mismatch"
        );
        let bank = ExpertBank::from_loaded(&kernels.dev, manifest, layer, weights.clone())?;
        ensure!(
            bank.complete_native_bank(),
            "FFN requires every native local expert"
        );
        // Validate replicated shared/router tensors before publishing a block.
        let prefix = format!("layers.{layer}.ffn");
        ensure!(
            weights.bf16(&format!("{prefix}.gate.weight"))?.len() == 384 * DIM
                && weights.f32(&format!("{prefix}.gate.bias"))?.len() == 384,
            "FFN router geometry"
        );
        for (name, n, k) in [("w1", INTER, DIM), ("w3", INTER, DIM), ("w2", DIM, INTER)] {
            ensure!(
                weights
                    .native_u8(&format!("{prefix}.shared_experts.{name}.weight"))?
                    .len()
                    == n * k
                    && weights
                        .native_u8(&format!("{prefix}.shared_experts.{name}.scale"))?
                        .len()
                        == n / 32 * (k / 32),
                "FFN shared native FP8 geometry"
            );
        }
        Ok(Self {
            kernels,
            weights,
            bank,
            layer,
        })
    }

    /// Prepare routing and the unweighted shared expert. Route IDs remain on
    /// the GPU; the routed executor never reads expert choices on the CPU.
    pub fn prepare<'a>(
        &'a self,
        stream: &'a CudaStream,
        input: &'a CudaSlice<bf16>,
        rows: usize,
    ) -> Result<PreparedMoe<'a>> {
        ensure!(
            (1..=128).contains(&rows) && input.len() == rows * DIM,
            "FFN input geometry"
        );
        let k = &self.kernels;
        ensure!(
            Arc::ptr_eq(&k.dev, &input.device()),
            "FFN input context mismatch"
        );
        // The closure catches even a nested `?` in a call argument. It fences
        // BEFORE returning an error, while every earlier local owner is alive.
        macro_rules! queued {
            ($expression:expr) => {
                match (|| $expression)() {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = k.dev.synchronize();
                        return Err(error.into());
                    }
                }
            };
        }
        let prefix = format!("layers.{}.ffn", self.layer);
        // Gate.forward explicitly promotes both operands to F32. Do not replace
        // this operation with a BF16-output projection or TF32 approximation.
        let mut promoted = queued!(k.dev.alloc_zeros::<f32>(rows * DIM));
        let mut status = queued!(k.dev.alloc_zeros::<u32>(1));
        queued!(k
            .position
            .cast_bf16_f32(stream, input, &mut promoted, &mut status));
        let logits = queued!(k.dense.f32_bf16_linear_f32(
            stream,
            &promoted,
            self.weights.bf16(&format!("{prefix}.gate.weight"))?,
            DenseShape {
                rows,
                outputs: 384,
                inner: DIM,
            },
        ));
        let mut ids = queued!(k.dev.alloc_zeros::<u32>(rows * 6));
        let mut route_weights = queued!(k.dev.alloc_zeros::<f32>(rows * 6));
        queued!(k.route.route(
            stream,
            unsafe { logits.device_values_unchecked() },
            self.weights.f32(&format!("{prefix}.gate.bias"))?,
            &mut ids,
            &mut route_weights,
            &mut status,
            rows,
        ));
        let quant_input = queued!(k
            .numerics
            .quantize_activations_bf16(stream, input, rows, DIM));
        let shared =
            |name: &str, activation: &QuantizedGpu, outputs, inner| -> Result<GemmOutput<bf16>> {
                k.gemm.fp8_complete_bf16(
                    stream,
                    unsafe { activation.device_values_unchecked() },
                    unsafe { activation.device_scales_unchecked() },
                    self.weights
                        .native_u8(&format!("{prefix}.shared_experts.{name}.weight"))?,
                    self.weights
                        .native_u8(&format!("{prefix}.shared_experts.{name}.scale"))?,
                    GemmShape {
                        rows,
                        outputs,
                        inner,
                    },
                )
            };
        let gate = queued!(shared("w1", &quant_input, INTER, DIM));
        let up = queued!(shared("w3", &quant_input, INTER, DIM));
        let unused_weights = queued!(k.dev.alloc_zeros::<f32>(rows));
        let mut hidden = queued!(k.dev.alloc_zeros::<bf16>(rows * INTER));
        queued!(k.route.swiglu(
            stream,
            unsafe { gate.device_values_unchecked() },
            unsafe { up.device_values_unchecked() },
            &unused_weights,
            false,
            &mut hidden,
            &mut status,
            rows,
            INTER,
        ));
        let quant_hidden = queued!(k
            .numerics
            .quantize_activations_bf16(stream, &hidden, rows, INTER));
        let shared_output = queued!(shared("w2", &quant_hidden, DIM, INTER));
        // These checks are deliberately explicit during correctness bringup.
        // Removing their synchronization needs an owned pooled execution graph.
        queued!(logits.check());
        queued!(quant_input.check());
        queued!(gate.check());
        queued!(up.check());
        queued!(quant_hidden.check());
        queued!(shared_output.check());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "FFN prepare numeric/status failure"
        );
        Ok(PreparedMoe {
            block: self,
            stream,
            input,
            rows,
            ids,
            route_weights,
            status,
            shared_output,
            _promoted: promoted,
            _logits: logits,
            _quant_input: quant_input,
            _gate: gate,
            _up: up,
            _unused_weights: unused_weights,
            _hidden: hidden,
            _quant_hidden: quant_hidden,
        })
    }
    pub fn workspace(&self, rows: usize) -> Result<ExpertWorkspace> {
        ExpertWorkspace::new(&self.kernels.dev, ExpertShape::native1017(), rows)
    }
    pub fn enqueue_routed<'a>(
        &'a self,
        prepared: &'a PreparedMoe<'a>,
        workspace: &'a mut ExpertWorkspace,
    ) -> Result<PendingExperts<'a>> {
        ensure!(
            std::ptr::eq(self, prepared.block),
            "FFN preparation belongs to a different layer"
        );
        self.kernels.experts.enqueue(
            prepared.stream,
            &self.bank,
            workspace,
            prepared.input,
            &prepared.ids,
            &prepared.route_weights,
            &prepared.status,
        )
    }
    /// Input must be the completed F32 all-rank routed-expert sum. The shared
    /// BF16 expert is promoted and added before the single final BF16 cast.
    pub fn finish<'a>(
        &'a self,
        prepared: &'a PreparedMoe<'a>,
        reduced: &'a CudaSlice<f32>,
        reduction_status: &CudaSlice<u32>,
        output: &'a mut CudaSlice<bf16>,
        status: &'a mut CudaSlice<u32>,
    ) -> Result<PendingCombined<'a>> {
        ensure!(
            std::ptr::eq(self, prepared.block) && reduced.len() == prepared.rows * DIM,
            "FFN reduced result identity/shape"
        );
        ensure!(
            Arc::ptr_eq(&self.kernels.dev, &reduction_status.device())
                && reduction_status.len() == 1,
            "FFN reduction status context/shape"
        );
        ensure!(
            self.kernels.dev.dtoh_sync_copy(reduction_status)? == [0],
            "FFN upstream collective failed"
        );
        self.kernels.experts.combine_reduced_shared(
            prepared.stream,
            reduced,
            unsafe { prepared.shared_output.device_values_unchecked() },
            output,
            status,
        )
    }
    /// Integration gate using the actual rank-local collective owner rather
    /// than a synthetic success status. Caller retains the routed producer and
    /// the exact mutable buffer passed to enqueue_reduce until this completes.
    pub fn finish_checked<'a>(
        &'a self,
        prepared: &'a PreparedMoe<'a>,
        reduced: &'a CudaSlice<f32>,
        collectives: &mut GpuCollectives,
        output: &'a mut CudaSlice<bf16>,
        status: &'a mut CudaSlice<u32>,
    ) -> Result<PendingCombined<'a>> {
        ensure!(
            std::ptr::eq(self, prepared.block)
                && reduced.len() == prepared.rows * DIM
                && collectives.rank() == self.bank.rank()
                && collectives.compute_stream().stream == prepared.stream.stream,
            "FFN completion belongs to another layer, rank, stream or shape"
        );
        collectives.synchronize()?;
        self.kernels.experts.combine_reduced_shared(
            prepared.stream,
            reduced,
            unsafe { prepared.shared_output.device_values_unchecked() },
            output,
            status,
        )
    }
}
pub struct PreparedMoe<'a> {
    block: &'a MoeBlock,
    stream: &'a CudaStream,
    input: &'a CudaSlice<bf16>,
    rows: usize,
    ids: CudaSlice<u32>,
    route_weights: CudaSlice<f32>,
    status: CudaSlice<u32>,
    shared_output: GemmOutput<bf16>,
    _promoted: CudaSlice<f32>,
    _logits: DenseOutput<f32>,
    _quant_input: QuantizedGpu,
    _gate: GemmOutput<bf16>,
    _up: GemmOutput<bf16>,
    _unused_weights: CudaSlice<f32>,
    _hidden: CudaSlice<bf16>,
    _quant_hidden: QuantizedGpu,
}
impl Drop for PreparedMoe<'_> {
    fn drop(&mut self) {
        let _ = self.block.kernels.dev.synchronize();
    }
}
