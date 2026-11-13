//! Isolated1017 dense linear components. No model loading/serving is enabled.
//! GenuineBF16 Linear uses nativeBF16 MMA with F32 accumulation; F32 Linear uses
//! explicit F32 multiply/add, neverTF32/FP16. StoredBF16 weights may be promoted
//! in registers, preserving exactly the values of official weight.float().
//! Source cast points are caller decisions: BF16 and F32 outputs are distinct APIs.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync,
    LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use sha2::{Digest, Sha256};
use std::sync::Arc;
pub const PTX: &str = include_str!("../ptx/dsv41_dense.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_dense.cu");
const MODULE: &str = "dsv41_dense_isolated";
const NAMES: &[&str] = &[
    "dsv41_dense_identity",
    "dsv41_dense_bf16_mma",
    "dsv41_dense_f32_bf16",
    "dsv41_dense_f32_f32",
    "dsv41_dense_fp8_to_bf16",
];
pub const MAX_MATRIX_ELEMENTS: usize = 128 * 1024 * 1024;
#[derive(Clone, Copy, Debug)]
pub struct DenseShape {
    pub rows: usize,
    pub outputs: usize,
    pub inner: usize,
}
impl DenseShape {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=128).contains(&self.rows) && self.outputs > 0 && self.inner > 0,
            "bounded dense rows1..128 and positive dimensions required"
        );
        for (a, b) in [
            (self.rows, self.inner),
            (self.outputs, self.inner),
            (self.rows, self.outputs),
        ] {
            ensure!(
                a.checked_mul(b).is_some_and(|x| x <= MAX_MATRIX_ELEMENTS),
                "dense component allocation policy exceeded"
            );
        }
        i32::try_from(self.rows)?;
        i32::try_from(self.outputs)?;
        i32::try_from(self.inner)?;
        Ok(())
    }
}
/// RequireExact proves the requested lossless WO-A conversion. OfficialBf16Round
/// deliberately permits the BF16 rounding defined by the reference conversion;
/// it is an explicit policy, never a silent precision-mode fallback.
#[derive(Clone, Copy, Debug)]
pub enum DequantCast {
    RequireExact,
    OfficialBf16Round,
}
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
pub struct DenseOutput<T> {
    dev: Arc<CudaDevice>,
    values: CudaSlice<T>,
    status: CudaSlice<u32>,
}
impl<T: DeviceRepr> DenseOutput<T> {
    pub(crate) fn status_device_word(&self) -> &CudaSlice<u32> {
        &self.status
    }
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(
            status == 0,
            "dense GPU status0x{status:x} (1=input,2=scale,4=overflow,8=lossycast)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<T>> {
        self.check()?;
        Ok(self.dev.dtoh_sync_copy(&self.values)?)
    }
    /// Caller must establish completion and check sticky status before consumption.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<T> {
        &self.values
    }
}
pub struct GpuDense {
    dev: Arc<CudaDevice>,
    source_id: u64,
}
impl GpuDense {
    fn stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit blocking dense stream required"
        );
        let mut flags = 0_u32;
        let rc = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "valid blocking dense stream required"
        );
        Ok(())
    }
    fn input<T>(&self, x: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &x.device()),
            "dense device/context mismatch"
        );
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let source_id =
            u64::from_str_radix(env!("DSV41_DENSE_SOURCE_ID").trim_start_matches("0x"), 16)?;
        let this = Self {
            dev: dev.clone(),
            source_id,
        };
        this.stream(stream)?;
        if dev.get_func(MODULE, "dsv41_dense_identity").is_none() {
            dev.load_ptx(Ptx::from_src(PTX), MODULE, NAMES)?;
        }
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        dev.memset_zeros(&mut identity)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_dense_identity")
                .context("denseidentity")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (ptr(&identity),),
                )?;
        }
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&identity)? == [source_id, 1],
            "dedicateddense source/ABI mismatch"
        );
        Ok(this)
    }
    pub fn source_id(&self) -> u64 {
        self.source_id
    }
    pub fn cuda_source_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(CUDA_SOURCE))
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    fn launch<I: DeviceRepr, W: DeviceRepr, T: DeviceRepr + ValidAsZeroBits>(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<I>,
        w: &CudaSlice<W>,
        shape: DenseShape,
        kernel: &'static str,
        bf16_output: bool,
        bf16_mma: bool,
    ) -> Result<DenseOutput<T>> {
        self.stream(stream)?;
        self.input(a)?;
        self.input(w)?;
        shape.validate()?;
        ensure!(
            a.len() == shape.rows * shape.inner && w.len() == shape.outputs * shape.inner,
            "dense matrix shape mismatch"
        );
        let values = self.dev.alloc_zeros::<T>(shape.rows * shape.outputs)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        let output = ptr(&values);
        unsafe {
            self.dev
                .get_func(MODULE, kernel)
                .with_context(|| format!("densekernel{kernel}"))?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(shape.outputs.div_ceil(if bf16_mma { 128 } else { 8 }))?,
                            u32::try_from(shape.rows.div_ceil(16))?,
                            1,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        if bf16_output { 0_u64 } else { output },
                        if bf16_output { output } else { 0_u64 },
                        ptr(&status),
                        ptr(a),
                        ptr(w),
                        shape.rows as i32,
                        shape.outputs as i32,
                        shape.inner as i32,
                    ),
                )?;
        }
        Ok(DenseOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn bf16_linear_f32(
        &self,
        s: &CudaStream,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        g: DenseShape,
    ) -> Result<DenseOutput<f32>> {
        self.launch(s, a, w, g, "dsv41_dense_bf16_mma", false, true)
    }
    pub fn bf16_linear_bf16(
        &self,
        s: &CudaStream,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        g: DenseShape,
    ) -> Result<DenseOutput<bf16>> {
        self.launch(s, a, w, g, "dsv41_dense_bf16_mma", true, true)
    }
    pub fn f32_bf16_linear_f32(
        &self,
        s: &CudaStream,
        a: &CudaSlice<f32>,
        w: &CudaSlice<bf16>,
        g: DenseShape,
    ) -> Result<DenseOutput<f32>> {
        self.launch(s, a, w, g, "dsv41_dense_f32_bf16", false, false)
    }
    pub fn f32_bf16_linear_bf16(
        &self,
        s: &CudaStream,
        a: &CudaSlice<f32>,
        w: &CudaSlice<bf16>,
        g: DenseShape,
    ) -> Result<DenseOutput<bf16>> {
        self.launch(s, a, w, g, "dsv41_dense_f32_bf16", true, false)
    }
    pub fn f32_linear_f32(
        &self,
        s: &CudaStream,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        g: DenseShape,
    ) -> Result<DenseOutput<f32>> {
        self.launch(s, a, w, g, "dsv41_dense_f32_f32", false, false)
    }
    pub fn f32_linear_bf16(
        &self,
        s: &CudaStream,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        g: DenseShape,
    ) -> Result<DenseOutput<bf16>> {
        self.launch(s, a, w, g, "dsv41_dense_f32_f32", true, false)
    }
    fn reconstruct(
        &self,
        stream: &CudaStream,
        codes: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        rows: usize,
        inner: usize,
        activation: bool,
        policy: DequantCast,
    ) -> Result<DenseOutput<bf16>> {
        self.stream(stream)?;
        self.input(codes)?;
        self.input(scales)?;
        ensure!(
            rows > 0 && inner > 0 && inner % 32 == 0,
            "group32 dequant shape required"
        );
        let count = rows
            .checked_mul(inner)
            .filter(|&x| x <= MAX_MATRIX_ELEMENTS)
            .context("dequant shape bound")?;
        ensure!(
            codes.len() == count
                && scales.len()
                    == (if activation { rows } else { rows.div_ceil(32) }) * (inner / 32),
            "dequant native byte/scale shape mismatch"
        );
        let values = self.dev.alloc_zeros::<bf16>(count)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_dense_fp8_to_bf16")
                .context("reconstructionkernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems(u32::try_from(count)?),
                    (
                        ptr(&values),
                        ptr(&status),
                        ptr(codes),
                        ptr(scales),
                        count as u64,
                        u32::try_from(inner)?,
                        u32::from(activation),
                        u32::from(matches!(policy, DequantCast::RequireExact)),
                    ),
                )?;
        }
        Ok(DenseOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn fp8_weights_to_bf16(
        &self,
        s: &CudaStream,
        codes: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        outputs: usize,
        inner: usize,
        policy: DequantCast,
    ) -> Result<DenseOutput<bf16>> {
        self.reconstruct(s, codes, scales, outputs, inner, false, policy)
    }
    pub fn quantized_activations_to_bf16(
        &self,
        s: &CudaStream,
        codes: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        rows: usize,
        inner: usize,
        policy: DequantCast,
    ) -> Result<DenseOutput<bf16>> {
        self.reconstruct(s, codes, scales, rows, inner, true, policy)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_actual_shapes_and_bounds_cpu_only() {
        for (outputs, inner) in [
            (35, 37),
            (384, 5120),
            (512, 5120),
            (128, 512),
            (1024, 4096),
            (16160, 5120),
        ] {
            for rows in [1, 2, 4, 8, 16, 128] {
                DenseShape {
                    rows,
                    outputs,
                    inner,
                }
                .validate()
                .unwrap();
            }
        }
        assert!(DenseShape {
            rows: 0,
            outputs: 1,
            inner: 1
        }
        .validate()
        .is_err());
        assert!(DenseShape {
            rows: 129,
            outputs: 1,
            inner: 1
        }
        .validate()
        .is_err());
        assert!(DenseShape {
            rows: 1,
            outputs: usize::MAX,
            inner: 2
        }
        .validate()
        .is_err());
    }
}
