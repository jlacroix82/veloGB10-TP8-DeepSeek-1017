//! Isolated official1017 GPU numerical components, on an explicit caller-owned blocking stream.
//! No model loader or serving route is enabled here. Status flags are mandatory: a returned
//! device buffer is pending until check/download succeeds. No checkpoint weights are quantized.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice,
    LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use sha2::{Digest, Sha256};
use std::sync::Arc;

use super::numerics::{ActivationQuant, GROUP, NORM_EPS};

pub const ABI_VERSION: u64 = 1;
pub const PTX: &str = include_str!("../ptx/dsv41_numerics.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_numerics.cu");
const MODULE: &str = "dsv41_numerics_isolated";
const NAMES: &[&str] = &[
    "dsv41_numerics_identity",
    "dsv41_act_quant_group32",
    "dsv41_act_quant_group128_negative",
    "dsv41_fp8_encode_scalars",
    "dsv41_ue8m0_decode_scalars",
    "dsv41_fp4_native_dequant",
    "dsv41_rmsnorm_bf16",
];
// Component allocation policy, not a GPU/vendor context-capacity specification.
pub const MAX_ELEMENTS: usize = 64 * 1024 * 1024;

fn ptr<T>(slice: &CudaSlice<T>) -> u64 {
    *slice.device_ptr()
}
fn dims(rows: usize, columns: usize) -> Result<usize> {
    ensure!(rows > 0 && columns > 0, "positive component shape required");
    rows.checked_mul(columns)
        .filter(|&x| x <= MAX_ELEMENTS)
        .context("component shape exceeds bounded allocation policy")
}
fn linear_grid(count: usize) -> Result<LaunchConfig> {
    ensure!(
        count > 0 && count <= MAX_ELEMENTS,
        "component element count outside bound"
    );
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(count.div_ceil(256))?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub struct DeviceOutput<T> {
    dev: Arc<CudaDevice>,
    values: CudaSlice<T>,
    status: CudaSlice<u32>,
}
impl<T: DeviceRepr> DeviceOutput<T> {
    pub(crate) fn status_device_word(&self) -> &CudaSlice<u32> {
        &self.status
    }
    pub fn check(&self) -> Result<()> {
        self.dev.synchronize()?;
        let status = self.dev.dtoh_sync_copy(&self.status)?[0];
        ensure!(
            status == 0,
            "dsv41 numerical GPU status0x{status:x} (1=nonfiniteinput,2=scale,4=overflow)"
        );
        Ok(())
    }
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn download_checked(&self) -> Result<Vec<T>> {
        self.check()?;
        Ok(self.dev.dtoh_sync_copy(&self.values)?)
    }
    /// Caller must first establish stream completion and check this operation's status.
    /// This accessor performs no synchronization, so dependent GPU integration stays possible.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<T> {
        &self.values
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

pub struct QuantizedGpu {
    dev: Arc<CudaDevice>,
    rows: usize,
    columns: usize,
    group: usize,
    values: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    status: CudaSlice<u32>,
}
impl QuantizedGpu {
    pub(crate) fn status_device_word(&self) -> &CudaSlice<u32> {
        &self.status
    }
    pub fn check(&self) -> Result<()> {
        self.dev.synchronize()?;
        let status = self.dev.dtoh_sync_copy(&self.status)?[0];
        ensure!(status == 0, "dsv41 activation GPU status0x{status:x}");
        Ok(())
    }
    pub fn download_checked(&self) -> Result<ActivationQuant> {
        self.check()?;
        Ok(ActivationQuant {
            rows: self.rows,
            columns: self.columns,
            group_size: self.group,
            e4m3_values: self.dev.dtoh_sync_copy(&self.values)?,
            ue8m0_scales: self.dev.dtoh_sync_copy(&self.scales)?,
        })
    }
    /// Caller must first establish stream completion and check this operation's status.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<u8> {
        &self.values
    }
    /// Caller must first establish stream completion and check this operation's status.
    pub unsafe fn device_scales_unchecked(&self) -> &CudaSlice<u8> {
        &self.scales
    }
}

pub struct GpuNumerics {
    dev: Arc<CudaDevice>,
    source_id: u64,
}
impl GpuNumerics {
    fn stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit caller-owned blocking stream required"
        );
        let mut flags = 0_u32;
        let result = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            result == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "dsv41 numerics requires a valid blocking stream (flags0), got{flags} {result:?}"
        );
        Ok(())
    }
    fn input<T>(&self, input: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &input.device()),
            "component input belongs to a different CUDA device/context handle"
        );
        Ok(())
    }
    fn function(&self, name: &'static str) -> Result<CudaFunction> {
        self.dev
            .get_func(MODULE, name)
            .with_context(|| format!("missing isolated dsv41 kernel{name}"))
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let id_text = env!("DSV41_NUMERICS_SOURCE_ID");
        let source_id = u64::from_str_radix(id_text.trim_start_matches("0x"), 16)
            .context("invalid dedicatedDSV41_NUMERICS_SOURCE_ID")?;
        let result = Self {
            dev: dev.clone(),
            source_id,
        };
        result.stream(stream)?;
        if dev.get_func(MODULE, "dsv41_numerics_identity").is_none() {
            dev.load_ptx(Ptx::from_src(PTX), MODULE, NAMES)
                .context("load isolated dsv41numerics PTX")?;
        }
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        dev.memset_zeros(&mut identity)?;
        unsafe {
            result
                .function("dsv41_numerics_identity")?
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
        let observed = dev.dtoh_sync_copy(&identity)?;
        ensure!(
            observed == [source_id, ABI_VERSION],
            "isolated dsv41numerics source/ABI identity mismatch"
        );
        Ok(result)
    }
    pub fn source_id(&self) -> u64 {
        self.source_id
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    pub fn cuda_source_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(CUDA_SOURCE))
    }
    fn quantize(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        rows: usize,
        columns: usize,
        group: usize,
    ) -> Result<QuantizedGpu> {
        self.stream(stream)?;
        self.input(input)?;
        let count = dims(rows, columns)?;
        ensure!(
            columns % group == 0 && input.len() == count,
            "activation group/shape mismatch"
        );
        ensure!(
            group == 32 || group == 128,
            "unsupported diagnostic activation group"
        );
        let groups = count / group;
        let values = self.dev.alloc_zeros::<u8>(count)?;
        let scales = self.dev.alloc_zeros::<u8>(groups)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        // Explicitly reset sticky atomicOr flags, including diagnostic controls.
        // cudarc alloc_zeros already clears these; the redundant reset makes the
        // initialization contract local. A blocking stream observes the memset.
        self.dev.memset_zeros(&mut status)?;
        let name = if group == GROUP {
            "dsv41_act_quant_group32"
        } else {
            "dsv41_act_quant_group128_negative"
        };
        unsafe {
            self.function(name)?.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (u32::try_from(groups.div_ceil(4))?, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    ptr(&values),
                    ptr(&scales),
                    ptr(&status),
                    ptr(input),
                    groups as u64,
                ),
            )?;
        }
        Ok(QuantizedGpu {
            dev: self.dev.clone(),
            rows,
            columns,
            group,
            values,
            scales,
            status,
        })
    }
    pub fn quantize_activations_bf16(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        rows: usize,
        columns: usize,
    ) -> Result<QuantizedGpu> {
        self.quantize(stream, input, rows, columns, GROUP)
    }
    /// Deliberately wrong for1017; exposed only to prove the oracle rejects oldgroup128 semantics.
    pub fn group128_negative_control(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        rows: usize,
        columns: usize,
    ) -> Result<QuantizedGpu> {
        self.quantize(stream, input, rows, columns, 128)
    }
    pub fn encode_f32_scalars(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
    ) -> Result<DeviceOutput<u8>> {
        self.stream(stream)?;
        self.input(input)?;
        let cfg = linear_grid(input.len())?;
        let values = self.dev.alloc_zeros::<u8>(input.len())?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        // Explicitly reset sticky atomicOr flags, including diagnostic controls.
        // cudarc alloc_zeros already clears these; the redundant reset makes the
        // initialization contract local. A blocking stream observes the memset.
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.function("dsv41_fp8_encode_scalars")?
                .launch_on_stream(
                    stream,
                    cfg,
                    (ptr(&values), ptr(&status), ptr(input), input.len() as u64),
                )?;
        }
        Ok(DeviceOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn decode_ue8m0_scalars(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<u8>,
    ) -> Result<DeviceOutput<f32>> {
        self.stream(stream)?;
        self.input(input)?;
        let cfg = linear_grid(input.len())?;
        let values = self.dev.alloc_zeros::<f32>(input.len())?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        // Explicitly reset sticky atomicOr flags, including diagnostic controls.
        // cudarc alloc_zeros already clears these; the redundant reset makes the
        // initialization contract local. A blocking stream observes the memset.
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.function("dsv41_ue8m0_decode_scalars")?
                .launch_on_stream(
                    stream,
                    cfg,
                    (ptr(&values), ptr(&status), ptr(input), input.len() as u64),
                )?;
        }
        Ok(DeviceOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn dequantize_fp4_native(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        rows: usize,
        columns: usize,
    ) -> Result<DeviceOutput<f32>> {
        self.stream(stream)?;
        self.input(packed)?;
        self.input(scales)?;
        let count = dims(rows, columns)?;
        ensure!(
            columns % GROUP == 0 && packed.len() == count / 2 && scales.len() == count / GROUP,
            "nativeFP4component shape mismatch"
        );
        let values = self.dev.alloc_zeros::<f32>(count)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        // Explicitly reset sticky atomicOr flags, including diagnostic controls.
        // cudarc alloc_zeros already clears these; the redundant reset makes the
        // initialization contract local. A blocking stream observes the memset.
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.function("dsv41_fp4_native_dequant")?
                .launch_on_stream(
                    stream,
                    linear_grid(count)?,
                    (
                        ptr(&values),
                        ptr(&status),
                        ptr(packed),
                        ptr(scales),
                        count as u64,
                        columns as u64,
                    ),
                )?;
        }
        Ok(DeviceOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    /// Epsilon is explicit for qualification. The1017 model call must pass NORM_EPS=1e-20.
    pub fn rms_norm_bf16(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        weights: &CudaSlice<f32>,
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<DeviceOutput<bf16>> {
        self.stream(stream)?;
        self.input(input)?;
        self.input(weights)?;
        let count = dims(rows, columns)?;
        ensure!(
            input.len() == count && weights.len() == columns,
            "RMScomponent shape mismatch"
        );
        ensure!(
            epsilon.is_finite() && epsilon > 0.0,
            "positive finite RMSepsilon required"
        );
        let values = self.dev.alloc_zeros::<bf16>(count)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        // Explicitly reset sticky atomicOr flags, including diagnostic controls.
        // cudarc alloc_zeros already clears these; the redundant reset makes the
        // initialization contract local. A blocking stream observes the memset.
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.function("dsv41_rmsnorm_bf16")?.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (u32::try_from(rows)?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    ptr(&values),
                    ptr(&status),
                    ptr(input),
                    ptr(weights),
                    rows as u64,
                    columns as u64,
                    epsilon,
                ),
            )?;
        }
        Ok(DeviceOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn rms_norm_1017(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        weights: &CudaSlice<f32>,
        rows: usize,
        columns: usize,
    ) -> Result<DeviceOutput<bf16>> {
        self.rms_norm_bf16(stream, input, weights, rows, columns, NORM_EPS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_shape_bounds_are_cpu_only_and_fail_closed() {
        assert_eq!(dims(16, 5120).unwrap(), 81920);
        assert_eq!(dims(4096, 5120).unwrap(), 20971520);
        assert!(dims(0, 5120).is_err());
        assert!(dims(usize::MAX, 2).is_err());
        assert!(dims(MAX_ELEMENTS, 2).is_err());
        assert!(linear_grid(0).is_err());
        assert_eq!(linear_grid(257).unwrap().grid_dim, (2, 1, 1));
    }
}
