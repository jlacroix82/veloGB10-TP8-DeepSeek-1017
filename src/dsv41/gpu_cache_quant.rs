//! Isolated native1017 FP4 activation/cache quantization. No serving backend.
//! All buffers are pending until sticky status and stream completion are checked.
use super::cache_quant::{self, CacheFormat, QuantizedCache};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync,
    LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub const PTX: &str = include_str!("../ptx/dsv41_cache_quant.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_cache_quant.cu");
const MODULE: &str = "dsv41_cache_quant_v1";
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}

pub struct CheckedOutput<T> {
    dev: Arc<CudaDevice>,
    values: CudaSlice<T>,
    status: CudaSlice<u32>,
}
impl<T: DeviceRepr> CheckedOutput<T> {
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(
            status == 0,
            "cache quant status0x{status:x} (1=input,2=scale,4=BF16 reconstruction overflow)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<T>> {
        self.check()?;
        Ok(self.dev.dtoh_sync_copy(&self.values)?)
    }
    /// Caller must establish completion and check sticky status before use.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<T> {
        &self.values
    }
}

pub struct PackedCacheGpu {
    dev: Arc<CudaDevice>,
    packed: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    status: CudaSlice<u32>,
    rows: usize,
    columns: usize,
    format: CacheFormat,
}
impl PackedCacheGpu {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn format(&self) -> CacheFormat {
        self.format
    }
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(
            status == 0,
            "cache quant status0x{status:x} (1=input,2=scale,4=BF16 reconstruction overflow)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.check()?;
        Ok((
            self.dev.dtoh_sync_copy(&self.packed)?,
            self.dev.dtoh_sync_copy(&self.scales)?,
        ))
    }
    /// Caller must establish completion and check sticky status before use.
    pub unsafe fn device_packed_unchecked(&self) -> &CudaSlice<u8> {
        &self.packed
    }
    /// Caller must establish completion and check sticky status before use.
    pub unsafe fn device_scales_unchecked(&self) -> &CudaSlice<u8> {
        &self.scales
    }
}
pub struct CacheQuantGpu {
    native: PackedCacheGpu,
    qat: CudaSlice<bf16>,
}
impl CacheQuantGpu {
    pub fn native(&self) -> &PackedCacheGpu {
        &self.native
    }
    pub fn download_checked(&self) -> Result<QuantizedCache> {
        let (packed, scales) = self.native.download_checked()?;
        Ok(QuantizedCache {
            rows: self.native.rows,
            columns: self.native.columns,
            format: self.native.format,
            packed,
            scales,
            qat: self.native.dev.dtoh_sync_copy(&self.qat)?,
        })
    }
    /// Caller must establish completion and check native sticky status before use.
    pub unsafe fn device_qat_unchecked(&self) -> &CudaSlice<bf16> {
        &self.qat
    }
}
pub struct GpuCacheQuant {
    dev: Arc<CudaDevice>,
    source_id: u64,
}
impl GpuCacheQuant {
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !s.stream.is_null(),
            "explicit blocking cache-quant stream required"
        );
        let mut flags = 0;
        let rc = unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "invalid/nonblocking cache-quant stream"
        );
        Ok(())
    }
    fn input<T>(&self, x: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &x.device()),
            "cache-quant CUDA device/context mismatch"
        );
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let source_id = u64::from_str_radix(
            env!("DSV41_CACHE_QUANT_SOURCE_ID").trim_start_matches("0x"),
            16,
        )?;
        let this = Self {
            dev: dev.clone(),
            source_id,
        };
        this.stream(stream)?;
        if dev.get_func(MODULE, "dsv41_cache_quant_identity").is_none() {
            dev.load_ptx(
                Ptx::from_src(PTX),
                MODULE,
                &[
                    "dsv41_cache_quant_identity",
                    "dsv41_cache_quant_e4_group16",
                    "dsv41_cache_quant_ue8_group32",
                    "dsv41_cache_dequant_bf16",
                    "dsv41_cache_fp4_encode",
                ],
            )?;
        }
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        dev.memset_zeros(&mut identity)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_cache_quant_identity")
                .context("cache identity kernel")?
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
            "stale cache-quant source/ABI"
        );
        Ok(this)
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
    fn run_quant(
        &self,
        stream: &CudaStream,
        input: u64,
        qat: u64,
        rows: usize,
        columns: usize,
        format: CacheFormat,
    ) -> Result<PackedCacheGpu> {
        self.stream(stream)?;
        let count = cache_quant::elements(rows, columns, format)?;
        let groups = count / format.group();
        let packed = self.dev.alloc_zeros::<u8>(count / 2)?;
        let scales = self.dev.alloc_zeros::<u8>(groups)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        let name = match format {
            CacheFormat::CompressedKvE4m3Group16 => "dsv41_cache_quant_e4_group16",
            CacheFormat::IndexUe8m0Group32 => "dsv41_cache_quant_ue8_group32",
        };
        unsafe {
            self.dev
                .get_func(MODULE, name)
                .context("cache quant kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (u32::try_from(groups.div_ceil(4))?, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        ptr(&packed),
                        ptr(&scales),
                        qat,
                        ptr(&status),
                        input,
                        groups as u64,
                    ),
                )?;
        }
        Ok(PackedCacheGpu {
            dev: self.dev.clone(),
            packed,
            scales,
            status,
            rows,
            columns,
            format,
        })
    }
    pub fn quantize(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        rows: usize,
        columns: usize,
        format: CacheFormat,
    ) -> Result<CacheQuantGpu> {
        self.input(input)?;
        let count = cache_quant::elements(rows, columns, format)?;
        ensure!(input.len() == count, "cache input extent mismatch");
        let qat = self.dev.alloc_zeros::<bf16>(count)?;
        let native = self.run_quant(stream, ptr(input), ptr(&qat), rows, columns, format)?;
        Ok(CacheQuantGpu { native, qat })
    }
    /// Exact aliasing is safe: each warp first loads its complete disjoint group
    /// into registers, then writes only that group's BF16 reconstruction.
    pub fn quantize_in_place(
        &self,
        stream: &CudaStream,
        input: &mut CudaSlice<bf16>,
        rows: usize,
        columns: usize,
        format: CacheFormat,
    ) -> Result<PackedCacheGpu> {
        self.input(input)?;
        ensure!(
            input.len() == cache_quant::elements(rows, columns, format)?,
            "in-place cache extent mismatch"
        );
        self.run_quant(stream, ptr(input), ptr(input), rows, columns, format)
    }
    pub fn dequantize(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        rows: usize,
        columns: usize,
        format: CacheFormat,
    ) -> Result<CheckedOutput<bf16>> {
        self.stream(stream)?;
        self.input(packed)?;
        self.input(scales)?;
        let count = cache_quant::elements(rows, columns, format)?;
        ensure!(
            packed.len() == count / 2 && scales.len() == count / format.group(),
            "cache native extent mismatch"
        );
        let values = self.dev.alloc_zeros::<bf16>(count)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_cache_dequant_bf16")
                .context("cache dequant kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems(u32::try_from(count)?),
                    (
                        ptr(&values),
                        ptr(&status),
                        ptr(packed),
                        ptr(scales),
                        count as u64,
                        format.tag(),
                    ),
                )?;
        }
        Ok(CheckedOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
    pub fn encode_scalars(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
    ) -> Result<CheckedOutput<u8>> {
        self.stream(stream)?;
        self.input(input)?;
        let count = input.len();
        ensure!((1..=4096).contains(&count), "scalar FP4 probe bound");
        let values = self.dev.alloc_zeros::<u8>(count)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_cache_fp4_encode")
                .context("native FP4 cast kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems(count as u32),
                    (ptr(&values), ptr(&status), ptr(input), count as u64),
                )?;
        }
        Ok(CheckedOutput {
            dev: self.dev.clone(),
            values,
            status,
        })
    }
}
