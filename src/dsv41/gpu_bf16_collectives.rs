//! Isolated packing/finite guards around the unchanged serial/v2 BF16 K2.
//! Allocation/module qualification occurs once at workspace construction.
//! Enqueue performs no CPU tensor readback, allocation or synchronization.
use super::{
    bf16_collectives::{MatrixBf16Index, EPOCH_BF16, MAX_ELEMENTS},
    gpu_collectives::GpuCollectives,
    gpu_module::Module,
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_bf16_collectives.ptx");
pub const SOURCE_ID: &str = env!("DSV41_BF16_COLLECTIVES_SOURCE_ID");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_bf16_collectives.cu");

pub struct Bf16Workspace {
    dev: Arc<CudaDevice>,
    _stream_owner: Arc<CudaStream>,
    stream: usize,
    module: Module,
    scratch: CudaSlice<bf16>,
    status: CudaSlice<u32>,
    poisoned: bool,
}
impl Bf16Workspace {
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: &Arc<CudaStream>,
        maximum_elements: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_ELEMENTS).contains(&maximum_elements),
            "BF16 workspace capacity bound"
        );
        let module = Module::load(
            dev,
            stream,
            PTX,
            &["dsv41_bf16_pack", "dsv41_bf16_check", "dsv41_bf16_unpack"],
            "dsv41_bf16_collectives_identity",
            u64::from_str_radix(SOURCE_ID.trim_start_matches("0x"), 16)?,
            1,
        )?;
        let mut scratch = dev.alloc_zeros::<bf16>(maximum_elements.div_ceil(4) * 4)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut scratch)?;
        dev.memset_zeros(&mut status)?;
        dev.synchronize()?;
        Ok(Self {
            dev: dev.clone(),
            _stream_owner: stream.clone(),
            stream: stream.stream as usize,
            module,
            scratch,
            status,
            poisoned: false,
        })
    }
    pub fn capacity_elements(&self) -> usize {
        self.scratch.len()
    }
    pub fn source_id(&self) -> &'static str {
        SOURCE_ID
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    pub fn cuda_source_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(CUDA_SOURCE))
    }
    pub(crate) fn matches(
        &self,
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        layout: MatrixBf16Index,
    ) -> bool {
        !self.poisoned
            && Arc::ptr_eq(dev, &self.dev)
            && self.stream == stream.stream as usize
            && layout.padded_elements() <= self.scratch.len()
    }
    /// The shared session owns ctx/global_status and has already bound shape and
    /// source context to its immutable program. Both buffers outlive GPU work.
    pub(crate) fn enqueue_async(
        &mut self,
        stream: &CudaStream,
        serial: &crate::tp_xport::Kernels,
        ctx: u64,
        global_status: u64,
        values: &mut CudaSlice<bf16>,
        layout: MatrixBf16Index,
    ) -> Result<()> {
        let result = (|| {
            self.module.check_stream(stream)?;
            self.module.check_buffer(values)?;
            ensure!(
                !self.poisoned
                    && stream.stream as usize == self.stream
                    && values.len() == layout.elements()
                    && layout.padded_elements() <= self.scratch.len(),
                "BF16 workspace/source shape or owner differs"
            );
            let logical = layout.elements() as u32;
            let padded = layout.padded_elements() as u32;
            let source = *values.device_ptr();
            let scratch = *self.scratch.device_ptr();
            let status = *self.status.device_ptr();
            let mut c = ctx;
            let mut src = source;
            let mut dst = scratch;
            let mut n = logical;
            let mut pad = padded;
            let mut st = status;
            let mut gs = global_status;
            let mut args = [
                &mut c as *mut _ as *mut c_void,
                &mut src as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut pad as *mut _ as *mut c_void,
                &mut st as *mut _ as *mut c_void,
                &mut gs as *mut _ as *mut c_void,
            ];
            unsafe {
                self.module.launch(
                    "dsv41_bf16_pack",
                    stream,
                    (padded.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
            let mut offset = 0;
            while offset < layout.padded_elements() {
                let count = (layout.padded_elements() - offset).min(EPOCH_BF16);
                let pointer = scratch + (offset * 2) as u64;
                for _ in 0..3 {
                    // K1 is byte-agnostic but copies32-bit words; n is guaranteed
                    // a multiple of4BF16words, so neither payload nor tail truncates.
                    crate::tp_xport::k1_serial(serial, stream, ctx, pointer, count / 2)?;
                    crate::tp_xport::k2_serial_bf16(serial, stream, ctx, pointer, count)?;
                    let mut c = ctx;
                    let mut p = scratch;
                    let mut n = logical;
                    let mut pad = padded;
                    let mut first = offset as u32;
                    let mut len = count as u32;
                    let mut st = status;
                    let mut gs = global_status;
                    let mut args = [
                        &mut c as *mut _ as *mut c_void,
                        &mut p as *mut _ as *mut c_void,
                        &mut n as *mut _ as *mut c_void,
                        &mut pad as *mut _ as *mut c_void,
                        &mut first as *mut _ as *mut c_void,
                        &mut len as *mut _ as *mut c_void,
                        &mut st as *mut _ as *mut c_void,
                        &mut gs as *mut _ as *mut c_void,
                    ];
                    unsafe {
                        self.module.launch(
                            "dsv41_bf16_check",
                            stream,
                            ((count as u32).div_ceil(256), 1, 1),
                            (256, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
                offset += count;
            }
            let mut c = ctx;
            let mut src = scratch;
            let mut dst = source;
            let mut n = logical;
            let mut pad = padded;
            let mut st = status;
            let mut gs = global_status;
            let mut args = [
                &mut c as *mut _ as *mut c_void,
                &mut src as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut pad as *mut _ as *mut c_void,
                &mut st as *mut _ as *mut c_void,
                &mut gs as *mut _ as *mut c_void,
            ];
            unsafe {
                self.module.launch(
                    "dsv41_bf16_unpack",
                    stream,
                    (logical.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn pending<'a>(
        &'a mut self,
        values: &'a mut CudaSlice<bf16>,
        layout: MatrixBf16Index,
        identity: [u8; 32],
        epoch: u64,
        rank: usize,
    ) -> PendingBf16Matrix<'a> {
        PendingBf16Matrix {
            workspace: self,
            values,
            layout,
            identity,
            epoch,
            rank,
            checked: false,
        }
    }
}
impl Drop for Bf16Workspace {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Bf16MatrixIdentity {
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
    pub logical_elements: usize,
    pub padded_elements: usize,
    pub complete_epoch: u64,
    pub wire_dtype: &'static str,
    pub reduction: &'static str,
    pub source_id_hex: &'static str,
    pub nccl_bit_equivalence_qualified: bool,
}
pub struct PendingBf16Matrix<'a> {
    workspace: &'a mut Bf16Workspace,
    values: &'a mut CudaSlice<bf16>,
    layout: MatrixBf16Index,
    identity: [u8; 32],
    epoch: u64,
    rank: usize,
    checked: bool,
}
impl PendingBf16Matrix<'_> {
    pub fn identity(&self) -> Bf16MatrixIdentity {
        Bf16MatrixIdentity {
            rank: self.rank,
            rows: self.layout.rows(),
            cols: self.layout.cols(),
            logical_elements: self.layout.elements(),
            padded_elements: self.layout.padded_elements(),
            complete_epoch: self.epoch,
            wire_dtype: "BF16 nativeu16",
            reduction: "serial TP8 XORphase1,2,0; F32add then BF16RNE EACHround",
            source_id_hex: SOURCE_ID,
            nccl_bit_equivalence_qualified: false,
        }
    }
    /// Exact same ordered stream only. Keep this pending owner and every
    /// downstream consumer alive until checked completion; no CPU data copy.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<bf16> {
        self.values
    }
    pub fn check(&mut self, session: &mut GpuCollectives) -> Result<()> {
        self.checked = false;
        let result = (|| {
            ensure!(
                !self.workspace.poisoned
                    && session.completion_identity() == self.identity
                    && session.rank() == self.rank
                    && session.expected_epoch() >= self.epoch
                    && session.compute_stream().stream as usize == self.workspace.stream,
                "BF16 completion owner/rank/stream/epoch differs"
            );
            session.synchronize()?;
            ensure!(
                self.workspace.dev.dtoh_sync_copy(&self.workspace.status)? == [0],
                "BF16 matrix finite/padding guard failed"
            );
            Ok(())
        })();
        if result.is_ok() {
            self.checked = true;
        } else {
            self.workspace.poisoned = true;
        }
        result
    }
    pub fn download_checked(&mut self, session: &mut GpuCollectives) -> Result<Vec<bf16>> {
        self.check(session)?;
        let result = self.workspace.dev.dtoh_sync_copy(self.values);
        if result.is_err() {
            self.checked = false;
            self.workspace.poisoned = true;
        }
        Ok(result?)
    }
}
impl Drop for PendingBf16Matrix<'_> {
    fn drop(&mut self) {
        let _ = self.workspace.dev.synchronize();
        if !self.checked {
            self.workspace.poisoned = true;
        }
    }
}
