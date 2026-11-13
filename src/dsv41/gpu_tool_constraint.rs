//! Additive native-token allowed set, checked before local greedy/maxloc. This
//! does not change the qualified finite FP32/full-pair TP8 transport.
use super::{
    gpu_module::Module,
    tool_constraint::{AllowedNativeTokens, VOCAB},
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::{ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_tool_constraint.ptx");
pub struct ToolConstraintKernels {
    module: Module,
    stream: Arc<CudaStream>,
}
impl ToolConstraintKernels {
    pub fn load(dev: &Arc<CudaDevice>, s: Arc<CudaStream>) -> Result<Self> {
        let module = Module::load(
            dev,
            &s,
            PTX,
            &["dsv41_tool_masked_maxloc"],
            "dsv41_tool_constraint_identity",
            u64::from_str_radix(env!("DSV41_TOOL_CONSTRAINT_SOURCE_ID"), 16)?,
            1,
        )?;
        Ok(Self { module, stream: s })
    }
    /// The caller retains all buffers and this module through its checked fence.
    pub fn enqueue(
        &self,
        s: &CudaStream,
        logits: &CudaSlice<f32>,
        allowed: &CudaSlice<u32>,
        pair: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
        rank: usize,
    ) -> Result<()> {
        self.module.check_stream(s)?;
        ensure!(
            s.stream == self.stream.stream
                && rank < 8
                && logits.len() == 16160
                && allowed.len() == VOCAB.div_ceil(32)
                && pair.len() == 2
                && status.len() == 1,
            "constrained local head stream/rank/shape differs"
        );
        self.module.check_buffer(logits)?;
        self.module.check_buffer(allowed)?;
        self.module.check_buffer(pair)?;
        self.module.check_buffer(status)?;
        let (mut lp, mut ap, mut pp, mut sp, mut count, mut first, mut words) = (
            *logits.device_ptr(),
            *allowed.device_ptr(),
            *pair.device_ptr(),
            *status.device_ptr(),
            16160u32,
            (rank * 16160) as u32,
            (VOCAB.div_ceil(32)) as u32,
        );
        let mut args = [
            &mut lp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut count as *mut _ as *mut c_void,
            &mut first as *mut _ as *mut c_void,
            &mut words as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module.launch(
                "dsv41_tool_masked_maxloc",
                s,
                (1, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }
}
pub fn checked_final_membership(mask: &AllowedNativeTokens, pair: &[u32]) -> Result<u32> {
    ensure!(
        pair.len() == 2
            && f32::from_bits(pair[0]).is_finite()
            && f32::from_bits(pair[0]) > f32::MIN
            && mask.allows(pair[1]),
        "global constrained winner invalid, bottom identity won, or native ID was not allowed"
    );
    Ok(pair[1])
}

pub(crate) fn checked_execution(
    mask: &AllowedNativeTokens,
    selected: u32,
) -> super::tool_constraint::ConstraintExecution {
    use sha2::{Digest, Sha256};
    super::tool_constraint::ConstraintExecution {
        mask: mask.identity().clone(),
        sampler_source_id_hex: env!("DSV41_TOOL_CONSTRAINT_SOURCE_ID").into(),
        sampler_ptx_sha256: format!("{:x}", Sha256::digest(PTX.as_bytes())),
        selected_native_id: selected,
        local_mask_kernel_status_checked: true,
        global_selected_allowed: true,
    }
}
