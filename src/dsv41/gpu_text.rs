//! Text-only last-position and exact greedy local-max helpers. Sampling with
//! nonzero temperature is deliberately outside this initial comparison path.
use super::{collectives::Maxloc, gpu_module::Module};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use std::{ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_text.ptx");
pub const VOCAB_SHARD: usize = 16160;

/// Torch argmax's first-index tie behavior, with nonfinite logits rejected.
pub fn local_max_reference(logits: &[f32], rank: usize) -> Result<Maxloc> {
    ensure!(
        rank < 8 && logits.len() == VOCAB_SHARD && logits.iter().all(|v| v.is_finite()),
        "native local vocabulary/logit domain"
    );
    let mut id = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[id] {
            id = i;
        }
    }
    Maxloc::new(logits[id], (rank * VOCAB_SHARD + id) as u32)
}

pub struct TextKernels {
    module: Module,
    stream: usize,
}
impl TextKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let module = Module::load(
            dev,
            stream,
            PTX,
            &["dsv41_text_last", "dsv41_text_maxloc"],
            "dsv41_text_identity",
            u64::from_str_radix(env!("DSV41_TEXT_SOURCE_ID"), 16)?,
            1,
        )?;
        Ok(Self {
            module,
            stream: stream.stream as usize,
        })
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.module.check_stream(s)?;
        ensure!(
            self.stream == s.stream as usize,
            "text helper stream differs"
        );
        Ok(())
    }
    pub fn last(
        &self,
        s: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        rows: usize,
    ) -> Result<()> {
        self.stream(s)?;
        ensure!(
            (1..=128).contains(&rows)
                && input.len() == rows * 5120
                && output.len() == 5120
                && status.len() == 1,
            "last hidden row geometry"
        );
        self.module.check_buffer(input)?;
        self.module.check_buffer(output)?;
        self.module.check_buffer(status)?;
        let (mut src, mut dst, mut st, mut n, mut dim) = (
            *input.device_ptr(),
            *output.device_ptr(),
            *status.device_ptr(),
            rows as u32,
            5120u32,
        );
        let mut args = [
            &mut src as *mut _ as *mut c_void,
            &mut dst as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut dim as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module
                .launch("dsv41_text_last", s, (20, 1, 1), (256, 1, 1), 0, &mut args)
        }
    }
    /// Writes the full F32-score/native-token-ID pair for a subsequent qualified
    /// TP8 maxloc collective. Status is sticky and must be checked before use.
    pub fn maxloc(
        &self,
        s: &CudaStream,
        logits: &CudaSlice<f32>,
        pair: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
        rank: usize,
    ) -> Result<()> {
        self.stream(s)?;
        ensure!(
            rank < 8 && logits.len() == VOCAB_SHARD && pair.len() == 2 && status.len() == 1,
            "native vocabulary/maxloc geometry"
        );
        self.module.check_buffer(logits)?;
        self.module.check_buffer(pair)?;
        self.module.check_buffer(status)?;
        let (mut src, mut dst, mut st, mut n, mut first) = (
            *logits.device_ptr(),
            *pair.device_ptr(),
            *status.device_ptr(),
            VOCAB_SHARD as u32,
            (rank * VOCAB_SHARD) as u32,
        );
        let mut args = [
            &mut src as *mut _ as *mut c_void,
            &mut dst as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut first as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module
                .launch("dsv41_text_maxloc", s, (1, 1, 1), (256, 1, 1), 0, &mut args)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_vocabulary_ties_negative_scores_and_last_rank() {
        let mut scores = vec![-3.; VOCAB_SHARD];
        scores[16159] = -1.;
        scores[31] = -1.;
        let v = local_max_reference(&scores, 7).unwrap();
        assert_eq!(v.value(), -1.);
        assert_eq!(v.token_id(), 7 * 16160 + 31);
        scores[31] = f32::NAN;
        assert!(local_max_reference(&scores, 0).is_err());
        assert!(local_max_reference(&scores[..2], 0).is_err());
        assert!(local_max_reference(&vec![0.; VOCAB_SHARD], 8).is_err());
    }
}
