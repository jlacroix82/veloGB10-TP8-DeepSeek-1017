//! Isolated B1..16 native16160-vocabulary greedy helper.
//! Qualified C1 per-row reduction/tie association is unchanged; this is not a model.
use super::{collectives::Maxloc, gpu_module::Module};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::Arc};
pub const VOCAB_SHARD: usize = 16160;
pub const MAX_LANES: usize = 16;
pub const PTX: &str = include_str!("../ptx/dsv41_text_batch.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_text_batch.cu");
pub struct TextBatchKernels {
    dev: Arc<CudaDevice>,
    module: Module,
    stream_id: usize,
    source_id: u64,
}
impl TextBatchKernels {
    pub fn load(dev: &Arc<CudaDevice>, s: &CudaStream) -> Result<Arc<Self>> {
        let source_id = u64::from_str_radix(
            env!("DSV41_TEXT_BATCH_SOURCE_ID").trim_start_matches("0x"),
            16,
        )?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            module: Module::load(
                dev,
                s,
                PTX,
                &["dsv41_text_batch_maxloc"],
                "dsv41_text_batch_identity",
                source_id,
                1,
            )?,
            stream_id: s.stream as usize,
            source_id,
        }))
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
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.module.check_stream(s)?;
        ensure!(
            self.stream_id == s.stream as usize,
            "exact batched-text owner stream required"
        );
        Ok(())
    }
    /// Internal composition primitive. Every buffer owner must remain alive until
    /// stream completion; no partial row may be published when global status!=0.
    pub unsafe fn maxloc_into(
        &self,
        s: &CudaStream,
        logits: &CudaSlice<f32>,
        pairs: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
        rows: usize,
        rank: usize,
    ) -> Result<()> {
        self.stream(s)?;
        for valid in [
            self.module.check_buffer(logits),
            self.module.check_buffer(pairs),
            self.module.check_buffer(status),
        ] {
            valid?;
        }
        ensure!(
            (1..=MAX_LANES).contains(&rows)
                && rank < 8
                && logits.len() == rows * VOCAB_SHARD
                && pairs.len() == rows * 2
                && status.len() == 1,
            "native batched text shape/rank differs"
        );
        let (mut lp, mut pp, mut sp, mut count, mut first) = (
            *logits.device_ptr(),
            *pairs.device_ptr(),
            *status.device_ptr(),
            VOCAB_SHARD as u32,
            (rank * VOCAB_SHARD) as u32,
        );
        let mut args = [
            &mut lp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut count as *mut _ as *mut c_void,
            &mut first as *mut _ as *mut c_void,
        ];
        self.module.launch(
            "dsv41_text_batch_maxloc",
            s,
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }
    pub fn local_pairs<'a>(
        self: &Arc<Self>,
        s: &'a CudaStream,
        logits: &'a CudaSlice<f32>,
        rows: usize,
        rank: usize,
    ) -> Result<PendingPairs<'a>> {
        self.stream(s)?;
        self.module.check_buffer(logits)?;
        ensure!(
            (1..=MAX_LANES).contains(&rows) && rank < 8 && logits.len() == rows * VOCAB_SHARD,
            "native batched text input shape/rank differs before allocation"
        );
        let mut pairs = self
            .dev
            .alloc_zeros::<u32>(rows.checked_mul(2).unwrap_or(0))?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        let result = unsafe { self.maxloc_into(s, logits, &mut pairs, &mut status, rows, rank) };
        if let Err(error) = result {
            let _ = self.dev.synchronize();
            return Err(error);
        }
        Ok(PendingPairs {
            kernel: self.clone(),
            _s: s,
            _logits: logits,
            pairs,
            status,
            rows,
            rank,
        })
    }
}
pub struct PendingPairs<'a> {
    kernel: Arc<TextBatchKernels>,
    _s: &'a CudaStream,
    _logits: &'a CudaSlice<f32>,
    pairs: CudaSlice<u32>,
    status: CudaSlice<u32>,
    rows: usize,
    rank: usize,
}
impl PendingPairs<'_> {
    pub fn status_code(&self) -> Result<u32> {
        self.kernel.dev.synchronize()?;
        Ok(self.kernel.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let flag = self.status_code()?;
        ensure!(
            flag == 0,
            "batched greedy status0x{flag:x} (1=nonfinite,2=no finite candidate)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<Maxloc>> {
        self.check()?;
        let pairs = self.kernel.dev.dtoh_sync_copy(&self.pairs)?;
        ensure!(pairs.len() == self.rows * 2, "pair row count differs");
        pairs
            .chunks_exact(2)
            .map(|p| {
                ensure!(
                    (self.rank * VOCAB_SHARD..(self.rank + 1) * VOCAB_SHARD)
                        .contains(&(p[1] as usize)),
                    "local nativeID belongs to another rank"
                );
                Maxloc::new(f32::from_bits(p[0]), p[1])
            })
            .collect()
    }
}
impl Drop for PendingPairs<'_> {
    fn drop(&mut self) {
        let _ = self.kernel.dev.synchronize();
    }
}
