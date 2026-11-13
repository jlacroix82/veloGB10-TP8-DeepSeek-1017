//! Candidate/top-k GPU primitive for source BF16 index scores. This component
//! does not implement index projections, BF16 reduction, a KV store or a model.
//! Outputs retain all queued source owners and use the fixed source budgets.
use super::gpu_module::Module;
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_index_select.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_index_select.cu");
pub const MAX_COLUMNS: usize = 131072;
pub const MAX_QUERIES: usize = 128;
pub const TOPK: usize = 512;

pub fn validate_shape(queries: usize, columns: usize, offset: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_QUERIES).contains(&queries) && (1..=MAX_COLUMNS).contains(&columns),
        "bounded BF16 selection matrix"
    );
    ensure!(
        offset <= MAX_COLUMNS
            && offset
                .checked_add(columns)
                .is_some_and(|n| n <= i32::MAX as usize),
        "selection index offset overflow"
    );
    Ok(queries * columns.min(TOPK))
}
pub struct IndexSelection {
    dev: Arc<CudaDevice>,
    module: Module,
    stream: usize,
}
impl IndexSelection {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let module = Module::load(
            dev,
            stream,
            PTX,
            &["dsv41_candidate_blocks", "dsv41_topk_positions"],
            "dsv41_index_select_identity",
            u64::from_str_radix(env!("DSV41_INDEX_SELECT_SOURCE_ID"), 16)?,
            1,
        )?;
        Ok(Self {
            dev: dev.clone(),
            module,
            stream: stream.stream as usize,
        })
    }
    pub fn source_id_hex(&self) -> &'static str {
        env!("DSV41_INDEX_SELECT_SOURCE_ID")
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    fn inputs(
        &self,
        s: &CudaStream,
        scores: &CudaSlice<bf16>,
        visible: &CudaSlice<u32>,
        status: &CudaSlice<u32>,
        queries: usize,
        columns: usize,
    ) -> Result<()> {
        validate_shape(queries, columns, 0)?;
        self.module.check_stream(s)?;
        ensure!(
            self.stream == s.stream as usize,
            "selection stream owner differs"
        );
        ensure!(
            scores.len() == queries * columns && visible.len() == queries && status.len() == 1,
            "selection input extent"
        );
        self.module.check_buffer(scores)?;
        self.module.check_buffer(visible)?;
        self.module.check_buffer(status)?;
        Ok(())
    }
    pub fn candidates<'a>(
        &'a self,
        s: &'a CudaStream,
        scores: &'a CudaSlice<bf16>,
        visible: &'a CudaSlice<u32>,
        output: &'a mut CudaSlice<u8>,
        status: &'a mut CudaSlice<u32>,
        queries: usize,
        columns: usize,
    ) -> Result<PendingCandidates<'a>> {
        self.inputs(s, scores, visible, status, queries, columns)?;
        self.module.check_buffer(output)?;
        ensure!(output.len() == queries * columns, "candidate output extent");
        let (mut sp, mut vp, mut op, mut ep, mut n) = (
            *scores.device_ptr(),
            *visible.device_ptr(),
            *output.device_ptr(),
            *status.device_ptr(),
            columns as u32,
        );
        let mut args = [
            &mut sp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let result = unsafe {
            self.module.launch(
                "dsv41_candidate_blocks",
                s,
                (queries as u32, 1, 1),
                (512, 1, 1),
                0,
                &mut args,
            )
        };
        if result.is_err() {
            let _ = self.dev.synchronize();
        }
        result?;
        Ok(PendingCandidates {
            owner: self,
            _stream: s,
            _scores: scores,
            _visible: visible,
            output,
            status,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn positions<'a>(
        &'a self,
        s: &'a CudaStream,
        scores: &'a CudaSlice<bf16>,
        visible: &'a CudaSlice<u32>,
        candidates: Option<&'a CudaSlice<u8>>,
        output: &'a mut CudaSlice<i32>,
        status: &'a mut CudaSlice<u32>,
        queries: usize,
        columns: usize,
        offset: usize,
    ) -> Result<PendingPositions<'a>> {
        let expected = validate_shape(queries, columns, offset)?;
        self.inputs(s, scores, visible, status, queries, columns)?;
        self.module.check_buffer(output)?;
        ensure!(output.len() == expected, "top512 output extent");
        if let Some(mask) = candidates {
            self.module.check_buffer(mask)?;
            ensure!(mask.len() == scores.len(), "candidate mask extent");
        }
        let (mut sp, mut vp, mut mp, mut op, mut ep, mut n, mut shift) = (
            *scores.device_ptr(),
            *visible.device_ptr(),
            candidates.map(|m| *m.device_ptr()).unwrap_or(0),
            *output.device_ptr(),
            *status.device_ptr(),
            columns as u32,
            offset as u32,
        );
        let mut args = [
            &mut sp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut shift as *mut _ as *mut c_void,
        ];
        let result = unsafe {
            self.module.launch(
                "dsv41_topk_positions",
                s,
                (queries as u32, 1, 1),
                (512, 1, 1),
                0,
                &mut args,
            )
        };
        if result.is_err() {
            let _ = self.dev.synchronize();
        }
        result?;
        Ok(PendingPositions {
            owner: self,
            _stream: s,
            _scores: scores,
            _visible: visible,
            _candidates: candidates,
            output,
            status,
        })
    }
    fn check(&self, status: &CudaSlice<u32>) -> Result<()> {
        self.dev.synchronize()?;
        let bits = self.dev.dtoh_sync_copy(status)?[0];
        ensure!(
            bits == 0,
            "index selection status{bits}:1=nonfinite,2=extent,4=inconsistentcandidatepolicy"
        );
        Ok(())
    }
}
pub struct PendingCandidates<'a> {
    owner: &'a IndexSelection,
    _stream: &'a CudaStream,
    _scores: &'a CudaSlice<bf16>,
    _visible: &'a CudaSlice<u32>,
    output: &'a mut CudaSlice<u8>,
    status: &'a mut CudaSlice<u32>,
}
impl PendingCandidates<'_> {
    pub fn check(&self) -> Result<()> {
        self.owner.check(self.status)
    }
    pub fn download_checked(&self) -> Result<Vec<u8>> {
        self.check()?;
        Ok(self.owner.dev.dtoh_sync_copy(self.output)?)
    }
    /// Retain this producer and check it before publishing any downstream use.
    pub unsafe fn device_mask_unchecked(&self) -> &CudaSlice<u8> {
        self.output
    }
}
impl Drop for PendingCandidates<'_> {
    fn drop(&mut self) {
        let _ = self.owner.dev.synchronize();
    }
}
pub struct PendingPositions<'a> {
    owner: &'a IndexSelection,
    _stream: &'a CudaStream,
    _scores: &'a CudaSlice<bf16>,
    _visible: &'a CudaSlice<u32>,
    _candidates: Option<&'a CudaSlice<u8>>,
    output: &'a mut CudaSlice<i32>,
    status: &'a mut CudaSlice<u32>,
}
impl PendingPositions<'_> {
    pub fn check(&self) -> Result<()> {
        self.owner.check(self.status)
    }
    pub fn download_checked(&self) -> Result<Vec<i32>> {
        self.check()?;
        Ok(self.owner.dev.dtoh_sync_copy(self.output)?)
    }
    /// Retain this owner and check the sticky status before consuming indices.
    pub unsafe fn device_indices_unchecked(&self) -> &CudaSlice<i32> {
        self.output
    }
}
impl Drop for PendingPositions<'_> {
    fn drop(&mut self) {
        let _ = self.owner.dev.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_is_fixed_to_source_index_budgets() {
        assert_eq!(validate_shape(128, 131072, 128).unwrap(), 65536);
        assert_eq!(validate_shape(1, 1, 0).unwrap(), 1);
        for (q, n, o) in [
            (0, 1, 0),
            (129, 1, 0),
            (1, 0, 0),
            (1, 131073, 0),
            (1, 1, usize::MAX),
        ] {
            assert!(validate_shape(q, n, o).is_err());
        }
    }
}
