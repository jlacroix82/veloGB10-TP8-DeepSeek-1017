//! PRIVATE narrow source-order decode shim. It reuses the unchanged qualified
//! original KV-index kernel; no numeric CUDA or472 predecessor is modified.
//! This entry is ONLY for an explicitly declared one-token DECODE while the
//!128-slot window is unfilled. Prefill needs its own full-prompt width policy.
use super::{chunk_plan::LayerChunk, gpu_module::Module};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::{ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_kv_store.ptx");
pub struct SourceDecodeIndices {
    module: Module,
    stream: Arc<CudaStream>,
}
pub fn applicable(plan: &LayerChunk) -> bool {
    plan.ticket.tokens == 1 && (1..127).contains(&plan.ticket.start)
}
pub fn source_decode_window(start: usize) -> Result<Vec<i32>> {
    ensure!(
        (1..127).contains(&start),
        "source decode shim requires unfilled positive one-token window"
    );
    Ok((0..128)
        .map(|j| {
            let physical = (start + 1 + j) % 128;
            if physical <= start {
                physical as i32
            } else {
                -1
            }
        })
        .collect())
}
fn shape(plan: &LayerChunk) -> Result<(usize, usize)> {
    ensure!(
        applicable(plan)
            && plan.window.snapshot_global_base == 0
            && plan.window.snapshot_rows == plan.ticket.start + 1
            && plan.window.query_capacity == 128,
        "explicit source decode/snapshot shape differs"
    );
    let n = plan.compressed.as_ref().map_or(0, |c| c.stored_rows);
    ensure!(
        n <= 127
            && !plan.index_scoring_required
            && matches!(plan.ratio, 0 | 1 | 2)
            && (plan.ratio != 0 || n == 0),
        "early decode must select every compressed position exactly"
    );
    Ok((128 + n, plan.window.snapshot_rows + n))
}
impl SourceDecodeIndices {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            module: Module::load(
                dev,
                &stream,
                PTX,
                &["dsv41_kv_store_indices"],
                "dsv41_kv_store_identity",
                u64::from_str_radix(
                    env!("DSV41_KV_STORE_SOURCE_ID").trim_start_matches("0x"),
                    16,
                )?,
                1,
            )?,
            stream,
        })
    }
    /// The original kernel lists physical ring slots. Before slot127, those
    /// slots equal global positions and the chronological snapshot offsets.
    /// Its compressed base is supplied as the NEW physical snapshot row count;
    /// the LOGICAL window width remains128, preserving every64-key tile slot.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        plan: &LayerChunk,
        out: &mut CudaSlice<i32>,
    ) -> Result<()> {
        let (topk, _) = shape(plan)?;
        ensure!(
            stream.stream == self.stream.stream,
            "source decode index stream differs from retained owner"
        );
        self.module.check_buffer(out)?;
        ensure!(
            out.len() == topk,
            "source decode index destination size differs"
        );
        let (
            mut op,
            mut start,
            mut tokens,
            mut ratio,
            mut physical_rows,
            mut logical_width,
            mut compressed,
        ) = (
            *out.device_ptr(),
            plan.ticket.start as u64,
            1u64,
            plan.ratio as u64,
            plan.window.snapshot_rows as u64,
            128u64,
            plan.compressed.as_ref().map_or(0, |c| c.stored_rows) as u64,
        );
        let mut args = [
            &mut op as *mut _ as *mut c_void,
            &mut start as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
            &mut ratio as *mut _ as *mut c_void,
            &mut physical_rows as *mut _ as *mut c_void,
            &mut logical_width as *mut _ as *mut c_void,
            &mut compressed as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module.launch(
                "dsv41_kv_store_indices",
                stream,
                (topk.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_leading_mask_keeps_native_tile_membership_at_actual_divergence() {
        let row = source_decode_window(86).unwrap();
        assert_eq!(row[..41], [-1; 41]);
        assert_eq!(row[41..], (0..=86).collect::<Vec<_>>());
        assert_eq!(row[..64].iter().filter(|i| **i >= 0).count(), 23);
        assert_eq!(row[64..].iter().filter(|i| **i >= 0).count(), 64);
        let mut predecessor = (0..=86).collect::<Vec<_>>();
        predecessor.resize(128, -1);
        assert_ne!(row, predecessor);
        assert!(source_decode_window(0).is_err());
        assert!(source_decode_window(127).is_err());
    }
    #[test]
    fn exact_source_AST_vectors_cover_every_unfilled_decode_and_ratio() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_source_decode_indices.json"
        ))
        .unwrap();
        assert_eq!(
            f["reference_model_sha256"],
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        let cases = f["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 378);
        for c in cases {
            let start = c["start"].as_u64().unwrap() as usize;
            let ratio = c["ratio"].as_u64().unwrap() as usize;
            let n = if ratio == 0 { 0 } else { (start + 1) / ratio };
            let mut expected = source_decode_window(start).unwrap();
            expected.extend((0..n).map(|i| (start + 1 + i) as i32));
            assert_eq!(
                serde_json::to_value(expected).unwrap(),
                c["chronological_snapshot_indices"]
            );
        }
    }
}
