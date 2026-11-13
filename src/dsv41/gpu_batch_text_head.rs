//! Shared packed native1017 temperature-zero head for <=16 real lanes.
//! It consumes the final collapsed HC stream, normalizes the last position,
//! computes local F32 logits from BF16 checkpoint weights promoted to F32, and
//! requires the qualified TP8 full-score/token-ID collective before publication.
use super::{
    batch_plan::{BatchTicket, PackedSegment},
    collectives::{RoutedLayout, RowKind},
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseOutput, DenseShape, GpuDense},
    gpu_numerics::{DeviceOutput, GpuNumerics},
    gpu_position::PositionKernels,
    gpu_text::VOCAB_SHARD,
    gpu_text_batch::TextBatchKernels,
    gpu_weights::{LoadScope, LoadedWeights},
    weights::RankManifest,
};
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use std::sync::Arc;

#[derive(Default)]
struct SelectionState {
    attempted: bool,
    completion_owner: Option<[u8; 32]>,
    epoch: u64,
    checked: bool,
}
impl SelectionState {
    fn reserve(&mut self) -> Result<()> {
        self.checked = false;
        ensure!(!self.attempted, "packed head selection already attempted");
        self.attempted = true;
        Ok(())
    }
    fn bind(&mut self, owner: [u8; 32], epoch: u64) {
        self.completion_owner = Some(owner);
        self.epoch = epoch;
    }
    fn begin_check(&mut self, owner: [u8; 32], epoch: u64) -> Result<()> {
        self.checked = false;
        ensure!(
            self.attempted && self.completion_owner == Some(owner) && epoch >= self.epoch,
            "packed head completion owner/epoch differs"
        );
        Ok(())
    }
    fn publication(&self) -> Result<()> {
        ensure!(self.checked, "global packed head output was not checked");
        Ok(())
    }
}

pub struct GpuBatchTextHead {
    dev: Arc<CudaDevice>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    stream: usize,
    numerics: GpuNumerics,
    dense: GpuDense,
    position: PositionKernels,
    text: Arc<TextBatchKernels>,
    norm_f32: CudaSlice<f32>,
    init_status: CudaSlice<u32>,
}
impl GpuBatchTextHead {
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        weights: Arc<LoadedWeights>,
        manifest: &RankManifest,
    ) -> Result<Self> {
        manifest.verify_identity()?;
        ensure!(
            weights.scope() == &LoadScope::Global
                && weights.manifest_sha256() == manifest.manifest_sha256
                && manifest.rank < 8,
            "text head source/global rank manifest mismatch"
        );
        ensure!(
            weights.bf16("head.weight")?.len() == VOCAB_SHARD * 5120
                && weights.bf16("norm.weight")?.len() == 5120,
            "text head native BF16 geometry"
        );
        let position = PositionKernels::load(dev, stream)?;
        let mut norm_f32 = dev.alloc_zeros::<f32>(5120)?;
        let mut init_status = dev.alloc_zeros::<u32>(1)?;
        // Fence the conversion before either error teardown or publication.
        let result = position.cast_bf16_f32(
            stream,
            weights.bf16("norm.weight")?,
            &mut norm_f32,
            &mut init_status,
        );
        dev.synchronize()?;
        result?;
        ensure!(
            dev.dtoh_sync_copy(&init_status)? == [0],
            "text head norm conversion failed"
        );
        Ok(Self {
            dev: dev.clone(),
            weights,
            rank: manifest.rank,
            stream: stream.stream as usize,
            numerics: GpuNumerics::load(dev, stream)?,
            dense: GpuDense::load(dev, stream)?,
            position,
            text: TextBatchKernels::load(dev, stream)?,
            norm_f32,
            init_status,
        })
    }
    pub fn prepare<'a>(
        &'a self,
        stream: &'a CudaStream,
        hidden: &CudaSlice<bf16>,
        ticket: &BatchTicket,
    ) -> Result<PendingBatchTextHead<'a>> {
        let rows = ticket.total_rows();
        let lanes = ticket.lanes();
        let selected_rows = selected_rows(ticket)?;
        ensure!(
            stream.stream as usize == self.stream
                && Arc::ptr_eq(&self.dev, &hidden.device())
                && (1..=128).contains(&rows)
                && hidden.len() == rows * 5120,
            "text head input context/stream/shape"
        );
        // Every error fences while previous local GPU owners remain alive,
        // including errors from nested fallible weight accessors.
        macro_rules! queued {
            ($expression:expr) => {
                match (|| $expression)() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.dev.synchronize();
                        return Err(e.into());
                    }
                }
            };
        }
        let mut last = queued!(self.dev.alloc_zeros::<bf16>(lanes * 5120));
        let mut status = queued!(self.dev.alloc_zeros::<u32>(1));
        // Gather only each member's final flattened query row. There is one
        // shared RMS/cast/dense/head launch, not a per-lane head pipeline.
        for (to, &from) in selected_rows.iter().enumerate() {
            let src = *hidden.device_ptr() + (from * 5120 * 2) as u64;
            let dst = *last.device_ptr() + (to * 5120 * 2) as u64;
            let rc = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, 5120 * 2, stream.stream) };
            queued!(if rc == sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(anyhow::anyhow!("packed head row gather failed {rc:?}"))
            });
        }
        let normalized =
            queued!(self
                .numerics
                .rms_norm_1017(stream, &last, &self.norm_f32, lanes, 5120));
        let mut promoted = queued!(self.dev.alloc_zeros::<f32>(lanes * 5120));
        queued!(self.position.cast_bf16_f32(
            stream,
            unsafe { normalized.device_values_unchecked() },
            &mut promoted,
            &mut status
        ));
        let logits = queued!(self.dense.f32_bf16_linear_f32(
            stream,
            &promoted,
            self.weights.bf16("head.weight")?,
            DenseShape {
                rows: lanes,
                outputs: VOCAB_SHARD,
                inner: 5120
            }
        ));
        let mut pair = queued!(self.dev.alloc_zeros::<u32>(lanes * 2));
        queued!(unsafe {
            self.text.maxloc_into(
                stream,
                unsafe { logits.device_values_unchecked() },
                &mut pair,
                &mut status,
                lanes,
                self.rank,
            )
        });
        // Input is not retained, so its last-row copy must finish before this
        // method returns. The remaining owned pipeline also completes here.
        queued!(self.dev.synchronize());
        queued!(normalized.check());
        queued!(logits.check());
        ensure!(
            queued!(self.dev.dtoh_sync_copy(&status)) == [0],
            "text head numeric status failed"
        );
        Ok(PendingBatchTextHead {
            head: self,
            stream,
            lanes,
            pair,
            status,
            normalized,
            logits,
            _last: last,
            _promoted: promoted,
            state: SelectionState::default(),
        })
    }
}
pub struct PendingBatchTextHead<'a> {
    head: &'a GpuBatchTextHead,
    stream: &'a CudaStream,
    lanes: usize,
    pair: CudaSlice<u32>,
    status: CudaSlice<u32>,
    normalized: DeviceOutput<bf16>,
    logits: DenseOutput<f32>,
    _last: CudaSlice<bf16>,
    _promoted: CudaSlice<f32>,
    state: SelectionState,
}
impl PendingBatchTextHead<'_> {
    pub fn enqueue_selection(
        &mut self,
        collectives: &mut GpuCollectives,
        operation: u64,
    ) -> Result<()> {
        self.state.reserve()?;
        ensure!(
            collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.stream.stream,
            "text head selection session mismatch"
        );
        collectives.enqueue_maxloc(
            operation,
            RoutedLayout::new(self.lanes, RowKind::IndependentDecodeLanes)?,
            &mut self.pair,
        )?;
        self.state.bind(
            collectives.completion_identity(),
            collectives.expected_epoch(),
        );
        Ok(())
    }
    pub fn check(&mut self, collectives: &mut GpuCollectives) -> Result<()> {
        self.state.checked = false;
        ensure!(
            collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.stream.stream,
            "text head selection incomplete/wrong session"
        );
        self.state.begin_check(
            collectives.completion_identity(),
            collectives.expected_epoch(),
        )?;
        collectives.synchronize()?;
        self.normalized.check()?;
        self.logits.check()?;
        ensure!(
            self.head.dev.dtoh_sync_copy(&self.status)? == [0]
                && self.head.dev.dtoh_sync_copy(&self.head.init_status)? == [0],
            "text head status failure"
        );
        self.state.checked = true;
        Ok(())
    }
    /// Only the checked, globally selected native token reaches the CPU.
    pub fn selected_tokens(&self) -> Result<Vec<u32>> {
        self.state.publication()?;
        let pair = self.head.dev.dtoh_sync_copy(&self.pair)?;
        ensure!(
            pair.len() == self.lanes * 2,
            "packed global pair count differs"
        );
        pair.chunks_exact(2)
            .map(|p| {
                ensure!(
                    f32::from_bits(p[0]).is_finite() && p[1] < 129280,
                    "invalid packed global logit/token pair"
                );
                Ok(p[1])
            })
            .collect()
    }
}
impl Drop for PendingBatchTextHead<'_> {
    fn drop(&mut self) {
        let _ = self.head.dev.synchronize();
    }
}

fn selected_rows(ticket: &BatchTicket) -> Result<Vec<usize>> {
    selected_segment_rows(ticket.segments(), ticket.total_rows())
}

fn selected_segment_rows(segments: &[PackedSegment], total_rows: usize) -> Result<Vec<usize>> {
    ensure!(
        (1..=16).contains(&segments.len()) && (1..=128).contains(&total_rows),
        "packed head lane/row bound"
    );
    let mut expected = 0usize;
    let mut selected = Vec::with_capacity(segments.len());
    for segment in segments {
        ensure!(
            segment.first_row == expected
                && segment.rows > 0
                && segment.rows == segment.ticket.tokens,
            "packed head row association differs"
        );
        expected = expected
            .checked_add(segment.rows)
            .ok_or_else(|| anyhow::anyhow!("packed head row overflow"))?;
        ensure!(
            expected <= total_rows,
            "packed head row outside hidden extent"
        );
        selected.push(expected - 1);
    }
    ensure!(
        expected == total_rows,
        "packed head does not cover source rows"
    );
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::{
        batch_plan::{BatchCursor, LaneSegment, PackedKind},
        config::Dsv41Config,
        scheduler::LaneKey,
    };
    use uuid::Uuid;

    fn ticket(widths: &[usize]) -> (BatchCursor, BatchTicket) {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let c = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let boot = Uuid::from_u128(214);
        let mut cursor = BatchCursor::new(&c, boot, 16).unwrap();
        let segments = widths
            .iter()
            .enumerate()
            .map(|(i, &tokens)| {
                let key = LaneKey {
                    boot,
                    slot: (5 * i + 7) % 16,
                    generation: 1,
                };
                cursor.reset_lane(key).unwrap();
                LaneSegment {
                    key,
                    start: 0,
                    tokens,
                }
            })
            .collect();
        let ticket = cursor.begin(PackedKind::PrefillSegments, segments).unwrap();
        (cursor, ticket)
    }

    #[test]
    fn ragged_last_rows_preserve_request_order_at_all_native_widths() {
        let cases = vec![
            vec![128],
            vec![75, 53],
            vec![1, 3, 7, 117],
            vec![1, 2, 3, 4, 5, 6, 7, 100],
            (1..=15).chain(std::iter::once(8)).collect(),
        ];
        for widths in cases {
            let (_, t) = ticket(&widths);
            let mut end = 0usize;
            let expected = widths
                .iter()
                .map(|n| {
                    end += n;
                    end - 1
                })
                .collect::<Vec<_>>();
            assert_eq!(selected_rows(&t).unwrap(), expected);
            assert_eq!(t.total_rows(), 128);
            assert_eq!(t.segments()[0].ticket.lane.slot, 7);
            // Simulate the D2D row address using distinguishable row contents.
            for (&selected, segment) in expected.iter().zip(t.segments()) {
                assert_eq!(selected, segment.first_row + segment.rows - 1);
                assert!(selected < t.total_rows());
            }
        }
    }

    #[test]
    fn decode_compaction_selects_each_one_token_row_in_current_order() {
        let (mut cursor, prefill) = ticket(&[75, 53]);
        for layer in 0..40 {
            let plan = cursor.layer(&prefill, layer).unwrap();
            cursor.commit_layer(&plan).unwrap();
        }
        cursor.commit(&prefill).unwrap();
        let segments = prefill
            .segments()
            .iter()
            .rev()
            .map(|s| LaneSegment {
                key: s.ticket.lane,
                start: s.rows,
                tokens: 1,
            })
            .collect();
        let decode = cursor.begin(PackedKind::DecodeLanes, segments).unwrap();
        assert_eq!(selected_rows(&decode).unwrap(), [0, 1]);
        assert_eq!(
            decode.segments()[0].ticket.lane,
            prefill.segments()[1].ticket.lane
        );
    }

    #[test]
    fn malformed_segment_geometry_never_yields_a_row_address() {
        let (_, t) = ticket(&[75, 53]);
        let segments = t.segments().to_vec();
        assert!(selected_segment_rows(&[], 128).is_err());
        for total in [0, 127, 129] {
            assert!(selected_segment_rows(&segments, total).is_err());
        }
        for change in 0..5 {
            let mut bad = segments.clone();
            match change {
                0 => bad[1].first_row -= 1,
                1 => bad[1].rows = 0,
                2 => bad[1].ticket.tokens -= 1,
                3 => {
                    bad[1].rows = usize::MAX;
                    bad[1].ticket.tokens = usize::MAX;
                }
                _ => {
                    bad[1].rows += 1;
                    bad[1].ticket.tokens += 1;
                }
            }
            assert!(selected_segment_rows(&bad, 128).is_err());
        }
    }

    #[test]
    fn any_failed_selection_attempt_or_check_revokes_publication() {
        let owner = [1u8; 32];
        let other = [2u8; 32];
        let mut state = SelectionState::default();
        assert!(state.publication().is_err());
        assert!(state.begin_check(owner, 0).is_err());
        state.reserve().unwrap();
        state.bind(owner, 6);
        state.begin_check(owner, 6).unwrap();
        state.checked = true;
        state.publication().unwrap();
        assert!(state.begin_check(other, 6).is_err());
        assert!(state.publication().is_err());
        state.begin_check(owner, 6).unwrap();
        state.checked = true;
        assert!(state.begin_check(owner, 3).is_err());
        assert!(state.publication().is_err());
        state.begin_check(owner, 6).unwrap();
        state.checked = true;
        assert!(state.reserve().is_err());
        assert!(state.publication().is_err());
    }
}
