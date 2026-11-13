//! Actual-global-weight native1017 greedy head for independent decode lanes.
//! Per-lane BF16 RMS -> F32 promotion -> source F32 projection with stored BF16
//! weights -> [score_bits,native_ID] pairs. No TF32/FP16 substitute or sampler.
//! This component does not qualify a full B-lane model/agent serving runtime.
use super::{
    collectives::{RoutedLayout, RowKind},
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseOutput, DenseShape, GpuDense},
    gpu_numerics::{DeviceOutput, GpuNumerics},
    gpu_position::PositionKernels,
    gpu_text_batch::{TextBatchKernels, MAX_LANES, VOCAB_SHARD},
    gpu_weights::{LoadScope, LoadedWeights},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::{collections::BTreeSet, sync::Arc};
use uuid::Uuid;
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct DecodeLane {
    pub key: LaneKey,
    pub prefix_tokens: usize,
}
pub fn validate_lanes(lanes: &[DecodeLane], boot: Uuid) -> Result<()> {
    ensure!(
        (1..=MAX_LANES).contains(&lanes.len())
            && lanes.iter().all(|l| l.key.boot == boot
                && l.key.generation > 0
                && l.key.slot < MAX_LANES
                && (1..=1_048_576).contains(&l.prefix_tokens))
            && lanes
                .iter()
                .map(|l| l.key.slot)
                .collect::<BTreeSet<_>>()
                .len()
                == lanes.len(),
        "batched greedy lanes require unique live slots, same boot and valid native prefix lengths"
    );
    Ok(())
}
#[derive(Default)]
struct SelectionState {
    attempted: bool,
    owner: Option<[u8; 32]>,
    epoch: u64,
    checked: bool,
}
impl SelectionState {
    fn invalidate(&mut self) {
        self.checked = false;
    }
    fn reserve(&mut self) -> Result<()> {
        self.invalidate();
        ensure!(!self.attempted, "batched greedy selection already enqueued");
        self.attempted = true;
        Ok(())
    }
    fn bind(&mut self, owner: [u8; 32], epoch: u64) {
        self.owner = Some(owner);
        self.epoch = epoch;
        self.checked = false;
    }
    fn begin_check(&mut self, owner: [u8; 32], epoch: u64) -> Result<()> {
        self.invalidate();
        ensure!(
            self.owner == Some(owner) && epoch >= self.epoch,
            "batched greedy completion owner/epoch differs"
        );
        Ok(())
    }
    fn publication(&self) -> Result<()> {
        ensure!(
            self.checked,
            "batched greedy output has not passed current completion/numeric checks"
        );
        Ok(())
    }
}
pub struct GpuTextHeadBatch {
    dev: Arc<CudaDevice>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    boot: Uuid,
    stream: usize,
    numerics: GpuNumerics,
    dense: GpuDense,
    position: PositionKernels,
    text: Arc<TextBatchKernels>,
    norm_f32: CudaSlice<f32>,
    init_status: CudaSlice<u32>,
    manifest_sha256: String,
}
impl GpuTextHeadBatch {
    pub fn new(
        dev: &Arc<CudaDevice>,
        s: &CudaStream,
        boot: Uuid,
        weights: Arc<LoadedWeights>,
        manifest: &RankManifest,
    ) -> Result<Self> {
        manifest.verify_identity()?;
        ensure!(
            weights.scope() == &LoadScope::Global
                && weights.manifest_sha256() == manifest.manifest_sha256
                && manifest.rank < 8,
            "batched head requires exact actual global-weight scope/manifest"
        );
        ensure!(
            weights.bf16("head.weight")?.len() == VOCAB_SHARD * 5120
                && weights.bf16("norm.weight")?.len() == 5120,
            "batched head native BF16 geometry"
        );
        let position = PositionKernels::load(dev, s)?;
        let mut norm_f32 = dev.alloc_zeros::<f32>(5120)?;
        let mut init_status = dev.alloc_zeros::<u32>(1)?;
        let conversion = position.cast_bf16_f32(
            s,
            weights.bf16("norm.weight")?,
            &mut norm_f32,
            &mut init_status,
        );
        dev.synchronize()?;
        conversion?;
        ensure!(
            dev.dtoh_sync_copy(&init_status)? == [0],
            "batched head norm initialization failed"
        );
        Ok(Self {
            dev: dev.clone(),
            weights,
            rank: manifest.rank,
            boot,
            stream: s.stream as usize,
            numerics: GpuNumerics::load(dev, s)?,
            dense: GpuDense::load(dev, s)?,
            position,
            text: TextBatchKernels::load(dev, s)?,
            norm_f32,
            init_status,
            manifest_sha256: manifest.manifest_sha256.clone(),
        })
    }
    /// Each hidden row is the FINAL collapsed HC vector of one decode lane.
    /// This is not multi-token prefill. Input copy/use is fenced before return.
    pub fn prepare_decode<'a>(
        &'a self,
        s: &'a CudaStream,
        hidden: &CudaSlice<bf16>,
        lanes: Vec<DecodeLane>,
    ) -> Result<PendingTextHeadBatch<'a>> {
        validate_lanes(&lanes, self.boot)?;
        let rows = lanes.len();
        ensure!(
            s.stream as usize == self.stream
                && Arc::ptr_eq(&self.dev, &hidden.device())
                && hidden.len() == rows * 5120,
            "batched head input stream/context/shape differs"
        );
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=self.dev.synchronize();return Err(e.into());}}};}
        let mut status = queued!(self.dev.alloc_zeros::<u32>(1));
        queued!(self.dev.memset_zeros(&mut status));
        let normalized =
            queued!(self
                .numerics
                .rms_norm_1017(s, hidden, &self.norm_f32, rows, 5120));
        let mut promoted = queued!(self.dev.alloc_zeros::<f32>(rows * 5120));
        queued!(self.position.cast_bf16_f32(
            s,
            unsafe { normalized.device_values_unchecked() },
            &mut promoted,
            &mut status
        ));
        let logits = queued!(self.dense.f32_bf16_linear_f32(
            s,
            &promoted,
            self.weights.bf16("head.weight")?,
            DenseShape {
                rows,
                outputs: VOCAB_SHARD,
                inner: 5120
            }
        ));
        let mut pairs = queued!(self.dev.alloc_zeros::<u32>(rows * 2));
        queued!(unsafe {
            self.text.maxloc_into(
                s,
                logits.device_values_unchecked(),
                &mut pairs,
                &mut status,
                rows,
                self.rank,
            )
        });
        queued!(self.dev.synchronize());
        queued!(normalized.check());
        queued!(logits.check());
        ensure!(
            queued!(self.dev.dtoh_sync_copy(&status)) == [0]
                && queued!(self.dev.dtoh_sync_copy(&self.init_status)) == [0],
            "batched head local numeric status failed; no partial lanes published"
        );
        Ok(PendingTextHeadBatch {
            head: self,
            s,
            lanes,
            pairs,
            status,
            normalized,
            logits,
            promoted,
            state: SelectionState::default(),
        })
    }
}
#[derive(Debug, Serialize)]
pub struct LaneSelection {
    pub lane: DecodeLane,
    pub selected_native_token: u32,
    pub global_logit_f32_bits: u32,
}
#[derive(Debug, Serialize)]
pub struct HeadBatchDiagnostics {
    pub normalized_bf16_bits: Vec<u16>,
    pub promoted_f32_bits: Vec<u32>,
    pub logits_f32_bits: Vec<u32>,
    pub local_pairs: Vec<[u32; 2]>,
}
pub struct PendingTextHeadBatch<'a> {
    head: &'a GpuTextHeadBatch,
    s: &'a CudaStream,
    lanes: Vec<DecodeLane>,
    pairs: CudaSlice<u32>,
    status: CudaSlice<u32>,
    normalized: DeviceOutput<bf16>,
    logits: DenseOutput<f32>,
    promoted: CudaSlice<f32>,
    state: SelectionState,
}
impl PendingTextHeadBatch<'_> {
    pub fn lanes(&self) -> &[DecodeLane] {
        &self.lanes
    }
    pub fn enqueue_selection(
        &mut self,
        collectives: &mut GpuCollectives,
        operation: u64,
    ) -> Result<()> {
        self.state.reserve()?;
        ensure!(
            collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.s.stream,
            "batched head selection rank/stream differs"
        );
        collectives.enqueue_maxloc(
            operation,
            RoutedLayout::new(self.lanes.len(), RowKind::IndependentDecodeLanes)?,
            &mut self.pairs,
        )?;
        self.state.bind(
            collectives.completion_identity(),
            collectives.expected_epoch(),
        );
        Ok(())
    }
    pub fn check(&mut self, collectives: &mut GpuCollectives) -> Result<()> {
        self.state.invalidate();
        ensure!(
            collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.s.stream,
            "batched head completion rank/stream differs"
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
            "batched head numeric status failed; every lane withheld"
        );
        self.state.checked = true;
        Ok(())
    }
    pub fn selected_tokens(&self) -> Result<Vec<LaneSelection>> {
        self.state.publication()?;
        let pairs = self.head.dev.dtoh_sync_copy(&self.pairs)?;
        ensure!(
            pairs.len() == self.lanes.len() * 2
                && pairs
                    .chunks_exact(2)
                    .all(|p| f32::from_bits(p[0]).is_finite() && p[1] < 129280),
            "invalid global batched native pair vector"
        );
        Ok(self
            .lanes
            .iter()
            .copied()
            .zip(pairs.chunks_exact(2))
            .map(|(lane, p)| LaneSelection {
                lane,
                selected_native_token: p[1],
                global_logit_f32_bits: p[0],
            })
            .collect())
    }
    /// Explicit untimed oracle readback only, BEFORE global maxloc mutates pairs.
    pub fn local_diagnostics(&self) -> Result<HeadBatchDiagnostics> {
        ensure!(
            !self.state.attempted,
            "local diagnostics requested after global pair mutation"
        );
        self.head.dev.synchronize()?;
        self.normalized.check()?;
        self.logits.check()?;
        ensure!(
            self.head.dev.dtoh_sync_copy(&self.status)? == [0],
            "local batched head status failed"
        );
        let raw = self.head.dev.dtoh_sync_copy(&self.pairs)?;
        Ok(HeadBatchDiagnostics {
            normalized_bf16_bits: self
                .normalized
                .download_checked()?
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            promoted_f32_bits: self
                .head
                .dev
                .dtoh_sync_copy(&self.promoted)?
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            logits_f32_bits: self
                .logits
                .download_checked()?
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            local_pairs: raw.chunks_exact(2).map(|p| [p[0], p[1]]).collect(),
        })
    }
    pub fn identity(&self) -> serde_json::Value {
        serde_json::json!({"rank":self.head.rank,"rows":self.lanes.len(),"lanes":self.lanes,"manifest_sha256":self.head.manifest_sha256,"text_batch_source_id":format!("{:016x}",self.head.text.source_id()),"text_batch_ptx_sha256":self.head.text.ptx_sha256(),"dense_source_id":format!("{:016x}",self.head.dense.source_id()),"dense_ptx_sha256":self.head.dense.ptx_sha256(),"numerics_source_id":format!("{:016x}",self.head.numerics.source_id()),"numerics_ptx_sha256":self.head.numerics.ptx_sha256(),"native_logits_per_rank":VOCAB_SHARD,"projection":"source F32 from exact promoted BF16 operands; no TF32/FP16","whole_model_batch_qualified":false,"serving_enabled":false})
    }
}
impl Drop for PendingTextHeadBatch<'_> {
    fn drop(&mut self) {
        let _ = self.head.dev.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lane_identity_and_row_association_reject_duplicates_and_stale_boot() {
        let boot = Uuid::new_v4();
        let lane = DecodeLane {
            key: LaneKey {
                boot,
                slot: 7,
                generation: 1,
            },
            prefix_tokens: 331,
        };
        validate_lanes(&[lane], boot).unwrap();
        assert!(validate_lanes(&[lane, lane], boot).is_err());
        assert!(validate_lanes(&[lane], Uuid::new_v4()).is_err());
        for bad in [
            DecodeLane {
                prefix_tokens: 0,
                ..lane
            },
            DecodeLane {
                prefix_tokens: 1_048_577,
                ..lane
            },
            DecodeLane {
                key: LaneKey {
                    generation: 0,
                    ..lane.key
                },
                ..lane
            },
        ] {
            assert!(validate_lanes(&[bad], boot).is_err());
        }
    }
    #[test]
    fn late_failed_completion_revokes_previous_publication() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        let mut state = SelectionState::default();
        assert!(state.publication().is_err());
        state.reserve().unwrap();
        state.bind(a, 6);
        state.begin_check(a, 6).unwrap();
        state.checked = true;
        state.publication().unwrap();
        assert!(state.begin_check(b, 6).is_err());
        assert!(state.publication().is_err());
        state.begin_check(a, 6).unwrap();
        assert!(state.publication().is_err());
        state.checked = true;
        assert!(state.begin_check(a, 3).is_err());
        assert!(state.publication().is_err());
        assert!(state.reserve().is_err());
        assert!(state.publication().is_err());
    }
}
