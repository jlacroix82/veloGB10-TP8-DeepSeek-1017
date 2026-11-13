//! PRIVATE CPU cursor actually used by the proposed request transport executor.
//! No method here supplies GPU/model/all-rank completion authority.
use super::{
    gpu_collectives::Operation,
    model_session_plan::{self, CpuRequestPlan, ExpectedBoundary},
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TransportStop {
    NativeEos,
    OutputAllowance,
    DiagnosticPrefix,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TransportPrefixClaim {
    pub boundary: ExpectedBoundary,
    pub stop: TransportStop,
    pub selected_native_id: Option<u32>,
    pub consumed_native_ids_sha256: String,
    pub generated_native_ids_sha256: String,
}
impl TransportPrefixClaim {
    pub fn digest(&self) -> Result<String> {
        let mut h = Sha256::new();
        h.update(b"dsv41-transport-prefix-claim-v1\0");
        h.update(serde_json::to_vec(self)?);
        Ok(format!("{:x}", h.finalize()))
    }
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Clone, Debug)]
pub struct RequestCursor {
    plan: Arc<CpuRequestPlan>,
    next: usize,
    epoch: u64,
    finished: bool,
    last: Option<TransportPrefixClaim>,
}
impl RequestCursor {
    pub fn new(plan: Arc<CpuRequestPlan>, actual_start_epoch: u64) -> Result<Self> {
        ensure!(
            plan.start_epoch() == actual_start_epoch
                && plan.steps().len() <= model_session_plan::MAX_STEPS
                && !plan.operations().is_empty()
                && plan.operations().len() <= model_session_plan::MAX_OPERATIONS,
            "request cursor start/shape bounds differ"
        );
        Ok(Self {
            plan,
            next: 0,
            epoch: actual_start_epoch,
            finished: false,
            last: None,
        })
    }
    pub fn plan(&self) -> &Arc<CpuRequestPlan> {
        &self.plan
    }
    pub fn queued_operations(&self) -> usize {
        self.next
    }
    pub fn expected_epoch(&self) -> u64 {
        self.epoch
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn last_claim(&self) -> Option<&TransportPrefixClaim> {
        self.last.as_ref()
    }
    pub fn expect(&self, op: Operation) -> Result<()> {
        ensure!(
            !self.finished
                && self
                    .plan
                    .operations()
                    .get(self.next)
                    .is_some_and(|p| p.wire == op),
            "operation differs from immutable request cursor"
        );
        Ok(())
    }
    pub fn queued(&mut self, op: Operation) -> Result<()> {
        self.expect(op)?;
        let epochs = match op {
            Operation::RoutedF32 { layout, .. } => layout.chunk_count() * 3,
            Operation::ShardedMaxloc { .. } => 3,
            Operation::MatrixF32 { layout, .. } => layout.chunk_count() * 3,
            Operation::MatrixBf16 { layout, .. } => layout.epochs(),
        };
        self.epoch = self
            .epoch
            .checked_add(epochs as u64)
            .context("request epochoverflow")?;
        self.next += 1;
        Ok(())
    }
    /// Transport-prefix shape validation only; these are declared hashes, not
    /// evidence that a model produced the IDs. The GPU layer checks real status.
    pub fn check_claim(&self, c: &TransportPrefixClaim) -> Result<()> {
        let step = self
            .plan
            .steps()
            .get(c.boundary.step_index)
            .context("prefix step outside request")?;
        ensure!(
            !self.finished
                && c.boundary == step.expected_boundary
                && self.next == step.operation_end
                && self.epoch == step.epoch_after,
            "not the actual complete queued model-step boundary"
        );
        ensure!(
            hash(&c.consumed_native_ids_sha256) && hash(&c.generated_native_ids_sha256),
            "prefix native-ID digest syntax"
        );
        let generated = c.boundary.generated_native_count;
        ensure!(
            c.selected_native_id.is_none_or(|x| x < 129280),
            "selected native ID outside vocabulary"
        );
        if step.output_ordinal.is_none() {
            ensure!(
                c.selected_native_id.is_none(),
                "ignored prefill head cannot be a generated output"
            );
        } else {
            ensure!(
                c.selected_native_id.is_some(),
                "output-producing step requires selected native ID"
            );
        }
        if step.end <= self.plan.input_ids().len() {
            ensure!(
                c.consumed_native_ids_sha256
                    == model_session_plan::native_ids_sha256(&self.plan.input_ids()[..step.end]),
                "prefill consumed native prefix digest differs"
            );
        }
        if generated == 0 {
            ensure!(
                c.generated_native_ids_sha256 == model_session_plan::native_ids_sha256(&[]),
                "intermediate head must not enter generation digest"
            );
        }
        if generated == 1 {
            ensure!(
                c.generated_native_ids_sha256
                    == model_session_plan::native_ids_sha256(&[c
                        .selected_native_id
                        .context("first output missing")?]),
                "first generated-ID digest differs"
            );
        }
        match c.stop {
            TransportStop::NativeEos => ensure!(
                generated > 0 && c.selected_native_id == Some(model_session_plan::EOS_ID),
                "natural EOS requires selected nativeEOS at output boundary"
            ),
            TransportStop::OutputAllowance => ensure!(
                generated == self.plan.binding().budget.output_allowance
                    && c.selected_native_id != Some(model_session_plan::EOS_ID),
                "cap requires full allowance withoutEOS"
            ),
            TransportStop::DiagnosticPrefix => (),
        }
        Ok(())
    }
    /// Called by GPU owner ONLY AFTER checked GPU state and actual all-rank
    /// agreement. This CPU state mutation itself proves neither external fact.
    pub fn mark_after_external_agreement(&mut self, c: &TransportPrefixClaim) -> Result<()> {
        self.check_claim(c)?;
        self.finished = true;
        self.last = Some(c.clone());
        Ok(())
    }
    pub fn check_next(&self, next: &CpuRequestPlan) -> Result<()> {
        ensure!(
            self.finished && self.last.is_some(),
            "prior request prefix not finished"
        );
        let old = self.plan.binding();
        let new = next.binding();
        ensure!(
            next.start_epoch() == self.epoch
                && next.first_operation_id() >= self.plan.next_operation_id()
                && next
                    .steps()
                    .first()
                    .context("newrequest has no step")?
                    .step_id
                    > self.plan.steps().last().unwrap().step_id,
            "nextrequest must preserve actual epoch and skip every reserved old operation/step ID"
        );
        ensure!(
            new.lane.boot == old.lane.boot
                && new.lane.slot == 0
                && new.lane.generation > old.lane.generation,
            "newrequest needs sameboot/newgeneration; prefix reuse unqualified"
        );
        ensure!(
            new.model_revision == old.model_revision
                && new.global_checkpoint_geometry_checksum_sha256
                    == old.global_checkpoint_geometry_checksum_sha256
                && new.execution_snapshot_sha256 == old.execution_snapshot_sha256
                && new.engine_binary_sha256 == old.engine_binary_sha256
                && new.tokenizer_sha256 == old.tokenizer_sha256
                && new.toolkit_revision == old.toolkit_revision
                && new.frequency == old.frequency,
            "residentdeployment/model/tokenizer/frequency changed across request rearm"
        );
        Ok(())
    }
    pub fn rearm_after_external_agreement(&mut self, next: Arc<CpuRequestPlan>) -> Result<()> {
        self.check_next(&next)?;
        *self = Self::new(next, self.epoch)?;
        Ok(())
    }
}
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::dsv41::{
        model_session_plan::{GreedySemantics, RequestBinding, RequestBudget, StopPolicy},
        native_api_contract::TOKENIZER_SHA256,
        position::FrequencyBackend,
        protocol::{MODEL_REVISION, TOOLKIT_REVISION},
        scheduler::LaneKey,
    };
    use uuid::Uuid;
    pub fn plan(
        prompt: usize,
        outputs: usize,
        first_op: u64,
        first_step: u64,
        epoch: u64,
        generation: u64,
    ) -> Arc<CpuRequestPlan> {
        Arc::new(
            CpuRequestPlan::new(
                RequestBinding {
                    model_revision: MODEL_REVISION.into(),
                    global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                    execution_snapshot_sha256: "b".repeat(64),
                    engine_binary_sha256: "c".repeat(64),
                    original_request_sha256: "d".repeat(64),
                    tokenizer_sha256: TOKENIZER_SHA256.into(),
                    toolkit_revision: TOOLKIT_REVISION.into(),
                    frequency: FrequencyBackend::CudaTorch.identity(),
                    sampling: GreedySemantics {
                        temperature: 0.,
                        top_p: 1.,
                        presence_penalty: 0.,
                        frequency_penalty: 0.,
                        seed: Some(42),
                        thinking: false,
                        effective_reasoning_score: None,
                        speculation: false,
                    },
                    stop: StopPolicy {
                        eos_native_id: 1,
                        user_sequences: vec![],
                    },
                    budget: RequestBudget {
                        context_tokens: 8192,
                        output_allowance: outputs,
                        wall_millis: 180000,
                    },
                    lane: LaneKey {
                        boot: Uuid::from_u128(417),
                        slot: 0,
                        generation,
                    },
                },
                vec![17; prompt],
                first_op,
                first_step,
                epoch,
            )
            .unwrap(),
        )
    }
    fn queue_to(c: &mut RequestCursor, index: usize) {
        let end = c.plan.steps()[index].operation_end;
        let ops = c.plan.operations()[c.next..end]
            .iter()
            .map(|x| x.wire)
            .collect::<Vec<_>>();
        for op in ops {
            c.queued(op).unwrap();
        }
    }
    fn claim(
        p: &CpuRequestPlan,
        index: usize,
        stop: TransportStop,
        ids: &[u32],
    ) -> TransportPrefixClaim {
        let b = p.steps()[index].expected_boundary.clone();
        let mut consumed =
            p.input_ids()[..b.consumed_native_context.min(p.input_ids().len())].to_vec();
        if b.consumed_native_context > p.input_ids().len() {
            consumed.extend_from_slice(&ids[..b.consumed_native_context - p.input_ids().len()]);
        }
        TransportPrefixClaim {
            boundary: b,
            stop,
            selected_native_id: ids.last().copied(),
            consumed_native_ids_sha256: model_session_plan::native_ids_sha256(&consumed),
            generated_native_ids_sha256: model_session_plan::native_ids_sha256(ids),
        }
    }
    #[test]
    fn actual_cursor_refuses_partial_wrong_operations_epochs_and_stop_classification() {
        let p = plan(75, 4, 1, 1, 0, 1);
        let mut c = RequestCursor::new(p.clone(), 0).unwrap();
        let good = claim(&p, 0, TransportStop::NativeEos, &[1]);
        assert!(c.check_claim(&good).is_err());
        let ops = p.operations()[..83]
            .iter()
            .map(|x| x.wire)
            .collect::<Vec<_>>();
        for op in ops {
            c.queued(op).unwrap();
        }
        assert!(c.check_claim(&good).is_err());
        let wrong = p.operations()[82].wire;
        assert!(c.queued(wrong).is_err());
        c.queued(p.operations()[83].wire).unwrap();
        c.check_claim(&good).unwrap();
        assert_eq!(c.expected_epoch(), 501);
        for field in 0..5 {
            let mut bad = good.clone();
            match field {
                0 => bad.boundary.epoch += 3,
                1 => bad.boundary.operations_queued -= 1,
                2 => bad.boundary.boundary_sha256 = "0".repeat(64),
                3 => bad.stop = TransportStop::OutputAllowance,
                _ => bad.selected_native_id = Some(2),
            };
            assert!(c.check_claim(&bad).is_err());
        }
        c.mark_after_external_agreement(&good).unwrap();
        assert!(c.queued(p.operations()[84].wire).is_err());
        assert!(c.mark_after_external_agreement(&good).is_err());
    }
    #[test]
    fn early_EOS_then_rearm_preserves_actual_epoch_and_skips_entire_reserved_tail() {
        let p = plan(75, 4096, 1, 1, 0, 1);
        let mut c = RequestCursor::new(p.clone(), 0).unwrap();
        queue_to(&mut c, 0);
        let eos = claim(&p, 0, TransportStop::NativeEos, &[1]);
        c.mark_after_external_agreement(&eos).unwrap();
        assert!(c.expected_epoch() < p.end_epoch());
        let epoch = c.expected_epoch();
        let first = p.next_operation_id();
        let step = p.steps().last().unwrap().step_id + 1;
        let n = plan(10, 2, first, step, epoch, 2);
        c.check_next(&n).unwrap();
        for bad in [
            plan(10, 2, p.operations()[83].wire.id() + 1, step, epoch, 2),
            plan(10, 2, first, step, p.end_epoch(), 2),
            plan(10, 2, first, step, epoch, 1),
        ] {
            assert!(c.check_next(&bad).is_err());
        }
        c.rearm_after_external_agreement(n.clone()).unwrap();
        assert_eq!(c.expected_epoch(), epoch);
        assert_eq!(c.queued_operations(), 0);
        assert!(!c.finished());
        queue_to(&mut c, 1);
        let cap = claim(&n, 1, TransportStop::OutputAllowance, &[2, 3]);
        c.mark_after_external_agreement(&cap).unwrap();
        assert_eq!(c.expected_epoch(), n.end_epoch());
    }
}
