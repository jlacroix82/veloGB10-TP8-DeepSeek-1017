//! Resident atomic-batch CPU schedule. This module grants neither execution nor
//! terminal authority. Native EOS changes the next agreed packed program; it is
//! never replaced by padding, forced EOS, shortened allowances or C1 fan-out.
use super::{
    atomic_request_admission::{CpuOwnedAtomicGroup, CpuSlotBook},
    batch_plan::PackedKind,
    gpu_batch_model::{BatchLaneInput, BatchModelOperationIds, MAX_LEARNED_PREFIX},
    gpu_collectives::Operation,
    model_session_plan::{native_ids_sha256, RequestBinding},
    native_api_contract::{EOS_ID, TOKENIZER_SHA256},
    position::FrequencyBackend,
    protocol::{MODEL_REVISION, TOOLKIT_REVISION},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Instant};
pub const MAX_MEMBERS: usize = 16;
pub const MAX_OUTPUT: usize = 256;
pub const MAX_PHASES: usize = 17;
pub const OPS_PER_STEP: usize = 84;
pub const MAX_DEVICE_EPOCH: u64 = (1u64 << 44) - 1;
pub fn digest<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(serde_json::to_vec(value)?);
    Ok(format!("{:x}", hash.finalize()))
}
pub fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchMemberBinding {
    pub original_index: usize,
    pub response_id: String,
    pub prompt_ids: Vec<u32>,
    pub effective_semantics: Value,
    pub request: RequestBinding,
}
/// Local CPU admission UUIDs/clock nonces deliberately do not enter this common
/// wire binding: each rank independently owns its own local slot book.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchGroupBinding {
    response_id: String,
    original_body_sha256: String,
    prepared_group_sha256: String,
    members: Vec<BatchMemberBinding>,
    sha256: String,
}
impl BatchGroupBinding {
    pub fn from_cpu(
        group: &CpuOwnedAtomicGroup,
        book: &CpuSlotBook,
        now: Instant,
    ) -> Result<Arc<Self>> {
        let record = group.publication(book, now)?;
        let mut out = Self {
            response_id: record.group_request_id.clone(),
            original_body_sha256: record.original_group_body_sha256.clone(),
            prepared_group_sha256: record.prepared_group_sha256.clone(),
            members: record
                .members
                .iter()
                .map(|m| BatchMemberBinding {
                    original_index: m.original_response_index,
                    response_id: m.member_request_id.clone(),
                    prompt_ids: m.native_prompt_ids.clone(),
                    effective_semantics: m.effective_semantics.clone(),
                    request: m.request.clone(),
                })
                .collect(),
            sha256: String::new(),
        };
        out.validate()?;
        out.sha256 = digest(
            b"dsv41-resident-atomic-group-v1\0",
            &(
                &out.response_id,
                &out.original_body_sha256,
                &out.prepared_group_sha256,
                &out.members,
            ),
        )?;
        Ok(Arc::new(out))
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.response_id.is_empty()
                && sha256(&self.original_body_sha256)
                && sha256(&self.prepared_group_sha256)
                && (1..=MAX_MEMBERS).contains(&self.members.len()),
            "explicit nonempty atomic group and bounded member identities required"
        );
        let mut slots = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let base = &self.members[0].request;
        for (i, m) in self.members.iter().enumerate() {
            let r = &m.request;
            let s = &r.sampling;
            ensure!(
                m.original_index == i
                    && !m.response_id.is_empty()
                    && ids.insert(&m.response_id)
                    && r.lane.slot < MAX_MEMBERS
                    && slots.insert(r.lane.slot)
                    && !r.lane.boot.is_nil()
                    && r.lane.generation > 0
                    && r.lane.boot == base.lane.boot,
                "atomic original order/unique response IDs/physical lanes/generations differ"
            );
            ensure!(
                r.model_revision == MODEL_REVISION
                    && r.tokenizer_sha256 == TOKENIZER_SHA256
                    && r.toolkit_revision == TOOLKIT_REVISION
                    && r.frequency == FrequencyBackend::parse(&r.frequency.backend)?.identity()
                    && [
                        &r.global_checkpoint_geometry_checksum_sha256,
                        &r.execution_snapshot_sha256,
                        &r.engine_binary_sha256,
                        &r.original_request_sha256
                    ]
                    .iter()
                    .all(|v| sha256(v)),
                "exact checkpoint/tokenizer/toolkit/frequency/source identity required"
            );
            ensure!(
                r.global_checkpoint_geometry_checksum_sha256
                    == base.global_checkpoint_geometry_checksum_sha256
                    && r.execution_snapshot_sha256 == base.execution_snapshot_sha256
                    && r.engine_binary_sha256 == base.engine_binary_sha256
                    && r.frequency == base.frequency,
                "all group members must use the same exact resident deployment"
            );
            ensure!(
                s.temperature.is_finite()
                    && s.temperature == 0.0
                    && s.top_p == 1.0
                    && s.presence_penalty == 0.0
                    && s.frequency_penalty == 0.0
                    && !s.speculation
                    && s.effective_reasoning_score
                        .is_none_or(|n| (1..=100).contains(&n))
                    && r.stop.eos_native_id == EOS_ID
                    && r.stop.user_sequences.is_empty(),
                "first resident atomic profile is native greedy/EOS-or-full-allowance only"
            );
            ensure!(
                !m.prompt_ids.is_empty()
                    && m.prompt_ids.iter().all(|&n| n < 129280)
                    && (1..=MAX_OUTPUT).contains(&r.budget.output_allowance)
                    && r.budget.context_tokens == MAX_LEARNED_PREFIX
                    && m.prompt_ids
                        .len()
                        .checked_add(r.budget.output_allowance)
                        .is_some_and(|n| n <= MAX_LEARNED_PREFIX)
                    && (1..=180000).contains(&r.budget.wall_millis)
                    && r.budget.wall_millis == base.budget.wall_millis,
                "complete original allowance must fit the qualified512 context; no clipping"
            );
            let e = &m.effective_semantics;
            ensure!(
                e["original_request_sha256"] == r.original_request_sha256
                    && e["prompt_ids_sha256_u32le"] == native_ids_sha256(&m.prompt_ids)
                    && e["output_allowance"].as_u64() == Some(r.budget.output_allowance as u64)
                    && e["effective_temperature"].as_f64() == Some(0.0)
                    && e["effective_top_p"].as_f64() == Some(1.0)
                    && e["thinking"].as_bool() == Some(s.thinking)
                    && e["effective_reasoning_score"].as_u64()
                        == s.effective_reasoning_score.map(u64::from)
                    && e["seed"].as_u64() == s.seed
                    && e["stream"].as_bool() == Some(false)
                    && e["ignore_eos"].as_bool() == Some(false)
                    && e["min_tokens"].as_u64() == Some(0)
                    && e["stop_sequences"].as_array().is_some_and(Vec::is_empty)
                    && e.get("forced_tool_constraint").is_none(),
                "member effective/native semantics changed after whole-group preparation"
            );
        }
        Ok(())
    }
    pub fn digest(&self) -> &str {
        &self.sha256
    }
    pub fn response_id(&self) -> &str {
        &self.response_id
    }
    pub fn original_body_sha256(&self) -> &str {
        &self.original_body_sha256
    }
    pub fn prepared_group_sha256(&self) -> &str {
        &self.prepared_group_sha256
    }
    pub fn members(&self) -> &[BatchMemberBinding] {
        &self.members
    }
    pub fn lanes(&self) -> Vec<LaneKey> {
        self.members.iter().map(|m| m.request.lane).collect()
    }
    pub fn wall_millis(&self) -> u64 {
        self.members[0].request.budget.wall_millis
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum BatchStopReason {
    NativeEos,
    OutputAllowance,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchMemberProgress {
    pub original_index: usize,
    pub lane: LaneKey,
    pub consumed_native_ids: Vec<u32>,
    pub generated_native_ids: Vec<u32>,
    pub stop_reason: Option<BatchStopReason>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchStepMember {
    pub original_index: usize,
    pub lane: LaneKey,
    pub start: usize,
    pub tokens: usize,
    pub full_prompt_tokens: Option<usize>,
    pub generated_output_ordinal: Option<usize>,
}
#[derive(Clone, Debug, Serialize)]
pub struct BatchStepSpec {
    pub step_id: u64,
    pub kind: PackedKind,
    pub members: Vec<BatchStepMember>,
    pub operations: BatchModelOperationIds,
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub queued_before: usize,
    pub queued_after: usize,
}
impl BatchStepSpec {
    pub fn rows(&self) -> usize {
        self.members.iter().map(|m| m.tokens).sum()
    }
}
fn serialize_group<S: serde::Serializer>(
    group: &Arc<BatchGroupBinding>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    group.as_ref().serialize(serializer)
}
#[derive(Clone, Debug, Serialize)]
pub struct BatchPhasePlan {
    #[serde(serialize_with = "serialize_group")]
    group: Arc<BatchGroupBinding>,
    phase: usize,
    kind: PackedKind,
    initial_progress: Vec<BatchMemberProgress>,
    steps: Vec<BatchStepSpec>,
    #[serde(skip)]
    operations: Vec<Operation>,
    start_epoch: u64,
    end_epoch: u64,
    next_operation_id: u64,
    next_step_id: u64,
    next_batch_version: u64,
    sha256: String,
}
impl BatchPhasePlan {
    pub fn first(
        group: Arc<BatchGroupBinding>,
        first_operation: u64,
        first_step: u64,
        epoch: u64,
        next_batch_version: u64,
    ) -> Result<Arc<Self>> {
        let progress = group
            .members
            .iter()
            .map(|m| BatchMemberProgress {
                original_index: m.original_index,
                lane: m.request.lane,
                consumed_native_ids: vec![],
                generated_native_ids: vec![],
                stop_reason: None,
            })
            .collect();
        Self::build(
            group,
            0,
            PackedKind::PrefillSegments,
            progress,
            first_operation,
            first_step,
            epoch,
            next_batch_version,
        )
    }
    fn build(
        group: Arc<BatchGroupBinding>,
        phase: usize,
        kind: PackedKind,
        initial_progress: Vec<BatchMemberProgress>,
        first_operation: u64,
        first_step: u64,
        epoch: u64,
        next_batch_version: u64,
    ) -> Result<Arc<Self>> {
        ensure!(
            phase < MAX_PHASES
                && epoch % 3 == 0
                && epoch <= MAX_DEVICE_EPOCH
                && next_batch_version > 0
                && initial_progress.len() == group.members.len(),
            "bounded aligned atomic phase/version"
        );
        let mut positions: Vec<_> = initial_progress
            .iter()
            .map(|p| p.consumed_native_ids.len())
            .collect();
        let mut outputs: Vec<_> = initial_progress
            .iter()
            .map(|p| p.generated_native_ids.len())
            .collect();
        let alive: Vec<_> = initial_progress
            .iter()
            .filter(|p| p.stop_reason.is_none())
            .map(|p| p.original_index)
            .collect();
        ensure!(
            !alive.is_empty(),
            "empty group cannot start a new model phase"
        );
        let mut steps = Vec::new();
        let mut operations = Vec::new();
        let mut op = first_operation;
        let mut sid = first_step;
        let mut at_epoch = epoch;
        loop {
            let participants: Vec<_> = match kind {
                PackedKind::PrefillSegments => alive
                    .iter()
                    .copied()
                    .filter(|&i| positions[i] < group.members[i].prompt_ids.len())
                    .collect(),
                PackedKind::DecodeLanes => alive
                    .iter()
                    .copied()
                    .filter(|&i| outputs[i] < group.members[i].request.budget.output_allowance)
                    .collect(),
            };
            if participants.is_empty() {
                break;
            }
            // Decode phases stop before ANY member hits its allowance. The
            // next live mask is agreed after that actual completed boundary.
            if kind == PackedKind::DecodeLanes && participants.len() != alive.len() {
                break;
            }
            let per_lane = 128 / participants.len();
            let mut members = Vec::new();
            for i in participants {
                let m = &group.members[i];
                let tokens = if kind == PackedKind::PrefillSegments {
                    per_lane.min(m.prompt_ids.len() - positions[i])
                } else {
                    1
                };
                let full = (kind == PackedKind::PrefillSegments).then_some(m.prompt_ids.len());
                let eligible = full.is_none_or(|n| positions[i] + tokens == n);
                if eligible {
                    outputs[i] += 1;
                }
                members.push(BatchStepMember {
                    original_index: i,
                    lane: m.request.lane,
                    start: positions[i],
                    tokens,
                    full_prompt_tokens: full,
                    generated_output_ordinal: eligible.then_some(outputs[i]),
                });
                positions[i] += tokens;
            }
            let ids = BatchModelOperationIds::consecutive(op)?;
            let rows = members.iter().map(|m| m.tokens).sum();
            let wire = ids.operations(rows, members.len(), kind)?;
            let before = at_epoch;
            for &w in &wire {
                at_epoch = at_epoch
                    .checked_add(w.wire_epochs() as u64)
                    .context("batch epoch overflow")?;
            }
            ensure!(
                at_epoch <= MAX_DEVICE_EPOCH && operations.len() + wire.len() <= 32768,
                "atomic phase exceeds bounded native program/epoch domain"
            );
            let queued_before = operations.len();
            operations.extend(wire);
            steps.push(BatchStepSpec {
                step_id: sid,
                kind,
                members,
                operations: ids,
                epoch_before: before,
                epoch_after: at_epoch,
                queued_before,
                queued_after: operations.len(),
            });
            op = op
                .checked_add(OPS_PER_STEP as u64)
                .context("batch operation tail overflow")?;
            sid = sid.checked_add(1).context("batch step tail overflow")?;
        }
        ensure!(
            !steps.is_empty(),
            "new atomic phase must have actual model work"
        );
        next_batch_version
            .checked_add(steps.len() as u64)
            .context("batch ticket version overflow")?;
        let mut plan = Self {
            group,
            phase,
            kind,
            initial_progress,
            steps,
            operations,
            start_epoch: epoch,
            end_epoch: at_epoch,
            next_operation_id: op,
            next_step_id: sid,
            next_batch_version,
            sha256: String::new(),
        };
        plan.sha256 = digest(b"dsv41-atomic-phase-plan-v1\0", &plan)?;
        Ok(Arc::new(plan))
    }
    pub fn group(&self) -> &Arc<BatchGroupBinding> {
        &self.group
    }
    pub fn phase(&self) -> usize {
        self.phase
    }
    pub fn kind(&self) -> PackedKind {
        self.kind
    }
    pub fn steps(&self) -> &[BatchStepSpec] {
        &self.steps
    }
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
    pub fn digest(&self) -> &str {
        &self.sha256
    }
    pub fn start_epoch(&self) -> u64 {
        self.start_epoch
    }
    pub fn end_epoch(&self) -> u64 {
        self.end_epoch
    }
    pub fn next_operation_id(&self) -> u64 {
        self.next_operation_id
    }
    pub fn next_step_id(&self) -> u64 {
        self.next_step_id
    }
    pub fn next_batch_version(&self) -> u64 {
        self.next_batch_version
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum BatchPhaseStop {
    PrefillFinished,
    LiveSetShrank,
    AllMembersStopped,
}
/// CPU declaration only. Public cloning/serialization does NOT grant transport,
/// reset, EOS or normal-response authority; the sealed GPU session checks its
/// source against actual opaque model results before calling transport.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchPhaseClaim {
    pub group_sha256: String,
    pub phase_sha256: String,
    pub phase: usize,
    pub completed_steps: usize,
    pub actual_epoch: u64,
    pub queued_operations: usize,
    pub next_operation_id: u64,
    pub next_step_id: u64,
    pub next_batch_version: u64,
    pub progress: Vec<BatchMemberProgress>,
    pub reason: BatchPhaseStop,
}
impl BatchPhaseClaim {
    pub fn digest(&self) -> Result<String> {
        digest(b"dsv41-atomic-phase-prefix-v1\0", self)
    }
    pub fn terminal(&self) -> bool {
        self.reason == BatchPhaseStop::AllMembersStopped
    }
}
#[derive(Clone, Debug)]
pub struct BatchPhaseCursor {
    plan: Arc<BatchPhasePlan>,
    progress: Vec<BatchMemberProgress>,
    next: usize,
    failed: bool,
    stopped: bool,
    last_claim: Option<BatchPhaseClaim>,
}
impl BatchPhaseCursor {
    pub fn new(plan: Arc<BatchPhasePlan>) -> Self {
        Self {
            progress: plan.initial_progress.clone(),
            plan,
            next: 0,
            failed: false,
            stopped: false,
            last_claim: None,
        }
    }
    pub fn next_spec(&self) -> Result<&BatchStepSpec> {
        ensure!(
            !self.failed && !self.stopped,
            "atomic phase failed/already at release boundary"
        );
        self.plan
            .steps
            .get(self.next)
            .context("atomic phase is complete")
    }
    pub fn inputs(&self) -> Result<Vec<BatchLaneInput>> {
        let step = self.next_spec()?;
        step.members
            .iter()
            .map(|m| {
                let p = &self.progress[m.original_index];
                let native_ids = if step.kind == PackedKind::PrefillSegments {
                    self.plan.group.members[m.original_index].prompt_ids
                        [m.start..m.start + m.tokens]
                        .to_vec()
                } else {
                    vec![*p
                        .generated_native_ids
                        .last()
                        .context("decode has no actual preceding generated token")?]
                };
                ensure!(
                    step.kind != PackedKind::DecodeLanes || native_ids[0] != EOS_ID,
                    "retired native-EOS lane cannot execute another decode"
                );
                Ok(BatchLaneInput {
                    lane: m.lane,
                    native_ids,
                    full_prompt_tokens: m.full_prompt_tokens,
                })
            })
            .collect()
    }
    /// Pure CPU state transition, not an attestation. Production calls only
    /// after exact opaque model/ticket/operation/prefix checks in GPU session.
    pub fn observe(&mut self, selected: &[u32]) -> Result<Option<BatchPhaseClaim>> {
        let result = self.observe_inner(selected);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn observe_inner(&mut self, selected: &[u32]) -> Result<Option<BatchPhaseClaim>> {
        let step = self.next_spec()?.clone();
        let inputs = self.inputs()?;
        ensure!(
            selected.len() == step.members.len() && selected.iter().all(|&n| n < 129280),
            "all packed output rows must have real native vocabulary IDs"
        );
        let mut progress = self.progress.clone();
        let alive_before: Vec<_> = progress
            .iter()
            .filter(|p| p.stop_reason.is_none())
            .map(|p| p.original_index)
            .collect();
        for ((m, input), &token) in step.members.iter().zip(&inputs).zip(selected) {
            let p = &mut progress[m.original_index];
            ensure!(
                p.stop_reason.is_none() && p.consumed_native_ids.len() == m.start,
                "retired/stale native member step"
            );
            p.consumed_native_ids.extend(&input.native_ids);
            if let Some(ordinal) = m.generated_output_ordinal {
                ensure!(
                    ordinal == p.generated_native_ids.len() + 1,
                    "native output ordinal drift"
                );
                p.generated_native_ids.push(token);
                if token == EOS_ID {
                    p.stop_reason = Some(BatchStopReason::NativeEos);
                } else if ordinal
                    == self.plan.group.members[m.original_index]
                        .request
                        .budget
                        .output_allowance
                {
                    p.stop_reason = Some(BatchStopReason::OutputAllowance);
                }
            }
        }
        self.progress = progress;
        self.next += 1;
        let all_stopped = self.progress.iter().all(|p| p.stop_reason.is_some());
        let alive_after: Vec<_> = self
            .progress
            .iter()
            .filter(|p| p.stop_reason.is_none())
            .map(|p| p.original_index)
            .collect();
        let phase_end = self.next == self.plan.steps.len();
        // Prefill shape is determined by remaining prompt rows, not by ignored
        // heads. EOS of a fully-prefilled member cannot affect later prefill.
        let reason = if all_stopped {
            Some(BatchPhaseStop::AllMembersStopped)
        } else if self.plan.kind == PackedKind::PrefillSegments && phase_end {
            Some(BatchPhaseStop::PrefillFinished)
        } else if self.plan.kind == PackedKind::DecodeLanes
            && (alive_before != alive_after || phase_end)
        {
            Some(BatchPhaseStop::LiveSetShrank)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.stopped = true;
            let claim = BatchPhaseClaim {
                group_sha256: self.plan.group.digest().into(),
                phase_sha256: self.plan.digest().into(),
                phase: self.plan.phase,
                completed_steps: self.next,
                actual_epoch: step.epoch_after,
                queued_operations: step.queued_after,
                next_operation_id: self.plan.next_operation_id,
                next_step_id: self.plan.next_step_id,
                next_batch_version: self.plan.next_batch_version + self.next as u64,
                progress: self.progress.clone(),
                reason,
            };
            self.last_claim = Some(claim.clone());
            return Ok(Some(claim));
        }
        Ok(None)
    }
    pub fn next_phase(&self, claim: &BatchPhaseClaim) -> Result<Arc<BatchPhasePlan>> {
        ensure!(
            !self.failed
                && self.stopped
                && self.last_claim.as_ref() == Some(claim)
                && !claim.terminal()
                && claim.phase_sha256 == self.plan.digest()
                && claim.progress == self.progress
                && claim.completed_steps == self.next
                && claim.next_operation_id == self.plan.next_operation_id
                && claim.next_step_id == self.plan.next_step_id,
            "next phase requires this exact released prefix; no tail or native state rewinds"
        );
        BatchPhasePlan::build(
            self.plan.group.clone(),
            self.plan.phase + 1,
            PackedKind::DecodeLanes,
            self.progress.clone(),
            claim.next_operation_id,
            claim.next_step_id,
            claim.actual_epoch,
            claim.next_batch_version,
        )
    }
    pub fn completed_steps(&self) -> usize {
        self.next
    }
    pub fn progress(&self) -> &[BatchMemberProgress] {
        &self.progress
    }
    pub fn plan(&self) -> &Arc<BatchPhasePlan> {
        &self.plan
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        atomic_request_admission::{
            AdmissionClock, CpuAtomicAdmission, PreparedAtomicRequest, ProposedBatchDeployment,
        },
        native_api_contract::{CpuBridge, ATOMIC_ROUTE},
    };
    use super::*;
    use std::{sync::OnceLock, time::Duration};
    use uuid::Uuid;
    fn bridge() -> &'static CpuBridge {
        static B: OnceLock<CpuBridge> = OnceLock::new();
        B.get_or_init(|| CpuBridge::from_verified_tokenizer(&std::fs::read(
            "test-data/tokenizer.json").unwrap()).unwrap())
    }
    fn binding(width: usize, cap: usize, mixed: bool) -> Arc<BatchGroupBinding> {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_actual_atomic_requests.json"
        ))
        .unwrap();
        let mut body = fixture["cases"][2]["request"].clone();
        body["messages"].as_array_mut().unwrap().truncate(width);
        if mixed {
            for i in 0..width {
                if i % 2 == 1 {
                    body["messages"][i] = fixture["cases"][3]["request"]["messages"][0].clone();
                }
            }
        }
        body["max_tokens"] = serde_json::json!(cap);
        let now = Instant::now();
        let boot = Uuid::from_u128(214);
        let mut book = CpuSlotBook::new(boot, 16).unwrap();
        let keys = (0..width)
            .map(|i| LaneKey {
                boot,
                slot: (i * 5 + 7) % 16,
                generation: 1,
            })
            .collect();
        let prepared =
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &body).unwrap();
        let mut transaction = CpuAtomicAdmission::new(
            &mut book,
            prepared,
            ProposedBatchDeployment {
                boot,
                global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                execution_snapshot_sha256: "b".repeat(64),
                engine_binary_sha256: "c".repeat(64),
                frequency: FrequencyBackend::CudaTorch.identity(),
            },
            keys,
            AdmissionClock::new(now, now + Duration::from_secs(100)).unwrap(),
            format!("batch-{width}-{cap}"),
            now,
        )
        .unwrap();
        transaction.commit(now).unwrap();
        let owned = transaction.take_published(now).unwrap();
        BatchGroupBinding::from_cpu(&owned, &book, now).unwrap()
    }
    #[test]
    fn original_bodies_keep_native75_68_effort75_seed42_and_full256_at_every_width() {
        for width in [1, 2, 4, 8, 16] {
            let group = binding(width, 256, true);
            for (i, m) in group.members().iter().enumerate() {
                assert_eq!(m.prompt_ids.len(), if i % 2 == 0 { 75 } else { 68 });
                assert_eq!(m.request.sampling.seed, Some(42));
                assert_eq!(m.request.sampling.effective_reasoning_score, Some(75));
                assert_eq!(m.request.budget.output_allowance, 256);
                assert_eq!(m.request.budget.wall_millis, 100000);
            }
            let first = BatchPhasePlan::first(group, 11, 19, 24, 1).unwrap();
            assert!(first.operations().len() <= 32768);
            assert!(first
                .steps()
                .iter()
                .all(|s| s.rows() <= 128 && s.members.len() <= width));
            assert_eq!(first.kind(), PackedKind::PrefillSegments);
            let mut cursor = BatchPhaseCursor::new(first);
            let prefill = loop {
                let n = cursor.next_spec().unwrap().members.len();
                if let Some(c) = cursor.observe(&vec![2581; n]).unwrap() {
                    break c;
                }
            };
            let decode = cursor.next_phase(&prefill).unwrap();
            assert_eq!(decode.steps().len(), 255);
            assert_eq!(decode.operations().len(), 255 * 84);
            let mut c = BatchPhaseCursor::new(decode);
            let terminal = loop {
                let n = c.next_spec().unwrap().members.len();
                if let Some(claim) = c.observe(&vec![2581; n]).unwrap() {
                    break claim;
                }
            };
            assert!(terminal.terminal());
            assert!(terminal
                .progress
                .iter()
                .all(|p| p.generated_native_ids.len() == 256
                    && p.stop_reason == Some(BatchStopReason::OutputAllowance)));
            for p in &terminal.progress {
                assert_eq!(
                    p.consumed_native_ids.len(),
                    binding(width, 256, true).members()[p.original_index]
                        .prompt_ids
                        .len()
                        + 255
                );
            }
        }
    }
    #[test]
    fn actual_early_eos_retires_lane_once_and_burns_entire_unused_phase_tail() {
        let first = BatchPhasePlan::first(binding(4, 256, true), 7, 13, 0, 1).unwrap();
        let mut cursor = BatchPhaseCursor::new(first);
        let prefill = loop {
            let n = cursor.next_spec().unwrap().members.len();
            if let Some(c) = cursor.observe(&vec![2581; n]).unwrap() {
                break c;
            }
        };
        let phase = cursor.next_phase(&prefill).unwrap();
        let reserved_op = phase.next_operation_id();
        let reserved_step = phase.next_step_id();
        let first_version = phase.next_batch_version();
        let mut cursor = BatchPhaseCursor::new(phase.clone());
        let claim = cursor.observe(&[2581, 1, 2581, 2581]).unwrap().unwrap();
        assert!(!claim.terminal());
        assert_eq!(claim.next_operation_id, reserved_op);
        assert_eq!(claim.next_step_id, reserved_step);
        assert_eq!(claim.next_batch_version, first_version + 1);
        assert!(claim.actual_epoch < phase.end_epoch());
        let next = cursor.next_phase(&claim).unwrap();
        assert_eq!(next.steps()[0].operations.embedding, reserved_op);
        assert_eq!(next.steps()[0].step_id, reserved_step);
        assert_eq!(next.start_epoch(), claim.actual_epoch);
        assert_eq!(
            next.steps()[0]
                .members
                .iter()
                .map(|m| m.original_index)
                .collect::<Vec<_>>(),
            [0, 2, 3]
        );
        assert_eq!(claim.progress[1].generated_native_ids, [2581, 1]);
        assert!(cursor.observe(&[2581; 4]).is_err());
    }
    #[test]
    fn ignored_prefill_eos_is_not_output_or_retirement_and_forged_rearm_is_rejected() {
        let first = BatchPhasePlan::first(binding(16, 256, true), 1, 1, 0, 1).unwrap();
        let mut cursor = BatchPhaseCursor::new(first);
        assert!(cursor
            .next_spec()
            .unwrap()
            .members
            .iter()
            .all(|m| m.generated_output_ordinal.is_none()));
        assert!(cursor.observe(&[EOS_ID; 16]).unwrap().is_none());
        assert!(cursor
            .progress()
            .iter()
            .all(|p| p.generated_native_ids.is_empty() && p.stop_reason.is_none()));
        let claim = loop {
            let n = cursor.next_spec().unwrap().members.len();
            if let Some(c) = cursor.observe(&vec![2581; n]).unwrap() {
                break c;
            }
        };
        let mut bad = claim.clone();
        bad.actual_epoch += 3;
        assert!(cursor.next_phase(&bad).is_err());
        let mut bad = claim.clone();
        bad.next_batch_version += 1;
        assert!(cursor.next_phase(&bad).is_err());
        cursor.next_phase(&claim).unwrap();
    }
    #[test]
    fn bad_last_row_fails_whole_cpu_transition_and_output_cap_is_not_shortened() {
        let plan = BatchPhasePlan::first(binding(2, 1, false), 1, 1, 0, 1).unwrap();
        let mut cursor = BatchPhaseCursor::new(plan);
        let before = cursor.progress().to_vec();
        assert!(cursor.observe(&[2581, 129280]).is_err());
        assert_eq!(cursor.progress(), before);
        assert!(cursor.inputs().is_err());
        assert!(BatchPhasePlan::first(binding(1, 256, false), u64::MAX - 1, 1, 0, 1).is_err());
        assert!(BatchPhasePlan::first(binding(1, 256, false), 1, 1, 1, 1).is_err());
        // CPU842 can prepare <=512, but the initial runtime profile explicitly
        // refuses >256 rather than replacing the caller's full requested cap.
        let mut too_large = (*binding(1, 256, false)).clone();
        too_large.members[0].request.budget.output_allowance = 257;
        assert!(too_large.validate().is_err());
    }
}
