//! Sealed resident atomic-batch session. CPU admissions/JSON declarations,
//! transport-only receipts and compute-only batch probes cannot construct the
//! normal terminal type. Dynamic EOS retires rows only at an actual checked
//! full-model boundary followed by all8 prefix agreement and lifetime release.
use super::{
    atomic_request_admission::{CpuOwnedAtomicGroup, CpuSlotBook},
    batch_plan::PackedKind,
    batch_request_plan::{
        self, BatchGroupBinding, BatchMemberProgress, BatchPhaseClaim, BatchPhaseCursor,
        BatchPhasePlan, BatchStopReason, MAX_PHASES, OPS_PER_STEP,
    },
    batch_request_release::{BatchPhaseReleaseReceipt, BatchReleaseConfig},
    gpu_batch_model::{
        BatchModelIdentity, CheckedBatchModel, CheckedBatchModelReset, GpuBatchModel,
    },
    gpu_collectives::GpuCollectives,
    model_session_plan::native_ids_sha256,
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
/// Private fields/no Deserialize/no Clone. A borrowed token cannot be edited,
/// reconstructed from audit JSON, substituted for a C1 token or minted by a
/// fixed8/32 compute-only probe.
#[derive(Debug, Serialize)]
pub struct CheckedBatchRequestCompletion {
    group: BatchGroupBinding,
    model: BatchModelIdentity,
    final_phase_program_sha256: String,
    phase_plan_sha256: Vec<String>,
    phase_program_sha256: Vec<String>,
    phase_releases: Vec<BatchPhaseReleaseReceipt>,
    final_claim: BatchPhaseClaim,
    cleanup_reset: CheckedBatchModelReset,
    cleanup_release: BatchPhaseReleaseReceipt,
    completed_model_steps: usize,
    completed_operations: usize,
    next_operation_id: u64,
    next_step_id: u64,
    elapsed_seconds: f64,
    original_wall_millis: u64,
    timing_scope: &'static str,
    all_rank_native_prefix_and_terminal_agreement: bool,
    original_group_order_and_allowances_preserved: bool,
    actual_fresh_group_cleanup_reset_checked: bool,
    benchmark_qualified: bool,
    model_or_task_quality_qualified: bool,
}
impl CheckedBatchRequestCompletion {
    pub fn binding(&self) -> &BatchGroupBinding {
        &self.group
    }
    pub fn model_identity(&self) -> &BatchModelIdentity {
        &self.model
    }
    pub fn progress(&self) -> &[BatchMemberProgress] {
        &self.final_claim.progress
    }
    pub fn final_claim(&self) -> &BatchPhaseClaim {
        &self.final_claim
    }
    pub fn actual_epoch(&self) -> u64 {
        self.final_claim.actual_epoch
    }
    pub fn next_batch_version(&self) -> u64 {
        self.final_claim.next_batch_version
    }
    pub fn next_operation_id(&self) -> u64 {
        self.next_operation_id
    }
    pub fn next_step_id(&self) -> u64 {
        self.next_step_id
    }
    pub fn completed_model_steps(&self) -> usize {
        self.completed_model_steps
    }
    pub fn completed_operations(&self) -> usize {
        self.completed_operations
    }
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }
    pub fn cleanup_reset(&self) -> &CheckedBatchModelReset {
        &self.cleanup_reset
    }
    pub fn final_program_sha256(&self) -> &str {
        &self.final_phase_program_sha256
    }
    pub fn all_rank_terminal_agreement(&self) -> bool {
        self.all_rank_native_prefix_and_terminal_agreement
    }
    pub fn transport_lifetime_released(&self) -> bool {
        self.phase_releases
            .iter()
            .all(BatchPhaseReleaseReceipt::lifetime_released)
            && self.cleanup_release.lifetime_released()
    }
    pub fn fresh_cleanup_checked(&self) -> bool {
        self.actual_fresh_group_cleanup_reset_checked
    }
}
/// Release ports are reserved per complete group, not per member. They are
/// bounded/fresh even when an earlyEOS burns an entire frozen program tail.
#[derive(Clone, Debug, Serialize)]
pub struct BatchSessionRelease {
    pub peers: [std::net::Ipv4Addr; 8],
    pub first_phase_port: u16,
    pub timeout_millis: u64,
}
impl BatchSessionRelease {
    pub fn validate(&self) -> Result<()> {
        let end = self
            .first_phase_port
            .checked_add(MAX_PHASES as u16)
            .context("batch release port overflow")?;
        ensure!(
            end <= 64000,
            "batch group reserves all17 phase ports plus cleanup"
        );
        self.at(0)?.validate()?;
        self.at(MAX_PHASES)?.validate()
    }
    fn at(&self, phase: usize) -> Result<BatchReleaseConfig> {
        ensure!(phase <= MAX_PHASES, "batch phase release count bound");
        Ok(BatchReleaseConfig {
            peers: self.peers,
            port: self
                .first_phase_port
                .checked_add(phase as u16)
                .context("batch release port overflow")?,
            timeout_millis: self.timeout_millis,
        })
    }
}
fn check_model(
    group: &BatchGroupBinding,
    actual: &BatchModelIdentity,
    comm: &GpuCollectives,
) -> Result<()> {
    let r = &group.members()[0].request;
    let runtime = actual
        .runtime
        .as_ref()
        .context("explicit actual service binary/snapshot binding required")?;
    ensure!(
        actual.rank == comm.rank()
            && actual.boot == r.lane.boot
            && actual.revision == r.model_revision
            && actual.global_checkpoint_geometry_binding
                == r.global_checkpoint_geometry_checksum_sha256
            && runtime.execution_snapshot_sha256 == r.execution_snapshot_sha256
            && runtime.engine_binary_sha256 == r.engine_binary_sha256
            && actual.frequency == r.frequency
            && actual.max_lanes >= group.members().len()
            && actual.max_learned_prefix == 512
            && actual.completion_owner == comm.completion_identity()
            && actual.stream_identity == comm.compute_stream().stream as usize as u64,
        "atomic actual model/boot/global checkpoint/source/ELF/frequency/session/stream differs"
    );
    Ok(())
}
/// Does not own CUDA producers: the synchronous owner-thread executor below
/// keeps the real model/collective owner alive until release/fence on all paths.
pub struct GpuBatchRequestSession<'a> {
    owned_admission: &'a CpuOwnedAtomicGroup,
    slot_book: &'a CpuSlotBook,
    group: Arc<BatchGroupBinding>,
    cursor: BatchPhaseCursor,
    model: BatchModelIdentity,
    admitted: Instant,
    deadline: Instant,
    release: BatchSessionRelease,
    phase_plan_sha256: Vec<String>,
    phase_program_sha256: Vec<String>,
    phase_releases: Vec<BatchPhaseReleaseReceipt>,
    completed_steps: usize,
    failed: bool,
}
impl<'a> GpuBatchRequestSession<'a> {
    pub fn attach(
        plan: Arc<BatchPhasePlan>,
        reset: CheckedBatchModelReset,
        model: &GpuBatchModel,
        comm: &GpuCollectives,
        release: BatchSessionRelease,
        owned_admission: &'a CpuOwnedAtomicGroup,
        slot_book: &'a CpuSlotBook,
    ) -> Result<Self> {
        release.validate()?;
        let now = Instant::now();
        let actual_group = BatchGroupBinding::from_cpu(owned_admission, slot_book, now)?;
        ensure!(
            actual_group.as_ref() == plan.group().as_ref()
                && owned_admission.clock().wall_millis() == plan.group().wall_millis(),
            "GPU session must borrow the exact current non-Clone CPU ownership and original clock"
        );
        let admitted = owned_admission.clock().admitted();
        let deadline = admitted
            .checked_add(Duration::from_millis(plan.group().wall_millis()))
            .context("batch admission wall overflow")?;
        ensure!(
            admitted <= now && now < deadline,
            "original before-body/queue batch admission is future/expired"
        );
        ensure!(
            plan.phase() == 0
                && plan.kind() == PackedKind::PrefillSegments
                && plan.start_epoch() == comm.expected_epoch()
                && comm.queued_operation_count() == 0
                && comm.batch_phase_digest() == Some(plan.digest()),
            "initial exact batch program/cursor required"
        );
        model.check_reset(&reset, comm)?;
        check_model(plan.group(), &reset.model(), comm)?;
        ensure!(
            reset.lanes() == plan.group().lanes()
                && reset.next_batch_version() == plan.next_batch_version(),
            "sealed all-member reset does not match the prepared group/version"
        );
        Ok(Self {
            owned_admission,
            slot_book,
            group: plan.group().clone(),
            model: reset.model().clone(),
            cursor: BatchPhaseCursor::new(plan.clone()),
            admitted,
            deadline,
            release,
            phase_plan_sha256: vec![plan.digest().into()],
            phase_program_sha256: vec![comm.program_digest_hex()],
            phase_releases: vec![],
            completed_steps: 0,
            failed: false,
        })
    }
    pub fn rearm(
        plan: Arc<BatchPhasePlan>,
        reset: CheckedBatchModelReset,
        previous: &CheckedBatchRequestCompletion,
        model: &GpuBatchModel,
        comm: &mut GpuCollectives,
        release: BatchSessionRelease,
        owned_admission: &'a CpuOwnedAtomicGroup,
        slot_book: &'a CpuSlotBook,
    ) -> Result<Self> {
        let result = (|| {
            model.check_reset(&reset, comm)?;
            ensure!(previous.all_rank_terminal_agreement() && previous.transport_lifetime_released()
                && previous.fresh_cleanup_checked() && previous.model == model.identity()
                && previous.actual_epoch() == comm.expected_epoch()
                && previous.final_program_sha256() == comm.program_digest_hex()
                && reset.reset_sequence() > previous.cleanup_reset.reset_sequence()
                && plan.steps()[0].operations.embedding == previous.next_operation_id()
                && plan.steps()[0].step_id == previous.next_step_id()
                && plan.next_batch_version() == previous.next_batch_version()
                && plan.group().digest() != previous.group.digest(),
                "new group lacks exact prior opaque terminal/cleanup/session and full-tail reservation");
            comm.begin_batch_phase(plan.clone())?;
            Self::attach(
                plan,
                reset,
                model,
                comm,
                release,
                owned_admission,
                slot_book,
            )
        })();
        if result.is_err() {
            comm.abort_session();
        }
        result
    }
    fn within_budget(&self) -> Result<()> {
        self.owned_admission
            .publication(self.slot_book, Instant::now())?;
        ensure!(
            !self.failed && Instant::now() < self.deadline,
            "atomic group failed/expired; no normal partial result or allowance shortening"
        );
        Ok(())
    }
    fn check_step(&self, proof: &CheckedBatchModel, comm: &GpuCollectives) -> Result<()> {
        self.within_budget()?;
        let s = self.cursor.next_spec()?;
        let expected_inputs = self.cursor.inputs()?;
        let flat: Vec<_> = expected_inputs
            .iter()
            .flat_map(|i| i.native_ids.iter().copied())
            .collect();
        ensure!(proof.all_checks_passed() && proof.rank() == self.model.rank
            && proof.manifest_sha256() == self.model.manifest_sha256
            && proof.geometry_binding() == self.model.global_checkpoint_geometry_binding
            && proof.completion_owner() == self.model.completion_owner
            && proof.model_instance_identity() == self.model.model_instance_identity
            && proof.stream_identity() == self.model.stream_identity
            && proof.program_sha256() == comm.program_digest_hex()
            && comm.batch_phase_digest() == Some(self.cursor.plan().digest())
            && proof.ticket().boot() == self.model.boot
            && proof.ticket().version() == self.cursor.plan().next_batch_version() + self.cursor.completed_steps() as u64
            && proof.ticket().kind() == s.kind && proof.ticket().total_rows() == s.rows()
            && proof.ticket().lanes() == s.members.len() && proof.lanes().len() == s.members.len()
            && proof.input_native_ids() == flat && proof.epoch_before() == s.epoch_before
            && proof.epoch_after() == s.epoch_after && proof.epoch_after() == comm.expected_epoch()
            && proof.operation_count_before() == s.queued_before && proof.operation_count_after() == s.queued_after
            && proof.operation_count_after() == comm.queued_operation_count()
            && serde_json::to_value(proof.operations())? == serde_json::to_value(&s.operations)?,
            "opaque packed model result differs from this exact group/phase/model/stream/operation/native input");
        for (((segment, row), expected), input) in proof
            .ticket()
            .segments()
            .iter()
            .zip(proof.lanes())
            .zip(&s.members)
            .zip(&expected_inputs)
        {
            let mut consumed = self.cursor.progress()[expected.original_index]
                .consumed_native_ids
                .clone();
            consumed.extend(&input.native_ids);
            ensure!(
                segment.ticket.lane == expected.lane
                    && segment.ticket.start == expected.start
                    && segment.ticket.tokens == expected.tokens
                    && segment.ticket.end == expected.start + expected.tokens
                    && segment.rows == expected.tokens
                    && row.lane == expected.lane
                    && row.consumed_prefix_tokens == consumed.len()
                    && row.prefix_native_ids_sha256 == native_ids_sha256(&consumed)
                    && row.generated_output_ordinal == expected.generated_output_ordinal
                    && row.intermediate_prefill_head_ignored
                        == expected.generated_output_ordinal.is_none(),
                "actual packed member/native prefix/head ordinal/full source association differs"
            );
            match (segment.phase, expected.full_prompt_tokens) {
                (super::batch_attention_layout::AttentionPhase::Decode, None) => (),
                (
                    super::batch_attention_layout::AttentionPhase::Prefill { full_prompt_tokens },
                    Some(expected),
                ) if full_prompt_tokens == expected => (),
                _ => {
                    anyhow::bail!("actual prefill/decode phase/full original prompt extent differs")
                }
            }
        }
        Ok(())
    }
    /// Executes one real packed model step at a time; every shape rearm waits
    /// for actual all8 RDMA agreement+TCP release. No partial member response.
    pub fn run(
        mut self,
        model: &mut GpuBatchModel,
        comm: &mut GpuCollectives,
        mut observe: impl FnMut(&CheckedBatchModel) -> Result<()>,
        mut local_health: impl FnMut() -> Result<()>,
    ) -> Result<CheckedBatchRequestCompletion> {
        let result = (|| loop {
            local_health()?;
            self.within_budget()?;
            check_model(&self.group, &model.identity(), comm)?;
            let s = self.cursor.next_spec()?.clone();
            ensure!(
                comm.expected_epoch() == s.epoch_before
                    && comm.queued_operation_count() == s.queued_before,
                "actual prelaunch batch cursor/epoch differs"
            );
            let inputs = self.cursor.inputs()?;
            let proof = model.execute_checked(s.kind, &inputs, s.operations, comm)?;
            self.check_step(&proof, comm)?;
            observe(&proof)?;
            local_health()?;
            self.within_budget()?;
            let selected = proof
                .lanes()
                .iter()
                .map(|r| r.selected_native_id)
                .collect::<Vec<_>>();
            let claim = self.cursor.observe(&selected)?;
            self.completed_steps += 1;
            let Some(claim) = claim else { continue };
            self.within_budget()?;
            let receipt = comm.finish_batch_phase(&claim, &self.release.at(claim.phase)?)?;
            self.within_budget()?;
            self.phase_releases.push(receipt);
            local_health()?;
            if claim.terminal() {
                validate_terminal(&self.group, &claim)?;
                let keys = self
                    .group
                    .lanes()
                    .iter()
                    .map(|key| {
                        Ok(LaneKey {
                            generation: key
                                .generation
                                .checked_add(1)
                                .context("terminal cleanup generation overflow")?,
                            ..*key
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let reset = model.reset_group_checked(&keys, comm)?;
                let cleanup =
                    comm.confirm_batch_cleanup_reset(&reset, &self.release.at(MAX_PHASES)?)?;
                local_health()?;
                self.within_budget()?;
                return Ok(CheckedBatchRequestCompletion {
                        group: (*self.group).clone(), model: self.model.clone(),
                        final_phase_program_sha256: comm.program_digest_hex(),
                        phase_plan_sha256: self.phase_plan_sha256, phase_program_sha256: self.phase_program_sha256,
                        phase_releases: self.phase_releases, final_claim: claim.clone(), cleanup_reset: reset,
                        cleanup_release: cleanup, completed_model_steps: self.completed_steps,
                        completed_operations: self.completed_steps.checked_mul(OPS_PER_STEP).context("batch op total overflow")?,
                        next_operation_id: claim.next_operation_id, next_step_id: claim.next_step_id,
                        elapsed_seconds: self.admitted.elapsed().as_secs_f64(), original_wall_millis: self.group.wall_millis(),
                        timing_scope: "original admission before HTTP body/prepare/queue through all8 terminal and fenced fresh reset; includes control waits",
                        all_rank_native_prefix_and_terminal_agreement: true,
                        original_group_order_and_allowances_preserved: true, actual_fresh_group_cleanup_reset_checked: true,
                        benchmark_qualified: false, model_or_task_quality_qualified: false,
                    });
            }
            let next = self.cursor.next_phase(&claim)?;
            comm.begin_batch_phase(next.clone())?;
            self.within_budget()?;
            self.phase_plan_sha256.push(next.digest().into());
            self.phase_program_sha256.push(comm.program_digest_hex());
            self.cursor = BatchPhaseCursor::new(next);
        })();
        if result.is_err() {
            model.poison(comm);
        }
        result
    }
}
fn validate_terminal(group: &BatchGroupBinding, claim: &BatchPhaseClaim) -> Result<()> {
    ensure!(
        claim.terminal()
            && claim.group_sha256 == group.digest()
            && claim.progress.len() == group.members().len(),
        "all original members require a real terminal reason"
    );
    for (m, p) in group.members().iter().zip(&claim.progress) {
        ensure!(
            p.original_index == m.original_index
                && p.lane == m.request.lane
                && !p.generated_native_ids.is_empty()
                && p.generated_native_ids.len() <= m.request.budget.output_allowance
                && p.consumed_native_ids.starts_with(&m.prompt_ids)
                && p.consumed_native_ids[m.prompt_ids.len()..]
                    == p.generated_native_ids[..p.generated_native_ids.len() - 1],
            "terminal member order/native prompt/generated consumed-prefix differs"
        );
        let eos = p.generated_native_ids.iter().filter(|&&n| n == 1).count();
        match p.stop_reason {
            Some(BatchStopReason::NativeEos) => ensure!(
                eos == 1 && p.generated_native_ids.last() == Some(&1),
                "native EOS must be actual final selected ID exactly once"
            ),
            Some(BatchStopReason::OutputAllowance) => ensure!(
                eos == 0 && p.generated_native_ids.len() == m.request.budget.output_allowance,
                "length is actual complete original allowance without forcedEOS"
            ),
            None => anyhow::bail!("unfinished member cannot yield normal group completion"),
        }
    }
    Ok(())
}
