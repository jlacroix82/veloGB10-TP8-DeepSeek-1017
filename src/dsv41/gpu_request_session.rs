//! PRIVATE runtime request/terminal authority. No public receipt constructor,
//! Deserialize, JSON terminal claim, CPU stop proposal, listener or SSE exists.
use super::{
    gpu_collectives::GpuCollectives,
    gpu_model_chunk::{CheckedModelChunk, CheckedModelReset, ModelChunkIdentity},
    model_session_plan::{
        native_ids_sha256, CpuRequestPlan, ExpectedBoundary, RequestBinding, EOS_ID,
    },
    request_session_release::{RequestReleaseConfig, RequestReleaseReceipt},
    request_session_state::{TransportPrefixClaim, TransportStop},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const ADMISSION_TIMING_SCOPE: &str = "trusted_HTTP_admission_before_body_queue_prepare_reset_through_checked_prefix_and_TCP_release; cleanup_may_exceed_failed_budget; not benchmark";
const TIMING_SCOPE:&str="session_attach_after_checked_reset/resident_readiness_through_checked_prefix_and_TCP_release; excludes cold_load/queue/client_parsing; not benchmark";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RequestStopReason {
    NativeEos,
    OutputAllowance,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SessionProgress {
    Continue { next_step_index: usize },
    StopReady { reason: RequestStopReason },
}
fn checked_admission_elapsed(admitted: Instant, budget: u64) -> Result<Duration> {
    let elapsed = Instant::now()
        .checked_duration_since(admitted)
        .context("future request admission clock")?;
    ensure!(
        elapsed <= Duration::from_millis(budget),
        "request wallbudget exceeded; no completion authority"
    );
    Ok(elapsed)
}
fn model_binding(m: &ModelChunkIdentity, b: &RequestBinding, rank: usize) -> Result<()> {
    ensure!(
        m.rank == rank
            && m.model_revision == b.model_revision
            && m.global_checkpoint_geometry_checksum_sha256
                == b.global_checkpoint_geometry_checksum_sha256
            && m.runtime.execution_snapshot_sha256 == b.execution_snapshot_sha256
            && m.runtime.engine_binary_sha256 == b.engine_binary_sha256
            && m.frequency_profile == b.frequency
            && m.tokenizer_sha256 == b.tokenizer_sha256
            && m.native_context == 8192
            && m.maximum_chunk_tokens == 128
            && m.concurrency == 1
            && m.temperature == 0.0
            && !m.speculation
            && !m.prefix_restore,
        "actual model/reset identity differs from global request/deployment binding"
    );
    Ok(())
}
/// Private fields prevent CPU compatibility results, serialized evidence or a
/// caller-supplied stop flag becoming response authority. Read-only metadata may
/// be serialized for audit; this type deliberately implements no Deserialize.
#[derive(Debug, Serialize)]
pub struct CheckedRequestCompletion {
    #[serde(skip)]
    plan: Arc<CpuRequestPlan>,
    binding: RequestBinding,
    model: ModelChunkIdentity,
    rank: usize,
    lane: LaneKey,
    owner: [u8; 32],
    model_instance: [u8; 32],
    stream: usize,
    program_sha256: String,
    generated: Vec<u32>,
    input_sha256: String,
    generated_sha256: String,
    consumed_sha256: String,
    reason: RequestStopReason,
    boundary: ExpectedBoundary,
    completed_steps: usize,
    operations: usize,
    epoch: u64,
    actual_all_rank_prefix_agreement: bool,
    release: RequestReleaseReceipt,
    elapsed_seconds: f64,
    timing_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    forced_tool_constraint: Option<super::tool_constraint::ConstraintExecution>,
}
impl CheckedRequestCompletion {
    pub fn forced_tool_constraint(&self) -> Option<&super::tool_constraint::ConstraintExecution> {
        self.forced_tool_constraint.as_ref()
    }
    pub fn binding(&self) -> &RequestBinding {
        &self.binding
    }
    pub fn model_identity(&self) -> &ModelChunkIdentity {
        &self.model
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn lane(&self) -> LaneKey {
        self.lane
    }
    pub fn completion_owner(&self) -> [u8; 32] {
        self.owner
    }
    pub fn model_instance_identity(&self) -> [u8; 32] {
        self.model_instance
    }
    pub fn plan_sha256(&self) -> &str {
        self.plan.digest()
    }
    pub fn program_sha256(&self) -> &str {
        &self.program_sha256
    }
    pub fn input_native_ids(&self) -> &[u32] {
        self.plan.input_ids()
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
    pub fn generated_native_ids(&self) -> &[u32] {
        &self.generated
    }
    pub fn generated_sha256(&self) -> &str {
        &self.generated_sha256
    }
    pub fn consumed_native_prefix_sha256(&self) -> &str {
        &self.consumed_sha256
    }
    pub fn stop_reason(&self) -> RequestStopReason {
        self.reason
    }
    pub fn completed_model_steps(&self) -> usize {
        self.completed_steps
    }
    pub fn completed_operations(&self) -> usize {
        self.operations
    }
    pub fn actual_epoch(&self) -> u64 {
        self.epoch
    }
    pub fn all_rank_prefix_agreement(&self) -> bool {
        self.actual_all_rank_prefix_agreement
    }
    pub fn transport_lifetime_release_confirmed(&self) -> bool {
        self.release.actual_TCP_release_completed
            && self.release.all_eight_RDMA_finish_arrivals_confirmed
    }
    pub fn elapsed_request_seconds(&self) -> f64 {
        self.elapsed_seconds
    }
    pub fn timing_scope(&self) -> &'static str {
        self.timing_scope
    }
    pub fn next_operation_id(&self) -> u64 {
        self.plan.next_operation_id()
    }
    pub fn next_step_id(&self) -> u64 {
        self.plan.steps().last().unwrap().step_id + 1
    }
}
pub struct GpuRequestSession {
    plan: Arc<CpuRequestPlan>,
    model: ModelChunkIdentity,
    rank: usize,
    owner: [u8; 32],
    model_instance: [u8; 32],
    stream: usize,
    program_sha256: String,
    release: RequestReleaseConfig,
    next_step: usize,
    generated: Vec<u32>,
    last_consumed_sha: String,
    terminal: Option<RequestStopReason>,
    last_boundary: Option<ExpectedBoundary>,
    began: Instant,
    failed: bool,
    timing_scope: &'static str,
    forced_tool_constraint: Option<super::tool_constraint::ConstraintExecution>,
}
impl GpuRequestSession {
    pub fn attach(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        comm: &GpuCollectives,
        release: RequestReleaseConfig,
    ) -> Result<Self> {
        ensure!(
            comm.queued_operation_count() == 0 && reset.queued_operation_count() == 0,
            "initial session reset/request cursor must be0"
        );
        Self::from_reset(plan, reset, comm, release, Instant::now(), TIMING_SCOPE)
    }
    fn from_reset(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        comm: &GpuCollectives,
        release: RequestReleaseConfig,
        began: Instant,
        timing_scope: &'static str,
    ) -> Result<Self> {
        checked_admission_elapsed(began, plan.binding().budget.wall_millis)?;
        release.validate()?;
        ensure!(
            plan.binding().stop.user_sequences.is_empty(),
            "native userstop sequences require official parser authority; unsupported, not ignored"
        );
        ensure!(
            reset.all_components_reset_checked()
                && reset.rank() == comm.rank()
                && reset.lane() == plan.binding().lane
                && reset.completion_owner() == comm.completion_identity()
                && reset.stream_identity() == comm.compute_stream().stream as usize
                && reset.actual_epoch() == comm.expected_epoch()
                && comm.expected_epoch() == plan.start_epoch()
                && comm.queued_operation_count() == 0
                && comm.request_plan_digest() == Some(plan.digest()),
            "checked reset/installed immutable request owner/cursor differs"
        );
        model_binding(reset.identity(), plan.binding(), comm.rank())?;
        let session = Self {
            model: reset.identity().clone(),
            rank: comm.rank(),
            owner: comm.completion_identity(),
            model_instance: reset.model_instance_identity(),
            stream: comm.compute_stream().stream as usize,
            program_sha256: comm.program_digest_hex(),
            release,
            plan,
            next_step: 0,
            generated: vec![],
            last_consumed_sha: String::new(),
            terminal: None,
            last_boundary: None,
            began,
            failed: false,
            timing_scope,
            forced_tool_constraint: None,
        };
        session.within_budget()?;
        Ok(session)
    }
    pub fn check_before_step(&mut self, comm: &mut GpuCollectives) -> Result<usize> {
        let result = (|| {
            ensure!(
                !self.failed && self.terminal.is_none(),
                "request cannot launch another step"
            );
            self.owner(comm)?;
            self.within_budget()?;
            let step = self
                .plan
                .steps()
                .get(self.next_step)
                .context("no remaining request step")?;
            ensure!(
                comm.queued_operation_count() == step.operation_begin
                    && comm.expected_epoch() == step.epoch_before,
                "not the next frozen request boundary"
            );
            Ok(self.next_step)
        })();
        if result.is_err() {
            self.failed = true;
            comm.abort_session();
        }
        result
    }
    pub fn next_step_index(&self) -> usize {
        self.next_step
    }
    pub fn generated_native_ids(&self) -> &[u32] {
        &self.generated
    }
    fn within_budget(&self) -> Result<()> {
        checked_admission_elapsed(self.began, self.plan.binding().budget.wall_millis)?;
        Ok(())
    }

    fn owner(&self, comm: &GpuCollectives) -> Result<()> {
        ensure!(
            comm.rank() == self.rank
                && comm.completion_identity() == self.owner
                && comm.compute_stream().stream as usize == self.stream
                && comm.request_plan_digest() == Some(self.plan.digest())
                && comm.program_digest_hex() == self.program_sha256,
            "request transport owner/program changed"
        );
        Ok(())
    }
    pub fn accept_step(
        &mut self,
        proof: &CheckedModelChunk,
        comm: &mut GpuCollectives,
    ) -> Result<SessionProgress> {
        let result = self.accept_inner(proof, comm);
        if result.is_err() {
            self.failed = true;
            comm.abort_session();
        }
        result
    }
    fn accept_inner(
        &mut self,
        proof: &CheckedModelChunk,
        comm: &mut GpuCollectives,
    ) -> Result<SessionProgress> {
        ensure!(
            !self.failed && self.terminal.is_none(),
            "request already failed/terminal"
        );
        self.within_budget()?;
        self.owner(comm)?;
        let step = self
            .plan
            .steps()
            .get(self.next_step)
            .context("no remaining planned modelstep")?;
        model_binding(proof.identity(), self.plan.binding(), self.rank)?;
        ensure!(
            proof.all_checks_passed()
                && proof.model_instance_identity() == self.model_instance
                && proof.rank() == self.rank
                && proof.lane() == self.plan.binding().lane
                && proof.completion_owner() == self.owner
                && proof.stream_identity() == self.stream
                && proof.plan_sha256() == self.plan.digest()
                && proof.expected_boundary() == &step.expected_boundary
                && proof.actual_epoch_before() == step.epoch_before
                && proof.actual_epoch_after() == step.epoch_after
                && proof.actual_completed_operation_count() == step.operation_end
                && comm.queued_operation_count() == step.operation_end
                && comm.expected_epoch() == step.epoch_after
                && proof.output_ordinal() == step.output_ordinal
                && proof.intermediate_head_ignored() == step.output_ordinal.is_none(),
            "opaque modelstep/cursor/epoch/owner association differs"
        );
        let input_len = self.plan.input_ids().len();
        let mut consumed = if step.end <= input_len {
            self.plan.input_ids()[..step.end].to_vec()
        } else {
            let n = step.end - input_len;
            ensure!(
                n == self.generated.len(),
                "decode must consume exactly previously generated prefix"
            );
            let mut v = self.plan.input_ids().to_vec();
            v.extend_from_slice(&self.generated);
            v
        };
        ensure!(
            consumed.len() == step.end
                && proof.consumed_native_ids() == consumed.as_slice()
                && proof.native_prefix_sha256() == native_ids_sha256(&consumed),
            "actual consumed model-native prefix differs"
        );
        let input = if step.end <= input_len {
            self.plan.input_ids()[step.start..step.end].to_vec()
        } else {
            vec![*self
                .generated
                .last()
                .context("decode has no prior generated nativeID")?]
        };
        ensure!(
            proof.input_native_ids() == input.as_slice()
                && proof.input_native_ids_sha256() == native_ids_sha256(&input),
            "actual current-chunk native IDs differ"
        );
        let selected = proof.selected_native_id();
        ensure!(selected < 129280, "selected native ID vocabulary");
        let mut generated = self.generated.clone();
        if step.output_ordinal.is_some() {
            if let Some(c) = proof.forced_tool_constraint() {
                ensure!(
                    c.mask.tokenizer_sha256 == self.plan.binding().tokenizer_sha256,
                    "opaque constrained sampler tokenizer differs from immutable request binding"
                );
            }
            super::tool_constraint::validate_execution_chain(
                self.forced_tool_constraint.as_ref(),
                self.generated.is_empty(),
                proof.forced_tool_constraint(),
                &self.plan.binding().original_request_sha256,
                self.plan.input_ids(),
                &self.generated,
                selected,
            )?;
        } else {
            ensure!(
                proof.forced_tool_constraint().is_none(),
                "ignored prefill head cannot carry generated-token constraint authority"
            );
        }
        if let Some(ordinal) = step.output_ordinal {
            ensure!(ordinal == generated.len() + 1, "outputordinal differs");
            generated.push(selected);
        }
        ensure!(
            proof.generated_native_ids() == generated.as_slice()
                && proof.generated_native_ids_sha256() == native_ids_sha256(&generated),
            "actual generated prefix/head-ignore differs"
        );
        comm.synchronize()?;
        self.within_budget()?;
        self.last_consumed_sha = native_ids_sha256(&consumed);
        consumed.clear();
        if step.output_ordinal.is_some() {
            self.forced_tool_constraint = proof.forced_tool_constraint().cloned();
        }
        self.generated = generated;
        self.next_step += 1;
        self.last_boundary = Some(step.expected_boundary.clone());
        if step.output_ordinal.is_some() {
            if selected == EOS_ID {
                self.terminal = Some(RequestStopReason::NativeEos);
            } else if self.generated.len() == self.plan.binding().budget.output_allowance {
                self.terminal = Some(RequestStopReason::OutputAllowance);
            }
        }
        Ok(match self.terminal {
            Some(reason) => SessionProgress::StopReady { reason },
            None => SessionProgress::Continue {
                next_step_index: self.next_step,
            },
        })
    }
    pub fn finish(mut self, comm: &mut GpuCollectives) -> Result<CheckedRequestCompletion> {
        let result = (|| {
            ensure!(!self.failed, "failed request has no terminal authority");
            self.owner(comm)?;
            self.within_budget()?;
            let reason = self
                .terminal
                .context("no actual checked EOS/cap boundary")?;
            let boundary = self
                .last_boundary
                .clone()
                .context("no completed modelstep")?;
            let claim = TransportPrefixClaim {
                boundary: boundary.clone(),
                stop: match reason {
                    RequestStopReason::NativeEos => TransportStop::NativeEos,
                    RequestStopReason::OutputAllowance => TransportStop::OutputAllowance,
                },
                selected_native_id: self.generated.last().copied(),
                consumed_native_ids_sha256: self.last_consumed_sha.clone(),
                generated_native_ids_sha256: native_ids_sha256(&self.generated),
            };
            let prefix = comm.finish_transport_request_prefix(&claim)?;
            let release = comm.complete_request_release(&self.release)?;
            self.within_budget()?;
            ensure!(
                prefix.rank == self.rank
                    && prefix.program_sha256 == self.program_sha256
                    && prefix.actual_epoch == boundary.epoch
                    && prefix.actual_queued_operations == boundary.operations_queued
                    && prefix.all_rank_RDMA_prefix_agreement
                    && !prefix.model_completion_authority
                    && release.claim_sha256 == claim.digest()?
                    && release.actual_TCP_release_completed,
                "actual all-rank prefix/release metadata differs"
            );
            Ok(CheckedRequestCompletion {
                binding: self.plan.binding().clone(),
                input_sha256: self.plan.input_digest().into(),
                generated_sha256: native_ids_sha256(&self.generated),
                consumed_sha256: self.last_consumed_sha.clone(),
                model: self.model.clone(),
                rank: self.rank,
                lane: self.plan.binding().lane,
                owner: self.owner,
                model_instance: self.model_instance,
                stream: self.stream,
                program_sha256: self.program_sha256.clone(),
                generated: std::mem::take(&mut self.generated),
                reason,
                boundary: boundary.clone(),
                completed_steps: self.next_step,
                operations: boundary.operations_queued,
                epoch: boundary.epoch,
                actual_all_rank_prefix_agreement: true,
                release,
                elapsed_seconds: self.began.elapsed().as_secs_f64(),
                timing_scope: self.timing_scope,
                forced_tool_constraint: self.forced_tool_constraint.clone(),
                plan: self.plan.clone(),
            })
        })();
        if result.is_err() {
            comm.abort_session();
        }
        result
    }
    pub fn rearm(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        previous: &CheckedRequestCompletion,
        comm: &mut GpuCollectives,
        release: RequestReleaseConfig,
    ) -> Result<Self> {
        Self::rearm_with_clock(
            plan,
            reset,
            previous,
            comm,
            release,
            Instant::now(),
            TIMING_SCOPE,
        )
    }
    fn rearm_with_clock(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        previous: &CheckedRequestCompletion,
        comm: &mut GpuCollectives,
        release: RequestReleaseConfig,
        began: Instant,
        timing_scope: &'static str,
    ) -> Result<Self> {
        let result = (|| {
            release.validate()?;
            checked_admission_elapsed(began, plan.binding().budget.wall_millis)?;
            ensure!(
                plan.binding().stop.user_sequences.is_empty(),
                "native userstops unsupported, not ignored"
            );
            ensure!(
                previous.all_rank_prefix_agreement()
                    && previous.transport_lifetime_release_confirmed()
                    && previous.rank == comm.rank()
                    && previous.owner == comm.completion_identity()
                    && previous.stream == comm.compute_stream().stream as usize
                    && previous.plan_sha256()
                        == comm.request_plan_digest().context("not request mode")?
                    && previous.program_sha256 == comm.program_digest_hex()
                    && previous.epoch == comm.expected_epoch()
                    && previous.operations == comm.queued_operation_count(),
                "old actual terminal/owner/prefix differs before newrequest"
            );
            ensure!(
                reset.all_components_reset_checked()
                    && reset.model_instance_identity() == previous.model_instance
                    && reset.rank() == comm.rank()
                    && reset.completion_owner() == previous.owner
                    && reset.stream_identity() == previous.stream
                    && reset.actual_epoch() == previous.epoch
                    && reset.queued_operation_count() == previous.operations
                    && reset.lane() == plan.binding().lane,
                "actual resident-model reset differs"
            );
            model_binding(reset.identity(), plan.binding(), comm.rank())?;
            ensure!(
                reset.identity().manifest_sha256 == previous.model.manifest_sha256,
                "resident model manifest changed"
            );
            // This may allocate/control-agree once between requests. It never
            // resets the GPU epoch or changes the transport completion owner.
            comm.begin_next_request(plan.clone())?;
            Self::from_reset(plan, reset, comm, release, began, timing_scope)
        })();
        if result.is_err() {
            comm.abort_session();
        }
        result
    }
    /// Trusted in-process actor clock, captured before HTTP body/queue/prepare.
    /// It is never accepted from user JSON. GPU work remains actor-owned after a
    /// timeout/disconnect, until checked failure/whole-cluster teardown completes.
    pub fn attach_from_admission(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        comm: &GpuCollectives,
        release: RequestReleaseConfig,
        admitted: Instant,
    ) -> Result<Self> {
        ensure!(
            comm.queued_operation_count() == 0 && reset.queued_operation_count() == 0,
            "initial request cursor must be0"
        );
        let session =
            Self::from_reset(plan, reset, comm, release, admitted, ADMISSION_TIMING_SCOPE)?;
        session.within_budget()?;
        Ok(session)
    }
    pub fn rearm_from_admission(
        plan: Arc<CpuRequestPlan>,
        reset: &CheckedModelReset,
        previous: &CheckedRequestCompletion,
        comm: &mut GpuCollectives,
        release: RequestReleaseConfig,
        admitted: Instant,
    ) -> Result<Self> {
        Self::rearm_with_clock(
            plan,
            reset,
            previous,
            comm,
            release,
            admitted,
            ADMISSION_TIMING_SCOPE,
        )
    }
    /// Local cancellation registration at an accepted complete boundary only.
    /// This is NOT a peer broadcast, a preemptible-GPU guarantee or a reusable
    /// worker state. Readiness stays false until coordinated all8 recovery.
    pub fn cancel_at_checked_boundary(&mut self, comm: &mut GpuCollectives) -> Result<()> {
        let result = (|| {
            self.owner(comm)?;
            let queued = self
                .last_boundary
                .as_ref()
                .map_or(0, |b| b.operations_queued);
            let epoch = self
                .last_boundary
                .as_ref()
                .map_or(self.plan.start_epoch(), |b| b.epoch);
            ensure!(comm.queued_operation_count()==queued&&comm.expected_epoch()==epoch,
                "cancel requires a checked accepted boundary; coordinate teardown for in-flight/failed work");
            Ok(())
        })();
        self.failed = true;
        comm.abort_session();
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trusted_admission_clock_counts_queue_and_rejects_expired_or_future_without_GPU() {
        let now = Instant::now();
        assert!(
            checked_admission_elapsed(now - Duration::from_millis(20), 180000).unwrap()
                >= Duration::from_millis(20)
        );
        assert!(checked_admission_elapsed(now - Duration::from_millis(180001), 180000).is_err());
        assert!(checked_admission_elapsed(now + Duration::from_secs(3600), 180000).is_err());
    }
}
