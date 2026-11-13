//! Child of gpu_collectives: separate atomic batch agreement/rearm domain.
//! No CUDA kernel or C1/cancellation state/digest is changed. CPU claim values
//! cannot produce a sealed batch model/request completion.
use super::*;
use crate::dsv41::{
    batch_request_plan::{BatchPhaseClaim, BatchPhasePlan, OPS_PER_STEP},
    batch_request_release::{self, BatchPhaseReleaseReceipt, BatchReleaseConfig},
};
#[derive(Debug)]
pub(super) struct BatchTransportState {
    plan: Arc<BatchPhasePlan>,
    claim: Option<BatchPhaseClaim>,
    release: Option<BatchPhaseReleaseReceipt>,
}
impl BatchTransportState {
    pub(super) fn new(plan: Arc<BatchPhasePlan>) -> Self {
        Self {
            plan,
            claim: None,
            release: None,
        }
    }
}
fn words(
    rank: usize,
    phase: u32,
    old: [u8; 32],
    new: [u8; 32],
    claim: &BatchPhaseClaim,
    epoch: u64,
    queued: usize,
) -> Result<Vec<u32>> {
    let mut out = vec![
        0x31424144,
        1,
        rank as u32,
        8,
        phase,
        epoch as u32,
        (epoch >> 32) as u32,
        queued as u32,
    ];
    for data in [old.to_vec(), new.to_vec(), claim.digest()?.into_bytes()] {
        out.extend(
            data.chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    Ok(out)
}
fn agree_batch(
    link: &mut TpLink,
    rank: usize,
    phase: u32,
    old: [u8; 32],
    new: [u8; 32],
    claim: &BatchPhaseClaim,
    epoch: u64,
    queued: usize,
) -> Result<()> {
    let mine = words(rank, phase, old, new, claim, epoch, queued)?;
    let all = net::hub_all(&mut Control { link, rank }, &mine, mine.len() + 2)?;
    ensure!(all.len() == WORLD, "batch agreement omitted physical rank");
    for (rank, actual) in all.iter().enumerate() {
        let mut expected = mine.clone();
        expected[2] = rank as u32;
        ensure!(
            *actual == expected,
            "rank{rank} atomic group/phase/native prefix/tail agreement differs"
        );
    }
    Ok(())
}
fn check_prefix(
    plan: &BatchPhasePlan,
    claim: &BatchPhaseClaim,
    epoch: u64,
    queued: usize,
) -> Result<()> {
    let step = claim
        .completed_steps
        .checked_sub(1)
        .and_then(|i| plan.steps().get(i))
        .context("batch claim must end after an actual complete model step")?;
    ensure!(
        claim.group_sha256 == plan.group().digest()
            && claim.phase_sha256 == plan.digest()
            && claim.phase == plan.phase()
            && claim.actual_epoch == epoch
            && step.epoch_after == epoch
            && claim.queued_operations == queued
            && step.queued_after == queued
            && queued == claim.completed_steps * OPS_PER_STEP
            && claim.next_operation_id == plan.next_operation_id()
            && claim.next_step_id == plan.next_step_id()
            && claim.next_batch_version == plan.next_batch_version() + claim.completed_steps as u64
            && claim.progress.len() == plan.group().members().len(),
        "batch claim is not the exact completed prefix/full reserved tail of this phase"
    );
    Ok(())
}
impl GpuCollectives {
    pub fn batch_phase_digest(&self) -> Option<&str> {
        self.batch.as_ref().map(|b| b.plan.digest())
    }
    /// Transport-only operation; the sealed GPU batch session supplies a claim
    /// derived from opaque model results, and retains model terminal authority.
    pub(crate) fn finish_batch_phase(
        &mut self,
        claim: &BatchPhaseClaim,
        release: &BatchReleaseConfig,
    ) -> Result<BatchPhaseReleaseReceipt> {
        let result = (|| {
            ensure!(
                !self.failed && !self.finished && self.request.is_none() && self.cycle.is_none(),
                "batch phase failed/closed or different request domain"
            );
            let state = self
                .batch
                .as_ref()
                .context("not an atomic batch transport")?;
            ensure!(
                state.claim.is_none() && state.release.is_none(),
                "batch prefix already released"
            );
            check_prefix(&state.plan, claim, self.expected_epoch, self.next)?;
            self.synchronize()?;
            agree_batch(
                &mut self.link,
                self.rank,
                2,
                self.program.digest,
                self.program.digest,
                claim,
                self.expected_epoch,
                self.next,
            )?;
            self.finished = true;
            self.batch.as_mut().unwrap().claim = Some(claim.clone());
            let receipt = batch_request_release::release(
                self.rank,
                release,
                &self.program_digest_hex(),
                claim,
            )?;
            self.batch.as_mut().unwrap().release = Some(receipt.clone());
            Ok(receipt)
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }
    /// Rearm only after concrete all8 prefix agreement AND TCP lifetime release.
    /// Unused old operations/step IDs remain burned; the actual epoch never rewinds.
    pub(crate) fn begin_batch_phase(&mut self, plan: Arc<BatchPhasePlan>) -> Result<()> {
        let result = (|| {
            ensure!(
                !self.failed && self.finished && self.request.is_none() && self.cycle.is_none(),
                "new atomic phase requires checked prior release on this healthy owner"
            );
            let state = self.batch.as_ref().context("not atomic batch mode")?;
            let claim = state
                .claim
                .as_ref()
                .context("old batch prefix missing")?
                .clone();
            let released = state
                .release
                .as_ref()
                .context("actual batch TCP release missing")?;
            ensure!(
                released.lifetime_released()
                    && released.rank() == self.rank
                    && released.program_sha256() == self.program_digest_hex()
                    && released.claim_sha256() == claim.digest()?,
                "old batch release owner/claim mismatch"
            );
            let first = plan.steps().first().context("new phase cannot be empty")?;
            ensure!(
                plan.start_epoch() == self.expected_epoch
                    && first.operations.embedding == claim.next_operation_id
                    && first.step_id == claim.next_step_id
                    && plan.next_batch_version() == claim.next_batch_version,
                "new phase rewinds actual epoch or abandoned ID/version tails"
            );
            if plan.group().digest() == state.plan.group().digest() {
                ensure!(
                    !claim.terminal() && plan.phase() == state.plan.phase() + 1,
                    "same group phase must follow a nonterminal prefix"
                );
            } else {
                ensure!(
                    claim.terminal() && plan.phase() == 0,
                    "new group requires old whole-group terminal release"
                );
            }
            let program = Program::new_batch_phase(plan.clone())?;
            self.synchronize()?;
            agree_batch(
                &mut self.link,
                self.rank,
                3,
                self.program.digest,
                program.digest,
                &claim,
                self.expected_epoch,
                self.next,
            )?;
            self.program = program;
            self.next = 0;
            self.finished = false;
            self.batch = Some(BatchTransportState::new(plan));
            Ok(())
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }
}

impl GpuCollectives {
    /// Every rank proves its own sealed physical cleanup reset, then agrees on
    /// the common reset lane generations/version before any next group is Ready.
    pub(crate) fn confirm_batch_cleanup_reset(
        &mut self,
        reset: &crate::dsv41::gpu_batch_model::CheckedBatchModelReset,
        config: &BatchReleaseConfig,
    ) -> Result<BatchPhaseReleaseReceipt> {
        let result = (|| {
            let state = self
                .batch
                .as_ref()
                .context("batch cleanup domain required")?;
            let claim = state
                .claim
                .as_ref()
                .context("terminal batch claim missing")?
                .clone();
            ensure!(
                !self.failed
                    && self.finished
                    && claim.terminal()
                    && state
                        .release
                        .as_ref()
                        .is_some_and(BatchPhaseReleaseReceipt::lifetime_released)
                    && reset.model().completion_owner == self.completion_identity
                    && reset.model().rank == self.rank
                    && reset.epoch() == self.expected_epoch,
                "batch cleanup requires actual terminal phase release and sealed reset owner/epoch"
            );
            let expected: Vec<_> = state
                .plan
                .group()
                .lanes()
                .iter()
                .map(|key| {
                    Ok(crate::dsv41::scheduler::LaneKey {
                        generation: key
                            .generation
                            .checked_add(1)
                            .context("cleanup generation overflow")?,
                        ..*key
                    })
                })
                .collect::<Result<_>>()?;
            ensure!(
                reset.lanes() == expected && reset.next_batch_version() == claim.next_batch_version,
                "all original lanes require exact next-generation cleanup reset"
            );
            let cleanup_sha = crate::dsv41::batch_request_plan::digest(
                b"dsv41-atomic-cleanup-reset-agreement-v1\0",
                &(
                    claim.digest()?,
                    reset.lanes(),
                    reset.epoch(),
                    reset.next_batch_version(),
                ),
            )?;
            self.synchronize()?;
            // New phase code and common cleanup digest, distinct from both C1
            // and phase-prefix release; no local model/stream IDs cross ranks.
            let mut mine = vec![
                0x31424144,
                1,
                self.rank as u32,
                8,
                4,
                self.expected_epoch as u32,
                (self.expected_epoch >> 32) as u32,
                self.next as u32,
            ];
            mine.extend(
                cleanup_sha
                    .as_bytes()
                    .chunks_exact(4)
                    .map(|x| u32::from_le_bytes(x.try_into().unwrap())),
            );
            let all = net::hub_all(
                &mut Control {
                    link: &mut self.link,
                    rank: self.rank,
                },
                &mine,
                mine.len() + 2,
            )?;
            ensure!(all.len() == 8, "batch reset agreement omitted a rank");
            for (rank, actual) in all.iter().enumerate() {
                let mut expected = mine.clone();
                expected[2] = rank as u32;
                ensure!(
                    *actual == expected,
                    "rank{rank} cleanup generations/native version mismatch"
                );
            }
            batch_request_release::release_digest(
                self.rank,
                config,
                &self.program_digest_hex(),
                &cleanup_sha,
            )
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }
}
