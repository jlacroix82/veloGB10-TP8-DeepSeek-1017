//! CPU-only 1017 request/lane state. No kernel, cache restore, HTTP or tool execution.
//! Atomic admission registers a complete group between steps; a later step can
//! still split its prefill under its explicit token budget. Nothing here proves
//! a GPU batch or qualifies model serving.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

use super::config::MODEL_REVISION;
use super::engram::{EngramHashState, TOKENIZER_SHA256, TOKEN_COUNT};

const MAX_CONTEXT: usize = 1_048_576;
const MAX_TERMINAL_RECORDS: usize = 1024;
const MAX_ACCEPTED_TOKENS_PER_STEP: usize = 6;

fn native_digest(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn valid_name(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct LaneKey {
    pub boot: Uuid,
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrefixIdentity {
    pub boot: Uuid,
    pub cache_generation: u64,
    pub engine_sha256: String,
    pub model_revision: String,
    pub tokenizer_sha256: String,
    pub session_namespace: String,
    pub native_prefix_tokens: usize,
    pub native_prefix_ids_sha256_u32le: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum WorkScope {
    CappedSpeed,
    NaturalCompletion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestSpec {
    pub request_id: String,
    pub scope: WorkScope,
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub prefix: PrefixIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum StepWork {
    Prefill { start: usize, end: usize },
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepLane {
    pub lane: LaneKey,
    pub admission_group: u64,
    pub work: StepWork,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepTicket {
    pub id: u64,
    pub lanes: Vec<StepLane>,
    /// Fresh starts actually selected by this CPU plan, grouped by atomic admission.
    pub fresh_prefill_groups: Vec<(u64, usize)>,
    pub prefill_tokens: usize,
    pub decode_lanes: usize,
    next_decode_cursor: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishSignal {
    BackendStop,
    BackendLength,
    ContentFilter,
    MatchedUserStop(String),
    UnannouncedEof,
    TimedOut,
    Canceled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneCommit {
    Prefilled {
        lane: LaneKey,
    },
    Decoded {
        lane: LaneKey,
        accepted_native_ids: Vec<u32>,
        finish: Option<FinishSignal>,
    },
}

impl LaneCommit {
    fn lane(&self) -> LaneKey {
        match self {
            Self::Prefilled { lane } | Self::Decoded { lane, .. } => *lane,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum TerminalReason {
    BackendStop,
    BackendLength,
    LocalOutputCap,
    MatchedUserStop(String),
    ContentFilter,
    UnannouncedEof,
    TimedOut,
    Canceled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TerminalRecord {
    pub request_id: String,
    pub lane: LaneKey,
    pub admission_group: u64,
    pub scope: WorkScope,
    pub prefix: PrefixIdentity,
    pub requested_prompt_tokens: usize,
    pub prefilled_native_tokens: usize,
    pub generated_native_tokens: usize,
    pub generated_ids_sha256_u32le: String,
    pub requested_output_cap: usize,
    pub reason: TerminalReason,
    /// Native-count/cap eligibility only. Output equivalence, cache state and
    /// whole-wave correctness must still be qualified by the measurement harness.
    pub capped_speed_count_eligible: bool,
    /// A normally terminated response is not proof of a correct coding/tool task.
    pub natural_response_complete: bool,
    pub externally_checked_task_success: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaneSnapshot {
    pub lane: LaneKey,
    pub request_id: String,
    pub scope: WorkScope,
    pub admission_group: u64,
    pub prefilled_native_tokens: usize,
    pub generated_native_tokens: usize,
    pub history_position: usize,
    /// No physical prefix restoration exists in this CPU state module.
    pub verified_cached_prompt_tokens: usize,
}

#[derive(Debug, Clone)]
struct Active {
    spec: RequestSpec,
    lane: LaneKey,
    group: u64,
    prefilled: usize,
    generated: Vec<u32>,
}

#[derive(Debug, Clone)]
struct Slot {
    generation: u64,
    active: Option<Active>,
}

#[derive(Debug, Clone)]
enum History {
    Verified(EngramHashState),
    // Only a call/position spy. It does not replace the Engram mathematical oracle.
    #[cfg(test)]
    Spy(Vec<Vec<u32>>),
}

impl History {
    fn position(&self, lane: usize) -> Result<usize> {
        match self {
            Self::Verified(v) => v.position(lane),
            #[cfg(test)]
            Self::Spy(v) => v.get(lane).map(Vec::len).context("unknown spy lane"),
        }
    }
    fn reset(&mut self, lane: usize) -> Result<()> {
        match self {
            Self::Verified(v) => v.reset(lane),
            #[cfg(test)]
            Self::Spy(v) => {
                v.get_mut(lane).context("unknown spy lane")?.clear();
                Ok(())
            }
        }
    }
    fn append(&mut self, lane: usize, start: usize, ids: &[u32]) -> Result<()> {
        match self {
            Self::Verified(v) => {
                v.append_at(lane, start, ids, None)?;
                Ok(())
            }
            #[cfg(test)]
            Self::Spy(v) => {
                let row = v.get_mut(lane).context("unknown spy lane")?;
                ensure!(row.len() == start, "stale spy position");
                row.extend_from_slice(ids);
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scheduler {
    boot: Uuid,
    cache_generation: u64,
    engine_sha256: String,
    context_limit: usize,
    slots: Vec<Slot>,
    history: History,
    next_group: u64,
    next_step: u64,
    decode_cursor: usize,
    pending: Option<StepTicket>,
    terminal: Vec<TerminalRecord>,
}

impl Scheduler {
    /// Caller supplies the measured binary/source digest; this metadata module
    /// binds that identity but does not inspect an executable or allocate KV.
    /// Public construction requires the Engram module's full verified token map.
    pub fn new(
        history: EngramHashState,
        capacity: usize,
        context_limit: usize,
        engine_sha256: &str,
    ) -> Result<Self> {
        Self::with_history(
            History::Verified(history),
            capacity,
            context_limit,
            engine_sha256,
        )
    }
    fn with_history(
        history: History,
        capacity: usize,
        context_limit: usize,
        engine_sha256: &str,
    ) -> Result<Self> {
        ensure!(
            [1, 8, 16].contains(&capacity),
            "CPU scheduler capacity must be1,8or16"
        );
        ensure!(
            (1..=MAX_CONTEXT).contains(&context_limit),
            "invalid1017 context limit"
        );
        ensure!(
            engine_sha256.len() == 64
                && engine_sha256
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "expected explicit lowercase binary/source SHA256"
        );
        for lane in 0..capacity {
            ensure!(
                history.position(lane)? == 0,
                "scheduler requires fresh Engram lanes"
            );
        }
        ensure!(
            history.position(capacity).is_err(),
            "Engram lane capacity differs from scheduler"
        );
        Ok(Self {
            boot: Uuid::new_v4(),
            cache_generation: 1,
            engine_sha256: engine_sha256.into(),
            context_limit,
            slots: vec![
                Slot {
                    generation: 0,
                    active: None
                };
                capacity
            ],
            history,
            next_group: 1,
            next_step: 1,
            decode_cursor: 0,
            pending: None,
            terminal: Vec::new(),
        })
    }
    pub fn serving_supported(&self) -> bool {
        false
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.active.is_some()).count()
    }
    pub fn cache_generation(&self) -> u64 {
        self.cache_generation
    }

    pub fn prefix_identity(
        &self,
        namespace: &str,
        prompt_ids: &[u32],
        prefix_tokens: usize,
    ) -> Result<PrefixIdentity> {
        ensure!(valid_name(namespace), "invalid private session namespace");
        ensure!(prefix_tokens <= prompt_ids.len(), "prefix exceeds prompt");
        ensure!(
            prompt_ids.iter().all(|&i| (i as usize) < TOKEN_COUNT),
            "token outside1017 vocabulary"
        );
        Ok(PrefixIdentity {
            boot: self.boot,
            cache_generation: self.cache_generation,
            engine_sha256: self.engine_sha256.clone(),
            model_revision: MODEL_REVISION.into(),
            tokenizer_sha256: TOKENIZER_SHA256.into(),
            session_namespace: namespace.into(),
            native_prefix_tokens: prefix_tokens,
            native_prefix_ids_sha256_u32le: native_digest(&prompt_ids[..prefix_tokens]),
        })
    }

    fn validate_request(&self, r: &RequestSpec) -> Result<()> {
        ensure!(valid_name(&r.request_id), "invalid request ID");
        ensure!(
            !r.prompt_ids.is_empty() && r.max_new_tokens > 0,
            "empty prompt or zero output budget"
        );
        ensure!(
            r.prompt_ids
                .len()
                .checked_add(r.max_new_tokens)
                .is_some_and(|v| v <= self.context_limit),
            "prompt/output budget exceeds declared context"
        );
        ensure!(
            r.prefix
                == self.prefix_identity(
                    &r.prefix.session_namespace,
                    &r.prompt_ids,
                    r.prefix.native_prefix_tokens
                )?,
            "stale or mismatched prefix/model/tokenizer/boot identity"
        );
        ensure!(
            !self
                .slots
                .iter()
                .filter_map(|s| s.active.as_ref())
                .any(|a| a.spec.request_id == r.request_id)
                && !self.terminal.iter().any(|a| a.request_id == r.request_id),
            "duplicate request ID"
        );
        Ok(())
    }

    /// All-or-none registration. No step can interleave these registrations.
    pub fn admit_atomic(&mut self, requests: Vec<RequestSpec>) -> Result<Vec<LaneKey>> {
        ensure!(
            self.pending.is_none(),
            "a step is in flight; register between steps"
        );
        ensure!(
            !requests.is_empty() && requests.len() <= self.capacity(),
            "invalid atomic group size"
        );
        ensure!(
            self.terminal.len() + self.active_count() + requests.len() <= MAX_TERMINAL_RECORDS,
            "drain terminal evidence before admitting more work"
        );
        let free: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active.is_none())
            .map(|(i, _)| i)
            .collect();
        ensure!(
            requests.len() <= free.len(),
            "insufficient lanes for the complete atomic group"
        );
        let mut names = BTreeSet::new();
        for r in &requests {
            self.validate_request(r)?;
            ensure!(
                names.insert(r.request_id.clone()),
                "duplicate ID within atomic group"
            );
        }
        let mut staged = self.clone();
        let group = staged.next_group;
        staged.next_group = group.checked_add(1).context("admission group overflow")?;
        let mut keys = Vec::new();
        for (slot, spec) in free.into_iter().zip(requests) {
            let generation = staged.slots[slot]
                .generation
                .checked_add(1)
                .context("lane generation overflow")?;
            let lane = LaneKey {
                boot: staged.boot,
                slot,
                generation,
            };
            staged.history.reset(slot)?;
            staged.slots[slot] = Slot {
                generation,
                active: Some(Active {
                    spec,
                    lane,
                    group,
                    prefilled: 0,
                    generated: Vec::new(),
                }),
            };
            keys.push(lane);
        }
        *self = staged;
        Ok(keys)
    }

    /// A budgeted CPU plan, not proof that any GPU work has happened.
    pub fn plan_step(
        &mut self,
        prefill_budget: usize,
        decode_lane_budget: usize,
    ) -> Result<Option<StepTicket>> {
        ensure!(self.pending.is_none(), "previous step has no commit/abort");
        ensure!(
            decode_lane_budget <= self.capacity(),
            "decode lane budget exceeds capacity"
        );
        if self.active_count() == 0 {
            return Ok(None);
        }
        let mut lanes = Vec::new();
        let mut next_cursor = self.decode_cursor;
        for offset in 0..self.capacity() {
            let slot = (self.decode_cursor + offset) % self.capacity();
            if let Some(a) = self.slots[slot]
                .active
                .as_ref()
                .filter(|a| a.prefilled == a.spec.prompt_ids.len())
            {
                if lanes.len() == decode_lane_budget {
                    break;
                }
                lanes.push(StepLane {
                    lane: a.lane,
                    admission_group: a.group,
                    work: StepWork::Decode,
                });
                next_cursor = (slot + 1) % self.capacity();
            }
        }
        let decode_lanes = lanes.len();
        let mut remaining = prefill_budget;
        let mut groups = BTreeMap::new();
        let mut prefill: Vec<_> = self
            .slots
            .iter()
            .filter_map(|s| s.active.as_ref())
            .filter(|a| a.prefilled < a.spec.prompt_ids.len())
            .collect();
        prefill.sort_by_key(|a| (a.group, a.lane.slot));
        for a in prefill {
            if remaining == 0 {
                break;
            }
            let count = (a.spec.prompt_ids.len() - a.prefilled).min(remaining);
            lanes.push(StepLane {
                lane: a.lane,
                admission_group: a.group,
                work: StepWork::Prefill {
                    start: a.prefilled,
                    end: a.prefilled + count,
                },
            });
            if a.prefilled == 0 {
                *groups.entry(a.group).or_insert(0) += 1;
            }
            remaining -= count;
        }
        ensure!(
            !lanes.is_empty(),
            "active work cannot advance under these zero/insufficient budgets"
        );
        let ticket = StepTicket {
            id: self.next_step,
            lanes,
            fresh_prefill_groups: groups.into_iter().collect(),
            prefill_tokens: prefill_budget - remaining,
            decode_lanes,
            next_decode_cursor: next_cursor,
        };
        self.next_step = self.next_step.checked_add(1).context("step ID overflow")?;
        self.pending = Some(ticket.clone());
        Ok(Some(ticket))
    }

    pub fn commit_step(&mut self, ticket: u64, results: Vec<LaneCommit>) -> Result<()> {
        let mut staged = self.clone();
        staged.commit_inner(ticket, results)?;
        *self = staged;
        Ok(())
    }
    fn commit_inner(&mut self, ticket: u64, results: Vec<LaneCommit>) -> Result<()> {
        let plan = self.pending.take().context("no step awaits commit")?;
        ensure!(plan.id == ticket, "stale step ticket");
        ensure!(
            results.len() == plan.lanes.len(),
            "not all planned lanes returned"
        );
        let mut by_slot = BTreeMap::new();
        for result in results {
            ensure!(
                by_slot.insert(result.lane().slot, result).is_none(),
                "duplicate lane result"
            );
        }
        for planned in &plan.lanes {
            let result = by_slot
                .remove(&planned.lane.slot)
                .context("missing planned lane result")?;
            ensure!(
                result.lane() == planned.lane,
                "wrong boot/lane/generation association"
            );
            let mut active = self.slots[planned.lane.slot]
                .active
                .take()
                .context("planned lane no longer active")?;
            ensure!(active.lane == planned.lane, "stale active lane");
            let mut terminal = None;
            match (&planned.work, result) {
                (StepWork::Prefill { start, end }, LaneCommit::Prefilled { .. }) => {
                    ensure!(
                        *start == active.prefilled && *end <= active.spec.prompt_ids.len(),
                        "stale prefill range"
                    );
                    self.history.append(
                        active.lane.slot,
                        *start,
                        &active.spec.prompt_ids[*start..*end],
                    )?;
                    active.prefilled = *end;
                }
                (
                    StepWork::Decode,
                    LaneCommit::Decoded {
                        accepted_native_ids,
                        finish,
                        ..
                    },
                ) => {
                    ensure!(
                        active.prefilled == active.spec.prompt_ids.len(),
                        "decode before complete prefill"
                    );
                    ensure!(
                        accepted_native_ids.len() <= MAX_ACCEPTED_TOKENS_PER_STEP,
                        "native commit exceeds bounded1017 step"
                    );
                    ensure!(
                        !accepted_native_ids.is_empty() || finish.is_some(),
                        "decode made no progress and supplied no finish"
                    );
                    ensure!(
                        active
                            .generated
                            .len()
                            .checked_add(accepted_native_ids.len())
                            .is_some_and(|v| v <= active.spec.max_new_tokens),
                        "native output cap exceeded"
                    );
                    ensure!(
                        accepted_native_ids
                            .iter()
                            .all(|&i| (i as usize) < TOKEN_COUNT),
                        "generated token outside1017 vocabulary"
                    );
                    self.history.append(
                        active.lane.slot,
                        active.prefilled + active.generated.len(),
                        &accepted_native_ids,
                    )?;
                    active.generated.extend_from_slice(&accepted_native_ids);
                    terminal = match finish {
                        Some(FinishSignal::BackendStop) => Some(TerminalReason::BackendStop),
                        Some(FinishSignal::BackendLength) => Some(TerminalReason::BackendLength),
                        Some(FinishSignal::ContentFilter) => Some(TerminalReason::ContentFilter),
                        Some(FinishSignal::MatchedUserStop(s)) => {
                            ensure!(
                                !s.is_empty() && s.len() <= 1024,
                                "invalid matched stop proof"
                            );
                            Some(TerminalReason::MatchedUserStop(s))
                        }
                        Some(FinishSignal::UnannouncedEof) => Some(TerminalReason::UnannouncedEof),
                        Some(FinishSignal::TimedOut) => Some(TerminalReason::TimedOut),
                        Some(FinishSignal::Canceled) => Some(TerminalReason::Canceled),
                        None if active.generated.len() == active.spec.max_new_tokens => {
                            Some(TerminalReason::LocalOutputCap)
                        }
                        None => None,
                    };
                }
                _ => ensure!(false, "lane result kind differs from planned work"),
            }
            if let Some(reason) = terminal {
                self.finish_active(active, reason)?;
            } else {
                self.slots[planned.lane.slot].active = Some(active);
            }
        }
        self.decode_cursor = plan.next_decode_cursor;
        Ok(())
    }

    fn finish_active(&mut self, active: Active, reason: TerminalReason) -> Result<()> {
        ensure!(
            self.terminal.len() < MAX_TERMINAL_RECORDS,
            "terminal evidence must be drained"
        );
        let natural = matches!(
            reason,
            TerminalReason::BackendStop | TerminalReason::MatchedUserStop(_)
        );
        let length = matches!(
            reason,
            TerminalReason::BackendLength | TerminalReason::LocalOutputCap
        );
        let record = TerminalRecord {
            request_id: active.spec.request_id,
            lane: active.lane,
            admission_group: active.group,
            scope: active.spec.scope,
            prefix: active.spec.prefix,
            requested_prompt_tokens: active.spec.prompt_ids.len(),
            prefilled_native_tokens: active.prefilled,
            generated_native_tokens: active.generated.len(),
            generated_ids_sha256_u32le: native_digest(&active.generated),
            requested_output_cap: active.spec.max_new_tokens,
            reason,
            capped_speed_count_eligible: active.spec.scope == WorkScope::CappedSpeed
                && length
                && active.generated.len() == active.spec.max_new_tokens,
            natural_response_complete: active.spec.scope == WorkScope::NaturalCompletion && natural,
            externally_checked_task_success: None,
        };
        self.history.reset(active.lane.slot)?;
        self.terminal.push(record);
        Ok(())
    }

    /// Caller must first resolve backend work; state never reclaims an in-flight lane.
    pub fn abort_request(&mut self, lane: LaneKey, reason: FinishSignal) -> Result<()> {
        ensure!(
            self.pending.is_none(),
            "resolve/abandon the pending step before reclaiming a lane"
        );
        ensure!(
            matches!(
                reason,
                FinishSignal::UnannouncedEof | FinishSignal::TimedOut | FinishSignal::Canceled
            ),
            "abort requires a failure/cancel reason"
        );
        let mut staged = self.clone();
        let slot = staged.slots.get_mut(lane.slot).context("unknown lane")?;
        let active = slot.active.take().context("lane is not active")?;
        ensure!(active.lane == lane, "stale lane handle");
        let why = match reason {
            FinishSignal::UnannouncedEof => TerminalReason::UnannouncedEof,
            FinishSignal::TimedOut => TerminalReason::TimedOut,
            _ => TerminalReason::Canceled,
        };
        staged.finish_active(active, why)?;
        *self = staged;
        Ok(())
    }

    /// Discard a failed backend step after its owner has stopped that work. Every
    /// participating lane is retained as failed; unscheduled lanes keep their state.
    pub fn abandon_step(&mut self, ticket: u64, reason: FinishSignal) -> Result<()> {
        let plan = self.pending.as_ref().context("no pending step")?;
        ensure!(plan.id == ticket, "stale failed-step ticket");
        let keys: Vec<_> = plan.lanes.iter().map(|p| p.lane).collect();
        let mut staged = self.clone();
        staged.pending = None;
        for key in keys {
            staged.abort_request(key, reason.clone())?;
        }
        *self = staged;
        Ok(())
    }

    pub fn snapshots(&self) -> Result<Vec<LaneSnapshot>> {
        self.slots
            .iter()
            .filter_map(|s| s.active.as_ref())
            .map(|a| {
                Ok(LaneSnapshot {
                    lane: a.lane,
                    request_id: a.spec.request_id.clone(),
                    scope: a.spec.scope,
                    admission_group: a.group,
                    prefilled_native_tokens: a.prefilled,
                    generated_native_tokens: a.generated.len(),
                    history_position: self.history.position(a.lane.slot)?,
                    verified_cached_prompt_tokens: 0,
                })
            })
            .collect()
    }
    pub fn terminal_records(&self) -> &[TerminalRecord] {
        &self.terminal
    }
    /// Ownership of all failures and completions passes to the evidence writer.
    pub fn take_terminal_records(&mut self) -> Vec<TerminalRecord> {
        std::mem::take(&mut self.terminal)
    }
    pub fn record_checked_outcome(&mut self, lane: LaneKey, passed: bool) -> Result<()> {
        let record = self
            .terminal
            .iter_mut()
            .find(|r| r.lane == lane)
            .context("unknown terminal attempt")?;
        ensure!(
            record.scope == WorkScope::NaturalCompletion,
            "capped speed is not task quality"
        );
        ensure!(
            !passed || record.natural_response_complete,
            "failed/capped/EOF attempt cannot be a successful completion"
        );
        ensure!(
            record.externally_checked_task_success.is_none(),
            "outcome already recorded"
        );
        record.externally_checked_task_success = Some(passed);
        Ok(())
    }
    pub fn rotate_prefix_generation(&mut self) -> Result<u64> {
        ensure!(
            self.active_count() == 0 && self.pending.is_none(),
            "rotate cache identity only when idle"
        );
        self.cache_generation = self
            .cache_generation
            .checked_add(1)
            .context("cache generation overflow")?;
        Ok(self.cache_generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scheduler(capacity: usize) -> Scheduler {
        Scheduler::with_history(
            History::Spy(vec![Vec::new(); capacity]),
            capacity,
            4096,
            &"a".repeat(64),
        )
        .unwrap()
    }
    fn request(s: &Scheduler, id: &str, scope: WorkScope) -> RequestSpec {
        let prompt = vec![0, 128803, 42];
        RequestSpec {
            request_id: id.into(),
            scope,
            prompt_ids: prompt.clone(),
            max_new_tokens: 3,
            prefix: s.prefix_identity("session-a", &prompt, 2).unwrap(),
        }
    }
    fn prefill(s: &mut Scheduler) {
        let p = s.plan_step(4096, 16.min(s.capacity())).unwrap().unwrap();
        let results = p
            .lanes
            .iter()
            .map(|r| LaneCommit::Prefilled { lane: r.lane })
            .collect();
        s.commit_step(p.id, results).unwrap();
    }

    #[test]
    fn atomic_groups_register_all_members_and_budgeted_plans_may_split() {
        for c in [1, 8, 16] {
            let mut s = scheduler(c);
            let requests = (0..c)
                .map(|i| request(&s, &format!("r{i}"), WorkScope::CappedSpeed))
                .collect();
            let keys = s.admit_atomic(requests).unwrap();
            assert_eq!(keys.len(), c);
            let p = s.plan_step(3 * c, c).unwrap().unwrap();
            assert_eq!(p.fresh_prefill_groups, vec![(1, c)]);
            assert_eq!(p.prefill_tokens, 3 * c);
            assert!(!s.serving_supported());
        }
        let mut s = scheduler(8);
        s.admit_atomic(
            (0..8)
                .map(|i| request(&s, &format!("r{i}"), WorkScope::CappedSpeed))
                .collect(),
        )
        .unwrap();
        let p = s.plan_step(12, 8).unwrap().unwrap();
        assert_eq!(p.fresh_prefill_groups, vec![(1, 4)]);
        s.commit_step(
            p.id,
            p.lanes
                .iter()
                .map(|r| LaneCommit::Prefilled { lane: r.lane })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            s.plan_step(12, 0).unwrap().unwrap().fresh_prefill_groups,
            vec![(1, 4)]
        );
    }

    #[test]
    fn bad_group_and_wrong_step_result_leave_every_lane_unchanged() {
        let mut s = scheduler(8);
        let mut bad = request(&s, "bad", WorkScope::NaturalCompletion);
        bad.prompt_ids[1] = TOKEN_COUNT as u32;
        assert!(s
            .admit_atomic(vec![request(&s, "good", WorkScope::NaturalCompletion), bad])
            .is_err());
        assert_eq!(s.active_count(), 0);
        let keys = s
            .admit_atomic(vec![
                request(&s, "one", WorkScope::NaturalCompletion),
                request(&s, "two", WorkScope::NaturalCompletion),
            ])
            .unwrap();
        let p = s.plan_step(6, 8).unwrap().unwrap();
        let before = s.snapshots().unwrap();
        assert!(s
            .commit_step(
                p.id,
                vec![
                    LaneCommit::Prefilled { lane: keys[0] },
                    LaneCommit::Prefilled { lane: keys[0] }
                ]
            )
            .is_err());
        assert_eq!(s.snapshots().unwrap(), before);
        s.commit_step(
            p.id,
            vec![
                LaneCommit::Prefilled { lane: keys[1] },
                LaneCommit::Prefilled { lane: keys[0] },
            ],
        )
        .unwrap();
        assert!(s.commit_step(p.id, Vec::new()).is_err());
    }

    #[test]
    fn native_counts_do_not_count_steps_and_speed_caps_are_not_quality() {
        let mut s = scheduler(1);
        let key = s
            .admit_atomic(vec![request(&s, "speed", WorkScope::CappedSpeed)])
            .unwrap()[0];
        prefill(&mut s);
        let p = s.plan_step(0, 1).unwrap().unwrap();
        s.commit_step(
            p.id,
            vec![LaneCommit::Decoded {
                lane: key,
                accepted_native_ids: vec![17, 18],
                finish: None,
            }],
        )
        .unwrap();
        assert_eq!(s.snapshots().unwrap()[0].generated_native_tokens, 2);
        assert_eq!(s.snapshots().unwrap()[0].history_position, 5);
        let p = s.plan_step(0, 1).unwrap().unwrap();
        s.commit_step(
            p.id,
            vec![LaneCommit::Decoded {
                lane: key,
                accepted_native_ids: vec![19],
                finish: None,
            }],
        )
        .unwrap();
        let r = &s.terminal_records()[0];
        assert_eq!(r.generated_native_tokens, 3);
        assert!(r.capped_speed_count_eligible);
        assert!(!r.natural_response_complete);
        assert!(s.record_checked_outcome(key, true).is_err());
    }

    #[test]
    fn eos_timeout_and_length_never_become_successful_natural_tasks() {
        for signal in [
            FinishSignal::UnannouncedEof,
            FinishSignal::TimedOut,
            FinishSignal::BackendLength,
        ] {
            let mut s = scheduler(1);
            let key = s
                .admit_atomic(vec![request(&s, "natural", WorkScope::NaturalCompletion)])
                .unwrap()[0];
            prefill(&mut s);
            let p = s.plan_step(0, 1).unwrap().unwrap();
            s.commit_step(
                p.id,
                vec![LaneCommit::Decoded {
                    lane: key,
                    accepted_native_ids: vec![17],
                    finish: Some(signal),
                }],
            )
            .unwrap();
            assert!(!s.terminal_records()[0].natural_response_complete);
            assert!(s.record_checked_outcome(key, true).is_err());
            s.record_checked_outcome(key, false).unwrap();
            assert_eq!(s.terminal_records()[0].generated_native_tokens, 1);
        }
    }

    #[test]
    fn lane_reuse_resets_history_and_rejects_old_generation() {
        let mut s = scheduler(1);
        let old = s
            .admit_atomic(vec![request(&s, "first", WorkScope::NaturalCompletion)])
            .unwrap()[0];
        prefill(&mut s);
        s.abort_request(old, FinishSignal::Canceled).unwrap();
        let new = s
            .admit_atomic(vec![request(&s, "second", WorkScope::NaturalCompletion)])
            .unwrap()[0];
        assert_eq!(old.slot, new.slot);
        assert!(new.generation > old.generation);
        assert_eq!(s.snapshots().unwrap()[0].history_position, 0);
        assert!(s.abort_request(old, FinishSignal::Canceled).is_err());
        prefill(&mut s);
        let p = s.plan_step(0, 1).unwrap().unwrap();
        assert!(s
            .commit_step(
                p.id,
                vec![LaneCommit::Decoded {
                    lane: old,
                    accepted_native_ids: vec![7],
                    finish: None
                }]
            )
            .is_err());
        s.commit_step(
            p.id,
            vec![LaneCommit::Decoded {
                lane: new,
                accepted_native_ids: vec![8],
                finish: Some(FinishSignal::BackendStop),
            }],
        )
        .unwrap();
        assert!(s.terminal_records()[1].natural_response_complete);
        s.record_checked_outcome(new, true).unwrap();
    }

    #[test]
    fn prefix_namespace_generation_and_boot_never_alias_or_claim_a_cache_hit() {
        let mut s = scheduler(8);
        let old = request(&s, "old", WorkScope::NaturalCompletion);
        let mut other = old.clone();
        other.request_id = "other".into();
        other.prefix.session_namespace = "session-b".into();
        assert_ne!(old.prefix, other.prefix);
        assert_ne!(old.prefix.boot, scheduler(8).boot);
        s.rotate_prefix_generation().unwrap();
        assert!(s.admit_atomic(vec![old]).is_err());
        let key = s
            .admit_atomic(vec![request(&s, "new", WorkScope::NaturalCompletion)])
            .unwrap()[0];
        assert_eq!(s.snapshots().unwrap()[0].verified_cached_prompt_tokens, 0);
        assert!(s.rotate_prefix_generation().is_err());
        let p = s.plan_step(3, 8).unwrap().unwrap();
        assert!(s
            .admit_atomic(vec![request(&s, "queued", WorkScope::NaturalCompletion)])
            .is_err());
        s.abandon_step(p.id, FinishSignal::TimedOut).unwrap();
        assert_eq!(s.terminal_records()[0].lane, key);
    }

    #[test]
    fn prefill_chunking_lane_isolation_and_decode_round_robin_are_bounded() {
        let mut s = scheduler(8);
        s.admit_atomic(vec![
            request(&s, "a", WorkScope::NaturalCompletion),
            request(&s, "b", WorkScope::NaturalCompletion),
        ])
        .unwrap();
        let p = s.plan_step(2, 8).unwrap().unwrap();
        s.commit_step(
            p.id,
            p.lanes
                .iter()
                .map(|r| LaneCommit::Prefilled { lane: r.lane })
                .collect(),
        )
        .unwrap();
        let snapshot = s.snapshots().unwrap();
        assert_eq!(snapshot[0].history_position, 2);
        assert_eq!(snapshot[1].history_position, 0);
        prefill(&mut s);
        let first = s.plan_step(0, 1).unwrap().unwrap();
        let a = first.lanes[0].lane;
        s.commit_step(
            first.id,
            vec![LaneCommit::Decoded {
                lane: a,
                accepted_native_ids: vec![5],
                finish: None,
            }],
        )
        .unwrap();
        let next = s.plan_step(0, 1).unwrap().unwrap();
        assert_ne!(a, next.lanes[0].lane);
    }
}
