//! CPU-only atomic ownership of the measured nested-conversation batch schema.
//! No GPU submission, model/reset proof, terminal proof or HTTP listener exists
//! here. A later sealed batch session must independently validate actual model
//! steps, dynamic EOS shapes, all-rank agreement and transport release.
use super::{
    config::MODEL_REVISION,
    model_session_plan::{GreedySemantics, RequestBinding, RequestBudget, StopPolicy},
    native_api_contract::{
        CpuBridge, PlannedAtomicGroup, PlannedChat, PlanningLimits, ATOMIC_ROUTE,
    },
    position::{FrequencyBackend, FrequencyProfileIdentity},
    scheduler::LaneKey,
};
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const MAX_MEMBERS: usize = 16;
pub const MAX_LEARNED_PREFIX: usize = 512;
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAX_WALL_MILLIS: u64 = 180_000;

pub fn capabilities() -> Value {
    json!({"CPU_atomic_preparation":true,"CPU_all_or_none_slot_ownership":true,
        "maximum_members":MAX_MEMBERS,"context_envelope_tokens":MAX_LEARNED_PREFIX,
        "model_execution":false,"GPU_batching_verified":false,"HTTP_serving":false,
        "terminal_authority":false,"runtime_reset_authority":false,
        "qualification":"CPU ownership only; actual packed scheduling and sealed runtime session remain unqualified"})
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex(v: &str) -> bool {
    v.len() == 64
        && v.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Immutable official rendering plus exact requested/effective semantics.
/// Returning it causes no logical-slot, device, tool or process side effect.
pub struct PreparedAtomicRequest {
    group: PlannedAtomicGroup,
    original_body_sha256: String,
    member_semantics: Vec<Value>,
    digest: String,
}
impl PreparedAtomicRequest {
    pub fn prepare(
        bridge: &CpuBridge,
        path: &str,
        header: Option<&str>,
        body: &Value,
    ) -> Result<Self> {
        let limits = PlanningLimits {
            context_tokens: 512,
            prompt_tokens: 512,
            output_tokens: 512,
            atomic_requests: MAX_MEMBERS,
            request_bytes: MAX_REQUEST_BYTES,
        };
        // Reuse the official bridge's ordered, full-group validation. There is
        // no partial group or reservation if the LAST member is invalid.
        let group = bridge.prepare_atomic(path, header, body.clone(), limits)?;
        let mut semantics = Vec::with_capacity(group.members().len());
        for (index, member) in group.members().iter().enumerate() {
            let s = member.semantics();
            ensure!(!s.stream && !s.ignore_eos && s.min_tokens == 0
                && s.effective_temperature == 0. && s.effective_top_p == 1.
                && s.presence_penalty == 0. && s.frequency_penalty == 0.,
                "member{index}: initial packed profile requires unmodified greedy EOS-or-cap semantics");
            ensure!(s.stop_sequences.is_empty() && s.forced_tool_constraint.is_none()
                && s.cache_namespace.is_none(),
                "member{index}: user stops, forced tools and cache reuse require a separately qualified packed session");
            let original = member.original_request();
            // Auto is unforced native generation: the original official
            // prompt/parser already carries definitions and call/result IDs.
            // A packed row has no qualified token mask; never treat a named,
            // required or strict request as Auto by dropping its constraint.
            ensure!(
                original.get("tool_choice").is_none_or(|v| v.is_null()
                    || matches!(v.as_str(), Some("auto" | "none"))),
                "member{index}: packed tools support unforced auto/none only; named/required constraints need a qualified per-row sampler"
            );
            let messages = original["messages"]
                .as_array()
                .context("prepared conversation must be array")?;
            // The SAME CpuBridge has already checked unknown fields, ordered
            // tool results, duplicate/missing call IDs and supported text
            // content. Rendering is never reimplemented at this boundary.
            ensure!(
                messages.iter().all(|m| matches!(
                    m["role"].as_str(),
                    Some("system" | "user" | "assistant" | "tool")
                ) && m
                    .get("content")
                    .is_none_or(|v| v.is_string() || v.is_null())),
                "member{index}: packed Auto tools still require official text-only message content"
            );
            ensure!(s.required_context_tokens as usize <= MAX_LEARNED_PREFIX
                && member.prompt_ids().len().checked_add(s.output_allowance as usize)
                    .is_some_and(|n| n <= MAX_LEARNED_PREFIX),
                "member{index}: full prompt plus unchanged allowance exceeds explicit512-token context envelope");
            semantics.push(serde_json::to_value(s)?);
        }
        let original_body_sha256 = hash(&serde_json::to_vec(group.original_request())?);
        let members: Vec<_> = group.members().iter().zip(&semantics).enumerate()
            .map(|(index,(chat,s))| json!({"original_response_index":index,
                "member_original_request":chat.original_request(),"native_prompt_ids":chat.prompt_ids(),"effective_semantics":s})).collect();
        let mut encoded = b"dsv41-CPU-atomic-prepared-group-v1\0".to_vec();
        encoded.extend(serde_json::to_vec(&(
            ATOMIC_ROUTE,
            "1",
            &original_body_sha256,
            &members,
        ))?);
        Ok(Self {
            group,
            original_body_sha256,
            member_semantics: semantics,
            digest: hash(&encoded),
        })
    }
    pub fn original_request(&self) -> &Value {
        self.group.original_request()
    }
    pub fn original_body_sha256(&self) -> &str {
        // As in CpuBridge's member identity, this hashes the preserved JSON
        // value's compact serialization, not an unavailable HTTP wire buffer.
        &self.original_body_sha256
    }
    pub fn members(&self) -> &[PlannedChat] {
        self.group.members()
    }
    pub fn member_semantics(&self) -> &[Value] {
        &self.member_semantics
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    /// Consume CPU data only. This does not authorize model execution or parse
    /// a transcript; an actual sealed batch completion is required separately.
    pub fn into_planned_group(self) -> super::native_api_contract::PlannedAtomicGroup {
        self.group
    }
    pub fn execution_permit(&self) -> Result<()> {
        bail!("CPU atomic plan is not a runtime/GPU/terminal execution permit")
    }
}

/// Independently expected identities supplied by a future controller. This is
/// caller metadata, not proof that weights, ELF or frequency tables were loaded.
#[derive(Clone, Debug, Serialize)]
pub struct ProposedBatchDeployment {
    pub boot: Uuid,
    pub global_checkpoint_geometry_checksum_sha256: String,
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
    pub frequency: FrequencyProfileIdentity,
}
impl ProposedBatchDeployment {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.boot.is_nil()
                && [
                    &self.global_checkpoint_geometry_checksum_sha256,
                    &self.execution_snapshot_sha256,
                    &self.engine_binary_sha256
                ]
                .iter()
                .all(|v| hex(v)),
            "explicit boot/checkpoint/snapshot/binary identities required"
        );
        ensure!(
            self.frequency == FrequencyBackend::parse(&self.frequency.backend)?.identity(),
            "exact named frequency profile required; CPU/CUDA profiles are not interchangeable"
        );
        Ok(())
    }
}

/// Absolute, trusted in-process admission clock, captured before preparation.
/// Its nonce/wall duration are canonical metadata; Instant values are never
/// serialized or misrepresented as comparable across physical hosts.
#[derive(Clone, Debug)]
pub struct AdmissionClock {
    id: Uuid,
    admitted: Instant,
    deadline: Instant,
    wall_millis: u64,
}
impl AdmissionClock {
    pub fn new(admitted: Instant, deadline: Instant) -> Result<Self> {
        let duration = deadline
            .checked_duration_since(admitted)
            .context("deadline precedes admission")?;
        ensure!(
            !duration.is_zero()
                && duration <= Duration::from_millis(MAX_WALL_MILLIS)
                && duration.subsec_nanos() % 1_000_000 == 0,
            "explicit integral-millisecond admission budget must be1..=180000ms"
        );
        Ok(Self {
            id: Uuid::new_v4(),
            admitted,
            deadline,
            wall_millis: duration.as_millis() as u64,
        })
    }
    fn check(&self, now: Instant) -> Result<()> {
        ensure!(
            now >= self.admitted && now < self.deadline,
            "admission is future/expired; never restart or shorten the original request clock"
        );
        Ok(())
    }
    pub fn admitted(&self) -> Instant {
        self.admitted
    }
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
    pub fn wall_millis(&self) -> u64 {
        self.wall_millis
    }
}

#[derive(Clone, Debug, Serialize)]
struct Slot {
    generation: u64,
    owner: Option<Uuid>,
}
pub struct CpuSlotBook {
    boot: Uuid,
    identity: Uuid,
    slots: Vec<Slot>,
    seen_request_ids: BTreeSet<String>,
    revision: u64,
}
impl CpuSlotBook {
    pub fn new(boot: Uuid, capacity: usize) -> Result<Self> {
        ensure!(
            !boot.is_nil() && (1..=MAX_MEMBERS).contains(&capacity),
            "CPU slot book needs explicit boot/bounded capacity"
        );
        Ok(Self {
            boot,
            identity: Uuid::new_v4(),
            slots: vec![
                Slot {
                    generation: 0,
                    owner: None
                };
                capacity
            ],
            seen_request_ids: BTreeSet::new(),
            revision: 0,
        })
    }
    pub fn snapshot(&self) -> Value {
        json!({"boot":self.boot,"identity":self.identity,"revision":self.revision,"slots":self.slots,"seen_request_ids":self.seen_request_ids,"CPU_only":true})
    }
    fn validate_free(&self, request_id: &str, keys: &[LaneKey]) -> Result<()> {
        ensure!(
            !request_id.is_empty()
                && request_id.len() <= 128
                && !self.seen_request_ids.contains(request_id),
            "fresh explicit group request ID required"
        );
        let mut seen = BTreeSet::new();
        for key in keys {
            let slot = self
                .slots
                .get(key.slot)
                .context("physical slot outside CPU book")?;
            ensure!(
                key.boot == self.boot
                    && key.generation > slot.generation
                    && seen.insert(key.slot)
                    && slot.owner.is_none(),
                "all members need distinct free physical slots and strictly newer generations"
            );
        }
        Ok(())
    }
    fn release_claims(&mut self, owner: Uuid) {
        let mut changed = false;
        for slot in &mut self.slots {
            if slot.owner == Some(owner) {
                slot.owner = None;
                changed = true;
            }
        }
        if changed {
            self.revision = self.revision.saturating_add(1);
        }
    }
    pub fn release_cpu_ownership(&mut self, group: &CpuOwnedAtomicGroup) -> Result<()> {
        group.validate_book(self)?;
        // Caller must separately prove runtime teardown/reset before reusing
        // actual GPU state. This method releases ONLY this CPU book's claims.
        self.release_claims(group.record.admission_id);
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct AtomicMemberBinding {
    pub original_response_index: usize,
    pub member_request_id: String,
    pub native_prompt_ids: Vec<u32>,
    pub effective_semantics: Value,
    /// Shared identity fields only. The existing C1 plan constructor correctly
    /// refuses nonzero slots; a future sealed batch session needs its own
    /// actual reset/packed-step/all-rank terminal validation.
    pub request: RequestBinding,
}
#[derive(Debug, Serialize)]
pub struct CpuAdmissionRecord {
    pub admission_id: Uuid,
    pub group_request_id: String,
    pub original_group_body_sha256: String,
    pub prepared_group_sha256: String,
    pub admission_clock_id: Uuid,
    pub wall_millis: u64,
    pub slot_book_revision: u64,
    pub members: Vec<AtomicMemberBinding>,
    pub digest: String,
    pub CPU_ownership_only: bool,
    pub runtime_or_terminal_authority: bool,
}
enum State {
    Prepared,
    Published,
    Revoked,
    Taken,
}
pub struct CpuAtomicAdmission<'a> {
    book: &'a mut CpuSlotBook,
    prepared: Option<PreparedAtomicRequest>,
    deployment: ProposedBatchDeployment,
    keys: Vec<LaneKey>,
    clock: AdmissionClock,
    request_id: String,
    owner: Uuid,
    state: State,
    record: Option<CpuAdmissionRecord>,
}
impl<'a> CpuAtomicAdmission<'a> {
    pub fn new(
        book: &'a mut CpuSlotBook,
        prepared: PreparedAtomicRequest,
        deployment: ProposedBatchDeployment,
        keys: Vec<LaneKey>,
        clock: AdmissionClock,
        request_id: String,
        now: Instant,
    ) -> Result<Self> {
        deployment.validate()?;
        clock.check(now)?;
        ensure!(
            deployment.boot == book.boot && keys.len() == prepared.members().len(),
            "boot/member/physical-slot mapping differs"
        );
        book.validate_free(&request_id, &keys)?;
        Ok(Self {
            book,
            prepared: Some(prepared),
            deployment,
            keys,
            clock,
            request_id,
            owner: Uuid::new_v4(),
            state: State::Prepared,
            record: None,
        })
    }
    fn revoke(&mut self) {
        self.book.release_claims(self.owner);
        self.record = None;
        self.state = State::Revoked;
    }
    pub fn commit(&mut self, now: Instant) -> Result<()> {
        if !matches!(self.state, State::Prepared) {
            self.revoke();
            bail!("CPU admission already attempted; previous publication revoked");
        }
        let result = self.commit_inner(now);
        if result.is_err() {
            self.revoke();
        }
        result
    }
    fn commit_inner(&mut self, now: Instant) -> Result<()> {
        self.clock.check(now)?;
        self.deployment.validate()?;
        self.book.validate_free(&self.request_id, &self.keys)?;
        let group = self
            .prepared
            .as_ref()
            .context("prepared group ownership missing")?;
        let revision = self
            .book
            .revision
            .checked_add(1)
            .context("CPU slot revision overflow")?;
        let mut members = Vec::with_capacity(self.keys.len());
        for (index, (chat, key)) in group.members().iter().zip(&self.keys).enumerate() {
            let s = chat.semantics();
            let request = RequestBinding {
                model_revision: MODEL_REVISION.into(),
                global_checkpoint_geometry_checksum_sha256: self
                    .deployment
                    .global_checkpoint_geometry_checksum_sha256
                    .clone(),
                execution_snapshot_sha256: self.deployment.execution_snapshot_sha256.clone(),
                engine_binary_sha256: self.deployment.engine_binary_sha256.clone(),
                original_request_sha256: s.original_request_sha256.clone(),
                tokenizer_sha256: s.tokenizer_sha256.into(),
                toolkit_revision: s.toolkit_revision.into(),
                frequency: self.deployment.frequency.clone(),
                sampling: GreedySemantics {
                    temperature: s.effective_temperature as f64,
                    top_p: s.effective_top_p as f64,
                    presence_penalty: s.presence_penalty as f64,
                    frequency_penalty: s.frequency_penalty as f64,
                    seed: s.seed,
                    thinking: s.thinking,
                    effective_reasoning_score: s.effective_reasoning_score,
                    speculation: false,
                },
                stop: StopPolicy {
                    eos_native_id: 1,
                    user_sequences: s.stop_sequences.clone(),
                },
                budget: RequestBudget {
                    context_tokens: MAX_LEARNED_PREFIX,
                    output_allowance: s.output_allowance as usize,
                    wall_millis: self.clock.wall_millis,
                },
                lane: *key,
            };
            members.push(AtomicMemberBinding {
                original_response_index: index,
                member_request_id: format!("{}-member-{index}", self.request_id),
                native_prompt_ids: chat.prompt_ids().to_vec(),
                effective_semantics: group.member_semantics[index].clone(),
                request,
            });
        }
        let mut encoded = b"dsv41-CPU-atomic-admission-v1\0".to_vec();
        encoded.extend(serde_json::to_vec(&(
            self.owner,
            &self.request_id,
            group.original_body_sha256(),
            group.digest(),
            self.clock.id,
            self.clock.wall_millis,
            revision,
            &members,
        ))?);
        let record = CpuAdmissionRecord {
            admission_id: self.owner,
            group_request_id: self.request_id.clone(),
            original_group_body_sha256: group.original_body_sha256.clone(),
            prepared_group_sha256: group.digest.clone(),
            admission_clock_id: self.clock.id,
            wall_millis: self.clock.wall_millis,
            slot_book_revision: revision,
            members,
            digest: hash(&encoded),
            CPU_ownership_only: true,
            runtime_or_terminal_authority: false,
        };
        let mut seen_request_ids = self.book.seen_request_ids.clone();
        seen_request_ids.insert(self.request_id.clone());
        // All checks, serialization and allocations completed before ANY slot
        // mutation. Exclusive Rust ownership prevents concurrent partial claims.
        for key in &self.keys {
            self.book.slots[key.slot] = Slot {
                generation: key.generation,
                owner: Some(self.owner),
            };
        }
        self.book.seen_request_ids = seen_request_ids;
        self.book.revision = revision;
        self.record = Some(record);
        self.state = State::Published;
        Ok(())
    }
    pub fn publication(&mut self, now: Instant) -> Result<&CpuAdmissionRecord> {
        if !matches!(self.state, State::Published) {
            bail!("whole CPU group has not been published");
        }
        if let Err(e) = self.clock.check(now) {
            self.revoke();
            return Err(e);
        }
        self.record.as_ref().context("published CPU record missing")
    }
    pub fn take_published(mut self, now: Instant) -> Result<CpuOwnedAtomicGroup> {
        self.publication(now)?;
        let group = CpuOwnedAtomicGroup {
            book_identity: self.book.identity,
            prepared: self.prepared.take().context("prepared group missing")?,
            record: self.record.take().context("published record missing")?,
            clock: self.clock.clone(),
        };
        self.state = State::Taken;
        Ok(group)
    }
}
impl Drop for CpuAtomicAdmission<'_> {
    fn drop(&mut self) {
        if matches!(self.state, State::Published) {
            self.revoke();
        }
    }
}
/// Non-cloneable CPU ownership only. Dropping it does not free physical GPU
/// state or the slot book; explicit release follows separate runtime teardown.
pub struct CpuOwnedAtomicGroup {
    book_identity: Uuid,
    prepared: PreparedAtomicRequest,
    record: CpuAdmissionRecord,
    clock: AdmissionClock,
}
impl CpuOwnedAtomicGroup {
    fn validate_book(&self, book: &CpuSlotBook) -> Result<()> {
        ensure!(
            self.book_identity == book.identity
                && self.record.members.iter().all(|m| book
                    .slots
                    .get(m.request.lane.slot)
                    .is_some_and(|s| s.owner == Some(self.record.admission_id)
                        && s.generation == m.request.lane.generation)),
            "CPU ownership was released/revoked or belongs to a different book"
        );
        Ok(())
    }
    pub fn publication<'a>(
        &'a self,
        book: &CpuSlotBook,
        now: Instant,
    ) -> Result<&'a CpuAdmissionRecord> {
        self.clock.check(now)?;
        self.validate_book(book)?;
        Ok(&self.record)
    }
    pub fn prepared(&self) -> &PreparedAtomicRequest {
        &self.prepared
    }
    pub fn clock(&self) -> &AdmissionClock {
        &self.clock
    }
    pub fn into_prepared(self) -> PreparedAtomicRequest {
        self.prepared
    }
    pub fn execution_permit(&self) -> Result<()> {
        bail!("CPU-owned atomic group cannot authorize model execution, actual batching or a normal terminal response")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    fn bridge() -> &'static CpuBridge {
        static B: OnceLock<CpuBridge> = OnceLock::new();
        B.get_or_init(|| CpuBridge::from_verified_tokenizer(&std::fs::read(
            "test-data/tokenizer.json").unwrap()).unwrap())
    }
    fn fixtures() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_actual_atomic_requests.json"
        ))
        .unwrap()
    }
    fn body(width: usize) -> Value {
        let mut b = fixtures()["cases"][2]["request"].clone();
        b["messages"].as_array_mut().unwrap().truncate(width);
        b
    }
    fn prepared(width: usize) -> PreparedAtomicRequest {
        PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &body(width)).unwrap()
    }
    fn boot() -> Uuid {
        Uuid::from_u128(214)
    }
    fn deployment() -> ProposedBatchDeployment {
        ProposedBatchDeployment {
            boot: boot(),
            global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
            execution_snapshot_sha256: "b".repeat(64),
            engine_binary_sha256: "c".repeat(64),
            frequency: FrequencyBackend::CudaTorch.identity(),
        }
    }
    fn keys(width: usize, generation: u64) -> Vec<LaneKey> {
        (0..width)
            .map(|i| LaneKey {
                boot: boot(),
                slot: (5 * i + 7) % 16,
                generation,
            })
            .collect()
    }
    fn clock(now: Instant) -> AdmissionClock {
        AdmissionClock::new(now, now + Duration::from_secs(100)).unwrap()
    }
    #[test]
    fn actual_retained_C8_C16_code_prose_bodies_keep_all_native_ids_defaults_and_allowances() {
        for case in fixtures()["cases"].as_array().unwrap() {
            let request = case["request"].clone();
            let group = PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &request)
                .unwrap();
            assert_eq!(group.original_request(), &request);
            assert_eq!(
                group.members().len(),
                case["concurrency"].as_u64().unwrap() as usize
            );
            let ids: Vec<u32> = serde_json::from_value(case["native_prompt_ids"].clone()).unwrap();
            for member in group.members() {
                assert_eq!(member.prompt_ids(), ids);
                assert_eq!(
                    member.semantics().prompt_ids_sha256_u32le,
                    case["native_prompt_sha256_u32le"]
                );
                let s = member.semantics();
                assert_eq!(s.output_allowance, 256);
                assert_eq!(s.seed, Some(42));
                assert!(s.thinking);
                assert_eq!(s.effective_reasoning_score, Some(75));
                assert_eq!(s.effective_temperature, 0.);
                assert_eq!(s.effective_top_p, 1.);
                assert_eq!(s.requested_top_p, None);
                assert!(s.return_token_ids && !s.ignore_eos && s.stop_sequences.is_empty());
            }
            assert!(group.execution_permit().is_err());
        }
    }
    #[test]
    fn B1_through_B16_prepare_without_slot_effect_and_bind_response_order_to_physical_lanes() {
        for width in 1..=MAX_MEMBERS {
            let now = Instant::now();
            let group = prepared(width);
            let again = prepared(width);
            assert_eq!(group.digest(), again.digest());
            let mut book = CpuSlotBook::new(boot(), 16).unwrap();
            let before = book.snapshot();
            let lane_keys = keys(width, 1);
            let mut transaction = CpuAtomicAdmission::new(
                &mut book,
                group,
                deployment(),
                lane_keys.clone(),
                clock(now),
                format!("group-{width}"),
                now,
            )
            .unwrap();
            assert!(transaction.publication(now).is_err());
            transaction.commit(now).unwrap();
            let record = transaction.publication(now).unwrap();
            assert!(record.CPU_ownership_only && !record.runtime_or_terminal_authority);
            for (index, member) in record.members.iter().enumerate() {
                assert_eq!(member.original_response_index, index);
                assert_eq!(member.request.lane, lane_keys[index]);
                assert_eq!(member.request.budget.output_allowance, 256);
                assert_eq!(member.request.budget.wall_millis, 100000);
                assert_eq!(
                    member.native_prompt_ids,
                    again.members()[index].prompt_ids()
                );
                assert_eq!(member.effective_semantics, again.member_semantics()[index]);
            }
            let owned = transaction.take_published(now).unwrap();
            assert_ne!(book.snapshot(), before);
            assert!(owned.publication(&book, now).is_ok());
            assert!(owned.execution_permit().is_err());
            book.release_cpu_ownership(&owned).unwrap();
            assert!(owned.publication(&book, now).is_err());
        }
    }
    #[test]
    fn any_invalid_member_is_rejected_before_any_slot_or_permit_effect() {
        let book = CpuSlotBook::new(boot(), 16).unwrap();
        let before = book.snapshot();
        for failure in 0..3 {
            let mut request = body(16);
            request["messages"][15] = match failure {
                0 => json!([{"role":"tool","tool_call_id":"orphan","content":"wrong"}]),
                1 => json!([{"role":"user","content":"far beyond explicit context ".repeat(600)}]),
                _ => {
                    json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://never-fetch.invalid/image"}}]}])
                }
            };
            assert!(
                PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &request)
                    .is_err()
            );
            assert_eq!(book.snapshot(), before);
        }
    }
    #[test]
    fn unsupported_constraints_routes_and_over_capacity_are_refused_without_clipping() {
        for (path, header) in [
            ("/v1/chat/completions", Some("1")),
            (ATOMIC_ROUTE, None),
            (ATOMIC_ROUTE, Some("true")),
        ] {
            assert!(PreparedAtomicRequest::prepare(bridge(), path, header, &body(1)).is_err());
        }
        for (field, value) in [
            ("temperature", json!(0.5)),
            ("stop", json!("DONE")),
            ("cache_salt", json!("unqualified")),
            ("ignore_eos", json!(true)),
            ("stream", json!(true)),
            ("max_tokens", json!(4096)),
        ] {
            let mut request = body(1);
            request[field] = value;
            assert!(
                PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &request)
                    .is_err(),
                "{field}"
            );
        }
        let mut too_many = body(16);
        let first = too_many["messages"][0].clone();
        too_many["messages"].as_array_mut().unwrap().push(first);
        assert!(
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &too_many).is_err()
        );
        let mut full_default_cannot_fit = body(1);
        full_default_cannot_fit
            .as_object_mut()
            .unwrap()
            .remove("max_tokens");
        assert!(PreparedAtomicRequest::prepare(
            bridge(),
            ATOMIC_ROUTE,
            Some("1"),
            &full_default_cannot_fit
        )
        .is_err());
    }
    #[test]
    fn busy_last_slot_and_invalid_identity_have_no_partial_claims() {
        let now = Instant::now();
        let mut book = CpuSlotBook::new(boot(), 16).unwrap();
        let mut one = keys(1, 1);
        one[0].slot = 0;
        let mut first = CpuAtomicAdmission::new(
            &mut book,
            prepared(1),
            deployment(),
            one,
            clock(now),
            "first".into(),
            now,
        )
        .unwrap();
        first.commit(now).unwrap();
        let owned = first.take_published(now).unwrap();
        let before = book.snapshot();
        let mut last_busy: Vec<_> = (8..15)
            .map(|slot| LaneKey {
                boot: boot(),
                slot,
                generation: 1,
            })
            .collect();
        last_busy.push(LaneKey {
            boot: boot(),
            slot: 0,
            generation: 2,
        });
        assert!(CpuAtomicAdmission::new(
            &mut book,
            prepared(8),
            deployment(),
            last_busy,
            clock(now),
            "busy".into(),
            now
        )
        .is_err());
        assert_eq!(book.snapshot(), before);
        let mut wrong = deployment();
        wrong.frequency.plain_table_sha256 = "0".repeat(64);
        assert!(CpuAtomicAdmission::new(
            &mut book,
            prepared(1),
            wrong,
            keys(1, 1),
            clock(now),
            "wrong".into(),
            now
        )
        .is_err());
        assert_eq!(book.snapshot(), before);
        let mut other = CpuSlotBook::new(boot(), 16).unwrap();
        let other_before = other.snapshot();
        assert!(other.release_cpu_ownership(&owned).is_err());
        assert_eq!(other.snapshot(), other_before);
    }
    #[test]
    fn double_commit_revokes_entire_publication_and_generations_never_roll_back() {
        let now = Instant::now();
        let mut book = CpuSlotBook::new(boot(), 16).unwrap();
        let mut transaction = CpuAtomicAdmission::new(
            &mut book,
            prepared(8),
            deployment(),
            keys(8, 1),
            clock(now),
            "one-shot".into(),
            now,
        )
        .unwrap();
        transaction.commit(now).unwrap();
        assert!(transaction.publication(now).is_ok());
        assert!(transaction.commit(now).is_err());
        assert!(transaction.publication(now).is_err());
        drop(transaction);
        assert!(book.slots.iter().all(|s| s.owner.is_none()));
        assert_eq!(book.slots.iter().filter(|s| s.generation == 1).count(), 8);
        assert!(CpuAtomicAdmission::new(
            &mut book,
            prepared(8),
            deployment(),
            keys(8, 2),
            clock(now),
            "one-shot".into(),
            now
        )
        .is_err());
        assert!(CpuAtomicAdmission::new(
            &mut book,
            prepared(8),
            deployment(),
            keys(8, 1),
            clock(now),
            "fresh-stale-generation".into(),
            now
        )
        .is_err());
        let mut fresh = CpuAtomicAdmission::new(
            &mut book,
            prepared(8),
            deployment(),
            keys(8, 2),
            clock(now),
            "fresh".into(),
            now,
        )
        .unwrap();
        fresh.commit(now).unwrap();
    }
    #[test]
    fn original_absolute_deadline_is_never_restarted_and_expiry_revokes_all_members() {
        let now = Instant::now();
        let mut book = CpuSlotBook::new(boot(), 16).unwrap();
        let before = book.snapshot();
        let mut expired = CpuAtomicAdmission::new(
            &mut book,
            prepared(2),
            deployment(),
            keys(2, 1),
            clock(now),
            "expires-before-commit".into(),
            now,
        )
        .unwrap();
        assert!(expired.commit(now + Duration::from_secs(100)).is_err());
        drop(expired);
        assert_eq!(book.snapshot(), before);
        let mut published = CpuAtomicAdmission::new(
            &mut book,
            prepared(2),
            deployment(),
            keys(2, 1),
            clock(now),
            "expires-after-commit".into(),
            now,
        )
        .unwrap();
        published.commit(now).unwrap();
        assert!(published
            .publication(now + Duration::from_secs(100))
            .is_err());
        drop(published);
        assert!(book.slots.iter().all(|s| s.owner.is_none()));
        assert!(AdmissionClock::new(now, now).is_err());
        assert!(AdmissionClock::new(now, now + Duration::from_secs(181)).is_err());
        assert!(AdmissionClock::new(now, now + Duration::from_nanos(1)).is_err());
    }
    #[test]
    fn CPU_capabilities_never_mint_runtime_or_terminal_authority() {
        let c = capabilities();
        for field in [
            "model_execution",
            "GPU_batching_verified",
            "HTTP_serving",
            "terminal_authority",
            "runtime_reset_authority",
        ] {
            assert_eq!(c[field], false);
        }
    }
}
