//! PRIVATE resident C1 request execution, without a listener or automatic launch.
//! All ranks must independently prepare the same exact request before entering
//! this method. Native transport rearm checks their frozen-plan agreement.
//! A failed/cancelled request poisons the owner; it cannot be silently reused.
use super::{
    gpu_collectives::GpuCollectives,
    gpu_model_chunk::{CheckedModelChunk, GpuModelChunk},
    gpu_request_session::{CheckedRequestCompletion, GpuRequestSession, SessionProgress},
    model_session_plan::{
        CpuRequestPlan, GreedySemantics, RequestBinding, RequestBudget, StepRole, StopPolicy,
    },
    native_api_contract::{CpuBridge, PlannedChat},
    native_response_bridge::{self, BufferedNativeResponse, ResponseAuthorityBinding},
    request_session_release::RequestReleaseConfig,
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

pub const REQUEST_WALL_MILLIS: u64 = 180_000;

/// Metadata-only construction; this value does not authorize model execution.
fn chat_binding(
    chat: &PlannedChat,
    deployment: &RequestBinding,
    lane: LaneKey,
) -> Result<RequestBinding> {
    let s = chat.semantics();
    ensure!(!s.stream && s.stop_sequences.is_empty() && !s.ignore_eos && s.min_tokens==0,
        "resident buffered path requires native EOS/cap; streaming/userstops/forcedlength are unsupported");
    ensure!(
        s.effective_temperature == 0.0
            && s.effective_top_p == 1.0
            && s.presence_penalty == 0.0
            && s.frequency_penalty == 0.0,
        "resident request requires exact qualified greedy semantics"
    );
    ensure!(
        s.model_revision == deployment.model_revision
            && s.tokenizer_sha256 == deployment.tokenizer_sha256
            && s.toolkit_revision == deployment.toolkit_revision,
        "request/checkpoint/tokenizer/toolkit identity differs"
    );
    let mut b = deployment.clone();
    b.original_request_sha256 = s.original_request_sha256.clone();
    b.lane = lane;
    b.sampling = GreedySemantics {
        temperature: 0.0,
        top_p: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: s.seed,
        thinking: s.thinking,
        effective_reasoning_score: s.effective_reasoning_score,
        speculation: false,
    };
    b.stop = StopPolicy {
        eos_native_id: 1,
        user_sequences: vec![],
    };
    b.budget = RequestBudget {
        context_tokens: s.planning_limits.context_tokens as usize,
        output_allowance: s.output_allowance as usize,
        wall_millis: REQUEST_WALL_MILLIS,
    };
    Ok(b)
}

/// Exact stream:true request proposal. It cannot issue terminal authority.
fn streaming_binding(
    chat: &PlannedChat,
    deployment: &RequestBinding,
    lane: LaneKey,
) -> Result<RequestBinding> {
    let s = chat.semantics();
    ensure!(
        s.stream
            && chat
                .original_request()
                .get("stream")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        "explicit unchanged stream:true required"
    );
    ensure!(
        s.stop_sequences.is_empty() && !s.ignore_eos && s.min_tokens == 0,
        "stream userstop/forcedlength unsupported"
    );
    ensure!(
        s.effective_temperature == 0.0
            && s.effective_top_p == 1.0
            && s.presence_penalty == 0.0
            && s.frequency_penalty == 0.0,
        "greedy stream semantics required"
    );
    ensure!(
        s.model_revision == deployment.model_revision
            && s.tokenizer_sha256 == deployment.tokenizer_sha256
            && s.toolkit_revision == deployment.toolkit_revision,
        "stream request identity differs"
    );
    let mut b = deployment.clone();
    b.original_request_sha256 = s.original_request_sha256.clone();
    b.lane = lane;
    b.sampling = GreedySemantics {
        temperature: 0.0,
        top_p: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: s.seed,
        thinking: s.thinking,
        effective_reasoning_score: s.effective_reasoning_score,
        speculation: false,
    };
    b.stop = StopPolicy {
        eos_native_id: 1,
        user_sequences: vec![],
    };
    b.budget = RequestBudget {
        context_tokens: s.planning_limits.context_tokens as usize,
        output_allowance: s.output_allowance as usize,
        wall_millis: REQUEST_WALL_MILLIS,
    };
    Ok(b)
}
pub fn next_stream_plan(
    chat: &PlannedChat,
    previous: &CheckedRequestCompletion,
) -> Result<Arc<CpuRequestPlan>> {
    ensure!(
        previous.all_rank_prefix_agreement() && previous.transport_lifetime_release_confirmed(),
        "prior all8 completion missing"
    );
    let old = previous.lane();
    let lane = LaneKey {
        boot: old.boot,
        slot: 0,
        generation: old
            .generation
            .checked_add(1)
            .context("stream generation exhausted")?,
    };
    Ok(Arc::new(CpuRequestPlan::new(
        streaming_binding(chat, previous.binding(), lane)?,
        chat.prompt_ids().to_vec(),
        previous.next_operation_id(),
        previous.next_step_id(),
        previous.actual_epoch(),
    )?))
}
/// A locally accepted actual generated ID. Private fields and construction keep
/// raw JSON/unaccepted/intermediate-prefill outputs out of the public SSE path.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AcceptedNativeToken {
    request_sha256: String,
    plan_sha256: String,
    ordinal: usize,
    native_id: u32,
}
impl AcceptedNativeToken {
    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub fn native_id(&self) -> u32 {
        self.native_id
    }
}
fn after_accept(
    plan: &CpuRequestPlan,
    checked: &CheckedModelChunk,
) -> Result<Option<AcceptedNativeToken>> {
    let Some(ordinal) = checked.output_ordinal() else {
        return Ok(None);
    };
    ensure!(
        checked.plan_sha256() == plan.digest()
            && checked.generated_native_ids().len() == ordinal
            && checked.generated_native_ids().last() == Some(&checked.selected_native_id()),
        "accepted native output ordinal/identity differs"
    );
    Ok(Some(AcceptedNativeToken {
        request_sha256: plan.binding().original_request_sha256.clone(),
        plan_sha256: plan.digest().into(),
        ordinal,
        native_id: checked.selected_native_id(),
    }))
}
#[cfg(test)]
pub(crate) fn accepted_token_CPU_fixture(
    request_sha256: &str,
    plan_sha256: &str,
    ordinal: usize,
    native_id: u32,
) -> AcceptedNativeToken {
    AcceptedNativeToken {
        request_sha256: request_sha256.into(),
        plan_sha256: plan_sha256.into(),
        ordinal,
        native_id,
    }
}

/// Uses only the previous actual completion's reserved IDs and actual epoch.
/// It preserves the full requested output allowance; no fixed diagnostic cap.
pub fn next_chat_plan(
    chat: &PlannedChat,
    previous: &CheckedRequestCompletion,
) -> Result<Arc<CpuRequestPlan>> {
    ensure!(
        previous.all_rank_prefix_agreement() && previous.transport_lifetime_release_confirmed(),
        "prior actual request has not completed all-rank transport release"
    );
    let old = previous.lane();
    let lane = LaneKey {
        boot: old.boot,
        slot: 0,
        generation: old
            .generation
            .checked_add(1)
            .context("request generation exhausted")?,
    };
    let b = chat_binding(chat, previous.binding(), lane)?;
    Ok(Arc::new(CpuRequestPlan::new(
        b,
        chat.prompt_ids().to_vec(),
        previous.next_operation_id(),
        previous.next_step_id(),
        previous.actual_epoch(),
    )?))
}

/// Executes on the ONE existing native owner thread. The observer receives
/// actual opaque full-step receipts, including ignored intermediate heads.
/// Cancellation records a local checked-boundary abort; the supervisor must
/// still drain all eight ranks before declaring recovery or accepting work.
#[allow(clippy::too_many_arguments)]
pub fn execute_next_request(
    model: &mut GpuModelChunk,
    comm: &mut GpuCollectives,
    previous: &CheckedRequestCompletion,
    plan: Arc<CpuRequestPlan>,
    release: RequestReleaseConfig,
    admitted: Instant,
    cancelled: &AtomicBool,
    observe: impl FnMut(&CheckedModelChunk) -> Result<()>,
) -> Result<CheckedRequestCompletion> {
    execute_next_request_controlled(
        model,
        comm,
        previous,
        plan,
        release,
        admitted,
        cancelled,
        observe,
        None,
        |_| Ok(()),
    )
}
/// Additive real constrained path. post_accept runs only after actual request
/// session acceptance; it is suitable for the separate opaque SSE adapter.
#[allow(clippy::too_many_arguments)]
pub fn execute_next_request_controlled(
    model: &mut GpuModelChunk,
    comm: &mut GpuCollectives,
    previous: &CheckedRequestCompletion,
    plan: Arc<CpuRequestPlan>,
    release: RequestReleaseConfig,
    admitted: Instant,
    cancelled: &AtomicBool,
    mut observe: impl FnMut(&CheckedModelChunk) -> Result<()>,
    mut forced: Option<super::tool_constraint::ForcedNamedCursor>,
    mut post_accept: impl FnMut(&CheckedModelChunk) -> Result<()>,
) -> Result<CheckedRequestCompletion> {
    let result = (|| {
        ensure!(
            plan.binding().budget.wall_millis == REQUEST_WALL_MILLIS,
            "native request must preserve the predeclared180s wall policy"
        );
        ensure!(
            !cancelled.load(Ordering::Acquire),
            "request cancelled before native reset"
        );
        ensure!(
            admitted.elapsed().as_millis() < REQUEST_WALL_MILLIS as u128,
            "original admission budget expired before native reset"
        );
        let reset = model.reset_lane(plan.binding().lane, comm)?;
        let mut session = GpuRequestSession::rearm_from_admission(
            plan.clone(),
            &reset,
            previous,
            comm,
            release,
            admitted,
        )?;
        loop {
            if cancelled.load(Ordering::Acquire) {
                session.cancel_at_checked_boundary(comm)?;
                anyhow::bail!("request cancelled; coordinated all-eight recovery required");
            }
            let index = session.check_before_step(comm)?;
            let step = plan
                .steps()
                .get(index)
                .context("native step outside immutable request")?;
            let native = match step.role {
                StepRole::IntermediatePrefillHeadIgnored | StepRole::FinalPrefillFirstOutput => {
                    plan.input_ids()
                        .get(step.start..step.end)
                        .context("native prompt segment differs")?
                        .to_vec()
                }
                StepRole::DecodeOutput => vec![*session
                    .generated_native_ids()
                    .last()
                    .context("native decode missing actual prior token")?],
            };
            let mask = if step.output_ordinal.is_some() {
                forced.as_ref().map(|c| c.allowed()).transpose()?
            } else {
                None
            };
            // CPU grammar work counts against the ORIGINAL admission budget.
            // Never launch a model step after a long mask-build crossed it.
            ensure!(
                !cancelled.load(Ordering::Acquire),
                "request cancelled during token-constraint planning"
            );
            ensure!(
                session.check_before_step(comm)? == index,
                "request expired or changed during token-constraint planning"
            );
            let checked =
                model.execute_step_with_constraint(&plan, index, &native, comm, mask.as_ref())?;
            observe(&checked)?;
            let progress = session.accept_step(&checked, comm)?;
            if step.output_ordinal.is_some() {
                if let Some(c) = forced.as_mut() {
                    c.accept(checked.selected_native_id())?;
                }
                post_accept(&checked)?;
            }
            match progress {
                SessionProgress::Continue { next_step_index } => ensure!(
                    next_step_index == index + 1,
                    "native request skipped a complete step"
                ),
                SessionProgress::StopReady { .. } => break,
            }
        }
        if cancelled.load(Ordering::Acquire) {
            session.cancel_at_checked_boundary(comm)?;
            anyhow::bail!(
                "request cancelled during final step; coordinated all-eight recovery required"
            );
        }
        if let Some(c) = forced.as_ref() {
            ensure!(
                c.finished(),
                "required named output reached cap without complete calls block/native EOS"
            );
        }
        let done = session.finish(comm)?;
        ensure!(!cancelled.load(Ordering::Acquire) && admitted.elapsed().as_millis()<REQUEST_WALL_MILLIS as u128,
            "request expired or cancelled during all-rank completion; checked evidence cannot become a normal response");
        ensure!(
            done.input_native_ids() == plan.input_ids() && done.plan_sha256() == plan.digest(),
            "actual terminal is associated with a different request"
        );
        Ok(done)
    })();
    if result.is_err() {
        comm.abort_session();
    }
    result
}

/// Streaming observer publishes only after successful actual session.accept_step.
#[allow(clippy::too_many_arguments)]
pub fn execute_next_request_stream(
    model: &mut GpuModelChunk,
    comm: &mut GpuCollectives,
    previous: &CheckedRequestCompletion,
    plan: Arc<CpuRequestPlan>,
    release: RequestReleaseConfig,
    admitted: Instant,
    cancelled: &AtomicBool,
    observe: impl FnMut(&CheckedModelChunk) -> Result<()>,
    forced: Option<super::tool_constraint::ForcedNamedCursor>,
    mut accepted: impl FnMut(&AcceptedNativeToken) -> Result<()>,
) -> Result<CheckedRequestCompletion> {
    let observed_plan = plan.clone();
    execute_next_request_controlled(
        model,
        comm,
        previous,
        plan,
        release,
        admitted,
        cancelled,
        observe,
        forced,
        |checked| {
            if let Some(token) = after_accept(&observed_plan, checked)? {
                accepted(&token)?;
            }
            Ok(())
        },
    )
}

/// Buffered response only. This consumes the SAME PlannedChat that the actor
/// encoded and dispatched; it cannot accept a serialized completion assertion.
#[allow(clippy::too_many_arguments)]
pub async fn parse_actual_response(
    bridge: &CpuBridge,
    chat: PlannedChat,
    plan: &CpuRequestPlan,
    expected: &ResponseAuthorityBinding,
    completion: &CheckedRequestCompletion,
    response_id: String,
) -> BufferedNativeResponse {
    native_response_bridge::buffered_response(bridge, chat, plan, expected, completion, response_id)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::{
        native_api_contract::PlanningLimits, position::FrequencyBackend, protocol,
        scheduler::LaneKey,
    };
    use serde_json::json;
    use uuid::Uuid;
    fn setup() -> (CpuBridge, RequestBinding) {
        let bytes=std::fs::read("test-data/tokenizer.json").unwrap();
        let bridge = CpuBridge::from_verified_tokenizer(&bytes).unwrap();
        let b = RequestBinding {
            model_revision: protocol::MODEL_REVISION.into(),
            global_checkpoint_geometry_checksum_sha256: "1".repeat(64),
            execution_snapshot_sha256: "2".repeat(64),
            engine_binary_sha256: "3".repeat(64),
            original_request_sha256: "4".repeat(64),
            tokenizer_sha256: super::super::native_api_contract::TOKENIZER_SHA256.into(),
            toolkit_revision: protocol::TOOLKIT_REVISION.into(),
            frequency: FrequencyBackend::CudaTorch.identity(),
            sampling: GreedySemantics {
                temperature: 0.0,
                top_p: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                seed: Some(42),
                thinking: true,
                effective_reasoning_score: Some(75),
                speculation: false,
            },
            stop: StopPolicy {
                eos_native_id: 1,
                user_sequences: vec![],
            },
            budget: RequestBudget {
                context_tokens: 8192,
                output_allowance: 32,
                wall_millis: 300000,
            },
            lane: LaneKey {
                boot: Uuid::from_u128(214),
                slot: 0,
                generation: 1,
            },
        };
        (bridge, b)
    }
    fn limits() -> PlanningLimits {
        PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 1,
            request_bytes: 1024 * 1024,
        }
    }
    #[test]
    fn actual_encoded_chat_preserves_full4096_seed_and_thinking_without_diagnostic_budget() {
        let (bridge, b) = setup();
        let chat=bridge.prepare_single(json!({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Read the fixture, fix its bug, run tests and report the result."}],"temperature":0.0,"top_p":1.0,"seed":29,"max_tokens":4096,"chat_template_kwargs":{"thinking":false}}),limits()).unwrap();
        let binding = chat_binding(&chat, &b, b.lane).unwrap();
        let plan = CpuRequestPlan::new(binding, chat.prompt_ids().to_vec(), 1000, 99, 300).unwrap();
        assert_eq!(plan.binding().budget.output_allowance, 4096);
        assert_eq!(plan.binding().budget.wall_millis, 180000);
        assert_eq!(plan.binding().sampling.seed, Some(29));
        assert!(!plan.binding().sampling.thinking);
        assert_eq!(
            plan.binding().original_request_sha256,
            chat.semantics().original_request_sha256
        );
        assert_eq!(plan.input_ids(), chat.prompt_ids());
        assert!(plan.operations().len() > 32768);
    }
    #[test]
    fn user_stop_and_stream_are_refused_before_touching_any_gpu_owner() {
        let (bridge, b) = setup();
        for extra in [json!({"stream":true}), json!({"stop":["STOP"]})] {
            let mut body = json!({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"temperature":0.0,"top_p":1.0,"max_tokens":4096});
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            if let Ok(chat) = bridge.prepare_single(body, limits()) {
                assert!(chat_binding(&chat, &b, b.lane).is_err());
            }
        }
    }
    #[test]
    fn token_and_toolkit_binding_cannot_be_replaced_by_previous_deployment_metadata() {
        let (bridge, b) = setup();
        let chat=bridge.prepare_single(json!({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"temperature":0.0,"top_p":1.0,"max_tokens":4096}),limits()).unwrap();
        for field in 0..3 {
            let mut wrong = b.clone();
            match field {
                0 => wrong.model_revision = "0".repeat(40),
                1 => wrong.tokenizer_sha256 = "0".repeat(64),
                _ => wrong.toolkit_revision = "0".repeat(40),
            };
            assert!(chat_binding(&chat, &wrong, b.lane).is_err());
        }
    }
}
