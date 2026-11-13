//! PRIVATE buffered OpenAI response adapter. No listener, GPU submission,
//! execution permit, live SSE, tool execution or serving/quality qualification.
//!
//! The sole production entry accepts the transport owner's opaque completion.
//! Caller JSON, CPU stop proposals and parser EOF cannot mint terminal authority.
use super::{
    gpu_model_chunk::ModelChunkIdentity,
    gpu_request_session::{CheckedRequestCompletion, RequestStopReason},
    model_session_plan::{self, CpuRequestPlan, RequestBinding},
    native_api_contract::{
        CpuBridge, CpuParseReport, CpuTerminal, CpuTranscript, PlannedChat, EOS_ID,
    },
    position::FrequencyProfileIdentity,
    scheduler::LaneKey,
    tool_constraint::{ConstraintExecution, POLICY as FORCED_POLICY},
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::Digest;

/// Independently expected source/supervisor/session configuration, not a proof
/// constructor. Supplying these values never creates a CheckedRequestCompletion.
#[derive(Clone, Debug, Serialize)]
pub struct ResponseAuthorityBinding {
    pub global_checkpoint_geometry_checksum_sha256: String,
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
    pub head_manifest_sha256: String,
    pub frequency: FrequencyProfileIdentity,
    pub lane: LaneKey,
    pub completion_owner: [u8; 32],
    pub program_sha256: String,
}

/// Native evidence remains available even if binding or official parsing fails.
#[derive(Debug, Serialize)]
pub struct BufferedNativeResponse {
    passed: bool,
    response: Option<Value>,
    error: Option<String>,
    native_prompt_ids: Vec<u32>,
    native_generated_ids: Vec<u32>,
    proof_audit: Value,
    parser_report: Option<CpuParseReport>,
    output_transport: &'static str,
    TTFT_measured: bool,
    runtime_terminal_authority_checked: bool,
    model_or_task_quality_qualified: bool,
    serving_backend_qualified: bool,
}

impl BufferedNativeResponse {
    pub fn passed(&self) -> bool {
        self.passed
    }
    pub fn response(&self) -> Option<&Value> {
        self.response.as_ref()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn native_prompt_ids(&self) -> &[u32] {
        &self.native_prompt_ids
    }
    pub fn native_generated_ids(&self) -> &[u32] {
        &self.native_generated_ids
    }
    pub fn proof_audit(&self) -> &Value {
        &self.proof_audit
    }
    pub fn parser_report(&self) -> Option<&CpuParseReport> {
        self.parser_report.as_ref()
    }
    pub fn runtime_terminal_authority_checked(&self) -> bool {
        self.runtime_terminal_authority_checked
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeTerminal {
    Eos,
    Allowance,
}

/// Private borrowed view exists only to test checking logic without a CUDA
/// context. It is neither exported nor a substitute public completion type.
struct CompletionFacts<'a> {
    binding: &'a RequestBinding,
    model: &'a ModelChunkIdentity,
    rank: usize,
    lane: LaneKey,
    owner: [u8; 32],
    plan_sha256: &'a str,
    program_sha256: &'a str,
    input: &'a [u32],
    input_sha256: &'a str,
    generated: &'a [u32],
    generated_sha256: &'a str,
    consumed_sha256: &'a str,
    terminal: NativeTerminal,
    completed_steps: usize,
    completed_operations: usize,
    actual_epoch: u64,
    all_rank_agreement: bool,
    lifetime_released: bool,
    elapsed: f64,
    timing_scope: &'a str,
    next_operation_id: u64,
    next_step_id: u64,
    forced_constraint: Option<&'a ConstraintExecution>,
}
fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Association check only: production facts originate in the opaque actual
/// completion, whose session has checked every constrained generated head.
fn validate_forced_constraint(
    expected: Option<&Value>,
    actual: Option<&ConstraintExecution>,
    request_sha256: &str,
    tokenizer_sha256: &str,
    prompt: &[u32],
    generated: &[u32],
) -> Result<()> {
    match (expected, actual) {
        (None, None) => return Ok(()),
        (Some(_), None) => {
            anyhow::bail!("forced-named request lacks actual constrained sampler authority")
        }
        (None, Some(_)) => {
            anyhow::bail!("unrequested forced-named sampler changed native decoding")
        }
        (Some(expected), Some(actual)) => {
            ensure!(
                expected.get("policy").and_then(Value::as_str) == Some(FORCED_POLICY)
                    && expected.get("name").and_then(Value::as_str)
                        == Some(actual.mask.named_tool.as_str())
                    && expected.get("schema_sha256").and_then(Value::as_str)
                        == Some(actual.mask.schema_sha256.as_str())
                    && expected.get("native_token_mask_before_greedy") == Some(&json!(true))
                    && expected.get("ordinary_bytes_cannot_impersonate_native_DSML")
                        == Some(&json!(true))
                    && expected.get("actual_name_JSON_schema_and_finish_checked")
                        == Some(&json!(true)),
                "actual constrained policy/name/schema differs from original request"
            );
            let (&selected, prefix) = generated
                .split_last()
                .context("forced native output is empty")?;
            super::tool_constraint::validate_execution_chain(
                None,
                true,
                Some(actual),
                request_sha256,
                prompt,
                prefix,
                selected,
            )?;
            ensure!(
                actual.mask.tokenizer_sha256 == tokenizer_sha256
                    && actual.sampler_source_id_hex == env!("DSV41_TOOL_CONSTRAINT_SOURCE_ID")
                    && actual.sampler_ptx_sha256
                        == format!(
                            "{:x}",
                            sha2::Sha256::digest(super::gpu_tool_constraint::PTX.as_bytes())
                        ),
                "actual constrained sampler/tokenizer source or PTX differs from this build"
            );
        }
    }
    Ok(())
}

fn validate_facts(
    chat: &PlannedChat,
    expected: &CpuRequestPlan,
    authority: &ResponseAuthorityBinding,
    f: &CompletionFacts<'_>,
) -> Result<()> {
    validate_facts_transport(chat, expected, authority, f, false)
}
fn validate_facts_transport(
    chat: &PlannedChat,
    expected: &CpuRequestPlan,
    authority: &ResponseAuthorityBinding,
    f: &CompletionFacts<'_>,
    streaming: bool,
) -> Result<()> {
    let s = chat.semantics();
    let b = f.binding;
    let m = f.model;
    validate_forced_constraint(
        s.forced_tool_constraint.as_ref(),
        f.forced_constraint,
        &b.original_request_sha256,
        &b.tokenizer_sha256,
        f.input,
        f.generated,
    )?;
    for value in [
        &authority.global_checkpoint_geometry_checksum_sha256,
        &authority.execution_snapshot_sha256,
        &authority.engine_binary_sha256,
        &authority.head_manifest_sha256,
        &authority.program_sha256,
    ] {
        ensure!(
            digest(value),
            "expected runtime/model/program identity is not SHA256"
        );
    }
    ensure!(
        authority.completion_owner != [0; 32],
        "empty expected collective session owner"
    );
    ensure!(
        f.rank == 0 && m.rank == 0,
        "only checked head-rank0 produces an API response"
    );
    ensure!(
        f.binding == expected.binding() && f.plan_sha256 == expected.digest(),
        "completion differs from exact immutable request plan/binding"
    );
    ensure!(
        f.program_sha256 == authority.program_sha256
            && f.owner == authority.completion_owner
            && f.lane == authority.lane
            && b.lane == authority.lane,
        "completion program/session/lane identity differs"
    );
    ensure!(
        b.model_revision == s.model_revision
            && b.tokenizer_sha256 == s.tokenizer_sha256
            && b.toolkit_revision == s.toolkit_revision
            && b.original_request_sha256 == s.original_request_sha256,
        "actual model/tokenizer/toolkit/original HTTP request identity differs"
    );
    ensure!(
        b.global_checkpoint_geometry_checksum_sha256
            == authority.global_checkpoint_geometry_checksum_sha256
            && b.execution_snapshot_sha256 == authority.execution_snapshot_sha256
            && b.engine_binary_sha256 == authority.engine_binary_sha256
            && b.frequency == authority.frequency,
        "actual checkpoint/binary/snapshot/frequency identity differs"
    );
    ensure!(
        m.model_revision == b.model_revision
            && m.tokenizer_sha256 == b.tokenizer_sha256
            && m.manifest_sha256 == authority.head_manifest_sha256
            && m.global_checkpoint_geometry_checksum_sha256
                == b.global_checkpoint_geometry_checksum_sha256
            && m.runtime.execution_snapshot_sha256 == b.execution_snapshot_sha256
            && m.runtime.engine_binary_sha256 == b.engine_binary_sha256
            && m.frequency_profile == b.frequency,
        "opaque model-step identity differs from request/runtime authority binding"
    );
    ensure!(
        m.concurrency == 1 && m.temperature == 0.0 && !m.speculation && !m.prefix_restore,
        "only C1 greedy/speculation-off/cold-prefix execution is supported"
    );
    ensure!(
        f.input == chat.prompt_ids()
            && f.input == expected.input_ids()
            && model_session_plan::native_ids_sha256(f.input) == f.input_sha256
            && f.input_sha256 == s.prompt_ids_sha256_u32le
            && f.input_sha256 == expected.input_digest(),
        "actual complete native prompt IDs/hash differ; no truncation/canonicalization allowed"
    );
    ensure!(
        f.input.len() == s.prompt_tokens as usize
            && b.budget.output_allowance == s.output_allowance as usize
            && b.budget.context_tokens == s.planning_limits.context_tokens as usize
            && s.required_context_tokens as usize <= b.budget.context_tokens
            && b.budget.context_tokens <= m.native_context,
        "full requested output/context allowance was changed or exceeds actual model context"
    );
    let sampling = &b.sampling;
    ensure!(
        sampling.temperature == s.effective_temperature as f64
            && sampling.temperature == 0.0
            && sampling.top_p == s.effective_top_p as f64
            && sampling.top_p == 1.0
            && sampling.presence_penalty == s.presence_penalty as f64
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == s.frequency_penalty as f64
            && sampling.frequency_penalty == 0.0
            && sampling.seed == s.seed
            && sampling.thinking == s.thinking
            && sampling.effective_reasoning_score == s.effective_reasoning_score
            && !sampling.speculation,
        "effective thinking/sampling/seed/penalty semantics differ"
    );
    ensure!(
        s.stream == streaming
            && if streaming {
                chat.original_request()
                    .get("stream")
                    .and_then(Value::as_bool)
                    == Some(true)
            } else {
                chat.original_request()
                    .get("stream")
                    .is_none_or(|x| x == false)
            },
        "native response transport differs from unchanged original request"
    );
    ensure!(b.stop.eos_native_id==EOS_ID && b.stop.user_sequences==s.stop_sequences
        && s.stop_sequences.is_empty() && !s.ignore_eos && s.min_tokens==0,
        "only actual native EOS/full allowance terminal authority is integrated; user stops unsupported");
    ensure!(
        f.all_rank_agreement && f.lifetime_released,
        "all-eight prefix agreement and concrete transport lifetime release are required"
    );
    ensure!(
        !f.generated.is_empty()
            && f.generated.len() <= b.budget.output_allowance
            && model_session_plan::native_ids_sha256(f.generated) == f.generated_sha256,
        "actual generated native count/hash differs"
    );
    let eos = f.generated.iter().filter(|id| **id == EOS_ID).count();
    match f.terminal {
        NativeTerminal::Eos => ensure!(
            eos == 1 && f.generated.last() == Some(&EOS_ID),
            "EOS authority requires exactly one final native EOS; no continuation after EOS"
        ),
        NativeTerminal::Allowance => ensure!(
            eos == 0 && f.generated.len() == b.budget.output_allowance,
            "cap authority requires unchanged complete allowance without EOS"
        ),
    }
    // The final selected ID is an output, not a future input. Intermediate
    // prefill head selections are absent from both generated IDs and this hash.
    let mut consumed = f.input.to_vec();
    consumed.extend_from_slice(&f.generated[..f.generated.len() - 1]);
    ensure!(
        model_session_plan::native_ids_sha256(&consumed) == f.consumed_sha256,
        "native consumed prefix differs from complete prompt plus prior generated IDs"
    );
    let at = expected
        .steps()
        .iter()
        .position(|step| step.output_ordinal == Some(f.generated.len()))
        .context("generated count has no actual complete model-step boundary")?;
    let step = &expected.steps()[at];
    ensure!(
        f.completed_steps == at + 1
            && f.completed_operations == step.operation_end
            && f.actual_epoch == step.epoch_after
            && step.expected_boundary.generated_native_count == f.generated.len()
            && step.expected_boundary.consumed_native_context == consumed.len(),
        "actual model-step/operation/epoch/native-prefix boundary differs"
    );
    ensure!(
        f.next_operation_id == expected.next_operation_id()
            && f.next_step_id
                == expected
                    .steps()
                    .last()
                    .context("empty expected schedule")?
                    .step_id
                    .checked_add(1)
                    .context("step reservation overflow")?,
        "reserved next operation/step IDs must follow the entire unused planned tail"
    );
    ensure!(
        f.elapsed.is_finite()
            && f.elapsed >= 0.0
            && f.elapsed <= b.budget.wall_millis as f64 / 1000.0
            && !f.timing_scope.is_empty(),
        "actual checked-request wall budget/timing scope invalid"
    );
    Ok(())
}

fn completion_facts(completion: &CheckedRequestCompletion) -> CompletionFacts<'_> {
    let terminal = match completion.stop_reason() {
        RequestStopReason::NativeEos => NativeTerminal::Eos,
        RequestStopReason::OutputAllowance => NativeTerminal::Allowance,
    };
    CompletionFacts {
        binding: completion.binding(),
        model: completion.model_identity(),
        rank: completion.rank(),
        lane: completion.lane(),
        owner: completion.completion_owner(),
        plan_sha256: completion.plan_sha256(),
        program_sha256: completion.program_sha256(),
        input: completion.input_native_ids(),
        input_sha256: completion.input_sha256(),
        generated: completion.generated_native_ids(),
        generated_sha256: completion.generated_sha256(),
        consumed_sha256: completion.consumed_native_prefix_sha256(),
        terminal,
        completed_steps: completion.completed_model_steps(),
        completed_operations: completion.completed_operations(),
        actual_epoch: completion.actual_epoch(),
        all_rank_agreement: completion.all_rank_prefix_agreement(),
        lifetime_released: completion.transport_lifetime_release_confirmed(),
        elapsed: completion.elapsed_request_seconds(),
        timing_scope: completion.timing_scope(),
        next_operation_id: completion.next_operation_id(),
        next_step_id: completion.next_step_id(),
        forced_constraint: completion.forced_tool_constraint(),
    }
}

/// Assemble only a BUFFERED response from a checked native request completion.
/// Ownership remains with the caller on every path; no borrowed GPU owner is
/// released here. An invalid or incomplete receipt produces retained evidence,
/// never a normal OpenAI completion. This does not open CPU execution_permit.
pub async fn buffered_response(
    bridge: &CpuBridge,
    chat: PlannedChat,
    expected: &CpuRequestPlan,
    authority: &ResponseAuthorityBinding,
    completion: &CheckedRequestCompletion,
    response_id: String,
) -> BufferedNativeResponse {
    let f = completion_facts(completion);
    let terminal = f.terminal;
    let audit = json!({"terminal_authority":"opaque CheckedRequestCompletion issued after checked GPU model steps and actual all-rank prefix agreement/release",
        "rank":f.rank,"lane":f.lane,"completion_owner":f.owner,"plan_sha256":f.plan_sha256,"program_sha256":f.program_sha256,
        "request_binding":f.binding,"model_identity":f.model,
        "input_native_ids_sha256_u32le":f.input_sha256,"generated_native_ids_sha256_u32le":f.generated_sha256,
        "consumed_native_prefix_sha256_u32le":f.consumed_sha256,"terminal":match terminal{NativeTerminal::Eos=>"native_eos",NativeTerminal::Allowance=>"full_output_allowance"},
        "completed_model_steps":f.completed_steps,"completed_operations":f.completed_operations,"actual_epoch":f.actual_epoch,
        "all_rank_prefix_agreement":f.all_rank_agreement,"transport_lifetime_release_confirmed":f.lifetime_released,
        "elapsed_checked_request_seconds":f.elapsed,"timing_scope":f.timing_scope,"client_TTFT_measured":false,"actual_forced_tool_constraint":f.forced_constraint,
        "next_operation_id_reserved_after_full_plan":f.next_operation_id,"next_step_id_reserved_after_full_plan":f.next_step_id,
        "checkpoint_identity_scope":"verified global checkpoint/header/geometry plus actual per-rank loader receipt; whole payload identity is only the loader's explicit flag",
        "artifact_identity_scope":"ELF/snapshot IDs are independently expected supervisor attestations; model runtime does not hash its own executable"});
    let mut report = BufferedNativeResponse {
        passed: false,
        response: None,
        error: None,
        native_prompt_ids: f.input.to_vec(),
        native_generated_ids: f.generated.to_vec(),
        proof_audit: audit,
        parser_report: None,
        output_transport: "buffered only; no live SSE or usable-output TTFT measurement",
        TTFT_measured: false,
        runtime_terminal_authority_checked: false,
        model_or_task_quality_qualified: false,
        serving_backend_qualified: false,
    };
    if let Err(e) = validate_facts(&chat, expected, authority, &f) {
        report.error = Some(format!(
            "native completion binding/terminal rejected: {e:#}"
        ));
        return report;
    }
    report.runtime_terminal_authority_checked = true;
    let transcript = CpuTranscript {
        token_chunks: vec![f.generated.to_vec()],
        terminal: match terminal {
            NativeTerminal::Eos => CpuTerminal::NativeEos,
            NativeTerminal::Allowance => CpuTerminal::OutputAllowance,
        },
        cached_prompt_tokens: 0,
    };
    match bridge
        .parse_cpu_transcript(chat, response_id, &transcript)
        .await
    {
        Ok(parsed) => {
            report.passed = parsed.passed;
            report.response = parsed.response.clone();
            report.error = parsed.error.clone();
            if report.passed {
                match report
                    .response
                    .take()
                    .context("official buffered response missing")
                    .and_then(buffered_openai_tool_calls)
                {
                    Ok((response, removed)) => {
                        report.response = Some(response);
                        report.proof_audit["buffered_stream_only_tool_call_indices_removed"] =
                            json!(removed);
                    }
                    Err(error) => {
                        report.passed = false;
                        report.error = Some(format!(
                            "official buffered tool-call shape rejected: {error:#}"
                        ));
                    }
                }
            }
            // Keep the CPU parser's original qualification/authority labels.
            // Also retain its unmodified accumulated response and stream events.
            // Only this outer typed adapter establishes checked native origin.
            report.parser_report = Some(parsed);
        }
        Err(e) => {
            report.error = Some(format!(
                "official parser rejected checked native transcript: {e:#}"
            ))
        }
    }
    report
}

/// The official chunk accumulator retains the delta's tool-call `index`.
/// OpenAI's full assistant message uses array order instead: publishing that
/// stream-only field breaks an unchanged historical tool-result round trip.
/// Normalize only validated sequential indices, retaining every semantic byte
/// (including argument JSON, IDs, names and Unicode) and the original parser
/// response separately. Historical-request unknown-field checks stay strict.
pub(crate) fn buffered_openai_tool_calls(mut response: Value) -> Result<(Value, usize)> {
    let choices = response
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .context("official buffered choices missing")?;
    let mut removed = 0usize;
    for choice in choices {
        let message = choice
            .get_mut("message")
            .and_then(Value::as_object_mut)
            .context("official buffered assistant message missing")?;
        if let Some(calls) = message.get_mut("tool_calls").filter(|v| !v.is_null()) {
            let calls = calls
                .as_array_mut()
                .context("official buffered tool calls must be an array")?;
            for (ordinal, call) in calls.iter_mut().enumerate() {
                let call = call
                    .as_object_mut()
                    .context("official buffered tool call must be an object")?;
                if let Some(index) = call.get("index") {
                    ensure!(
                        index.as_u64() == Some(ordinal as u64),
                        "official buffered tool-call index/order differs"
                    );
                    call.remove("index");
                    removed += 1;
                }
            }
        }
    }
    Ok((response, removed))
}

pub fn capabilities() -> Value {
    json!({"opaque_native_completion_adapter":true,"buffered_OpenAI_response_only":true,
        "execution_permit":false,"HTTP_listener":false,"SSE_transport":false,"TTFT_measurement":false,
        "C1_backend_qualified":false,"8192_context_quality_qualified":false,"atomic_batching_qualified":false,
        "tool_execution":false,"model_or_task_quality_qualified":false})
}

#[cfg(test)]
mod tests {
    use super::super::{
        gpu_model_chunk::ModelChunkRuntimeBinding,
        model_session_plan::{GreedySemantics, RequestBudget, StopPolicy},
        native_api_contract::{PlanningLimits, SERVED_MODEL},
        position::FrequencyBackend,
    };
    use super::*;
    use std::sync::OnceLock;
    use uuid::Uuid;
    const TOKENIZER: &str =
        "test-data/tokenizer.json";
    fn bridge() -> &'static CpuBridge {
        static B: OnceLock<CpuBridge> = OnceLock::new();
        B.get_or_init(|| {
            CpuBridge::from_verified_tokenizer(&std::fs::read(TOKENIZER).unwrap()).unwrap()
        })
    }
    #[test]
    fn actual_native_buffered_tool_result_roundtrip_preserves_official_prompt_ids() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_actual_toolcall_roundtrip.json"
        ))
        .unwrap();
        let limits = PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 1,
            request_bytes: 1024 * 1024,
        };
        // The original strict consumer must continue refusing unknown historical
        // fields. The actual producer is fixed instead of ignoring request data.
        let original = fixture["actual_failed_followup"].clone();
        assert!(bridge().prepare_single(original.clone(), limits).is_err());
        let (normal, removed) =
            buffered_openai_tool_calls(fixture["actual_response"].clone()).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(normal, fixture["normalized_response"]);
        // This expected body was generated from the actual response through the
        // unchanged scripts/client.py clean_message, whose exact SHA is retained.
        let follow = fixture["same_original_clean_message_followup"].clone();
        assert_eq!(follow["messages"][1], normal["choices"][0]["message"]);
        let planned = bridge().prepare_single(follow.clone(), limits).unwrap();
        let tokenizer = tokenizers_recipe::Tokenizer::from_file(TOKENIZER).unwrap();
        let encoder = deepseek_recipe_encoding::PromptTokenizer::new(&tokenizer);
        let official = super::super::protocol::prepare_chat(original).unwrap();
        assert_eq!(planned.prompt_ids(), official.encode_ids(&encoder).unwrap());
        assert_eq!(planned.original_request(), &follow);
        assert_eq!(planned.semantics().output_allowance, 512);
        let mut unexpected = follow;
        unexpected["messages"][1]["tool_calls"][0]["unknown_metadata"] = json!(true);
        assert!(bridge().prepare_single(unexpected, limits).is_err());
    }

    #[test]
    fn buffered_index_normalization_preserves_nested_unicode_and_call_associations() {
        let response = json!({"choices":[{"message":{"role":"assistant","content":"",
            "tool_calls":[{"index":0,"id":"first","type":"function","function":{"name":"echo","arguments":"{\"value\":{\"text\":\"café 東京\",\"nested\":[null,{\"index\":7}]}}"}},
                          {"index":1,"id":"second","type":"function","function":{"name":"echo","arguments":"{\"value\":null}"}}]}}]});
        let (normalized, removed) = buffered_openai_tool_calls(response.clone()).unwrap();
        assert_eq!(removed, 2);
        for (before, after) in response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .zip(
                normalized["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .unwrap(),
            )
        {
            let mut expected = before.clone();
            expected.as_object_mut().unwrap().remove("index");
            assert_eq!(*after, expected);
        }
        assert_eq!(
            buffered_openai_tool_calls(normalized.clone()).unwrap(),
            (normalized, 0)
        );
        for invalid in [json!(true), json!(-1), json!(0.0), json!(1), json!("0")] {
            let mut bad = response.clone();
            bad["choices"][0]["message"]["tool_calls"][0]["index"] = invalid;
            assert!(buffered_openai_tool_calls(bad).is_err());
        }
    }
    // These tests create PRIVATE CPU facts, never an opaque runtime receipt.
    // No public JSON/CPU constructor is added to CheckedRequestCompletion.
    struct Fixture {
        chat: PlannedChat,
        plan: CpuRequestPlan,
        authority: ResponseAuthorityBinding,
        model: ModelChunkIdentity,
        generated: Vec<u32>,
        generated_sha: String,
        consumed_sha: String,
        terminal: NativeTerminal,
        completed_steps: usize,
        completed_operations: usize,
        actual_epoch: u64,
    }
    impl Fixture {
        fn new(
            thinking: bool,
            allowance: u32,
            generated: Vec<u32>,
            terminal: NativeTerminal,
        ) -> Self {
            let chat=bridge().prepare_single(json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":"Answer briefly."}],
                "temperature":0,"top_p":1,"seed":11,"max_tokens":allowance,"return_token_ids":true,
                "chat_template_kwargs":{"thinking":thinking,"reasoning_effort":75}}),PlanningLimits{
                    context_tokens:8192,prompt_tokens:8192,output_tokens:4096,atomic_requests:1,request_bytes:1024*1024}).unwrap();
            let s = chat.semantics();
            let lane = LaneKey {
                boot: Uuid::from_u128(214),
                slot: 0,
                generation: 7,
            };
            let frequency = FrequencyBackend::CudaTorch.identity();
            let binding = RequestBinding {
                model_revision: s.model_revision.into(),
                global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                execution_snapshot_sha256: "b".repeat(64),
                engine_binary_sha256: "c".repeat(64),
                original_request_sha256: s.original_request_sha256.clone(),
                tokenizer_sha256: s.tokenizer_sha256.into(),
                toolkit_revision: s.toolkit_revision.into(),
                frequency: frequency.clone(),
                sampling: GreedySemantics {
                    temperature: 0.,
                    top_p: 1.,
                    presence_penalty: 0.,
                    frequency_penalty: 0.,
                    seed: s.seed,
                    thinking: s.thinking,
                    effective_reasoning_score: s.effective_reasoning_score,
                    speculation: false,
                },
                stop: StopPolicy {
                    eos_native_id: 1,
                    user_sequences: vec![],
                },
                budget: RequestBudget {
                    context_tokens: 8192,
                    output_allowance: allowance as usize,
                    wall_millis: 180_000,
                },
                lane,
            };
            let plan =
                CpuRequestPlan::new(binding, chat.prompt_ids().to_vec(), 71, 19, 300).unwrap();
            let authority = ResponseAuthorityBinding {
                global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                execution_snapshot_sha256: "b".repeat(64),
                engine_binary_sha256: "c".repeat(64),
                head_manifest_sha256: "d".repeat(64),
                frequency: frequency.clone(),
                lane,
                completion_owner: [7; 32],
                program_sha256: "e".repeat(64),
            };
            let model = ModelChunkIdentity {
                model_revision: s.model_revision.into(),
                rank: 0,
                manifest_sha256: "d".repeat(64),
                global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                full_checkpoint_payload_identity_verified_by_loader: false,
                runtime: ModelChunkRuntimeBinding {
                    execution_snapshot_sha256: "b".repeat(64),
                    engine_binary_sha256: "c".repeat(64),
                },
                runtime_artifacts_self_verified: false,
                frequency_profile: frequency,
                tokenizer_sha256: s.tokenizer_sha256.into(),
                native_context: 8192,
                maximum_chunk_tokens: 128,
                concurrency: 1,
                temperature: 0.,
                speculation: false,
                prefix_restore: false,
                model_or_quality_qualified: false,
                HTTP_serving_enabled: false,
            };
            let generated_sha = model_session_plan::native_ids_sha256(&generated);
            let mut consumed = chat.prompt_ids().to_vec();
            consumed.extend_from_slice(&generated[..generated.len() - 1]);
            let consumed_sha = model_session_plan::native_ids_sha256(&consumed);
            let at = plan
                .steps()
                .iter()
                .position(|x| x.output_ordinal == Some(generated.len()))
                .unwrap();
            let step = &plan.steps()[at];
            let (completed_steps, completed_operations, actual_epoch) =
                (at + 1, step.operation_end, step.epoch_after);
            Self {
                chat,
                plan,
                authority,
                model,
                generated,
                generated_sha,
                consumed_sha,
                terminal,
                completed_steps,
                completed_operations,
                actual_epoch,
            }
        }
        fn facts(&self) -> CompletionFacts<'_> {
            CompletionFacts {
                binding: self.plan.binding(),
                model: &self.model,
                rank: 0,
                lane: self.authority.lane,
                owner: self.authority.completion_owner,
                plan_sha256: self.plan.digest(),
                program_sha256: &self.authority.program_sha256,
                input: self.plan.input_ids(),
                input_sha256: self.plan.input_digest(),
                generated: &self.generated,
                generated_sha256: &self.generated_sha,
                consumed_sha256: &self.consumed_sha,
                terminal: self.terminal,
                completed_steps: self.completed_steps,
                completed_operations: self.completed_operations,
                actual_epoch: self.actual_epoch,
                all_rank_agreement: true,
                lifetime_released: true,
                elapsed: 1.,
                timing_scope: "CPU test fixture only; no runtime authority",
                next_operation_id: self.plan.next_operation_id(),
                next_step_id: self.plan.steps().last().unwrap().step_id + 1,
                forced_constraint: None,
            }
        }
        fn valid(&self) -> Result<()> {
            validate_facts(&self.chat, &self.plan, &self.authority, &self.facts())
        }
    }
    #[test]
    fn private_cpu_eos_and_full_cap_preserve_quality_and_speed_allowances() {
        let quality = Fixture::new(false, 4096, vec![17, 1], NativeTerminal::Eos);
        quality.valid().unwrap();
        assert_eq!(quality.plan.binding().budget.output_allowance, 4096);
        assert!(!quality.chat.semantics().thinking);
        let speed = Fixture::new(true, 256, vec![17; 256], NativeTerminal::Allowance);
        speed.valid().unwrap();
        assert_eq!(speed.chat.semantics().effective_reasoning_score, Some(75));
        assert!(quality.chat.execution_permit().is_err());
        assert!(speed.chat.execution_permit().is_err());
    }
    #[test]
    fn native_terminal_hash_step_release_and_budget_failures_are_not_normal_completions() {
        let f = Fixture::new(false, 32, vec![17, 1], NativeTerminal::Eos);
        let mut checked = 0;
        macro_rules! bad {
            ($edit:expr) => {{
                let mut view = f.facts();
                $edit(&mut view);
                assert!(validate_facts(&f.chat, &f.plan, &f.authority, &view).is_err());
                checked += 1;
            }};
        }
        bad!(|x: &mut CompletionFacts<'_>| x.rank = 1);
        bad!(|x: &mut CompletionFacts<'_>| x.lane.generation += 1);
        bad!(|x: &mut CompletionFacts<'_>| x.owner = [8; 32]);
        bad!(|x: &mut CompletionFacts<'_>| x.plan_sha256 = "bad");
        bad!(|x: &mut CompletionFacts<'_>| x.program_sha256 = "bad");
        bad!(|x: &mut CompletionFacts<'_>| x.input_sha256 = "bad");
        bad!(|x: &mut CompletionFacts<'_>| x.generated_sha256 = "bad");
        bad!(|x: &mut CompletionFacts<'_>| x.consumed_sha256 = "bad");
        bad!(|x: &mut CompletionFacts<'_>| x.generated = &[17, 1, 18]);
        bad!(|x: &mut CompletionFacts<'_>| x.generated = &[1, 1]);
        bad!(|x: &mut CompletionFacts<'_>| x.terminal = NativeTerminal::Allowance);
        bad!(|x: &mut CompletionFacts<'_>| x.completed_steps -= 1);
        bad!(|x: &mut CompletionFacts<'_>| x.completed_operations -= 1);
        bad!(|x: &mut CompletionFacts<'_>| x.actual_epoch += 3);
        bad!(|x: &mut CompletionFacts<'_>| x.all_rank_agreement = false);
        bad!(|x: &mut CompletionFacts<'_>| x.lifetime_released = false);
        bad!(|x: &mut CompletionFacts<'_>| x.elapsed = f64::NAN);
        bad!(|x: &mut CompletionFacts<'_>| x.elapsed = 181.);
        bad!(|x: &mut CompletionFacts<'_>| x.timing_scope = "");
        bad!(|x: &mut CompletionFacts<'_>| x.next_operation_id -= 1);
        bad!(|x: &mut CompletionFacts<'_>| x.next_step_id -= 1);
        assert_eq!(checked, 21);
    }
    #[test]
    fn actual_model_identity_and_independent_runtime_expectations_cannot_be_substituted() {
        let mutations: Vec<Box<dyn Fn(&mut Fixture)>> = vec![
            Box::new(|f| f.model.runtime.engine_binary_sha256 = "f".repeat(64)),
            Box::new(|f| f.model.runtime.execution_snapshot_sha256 = "f".repeat(64)),
            Box::new(|f| f.model.manifest_sha256 = "f".repeat(64)),
            Box::new(|f| f.model.global_checkpoint_geometry_checksum_sha256 = "f".repeat(64)),
            Box::new(|f| f.model.frequency_profile.yarn_table_sha256 = "f".repeat(64)),
            Box::new(|f| f.model.native_context = 512),
            Box::new(|f| f.model.prefix_restore = true),
            Box::new(|f| f.model.speculation = true),
            Box::new(|f| f.model.concurrency = 8),
            Box::new(|f| f.model.temperature = 0.5),
            Box::new(|f| f.model.rank = 7),
            Box::new(|f| f.authority.lane.generation += 1),
            Box::new(|f| f.authority.engine_binary_sha256 = "f".repeat(64)),
        ];
        for change in mutations {
            let mut f = Fixture::new(false, 4096, vec![17, 1], NativeTerminal::Eos);
            change(&mut f);
            assert!(f.valid().is_err());
        }
    }
    #[test]
    fn complete_request_binding_preserves_full_budget_thinking_and_exact_native_prompt() {
        let changes: Vec<Box<dyn Fn(&mut RequestBinding)>> = vec![
            Box::new(|b| b.original_request_sha256 = "f".repeat(64)),
            Box::new(|b| b.budget.output_allowance = 32),
            Box::new(|b| b.budget.context_tokens = 4096),
            Box::new(|b| b.sampling.thinking = true),
            Box::new(|b| b.sampling.effective_reasoning_score = Some(1)),
            Box::new(|b| b.sampling.seed = Some(47)),
            Box::new(|b| b.stop.user_sequences.push("STOP".into())),
        ];
        for change in changes {
            let mut f = Fixture::new(false, 256, vec![17, 1], NativeTerminal::Eos);
            let mut binding = f.plan.binding().clone();
            change(&mut binding);
            f.plan =
                CpuRequestPlan::new(binding, f.chat.prompt_ids().to_vec(), 71, 19, 300).unwrap();
            assert!(f.valid().is_err());
        }
        let mut f = Fixture::new(false, 256, vec![17, 1], NativeTerminal::Eos);
        let mut input = f.chat.prompt_ids().to_vec();
        input[0] = if input[0] == 17 { 18 } else { 17 };
        f.plan = CpuRequestPlan::new(f.plan.binding().clone(), input, 71, 19, 300).unwrap();
        assert!(f.valid().is_err());
        // Hash-consistent malformed terminals must fail on EOS/cap rules,
        // rather than depending on a stale generated hash to reject them.
        for (ids, terminal) in [
            (vec![17, 1, 18], NativeTerminal::Eos),
            (vec![1, 1], NativeTerminal::Eos),
            (vec![17, 18], NativeTerminal::Eos),
            (vec![17, 18], NativeTerminal::Allowance),
            (vec![17, 1], NativeTerminal::Allowance),
        ] {
            let f = Fixture::new(false, 32, ids, terminal);
            assert!(f.valid().is_err());
        }
    }

    #[test]
    fn opaque_runtime_adapter_does_not_claim_listener_batching_quality_or_ttft() {
        let c = capabilities();
        for name in [
            "execution_permit",
            "HTTP_listener",
            "SSE_transport",
            "TTFT_measurement",
            "C1_backend_qualified",
            "8192_context_quality_qualified",
            "atomic_batching_qualified",
            "tool_execution",
            "model_or_task_quality_qualified",
        ] {
            assert_eq!(c[name], false);
        }
    }
    #[test]
    fn forced_native_authority_requires_exact_requested_schema_prefix_and_compiled_sampler() {
        use super::super::tool_constraint::{ids_hash, ConstraintIdentity};
        let request = "a".repeat(64);
        let tokenizer = super::super::engram::TOKENIZER_SHA256;
        let prompt = vec![17, 33];
        let generated = vec![2581, 1];
        let expected = json!({"policy":FORCED_POLICY,"name":"echo","schema_sha256":"b".repeat(64),
            "native_token_mask_before_greedy":true,"ordinary_bytes_cannot_impersonate_native_DSML":true,
            "actual_name_JSON_schema_and_finish_checked":true,"qualified":false});
        let actual = ConstraintExecution {
            mask: ConstraintIdentity {
                policy: FORCED_POLICY,
                named_tool: "echo".into(),
                schema_sha256: "b".repeat(64),
                request_sha256: request.clone(),
                tokenizer_sha256: tokenizer.into(),
                prompt_ids_sha256: ids_hash(&prompt),
                generated_prefix_sha256: ids_hash(&generated[..1]),
                generated_prefix_tokens: 1,
                allowed_native_tokens: 1,
                mask_sha256: "c".repeat(64),
                build_cpu_millis: 1.0,
                constrained_greedy_before_local_maxloc: true,
            },
            sampler_source_id_hex: env!("DSV41_TOOL_CONSTRAINT_SOURCE_ID").into(),
            sampler_ptx_sha256: format!(
                "{:x}",
                sha2::Sha256::digest(super::super::gpu_tool_constraint::PTX.as_bytes())
            ),
            selected_native_id: 1,
            local_mask_kernel_status_checked: true,
            global_selected_allowed: true,
        };
        let valid = |expected: Option<&Value>, actual: Option<&ConstraintExecution>| {
            validate_forced_constraint(expected, actual, &request, tokenizer, &prompt, &generated)
        };
        valid(Some(&expected), Some(&actual)).unwrap();
        valid(None, None).unwrap();
        assert!(valid(Some(&expected), None).is_err());
        assert!(valid(None, Some(&actual)).is_err());
        for which in 0..14 {
            let mut bad = actual.clone();
            match which {
                0 => bad.mask.named_tool = "other".into(),
                1 => bad.mask.schema_sha256 = "d".repeat(64),
                2 => bad.mask.request_sha256 = "d".repeat(64),
                3 => bad.mask.tokenizer_sha256 = "d".repeat(64),
                4 => bad.mask.prompt_ids_sha256 = "d".repeat(64),
                5 => bad.mask.generated_prefix_sha256 = ids_hash(&generated),
                6 => bad.mask.generated_prefix_tokens = 2,
                7 => bad.mask.allowed_native_tokens = 0,
                8 => bad.mask.constrained_greedy_before_local_maxloc = false,
                9 => bad.sampler_source_id_hex = "0".repeat(16),
                10 => bad.sampler_ptx_sha256 = "0".repeat(64),
                11 => bad.selected_native_id = 2581,
                12 => bad.local_mask_kernel_status_checked = false,
                _ => bad.global_selected_allowed = false,
            };
            assert!(
                valid(Some(&expected), Some(&bad)).is_err(),
                "accepted invalid constrained receipt{which}"
            );
        }
        let mut wrong = expected.clone();
        wrong["native_token_mask_before_greedy"] = json!(false);
        assert!(valid(Some(&wrong), Some(&actual)).is_err());
    }
}

/// Opaque transport footer authority. Only an actual checked native completion
/// can reach this constructor; token/JSON/EOF assertions cannot mint one.
#[derive(Debug, Serialize)]
pub struct CheckedStreamFinish {
    response_id: String,
    original_request_sha256: String,
    plan_sha256: String,
    input_native_ids: Vec<u32>,
    generated_native_ids: Vec<u32>,
    eos: bool,
    proof_audit: Value,
}
impl CheckedStreamFinish {
    pub fn response_id(&self) -> &str {
        &self.response_id
    }
    pub fn request_sha256(&self) -> &str {
        &self.original_request_sha256
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn input_native_ids(&self) -> &[u32] {
        &self.input_native_ids
    }
    pub fn generated_native_ids(&self) -> &[u32] {
        &self.generated_native_ids
    }
    pub fn eos(&self) -> bool {
        self.eos
    }
    pub fn proof_audit(&self) -> &Value {
        &self.proof_audit
    }
}
pub fn checked_stream_finish(
    chat: &PlannedChat,
    expected: &CpuRequestPlan,
    authority: &ResponseAuthorityBinding,
    completion: &CheckedRequestCompletion,
    response_id: String,
) -> Result<CheckedStreamFinish> {
    ensure!(!response_id.is_empty(), "stream response ID missing");
    let facts = completion_facts(completion);
    validate_facts_transport(chat, expected, authority, &facts, true)?;
    Ok(CheckedStreamFinish {
        response_id,
        original_request_sha256: facts.binding.original_request_sha256.clone(),
        plan_sha256: facts.plan_sha256.into(),
        input_native_ids: facts.input.to_vec(),
        generated_native_ids: facts.generated.to_vec(),
        eos: facts.terminal == NativeTerminal::Eos,
        proof_audit: json!({"actual_checked_native_completion":completion,
            "all_rank_prefix_agreement":facts.all_rank_agreement,
            "transport_lifetime_release_confirmed":facts.lifetime_released,
            "streaming_terminal_authority_checked":true,"model_or_task_quality_qualified":false}),
    })
}

#[cfg(test)]
pub(crate) fn stream_finish_CPU_fixture(
    response_id: &str,
    request_sha256: &str,
    plan_sha256: &str,
    input_native_ids: Vec<u32>,
    generated_native_ids: Vec<u32>,
    eos: bool,
) -> CheckedStreamFinish {
    CheckedStreamFinish {
        response_id: response_id.into(),
        original_request_sha256: request_sha256.into(),
        plan_sha256: plan_sha256.into(),
        input_native_ids,
        generated_native_ids,
        eos,
        proof_audit: json!({"CPU_fixture_only":true,"actual_GPU_or_allrank_proof":false}),
    }
}
