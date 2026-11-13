//! Buffered batch response from a genuine sealed native group completion.
//! The official parser consumes the SAME owned per-member chat plans; this
//! adapter never decodes or fabricates native tokens and exposes no SSE/TTFT.
use super::{
    atomic_request_admission::PreparedAtomicRequest,
    batch_request_plan::{digest, BatchStopReason},
    gpu_batch_request_session::CheckedBatchRequestCompletion,
    native_api_contract::{CpuBridge, CpuParseReport, CpuTerminal, CpuTranscript, SERVED_MODEL},
    native_response_bridge::buffered_openai_tool_calls,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
#[derive(Debug, Serialize)]
pub struct BufferedBatchResponse {
    response: Value,
    parser_reports: Vec<CpuParseReport>,
    completion_audit: Value,
    buffered_tool_call_indices_removed_per_member: Vec<usize>,
    buffered: bool,
    TTFT_measured: bool,
    actual_all8_terminal_and_cleanup_checked: bool,
    benchmark_or_task_quality_qualified: bool,
}
impl BufferedBatchResponse {
    pub fn response(&self) -> &Value {
        &self.response
    }
    pub fn into_response(self) -> Value {
        self.response
    }
    pub fn audit(&self) -> &Value {
        &self.completion_audit
    }
}
fn within_budget(admitted: Instant, wall_millis: u64) -> Result<()> {
    let deadline = admitted
        .checked_add(Duration::from_millis(wall_millis))
        .context("original group deadline overflow")?;
    ensure!(
        admitted <= Instant::now() && Instant::now() < deadline,
        "batch response exceeded original admission deadline; no late normal200"
    );
    Ok(())
}
/// Apply the exact qualified C1 buffered shape normalization to one member.
/// The original official report remains intact in private evidence; only the
/// normal response drops validated stream-only call indices. Call IDs, names,
/// arguments, native tokens and EOS-versus-cap classification are unchanged.
fn normal_member(report: &CpuParseReport, original_index: usize) -> Result<(Value, usize)> {
    ensure!(
        report.passed,
        "failed member cannot produce a normal choice"
    );
    let raw = report
        .response
        .as_ref()
        .context("successful parser response missing")?;
    let (response, removed) = buffered_openai_tool_calls(raw.clone())?;
    let choices = response["choices"]
        .as_array()
        .context("official choices missing")?;
    ensure!(
        choices.len() == 1,
        "one original atomic member yields one native completion"
    );
    let mut choice = choices[0].clone();
    choice["index"] = json!(original_index);
    Ok((choice, removed))
}
pub async fn buffered_response(
    bridge: &CpuBridge,
    prepared: PreparedAtomicRequest,
    completion: &CheckedBatchRequestCompletion,
    admitted: Instant,
    mut observe_parse: impl FnMut(&CpuParseReport) -> Result<()>,
) -> Result<BufferedBatchResponse> {
    let group = completion.binding();
    within_budget(admitted, group.wall_millis())?;
    ensure!(
        completion.model_identity().rank == 0
            && completion.all_rank_terminal_agreement()
            && completion.transport_lifetime_released()
            && completion.fresh_cleanup_checked()
            && group.original_body_sha256() == prepared.original_body_sha256()
            && group.prepared_group_sha256() == prepared.digest()
            && group.members().len() == prepared.members().len(),
        "actual sealed head group identity/order/all8 terminal+fresh cleanup differs"
    );
    for (member, plan) in group.members().iter().zip(prepared.members()) {
        ensure!(
            member.prompt_ids == plan.prompt_ids()
                && member.effective_semantics == serde_json::to_value(plan.semantics())?,
            "normal batch response requires SAME original ordered owned native chat plan"
        );
    }
    let chats = prepared.into_planned_group().into_members();
    let mut reports = Vec::with_capacity(chats.len());
    for ((plan, member), progress) in chats
        .into_iter()
        .zip(group.members())
        .zip(completion.progress())
    {
        let terminal = match progress
            .stop_reason
            .context("sealed terminal member missing reason")?
        {
            BatchStopReason::NativeEos => CpuTerminal::NativeEos,
            BatchStopReason::OutputAllowance => CpuTerminal::OutputAllowance,
        };
        let transcript = CpuTranscript {
            token_chunks: vec![progress.generated_native_ids.clone()],
            terminal,
            cached_prompt_tokens: 0,
        };
        let parsed = bridge
            .parse_cpu_transcript(plan, member.response_id.clone(), &transcript)
            .await?;
        observe_parse(&parsed)?;
        // Keep the parser's own counts/events separate from native EOS-inclusive
        // usage. Failed parsing must not fabricate an ordinary batch completion.
        ensure!(
            parsed.passed && parsed.response.is_some(),
            "official batch member{} parser failed:{:?}",
            member.original_index,
            parsed.error
        );
        ensure!(
            parsed.native_output_ids == progress.generated_native_ids
                && parsed.native_prompt_ids == member.prompt_ids
                && serde_json::to_value(&parsed.semantics)? == member.effective_semantics,
            "official parser/native transcript/member association differs"
        );
        reports.push(parsed);
    }
    let mut choices = Vec::with_capacity(reports.len());
    let mut normalized_indices = Vec::with_capacity(reports.len());
    let mut prompt = 0usize;
    let mut output = 0usize;
    for (index, report) in reports.iter().enumerate() {
        let member = report
            .response
            .as_ref()
            .context("successful parser response missing")?;
        ensure!(
            member["id"] == group.members()[index].response_id,
            "member response ID differs"
        );
        let (choice, removed) = normal_member(report, index)?;
        choices.push(choice);
        normalized_indices.push(removed);
        prompt = prompt
            .checked_add(report.native_prompt_ids.len())
            .context("batch prompt usage overflow")?;
        output = output
            .checked_add(report.native_generated_tokens)
            .context("batch native usage overflow")?;
    }
    let response = json!({"id":group.response_id(),"object":"chat.completion","model":SERVED_MODEL,
        "created":reports[0].response.as_ref().unwrap()["created"],"choices":choices,
        "usage":{"prompt_tokens":prompt,"completion_tokens":output,
            "total_tokens":prompt.checked_add(output).context("batch total usage overflow")?}});
    let audit = json!({"completion":completion,"group_response_id":group.response_id(),
        "normal_response_sha256":digest(b"dsv41-native-buffered-batch-response-v1\0", &response)?,
        "native_counts_sum_members_once":true,"runtime_terminal_authority_checked":true,
        "buffered_tool_call_indices_removed_per_member":normalized_indices,
        "buffered":true,"TTFT_measured":false,"task_quality_qualified":false});
    within_budget(admitted, group.wall_millis())?;
    Ok(BufferedBatchResponse {
        response,
        parser_reports: reports,
        completion_audit: audit,
        buffered_tool_call_indices_removed_per_member: normalized_indices,
        buffered: true,
        TTFT_measured: false,
        actual_all8_terminal_and_cleanup_checked: true,
        benchmark_or_task_quality_qualified: false,
    })
}

#[cfg(test)]
mod auto_tool_tests {
    use super::*;
    use crate::dsv41::{atomic_request_admission::CpuSlotBook, native_api_contract::ATOMIC_ROUTE};
    use std::sync::OnceLock;
    use uuid::Uuid;
    const TOKENIZER: &str =
        "test-data/tokenizer.json";
    const CALLS: &str = "<｜DSML｜ calls><｜DSML｜ invoke name=\"read\"><｜DSML｜ parameter name=\"label\" string=\"true\">東京</｜DSML｜ parameter><｜DSML｜ parameter name=\"optional\" string=\"false\">{\"nested\":[null,{\"index\":7}]}</｜DSML｜ parameter></｜DSML｜ invoke><｜DSML｜ invoke name=\"read\"><｜DSML｜ parameter name=\"label\" string=\"true\">café</｜DSML｜ parameter><｜DSML｜ parameter name=\"optional\" string=\"false\">null</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    fn bridge() -> &'static CpuBridge {
        static BRIDGE: OnceLock<CpuBridge> = OnceLock::new();
        BRIDGE.get_or_init(|| {
            CpuBridge::from_verified_tokenizer(&std::fs::read(TOKENIZER).unwrap()).unwrap()
        })
    }
    fn body() -> Value {
        // Explicit bounded CPU plumbing fixture, not an edited historical
        // protocol/quality body or a claim that the model chose these calls.
        json!({"model":SERVED_MODEL,"messages":[
            [{"role":"user","content":"Call read twice, preserving each result ID."}],
            [{"role":"user","content":"Do not call a tool. Reply READY_47."}]],
            "tools":[{"type":"function","function":{"name":"read","parameters":{
                "type":"object","properties":{"label":{"type":"string"},"optional":{}}}}}],
            "tool_choice":"auto","parallel_tool_calls":true,"temperature":0,"top_p":1,
            "seed":42,"max_tokens":128,"return_token_ids":true,
            "chat_template_kwargs":{"thinking":false}})
    }
    fn transcript(text: &str) -> CpuTranscript {
        let tokenizer = tokenizers_recipe::Tokenizer::from_file(TOKENIZER).unwrap();
        let mut ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        ids.push(1);
        CpuTranscript {
            token_chunks: ids.chunks(1).map(|v| v.to_vec()).collect(),
            terminal: CpuTerminal::NativeEos,
            cached_prompt_tokens: 0,
        }
    }
    // Exact field filter from unchanged scripts/client.py::clean_message.
    // Argument bytes, IDs and all nested values stay untouched.
    fn clean_message(message: &Value) -> Value {
        Value::Object(
            message
                .as_object()
                .unwrap()
                .iter()
                .filter(|(k, v)| {
                    matches!(
                        k.as_str(),
                        "role" | "content" | "tool_calls" | "reasoning_content"
                    ) && !v.is_null()
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }
    async fn reports(request: &Value) -> Vec<CpuParseReport> {
        let prepared =
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), request).unwrap();
        assert_eq!(prepared.original_request(), request);
        assert!(prepared.execution_permit().is_err());
        let mut out = Vec::new();
        for (index, plan) in prepared
            .into_planned_group()
            .into_members()
            .into_iter()
            .enumerate()
        {
            let raw = transcript(if index == 0 { CALLS } else { "READY_47" });
            let report = bridge()
                .parse_cpu_transcript(plan, format!("CPU-auto-member-{index}"), &raw)
                .await
                .unwrap();
            assert!(report.passed, "{:?}", report.error);
            out.push(report);
        }
        out
    }
    fn followup(request: &Value, reports: &[CpuParseReport]) -> Value {
        let mut next = request.clone();
        next["max_tokens"] = json!(32); // A distinct explicit next request, no clipping.
        next["tool_choice"] = json!("none");
        for (index, report) in reports.iter().enumerate() {
            let (choice, _) = normal_member(report, index).unwrap();
            let message = clean_message(&choice["message"]);
            let history = next["messages"][index].as_array_mut().unwrap();
            history.push(message.clone());
            if let Some(calls) = message["tool_calls"].as_array() {
                for call in calls.iter().rev() {
                    history.push(json!({"role":"tool","tool_call_id":call["id"],
                        "content":format!("result:{}",call["function"]["arguments"].as_str().unwrap())}));
                }
            }
            history.push(json!({"role":"user","content":"Now reply DONE_47."}));
        }
        next
    }
    #[tokio::test]
    async fn actual_batch_bridge_parser_and_tool_followup_keep_member_ids_and_argument_bytes() {
        let request = body();
        let reports = reports(&request).await;
        let (first, removed) = normal_member(&reports[0], 0).unwrap();
        let (second, other_removed) = normal_member(&reports[1], 1).unwrap();
        assert_eq!((removed, other_removed), (2, 0));
        assert_eq!(first["finish_reason"], "tool_calls");
        assert_eq!(second["finish_reason"], "stop");
        assert_eq!(second["index"], 1);
        assert_eq!(second["message"]["content"], "READY_47");
        let raw_calls = reports[0].response.as_ref().unwrap()["choices"][0]["message"]
            ["tool_calls"]
            .as_array()
            .unwrap();
        let calls = first["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        for (raw, normal) in raw_calls.iter().zip(calls) {
            assert!(raw.get("index").is_some());
            let mut expected = raw.clone();
            expected.as_object_mut().unwrap().remove("index");
            assert_eq!(&expected, normal);
            assert_eq!(normal["function"]["name"], "read");
        }
        assert_eq!(
            serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"label":"東京","optional":{"nested":[null,{"index":7}]}})
        );
        assert_eq!(
            serde_json::from_str::<Value>(calls[1]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"label":"café","optional":null})
        );
        let next = followup(&request, &reports);
        let group =
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &next).unwrap();
        assert_eq!(group.original_request(), &next);
        for (index, member) in group.members().iter().enumerate() {
            let mut single = next.clone();
            single["messages"] = next["messages"][index].clone();
            let official = bridge()
                .prepare_single(single, member.semantics().planning_limits)
                .unwrap();
            assert_eq!(member.prompt_ids(), official.prompt_ids());
            assert_eq!(member.semantics().output_allowance, 32);
            assert_eq!(member.semantics().tool_choice, "none");
            assert!(member.forced_named_cursor().unwrap().is_none());
        }
        assert!(group.execution_permit().is_err());
    }
    #[tokio::test]
    async fn invalid_last_tool_history_rejects_whole_group_before_any_slot_effect() {
        let request = body();
        let reports = reports(&request).await;
        let good = followup(&request, &reports);
        let book = CpuSlotBook::new(Uuid::from_u128(214), 16).unwrap();
        let snapshot = book.snapshot();
        for failure in 0..5 {
            let mut bad = good.clone();
            // Move the tool conversation last, so an earlier valid member has
            // already been rendered when the actual official converter rejects it.
            bad["messages"].as_array_mut().unwrap().swap(0, 1);
            match failure {
                0 => bad["messages"][1][2]["tool_call_id"] = json!("foreign-member-call"),
                1 => {
                    bad["messages"][1][1]["tool_calls"][1]["id"] =
                        bad["messages"][1][1]["tool_calls"][0]["id"].clone()
                }
                2 => {
                    bad["messages"][1][3]["tool_call_id"] =
                        bad["messages"][1][2]["tool_call_id"].clone()
                }
                3 => {
                    bad["messages"][1].as_array_mut().unwrap().remove(2);
                }
                _ => bad["messages"][1][1]["tool_calls"][0]["index"] = json!(0),
            }
            assert!(
                PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &bad).is_err(),
                "failure{failure}"
            );
            assert_eq!(book.snapshot(), snapshot);
        }
    }
    #[test]
    fn auto_tools_never_silently_drop_constraints_stream_or_full_budgets() {
        for (key, value) in [
            ("tool_choice", json!("required")),
            (
                "tool_choice",
                json!({"type":"function","function":{"name":"read"}}),
            ),
            ("parallel_tool_calls", json!(false)),
            ("stream", json!(true)),
            ("stop", json!("DONE")),
            ("max_tokens", json!(512)),
            ("max_tokens", json!(4096)),
        ] {
            let mut bad = body();
            bad[key] = value;
            assert!(
                PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &bad).is_err(),
                "{key}"
            );
        }
        let mut strict = body();
        strict["tools"][0]["function"]["strict"] = json!(true);
        assert!(
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &strict).is_err()
        );
    }
    #[tokio::test]
    async fn parser_eof_never_becomes_normal_member_or_tool_finish() {
        let request = body();
        let group =
            PreparedAtomicRequest::prepare(bridge(), ATOMIC_ROUTE, Some("1"), &request).unwrap();
        let plan = group.into_planned_group().into_members().remove(0);
        let mut raw = transcript(CALLS);
        raw.token_chunks.pop();
        raw.terminal = CpuTerminal::UnannouncedEof;
        let failed = bridge()
            .parse_cpu_transcript(plan, "CPU-eof".into(), &raw)
            .await
            .unwrap();
        assert!(!failed.passed && failed.response.is_none());
        assert!(normal_member(&failed, 0).is_err());
    }
}
