//! Opt-in native SSE. Tokens have checked-step origin; only an opaque final
//! completion releases finish/usage/DONE. No listener or GPU is launched here.
use super::{
    native_api_contract::{CpuBridge, PlannedChat, SERVED_MODEL},
    native_http_actor::ResidentControl,
    native_request_executor::AcceptedNativeToken,
    native_response_bridge::CheckedStreamFinish,
};
use anyhow::{ensure, Context, Result};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    fs::OpenOptions,
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

const TOKEN_CAPACITY: usize = 32;
const INFERENCE_CAPACITY: usize = 32;
const SSE_CAPACITY: usize = 8;
enum NativeInput {
    Token(AcceptedNativeToken),
    Finish(CheckedStreamFinish),
    Error(String),
}
enum Wire {
    Chunk(Value),
    Error,
    Done,
}

struct ContextBinding {
    response_id: String,
    request_sha256: String,
    prompt_ids: Vec<u32>,
    output_allowance: usize,
    include_usage: bool,
    admitted: Instant,
    deadline: Instant,
    cancelled: Arc<AtomicBool>,
    drained: Arc<AtomicBool>,
    control: ResidentControl,
    evidence: PathBuf,
}
impl ContextBinding {
    fn valid(&self) -> Result<()> {
        ensure!(
            !self.cancelled.load(Ordering::Acquire) && Instant::now() < self.deadline,
            "stream admission deadline or cancellation"
        );
        Ok(())
    }
    fn fail(&self, error: &str) {
        self.cancelled.store(true, Ordering::Release);
        self.control.block_until_coordinated_native_recovery();
        let _ = write_new(
            &self.evidence.join("stream-error.json"),
            &json!({"error":error,
            "elapsed_seconds":self.admitted.elapsed().as_secs_f64(),"normal_finish_sent":false,
            "all8_teardown_required":true,"GPU_owners_dropped_by_HTTP":false}),
        );
    }
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn log_accepted(
    file: &mut std::fs::File,
    token: &AcceptedNativeToken,
    context: &ContextBinding,
) -> Result<()> {
    serde_json::to_writer(
        &mut *file,
        &json!({"event":"actual_native_token_after_session_accept",
        "token":token,"elapsed_since_HTTP_admission_s":context.admitted.elapsed().as_secs_f64(),
        "terminal_or_quality_proof":false}),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
async fn send_async<T>(tx: &mpsc::Sender<T>, value: T, context: &ContextBinding) -> Result<()> {
    context.valid()?;
    tokio::time::timeout_at(
        tokio::time::Instant::from_std(context.deadline),
        tx.send(value),
    )
    .await
    .context("stream backpressure exceeded original admission deadline")?
    .map_err(|_| anyhow::anyhow!("stream consumer disconnected"))?;
    context.valid()
}

/// Owned by the native model thread. No API accepts a token ID or a normal
/// completion JSON; both public inputs are opaque runtime-issued Rust types.
pub struct NativeStreamPublisher {
    sender: mpsc::Sender<NativeInput>,
    context: Arc<ContextBinding>,
    final_submitted: bool,
    complete: bool,
}
impl NativeStreamPublisher {
    fn send(&mut self, mut value: NativeInput) -> Result<()> {
        loop {
            self.context.valid()?;
            match self.sender.try_send(value) {
                Ok(()) => return Ok(()),
                Err(mpsc::error::TrySendError::Full(returned)) => {
                    value = returned;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    anyhow::bail!("native stream receiver closed")
                }
            }
        }
    }
    pub fn accepted(&mut self, token: &AcceptedNativeToken) -> Result<()> {
        ensure!(!self.final_submitted, "token after stream final authority");
        self.send(NativeInput::Token(token.clone()))
    }
    pub fn finish(&mut self, finish: CheckedStreamFinish) -> Result<()> {
        ensure!(!self.final_submitted, "duplicate stream final authority");
        self.send(NativeInput::Finish(finish))?;
        self.final_submitted = true;
        // Retain the resident checked boundary until the HTTP body performs its
        // checked local DONE handoff. This is NOT remote delivery/acknowledgment.
        while !self.context.drained.load(Ordering::Acquire) {
            self.context.valid()?;
            std::thread::sleep(Duration::from_millis(1));
        }
        // Body handoff was checked against the deadline immediately before
        // publishing DONE. Do not turn that completed handoff into a later
        // timeout merely because this native thread was rescheduled afterwards.
        ensure!(
            !self.context.cancelled.load(Ordering::Acquire),
            "stream cancelled at completion handoff"
        );
        self.complete = true;
        Ok(())
    }
    pub fn finished_on_time(&self) -> bool {
        self.complete
    }
    pub fn failed(&mut self, detail: String) -> Result<()> {
        self.context.fail(&detail);
        let _ = self.sender.try_send(NativeInput::Error(detail));
        Ok(())
    }
}
impl Drop for NativeStreamPublisher {
    fn drop(&mut self) {
        if !self.complete {
            self.context
                .fail("native streaming publisher ended without checked delivery");
        }
    }
}

pub struct NativeSseBody {
    receiver: mpsc::Receiver<Wire>,
    context: Arc<ContextBinding>,
}
struct DisconnectGuard {
    context: Arc<ContextBinding>,
    armed: bool,
}
impl Drop for DisconnectGuard {
    fn drop(&mut self) {
        if self.armed {
            self.context
                .fail("HTTP stream body disconnected or ended without DONE");
        }
    }
}
impl NativeSseBody {
    pub fn into_response(self) -> Response {
        let context = self.context;
        let mut rx = self.receiver;
        let disconnect = DisconnectGuard {
            context: context.clone(),
            armed: true,
        };
        let output = async_stream::stream! {
            let mut guard=disconnect;
            loop {
                let received=tokio::time::timeout_at(tokio::time::Instant::from_std(context.deadline),rx.recv()).await;
                if context.valid().is_err() {
                    context.fail("stream body admission deadline/cancellation before emission");
                    yield Ok(Event::default().event("error").data("{\"error\":{\"type\":\"native_stream_deadline\"}}"));break;
                }
                match received {
                    Ok(Some(Wire::Chunk(value)))=>yield Ok::<Event,Infallible>(Event::default().data(value.to_string())),
                    Ok(Some(Wire::Done))=>{
                        if context.valid().is_err() {yield Ok(Event::default().event("error").data("{\"error\":{\"type\":\"native_stream_deadline\"}}"));break;}
                        // This is a checked handoff to Hyper, not evidence that
                        // a remote application received/acknowledged the bytes.
                        context.drained.store(true,Ordering::Release);guard.armed=false;
                        yield Ok(Event::default().data("[DONE]"));break;
                    }
                    _=>{
                        context.fail("stream deadline, native failure or missing normal footer");
                        yield Ok(Event::default().event("error").data("{\"error\":{\"type\":\"native_stream_failed\",\"message\":\"Partial output; no checked normal completion. Private evidence retained.\"}}"));
                        break;
                    }
                }
            }
        };
        Sse::new(output).into_response()
    }
}

/// Called after the SAME PlannedChat was prepared once. The official generator
/// and tokenizer clone configuration; they never re-render/canonicalize input.
#[allow(clippy::too_many_arguments)]
pub fn start(
    bridge: &CpuBridge,
    chat: &PlannedChat,
    response_id: String,
    admitted: Instant,
    deadline: Instant,
    cancelled: Arc<AtomicBool>,
    control: ResidentControl,
    evidence: PathBuf,
) -> Result<(NativeStreamPublisher, NativeSseBody)> {
    ensure!(
        chat.semantics().stream && chat.original_request().get("stream") == Some(&json!(true)),
        "original stream:true missing"
    );
    ensure!(
        deadline > Instant::now() && deadline <= admitted + Duration::from_secs(180),
        "stream original180s budget differs"
    );
    let include_usage = chat
        .original_request()
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let context = Arc::new(ContextBinding {
        response_id: response_id.clone(),
        request_sha256: chat.semantics().original_request_sha256.clone(),
        prompt_ids: chat.prompt_ids().to_vec(),
        output_allowance: chat.semantics().output_allowance as usize,
        include_usage,
        admitted,
        deadline,
        cancelled,
        drained: Arc::new(AtomicBool::new(false)),
        control,
        evidence,
    });
    context.valid()?;
    write_new(
        &context.evidence.join("stream-policy.json"),
        &json!({"actual_native_tokens_only":true,
        "publish_only_after_session_accept_step":true,"terminal_requires_opaque_all8_completion":true,
        "token_channel_capacity":TOKEN_CAPACITY,"inference_channel_capacity":INFERENCE_CAPACITY,"SSE_channel_capacity":SSE_CAPACITY,
        "wall_millis":180000,"native_allowance":context.output_allowance,"thinking":chat.semantics().thinking,
        "include_usage":include_usage,"buffered_response_replay":false,"liveSSE_hardware_qualified":false,
        "role_only_start_is_not_generated_TTFT":true,"local_DONE_handoff_is_not_remote_client_ack":true}),
    )?;
    let log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(context.evidence.join("accepted-native-tokens.jsonl"))?;
    let (native_tx, native_rx) = mpsc::channel(TOKEN_CAPACITY);
    let (inference_tx, inference_rx) = mpsc::channel(INFERENCE_CAPACITY);
    let (wire_tx, wire_rx) = mpsc::channel(SSE_CAPACITY);
    let (final_tx, final_rx) = oneshot::channel();
    let parser = bridge.incremental_parser(chat, response_id, ReceiverStream::new(inference_rx));
    let pump_context = context.clone();
    tokio::spawn(async move {
        let result = pump(native_rx, inference_tx, final_tx, log, pump_context.clone()).await;
        if let Err(error) = result {
            pump_context.fail(&format!("{error:#}"));
        }
    });
    let parser_context = context.clone();
    tokio::spawn(async move {
        let result =
            forward_parser(parser, wire_tx.clone(), final_rx, parser_context.clone()).await;
        if let Err(error) = result {
            parser_context.fail(&format!("{error:#}"));
            let _ = wire_tx.try_send(Wire::Error);
        }
    });
    Ok((
        NativeStreamPublisher {
            sender: native_tx,
            context: context.clone(),
            final_submitted: false,
            complete: false,
        },
        NativeSseBody {
            receiver: wire_rx,
            context,
        },
    ))
}

async fn pump(
    mut native: mpsc::Receiver<NativeInput>,
    inference: mpsc::Sender<InferenceChunk>,
    final_sender: oneshot::Sender<CheckedStreamFinish>,
    mut log: std::fs::File,
    context: Arc<ContextBinding>,
) -> Result<()> {
    send_async(
        &inference,
        InferenceChunk::Ready {
            system_fingerprint: None,
            prompt_usage: PromptUsage {
                prompt_tokens: context.prompt_ids.len(),
                prompt_cache_hit_tokens: 0,
            },
        },
        &context,
    )
    .await?;
    let mut ids = Vec::new();
    let mut plan = None;
    let mut decoder_open = true;
    loop {
        context.valid()?;
        let event = tokio::time::timeout_at(
            tokio::time::Instant::from_std(context.deadline),
            native.recv(),
        )
        .await
        .context("native token producer deadline")?
        .context("native producer EOF without completion")?;
        context.valid()?;
        match event {
            NativeInput::Token(token) => {
                ensure!(
                    token.request_sha256() == context.request_sha256
                        && token.ordinal() == ids.len() + 1
                        && ids.len() < context.output_allowance
                        && !ids.contains(&1),
                    "native streamed token request/ordinal/EOS differs"
                );
                if let Some(old) = &plan {
                    ensure!(old == token.plan_sha256(), "native streamed plan changed");
                } else {
                    plan = Some(token.plan_sha256().to_owned());
                }
                ensure!(
                    token.native_id() < 129280,
                    "native streamed ID outofvocabulary"
                );
                log_accepted(&mut log, &token, &context)?;
                ids.push(token.native_id());
                if token.native_id() != 1 && decoder_open {
                    // Parser may finish early on a matched marker. Drain actual
                    // subsequent native tokens regardless; terminal still waits
                    // for the separate opaque actual completion on this channel.
                    match tokio::time::timeout_at(
                        tokio::time::Instant::from_std(context.deadline),
                        inference.send(InferenceChunk::Tokens {
                            token_ids: vec![token.native_id()],
                            logprobs: None,
                        }),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => decoder_open = false,
                        Err(_) => anyhow::bail!("decoder backpressure exceeded admission deadline"),
                    }
                }
            }
            NativeInput::Finish(finish) => {
                ensure!(
                    finish.response_id() == context.response_id
                        && finish.request_sha256() == context.request_sha256
                        && finish.input_native_ids() == context.prompt_ids
                        && Some(finish.plan_sha256()) == plan.as_deref()
                        && finish.generated_native_ids() == ids,
                    "actual final proof differs from every published native token/request"
                );
                ensure!(
                    !ids.is_empty()
                        && if finish.eos() {
                            ids.last() == Some(&1) && ids.iter().filter(|&&x| x == 1).count() == 1
                        } else {
                            ids.len() == context.output_allowance && !ids.contains(&1)
                        },
                    "native EOS/fullcap differs"
                );
                context.valid()?;
                write_new(
                    &context.evidence.join("stream-checked-terminal.json"),
                    &serde_json::to_value(&finish)?,
                )?;
                if decoder_open {
                    let reason = if finish.eos() {
                        InferenceFinishReason::Stop
                    } else {
                        InferenceFinishReason::Length
                    };
                    let result = tokio::time::timeout_at(
                        tokio::time::Instant::from_std(context.deadline),
                        inference.send(InferenceChunk::Finish {
                            finish_reason: reason,
                        }),
                    )
                    .await;
                    ensure!(result.is_ok(), "finish decoder backpressure deadline");
                }
                context.valid()?;
                final_sender.send(finish).map_err(|_| {
                    anyhow::anyhow!("official parser stopped without final consumer")
                })?;
                return Ok(());
            }
            NativeInput::Error(detail) => anyhow::bail!("native stream failed: {detail}"),
        }
    }
}
async fn forward_parser<S>(
    parser: S,
    wire: mpsc::Sender<Wire>,
    final_receiver: oneshot::Receiver<CheckedStreamFinish>,
    context: Arc<ContextBinding>,
) -> Result<()>
where
    S: tokio_stream::Stream<
            Item = std::result::Result<
                deepseek_recipe::openai::chat_completion::response::ChatCompletionChunk,
                deepseek_recipe::stream::StreamError,
            >,
        > + Send,
{
    let mut parser = Box::pin(parser);
    let mut footer = None;
    while let Some(chunk) = tokio::time::timeout_at(
        tokio::time::Instant::from_std(context.deadline),
        parser.next(),
    )
    .await
    .context("official incremental parser deadline")?
    {
        let mut value = serde_json::to_value(chunk?)?;
        ensure!(
            value.get("id").and_then(Value::as_str) == Some(context.response_id.as_str())
                && value.get("model").and_then(Value::as_str) == Some(SERVED_MODEL),
            "official stream response identity differs"
        );
        let terminal = value["choices"]
            .as_array()
            .context("official stream choices")?
            .iter()
            .any(|c| c.get("finish_reason").is_some_and(|v| !v.is_null()));
        if terminal {
            ensure!(footer.is_none(), "duplicate official finish");
            footer = Some(value);
        } else {
            ensure!(
                value.get("usage").is_none_or(Value::is_null),
                "premature nonnull usage"
            );
            if !context.include_usage {
                value
                    .as_object_mut()
                    .context("chunk object")?
                    .remove("usage");
            }
            send_async(&wire, Wire::Chunk(value), &context).await?;
        }
    }
    drop(parser); // Close inference receiver even when official parser stops early.
                  // Even a parser-matched stop cannot authorize protocol success. Native
                  // producer/opaque all8 proof continues independently after parser closure.
    let finish = tokio::time::timeout_at(
        tokio::time::Instant::from_std(context.deadline),
        final_receiver,
    )
    .await
    .context("all8 stream completion deadline")?
    .context("no opaque all8 stream completion")?;
    context.valid()?;
    let mut footer = footer.context("official parser missing explicit finish")?;
    let choices = footer["choices"]
        .as_array()
        .context("official finish choices")?;
    ensure!(
        choices.len() == 1
            && if finish.eos() {
                matches!(
                    choices[0]["finish_reason"].as_str(),
                    Some("stop" | "tool_calls")
                )
            } else {
                choices[0]["finish_reason"].as_str() == Some("length")
            },
        "official stop/length differs from actual native terminal"
    );
    let parser_usage = footer.get("usage").cloned();
    if context.include_usage {
        let usage = footer
            .get_mut("usage")
            .and_then(Value::as_object_mut)
            .context("official finish usage missing")?;
        usage.insert(
            "prompt_tokens".into(),
            json!(finish.input_native_ids().len()),
        );
        usage.insert(
            "completion_tokens".into(),
            json!(finish.generated_native_ids().len()),
        );
        usage.insert(
            "total_tokens".into(),
            json!(finish.input_native_ids().len() + finish.generated_native_ids().len()),
        );
    } else {
        footer
            .as_object_mut()
            .context("official footer object")?
            .remove("usage");
    }
    write_new(
        &context.evidence.join("stream-footer.json"),
        &json!({"footer":footer,"official_parser_usage":parser_usage,
        "native_generated_tokens":finish.generated_native_ids().len(),"native_prompt_tokens":finish.input_native_ids().len(),
        "reasoning_usage_scope":"official parser/tokenizer re-encoding, not a per-native-ID classification",
        "opaque_all8_terminal_verified":true,"model_or_task_quality_qualified":false}),
    )?;
    send_async(&wire, Wire::Chunk(footer), &context).await?;
    send_async(&wire, Wire::Done, &context).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        native_api_contract::PlanningLimits, native_request_executor::accepted_token_CPU_fixture,
        native_response_bridge::stream_finish_CPU_fixture,
    };
    use super::*;
    use std::sync::OnceLock;
    use uuid::Uuid;
    fn bridge() -> Arc<CpuBridge> {
        static B: OnceLock<Arc<CpuBridge>> = OnceLock::new();
        B.get_or_init(||Arc::new(CpuBridge::from_verified_tokenizer(&std::fs::read(
            "test-data/tokenizer.json").unwrap()).unwrap())).clone()
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("native-stream-{}", Uuid::new_v4()));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn request(max: usize, thinking: bool) -> Value {
        json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":"CPU streaming fixture"}],"temperature":0,"top_p":1,"max_tokens":max,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"thinking":thinking}})
    }
    fn plan(body: Value) -> PlannedChat {
        bridge()
            .prepare_stream(
                body,
                PlanningLimits {
                    context_tokens: 8192,
                    prompt_tokens: 8192,
                    output_tokens: 4096,
                    atomic_requests: 1,
                    request_bytes: 1024 * 1024,
                },
            )
            .unwrap()
    }
    fn native(text: &str) -> Vec<u32> {
        let t=tokenizers_recipe::Tokenizer::from_bytes(&std::fs::read("test-data/tokenizer.json").unwrap()).unwrap();
        t.encode(text, false).unwrap().get_ids().to_vec()
    }
    fn context_control() -> ResidentControl {
        super::super::native_http_actor::CPU_fixture_control_ready()
    }
    fn chunks(raw: &str) -> Vec<Value> {
        raw.lines()
            .filter_map(|s| s.strip_prefix("data: "))
            .filter(|s| *s != "[DONE]")
            .filter_map(|s| serde_json::from_str(s).ok())
            .collect()
    }
    async fn collect(body: axum::body::Body) -> String {
        let mut stream = body.into_data_stream();
        let mut data = Vec::new();
        while let Some(c) = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .unwrap()
        {
            data.extend_from_slice(&c.unwrap());
        }
        String::from_utf8(data).unwrap()
    }
    #[tokio::test]
    async fn original_stream_true_full_budget_and_prompt_are_preserved() {
        let body = request(4096, false);
        let p = plan(body.clone());
        assert_eq!(p.original_request(), &body);
        assert!(p.semantics().stream);
        assert_eq!(p.semantics().output_allowance, 4096);
        assert!(bridge()
            .prepare_single(
                body,
                PlanningLimits {
                    context_tokens: 8192,
                    prompt_tokens: 8192,
                    output_tokens: 4096,
                    atomic_requests: 1,
                    request_bytes: 1024 * 1024
                }
            )
            .is_err());
        let mut bad = request(32, false);
        bad["stream_options"]["obfuscation"] = json!(true);
        assert!(bridge()
            .prepare_stream(
                bad,
                PlanningLimits {
                    context_tokens: 8192,
                    prompt_tokens: 8192,
                    output_tokens: 4096,
                    atomic_requests: 1,
                    request_bytes: 1024 * 1024
                }
            )
            .is_err());
    }
    #[tokio::test]
    async fn visible_delta_arrives_before_final_authority_but_footer_waits() {
        let temp = Temp::new();
        let p = plan(request(512, false));
        let id = "chatcmpl-early";
        let sha = p.semantics().original_request_sha256.clone();
        let prompt = p.prompt_ids().to_vec();
        let control = context_control();
        let cancel = Arc::new(AtomicBool::new(false));
        let admitted = Instant::now();
        let (mut pub_, body) = start(
            &bridge(),
            &p,
            id.into(),
            admitted,
            admitted + Duration::from_secs(10),
            cancel,
            control,
            temp.0.clone(),
        )
        .unwrap();
        let final_allowed = Arc::new(AtomicBool::new(false));
        let release = final_allowed.clone();
        let mut ids = native("EARLY_47 visible ");
        ids.push(1);
        let expected = ids.len();
        let producer = tokio::task::spawn_blocking(move || -> Result<()> {
            for (i, &token) in ids.iter().enumerate() {
                pub_.accepted(&accepted_token_CPU_fixture(
                    &sha,
                    &"a".repeat(64),
                    i + 1,
                    token,
                ))?;
            }
            let fixture_deadline = Instant::now() + Duration::from_secs(5);
            while !release.load(Ordering::Acquire) {
                ensure!(
                    Instant::now() < fixture_deadline,
                    "CPU fixture release deadline"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            pub_.finish(stream_finish_CPU_fixture(
                id,
                &sha,
                &"a".repeat(64),
                prompt,
                ids,
                true,
            ))
        });
        let mut stream = body.into_response().into_body().into_data_stream();
        let mut raw = String::new();
        loop {
            let next = tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            raw.push_str(std::str::from_utf8(&next).unwrap());
            let visible: String = chunks(&raw)
                .iter()
                .flat_map(|v| v["choices"].as_array().into_iter().flatten())
                .filter_map(|c| c["delta"]["content"].as_str())
                .collect();
            if visible.contains("EARLY_47") {
                break;
            }
        }
        assert!(!raw.contains("[DONE]"));
        assert!(chunks(&raw).iter().all(|v| v["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["finish_reason"].is_null())));
        final_allowed.store(true, Ordering::Release);
        while let Some(next) = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .unwrap()
        {
            raw.push_str(std::str::from_utf8(&next.unwrap()).unwrap());
        }
        producer.await.unwrap().unwrap();
        assert!(raw.contains("[DONE]"));
        let values = chunks(&raw);
        let last = values.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["usage"]["completion_tokens"], expected);
    }
    #[tokio::test]
    async fn reasoning_unicode_nested_null_and_repeated_tool_ids_stream_incrementally() {
        let temp = Temp::new();
        let mut body = request(512, true);
        body["tools"] = json!([{"type":"function","function":{"name":"echo","description":"fixture","parameters":{"type":"object","properties":{"value":{}},"required":["value"]}}}]);
        body["tool_choice"] = json!("auto");
        let p = plan(body);
        let id = "chatcmpl-tools";
        let sha = p.semantics().original_request_sha256.clone();
        let prompt = p.prompt_ids().to_vec();
        let control = context_control();
        let began = Instant::now();
        let (mut publisher, output) = start(
            &bridge(),
            &p,
            id.into(),
            began,
            began + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
            control,
            temp.0.clone(),
        )
        .unwrap();
        let text="Reason_47</think><｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"value\" string=\"false\">{\"text\":\"café 東京 🦊\",\"optional\":null}</｜DSML｜ parameter></｜DSML｜ invoke><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"value\" string=\"false\">[1,null,{\"ok\":true}]</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut ids = native(text);
        ids.push(1);
        let job = tokio::task::spawn_blocking(move || -> Result<()> {
            for (i, &t) in ids.iter().enumerate() {
                publisher.accepted(&accepted_token_CPU_fixture(&sha, &"b".repeat(64), i + 1, t))?;
            }
            publisher.finish(stream_finish_CPU_fixture(
                id,
                &sha,
                &"b".repeat(64),
                prompt,
                ids,
                true,
            ))
        });
        let raw = collect(output.into_response().into_body()).await;
        job.await.unwrap().unwrap();
        let values = chunks(&raw);
        let mut reasoning = String::new();
        let mut arguments = [String::new(), String::new()];
        let mut names = [String::new(), String::new()];
        let mut callids = [String::new(), String::new()];
        let mut argument_events = 0;
        for v in &values {
            for c in v["choices"].as_array().unwrap() {
                let d = &c["delta"];
                if let Some(s) = d["reasoning_content"].as_str() {
                    reasoning.push_str(s);
                }
                if let Some(calls) = d["tool_calls"].as_array() {
                    for call in calls {
                        let i = call["index"].as_u64().unwrap() as usize;
                        if let Some(s) = call["id"].as_str() {
                            callids[i].push_str(s);
                        }
                        if let Some(s) = call["function"]["name"].as_str() {
                            names[i].push_str(s);
                        }
                        if let Some(s) = call["function"]["arguments"].as_str() {
                            arguments[i].push_str(s);
                            argument_events += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(reasoning, "Reason_47");
        assert_eq!(names, ["echo", "echo"]);
        assert!(!callids[0].is_empty() && !callids[1].is_empty() && callids[0] != callids[1]);
        assert!(argument_events > 2);
        assert_eq!(
            serde_json::from_str::<Value>(&arguments[0]).unwrap(),
            json!({"value":{"text":"café 東京 🦊","optional":null}})
        );
        assert_eq!(
            serde_json::from_str::<Value>(&arguments[1]).unwrap(),
            json!({"value":[1,null,{"ok":true}]})
        );
        assert_eq!(
            values.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert!(raw.contains("[DONE]"));
    }
    #[tokio::test]
    async fn native_full_cap_footer_is_length_and_exact_native_count() {
        let temp = Temp::new();
        let p = plan(request(8, false));
        let id = "chatcmpl-cap";
        let sha = p.semantics().original_request_sha256.clone();
        let prompt = p.prompt_ids().to_vec();
        let began = Instant::now();
        let (mut publisher, output) = start(
            &bridge(),
            &p,
            id.into(),
            began,
            began + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
            context_control(),
            temp.0.clone(),
        )
        .unwrap();
        let ids = native("1\n2\n3\n4\n5\n6\n7\n8\n")[..8].to_vec();
        let job = tokio::task::spawn_blocking(move || -> Result<()> {
            for (i, &t) in ids.iter().enumerate() {
                publisher.accepted(&accepted_token_CPU_fixture(&sha, &"c".repeat(64), i + 1, t))?;
            }
            publisher.finish(stream_finish_CPU_fixture(
                id,
                &sha,
                &"c".repeat(64),
                prompt,
                ids,
                false,
            ))
        });
        let raw = collect(output.into_response().into_body()).await;
        job.await.unwrap().unwrap();
        let values = chunks(&raw);
        assert_eq!(
            values.last().unwrap()["choices"][0]["finish_reason"],
            "length"
        );
        assert_eq!(values.last().unwrap()["usage"]["completion_tokens"], 8);
        assert!(raw.contains("[DONE]"));
    }
    #[tokio::test]
    async fn no_opaque_terminal_or_wrong_prefix_never_emits_finish_or_done() {
        for wrong in [false, true] {
            let temp = Temp::new();
            let p = plan(request(512, false));
            let id = "chatcmpl-failure";
            let sha = p.semantics().original_request_sha256.clone();
            let prompt = p.prompt_ids().to_vec();
            let began = Instant::now();
            let control = context_control();
            let (mut publisher, body) = start(
                &bridge(),
                &p,
                id.into(),
                began,
                began + Duration::from_secs(5),
                Arc::new(AtomicBool::new(false)),
                control.clone(),
                temp.0.clone(),
            )
            .unwrap();
            let mut ids = native("partial text ");
            ids.push(1);
            let job = tokio::task::spawn_blocking(move || {
                for (i, &t) in ids.iter().enumerate() {
                    publisher
                        .accepted(&accepted_token_CPU_fixture(&sha, &"d".repeat(64), i + 1, t))
                        .unwrap();
                }
                if wrong {
                    let mut bad = ids;
                    bad[0] += 1;
                    assert!(publisher
                        .finish(stream_finish_CPU_fixture(
                            id,
                            &sha,
                            &"d".repeat(64),
                            prompt,
                            bad,
                            true
                        ))
                        .is_err());
                }
            });
            let raw = collect(body.into_response().into_body()).await;
            job.await.unwrap();
            assert!(!raw.contains("[DONE]"));
            assert!(chunks(&raw).iter().all(|v| v["choices"]
                .as_array()
                .is_none_or(|cs| cs.iter().all(|c| c["finish_reason"].is_null()))));
            assert!(!control.ready());
        }
    }
    #[tokio::test]
    async fn unpolled_disconnect_and_bounded_backpressure_poison_readiness() {
        let temp = Temp::new();
        let p = plan(request(4096, false));
        let began = Instant::now();
        let control = context_control();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (publisher, body) = start(
            &bridge(),
            &p,
            "drop".into(),
            began,
            began + Duration::from_secs(5),
            cancelled.clone(),
            control.clone(),
            temp.0.clone(),
        )
        .unwrap();
        let response = body.into_response();
        drop(response);
        assert!(cancelled.load(Ordering::Acquire));
        assert!(!control.ready());
        drop(publisher);
        let temp = Temp::new();
        let began = Instant::now();
        let control = context_control();
        let sha = p.semantics().original_request_sha256.clone();
        let (mut publisher, body) = start(
            &bridge(),
            &p,
            "blocked".into(),
            began,
            began + Duration::from_millis(150),
            Arc::new(AtomicBool::new(false)),
            control.clone(),
            temp.0.clone(),
        )
        .unwrap();
        let response = body.into_response();
        let job = tokio::task::spawn_blocking(move || {
            for n in 1..4097 {
                if publisher
                    .accepted(&accepted_token_CPU_fixture(&sha, &"e".repeat(64), n, 2581))
                    .is_err()
                {
                    return;
                }
            }
            panic!("unbounded consumer accepted full native window")
        });
        tokio::time::timeout(Duration::from_secs(2), job)
            .await
            .unwrap()
            .unwrap();
        assert!(!control.ready());
        drop(response);
    }
    #[tokio::test]
    async fn queued_footer_polled_after_deadline_never_becomes_normal_finish() {
        let temp = Temp::new();
        let p = plan(request(512, false));
        let id = "chatcmpl-late-footer";
        let sha = p.semantics().original_request_sha256.clone();
        let prompt = p.prompt_ids().to_vec();
        let mut ids = native("OK");
        ids.push(1);
        let began = Instant::now();
        let control = context_control();
        let (mut publisher, body) = start(
            &bridge(),
            &p,
            id.into(),
            began,
            began + Duration::from_secs(2),
            Arc::new(AtomicBool::new(false)),
            control.clone(),
            temp.0.clone(),
        )
        .unwrap();
        let response = body.into_response();
        let job = tokio::task::spawn_blocking(move || {
            for (i, &t) in ids.iter().enumerate() {
                publisher
                    .accepted(&accepted_token_CPU_fixture(&sha, &"f".repeat(64), i + 1, t))
                    .unwrap();
            }
            assert!(publisher
                .finish(stream_finish_CPU_fixture(
                    id,
                    &sha,
                    &"f".repeat(64),
                    prompt,
                    ids,
                    true
                ))
                .is_err());
        });
        for _ in 0..500 {
            if temp.0.join("stream-footer.json").exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(
            temp.0.join("stream-footer.json").exists(),
            "CPU fixture never prepared queued footer"
        );
        tokio::time::sleep_until(tokio::time::Instant::from_std(
            began + Duration::from_millis(2050),
        ))
        .await;
        let raw = collect(response.into_body()).await;
        job.await.unwrap();
        assert!(!raw.contains("[DONE]"));
        assert!(!control.ready());
        assert!(chunks(&raw).iter().all(|v| v["choices"]
            .as_array()
            .is_none_or(|cs| cs.iter().all(|c| c["finish_reason"].is_null()))));
    }
    #[tokio::test]
    async fn forced_named_stream_keeps_original_body_prompt_and_mask_binding() {
        let bodies: Vec<Value> = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_forced_requests.json"
        ))
        .unwrap();
        let original = bodies[1].clone();
        let baseline = bridge()
            .prepare_single(
                original.clone(),
                PlanningLimits {
                    context_tokens: 8192,
                    prompt_tokens: 8192,
                    output_tokens: 4096,
                    atomic_requests: 1,
                    request_bytes: 1024 * 1024,
                },
            )
            .unwrap();
        let mut streamed = original;
        streamed["stream"] = json!(true);
        streamed["stream_options"] = json!({"include_usage":true});
        let chat = plan(streamed.clone());
        assert_eq!(chat.original_request(), &streamed);
        assert!(chat.semantics().stream);
        assert_eq!(chat.prompt_ids(), baseline.prompt_ids());
        assert_eq!(chat.semantics().output_allowance, 512);
        let expected = chat.semantics().forced_tool_constraint.as_ref().unwrap();
        assert_eq!(expected["name"], "echo");
        let cursor = chat.forced_named_cursor().unwrap().unwrap();
        let mask = cursor.allowed().unwrap();
        mask.check_binding(
            &chat.semantics().original_request_sha256,
            chat.prompt_ids(),
            &[],
        )
        .unwrap();
        assert!(mask.identity().allowed_native_tokens > 0);
        assert_eq!(mask.identity().named_tool, "echo");
    }
}
