//! PRIVATE CPU draft. No HTTP listener, inference, tool execution or permit to
//! execute exists here. The numeric limits below are supplied planning inputs,
//! never a statement that a native engine has qualified those capabilities.
//!
//! Intended parent integration: `crate::dsv41::protocol` remains the sole chat
//! normalizer/official renderer. This file adds strict request-field checks,
//! exact full-budget admission and auditable native-ID response accounting.

use crate::dsv41::protocol::{self, PreparedChat};
use crate::dsv41::tool_constraint::{ConstraintVocabulary, ForcedNamedCursor, NamedToolSpec};
use anyhow::{bail, ensure, Context, Result};
use deepseek_recipe::openai::chat_completion::response::{
    ChatCompletionChunk, ChatCompletionChunkGenerator, ChatCompletionResponse,
};
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{
    ChunkGenerator, FinishReason, InferenceChunk, InferenceFinishReason, OutputChunk, PromptUsage,
    StreamProcessor,
};
use deepseek_recipe_core::conversation::ReasoningEffort;
use deepseek_recipe_encoding::PromptTokenizer;
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use tokio_stream::StreamExt;

pub const SERVED_MODEL: &str = "deepseek-v4.1-flash";
pub const TOKENIZER_SHA256: &str =
    "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
pub const CHAT_ROUTE: &str = "/v1/chat/completions";
pub const ATOMIC_ROUTE: &str = "/v1/chat/completions/batch";
pub const ATOMIC_HEADER: &str = "X-TP8-Atomic-Admission";
pub const EOS_ID: u32 = 1;

pub fn capabilities() -> Value {
    json!({"cpu_request_planning":true,"cpu_native_transcript_parsing":true,
        "model_serving":false,"network_listener":false,"SSE_transport":false,
        "C1_backend_qualified":false,"context8192_qualified":false,
        "atomic_backend_admission_qualified":false,"GPU_batching_qualified":false,
        "prefix_cache_qualified":false,"tool_execution":false,
        "runtime_terminal_authority_integrated":false,
        "qualification":"private CPU contract only; supplied limits and transcript terminals are unqualified planning/test data"})
}

/// An explicit proposal supplied by the CPU caller. Cannot enable execution.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlanningLimits {
    pub context_tokens: u32,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub atomic_requests: usize,
    pub request_bytes: usize,
}

impl PlanningLimits {
    fn validate(self) -> Result<()> {
        ensure!(
            self.context_tokens > 0
                && self.prompt_tokens > 0
                && self.output_tokens > 0
                && self.request_bytes > 0,
            "planning token/byte limits must be positive"
        );
        ensure!(
            (1..=16).contains(&self.atomic_requests),
            "only bounded logical groups of1..16 are planned"
        );
        ensure!(
            self.prompt_tokens <= self.context_tokens && self.output_tokens <= self.context_tokens,
            "planning per-side limits exceed proposed context"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct EffectiveSemantics {
    pub original_request_sha256: String,
    pub model_revision: &'static str,
    pub tokenizer_sha256: &'static str,
    pub toolkit_revision: &'static str,
    pub thinking: bool,
    pub requested_reasoning_effort: Option<Value>,
    pub effective_reasoning_score: Option<u8>,
    pub requested_temperature: Option<f32>,
    pub effective_temperature: f32,
    pub requested_top_p: Option<f32>,
    pub effective_top_p: f32,
    pub seed: Option<u64>,
    pub seed_policy: &'static str,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub requested_max_tokens: Option<u32>,
    pub output_allowance: u32,
    pub budget_origin: &'static str,
    pub prompt_tokens: u32,
    pub prompt_ids_sha256_u32le: String,
    pub required_context_tokens: u32,
    pub planning_limits: PlanningLimits,
    pub stop_sequences: Vec<String>,
    pub tool_choice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forced_tool_constraint: Option<Value>,
    pub parallel_tool_calls: &'static str,
    pub cache_namespace: Option<String>,
    pub cache_policy: &'static str,
    pub stream: bool,
    pub min_tokens: u32,
    pub ignore_eos: bool,
    pub return_token_ids: bool,
}

/// Fields remain private so a plan cannot be changed after encoding.
#[derive(Debug)]
pub struct PlannedChat {
    original_request: Value,
    prepared: PreparedChat,
    prompt_ids: Vec<u32>,
    semantics: EffectiveSemantics,
    forced: Option<(Arc<ConstraintVocabulary>, Arc<NamedToolSpec>)>,
}

impl PlannedChat {
    pub fn original_request(&self) -> &Value {
        &self.original_request
    }
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    pub fn semantics(&self) -> &EffectiveSemantics {
        &self.semantics
    }
    pub fn forced_named_cursor(&self) -> Result<Option<ForcedNamedCursor>> {
        self.forced
            .as_ref()
            .map(|(v, s)| {
                ForcedNamedCursor::new(
                    v.clone(),
                    s.clone(),
                    self.semantics.original_request_sha256.clone(),
                    &self.prompt_ids,
                )
            })
            .transpose()
    }
    pub fn execution_permit(&self) -> Result<()> {
        bail!("CPU contract has no qualified backend or runtime terminal authority; execution refused")
    }
}

/// All members validate before this value exists. Returning this plan neither
/// admits anything nor proves that a backend will schedule a common GPU batch.
#[derive(Debug)]
pub struct PlannedAtomicGroup {
    original_request: Value,
    members: Vec<PlannedChat>,
}

impl PlannedAtomicGroup {
    pub fn original_request(&self) -> &Value {
        &self.original_request
    }
    pub fn members(&self) -> &[PlannedChat] {
        &self.members
    }
    /// CPU data transfer, never an execution or terminal permit.
    pub fn into_members(self) -> Vec<PlannedChat> {
        self.members
    }
    pub fn execution_permit(&self) -> Result<()> {
        bail!("atomic registration/scheduler/backend unqualified; execution refused")
    }
}

/// Both prompt encoding and output decoding are tied to the same byte-verified
/// official tokenizer. No file reads, fallback tokenizer or downloads occur.
pub struct CpuBridge {
    tokenizer: tokenizers_recipe::Tokenizer,
    prompt_tokenizer: PromptTokenizer,
    constraint_vocabulary: Arc<ConstraintVocabulary>,
}

impl CpuBridge {
    pub fn from_verified_tokenizer(bytes: &[u8]) -> Result<Self> {
        ensure!(
            sha(bytes) == TOKENIZER_SHA256,
            "unverified1017 tokenizer bytes"
        );
        let tokenizer =
            tokenizers_recipe::Tokenizer::from_bytes(bytes).map_err(anyhow::Error::msg)?;
        ensure!(
            tokenizer.get_padding().is_none() && tokenizer.get_truncation().is_none(),
            "tokenizer padding/truncation would change native prompt IDs"
        );
        let prompt_tokenizer = PromptTokenizer::new(&tokenizer);
        static CONSTRAINT_VOCABULARY: std::sync::OnceLock<Arc<ConstraintVocabulary>> =
            std::sync::OnceLock::new();
        let constraint_vocabulary = if let Some(v) = CONSTRAINT_VOCABULARY.get() {
            v.clone()
        } else {
            let v = ConstraintVocabulary::from_verified_json(bytes, TOKENIZER_SHA256)?;
            let _ = CONSTRAINT_VOCABULARY.set(v.clone());
            v
        };
        Ok(Self {
            tokenizer,
            prompt_tokenizer,
            constraint_vocabulary,
        })
    }

    pub fn prepare_single(&self, body: Value, limits: PlanningLimits) -> Result<PlannedChat> {
        self.prepare_transport(body, limits, false)
    }

    /// Additive native streaming preparation. Original stream:true/usage/body
    /// are retained in request identity; default buffered preparation stays closed.
    pub fn prepare_stream(&self, body: Value, limits: PlanningLimits) -> Result<PlannedChat> {
        self.prepare_transport(body, limits, true)
    }

    fn prepare_transport(
        &self,
        body: Value,
        limits: PlanningLimits,
        streaming: bool,
    ) -> Result<PlannedChat> {
        limits.validate()?;
        ensure!(
            serde_json::to_vec(&body)?.len() <= limits.request_bytes,
            "request exceeds explicit CPU byte bound"
        );
        let mut normalized = body.clone();
        validate_single_fields_transport(&normalized, streaming)?;
        let object = normalized.as_object_mut().unwrap();
        let return_ids = optional_bool(object, "return_token_ids")?.unwrap_or(false);
        let cache_namespace = optional_string(object, "cache_salt")?;
        for name in ["return_token_ids", "cache_salt", "min_tokens", "ignore_eos"] {
            object.remove(name);
        }
        if let Some(alias) = object.remove("max_completion_tokens") {
            if let Some(original) = object.get("max_tokens") {
                ensure!(
                    original == &alias,
                    "max_tokens and max_completion_tokens conflict; neither wins silently"
                );
            }
            object.insert("max_tokens".into(), alias);
        }
        let prepared = protocol::prepare_chat(normalized)?;
        let options = &prepared.converted.inference_options;
        ensure!(prepared.sampling.effective_temperature == Some(0.0),
            "native backend supports greedy temperature0 only; requested/default sampling is unsupported");
        ensure!(
            prepared.sampling.effective_top_p == Some(1.0),
            "native backend supports top_p1 only"
        );
        ensure!(
            !options.logprobs && options.top_logprobs.is_none(),
            "native token logprobs are not implemented"
        );
        // The toolkit itself performs Named→single selected definition plus
        // Required, required-prefix rendering and parser initial-state changes.
        // Assert that exact conversion, then retain a real sampler constraint.
        let forced = if let Some(name) = named_choice(&body)? {
            use deepseek_recipe::request::OutputConstraint;
            use deepseek_recipe_core::tools::ToolChoice;
            ensure!(
                !prepared.converted.conversation.thinking_mode
                    && prepared.converted.conversation.tool_choice == ToolChoice::Required
                    && prepared.converted.parsing_options.starts_in_tool_calls
                    && prepared.rendered.prompt.ends_with("\n\n<｜DSML｜ calls>\n"),
                "pinned official forced-tool prompt/parser semantics changed"
            );
            let tools = match &prepared.converted.inference_options.output_constraint {
                Some(OutputConstraint::ToolCalls {
                    tools,
                    required: true,
                }) => tools,
                _ => bail!("named tool lacks official Required output constraint"),
            };
            ensure!(
                tools.len() == 1 && tools[0].name == name,
                "official named constraint did not retain exactly the selected function"
            );
            let spec = NamedToolSpec::new(name, tools[0].parameters.clone())?;
            Some((self.constraint_vocabulary.clone(), spec))
        } else {
            None
        };
        let thinking = prepared.converted.conversation.thinking_mode;
        let score = if thinking {
            Some(match prepared.converted.conversation.reasoning_effort {
                Some(ReasoningEffort::Low) => 50,
                Some(ReasoningEffort::Max) => 100,
                Some(ReasoningEffort::Score(score)) => score.get(),
                None | Some(ReasoningEffort::High | ReasoningEffort::Xhigh) => 75,
            })
        } else {
            None
        };
        let requested = options.max_tokens;
        let allowance = requested.unwrap_or(if !thinking {
            8192
        } else if score == Some(100) {
            131072
        } else {
            65536
        });
        // Do not call PreparedChat::resolve_max_tokens: unspecified defaults
        // there are clipped to remaining context. This contract never clips.
        let prompt_ids = prepared.encode_ids(&self.prompt_tokenizer)?;
        let prompt_tokens =
            u32::try_from(prompt_ids.len()).context("native prompt token count exceeds u32")?;
        let required = prompt_tokens
            .checked_add(allowance)
            .context("prompt+full output allowance overflow")?;
        ensure!(
            prompt_tokens <= limits.prompt_tokens,
            "native prompt{prompt_tokens} exceeds proposed prompt limit{}; truncation refused",
            limits.prompt_tokens
        );
        ensure!(
            allowance <= limits.output_tokens,
            "full output allowance{allowance} exceeds proposed output limit{}; clipping refused",
            limits.output_tokens
        );
        ensure!(required <= limits.context_tokens,
            "native prompt{prompt_tokens}+full allowance{allowance} requires{required} tokens, proposed context{}; clipping refused",
            limits.context_tokens);
        let raw = body.as_object().unwrap();
        let effort = raw.get("reasoning_effort").cloned().or_else(|| {
            raw.get("chat_template_kwargs")?
                .get("reasoning_effort")
                .cloned()
        });
        let semantics = EffectiveSemantics {
            original_request_sha256: sha(&serde_json::to_vec(&body)?),
            model_revision: protocol::MODEL_REVISION,
            tokenizer_sha256: TOKENIZER_SHA256,
            toolkit_revision: protocol::TOOLKIT_REVISION,
            thinking,
            requested_reasoning_effort: effort,
            effective_reasoning_score: score,
            requested_temperature: prepared.sampling.requested_temperature,
            effective_temperature: 0.0,
            requested_top_p: prepared.sampling.requested_top_p,
            effective_top_p: 1.0,
            seed: options.seed,
            seed_policy: "preserved as metadata; greedy path draws no random samples",
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            requested_max_tokens: requested,
            output_allowance: allowance,
            budget_origin: if requested.is_some() {
                "explicit request"
            } else {
                "full official thinking-mode default; never context-clipped"
            },
            prompt_tokens,
            prompt_ids_sha256_u32le: ids_sha(&prompt_ids),
            required_context_tokens: required,
            planning_limits: limits,
            stop_sequences: prepared.converted.parsing_options.stop_sequences.clone(),
            tool_choice: raw.get("tool_choice").filter(|v|!v.is_null())
                .map(|v|v.as_str().map(str::to_owned).unwrap_or_else(||v.to_string())).unwrap_or_else(||"auto".into()),
            forced_tool_constraint: forced.as_ref().map(|(_,s)|json!({"policy":crate::dsv41::tool_constraint::POLICY,
                "name":s.name(),"schema_sha256":s.sha256(),"native_token_mask_before_greedy":true,
                "ordinary_bytes_cannot_impersonate_native_DSML":true,"actual_name_JSON_schema_and_finish_checked":true,
                "qualified":false})),
            parallel_tool_calls:
                "multiple calls permitted; model selects; fixture execution is outside this module",
            cache_namespace,
            cache_policy: "namespace preserved; prefix/cache reuse unqualified and no hits claimed",
            stream: streaming,
            min_tokens: 0,
            ignore_eos: false,
            return_token_ids: return_ids,
        };
        Ok(PlannedChat {
            original_request: body,
            prepared,
            prompt_ids,
            semantics,
            forced,
        })
    }

    pub(crate) fn incremental_parser(
        &self,
        chat: &PlannedChat,
        response_id: String,
        inference: impl tokio_stream::Stream<Item = InferenceChunk> + Send + 'static,
    ) -> impl tokio_stream::Stream<
        Item = std::result::Result<
            deepseek_recipe::openai::chat_completion::response::ChatCompletionChunk,
            deepseek_recipe::stream::StreamError,
        >,
    > + Send
           + 'static {
        let generator = ChatCompletionRequest::chunk_generator(
            &chat.prepared.converted,
            response_id,
            SERVED_MODEL.into(),
        );
        StreamProcessor::new(generator, chat.prepared.converted.parsing_options.clone())
            .with_strict_finish(true)
            .with_tokenizer(self.tokenizer.clone())
            .into_protocol_stream(inference)
    }

    /// Exact historic envelope: common fields plus nested conversations in
    /// `messages`, accompanied by the explicit atomic route/header contract.
    pub fn prepare_atomic(
        &self,
        path: &str,
        atomic_header: Option<&str>,
        body: Value,
        limits: PlanningLimits,
    ) -> Result<PlannedAtomicGroup> {
        limits.validate()?;
        ensure!(
            path == ATOMIC_ROUTE && atomic_header == Some("1"),
            "nested batch requires exact atomic route and X-TP8-Atomic-Admission:1"
        );
        ensure!(
            serde_json::to_vec(&body)?.len() <= limits.request_bytes,
            "atomic request exceeds explicit CPU byte bound"
        );
        let object = body.as_object().context("atomic request must be object")?;
        let conversations = object
            .get("messages")
            .and_then(Value::as_array)
            .context("atomic messages must be a nested conversation array")?;
        ensure!(
            !conversations.is_empty() && conversations.len() <= limits.atomic_requests,
            "atomic group outside supplied logical capacity; group not admitted"
        );
        let mut members = Vec::with_capacity(conversations.len());
        for (index, messages) in conversations.iter().enumerate() {
            ensure!(
                messages.is_array(),
                "atomic messages[{index}] must be a conversation array"
            );
            let mut member = object.clone();
            member.insert("messages".into(), messages.clone());
            members.push(
                self.prepare_single(Value::Object(member), limits)
                    .with_context(|| format!("atomic member{index} failed; no member admitted"))?,
            );
        }
        Ok(PlannedAtomicGroup {
            original_request: body,
            members,
        })
    }

    /// Parse CPU-supplied native transcripts with the SAME official parser and
    /// OpenAI chunk generator as protocol.rs. The transcript is not evidence
    /// that an actual engine generated these IDs or issued the terminal proof.
    /// Caller retains the borrowed raw transcript on every failure.
    pub async fn parse_cpu_transcript(
        &self,
        plan: PlannedChat,
        response_id: String,
        transcript: &CpuTranscript,
    ) -> Result<CpuParseReport> {
        ensure!(!response_id.is_empty(), "response ID must be nonempty");
        ensure!(
            transcript.cached_prompt_tokens <= plan.prompt_ids.len(),
            "cached prompt count exceeds exact native prompt"
        );
        let native_count = transcript
            .token_chunks
            .iter()
            .try_fold(0usize, |sum, chunk| {
                sum.checked_add(chunk.len())
                    .context("native output count overflow")
            })?;
        ensure!(
            native_count <= plan.semantics.output_allowance as usize,
            "native output exceeds unchanged full allowance"
        );
        let raw_ids: Vec<u32> = transcript.token_chunks.iter().flatten().copied().collect();
        for id in &raw_ids {
            ensure!(
                self.tokenizer.id_to_token(*id).is_some(),
                "unknown native output ID{id}"
            );
        }
        let eos_count = raw_ids.iter().filter(|id| **id == EOS_ID).count();
        let reason = match &transcript.terminal {
            CpuTerminal::NativeEos => {
                ensure!(
                    raw_ids.last() == Some(&EOS_ID) && eos_count == 1,
                    "EOS terminal requires exactly one EOS at final native position"
                );
                Some(InferenceFinishReason::Stop)
            }
            CpuTerminal::OutputAllowance => {
                ensure!(
                    raw_ids.len() == plan.semantics.output_allowance as usize && eos_count == 0,
                    "length terminal must consume exact requested allowance without EOS"
                );
                Some(InferenceFinishReason::Length)
            }
            CpuTerminal::UnannouncedEof | CpuTerminal::BackendError { .. } => {
                ensure!(eos_count == 0, "EOS IDs without EOS terminal authority");
                None
            }
        };
        let mut chunks = vec![InferenceChunk::Ready {
            system_fingerprint: None,
            prompt_usage: PromptUsage {
                prompt_tokens: plan.prompt_ids.len(),
                prompt_cache_hit_tokens: transcript.cached_prompt_tokens,
            },
        }];
        for token_chunk in &transcript.token_chunks {
            let ids: Vec<u32> = token_chunk
                .iter()
                .copied()
                .filter(|id| *id != EOS_ID)
                .collect();
            if !ids.is_empty() {
                chunks.push(InferenceChunk::Tokens {
                    token_ids: ids,
                    logprobs: None,
                });
            }
        }
        if let Some(finish_reason) = reason {
            chunks.push(InferenceChunk::Finish { finish_reason });
        }
        let generator = ChatCompletionRequest::chunk_generator(
            &plan.prepared.converted,
            response_id.clone(),
            SERVED_MODEL.into(),
        );
        let finish = Arc::new(Mutex::new(None));
        let generator = ObserveFinish {
            inner: generator,
            finish: Arc::clone(&finish),
        };
        let stream = StreamProcessor::new(generator, plan.prepared.converted.parsing_options)
            .with_strict_finish(true)
            .with_tokenizer(self.tokenizer.clone())
            .into_protocol_stream(tokio_stream::iter(chunks));
        tokio::pin!(stream);
        let mut response = ChatCompletionResponse::new(response_id, SERVED_MODEL.into(), 0, 0, 0);
        let mut events = Vec::new();
        let mut failure = None;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    response.created = chunk.created;
                    events.push(serde_json::to_value(&chunk)?);
                    response.accumulate(chunk);
                }
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }
        }
        if let CpuTerminal::BackendError { detail } = &transcript.terminal {
            failure = Some(format!(
                "backend error retained, even if a parser stop preceded it: {detail}"
            ));
        }
        let observed = finish
            .lock()
            .map_err(|_| anyhow::anyhow!("terminal observation poisoned"))?
            .clone();
        if observed.is_none() && failure.is_none() {
            failure = Some("no official parser terminal observed".into());
        }
        let mut wire = serde_json::to_value(response)?;
        let parser_usage = wire.get("usage").cloned();
        if let Some((v, spec)) = &plan.forced {
            let checked = (|| -> Result<()> {
                let mut cursor = ForcedNamedCursor::new(
                    v.clone(),
                    spec.clone(),
                    plan.semantics.original_request_sha256.clone(),
                    &plan.prompt_ids,
                )?;
                for &id in &raw_ids {
                    cursor.accept(id)?;
                }
                ensure!(cursor.finished(),"forced named output must finish its complete calls block with actual native EOS");
                let choices = wire["choices"]
                    .as_array()
                    .context("forced parsed choices missing")?;
                ensure!(
                    choices.len() == 1 && choices[0]["finish_reason"] == "tool_calls",
                    "forced named response requires tool_calls finish"
                );
                let calls = choices[0]["message"]["tool_calls"]
                    .as_array()
                    .context("forced parsed tool calls missing")?;
                ensure!(
                    !calls.is_empty(),
                    "required named response emitted no invocation"
                );
                for call in calls {
                    ensure!(
                        call["function"]["name"] == spec.name(),
                        "parsed function differs from forced name"
                    );
                    let arguments = serde_json::from_str(
                        call["function"]["arguments"]
                            .as_str()
                            .context("forced argument JSON string missing")?,
                    )?;
                    spec.validate_arguments(&arguments)?;
                }
                Ok(())
            })();
            if let Err(e) = checked {
                failure = Some(format!("forced named constraint failed:{e:#}"));
            }
        }
        let passed = failure.is_none();
        if passed {
            // Standard usage reports native generated IDs. Parser consumption
            // can exclude EOS, post-stop IDs and undecoded trailing IDs; keep
            // its original snapshot separately, never silently equate counts.
            if let Some(usage) = wire.get_mut("usage").and_then(Value::as_object_mut) {
                usage.insert("completion_tokens".into(), json!(raw_ids.len()));
                usage.insert(
                    "total_tokens".into(),
                    json!(plan.prompt_ids.len() + raw_ids.len()),
                );
            }
            if plan.semantics.return_token_ids {
                for choice in wire["choices"]
                    .as_array_mut()
                    .context("official choices missing")?
                {
                    choice["token_ids"] = json!(&raw_ids);
                }
            }
        }
        Ok(CpuParseReport {
            passed, response: passed.then_some(wire), partial_protocol_events: events,
            error: failure, native_prompt_ids: plan.prompt_ids,
            native_output_ids_sha256_u32le: ids_sha(&raw_ids),
            native_generated_tokens: raw_ids.len(), native_output_ids: raw_ids,
            native_eos_tokens: eos_count, parser_usage, parser_terminal: observed,
            semantics: plan.semantics,
            runtime_qualified: false,
            terminal_authority: "CPU transcript only; opaque engine-issued terminal proof must be integrated and qualified later",
        })
    }
}

/// These variants classify CPU fixtures, not trusted runtime completion tokens.
/// A future backend must mint an opaque terminal receipt bound to request/lane,
/// generation, token sequence and checked collective completion; JSON assertions
/// alone can never qualify inference or authorize fixture side effects.
#[derive(Clone, Debug, Serialize)]
pub enum CpuTerminal {
    NativeEos,
    OutputAllowance,
    UnannouncedEof,
    BackendError { detail: String },
}

#[derive(Debug, Serialize)]
pub struct CpuTranscript {
    pub token_chunks: Vec<Vec<u32>>,
    pub terminal: CpuTerminal,
    pub cached_prompt_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParserTerminal {
    pub origin: String,
    pub stop_sequence: Option<String>,
    pub consumed_native_tokens: usize,
    pub reasoning_reencoded_tokens: Option<usize>,
}

struct ObserveFinish {
    inner: ChatCompletionChunkGenerator,
    finish: Arc<Mutex<Option<ParserTerminal>>>,
}

impl ChunkGenerator for ObserveFinish {
    type Chunk = ChatCompletionChunk;
    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        if let OutputChunk::Finish {
            finish_reason,
            stop_sequence,
            completion_usage,
        } = &chunk
        {
            let origin = match finish_reason {
                FinishReason::Stop => "native_eos",
                FinishReason::ToolCalls => "native_eos_with_parsed_tool_calls",
                FinishReason::Length => "native_output_allowance",
                FinishReason::StopSequence => "official_parser_user_stop_sequence",
                FinishReason::ContentFilter => "content_filter",
                FinishReason::EndOfStream => "unannounced_EOF_INVALID",
            };
            // A poisoned observation is handled as a failure after parsing.
            if let Ok(mut observation) = self.finish.lock() {
                *observation = Some(ParserTerminal {
                    origin: origin.into(),
                    stop_sequence: stop_sequence.clone(),
                    consumed_native_tokens: completion_usage.completion_tokens,
                    reasoning_reencoded_tokens: completion_usage.reasoning_tokens,
                });
            }
        }
        self.inner.generate(chunk).await
    }
}

#[derive(Debug, Serialize)]
pub struct CpuParseReport {
    pub passed: bool,
    pub response: Option<Value>,
    pub partial_protocol_events: Vec<Value>,
    pub error: Option<String>,
    pub native_prompt_ids: Vec<u32>,
    pub native_output_ids: Vec<u32>,
    pub native_output_ids_sha256_u32le: String,
    pub native_generated_tokens: usize,
    pub native_eos_tokens: usize,
    pub parser_usage: Option<Value>,
    pub parser_terminal: Option<ParserTerminal>,
    pub semantics: EffectiveSemantics,
    pub runtime_qualified: bool,
    pub terminal_authority: &'static str,
}

/// Shape-only batch aggregation. A failed member never yields a normal batch
/// response. Return every per-member report to the caller regardless of failure.
pub fn aggregate_cpu_atomic_response(
    group: &PlannedAtomicGroup,
    id: &str,
    reports: &[CpuParseReport],
) -> Result<Value> {
    ensure!(
        !id.is_empty() && !reports.is_empty(),
        "nonempty batch ID/reports required"
    );
    ensure!(
        reports.len() == group.members.len(),
        "atomic member count changed"
    );
    ensure!(
        reports.iter().all(|r| r.passed && r.response.is_some()),
        "failed member retained; normal batch completion refused"
    );
    let mut choices = Vec::with_capacity(reports.len());
    let mut prompt = 0usize;
    let mut output = 0usize;
    for (index, report) in reports.iter().enumerate() {
        ensure!(
            report.native_prompt_ids == group.members[index].prompt_ids
                && serde_json::to_value(&report.semantics)?
                    == serde_json::to_value(&group.members[index].semantics)?,
            "atomic response member{index} differs from ordered admitted CPU plan"
        );
        let response = report.response.as_ref().unwrap();
        let member = response["choices"]
            .as_array()
            .context("member choices missing")?;
        ensure!(
            member.len() == 1,
            "each atomic member must contain one completion"
        );
        let mut choice = member[0].clone();
        choice["index"] = json!(index);
        choices.push(choice);
        prompt = prompt
            .checked_add(report.native_prompt_ids.len())
            .context("batch prompt overflow")?;
        output = output
            .checked_add(report.native_generated_tokens)
            .context("batch output overflow")?;
    }
    Ok(
        json!({"id":id,"object":"chat.completion","model":SERVED_MODEL,
        "created":reports[0].response.as_ref().unwrap()["created"],"choices":choices,
        "usage":{"prompt_tokens":prompt,"completion_tokens":output,
            "total_tokens":prompt.checked_add(output).context("batch total overflow")?}}),
    )
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn ids_sha(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn only_fields(object: &Map<String, Value>, fields: &[&str], label: &str) -> Result<()> {
    for name in object.keys() {
        ensure!(
            fields.contains(&name.as_str()),
            "unsupported {label} field{name}; ignoring it would change request semantics"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    // Existing pinned local artifact, read only. Missing/wrong bytes FAIL the
    // test rather than silently substituting synthetic tokenizer/vocabulary.
    const VERIFIED_TOKENIZER_PATH: &str =
        "test-data/tokenizer.json";
    const PROMPT_REFERENCE: &str = include_str!("../../tests/fixtures/dsv41_reference_prompt_ids.json");
    fn bridge() -> &'static CpuBridge {
        static BRIDGE: OnceLock<CpuBridge> = OnceLock::new();
        BRIDGE.get_or_init(|| {
            CpuBridge::from_verified_tokenizer(
                &std::fs::read(VERIFIED_TOKENIZER_PATH).expect("retained1017 tokenizer required"),
            )
            .expect("verified native tokenizer")
        })
    }
    fn limits() -> PlanningLimits {
        PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 8192,
            atomic_requests: 16,
            request_bytes: 16 * 1024 * 1024,
        }
    }
    fn request(thinking: bool) -> Value {
        json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":"Answer briefly."}],
            "temperature":0,"top_p":1,"max_tokens":4096,
            "chat_template_kwargs":{"thinking":thinking}})
    }
    fn fixture_ids(text: &str) -> Vec<u32> {
        bridge()
            .tokenizer
            .encode(text, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }
    fn eos(text: &str, width: usize) -> CpuTranscript {
        let mut ids = fixture_ids(text);
        ids.push(EOS_ID);
        CpuTranscript {
            token_chunks: ids.chunks(width).map(|s| s.to_vec()).collect(),
            terminal: CpuTerminal::NativeEos,
            cached_prompt_tokens: 0,
        }
    }

    #[test]
    fn stock_model_and_all_execution_capabilities_fail_closed() {
        let mut r = request(false);
        let p = bridge().prepare_single(r.clone(), limits()).unwrap();
        assert!(p.execution_permit().is_err());
        for model in [
            "DeepSeek-V4.1-Flash-1017",
            "deepseek-v4.1-flash-glp",
            "Qwen",
        ] {
            r["model"] = json!(model);
            assert!(bridge().prepare_single(r.clone(), limits()).is_err());
        }
        assert_eq!(capabilities()["model_serving"], false);
        assert_eq!(capabilities()["context8192_qualified"], false);
        assert_eq!(capabilities()["atomic_backend_admission_qualified"], false);
        assert!(CpuBridge::from_verified_tokenizer(b"not the1017 tokenizer").is_err());
    }

    #[test]
    fn full_quality_and_capped_budgets_remain_distinct_and_never_clip() {
        let q = bridge().prepare_single(request(false), limits()).unwrap();
        assert!(!q.semantics.thinking);
        assert_eq!(q.semantics.effective_reasoning_score, None);
        assert_eq!(q.semantics.output_allowance, 4096);
        let mut capped = request(true);
        capped["max_tokens"] = json!(256);
        let c = bridge().prepare_single(capped.clone(), limits()).unwrap();
        assert_eq!(c.semantics.effective_reasoning_score, Some(75));
        assert_eq!(c.semantics.output_allowance, 256);
        let mut insufficient = limits();
        insufficient.context_tokens = c.semantics.required_context_tokens - 1;
        insufficient.prompt_tokens = insufficient.context_tokens;
        insufficient.output_tokens = insufficient.context_tokens;
        assert!(bridge()
            .prepare_single(capped, insufficient)
            .unwrap_err()
            .to_string()
            .contains("clipping refused"));
        for (thinking, effort, expected) in
            [(false, 75, 8192), (true, 75, 65536), (true, 100, 131072)]
        {
            let mut r = request(thinking);
            r.as_object_mut().unwrap().remove("max_tokens");
            r["reasoning_effort"] = json!(effort);
            assert!(bridge().prepare_single(r.clone(), limits()).is_err());
            let roomy = PlanningLimits {
                context_tokens: 1_048_576,
                prompt_tokens: 1_048_576,
                output_tokens: 131072,
                ..limits()
            };
            let p = bridge().prepare_single(r, roomy).unwrap();
            assert_eq!(p.semantics.requested_max_tokens, None);
            assert_eq!(p.semantics.output_allowance, expected);
        }
    }

    #[test]
    fn preserve_supported_request_controls_reject_every_unsupported_control() {
        let mut r = request(false);
        r["seed"] = json!(47);
        r["cache_salt"] = json!("paired-attempt-東京");
        r["parallel_tool_calls"] = json!(true);
        r["stop"] = json!(["END", "STOP"]);
        r["max_completion_tokens"] = json!(4096);
        let p = bridge().prepare_single(r.clone(), limits()).unwrap();
        assert_eq!(p.original_request(), &r);
        assert_eq!(p.semantics.seed, Some(47));
        assert_eq!(
            p.semantics.cache_namespace.as_deref(),
            Some("paired-attempt-東京")
        );
        assert_eq!(p.semantics.stop_sequences, ["END", "STOP"]);
        for (field, value) in [
            ("temperature", json!(0.1)),
            ("temperature", json!(1.0e-50)),
            ("top_p", json!(0.95)),
            ("top_p", json!(0.99999999)),
            ("presence_penalty", json!(0.1)),
            ("frequency_penalty", json!(-0.1)),
            ("min_tokens", json!(1)),
            ("ignore_eos", json!(true)),
            ("stream", json!(true)),
            ("parallel_tool_calls", json!(false)),
            ("tool_choice", json!("required")),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(0)),
            ("truncate_prompt_tokens", json!(100)),
            ("response_format", json!({"type":"json_object"})),
            ("max_completion_tokens", json!(10)),
            ("stop", json!(["valid", 7])),
            ("stop", json!([""])),
            ("n", json!(2)),
            ("top_k", json!(1)),
            ("unknown_future_setting", json!(true)),
        ] {
            let mut changed = r.clone();
            changed[field] = value;
            assert!(
                bridge().prepare_single(changed, limits()).is_err(),
                "accepted unsupported{field}"
            );
        }
        r.as_object_mut().unwrap().remove("temperature");
        assert!(bridge()
            .prepare_single(r, limits())
            .unwrap_err()
            .to_string()
            .contains("sampling is unsupported"));
    }

    #[test]
    fn exact_historical_code75_prose68_ids_from_independent_receipts() {
        let reference: Value = serde_json::from_str(PROMPT_REFERENCE).unwrap();
        for (kind, count, hash) in [
            (
                "code",
                75,
                "f5cd9db5cc738c2a8b7eafadd34de50dc3ba70cb18c782d380517ea093975c4b",
            ),
            (
                "prose",
                68,
                "c2df86f184dc4c8fb2f1f684315808386ca2d6b65758ef3a51cbd70209ead6e3",
            ),
        ] {
            let golden = &reference["capped_speed"]["prompts_and_native_IDs"][kind];
            let r = json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":golden["text"]}],
                "temperature":0,"seed":42,"max_tokens":256,"return_token_ids":true,"stream":false});
            let p = bridge().prepare_single(r, limits()).unwrap();
            assert_eq!(p.prompt_ids.len(), count);
            assert_eq!(json!(p.prompt_ids), golden["native_input_ids"]);
            assert_eq!(p.semantics.prompt_ids_sha256_u32le, hash);
            assert_eq!(p.semantics.effective_reasoning_score, Some(75));
            assert_eq!(p.semantics.requested_top_p, None);
            assert_eq!(p.semantics.effective_top_p, 1.0);
        }
    }

    #[test]
    #[ignore = "requires the operator's private original-request archive"]
    fn all573_frozen_quality_requests_convert_without_payload_edits() {
        let source = std::env::var_os("DSV41_PRIVATE_REQUEST_AUDIT")
            .expect("set DSV41_PRIVATE_REQUEST_AUDIT to the original request manifest");
        let audit: Value = serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
        let records = audit["source_request_records"].as_array().unwrap();
        assert_eq!(records.len(), 573);
        let roomy = PlanningLimits {
            context_tokens: 1_048_576,
            prompt_tokens: 1_048_576,
            output_tokens: 4096,
            ..limits()
        };
        let mut salts = 0;
        for row in records {
            let bytes = std::fs::read(row["path"].as_str().unwrap()).unwrap();
            assert_eq!(sha(&bytes), row["sha256"].as_str().unwrap());
            let raw: Value = serde_json::from_slice(&bytes).unwrap();
            let body = raw["request"].clone();
            let plan = bridge()
                .prepare_single(body.clone(), roomy)
                .unwrap_or_else(|e| panic!("{}:{e:#}", row["path"]));
            assert_eq!(plan.original_request(), &body);
            assert_eq!(plan.semantics.output_allowance, 4096);
            assert!(!plan.semantics.thinking);
            assert_eq!(plan.semantics.tool_choice, "auto");
            salts += usize::from(plan.semantics.cache_namespace.is_some());
        }
        assert_eq!(salts, 213);
    }

    #[test]
    fn atomic_exact_nested_envelope_is_ordered_and_all_or_none() {
        let base = request(false);
        let mut batch = base.clone();
        batch["messages"] = json!([base["messages"],
            [{"role":"user","content":"second distinct conversation"}]]);
        let group = bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), batch.clone(), limits())
            .unwrap();
        assert_eq!(group.members.len(), 2);
        assert_ne!(group.members[0].prompt_ids, group.members[1].prompt_ids);
        assert!(group.execution_permit().is_err());
        for (path, header) in [
            (CHAT_ROUTE, Some("1")),
            (ATOMIC_ROUTE, None),
            (ATOMIC_ROUTE, Some("true")),
        ] {
            assert!(bridge()
                .prepare_atomic(path, header, batch.clone(), limits())
                .is_err());
        }
        batch["messages"][1] =
            json!([{"role":"tool","content":"orphan result","tool_call_id":"unknown"}]);
        assert!(bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), batch, limits())
            .is_err());
        assert!(bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), base, limits())
            .is_err());
        // Actual retained C8 request, not a newly invented batch envelope.
        let frozen: Value = serde_json::from_str(include_str!("../../tests/fixtures/dsv41_actual_atomic_requests.json")).unwrap();
        let old = &frozen["cases"][0];
        assert_eq!(old["concurrency"], 8);
        let retained = bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), old["request"].clone(), limits())
            .unwrap();
        assert_eq!(retained.members.len(), 8);
        assert!(retained.members.iter().all(|p| p.prompt_ids.len() == 75
            && p.semantics.output_allowance == 256
            && p.semantics.effective_reasoning_score == Some(75)));
        let mut c16 = old["request"].clone();
        c16["messages"] = json!(vec![old["request"]["messages"][0].clone(); 16]);
        assert_eq!(
            bridge()
                .prepare_atomic(ATOMIC_ROUTE, Some("1"), c16.clone(), limits())
                .unwrap()
                .members
                .len(),
            16
        );
        c16["messages"]
            .as_array_mut()
            .unwrap()
            .push(old["request"]["messages"][0].clone());
        assert!(bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), c16, limits())
            .is_err());
    }

    #[tokio::test]
    async fn frozen_capped_native_outputs_use_official_reasoning_parser_unchanged() {
        let reference: Value = serde_json::from_str(PROMPT_REFERENCE).unwrap();
        for kind in ["code", "prose"] {
            let golden = &reference["capped_speed"]["prompts_and_native_IDs"][kind];
            let ids: Vec<u32> =
                serde_json::from_value(golden["native_output_reference"]["ids"].clone()).unwrap();
            assert_eq!(ids.len(), 256);
            let request = json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":golden["text"]}],
                "temperature":0,"seed":42,"max_tokens":256,"return_token_ids":true});
            let transcript = CpuTranscript {
                token_chunks: ids.chunks(3).map(|r| r.to_vec()).collect(),
                terminal: CpuTerminal::OutputAllowance,
                cached_prompt_tokens: 0,
            };
            let parsed = bridge()
                .parse_cpu_transcript(
                    bridge().prepare_single(request, limits()).unwrap(),
                    format!("cpu-historical-{kind}"),
                    &transcript,
                )
                .await
                .unwrap();
            assert!(parsed.passed, "{:?}", parsed.error);
            let response = parsed.response.unwrap();
            assert_eq!(response["choices"][0]["finish_reason"], "length");
            assert_eq!(response["choices"][0]["token_ids"], json!(ids));
            let message = &response["choices"][0]["message"];
            assert!(message["content"].as_str().is_none_or(str::is_empty));
            let text = message["reasoning_content"].as_str().unwrap();
            assert_eq!(
                sha(text.as_bytes()),
                golden["native_output_reference"]["text_sha256"]
                    .as_str()
                    .unwrap()
            );
            assert_eq!(response["usage"]["completion_tokens"], 256);
            assert_eq!(parsed.semantics.output_allowance, 256);
        }
    }

    #[tokio::test]
    async fn reject_forged_terminal_shapes_and_invalid_native_usage() {
        let mut r = request(false);
        r["max_tokens"] = json!(4);
        for raw in [
            CpuTranscript {
                token_chunks: vec![vec![2581]],
                terminal: CpuTerminal::NativeEos,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![1, 2581]],
                terminal: CpuTerminal::NativeEos,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![1, 1]],
                terminal: CpuTerminal::NativeEos,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![2581, 1]],
                terminal: CpuTerminal::UnannouncedEof,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![2581]],
                terminal: CpuTerminal::OutputAllowance,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![u32::MAX, 1]],
                terminal: CpuTerminal::NativeEos,
                cached_prompt_tokens: 0,
            },
            CpuTranscript {
                token_chunks: vec![vec![2581, 1]],
                terminal: CpuTerminal::NativeEos,
                cached_prompt_tokens: usize::MAX,
            },
        ] {
            assert!(bridge()
                .parse_cpu_transcript(
                    bridge().prepare_single(r.clone(), limits()).unwrap(),
                    "cpu-invalid-terminal".into(),
                    &raw
                )
                .await
                .is_err());
        }
    }

    #[tokio::test]
    async fn native_eos_counts_are_distinct_from_parser_usage_and_eof_fails() {
        let r = request(false);
        let raw = eos("東京🙂", 1);
        let parsed = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r.clone(), limits()).unwrap(),
                "cpu-eos".into(),
                &raw,
            )
            .await
            .unwrap();
        assert!(parsed.passed);
        assert_eq!(
            parsed.response.as_ref().unwrap()["choices"][0]["message"]["content"],
            "東京🙂"
        );
        assert_eq!(
            parsed.native_generated_tokens,
            raw.token_chunks.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(parsed.native_eos_tokens, 1);
        assert_eq!(
            parsed.parser_terminal.as_ref().unwrap().origin,
            "native_eos"
        );
        assert_eq!(
            parsed.parser_usage.as_ref().unwrap()["completion_tokens"]
                .as_u64()
                .unwrap() as usize,
            parsed.native_generated_tokens - 1
        );
        assert_eq!(
            parsed.response.as_ref().unwrap()["usage"]["completion_tokens"],
            json!(parsed.native_generated_tokens)
        );
        assert!(!parsed.runtime_qualified);
        let incomplete = CpuTranscript {
            token_chunks: vec![fixture_ids("answer")],
            terminal: CpuTerminal::UnannouncedEof,
            cached_prompt_tokens: 0,
        };
        let failed = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r, limits()).unwrap(),
                "cpu-eof".into(),
                &incomplete,
            )
            .await
            .unwrap();
        assert!(!failed.passed && failed.response.is_none());
        assert!(failed.error.unwrap().contains("without explicit Finish"));
        assert!(!failed.partial_protocol_events.is_empty());
    }

    #[tokio::test]
    async fn parser_stop_length_and_error_have_distinct_terminal_origins() {
        let mut r = request(false);
        r["stop"] = json!("END");
        let raw = CpuTranscript {
            token_chunks: vec![fixture_ids("answer END ignored")],
            terminal: CpuTerminal::UnannouncedEof,
            cached_prompt_tokens: 0,
        };
        let stopped = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r.clone(), limits()).unwrap(),
                "cpu-user-stop".into(),
                &raw,
            )
            .await
            .unwrap();
        assert!(stopped.passed);
        assert_eq!(
            stopped.parser_terminal.as_ref().unwrap().origin,
            "official_parser_user_stop_sequence"
        );
        assert_eq!(
            stopped.parser_terminal.unwrap().stop_sequence.as_deref(),
            Some("END")
        );
        let errored = CpuTranscript {
            terminal: CpuTerminal::BackendError {
                detail: "fixture failure".into(),
            },
            ..raw
        };
        let failed = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r, limits()).unwrap(),
                "cpu-error".into(),
                &errored,
            )
            .await
            .unwrap();
        assert!(!failed.passed && failed.response.is_none());
        let mut r = request(false);
        r["max_tokens"] = json!(1);
        let capped = CpuTranscript {
            token_chunks: vec![vec![2581]],
            terminal: CpuTerminal::OutputAllowance,
            cached_prompt_tokens: 0,
        };
        let result = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r.clone(), limits()).unwrap(),
                "cpu-length".into(),
                &capped,
            )
            .await
            .unwrap();
        assert!(result.passed);
        assert_eq!(
            result.response.unwrap()["choices"][0]["finish_reason"],
            "length"
        );
        let short = CpuTranscript {
            token_chunks: vec![],
            ..capped
        };
        assert!(bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(r, limits()).unwrap(),
                "cpu-false-length".into(),
                &short
            )
            .await
            .is_err());
    }

    fn tool_request() -> Value {
        let mut r = request(false);
        r["tools"] = json!([{"type":"function","function":{"name":"read",
            "parameters":{"type":"object","properties":{"label":{"type":"string"},
            "optional":{"type":["object","null"]}},"additionalProperties":false}}}]);
        r["tool_choice"] = json!("auto");
        r
    }

    #[tokio::test]
    async fn split_native_tool_calls_unicode_null_nested_results_and_eof() {
        let dsml = "<｜DSML｜ calls><｜DSML｜ invoke name=\"read\"><｜DSML｜ parameter name=\"label\" string=\"true\">東京</｜DSML｜ parameter><｜DSML｜ parameter name=\"optional\" string=\"false\">{\"nested\":[null,{\"ok\":true}]}</｜DSML｜ parameter></｜DSML｜ invoke><｜DSML｜ invoke name=\"read\"><｜DSML｜ parameter name=\"label\" string=\"true\">二</｜DSML｜ parameter><｜DSML｜ parameter name=\"optional\" string=\"false\">null</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        let raw = eos(dsml, 1);
        let parsed = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(tool_request(), limits()).unwrap(),
                "cpu-tools".into(),
                &raw,
            )
            .await
            .unwrap();
        assert!(parsed.passed, "{:?}", parsed.error);
        let response = parsed.response.unwrap();
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        let calls = response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            args,
            json!({"label":"東京","optional":{"nested":[null,{"ok":true}]}})
        );
        let mut history = tool_request();
        history["messages"].as_array_mut().unwrap().extend([
            json!({"role":"assistant","content":null,"tool_calls":calls}),
            json!({"role":"tool","tool_call_id":calls[1]["id"],"content":"second result"}),
            json!({"role":"tool","tool_call_id":calls[0]["id"],"content":"first result"}),
        ]);
        // Indexed output-only extension is not a request tool-call field.
        for call in history["messages"][1]["tool_calls"].as_array_mut().unwrap() {
            call.as_object_mut().unwrap().remove("index");
        }
        bridge().prepare_single(history.clone(), limits()).unwrap();
        history["messages"][2]["tool_call_id"] = json!("wrong");
        assert!(bridge().prepare_single(history, limits()).is_err());
        let mut incomplete = raw;
        incomplete.token_chunks.last_mut().unwrap().pop();
        incomplete.terminal = CpuTerminal::UnannouncedEof;
        let failed = bridge()
            .parse_cpu_transcript(
                bridge().prepare_single(tool_request(), limits()).unwrap(),
                "cpu-tool-eof".into(),
                &incomplete,
            )
            .await
            .unwrap();
        assert!(!failed.passed && failed.response.is_none());
    }

    #[tokio::test]
    async fn atomic_response_indexes_native_usage_and_order_binding() {
        let base = request(false);
        let mut body = base.clone();
        body["messages"] = json!([base["messages"],[{"role":"user","content":"other"}]]);
        let group = bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), body.clone(), limits())
            .unwrap();
        let members = bridge()
            .prepare_atomic(ATOMIC_ROUTE, Some("1"), body, limits())
            .unwrap()
            .members;
        let mut reports = Vec::new();
        for (index, member) in members.into_iter().enumerate() {
            reports.push(
                bridge()
                    .parse_cpu_transcript(member, format!("cpu-atomic-{index}"), &eos("ok", 2))
                    .await
                    .unwrap(),
            );
        }
        let response = aggregate_cpu_atomic_response(&group, "cpu-group", &reports).unwrap();
        assert_eq!(response["choices"][0]["index"], 0);
        assert_eq!(response["choices"][1]["index"], 1);
        assert_eq!(
            response["usage"]["completion_tokens"],
            json!(reports
                .iter()
                .map(|r| r.native_generated_tokens)
                .sum::<usize>())
        );
        reports.swap(0, 1);
        assert!(aggregate_cpu_atomic_response(&group, "cpu-wrong-order", &reports).is_err());
        reports.swap(0, 1);
        reports[1].passed = false;
        assert!(aggregate_cpu_atomic_response(&group, "cpu-failed-member", &reports).is_err());
    }
}
fn optional_bool(object: &Map<String, Value>, name: &str) -> Result<Option<bool>> {
    object
        .get(name)
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_bool()
                .with_context(|| format!("{name} must be boolean"))
        })
        .transpose()
}
fn optional_string(object: &Map<String, Value>, name: &str) -> Result<Option<String>> {
    object
        .get(name)
        .filter(|v| !v.is_null())
        .map(|v| {
            let value = v
                .as_str()
                .with_context(|| format!("{name} must be string"))?;
            ensure!(!value.is_empty(), "{name} must not be empty");
            Ok(value.into())
        })
        .transpose()
}

/// Additional strictness is intentional: the toolkit accepts unknown fields
/// and strips some vendor metadata. A native bridge must never promise that
/// ignored controls affected inference. Tool schemas' own JSON remains opaque
/// and is validated/rendered by the official toolkit, not rewritten here.
fn named_choice(body: &Value) -> Result<Option<String>> {
    let Some(choice) = body.get("tool_choice").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if choice.is_string() {
        return Ok(None);
    }
    let m = choice
        .as_object()
        .context("tool_choice must be a string or named function object")?;
    only_fields(m, &["type", "function"], "named tool_choice")?;
    ensure!(
        m.get("type").and_then(Value::as_str) == Some("function"),
        "named tool_choice type must be function"
    );
    let f = m
        .get("function")
        .and_then(Value::as_object)
        .context("named tool_choice function object missing")?;
    only_fields(f, &["name"], "named tool_choice function")?;
    let name = f
        .get("name")
        .and_then(Value::as_str)
        .context("named function name missing")?;
    ensure!(!name.is_empty(), "empty named function");
    Ok(Some(name.into()))
}

fn validate_single_fields(body: &Value) -> Result<()> {
    validate_single_fields_transport(body, false)
}
fn validate_single_fields_transport(body: &Value, streaming: bool) -> Result<()> {
    let object = body.as_object().context("chat request must be object")?;
    only_fields(
        object,
        &[
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "thinking",
            "reasoning_effort",
            "chat_template_kwargs",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "parallel_tool_calls",
            "response_format",
            "logprobs",
            "top_logprobs",
            "return_token_ids",
            "cache_salt",
            "min_tokens",
            "ignore_eos",
        ],
        "chat request",
    )?;
    ensure!(
        object.get("model").and_then(Value::as_str) == Some(SERVED_MODEL),
        "model must be exact stock served name{SERVED_MODEL}"
    );
    // Check the original JSON number before toolkit f32 deserialization can
    // round a nonzero temperature to0 or a nonunit top_p to1.
    for (name, expected) in [("temperature", 0.0), ("top_p", 1.0)] {
        if let Some(value) = object.get(name).filter(|v| !v.is_null()) {
            ensure!(
                value.as_f64() == Some(expected),
                "unsupported raw {name}; greedy path requires exactly {expected}"
            );
        }
    }
    ensure!(
        optional_bool(object, "stream")?.unwrap_or(false) == streaming,
        "request stream mode differs from explicit preparation API"
    );
    if streaming {
        ensure!(
            object.get("stream").and_then(Value::as_bool) == Some(true),
            "explicit original stream:true required"
        );
        if let Some(options) = object.get("stream_options").filter(|v| !v.is_null()) {
            let options = options
                .as_object()
                .context("stream_options must be object")?;
            only_fields(options, &["include_usage"], "stream_options")?;
            optional_bool(options, "include_usage")?;
        }
        ensure!(
            !optional_bool(object, "return_token_ids")?.unwrap_or(false),
            "stream return_token_ids unsupported; actual IDs retained in private native evidence"
        );
    } else {
        ensure!(
            !object.contains_key("stream_options"),
            "stream_options requires explicit streaming preparation"
        );
    }
    ensure!(
        !optional_bool(object, "ignore_eos")?.unwrap_or(false),
        "ignore_eos unsupported"
    );
    if let Some(value) = object.get("min_tokens").filter(|v| !v.is_null()) {
        ensure!(
            value.as_u64() == Some(0),
            "positive/invalid min_tokens unsupported"
        );
    }
    for penalty in ["presence_penalty", "frequency_penalty"] {
        if let Some(value) = object.get(penalty).filter(|v| !v.is_null()) {
            ensure!(
                value.as_f64() == Some(0.0),
                "nonzero/invalid {penalty} unsupported"
            );
        }
    }
    ensure!(
        !optional_bool(object, "logprobs")?.unwrap_or(false),
        "native logprobs unsupported"
    );
    ensure!(
        object.get("top_logprobs").is_none_or(Value::is_null),
        "native top_logprobs unsupported"
    );
    ensure!(optional_bool(object,"parallel_tool_calls")? != Some(false),
        "parallel_tool_calls=false requires unsupported generation constraint; true/default permits multiple calls");
    if let Some(choice) = object.get("tool_choice").filter(|v| !v.is_null()) {
        ensure!(matches!(choice.as_str(),Some("auto"|"none")) || named_choice(body)?.is_some(),
            "tool_choice must be auto/none or an explicitly named supported function; required multi-tool constraints remain unsupported");
    }
    if let Some(format) = object.get("response_format").filter(|v| !v.is_null()) {
        ensure!(
            format == &json!({"type":"text"}),
            "JSON/regex constrained decoding unsupported"
        );
    }
    if let Some(thinking) = object.get("thinking").filter(|v| !v.is_null()) {
        only_fields(
            thinking.as_object().context("thinking must be object")?,
            &["type"],
            "thinking",
        )?;
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .context("messages must be array")?;
    ensure!(!messages.is_empty(), "messages cannot be empty");
    for message in messages {
        let m = message.as_object().context("each message must be object")?;
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .context("message role required")?;
        let allowed: &[&str] = match role {
            "system" | "user" => &["role", "content"],
            "assistant" => &[
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "prefix",
            ],
            "tool" => &["role", "content", "tool_call_id"],
            "latest_reminder" => &["role", "content"],
            _ => bail!("unsupported message role{role}"),
        };
        only_fields(m, allowed, "message")?;
        if let Some(content) = m.get("content") {
            if role == "system"
                && content
                    .as_str()
                    .is_some_and(|s| s.starts_with("x-anthropic-billing-header"))
            {
                bail!("official toolkit strips Anthropic billing headers; implicit prompt edits refused");
            }
            if let Some(parts) = content.as_array() {
                for part in parts {
                    let part = part.as_object().context("content part must be object")?;
                    only_fields(part, &["type", "text"], "text content")?;
                    ensure!(
                        part.get("type").and_then(Value::as_str) == Some("text"),
                        "only text content is qualified for CPU planning"
                    );
                }
            }
        }
        if let Some(calls) = m.get("tool_calls").filter(|v| !v.is_null()) {
            for call in calls.as_array().context("tool_calls must be array")? {
                let call = call.as_object().context("tool call must be object")?;
                only_fields(call, &["id", "type", "function"], "historical tool call")?;
                only_fields(
                    call.get("function")
                        .and_then(Value::as_object)
                        .context("tool call function required")?,
                    &["name", "arguments"],
                    "historical tool function",
                )?;
            }
        }
    }
    if let Some(tools) = object.get("tools").filter(|v| !v.is_null()) {
        for tool in tools.as_array().context("tools must be array")? {
            let tool = tool.as_object().context("tool must be object")?;
            only_fields(tool, &["type", "function"], "tool")?;
            let function = tool
                .get("function")
                .and_then(Value::as_object)
                .context("tool function required")?;
            only_fields(
                function,
                &["name", "description", "parameters", "strict"],
                "tool definition",
            )?;
            ensure!(
                optional_bool(function, "strict")? != Some(true),
                "strict tool schema enforcement requires unsupported constrained decoding"
            );
        }
    }
    Ok(())
}
