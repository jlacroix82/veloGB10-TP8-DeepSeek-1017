//! PRIVATE bounded C1 buffered frontend. Defining this module opens no socket,
//! loads no weights, launches no worker, and grants no execution qualification.
//! One native thread must retain its CUDA/model/session owners independently of
//! HTTP futures. Only the opaque response bridge can supply a normal response.
use super::{
    native_api_contract::{CpuBridge, PlannedChat, PlanningLimits, SERVED_MODEL},
    native_response_bridge::BufferedNativeResponse,
};
use anyhow::{ensure, Context, Result};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Extension, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::{DirBuilder, OpenOptions},
    io::Write,
    net::{Ipv4Addr, SocketAddr},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

const WALL: Duration = Duration::from_secs(180);
const MAX_BODY: usize = 16 * 1024 * 1024;
const QUEUED_REQUESTS: usize = 1;

fn record(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(&bytes)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn raw_record(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)?;
    Ok(())
}
fn wire_error(status: StatusCode, code: &str, detail: &str, id: &str) -> Response {
    (status,Json(json!({"error":{"message":detail,"type":"native_request_error","code":code},"request_id":id}))).into_response()
}

#[derive(Clone)]
struct Admission {
    request_id: String,
    admitted: Instant,
    deadline: Instant,
    path: PathBuf,
    cancelled: Arc<AtomicBool>,
}
impl Admission {
    fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
    fn error(&self, status: StatusCode, code: &str, detail: &str) -> Response {
        let evidence = json!({"request_id":self.request_id,"HTTP_status":status.as_u16(),"class":code,"detail":detail,
            "server_elapsed_seconds":self.admitted.elapsed().as_secs_f64(),"normal_completion":false,
            "cleanup_may_continue_after_HTTP_response":true,"client_TTFT_measured":false});
        // An evidence failure cannot turn a failed attempt into a completion.
        let _ = record(&self.path.join("http-error.json"), &evidence);
        wire_error(status, code, detail, &self.request_id)
    }
}
struct CancelOnDrop {
    admission: Admission,
    control: ResidentControl,
    armed: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.admission.cancelled.store(true, Ordering::Release);
            self.control.block_until_coordinated_native_recovery();
            let _ = record(
                &self.admission.path.join("http-detached.json"),
                &json!({
                "request_id":self.admission.request_id,"cancel_requested":true,"native_owners_dropped_by_HTTP":false,
                "normal_completion":false,"elapsed_seconds":self.admission.admitted.elapsed().as_secs_f64()}),
            );
        }
    }
}

/// Readiness is metadata controlled by the resident supervisor after actual
/// all-eight model/reset checks. It never creates model or terminal authority.
#[derive(Clone)]
pub struct ResidentControl {
    ready: Arc<AtomicBool>,
}
impl ResidentControl {
    pub fn block_until_coordinated_native_recovery(&self) {
        self.ready.store(false, Ordering::Release)
    }
    pub fn mark_ready_after_coordinated_native_reset(&self) {
        self.ready.store(true, Ordering::Release)
    }
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

/// Only this owned item crosses to the ONE initializing native CUDA thread.
/// Its absolute admission Instant/deadline include body/prepare/queue/reset.
/// Neither HTTP disconnect nor dropping the response receiver drops its GPU
/// work. The worker must finish a checked boundary or coordinate all8 teardown,
/// retain cleanup time, and recover before marking readiness again.
pub struct ResidentWork {
    pub planned_chat: PlannedChat,
    pub response_id: String,
    pub admitted: Instant,
    pub deadline: Instant,
    pub raw_request_path: PathBuf,
    cancelled: Arc<AtomicBool>,
    reply: ResidentReply,
    streaming: Option<super::native_streaming::NativeStreamPublisher>,
}
impl ResidentWork {
    pub fn cancellation_requested(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    pub fn deadline_expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
    pub fn into_parts_with_stream(
        self,
    ) -> (
        PlannedChat,
        String,
        Instant,
        Instant,
        PathBuf,
        Arc<AtomicBool>,
        ResidentReply,
        Option<super::native_streaming::NativeStreamPublisher>,
    ) {
        (
            self.planned_chat,
            self.response_id,
            self.admitted,
            self.deadline,
            self.raw_request_path,
            self.cancelled,
            self.reply,
            self.streaming,
        )
    }
    pub fn into_parts(
        self,
    ) -> (
        PlannedChat,
        String,
        Instant,
        Instant,
        PathBuf,
        Arc<AtomicBool>,
        ResidentReply,
    ) {
        assert!(
            self.streaming.is_none(),
            "streaming work requires native streaming owner"
        );
        (
            self.planned_chat,
            self.response_id,
            self.admitted,
            self.deadline,
            self.raw_request_path,
            self.cancelled,
            self.reply,
        )
    }
}

/// Non-clone receiver enforces one resident executor. Blocking receive belongs
/// on its owning native thread; it must not run on an async HTTP task.
pub struct ResidentInbox {
    receiver: mpsc::Receiver<ResidentWork>,
}
impl ResidentInbox {
    pub fn blocking_recv(&mut self) -> Option<ResidentWork> {
        self.receiver.blocking_recv()
    }
    pub async fn recv(&mut self) -> Option<ResidentWork> {
        self.receiver.recv().await
    }
}

#[derive(Debug, Serialize)]
pub struct NativeFailure {
    pub class: String,
    pub detail: String,
    pub all_eight_cleanup_confirmed: bool,
    pub cleanup_elapsed_seconds: Option<f64>,
}
enum WorkerOutcome {
    Checked(BufferedNativeResponse),
    Failed(NativeFailure),
}

/// No method accepts a normal completion JSON. `checked` requires the opaque
/// bridge result; failures can be freely reported but never become HTTP200.
/// Evidence is written by the worker before delivery, including when the client
/// has disconnected and the oneshot receiver no longer exists.
pub struct ResidentReply {
    sender: oneshot::Sender<WorkerOutcome>,
    admission: Admission,
    control: ResidentControl,
    expected_request_sha256: String,
    expected_prompt_ids: Vec<u32>,
}
impl ResidentReply {
    pub fn checked(self, result: BufferedNativeResponse) -> Result<()> {
        let belongs = result
            .parser_report()
            .is_some_and(|p| p.semantics.original_request_sha256 == self.expected_request_sha256)
            && result.native_prompt_ids() == self.expected_prompt_ids
            && result.response().is_none_or(|v| {
                v.get("id").and_then(Value::as_str) == Some(self.admission.request_id.as_str())
            });
        if result.passed() && !belongs {
            self.control.block_until_coordinated_native_recovery();
            record(
                &self
                    .admission
                    .path
                    .join("worker-rejected-bridge-result.json"),
                &result,
            )?;
            return self.failed(NativeFailure {
                class: "response_request_mismatch".into(),
                detail: "opaque bridge result does not belong to this exact HTTP request".into(),
                all_eight_cleanup_confirmed: false,
                cleanup_elapsed_seconds: None,
            });
        }
        record(
            &self.admission.path.join("worker-checked-result.json"),
            &result,
        )?;
        if self.admission.expired() || self.admission.cancelled.load(Ordering::Acquire) {
            return self.failed(NativeFailure {
                class: "request_expired_or_cancelled_after_checked_native_result".into(),
                detail: "checked native evidence retained; buffered response cannot be delivered after admission deadline/cancellation".into(),
                all_eight_cleanup_confirmed: false,
                cleanup_elapsed_seconds: None,
            });
        }
        if !result.passed() || !result.runtime_terminal_authority_checked() {
            self.control.block_until_coordinated_native_recovery();
        }
        let _ = self.sender.send(WorkerOutcome::Checked(result));
        Ok(())
    }
    pub fn failed(self, failure: NativeFailure) -> Result<()> {
        self.control.block_until_coordinated_native_recovery();
        record(&self.admission.path.join("worker-failure.json"), &failure)?;
        let _ = self.sender.send(WorkerOutcome::Failed(failure));
        Ok(())
    }
}

struct FrontendState {
    bridge: Arc<CpuBridge>,
    sender: mpsc::Sender<ResidentWork>,
    control: ResidentControl,
    evidence: PathBuf,
    streaming_enabled: bool,
}

/// Create fresh private evidence and a bounded typed actor interface only.
/// The caller must separately initialize the resident model on ONE native
/// thread and keep readiness false until all8 construction/reset checks pass.
pub fn build_frontend(
    bridge: Arc<CpuBridge>,
    fresh_evidence: PathBuf,
) -> Result<(Router, ResidentInbox, ResidentControl)> {
    build_frontend_mode(bridge, fresh_evidence, false)
}
pub fn build_streaming_frontend(
    bridge: Arc<CpuBridge>,
    fresh_evidence: PathBuf,
) -> Result<(Router, ResidentInbox, ResidentControl)> {
    build_frontend_mode(bridge, fresh_evidence, true)
}
fn build_frontend_mode(
    bridge: Arc<CpuBridge>,
    fresh_evidence: PathBuf,
    streaming_enabled: bool,
) -> Result<(Router, ResidentInbox, ResidentControl)> {
    DirBuilder::new()
        .mode(0o700)
        .create(&fresh_evidence)
        .context("frontend evidence directory must be fresh/private")?;
    record(
        &fresh_evidence.join("frontend-policy.json"),
        &json!({"model":SERVED_MODEL,"buffered_only":!streaming_enabled,"C1":true,
        "queued_requests":QUEUED_REQUESTS,"wall_millis":180000,"wall_scope":"server admission before body reception through checked completion; includes queue, preparation and native reset/rearm",
        "output_allowance":"exact request max_tokens, including4096; never shortened by frontend",
        "evidence_private":true,"SSE":streaming_enabled,"atomic_batching":false,"tool_execution":false,
        "model_or_quality_qualified":false,"client_TTFT_measured":false}),
    )?;
    let (sender, receiver) = mpsc::channel(QUEUED_REQUESTS);
    let control = ResidentControl {
        ready: Arc::new(AtomicBool::new(false)),
    };
    let state = Arc::new(FrontendState {
        bridge,
        sender,
        control: control.clone(),
        evidence: fresh_evidence,
        streaming_enabled,
    });
    let chat = Router::new()
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/chat/completions/batch", post(unsupported_atomic))
        .route_layer(middleware::from_fn_with_state(state.clone(), admit_request));
    let app = Router::new()
        .merge(chat)
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);
    Ok((app, ResidentInbox { receiver }, control))
}

async fn admit_request(
    State(state): State<Arc<FrontendState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let admitted = Instant::now();
    let request_id = format!("chatcmpl-native-{}", Uuid::new_v4().simple());
    let path = state.evidence.join(&request_id);
    if let Err(e) = DirBuilder::new().mode(0o700).create(&path) {
        return wire_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "evidence_failed",
            &format!("cannot retain request: {e}"),
            &request_id,
        );
    }
    let a = Admission {
        request_id,
        admitted,
        deadline: admitted + WALL,
        path,
        cancelled: Arc::new(AtomicBool::new(false)),
    };
    if let Err(e) = record(
        &a.path.join("admission.json"),
        &json!({"request_id":a.request_id,"unix_millis":SystemTime::now().duration_since(UNIX_EPOCH).map(|x|x.as_millis()).unwrap_or(0),
        "path":request.uri().path(),"wall_millis":180000,"queue_capacity":QUEUED_REQUESTS,"headers_or_credentials_recorded":false}),
    ) {
        return a.error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "evidence_failed",
            &e.to_string(),
        );
    }
    request.extensions_mut().insert(a.clone());
    match tokio::time::timeout_at(
        tokio::time::Instant::from_std(a.deadline),
        next.run(request),
    )
    .await
    {
        Ok(response) => {
            let _ = record(
                &a.path.join("http-status.json"),
                &json!({
                "request_id":a.request_id,"status":response.status().as_u16(),
                "server_elapsed_seconds":a.admitted.elapsed().as_secs_f64(),
                "native_success_requires_worker_checked_result_and_http200":true,
                "stream_HTTP200_is_headers_only_not_terminal":response.headers().get(axum::http::header::CONTENT_TYPE).and_then(|v|v.to_str().ok()).is_some_and(|v|v.starts_with("text/event-stream"))}),
            );
            response
        }
        Err(_) => {
            a.cancelled.store(true, Ordering::Release);
            state.control.block_until_coordinated_native_recovery();
            a.error(StatusCode::GATEWAY_TIMEOUT,"request_deadline","180-second server request deadline; native cleanup may continue; no partial normal completion")
        }
    }
}

async fn stream_completion(state: Arc<FrontendState>, a: Admission, body: Value) -> Response {
    let plan = match state.bridge.prepare_stream(
        body,
        PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 1,
            request_bytes: MAX_BODY,
        },
    ) {
        Ok(p) => p,
        Err(e) => {
            return a.error(
                StatusCode::BAD_REQUEST,
                "unsupported_or_invalid_stream_request",
                &format!("{e:#}"),
            )
        }
    };
    if let Err(e) = record(&a.path.join("planned-request.json"), plan.semantics()) {
        return a.error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "evidence_failed",
            &e.to_string(),
        );
    }
    if !state.control.ready() {
        return a.error(
            StatusCode::SERVICE_UNAVAILABLE,
            "native_not_ready",
            "all8 native initialization/recovery required",
        );
    }
    if a.expired() {
        return a.error(
            StatusCode::GATEWAY_TIMEOUT,
            "request_deadline",
            "original admission deadline",
        );
    }
    // Reserve before constructing a pipeline: rejected queue-full requests must
    // not poison another request's live native session via an unadmitted Drop.
    let permit = match state.sender.try_reserve() {
        Ok(p) => p,
        Err(mpsc::error::TrySendError::Full(_)) => {
            return a.error(
                StatusCode::TOO_MANY_REQUESTS,
                "native_queue_full",
                "one native request already queued",
            )
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return a.error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native_worker_closed",
                "native worker unavailable",
            )
        }
    };
    let (publisher, stream) = match super::native_streaming::start(
        &state.bridge,
        &plan,
        a.request_id.clone(),
        a.admitted,
        a.deadline,
        a.cancelled.clone(),
        state.control.clone(),
        a.path.clone(),
    ) {
        Ok(pair) => pair,
        Err(e) => {
            return a.error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "stream_setup_failed",
                &format!("{e:#}"),
            )
        }
    };
    let (sender, _receiver) = oneshot::channel();
    let reply = ResidentReply {
        sender,
        admission: a.clone(),
        control: state.control.clone(),
        expected_request_sha256: plan.semantics().original_request_sha256.clone(),
        expected_prompt_ids: plan.prompt_ids().to_vec(),
    };
    permit.send(ResidentWork {
        planned_chat: plan,
        response_id: a.request_id.clone(),
        admitted: a.admitted,
        deadline: a.deadline,
        raw_request_path: a.path.join("request-body.json"),
        cancelled: a.cancelled.clone(),
        reply,
        streaming: Some(publisher),
    });
    stream.into_response()
}

async fn chat_completion(
    State(state): State<Arc<FrontendState>>,
    Extension(a): Extension<Admission>,
    body: Bytes,
) -> Response {
    if let Err(e) = raw_record(&a.path.join("request-body.json"), &body) {
        return a.error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "evidence_failed",
            &e.to_string(),
        );
    }
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return a.error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };
    if state.streaming_enabled && body.get("stream").and_then(Value::as_bool) == Some(true) {
        return stream_completion(state, a, body).await;
    }
    let plan = match state.bridge.prepare_single(
        body,
        PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 1,
            request_bytes: MAX_BODY,
        },
    ) {
        Ok(p) => p,
        Err(e) => {
            return a.error(
                StatusCode::BAD_REQUEST,
                "unsupported_or_invalid_request",
                &format!("{e:#}"),
            )
        }
    };
    if let Err(e) = record(&a.path.join("planned-request.json"), plan.semantics()) {
        return a.error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "evidence_failed",
            &e.to_string(),
        );
    }
    if !state.control.ready() {
        return a.error(StatusCode::SERVICE_UNAVAILABLE,"native_not_ready","resident native worker is not ready; coordinated all8 initialization/recovery required");
    }
    if a.expired() {
        return a.error(
            StatusCode::GATEWAY_TIMEOUT,
            "request_deadline",
            "deadline expired before native admission",
        );
    }
    let expected_sha = plan.semantics().original_request_sha256.clone();
    let expected_prompt = plan.prompt_ids().to_vec();
    let (sender, receiver) = oneshot::channel();
    let reply = ResidentReply {
        sender,
        admission: a.clone(),
        control: state.control.clone(),
        expected_request_sha256: expected_sha.clone(),
        expected_prompt_ids: expected_prompt.clone(),
    };
    let work = ResidentWork {
        planned_chat: plan,
        response_id: a.request_id.clone(),
        admitted: a.admitted,
        deadline: a.deadline,
        raw_request_path: a.path.join("request-body.json"),
        cancelled: a.cancelled.clone(),
        reply,
        streaming: None,
    };
    match state.sender.try_send(work) {
        Ok(()) => (),
        Err(mpsc::error::TrySendError::Full(_)) => {
            return a.error(
                StatusCode::TOO_MANY_REQUESTS,
                "native_queue_full",
                "one C1 request is already queued; request was not executed",
            )
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return a.error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native_worker_closed",
                "resident native worker is unavailable; request was not executed",
            )
        }
    }
    let mut cancel = CancelOnDrop {
        admission: a.clone(),
        control: state.control.clone(),
        armed: true,
    };
    let outcome = match receiver.await {
        Ok(r) => r,
        Err(_) => {
            return a.error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native_worker_lost",
                "native worker reply was lost; no completion authority",
            )
        }
    };
    cancel.armed = false;
    if a.expired() || a.cancelled.load(Ordering::Acquire) {
        a.cancelled.store(true, Ordering::Release);
        state.control.block_until_coordinated_native_recovery();
        return a.error(
            StatusCode::GATEWAY_TIMEOUT,
            "request_deadline_or_cancelled",
            "checked worker evidence retained; no response after deadline/cancellation",
        );
    }
    match outcome {
        WorkerOutcome::Failed(f) => a.error(
            StatusCode::SERVICE_UNAVAILABLE,
            "native_execution_failed",
            &format!("{}: {}", f.class, f.detail),
        ),
        WorkerOutcome::Checked(r) => {
            let matches = r
                .parser_report()
                .is_some_and(|p| p.semantics.original_request_sha256 == expected_sha)
                && r.native_prompt_ids() == expected_prompt
                && r.response().is_none_or(|v| {
                    v.get("id").and_then(Value::as_str) == Some(a.request_id.as_str())
                });
            if !matches || !r.passed() || !r.runtime_terminal_authority_checked() {
                state.control.block_until_coordinated_native_recovery();
                return a.error(
                    StatusCode::BAD_GATEWAY,
                    "checked_native_response_rejected",
                    r.error()
                        .unwrap_or("opaque result or exact request association failed"),
                );
            }
            match r.response() {
                Some(response) => match record(&a.path.join("http-response.json"), response) {
                    Ok(()) => {
                        if a.expired() || a.cancelled.load(Ordering::Acquire) {
                            a.cancelled.store(true, Ordering::Release);
                            state.control.block_until_coordinated_native_recovery();
                            return a.error(StatusCode::GATEWAY_TIMEOUT, "request_deadline_or_cancelled", "response evidence crossed admission deadline/cancellation; no normal response");
                        }
                        (StatusCode::OK, Json(response.clone())).into_response()
                    }
                    Err(e) => a.error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "evidence_failed",
                        &e.to_string(),
                    ),
                },
                None => a.error(
                    StatusCode::BAD_GATEWAY,
                    "no_normal_response",
                    "checked native result had no normal response",
                ),
            }
        }
    }
}
async fn unsupported_atomic(Extension(a): Extension<Admission>, body: Bytes) -> Response {
    let _ = raw_record(&a.path.join("request-body.json"), &body);
    a.error(
        StatusCode::BAD_REQUEST,
        "unsupported_atomic_batch",
        "atomic GPU batching is not implemented by this C1 endpoint; requests were not executed",
    )
}
async fn models(State(state): State<Arc<FrontendState>>) -> Response {
    if !state.control.ready() {
        return wire_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "native_not_ready",
            "resident native worker is not ready",
            "",
        );
    }
    Json(json!({"object":"list","data":[{"id":SERVED_MODEL,"object":"model","owned_by":"deepseek"}]})).into_response()
}
async fn health(State(state): State<Arc<FrontendState>>) -> Response {
    let ready = state.control.ready();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"status":if ready{"ready"}else{"unavailable"},"model":SERVED_MODEL})),
    )
        .into_response()
}

/// Explicit private/loopback listener only. Calling this is a separate parent
/// launch action; module construction and tests never bind or launch it.
pub async fn serve_private(
    address: SocketAddr,
    head: Ipv4Addr,
    router: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    super::lab_network::validate_http_bind(address, head)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{os::unix::fs::PermissionsExt, sync::OnceLock};
    fn bridge() -> Arc<CpuBridge> {
        static B: OnceLock<Arc<CpuBridge>> = OnceLock::new();
        B.get_or_init(||Arc::new(CpuBridge::from_verified_tokenizer(&std::fs::read(
            "test-data/tokenizer.json").unwrap()).unwrap())).clone()
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "native-actor-{}-{}",
                std::process::id(),
                Uuid::new_v4().simple()
            ));
            DirBuilder::new().mode(0o700).create(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> (Temp, Arc<FrontendState>, ResidentInbox) {
        let tmp = Temp::new();
        let (sender, receiver) = mpsc::channel(QUEUED_REQUESTS);
        let control = ResidentControl {
            ready: Arc::new(AtomicBool::new(true)),
        };
        let state = Arc::new(FrontendState {
            bridge: bridge(),
            sender,
            control,
            evidence: tmp.0.clone(),
            streaming_enabled: false,
        });
        (tmp, state, ResidentInbox { receiver })
    }
    fn admission(root: &Path) -> Admission {
        let request_id = format!("chatcmpl-native-{}", Uuid::new_v4().simple());
        let path = root.join(&request_id);
        DirBuilder::new().mode(0o700).create(&path).unwrap();
        let admitted = Instant::now();
        Admission {
            request_id,
            admitted,
            deadline: admitted + WALL,
            path,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
    fn body() -> Value {
        json!({"model":SERVED_MODEL,"messages":[{"role":"user","content":"Answer briefly."}],
        "temperature":0,"top_p":1,"max_tokens":4096,"seed":11,"tool_choice":"auto","chat_template_kwargs":{"thinking":false}})
    }
    #[test]
    fn explicit_bind_scope_rejects_public_wildcard_and_privileged_ports() {
        let head = Ipv4Addr::new(172, 31, 240, 11);
        for address in ["127.0.0.1:46111", "172.31.240.11:46111"] {
            super::super::lab_network::validate_http_bind(address.parse().unwrap(), head).unwrap();
        }
        for address in [
            "0.0.0.0:46111",
            "8.8.8.8:46111",
            "172.31.240.10:46111",
            "198.51.100.11:46111",
            "127.0.0.1:80",
            "[::]:46111",
        ] {
            assert!(super::super::lab_network::validate_http_bind(address.parse().unwrap(), head).is_err());
        }
    }
    #[test]
    fn constructor_is_private_fresh_and_never_claims_native_ready() {
        let temp = Temp::new();
        let path = temp.0.join("frontend");
        let (_router, _inbox, control) = build_frontend(bridge(), path.clone()).unwrap();
        assert!(!control.ready());
        assert!(build_frontend(bridge(), path.clone()).is_err());
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(path.join("frontend-policy.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        fn send<T: Send>() {}
        send::<ResidentWork>();
        send::<ResidentReply>();
    }
    #[tokio::test]
    async fn failed_native_worker_never_returns_a_normal_completion_and_keeps_full4096() {
        let (_tmp, state, mut inbox) = fixture();
        let a = admission(&state.evidence);
        let folder = a.path.clone();
        let raw = serde_json::to_vec(&body()).unwrap();
        let original = raw.clone();
        let ss = state.clone();
        let aa = a.clone();
        let request = tokio::spawn(async move {
            chat_completion(State(ss), Extension(aa), Bytes::from(raw)).await
        });
        let work = tokio::time::timeout(Duration::from_secs(10), inbox.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(work.planned_chat.semantics().output_allowance, 4096);
        assert!(!work.planned_chat.semantics().thinking);
        assert!(work.planned_chat.execution_permit().is_err());
        assert_eq!(work.deadline - work.admitted, WALL);
        assert_eq!(std::fs::read(&work.raw_request_path).unwrap(), original);
        work.reply
            .failed(NativeFailure {
                class: "test_no_gpu".into(),
                detail: "CPU-only fixture; no model execution".into(),
                all_eight_cleanup_confirmed: false,
                cleanup_elapsed_seconds: None,
            })
            .unwrap();
        assert_eq!(
            request.await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(!state.control.ready());
        assert!(folder.join("worker-failure.json").is_file());
        assert!(folder.join("http-error.json").is_file());
    }
    #[tokio::test]
    async fn one_queued_request_is_bounded_and_disconnect_requests_cancel_without_dropping_work() {
        let (_tmp, state, mut inbox) = fixture();
        let a = admission(&state.evidence);
        let cancel = a.cancelled.clone();
        let ss = state.clone();
        let first = tokio::spawn(async move {
            chat_completion(
                State(ss),
                Extension(a),
                Bytes::from(serde_json::to_vec(&body()).unwrap()),
            )
            .await
        });
        for _ in 0..100 {
            if state.sender.capacity() == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(state.sender.capacity(), 0);
        let second = chat_completion(
            State(state.clone()),
            Extension(admission(&state.evidence)),
            Bytes::from(serde_json::to_vec(&body()).unwrap()),
        )
        .await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(state.control.ready());
        first.abort();
        let _ = first.await;
        assert!(cancel.load(Ordering::Acquire));
        assert!(!state.control.ready());
        let work = inbox.recv().await.unwrap();
        assert!(work.cancellation_requested());
        // The cancelled handler did not destroy the worker-owned queued item.
        // A real executor must coordinate all8 cleanup before recovery.
        work.reply
            .failed(NativeFailure {
                class: "cancel_before_gpu".into(),
                detail: "CPU-only queued cancellation".into(),
                all_eight_cleanup_confirmed: false,
                cleanup_elapsed_seconds: None,
            })
            .unwrap();
    }
    #[tokio::test]
    async fn unsupported_stream_atomic_and_expired_requests_never_enter_native_queue() {
        let (_tmp, state, mut inbox) = fixture();
        let mut stream = body();
        stream["stream"] = json!(true);
        let r = chat_completion(
            State(state.clone()),
            Extension(admission(&state.evidence)),
            Bytes::from(serde_json::to_vec(&stream).unwrap()),
        )
        .await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let r = unsupported_atomic(
            Extension(admission(&state.evidence)),
            Bytes::from(serde_json::to_vec(&json!({"messages":[body(),body()]})).unwrap()),
        )
        .await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let mut a = admission(&state.evidence);
        a.admitted = Instant::now() - Duration::from_secs(181);
        a.deadline = a.admitted + WALL;
        let r = chat_completion(
            State(state),
            Extension(a),
            Bytes::from(serde_json::to_vec(&body()).unwrap()),
        )
        .await;
        assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(inbox.receiver.try_recv().is_err());
    }
    #[tokio::test]
    async fn rejected_stream_queue_full_does_not_cancel_other_native_work() {
        let (_temp, state, _inbox) = fixture();
        state.control.mark_ready_after_coordinated_native_reset();
        let permit = state.sender.try_reserve().unwrap();
        let a = admission(&state.evidence);
        let path = a.path.clone();
        let mut request = body();
        request["stream"] = json!(true);
        let response = stream_completion(state.clone(), a, request).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(state.control.ready());
        assert!(!path.join("stream-policy.json").exists());
        drop(permit);
    }
}

#[cfg(test)]
pub(crate) fn CPU_fixture_control_ready() -> ResidentControl {
    let control = ResidentControl {
        ready: Arc::new(AtomicBool::new(false)),
    };
    control.mark_ready_after_coordinated_native_reset();
    control
}
