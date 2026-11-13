//! Experimental buffered-only atomic HTTP actor + concrete resident owner loop.
//! It owns no CUDA thread: exactly one initializing model thread/rank executes
//! real packed groups. Client failure is fatal for this initial batch profile;
//! no C1 cancellation token, fakeEOS, partial200, liveSSE or ready-on-error path.
use super::{
    atomic_request_admission::{CpuSlotBook, PreparedAtomicRequest, ProposedBatchDeployment},
    gpu_batch_model::GpuBatchModel,
    gpu_batch_request_session::{
        BatchSessionRelease, CheckedBatchRequestCompletion, GpuBatchRequestSession,
    },
    gpu_collectives::GpuCollectives,
    native_api_contract::{CpuBridge, ATOMIC_HEADER, ATOMIC_ROUTE, SERVED_MODEL},
    native_batch_dispatch::{
        BatchDispatchConfig, HeadBatchDispatch, PeerBatchDispatch, PreparedBatchWave,
    },
    native_batch_response::{self, BufferedBatchResponse},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use uuid::Uuid;
const MAX_BODY: usize = 1024 * 1024;
const WALL: Duration = Duration::from_secs(180);
/// Strict duplicate-key decoding preserves source object order. No map sorting,
/// message cleanup, inserted sampling field or tokenizer rewrite is performed.
struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON without duplicate object fields")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(json!(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(json!(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(json!(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite JSON"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v.into())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v)))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = vec![];
                while let Some(v) = a.next_element::<UniqueJson>()? {
                    out.push(v.0);
                }
                Ok(UniqueJson(Value::Array(out)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some(key) = a.next_key::<String>()? {
                    if out.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate JSON field{key}")));
                    }
                    out.insert(key, a.next_value::<UniqueJson>()?.0);
                }
                Ok(UniqueJson(Value::Object(out)))
            }
        }
        d.deserialize_any(V)
    }
}
pub fn parse_original_body(raw: &[u8]) -> Result<Value> {
    ensure!(raw.len() <= MAX_BODY, "atomic body limit1MiB");
    let UniqueJson(v) = serde_json::from_slice(raw)?;
    Ok(v)
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    use std::{fs::OpenOptions, io::Write, os::unix::fs::OpenOptionsExt};
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn request_dir(root: &Path, id: &str) -> Result<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    ensure!(
        id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "safe private request ID"
    );
    let dir = root.join(id);
    std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
    Ok(dir)
}
pub struct BatchHttpControl {
    ready: AtomicBool,
    fatal: AtomicBool,
    stopping: AtomicBool,
}
impl BatchHttpControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            ready: AtomicBool::new(false),
            fatal: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
        })
    }
    fn block(&self) {
        self.fatal.store(true, Ordering::SeqCst);
        self.ready.store(false, Ordering::SeqCst);
    }
    fn ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
            && !self.fatal.load(Ordering::SeqCst)
            && !self.stopping.load(Ordering::SeqCst)
    }
    fn publish_ready(&self) -> Result<()> {
        ensure!(
            !self.fatal.load(Ordering::SeqCst) && !self.stopping.load(Ordering::SeqCst),
            "batch fatal/stop is sticky"
        );
        self.ready.store(true, Ordering::SeqCst);
        Ok(())
    }
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.ready.store(false, Ordering::SeqCst);
    }
}
struct Work {
    prepared: PreparedAtomicRequest,
    id: String,
    admitted: Instant,
    cancelled: Arc<AtomicBool>,
    evidence: PathBuf,
    reply: oneshot::Sender<Result<BufferedBatchResponse, String>>,
}
#[derive(Clone)]
pub struct BatchHttpActor {
    bridge: Arc<CpuBridge>,
    sender: mpsc::SyncSender<Work>,
    control: Arc<BatchHttpControl>,
    evidence: Arc<PathBuf>,
    service_deadline: Instant,
}
pub struct BatchInbox {
    receiver: mpsc::Receiver<Work>,
}
impl BatchHttpActor {
    pub fn new(
        bridge: Arc<CpuBridge>,
        control: Arc<BatchHttpControl>,
        evidence: PathBuf,
        service_deadline: Instant,
    ) -> Result<(Self, BatchInbox)> {
        ensure!(
            evidence.is_dir(),
            "owned private evidence directory must exist"
        );
        let (sender, receiver) = mpsc::sync_channel(1);
        Ok((
            Self {
                bridge,
                sender,
                control,
                evidence: Arc::new(evidence),
                service_deadline,
            },
            BatchInbox { receiver },
        ))
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route(ATOMIC_ROUTE, post(batch))
            .with_state(self.clone())
    }
    pub async fn serve(
        self,
        bind: SocketAddr,
        head: std::net::Ipv4Addr,
        shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> Result<()> {
        super::lab_network::validate_http_bind(bind, head)?;
        let listener = tokio::net::TcpListener::bind(bind).await?;
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await?;
        Ok(())
    }
}
async fn health(State(a): State<BatchHttpActor>) -> Response {
    if a.control.ready() {
        (
            StatusCode::OK,
            Json(json!({"status":"ready","experimental_atomic_batch":true})),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"unavailable"})),
        )
            .into_response()
    }
}
async fn models() -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":SERVED_MODEL,"object":"model"}]}))
}
fn error(status: StatusCode, detail: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":detail,"type":"native_atomic_batch_error"}})),
    )
        .into_response()
}
struct CancelOnDrop {
    cancelled: Arc<AtomicBool>,
    control: Arc<BatchHttpControl>,
    armed: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::SeqCst);
            self.control.block();
        }
    }
}
async fn batch(State(a): State<BatchHttpActor>, request: Request<Body>) -> Response {
    let admitted = Instant::now();
    let deadline = tokio::time::Instant::from_std(admitted + WALL);
    if admitted + WALL > a.service_deadline {
        return error(StatusCode::SERVICE_UNAVAILABLE, "full original180s request budget cannot fit this bounded service window; no shortened admission");
    }
    if !a.control.ready() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "resident atomic worker is not at a checked reusable boundary",
        );
    }
    let headers = request
        .headers()
        .get_all(ATOMIC_HEADER)
        .iter()
        .collect::<Vec<_>>();
    if headers.len() != 1 || headers[0].as_bytes() != b"1" {
        return error(
            StatusCode::BAD_REQUEST,
            "exact X-TP8-Atomic-Admission: 1 required",
        );
    }
    let raw = match tokio::time::timeout_at(deadline, to_bytes(request.into_body(), MAX_BODY)).await
    {
        Ok(Ok(raw)) => raw,
        Ok(Err(_)) => {
            return error(
                StatusCode::BAD_REQUEST,
                "bounded original JSON body required",
            )
        }
        Err(_) => {
            return error(
                StatusCode::REQUEST_TIMEOUT,
                "original admission deadline reading body",
            )
        }
    };
    let body = match parse_original_body(&raw) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let prepared = match PreparedAtomicRequest::prepare(&a.bridge, ATOMIC_ROUTE, Some("1"), &body) {
        Ok(p)
            if p.members()
                .iter()
                .all(|m| (1..=256).contains(&m.semantics().output_allowance)) =>
        {
            p
        }
        Ok(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "full original allowance must be1..256 in qualified512-context profile",
            )
        }
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let id = format!("chatcmpl-atomic-{}", Uuid::new_v4());
    let evidence = match request_dir(&a.evidence, &id) {
        Ok(d) => d,
        Err(e) => {
            a.control.block();
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("evidence:{e:#}"),
            );
        }
    };
    if let Err(e) = write_new(
        &evidence.join("request.json"),
        &json!({"id":id,"original_wire_utf8":String::from_utf8_lossy(&raw),
        "original_wire_sha256":format!("{:x}",Sha256::digest(&raw)),"original_body":body,"prepared_digest":prepared.digest(),
        "effective_semantics":prepared.member_semantics(),"native_prompt_ids":prepared.members().iter().map(|m|m.prompt_ids()).collect::<Vec<_>>(),
        "full_native_wall_millis":180000,"historical_matched_client_timeout_seconds":100,"stream":false}),
    ) {
        a.control.block();
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("evidence:{e:#}"),
        );
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let (reply, receive) = oneshot::channel();
    let work = Work {
        prepared,
        id: id.clone(),
        admitted,
        cancelled: cancelled.clone(),
        evidence: evidence.clone(),
        reply,
    };
    if Instant::now() >= admitted + WALL {
        return error(
            StatusCode::REQUEST_TIMEOUT,
            "original atomic admission expired before queue; no slot effects",
        );
    }
    if a.sender.try_send(work).is_err() {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "one bounded atomic group queue slot already occupied",
        );
    }
    let mut guard = CancelOnDrop {
        cancelled,
        control: a.control.clone(),
        armed: true,
    };
    let outcome = tokio::time::timeout_at(deadline, receive).await;
    let response = match outcome {
        Ok(Ok(Ok(value))) => {
            if Instant::now() >= admitted + WALL
                || guard.cancelled.load(Ordering::SeqCst)
                || a.control.fatal.load(Ordering::SeqCst)
            {
                a.control.block();
                error(
                    StatusCode::GATEWAY_TIMEOUT,
                    "atomic normal response expired/cancelled after checked worker result",
                )
            } else if value.response()["id"] != id {
                a.control.block();
                error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "opaque batch response belongs to a different admission",
                )
            } else if let Err(e) = write_new(&evidence.join("http-response.json"), value.response())
            {
                a.control.block();
                error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("response evidence:{e:#}"),
                )
            } else if Instant::now() >= admitted + WALL || guard.cancelled.load(Ordering::SeqCst) {
                a.control.block();
                error(
                    StatusCode::GATEWAY_TIMEOUT,
                    "atomic original deadline after response evidence",
                )
            } else {
                (StatusCode::OK, Json(value.into_response())).into_response()
            }
        }
        Ok(Ok(Err(e))) => {
            a.control.block();
            error(StatusCode::INTERNAL_SERVER_ERROR, &e)
        }
        Ok(Err(_)) => {
            a.control.block();
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native atomic owner stopped",
            )
        }
        Err(_) => {
            a.control.block();
            guard.cancelled.store(true, Ordering::SeqCst);
            error(
                StatusCode::GATEWAY_TIMEOUT,
                "atomic original180s deadline; no partial completion",
            )
        }
    };
    guard.armed = false;
    response
}
fn deployment(previous: &CheckedBatchRequestCompletion) -> Result<ProposedBatchDeployment> {
    let m = previous.model_identity();
    let runtime = m.runtime.as_ref().context("batch runtime missing")?;
    Ok(ProposedBatchDeployment {
        boot: m.boot,
        global_checkpoint_geometry_checksum_sha256: m.global_checkpoint_geometry_binding.clone(),
        execution_snapshot_sha256: runtime.execution_snapshot_sha256.clone(),
        engine_binary_sha256: runtime.engine_binary_sha256.clone(),
        frequency: m.frequency.clone(),
    })
}
fn release(config: &BatchDispatchConfig, sequence: usize) -> Result<BatchSessionRelease> {
    Ok(BatchSessionRelease {
        peers: config.peers,
        first_phase_port: config.release_port(sequence)?,
        timeout_millis: 30000,
    })
}
fn emit(value: &Value) -> Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
/// Called on the original CUDA owner thread after an actual checked bootstrap
/// terminal/reset. CPU queue tokens never substitute for that opaque boundary.
pub fn run_head_worker(
    model: &mut GpuBatchModel,
    comm: &mut GpuCollectives,
    mut previous: CheckedBatchRequestCompletion,
    bridge: &CpuBridge,
    config: BatchDispatchConfig,
    inbox: BatchInbox,
    control: Arc<BatchHttpControl>,
    service_deadline: Instant,
) -> Result<CheckedBatchRequestCompletion> {
    let result = (|| {
        let mut dispatch = HeadBatchDispatch::bind(config.clone(), &previous, service_deadline)?;
        let mut book = CpuSlotBook::new(config.boot, 16)?;
        let mut generations = vec![0; 16];
        for l in previous.cleanup_reset().lanes() {
            generations[l.slot] = l.generation;
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mut sequence = 2;
        control.publish_ready()?;
        loop {
            if Instant::now() >= service_deadline || control.stopping.load(Ordering::SeqCst) {
                break;
            }
            if control.fatal.load(Ordering::SeqCst) {
                anyhow::bail!("fatal atomic actor requires whole-cluster teardown");
            }
            let work = match inbox.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(w) => w,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            control.ready.store(false, Ordering::SeqCst);
            let result = (|| {
                ensure!(
                    !work.cancelled.load(Ordering::SeqCst)
                        && Instant::now() < work.admitted + WALL
                        && work.admitted + WALL <= service_deadline,
                    "queued original atomic admission expired/cancelled"
                );
                let keys = (0..work.prepared.members().len())
                    .map(|slot| {
                        Ok(LaneKey {
                            boot: config.boot,
                            slot,
                            generation: generations[slot]
                                .checked_add(1)
                                .context("batch physical generation overflow")?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let wave = PreparedBatchWave::prepare(
                    work.prepared,
                    &mut book,
                    deployment(&previous)?,
                    keys,
                    work.id.clone(),
                    work.admitted,
                    sequence,
                    &previous,
                )?;
                model.validate_group_reset(&wave.binding.lanes(), comm)?;
                dispatch.begin(&wave, &previous)?;
                let reset = model.reset_group_checked(&wave.binding.lanes(), comm)?;
                emit(
                    &json!({"event":"ATOMIC_GROUP_RESET","response_id":work.id,"sequence":sequence,"reset":reset,"group":wave.binding.as_ref()}),
                )?;
                let session = GpuBatchRequestSession::rearm(
                    wave.plan.clone(),
                    reset,
                    &previous,
                    model,
                    comm,
                    release(&config, sequence)?,
                    &wave.owned,
                    &book,
                )?;
                let completed=session.run(model,comm,|proof|emit(&json!({"event":"ATOMIC_CHECKED_MODEL_STEP","response_id":work.id,"sequence":sequence,"proof":proof})),||{
                    ensure!(!work.cancelled.load(Ordering::SeqCst)&&!control.fatal.load(Ordering::SeqCst),"batch client disappeared/internal actor failure; fatal all8 teardown");Ok(())})?;
                emit(
                    &json!({"event":"ATOMIC_GROUP_COMPLETION","response_id":work.id,"sequence":sequence,"completion":completed}),
                )?;
                dispatch.finish(&completed, work.admitted)?;
                book.release_cpu_ownership(&wave.owned)?;
                for key in completed.cleanup_reset().lanes() {
                    generations[key.slot] = key.generation;
                }
                let response=rt.block_on(native_batch_response::buffered_response(bridge,wave.owned.into_prepared(),&completed,work.admitted,
                    |report|emit(&json!({"event":"ATOMIC_MEMBER_PARSED","response_id":work.id,"sequence":sequence,"report":report}))))?;
                write_new(
                    &work.evidence.join("native-response-audit.json"),
                    &serde_json::to_value(&response)?,
                )?;
                ensure!(
                    !work.cancelled.load(Ordering::SeqCst) && Instant::now() < work.admitted + WALL,
                    "original atomic deadline/client disappeared after official parsing"
                );
                Ok((completed, response))
            })();
            match result {
                Ok((completed, response)) => {
                    previous = completed;
                    sequence += 1;
                    control.publish_ready()?;
                    if work.reply.send(Ok(response)).is_err() {
                        control.block();
                        anyhow::bail!("client detached after actual atomic completion");
                    }
                }
                Err(error) => {
                    control.block();
                    let detail = format!("{error:#}");
                    let _ = write_new(
                        &work.evidence.join("failure.json"),
                        &json!({"status":"FAIL","error":detail,"partial_normal_response":false}),
                    );
                    let _ = work.reply.send(Err(detail.clone()));
                    anyhow::bail!(detail);
                }
            }
        }
        control.stop();
        dispatch.stop(&previous, service_deadline + Duration::from_secs(30))?;
        Ok(previous)
    })();
    if result.is_err() {
        control.block();
        model.poison(comm);
    }
    result
}
pub fn run_peer_worker(
    model: &mut GpuBatchModel,
    comm: &mut GpuCollectives,
    mut previous: CheckedBatchRequestCompletion,
    bridge: &CpuBridge,
    config: BatchDispatchConfig,
    service_deadline: Instant,
) -> Result<CheckedBatchRequestCompletion> {
    let result = (|| {
        let mut dispatch =
            PeerBatchDispatch::connect(config.clone(), comm.rank(), &previous, service_deadline)?;
        let mut book = CpuSlotBook::new(config.boot, 16)?;
        loop {
            let Some(wave) =
                dispatch.receive(bridge, &mut book, &previous, model, comm, service_deadline)?
            else {
                break;
            };
            let reset = model.reset_group_checked(&wave.binding.lanes(), comm)?;
            emit(
                &json!({"event":"ATOMIC_GROUP_RESET","rank":comm.rank(),"sequence":wave.sequence,"reset":reset,"group":wave.binding.as_ref()}),
            )?;
            let session = GpuBatchRequestSession::rearm(
                wave.plan.clone(),
                reset,
                &previous,
                model,
                comm,
                release(&config, wave.sequence)?,
                &wave.owned,
                &book,
            )?;
            let completed=session.run(model,comm,|proof|emit(&json!({"event":"ATOMIC_CHECKED_MODEL_STEP","sequence":wave.sequence,"proof":proof})),||Ok(()))?;
            emit(
                &json!({"event":"ATOMIC_GROUP_COMPLETION","sequence":wave.sequence,"completion":completed}),
            )?;
            dispatch.finish(&completed, wave.admitted)?;
            book.release_cpu_ownership(&wave.owned)?;
            previous = completed;
        }
        Ok(previous)
    })();
    if result.is_err() {
        model.poison(comm);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    fn bridge() -> Arc<CpuBridge> {
        static BRIDGE: OnceLock<Arc<CpuBridge>> = OnceLock::new();
        BRIDGE.get_or_init(||Arc::new(CpuBridge::from_verified_tokenizer(&std::fs::read(
            "test-data/tokenizer.json").unwrap()).unwrap())).clone()
    }
    fn body() -> Value {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_actual_atomic_requests.json"
        ))
        .unwrap();
        fixture["cases"][2]["request"].clone()
    }
    struct PrivateTemporary(PathBuf);
    impl PrivateTemporary {
        fn new() -> Self {
            use std::os::unix::fs::DirBuilderExt;
            let p = std::env::temp_dir().join(format!("native-batch-CPU-{}", Uuid::new_v4()));
            std::fs::DirBuilder::new().mode(0o700).create(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for PrivateTemporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn raw_json_rejects_duplicate_nested_keys_and_preserves_original_order_unicode_null() {
        let raw=br#"{"model":"deepseek-v4.1-flash","messages":[[{"role":"user","content":null}]],"z":1,"a":2}"#;
        let parsed = parse_original_body(raw).unwrap();
        assert_eq!(serde_json::to_vec(&parsed).unwrap(), raw);
        for raw in [
            br#"{"messages":[],"messages":[]}"#.as_slice(),
            br#"{"nested":[{"a":1,"a":1}]}"#,
            br#"{"a":NaN}"#,
            br#"{} trailing"#,
        ] {
            assert!(parse_original_body(raw).is_err());
        }
        assert!(parse_original_body(&vec![b' '; MAX_BODY + 1]).is_err());
    }
    #[tokio::test]
    async fn atomic_header_and_invalid_last_member_never_reach_queue() {
        let temp = PrivateTemporary::new();
        let control = BatchHttpControl::new();
        let (actor, inbox) = BatchHttpActor::new(
            bridge(),
            control.clone(),
            temp.0.clone(),
            Instant::now() + Duration::from_secs(600),
        )
        .unwrap();
        control.publish_ready().unwrap();
        for mode in 0..5 {
            let mut payload = body();
            if mode == 3 {
                payload["messages"][15] =
                    json!([{"role":"tool","tool_call_id":"orphan","content":"invalid"}]);
            }
            if mode == 4 {
                payload["max_tokens"] = 257.into();
            }
            let mut request = Request::builder().method("POST").uri(ATOMIC_ROUTE);
            if mode != 0 {
                request = request.header(ATOMIC_HEADER, if mode == 1 { "true" } else { "1" });
            }
            if mode == 2 {
                request = request.header(ATOMIC_HEADER, "1");
            }
            let response = batch(
                State(actor.clone()),
                request
                    .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                    .unwrap(),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "mode{mode}");
            assert!(matches!(
                inbox.receiver.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            assert!(control.ready());
        }
    }
    #[tokio::test]
    async fn actual_original_C16_body_is_queued_once_without_terminal_authority_and_disconnect_is_fatal(
    ) {
        let temp = PrivateTemporary::new();
        let control = BatchHttpControl::new();
        let (actor, inbox) = BatchHttpActor::new(
            bridge(),
            control.clone(),
            temp.0.clone(),
            Instant::now() + Duration::from_secs(600),
        )
        .unwrap();
        control.publish_ready().unwrap();
        let original = body();
        let request = Request::builder()
            .method("POST")
            .uri(ATOMIC_ROUTE)
            .header(ATOMIC_HEADER, "1")
            .body(Body::from(serde_json::to_vec(&original).unwrap()))
            .unwrap();
        let task = tokio::spawn(batch(State(actor), request));
        let work = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match inbox.receiver.try_recv() {
                    Ok(work) => break work,
                    Err(mpsc::TryRecvError::Empty) => tokio::task::yield_now().await,
                    Err(_) => panic!("queue unexpectedly disconnected"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(work.prepared.original_request(), &original);
        assert_eq!(work.prepared.members().len(), 16);
        assert!(work
            .prepared
            .members()
            .iter()
            .all(|m| m.prompt_ids().len() == 75 && m.semantics().output_allowance == 256));
        assert!(work.prepared.execution_permit().is_err());
        task.abort();
        let _ = task.await;
        assert!(work.cancelled.load(Ordering::SeqCst));
        assert!(!control.ready());
        assert!(control.publish_ready().is_err());
        assert!(work
            .reply
            .send(Err("CPU-only no native execution".into()))
            .is_err());
    }
    #[test]
    fn initial_fatal_and_stop_cannot_publish_admission_or_normal_response() {
        let c = BatchHttpControl::new();
        assert!(!c.ready());
        c.publish_ready().unwrap();
        assert!(c.ready());
        c.block();
        assert!(!c.ready());
        assert!(c.publish_ready().is_err());
        let c = BatchHttpControl::new();
        c.stop();
        assert!(c.publish_ready().is_err());
        assert!(!c.ready());
    }
}
