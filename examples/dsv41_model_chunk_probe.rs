//! PRIVATE full8rank resident C1 native chunk controls. No HTTP/tool task or
//! performance claim; natural nativeEOS/cap only through opaque session proof.
//! Parent owns launches and strict1200s/all-rank cleanup/thermal/memory guards.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    config::{Dsv41Config, MODEL_REVISION},
    engram::CompressedTokenMap,
    gpu_collectives::{GpuCollectives, Program},
    gpu_model_chunk::{
        global_checkpoint_binding, official_global_checkpoint_binding, GpuModelChunk,
        ModelChunkKernels, ModelChunkRuntimeBinding,
    },
    gpu_model_weights::{plan_rank_load, RankModelWeights},
    gpu_reader::{GpuWeightBudget, WeightLoadLimits},
    gpu_request_session::{GpuRequestSession, RequestStopReason, SessionProgress},
    model_session_plan::{
        CpuRequestPlan, GreedySemantics, RequestBinding, RequestBudget, StepRole, StopPolicy,
    },
    position::{FrequencyBackend, RopeProfile, VerifiedFrequencyTable},
    reader::CheckpointReader,
    request_session_release::RequestReleaseConfig,
    scheduler::LaneKey,
    weights::{build_rank_manifest, WoaReconstructionPolicy},
};
use gb10_inference::net::TpLink;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read, Write},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const WORLD: usize = 8;
const WALL_SECONDS: u64 = 1200;
const LOAD_READY_SECONDS: u64 = 900;
const MODEL_READY_SECONDS: u64 = 300;
const GIB: u64 = 1024 * 1024 * 1024;
struct Args {
    rank: usize,
    ips: Vec<IpAddr>,
    port: u16,
    release_port: u16,
    request_release_port: u16,
    dev: String,
    gid: i32,
    proxy_core: i32,
    main_core: i32,
    model_dir: PathBuf,
    freq_plain: PathBuf,
    freq_yarn: PathBuf,
    frequency_backend: FrequencyBackend,
    compressed_map: PathBuf,
    requests_file: PathBuf,
    metadata_only: bool,
    global_checkpoint_binding: Option<String>,
    execution_snapshot_sha256: String,
    engine_binary_sha256: String,
    boot: Uuid,
    load_release: SocketAddr,
}
fn valid_sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
fn args() -> Result<Args> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(raw.len() % 2 == 0, "explicit --flag value pairs required");
    let allowed = [
        "--rank",
        "--world",
        "--peer-ips",
        "--port",
        "--release-port",
        "--request-release-port",
        "--dev",
        "--gid",
        "--proxy-core",
        "--main-core",
        "--model-dir",
        "--freq-plain",
        "--freq-yarn",
        "--frequency-backend",
        "--compressed-map",
        "--requests-file",
        "--metadata-only",
        "--global-checkpoint-binding",
        "--execution-snapshot-sha256",
        "--engine-binary-sha256",
        "--boot",
        "--load-release-address",
        "--load-release-port",
    ];
    let mut f = BTreeMap::new();
    for p in raw.chunks_exact(2) {
        ensure!(
            allowed.contains(&p[0].as_str()) && f.insert(p[0].clone(), p[1].clone()).is_none(),
            "unknown/repeated chunk probe argument {}",
            p[0]
        );
    }
    let v = |k: &str| f.get(k).with_context(|| format!("missing {k}"));
    ensure!(
        v("--world")?.parse::<usize>()? == 8,
        "native chunk caller requires world8"
    );
    let rank = v("--rank")?.parse::<usize>()?;
    let ips = v("--peer-ips")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<IpAddr>, _>>()?;
    ensure!(
        rank < 8 && ips.len() == 8 && ips.iter().collect::<BTreeSet<_>>().len() == 8,
        "eight distinct rank-ordered RoCE IPs required"
    );
    let port = v("--port")?.parse::<u16>()?;
    let release_port = v("--release-port")?.parse::<u16>()?;
    let load_port = v("--load-release-port")?.parse::<u16>()?;
    let request_release_port = v("--request-release-port")?.parse::<u16>()?;
    let ports = gb10_inference::net::nway_link_ports(port, 8);
    ensure!(
        [port, release_port, load_port].iter().all(|p| *p >= 1024)
            && [port, release_port, load_port]
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                == 3
            && ports.iter().all(|p| *p <= u16::MAX as u32)
            && !ports.contains(&(release_port as u32))
            && !ports.contains(&(load_port as u32)),
        "bootstrap/release/load ports overlap or overflow"
    );
    ensure!(
        (20000..=63993).contains(&request_release_port)
            && ips.iter().all(|ip| matches!(ip, IpAddr::V4(_)))
            && (request_release_port..=request_release_port + 7).all(|p| ![
                port,
                release_port,
                load_port
            ]
            .contains(&p)
                && !ports.contains(&(p as u32))),
        "IPv4 and distinct reserved request-release ports required"
    );
    let load_ip = v("--load-release-address")?.parse::<IpAddr>()?;
    ensure!(
        !load_ip.is_unspecified() && !load_ip.is_multicast(),
        "explicit controller management address required"
    );
    let gid = v("--gid")?.parse::<i32>()?;
    ensure!(
        gid >= 0 && !v("--dev")?.is_empty(),
        "explicit HCA/GID required"
    );
    ensure!(
        valid_sha(v("--execution-snapshot-sha256")?) && valid_sha(v("--engine-binary-sha256")?),
        "snapshot/ELF identities must be SHA256"
    );
    let boot = Uuid::parse_str(v("--boot")?)?;
    ensure!(!boot.is_nil(), "nonnull boot required");
    Ok(Args {
        rank,
        ips,
        port,
        release_port,
        request_release_port,
        dev: v("--dev")?.clone(),
        gid,
        proxy_core: f
            .get("--proxy-core")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(-1),
        main_core: f
            .get("--main-core")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(-1),
        model_dir: v("--model-dir")?.into(),
        freq_plain: v("--freq-plain")?.into(),
        freq_yarn: v("--freq-yarn")?.into(),
        frequency_backend: FrequencyBackend::parse(v("--frequency-backend")?)?,
        compressed_map: v("--compressed-map")?.into(),
        requests_file: v("--requests-file")?.into(),
        metadata_only: f
            .get("--metadata-only")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(false),
        global_checkpoint_binding: f.get("--global-checkpoint-binding").cloned(),
        execution_snapshot_sha256: v("--execution-snapshot-sha256")?.clone(),
        engine_binary_sha256: v("--engine-binary-sha256")?.clone(),
        boot,
        load_release: SocketAddr::new(load_ip, load_port),
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    Diagnostic,
    FullBudgetNativeRequest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sampling {
    temperature: f64,
    top_p: f64,
    presence_penalty: f64,
    frequency_penalty: f64,
    seed: Option<u64>,
    thinking: bool,
    effective_reasoning_score: Option<u8>,
    speculation: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRequest {
    id: String,
    scope: Scope,
    prompt_native_ids: Vec<u32>,
    output_allowance: usize,
    context_tokens: usize,
    wall_millis: u64,
    sampling: Sampling,
    user_stop_sequences: Vec<String>,
    expected_output_native_ids: Option<Vec<u32>>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestPack {
    schema_version: u32,
    model_revision: String,
    requests: Vec<NativeRequest>,
}
fn read_pack(a: &Args) -> Result<(RequestPack, String)> {
    let raw = bounded_file(&a.requests_file, 1024 * 1024)?;
    let p: RequestPack = serde_json::from_slice(&raw)?;
    validate_pack(&p)?;
    Ok((p, sha(&raw)))
}
fn validate_pack(p: &RequestPack) -> Result<()> {
    ensure!(
        p.schema_version == 1
            && p.model_revision == MODEL_REVISION
            && (1..=8).contains(&p.requests.len()),
        "exact model/schema and1..8 resident diagnostic requests required"
    );
    let mut ids = BTreeSet::new();
    for r in &p.requests {
        ensure!(
            !r.id.is_empty()
                && r.id.len() <= 128
                && !r.id.chars().any(char::is_control)
                && ids.insert(&r.id),
            "distinct bounded request IDs required"
        );
        ensure!(
            (1..=8191).contains(&r.prompt_native_ids.len())
                && r.prompt_native_ids.iter().all(|id| *id < 129280)
                && (1..=4096).contains(&r.output_allowance)
                && r.context_tokens <= 8192
                && r.prompt_native_ids
                    .len()
                    .checked_add(r.output_allowance)
                    .is_some_and(|n| n <= r.context_tokens)
                && r.wall_millis > 0
                && r.wall_millis <= WALL_SECONDS * 1000,
            "full declared native input/output/context/wall bounds; no clipping"
        );
        let s = &r.sampling;
        ensure!(
            s.temperature.is_finite()
                && s.temperature == 0.
                && s.top_p == 1.
                && s.presence_penalty == 0.
                && s.frequency_penalty == 0.
                && !s.speculation
                && s.effective_reasoning_score
                    .is_none_or(|n| (1..=100).contains(&n)),
            "explicit greedy sampling only; no coercion"
        );
        ensure!(
            r.user_stop_sequences.is_empty(),
            "native-only caller has no official parser stop authority; user stops unsupported"
        );
        if matches!(&r.scope, Scope::Diagnostic) {
            ensure!([1,5,32].contains(&r.output_allowance),"diagnostics must predeclare1/5/32; original full-budget requests use their own scope");
        }
        if let Some(expected) = &r.expected_output_native_ids {
            ensure!(
                expected.len() == r.output_allowance && expected.iter().all(|id| *id < 129280),
                "reference must cover the complete declared diagnostic allowance"
            );
        }
    }
    Ok(())
}
fn native_sha(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn plan(
    a: &Args,
    r: &NativeRequest,
    pack_sha: &str,
    global_sha: &str,
    operation: u64,
    step: u64,
    epoch: u64,
    generation: u64,
) -> Result<Arc<CpuRequestPlan>> {
    let original = sha(&serde_json::to_vec(&(
        "native-resident-request-v1",
        pack_sha,
        r,
    ))?);
    let s = &r.sampling;
    Ok(Arc::new(CpuRequestPlan::new(
        RequestBinding {
            model_revision: MODEL_REVISION.into(),
            global_checkpoint_geometry_checksum_sha256: global_sha.into(),
            execution_snapshot_sha256: a.execution_snapshot_sha256.clone(),
            engine_binary_sha256: a.engine_binary_sha256.clone(),
            original_request_sha256: original,
            tokenizer_sha256: gb10_inference::dsv41::engram::TOKENIZER_SHA256.into(),
            toolkit_revision: gb10_inference::dsv41::protocol::TOOLKIT_REVISION.into(),
            frequency: a.frequency_backend.identity(),
            sampling: GreedySemantics {
                temperature: s.temperature,
                top_p: s.top_p,
                presence_penalty: s.presence_penalty,
                frequency_penalty: s.frequency_penalty,
                seed: s.seed,
                thinking: s.thinking,
                effective_reasoning_score: s.effective_reasoning_score,
                speculation: s.speculation,
            },
            stop: StopPolicy {
                eos_native_id: 1,
                user_sequences: r.user_stop_sequences.clone(),
            },
            budget: RequestBudget {
                context_tokens: r.context_tokens,
                output_allowance: r.output_allowance,
                wall_millis: r.wall_millis,
            },
            lane: LaneKey {
                boot: a.boot,
                slot: 0,
                generation,
            },
        },
        r.prompt_native_ids.clone(),
        operation,
        step,
        epoch,
    )?))
}
fn emit<T: Serialize>(v: &T) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, v)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn bounded_file(path: &PathBuf, maximum: usize) -> Result<Vec<u8>> {
    let n = std::fs::metadata(path)?.len();
    ensure!(n <= maximum as u64, "metadata file exceeds declared bound");
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() == n as usize, "metadata changed while reading");
    Ok(bytes)
}

fn controller_gate(a: &Args, ready: Value, deadline: Instant) -> Result<()> {
    emit(&ready)?;
    let mut stream = loop {
        ensure!(
            Instant::now() < deadline,
            "controller startup release deadline"
        );
        match TcpStream::connect_timeout(&a.load_release, Duration::from_secs(1)) {
            Ok(s) => break s,
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    };
    stream.set_read_timeout(Some(deadline.saturating_duration_since(Instant::now())))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    serde_json::to_writer(&mut stream, &ready)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let mut line = String::new();
    BufReader::new(&mut stream)
        .take(16 * 1024)
        .read_line(&mut line)?;
    ensure!(
        line.ends_with('\n') && line.len() < 16 * 1024,
        "bounded controller release JSONL required"
    );
    let ack: Value = serde_json::from_str(&line)?;
    let release = if ready["event"] == "WEIGHTS_READY" {
        "RELEASE_WEIGHTS"
    } else {
        "RELEASE_MODEL"
    };
    ensure!(
        ack["event"] == release
            && ack["rank"] == a.rank
            && ack["world"] == 8
            && ack["boot"] == a.boot.to_string()
            && ack["program_sha256"] == ready["program_sha256"]
            && ack["input_ids_sha256"] == ready["input_ids_sha256"]
            && ack["frequency_profile"] == ready["frequency_profile"]
            && ack["request_pack_sha256"] == ready["request_pack_sha256"]
            && ack["execution_snapshot_sha256"] == ready["execution_snapshot_sha256"]
            && ack["engine_binary_sha256"] == ready["engine_binary_sha256"],
        "controller release identity/phase mismatch"
    );
    emit(
        &json!({"event":"CONTROLLER_RELEASE_ACCEPTED","phase":ready["event"],"rank":a.rank,"boot":a.boot,"program_sha256":ready["program_sha256"],"frequency_profile":ready["frequency_profile"]}),
    )
}

struct Release {
    rank: usize,
    address: SocketAddr,
    listener: Option<TcpListener>,
    digest: [u8; 32],
}
impl Release {
    fn new(a: &Args, digest: [u8; 32]) -> Result<Self> {
        let address = SocketAddr::new(a.ips[0], a.release_port);
        let listener = if a.rank == 0 {
            let l = TcpListener::bind(address).context("bind private post-RDMA release barrier")?;
            l.set_nonblocking(true)?;
            Some(l)
        } else {
            None
        };
        Ok(Self {
            rank: a.rank,
            address,
            listener,
            digest,
        })
    }
    fn wait(self, deadline: Instant) -> Result<()> {
        let mut message = [0_u8; 48];
        message[..8].copy_from_slice(b"DS41END1");
        message[8..16].copy_from_slice(&(self.rank as u64).to_le_bytes());
        message[16..].copy_from_slice(&self.digest);
        if let Some(listener) = self.listener {
            let mut streams = BTreeMap::new();
            while streams.len() < WORLD - 1 {
                ensure!(
                    Instant::now() < deadline,
                    "release barrier did not receive all seven completed ranks"
                );
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                        let mut frame = [0_u8; 48];
                        stream.read_exact(&mut frame)?;
                        let source = u64::from_le_bytes(frame[8..16].try_into().unwrap()) as usize;
                        ensure!(
                            &frame[..8] == b"DS41END1"
                                && frame[16..] == self.digest
                                && source > 0
                                && source < WORLD
                                && !streams.contains_key(&source),
                            "invalid or duplicate release-barrier source/program"
                        );
                        streams.insert(source, stream);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            // All eight ranks have finished every RDMA operation before any
            // release. Subsequent cooperative proxy shutdown cannot interrupt
            // another rank's RDMA work, and cannot interrupt this TCP fan-out.
            for (_, mut stream) in streams {
                stream.write_all(b"DS41FREE")?;
            }
        } else {
            let mut stream = loop {
                ensure!(
                    Instant::now() < deadline,
                    "release barrier connection deadline"
                );
                match TcpStream::connect_timeout(&self.address, Duration::from_secs(1)) {
                    Ok(stream) => break stream,
                    Err(_) => std::thread::sleep(Duration::from_millis(50)),
                }
            };
            stream.set_read_timeout(Some(deadline.saturating_duration_since(Instant::now())))?;
            stream.set_write_timeout(Some(Duration::from_secs(5)))?;
            stream.write_all(&message)?;
            let mut ack = [0_u8; 8];
            stream.read_exact(&mut ack)?;
            ensure!(
                &ack == b"DS41FREE",
                "incorrect release-barrier acknowledgement"
            );
        }
        Ok(())
    }
}

fn executable_sha() -> Result<String> {
    let path = std::env::current_exe()?;
    let mut file = std::fs::File::open(path)?;
    let maximum = 1024 * 1024 * 1024u64;
    let size = file.metadata()?.len();
    ensure!(size > 0 && size <= maximum, "bounded actualELF required");
    let mut h = Sha256::new();
    let mut block = vec![0u8; 4 * 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= size, "ELF changed during hashing");
        h.update(&block[..n]);
    }
    ensure!(total == size, "ELF extent changed");
    Ok(format!("{:x}", h.finalize()))
}
fn request_release(a: &Args, index: usize) -> Result<RequestReleaseConfig> {
    let peers = a
        .ips
        .iter()
        .map(|ip| match ip {
            IpAddr::V4(v) => Ok(*v),
            _ => Err(anyhow::anyhow!("IPv4 required")),
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("eight release peers required"))?;
    Ok(RequestReleaseConfig {
        peers,
        port: a
            .request_release_port
            .checked_add(index as u16)
            .context("request release port exhausted")?,
        timeout_millis: 30000,
    })
}
fn kernel_identities() -> Value {
    let entries = [
        (
            "numerics",
            env!("DSV41_NUMERICS_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_numerics.ptx"),
            true,
        ),
        (
            "engram",
            env!("DSV41_ENGRAM_SOURCE_ID"),
            include_str!("../src/ptx/gpu_dsv41_engram.ptx"),
            true,
        ),
        (
            "collectives",
            env!("DSV41_COLLECTIVES_SOURCE_ID"),
            include_str!("../src/ptx/gpu_dsv41_collectives.ptx"),
            true,
        ),
        (
            "gemm",
            env!("DSV41_GEMM_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_gemm.ptx"),
            true,
        ),
        (
            "hc",
            env!("DSV41_HC_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_hc.ptx"),
            true,
        ),
        (
            "dense",
            env!("DSV41_DENSE_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_dense.ptx"),
            true,
        ),
        (
            "moe",
            env!("DSV41_MOE_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_moe.ptx"),
            true,
        ),
        (
            "cache_quant",
            env!("DSV41_CACHE_QUANT_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_cache_quant.ptx"),
            true,
        ),
        (
            "attention",
            env!("DSV41_ATTENTION_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_attention.ptx"),
            true,
        ),
        (
            "position",
            env!("DSV41_POSITION_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_position.ptx"),
            true,
        ),
        (
            "compression",
            env!("DSV41_COMPRESSION_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_compression.ptx"),
            true,
        ),
        (
            "experts",
            env!("DSV41_EXPERTS_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_experts.ptx"),
            true,
        ),
        (
            "kv_store",
            env!("DSV41_KV_STORE_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_kv_store.ptx"),
            true,
        ),
        (
            "text",
            env!("DSV41_TEXT_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_text.ptx"),
            true,
        ),
        (
            "text_batch",
            env!("DSV41_TEXT_BATCH_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_text_batch.ptx"),
            false,
        ),
        (
            "index_select",
            env!("DSV41_INDEX_SELECT_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_index_select.ptx"),
            true,
        ),
        (
            "chunk_store",
            env!("DSV41_CHUNK_STORE_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_chunk_store.ptx"),
            true,
        ),
        (
            "bf16_collectives",
            env!("DSV41_BF16_COLLECTIVES_SOURCE_ID"),
            include_str!("../src/ptx/dsv41_bf16_collectives.ptx"),
            true,
        ),
    ];
    let mut kernels = serde_json::Map::new();
    for (name, source_id, ptx, required) in entries {
        kernels.insert(name.into(),json!({"source_id":source_id,"ptx_sha256":sha(ptx.as_bytes()),"required_by_live_C1_chunk_path":required}));
    }
    json!({"schema_version":1,"compiled_components":18,"required_live_C1_chunk_components":17,"kernels":kernels,"legacy_transport_gpu_batch_ptx_sha256":sha(include_str!("../src/ptx/gpu_batch.ptx").as_bytes()),"unused_in_C1_chunk_path":["text_batch"],"scope":"Compiled identities; required modules are checked by their native load guards before MODEL_READY. Inclusion alone is not execution/quality qualification."})
}

fn startup(a: &Args, event: &str, program: &str, input: &str, pack: &str, body: Value) -> Value {
    let mut v = body;
    let object = v.as_object_mut().expect("literal object");
    object.insert("kernel_identities".into(), kernel_identities());
    object.insert(
        "complete_chunk_component_construction_checked".into(),
        json!(event == "MODEL_READY" || event == "MODEL_CHUNK_PACK_RESULT"),
    );
    for (key, value) in [
        ("event", json!(event)),
        ("rank", json!(a.rank)),
        ("world", json!(8)),
        ("boot", json!(a.boot)),
        ("program_sha256", json!(program)),
        ("input_ids_sha256", json!(input)),
        ("request_pack_sha256", json!(pack)),
        ("frequency_profile", json!(a.frequency_backend.identity())),
        (
            "execution_snapshot_sha256",
            json!(a.execution_snapshot_sha256),
        ),
        ("engine_binary_sha256", json!(a.engine_binary_sha256)),
    ] {
        object.insert(key.into(), value);
    }
    v
}
/// Pure CPU branch. All calls here are file/JSON/hash/typed-plan arithmetic.
/// It exits before device construction, checkpoint/table access, CPU pinning,
/// any stream/link/listener or startup network gate.
fn metadata(a: &Args) -> Result<Value> {
    ensure!(
        executable_sha()? == a.engine_binary_sha256,
        "metadata helper actualELF differs"
    );
    let (pack, pack_sha) = read_pack(a)?;
    let global = official_global_checkpoint_binding(WoaReconstructionPolicy::OfficialBf16Round)?;
    if let Some(expected) = &a.global_checkpoint_binding {
        ensure!(
            expected == &global,
            "supplied global checkpoint metadata binding differs"
        );
    }
    let first = plan(a, &pack.requests[0], &pack_sha, &global, 1, 1, 0, 1)?;
    let program = Program::new_request_session(first.clone())?;
    let releases = (0..pack.requests.len())
        .map(|i| {
            let c = request_release(a, i)?;
            c.validate()?;
            Ok(c)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(startup(
        a,
        "MODEL_CHUNK_CPU_METADATA",
        &program
            .digest()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>(),
        &native_sha(&pack.requests[0].prompt_native_ids),
        &pack_sha,
        json!({"schema_version":1,"kind":"native1017_TP8_C1_chunk_CPU_startup_metadata","CPU_only":true,"CUDA_initialized":false,"links_or_listeners_opened":false,"model_payload_or_frequency_tables_read":false,
        "global_checkpoint_binding":global,"source_index_sha256":gb10_inference::dsv41::weights::INDEX_SHA256,"source_header_geometry_sha256":gb10_inference::dsv41::weights::HEADER_GEOMETRY_SHA256,"WO_A_policy":"OfficialBf16Round",
        "first_request_plan_sha256":first.digest(),"first_request_binding":first.binding(),"first_request_native_ids":first.input_ids(),"first_request_steps":first.steps(),"first_request_planned_operations":first.operations(),
        "first_request_start_epoch":first.start_epoch(),"first_request_end_epoch_at_allowance":first.end_epoch(),"first_request_next_operation_id_reserved":first.next_operation_id(),
        "request_release_endpoints":releases,"request_count":pack.requests.len(),"request_pack":pack,
        "subsequent_request_plan_note":"Subsequent actual start epochs depend on checked early EOS/cap; runtime constructs/agrees those plans only after prior opaque completion and resident reset.",
        "execution_authority":false,"GPU_or_model_or_quality_qualified":false,"benchmark":false}),
    ))
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let a = args()?;
    if a.metadata_only {
        return metadata(&a);
    }
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "main CPU placement failed"
        );
    }
    let actual_elf = executable_sha()?;
    ensure!(
        actual_elf == a.engine_binary_sha256,
        "actualELF differs from pinned request identity"
    );
    let (pack, pack_sha) = read_pack(&a)?;
    for i in 0..pack.requests.len() {
        request_release(&a, i)?.validate()?;
    }
    let config = Dsv41Config::load_dir(&a.model_dir)?;
    let plain = VerifiedFrequencyTable::open_with_backend(
        &a.freq_plain,
        RopeProfile::Plain,
        a.frequency_backend,
    )?;
    let yarn = VerifiedFrequencyTable::open_with_backend(
        &a.freq_yarn,
        RopeProfile::Yarn,
        a.frequency_backend,
    )?;
    let map = CompressedTokenMap::from_u32le(&bounded_file(&a.compressed_map, 129280 * 4)?)?;
    let mut reader = CheckpointReader::open(&a.model_dir, 4 * 1024 * 1024)?;
    let manifest = Arc::new(build_rank_manifest(
        &config,
        reader.inventory(),
        a.rank,
        WoaReconstructionPolicy::OfficialBf16Round,
    )?);
    let global = global_checkpoint_binding(&manifest)?;
    if let Some(expected) = &a.global_checkpoint_binding {
        ensure!(
            expected == &global,
            "actual checkpoint metadata differs from supplied global binding"
        );
    }
    let mut current_plan = plan(&a, &pack.requests[0], &pack_sha, &global, 1, 1, 0, 1)?;
    let program = Program::new_request_session(current_plan.clone())?;
    let program_sha = program
        .digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let first_input_sha = native_sha(&pack.requests[0].prompt_native_ids);
    let dev = CudaDevice::new(0)?;
    let load_stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    let limits = WeightLoadLimits {
        maximum_resident_bytes: 72 * GIB,
        maximum_single_allocation_bytes: 16 * GIB,
        maximum_staging_bytes: 4 * 1024 * 1024,
        available_floor_bytes: 8 * GIB,
        reserved_nonweight_bytes: 4 * GIB,
        maximum_load_seconds: LOAD_READY_SECONDS,
    };
    let budget = GpuWeightBudget::new(limits.clone())?;
    emit(&startup(
        &a,
        "MODEL_LOAD_BEGIN",
        &program_sha,
        &first_input_sha,
        &pack_sha,
        json!({"load_plan":plan_rank_load(&manifest)?,"limits":limits,"native_request_pack":pack,"global_checkpoint_binding":global,"first_request_plan_sha256":current_plan.digest(),"first_request_binding":current_plan.binding(),"first_request_steps":current_plan.steps(),"frozen_operations":program.operations(),"model_execution":false,"timed":false}),
    ))?;
    let weights = RankModelWeights::load(
        &dev,
        &load_stream,
        &mut reader,
        manifest.clone(),
        &budget,
        |e| {
            emit(
                &json!({"event":"MODEL_SCOPE_LOAD_EVIDENCE","rank":a.rank,"boot":a.boot,"timed":false,"evidence":e}),
            )
        },
    )?;
    let load_stats = reader.statistics().clone();
    controller_gate(
        &a,
        startup(
            &a,
            "WEIGHTS_READY",
            &program_sha,
            &first_input_sha,
            &pack_sha,
            json!({"model_revision":MODEL_REVISION,"manifest_sha256":manifest.manifest_sha256,"load_elapsed_seconds":began.elapsed().as_secs_f64(),"budget":budget.snapshot()?,"output_tokens":pack.requests[0].output_allowance,"resident_request_count":pack.requests.len(),"untimed":true}),
        ),
        began + Duration::from_secs(LOAD_READY_SECONDS),
    )?;
    let model_deadline = (Instant::now() + Duration::from_secs(MODEL_READY_SECONDS))
        .min(began + Duration::from_secs(WALL_SECONDS));
    let final_release = Release::new(&a, program.digest())?;
    let link = TpLink::connect_nway(
        a.rank as i32,
        8,
        &a.ips,
        a.port,
        &a.dev,
        a.gid,
        gb10_inference::tp::TP_SLOT_BYTES,
    )?;
    let mut comm = GpuCollectives::new(dev.clone(), link, a.rank, program, a.proxy_core)?;
    let stream = comm.compute_stream_owner();
    let kernels = ModelChunkKernels::load(&dev, stream, a.boot)?;
    let mut model = GpuModelChunk::new(
        kernels,
        weights,
        config,
        map,
        plain,
        yarn,
        ModelChunkRuntimeBinding {
            execution_snapshot_sha256: a.execution_snapshot_sha256.clone(),
            engine_binary_sha256: a.engine_binary_sha256.clone(),
        },
        &comm,
    )?;
    let initial_reset = model.reset_lane(current_plan.binding().lane, &mut comm)?;
    controller_gate(
        &a,
        startup(
            &a,
            "MODEL_READY",
            &program_sha,
            &first_input_sha,
            &pack_sha,
            json!({"model_identity":model.identity(),"checked_reset":initial_reset,"output_tokens":pack.requests[0].output_allowance,"resident_request_count":pack.requests.len(),"elapsed_seconds":began.elapsed().as_secs_f64(),"untimed":true}),
        ),
        model_deadline,
    )?;
    let mut session = GpuRequestSession::attach(
        current_plan.clone(),
        &initial_reset,
        &comm,
        request_release(&a, 0)?,
    )?;
    let mut request_began = Instant::now();
    let mut results = Vec::new();
    let mut agreement = Sha256::new();
    agreement.update(b"dsv41-resident-native-chunk-pack-final-v1\0");
    agreement.update(pack_sha.as_bytes());
    agreement.update(a.boot.as_bytes());
    for (request_index, request) in pack.requests.iter().enumerate() {
        emit(
            &json!({"event":"MODEL_REQUEST_BEGIN","rank":a.rank,"boot":a.boot,"request_index":request_index,"request_id":request.id,"plan_sha256":current_plan.digest(),"declared_request":request,"weights_loaded_again":false,"benchmark":false}),
        )?;
        let mut step_index = 0usize;
        let mut generated = Vec::new();
        let stop_label;
        loop {
            ensure!(
                began.elapsed() < Duration::from_secs(WALL_SECONDS)
                    && request_began.elapsed().as_millis() <= request.wall_millis as u128,
                "native request/global wall budget exhausted"
            );
            let step = current_plan
                .steps()
                .get(step_index)
                .context("session requested step outside frozen plan")?;
            let native = match step.role {
                StepRole::IntermediatePrefillHeadIgnored | StepRole::FinalPrefillFirstOutput => {
                    request.prompt_native_ids[step.start..step.end].to_vec()
                }
                StepRole::DecodeOutput => {
                    vec![*generated.last().context("missing prior native output")?]
                }
            };
            ensure!(
                session.check_before_step(&mut comm)? == step_index,
                "session prelaunch cursor differs"
            );
            let output = model.execute_step(&current_plan, step_index, &native, &mut comm)?;
            generated = output.generated_native_ids().to_vec();
            emit(
                &json!({"event":"MODEL_CHUNK_STEP","rank":a.rank,"boot":a.boot,"request_index":request_index,"request_id":request.id,"checked_forward":output,"benchmark":false,"quality_scored":false}),
            )?;
            if let (Some(expected), Some(ordinal)) =
                (&request.expected_output_native_ids, output.output_ordinal())
            {
                let wanted = expected[ordinal - 1];
                let observed = output.selected_native_id();
                emit(
                    &json!({"event":"MODEL_NATIVE_REFERENCE_COMPARISON","rank":a.rank,"request_index":request_index,"ordinal":ordinal,"expected":wanted,"observed":observed,"passed":wanted==observed}),
                )?;
                ensure!(wanted==observed,"native historical reference differs at output{ordinal}: expected{wanted} observed{observed}");
            }
            ensure!(
                request_began.elapsed().as_millis() <= request.wall_millis as u128,
                "native request wall budget exhausted after completed step"
            );
            match session.accept_step(&output, &mut comm)? {
                SessionProgress::Continue { next_step_index } => {
                    ensure!(
                        next_step_index == step_index + 1,
                        "session skipped a complete step"
                    );
                    step_index = next_step_index;
                }
                SessionProgress::StopReady { reason } => {
                    stop_label = match reason {
                        RequestStopReason::NativeEos => "native_eos",
                        RequestStopReason::OutputAllowance => "output_allowance",
                    };
                    break;
                }
            }
        }
        let completed = session.finish(&mut comm)?;
        ensure!(
            request_began.elapsed().as_millis() <= request.wall_millis as u128,
            "native request wall budget exhausted during final agreement"
        );
        ensure!(
            completed.plan_sha256() == current_plan.digest()
                && completed.generated_native_ids() == generated
                && completed.actual_epoch() == comm.expected_epoch(),
            "opaque request completion differs from model outputs"
        );
        if let Some(expected) = &request.expected_output_native_ids {
            ensure!(
                expected == &generated,
                "full declared historical reference length/content differs"
            );
        }
        agreement.update(current_plan.digest().as_bytes());
        agreement.update(native_sha(&generated).as_bytes());
        agreement.update(stop_label.as_bytes());
        let result = json!({"request_index":request_index,"request_id":request.id,"scope":request.scope,"plan_sha256":current_plan.digest(),"output_native_ids":generated,"native_generated_tokens":generated.len(),"declared_output_allowance":request.output_allowance,"terminal_reason":stop_label,"naturally_completed_native_response":stop_label=="native_eos","capped":stop_label=="output_allowance","historical_reference_checked":request.expected_output_native_ids.is_some(),"request_elapsed_seconds":request_began.elapsed().as_secs_f64(),"checked_request_completion":completed,"actual_context_consumed":model.position(),"native_prefix_sha256":native_sha(model.prefix_native_ids()),"weights_loaded_again":false,"coding_or_tool_quality_scored":false,"benchmark":false});
        emit(
            &json!({"event":"MODEL_REQUEST_COMPLETE","rank":a.rank,"boot":a.boot,"result":result}),
        )?;
        results.push(result);
        if let Some(next) = pack.requests.get(request_index + 1) {
            // The next declared request budget includes plan/reset/rearm work.
            request_began = Instant::now();
            let next_plan = plan(
                &a,
                next,
                &pack_sha,
                &global,
                completed.next_operation_id(),
                completed.next_step_id(),
                completed.actual_epoch(),
                current_plan
                    .binding()
                    .lane
                    .generation
                    .checked_add(1)
                    .context("lane generation exhausted")?,
            )?;
            let reset = model.reset_lane(next_plan.binding().lane, &mut comm)?;
            emit(
                &json!({"event":"MODEL_REQUEST_RESET_CHECKED","rank":a.rank,"boot":a.boot,"next_request_index":request_index+1,"checked_reset":reset,"next_plan_sha256":next_plan.digest(),"weights_loaded_again":false}),
            )?;
            session = GpuRequestSession::rearm(
                next_plan.clone(),
                &reset,
                &completed,
                &mut comm,
                request_release(&a, request_index + 1)?,
            )?;
            current_plan = next_plan;
        } else {
            break;
        }
    }
    let mut release = final_release;
    release.digest = agreement.finalize().into();
    release.wait(began + Duration::from_secs(WALL_SECONDS))?;
    let result = startup(
        &a,
        "MODEL_CHUNK_PACK_RESULT",
        &program_sha,
        &first_input_sha,
        &pack_sha,
        json!({"schema_version":1,"kind":"native1017_TP8_C1_resident_chunk_controls","status":"PASS","all_passed":true,"model_execution":true,"resident_requests":results.len(),"requests":results,"model_identity":model.identity(),"reader_statistics":load_stats,"budget":budget.snapshot()?,"load_count":1,"elapsed_seconds":began.elapsed().as_secs_f64(),"hard_wall_seconds":WALL_SECONDS,"natural_completion_quality":false,"benchmark":false,"HTTP_serving_enabled":false,"scope":"Actual complete learned chunk-forward and resident reset/rearm execution gate; coding/tool quality and speed require separate unchanged harnesses."}),
    );
    drop(model);
    drop(comm);
    dev.synchronize()?;
    Ok(result)
}
fn main() {
    if let Err(error) = (|| -> Result<()> {
        let result = run()?;
        emit(&result)
    })() {
        let _ = emit(
            &json!({"schema_version":1,"status":"FAIL","all_passed":false,"model_execution_may_be_partial":true,"error":format!("{error:#}"),"benchmark":false,"quality_scored":false}),
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(n: usize, output: usize, scope: Scope) -> NativeRequest {
        NativeRequest {
            id: "fixed-native-control".into(),
            scope,
            prompt_native_ids: vec![17; n],
            output_allowance: output,
            context_tokens: 8192,
            wall_millis: 90000,
            sampling: Sampling {
                temperature: 0.,
                top_p: 1.,
                presence_penalty: 0.,
                frequency_penalty: 0.,
                seed: Some(42),
                thinking: true,
                effective_reasoning_score: Some(75),
                speculation: false,
            },
            user_stop_sequences: vec![],
            expected_output_native_ids: None,
        }
    }
    fn pack(r: NativeRequest) -> RequestPack {
        RequestPack {
            schema_version: 1,
            model_revision: MODEL_REVISION.into(),
            requests: vec![r],
        }
    }
    #[test]
    fn original_full4096_allowance_is_preserved_and_never_coerced_to_diagnostic32() {
        let p = pack(request(1480, 4096, Scope::FullBudgetNativeRequest));
        validate_pack(&p).unwrap();
        assert_eq!(p.requests[0].output_allowance, 4096);
        assert!(validate_pack(&pack(request(1480, 4096, Scope::Diagnostic))).is_err());
        assert!(validate_pack(&pack(request(1480, 4097, Scope::FullBudgetNativeRequest))).is_err());
        let mut r = request(1480, 4096, Scope::FullBudgetNativeRequest);
        r.context_tokens = 5575;
        assert!(validate_pack(&pack(r)).is_err());
        for (n, allowance) in [(75, 32), (68, 32), (129, 5), (257, 5), (8191, 1)] {
            validate_pack(&pack(request(n, allowance, Scope::Diagnostic))).unwrap();
        }
    }
    #[test]
    fn malformed_native_budgets_sampling_stops_and_references_refuse_before_loading() {
        for change in 0..11 {
            let mut r = request(75, 32, Scope::Diagnostic);
            match change {
                0 => r.prompt_native_ids[0] = 129280,
                1 => r.prompt_native_ids.clear(),
                2 => r.wall_millis = 0,
                3 => r.wall_millis = WALL_SECONDS * 1000 + 1,
                4 => r.sampling.temperature = 1e-50,
                5 => r.sampling.top_p = 0.99,
                6 => r.sampling.speculation = true,
                7 => r.sampling.effective_reasoning_score = Some(101),
                8 => r.user_stop_sequences.push("unsupported parser stop".into()),
                9 => r.expected_output_native_ids = Some(vec![17; 31]),
                _ => r.expected_output_native_ids = Some(vec![129280; 32]),
            }
            assert!(validate_pack(&pack(r)).is_err());
        }
        let r = request(75, 32, Scope::Diagnostic);
        let mut p = pack(r.clone());
        p.requests.push(r);
        assert!(validate_pack(&p).is_err());
    }
    #[test]
    fn strict_schema_rejects_unknown_duplicate_and_typed_integer_alias_fields() {
        let p = pack(request(75, 32, Scope::Diagnostic));
        let good = serde_json::to_string(&p).unwrap();
        let roundtrip: RequestPack = serde_json::from_str(&good).unwrap();
        validate_pack(&roundtrip).unwrap();
        for bad in [
            good.replacen(
                "\"schema_version\":1",
                "\"schema_version\":1,\"schema_version\":1",
                1,
            ),
            good.replacen("\"schema_version\":1", "\"schema_version\":true", 1),
            good.replacen(
                "\"schema_version\":1",
                "\"schema_version\":1,\"silent_cap\":32",
                1,
            ),
            good.replacen("\"output_allowance\":32", "\"output_allowance\":true", 1),
            good.replacen("\"diagnostic\"", "\"unknown_scope\"", 1),
        ] {
            assert!(serde_json::from_str::<RequestPack>(&bad).is_err());
        }
    }
}
