//! Private resident native atomic-batch service. One actual packed model per
//! rank, no C1 fan-out, no output clipping, no streaming or task/speed claim.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    atomic_request_admission::{
        AdmissionClock, CpuAtomicAdmission, CpuOwnedAtomicGroup, CpuSlotBook,
        PreparedAtomicRequest, ProposedBatchDeployment,
    },
    batch_request_plan::{digest, BatchGroupBinding, BatchPhasePlan},
    config::{Dsv41Config, MODEL_REVISION},
    engram::CompressedTokenMap,
    gpu_batch_block::DecodeHcScratchMode,
    gpu_batch_model::{BatchModelKernels, BatchRuntimeBinding, GpuBatchModel},
    gpu_batch_request_session::{BatchSessionRelease, GpuBatchRequestSession},
    gpu_collectives::{GpuCollectives, Program},
    gpu_gemm::GemmBackend,
    gpu_model_chunk::official_global_checkpoint_binding,
    gpu_model_weights::{plan_rank_load, RankModelWeights},
    gpu_reader::{GpuWeightBudget, WeightLoadLimits},
    gpu_status_cut::ProjectionStatusBackend,
    lab_network::management_peers,
    native_api_contract::{CpuBridge, ATOMIC_ROUTE},
    native_batch_dispatch::{BatchDispatchConfig, InstantMetadata},
    native_batch_http::{self, BatchHttpActor, BatchHttpControl},
    position::{FrequencyBackend, RopeProfile, VerifiedFrequencyTable},
    reader::CheckpointReader,
    scheduler::LaneKey,
    weights::{build_rank_manifest, WoaReconstructionPolicy},
};
use gb10_inference::net::TpLink;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read, Write},
    net::{IpAddr, SocketAddr, TcpStream},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const HARD: u64 = 1200;
const GIB: u64 = 1024 * 1024 * 1024;
struct Args {
    rank: usize,
    ips: Vec<IpAddr>,
    port: u16,
    dev: String,
    gid: i32,
    proxy_core: i32,
    main_core: i32,
    model_dir: PathBuf,
    freq_plain: PathBuf,
    freq_yarn: PathBuf,
    compressed_map: PathBuf,
    tokenizer: PathBuf,
    warmup: PathBuf,
    boot: Uuid,
    load_release: SocketAddr,
    snapshot: String,
    dispatch: BatchDispatchConfig,
    http_bind: SocketAddr,
    service_seconds: u64,
    evidence: PathBuf,
    gemm_backend: GemmBackend,
    projection_status_backend: ProjectionStatusBackend,
    decode_hc_scratch: DecodeHcScratchMode,
    decode_hc_graph: bool,
    metadata_only: bool,
}
fn args() -> Result<Args> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(raw.len() % 2 == 0, "explicit flag/value pairs");
    let allowed = [
        "--fp8-gemm-backend",
        "--projection-status-backend",
        "--decode-hc-scratch",
        "--decode-hc-graph",
        "--rank",
        "--world",
        "--peer-ips",
        "--management-ips",
        "--port",
        "--dev",
        "--gid",
        "--proxy-core",
        "--main-core",
        "--model-dir",
        "--freq-plain",
        "--freq-yarn",
        "--frequency-backend",
        "--compressed-map",
        "--tokenizer",
        "--warmup-request-file",
        "--boot",
        "--load-release-address",
        "--load-release-port",
        "--execution-snapshot-sha256",
        "--dispatch-port",
        "--release-base",
        "--http-bind",
        "--service-seconds",
        "--maximum-groups",
        "--evidence-dir",
        "--metadata-only",
    ];
    let mut f = BTreeMap::new();
    for pair in raw.chunks_exact(2) {
        ensure!(
            allowed.contains(&pair[0].as_str())
                && f.insert(pair[0].clone(), pair[1].clone()).is_none(),
            "unknown/repeated argument{}",
            pair[0]
        );
    }
    let v = |key: &str| f.get(key).with_context(|| format!("missing{key}"));
    ensure!(
        v("--world")? == "8" && v("--frequency-backend")? == "cuda-torch",
        "world8 and measured CUDA frequency backend required"
    );
    let rank = v("--rank")?.parse::<usize>()?;
    let ips = v("--peer-ips")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<IpAddr>, _>>()?;
    ensure!(
        rank < 8 && ips.len() == 8 && ips.iter().collect::<BTreeSet<_>>().len() == 8,
        "ordered distinct8 RoCEIPs/rank required"
    );
    let port = v("--port")?.parse::<u16>()?;
    let boot = Uuid::parse_str(v("--boot")?)?;
    let load_release = SocketAddr::new(
        v("--load-release-address")?.parse()?,
        v("--load-release-port")?.parse()?,
    );
    let dispatch = BatchDispatchConfig {
        peers: management_peers(v("--management-ips")?)?,
        port: v("--dispatch-port")?.parse()?,
        release_base: v("--release-base")?.parse()?,
        maximum_groups: v("--maximum-groups")?.parse()?,
        boot,
        service_deadline: InstantMetadata {
            idle_millis: 1200000,
        },
    };
    dispatch.validate()?;
    let http_bind = v("--http-bind")?.parse::<SocketAddr>()?;
    ensure!(
        http_bind.ip() == IpAddr::V4(dispatch.peers[0]) && http_bind.port() >= 1024,
        "private rank-zero API bind required"
    );
    let mut ports = gb10_inference::net::nway_link_ports(port, 8);
    ports.extend([
        load_release.port() as u32,
        dispatch.port as u32,
        http_bind.port() as u32,
    ]);
    ports.extend(
        (dispatch.release_base as usize
            ..dispatch.release_base as usize + dispatch.maximum_groups * 20)
            .map(|p| p as u32),
    );
    ensure!(
        ports.iter().all(|p| *p >= 1024 && *p <= 65535)
            && ports.iter().collect::<BTreeSet<_>>().len() == ports.len(),
        "all data/startup/HTTP/phase ports distinct and bounded"
    );
    let snapshot = v("--execution-snapshot-sha256")?.clone();
    ensure!(
        gb10_inference::dsv41::batch_request_plan::sha256(&snapshot),
        "snapshot SHA256 required"
    );
    let service_seconds = v("--service-seconds")?.parse::<u64>()?;
    ensure!(
        (1..=600).contains(&service_seconds),
        "initial bounded service window1..600s"
    );
    ensure!(
        v("--metadata-only")? == "true" || v("--metadata-only")? == "false",
        "metadata-only true/false"
    );
    Ok(Args {
        gemm_backend: GemmBackend::parse(
            f.get("--fp8-gemm-backend")
                .map(String::as_str)
                .unwrap_or("original"),
        )?,
        projection_status_backend: ProjectionStatusBackend::parse(
            f.get("--projection-status-backend")
                .map(String::as_str)
                .unwrap_or("original"),
        )?,
        decode_hc_scratch: DecodeHcScratchMode::parse(
            f.get("--decode-hc-scratch")
                .map(String::as_str)
                .unwrap_or("off"),
        )?,
        decode_hc_graph: match f.get("--decode-hc-graph").map(String::as_str) {
            None | Some("off") => false,
            Some("on") => true,
            _ => anyhow::bail!("decode HC graph must be off or on"),
        },
        rank,
        ips,
        port,
        boot,
        load_release,
        dispatch,
        http_bind,
        snapshot,
        service_seconds,
        dev: v("--dev")?.clone(),
        gid: v("--gid")?.parse()?,
        proxy_core: v("--proxy-core")?.parse()?,
        main_core: v("--main-core")?.parse()?,
        model_dir: v("--model-dir")?.into(),
        freq_plain: v("--freq-plain")?.into(),
        freq_yarn: v("--freq-yarn")?.into(),
        compressed_map: v("--compressed-map")?.into(),
        tokenizer: v("--tokenizer")?.into(),
        warmup: v("--warmup-request-file")?.into(),
        evidence: v("--evidence-dir")?.into(),
        metadata_only: v("--metadata-only")? == "true",
    })
}
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn read(path: &PathBuf, limit: usize) -> Result<Vec<u8>> {
    let n = std::fs::metadata(path)?.len();
    ensure!(n <= limit as u64, "bounded file");
    let b = std::fs::read(path)?;
    ensure!(b.len() == n as usize, "file changed duringread");
    Ok(b)
}
fn binary_sha(path: &PathBuf) -> Result<String> {
    // CPU debug ELFs can exceed256MiB. Stream a bounded file instead of loading
    // it into RAM; the exact same bytes are pinned by the parent supervisor.
    let mut file = std::fs::File::open(path)?;
    let before = file.metadata()?;
    ensure!(
        before.is_file() && before.len() <= 2 * GIB,
        "bounded actual ELF"
    );
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut count = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count.checked_add(n as u64).context("ELF size overflow")?;
        ensure!(count <= before.len(), "ELF grew during hashing");
        hash.update(&buffer[..n]);
    }
    let after = file.metadata()?;
    ensure!(
        count == before.len()
            && before.len() == after.len()
            && before.modified()? == after.modified()?,
        "actual ELF changed during streaming hash"
    );
    Ok(format!("{:x}", hash.finalize()))
}
fn emit<T: Serialize>(v: &T) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, v)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn bootstrap(
    a: &Args,
    bridge: &CpuBridge,
    binary: &str,
    admitted: Instant,
) -> Result<(CpuSlotBook, CpuOwnedAtomicGroup, Arc<BatchPhasePlan>)> {
    let body = native_batch_http::parse_original_body(&read(&a.warmup, 1024 * 1024)?)?;
    let prepared = PreparedAtomicRequest::prepare(bridge, ATOMIC_ROUTE, Some("1"), &body)?;
    ensure!(
        prepared.members().len() == 1
            && prepared.members()[0].prompt_ids().len() == 75
            && prepared.members()[0].semantics().output_allowance == 1
            && prepared.members()[0].semantics().thinking
            && prepared.members()[0].semantics().effective_reasoning_score == Some(75),
        "explicit code75/1 warmup only, never shorten benchmark bodies"
    );
    let fixture: Value = serde_json::from_str(include_str!(
        "../tests/fixtures/dsv41_actual_atomic_requests.json"
    ))?;
    let expected: Vec<u32> =
        serde_json::from_value(fixture["cases"][2]["native_prompt_ids"].clone())?;
    ensure!(
        prepared.members()[0].prompt_ids() == expected,
        "warmup must use exact original code75 native IDs"
    );
    let mut book = CpuSlotBook::new(a.boot, 16)?;
    let deployment = ProposedBatchDeployment {
        boot: a.boot,
        global_checkpoint_geometry_checksum_sha256: official_global_checkpoint_binding(
            WoaReconstructionPolicy::OfficialBf16Round,
        )?,
        execution_snapshot_sha256: a.snapshot.clone(),
        engine_binary_sha256: binary.into(),
        frequency: FrequencyBackend::CudaTorch.identity(),
    };
    let mut tx = CpuAtomicAdmission::new(
        &mut book,
        prepared,
        deployment,
        vec![LaneKey {
            boot: a.boot,
            slot: 0,
            generation: 1,
        }],
        AdmissionClock::new(admitted, admitted + Duration::from_secs(180))?,
        format!("native-atomic-{}-warmup", a.boot),
        admitted,
    )?;
    tx.commit(admitted)?;
    let owned = tx.take_published(admitted)?;
    let binding = BatchGroupBinding::from_cpu(&owned, &book, admitted)?;
    let plan = BatchPhasePlan::first(binding, 1, 1, 0, 1)?;
    Ok((book, owned, plan))
}
fn gate(a: &Args, ready: Value, deadline: Instant) -> Result<()> {
    emit(&ready)?;
    let mut stream = loop {
        ensure!(Instant::now() < deadline, "startup deadline");
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
        .take(16384)
        .read_line(&mut line)?;
    ensure!(
        line.ends_with('\n') && line.len() < 16384,
        "bounded startupACK"
    );
    let ack: Value = serde_json::from_str(&line)?;
    let phase = if ready["event"] == "WEIGHTS_READY" {
        "RELEASE_WEIGHTS"
    } else {
        "RELEASE_MODEL"
    };
    ensure!(
        ack["event"] == phase
            && ack["rank"] == a.rank
            && ack["world"] == 8
            && ack["boot"] == a.boot.to_string()
            && ack["program_sha256"] == ready["program_sha256"]
            && ack["input_ids_sha256"] == ready["input_ids_sha256"]
            && ack["frequency_profile"] == ready["frequency_profile"]
            && ack["request_pack_sha256"] == ready["request_pack_sha256"]
            && ack["binary_sha256"] == ready["binary_sha256"]
            && ack["execution_snapshot_sha256"] == ready["execution_snapshot_sha256"]
            && ack["atomic_batch_enabled"] == true
            && ack["compiled_isolated_components"] == 22,
        "exact atomic startup mode/program/input/source/profile ACK required"
    );
    emit(
        &json!({"event":"CONTROLLER_RELEASE_ACCEPTED","phase":ready["event"],"rank":a.rank,"boot":a.boot,"program_sha256":ready["program_sha256"],"atomic_batch_enabled":true}),
    )
}
fn kernel_identities() -> Value {
    macro_rules! component {($e:literal,$p:literal)=>{json!({"source_id_hex":env!($e),"ptx_sha256":sha(include_str!($p).as_bytes())})};}
    json!({
    "numerics":component!("DSV41_NUMERICS_SOURCE_ID","../src/ptx/dsv41_numerics.ptx"),
    "engram":component!("DSV41_ENGRAM_SOURCE_ID","../src/ptx/gpu_dsv41_engram.ptx"),
    "collectives":component!("DSV41_COLLECTIVES_SOURCE_ID","../src/ptx/gpu_dsv41_collectives.ptx"),
    "gemm":component!("DSV41_GEMM_SOURCE_ID","../src/ptx/dsv41_gemm.ptx"),
    "dense":component!("DSV41_DENSE_SOURCE_ID","../src/ptx/dsv41_dense.ptx"),
    "hc":component!("DSV41_HC_SOURCE_ID","../src/ptx/dsv41_hc.ptx"),
    "moe":component!("DSV41_MOE_SOURCE_ID","../src/ptx/dsv41_moe.ptx"),
    "cache_quant":component!("DSV41_CACHE_QUANT_SOURCE_ID","../src/ptx/dsv41_cache_quant.ptx"),
    "attention":component!("DSV41_ATTENTION_SOURCE_ID","../src/ptx/dsv41_attention.ptx"),
    "compression":component!("DSV41_COMPRESSION_SOURCE_ID","../src/ptx/dsv41_compression.ptx"),
    "position":component!("DSV41_POSITION_SOURCE_ID","../src/ptx/dsv41_position.ptx"),
    "experts":component!("DSV41_EXPERTS_SOURCE_ID","../src/ptx/dsv41_experts.ptx"),
    "kv_store":component!("DSV41_KV_STORE_SOURCE_ID","../src/ptx/dsv41_kv_store.ptx"),
    "text":component!("DSV41_TEXT_SOURCE_ID","../src/ptx/dsv41_text.ptx"),
    "chunk_store":component!("DSV41_CHUNK_STORE_SOURCE_ID","../src/ptx/dsv41_chunk_store.ptx"),
    "text_batch":component!("DSV41_TEXT_BATCH_SOURCE_ID","../src/ptx/dsv41_text_batch.ptx"),
    "index_select":component!("DSV41_INDEX_SELECT_SOURCE_ID","../src/ptx/dsv41_index_select.ptx"),
    "bf16_collectives":component!("DSV41_BF16_COLLECTIVES_SOURCE_ID","../src/ptx/dsv41_bf16_collectives.ptx"),
    "tool_constraint":component!("DSV41_TOOL_CONSTRAINT_SOURCE_ID","../src/ptx/dsv41_tool_constraint.ptx"),
    "batch_layout":component!("DSV41_BATCH_LAYOUT_SOURCE_ID","../src/ptx/dsv41_batch_layout.ptx"),
    "batch_attention":component!("DSV41_BATCH_ATTENTION_SOURCE_ID","../src/ptx/dsv41_batch_attention.ptx"),
    "batch_compression":component!("DSV41_BATCH_COMPRESSION_SOURCE_ID","../src/ptx/dsv41_batch_compression.ptx"),
    })
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let a = args()?;
    let binary = binary_sha(&std::env::current_exe()?)?;
    let bridge = Arc::new(CpuBridge::from_verified_tokenizer(&read(
        &a.tokenizer,
        64 * 1024 * 1024,
    )?)?);
    let (_, _, prototype) = bootstrap(&a, &bridge, &binary, Instant::now())?;
    let program = Program::new_batch_phase(prototype.clone())?;
    let program_sha = program
        .digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let common = json!({"rank":a.rank,"world":8,"boot":a.boot,"model_revision":MODEL_REVISION,
        "program_sha256":program_sha,"request_pack_sha256":prototype.group().original_body_sha256(),
        "input_ids_sha256":gb10_inference::dsv41::model_session_plan::native_ids_sha256(&prototype.group().members()[0].prompt_ids),
        "frequency_profile":FrequencyBackend::CudaTorch.identity(),"binary_sha256":binary,
        "execution_snapshot_sha256":a.snapshot,"source_build_id":env!("SOURCE_BUILD_ID"),"source_commit":env!("TUNE_GIT_COMMIT"),
        "fp8_gemm_backend":a.gemm_backend,"fp8_gemm_policy":{"eligible_rows":[1,16],"eligible_inner":5120,"eligible_outputs":[512,1280,2304],"persistent_contribution_bytes":if a.gemm_backend==GemmBackend::Original{0}else{gb10_inference::dsv41::gpu_gemm::HOT_GROUP_WORKSPACE_ELEMENTS*4},"other_shapes_and_FP4":"original"},
        "projection_status_backend":a.projection_status_backend,"decode_hc_scratch_mode":a.decode_hc_scratch,"decode_hc_graph_enabled":a.decode_hc_graph,"projection_status_scope":"initial packed attention projection cut; ten individual original status words; no kernel/math changes","kernel_identities":kernel_identities(),"compiled_isolated_components":22,
        "intended_actual_model_loaded_modules":["numerics","engram","collectives","gemm","dense","hc","moe","cache_quant","position","experts","chunk_store","text_batch","batch_layout","batch_attention","batch_compression"],
        "atomic_batch_enabled":true,"buffered_only":true,"HTTP_streaming_enabled":false,
        "maximum_members":16,"maximum_learned_prefix":512,"maximum_output_allowance":256,
        "native_wall_seconds":180,"historical_matched_client_timeout_seconds":100,"benchmark_qualified":false});
    if a.metadata_only {
        return Ok(
            json!({"event":"ATOMIC_SERVICE_CPU_METADATA","common":common,
        "initial_phase":prototype.as_ref(),"initial_program":program.operations(),"GPU_initialized":false,
        "checkpoint_payload_read":false,"network_opened":false,"metadata_only":true}),
        );
    }
    if a.main_core >= 0 {
        ensure!(gb10_inference::net::pin_thread(a.main_core), "CPUplacement");
    }
    let config = Dsv41Config::load_dir(&a.model_dir)?;
    let plain = VerifiedFrequencyTable::open_with_backend(
        &a.freq_plain,
        RopeProfile::Plain,
        FrequencyBackend::CudaTorch,
    )?;
    let yarn = VerifiedFrequencyTable::open_with_backend(
        &a.freq_yarn,
        RopeProfile::Yarn,
        FrequencyBackend::CudaTorch,
    )?;
    let map = CompressedTokenMap::from_u32le(&read(&a.compressed_map, 129280 * 4)?)?;
    let mut reader = CheckpointReader::open(&a.model_dir, 4 * 1024 * 1024)?;
    let manifest = Arc::new(build_rank_manifest(
        &config,
        reader.inventory(),
        a.rank,
        WoaReconstructionPolicy::OfficialBf16Round,
    )?);
    let plan = plan_rank_load(&manifest)?;
    let dev = CudaDevice::new(0)?;
    let load_stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let limits = WeightLoadLimits {
        maximum_resident_bytes: 72 * GIB,
        maximum_single_allocation_bytes: 16 * GIB,
        maximum_staging_bytes: 4 * 1024 * 1024,
        available_floor_bytes: 8 * GIB,
        reserved_nonweight_bytes: 4 * GIB,
        maximum_load_seconds: 900,
    };
    let budget = GpuWeightBudget::new(limits.clone())?;
    emit(&event(
        &common,
        "MODEL_LOAD_BEGIN",
        json!({"load_plan":plan,"limits":limits,"timed":false}),
    ))?;
    let weights = RankModelWeights::load(
        &dev,
        &load_stream,
        &mut reader,
        manifest.clone(),
        &budget,
        |e| {
            emit(&event(
                &common,
                "MODEL_SCOPE_LOAD_EVIDENCE",
                json!({"evidence":e,"timed":false}),
            ))
        },
    )?;
    gate(
        &a,
        event(
            &common,
            "WEIGHTS_READY",
            json!({"manifest_sha256":manifest.manifest_sha256,"budget":budget.snapshot()?,"untimed":true}),
        ),
        began + Duration::from_secs(900),
    )?;
    let model_deadline =
        (Instant::now() + Duration::from_secs(300)).min(began + Duration::from_secs(HARD));
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
    let k = BatchModelKernels::load_with_decode_graph(
        &dev,
        comm.compute_stream_owner(),
        a.boot,
        a.gemm_backend,
        a.projection_status_backend,
        a.decode_hc_scratch,
        a.decode_hc_graph,
    )?;
    let mut model = GpuBatchModel::new(k, weights, config, map, plain, yarn, &comm, 16)?;
    model.bind_runtime(BatchRuntimeBinding {
        execution_snapshot_sha256: a.snapshot.clone(),
        engine_binary_sha256: binary,
    })?;
    gate(
        &a,
        event(
            &common,
            "MODEL_READY",
            json!({"model_identity":model.identity(),"loaded_model_count":1,"model_constructed_count":1,"decode_hc_scratch_bytes":model.decode_hc_scratch_bytes(),"untimed":true}),
        ),
        model_deadline,
    )?;
    let admitted = Instant::now();
    let (mut book, owned, actual) = bootstrap(
        &a,
        &bridge,
        common["binary_sha256"].as_str().unwrap(),
        admitted,
    )?;
    ensure!(
        actual.digest() == prototype.digest(),
        "real bootstrap preparation differs from agreed CPU program"
    );
    let reset = model.reset_group_checked(&actual.group().lanes(), &mut comm)?;
    emit(&event(
        &common,
        "ATOMIC_BOOTSTRAP_RESET",
        json!({"reset":reset}),
    ))?;
    let session = GpuBatchRequestSession::attach(
        actual,
        reset,
        &model,
        &comm,
        BatchSessionRelease {
            peers: a.dispatch.peers,
            first_phase_port: a.dispatch.release_port(1)?,
            timeout_millis: 30000,
        },
        &owned,
        &book,
    )?;
    let previous = session.run(
        &mut model,
        &mut comm,
        |p| emit(&event(&common, "ATOMIC_BOOTSTRAP_STEP", json!({"proof":p}))),
        || Ok(()),
    )?;
    emit(&event(
        &common,
        "ATOMIC_BOOTSTRAP_COMPLETE",
        json!({"completion":previous}),
    ))?;
    ensure!(
        previous.progress()[0].generated_native_ids == [2581],
        "actual code75 warmup first native token must match historicalWe2581"
    );
    book.release_cpu_ownership(&owned)?;
    let service_deadline = Instant::now() + Duration::from_secs(a.service_seconds);
    ensure!(
        service_deadline + Duration::from_secs(30) < began + Duration::from_secs(HARD),
        "full declared service window plus cleanup must fit process hardwall; do not silently clip"
    );
    let final_boundary = if a.rank == 0 {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&a.evidence)?;
        let control = BatchHttpControl::new();
        let (actor, inbox) = BatchHttpActor::new(
            bridge.clone(),
            control.clone(),
            a.evidence.clone(),
            service_deadline,
        )?;
        let control_for_server = control.clone();
        let control_for_failure = control.clone();
        let bind = a.http_bind;
        let head = a.dispatch.peers[0];
        let (stop_http, stopped_http) = tokio::sync::oneshot::channel::<()>();
        let http = std::thread::spawn(move || -> Result<()> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(actor.serve(bind, head, async move {
                tokio::select! { _ = tokio::time::sleep_until(tokio::time::Instant::from_std(service_deadline)) => (), _ = stopped_http => () };
                control_for_server.stop();
            }));
            if result.is_err() {
                control_for_failure.stop();
            }
            result
        });
        let result = native_batch_http::run_head_worker(
            &mut model,
            &mut comm,
            previous,
            &bridge,
            a.dispatch.clone(),
            inbox,
            control.clone(),
            service_deadline,
        );
        control.stop();
        let _ = stop_http.send(());
        // Graceful HTTP shutdown has a finite declared service window; failure
        // still poisons the model and parent independently tears all8 down.
        let http_result = http
            .join()
            .map_err(|_| anyhow::anyhow!("HTTPthread panic"))?;
        http_result?;
        result?
    } else {
        native_batch_http::run_peer_worker(
            &mut model,
            &mut comm,
            previous,
            &bridge,
            a.dispatch.clone(),
            service_deadline + Duration::from_secs(30),
        )?
    };
    let result = event(
        &common,
        "ATOMIC_SERVICE_COMPLETE",
        json!({"status":"PASS","all_passed":true,
        "kind":"native1017_resident_atomic_batch_service","last_completion":final_boundary,
        "fp8_gemm_dispatch":model.gemm_dispatch_snapshot(),"projection_status_cuts":model.projection_status_snapshot(),"decode_hc_scratch_bytes":model.decode_hc_scratch_bytes(),"loaded_model_count":1,"reader_statistics":reader.statistics(),"budget":budget.snapshot()?,
        "benchmark_or_quality_qualified":false,"elapsed_seconds":began.elapsed().as_secs_f64()}),
    );
    drop(model);
    drop(comm);
    dev.synchronize()?;
    Ok(result)
}
fn event(common: &Value, name: &str, extra: Value) -> Value {
    let mut out = common.clone();
    let o = out.as_object_mut().unwrap();
    o.insert("event".into(), name.into());
    for (k, v) in extra.as_object().unwrap() {
        o.insert(k.clone(), v.clone());
    }
    out
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(HARD));
        eprintln!("atomic service hard1200s deadline");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"status":"FAIL","all_passed":false,"error":format!("{e:#}"),"kind":"native1017_resident_atomic_batch_service"})
        }
        Err(_) => {
            json!({"status":"FAIL","all_passed":false,"error":"panic","kind":"native1017_resident_atomic_batch_service"})
        }
    };
    let passed = result["all_passed"] == true || result["metadata_only"] == true;
    if emit(&result).is_err() || !passed {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cpu_args() -> Args {
        Args{rank:0,ips:(14..22).map(|n|format!("198.51.100.{n}").parse().unwrap()).collect(),port:47300,
            dev:"CPU_ONLY_NO_HCA_OPEN".into(),gid:3,proxy_core:-1,main_core:-1,
            model_dir:"/CPU_SENTINEL_MUST_NOT_OPEN_MODEL".into(),freq_plain:"/CPU_SENTINEL_PLAIN".into(),
            freq_yarn:"/CPU_SENTINEL_YARN".into(),compressed_map:"/CPU_SENTINEL_MAP".into(),
            tokenizer:"test-data/tokenizer.json".into(),
            warmup:"tests/fixtures/dsv41_atomic_warmup.json".into(),
            boot:Uuid::from_u128(214),load_release:"172.31.240.10:47401".parse().unwrap(),snapshot:"b".repeat(64),
            dispatch:BatchDispatchConfig{peers:std::array::from_fn(|i|std::net::Ipv4Addr::new(172, 31, 240, 11 +i as u8)),
                port:47430,release_base:62000,maximum_groups:32,boot:Uuid::from_u128(214),service_deadline:InstantMetadata{idle_millis:1200000}},
            http_bind:"172.31.240.11:46111".parse().unwrap(),service_seconds:600,gemm_backend:GemmBackend::Original,projection_status_backend:ProjectionStatusBackend::Original,decode_hc_scratch:DecodeHcScratchMode::Off,decode_hc_graph:false,evidence:"/CPU_UNUSED_HTTP_EVIDENCE".into(),metadata_only:true}
    }
    #[test]
    fn repeated_rank_local_CPU_nonce_clocks_do_not_change_common_warmup_native_program() {
        let a = cpu_args();
        let bridge =
            CpuBridge::from_verified_tokenizer(&read(&a.tokenizer, 64 * 1024 * 1024).unwrap())
                .unwrap();
        let (_, _, one) = bootstrap(&a, &bridge, &"a".repeat(64), Instant::now()).unwrap();
        let (_, _, two) = bootstrap(&a, &bridge, &"a".repeat(64), Instant::now()).unwrap();
        assert_eq!(one.digest(), two.digest());
        assert_eq!(one.group().digest(), two.group().digest());
        assert_eq!(one.group().members()[0].prompt_ids.len(), 75);
        assert_eq!(one.group().members()[0].request.budget.output_allowance, 1);
        let p = Program::new_batch_phase(one).unwrap();
        assert_eq!(p.operations().len(), 84);
        assert_eq!(p.digest(), Program::new_batch_phase(two).unwrap().digest());
        assert_ne!(
            p.digest(),
            Program::new_model_sequence(p.operations().to_vec())
                .unwrap()
                .digest()
        );
        assert!(!a.model_dir.exists());
    }
    #[test]
    fn actual_entrypoint_declares22_compiled_and15_loaded_without_optional_sampler_load() {
        let ids = kernel_identities();
        assert_eq!(ids.as_object().unwrap().len(), 22);
        let source = include_str!("dsv41_resident_atomic_http.rs");
        let live = [
            "numerics",
            "engram",
            "collectives",
            "gemm",
            "dense",
            "hc",
            "moe",
            "cache_quant",
            "position",
            "experts",
            "chunk_store",
            "text_batch",
            "batch_layout",
            "batch_attention",
            "batch_compression",
        ];
        assert_eq!(live.len(), 15);
        assert!(live.iter().all(|n| ids.get(*n).is_some()));
        assert!(!live.contains(&"tool_constraint"));
        assert!(source.contains("historical_matched_client_timeout_seconds"));
    }
}
