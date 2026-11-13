//! Full8-rank native1017 C1 diagnostic. Parent owns exclusive GPUs, recoverable
//! guards and independently supervised1200s process groups. This is fixed-count
//! continuation (even afterEOS), not a natural-completion quality or speed test.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    config::{Dsv41Config, MODEL_REVISION},
    engram::CompressedTokenMap,
    gpu_block::BlockOperationIds,
    gpu_collectives::{GpuCollectives, Program},
    gpu_model::{GpuModel, ModelKernels, ModelOperationIds, OPERATIONS_PER_FORWARD},
    gpu_model_weights::{plan_rank_load, RankModelWeights},
    gpu_reader::{GpuWeightBudget, WeightLoadLimits},
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
    dev: String,
    gid: i32,
    proxy_core: i32,
    main_core: i32,
    model_dir: PathBuf,
    freq_plain: PathBuf,
    freq_yarn: PathBuf,
    frequency_backend: FrequencyBackend,
    compressed_map: PathBuf,
    tokens_file: PathBuf,
    output_tokens: usize,
    boot: Uuid,
    load_release: SocketAddr,
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
        "--dev",
        "--gid",
        "--proxy-core",
        "--main-core",
        "--model-dir",
        "--freq-plain",
        "--freq-yarn",
        "--frequency-backend",
        "--compressed-map",
        "--tokens-file",
        "--output-tokens",
        "--boot",
        "--load-release-address",
        "--load-release-port",
    ];
    let mut f = BTreeMap::new();
    for p in raw.chunks_exact(2) {
        ensure!(
            allowed.contains(&p[0].as_str()) && f.insert(p[0].clone(), p[1].clone()).is_none(),
            "unknown/repeated probe argument {}",
            p[0]
        );
    }
    let v = |k: &str| f.get(k).with_context(|| format!("missing {k}"));
    ensure!(
        v("--world")?.parse::<usize>()? == WORLD,
        "native probe requires world8"
    );
    let rank = v("--rank")?.parse::<usize>()?;
    let ips = v("--peer-ips")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<IpAddr>, _>>()?;
    let port = v("--port")?.parse::<u16>()?;
    let release_port = v("--release-port")?.parse::<u16>()?;
    let load_ip = v("--load-release-address")?.parse::<IpAddr>()?;
    let load_port = v("--load-release-port")?.parse::<u16>()?;
    ensure!(
        rank < 8 && ips.len() == 8 && ips.iter().collect::<BTreeSet<_>>().len() == 8,
        "eight distinct rank-ordered RoCE IPs/rank0..7 required"
    );
    ensure!(
        [port, release_port, load_port].iter().all(|p| *p >= 1024)
            && [port, release_port, load_port]
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                == 3,
        "private unprivileged distinct bootstrap/release/load ports required"
    );
    let ports = gb10_inference::net::nway_link_ports(port, 8);
    ensure!(
        ports.iter().all(|p| *p <= u16::MAX as u32)
            && !ports.contains(&(release_port as u32))
            && !ports.contains(&(load_port as u32)),
        "bootstrap ports overlap/overflow"
    );
    ensure!(
        !load_ip.is_unspecified() && !load_ip.is_multicast(),
        "explicit controller management address required"
    );
    let output_tokens = v("--output-tokens")?.parse::<usize>()?;
    ensure!(
        (1..=32).contains(&output_tokens),
        "initial diagnostic outputcount1..32"
    );
    let gid = v("--gid")?.parse::<i32>()?;
    ensure!(
        gid >= 0 && !v("--dev")?.is_empty(),
        "explicit HCA/GID required"
    );
    Ok(Args {
        rank,
        ips,
        port,
        release_port,
        dev: v("--dev")?.clone(),
        gid,
        proxy_core: f
            .get("--proxy-core")
            .map(|x| x.parse())
            .transpose()?
            .unwrap_or(-1),
        main_core: f
            .get("--main-core")
            .map(|x| x.parse())
            .transpose()?
            .unwrap_or(-1),
        model_dir: v("--model-dir")?.into(),
        freq_plain: v("--freq-plain")?.into(),
        freq_yarn: v("--freq-yarn")?.into(),
        frequency_backend: FrequencyBackend::parse(
            f.get("--frequency-backend")
                .map(String::as_str)
                .unwrap_or("cpu-torch"),
        )?,
        compressed_map: v("--compressed-map")?.into(),
        tokens_file: v("--tokens-file")?.into(),
        output_tokens,
        boot: Uuid::parse_str(v("--boot")?)?,
        load_release: SocketAddr::new(load_ip, load_port),
    })
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
fn tokens(a: &Args) -> Result<(Vec<u32>, String)> {
    let raw = bounded_file(&a.tokens_file, 64 * 1024)?;
    let value: Value = serde_json::from_slice(&raw)?;
    ensure!(
        value["schema_version"] == 1 && value["model_revision"] == MODEL_REVISION,
        "frozen native input schema/model differs"
    );
    let ids: Vec<u32> = serde_json::from_value(value["native_ids"].clone())?;
    ensure!(
        (1..=128).contains(&ids.len()) && ids.iter().all(|x| *x < 129280),
        "initial native input1..128/vocabulary bound"
    );
    Ok((ids, sha(&raw)))
}
fn operation_ids(first: u64) -> ModelOperationIds {
    let mut n = first;
    let embedding = n;
    n += 1;
    let mut engram = [0; 2];
    let mut blocks = Vec::with_capacity(40);
    for layer in 0..40 {
        if layer == 1 || layer == 14 {
            engram[usize::from(layer == 14)] = n;
            n += 1;
        }
        blocks.push(BlockOperationIds {
            attention: n,
            routed_experts: n + 1,
        });
        n += 2;
    }
    ModelOperationIds {
        embedding,
        engram,
        blocks,
        head: n,
    }
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
            && ack["frequency_profile"] == ready["frequency_profile"],
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

fn run() -> Result<Value> {
    let began = Instant::now();
    let a = args()?;
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "main CPU placement failed"
        );
    }
    let (input, input_file_sha) = tokens(&a)?;
    let input_sha = gb10_inference::dsv41::gpu_model::native_ids_sha256(&input);
    let mut step_ids = Vec::new();
    let mut ops = Vec::new();
    for step in 0..a.output_tokens {
        let ids = operation_ids(1 + (step * OPERATIONS_PER_FORWARD) as u64);
        ops.extend(ids.operations(
            if step == 0 { 0 } else { input.len() + step - 1 },
            if step == 0 { input.len() } else { 1 },
        )?);
        step_ids.push(ids);
    }
    let program = Program::new_model_sequence(ops)?;
    let program_sha = program
        .digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
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
    let frequency_profile = a.frequency_backend.identity();
    let map = CompressedTokenMap::from_u32le(&bounded_file(&a.compressed_map, 129280 * 4)?)?;
    let mut reader = CheckpointReader::open(&a.model_dir, 4 * 1024 * 1024)?;
    let manifest = Arc::new(build_rank_manifest(
        &config,
        reader.inventory(),
        a.rank,
        WoaReconstructionPolicy::OfficialBf16Round,
    )?);
    let load_plan = plan_rank_load(&manifest)?;
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
    emit(
        &json!({"event":"MODEL_LOAD_BEGIN","rank":a.rank,"boot":a.boot,"load_plan":load_plan,"limits":limits,"frequency_profile":frequency_profile,"input_native_ids":input,"input_ids_sha256":input_sha,"input_file_sha256":input_file_sha,"program_sha256":program_sha,"frozen_operations":program.operations(),"model_execution":false,"timed":false}),
    )?;
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
        json!({"event":"WEIGHTS_READY","frequency_profile":frequency_profile,"rank":a.rank,"world":8,"boot":a.boot,"program_sha256":program_sha,"input_ids_sha256":input_sha,"output_tokens":a.output_tokens,"model_revision":MODEL_REVISION,"manifest_sha256":manifest.manifest_sha256,"load_elapsed_seconds":began.elapsed().as_secs_f64(),"budget":budget.snapshot()?,"untimed":true}),
        began + Duration::from_secs(LOAD_READY_SECONDS),
    )?;
    let model_ready_deadline = (Instant::now() + Duration::from_secs(MODEL_READY_SECONDS))
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
    let mut collectives =
        GpuCollectives::new(dev.clone(), link, a.rank, program.clone(), a.proxy_core)?;
    let stream = collectives.compute_stream_owner();
    let kernels = ModelKernels::load(&dev, &stream, a.boot)?;
    let mut model = GpuModel::new(kernels, &stream, weights, config, map, plain, yarn)?;
    let lane = LaneKey {
        boot: a.boot,
        slot: 0,
        generation: 1,
    };
    model.reset_lane(&stream, lane)?;
    controller_gate(
        &a,
        json!({"event":"MODEL_READY","frequency_profile":frequency_profile,"rank":a.rank,"world":8,"boot":a.boot,"program_sha256":program_sha,"input_ids_sha256":input_sha,"output_tokens":a.output_tokens,"model_identity":model.identity(),"elapsed_seconds":began.elapsed().as_secs_f64(),"untimed":true}),
        model_ready_deadline,
    )?;
    let mut generated = Vec::new();
    let mut eos_steps = Vec::new();
    let mut events = Vec::new();
    for step in 0..a.output_tokens {
        let current = if step == 0 {
            input.clone()
        } else {
            vec![*generated
                .last()
                .context("missing autoregressive native token")?]
        };
        let start = if step == 0 { 0 } else { input.len() + step - 1 };
        let output = model.forward_greedy(
            &stream,
            lane,
            start,
            &current,
            step_ids[step].clone(),
            &mut collectives,
        )?;
        let token = output.selected_native_token;
        if token == 1 {
            eos_steps.push(step);
        }
        generated.push(token);
        emit(
            &json!({"event":"MODEL_NATIVE_TOKEN","frequency_profile":frequency_profile,"rank":a.rank,"world":8,"boot":a.boot,"step":step,"selected_native_token":token,"EOS":token==1,"mode":"fixed_count_diagnostic_continuation_even_after_EOS","natural_completion_quality":false,"benchmark":false,"checked_forward":output}),
        )?;
        events.push(json!({"step":step,"native_token":token,"prefix_tokens":model.position()}));
    }
    let completion = collectives.finish()?;
    ensure!(
        completion.completed_operations == a.output_tokens * OPERATIONS_PER_FORWARD,
        "full model collective operation count differs"
    );
    // All ranks bind the ENTIRE native continuation at final release, after every
    // RDMA op has finished. No per-token CPU/network barrier changes execution.
    let mut agreement = Sha256::new();
    agreement.update(b"dsv41-model-final-native-sequence-v1\0");
    agreement.update(program.digest());
    agreement.update(a.boot.as_bytes());
    agreement.update(input_sha.as_bytes());
    for id in &generated {
        agreement.update(id.to_le_bytes());
    }
    let mut release = final_release;
    release.digest = agreement.finalize().into();
    release.wait(began + Duration::from_secs(WALL_SECONDS))?;
    let result = json!({"schema_version":1,"kind":"native1017_TP8_C1_fixed_continuation_probe","frequency_profile":frequency_profile,"status":"PASS","all_passed":true,"rank":a.rank,"world":8,"boot":a.boot,"model_revision":MODEL_REVISION,"model_execution":true,"external_reference_output_qualified":false,"natural_completion_quality":false,"benchmark":false,"serving_enabled":false,"input_native_ids":input,"input_ids_sha256":input_sha,"input_file_sha256":input_file_sha,"output_native_ids":generated,"native_generated_tokens":a.output_tokens,"EOS_steps":eos_steps,"continued_after_EOS_by_predeclared_design":true,"program_sha256":program_sha,"frozen_operations":program.operations(),"collective_completion":completion,"per_step_tokens":events,"model_identity":model.identity(),"reader_statistics":load_stats,"budget":budget.snapshot()?,"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"hard_wall_seconds":WALL_SECONDS,"weights_and_controller_deadline_seconds":LOAD_READY_SECONDS,"model_construction_release_max_seconds":MODEL_READY_SECONDS,"maximum_native_context":512,"maximum_initial_prefill":128,"maximum_diagnostic_output":32},"scope":"Real official1017 complete40layer+Engram text forward, temp0C1 bounded native continuation; success is an execution/internal-consistency gate only; external-reference numerical correctness, task quality and speed improvement remain unqualified."});
    drop(model);
    drop(collectives);
    dev.synchronize()?;
    Ok(result)
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("model probe hard1200s deadline");
        std::process::exit(124);
    });
    let outcome = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"native1017_TP8_C1_fixed_continuation_probe","status":"FAIL","all_passed":false,"error":format!("{e:#}"),"benchmark":false,"natural_completion_quality":false})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"native1017_TP8_C1_fixed_continuation_probe","status":"FAIL","all_passed":false,"error":"panic","benchmark":false,"natural_completion_quality":false})
        }
    };
    let passed = outcome["all_passed"] == true;
    if let Err(error) = emit(&outcome) {
        eprintln!("final probe evidence write failed {error:#}");
        std::process::exit(1);
    }
    if !passed {
        std::process::exit(1);
    }
}
