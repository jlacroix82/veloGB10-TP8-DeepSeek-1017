//! True packed official1017 TP8 diagnostic. Six frozen cases C1/B2/4/8/16,
//! eight native outputs per lane, one resident model and one frozen RoCE program.
//! Actual EOS is a retained failure; no forced continuation, quality or speed claim.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    batch_plan::PackedKind,
    config::{Dsv41Config, MODEL_REVISION},
    engram::CompressedTokenMap,
    gpu_batch_model::{BatchLaneInput, BatchModelKernels, BatchModelOperationIds, GpuBatchModel},
    gpu_collectives::{GpuCollectives, Operation, Program},
    gpu_gemm::GemmBackend,
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
    requests_file: PathBuf,
    output_tokens: usize,
    boot: Uuid,
    load_release: SocketAddr,
    gemm_backend: GemmBackend,
    metadata_only: bool,
    execution_snapshot_sha256: String,
}
fn args() -> Result<Args> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(raw.len() % 2 == 0, "explicit --flag value pairs required");
    let allowed = [
        "--fp8-gemm-backend",
        "--metadata-only",
        "--execution-snapshot-sha256",
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
        "--requests-file",
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
        [8, 32].contains(&output_tokens),
        "frozen packed diagnostic requires explicit8 or32 outputs per lane"
    );
    let gid = v("--gid")?.parse::<i32>()?;
    ensure!(
        gid >= 0 && !v("--dev")?.is_empty(),
        "explicit HCA/GID required"
    );
    ensure!(
        f.get("--metadata-only")
            .is_none_or(|v| v == "true" || v == "false"),
        "metadata-only must be explicit true/false"
    );
    let boot = Uuid::parse_str(v("--boot")?)?;
    ensure!(!boot.is_nil(), "nonzero actual run bootUUID required");
    let snapshot = v("--execution-snapshot-sha256")?.clone();
    ensure!(
        snapshot.len() == 64 && snapshot.bytes().all(|b| b.is_ascii_hexdigit()),
        "exact frozen execution snapshotSHA256 required"
    );
    Ok(Args {
        gemm_backend: GemmBackend::parse(
            f.get("--fp8-gemm-backend")
                .map(String::as_str)
                .unwrap_or("original"),
        )?,
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
        frequency_backend: FrequencyBackend::parse(v("--frequency-backend")?)?,
        compressed_map: v("--compressed-map")?.into(),
        requests_file: v("--requests-file")?.into(),
        output_tokens,
        boot,
        load_release: SocketAddr::new(load_ip, load_port),
        metadata_only: f
            .get("--metadata-only")
            .map(|v| v == "true")
            .unwrap_or(false),
        execution_snapshot_sha256: snapshot,
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
            && ack["binary_sha256"] == ready["binary_sha256"]
            && ack["execution_snapshot_sha256"] == ready["execution_snapshot_sha256"],
        "controller release identity/phase mismatch"
    );
    emit(
        &json!({"event":"CONTROLLER_RELEASE_ACCEPTED","phase":ready["event"],"rank":a.rank,"boot":a.boot,"program_sha256":ready["program_sha256"],"frequency_profile":ready["frequency_profile"]}),
    )
}

struct Release {
    rank: usize,
    address: SocketAddr,
    peers: Vec<IpAddr>,
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
            peers: a.ips.clone(),
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
                    Ok((mut stream, peer)) => {
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
                                && peer.ip() == self.peers[source]
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

const PACK8_SHA: &str = "bfe74b8d5d23ccbd0d33a08d876d6cabd0dcaafc603239edbe79f24f76ce1c46";
const PACK32_SHA: &str = "d8636a1c990554ca32bfcd1d7bfb791d419f696544feb1c0d8d329d0639bbfdc";
// The independent packed reference schedule binds the actual source d412/8da
// helper used by the qualified atomic ELF. The previous constants acc5/f56f
// bound an older77f/4766 helper; operation/payload/epoch bytes are unchanged.
const PROGRAM32_SHA: &str = "3e6763da0554978d1b041e62f332880e5135bc04ac63c3f147fcd0b81db27ef9";
const KIND: &str = "native1017_TP8_true_packed_reference_probe_v1";
#[derive(Clone)]
struct PlannedLane {
    key: LaneKey,
    ids: Vec<u32>,
    full: Option<usize>,
    ordinal: Option<usize>,
    expected: Option<u32>,
    prefix: Vec<u32>,
}
#[derive(Clone)]
struct Step {
    global: usize,
    case: usize,
    kind: PackedKind,
    lanes: Vec<PlannedLane>,
    ops: BatchModelOperationIds,
    epoch_before: u64,
    epoch_after: u64,
}
struct Pack {
    sha256: String,
    value: Value,
    steps: Vec<Step>,
    program: Program,
    generated: usize,
    ignored: usize,
}
fn native_sha(ids: &[u32]) -> String {
    sha(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
fn ids(v: &Value) -> Result<Vec<u32>> {
    let ids: Vec<u32> = serde_json::from_value(v.clone())?;
    ensure!(
        !ids.is_empty() && ids.iter().all(|&v| v < 129280),
        "exact native vocabulary IDs required"
    );
    Ok(ids)
}
fn load_pack(a: &Args) -> Result<Pack> {
    let raw = bounded_file(&a.requests_file, 64 * 1024)?;
    let pack_sha = sha(&raw);
    ensure!(
        pack_sha
            == if a.output_tokens == 8 {
                PACK8_SHA
            } else {
                PACK32_SHA
            },
        "frozen six-case native pack changed"
    );
    let value: Value = serde_json::from_slice(&raw)?;
    ensure!(
        value["model_revision"] == MODEL_REVISION
            && value["output_tokens_per_lane"] == a.output_tokens
            && value["forced_continuation_after_EOS"] == false,
        "frozen pack semantics differ"
    );
    let mut steps = Vec::new();
    let mut operations = Vec::new();
    let mut generated = 0;
    let mut ignored = 0;
    let mut epoch = 0_u64;
    for (case, entry) in value["cases"]
        .as_array()
        .context("six cases")?
        .iter()
        .enumerate()
    {
        let members = entry["members"].as_array().context("case members")?;
        let mut prefix: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut outputs: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for m in members {
            let slot = m["slot"].as_u64().context("slot")? as usize;
            prefix.insert(slot, Vec::new());
            outputs.insert(slot, Vec::new());
        }
        let mut append = |kind: PackedKind,
                          current: Vec<(Value, Vec<u32>, Option<usize>)>,
                          prefix: &mut BTreeMap<usize, Vec<u32>>,
                          outputs: &mut BTreeMap<usize, Vec<u32>>|
         -> Result<()> {
            let mut lanes = Vec::new();
            let mut rows = 0;
            for (m, input, full) in current {
                let slot = m["slot"].as_u64().context("slot")? as usize;
                let generation = m["generation"].as_u64().context("generation")?;
                prefix.get_mut(&slot).context("prefix")?.extend(&input);
                let p = prefix[&slot].clone();
                ensure!(p.len() <= 512, "explicit learned prefix ceiling");
                let finished = kind == PackedKind::DecodeLanes || full == Some(p.len());
                let out = outputs.get_mut(&slot).context("outputs")?;
                let expected = if finished {
                    let work = m["workload"].as_str().context("workload")?;
                    let reference = ids(&value["corpora"][work]["expected_output_native_ids"])?;
                    let v = reference[out.len()];
                    ensure!(v != 1, "reference cannot force EOS continuation");
                    out.push(v);
                    generated += 1;
                    Some(v)
                } else {
                    ignored += 1;
                    None
                };
                rows += input.len();
                lanes.push(PlannedLane {
                    key: LaneKey {
                        boot: a.boot,
                        slot,
                        generation,
                    },
                    ids: input,
                    full,
                    ordinal: if finished { Some(out.len()) } else { None },
                    expected,
                    prefix: p,
                });
            }
            let ops = BatchModelOperationIds::consecutive(operations.len() as u64 + 1)?;
            let wire = ops.operations(rows, lanes.len(), kind)?;
            // Derive the source's three serial rounds from each typed
            // chunk layout; hash the 7MiB immutable transport PTX ONCE in
            // the final Program factory rather than once per prefix.
            let span: u64 = wire
                .iter()
                .map(|op| match op {
                    Operation::RoutedF32 { layout, .. } => 3 * layout.chunk_count() as u64,
                    Operation::ShardedMaxloc { .. } => 3,
                    Operation::MatrixF32 { layout, .. } => 3 * layout.chunk_count() as u64,
                    Operation::MatrixBf16 { .. } => {
                        unreachable!("short packed all-key plan has no BF16 network operation")
                    }
                })
                .sum();
            operations.extend(wire);
            let end = epoch.checked_add(span).context("packed epoch overflow")?;
            steps.push(Step {
                global: steps.len(),
                case,
                kind,
                lanes,
                ops,
                epoch_before: epoch,
                epoch_after: end,
            });
            epoch = end;
            Ok(())
        };
        loop {
            let active = members
                .iter()
                .filter(|m| {
                    let slot = m["slot"].as_u64().unwrap() as usize;
                    let w = m["workload"].as_str().unwrap();
                    prefix[&slot].len()
                        < value["corpora"][w]["prompt_native_ids"]
                            .as_array()
                            .unwrap()
                            .len()
                })
                .cloned()
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }
            let chunk = 128 / active.len();
            let mut current = Vec::new();
            for m in active {
                let slot = m["slot"].as_u64().unwrap() as usize;
                let w = m["workload"].as_str().unwrap();
                let prompt = ids(&value["corpora"][w]["prompt_native_ids"])?;
                let start = prefix[&slot].len();
                current.push((
                    m,
                    prompt[start..(start + chunk).min(prompt.len())].to_vec(),
                    Some(prompt.len()),
                ));
            }
            append(
                PackedKind::PrefillSegments,
                current,
                &mut prefix,
                &mut outputs,
            )?;
        }
        for wave in 0..a.output_tokens - 1 {
            let off = (wave + 1) % members.len();
            let mut order = members[off..]
                .iter()
                .chain(&members[..off])
                .cloned()
                .collect::<Vec<_>>();
            if wave % 2 == 1 {
                order.reverse();
            }
            let current = order
                .into_iter()
                .map(|m| {
                    let slot = m["slot"].as_u64().unwrap() as usize;
                    let token = *outputs[&slot].last().unwrap();
                    (m, vec![token], None)
                })
                .collect();
            append(PackedKind::DecodeLanes, current, &mut prefix, &mut outputs)?;
        }
    }
    let program = Program::new_model_sequence(operations)?;
    ensure!(
        steps.len() == 22 + 6 * (a.output_tokens - 1)
            && program.operations().len() == steps.len() * 84
            && program.end_epoch(0)? == 24690 + (a.output_tokens as u64 - 8) * 1512
            && generated == 32 * a.output_tokens
            && ignored == 178,
        "frozen packed schedule counts differ"
    );
    ensure!(
        sha_program(&program)
            == if a.output_tokens == 8 {
                "78232822391f72e1daa963ed54428c4d8440bfac8509ed92c2110cb30f7a0396"
            } else {
                PROGRAM32_SHA
            },
        "actual compiled wire program differs from frozen independent CPU witness"
    );
    Ok(Pack {
        sha256: pack_sha,
        value,
        steps,
        program,
        generated,
        ignored,
    })
}
fn sha_program(p: &Program) -> String {
    p.digest().iter().map(|b| format!("{b:02x}")).collect()
}
fn step_metadata(s: &Step) -> Value {
    json!({"global_step":s.global,"case_index":s.case,"kind":s.kind,"epoch_before":s.epoch_before,"epoch_after":s.epoch_after,"operation_ids":s.ops,"rows":s.lanes.iter().map(|l|l.ids.len()).sum::<usize>(),"lanes":s.lanes.iter().map(|l|json!({"lane":l.key,"input_native_ids":l.ids,"full_prompt_tokens":l.full,"expected_output_ordinal":l.ordinal,"expected_selected_native_id":l.expected,"expected_prefix_native_ids_sha256":native_sha(&l.prefix)})).collect::<Vec<_>>()})
}
fn binary_sha() -> Result<String> {
    let path = std::env::current_exe()?;
    Ok(sha(&bounded_file(&path, 256 * 1024 * 1024)?))
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
    "batch_layout":component!("DSV41_BATCH_LAYOUT_SOURCE_ID","../src/ptx/dsv41_batch_layout.ptx"),
    "batch_attention":component!("DSV41_BATCH_ATTENTION_SOURCE_ID","../src/ptx/dsv41_batch_attention.ptx"),
    "batch_compression":component!("DSV41_BATCH_COMPRESSION_SOURCE_ID","../src/ptx/dsv41_batch_compression.ptx"),
    })
}
fn common(a: &Args, pack: &Pack, binary: &str) -> Value {
    json!({"rank":a.rank,"world":8,"boot":a.boot,"model_revision":MODEL_REVISION,"request_pack_sha256":pack.sha256,"program_sha256":sha_program(&pack.program),"input_ids_sha256":pack.value["corpora"]["code"]["prompt_native_ids_sha256_u32le"],"input_ids_sha256_scope":"code75 reference only; complete mixed inputs are bound by request_pack_sha256","frequency_profile":a.frequency_backend.identity(),"global_checkpoint_geometry_binding":pack.value["global_checkpoint_geometry_binding"],"binary_sha256":binary,"execution_snapshot_sha256":a.execution_snapshot_sha256,"source_build_id":env!("SOURCE_BUILD_ID"),"source_commit":env!("TUNE_GIT_COMMIT"),"fp8_gemm_backend":a.gemm_backend,"fp8_gemm_policy":{"eligible_rows":[1,16],"eligible_inner":5120,"eligible_outputs":[512,1280,2304],"persistent_contribution_bytes":if a.gemm_backend==GemmBackend::Original{0}else{gb10_inference::dsv41::gpu_gemm::HOT_GROUP_WORKSPACE_ELEMENTS*4},"other_shapes_and_FP4":"original"},"kernel_identities":kernel_identities(),"compiled_isolated_components":21,"intended_actual_model_loaded_modules":["numerics","engram","collectives","gemm","dense","hc","moe","cache_quant","position","experts","chunk_store","text_batch","batch_layout","batch_attention","batch_compression"],"max_lanes":16,"maximum_learned_prefix":512,"output_tokens_per_lane":a.output_tokens,"benchmark":false,"natural_completion_quality":false})
}
fn with_event(common: &Value, event: &str, extra: Value) -> Value {
    let mut value = common.clone();
    let object = value.as_object_mut().unwrap();
    object.insert("event".into(), event.into());
    for (k, v) in extra.as_object().unwrap() {
        object.insert(k.clone(), v.clone());
    }
    value
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let a = args()?;
    ensure!(
        a.frequency_backend == FrequencyBackend::CudaTorch,
        "qualified comparison requires explicit cuda-torch frequency backend"
    );
    let pack = load_pack(&a)?;
    let binary = binary_sha()?;
    let common = common(&a, &pack, &binary);
    if a.metadata_only {
        return Ok(with_event(
            &common,
            "PACKED_MODEL_CPU_METADATA",
            json!({"schema_version":1,"kind":"CPU_only_packed_model_probe_metadata","all_passed":true,"metadata_only":true,"GPU_initialized":false,"checkpoint_reader_opened":false,"network_opened":false,"operation_count":pack.program.operations().len(),"step_count":pack.steps.len(),"expected_end_epoch":pack.program.end_epoch(0)?,"cluster_generated_native_tokens":pack.generated,"ignored_intermediate_prefill_head_values":pack.ignored,"frozen_operations":pack.program.operations(),"planned_steps":pack.steps.iter().map(step_metadata).collect::<Vec<_>>(),"limits":{"hard_wall_seconds":WALL_SECONDS,"load_ready_seconds":LOAD_READY_SECONDS,"model_ready_seconds":MODEL_READY_SECONDS}}),
        ));
    }
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "main CPU placement failed"
        );
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
    ensure!(
        gb10_inference::dsv41::gpu_model_chunk::global_checkpoint_binding(&manifest)?
            == pack.value["global_checkpoint_geometry_binding"]
                .as_str()
                .context("pack geometry")?,
        "actual manifest/global reference binding differs"
    );
    let load_plan = plan_rank_load(&manifest)?;
    let dev = CudaDevice::new(0)?;
    let load_stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let limits = WeightLoadLimits {
        maximum_resident_bytes: 72 * GIB,
        maximum_single_allocation_bytes: 16 * GIB,
        maximum_staging_bytes: 4 * 1024 * 1024,
        available_floor_bytes: 8 * GIB,
        reserved_nonweight_bytes: 4 * GIB,
        maximum_load_seconds: LOAD_READY_SECONDS,
    };
    let budget = GpuWeightBudget::new(limits.clone())?;
    emit(&with_event(
        &common,
        "MODEL_LOAD_BEGIN",
        json!({"load_plan":load_plan,"limits":limits,"timed":false,"model_execution":false}),
    ))?;
    let weights = RankModelWeights::load(
        &dev,
        &load_stream,
        &mut reader,
        manifest.clone(),
        &budget,
        |e| {
            emit(&with_event(
                &common,
                "MODEL_SCOPE_LOAD_EVIDENCE",
                json!({"timed":false,"evidence":e}),
            ))
        },
    )?;
    let load_stats = reader.statistics().clone();
    controller_gate(
        &a,
        with_event(
            &common,
            "WEIGHTS_READY",
            json!({"manifest_sha256":manifest.manifest_sha256,"load_elapsed_seconds":began.elapsed().as_secs_f64(),"budget":budget.snapshot()?,"untimed":true}),
        ),
        began + Duration::from_secs(LOAD_READY_SECONDS),
    )?;
    let model_deadline = (Instant::now() + Duration::from_secs(MODEL_READY_SECONDS))
        .min(began + Duration::from_secs(WALL_SECONDS));
    let mut release = Release::new(&a, pack.program.digest())?;
    let link = TpLink::connect_nway(
        a.rank as i32,
        8,
        &a.ips,
        a.port,
        &a.dev,
        a.gid,
        gb10_inference::tp::TP_SLOT_BYTES,
    )?;
    let mut comm = GpuCollectives::new(
        dev.clone(),
        link,
        a.rank,
        pack.program.clone(),
        a.proxy_core,
    )?;
    let stream = comm.compute_stream_owner();
    let k =
        BatchModelKernels::load_with_gemm_backend(&dev, stream.clone(), a.boot, a.gemm_backend)?;
    let mut model = GpuBatchModel::new(k, weights, config, map, plain, yarn, &comm, 16)?;
    let owner = comm.completion_identity();
    controller_gate(
        &a,
        with_event(
            &common,
            "MODEL_READY",
            json!({"manifest_sha256":manifest.manifest_sha256,"completion_owner":owner,"loaded_model_count":1,"model_constructed_count":1,"elapsed_seconds":began.elapsed().as_secs_f64(),"untimed":true}),
        ),
        model_deadline,
    )?;
    let cases = pack.value["cases"].as_array().unwrap();
    let mut actual: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
    let mut active_case = None;
    let mut actual_ignored = 0;
    let mut checked_steps = 0;
    for step in &pack.steps {
        ensure!(
            Instant::now() < began + Duration::from_secs(WALL_SECONDS),
            "packed whole-process deadline before next native launch"
        );
        if active_case != Some(step.case) {
            for m in cases[step.case]["members"].as_array().unwrap() {
                let key = LaneKey {
                    boot: a.boot,
                    slot: m["slot"].as_u64().unwrap() as usize,
                    generation: m["generation"].as_u64().unwrap(),
                };
                model.reset_lane(key)?;
                actual.insert((step.case, key.slot), Vec::new());
                emit(&with_event(
                    &common,
                    "PACKED_MODEL_RESET",
                    json!({"case_id":cases[step.case]["case_id"],"lane":key,"completion_owner":owner,"actual_epoch":comm.expected_epoch(),"checked_reset_returned":true,"checkpoint_load_count":1}),
                ))?;
            }
            active_case = Some(step.case);
        }
        let mut inputs = Vec::new();
        for lane in &step.lanes {
            let native = if step.kind == PackedKind::DecodeLanes {
                vec![*actual[&(step.case, lane.key.slot)]
                    .last()
                    .context("missing actual autoregressive output")?]
            } else {
                lane.ids.clone()
            };
            ensure!(
                native == lane.ids,
                "actual previous native output differs from frozen next-step plan"
            );
            inputs.push(BatchLaneInput {
                lane: lane.key,
                native_ids: native,
                full_prompt_tokens: lane.full,
            });
        }
        ensure!(
            comm.expected_epoch() == step.epoch_before,
            "actual packed operation cursor differs before step"
        );
        let proof = model.execute_checked(step.kind, &inputs, step.ops.clone(), &mut comm)?;
        emit(&with_event(
            &common,
            "PACKED_MODEL_STEP",
            json!({"global_step":step.global,"case_id":cases[step.case]["case_id"],"checked_forward":proof}),
        ))?;
        ensure!(
            proof.all_checks_passed()
                && proof.epoch_after() == step.epoch_after
                && proof.lanes().len() == step.lanes.len()
                && comm.completion_identity() == owner,
            "checked packed full-step owner/count/epoch differs"
        );
        for (observed, planned) in proof.lanes().iter().zip(&step.lanes) {
            ensure!(
                observed.lane == planned.key
                    && observed.consumed_prefix_tokens == planned.prefix.len()
                    && observed.prefix_native_ids_sha256 == native_sha(&planned.prefix)
                    && observed.generated_output_ordinal == planned.ordinal
                    && observed.intermediate_prefill_head_ignored == planned.ordinal.is_none(),
                "packed lane/prefix/output association differs"
            );
            if let Some(want) = planned.expected {
                ensure!(observed.selected_native_id!=1,"actual generated native EOS retained as an early diagnostic stop; no continuation");
                ensure!(
                    observed.selected_native_id == want,
                    "native reference mismatch case{} slot{} ordinal{:?}: observed{} expected{}",
                    step.case,
                    planned.key.slot,
                    planned.ordinal,
                    observed.selected_native_id,
                    want
                );
                actual
                    .get_mut(&(step.case, planned.key.slot))
                    .unwrap()
                    .push(observed.selected_native_id);
            } else {
                actual_ignored += 1;
            }
        }
        checked_steps += 1;
    }
    let completion = comm.finish()?;
    ensure!(
        completion.completed_operations == pack.program.operations().len()
            && checked_steps == pack.steps.len()
            && actual_ignored == 178,
        "full packed program completion counts differ"
    );
    let case_results=cases.iter().enumerate().map(|(index,case)|{let members=case["members"].as_array().unwrap().iter().map(|m|{let slot=m["slot"].as_u64().unwrap()as usize;let output=&actual[&(index,slot)];json!({"request_id":m["request_id"],"lane":{"boot":a.boot,"slot":slot,"generation":m["generation"]},"workload":m["workload"],"output_native_ids":output,"output_native_ids_sha256_u32le":native_sha(output)})}).collect::<Vec<_>>();json!({"case_id":case["case_id"],"lanes":case["lanes"],"generated_native_tokens":case["lanes"].as_u64().unwrap()*a.output_tokens as u64,"actual_full_width_decode_steps":a.output_tokens-1,"members":members})}).collect::<Vec<_>>();
    ensure!(
        actual.values().all(|v| v.len() == a.output_tokens)
            && actual.values().map(Vec::len).sum::<usize>() == pack.generated,
        "every real lane must produce its complete predeclared diagnostic native count"
    );
    let mut h = Sha256::new();
    h.update(b"dsv41-packed-model-final-native-sequences-v1\0");
    h.update(pack.program.digest());
    h.update(a.boot.as_bytes());
    h.update(pack.sha256.as_bytes());
    h.update(serde_json::to_vec(&case_results)?);
    release.digest = h.finalize().into();
    let agreement_sha = release
        .digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    release.wait(began + Duration::from_secs(WALL_SECONDS))?;
    let result = with_event(
        &common,
        "PACKED_MODEL_COMPLETE",
        json!({"schema_version":1,"kind":KIND,"status":"PASS","all_passed":true,"model_execution":true,"HTTP_serving_enabled":false,"loaded_model_count":1,"model_constructed_count":1,"completion_owner":owner,"checked_model_steps":checked_steps,"cluster_generated_native_tokens":pack.generated,"ignored_intermediate_prefill_head_values":actual_ignored,"duplicated_rank_tokens_never_summed":true,"cases":case_results,"collective_completion":completion,"all_rank_terminal_agreement_confirmed":true,"all8_sequence_release_sha256":agreement_sha,"forced_continuation_after_EOS":false,"finish_classification":format!("predeclared{}token diagnostic output limit; no natural quality completion claim",a.output_tokens),"fp8_gemm_dispatch":model.gemm_dispatch_snapshot(),"reader_statistics":load_stats,"budget":budget.snapshot()?,"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"hard_wall_seconds":WALL_SECONDS,"weights_and_controller_deadline_seconds":LOAD_READY_SECONDS,"model_construction_release_max_seconds":MODEL_READY_SECONDS,"maximum_learned_prefix":512,"maximum_flat_rows":128,"maximum_lanes":16,"output_tokens_per_lane":a.output_tokens},"scope":"One resident official1017 model, true shared packed full40layer+Engram computation and per-lane native maxloc. Exact predeclared8-or32-native-token reference gate only; no throughput/task-quality/promotion claim."}),
    );
    drop(model);
    drop(comm);
    dev.synchronize()?;
    Ok(result)
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("packed model hard1200s deadline");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":KIND,"status":"FAIL","all_passed":false,"error":format!("{e:#}"),"benchmark":false,"natural_completion_quality":false})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":KIND,"status":"FAIL","all_passed":false,"error":"panic","benchmark":false,"natural_completion_quality":false})
        }
    };
    let okay = result["all_passed"] == true;
    if let Err(e) = emit(&result) {
        eprintln!("packed probe final evidence error {e:#}");
        std::process::exit(1);
    }
    if !okay {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_current_helper_and_full_program_identity_are_pinned() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dsv41_fp8_hot_dispatch_wire_identity.json"
        ))
        .unwrap();
        assert_eq!(
            fixture["helper_source_id"],
            gb10_inference::dsv41::gpu_collectives::SOURCE_ID
        );
        assert_eq!(
            fixture["helper_ptx_sha256"],
            sha(gb10_inference::dsv41::gpu_collectives::PTX.as_bytes())
        );
        assert_eq!(
            fixture["transport_ptx_sha256"],
            sha(gb10_inference::ptx_embed::GPU_BATCH.as_bytes())
        );
        for count in [8, 32] {
            let mut a = cpu_args();
            a.output_tokens = count;
            if count == 32 {
                a.requests_file="tests/fixtures/dsv41_packed_reference_32.json".into();
            }
            let p = load_pack(&a).unwrap();
            let want = &fixture["schedules"][count.to_string()];
            assert_eq!(sha_program(&p.program), want["program_sha256"]);
            assert_eq!(
                p.program.operations().len(),
                want["operations"].as_u64().unwrap() as usize
            );
            assert_eq!(
                p.program.end_epoch(0).unwrap(),
                want["end_epoch"].as_u64().unwrap()
            );
            assert_eq!(p.steps.len(), want["steps"].as_u64().unwrap() as usize);
            assert_eq!(
                p.generated,
                want["unique_generated_native_tokens"].as_u64().unwrap() as usize
            );
        }
    }
    fn cpu_args() -> Args {
        Args {
            rank:0, ips:(14..22).map(|n|format!("198.51.100.{n}").parse().unwrap()).collect(),
            port:47300,release_port:47400,dev:"CPU_METADATA_DOES_NOT_OPEN_HCA".into(),gid:3,
            proxy_core:-1,main_core:-1,model_dir:"/nonexistent-cpu-metadata-model".into(),
            freq_plain:"/nonexistent-cpu-metadata-plain".into(),freq_yarn:"/nonexistent-cpu-metadata-yarn".into(),
            frequency_backend:FrequencyBackend::CudaTorch,compressed_map:"/nonexistent-cpu-metadata-map".into(),
            requests_file:"tests/fixtures/dsv41_packed_reference_8.json".into(),
            output_tokens:8,boot:Uuid::from_u128(1),load_release:"172.31.240.10:47401".parse().unwrap(),
            gemm_backend:GemmBackend::Original,metadata_only:true,execution_snapshot_sha256:"0".repeat(64),
        }
    }
    #[test]
    fn actual_compiled_wire_plan_matches_independent_frozen64step_witness() {
        let a = cpu_args();
        let p = load_pack(&a).unwrap();
        assert_eq!(p.program.operations().len(), 5376);
        assert_eq!(p.program.end_epoch(0).unwrap(), 24690);
        assert_eq!(p.generated, 256);
        assert_eq!(p.ignored, 178);
        for case in 0..6 {
            let decode = p
                .steps
                .iter()
                .filter(|s| s.case == case && s.kind == PackedKind::DecodeLanes)
                .collect::<Vec<_>>();
            assert_eq!(decode.len(), 7);
            let width = p.value["cases"][case]["lanes"].as_u64().unwrap() as usize;
            for step in decode {
                assert_eq!(step.lanes.len(), width);
                for lane in &step.lanes {
                    let ordinal = lane.ordinal.unwrap();
                    assert!((2..=8).contains(&ordinal));
                    let m = p.value["cases"][case]["members"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|m| m["slot"] == lane.key.slot)
                        .unwrap();
                    let corpus = &p.value["corpora"][m["workload"].as_str().unwrap()];
                    let reference = ids(&corpus["expected_output_native_ids"]).unwrap();
                    assert_eq!(lane.ids, vec![reference[ordinal - 2]]);
                    assert_eq!(lane.expected, Some(reference[ordinal - 1]));
                }
            }
        }
        assert!(p.steps.iter().any(|s| s.kind == PackedKind::PrefillSegments
            && s.lanes.len() == 16
            && s.lanes.iter().map(|l| l.ids.len()).sum::<usize>() == 128));
    }
    #[test]
    fn modified_request_or_allowance_cannot_replace_frozen_comparison_work() {
        let mut a = cpu_args();
        let raw = std::fs::read(&a.requests_file).unwrap();
        let mut pack: Value = serde_json::from_slice(&raw).unwrap();
        struct CpuTemporary(std::path::PathBuf);
        impl Drop for CpuTemporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        use std::os::unix::fs::DirBuilderExt;
        let temporary =
            CpuTemporary(std::env::temp_dir().join(format!("dsv41-packed-cpu-{}", Uuid::new_v4())));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&temporary.0)
            .unwrap();
        a.requests_file = temporary.0.join("changed-pack.json");
        for change in 0..3 {
            let original = pack.clone();
            match change {
                0 => pack["output_tokens_per_lane"] = 7.into(),
                1 => pack["corpora"]["code"]["prompt_native_ids"][0] = 1.into(),
                _ => pack["cases"][5]["members"]
                    .as_array_mut()
                    .unwrap()
                    .reverse(),
            }
            std::fs::write(&a.requests_file, serde_json::to_vec(&pack).unwrap()).unwrap();
            assert!(load_pack(&a).is_err());
            pack = original;
        }
    }
    #[test]
    fn separate_frozen32_gate_reaches_known_output13_and_reserves_complete208steps() {
        let mut a = cpu_args();
        a.output_tokens = 32;
        a.requests_file="tests/fixtures/dsv41_packed_reference_32.json".into();
        let p = load_pack(&a).unwrap();
        assert_eq!(p.sha256, PACK32_SHA);
        assert_eq!(p.steps.len(), 208);
        assert_eq!(p.program.operations().len(), 17472);
        assert_eq!(p.program.end_epoch(0).unwrap(), 60978);
        assert_eq!(p.generated, 1024);
        assert_eq!(p.ignored, 178);
        for case in 0..6 {
            assert_eq!(
                p.steps
                    .iter()
                    .filter(|s| s.case == case && s.kind == PackedKind::DecodeLanes)
                    .count(),
                31
            );
        }
        assert!(p
            .steps
            .iter()
            .flat_map(|s| &s.lanes)
            .any(|l| l.ordinal == Some(13)));
        a.output_tokens = 8;
        assert!(load_pack(&a).is_err());
    }
}
