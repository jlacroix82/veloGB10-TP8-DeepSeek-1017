//! Model-free native16160-logit GPU argmax -> same-stream8-rank full-pair maxloc.
//! This is composition qualification, not a complete model/head-GEMM benchmark.
//! Parent independently owns exclusivity,8GiB/80C/90C guards and195/210s deadlines.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use gb10_inference::dsv41::{
    collectives::{self as cpu, Maxloc, MaxlocFrame, RoutedLayout, RowKind, HIDDEN, VOCAB, WORLD},
    gpu_collectives::{GpuCollectives, Operation, Program},
    gpu_text::local_max_reference,
    gpu_text_batch::{TextBatchKernels, VOCAB_SHARD},
};
use gb10_inference::net::TpLink;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
};
const WALL_SECONDS: u64 = 180;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const PATTERNS: usize = 5;
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
const ARRAY_LIMIT: usize = 16 * 1024 * 1024;
const HOST_LIMIT: usize = 128 * 1024 * 1024;
const LANE_BOOT: &str = "48d5a0fc-9fd8-4b24-a2bc-ceee8f0d0615";
struct Args {
    rank: usize,
    ips: Vec<IpAddr>,
    port: u16,
    release_port: u16,
    dev: String,
    gid: i32,
    proxy_core: i32,
    main_core: i32,
}

fn args() -> Result<Args> {
    let raw: Vec<_> = std::env::args().skip(1).collect();
    ensure!(raw.len() % 2 == 0, "expected explicit --flag value pairs");
    let allowed = [
        "--rank",
        "--peer-ips",
        "--port",
        "--release-port",
        "--dev",
        "--gid",
        "--proxy-core",
        "--main-core",
    ];
    let mut fields = BTreeMap::new();
    for pair in raw.chunks_exact(2) {
        ensure!(
            allowed.contains(&pair[0].as_str())
                && fields.insert(pair[0].clone(), pair[1].clone()).is_none(),
            "unknown or repeated argument {}",
            pair[0]
        );
    }
    let value = |name: &str| fields.get(name).with_context(|| format!("missing {name}"));
    let rank: usize = value("--rank")?.parse()?;
    let ips: Vec<IpAddr> = value("--peer-ips")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let port: u16 = value("--port")?.parse()?;
    let release_port: u16 = value("--release-port")?.parse()?;
    ensure!(
        rank < WORLD && ips.len() == WORLD && ips.iter().collect::<BTreeSet<_>>().len() == WORLD,
        "require eight distinct rank-ordered IPs and rank0..7"
    );
    ensure!(
        port >= 1024 && release_port >= 1024,
        "use explicit unprivileged private ports"
    );
    let native_ports = gb10_inference::net::nway_link_ports(port, WORLD as i32);
    ensure!(
        native_ports.iter().all(|&p| p <= u16::MAX as u32)
            && release_port != port
            && !native_ports.contains(&(release_port as u32)),
        "TP bootstrap ports overflow or release port overlaps a TP bootstrap/liveness port"
    );
    Ok(Args {
        rank,
        ips,
        port,
        release_port,
        dev: value("--dev")?.clone(),
        gid: value("--gid")?.parse()?,
        proxy_core: fields
            .get("--proxy-core")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(-1),
        main_core: fields
            .get("--main-core")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(-1),
    })
}

fn layout(rows: usize) -> Result<RoutedLayout> {
    RoutedLayout::new(rows, RowKind::IndependentDecodeLanes)
}
fn program() -> Result<Program> {
    let mut ops = Vec::new();
    for rows in WIDTHS {
        ops.push(Operation::RoutedF32 {
            id: ops.len() as u64,
            layout: layout(rows)?,
        });
        for _ in 0..PATTERNS {
            ops.push(Operation::ShardedMaxloc {
                id: ops.len() as u64,
                layout: layout(rows)?,
            });
        }
    }
    Program::new(ops)
}
fn lane_keys(rows: usize) -> Vec<Value> {
    ORDER[..rows]
        .iter()
        .map(|&lane| json!({"boot":LANE_BOOT,"slot":lane,"generation":100+lane}))
        .collect()
}
fn logits(rank: usize, lane: usize, pattern: usize) -> Vec<f32> {
    let mut values = vec![f32::from_bits(0xff7fffff); VOCAB_SHARD];
    let first = (lane * 149 + rank * 19 + 3) % 8192;
    let second = first + 8192;
    let score = match pattern {
        0 => {
            if rank == (lane * 3 + 1) % WORLD {
                f32::from_bits(0x41000001 + lane as u32 * 17)
            } else {
                -(rank as f32) - 1.0
            }
        }
        1 => {
            if rank % 2 == 0 {
                -0.0
            } else {
                0.0
            }
        }
        2 => f32::from_bits(0x40800001 + lane as u32 * 19 + rank as u32),
        3 => match lane % 4 {
            0 => f32::from_bits(1 + rank as u32),
            1 => f32::from_bits(0x80000000 + 8 - rank as u32),
            2 => f32::from_bits(0x7f7ffff8 + rank as u32),
            _ => f32::from_bits(0x3f800000 + ((rank + lane) % WORLD) as u32),
        },
        4 => {
            if rank == (lane + 5) % WORLD {
                f32::from_bits(0x3f000001 + lane as u32 * 23)
            } else {
                return values;
            }
        }
        _ => unreachable!(),
    };
    values[first] = score;
    values[second] = score;
    if pattern == 2 && rank == WORLD - 1 {
        values[VOCAB_SHARD - 1] = f32::from_bits(0x41800001 + lane as u32 * 29);
    }
    values
}
fn head_reference(rows: usize, pattern: usize, id: u64, start: u64) -> Result<Vec<u32>> {
    let mut words = Vec::with_capacity(2 * rows);
    for &lane in &ORDER[..rows] {
        let frames = (0..WORLD)
            .map(|rank| {
                Ok(MaxlocFrame {
                    rank,
                    operation: id,
                    pair: local_max_reference(&logits(rank, lane, pattern), rank)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let value = cpu::maxloc_reference(&frames, start)?.ranks[0];
        words.extend([value.value().to_bits(), value.token_id()]);
    }
    Ok(words)
}
fn control_partial(rank: usize, rows: usize) -> Vec<f32> {
    (0..rows * HIDDEN)
        .map(|i| rank as f32 + ((i / HIDDEN) % 3) as f32 / 8.0 + (i % HIDDEN % 17) as f32 / 1024.0)
        .collect()
}
fn control_reference(rows: usize) -> Vec<f32> {
    (0..rows * HIDDEN)
        .map(|i| 28.0 + ((i / HIDDEN) % 3) as f32 + (i % HIDDEN % 17) as f32 / 128.0)
        .collect()
}
fn words_sha(words: &[u32]) -> String {
    let mut h = Sha256::new();
    for word in words {
        h.update(word.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn floats_sha(values: &[f32]) -> String {
    words_sha(&values.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
}
fn explicit_arrays() -> usize {
    WIDTHS
        .iter()
        .map(|rows| rows * HIDDEN * 4 + PATTERNS * (rows * VOCAB_SHARD * 4 + rows * 2 * 4 + 4))
        .sum()
}
struct Case {
    operation: Operation,
    pattern: Option<usize>,
    input: CudaSlice<f32>,
    pairs: Option<CudaSlice<u32>>,
    status: Option<CudaSlice<u32>>,
    input_sha: String,
    expected_words: Vec<u32>,
}
struct Release {
    rank: usize,
    address: SocketAddr,
    listener: Option<TcpListener>,
    digest: [u8; 32],
    ips: Vec<IpAddr>,
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
            ips: a.ips.clone(),
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
                    Ok((mut stream, peer_address)) => {
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
                                && !streams.contains_key(&source)
                                && peer_address.ip() == self.ips[source],
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
    let started = Instant::now();
    let a = args()?;
    ensure!(
        VOCAB_SHARD == 16160 && VOCAB as usize == WORLD * VOCAB_SHARD,
        "native vocabulary geometry differs"
    );
    ensure!(
        explicit_arrays() <= ARRAY_LIMIT,
        "explicit array bound differs"
    );
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "could not pin oracle main thread"
        );
    }
    let plan = program()?;
    ensure!(plan.end_epoch(0)? == 90, "frozen program epoch differs");
    let release = Release::new(&a, plan.digest())?;
    let dev = CudaDevice::new(0)?;
    let link = TpLink::connect_nway(
        a.rank as i32,
        WORLD as i32,
        &a.ips,
        a.port,
        &a.dev,
        a.gid,
        gb10_inference::tp::TP_SLOT_BYTES,
    )?;
    let mut gpu = GpuCollectives::new(dev.clone(), link, a.rank, plan.clone(), a.proxy_core)?;
    let stream = gpu.compute_stream_owner();
    let head = TextBatchKernels::load(&dev, &stream)?;
    // Every producer and destination is preallocated, synchronously uploaded and
    // retained before the first distributed enqueue. No allocator/readback occurs
    // between a local GPU argmax and its paired RDMA operation.
    let mut cases = Vec::new();
    for op in plan.operations() {
        let rows = op.layout().rows();
        let pattern = if matches!(op, Operation::ShardedMaxloc { .. }) {
            Some((op.id() as usize % 6) - 1)
        } else {
            None
        };
        let input = if let Some(pattern) = pattern {
            ORDER[..rows]
                .iter()
                .flat_map(|&lane| logits(a.rank, lane, pattern))
                .collect()
        } else {
            control_partial(a.rank, rows)
        };
        let expected_words = if let Some(pattern) = pattern {
            head_reference(rows, pattern, op.id(), op.id() * 3)?
        } else {
            control_reference(rows)
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let input_sha = floats_sha(&input);
        let input = dev.htod_sync_copy(&input)?;
        let (pairs, status) = if pattern.is_some() {
            let pairs = dev.alloc_zeros::<u32>(rows * 2)?;
            let mut status = dev.alloc_zeros::<u32>(1)?;
            dev.memset_zeros(&mut status)?;
            (Some(pairs), Some(status))
        } else {
            (None, None)
        };
        cases.push(Case {
            operation: *op,
            pattern,
            input,
            pairs,
            status,
            input_sha,
            expected_words,
        });
    }
    let execution = (|| -> Result<Value> {
        for case in &mut cases {
            let op = case.operation;
            let before = gpu.expected_epoch();
            let rows = op.layout().rows();
            if let Some(pattern) = case.pattern {
                let pairs = case.pairs.as_mut().context("missing owned pair buffer")?;
                let status = case
                    .status
                    .as_mut()
                    .context("missing owned status buffer")?;
                unsafe {
                    head.maxloc_into(&stream, &case.input, pairs, status, rows, a.rank)?;
                }
                gpu.enqueue_maxloc(op.id(), op.layout(), pairs)?;
                println!(
                    "{}",
                    json!({"schema_version":1,"kind":"head_collective_queued","id":op.id(),"operation":"head_maxloc","rows":rows,"pattern":pattern,"lane_keys":lane_keys(rows),"input_sha256_f32le":case.input_sha,"start_epoch":before,"end_epoch":gpu.expected_epoch(),"same_owned_stream":true,"CPU_local_pair_readback_before_RDMA":false})
                );
            } else {
                gpu.enqueue_reduce(op.id(), op.layout(), &mut case.input)?;
                println!(
                    "{}",
                    json!({"schema_version":1,"kind":"head_collective_queued","id":op.id(),"operation":"routed_f32_control","rows":rows,"input_sha256_f32le":case.input_sha,"start_epoch":before,"end_epoch":gpu.expected_epoch()})
                );
            }
        }
        let completion = gpu.finish()?;
        // All distributed GPU work is complete before any result/status readback.
        // Keep the transport owner alive through exact all-rank post-work release.
        let mut checks = Vec::new();
        for case in &cases {
            let op = case.operation;
            let rows = op.layout().rows();
            let actual = if let Some(pairs) = &case.pairs {
                dev.dtoh_sync_copy(pairs)?
            } else {
                dev.dtoh_sync_copy(&case.input)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            };
            let status = if let Some(status) = &case.status {
                dev.dtoh_sync_copy(status)?[0]
            } else {
                0
            };
            let mismatches = usize::from(actual.len() != case.expected_words.len())
                + actual
                    .iter()
                    .zip(&case.expected_words)
                    .filter(|(a, b)| a != b)
                    .count();
            let report = if let Some(pattern) = case.pattern {
                json!({"id":op.id(),"kind":"head_maxloc","rows":rows,"pattern":pattern,"lane_keys":lane_keys(rows),"input_sha256_f32le":case.input_sha,"start_epoch":op.id()*3,"end_epoch":(op.id()+1)*3,"local_head_status":status,"actual_score_id_words_u32le":actual,"reference_score_id_words_u32le":case.expected_words,"actual_sha256_u32le":words_sha(&actual),"reference_sha256_u32le":words_sha(&case.expected_words),"bitwise_mismatches":mismatches})
            } else {
                json!({"id":op.id(),"kind":"routed_f32_control","rows":rows,"input_sha256_f32le":case.input_sha,"start_epoch":op.id()*3,"end_epoch":(op.id()+1)*3,"local_head_status":status,"elements":actual.len(),"actual_sha256_u32le":words_sha(&actual),"reference_sha256_u32le":words_sha(&case.expected_words),"bitwise_mismatches":mismatches})
            };
            println!(
                "{}",
                json!({"schema_version":1,"kind":"head_collective_result","rank":a.rank,"check":report})
            );
            checks.push(report);
            ensure!(
                status == 0 && mismatches == 0,
                "head/collective full result or lane association differs"
            );
        }
        release.wait(started + Duration::from_secs(WALL_SECONDS - 10))?;
        ensure!(
            started.elapsed() < Duration::from_secs(WALL_SECONDS),
            "oracle deadline"
        );
        Ok(
            json!({"schema_version":1,"kind":"dsv41_head_collective_oracle","status":"PASS","all_passed":true,"rank":a.rank,"world":WORLD,"RDMA":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"full_batched_model_qualified":false,"native_vocab_per_rank":VOCAB_SHARD,"widths":WIDTHS,"checks":checks,"completion":completion,"head_kernel":{"source_id_hex":format!("{:016x}",head.source_id()),"ptx_sha256":head.ptx_sha256(),"cuda_source_sha256":head.cuda_source_sha256()},"composition":{"same_owned_stream":true,"all_inputs_preuploaded":true,"CPU_local_pair_readback_before_RDMA":false,"CPU_result_readback_after_all_RDMA":true,"all_rank_release":true},"limits":{"wall_seconds":WALL_SECONDS,"explicit_device_array_bytes":explicit_arrays(),"maximum_device_array_bytes":ARRAY_LIMIT,"maximum_host_array_bytes":HOST_LIMIT,"scope":"excludes native rings,Cuda context,module and small transport guard arrays"},"elapsed_seconds":started.elapsed().as_secs_f64(),"scope":"native16160-logit local GPU greedy helper composed with actual8-rank full-score/index transport, independent lane association only; no learned head weights, model forward, natural tasks or speed claim"}),
        )
    })();
    if execution.is_err() {
        gpu.abort_session();
        let _ = dev.synchronize();
    }
    // Queued owners remain alive until the abort-before-fence error path above or
    // complete checked transport + release. No fake model receipt is constructed.
    execution
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_head_collective_oracle","status":"FAIL","all_passed":false,"error":"hard wall-time limit"})
        );
        std::process::exit(124);
    });
    let report = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_head_collective_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_head_collective_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{report}");
    if report["all_passed"] != true {
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn composed_fixture_matches_independent_python_words() {
        let fixture: Value =
            serde_json::from_str(include_str!("dsv41_head_collective_expected.json")).unwrap();
        let cases = fixture["checks"].as_array().unwrap();
        assert_eq!(cases.len(), 30);
        for (operation, case) in program().unwrap().operations().iter().zip(cases) {
            let rows = operation.layout().rows();
            let expected = if matches!(operation, Operation::ShardedMaxloc { .. }) {
                head_reference(
                    rows,
                    (operation.id() as usize % 6) - 1,
                    operation.id(),
                    operation.id() * 3,
                )
                .unwrap()
            } else {
                control_reference(rows)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            };
            assert_eq!(
                words_sha(&expected),
                case["actual_sha256_u32le"].as_str().unwrap()
            );
            if case["kind"] == "head_maxloc" {
                assert_eq!(
                    serde_json::to_value(expected).unwrap(),
                    case["actual_score_id_words_u32le"]
                );
            }
        }
    }
    #[test]
    fn association_negative_controls_and_bounds() {
        let base = head_reference(16, 0, 1, 3).unwrap();
        let mut rotated = base.clone();
        rotated.rotate_left(2);
        assert_ne!(base, rotated);
        let mut index_only = base.clone();
        for pair in index_only.chunks_exact_mut(2) {
            pair[0] = 0;
        }
        assert_ne!(base, index_only);
        let end = head_reference(16, 2, 1, 3).unwrap();
        assert!(end.chunks_exact(2).all(|pair| pair[1] == 129279));
        let zero = head_reference(16, 1, 1, 3).unwrap();
        assert!(zero
            .chunks_exact(2)
            .all(|pair| pair[0] == 0x80000000 && pair[1] < 16160));
        let words = head_reference(16, 3, 1, 3).unwrap();
        assert!(words.chunks_exact(2).any(|pair| pair[0] == 8));
        assert!(words.chunks_exact(2).any(|pair| pair[0] == 0x80000001));
        assert_eq!(program().unwrap().operations().len(), 30);
        assert_eq!(program().unwrap().end_epoch(0).unwrap(), 90);
        assert!(explicit_arrays() < ARRAY_LIMIT);
    }
}
