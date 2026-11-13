//! Eight-process, model-free native-1017 collective oracle. Each rank needs its
//! own GB10 and the same binary/program; no model, clocks or network is changed.
//! Required flags: --rank R --peer-ips IP0,...,IP7 --port P --release-port Q
//!                 --dev HCA --gid N [--proxy-core CPU] [--main-core CPU].
//! The private release TCP port is separate from every native TP bootstrap port.
//! A parent must independently enforce memory/thermal limits and a wall timeout.

use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::collectives::{
    self as cpu, Maxloc, MaxlocFrame, RoutedFrame, RoutedLayout, RowKind, HIDDEN, VOCAB, WORLD,
};
use gb10_inference::dsv41::gpu_collectives::{GpuCollectives, Operation, Program};
use gb10_inference::net::TpLink;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

const WALL_SECONDS: u64 = 180;
const WIDTHS: [usize; 7] = [1, 2, 4, 8, 16, 52, 128];
const PATTERNS: usize = 3;
const SHARD: u32 = VOCAB / WORLD as u32;
const MAX_DEVICE_ARRAY_BYTES: usize = 128 * HIDDEN * 4;

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
    RoutedLayout::new(
        rows,
        if rows <= 16 {
            RowKind::IndependentDecodeLanes
        } else {
            RowKind::ContiguousPrefillRows
        },
    )
}

fn program() -> Result<Program> {
    let mut operations = Vec::new();
    for rows in WIDTHS {
        for _ in 0..PATTERNS {
            let id = operations.len() as u64;
            operations.push(Operation::RoutedF32 {
                id,
                layout: layout(rows)?,
            });
        }
    }
    for rows in WIDTHS {
        for _ in 0..PATTERNS {
            let id = operations.len() as u64;
            operations.push(Operation::ShardedMaxloc {
                id,
                layout: layout(rows)?,
            });
        }
    }
    Program::new(operations)
}

fn partials(layout: RoutedLayout, id: u64, pattern: usize) -> Result<Vec<RoutedFrame>> {
    let mut values = vec![vec![0.0_f32; layout.elements()]; WORLD];
    for rank in 0..WORLD {
        for i in 0..layout.elements() {
            values[rank][i] = match pattern {
                0 => rank as f32 + (i % 97) as f32 / 1024.0,
                1 => match i % 2 {
                    0 => match rank {
                        0 => 1.003_906_25,
                        1 => -1.0,
                        _ => 0.0,
                    },
                    _ => match rank {
                        0 => 16_777_216.0,
                        1 => -16_777_216.0,
                        2 | 3 => 1.0,
                        _ => 0.0,
                    },
                },
                _ => 0.0,
            };
        }
    }
    if pattern == 2 {
        // Routed top6 across expert partition boundaries, with three empty ranks.
        for row in 0..layout.rows() {
            for (slot, expert) in [0, 47, 48, 143, 240, 383].into_iter().enumerate() {
                let owner = layout.expert_owner(expert)?;
                for d in 0..HIDDEN {
                    values[owner][row * HIDDEN + d] +=
                        ((slot + 1) * (row + 1)) as f32 + (d % 17) as f32 / 1024.0;
                }
            }
        }
        ensure!(
            [3, 4, 6]
                .iter()
                .all(|&rank| values[rank].iter().all(|&v| v == 0.0)),
            "empty-rank fixture is invalid"
        );
    }
    values
        .into_iter()
        .enumerate()
        .map(|(rank, values)| RoutedFrame::new(layout, rank, id, values))
        .collect()
}

fn pairs(rows: usize, id: u64, pattern: usize) -> Result<Vec<Vec<MaxlocFrame>>> {
    (0..rows)
        .map(|row| {
            (0..WORLD)
                .map(|rank| {
                    let token = rank as u32 * SHARD + ((row * 17 + rank + 1) as u32 % SHARD);
                    let score = match pattern {
                        0 => ((rank + row) % WORLD) as f32 + 0.125,
                        1 => {
                            if rank == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        _ => [5.0, 7.0, 9.0, 1.0, 2.0, 3.0, 4.0, 0.0][rank],
                    };
                    Ok(MaxlocFrame {
                        rank,
                        operation: id,
                        pair: Maxloc::new(score, token)?,
                    })
                })
                .collect()
        })
        .collect()
}

fn f32_sha(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for v in values {
        hash.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn u32_sha(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for v in values {
        hash.update(v.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

// Pure CPU negative controls prove the chosen fixtures expose the old failure
// classes; success cannot mean that a numerically insensitive fixture was used.
fn negative_controls() -> Result<Value> {
    let layout = layout(1)?;
    let frames = partials(layout, 0, 1)?;
    let correct = cpu::reduce_reference(&frames, 0)?;
    let mut legacy: Vec<Vec<f32>> = frames.iter().map(|f| f.values()[..2].to_vec()).collect();
    for round in cpu::round_order(0)? {
        let old = legacy.clone();
        let bit = 1 << round;
        for rank in 0..WORLD {
            let lower = rank & !bit;
            for d in 0..2 {
                legacy[rank][d] =
                    half::bf16::from_f32(old[lower][d] + old[lower | bit][d]).to_f32();
            }
        }
    }
    ensure!(
        correct.ranks[0][0] == 0.003_906_25 && legacy[0][0] == 0.0,
        "BF16 negative-control fixture does not reject early rounding"
    );
    let source = pairs(1, 1, 2)?.remove(0);
    let winner = cpu::maxloc_reference(&source, 0)?.ranks[0];
    let mut old: Vec<_> = source.iter().map(|f| f.pair).collect();
    for round in cpu::round_order(0)? {
        let before = old.clone();
        let bit = 1 << round;
        for rank in 0..WORLD {
            let lower = rank & !bit;
            let elected = cpu::maxloc_combine(before[lower], before[lower | bit])?;
            old[rank] = Maxloc::new(before[rank].value(), elected.token_id())?;
        }
    }
    ensure!(
        old[0].token_id() != winner.token_id(),
        "index-only maxloc negative-control fixture is insensitive"
    );
    Ok(
        json!({"early_bf16":{"reference":correct.ranks[0][0],"legacy":legacy[0][0],"rejected":true},"index_only_maxloc":{"reference_id":winner.token_id(),"legacy_id":old[0].token_id(),"rejected":true},"scope":"CPU fixture controls; real GPU results are checked separately"}),
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
    let started = Instant::now();
    let a = args()?;
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "could not pin oracle main thread"
        );
    }
    let negative = negative_controls()?;
    let plan = program()?;
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
    let mut checks = Vec::new();
    for operation in plan.operations() {
        let begin = gpu.expected_epoch();
        let pattern = operation.id() as usize % PATTERNS;
        let layout = operation.layout();
        let timer = Instant::now();
        match *operation {
            Operation::RoutedF32 { id, .. } => {
                let source = partials(layout, id, pattern)?;
                let expected = cpu::reduce_reference(&source, begin)?;
                let mut values = dev.htod_copy(source[a.rank].values().to_vec())?;
                gpu.enqueue_reduce(id, layout, &mut values)?;
                gpu.synchronize()?;
                let actual = dev.dtoh_sync_copy(&values)?;
                let reference = &expected.ranks[a.rank];
                let mismatches = actual
                    .iter()
                    .zip(reference)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_error = actual
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .fold(0.0_f64, f64::max);
                let report = json!({"id":id,"kind":"routed_f32","rows":layout.rows(),"row_kind":layout.kind(),"pattern":pattern,"start_epoch":begin,"end_epoch":gpu.expected_epoch(),"chunk_count":layout.chunk_count(),"elements":actual.len(),"bitwise_mismatches":mismatches,"max_abs_error":max_error,"reference_sha256":f32_sha(reference),"actual_sha256":f32_sha(&actual),"elapsed_seconds":timer.elapsed().as_secs_f64()});
                println!(
                    "{}",
                    json!({"kind":"dsv41_collective_check","rank":a.rank,"check":report})
                );
                checks.push(report);
                ensure!(
                    mismatches == 0 && gpu.expected_epoch() == expected.end_epoch,
                    "FP32 numerical/epoch mismatch at operation {id}"
                );
            }
            Operation::ShardedMaxloc { id, .. } => {
                let source = pairs(layout.rows(), id, pattern)?;
                let mut local = Vec::new();
                let mut reference = Vec::new();
                for row in &source {
                    let pair = row[a.rank].pair;
                    local.extend([pair.value().to_bits(), pair.token_id()]);
                    let expected = cpu::maxloc_reference(row, begin)?.ranks[a.rank];
                    reference.extend([expected.value().to_bits(), expected.token_id()]);
                }
                let mut values = dev.htod_copy(local)?;
                gpu.enqueue_maxloc(id, layout, &mut values)?;
                gpu.synchronize()?;
                let actual = dev.dtoh_sync_copy(&values)?;
                let mismatches = actual
                    .iter()
                    .zip(&reference)
                    .filter(|(a, b)| a != b)
                    .count();
                let report = json!({"id":id,"kind":"sharded_maxloc","rows":layout.rows(),"row_kind":layout.kind(),"pattern":pattern,"start_epoch":begin,"end_epoch":gpu.expected_epoch(),"tuple_words":actual.len(),"bitwise_mismatches":mismatches,"reference_sha256":u32_sha(&reference),"actual_sha256":u32_sha(&actual),"reference_first_pair":&reference[..2],"actual_first_pair":&actual[..2],"elapsed_seconds":timer.elapsed().as_secs_f64()});
                println!(
                    "{}",
                    json!({"kind":"dsv41_collective_check","rank":a.rank,"check":report})
                );
                checks.push(report);
                ensure!(
                    mismatches == 0,
                    "full-pair maxloc mismatch at operation {id}"
                );
            }
            Operation::MatrixF32 { .. } | Operation::MatrixBf16 { .. } => {
                anyhow::bail!("phase1 oracle program must not contain matrix operations")
            }
        }
    }
    let completion = gpu.finish()?;
    release.wait(started + Duration::from_secs(WALL_SECONDS - 10))?;
    drop(gpu);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_collective_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":true,"rank":a.rank,"world":WORLD,"hidden":HIDDEN,"widths":WIDTHS,"checks":checks,"completion":completion,"negative_controls":negative,"limits":{"wall_seconds":WALL_SECONDS,"largest_device_payload_bytes":MAX_DEVICE_ARRAY_BYTES,"scope":"payload excludes native transport rings, CUDA context/module and small guard arrays"},"elapsed_seconds":started.elapsed().as_secs_f64(),"shutdown":"all ranks completed RDMA, then independent TCP release, then cooperative abort/join of owned proxies","scope":"model-free eager FP32 and full-pair maxloc component over actual eight-rank transport; not model fidelity, serving or speed qualification"}),
    )
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_collective_oracle","status":"FAIL","all_passed":false,"error":"hard wall-time limit","limit_seconds":WALL_SECONDS})
        );
        std::process::exit(124);
    });
    let result = std::panic::catch_unwind(run);
    let report = match result {
        Ok(Ok(report)) => report,
        Ok(Err(error)) => {
            json!({"schema_version":1,"kind":"dsv41_collective_oracle","status":"FAIL","all_passed":false,"error":format!("{error:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_collective_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{report}");
    if report["all_passed"] != true {
        std::process::exit(1);
    }
}
