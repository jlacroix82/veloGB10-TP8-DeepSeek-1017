//! New actual-eight-rank matrix/cycle collective oracle; no model execution.
//! Reuses the phase1 CLI and final TCP release protocol verbatim. All GPU
//! buffers are prepared before submission; cycles have no synchronization or
//! control exchange. Only finish synchronizes/checks the full queued program.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use gb10_inference::dsv41::collectives::{
    self as cpu, MatrixF32, MatrixFrame, MatrixRole, Maxloc, MaxlocFrame, RoutedFrame,
    RoutedLayout, RowKind, HIDDEN, VOCAB, WORLD,
};
use gb10_inference::dsv41::gpu_collectives::{GpuCollectives, Operation, Program};
use gb10_inference::net::TpLink;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};
const WALL_SECONDS: u64 = 180;
const STEPS: [u64; 3] = [11, 19, 42];
const OP_COUNT: usize = 12;
const FINAL_EPOCH: u64 = 135;
const SHARD: u32 = VOCAB / WORLD as u32;
const ARRAY_BOUND: usize = 32 * 1024 * 1024;
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

fn program() -> Result<Program> {
    let mut operations = Vec::new();
    for rows in [1, 8, 16] {
        for (role, cols) in [
            (MatrixRole::EngramLookup, 6144),
            (MatrixRole::IndexScore, 31),
            (MatrixRole::IndexScore, 513),
        ] {
            operations.push(Operation::MatrixF32 {
                id: operations.len() as u64,
                layout: MatrixF32::new(role, rows, cols)?,
            });
        }
    }
    operations.push(Operation::MatrixF32 {
        id: 9,
        layout: MatrixF32::new(MatrixRole::EngramLookup, 128, 6144)?,
    });
    let routed = RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?;
    operations.push(Operation::RoutedF32 {
        id: 10,
        layout: routed,
    });
    operations.push(Operation::ShardedMaxloc {
        id: 11,
        layout: routed,
    });
    ensure!(operations.len() == OP_COUNT, "matrix program cardinality");
    Program::new_cyclic(operations, STEPS.len() as u64)
}
enum Buffer {
    F32(CudaSlice<f32>),
    Pair(CudaSlice<u32>),
}
struct Prepared {
    step: u64,
    operation: Operation,
    start: u64,
    end: u64,
    buffer: Buffer,
    expected: Vec<u32>,
    source_sha: String,
}
fn words_sha(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn prepare(
    dev: &Arc<CudaDevice>,
    rank: usize,
    step: u64,
    operation: Operation,
    start: u64,
) -> Result<Prepared> {
    let (buffer, expected, end, source) = match operation {
        Operation::MatrixF32 { id, layout } => {
            let frames: Vec<_> = (0..WORLD)
                .map(|source| {
                    let values = (0..layout.elements())
                        .map(|i| {
                            let row = i / layout.cols();
                            let col = i % layout.cols();
                            match layout.role() {
                                MatrixRole::EngramLookup => {
                                    let owner = (row * 24 + col / 256 + step as usize) % WORLD;
                                    if source == owner {
                                        bf16::from_f32(
                                            ((col % 256) as f32 - 128.0) / 16.0
                                                + row as f32 / 4.0
                                                + step as f32 / 64.0,
                                        )
                                        .to_f32()
                                    } else {
                                        0.0
                                    }
                                }
                                MatrixRole::IndexScore => {
                                    if col >= layout.cols() - 3 {
                                        0.0
                                    } else {
                                        source as f32
                                            + step as f32 / 64.0
                                            + row as f32 / 16.0
                                            + (col % 17) as f32 / 1024.0
                                    }
                                }
                            }
                        })
                        .collect();
                    MatrixFrame::new(layout, source, step, id, values)
                })
                .collect::<Result<_>>()?;
            let reference = cpu::reduce_matrix_reference(&frames, start)?;
            let source: Vec<_> = frames[rank].values().iter().map(|v| v.to_bits()).collect();
            (
                Buffer::F32(dev.htod_sync_copy(frames[rank].values())?),
                reference.ranks[rank].iter().map(|v| v.to_bits()).collect(),
                reference.end_epoch,
                source,
            )
        }
        Operation::RoutedF32 { id, layout } => {
            let frames: Vec<_> = (0..WORLD)
                .map(|source| {
                    RoutedFrame::new(
                        layout,
                        source,
                        id,
                        (0..layout.elements())
                            .map(|i| source as f32 + step as f32 / 128.0 + (i % 13) as f32 / 1024.0)
                            .collect(),
                    )
                })
                .collect::<Result<_>>()?;
            let reference = cpu::reduce_reference(&frames, start)?;
            let source: Vec<_> = frames[rank].values().iter().map(|v| v.to_bits()).collect();
            (
                Buffer::F32(dev.htod_sync_copy(frames[rank].values())?),
                reference.ranks[rank].iter().map(|v| v.to_bits()).collect(),
                reference.end_epoch,
                source,
            )
        }
        Operation::ShardedMaxloc { id, .. } => {
            let frames: Vec<_> = (0..WORLD)
                .map(|source| {
                    Ok(MaxlocFrame {
                        rank: source,
                        operation: id,
                        pair: Maxloc::new(
                            ((source + step as usize) % WORLD) as f32 + 0.125,
                            source as u32 * SHARD + ((step + source as u64) % SHARD as u64) as u32,
                        )?,
                    })
                })
                .collect::<Result<_>>()?;
            let reference = cpu::maxloc_reference(&frames, start)?;
            let winner = reference.ranks[rank];
            let local = frames[rank].pair;
            let source = vec![local.value().to_bits(), local.token_id()];
            (
                Buffer::Pair(dev.htod_sync_copy(&source)?),
                vec![winner.value().to_bits(), winner.token_id()],
                reference.end_epoch,
                source,
            )
        }
        Operation::MatrixBf16 { .. } => {
            anyhow::bail!("frozen F32 matrix oracle refuses BF16 operations")
        }
    };
    Ok(Prepared {
        step,
        operation,
        start,
        end,
        buffer,
        expected,
        source_sha: words_sha(&source),
    })
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let a = args()?;
    if a.main_core >= 0 {
        ensure!(
            gb10_inference::net::pin_thread(a.main_core),
            "main CPU placement"
        );
    }
    let plan = program()?;
    let release = Release::new(&a, plan.digest())?;
    let dev = CudaDevice::new(0)?;
    let mut prepared = Vec::new();
    let mut epoch = 0;
    // Upload every distinct source before the first queued cycle. No D2H/H2D
    // transfer, host agreement, stream sync or GPU epoch reset occurs between cycles.
    for step in STEPS {
        for operation in plan.operations() {
            let item = prepare(&dev, a.rank, step, *operation, epoch)?;
            epoch = item.end;
            prepared.push(item);
        }
    }
    ensure!(
        prepared.len() == OP_COUNT * STEPS.len() && epoch == FINAL_EPOCH,
        "CPU reference program extent changed"
    );
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
    let mut submission = Vec::new();
    let mut cursor = 0;
    for step in STEPS {
        gpu.begin_cycle(step)?;
        submission
            .push(json!({"event":"begin_cycle","step":step,"queued_epoch":gpu.expected_epoch()}));
        for _ in 0..OP_COUNT {
            let item = &mut prepared[cursor];
            ensure!(
                item.step == step && item.start == gpu.expected_epoch(),
                "local source/epoch association"
            );
            match (item.operation, &mut item.buffer) {
                (Operation::MatrixF32 { id, layout }, Buffer::F32(values)) => {
                    gpu.enqueue_matrix_for_step(step, id, layout, values)?
                }
                (Operation::RoutedF32 { id, layout }, Buffer::F32(values)) => {
                    gpu.enqueue_reduce(id, layout, values)?
                }
                (Operation::ShardedMaxloc { id, layout }, Buffer::Pair(values)) => {
                    gpu.enqueue_maxloc(id, layout, values)?
                }
                _ => anyhow::bail!("wrong typed source buffer"),
            }
            ensure!(gpu.expected_epoch() == item.end, "queued epoch accounting");
            cursor += 1;
        }
        gpu.end_cycle(step)?;
        submission
            .push(json!({"event":"end_cycle","step":step,"queued_epoch":gpu.expected_epoch()}));
    }
    let completion = gpu.finish()?;
    ensure!(
        completion.device_epoch == FINAL_EPOCH && completion.completed_operations == prepared.len(),
        "actual final device extent"
    );
    let mut checks = Vec::new();
    for item in &prepared {
        let actual: Vec<u32> = match &item.buffer {
            Buffer::F32(values) => dev
                .dtoh_sync_copy(values)?
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            Buffer::Pair(values) => dev.dtoh_sync_copy(values)?,
        };
        let mismatch = usize::from(actual.len() != item.expected.len())
            + actual
                .iter()
                .zip(&item.expected)
                .filter(|(a, b)| a != b)
                .count();
        let (kind, rows, cols, role) = match item.operation {
            Operation::MatrixF32 { layout, .. } => (
                "matrix_f32",
                layout.rows(),
                layout.cols(),
                Some(layout.role()),
            ),
            Operation::RoutedF32 { layout, .. } => ("routed_f32", layout.rows(), HIDDEN, None),
            Operation::ShardedMaxloc { layout, .. } => ("sharded_maxloc", layout.rows(), 2, None),
            Operation::MatrixBf16 { .. } => {
                anyhow::bail!("frozen F32 matrix oracle refuses BF16 operations")
            }
        };
        let report = json!({"step":item.step,"id":item.operation.id(),"kind":kind,"role":role,"rows":rows,"cols":cols,"start_epoch":item.start,"end_epoch":item.end,
            "elements":actual.len(),"bitwise_mismatches":mismatch,"input_partial_sha256":item.source_sha,"reference_sha256":words_sha(&item.expected),"actual_sha256":words_sha(&actual),
            "reference_first_words":&item.expected[..item.expected.len().min(4)],"actual_first_words":&actual[..actual.len().min(4)],"finite_index_padding_columns":if role==Some(MatrixRole::IndexScore){3}else{0}});
        println!(
            "{}",
            json!({"kind":"dsv41_matrix_cycle_check","rank":a.rank,"check":report})
        );
        checks.push(report);
        ensure!(
            mismatch == 0,
            "matrix/cycle numerical mismatch at step{} op{}",
            item.step,
            item.operation.id()
        );
    }
    release.wait(began + Duration::from_secs(WALL_SECONDS - 10))?;
    drop(gpu);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_matrix_cycle_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":true,
        "rank":a.rank,"world":WORLD,"cycle_steps":STEPS,"operations_per_cycle":OP_COUNT,"checks":checks,"submission_events":submission,"completion":completion,
        "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL_SECONDS,"explicit_device_array_upper_bound_bytes":ARRAY_BOUND},
        "synchronization_scope":"all buffers preloaded; begin/end cycles are CPU cursor changes only; no per-cycle GPU synchronization, control exchange or epoch reset; finish is the final GPU/global check",
        "scope":"new matrix roles and repeated fixed-program component only; no full engine, graphs, dynamic shape bucket choice or model correctness/performance qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_matrix_cycle_oracle","status":"FAIL","all_passed":false,"error":"hard wall-time limit","limit_seconds":WALL_SECONDS})
        );
        std::process::exit(124);
    });
    let result = std::panic::catch_unwind(run);
    let report = match result {
        Ok(Ok(report)) => report,
        Ok(Err(error)) => {
            json!({"schema_version":1,"kind":"dsv41_matrix_cycle_oracle","status":"FAIL","all_passed":false,"error":format!("{error:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_matrix_cycle_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{report}");
    if report["all_passed"] != true {
        std::process::exit(1);
    }
}
