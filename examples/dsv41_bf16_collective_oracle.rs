//! Actual-eight-rank source-BF16 index-score collective oracle.
//! One immutable mixed BF16/F32/full-maxloc program runs3cycles with monotonically
//! advancing shared GPU epochs and no per-cycle sync/control exchange. This is
//! model-free correctness evidence, not NCCL bit equivalence or model/serving.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use gb10_inference::dsv41::{
    bf16_collectives::{self as bf, Bf16IndexFrame, MatrixBf16Index},
    collectives::{
        self as cpu, Maxloc, MaxlocFrame, RoutedFrame, RoutedLayout, RowKind, HIDDEN, VOCAB, WORLD,
    },
    gpu_bf16_collectives::{Bf16Workspace, PendingBf16Matrix},
    gpu_collectives::{GpuCollectives, Operation, Program},
};
use gb10_inference::net::TpLink;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    sync::Arc,
    time::{Duration, Instant},
};
const WALL_SECONDS: u64 = 180;
const STEPS: [u64; 3] = [11, 19, 42];
const OP_COUNT: usize = 14;
const FINAL_EPOCH: u64 = 135;
const SHARD: u32 = VOCAB / WORLD as u32;
const ARRAY_BOUND: usize = 256 * 1024 * 1024;
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
    let mut ops = Vec::new();
    for rows in [1, 2, 4, 8, 16] {
        for cols in [31, 513] {
            ops.push(Operation::MatrixBf16 {
                id: ops.len() as u64,
                layout: MatrixBf16Index::new(rows, cols)?,
            });
        }
    }
    for (rows, cols) in [(128, 4097), (1, 131071)] {
        ops.push(Operation::MatrixBf16 {
            id: ops.len() as u64,
            layout: MatrixBf16Index::new(rows, cols)?,
        });
    }
    let layout = RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?;
    ops.push(Operation::RoutedF32 { id: 12, layout });
    ops.push(Operation::ShardedMaxloc { id: 13, layout });
    ensure!(ops.len() == OP_COUNT, "BF16 oracle program cardinality");
    Program::new_cyclic(ops, STEPS.len() as u64)
}
fn source_value(rank: usize, i: usize, cols: usize, step: u64) -> bf16 {
    let scale = 2f32.powi((step % 3) as i32);
    match i % 8 {
        0 => bf16::from_f32([256., -256., 1., 1., 0., 0., 0., 0.][rank] * scale),
        1 => bf16::from_f32(bf16::from_bits((rank + 1) as u16).to_f32() * scale),
        2 => bf16::from_bits(0x8000),
        3 => bf16::from_f32([65536., -65536., 256., 128., -64., 32., -16., 8.][rank] * scale),
        4 => bf16::from_f32(
            rank as f32 + (i / cols) as f32 / 16. + (i % 7) as f32 / 32. + (step % 16) as f32 / 32.,
        ),
        5 => {
            if rank == (i / cols + i % cols + step as usize) % 8 {
                bf16::from_f32(((i % 63) as f32 - 31.) / 16.)
            } else {
                bf16::ZERO
            }
        }
        6 => bf16::from_f32(if rank % 2 == 0 { 1. } else { -1. }),
        _ => bf16::from_f32(
            [
                1.,
                1. / 256.,
                1. / 256.,
                1. / 256.,
                1. / 256.,
                1. / 256.,
                1. / 256.,
                1. / 256.,
            ][rank]
                * scale,
        ),
    }
}
fn words_sha(words: &[u32]) -> String {
    let mut h = Sha256::new();
    for x in words {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
struct PreparedBf16 {
    step: u64,
    operation: Operation,
    start: u64,
    end: u64,
    values: CudaSlice<bf16>,
    workspace: Bf16Workspace,
    expected: Vec<bf16>,
    source_sha: String,
    stages: Vec<bf::Bf16StageProof>,
}
enum OtherBuffer {
    F32(CudaSlice<f32>),
    Pair(CudaSlice<u32>),
}
impl OtherBuffer {
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Pair(v) => v.len(),
        }
    }
}
struct PreparedOther {
    step: u64,
    operation: Operation,
    start: u64,
    end: u64,
    values: OtherBuffer,
    expected: Vec<u32>,
    source_sha: String,
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
    let p = program()?;
    let release = Release::new(&a, p.digest())?;
    let dev = CudaDevice::new(0)?;
    let link = TpLink::connect_nway(
        a.rank as i32,
        8,
        &a.ips,
        a.port,
        &a.dev,
        a.gid,
        gb10_inference::tp::TP_SLOT_BYTES,
    )?;
    let mut gpu = GpuCollectives::new(dev.clone(), link, a.rank, p.clone(), a.proxy_core)?;
    ensure!(
        gpu.expected_epoch() == 0,
        "oracle requires fresh epoch0 context"
    );
    let stream = gpu.compute_stream_owner();
    let mut matrices = Vec::new();
    let mut other = Vec::new();
    let mut epoch = 0;
    for step in STEPS {
        for operation in p.operations().iter().copied() {
            match operation {
                Operation::MatrixBf16 { id, layout } => {
                    let frames = (0..8)
                        .map(|rank| {
                            Bf16IndexFrame::new(
                                layout,
                                rank,
                                id,
                                step,
                                (0..layout.elements())
                                    .map(|i| source_value(rank, i, layout.cols(), step))
                                    .collect(),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let r = bf::serial_bf16_reference(&frames, epoch)?;
                    let values = dev.htod_sync_copy(&frames[a.rank].values)?;
                    let workspace = Bf16Workspace::new(&dev, &stream, layout.elements())?;
                    matrices.push(PreparedBf16 {
                        step,
                        operation,
                        start: epoch,
                        end: r.end_epoch,
                        values,
                        workspace,
                        expected: r.outputs[a.rank].clone(),
                        source_sha: bf::values_sha256(&frames[a.rank].values),
                        stages: r.stages,
                    });
                    epoch = r.end_epoch;
                }
                Operation::RoutedF32 { id, layout } => {
                    let frames = (0..8)
                        .map(|rank| {
                            RoutedFrame::new(
                                layout,
                                rank,
                                id,
                                (0..HIDDEN)
                                    .map(|i| {
                                        rank as f32 + step as f32 / 32. + (i % 31) as f32 / 64.
                                    })
                                    .collect(),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let r = cpu::reduce_reference(&frames, epoch)?;
                    let source = frames[a.rank]
                        .values()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    other.push(PreparedOther {
                        step,
                        operation,
                        start: epoch,
                        end: r.end_epoch,
                        values: OtherBuffer::F32(dev.htod_sync_copy(frames[a.rank].values())?),
                        expected: r.ranks[a.rank].iter().map(|x| x.to_bits()).collect(),
                        source_sha: words_sha(&source),
                    });
                    epoch = r.end_epoch;
                }
                Operation::ShardedMaxloc { id, .. } => {
                    let frames = (0..8)
                        .map(|rank| {
                            Ok(MaxlocFrame {
                                rank,
                                operation: id,
                                pair: Maxloc::new(
                                    ((rank + step as usize) % 8) as f32 + 0.25,
                                    rank as u32 * SHARD
                                        + ((step + rank as u64) % SHARD as u64) as u32,
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let r = cpu::maxloc_reference(&frames, epoch)?;
                    let local = frames[a.rank].pair;
                    let winner = r.ranks[a.rank];
                    let source = vec![local.value().to_bits(), local.token_id()];
                    other.push(PreparedOther {
                        step,
                        operation,
                        start: epoch,
                        end: r.end_epoch,
                        values: OtherBuffer::Pair(dev.htod_sync_copy(&source)?),
                        expected: vec![winner.value().to_bits(), winner.token_id()],
                        source_sha: words_sha(&source),
                    });
                    epoch = r.end_epoch;
                }
                _ => anyhow::bail!("unqualified oracle operation"),
            }
        }
    }
    ensure!(
        epoch == FINAL_EPOCH && matrices.len() == 36 && other.len() == 6,
        "preparedBF16 program/epoch count"
    );
    let native_bytes = matrices
        .iter()
        .map(|x| x.values.len() * 2 + x.workspace.capacity_elements() * 2 + x.expected.len() * 2)
        .sum::<usize>()
        + other
            .iter()
            .map(|x| (x.values.len() + x.expected.len()) * 4)
            .sum::<usize>();
    ensure!(
        native_bytes + 8 * 2 * 128 * 4097 * 4 < ARRAY_BOUND,
        "explicit host/device oracle array bound"
    );
    let mut pending: Vec<PendingBf16Matrix<'_>> = Vec::with_capacity(matrices.len());
    let mut matrix_iter = matrices.iter_mut();
    let mut other_iter = other.iter_mut();
    let mut submission = Vec::new();
    for step in STEPS {
        gpu.begin_cycle(step)?;
        submission.push(json!({"event":"begin_cycle","step":step,"epoch":gpu.expected_epoch()}));
        for operation in p.operations().iter().copied() {
            match operation {
                Operation::MatrixBf16 { id, layout } => {
                    let item = matrix_iter.next().context("missing BF16 source")?;
                    ensure!(
                        item.step == step
                            && item.operation == operation
                            && item.start == gpu.expected_epoch(),
                        "BF16 input/source epoch association"
                    );
                    pending.push(gpu.enqueue_bf16_matrix_for_step(
                        step,
                        id,
                        layout,
                        &mut item.values,
                        &mut item.workspace,
                    )?);
                    ensure!(gpu.expected_epoch() == item.end, "BF16 queuedepoch");
                }
                Operation::RoutedF32 { id, layout } => {
                    let item = other_iter.next().context("missing routed source")?;
                    ensure!(
                        item.step == step
                            && item.operation == operation
                            && item.start == gpu.expected_epoch(),
                        "routed sourceassociation"
                    );
                    match &mut item.values {
                        OtherBuffer::F32(values) => gpu.enqueue_reduce(id, layout, values)?,
                        _ => anyhow::bail!("wrongtypedroutedsource"),
                    };
                }
                Operation::ShardedMaxloc { id, layout } => {
                    let item = other_iter.next().context("missing maxloc source")?;
                    ensure!(
                        item.step == step
                            && item.operation == operation
                            && item.start == gpu.expected_epoch(),
                        "maxloc sourceassociation"
                    );
                    match &mut item.values {
                        OtherBuffer::Pair(values) => gpu.enqueue_maxloc(id, layout, values)?,
                        _ => anyhow::bail!("wrongtypedmaxlocsource"),
                    };
                }
                _ => anyhow::bail!("unsupported BF16 oracle operation"),
            }
            submission.push(json!({"event":"queued","step":step,"operation_id":operation.id(),"epoch":gpu.expected_epoch()}));
        }
        gpu.end_cycle(step)?;
        submission.push(json!({"event":"end_cycle","step":step,"epoch":gpu.expected_epoch()}));
    }
    drop(matrix_iter);
    drop(other_iter);
    let completion = gpu.finish()?;
    ensure!(
        completion.device_epoch == FINAL_EPOCH && completion.completed_operations == 42,
        "actualsharedBF16/F32/maxloccompletion"
    );
    let mut actuals = Vec::with_capacity(pending.len());
    for mut owner in pending {
        actuals.push(owner.download_checked(&mut gpu)?);
    }
    let mut checks = Vec::new();
    for (item, actual) in matrices.iter().zip(actuals) {
        let layout = item.operation.bf16_layout().unwrap();
        let mismatches = actual
            .iter()
            .zip(&item.expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
            + usize::from(actual.len() != item.expected.len());
        let check = json!({"step":item.step,"id":item.operation.id(),"kind":"matrix_bf16_index","rows":layout.rows(),"cols":layout.cols(),"logical_elements":layout.elements(),"padded_elements":layout.padded_elements(),"padding_elements":layout.padding_elements(),"chunk_count":layout.chunk_count(),"start_epoch":item.start,"end_epoch":item.end,"input_partial_sha256_u16le":item.source_sha,"reference_sha256_u16le":bf::values_sha256(&item.expected),"actual_sha256_u16le":bf::values_sha256(&actual),"bitwise_mismatches":mismatches,"CPU_stage_proof":item.stages,"includes_BF16_subnormals":true,"includes_negative_zero":true,"includes_perround_rounding_canary":true,"passed":mismatches==0});
        println!(
            "{}",
            json!({"kind":"dsv41_bf16_collective_check","rank":a.rank,"check":check})
        );
        checks.push(check);
        ensure!(mismatches == 0, "BF16 fullmatrix differs");
    }
    for item in &other {
        let actual = match &item.values {
            OtherBuffer::F32(v) => dev
                .dtoh_sync_copy(v)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            OtherBuffer::Pair(v) => dev.dtoh_sync_copy(v)?,
        };
        let mismatch = actual
            .iter()
            .zip(&item.expected)
            .filter(|(a, b)| a != b)
            .count()
            + usize::from(actual.len() != item.expected.len());
        let kind = if matches!(item.operation, Operation::RoutedF32 { .. }) {
            "routed_f32"
        } else {
            "full_maxloc"
        };
        let check = json!({"step":item.step,"id":item.operation.id(),"kind":kind,"elements":actual.len(),"start_epoch":item.start,"end_epoch":item.end,"input_partial_sha256":item.source_sha,"reference_sha256":words_sha(&item.expected),"actual_sha256":words_sha(&actual),"bitwise_mismatches":mismatch,"passed":mismatch==0});
        println!(
            "{}",
            json!({"kind":"dsv41_bf16_collective_check","rank":a.rank,"check":check})
        );
        checks.push(check);
        ensure!(mismatch == 0, "mixed F32/maxloc changed");
    }
    release.wait(began + Duration::from_secs(WALL_SECONDS - 10))?;
    let result = json!({"schema_version":1,"kind":"dsv41_source_bf16_collective_oracle","rank":a.rank,"world":8,"status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"RDMA":true,"serving_enabled":false,"NCCL_bit_equivalence_qualified":false,"steps":STEPS,"operations_per_cycle":OP_COUNT,"checks":checks,"completion":completion,"submission":submission,"helper_source_id_hex":gb10_inference::dsv41::gpu_bf16_collectives::SOURCE_ID,"helper_ptx_sha256":format!("{:x}",Sha256::digest(gb10_inference::dsv41::gpu_bf16_collectives::PTX.as_bytes())),"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL_SECONDS,"explicit_array_bound_bytes":ARRAY_BOUND},"scope":"Actual8rank eager serial sourceBF16 perroundRNE plus unchangedF32/fullpairmaxloc onone sharedmonotonicsession. Noindex/topk/fullmodel/speed qualification; CPUround-stageproof is not GPUintermediatereadback."});
    drop(gpu);
    ensure!(
        began.elapsed() < Duration::from_secs(WALL_SECONDS),
        "BF16 oraclewall"
    );
    Ok(result)
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("BF16 collectiveharddeadline");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"kind":"dsv41_source_bf16_collective_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"kind":"dsv41_source_bf16_collective_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    let passed = result["all_passed"] == true;
    println!("{result}");
    if !passed {
        std::process::exit(1);
    }
}
