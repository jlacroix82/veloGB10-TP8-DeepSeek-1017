//! Model-free shortKV store oracle. Synthetic BF16 payloads are exact dyadic
//! FP4-grid values; this checks cache/ownership/index assembly, not QAT kernels.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    gpu_kv_store::{GpuKvStore, OwnedKvSnapshot},
    kv_store::{self, KvPlan},
    scheduler::LaneKey,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const WALL: u64 = 180;
const ARRAY_BOUND: usize = 64 * 1024 * 1024;
const ACTIVE: [usize; 11] = [0, 2, 3, 8, 9, 14, 15, 20, 21, 24, 39];
fn key(slot: usize, generation: u64) -> LaneKey {
    LaneKey {
        boot: Uuid::from_u128(0x410017),
        slot,
        generation,
    }
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn value(k: LaneKey, layer: usize, row: usize, col: usize, compressed: bool) -> bf16 {
    let seed = mix((k.slot as u64) << 56
        ^ (k.generation << 44)
        ^ (layer as u64) << 32
        ^ (row as u64) << 12
        ^ (col / 16) as u64
        ^ if compressed { 0xc0000000 } else { 0x30000000 });
    let h = mix(seed ^ col as u64);
    let table = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let m = if col % 16 == 0 {
        6.0
    } else {
        table[(h as usize) & 7]
    };
    let sign = if h & 8 != 0 { -1.0 } else { 1.0 };
    let scale = f32::from_bits(((124 + (seed % 4)) as u32) << 23);
    bf16::from_f32(m * sign * scale)
}
fn payload(k: LaneKey, layer: usize, first: usize, rows: usize, compressed: bool) -> Vec<bf16> {
    (0..rows * 512)
        .map(|i| value(k, layer, first + i / 512, i % 512, compressed))
        .collect()
}
fn expected(p: &KvPlan) -> Vec<bf16> {
    let mut out = Vec::new();
    for row in 0..p.window_rows {
        if p.start == 0 {
            out.extend(payload(p.key, p.layer, row, 1, false));
        } else if row > p.start {
            out.extend(vec![bf16::ZERO; 512]);
        } else {
            let absolute = p.end - 1 - ((p.end - 1 + 128 - row) % 128);
            out.extend(payload(p.key, p.layer, absolute, 1, false));
        }
    }
    if let Some(owner) = p.compressed_owner {
        out.extend(payload(p.key, owner, 0, p.compressed_rows, true));
    }
    out
}
fn sha_b(x: &[bf16]) -> String {
    let mut h = Sha256::new();
    for v in x {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn sha_i(x: &[i32]) -> String {
    let mut h = Sha256::new();
    for v in x {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn record(snapshot: &OwnedKvSnapshot, label: &str, checks: &mut Vec<Value>) -> Result<()> {
    let (actual, indices) = snapshot.download_checked()?;
    let wanted = expected(snapshot.plan());
    let wanted_i = kv_store::indices_reference(snapshot.plan())?;
    let mismatch = usize::from(actual.len() != wanted.len())
        + actual
            .iter()
            .zip(&wanted)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    let imismatch = usize::from(indices.len() != wanted_i.len())
        + indices
            .iter()
            .zip(&wanted_i)
            .filter(|(a, b)| a != b)
            .count();
    ensure!(
        mismatch == 0 && imismatch == 0,
        "KV store {label} mismatch data{mismatch} indices{imismatch}"
    );
    let v = json!({"case":label,"plan":snapshot.plan(),"serial":snapshot.serial(),"kv_elements":actual.len(),"index_elements":indices.len(),"BF16_bit_mismatches":mismatch,"index_mismatches":imismatch,"actual_kv_sha256":sha_b(&actual),"reference_kv_sha256":sha_b(&wanted),"actual_indices_sha256":sha_i(&indices),"reference_indices_sha256":sha_i(&wanted_i),"passed":true});
    println!("{}", json!({"kind":"dsv41_kv_store_check","check":v}));
    checks.push(v);
    Ok(())
}
fn prepare(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    store: &mut GpuKvStore,
    layer: usize,
    k: LaneKey,
    start: usize,
    tokens: usize,
) -> Result<OwnedKvSnapshot> {
    let window = dev.htod_sync_copy(&payload(k, layer, start, tokens, false))?;
    let r = kv_store::ratio(layer)?;
    let new_rows = if r > 0 && kv_store::owner(layer)? == Some(layer) {
        (start + tokens) / r - start / r
    } else {
        0
    };
    let compressed = if new_rows > 0 {
        Some(dev.htod_sync_copy(&payload(k, layer, start / r, new_rows, true))?)
    } else {
        None
    };
    store.prepare(s, layer, k, start, tokens, &window, compressed.as_ref())
}
fn rejected(checks: &mut Vec<Value>, name: &str, passed: bool) -> Result<()> {
    ensure!(passed, "KV negative escaped:{name}");
    checks.push(json!({"case":name,"rejected":true,"passed":true}));
    Ok(())
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "zero-argument standalone KV oracle"
    );
    let began = Instant::now();
    ensure!(
        format!("{:x}", Sha256::digest(kv_store::FIXTURE.as_bytes())) == kv_store::FIXTURE_SHA,
        "official source AST fixture changed"
    );
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let mut store = GpuKvStore::new(&dev, &stream, key(0, 1).boot, 2)?;
    let mut checks = Vec::new();
    let mut submissions = 0_usize;
    let mut immutable = None;
    for (g, tokens) in [(1, 1), (2, 2), (3, 3), (4, 127), (5, 128)] {
        let k = key(0, g);
        store.reset_lane(&stream, k)?;
        let layers: Vec<_> = if tokens == 128 {
            (0..40).collect()
        } else {
            ACTIVE.to_vec()
        };
        for layer in layers {
            let snap = prepare(&dev, &stream, &mut store, layer, k, 0, tokens)?;
            submissions += 1;
            record(&snap, &format!("cold{tokens}_L{layer}"), &mut checks)?;
            if tokens == 128 && layer == 2 {
                immutable = Some(snap);
            }
        }
        if tokens < 128 {
            for layer in ACTIVE {
                let snap = prepare(&dev, &stream, &mut store, layer, k, tokens, 1)?;
                submissions += 1;
                record(&snap, &format!("edge_append{tokens}_L{layer}"), &mut checks)?;
            }
        }
    }
    let k = key(0, 5);
    for start in 128..512 {
        for layer in ACTIVE {
            let snap = prepare(&dev, &stream, &mut store, layer, k, start, 1)?;
            submissions += 1;
            if [128, 129, 255, 256, 510, 511].contains(&start) {
                record(&snap, &format!("append{start}_L{layer}"), &mut checks)?;
            } else {
                snap.check()?;
            }
        }
    }
    record(
        immutable.as_ref().unwrap(),
        "immutable_old_snapshot_after512",
        &mut checks,
    )?;
    store.reset_lane(&stream, key(1, 1))?;
    for layer in ACTIVE {
        let snap = prepare(&dev, &stream, &mut store, layer, key(1, 1), 0, 3)?;
        submissions += 1;
        record(&snap, &format!("isolated_lane1_L{layer}"), &mut checks)?;
    }
    let mut controls = Vec::new();
    let one = dev.htod_sync_copy(&payload(k, 0, 512, 1, false))?;
    rejected(
        &mut controls,
        "context513_rejected",
        store.prepare(&stream, 0, k, 512, 1, &one, None).is_err(),
    )?;
    // Layer1 was initialized at128 but not advanced by ACTIVE. The resulting
    // end130 stays below512, isolating the positive-start multi-token rule.
    let two = dev.htod_sync_copy(&payload(k, 1, 128, 2, false))?;
    rejected(
        &mut controls,
        "positive_start_multitoken_rejected",
        store.prepare(&stream, 1, k, 128, 2, &two, None).is_err(),
    )?;
    let other = gb10_inference::gpu::fork_blocking_stream(&dev);
    rejected(
        &mut controls,
        "other_blocking_stream_rejected",
        store.reset_lane(&other, key(0, 6)).is_err(),
    )?;
    store.reset_lane(&stream, key(0, 6))?;
    rejected(
        &mut controls,
        "old_snapshot_generation_rejected",
        immutable.as_ref().unwrap().check().is_err(),
    )?;
    drop(immutable);
    rejected(
        &mut controls,
        "consumer_before_owner_rejected",
        prepare(&dev, &stream, &mut store, 3, key(0, 6), 0, 3).is_err(),
    )?;
    let window = dev.htod_sync_copy(&payload(key(0, 6), 2, 0, 3, false))?;
    let wrong = dev.htod_sync_copy(&payload(key(0, 6), 2, 0, 2, true))?;
    rejected(
        &mut controls,
        "wrong_new_compressed_rows_rejected",
        store
            .prepare(&stream, 2, key(0, 6), 0, 3, &window, Some(&wrong))
            .is_err(),
    )?;
    let snap = prepare(&dev, &stream, &mut store, 2, key(0, 6), 0, 3)?;
    snap.check()?;
    submissions += 1;
    let window3 = dev.htod_sync_copy(&payload(key(0, 6), 3, 0, 3, false))?;
    let extra = dev.htod_sync_copy(&payload(key(0, 6), 2, 0, 1, true))?;
    rejected(
        &mut controls,
        "consumer_cannot_append_source_rejected",
        store
            .prepare(&stream, 3, key(0, 6), 0, 3, &window3, Some(&extra))
            .is_err(),
    )?;
    let next = prepare(&dev, &stream, &mut store, 2, key(0, 6), 3, 1)?;
    next.check()?;
    submissions += 1;
    rejected(
        &mut controls,
        "old_source_forward_rejected",
        prepare(&dev, &stream, &mut store, 4, key(0, 6), 0, 3).is_err(),
    )?;
    let huge = dev.htod_sync_copy(&payload(key(0, 6), 0, 0, 129, false))?;
    rejected(
        &mut controls,
        "prefill129_rejected",
        store
            .prepare(&stream, 0, key(0, 6), 0, 129, &huge, None)
            .is_err(),
    )?;
    let mut badkey = key(1, 1);
    badkey.boot = Uuid::from_u128(9);
    rejected(
        &mut controls,
        "wrong_boot_rejected",
        store.prepare(&stream, 0, badkey, 0, 1, &one, None).is_err(),
    )?;
    // Error status poisons the whole lane generation; only a newer reset can recover.
    store.reset_lane(&stream, key(1, 2))?;
    let mut nan = payload(key(1, 2), 0, 0, 1, false);
    nan[17] = bf16::NAN;
    let nan = dev.htod_sync_copy(&nan)?;
    let bad = store.prepare(&stream, 0, key(1, 2), 0, 1, &nan, None)?;
    let code = bad.status_code()?;
    rejected(
        &mut controls,
        "nonfinite_window_poison",
        code == 1 && bad.check().is_err(),
    )?;
    rejected(
        &mut controls,
        "same_generation_poison_blocks",
        prepare(&dev, &stream, &mut store, 1, key(1, 2), 0, 1).is_err(),
    )?;
    store.reset_lane(&stream, key(1, 3))?;
    let snap = prepare(&dev, &stream, &mut store, 0, key(1, 3), 0, 1)?;
    submissions += 1;
    record(&snap, "new_generation_recovers", &mut checks)?;
    let mut nan = payload(key(1, 3), 2, 0, 1, true);
    nan[255] = bf16::NAN;
    let compressed = dev.htod_sync_copy(&nan)?;
    let win = dev.htod_sync_copy(&payload(key(1, 3), 2, 0, 2, false))?;
    let bad = store.prepare(&stream, 2, key(1, 3), 0, 2, &win, Some(&compressed))?;
    rejected(
        &mut controls,
        "nonfinite_compressed_poison",
        bad.status_code()? == 1 && bad.check().is_err(),
    )?;
    ensure!(
        checks.len() == 207 && controls.len() == 13,
        "KV oracle coverage changed {} {}",
        checks.len(),
        controls.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "KV oracle wall bound"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_kv_store_gpu_component_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":false,"checkpoint_payload_read":false,
        "source_id_hex":format!("{:016x}",store.source_id()),"ptx_sha256":store.ptx_sha256(),"cuda_source_sha256":store.cuda_source_sha256(),"fixture_sha256":kv_store::FIXTURE_SHA,"hard_context":512,"lane_slots":2,"submissions":submissions,"elapsed_seconds":began.elapsed().as_secs_f64(),"checks":checks,"negative_controls":controls,
        "limits":{"wall_seconds":WALL,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND},"scope":"Fixed512 shortcontext KV store/addressing/causal indices only; synthetic already-QAT-grid BF16 payload. All compressed positions fit source topk512 and all candidate blocks fit2048; independent officialCPUAST fixture. No scoring/indexnetwork, complete attention/model, prefixrestore, largercontext, serving or speed qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("KV oracle hard wall limit");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_kv_store_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_kv_store_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
