//! Model-free source decode logical-order composition. Old and fixed paths
//! use identical attention PTX and identical gathered BF16 tile values. The
//! retained chronological-front predecessor is an actual GPU negative control.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use gb10_inference::dsv41::{
    chunk_plan::{ChunkCursor, LayerChunk},
    config::Dsv41Config,
    gpu_attention::AttentionKernels,
    gpu_chunk_source_indices::{self, SourceDecodeIndices},
    gpu_module::Module,
    scheduler::LaneKey,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const D: usize = 512;
const H: usize = 8;
const LIMIT: usize = 16 * 1024 * 1024;
const WALL: u64 = 60;
const OLD: &str = include_str!("../src/ptx/dsv41_kv_store.ptx");
const RAW: &str = include_str!("../src/ptx/dsv41_chunk_store.ptx");
const ATTN: &str = include_str!("../src/ptx/dsv41_attention.ptx");
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn config() -> Result<Dsv41Config> {
    let f: Value = serde_json::from_str(include_str!("../tests/fixtures/dsv41_1017_config.json"))?;
    Dsv41Config::from_values(&f["hf_config"], &f["inference_config"])
}
fn plans(start: usize) -> Result<Vec<LayerChunk>> {
    let key = LaneKey {
        boot: Uuid::from_u128(1401),
        slot: 0,
        generation: 1,
    };
    let mut c = ChunkCursor::new(&config()?, key.boot, 1)?;
    c.reset(key)?;
    let mut at = 0;
    while at < start {
        let n = (start - at).min(128);
        let t = c.begin(key, at, n)?;
        for layer in 0..40 {
            let p = c.layer(&t, layer)?;
            c.commit_layer(&p)?;
        }
        c.commit(&t)?;
        at += n;
    }
    let t = c.begin(key, start, 1)?;
    let mut result = Vec::new();
    for layer in 0..40 {
        let p = c.layer(&t, layer)?;
        if [0, 2, 20].contains(&layer) {
            result.push(p.clone());
        }
        c.commit_layer(&p)?;
    }
    Ok(result)
}
fn old_indices(
    module: &Module,
    s: &CudaStream,
    plan: &LayerChunk,
    window_rows: usize,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let (mut op, mut start, mut tokens, mut ratio, mut wr, mut wk, mut cr) = (
        *out.device_ptr(),
        plan.ticket.start as u64,
        1u64,
        plan.ratio as u64,
        window_rows as u64,
        128u64,
        plan.compressed.as_ref().map_or(0, |c| c.stored_rows) as u64,
    );
    let mut a = [
        &mut op as *mut _ as *mut c_void,
        &mut start as *mut _ as *mut c_void,
        &mut tokens as *mut _ as *mut c_void,
        &mut ratio as *mut _ as *mut c_void,
        &mut wr as *mut _ as *mut c_void,
        &mut wk as *mut _ as *mut c_void,
        &mut cr as *mut _ as *mut c_void,
    ];
    unsafe {
        module.launch(
            "dsv41_kv_store_indices",
            s,
            (out.len().div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut a,
        )
    }
}
fn raw_indices(
    module: &Module,
    s: &CudaStream,
    plan: &LayerChunk,
    out: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
) -> Result<()> {
    let (mut op, mut selection, mut ep, mut start, mut tokens, mut ratio, mut cr) = (
        *out.device_ptr(),
        0u64,
        *status.device_ptr(),
        plan.ticket.start as u64,
        1u64,
        plan.ratio as u64,
        plan.compressed.as_ref().map_or(0, |c| c.stored_rows) as u64,
    );
    let mut a = [
        &mut op as *mut _ as *mut c_void,
        &mut selection as *mut _ as *mut c_void,
        &mut ep as *mut _ as *mut c_void,
        &mut start as *mut _ as *mut c_void,
        &mut tokens as *mut _ as *mut c_void,
        &mut ratio as *mut _ as *mut c_void,
        &mut cr as *mut _ as *mut c_void,
    ];
    unsafe {
        module.launch(
            "dsv41_chunk_store_indices",
            s,
            (out.len().div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut a,
        )
    }
}
fn ids_sha(x: &[i32]) -> String {
    sha(&x.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>())
}
fn bits_sha(x: &[bf16]) -> String {
    sha(&x
        .iter()
        .flat_map(|n| n.to_bits().to_le_bytes())
        .collect::<Vec<_>>())
}
fn mismatches(a: &[bf16], b: &[bf16]) -> usize {
    usize::from(a.len() != b.len())
        + a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
}
fn global_for_slot(slot: usize, position: usize) -> usize {
    position - (position % 128 + 128 - slot) % 128
}
fn kv_value(tag: usize, column: usize, pattern: usize) -> bf16 {
    let score = if pattern == 0 {
        ((tag % 37) as i32 - 18) as f32 * 8.
    } else {
        (tag % 257) as f32 * 4.
    };
    bf16::from_f32(if column == 0 {
        score
    } else {
        (((tag * 17 + column * 7) % 61) as i32 - 30) as f32 * 0.25
    })
}
fn host_inputs(p: &LayerChunk, pattern: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let pos = p.ticket.start;
    let n = p.compressed.as_ref().map_or(0, |c| c.stored_rows);
    let w = p.window.snapshot_rows;
    let base = p.window.snapshot_global_base;
    let mut q = vec![bf16::ZERO; H * D];
    for h in 0..H {
        q[h * D] = bf16::from_f32((h + 1) as f32 * 0.25);
    }
    let mut old = vec![bf16::ZERO; (128 + n) * D];
    let mut new = vec![bf16::ZERO; (w + n) * D];
    for physical in 0..128 {
        if pos >= 127 || physical <= pos {
            let tag = if pos < 127 {
                physical
            } else {
                global_for_slot(physical, pos)
            };
            for d in 0..D {
                old[physical * D + d] = kv_value(tag, d, pattern);
            }
        }
    }
    for row in 0..w {
        for d in 0..D {
            new[row * D + d] = kv_value(base + row, d, pattern);
        }
    }
    for row in 0..n {
        for d in 0..D {
            let value = kv_value(10000 + row, d, pattern);
            old[(128 + row) * D + d] = value;
            new[(w + row) * D + d] = value;
        }
    }
    (q, old, new)
}
struct Owners {
    dev: Arc<CudaDevice>,
}
impl Drop for Owners {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let fixture_raw = include_bytes!("../tests/fixtures/dsv41_source_decode_indices.json");
    let fixture: Value = serde_json::from_slice(fixture_raw)?;
    let dev = CudaDevice::new(0)?;
    let stream = Arc::new(gb10_inference::dsv4_gpu::blocking_compute_stream(&dev));
    let old = Module::load(
        &dev,
        &stream,
        OLD,
        &["dsv41_kv_store_indices"],
        "dsv41_kv_store_identity",
        u64::from_str_radix(env!("DSV41_KV_STORE_SOURCE_ID"), 16)?,
        1,
    )?;
    let raw = Module::load(
        &dev,
        &stream,
        RAW,
        &["dsv41_chunk_store_indices"],
        "dsv41_chunk_store_identity",
        u64::from_str_radix(env!("DSV41_CHUNK_STORE_SOURCE_ID"), 16)?,
        1,
    )?;
    let fixed = SourceDecodeIndices::load(&dev, stream.clone())?;
    let attention = AttentionKernels::load(&dev, &stream)?;
    let wrong_stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    let plan = plans(1)?.remove(0);
    let mut untouched = dev.htod_sync_copy(&vec![12345i32; plan.attention_index_capacity])?;
    ensure!(
        fixed.enqueue(&wrong_stream, &plan, &mut untouched).is_err(),
        "substituted stream was accepted"
    );
    dev.synchronize()?;
    ensure!(
        dev.dtoh_sync_copy(&untouched)?.iter().all(|v| *v == 12345),
        "rejected stream mutated destination"
    );
    drop(untouched);
    drop(wrong_stream);
    let sink = dev.htod_sync_copy(&[0.0f32; H])?;
    let mut index_checks = Vec::new();
    for start in 1..127 {
        for plan in plans(start)? {
            let topk = plan.attention_index_capacity;
            let mut ids = dev.alloc_zeros::<i32>(topk)?;
            let _owners = Owners { dev: dev.clone() };
            fixed.enqueue(&stream, &plan, &mut ids)?;
            dev.synchronize()?;
            let observed = dev.dtoh_sync_copy(&ids)?;
            let reference = fixture["cases"]
                .as_array()
                .context("missing source fixture cases")?
                .iter()
                .find(|c| c["start"] == start && c["ratio"] == plan.ratio)
                .context("missing exact source fixture")?;
            let expected: Vec<i32> =
                serde_json::from_value(reference["chronological_snapshot_indices"].clone())?;
            ensure!(
                observed == expected,
                "GPU source-order indices differ from official AST"
            );
            index_checks.push(json!({"start":start,"ratio":plan.ratio,"elements":topk,"source_index_sha256_i32le":ids_sha(&observed),"mismatches":0,"passed":true}));
        }
    }
    ensure!(index_checks.len() == 378, "full unfilled index coverage");
    let mut checks = Vec::new();
    let mut negative_cases = 0usize;
    let mut peak = 0usize;
    for start in [1, 31, 63, 64, 75, 86, 95, 126, 127, 128, 255, 511] {
        for plan in plans(start)? {
            for pattern in 0..2 {
                ensure!(
                    began.elapsed() < Duration::from_secs(WALL),
                    "oracle bounded wall"
                );
                let t = plan.attention_index_capacity;
                let n = plan.compressed.as_ref().map_or(0, |c| c.stored_rows);
                let (qh, oh, nh) = host_inputs(&plan, pattern);
                let bytes =
                    (qh.len() + oh.len() + nh.len() + 3 * H * D) * 2 + 3 * t * 4 + 4 + H * 4;
                peak = peak.max(bytes);
                ensure!(bytes <= LIMIT, "explicit GPU array bound");
                let q = dev.htod_sync_copy(&qh)?;
                let old_kv = dev.htod_sync_copy(&oh)?;
                let new_kv = dev.htod_sync_copy(&nh)?;
                let (mut old_ids, mut fixed_ids, mut bad_ids) = (
                    dev.alloc_zeros::<i32>(t)?,
                    dev.alloc_zeros::<i32>(t)?,
                    dev.alloc_zeros::<i32>(t)?,
                );
                let (mut out_old, mut out_fixed, mut out_bad) = (
                    dev.alloc_zeros::<bf16>(H * D)?,
                    dev.alloc_zeros::<bf16>(H * D)?,
                    dev.alloc_zeros::<bf16>(H * D)?,
                );
                let mut status = dev.alloc_zeros::<u32>(1)?;
                dev.memset_zeros(&mut status)?;
                let _owners = Owners { dev: dev.clone() };
                old_indices(&old, &stream, &plan, 128, &mut old_ids)?;
                raw_indices(&raw, &stream, &plan, &mut bad_ids, &mut status)?;
                if gpu_chunk_source_indices::applicable(&plan) {
                    fixed.enqueue(&stream, &plan, &mut fixed_ids)?;
                } else {
                    raw_indices(&raw, &stream, &plan, &mut fixed_ids, &mut status)?;
                }
                attention.sparse(
                    &stream,
                    &q,
                    &old_kv,
                    &sink,
                    &old_ids,
                    &mut out_old,
                    &mut status,
                    1,
                    1,
                    H,
                    128 + n,
                    t,
                )?;
                attention.sparse(
                    &stream,
                    &q,
                    &new_kv,
                    &sink,
                    &fixed_ids,
                    &mut out_fixed,
                    &mut status,
                    1,
                    1,
                    H,
                    plan.window.snapshot_rows + n,
                    t,
                )?;
                attention.sparse(
                    &stream,
                    &q,
                    &new_kv,
                    &sink,
                    &bad_ids,
                    &mut out_bad,
                    &mut status,
                    1,
                    1,
                    H,
                    plan.window.snapshot_rows + n,
                    t,
                )?;
                dev.synchronize()?;
                ensure!(
                    dev.dtoh_sync_copy(&status)? == [0],
                    "GPU index/attention status failed"
                );
                let a = dev.dtoh_sync_copy(&out_old)?;
                let b = dev.dtoh_sync_copy(&out_fixed)?;
                let c = dev.dtoh_sync_copy(&out_bad)?;
                let good = mismatches(&a, &b);
                let bad = mismatches(&a, &c);
                ensure!(good == 0, "source vs fixed attention BF16 bits differ");
                if start < 127 && bad > 0 {
                    negative_cases += 1;
                }
                if start >= 127 {
                    ensure!(
                        bad == 0,
                        "full ring predecessor should already match source"
                    );
                }
                checks.push(json!({"start":start,"ratio":plan.ratio,"pattern":pattern,"elements":H*D,"source_decode_route":start<127,"old_source_sha256_bf16le":bits_sha(&a),"fixed_sha256_bf16le":bits_sha(&b),"predecessor_sha256_bf16le":bits_sha(&c),"source_fixed_bit_mismatches":good,"negative_predecessor_bit_mismatches":bad,"passed":true}));
            }
        }
    }
    ensure!(
        checks.len() == 72 && negative_cases > 0,
        "complete GPU attention coverage and real predecessor negative control required"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_source_decode_order_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"benchmark":false,
 "exact_stream_rejection_passed":true,"rejected_stream_destination_unchanged":true,"index_checks":index_checks,"attention_checks":checks,"negative_predecessor_cases":negative_cases,
 "identities":{"wrapper_sha256":sha(include_bytes!("../src/dsv41/gpu_chunk_source_indices.rs")),"reference_fixture_sha256":sha(fixture_raw),"old_KV_source_id":env!("DSV41_KV_STORE_SOURCE_ID"),"old_KV_ptx_sha256":sha(OLD.as_bytes()),"raw_chunk_source_id":env!("DSV41_CHUNK_STORE_SOURCE_ID"),"raw_chunk_ptx_sha256":sha(RAW.as_bytes()),"attention_source_id":env!("DSV41_ATTENTION_SOURCE_ID"),"attention_ptx_sha256":sha(ATTN.as_bytes())},
 "maximum_device_array_bytes":LIMIT,"peak_explicit_device_array_bytes":peak,"wall_seconds":WALL,"elapsed_seconds":began.elapsed().as_secs_f64(),
 "scope":"Actual same-PTX attention/source-order composition with zero BF16-bit tolerance and retained bad-padding negative control. Not full learned model, quality, RoCE or speed qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("source decode oracle hard deadline");
        std::process::exit(124);
    });
    let report = match run() {
        Ok(v) => v,
        Err(e) => {
            json!({"schema_version":1,"kind":"dsv41_source_decode_order_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
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
    fn every_source_slot_gathers_exact_chronological_bf16_values() {
        for start in [1, 31, 63, 64, 75, 86, 95, 126, 127, 128, 255, 511] {
            for p in plans(start).unwrap() {
                for pattern in 0..2 {
                    let (_, old, new) = host_inputs(&p, pattern);
                    let n = p.compressed.as_ref().map_or(0, |c| c.stored_rows);
                    for slot in 0..128 + n {
                        let pair = if slot < 128 {
                            let ring = (start + 1 + slot) % 128;
                            if ring > start {
                                None
                            } else {
                                let global = if start < 127 {
                                    ring
                                } else {
                                    global_for_slot(ring, start)
                                };
                                Some((ring, global - p.window.snapshot_global_base))
                            }
                        } else {
                            Some((slot, p.window.snapshot_rows + slot - 128))
                        };
                        if let Some((a, b)) = pair {
                            assert_eq!(old[a * D..(a + 1) * D], new[b * D..(b + 1) * D]);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn retained_wrong_padding_changes_key_tiles_without_changing_visible_set() {
        let source = gpu_chunk_source_indices::source_decode_window(86).unwrap();
        let mut wrong: Vec<i32> = (0..=86).collect();
        wrong.resize(128, -1);
        let mut a: Vec<_> = source.iter().copied().filter(|x| *x >= 0).collect();
        let mut b: Vec<_> = wrong.iter().copied().filter(|x| *x >= 0).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert_ne!(source[..64], wrong[..64]);
        assert_eq!(source[..64].iter().filter(|x| **x >= 0).count(), 23);
        assert_eq!(wrong[..64].iter().filter(|x| **x >= 0).count(), 64);
    }
}
