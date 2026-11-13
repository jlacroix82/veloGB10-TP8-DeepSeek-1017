//! Model-free packed projected-pool oracle. No model weights, Dense projection,
//! RoPE/cacheQAT, collectives, serving or throughput measurement.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr};
use gb10_inference::dsv41::{
    batch_compression_plan::{build, PackedCompressionPlan},
    batch_plan::{BatchCursor, BatchTicket, LaneSegment},
    compression::Ratio,
    config::Dsv41Config,
    gpu_compression::{CompressionKernels, GpuCompressor},
    gpu_module::Module,
    gpu_numerics::GpuNumerics,
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
const LANES: usize = 16;
const WALL: u64 = 120;
const BOUND: usize = 16 * 1024 * 1024;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const SIZES: [usize; 7] = [1, 2, 1, 7, 2, 1, 8];
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
const PTX: &str = include_str!("../src/ptx/dsv41_batch_compression.ptx");
struct Fence(Arc<CudaDevice>);
impl Drop for Fence {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn u16hash(x: &[bf16]) -> String {
    sha(&x
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>())
}
fn f32hash(x: &[f32]) -> String {
    sha(&x
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>())
}
fn mismatches<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.len().abs_diff(b.len()) + a.iter().zip(b).filter(|(a, b)| a != b).count()
}
fn bits(x: &[bf16]) -> Vec<u16> {
    x.iter().map(|x| x.to_bits()).collect()
}
fn fbits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|x| x.to_bits()).collect()
}
fn key(boot: Uuid, slot: usize) -> LaneKey {
    LaneKey {
        boot,
        slot,
        generation: 1,
    }
}
fn config() -> Result<Dsv41Config> {
    let f: Value = serde_json::from_str(include_str!("../tests/fixtures/dsv41_1017_config.json"))?;
    Dsv41Config::from_values(&f["hf_config"], &f["inference_config"])
}
fn layer(
    c: &mut BatchCursor,
    t: &BatchTicket,
) -> Result<gb10_inference::dsv41::batch_plan::BatchLayerPlan> {
    for l in 0..2 {
        let p = c.layer(t, l)?;
        c.commit_layer(&p)?;
    }
    c.layer(t, 2)
}
fn finish(c: &mut BatchCursor, t: &BatchTicket) -> Result<()> {
    for l in 2..40 {
        let p = c.layer(t, l)?;
        c.commit_layer(&p)?;
    }
    c.commit(t)
}
fn launch(
    m: &Module,
    s: &CudaStream,
    name: &str,
    n: usize,
    args: &mut [*mut c_void],
) -> Result<()> {
    ensure!(n > 0 && n <= 128 * D, "bounded nonempty oracle launch");
    unsafe {
        m.launch(
            name,
            s,
            (n.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            args,
        )
    }
}
fn reset(
    m: &Module,
    s: &CudaStream,
    tk: &CudaSlice<f32>,
    ts: &CudaSlice<f32>,
    status: &CudaSlice<u32>,
    lane: usize,
) -> Result<()> {
    let (mut a, mut b, mut st, mut ln, mut lanes) = (
        *tk.device_ptr(),
        *ts.device_ptr(),
        *status.device_ptr(),
        lane as u64,
        LANES as u64,
    );
    launch(
        m,
        s,
        "dsv41_batch_compression_reset",
        D,
        &mut [
            &mut a as *mut _ as _,
            &mut b as *mut _ as _,
            &mut st as *mut _ as _,
            &mut ln as *mut _ as _,
            &mut lanes as *mut _ as _,
        ],
    )
}
fn pool(
    m: &Module,
    s: &CudaStream,
    kv: &CudaSlice<f32>,
    scores: &CudaSlice<f32>,
    tk: &CudaSlice<f32>,
    ts: &CudaSlice<f32>,
    desc: &CudaSlice<u64>,
    out: &CudaSlice<bf16>,
    status: &CudaSlice<u32>,
    rows: usize,
    groups: usize,
) -> Result<()> {
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut st, mut r, mut g, mut l) = (
        *kv.device_ptr(),
        *scores.device_ptr(),
        *tk.device_ptr(),
        *ts.device_ptr(),
        *desc.device_ptr(),
        *out.device_ptr(),
        *status.device_ptr(),
        rows as u64,
        groups as u64,
        LANES as u64,
    );
    launch(
        m,
        s,
        "dsv41_batch_compression_pool2",
        groups.max(1) * D,
        &mut [
            &mut a as *mut _ as _,
            &mut b as *mut _ as _,
            &mut c as *mut _ as _,
            &mut d as *mut _ as _,
            &mut e as *mut _ as _,
            &mut f as *mut _ as _,
            &mut st as *mut _ as _,
            &mut r as *mut _ as _,
            &mut g as *mut _ as _,
            &mut l as *mut _ as _,
        ],
    )
}
fn tail(
    m: &Module,
    s: &CudaStream,
    kv: &CudaSlice<f32>,
    scores: &CudaSlice<f32>,
    tk: &CudaSlice<f32>,
    ts: &CudaSlice<f32>,
    desc: &CudaSlice<u64>,
    status: &CudaSlice<u32>,
    rows: usize,
    tails: usize,
) -> Result<()> {
    let (mut a, mut b, mut c, mut d, mut e, mut st, mut r, mut t, mut l) = (
        *kv.device_ptr(),
        *scores.device_ptr(),
        *tk.device_ptr(),
        *ts.device_ptr(),
        *desc.device_ptr(),
        *status.device_ptr(),
        rows as u64,
        tails as u64,
        LANES as u64,
    );
    launch(
        m,
        s,
        "dsv41_batch_compression_tail2",
        tails.max(1) * D,
        &mut [
            &mut a as *mut _ as _,
            &mut b as *mut _ as _,
            &mut c as *mut _ as _,
            &mut d as *mut _ as _,
            &mut e as *mut _ as _,
            &mut st as *mut _ as _,
            &mut r as *mut _ as _,
            &mut t as *mut _ as _,
            &mut l as *mut _ as _,
        ],
    )
}
fn data(slot: usize, position: usize, d: usize) -> (f32, f32) {
    let mut v = ((slot * 19 + position * 23 + d * 7) % 97) as f32 / 16. - 3.;
    if (position + d + slot) % 79 == 0 {
        v = -0.;
    }
    // Wide per-channel score differences exercise exp underflow as well as ties.
    let score = match d % 31 {
        0 => {
            if position % 2 == 0 {
                90.
            } else {
                -90.
            }
        }
        1 => 0.,
        _ => ((slot * 3 + position * 11 + d * 13) % 37) as f32 / 4. - 4.5,
    };
    (v, score)
}
fn projected(g: &PackedCompressionPlan) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for member in &g.members {
        for p in member.start..member.end {
            for d in 0..D {
                let (v, s) = data(member.key.slot, p, d);
                a.push(v);
                b.push(s);
            }
        }
    }
    (a, b)
}
fn cases(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    m: &Module,
    old: &Arc<CompressionKernels>,
    norm: &GpuNumerics,
    boot: Uuid,
    checks: &mut Vec<Value>,
) -> Result<()> {
    let nw = (0..D)
        .map(|d| bf16::from_f32(0.75 + (d % 7) as f32 / 16.).to_f32())
        .collect::<Vec<_>>();
    let dw = dev.htod_sync_copy(&nw)?;
    for width in WIDTHS {
        let mut c = BatchCursor::new(&config()?, boot, LANES)?;
        let mut reference = GpuCompressor::new(old.clone(), s, Ratio::Two, D, LANES)?;
        let tk = dev.alloc_zeros::<f32>(LANES * D)?;
        let ts = dev.alloc_zeros::<f32>(LANES * D)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        let _f = Fence(dev.clone());
        for slot in 0..LANES {
            c.reset_lane(key(boot, slot))?;
            reference.reset_lane(s, key(boot, slot))?;
            reset(m, s, &tk, &ts, &status, slot)?;
        }
        dev.synchronize()?;
        let mut start = 0;
        for (event, tokens) in SIZES.into_iter().enumerate() {
            let mut order = ORDER[..width].to_vec();
            if event % 2 == 1 {
                order.reverse();
            }
            let t = c.begin_prefill_segments(
                order
                    .iter()
                    .map(|&slot| {
                        (
                            LaneSegment {
                                key: key(boot, slot),
                                start,
                                tokens,
                            },
                            SIZES.iter().sum(),
                        )
                    })
                    .collect(),
            )?;
            let p = layer(&mut c, &t)?;
            let g = build(&p)?;
            let (kv, score) = projected(&g);
            let dk = dev.htod_sync_copy(&kv)?;
            let ds = dev.htod_sync_copy(&score)?;
            let dg = dev.htod_sync_copy(&if g.groups.is_empty() {
                vec![0; 3]
            } else {
                g.groups.iter().flatten().copied().collect()
            })?;
            let dt = dev.htod_sync_copy(&if g.tails.is_empty() {
                vec![0; 2]
            } else {
                g.tails.iter().flatten().copied().collect()
            })?;
            let out = dev.alloc_zeros::<bf16>((g.output_rows * D).max(1))?;
            let before_k = dev.dtoh_sync_copy(&tk)?;
            let before_s = dev.dtoh_sync_copy(&ts)?;
            let _f = Fence(dev.clone());
            dev.memset_zeros(&mut status)?;
            if g.output_rows > 0 {
                pool(
                    m,
                    s,
                    &dk,
                    &ds,
                    &tk,
                    &ts,
                    &dg,
                    &out,
                    &status,
                    g.input_rows,
                    g.output_rows,
                )?;
            }
            if !g.tails.is_empty() {
                tail(
                    m,
                    s,
                    &dk,
                    &ds,
                    &tk,
                    &ts,
                    &dt,
                    &status,
                    g.input_rows,
                    g.tails.len(),
                )?;
            }
            dev.synchronize()?;
            ensure!(
                dev.dtoh_sync_copy(&status)? == [0],
                "packed pool/tail status"
            );
            let got = dev.dtoh_sync_copy(&out)?[..g.output_rows * D].to_vec();
            let after_k = dev.dtoh_sync_copy(&tk)?;
            let after_s = dev.dtoh_sync_copy(&ts)?;
            // Independent old state processes every new row in native time order.
            // Pool results are collected in exactly the packed member/group order.
            let mut want = Vec::new();
            let mut want_norm = Vec::new();
            let mut reference_positions = Vec::new();
            let mut tail_bad = 0;
            let mut fixed_rows = 0;
            for member in &g.members {
                for position in member.start..member.end {
                    let (a, b): (Vec<_>, Vec<_>) =
                        (0..D).map(|d| data(member.key.slot, position, d)).unzip();
                    let da = dev.htod_sync_copy(&a)?;
                    let db = dev.htod_sync_copy(&b)?;
                    let result = reference.push_ratio2(s, member.key, position, 1, &da, &db)?;
                    let (v, _) = result.download_checked()?;
                    reference_positions.extend(result.plan.positions.iter().copied());
                    want.extend(v);
                    fixed_rows += result.plan.output_rows;
                    let normalized = result.normalize(s, norm, &dw)?;
                    normalized.check()?;
                    if let Some(latent) = normalized.latent() {
                        want_norm.extend(latent.download_checked()?);
                    }
                }
                if member.retained_new_tail.is_some() {
                    let (a, b) = reference.download_tail_checked(member.key)?;
                    let base = member.key.slot * D;
                    tail_bad += mismatches(&fbits(&after_k[base..base + D]), &fbits(&a[..D]));
                    tail_bad += mismatches(&fbits(&after_s[base..base + D]), &fbits(&b[..D]));
                }
            }
            let mut expected_tail_k = before_k.clone();
            let mut expected_tail_s = before_s.clone();
            for &[slot, row] in &g.tails {
                let base = slot as usize * D;
                let src = row as usize * D;
                expected_tail_k[base..base + D].copy_from_slice(&kv[src..src + D]);
                expected_tail_s[base..base + D].copy_from_slice(&score[src..src + D]);
            }
            let preservation = mismatches(&fbits(&after_k), &fbits(&expected_tail_k))
                + mismatches(&fbits(&after_s), &fbits(&expected_tail_s));
            let actual_norm = if g.output_rows > 0 {
                norm.rms_norm_1017(s, &out, &dw, g.output_rows, D)?
                    .download_checked()?
            } else {
                vec![]
            };
            let pool_bad = mismatches(&bits(&got), &bits(&want));
            let rms_bad = mismatches(&bits(&actual_norm), &bits(&want_norm));
            let positions_bad = mismatches(
                &g.rope_positions
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>(),
                &reference_positions,
            );
            ensure!(
                fixed_rows == g.output_rows
                    && pool_bad + rms_bad + tail_bad + preservation + positions_bad == 0,
                "packed projected pool differs from consecutive qualified source"
            );
            checks.push(json!({"case":format!("B{width}_event{event}"),"width":width,"start":start,"tokens_per_lane":tokens,"physical_slots":order,"input_rows":g.input_rows,"output_rows":g.output_rows,"old_tail_groups":g.groups.iter().filter(|g|g[1]==u64::MAX).count(),"retained_tail_lanes":g.tails.len(),"rope_positions":g.rope_positions,"pool_BF16_bit_mismatches":pool_bad,"normalized_BF16_bit_mismatches":rms_bad,"live_tail_F32_bit_mismatches":tail_bad,"complete_tail_preservation_mismatches":preservation,"position_mismatches":positions_bad,"projected_KV_sha256":f32hash(&kv),"projected_score_sha256":f32hash(&score),"packed_norm_input_sha256":u16hash(&got),"source_norm_input_sha256":u16hash(&want),"packed_normalized_sha256":u16hash(&actual_norm),"source_normalized_sha256":u16hash(&want_norm),"tail_KV_sha256":f32hash(&after_k),"tail_score_sha256":f32hash(&after_s),"passed":true}));
            finish(&mut c, &t)?;
            start += tokens;
        }
    }
    Ok(())
}
fn controls(dev: &Arc<CudaDevice>, s: &CudaStream, m: &Module) -> Result<Vec<Value>> {
    let mut checks = Vec::new();
    let old = vec![1.; LANES * D];
    let old_s = vec![0.; LANES * D];
    let kv = dev.htod_sync_copy(&[vec![2.; D], vec![9.; D]].concat())?;
    let scores = dev.htod_sync_copy(&vec![0.; 2 * D])?;
    let group = dev.htod_sync_copy(&[0u64, u64::MAX, 0])?;
    let tail_desc = dev.htod_sync_copy(&[0u64, 1])?;
    let mut tk = dev.htod_sync_copy(&old)?;
    let ts = dev.htod_sync_copy(&old_s)?;
    let out = dev.alloc_zeros::<bf16>(D)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    let _f = Fence(dev.clone());
    pool(m, s, &kv, &scores, &tk, &ts, &group, &out, &status, 2, 1)?;
    tail(m, s, &kv, &scores, &tk, &ts, &tail_desc, &status, 2, 1)?;
    dev.synchronize()?;
    ensure!(dev.dtoh_sync_copy(&status)? == [0], "ordered canary status");
    let correct = dev.dtoh_sync_copy(&out)?;
    ensure!(
        correct
            .iter()
            .all(|v| v.to_bits() == bf16::from_f32(1.5).to_bits()),
        "CPU canary exact mean"
    );
    dev.htod_sync_copy_into(&old, &mut tk)?;
    tail(m, s, &kv, &scores, &tk, &ts, &tail_desc, &status, 2, 1)?;
    pool(m, s, &kv, &scores, &tk, &ts, &group, &out, &status, 2, 1)?;
    dev.synchronize()?;
    let wrong = dev.dtoh_sync_copy(&out)?;
    let n = mismatches(&bits(&correct), &bits(&wrong));
    ensure!(
        n == D
            && wrong
                .iter()
                .all(|v| v.to_bits() == bf16::from_f32(5.5).to_bits()),
        "actual wrong-order sensitivity required"
    );
    checks.push(json!({"case":"actual_tail_write_before_old_tail_read_negative","mismatches":n,"expected_mismatches":D,"correct_BF16_bits":bf16::from_f32(1.5).to_bits(),"wrong_BF16_bits":bf16::from_f32(5.5).to_bits(),"passed":true}));
    for (name, desc, input_rows, groups, kv_values, score_values, wanted) in [
        (
            "invalid_lane",
            vec![16, u64::MAX, 0],
            2,
            1,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "invalid_first_row",
            vec![0, 2, 0],
            2,
            1,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "invalid_second_row",
            vec![0, 0, 2],
            2,
            1,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "zero_input_rows",
            vec![0, 0, 0],
            0,
            1,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "too_many_input_rows",
            vec![0, 0, 0],
            129,
            1,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "zero_groups",
            vec![0, 0, 0],
            2,
            0,
            vec![2.; 2 * D],
            vec![0.; 2 * D],
            4,
        ),
        (
            "nonfinite_KV",
            vec![0, 0, 1],
            2,
            1,
            vec![f32::NAN; 2 * D],
            vec![0.; 2 * D],
            1,
        ),
        (
            "nonfinite_score",
            vec![0, 0, 1],
            2,
            1,
            vec![2.; 2 * D],
            vec![f32::NAN; 2 * D],
            1,
        ),
        (
            "BF16_overflow",
            vec![0, 0, 1],
            2,
            1,
            vec![f32::MAX; 2 * D],
            vec![0.; 2 * D],
            2,
        ),
    ] {
        let dg = dev.htod_sync_copy(&desc)?;
        let a = dev.htod_sync_copy(&kv_values)?;
        let b = dev.htod_sync_copy(&score_values)?;
        let _f = Fence(dev.clone());
        dev.memset_zeros(&mut status)?;
        pool(
            m, s, &a, &b, &tk, &ts, &dg, &out, &status, input_rows, groups,
        )?;
        dev.synchronize()?;
        let observed = dev.dtoh_sync_copy(&status)?[0];
        ensure!(
            observed == wanted,
            "negative {name} status differs: {observed}"
        );
        checks.push(
            json!({"case":name,"expected_status":wanted,"observed_status":observed,"passed":true}),
        );
    }
    Ok(checks)
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "zero argument bounded oracle only"
    );
    let began = Instant::now();
    let boot = Uuid::from_u128(0x1641_00ba_7c02);
    let dev = CudaDevice::new(0)?;
    let s = dev.fork_default_stream()?;
    let m = Module::load(
        &dev,
        &s,
        PTX,
        &[
            "dsv41_batch_compression_reset",
            "dsv41_batch_compression_pool2",
            "dsv41_batch_compression_tail2",
        ],
        "dsv41_batch_compression_identity",
        u64::from_str_radix(
            env!("DSV41_BATCH_COMPRESSION_SOURCE_ID").trim_start_matches("0x"),
            16,
        )?,
        1,
    )?;
    let old = CompressionKernels::load(&dev, &s, boot)?;
    let norm = GpuNumerics::load(&dev, &s)?;
    let mut checks = Vec::new();
    cases(&dev, &s, &m, &old, &norm, boot, &mut checks)?;
    let negative = controls(&dev, &s, &m)?;
    ensure!(
        checks.len() == 35 && negative.len() == 10,
        "complete bounded oracle coverage"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_packed_compression_oracle","status":"PASS","all_passed":true,"model_free":true,"actual_weights":false,"Dense_projection_exercised":false,"typed_compressor_exercised":false,"network":false,"benchmark":false,"checks":checks,"negative_controls":negative,"maximum_GPU_array_bytes":BOUND,"wall_seconds":WALL,"elapsed_seconds":began.elapsed().as_secs_f64(),"identities":{"batch_compression_source_id":env!("DSV41_BATCH_COMPRESSION_SOURCE_ID"),"batch_compression_ptx_sha256":sha(PTX.as_bytes()),"qualified_compression_source_id":old.source_id(),"qualified_compression_ptx_sha256":old.ptx_sha256(),"qualified_numerics_ptx_sha256":sha(gb10_inference::dsv41::gpu_numerics::PTX.as_bytes())},"scope":"One packed pool launch per event with ordered tail commit, versus qualified per-lane consecutive one-token source controls. Exact BF16 pool/RMS and F32 live tails; no checkpoint projection, typed loader/compressor, full model, collective or performance qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("packed compression hard deadline");
        std::process::exit(124);
    });
    let result = match run() {
        Ok(v) => v,
        Err(e) => {
            json!({"schema_version":1,"kind":"dsv41_packed_compression_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_pack_has_actual_B16_128_rows_and_read_write_tail_boundary() {
        assert_eq!(
            WIDTHS.iter().sum::<usize>() * SIZES.iter().sum::<usize>(),
            682
        );
        assert_eq!(WIDTHS.len() * SIZES.len(), 35);
        assert_eq!(SIZES.iter().sum::<usize>(), 22);
        assert_eq!(16 * SIZES.iter().max().unwrap(), 128);
        let mut p = 0;
        let mut old_and_new = false;
        let mut no_output = false;
        for t in SIZES {
            old_and_new |= p % 2 == 1 && (p + t) % 2 == 1;
            no_output |= p / 2 == (p + t) / 2;
            p += t;
        }
        assert!(old_and_new && no_output);
    }
}
