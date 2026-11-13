//! Bounded model-free batch-head component oracle. No model payload/RDMA.
//! All8 logical vocabulary offsets are tested; transport itself is separately
//! qualified. Synthetic actual-shape BF16 head data have exact dyadic F32 sums.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    gpu_dense::{DenseShape, GpuDense},
    gpu_numerics::GpuNumerics,
    gpu_position::PositionKernels,
    gpu_text::{local_max_reference, TextKernels},
    gpu_text_batch::{TextBatchKernels, VOCAB_SHARD},
};
use half::bf16;
use serde_json::{json, Value};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const WALL: u64 = 120;
const ARRAY_BOUND: usize = 512 * 1024 * 1024;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
fn record(checks: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    checks.push(json!({"name":name,"passed":passed,"details":details}));
}
fn bad<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    usize::from(a.len() != b.len()) + a.iter().zip(b).filter(|(a, b)| a != b).count()
}
fn logical_logits(rank: usize, lane: usize) -> Vec<f32> {
    let mut v = vec![-100f32; VOCAB_SHARD];
    let first = if lane == 0 {
        0
    } else {
        (lane * 37 + rank * 11) % 8192
    };
    let score = match lane % 4 {
        0 => -3.,
        1 => {
            if rank % 2 == 0 {
                -0.0
            } else {
                0.0
            }
        }
        2 => 7. + lane as f32 / 8.,
        _ => -1. + rank as f32 / 16.,
    };
    v[first] = score;
    v[16159] = score;
    if lane == 15 {
        v[16159] = score + 0.125;
    } // Last native ID129279 on rank7.
    v
}
fn pairs(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<TextBatchKernels>,
    checks: &mut Vec<Value>,
) -> Result<()> {
    let old = TextKernels::load(dev, s)?;
    for rank in 0..8 {
        let corpus = (0..16)
            .map(|lane| logical_logits(rank, lane))
            .collect::<Vec<_>>();
        for width in WIDTHS {
            let input = ORDER[..width]
                .iter()
                .flat_map(|&lane| corpus[lane].iter().copied())
                .collect::<Vec<_>>();
            let data = dev.htod_sync_copy(&input)?;
            let output = g.local_pairs(s, &data, width, rank)?.download_checked()?;
            let mut cpu_bad = 0;
            let mut old_bad = 0;
            let mut native_pairs = Vec::new();
            for (row, &lane) in ORDER[..width].iter().enumerate() {
                let expected = local_max_reference(&corpus[lane], rank)?;
                let actual = output[row];
                cpu_bad += usize::from(actual.encode_pair() != expected.encode_pair());
                let source = dev.htod_sync_copy(&corpus[lane])?;
                let mut pair = dev.alloc_zeros::<u32>(2)?;
                let mut status = dev.alloc_zeros::<u32>(1)?;
                old.maxloc(s, &source, &mut pair, &mut status, rank)?;
                dev.synchronize()?;
                ensure!(
                    dev.dtoh_sync_copy(&status)? == [0],
                    "qualified C1 helper unexpectedly failed"
                );
                let c1 = dev.dtoh_sync_copy(&pair)?;
                old_bad += bad(&[actual.value().to_bits(), actual.token_id()], &c1);
                native_pairs.push([actual.value().to_bits(), actual.token_id()]);
            }
            record(
                checks,
                format!("native_rank{rank}_width{width}"),
                cpu_bad + old_bad == 0,
                json!({"rank":rank,"width":width,"logical_lanes":&ORDER[..width],"native_pairs":native_pairs,"CPU_pair_mismatches":cpu_bad,"qualified_C1_pair_mismatches":old_bad,"native_vocab_per_rank":VOCAB_SHARD,"tolerance":"bitwise including signed-zero winning-score bits"}),
            );
        }
    }
    Ok(())
}
fn parity(v: usize) -> bool {
    v.count_ones() % 2 == 0
}
fn hidden(lane: usize) -> Vec<bf16> {
    let amplitude = 2f32.powi((lane % 7) as i32 - 3);
    (0..5120)
        .map(|k| {
            bf16::from_f32(if parity(lane ^ (k & 15)) {
                amplitude
            } else {
                -amplitude
            })
        })
        .collect()
}
fn weight(out: usize, k: usize) -> bf16 {
    bf16::from_f32((((out * 7 + k * 11 + out / 31) % 3) as i32 - 1) as f32 / 32.)
}
fn float_stages(dev: &Arc<CudaDevice>, s: &CudaStream, checks: &mut Vec<Value>) -> Result<()> {
    let norm = GpuNumerics::load(dev, s)?;
    let dense = GpuDense::load(dev, s)?;
    let pos = PositionKernels::load(dev, s)?;
    let norm_weights = (0..5120)
        .map(|k| [0.5f32, 0.75, 1., 1.25][k % 4])
        .collect::<Vec<_>>();
    let dnorm = dev.htod_sync_copy(&norm_weights)?;
    let weights = (0..VOCAB_SHARD)
        .flat_map(|o| (0..5120).map(move |k| weight(o, k)))
        .collect::<Vec<_>>();
    ensure!(
        // Both host+device native BF16 weights, reference/current float arrays,
        // packed inputs/norm outputs and conservative intermediate slack.
        weights.len() * 4 + 8 * 16 * VOCAB_SHARD * 4 + 16 * 5120 * 32 < ARRAY_BOUND,
        "declared explicit array guard differs"
    );
    let dw = dev.htod_sync_copy(&weights)?;
    let corpus = (0..16).map(hidden).collect::<Vec<_>>();
    let mut c1_norm = Vec::new();
    let mut c1_logits = Vec::new();
    for lane in 0..16 {
        let input = dev.htod_sync_copy(&corpus[lane])?;
        let n = norm.rms_norm_1017(s, &input, &dnorm, 1, 5120)?;
        let normed = n.download_checked()?;
        let expected = (0..5120)
            .map(|k| {
                bf16::from_f32(if parity(lane ^ (k & 15)) {
                    norm_weights[k]
                } else {
                    -norm_weights[k]
                })
            })
            .collect::<Vec<_>>();
        ensure!(normed == expected, "C1 normalized dyadic vector differs");
        let mut fp = dev.alloc_zeros::<f32>(5120)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        unsafe {
            pos.cast_bf16_f32(s, n.device_values_unchecked(), &mut fp, &mut status)?;
        }
        let output = dense.f32_bf16_linear_f32(
            s,
            &fp,
            &dw,
            DenseShape {
                rows: 1,
                outputs: VOCAB_SHARD,
                inner: 5120,
            },
        )?;
        let logits = output.download_checked()?;
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&status)? == [0],
            "C1 float-stage status failed"
        );
        let mut errors = 0;
        for out in [0, 1, 31, 32, 511, 1023, 4095, 8191, 16159] {
            let exact = (0..5120)
                .map(|k| normed[k].to_f32() as f64 * weights[out * 5120 + k].to_f32() as f64)
                .sum::<f64>() as f32;
            errors += usize::from(logits[out].to_bits() != exact.to_bits());
        }
        record(
            checks,
            format!("source_F32_head_C1_lane{lane}"),
            errors == 0,
            json!({"logical_lane":lane,"sampled_outputs":9,"FP64_diagnostic_mismatches":errors,"tolerance":"bitwise exact dyadic products/sums","head_shape":[16160,5120],"TF32_or_FP16":false}),
        );
        c1_norm.push(normed);
        c1_logits.push(logits);
    }
    for width in WIDTHS {
        let host = ORDER[..width]
            .iter()
            .flat_map(|&l| corpus[l].iter().copied())
            .collect::<Vec<_>>();
        let input = dev.htod_sync_copy(&host)?;
        let n = norm.rms_norm_1017(s, &input, &dnorm, width, 5120)?;
        let normed = n.download_checked()?;
        let mut fp = dev.alloc_zeros::<f32>(width * 5120)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        unsafe {
            pos.cast_bf16_f32(s, n.device_values_unchecked(), &mut fp, &mut status)?;
        }
        let projection = dense.f32_bf16_linear_f32(
            s,
            &fp,
            &dw,
            DenseShape {
                rows: width,
                outputs: VOCAB_SHARD,
                inner: 5120,
            },
        )?;
        let logits = projection.download_checked()?;
        let promoted = dev.dtoh_sync_copy(&fp)?;
        ensure!(
            dev.dtoh_sync_copy(&status)? == [0],
            "batched float-stage status failed"
        );
        let mut norm_bad = 0;
        let mut fp_bad = 0;
        let mut logits_bad = 0;
        for (row, &lane) in ORDER[..width].iter().enumerate() {
            norm_bad += bad(&normed[row * 5120..(row + 1) * 5120], &c1_norm[lane]);
            let expected = c1_norm[lane]
                .iter()
                .map(|v| v.to_f32().to_bits())
                .collect::<Vec<_>>();
            let actual = promoted[row * 5120..(row + 1) * 5120]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            fp_bad += bad(&actual, &expected);
            let a = logits[row * VOCAB_SHARD..(row + 1) * VOCAB_SHARD]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            let b = c1_logits[lane]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            logits_bad += bad(&a, &b);
        }
        record(
            checks,
            format!("source_float_stages_permuted_width{width}"),
            norm_bad + fp_bad + logits_bad == 0,
            json!({"width":width,"logical_lanes":&ORDER[..width],"normalized_BF16_mismatches":norm_bad,"promoted_F32_mismatches":fp_bad,"complete_logits_F32_mismatches":logits_bad,"logits_compared":width*VOCAB_SHARD,"tolerance":"bitwise all columns against independent C1 executions"}),
        );
    }
    Ok(())
}
fn controls(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<TextBatchKernels>,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for (name, rows, index, value, expected) in [
        ("NaN_first_row", 1, 0, f32::NAN, 1u32),
        ("positive_infinity", 1, 17, f32::INFINITY, 1),
        ("negative_infinity", 1, 16159, f32::NEG_INFINITY, 1),
        ("late_NaN_row15", 16, 15 * VOCAB_SHARD + 16159, f32::NAN, 1),
    ] {
        let mut host = vec![0f32; rows * VOCAB_SHARD];
        host[index] = value;
        let source = dev.htod_sync_copy(&host)?;
        let pending = g.local_pairs(s, &source, rows, 7)?;
        let flag = pending.status_code()?;
        let rejected = pending.download_checked().is_err();
        record(
            checks,
            format!("negative_{name}"),
            flag == expected && rejected,
            json!({"width":rows,"status":flag,"expected_status":expected,"all_lane_publication_rejected":rejected}),
        );
    }
    let source = dev.htod_sync_copy(&vec![f32::NEG_INFINITY; VOCAB_SHARD])?;
    let p = g.local_pairs(s, &source, 1, 0)?;
    let flag = p.status_code()?;
    let rejected = p.download_checked().is_err();
    record(
        checks,
        "negative_no_finite_candidate".into(),
        flag == 3 && rejected,
        json!({"status":flag,"expected_status":3,"all_lane_publication_rejected":rejected}),
    );
    let good = dev.htod_sync_copy(&vec![0f32; VOCAB_SHARD])?;
    for (name, width, rank) in [("width17", 17, 0), ("rank8", 1, 8), ("extent", 2, 0)] {
        let rejected = g.local_pairs(s, &good, width, rank).is_err();
        record(
            checks,
            format!("negative_{name}"),
            rejected,
            json!({"rejected_before_allocation":rejected}),
        );
    }
    let mut pair = dev.alloc_zeros::<u32>(2)?;
    let mut status = dev.htod_sync_copy(&[1u32])?;
    unsafe {
        g.maxloc_into(s, &good, &mut pair, &mut status, 1, 0)?;
    }
    dev.synchronize()?;
    let sticky = dev.dtoh_sync_copy(&status)? == [1];
    record(
        checks,
        "negative_sticky_failure_not_silently_cleared".into(),
        sticky,
        json!({"previous_error_retained":sticky}),
    );
    Ok(())
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "component oracle accepts zero arguments"
    );
    let began = Instant::now();
    let dev = CudaDevice::new(0)?;
    let s = gb10_inference::gpu::fork_blocking_stream(&dev);
    let g = TextBatchKernels::load(&dev, &s)?;
    let mut checks = Vec::new();
    pairs(&dev, &s, &g, &mut checks)?;
    float_stages(&dev, &s, &mut checks)?;
    controls(&dev, &s, &g, &mut checks)?;
    ensure!(
        checks.len() == 70,
        "batch-head oracle coverage changed:{}",
        checks.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "batch-head component wall deadline"
    );
    let passed = checks.iter().all(|c| c["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_text_batch_component_oracle","status":if passed{"PASS"}else{"FAIL"},"all_passed":passed,
 "model_free":true,"model_execution":false,"checkpoint_payload_read":false,"RDMA":false,"physical_cluster_execution":false,"logical_vocab_ranks_tested":8,"production_actual_global_weight_composition_exercised":false,
 "source_id_hex":format!("{:016x}",g.source_id()),"ptx_sha256":g.ptx_sha256(),"cuda_source_sha256":g.cuda_source_sha256(),
 "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL,"explicit_array_bound_bytes":ARRAY_BOUND},"checks":checks,
 "scope":"Native rowwise greedy helper and source RMS/F32-head batch/packing invariance only. All8 logical rank offsets are covered on this device; existing maxloc transport is separately qualified. No full-model batched decode, actual-checkpoint head, quality or speed claim."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("text batch component hard deadline");
        std::process::exit(124);
    });
    let out = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_text_batch_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_text_batch_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{out}");
    if out["all_passed"] != true {
        std::process::exit(1);
    }
}
