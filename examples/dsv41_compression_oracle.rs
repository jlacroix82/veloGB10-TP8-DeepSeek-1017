//! Model-free official1017 projected compression/local-index oracle.
//! No checkpoint bytes, dense projection, RoPE/QAT, allreduce or model execution.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    compression::{self, CompressionState, IndexNetworkPolicy, IndexShape, Ratio},
    gpu_compression::{CompressionKernels, GpuCompressor, IndexBatchIdentity},
    gpu_numerics::GpuNumerics,
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
const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_compression_vectors.json");
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND_BYTES: usize = 16 * 1024 * 1024;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
fn record(c: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    c.push(json!({"name":name,"passed":passed,"details":details}));
}
fn bits(v: &[bf16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn fbits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn bad<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    usize::from(a.len() != b.len()) + a.iter().zip(b).filter(|(x, y)| x != y).count()
}
fn b(v: &Value, k: &str) -> Result<Vec<bf16>> {
    Ok(serde_json::from_value::<Vec<u16>>(v[k].clone())?
        .into_iter()
        .map(bf16::from_bits)
        .collect())
}
fn f(v: &Value, k: &str) -> Result<Vec<f32>> {
    Ok(serde_json::from_value::<Vec<u32>>(v[k].clone())?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}
fn usize_at(v: &Value, k: &str) -> Result<usize> {
    Ok(v[k].as_u64().with_context(|| format!("missing {k}"))? as usize)
}
fn key(boot: Uuid, slot: usize, generation: u64) -> LaneKey {
    LaneKey {
        boot,
        slot,
        generation,
    }
}
fn norm_bad(a: &[bf16], want: &[bf16]) -> usize {
    usize::from(a.len() != want.len())
        + a.iter()
            .zip(want)
            .filter(|(a, b)| !a.is_finite() || a.to_bits().abs_diff(b.to_bits()) > 1)
            .count()
}
fn pool_bad(a: &[f32], want: &[f32]) -> usize {
    usize::from(a.len() != want.len())
        + a.iter()
            .zip(want)
            .filter(|(a, b)| !a.is_finite() || (*a - *b).abs() > 1e-5 + 1e-6 * b.abs())
            .count()
}
fn fixture_compression(
    dev: &Arc<CudaDevice>,
    stream: &CudaStream,
    gpu: &Arc<CompressionKernels>,
    norm: &GpuNumerics,
    boot: Uuid,
    fixture: &Value,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for case in fixture["compression_cases"]
        .as_array()
        .context("compression cases")?
    {
        let name = case["name"].as_str().context("case name")?;
        let ratio = if case["ratio"] == 1 {
            Ratio::One
        } else {
            Ratio::Two
        };
        let dim = usize_at(case, "dim")?;
        let weights = b(case, "norm_weight_bf16_bits")?
            .into_iter()
            .map(bf16::to_f32)
            .collect::<Vec<_>>();
        let dw = dev.htod_sync_copy(&weights)?;
        let lane = key(boot, 3, 1);
        let mut state = GpuCompressor::new(gpu.clone(), stream, ratio, dim, 16)?;
        state.reset_lane(stream, lane)?;
        for (event_no, event) in case["events"]
            .as_array()
            .context("events")?
            .iter()
            .enumerate()
        {
            let start = usize_at(event, "start")?;
            let tokens = usize_at(event, "tokens")?;
            let bb = if ratio == Ratio::One {
                b(event, "projected_kv_bits")?
            } else {
                vec![]
            };
            let ff = if ratio == Ratio::Two {
                f(event, "projected_kv_bits")?
            } else {
                vec![]
            };
            let ss = if ratio == Ratio::Two {
                f(event, "projected_score_bits")?
            } else {
                vec![]
            };
            let db = dev.htod_sync_copy(&if bb.is_empty() { vec![bf16::ZERO] } else { bb })?;
            let df = dev.htod_sync_copy(&if ff.is_empty() { vec![0.] } else { ff })?;
            let ds = dev.htod_sync_copy(&if ss.is_empty() { vec![0.] } else { ss })?;
            let out = if ratio == Ratio::One {
                state.push_ratio1(stream, lane, start, tokens, &db)?
            } else {
                state.push_ratio2(stream, lane, start, tokens, &df, &ds)?
            };
            let expected_positions: Vec<usize> =
                serde_json::from_value(event["positions"].clone())?;
            let position_bad = usize::from(out.plan.positions != expected_positions);
            let (prepared, pool) = out.download_checked()?;
            let input_bad = bad(&bits(&prepared), &bits(&b(event, "norm_input_bf16_bits")?));
            let pbad = if ratio == Ratio::Two && !prepared.is_empty() {
                pool_bad(pool.as_ref().unwrap(), &f(event, "pooled_f32_bits")?)
            } else {
                0
            };
            let tail_bad = if ratio == Ratio::Two {
                let (a, z) = state.download_tail_checked(lane)?;
                bad(&fbits(&a), &fbits(&f(event, "tail_kv_f32_bits")?))
                    + bad(&fbits(&z), &fbits(&f(event, "tail_score_f32_bits")?))
            } else {
                0
            };
            let normalized = out.normalize(stream, norm, &dw)?;
            normalized.check()?;
            let mut copy_bad = 0;
            let actual = if let Some(latent) = normalized.latent() {
                let before = latent.download_checked()?;
                let copied = unsafe { latent.enqueue_rotary_copy_unchecked(stream)? };
                dev.synchronize()?;
                copy_bad += bad(&bits(&before), &bits(&dev.dtoh_sync_copy(&copied)?));
                copy_bad += bad(&bits(&before), &bits(&latent.download_checked()?));
                before
            } else {
                vec![]
            };
            let nbad = norm_bad(&actual, &b(event, "normalized_unrotated_bf16_bits")?);
            record(
                checks,
                format!("fixture_{name}_event{event_no}"),
                position_bad + input_bad + pbad + tail_bad + copy_bad + nbad == 0,
                json!({"ratio":ratio,"dim":dim,"start":start,"tokens":tokens,"output_rows":actual.len()/dim,"positions":expected_positions,"position_mismatches":position_bad,"norm_input_mismatches":input_bad,"pool_tolerance_mismatches":pbad,"tail_bit_mismatches":tail_bad,"normalized_over_1_BF16_ULP":nbad,"immutable_unrotated_copy_mismatches":copy_bad}),
            );
        }
    }
    Ok(())
}
fn compare_index(a: &compression::IndexScores, w: &compression::IndexScores) -> usize {
    bad(&bits(&a.scaled_weights), &bits(&w.scaled_weights))
        + bad(&bits(&a.dot), &bits(&w.dot))
        + bad(&bits(&a.relu), &bits(&w.relu))
        + bad(&bits(&a.product), &bits(&w.product))
        + bad(&bits(&a.local_bf16), &bits(&w.local_bf16))
        + bad(&fbits(&a.promoted_f32), &fbits(&w.promoted_f32))
}
fn run_index(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<CompressionKernels>,
    boot: Uuid,
    shape: IndexShape,
    q: &[bf16],
    k: &[bf16],
    w: &[bf16],
) -> Result<compression::IndexScores> {
    let dq = dev.htod_sync_copy(q)?;
    let dk = dev.htod_sync_copy(k)?;
    let dw = dev.htod_sync_copy(w)?;
    let id = IndexBatchIdentity {
        shape,
        lanes: ORDER[..shape.batch]
            .iter()
            .map(|&slot| key(boot, slot, 1))
            .collect(),
        start_positions: (0..shape.batch).map(|n| n * 3).collect(),
    };
    let output = g.index_scores(
        s,
        &dq,
        &dk,
        &dw,
        id,
        IndexNetworkPolicy::SourceBf16LocalAndNetwork,
    )?;
    output.download_checked()
}
fn fixture_index(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<CompressionKernels>,
    boot: Uuid,
    fixture: &Value,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for case in fixture["index_cases"].as_array().context("index cases")? {
        let name = case["name"].as_str().context("index name")?;
        let shape = IndexShape {
            batch: usize_at(case, "batch")?,
            queries: usize_at(case, "queries")?,
            keys: usize_at(case, "keys")?,
        };
        let q = b(case, "projected_q_bf16_bits")?;
        let k = b(case, "key_bf16_bits")?;
        let w = b(case, "projected_weight_bf16_bits")?;
        let cpu = compression::index_scores(&q, &k, &w, shape)?;
        let expected = compression::IndexScores {
            scaled_weights: b(case, "scaled_weight_bf16_bits")?,
            dot: b(case, "dot_bf16_bits")?,
            relu: b(case, "relu_bf16_bits")?,
            product: b(case, "product_bf16_bits")?,
            local_bf16: b(case, "local_sum_bf16_bits")?,
            promoted_f32: f(case, "source_local_promoted_f32_bits")?,
        };
        ensure!(
            compare_index(&cpu, &expected) == 0,
            "CPU index math differs from exact AST {name}"
        );
        let actual = run_index(dev, s, g, boot, shape, &q, &k, &w)?;
        let mismatch = compare_index(&actual, &expected);
        record(
            checks,
            format!("fixture_{name}_all_BF16_stages"),
            mismatch == 0,
            json!({"shape":shape,"stage_bit_mismatches":mismatch,"elements":shape.scores(),"tolerance":"bitwise all source BF16 boundaries"}),
        );
        let alt = f(case, "negative_unrounded_product_or_sum_f32_bits")?;
        let changed = bad(&fbits(&expected.promoted_f32), &fbits(&alt));
        let wanted = usize_at(case, "negative_unrounded_local_mismatches")?;
        record(
            checks,
            format!("fixture_{name}_unrounded_local_negative"),
            changed == wanted && changed > 0,
            json!({"different_elements":changed,"expected_different_elements":wanted,"note":"Unrounded local products/sum are not the source BF16 local algorithm; F32 NETWORK alternative only widens already-rounded locals."}),
        );
    }
    Ok(())
}
fn generated_index(shape: IndexShape) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let q = (0..shape.batch * shape.queries * 4 * 128)
        .map(|i| bf16::from_f32(((i * 13 + i / 128 * 7) % 31) as f32 / 16. - 0.9375))
        .collect();
    let k = (0..shape.batch * shape.keys * 128)
        .map(|i| bf16::from_f32(((i * 11 + i / 128 * 3) % 23) as f32 / 16. - 0.6875))
        .collect();
    let w = (0..shape.batch * shape.queries * 4)
        .map(|i| bf16::from_f32(((i * 7) % 29) as f32 / 8. - 1.75))
        .collect();
    (q, k, w)
}
fn index_widths(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<CompressionKernels>,
    boot: Uuid,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for queries in [1, 128] {
        let max = IndexShape {
            batch: 16,
            queries,
            keys: 19,
        };
        let (q, k, w) = generated_index(max);
        let all = run_index(dev, s, g, boot, max, &q, &k, &w)?;
        for width in WIDTHS {
            let shape = IndexShape {
                batch: width,
                ..max
            };
            let qq = &q[..width * queries * 512];
            let kk = &k[..width * 19 * 128];
            let ww = &w[..width * queries * 4];
            let cpu = compression::index_scores(qq, kk, ww, shape)?;
            let actual = run_index(dev, s, g, boot, shape, qq, kk, ww)?;
            let mismatches = compare_index(&actual, &cpu);
            let n = shape.scores();
            let cross = bad(&bits(&actual.local_bf16), &bits(&all.local_bf16[..n]))
                + bad(&fbits(&actual.promoted_f32), &fbits(&all.promoted_f32[..n]));
            record(
                checks,
                format!("index_width{width}_queries{queries}"),
                mismatches + cross == 0,
                json!({"width":width,"queries":queries,"keys":19,"permuted_slots":&ORDER[..width],"CPU_stage_mismatches":mismatches,"cross_width_bit_mismatches":cross,"elements":n,"dot_diagnostic":"FP64 sum then FP32 and source BF16 cast; dyadic inputs exact in FP32"}),
            );
        }
    }
    Ok(())
}
fn compression_widths(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<CompressionKernels>,
    norm: &GpuNumerics,
    boot: Uuid,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for ratio in [Ratio::One, Ratio::Two] {
        for width in WIDTHS {
            let dim = 512;
            let dw = dev.htod_sync_copy(&vec![1f32; dim])?;
            let mut gpu = GpuCompressor::new(g.clone(), s, ratio, dim, 16)?;
            let mut cpu = CompressionState::new(boot, 16, ratio, dim)?;
            for &slot in &ORDER[..width] {
                let lane = key(boot, slot, 1);
                gpu.reset_lane(s, lane)?;
                cpu.reset(lane)?;
            }
            let mut input_bad = 0;
            let mut pbad = 0;
            let mut nbad = 0;
            let mut tail_bad = 0;
            let mut position_bad = 0;
            let mut empty = 0;
            let mut outputs = 0;
            for (start, tokens) in [(0, 127), (127, 1), (128, 1), (129, 1)] {
                let mut bs = Vec::new();
                let mut fs = Vec::new();
                let mut ss = Vec::new();
                let mut expected = Vec::new();
                for &slot in &ORDER[..width] {
                    let kv = (0..tokens * dim)
                        .map(|i| ((i * 13 + slot * 11 + start * 7) % 251) as f32 / 64. - 1.9375)
                        .collect::<Vec<_>>();
                    let score = (0..tokens * dim)
                        // Pair-equal gates give an exact dyadic lane/width control;
                        // unequal learned gates are independently covered by AST fixtures.
                        .map(|i| {
                            ((i % dim * 7 + slot * 3 + (start + i / dim) / 2) % 17) as f32 / 8. - 1.
                        })
                        .collect::<Vec<_>>();
                    let bf = kv.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
                    let lane = key(boot, slot, 1);
                    let ex = if ratio == Ratio::One {
                        cpu.push_ratio1(lane, start, tokens, &bf)?
                    } else {
                        cpu.push_ratio2(lane, start, tokens, &kv, &score)?
                    };
                    bs.push(dev.htod_sync_copy(&bf)?);
                    fs.push(dev.htod_sync_copy(&kv)?);
                    ss.push(dev.htod_sync_copy(&score)?);
                    expected.push(ex);
                }
                let mut queued = Vec::new();
                for (i, &slot) in ORDER[..width].iter().enumerate() {
                    let lane = key(boot, slot, 1);
                    queued.push(if ratio == Ratio::One {
                        gpu.push_ratio1(s, lane, start, tokens, &bs[i])?
                    } else {
                        gpu.push_ratio2(s, lane, start, tokens, &fs[i], &ss[i])?
                    });
                }
                for (i, out) in queued.into_iter().enumerate() {
                    let ex = &expected[i];
                    let lane = key(boot, ORDER[i], 1);
                    position_bad += usize::from(out.plan != ex.plan);
                    let (prepared, pool) = out.download_checked()?;
                    input_bad += bad(&bits(&prepared), &bits(&ex.norm_input_bf16));
                    if ratio == Ratio::Two {
                        pbad += pool_bad(pool.as_ref().unwrap(), ex.pooled_f32.as_ref().unwrap());
                        let (a, b) = gpu.download_tail_checked(lane)?;
                        let (x, y) = cpu.tail(lane)?;
                        tail_bad += bad(&fbits(&a), &fbits(x)) + bad(&fbits(&b), &fbits(y));
                    }
                    let normalized = out.normalize(s, norm, &dw)?;
                    normalized.check()?;
                    let actual = if let Some(n) = normalized.latent() {
                        outputs += 1;
                        n.download_checked()?
                    } else {
                        empty += 1;
                        vec![]
                    };
                    let wanted = ex.normalized_unrotated(&vec![1.; dim], dim)?;
                    nbad += norm_bad(&actual, &wanted);
                }
            }
            record(
                checks,
                format!(
                    "compression_ratio{}_width{width}_isolated_lanes",
                    ratio.value()
                ),
                input_bad + pbad + nbad + tail_bad + position_bad == 0,
                json!({"ratio":ratio,"width":width,"dim":dim,"permuted_slots":&ORDER[..width],"prefill_tokens":127,"appends":3,"norm_input_mismatches":input_bad,"pool_tolerance_mismatches":pbad,"normalized_over_1_BF16_ULP":nbad,"tail_bit_mismatches":tail_bad,"plan_mismatches":position_bad,"nonempty_outputs":outputs,"incomplete_tail_outputs":empty,"launch_scope":"independent lane kernels queued on one stream; not concurrent-request serving"}),
            );
        }
    }
    Ok(())
}
fn controls(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &Arc<CompressionKernels>,
    norm: &GpuNumerics,
    boot: Uuid,
    checks: &mut Vec<Value>,
) -> Result<()> {
    let lane = key(boot, 2, 1);
    let mut st = GpuCompressor::new(g.clone(), s, Ratio::Two, 8, 16)?;
    st.reset_lane(s, lane)?;
    let other_stream = gb10_inference::gpu::fork_blocking_stream(dev);
    let other_rejected = st.reset_lane(&other_stream, key(boot, 2, 2)).is_err();
    record(
        checks,
        "control_other_blocking_stream_rejected".into(),
        other_rejected,
        json!({"rejected":other_rejected}),
    );
    let v = dev.htod_sync_copy(&vec![1f32; 8])?;
    let z = dev.htod_sync_copy(&vec![0f32; 8])?;
    let out = st.push_ratio2(s, lane, 0, 1, &v, &z)?;
    out.check()?;
    let missing = st.push_ratio2(s, lane, 2, 1, &v, &z).is_err();
    record(
        checks,
        "control_noncontiguous_append".into(),
        missing,
        json!({"rejected":missing}),
    );
    let norm_weight = dev.htod_sync_copy(&vec![1f32; 8])?;
    let empty = out.normalize(s, norm, &norm_weight)?;
    let retained = empty.latent().is_none() && empty.check().is_ok();
    record(
        checks,
        "control_incomplete_tail_owner_retained".into(),
        retained,
        json!({"empty_but_status_checked":retained}),
    );
    let new = key(boot, 2, 2);
    st.reset_lane(s, new)?;
    let stale = empty.check().is_err();
    record(
        checks,
        "control_reset_invalidates_prior_output".into(),
        stale,
        json!({"rejected":stale}),
    );
    let generation = st.reset_lane(s, new).is_err();
    record(
        checks,
        "control_reset_requires_new_generation".into(),
        generation,
        json!({"rejected":generation}),
    );
    let mut nan = vec![0f32; 8];
    nan[3] = f32::NAN;
    let dn = dev.htod_sync_copy(&nan)?;
    let failed = st.push_ratio2(s, new, 0, 1, &dn, &z)?;
    let status = failed.status_code()?;
    let rejected = failed.check().is_err();
    let poison = st.push_ratio2(s, new, 1, 1, &v, &z).is_err();
    record(
        checks,
        "control_nonfinite_tail_poison".into(),
        status == 1 && rejected && poison,
        json!({"status":status,"rejected":rejected,"same_generation_blocked":poison}),
    );
    st.reset_lane(s, key(boot, 2, 3))?;
    let recovered = st
        .push_ratio2(s, key(boot, 2, 3), 0, 1, &v, &z)?
        .check()
        .is_ok();
    record(
        checks,
        "control_new_generation_recovers".into(),
        recovered,
        json!({"recovered":recovered}),
    );
    let mut r1 = GpuCompressor::new(g.clone(), s, Ratio::One, 8, 16)?;
    r1.reset_lane(s, key(boot, 0, 1))?;
    let db = dev.htod_sync_copy(&vec![bf16::ZERO; 129 * 8])?;
    let cap = r1.push_ratio1(s, key(boot, 0, 1), 0, 129, &db).is_err();
    record(
        checks,
        "control_prefill129_rejected".into(),
        cap,
        json!({"rejected":cap}),
    );
    let shape = IndexShape {
        batch: 2,
        queries: 1,
        keys: 1,
    };
    let dq = dev.htod_sync_copy(&vec![bf16::ZERO; 2 * 512])?;
    let dk = dev.htod_sync_copy(&vec![bf16::ZERO; 2 * 128])?;
    let dw = dev.htod_sync_copy(&vec![bf16::ZERO; 8])?;
    let duplicate = IndexBatchIdentity {
        shape,
        lanes: vec![key(boot, 1, 1); 2],
        start_positions: vec![0; 2],
    };
    let invalid = g
        .index_scores(
            s,
            &dq,
            &dk,
            &dw,
            duplicate,
            IndexNetworkPolicy::SourceBf16LocalAndNetwork,
        )
        .is_err();
    record(
        checks,
        "control_duplicate_index_lane_rejected".into(),
        invalid,
        json!({"rejected":invalid}),
    );
    let mut nq = vec![bf16::ZERO; 512];
    nq[1] = bf16::NAN;
    let nq = dev.htod_sync_copy(&nq)?;
    let nk = dev.htod_sync_copy(&vec![bf16::ONE; 128])?;
    let nw = dev.htod_sync_copy(&vec![bf16::ONE; 4])?;
    let ident = IndexBatchIdentity {
        shape: IndexShape {
            batch: 1,
            queries: 1,
            keys: 1,
        },
        lanes: vec![key(boot, 0, 1)],
        start_positions: vec![0],
    };
    let negative = g.index_scores(
        s,
        &nq,
        &nk,
        &nw,
        ident.clone(),
        IndexNetworkPolicy::SourceBf16LocalAndNetwork,
    )?;
    let status = negative.status_code()?;
    let rejected = negative.check().is_err();
    record(
        checks,
        "control_nonfinite_index_rejected".into(),
        status == 1 && rejected,
        json!({"status":status,"rejected":rejected}),
    );
    let cleanq = dev.htod_sync_copy(&vec![bf16::ONE; 512])?;
    let mut alt = g.index_scores(
        s,
        &cleanq,
        &nk,
        &nw,
        ident,
        IndexNetworkPolicy::SourceBf16LocalPromotedF32NetworkAlternative,
    )?;
    let source_rejected = unsafe { alt.source_bf16_unchecked() }.is_err();
    let alt_allowed = unsafe { alt.alternative_f32_unchecked() }.is_ok();
    alt.check()?;
    record(
        checks,
        "control_network_policy_explicit".into(),
        source_rejected && alt_allowed,
        json!({"source_BF16_accessor_rejected":source_rejected,"alternative_F32_accessor_allowed":alt_allowed,"F32_network_is_source_rounding_alternative":true}),
    );
    Ok(())
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    ensure!(
        fixture["model_source_sha256"] == compression::MODEL_REFERENCE_SHA256
            && fixture["compression_cases"]
                .as_array()
                .context("compression cases")?
                .len()
                == 8
            && fixture["index_cases"]
                .as_array()
                .context("index cases")?
                .len()
                == 4,
        "fixture identity/cardinality"
    );
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let boot = Uuid::new_v4();
    let gpu = CompressionKernels::load(&dev, &stream, boot)?;
    let norm = GpuNumerics::load(&dev, &stream)?;
    let mut checks = Vec::new();
    fixture_compression(&dev, &stream, &gpu, &norm, boot, &fixture, &mut checks)?;
    fixture_index(&dev, &stream, &gpu, boot, &fixture, &mut checks)?;
    compression_widths(&dev, &stream, &gpu, &norm, boot, &mut checks)?;
    index_widths(&dev, &stream, &gpu, boot, &mut checks)?;
    controls(&dev, &stream, &gpu, &norm, boot, &mut checks)?;
    ensure!(
        checks.len() == 53,
        "compression oracle coverage changed: {}",
        checks.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL_SECONDS),
        "compression oracle wall deadline"
    );
    let passed = checks.iter().all(|c| c["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_compression_index_gpu_component_oracle","status":if passed{"PASS"}else{"FAIL"},"all_passed":passed,
    "model_free":true,"model_execution":false,"checkpoint_payload_read":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":false,
 "source_id_hex":format!("{:016x}",gpu.source_id()),"cuda_source_sha256":gpu.cuda_source_sha256(),"ptx_sha256":gpu.ptx_sha256(),"fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),"model_reference_sha256":compression::MODEL_REFERENCE_SHA256,
 "numerics_source_id_hex":format!("{:016x}",norm.source_id()),"numerics_ptx_sha256":norm.ptx_sha256(),
 "elapsed_seconds":began.elapsed().as_secs_f64(),"checks":checks,
 "limits":{"wall_seconds":WALL_SECONDS,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND_BYTES},
 "dtype_contract":{"ratio1_projection":"BF16","ratio2_projection_pool_tail":"F32","RMS_input":"BF16 after pool","RMS_epsilon":1e-20,"index_dot_relu_scaled_weight_product_local_sum":"BF16 at every source boundary","source_network":"BF16, executor external","alternative_network":"source BF16 locals widened to F32, changes network rounding"},
 "scope":"Supplied projection compression, qualified RMS composition and supplied BF16 projected/QAT index scores only. Dot association and Torch CPU/CUDA backend differences on general mixed-magnitude data, network rounding and whole-model quality remain unqualified."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("compression oracle hard deadline");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_compression_index_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_compression_index_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
