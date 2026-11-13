//! Bounded model-free1017 coefficient-RoPE/embedding/cast/ring GPU oracle.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use gb10_inference::dsv41::{
    gpu_position::{PositionKernels, PTX},
    position::{self, EmbeddingSpan, FrequencyRows, RingPlan, RingWrite, RopeProfile},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const WALL: u64 = 60;
const BOUND: usize = 16 * 1024 * 1024;
fn bvalues(v: &Value, key: &str) -> Vec<bf16> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| bf16::from_bits(v.as_u64().unwrap() as u16))
        .collect()
}
fn mismatch_bf16(a: &[bf16], b: &[bf16]) -> usize {
    usize::from(a.len() != b.len())
        + a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
}
fn mismatch_f32(a: &[f32], b: &[f32]) -> usize {
    usize::from(a.len() != b.len())
        + a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
}
fn flag(dev: &Arc<CudaDevice>, f: &CudaSlice<u32>) -> Result<u32> {
    dev.synchronize()?;
    Ok(dev.dtoh_sync_copy(f)?[0])
}
fn check(dev: &Arc<CudaDevice>, f: &CudaSlice<u32>) -> Result<()> {
    ensure!(flag(dev, f)? == 0, "position GPU sticky rejection status");
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn rotate(
    dev: &Arc<CudaDevice>,
    k: &PositionKernels,
    s: &CudaStream,
    input: &[bf16],
    freq: &FrequencyRows,
    b: usize,
    m: usize,
    h: usize,
    width: usize,
    inverse: bool,
) -> Result<Vec<bf16>> {
    let mut x = dev.htod_sync_copy(input)?;
    let f = k.upload_frequencies(s, freq)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    k.rotate(
        s,
        &mut x,
        &f,
        freq.profile(),
        freq.positions(),
        &mut status,
        b,
        m,
        h,
        width,
        inverse,
    )?;
    check(dev, &status)?;
    Ok(dev.dtoh_sync_copy(&x)?)
}
/// Retain stage counts and first native-bit disagreements even when the
/// diagnostic exits before its final PASS record. Inputs are synthetic only.
fn rope_progress(
    case: &Value,
    operation: &str,
    input: &[bf16],
    actual: &[bf16],
    expected: &[bf16],
) {
    let positions: Vec<u32> = serde_json::from_value(case["positions"].clone()).unwrap();
    let mut differences = Vec::new();
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits() != e.to_bits() && differences.len() < 32 {
            let query = index / (3 * 512);
            let head = (index / 512) % 3;
            let column = index % 512;
            let pair = column.checked_sub(448).map(|c| c / 2);
            let base = index / 2 * 2;
            let coefficients = pair.map(|p| {
                vec![
                    case["coefficients_f32_bits"][query * 64 + p * 2].clone(),
                    case["coefficients_f32_bits"][query * 64 + p * 2 + 1].clone(),
                ]
            });
            differences.push(json!({"index":index,"query":query,"position":positions[query],"head":head,"column":column,"pair":pair,
                "actual_bf16_bits":a.to_bits(),"expected_bf16_bits":e.to_bits(),"actual":a.to_f32(),"expected":e.to_f32(),
                "input_pair_bf16_bits":[input[base].to_bits(),input[base+1].to_bits()],"coefficients_f32_bits":coefficients}));
        }
    }
    fn sha(values: &[bf16]) -> String {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        format!("{:x}", Sha256::digest(bytes))
    }
    println!(
        "{}",
        json!({"event":"position_Rope_AST_diagnostic","profile":case["profile"],"operation":operation,
        "elements":actual.len(),"expected_elements":expected.len(),"bit_mismatches":mismatch_bf16(actual,expected),
        "actual_native_sha256":sha(actual),"expected_native_sha256":sha(expected),"expected_backend":"official_TorchCUDA","CPU_complex_bit_mismatches":mismatch_bf16(actual,&bvalues(case,&format!("{operation}_bf16_bits"))),"first_mismatches":differences})
    );
}
/// Deliberately keep the previous separate-products schedule as a diagnostic
/// canary. It is not an available production rotation policy.
fn legacy_separate_roundtrip(case: &Value) -> Vec<bf16> {
    let mut out = bvalues(case, "forward_bf16_bits");
    let coeff = case["coefficients_f32_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
        .collect::<Vec<_>>();
    for row in 0..15 {
        for head in 0..3 {
            for pair in 0..32 {
                let index = (row * 3 + head) * 512 + 448 + pair * 2;
                let a = out[index].to_f32();
                let b = out[index + 1].to_f32();
                let c = coeff[row * 64 + pair * 2];
                let s = -coeff[row * 64 + pair * 2 + 1];
                let ac = std::hint::black_box(a * c);
                let bs = std::hint::black_box(b * s);
                let az = std::hint::black_box(a * s);
                let bc = std::hint::black_box(b * c);
                out[index] = bf16::from_f32(ac - bs);
                out[index + 1] = bf16::from_f32(az + bc);
            }
        }
    }
    out
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "zero-argument standalone oracle"
    );
    let began = Instant::now();
    ensure!(
        format!("{:x}", Sha256::digest(position::FIXTURE.as_bytes())) == position::FIXTURE_SHA
            && format!("{:x}", Sha256::digest(position::BASIC_FIXTURE.as_bytes()))
                == position::BASIC_FIXTURE_SHA
            && format!("{:x}", Sha256::digest(position::CUDA_REFERENCE.as_bytes()))
                == position::CUDA_REFERENCE_SHA,
        "source fixtures changed"
    );
    let rope: Value = serde_json::from_str(position::FIXTURE)?;
    let cuda: Value = serde_json::from_str(position::CUDA_REFERENCE)?;
    let basic: Value = serde_json::from_str(position::BASIC_FIXTURE)?;
    let dev = CudaDevice::new(0)?;
    let s = gb10_inference::gpu::fork_blocking_stream(&dev);
    let k = PositionKernels::load(&dev, &s)?;
    let mut checks = Vec::new();
    let mut legacy_separate_canaries = Vec::new();
    for c in rope["cases"].as_array().unwrap() {
        let profile = if c["profile"] == "plain" {
            RopeProfile::Plain
        } else {
            RopeProfile::Yarn
        };
        let reference = cuda["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["profile"] == c["profile"])
            .context("missing official CUDA profile")?;
        let positions: Vec<u32> = serde_json::from_value(c["positions"].clone())?;
        let freq = FrequencyRows::from_reference_fixture(profile, &positions)?;
        let input = bvalues(c, "input_bf16_bits");
        for (inverse, label, key) in [
            (false, "forward", "forward_bf16_bits"),
            (true, "inverse", "inverse_bf16_bits"),
        ] {
            let out = rotate(
                &dev,
                &k,
                &s,
                &input,
                &freq,
                1,
                positions.len(),
                3,
                512,
                inverse,
            )?;
            let expected = bvalues(reference, key);
            rope_progress(c, label, &input, &out, &expected);
            let errors = mismatch_bf16(&out, &expected);
            ensure!(errors == 0, "official last64{label} differs");
            checks.push(json!({"case":"Rope_AST","profile":profile.name(),"operation":label,"queries":positions.len(),"heads":3,"width":512,"elements":out.len(),"bit_mismatches":errors,"CPU_complex_bit_mismatches":mismatch_bf16(&out,&bvalues(c,key)),"passed":true}));
        }
        let forward = rotate(
            &dev,
            &k,
            &s,
            &input,
            &freq,
            1,
            positions.len(),
            3,
            512,
            false,
        )?;
        let out = rotate(
            &dev,
            &k,
            &s,
            &forward,
            &freq,
            1,
            positions.len(),
            3,
            512,
            true,
        )?;
        let expected = bvalues(reference, "roundtrip_bf16_bits");
        rope_progress(c, "roundtrip", &forward, &out, &expected);
        let errors = mismatch_bf16(&out, &expected);
        ensure!(
            errors == 0,
            "official BF16forward/inverse roundtrip differs"
        );
        checks.push(json!({"case":"Rope_AST","profile":profile.name(),"operation":"roundtrip","queries":positions.len(),"heads":3,"width":512,"elements":out.len(),"bit_mismatches":errors,"CPU_complex_bit_mismatches":mismatch_bf16(&out,&bvalues(c,"roundtrip_bf16_bits")),"passed":true}));
        let legacy = legacy_separate_roundtrip(c);
        let legacy_errors = mismatch_bf16(&legacy, &expected);
        println!(
            "{}",
            json!({"event":"position_legacy_separate_complex_canary","profile":profile.name(),"bit_mismatches":legacy_errors})
        );
        ensure!(
            legacy_errors == 1,
            "legacy separate-products canary lost exact sensitivity"
        );
        legacy_separate_canaries.push(json!({"profile":profile.name(),"operation":"roundtrip","bit_mismatches":legacy_errors,"rejected":true}));
        // Shared source prefix rows differ in each lane/query, including near1M.
        let allpos = (0..16 * 3)
            .map(|i| positions[(i * 7 + 3) % positions.len()])
            .collect::<Vec<_>>();
        let freq = FrequencyRows::from_reference_fixture(profile, &allpos)?;
        let input = (0..16 * 3 * 8 * 512)
            .map(|i| bf16::from_f32(((i * 13 + 7) % 71) as f32 * 0.03125 - 1.09375))
            .collect::<Vec<_>>();
        let baseline = rotate(&dev, &k, &s, &input, &freq, 16, 3, 8, 512, false)?;
        let mut cpu = input.clone();
        position::rotate_in_place(&mut cpu, &freq, profile, &allpos, 16, 3, 8, 512, false)?;
        ensure!(
            mismatch_bf16(&baseline, &cpu) == 0,
            "C16RoPE CPU arithmetic differs"
        );
        for width in [1, 2, 4, 8, 16] {
            for start in (0..16).step_by(width) {
                let pitch = 3 * 8 * 512;
                let part = freq.slice_rows(start * 3, width * 3)?;
                let out = rotate(
                    &dev,
                    &k,
                    &s,
                    &input[start * pitch..(start + width) * pitch],
                    &part,
                    width,
                    3,
                    8,
                    512,
                    false,
                )?;
                let errors = mismatch_bf16(&out, &baseline[start * pitch..(start + width) * pitch]);
                ensure!(errors == 0, "lane-width RoPE output changed");
                checks.push(json!({"case":"Rope_width","profile":profile.name(),"width":width,"start_lane":start,"queries":3,"heads":8,"elements":out.len(),"bit_mismatches":errors,"CPU_bit_mismatches":0,"passed":true}));
            }
        }
    }
    // Actual source AST full owner range with toy hidden8; simulated rank loop,
    // CPU one-owner sum only. This is explicitly not a network collective.
    let e = &basic["embedding"];
    let ids: Vec<u32> = serde_json::from_value(e["ids"].clone())?;
    let mut sum = vec![0_f32; ids.len() * 8];
    for rank in 0..8 {
        let span = EmbeddingSpan::new(rank, rank * position::VOCAB_ROWS, position::VOCAB_ROWS, 8)?;
        let weights = (0..position::VOCAB_ROWS * 8)
            .map(|i| {
                let row = rank * position::VOCAB_ROWS + i / 8;
                let col = i % 8;
                bf16::from_f32((((row * 13 + col * 7) % 61 * 2) as f32 - 59.0) / 16.0)
            })
            .collect::<Vec<_>>();
        let dw = dev.htod_sync_copy(&weights)?;
        let di = dev.htod_sync_copy(&ids)?;
        let mut out = dev.alloc_zeros::<f32>(sum.len())?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        k.embedding(&s, &dw, &di, &mut out, &mut status, span)?;
        check(&dev, &status)?;
        let actual = dev.dtoh_sync_copy(&out)?;
        let bits: Vec<u32> = serde_json::from_value(e["partial_f32_bits"][rank].clone())?;
        let expected = bits.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let errors = mismatch_f32(&actual, &expected);
        ensure!(errors == 0, "embedding owner partial differs");
        for (a, b) in sum.iter_mut().zip(actual) {
            *a += b;
        }
        checks.push(json!({"case":"embedding_AST","rank":rank,"tokens":ids.len(),"dim":8,"elements":out.len(),"bit_mismatches":errors,"passed":true}));
    }
    let input = dev.htod_sync_copy(&sum)?;
    let mut out = dev.alloc_zeros::<bf16>(sum.len() * 4)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    k.expand(&s, &input, &mut out, &mut status, ids.len(), 8)?;
    check(&dev, &status)?;
    let errors = mismatch_bf16(
        &dev.dtoh_sync_copy(&out)?,
        &bvalues(e, "expanded_hc4_bf16_bits"),
    );
    ensure!(errors == 0, "ASTHC4 expansion differs");
    checks.push(json!({"case":"embedding_AST_expand","tokens":ids.len(),"dim":8,"elements":sum.len()*4,"bit_mismatches":errors,"passed":true}));
    drop((input, out, status));
    // Real hidden5120, bounded staged row intervals. Off-owner requests zero;
    // an owned missing row is tested as an error below.
    let ids = (0..8)
        .flat_map(|r| {
            [
                (r * position::VOCAB_ROWS) as u32,
                (r * position::VOCAB_ROWS + 1) as u32,
            ]
        })
        .collect::<Vec<_>>();
    let mut sum = vec![0_f32; ids.len() * 5120];
    for rank in 0..8 {
        let span = EmbeddingSpan::new(rank, rank * position::VOCAB_ROWS, 2, 5120)?;
        let weights = (0..2 * 5120)
            .map(|i| bf16::from_f32(((i * 7 + rank * 13) % 61) as f32 * 0.0625 - 1.875))
            .collect::<Vec<_>>();
        let wanted = position::embedding_partial(&weights, &ids, span)?;
        let w = dev.htod_sync_copy(&weights)?;
        let di = dev.htod_sync_copy(&ids)?;
        let mut o = dev.alloc_zeros::<f32>(wanted.len())?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        k.embedding(&s, &w, &di, &mut o, &mut f, span)?;
        check(&dev, &f)?;
        let actual = dev.dtoh_sync_copy(&o)?;
        let errors = mismatch_f32(&actual, &wanted);
        ensure!(errors == 0, "hidden5120 embedding differs");
        for (a, b) in sum.iter_mut().zip(actual) {
            *a += b;
        }
        checks.push(json!({"case":"embedding_hidden5120","rank":rank,"tokens":16,"dim":5120,"loaded_rows":2,"elements":wanted.len(),"bit_mismatches":errors,"passed":true}));
    }
    let input = dev.htod_sync_copy(&sum)?;
    let mut out = dev.alloc_zeros::<bf16>(sum.len() * 4)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    k.expand(&s, &input, &mut out, &mut status, 16, 5120)?;
    check(&dev, &status)?;
    let errors = mismatch_bf16(
        &dev.dtoh_sync_copy(&out)?,
        &position::expand_embedding(&sum, 16, 5120)?,
    );
    ensure!(errors == 0, "hidden5120HC4 expansion differs");
    checks.push(json!({"case":"embedding_hidden5120_expand","tokens":16,"dim":5120,"elements":sum.len()*4,"bit_mismatches":errors,"passed":true}));
    drop((input, out, status));
    // Native casting retains signed zero, BF16 ties and gradual underflow.
    let bits: Vec<u32> = serde_json::from_value(basic["cast"]["input_f32_bits"].clone())?;
    let input = bits.into_iter().map(f32::from_bits).collect::<Vec<_>>();
    let x = dev.htod_sync_copy(&input)?;
    let mut out = dev.alloc_zeros::<bf16>(input.len())?;
    let mut f = dev.alloc_zeros::<u32>(1)?;
    k.cast_f32_bf16(&s, &x, &mut out, &mut f)?;
    check(&dev, &f)?;
    let errors = mismatch_bf16(
        &dev.dtoh_sync_copy(&out)?,
        &bvalues(&basic["cast"], "expected_bf16_bits"),
    );
    ensure!(errors == 0, "native BF16RNE cast differs");
    checks.push(json!({"case":"F32_BF16_cast_AST","elements":input.len(),"bit_mismatches":errors,"passed":true}));
    let mut back = dev.alloc_zeros::<f32>(input.len())?;
    k.cast_bf16_f32(&s, &out, &mut back, &mut f)?;
    check(&dev, &f)?;
    let bits: Vec<u32> = serde_json::from_value(basic["cast"]["roundtrip_f32_bits"].clone())?;
    let errors = mismatch_f32(
        &dev.dtoh_sync_copy(&back)?,
        &bits.into_iter().map(f32::from_bits).collect::<Vec<_>>(),
    );
    ensure!(errors == 0, "native BF16F32 widening differs");
    checks.push(json!({"case":"BF16_F32_cast_AST","elements":input.len(),"bit_mismatches":errors,"passed":true}));
    drop((x, out, f, back));
    for c in basic["ring"].as_array().unwrap() {
        let count = c["count"].as_u64().unwrap() as usize;
        let start = c["start_pos"].as_u64().unwrap() as u32;
        let markers = bvalues(c, "input_row_bf16_bits");
        let input = markers
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, 512))
            .collect::<Vec<_>>();
        let cache = if c["initial_cache_row_bf16_bits"].is_array() {
            bvalues(c, "initial_cache_row_bf16_bits")
                .iter()
                .flat_map(|v| std::iter::repeat_n(*v, 512))
                .collect::<Vec<_>>()
        } else {
            vec![bf16::from_f32(-99.0); 2 * 128 * 512]
        };
        let plan = RingPlan::new(
            2,
            2 * count,
            (0..2)
                .map(|lane| RingWrite {
                    lane,
                    start_pos: start,
                    count,
                    input_row: lane * count,
                })
                .collect(),
        )?;
        let gp = k.upload_ring_plan(&s, &plan)?;
        let input = dev.htod_sync_copy(&input)?;
        let mut cache = dev.htod_sync_copy(&cache)?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        k.write_ring(&s, &input, &mut cache, &gp, &mut f)?;
        check(&dev, &f)?;
        let expected = bvalues(c, "expected_cache_row_bf16_bits")
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, 512))
            .collect::<Vec<_>>();
        let errors = mismatch_bf16(&dev.dtoh_sync_copy(&cache)?, &expected);
        ensure!(errors == 0, "official ring slot copy differs");
        checks.push(json!({"case":"ring_AST","name":c["name"],"lanes":2,"count":count,"start_pos":start,"elements":expected.len(),"bit_mismatches":errors,"passed":true}));
    }
    // Heterogeneous initial and decode plans, physically reordered lanes.
    for prefill in [true, false] {
        let counts = if prefill {
            vec![1, 127, 128, 129, 256, 257, 511, 512]
        } else {
            vec![1; 16]
        };
        let n = counts.len();
        let mut offset = 0;
        let mut writes = Vec::new();
        for (i, &count) in counts.iter().enumerate() {
            writes.push(RingWrite {
                lane: n - 1 - i,
                start_pos: if prefill {
                    0
                } else {
                    [
                        1, 127, 128, 129, 257, 65535, 65536, 99999, 100000, 1048575, 4, 16, 64,
                        512, 1024, 2048,
                    ][i]
                },
                count,
                input_row: offset,
            });
            offset += count;
        }
        let plan = RingPlan::new(n, offset, writes)?;
        let input = (0..offset * 512)
            .map(|i| bf16::from_f32(((i * 11 + 7) % 97) as f32 * 0.03125 - 1.5))
            .collect::<Vec<_>>();
        let cache = vec![bf16::from_f32(-99.0); n * 128 * 512];
        let mut expected = cache.clone();
        position::write_ring(&mut expected, &input, &plan)?;
        let gp = k.upload_ring_plan(&s, &plan)?;
        let input = dev.htod_sync_copy(&input)?;
        let mut cache = dev.htod_sync_copy(&cache)?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        k.write_ring(&s, &input, &mut cache, &gp, &mut f)?;
        check(&dev, &f)?;
        let errors = mismatch_bf16(&dev.dtoh_sync_copy(&cache)?, &expected);
        ensure!(errors == 0, "heterogeneous/reordered ring writes differ");
        checks.push(json!({"case":"ring_heterogeneous","prefill":prefill,"lanes":n,"input_rows":offset,"elements":expected.len(),"bit_mismatches":errors,"passed":true}));
    }
    let mut negative = serde_json::Map::new();
    let freq = FrequencyRows::from_reference_fixture(RopeProfile::Plain, &[1])?;
    let gf = k.upload_frequencies(&s, &freq)?;
    let mut x = dev.alloc_zeros::<bf16>(512)?;
    let mut f = dev.alloc_zeros::<u32>(1)?;
    ensure!(
        k.rotate(
            &s,
            &mut x,
            &gf,
            RopeProfile::Yarn,
            &[1],
            &mut f,
            1,
            1,
            1,
            512,
            false
        )
        .is_err(),
        "profile mismatch escaped"
    );
    negative.insert("wrong_profile_rejected".into(), json!(true));
    ensure!(
        k.rotate(
            &s,
            &mut x,
            &gf,
            RopeProfile::Plain,
            &[0],
            &mut f,
            1,
            1,
            1,
            512,
            false
        )
        .is_err(),
        "query position mismatch escaped"
    );
    negative.insert("wrong_query_positions_rejected".into(), json!(true));
    for (name, col, huge) in [
        ("nonfinite_prefix_status", 0, false),
        ("nonfinite_tail_status", 448, false),
        ("rotation_overflow_status", 448, true),
    ] {
        let mut input = vec![bf16::ZERO; 512];
        input[col] = if huge { bf16::MAX } else { bf16::NAN };
        if huge {
            input[col + 1] = bf16::MAX;
        }
        let mut x = dev.htod_sync_copy(&input)?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        k.rotate(
            &s,
            &mut x,
            &gf,
            RopeProfile::Plain,
            &[1],
            &mut f,
            1,
            1,
            1,
            512,
            false,
        )?;
        let status = flag(&dev, &f)?;
        ensure!(
            status & if huge { 2 } else { 1 } != 0,
            "nonfinite/overflow RoPE guard escaped"
        );
        negative.insert(name.into(), json!(status));
    }
    let span = EmbeddingSpan::new(0, 0, 2, 8)?;
    for (name, id, nan) in [
        ("invalid_embedding_id_status", position::VOCAB as u32, false),
        ("missing_owned_embedding_status", 3, false),
        ("nonfinite_embedding_status", 0, true),
    ] {
        let mut w = vec![bf16::ONE; 16];
        if nan {
            w[0] = bf16::NAN;
        }
        let w = dev.htod_sync_copy(&w)?;
        let ids = dev.htod_sync_copy(&[id])?;
        let mut out = dev.alloc_zeros::<f32>(8)?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        k.embedding(&s, &w, &ids, &mut out, &mut f, span)?;
        let status = flag(&dev, &f)?;
        ensure!(
            status & if id == 3 { 4 } else { 1 } != 0,
            "bad embedding guard escaped"
        );
        negative.insert(name.into(), json!(status));
    }
    let source = dev.htod_sync_copy(&[f32::MAX])?;
    let mut out = dev.alloc_zeros::<bf16>(1)?;
    let mut f = dev.alloc_zeros::<u32>(1)?;
    k.cast_f32_bf16(&s, &source, &mut out, &mut f)?;
    let status = flag(&dev, &f)?;
    ensure!(status & 2 != 0, "BF16cast overflow escaped");
    negative.insert("cast_overflow_status".into(), json!(status));
    let plan = RingPlan::new(
        1,
        1,
        vec![RingWrite {
            lane: 0,
            start_pos: 1,
            count: 1,
            input_row: 0,
        }],
    )?;
    let gp = k.upload_ring_plan(&s, &plan)?;
    let input = dev.htod_sync_copy(&vec![bf16::NAN; 512])?;
    let mut cache = dev.alloc_zeros::<bf16>(128 * 512)?;
    let mut f = dev.alloc_zeros::<u32>(1)?;
    k.write_ring(&s, &input, &mut cache, &gp, &mut f)?;
    let status = flag(&dev, &f)?;
    ensure!(status & 1 != 0, "ring nonfinite guard escaped");
    negative.insert("nonfinite_ring_status".into(), json!(status));
    ensure!(
        RingPlan::new(
            1,
            2,
            vec![
                RingWrite {
                    lane: 0,
                    start_pos: 0,
                    count: 1,
                    input_row: 0
                },
                RingWrite {
                    lane: 0,
                    start_pos: 1,
                    count: 1,
                    input_row: 1
                }
            ]
        )
        .is_err(),
        "duplicate writer escaped"
    );
    negative.insert("duplicate_ring_writer_rejected".into(), json!(true));
    ensure!(
        RingPlan::new(
            1,
            2,
            vec![RingWrite {
                lane: 0,
                start_pos: 1,
                count: 2,
                input_row: 0
            }]
        )
        .is_err(),
        "positive-position chunk escaped"
    );
    negative.insert("positive_start_multi_token_rejected".into(), json!(true));
    let nonblocking = dev.fork_default_stream()?;
    ensure!(
        k.cast_f32_bf16(&nonblocking, &source, &mut out, &mut f)
            .is_err(),
        "nonblocking stream escaped"
    );
    negative.insert("nonblocking_stream_rejected".into(), json!(true));
    ensure!(
        checks.len() == 102,
        "position check coverage changed {}",
        checks.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "wall time limit"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_position_gpu_component_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":false,"checkpoint_payload_read":false,
 "source_id_hex":env!("DSV41_POSITION_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),"rope_fixture_sha256":position::FIXTURE_SHA,"basic_fixture_sha256":position::BASIC_FIXTURE_SHA,"cuda_reference_sha256":position::CUDA_REFERENCE_SHA,"rotation_reference_backend":"official_TorchCUDA","legacy_separate_complex_canaries":legacy_separate_canaries,
 "full_table_sha256":{"plain":RopeProfile::Plain.table_sha(),"yarn":RopeProfile::Yarn.table_sha()},"full_table_files_read":false,"coefficient_source":"SHA-bound exact officialCPUtorchrepresentative rows; no pow/sincos recomputation",
 "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL,"explicit_array_allocation_upper_bound_bytes":BOUND},"checks":checks,"negative_controls":negative,
 "scope":"Isolated last64rotation/basicstate operations only. Embedding8owners are simulated on oneGPU with CPUF32referenceassembly, no network collective. Full1Mfrequency artifacts were generated separately, not fully staged/tested here. No model, longcontextquality, prefixrestore, serving or speed qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("position oracle hard60s limit");
        std::process::exit(124);
    });
    let v = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_position_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_position_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{v}");
    if v["all_passed"] != true {
        std::process::exit(1);
    }
}
