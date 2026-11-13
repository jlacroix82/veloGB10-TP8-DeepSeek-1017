//! Model-free GPU oracle for the isolated1017 numerical components.
//! Zero arguments. Root must run with exclusive GPU ownership and an external60-second timeout.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{gpu_numerics::GpuNumerics, numerics};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_numerics_vectors.json");
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND_BYTES: usize = 16 * 1024 * 1024;
fn bytes(v: &Value, key: &str) -> Result<Vec<u8>> {
    Ok(serde_json::from_value(v[key].clone())?)
}
fn words(v: &Value, key: &str) -> Result<Vec<u16>> {
    Ok(serde_json::from_value(v[key].clone())?)
}
fn floats(v: &Value, key: &str) -> Result<Vec<f32>> {
    let bits: Vec<u32> = serde_json::from_value(v[key].clone())?;
    Ok(bits.into_iter().map(f32::from_bits).collect())
}
fn bfloat(v: &Value, key: &str) -> Result<Vec<bf16>> {
    Ok(words(v, key)?.into_iter().map(bf16::from_bits).collect())
}
fn n(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(v[key].as_u64().context("missing fixture dimension")?).map_err(Into::into)
}
fn record(checks: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    checks.push(json!({"name":name,"passed":passed,"details":details}));
}
fn first_byte_difference(a: &[u8], b: &[u8]) -> Value {
    json!(a
        .iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| x != y)
        .map(|(i, (x, y))| json!({"index":i,"actual":x,"expected":y})))
}
fn f32_bits_match(got: &[f32], want: &[u32]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| a.to_bits() == *b)
}
fn compare_bf16(got: &[bf16], want: &[bf16], allow_ulp: u16) -> (bool, usize, u16) {
    let mut wrong = 0;
    let mut max_ulp = 0;
    if got.len() != want.len() {
        return (false, got.len().abs_diff(want.len()), u16::MAX);
    }
    for (actual, expected) in got.iter().zip(want) {
        let a = actual.to_bits();
        let b = expected.to_bits();
        let same_sign = (a & 0x8000) == (b & 0x8000);
        let ulp = if same_sign {
            (a & 0x7fff).abs_diff(b & 0x7fff)
        } else {
            u16::MAX
        };
        max_ulp = max_ulp.max(ulp);
        if !actual.is_finite() || !expected.is_finite() || ulp > allow_ulp {
            wrong += 1;
        }
    }
    (wrong == 0, wrong, max_ulp)
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "bounded oracle accepts no arguments"
    );
    let began = Instant::now();
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    ensure!(
        fixture["oracle"]["device"] == "cpu",
        "expected pinned CPU/Torch reference fixture"
    );
    let dev = CudaDevice::new(0).context("CUDA device0")?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let gpu = GpuNumerics::load(&dev, &stream)?;
    let mut checks = Vec::new();

    let cast = fixture["scalar_e4m3_cast_cases"]
        .as_array()
        .context("FP8cast cases")?;
    let values: Vec<f32> = cast
        .iter()
        .map(|r| f32::from_bits(r["input_f32_bits"].as_u64().unwrap() as u32))
        .collect();
    let wanted: Vec<u8> = cast
        .iter()
        .map(|r| r["expected_byte"].as_u64().unwrap() as u8)
        .collect();
    let input = dev.htod_sync_copy(&values)?;
    let actual = gpu
        .encode_f32_scalars(&stream, &input)?
        .download_checked()?;
    record(
        &mut checks,
        "e4m3_RNE_ties_clipping_signed_zero".into(),
        actual == wanted,
        json!({"cases":cast.len(),"first_difference":first_byte_difference(&actual,&wanted),"tolerance":"bitwise FP8bytes"}),
    );

    let cases = fixture["ue8m0_decode_cases"]
        .as_array()
        .context("UE8M0cases")?;
    let values: Vec<u8> = cases
        .iter()
        .map(|r| r["byte"].as_u64().unwrap() as u8)
        .collect();
    let wanted: Vec<u32> = cases
        .iter()
        .map(|r| r["expected_f32_bits"].as_u64().unwrap() as u32)
        .collect();
    let input = dev.htod_sync_copy(&values)?;
    let actual = gpu
        .decode_ue8m0_scalars(&stream, &input)?
        .download_checked()?;
    record(
        &mut checks,
        "UE8M0_decode_including_subnormal_scale0".into(),
        f32_bits_match(&actual, &wanted),
        json!({"cases":cases.len(),"tolerance":"bitwise FP32","scale0_reference_bits":1_u32<<22}),
    );

    for case in fixture["activation_cases"]
        .as_array()
        .context("activationcases")?
    {
        let rows = n(case, "rows")?;
        let columns = n(case, "columns")?;
        let input = dev.htod_sync_copy(&bfloat(case, "input_bf16_bits")?)?;
        let actual = gpu
            .quantize_activations_bf16(&stream, &input, rows, columns)?
            .download_checked()?;
        let wanted = bytes(case, "expected_e4m3_bytes")?;
        let scales = bytes(case, "expected_ue8m0_scale_bytes")?;
        record(
            &mut checks,
            format!("group32_activation_{}", case["name"].as_str().unwrap()),
            actual.e4m3_values == wanted && actual.ue8m0_scales == scales,
            json!({"rows":rows,"columns":columns,"first_value_difference":first_byte_difference(&actual.e4m3_values,&wanted),
                "actual_scale_bytes":actual.ue8m0_scales,"expected_scale_bytes":scales,"tolerance":"bitwise FP8+UE8M0"}),
        );
        if case["old_group128_must_differ"] == true {
            let old = gpu
                .group128_negative_control(&stream, &input, rows, columns)?
                .download_checked()?;
            let old_values = bytes(case, "old_group128_e4m3_bytes")?;
            let old_scales = bytes(case, "old_group128_scale_bytes")?;
            record(
                &mut checks,
                "negative_old_group128_is_detected".into(),
                old.e4m3_values == old_values
                    && old.ue8m0_scales == old_scales
                    && old.e4m3_values != actual.e4m3_values,
                json!({"matches_old_Torch_reference":old.e4m3_values==old_values&&old.ue8m0_scales==old_scales,
                    "differs_from1017group32":old.e4m3_values!=actual.e4m3_values}),
            );
        }
    }
    for case in fixture["native_fp4_cases"].as_array().context("FP4cases")? {
        let packed = dev.htod_sync_copy(&bytes(case, "packed_bytes")?)?;
        let scales = dev.htod_sync_copy(&bytes(case, "ue8m0_scale_bytes")?)?;
        let actual = gpu
            .dequantize_fp4_native(
                &stream,
                &packed,
                &scales,
                n(case, "rows")?,
                n(case, "columns")?,
            )?
            .download_checked()?;
        let expected: Vec<u32> = serde_json::from_value(case["expected_f32_bits"].clone())?;
        record(
            &mut checks,
            format!("native_FP4_{}", case["name"].as_str().unwrap()),
            f32_bits_match(&actual, &expected),
            json!({"elements":actual.len(),"tolerance":"bitwise FP32 including source+zero-code8"}),
        );
    }
    for case in fixture["rms_cases"].as_array().context("RMScases")? {
        let input = dev.htod_sync_copy(&bfloat(case, "input_bf16_bits")?)?;
        let weights = dev.htod_sync_copy(&floats(case, "weight_f32_bits")?)?;
        let actual = gpu
            .rms_norm_1017(
                &stream,
                &input,
                &weights,
                n(case, "rows")?,
                n(case, "columns")?,
            )?
            .download_checked()?;
        let expected = bfloat(case, "expected_bf16_bits")?;
        let (passed, wrong, max_ulp) = compare_bf16(&actual, &expected, 0);
        record(
            &mut checks,
            format!("RMS_eps1e20_{}", case["name"].as_str().unwrap()),
            passed,
            json!({"elements":actual.len(),"mismatches":wrong,"max_bf16_ulp":max_ulp,"tolerance":"bitwise BF16 on pinnedTorchgoldens"}),
        );
        if case["old_eps_must_differ"] == true {
            let old = gpu
                .rms_norm_bf16(
                    &stream,
                    &input,
                    &weights,
                    n(case, "rows")?,
                    n(case, "columns")?,
                    1e-6,
                )?
                .download_checked()?;
            let wanted_old = bfloat(case, "old_eps1e6_bf16_bits")?;
            let old_matches = compare_bf16(&old, &wanted_old, 0).0;
            record(
                &mut checks,
                "negative_old_RMS_eps1e6_is_detected".into(),
                old_matches && !compare_bf16(&old, &actual, 0).0,
                json!({"matches_old_Torch_reference":old_matches,"differs_from1017epsilon":!compare_bf16(&old,&actual,0).0}),
            );
        }
    }

    // Actual1017 hidden geometry and independent batch rows; no throughput measurement.
    for rows in [1_usize, 2, 4, 8, 16] {
        let columns = 5120_usize;
        let data: Vec<bf16> = (0..rows * columns)
            .map(|i| {
                let row = i / columns;
                let col = i % columns;
                bf16::from_f32(((row * 19 + col * 7) % 31) as f32 - 15.0)
                    * bf16::from_f32(2_f32.powi(((col / 32) % 4) as i32 - 2))
            })
            .collect();
        let input = dev.htod_sync_copy(&data)?;
        let cpu = numerics::quantize_activations_bf16(&data, rows, columns)?;
        let actual = gpu
            .quantize_activations_bf16(&stream, &input, rows, columns)?
            .download_checked()?;
        record(
            &mut checks,
            format!("hidden5120_group32_rows{rows}"),
            actual == cpu,
            json!({"rows":rows,"columns":columns,"tolerance":"bitwise against qualifiedCPUreference"}),
        );
        let weights: Vec<f32> = (0..columns).map(|i| ((i % 3) + 1) as f32 * 0.5).collect();
        let w = dev.htod_sync_copy(&weights)?;
        let cpu = numerics::rms_norm_bf16(&data, &weights, rows, columns)?;
        let actual = gpu
            .rms_norm_1017(&stream, &input, &w, rows, columns)?
            .download_checked()?;
        let (passed, wrong, max_ulp) = compare_bf16(&actual, &cpu, 1);
        record(
            &mut checks,
            format!("hidden5120_RMS_rows{rows}"),
            passed,
            json!({"rows":rows,"columns":columns,"mismatches":wrong,"max_bf16_ulp":max_ulp,
                "tolerance":"predeclared <=1BF16code step, finite same-sign; GPUreductiontree differs fromCPUserialsum"}),
        );
    }
    {
        let rows = 2;
        let columns = 5120;
        let packed: Vec<u8> = (0..rows * columns / 2)
            .map(|i| ((i * 17 + 3) % 256) as u8)
            .collect();
        let scales: Vec<u8> = (0..rows * columns / 32)
            .map(|i| 126 + (i % 3) as u8)
            .collect();
        let cpu = numerics::dequantize_fp4_native(&packed, &scales, rows, columns)?;
        let p = dev.htod_sync_copy(&packed)?;
        let s = dev.htod_sync_copy(&scales)?;
        let actual = gpu
            .dequantize_fp4_native(&stream, &p, &s, rows, columns)?
            .download_checked()?;
        record(
            &mut checks,
            "native_FP4_columns5120".into(),
            actual
                .iter()
                .map(|x| x.to_bits())
                .eq(cpu.iter().map(|x| x.to_bits())),
            json!({"elements":actual.len(),"tolerance":"bitwise FP32"}),
        );
    }

    // Real device-side rejection controls; placeholders can never count as valid results.
    let invalid = dev.htod_sync_copy(&[f32::NAN])?;
    let result = gpu.encode_f32_scalars(&stream, &invalid)?;
    record(
        &mut checks,
        "negative_NaN_activation_rejected".into(),
        result.download_checked().is_err() && result.status_code()? != 0,
        json!({"status":result.status_code()?}),
    );
    let invalid = dev.htod_sync_copy(&[255_u8])?;
    let result = gpu.decode_ue8m0_scalars(&stream, &invalid)?;
    record(
        &mut checks,
        "negative_NaN_scale_rejected".into(),
        result.download_checked().is_err() && result.status_code()? != 0,
        json!({"status":result.status_code()?}),
    );
    let packed = dev.htod_sync_copy(&[0x77_u8; 16])?;
    let scale = dev.htod_sync_copy(&[254_u8])?;
    let result = gpu.dequantize_fp4_native(&stream, &packed, &scale, 1, 32)?;
    record(
        &mut checks,
        "negative_FP4_FP32_overflow_rejected".into(),
        result.download_checked().is_err() && result.status_code()? != 0,
        json!({"status":result.status_code()?}),
    );
    let input = dev.htod_sync_copy(&[bf16::MAX; 32])?;
    let weights = dev.htod_sync_copy(&[1_f32; 32])?;
    let result = gpu.rms_norm_1017(&stream, &input, &weights, 1, 32)?;
    record(
        &mut checks,
        "negative_RMS_variance_overflow_rejected".into(),
        result.download_checked().is_err() && result.status_code()? != 0,
        json!({"status":result.status_code()?}),
    );
    let nonblocking = dev.fork_default_stream()?;
    record(
        &mut checks,
        "negative_nonblocking_stream_rejected".into(),
        GpuNumerics::load(&dev, &nonblocking).is_err(),
        json!({"requires_explicit_blocking_stream":true}),
    );

    let all_passed = checks.iter().all(|c| c["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_numerics_gpu_component_oracle",
        "status":if all_passed{"PASS"}else{"FAIL"},"all_passed":all_passed,
        "serve_supported":false,"model_execution":false,"source_id_hex":format!("{:016x}",gpu.source_id()),
        "cuda_source_sha256":gpu.cuda_source_sha256(),"ptx_sha256":gpu.ptx_sha256(),
        "cpu_fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),
        "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL_SECONDS,
            "explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND_BYTES,
            "memory_scope":"smallfixture+hidden5120rows<=16 vectors; excludesCUDAcontext/moduleoverhead"},
        "checks":checks,"scope":"IsolatedGPUquantization/dequantization/RMScomponents. No modelweights, modelinference, toolprotocol, TP8engine, speedclaim or GPUW8A8GEMMqualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_numerics_gpu_component_oracle",
            "status":"FAIL","all_passed":false,"error":"hard wall-time limit"})
        );
        std::process::exit(124);
    });
    let value = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({"schema_version":1,"kind":"dsv41_numerics_gpu_component_oracle",
            "status":"FAIL","all_passed":false,"error":format!("{e:#}")}),
        Err(_) => json!({"schema_version":1,"kind":"dsv41_numerics_gpu_component_oracle",
            "status":"FAIL","all_passed":false,"error":"panic"}),
    };
    println!("{}", value);
    if value["all_passed"] != true {
        std::process::exit(1);
    }
}
