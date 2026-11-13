//! Native1017 model-free FP4 cache/activation oracle. No model payload or RDMA.
//! Parent owns GPU exclusivity, memory/thermal guards and independent wall bounds.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    cache_quant::{self, CacheFormat},
    gpu_cache_quant::GpuCacheQuant,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_cache_quant_vectors.json");
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND_BYTES: usize = 16 * 1024 * 1024;
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];

fn differences<T: PartialEq>(actual: &[T], expected: &[T]) -> usize {
    usize::from(actual.len() != expected.len())
        + actual.iter().zip(expected).filter(|(a, b)| a != b).count()
}
fn bf_bits(values: &[bf16]) -> Vec<u16> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn record(checks: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    checks.push(json!({"name":name,"passed":passed,"details":details}));
}
fn format(value: &Value) -> Result<CacheFormat> {
    match value.as_str() {
        Some("compressed_kv") => Ok(CacheFormat::CompressedKvE4m3Group16),
        Some("index_qk") => Ok(CacheFormat::IndexUe8m0Group32),
        _ => anyhow::bail!("unknown cache fixture format"),
    }
}
fn fixture_cases(
    dev: &Arc<CudaDevice>,
    gpu: &GpuCacheQuant,
    stream: &cudarc::driver::CudaStream,
    checks: &mut Vec<Value>,
    fixture: &Value,
) -> Result<()> {
    let cases = fixture["cases"].as_array().context("cache fixture cases")?;
    ensure!(cases.len() == 10, "cache fixture cardinality");
    for case in cases {
        let name = case["name"].as_str().context("case name")?;
        let mode = format(&case["format"])?;
        let rows = case["rows"].as_u64().context("rows")? as usize;
        let columns = case["columns"].as_u64().context("columns")? as usize;
        let input_bits: Vec<u16> = serde_json::from_value(case["input_bf16_bits"].clone())?;
        let input: Vec<_> = input_bits.iter().copied().map(bf16::from_bits).collect();
        let packed: Vec<u8> = serde_json::from_value(case["packed_bytes"].clone())?;
        let scales: Vec<u8> = serde_json::from_value(case["scale_bytes"].clone())?;
        let qat: Vec<u16> = serde_json::from_value(case["qat_bf16_bits"].clone())?;
        let count = cache_quant::elements(rows, columns, mode)?;
        ensure!(
            input.len() == count
                && packed.len() == count / 2
                && scales.len() == count / mode.group()
                && qat.len() == count,
            "cache golden full output geometry"
        );
        let cpu = cache_quant::quantize_bf16(&input, rows, columns, mode)?;
        ensure!(
            cpu.packed == packed && cpu.scales == scales && bf_bits(&cpu.qat) == qat,
            "Rust CPU differs from independent cache vectors"
        );
        let original = dev.htod_sync_copy(&input)?;
        let result = gpu.quantize(stream, &original, rows, columns, mode)?;
        let output = result.download_checked()?;
        let p = differences(&output.packed, &packed);
        let s = differences(&output.scales, &scales);
        let q = differences(&bf_bits(&output.qat), &qat);
        let changed = differences(&bf_bits(&dev.dtoh_sync_copy(&original)?), &input_bits);
        record(
            checks,
            format!("{name}_packed_and_QAT"),
            p + s + q + changed == 0,
            json!({"format":mode,"rows":rows,"columns":columns,"packed_mismatches":p,"scale_mismatches":s,"QAT_bf16_mismatches":q,"immutable_input_mismatches":changed,"elements":count,"tolerance":"bitwise including signed zero"}),
        );
        let mut inplace = dev.htod_sync_copy(&input)?;
        let native = gpu.quantize_in_place(stream, &mut inplace, rows, columns, mode)?;
        let (ip, is) = native.download_checked()?;
        let iq = dev.dtoh_sync_copy(&inplace)?;
        let mismatch = differences(&ip, &packed)
            + differences(&is, &scales)
            + differences(&bf_bits(&iq), &qat);
        record(
            checks,
            format!("{name}_actual_inplace"),
            mismatch == 0,
            json!({"elements":count,"mismatches":mismatch,"tolerance":"bitwise"}),
        );
        result.native().check()?;
        let rebuilt = unsafe {
            gpu.dequantize(
                stream,
                result.native().device_packed_unchecked(),
                result.native().device_scales_unchecked(),
                rows,
                columns,
                mode,
            )
        }?;
        let decoded = rebuilt.download_checked()?;
        let mismatch = differences(&bf_bits(&decoded), &qat);
        record(
            checks,
            format!("{name}_native_GPU_roundtrip"),
            mismatch == 0,
            json!({"elements":count,"mismatches":mismatch,"tolerance":"bitwise"}),
        );
    }
    Ok(())
}

fn widths(
    dev: &Arc<CudaDevice>,
    gpu: &GpuCacheQuant,
    stream: &cudarc::driver::CudaStream,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for mode in [
        CacheFormat::CompressedKvE4m3Group16,
        CacheFormat::IndexUe8m0Group32,
    ] {
        let columns = if mode == CacheFormat::CompressedKvE4m3Group16 {
            512
        } else {
            128
        };
        let input: Vec<_> = (0..128 * columns)
            .map(|i| {
                let row = i / columns;
                let col = i % columns;
                bf16::from_f32(
                    (((row * 113 + col * 7) % 97) as f32 - 48.0) / 16.0
                        * 2.0_f32.powi(((col / mode.group()) % 12) as i32 - 6),
                )
            })
            .collect();
        let source = dev.htod_sync_copy(&input)?;
        let base = gpu
            .quantize(stream, &source, 128, columns, mode)?
            .download_checked()?;
        let cpu = cache_quant::quantize_bf16(&input, 128, columns, mode)?;
        let errors = differences(&base.packed, &cpu.packed)
            + differences(&base.scales, &cpu.scales)
            + differences(&bf_bits(&base.qat), &bf_bits(&cpu.qat));
        record(
            checks,
            format!("{mode:?}_baseline128"),
            errors == 0,
            json!({"rows":128,"columns":columns,"mismatches":errors,"tolerance":"bitwise"}),
        );
        for rows in [1, 2, 4, 8, 16, 128] {
            let order: Vec<_> = if rows == 128 {
                (0..128).map(|i| (17 * i + 7) % 128).collect()
            } else {
                ORDER[..rows].to_vec()
            };
            let mut selected = Vec::new();
            let mut expected_p = Vec::new();
            let mut expected_s = Vec::new();
            let mut expected_q = Vec::new();
            for &row in &order {
                selected.extend_from_slice(&input[row * columns..(row + 1) * columns]);
                expected_p
                    .extend_from_slice(&base.packed[row * columns / 2..(row + 1) * columns / 2]);
                expected_s.extend_from_slice(
                    &base.scales[row * columns / mode.group()..(row + 1) * columns / mode.group()],
                );
                expected_q.extend_from_slice(&base.qat[row * columns..(row + 1) * columns]);
            }
            let data = dev.htod_sync_copy(&selected)?;
            let output = gpu
                .quantize(stream, &data, rows, columns, mode)?
                .download_checked()?;
            let mismatch = differences(&output.packed, &expected_p)
                + differences(&output.scales, &expected_s)
                + differences(&bf_bits(&output.qat), &bf_bits(&expected_q));
            record(
                checks,
                format!("{mode:?}_permuted_rows{rows}"),
                mismatch == 0,
                json!({"rows":rows,"columns":columns,"source_rows":order,"elements":rows*columns,"mismatches":mismatch,"tolerance":"bitwise all packed bytes/scales/BF16 values"}),
            );
        }
    }
    Ok(())
}

fn controls(
    dev: &Arc<CudaDevice>,
    gpu: &GpuCacheQuant,
    stream: &cudarc::driver::CudaStream,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for mode in [
        CacheFormat::CompressedKvE4m3Group16,
        CacheFormat::IndexUe8m0Group32,
    ] {
        let mut input = vec![bf16::ONE; mode.group()];
        input[0] = bf16::NAN;
        let data = dev.htod_sync_copy(&input)?;
        let result = gpu.quantize(stream, &data, 1, mode.group(), mode)?;
        let status = result.native().status_code()?;
        let rejected = result.download_checked().is_err();
        record(
            checks,
            format!("negative_{mode:?}_NaN_input"),
            status == 1 && rejected,
            json!({"status":status,"expected_status":1,"checked_download_rejected":rejected}),
        );
    }
    let data = dev.htod_sync_copy(&[bf16::MAX; 32])?;
    let result = gpu.quantize(stream, &data, 1, 32, CacheFormat::IndexUe8m0Group32)?;
    let status = result.native().status_code()?;
    let rejected = result.download_checked().is_err();
    record(
        checks,
        "negative_index_QAT_overflow".into(),
        status == 4 && rejected,
        json!({"status":status,"expected_status":4,"checked_download_rejected":rejected}),
    );
    for (name, mode, code, wanted) in [
        (
            "E4M3_NaN",
            CacheFormat::CompressedKvE4m3Group16,
            127u8,
            2u32,
        ),
        ("UE8M0_NaN", CacheFormat::IndexUe8m0Group32, 255, 2),
        ("E4M3_zero", CacheFormat::CompressedKvE4m3Group16, 0, 2),
        ("UE8M0_QAT_overflow", CacheFormat::IndexUe8m0Group32, 254, 4),
    ] {
        let packed = dev.htod_sync_copy(&vec![0x77u8; mode.group() / 2])?;
        let scales = dev.htod_sync_copy(&[code])?;
        let result = gpu.dequantize(stream, &packed, &scales, 1, mode.group(), mode)?;
        let status = result.status_code()?;
        let rejected = result.download_checked().is_err();
        record(
            checks,
            format!("negative_scale_{name}"),
            status == wanted && rejected,
            json!({"status":status,"expected_status":wanted,"checked_download_rejected":rejected}),
        );
    }
    for mode in [
        CacheFormat::CompressedKvE4m3Group16,
        CacheFormat::IndexUe8m0Group32,
    ] {
        let count = mode.group();
        let packed: Vec<_> = (0..count / 2)
            .map(|i| ((2 * i) % 16) as u8 | ((((2 * i + 1) % 16) as u8) << 4))
            .collect();
        let scale = if mode == CacheFormat::CompressedKvE4m3Group16 {
            56
        } else {
            127
        };
        let p = dev.htod_sync_copy(&packed)?;
        let s = dev.htod_sync_copy(&[scale])?;
        let actual = gpu
            .dequantize(stream, &p, &s, 1, count, mode)?
            .download_checked()?;
        let expected = cache_quant::dequantize_bf16(&packed, &[scale], 1, count, mode)?;
        let mismatch = differences(&bf_bits(&actual), &bf_bits(&expected));
        record(
            checks,
            format!("{mode:?}_all16_native_codes_signed_zero"),
            mismatch == 0 && actual[8].to_bits() == 0x8000,
            json!({"elements":count,"mismatches":mismatch,"code8_bf16_bits":actual[8].to_bits(),"tolerance":"bitwise"}),
        );
    }
    let packed: Vec<_> = (0..16)
        .map(|i| (2 * i % 16) as u8 | (((2 * i + 1) % 16) as u8) << 4)
        .collect();
    let p = dev.htod_sync_copy(&packed)?;
    let s = dev.htod_sync_copy(&[0u8])?;
    let actual = gpu
        .dequantize(stream, &p, &s, 1, 32, CacheFormat::IndexUe8m0Group32)?
        .download_checked()?;
    let expected =
        cache_quant::dequantize_bf16(&packed, &[0], 1, 32, CacheFormat::IndexUe8m0Group32)?;
    let mismatch = differences(&bf_bits(&actual), &bf_bits(&expected));
    record(
        checks,
        "native_UE8M0_scale0_gradual_underflow".into(),
        mismatch == 0,
        json!({"elements":32,"mismatches":mismatch,"tolerance":"bitwise"}),
    );
    let p = dev.htod_sync_copy(&[0u8; 64])?;
    let s = dev.htod_sync_copy(&[127u8])?;
    let rejected = gpu
        .dequantize(stream, &p, &s, 1, 128, CacheFormat::IndexUe8m0Group32)
        .is_err();
    record(
        checks,
        "negative_old128_scale_shape".into(),
        rejected,
        json!({"rejected":rejected}),
    );
    let nonblocking = dev.fork_default_stream()?;
    let input = dev.htod_sync_copy(&[bf16::ONE; 32])?;
    let rejected = gpu
        .quantize(&nonblocking, &input, 1, 32, CacheFormat::IndexUe8m0Group32)
        .is_err();
    record(
        checks,
        "negative_nonblocking_stream".into(),
        rejected,
        json!({"rejected":rejected}),
    );
    let input = dev.htod_sync_copy(&[f32::NAN])?;
    let result = gpu.encode_scalars(stream, &input)?;
    let status = result.status_code()?;
    let rejected = result.download_checked().is_err();
    record(
        checks,
        "negative_native_FP4_NaN_cast".into(),
        status == 1 && rejected,
        json!({"status":status,"expected_status":1,"checked_download_rejected":rejected}),
    );
    Ok(())
}

fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "cache oracle takes no arguments"
    );
    let began = Instant::now();
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    ensure!(
        fixture["kernel_reference_sha256"] == cache_quant::KERNEL_REFERENCE_SHA256
            && fixture["cases"].as_array().context("cases")?.len() == 10,
        "unqualified cache fixture"
    );
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let gpu = GpuCacheQuant::load(&dev, &stream)?;
    let mut checks = Vec::new();
    let bits: Vec<u32> = serde_json::from_value(fixture["scalar_cast"]["input_f32_bits"].clone())?;
    let expected: Vec<u8> =
        serde_json::from_value(fixture["scalar_cast"]["expected_nibbles"].clone())?;
    ensure!(
        bits.len() == 46 && expected.len() == 46,
        "native scalar fixture cardinality"
    );
    let input: Vec<_> = bits.into_iter().map(f32::from_bits).collect();
    let data = dev.htod_sync_copy(&input)?;
    let actual = gpu.encode_scalars(&stream, &data)?.download_checked()?;
    let mismatch = differences(&actual, &expected);
    record(
        &mut checks,
        "native_FP4_RNE_ties_neighbors_signed_zero".into(),
        mismatch == 0,
        json!({"elements":46,"mismatches":mismatch,"actual_nibbles":actual,"expected_nibbles":expected,"tolerance":"bitwise native GPU cast vs explicit CPU RNE table"}),
    );
    fixture_cases(&dev, &gpu, &stream, &mut checks, &fixture)?;
    widths(&dev, &gpu, &stream, &mut checks)?;
    controls(&dev, &gpu, &stream, &mut checks)?;
    ensure!(
        checks.len() == 58,
        "cache oracle coverage changed: {}",
        checks.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL_SECONDS),
        "cache oracle wall deadline"
    );
    let passed = checks.iter().all(|c| c["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_cache_quant_gpu_component_oracle","status":if passed{"PASS"}else{"FAIL"},"all_passed":passed,
        "model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":false,
        "source_id_hex":format!("{:016x}",gpu.source_id()),"cuda_source_sha256":gpu.cuda_source_sha256(),"ptx_sha256":gpu.ptx_sha256(),
        "fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),"kernel_reference_sha256":cache_quant::KERNEL_REFERENCE_SHA256,
        "elapsed_seconds":began.elapsed().as_secs_f64(),"checks":checks,
        "limits":{"wall_seconds":WALL_SECONDS,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND_BYTES},
        "formats":{"compressed_KV":"group16 FP4 E2M1 +RNE satfinite E4M3scale, floor6*2^-9","index_QK":"group32 FP4 E2M1 +UE8M0pow2 scale, floor6*2^-126"},
        "scope":"synthetic activation/cache packing, native scalar casts, actual in-place QAT and BF16 reconstruction only; no checkpoint payload, cache residency, attention/index forward, model serving or speed qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("cache quant oracle hard deadline");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_cache_quant_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_cache_quant_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
