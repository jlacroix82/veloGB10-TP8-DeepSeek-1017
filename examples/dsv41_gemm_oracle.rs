//! Model-free native group32 GEMM oracle. No checkpoint payload or serving route.
//! The independent CPU formula follows official kernel.py, retaining each K32 scale.
//! Run release profile only, with exclusive device ownership and external60s timeout.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    gpu_gemm::{GemmShape, GpuGemm},
    numerics,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND_BYTES: usize = 128 * 1024 * 1024;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
const KERNEL_REF: &str = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455";
const MODEL_REF: &str = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d";
#[derive(Clone)]
struct Pack {
    rows: usize,
    outputs: usize,
    inner: usize,
    fp4: bool,
    a: Vec<u8>,
    a_scales: Vec<u8>,
    w: Vec<u8>,
    w_scales: Vec<u8>,
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn pack(rows: usize, outputs: usize, inner: usize, fp4: bool) -> Pack {
    // Dyadic validation family: |activation*weight|<=9, per-side exponents[-2,1].
    // Even at K6144, scaled absolute sum / smallest product quantum <2^24.
    // Thus CPUFP32, CPUFP64 and correct FP32 MMA all have an exact diagnostic target.
    let codes = [0_u8, 0x30, 0x38, 0x3c, 0x40, 0x44];
    let a = (0..rows * inner)
        .map(|i| {
            let z = mix(i as u64 ^ 0xa37b39);
            let mag = codes[(z as usize) % (if fp4 { 4 } else { 6 })];
            mag | if (z >> 17) & 1 != 0 && mag != 0 {
                0x80
            } else {
                0
            }
        })
        .collect();
    let groups = inner / 32;
    let a_scales = (0..rows * groups)
        .map(|i| 125 + (mix(i as u64 ^ 0x88377) % 4) as u8)
        .collect();
    let w = if fp4 {
        (0..outputs * inner / 2)
            .map(|i| {
                let lo = (mix((2 * i) as u64 ^ 0x775533) % 16) as u8;
                let hi = (mix((2 * i + 1) as u64 ^ 0x775533) % 16) as u8;
                lo | (hi << 4)
            })
            .collect()
    } else {
        (0..outputs * inner)
            .map(|i| {
                let z = mix(i as u64 ^ 0x775533);
                let mag = codes[(z as usize) % 6];
                mag | if (z >> 19) & 1 != 0 && mag != 0 {
                    0x80
                } else {
                    0
                }
            })
            .collect()
    };
    let scale_rows = if fp4 { outputs } else { outputs.div_ceil(32) };
    let w_scales = (0..scale_rows * groups)
        .map(|i| 125 + (mix(i as u64 ^ 0x224466) % 4) as u8)
        .collect();
    Pack {
        rows,
        outputs,
        inner,
        fp4,
        a,
        a_scales,
        w,
        w_scales,
    }
}
fn fp4_value(code: u8) -> f32 {
    let table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let x = table[(code & 7) as usize];
    if code & 8 != 0 && code & 7 != 0 {
        -x
    } else {
        x
    }
}
fn weight(p: &Pack, out: usize, k: usize) -> f32 {
    if p.fp4 {
        let b = p.w[out * (p.inner / 2) + k / 2];
        fp4_value(if k & 1 == 0 { b & 15 } else { b >> 4 })
    } else {
        numerics::decode_e4m3_finite(p.w[out * p.inner + k]).unwrap()
    }
}
#[derive(Clone, Copy)]
struct Ref {
    f32_value: f32,
    f64_value: f64,
}
fn reference(p: &Pack, row: usize, out: usize) -> Ref {
    let groups = p.inner / 32;
    let mut answer = 0_f32;
    let mut exact = 0_f64;
    for g in 0..groups {
        let mut raw = 0_f32;
        let mut raw64 = 0_f64;
        for k in g * 32..(g + 1) * 32 {
            let a = numerics::decode_e4m3_finite(p.a[row * p.inner + k]).unwrap();
            let w = weight(p, out, k);
            raw += a * w;
            raw64 += (a as f64) * (w as f64);
        }
        let sa = numerics::decode_ue8m0(p.a_scales[row * groups + g]).unwrap();
        let wr = if p.fp4 { out } else { out / 32 };
        let sw = numerics::decode_ue8m0(p.w_scales[wr * groups + g]).unwrap();
        answer += (raw * sa) * sw;
        exact += (raw64 * sa as f64) * sw as f64;
    }
    Ref {
        f32_value: answer,
        f64_value: exact,
    }
}
fn probes(outputs: usize, full: bool) -> Vec<usize> {
    if full {
        return (0..outputs).collect();
    }
    let mut p = vec![
        0,
        1,
        7,
        15,
        16,
        31,
        32,
        63,
        127,
        128,
        255,
        outputs / 2,
        outputs - 2,
        outputs - 1,
    ];
    p.retain(|&x| x < outputs);
    p.sort_unstable();
    p.dedup();
    p
}
fn shuffled(p: &Pack, order: &[usize]) -> Pack {
    let mut q = p.clone();
    q.rows = order.len();
    q.a.clear();
    q.a_scales.clear();
    let groups = p.inner / 32;
    for &row in order {
        q.a.extend_from_slice(&p.a[row * p.inner..(row + 1) * p.inner]);
        q.a_scales
            .extend_from_slice(&p.a_scales[row * groups..(row + 1) * groups]);
    }
    q
}
fn hash_inputs(p: &Pack) -> String {
    let mut h = Sha256::new();
    for x in [&p.a, &p.a_scales, &p.w, &p.w_scales] {
        h.update((x.len() as u64).to_le_bytes());
        h.update(x);
    }
    format!("{:x}", h.finalize())
}
fn record(checks: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    checks.push(json!({"name":name,"passed":passed,"details":details}));
}
fn device_run(
    dev: &std::sync::Arc<CudaDevice>,
    gpu: &GpuGemm,
    stream: &cudarc::driver::CudaStream,
    p: &Pack,
) -> Result<(Vec<f32>, Vec<bf16>)> {
    let a = dev.htod_sync_copy(&p.a)?;
    let sa = dev.htod_sync_copy(&p.a_scales)?;
    let w = dev.htod_sync_copy(&p.w)?;
    let sw = dev.htod_sync_copy(&p.w_scales)?;
    let shape = GemmShape {
        rows: p.rows,
        outputs: p.outputs,
        inner: p.inner,
    };
    let full = if p.fp4 {
        gpu.fp4_partial_f32(stream, &a, &sa, &w, &sw, shape)?
    } else {
        gpu.fp8_partial_f32(stream, &a, &sa, &w, &sw, shape)?
    };
    let rounded = if p.fp4 {
        gpu.fp4_complete_bf16(stream, &a, &sa, &w, &sw, shape)?
    } else {
        gpu.fp8_complete_bf16(stream, &a, &sa, &w, &sw, shape)?
    };
    Ok((full.download_checked()?, rounded.download_checked()?))
}
fn case(
    dev: &std::sync::Arc<CudaDevice>,
    gpu: &GpuGemm,
    stream: &cudarc::driver::CudaStream,
    checks: &mut Vec<Value>,
    name: &str,
    outputs: usize,
    inner: usize,
    fp4: bool,
    full: bool,
) -> Result<()> {
    let p = pack(16, outputs, inner, fp4);
    let inputs_sha = hash_inputs(&p);
    let (base, base_bf) = device_run(dev, gpu, stream, &p)?;
    let selected = probes(outputs, full);
    let mut refs = Vec::new();
    let mut base_wrong = 0;
    let mut f64_wrong = 0;
    let mut max_fp64_error = 0_f64;
    for row in 0..16 {
        let mut rr = Vec::new();
        for &out in &selected {
            let r = reference(&p, row, out);
            let got = base[row * outputs + out];
            if got.to_bits() != r.f32_value.to_bits() {
                base_wrong += 1;
            }
            if r.f32_value.to_bits() != (r.f64_value as f32).to_bits() {
                f64_wrong += 1;
            }
            max_fp64_error = max_fp64_error.max((got as f64 - r.f64_value).abs());
            rr.push(r);
        }
        refs.push(rr);
    }
    let mut ref_digest = Sha256::new();
    for row in &refs {
        for value in row {
            ref_digest.update(value.f32_value.to_le_bytes());
        }
    }
    let cpu_reference_hash = format!("{:x}", ref_digest.finalize());
    // Independently generated NumPyCPU formula goldens for complete35x96 tinycases.
    // Provenance: private/velogb10-tp8/gemm-cpu-preflight.json; no GPU/Torch claim.
    let independent_hash = if fp4 {
        "59f2992ee2099dd31b64b171872fa7123e13ab27d43b171d2d1d3807065737a9"
    } else {
        "16953c58cf9b3262288ec5df91eea43c7cf8828ce9a6a91929deec62ebe917bc"
    };
    let independent_cpu_golden_matches = !full || cpu_reference_hash == independent_hash;
    let base_round_wrong = base
        .iter()
        .zip(&base_bf)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    let non_bf16 = base
        .iter()
        .filter(|&&x| bf16::from_f32(x).to_f32() != x)
        .count();
    record(
        checks,
        format!("{name}_CPU_reference"),
        base_wrong == 0
            && f64_wrong == 0
            && base_round_wrong == 0
            && independent_cpu_golden_matches,
        json!({"rows":16,"outputs":outputs,"inner":inner,"native_fp4":fp4,"input_sha256":inputs_sha,
            "CPU_outputs_checked":16*selected.len(),"CPU_output_indices":selected,
            "CPU_reference_sha256":cpu_reference_hash,"independent_CPU_golden_applies":full,
            "independent_CPU_golden_matches":independent_cpu_golden_matches,"CPU_mismatches":base_wrong,
            "CPU_FP32_vs_FP64_mismatches":f64_wrong,"max_absolute_FP64_error":max_fp64_error,
            "BF16_complete_round_mismatches":base_round_wrong,"FP32_outputs_not_BF16_representable":non_bf16,
            "reference":"independent ascendingK32 FP32 formula plus FP64 diagnostic; exact dyadic family",
            "tolerance":"bitwise FP32 and BF16"}),
    );
    for width in WIDTHS {
        let order = &ORDER[..width];
        let q = shuffled(&p, order);
        let (got, rounded) = device_run(dev, gpu, stream, &q)?;
        let mut wrong = 0;
        let mut round_wrong = 0;
        let mut cpu_wrong = 0;
        for row in 0..width {
            let source = order[row];
            for out in 0..outputs {
                let i = row * outputs + out;
                let b = source * outputs + out;
                if got[i].to_bits() != base[b].to_bits() {
                    wrong += 1;
                }
                if rounded[i].to_bits() != base_bf[b].to_bits()
                    || rounded[i].to_bits() != bf16::from_f32(got[i]).to_bits()
                {
                    round_wrong += 1;
                }
            }
            for (j, &out) in selected.iter().enumerate() {
                if got[row * outputs + out].to_bits() != refs[source][j].f32_value.to_bits() {
                    cpu_wrong += 1;
                }
            }
        }
        record(
            checks,
            format!("{name}_permuted_C{width}"),
            wrong == 0 && round_wrong == 0 && cpu_wrong == 0,
            json!({"width":width,"rows_from_reference":order,"outputs":outputs,"inner":inner,
                "elements_compared":got.len(),"cross_width_FP32_mismatches":wrong,
                "cross_width_BF16_mismatches":round_wrong,"CPU_mismatches":cpu_wrong,
                "CPU_outputs_checked":width*selected.len(),"tolerance":"bitwise"}),
        );
    }
    if full {
        record(
            checks,
            format!("{name}_negative_premature_BF16_partial"),
            non_bf16 > 0,
            json!({"non_BF16_FP32_outputs":non_bf16,"detected":non_bf16>0,
                "meaning":"partial path preserves FP32 rather than rounding each projection toBF16"}),
        );
    }
    Ok(())
}
fn bad_status(
    dev: &std::sync::Arc<CudaDevice>,
    gpu: &GpuGemm,
    stream: &cudarc::driver::CudaStream,
    p: &Pack,
) -> Result<(u32, bool)> {
    let a = dev.htod_sync_copy(&p.a)?;
    let sa = dev.htod_sync_copy(&p.a_scales)?;
    let w = dev.htod_sync_copy(&p.w)?;
    let sw = dev.htod_sync_copy(&p.w_scales)?;
    let shape = GemmShape {
        rows: p.rows,
        outputs: p.outputs,
        inner: p.inner,
    };
    let result = if p.fp4 {
        gpu.fp4_partial_f32(stream, &a, &sa, &w, &sw, shape)?
    } else {
        gpu.fp8_partial_f32(stream, &a, &sa, &w, &sw, shape)?
    };
    Ok((result.status_code()?, result.download_checked().is_err()))
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "oracle accepts no arguments"
    );
    let began = Instant::now();
    let dev = CudaDevice::new(0).context("CUDAdevice0")?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let gpu = GpuGemm::load(&dev, &stream)?;
    let mut checks = Vec::new();
    for (name, outputs, inner, fp4, full) in [
        ("tiny_FP8_tail", 35, 96, false, true),
        ("tiny_native_FP4_tail", 35, 96, true, true),
        ("Q_a_FP8", 1280, 5120, false, false),
        ("Q_b_TP8_FP8", 4096, 1280, false, false),
        ("WO_a_shape_FP8_primitive_only", 1024, 4096, false, false),
        ("WO_b_partial_FP8", 5120, 1024, false, false),
        ("Engram_projection_TP8_FP8", 3200, 6144, false, false),
        ("FFN_gate_native_FP4", 2304, 5120, true, false),
        ("FFN_down_native_FP4", 5120, 2304, true, false),
    ] {
        case(
            &dev,
            &gpu,
            &stream,
            &mut checks,
            name,
            outputs,
            inner,
            fp4,
            full,
        )?;
        ensure!(
            began.elapsed() < Duration::from_secs(WALL_SECONDS),
            "component deadline exceeded"
        );
    }
    // An outer-product basis covers EVERY finite E4M3 byte, not just the bounded
    // dyadic stress family. One scale0 block also checks gradual underflow inFP32.
    let codes: Vec<u8> = (0_u8..=254).filter(|&b| b & 127 != 127).collect();
    for (name, activation_basis) in [
        ("all_finite_FP8_weight_codes", false),
        ("all_finite_FP8_activation_codes_positive", true),
        ("all_finite_FP8_activation_codes_negative", true),
    ] {
        let activation_codes: Vec<u8> = if name.ends_with("negative") {
            codes.iter().copied().filter(|b| b & 128 != 0).collect()
        } else {
            codes.iter().copied().filter(|b| b & 128 == 0).collect()
        };
        let mut p = if activation_basis {
            pack(127, 35, 32, false)
        } else {
            pack(1, codes.len(), 32, false)
        };
        p.a.fill(0);
        p.w.fill(0);
        p.a_scales.fill(127);
        p.w_scales.fill(127);
        if activation_basis {
            for row in 0..127 {
                p.a[row * 32] = activation_codes[row];
            }
            for out in 0..p.outputs {
                p.w[out * 32] = 0x38;
            }
        } else {
            p.a[0] = 0x38;
            for (out, &code) in codes.iter().enumerate() {
                p.w[out * 32] = code;
            }
            p.w_scales[0] = 0;
        }
        let (got, rounded) = device_run(&dev, &gpu, &stream, &p)?;
        let mut wrong = 0;
        let mut round_wrong = 0;
        for row in 0..p.rows {
            for out in 0..p.outputs {
                let r = reference(&p, row, out);
                let i = row * p.outputs + out;
                if got[i].to_bits() != r.f32_value.to_bits() {
                    wrong += 1;
                }
                if rounded[i].to_bits() != bf16::from_f32(r.f32_value).to_bits() {
                    round_wrong += 1;
                }
            }
        }
        record(
            &mut checks,
            name.into(),
            wrong == 0 && round_wrong == 0,
            json!({"rows":p.rows,"outputs":p.outputs,"inner":32,"elements":got.len(),
                "CPU_mismatches":wrong,"BF16_mismatches":round_wrong,"tolerance":"bitwise",
                "UE8M0_scale0_in_weight_basis":!activation_basis}),
        );
    }
    // A second token tile proves that prefill rows do not alias tile0 or leave tails unwritten.
    for fp4 in [false, true] {
        let p = pack(128, 35, 96, fp4);
        let (got, rounded) = device_run(&dev, &gpu, &stream, &p)?;
        let mut wrong = 0;
        let mut round_wrong = 0;
        for row in 0..128 {
            for out in 0..35 {
                let r = reference(&p, row, out);
                let i = row * 35 + out;
                if got[i].to_bits() != r.f32_value.to_bits() {
                    wrong += 1;
                }
                if rounded[i].to_bits() != bf16::from_f32(r.f32_value).to_bits() {
                    round_wrong += 1;
                }
            }
        }
        record(
            &mut checks,
            format!("prefill128_{}", if fp4 { "native_FP4" } else { "FP8" }),
            wrong == 0 && round_wrong == 0,
            json!({"rows":128,"outputs":35,"inner":96,"elements":got.len(),
                "CPU_mismatches":wrong,"BF16_mismatches":round_wrong,"tolerance":"bitwise"}),
        );
    }
    // Group32 geometry must not accept old128-scale buffers even if their values look plausible.
    let p = pack(1, 35, 128, false);
    let a = dev.htod_sync_copy(&p.a)?;
    let sa = dev.htod_sync_copy(&vec![127_u8; 1])?;
    let w = dev.htod_sync_copy(&p.w)?;
    let sw = dev.htod_sync_copy(&p.w_scales)?;
    let wrong = gpu
        .fp8_partial_f32(
            &stream,
            &a,
            &sa,
            &w,
            &sw,
            GemmShape {
                rows: 1,
                outputs: 35,
                inner: 128,
            },
        )
        .is_err();
    record(
        &mut checks,
        "negative_old128_scale_shape_rejected".into(),
        wrong,
        json!({"rejected":wrong}),
    );
    for (name, fp4, kind, expected) in [
        ("negative_FP8_NaN_activation_rejected", false, 0, 1_u32),
        ("negative_FP8_NaN_weight_rejected", false, 1, 1),
        ("negative_native_FP4_NaN_scale_rejected", true, 2, 2),
        ("negative_FP8_NaN_scale_rejected", false, 2, 2),
        ("negative_FP8_FP32_overflow_rejected", false, 3, 4),
        ("negative_native_FP4_FP32_overflow_rejected", true, 3, 4),
    ] {
        let mut q = pack(1, 35, 32, fp4);
        match kind {
            0 => q.a[0] = 0x7f,
            1 => q.w[0] = 0xff,
            2 => q.w_scales[0] = 255,
            _ => {
                q.a.fill(0x44);
                q.w.fill(if fp4 { 0x77 } else { 0x44 });
                q.a_scales.fill(254);
                q.w_scales.fill(254);
            }
        }
        let (status, rejected) = bad_status(&dev, &gpu, &stream, &q)?;
        record(
            &mut checks,
            name.into(),
            status == expected && rejected,
            json!({"status":status,"expected_status":expected,"checked_download_rejected":rejected}),
        );
    }
    let all_passed = checks.iter().all(|x| x["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_native_group32_gemm_oracle",
        "status":if all_passed{"PASS"}else{"FAIL"},"all_passed":all_passed,
        "serve_supported":false,"model_execution":false,"checkpoint_payload_read":false,
        "source_id_hex":format!("{:016x}",gpu.source_id()),"cuda_source_sha256":gpu.cuda_source_sha256(),
        "ptx_sha256":gpu.ptx_sha256(),"kernel_reference_sha256":KERNEL_REF,"model_reference_sha256":MODEL_REF,
        "independent_CPU_preflight_sha256":"2b1a579ff833f82bdbe05cd667e672a122db772347d059c70fd5329d203bdff9",
        "elapsed_seconds":began.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":WALL_SECONDS,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND_BYTES,
            "memory_scope":"synthetic component arrays only; excludes CUDAcontext/moduleoverhead"},
        "quantization":{"activation":"native E4M3+UE8M0 perrowK32","FP8weight":"native E4M3+UE8M0 per32outputrowsK32",
            "FP4weight":"original packed E2M1 loweven/highodd+UE8M0 peroutputrowK32",
            "conversion":"FP4 raw values expand losslessly toE4M3 registers only, perofficialkernel; no tensor conversion"},
        "checks":checks,"reference_schedule_limits":{"WO_a":"shape coverage only; official WO-A dequantizesweights toBF16 and applies BF16 einsum after inplace activation quant-dequant; this oracle does not execute that complete algorithm",
            "rowparallel":"FP32partial output is an explicit alternative precision path; official RowParallelLinear rounds local Linear outputs toBF16 before FP32allreduce. CompleteBF16 API remains available."},
        "meaning":"component mathematical/layout gates only; no fullmodel correctness, speed or quality qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("dsv41 GEMM component exceeded60s");
        std::process::exit(124);
    });
    let value = match run() {
        Ok(v) => v,
        Err(e) => json!({"schema_version":1,"kind":"dsv41_native_group32_gemm_oracle",
        "status":"FAIL","all_passed":false,"serve_supported":false,"model_execution":false,
        "error":format!("{e:#}")}),
    };
    println!("{}", serde_json::to_string(&value).unwrap());
    if value["all_passed"] != true {
        std::process::exit(1);
    }
}
