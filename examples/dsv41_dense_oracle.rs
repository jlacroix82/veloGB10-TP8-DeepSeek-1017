//! Model-free1017 dense BF16/F32 and WO-A reconstruction oracle.
//! Synthetic arrays only. Releaseprofile, exclusiveGPU, external60s timeout.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use gb10_inference::dsv41::{
    gpu_dense::{DenseShape, DequantCast, GpuDense},
    gpu_numerics::GpuNumerics,
    numerics,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND_BYTES: usize = 512 * 1024 * 1024;
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
#[derive(Clone, Copy, Debug)]
enum Mode {
    Bf16,
    PromoteBf16,
    F32,
}
enum HostWeights {
    B(Vec<bf16>),
    F(Vec<f32>),
}
impl HostWeights {
    fn value(&self, i: usize) -> f32 {
        match self {
            Self::B(v) => v[i].to_f32(),
            Self::F(v) => v[i],
        }
    }
}
enum DeviceWeights {
    B(CudaSlice<bf16>),
    F(CudaSlice<f32>),
}
struct Pack {
    rows: usize,
    outputs: usize,
    inner: usize,
    mode: Mode,
    a: Vec<f32>,
    w: HostWeights,
}
fn signed(i: usize, salt: usize, span: usize) -> i32 {
    ((i.wrapping_mul(13)
        .wrapping_add((i / 37).wrapping_mul(7))
        .wrapping_add(salt))
        % (2 * span + 1)) as i32
        - span as i32
}
fn pack(rows: usize, outputs: usize, inner: usize, mode: Mode) -> Pack {
    let a = (0..rows * inner)
        .map(|i| match mode {
            Mode::Bf16 => signed(i, 17, 31) as f32 / 16.,
            Mode::PromoteBf16 => (signed(i, 11, 384) as f32 * 8. + 1.) / 8192.,
            Mode::F32 => signed(i, 11, 1023) as f32 / 1024.,
        })
        .collect();
    let w = match mode {
        Mode::Bf16 => HostWeights::B(
            (0..outputs * inner)
                .map(|i| bf16::from_f32(signed(i, 29, 15) as f32 / 16.))
                .collect(),
        ),
        Mode::PromoteBf16 => HostWeights::B(
            (0..outputs * inner)
                .map(|i| bf16::from_f32(signed(i, 29, 1) as f32 / 16.))
                .collect(),
        ),
        Mode::F32 => HostWeights::F(
            (0..outputs * inner)
                .map(|i| signed(i, 29, 4095) as f32 / 4096.)
                .collect(),
        ),
    };
    Pack {
        rows,
        outputs,
        inner,
        mode,
        a,
        w,
    }
}
fn reference(p: &Pack, row: usize, out: usize, narrow: bool, tf32: bool) -> (f32, f64) {
    let mut sum = 0_f32;
    let mut exact = 0_f64;
    for k in 0..p.inner {
        let mut a = p.a[row * p.inner + k];
        let mut w = p.w.value(out * p.inner + k);
        if narrow {
            a = bf16::from_f32(a).to_f32();
            w = bf16::from_f32(w).to_f32();
        }
        if tf32 {
            let round = |x: f32| {
                f32::from_bits(
                    x.to_bits().wrapping_add(0xfff + ((x.to_bits() >> 13) & 1)) & !0x1fff,
                )
            };
            a = round(a);
            w = round(w);
        }
        sum += a * w;
        exact += a as f64 * w as f64;
    }
    (sum, exact)
}
fn upload(dev: &Arc<CudaDevice>, w: &HostWeights) -> Result<DeviceWeights> {
    Ok(match w {
        HostWeights::B(v) => DeviceWeights::B(dev.htod_sync_copy(v)?),
        HostWeights::F(v) => DeviceWeights::F(dev.htod_sync_copy(v)?),
    })
}
fn device_run(
    dev: &Arc<CudaDevice>,
    gpu: &GpuDense,
    s: &CudaStream,
    p: &Pack,
    dw: &DeviceWeights,
    order: &[usize],
) -> Result<(Vec<f32>, Vec<bf16>)> {
    let a: Vec<f32> = order
        .iter()
        .flat_map(|&r| p.a[r * p.inner..(r + 1) * p.inner].iter().copied())
        .collect();
    let g = DenseShape {
        rows: order.len(),
        outputs: p.outputs,
        inner: p.inner,
    };
    match (p.mode, dw) {
        (Mode::Bf16, DeviceWeights::B(w)) => {
            let a = dev.htod_sync_copy(&a.into_iter().map(bf16::from_f32).collect::<Vec<_>>())?;
            Ok((
                gpu.bf16_linear_f32(s, &a, w, g)?.download_checked()?,
                gpu.bf16_linear_bf16(s, &a, w, g)?.download_checked()?,
            ))
        }
        (Mode::PromoteBf16, DeviceWeights::B(w)) => {
            let a = dev.htod_sync_copy(&a)?;
            Ok((
                gpu.f32_bf16_linear_f32(s, &a, w, g)?.download_checked()?,
                gpu.f32_bf16_linear_bf16(s, &a, w, g)?.download_checked()?,
            ))
        }
        (Mode::F32, DeviceWeights::F(w)) => {
            let a = dev.htod_sync_copy(&a)?;
            Ok((
                gpu.f32_linear_f32(s, &a, w, g)?.download_checked()?,
                gpu.f32_linear_bf16(s, &a, w, g)?.download_checked()?,
            ))
        }
        _ => anyhow::bail!("invalid oracle dtypecombination"),
    }
}
fn probes(n: usize, full: bool) -> Vec<usize> {
    if full {
        return (0..n).collect();
    }
    let mut p = vec![
        0,
        1,
        3,
        7,
        15,
        16,
        31,
        32,
        63,
        127,
        128,
        255,
        n / 2,
        n - 2,
        n - 1,
    ];
    p.retain(|&i| i < n);
    p.sort_unstable();
    p.dedup();
    p
}
fn record(checks: &mut Vec<Value>, name: String, passed: bool, details: Value) {
    checks.push(json!({"name":name,"passed":passed,"details":details}));
}
fn case(
    dev: &Arc<CudaDevice>,
    gpu: &GpuDense,
    s: &CudaStream,
    checks: &mut Vec<Value>,
    name: &str,
    p: Pack,
    full: bool,
) -> Result<()> {
    let dw = upload(dev, &p.w)?;
    let (baseline, rounded) = device_run(dev, gpu, s, &p, &dw, &(0..16).collect::<Vec<_>>())?;
    let selected = probes(p.outputs, full);
    let mut cpu_wrong = 0;
    let mut f64_wrong = 0;
    let mut max_error = 0_f64;
    let mut premature = 0;
    let mut narrowing = 0;
    let mut tf32wrong = 0;
    let mut h = Sha256::new();
    for row in 0..16 {
        for &out in &selected {
            let (wanted, exact) = reference(&p, row, out, false, false);
            let got = baseline[row * p.outputs + out];
            if got.to_bits() != wanted.to_bits() {
                cpu_wrong += 1;
            }
            if wanted.to_bits() != (exact as f32).to_bits() {
                f64_wrong += 1;
            }
            max_error = max_error.max((got as f64 - exact).abs());
            h.update(wanted.to_le_bytes());
            if full {
                if reference(&p, row, out, true, false).0.to_bits() != wanted.to_bits() {
                    narrowing += 1;
                }
                if reference(&p, row, out, false, true).0.to_bits() != wanted.to_bits() {
                    tf32wrong += 1;
                }
            }
        }
    }
    let cpu_reference_sha = format!("{:x}", h.finalize());
    // Independent NumPy/TorchCPU preflight, not generated by this Rust reference.
    let expected_cpu_hash = match name {
        "tiny_BF16_tail" => {
            Some("92cc7d8ffd4b277bef609e6d4060fdc9adbbcb13e0c9f94f592e7deb1bbcdb2c")
        }
        "tiny_F32_stored_BF16" => {
            Some("4bddaf74d326d1a12927ec00792442a0c44aeb85161ca5f83590eebb35d0c42d")
        }
        "tiny_F32_weights" => {
            Some("973b2a5895a750286123050ca444a5b0c7eebad2101cc58c6d9d35315075bb8e")
        }
        "head_TP8_F32" => Some("951af0cb8cf47354dc6f305cc696a32d6d4e4fbfc5c03c140fecb5036339d821"),
        "WO_A_BF16_QAT_projection" => {
            Some("cb8ce74283bbe66f06741b72fefd6c8fe3e6485735c5da41fa907b799ae6e1c9")
        }
        _ => None,
    };
    let independent_cpu_golden_matches = expected_cpu_hash.is_none_or(|h| h == cpu_reference_sha);
    let round_wrong = baseline
        .iter()
        .zip(&rounded)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    for &a in &baseline {
        if bf16::from_f32(a).to_f32() != a {
            premature += 1;
        }
    }
    record(
        checks,
        format!("{name}_CPU_reference"),
        cpu_wrong == 0 && f64_wrong == 0 && round_wrong == 0 && independent_cpu_golden_matches,
        json!({"rows":16,"outputs":p.outputs,"inner":p.inner,"arithmetic":format!("{:?}",p.mode),
            "CPU_output_indices":selected,"CPU_outputs_checked":16*selected.len(),
            "CPU_mismatches":cpu_wrong,"CPU_FP32_vs_FP64_mismatches":f64_wrong,"max_absolute_FP64_error":max_error,
            "BF16_round_mismatches":round_wrong,"CPU_reference_sha256":cpu_reference_sha,
            "independent_CPU_golden_applies":expected_cpu_hash.is_some(),"independent_CPU_golden_matches":independent_cpu_golden_matches,
            "non_BF16_FP32_outputs":premature,"tolerance":"bitwise, boundedexactdyadicfamily"}),
    );
    for width in WIDTHS {
        let (got, b) = device_run(dev, gpu, s, &p, &dw, &ORDER[..width])?;
        let mut wrong = 0;
        let mut bw = 0;
        let mut cw = 0;
        for row in 0..width {
            for out in 0..p.outputs {
                let i = row * p.outputs + out;
                let j = ORDER[row] * p.outputs + out;
                if got[i].to_bits() != baseline[j].to_bits() {
                    wrong += 1;
                }
                if b[i].to_bits() != rounded[j].to_bits()
                    || b[i].to_bits() != bf16::from_f32(got[i]).to_bits()
                {
                    bw += 1;
                }
            }
            for &out in &selected {
                if got[row * p.outputs + out].to_bits()
                    != reference(&p, ORDER[row], out, false, false).0.to_bits()
                {
                    cw += 1;
                }
            }
        }
        record(
            checks,
            format!("{name}_permuted_C{width}"),
            wrong == 0 && bw == 0 && cw == 0,
            json!({"width":width,"outputs":p.outputs,"inner":p.inner,"rows_from_reference":&ORDER[..width],
                "elements_compared":got.len(),"FP32_mismatches":wrong,"BF16_mismatches":bw,
                "CPU_mismatches":cw,"CPU_outputs_checked":width*selected.len(),"tolerance":"bitwise"}),
        );
    }
    if full {
        let requires_narrow = !matches!(p.mode, Mode::Bf16);
        record(
            checks,
            format!("{name}_negative_precision_narrowing"),
            premature > 0 && (!requires_narrow || (narrowing > 0 && tf32wrong > 0)),
            json!({"premature_BF16_partial_differences":premature,"BF16_operand_narrowing_differences":narrowing,
                "TF32_operand_narrowing_differences":tf32wrong,"operand_narrowing_applies":requires_narrow}),
        );
    }
    Ok(())
}
fn woa(
    dev: &Arc<CudaDevice>,
    gpu: &GpuDense,
    s: &CudaStream,
    checks: &mut Vec<Value>,
) -> Result<()> {
    let rows = 16;
    let outputs = 1024;
    let inner = 4096;
    let groups = inner / 32;
    let original: Vec<bf16> = (0..rows * inner)
        .map(|i| bf16::from_f32(signed(i, 3, 18) as f32 / 16. + 1. / 128.))
        .collect();
    let cpu = numerics::quantize_activations_bf16(&original, rows, inner)?;
    let cpu_a: Vec<bf16> = numerics::dequantize_activations_f32(&cpu)?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let codebook = [0_u8, 0x30, 0x38];
    let codes: Vec<u8> = (0..outputs * inner)
        .map(|i| {
            let v = codebook[(i * 7 + i / 31) % 3];
            v | if i % 7 < 3 && v != 0 { 128 } else { 0 }
        })
        .collect();
    let scales: Vec<u8> = (0..outputs.div_ceil(32) * groups)
        .map(|i| 125 + (i % 3) as u8)
        .collect();
    let cpu_w: Vec<bf16> = (0..outputs * inner)
        .map(|i| {
            let row = i / inner;
            let col = i % inner;
            bf16::from_f32(
                numerics::decode_e4m3_finite(codes[i]).unwrap()
                    * numerics::decode_ue8m0(scales[(row / 32) * groups + col / 32]).unwrap(),
            )
        })
        .collect();
    let numerics = GpuNumerics::load(dev, s)?;
    let input = dev.htod_sync_copy(&original)?;
    let native = numerics.quantize_activations_bf16(s, &input, rows, inner)?;
    let native_cpu = native.download_checked()?;
    let a = gpu
        .quantized_activations_to_bf16(
            s,
            unsafe { native.device_values_unchecked() },
            unsafe { native.device_scales_unchecked() },
            rows,
            inner,
            DequantCast::RequireExact,
        )?
        .download_checked()?;
    let wc = dev.htod_sync_copy(&codes)?;
    let ws = dev.htod_sync_copy(&scales)?;
    let w = gpu
        .fp8_weights_to_bf16(s, &wc, &ws, outputs, inner, DequantCast::RequireExact)?
        .download_checked()?;
    let a_wrong = a
        .iter()
        .zip(&cpu_a)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let w_wrong = w
        .iter()
        .zip(&cpu_w)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let changed = a
        .iter()
        .zip(&original)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    record(
        checks,
        "WO_A_native_reconstruction_and_QAT".into(),
        native_cpu.e4m3_values == cpu.e4m3_values
            && native_cpu.ue8m0_scales == cpu.ue8m0_scales
            && a_wrong == 0
            && w_wrong == 0
            && changed > 0,
        json!({"rows":rows,"outputs":outputs,"inner":inner,"activation_elements":a.len(),"weight_elements":w.len(),
            "activation_reconstruction_mismatches":a_wrong,"weight_reconstruction_mismatches":w_wrong,
            "QAT_changed_BF16_inputs":changed,"native_quantized_bytes_match":native_cpu==cpu,
            "exact_BF16_cast_required":true,"tolerance":"bitwise"}),
    );
    drop((input, native, wc, ws));
    case(
        dev,
        gpu,
        s,
        checks,
        "WO_A_BF16_QAT_projection",
        Pack {
            rows,
            outputs,
            inner,
            mode: Mode::Bf16,
            a: a.into_iter().map(bf16::to_f32).collect(),
            w: HostWeights::B(w),
        },
        false,
    )
}
fn status_test(
    dev: &Arc<CudaDevice>,
    gpu: &GpuDense,
    s: &CudaStream,
    checks: &mut Vec<Value>,
) -> Result<()> {
    let shape = DenseShape {
        rows: 1,
        outputs: 1,
        inner: 32,
    };
    let b = dev.htod_sync_copy(&[bf16::ONE; 32])?;
    for (name, values, wanted) in [
        ("negative_F32_NaN_input", vec![f32::NAN; 32], 1_u32),
        ("negative_F32_finite_overflow", vec![f32::MAX; 32], 4_u32),
    ] {
        let a = dev.htod_sync_copy(&values)?;
        let result = gpu.f32_bf16_linear_f32(s, &a, &b, shape)?;
        let status = result.status_code()?;
        let rejected = result.download_checked().is_err();
        record(
            checks,
            name.into(),
            status == wanted && rejected,
            json!({"status":status,"expected_status":wanted,"checked_download_rejected":rejected}),
        );
    }
    let a = dev.htod_sync_copy(&[bf16::MAX; 32])?;
    let result = gpu.bf16_linear_bf16(s, &a, &b, shape)?;
    let code = result.status_code()?;
    let rejected = result.download_checked().is_err();
    record(
        checks,
        "negative_BF16_output_overflow".into(),
        code == 4 && rejected,
        json!({"status":code,"expected_status":4,"checked_download_rejected":rejected}),
    );
    for (name, code, scale, policy, wanted) in [
        (
            "negative_FP8_NaN_reconstruction",
            0x7f,
            127,
            DequantCast::RequireExact,
            1,
        ),
        (
            "negative_UE8_NaN_reconstruction",
            0x38,
            255,
            DequantCast::RequireExact,
            2,
        ),
        (
            "negative_FP8_reconstruction_overflow",
            0x7e,
            254,
            DequantCast::RequireExact,
            4,
        ),
        (
            "negative_lossy_BF16_cast_rejected",
            0x01,
            0,
            DequantCast::RequireExact,
            8,
        ),
    ] {
        let c = dev.htod_sync_copy(&vec![code; 32])?;
        let scale = dev.htod_sync_copy(&[scale])?;
        let result = gpu.fp8_weights_to_bf16(s, &c, &scale, 1, 32, policy)?;
        let status = result.status_code()?;
        let rejected = result.download_checked().is_err();
        record(
            checks,
            name.into(),
            status == wanted && rejected,
            json!({"status":status,"expected_status":wanted,"checked_download_rejected":rejected}),
        );
    }
    let codes = dev.htod_sync_copy(&vec![0x01_u8; 32])?;
    let scales = dev.htod_sync_copy(&[0_u8])?;
    let round = gpu
        .fp8_weights_to_bf16(s, &codes, &scales, 1, 32, DequantCast::OfficialBf16Round)?
        .download_checked()?;
    record(
        checks,
        "explicit_official_BF16_round_policy".into(),
        round == vec![bf16::ZERO; 32],
        json!({"elements":round.len(),"all_zero_after_source_BF16_round":round==vec![bf16::ZERO;32],"precision_policy":"explicitOfficialBf16Round"}),
    );
    let nonblocking = dev.fork_default_stream()?;
    let wrong = gpu.bf16_linear_f32(&nonblocking, &b, &b, shape).is_err();
    record(
        checks,
        "negative_nonblocking_stream_rejected".into(),
        wrong,
        json!({"rejected":wrong}),
    );
    let a = dev.htod_sync_copy(&[bf16::ONE; 31])?;
    let wrong = gpu.bf16_linear_f32(s, &a, &b, shape).is_err();
    record(
        checks,
        "negative_wrong_shape_rejected".into(),
        wrong,
        json!({"rejected":wrong}),
    );
    Ok(())
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "oracle accepts zeroarguments"
    );
    let began = Instant::now();
    let dev = CudaDevice::new(0).context("CUDAdevice0")?;
    let s = gb10_inference::gpu::fork_blocking_stream(&dev);
    let gpu = GpuDense::load(&dev, &s)?;
    let mut checks = Vec::new();
    for (name, n, k, mode, full) in [
        ("tiny_BF16_tail", 35, 37, Mode::Bf16, true),
        ("tiny_F32_stored_BF16", 35, 37, Mode::PromoteBf16, true),
        ("tiny_F32_weights", 35, 3, Mode::F32, true),
        ("ratio1_compressor_BF16", 512, 5120, Mode::Bf16, false),
        ("index_key_BF16", 128, 512, Mode::Bf16, false),
        ("index_weights_TP8_BF16", 4, 5120, Mode::Bf16, false),
        ("ratio2_compressor_F32", 512, 5120, Mode::PromoteBf16, false),
        ("routing_gate_F32", 384, 5120, Mode::PromoteBf16, false),
        ("head_TP8_F32", 16160, 5120, Mode::PromoteBf16, false),
    ] {
        case(
            &dev,
            &gpu,
            &s,
            &mut checks,
            name,
            pack(16, n, k, mode),
            full,
        )?;
        ensure!(
            began.elapsed() < Duration::from_secs(WALL_SECONDS),
            "dense deadline exceeded"
        );
    }
    woa(&dev, &gpu, &s, &mut checks)?;
    // Token tiles after first16 must use independent source rows.
    for mode in [Mode::Bf16, Mode::PromoteBf16] {
        let p = pack(128, 35, 37, mode);
        let w = upload(&dev, &p.w)?;
        let (got, b) = device_run(&dev, &gpu, &s, &p, &w, &(0..128).collect::<Vec<_>>())?;
        let mut wrong = 0;
        let mut bw = 0;
        for row in 0..128 {
            for out in 0..35 {
                let want = reference(&p, row, out, false, false).0;
                let i = row * 35 + out;
                if got[i].to_bits() != want.to_bits() {
                    wrong += 1;
                }
                if b[i].to_bits() != bf16::from_f32(want).to_bits() {
                    bw += 1;
                }
            }
        }
        record(
            &mut checks,
            format!("prefill128_{mode:?}"),
            wrong == 0 && bw == 0,
            json!({"rows":128,"outputs":35,"inner":37,"elements":got.len(),"CPU_mismatches":wrong,"BF16_mismatches":bw,"tolerance":"bitwise"}),
        );
    }
    status_test(&dev, &gpu, &s, &mut checks)?;
    ensure!(
        began.elapsed() < Duration::from_secs(WALL_SECONDS),
        "dense deadline exceeded"
    );
    let passed = checks.iter().all(|c| c["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_dense_component_oracle","status":if passed{"PASS"}else{"FAIL"},
        "all_passed":passed,"serve_supported":false,"model_execution":false,"checkpoint_payload_read":false,
        "source_id_hex":format!("{:016x}",gpu.source_id()),"cuda_source_sha256":gpu.cuda_source_sha256(),"ptx_sha256":gpu.ptx_sha256(),
        "numerics_source_id_hex":format!("{:016x}",GpuNumerics::load(&dev,&s)?.source_id()),
        "numerics_ptx_sha256":GpuNumerics::load(&dev,&s)?.ptx_sha256(),
        "independent_CPU_preflight_sha256":"4764f53da712a02f2d171d0476381afa92f6607ca238777a7467fb54e649c362",
        "elapsed_seconds":began.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":WALL_SECONDS,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND_BYTES,
            "scope":"syntheticBF16 head shard plus componentarrays only; excludesCUDAcontext/moduleoverhead"},
        "arithmetic":{"BF16":"native m16n8k16 BF16×BF16/F32accumulation",
            "F32":"F32multiply+add fixedwarpKtree; originalBF16weights promotedexactly inregisters, ornativeF32weights",
            "forbidden_downgrades":["TF32","FP16","F32operand_toBF16"],
            "source_dispatch":"Gate/head/ratio2compressor explicitF32; ratio1compressor/indexwk/WO-A genuineBF16",
            "WO_A":"group32QATactivation reconstruction + originalFP8weightBF16dequant +pergroupBF16projection; unrotation/attention outside scope"},
        "checks":checks,"scope":"model-free dense/conversion primitiveoracles only; WO-A stages compare audited downloads/reuploads, not fused attention; no payloadreads, fullattention/MoE, TP8engine, speed orquality claim"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("dense oracle hard60s deadline");
        std::process::exit(124);
    });
    let value = match run() {
        Ok(v) => v,
        Err(e) => json!({"schema_version":1,"kind":"dsv41_dense_component_oracle",
        "status":"FAIL","all_passed":false,"serve_supported":false,"model_execution":false,"error":format!("{e:#}")}),
    };
    println!("{}", serde_json::to_string(&value).unwrap());
    if value["all_passed"] != true {
        std::process::exit(1);
    }
}
