//! Model-free BF16 attention oracle. Run this executable with no arguments.
//!
//! The checked-in PTX is embedded at compile time. This directly tests the existing kernels
//! and the rebased per-column launch plan, NOT gpu.rs dispatch or end-to-end model quality.
//! A parent process should also impose a 60-second wall limit and exclusive GPU ownership.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PTX: &str = include_str!("../src/ptx/gpu_batch.ptx");
const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const STRIDE: usize = 4096;
const SEG: usize = 48;
const LANES: usize = 8;
const SLOTS: usize = 8;
// Deliberate padding catches the common q_pitch == nh*hd assumption.
const Q_PITCH: usize = NH * HD + 32;
const POS: [i32; LANES] = [99, 102, 86, 128, 170, 511, 1023, 4095];
const SLOT: [i32; LANES] = [0, 5, 2, 7, 1, 6, 3, 4];
const INITIALIZED_TOKENS: usize = STRIDE;
const SCALAR_SMEM: u32 = ((8 * HD + 16) * 4) as u32;
const WALL_SECONDS: u64 = 60;
// Live device arrays: K,V,Q,positions,slots plus the largest one-call scratch/output.
const GPU_BYTES: usize = 2 * SLOTS * NKV * STRIDE * HD * 2
    + LANES * Q_PITCH * 2
    + 2 * LANES * 4
    + LANES * NH * SEG * (HD + 2) * 4
    + LANES * NH * HD * 2;

#[derive(Clone, Copy, Debug)]
enum Mode {
    TensorCore,
    Scalar,
    LegacyHybrid,
}

struct Inputs {
    q_host: Vec<bf16>,
    k_host: Vec<bf16>,
    v_host: Vec<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    pos: CudaSlice<i32>,
    slots: CudaSlice<i32>,
}

fn pointer<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr() as u64
}

fn pack_bs(batch: usize) -> u64 {
    assert!(STRIDE < (1 << 19) && SEG < 64 && batch < 64 && Q_PITCH < (1 << 19));
    STRIDE as u64 | ((SEG as u64) << 19) | ((batch as u64) << 25) | ((Q_PITCH as u64) << 31)
}

fn pack_nh() -> i32 {
    (NKV | (HD << 10) | (NH << 20)) as i32
}

fn kv_index(slot: usize, head: usize, token: usize, d: usize) -> usize {
    (((slot * NKV + head) * STRIDE + token) * HD) + d
}

fn make_inputs(dev: &Arc<CudaDevice>, nonuniform: bool) -> Result<Inputs> {
    let mut q = vec![bf16::from_f32(19.0); LANES * Q_PITCH];
    let mut k = vec![bf16::ZERO; SLOTS * NKV * STRIDE * HD];
    let mut v = vec![bf16::ZERO; k.len()];
    for lane in 0..LANES {
        for head in 0..NH {
            for d in 0..HD {
                let x = if nonuniform {
                    match d {
                        0 => (lane as f32 - 3.5) * 2.0 + head as f32 * 0.0625,
                        1 => (lane + 1) as f32 * 0.125 + (head % 6) as f32 * 0.03125,
                        2 => {
                            if lane % 2 == 0 {
                                0.5
                            } else {
                                -0.75
                            }
                        }
                        _ => 0.0,
                    }
                } else {
                    0.0
                };
                q[lane * Q_PITCH + head * HD + d] = bf16::from_f32(x);
            }
        }
    }
    // All live positions plus future sentinels. Readable future values must stay masked.
    for slot in 0..SLOTS {
        for head in 0..NKV {
            for token in 0..INITIALIZED_TOKENS {
                for d in 0..HD {
                    let index = kv_index(slot, head, token, d);
                    let key = if nonuniform {
                        match d {
                            0 => {
                                ((token % 37) as f32 - 18.0) * 0.25
                                    + slot as f32 * 0.125
                                    + head as f32 * 0.0625
                            }
                            1 => ((token / 37) as f32 - 3.0) * 0.25 + slot as f32 * 0.03125,
                            2 => ((token % 11) as f32 - 5.0) * 0.125,
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };
                    let value = if nonuniform {
                        ((token % 37) as f32 - 18.0) * 0.125
                            + slot as f32 * 0.25
                            + head as f32 * 0.0625
                            + (d % 7) as f32 * 0.03125
                    } else {
                        token as f32 + slot as f32 * 2.0
                    };
                    k[index] = bf16::from_f32(key);
                    v[index] = bf16::from_f32(value);
                }
            }
        }
    }
    Ok(Inputs {
        q: dev.htod_sync_copy(&q)?,
        k: dev.htod_sync_copy(&k)?,
        v: dev.htod_sync_copy(&v)?,
        pos: dev.htod_sync_copy(&POS)?,
        slots: dev.htod_sync_copy(&SLOT)?,
        q_host: q,
        k_host: k,
        v_host: v,
    })
}

// Independent f64 softmax over the exact BF16 values uploaded, not a kernel partition model.
fn cpu_oracle(input: &Inputs, lane: usize) -> Vec<f64> {
    let pc = (POS[lane] + 1) as usize;
    let slot = SLOT[lane] as usize;
    let mut output = vec![0.0; NH * HD];
    for qh in 0..NH {
        let kh = qh / (NH / NKV);
        let mut scores = vec![0.0_f64; pc];
        for (token, score) in scores.iter_mut().enumerate() {
            for d in 0..HD {
                *score += input.q_host[lane * Q_PITCH + qh * HD + d].to_f32() as f64
                    * input.k_host[kv_index(slot, kh, token, d)].to_f32() as f64;
            }
            *score /= (HD as f64).sqrt();
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut denominator = 0.0;
        for score in &mut scores {
            *score = (*score - max).exp();
            denominator += *score;
        }
        for d in 0..HD {
            let mut numerator = 0.0_f64;
            for (token, weight) in scores.iter().enumerate() {
                numerator += weight * input.v_host[kv_index(slot, kh, token, d)].to_f32() as f64;
            }
            output[qh * HD + d] = numerator / denominator;
        }
    }
    output
}

fn launch(
    dev: &Arc<CudaDevice>,
    input: &Inputs,
    p3: bool,
    mode: Mode,
    start: usize,
    width: usize,
) -> Result<Vec<bf16>> {
    ensure!(width > 0 && start + width <= LANES, "invalid column range");
    let parts = width * NH * SEG;
    let pm = dev.alloc_zeros::<f32>(parts)?;
    let pl = dev.alloc_zeros::<f32>(parts)?;
    let pa = dev.alloc_zeros::<f32>(parts * HD)?;
    let out = dev.alloc_zeros::<bf16>(width * NH * HD)?;
    let pos = pointer(&input.pos) + (start * 4) as u64;
    let slots = pointer(&input.slots) + (start * 4) as u64;
    let query = pointer(&input.q) + (start * Q_PITCH * 2) as u64;
    let kernel = if p3 {
        "gqa_attn_verify_e_p3"
    } else {
        "gqa_attn_verify_e"
    };
    if matches!(mode, Mode::TensorCore | Mode::LegacyHybrid) {
        for col in 0..width {
            let partial_bytes = (col * NH * SEG * 4) as u64;
            // Each column is a complete batch-1 launch: Q, positions, physical slots and
            // all partial buffers are rebased together. No slot can be shared implicitly.
            unsafe {
                dev.get_func("gqa_oracle", kernel)
                    .context("missing tensor-core kernel")?
                    .launch(
                        LaunchConfig {
                            grid_dim: (NKV as u32, SEG as u32, 1),
                            block_dim: (192, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            pointer(&pm) + partial_bytes,
                            pointer(&pl) + partial_bytes,
                            pointer(&pa) + partial_bytes * HD as u64,
                            query + (col * Q_PITCH * 2) as u64,
                            pointer(&input.k),
                            pointer(&input.v),
                            pos + (col * 4) as u64,
                            pack_bs(1),
                            pack_nh(),
                            slots + (col * 4) as u64,
                            0_u64,
                        ),
                    )?;
            }
        }
    }
    if matches!(mode, Mode::Scalar | Mode::LegacyHybrid) {
        unsafe {
            dev.get_func("gqa_oracle", "gqa_attn_splitk_gq")
                .context("missing scalar kernel")?
                .launch(
                    LaunchConfig {
                        grid_dim: ((width * NKV * SEG) as u32, 1, 1),
                        block_dim: (HD as u32, 1, 1),
                        shared_mem_bytes: SCALAR_SMEM,
                    },
                    (
                        pointer(&pm),
                        pointer(&pl),
                        pointer(&pa),
                        query,
                        pointer(&input.k),
                        pointer(&input.v),
                        pos,
                        pack_bs(width),
                        pack_nh(),
                        slots,
                        0_u64,
                        0_u64,
                    ),
                )?;
        }
    }
    let reducer = if matches!(mode, Mode::Scalar) {
        "gqa_attn_reduce"
    } else {
        "gqa_attn_reduce_e"
    };
    unsafe {
        dev.get_func("gqa_oracle", reducer)
            .context("missing reducer")?
            .launch(
                LaunchConfig {
                    grid_dim: ((width * NH) as u32, 1, 1),
                    block_dim: (HD as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    pointer(&out),
                    pointer(&pm),
                    pointer(&pl),
                    pointer(&pa),
                    pos,
                    SEG as i32,
                    width as i32,
                    pack_nh(),
                ),
            )?;
    }
    dev.synchronize()?;
    Ok(dev.dtoh_sync_copy(&out)?)
}

fn bf16_ulp(value: f64) -> f64 {
    let rounded = bf16::from_f32(value.abs() as f32);
    let next = bf16::from_bits(rounded.to_bits() + 1);
    (next.to_f32() - rounded.to_f32()) as f64
}

fn compare_cpu(got: &[bf16], reference: &[f64], nonuniform: bool) -> Value {
    assert_eq!(got.len(), reference.len());
    let mut mismatches = 0;
    let mut max_absolute_error = 0.0_f64;
    for (actual, expected) in got.iter().zip(reference) {
        let actual = actual.to_f32() as f64;
        // GVE rounds its softmax P tiles to BF16 before PV. This nonuniform tolerance is
        // predeclared for that approximation; cross-width checks below remain bitwise.
        let tolerance = if nonuniform {
            0.025 + 0.01 * expected.abs()
        } else {
            0.5 * bf16_ulp(*expected) + 1e-5
        };
        let error = (actual - expected).abs();
        if !actual.is_finite() || error > tolerance {
            mismatches += 1;
        }
        if error.is_finite() {
            max_absolute_error = max_absolute_error.max(error);
        }
    }
    json!({"passed":mismatches==0,"elements":got.len(),"mismatches":mismatches,
        "max_absolute_error":max_absolute_error})
}

fn bitwise_mismatches(actual: &[bf16], expected: &[bf16]) -> usize {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn check_pass(value: &Value) -> bool {
    value["passed"].as_bool() == Some(true)
}

fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "this bounded oracle accepts no arguments"
    );
    ensure!(
        GPU_BYTES < 1024 * 1024 * 1024,
        "GPU allocation bound exceeds 1 GiB"
    );
    let started = Instant::now();
    let names = [
        "gqa_attn_verify_e",
        "gqa_attn_verify_e_p3",
        "gqa_attn_splitk_gq",
        "gqa_attn_reduce_e",
        "gqa_attn_reduce",
    ];
    let dev = CudaDevice::new(0).context("CUDA device 0")?;
    dev.load_ptx(Ptx::from_src(PTX), "gqa_oracle", &names)
        .context("embedded PTX load")?;
    let mut variants: Vec<Value> = vec![
        json!({"p3":false,"widths":[1,2,4,8],"checks":[]}),
        json!({"p3":true,"widths":[1,2,4,8],"checks":[]}),
    ];
    let mut anchors = Vec::new();
    for nonuniform in [false, true] {
        let input = make_inputs(&dev, nonuniform)?;
        let references: Vec<Vec<f64>> = (0..LANES).map(|lane| cpu_oracle(&input, lane)).collect();
        if !nonuniform {
            ensure!(
                references[0].iter().all(|&x| x == 49.5),
                "CPU anchor not49.5"
            );
        }
        for (index, p3) in [false, true].into_iter().enumerate() {
            let mut single_e = Vec::new();
            let mut single_scalar = Vec::new();
            for lane in 0..LANES {
                single_e.push(launch(&dev, &input, p3, Mode::TensorCore, lane, 1)?);
                single_scalar.push(launch(&dev, &input, p3, Mode::Scalar, lane, 1)?);
            }
            if !nonuniform {
                let hybrid = launch(&dev, &input, p3, Mode::LegacyHybrid, 0, 1)?;
                // Deliberately reproduce the old dispatch: scalar overwrites segment0 with
                // all100 keys; the still-live E segment1 contributes keys86..99 again.
                let old_reference = ((0..100).sum::<i32>() + (86..100).sum::<i32>()) as f64 / 114.0;
                let old_bf16 = bf16::from_f32(old_reference as f32).to_f32() as f64;
                let true_ref = &references[0];
                let e = compare_cpu(&single_e[0], true_ref, false);
                let scalar = compare_cpu(&single_scalar[0], true_ref, false);
                let incorrect = compare_cpu(&hybrid, true_ref, false);
                let expected_hybrid = vec![old_reference; NH * HD];
                let hybrid_matches_legacy = compare_cpu(&hybrid, &expected_hybrid, false);
                anchors.push(json!({"p3":p3,"reference":49.5,"legacy_reference":old_reference,
                    "legacy_reference_bf16":old_bf16,"pure_e":e,"pure_scalar":scalar,
                    "legacy_against_correct":incorrect,"legacy_matches_expected":hybrid_matches_legacy,
                    "pure_e_value":single_e[0][0].to_f32(),
                    "pure_scalar_value":single_scalar[0][0].to_f32(),
                    "legacy_value":hybrid[0].to_f32(),
                    "pure_e_passed":check_pass(&e),"pure_scalar_passed":check_pass(&scalar),
                    "legacy_negative_rejected":!check_pass(&incorrect)&&check_pass(&hybrid_matches_legacy)}));
            }
            for width in [1_usize, 2, 4, 8] {
                for start in (0..LANES).step_by(width) {
                    let e = launch(&dev, &input, p3, Mode::TensorCore, start, width)?;
                    let scalar = launch(&dev, &input, p3, Mode::Scalar, start, width)?;
                    let expected: Vec<f64> = references[start..start + width]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    let expected_e: Vec<bf16> = single_e[start..start + width]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    let expected_scalar: Vec<bf16> = single_scalar[start..start + width]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    let e_cpu = compare_cpu(&e, &expected, nonuniform);
                    let scalar_cpu = compare_cpu(&scalar, &expected, nonuniform);
                    let e_bitwise = bitwise_mismatches(&e, &expected_e);
                    let scalar_bitwise = bitwise_mismatches(&scalar, &expected_scalar);
                    variants[index]["checks"].as_array_mut().unwrap().push(json!({
                        "case":if nonuniform{"nonuniform"}else{"uniform"},"width":width,
                        "start_lane":start,"positions":&POS[start..start+width],
                        "physical_slots":&SLOT[start..start+width],"pure_e":e_cpu,
                        "pure_scalar":scalar_cpu,"e_cross_width_bitwise_mismatches":e_bitwise,
                        "scalar_cross_width_bitwise_mismatches":scalar_bitwise,
                        "passed":check_pass(&e_cpu)&&check_pass(&scalar_cpu)&&e_bitwise==0&&scalar_bitwise==0}));
                }
            }
        }
    }
    for variant in &mut variants {
        let checks = variant["checks"].as_array().unwrap();
        let uniform = checks
            .iter()
            .filter(|x| x["case"] == "uniform")
            .all(check_pass);
        let nonuniform = checks
            .iter()
            .filter(|x| x["case"] == "nonuniform")
            .all(check_pass);
        let bitwise = checks.iter().all(|x| {
            x["e_cross_width_bitwise_mismatches"] == 0
                && x["scalar_cross_width_bitwise_mismatches"] == 0
        });
        variant["uniform_passed"] = json!(uniform);
        variant["nonuniform_passed"] = json!(nonuniform);
        variant["cross_width_bitwise"] = json!(bitwise);
        variant["all_passed"] = json!(uniform && nonuniform && bitwise);
    }
    let e_passed = anchors.iter().all(|a| a["pure_e_passed"] == true);
    let scalar_passed = anchors.iter().all(|a| a["pure_scalar_passed"] == true);
    let negative = anchors
        .iter()
        .all(|a| a["legacy_negative_rejected"] == true);
    let all_passed =
        e_passed && scalar_passed && negative && variants.iter().all(|x| x["all_passed"] == true);
    Ok(json!({"schema_version":1,"kind":"gqa_bf16_kernel_oracle",
        "status":if all_passed{"PASS"}else{"FAIL"},"all_passed":all_passed,
        "geometry":{"query_heads":NH,"kv_heads":NKV,"head_dim":HD,"stride":STRIDE,
            "segments":SEG,"query_pitch":Q_PITCH,"slots":SLOTS},
        "ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":WALL_SECONDS,"gpu_bytes_upper_bound":GPU_BYTES,
            "gpu_bytes_scope":"explicit arrays; excludes CUDA context/module overhead"},
        "tolerances":{"uniform":"0.5 BF16 ULP + 1e-5 vs independent f64 mean",
            "nonuniform":"0.025 absolute + 0.01*abs(reference) vs independent f64 softmax",
            "cross_width":"bitwise BF16 equality within the same kernel variant/path"},
        "uniform_anchor":{"reference":49.5,"legacy_reference":6245.0/114.0,
            "legacy_reference_bf16":54.75,"pure_e_passed":e_passed,
            "pure_scalar_passed":scalar_passed,"legacy_negative_rejected":negative,
            "checks":anchors},
        "variants":variants,
        "scope":"unmodified embedded BF16 kernels and explicit per-column rebasing; actual core dispatch and model end-to-end behavior require separate tests"}))
}

fn main() {
    // A CUDA hang must not turn a tiny model-free probe into an unbounded cluster job.
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"gqa_bf16_kernel_oracle",
            "status":"FAIL","all_passed":false,"error":"hard wall-time limit",
            "limit_seconds":WALL_SECONDS})
        );
        std::process::exit(124);
    });
    let outcome = std::panic::catch_unwind(run);
    let report = match outcome {
        Ok(Ok(report)) => report,
        Ok(Err(error)) => json!({"schema_version":1,"kind":"gqa_bf16_kernel_oracle",
            "status":"FAIL","all_passed":false,"error":format!("{error:#}")}),
        Err(_) => json!({"schema_version":1,"kind":"gqa_bf16_kernel_oracle",
            "status":"FAIL","all_passed":false,"error":"panic"}),
    };
    println!("{}", report);
    if report["all_passed"] != true {
        std::process::exit(1);
    }
}
