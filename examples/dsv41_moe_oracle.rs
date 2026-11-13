//! Model-free1017 routing/SwiGLU GPU oracle; no learned experts or dense linears.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    gpu_moe::{MoeKernels, PTX},
    moe::{self, Routing},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_moe_vectors.json");
const WALL: u64 = 60;
const ARRAY_BOUND: usize = 16 * 1024 * 1024;
const WIDTHS: [usize; 6] = [1, 2, 4, 8, 16, 128];
fn fs(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn bs(v: &Value, k: &str) -> Vec<bf16> {
    v[k].as_array()
        .unwrap()
        .iter()
        .map(|x| bf16::from_bits(x.as_u64().unwrap() as u16))
        .collect()
}
fn ordered(x: bf16) -> i32 {
    let b = x.to_bits() as i32;
    if b & 0x8000 != 0 {
        0x8000 - (b & 0x7fff)
    } else {
        0x8000 + b
    }
}
fn compare(a: &[bf16], b: &[bf16], allowed: u32) -> Value {
    let mut mismatches = usize::from(a.len() != b.len());
    let mut max = 0_u32;
    for (a, b) in a.iter().zip(b) {
        let d = ordered(*a).abs_diff(ordered(*b));
        max = max.max(d);
        if !a.is_finite() || !b.is_finite() || d > allowed {
            mismatches += 1;
        }
    }
    json!({"elements":a.len(),"mismatches":mismatches,"max_bf16_ulp":max,"allowed_bf16_ulp":allowed,"passed":mismatches==0})
}
fn checked(dev: &Arc<CudaDevice>, s: &cudarc::driver::CudaSlice<u32>) -> Result<()> {
    dev.synchronize()?;
    ensure!(
        dev.dtoh_sync_copy(s)? == [0],
        "sticky MoE GPU rejection status"
    );
    Ok(())
}
fn gpu_route(
    dev: &Arc<CudaDevice>,
    k: &MoeKernels,
    s: &CudaStream,
    l: &[f32],
    b: &[f32],
    rows: usize,
) -> Result<Routing> {
    let dl = dev.htod_sync_copy(l)?;
    let db = dev.htod_sync_copy(b)?;
    let mut ids = dev.alloc_zeros::<u32>(rows * 6)?;
    let mut w = dev.alloc_zeros::<f32>(rows * 6)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    k.route(s, &dl, &db, &mut ids, &mut w, &mut flag, rows)?;
    checked(dev, &flag)?;
    Ok(Routing {
        indices: dev.dtoh_sync_copy(&ids)?,
        weights: dev.dtoh_sync_copy(&w)?,
    })
}
#[allow(clippy::too_many_arguments)]
fn gpu_swiglu(
    dev: &Arc<CudaDevice>,
    k: &MoeKernels,
    s: &CudaStream,
    g: &[bf16],
    u: &[bf16],
    w: Option<&[f32]>,
    rows: usize,
    columns: usize,
) -> Result<Vec<bf16>> {
    let dg = dev.htod_sync_copy(g)?;
    let du = dev.htod_sync_copy(u)?;
    let ones = vec![1.0_f32; rows];
    let dw = dev.htod_sync_copy(w.unwrap_or(&ones))?;
    let mut out = dev.alloc_zeros::<bf16>(g.len())?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    k.swiglu(
        s,
        &dg,
        &du,
        &dw,
        w.is_some(),
        &mut out,
        &mut flag,
        rows,
        columns,
    )?;
    checked(dev, &flag)?;
    Ok(dev.dtoh_sync_copy(&out)?)
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "standalone oracle accepts zero arguments"
    );
    let began = Instant::now();
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    ensure!(
        fixture["kind"] == "official1017_MoE_CPU_AST_goldens"
            && fixture["route_cases"].as_array().unwrap().len() == 6
            && fixture["swiglu_cases"].as_array().unwrap().len() == 5,
        "exact official AST fixture coverage"
    );
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let kernels = MoeKernels::load(&dev, &stream)?;
    let cuda_reference = moe::official_cuda_reference()?;
    let mut route_checks = Vec::new();
    let mut bias_negative = false;
    let mut cpu_vector_known_mismatches = 0;
    for c in fixture["route_cases"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        let l = fs(c, "logits");
        let b = fs(c, "bias");
        let reference = cuda_reference["route_cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == c["name"])
            .unwrap();
        let canonical = Routing {
            indices: reference["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
            weights: fs(reference, "weights"),
        };
        ensure!(
            reference["rows"] == c["rows"],
            "CUDA reference row identity"
        );
        moe::validate_witness(&l, &b, rows, &canonical)?;
        let cpu_original = Routing {
            indices: c["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
            weights: fs(c, "weights"),
        };
        let cpu_vector_known_mismatch = moe::validate_witness(&l, &b, rows, &cpu_original).is_err();
        ensure!(
            cpu_vector_known_mismatch == (c["name"] == "tiny_score_eps"),
            "CPU vector diagnostic classification changed"
        );
        cpu_vector_known_mismatches += usize::from(cpu_vector_known_mismatch);
        let actual = gpu_route(&dev, &kernels, &stream, &l, &b, rows)?;
        let witness = moe::validate_witness(&l, &b, rows, &actual)?;
        let cpu = moe::route(&l, &b, rows)?;
        ensure!(
            actual.indices == cpu.indices,
            "explicit lowest-ID tie policy differs"
        );
        let unambiguous = c["tie_policy"] == "unambiguous exact expert IDs";
        let mut error = 0_f32;
        if unambiguous {
            ensure!(
                actual.indices == canonical.indices,
                "unambiguous official CUDA expert IDs differ"
            );
        }
        for (a, b) in actual.weights.iter().zip(&canonical.weights) {
            error = error.max((a - b).abs());
            ensure!(
                (a - b).abs() <= 2e-6 + 2e-6 * b.abs(),
                "official CUDA routing weight differs"
            );
        }
        let cpu_vector_difference = actual
            .weights
            .iter()
            .zip(&cpu_original.weights)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        if cpu_vector_known_mismatch {
            ensure!(
                cpu_vector_difference > 0.0004,
                "known CPU vector mismatch canary disappeared"
            );
        }
        if c["name"] == "selection_bias" {
            let mut wrong = actual.clone();
            for row in 0..rows {
                let mut selected = Vec::new();
                for id in &actual.indices[row * 6..(row + 1) * 6] {
                    selected
                        .push(moe::unbiased_score(l[row * 384 + *id as usize])? + b[*id as usize]);
                }
                let sum = selected.iter().sum::<f32>();
                for i in 0..6 {
                    wrong.weights[row * 6 + i] = (selected[i] / (sum + 1e-20)) * 1.5;
                }
            }
            bias_negative = moe::validate_witness(&l, &b, rows, &wrong).is_err();
            ensure!(bias_negative, "biased-output-weight control escaped");
        }
        route_checks.push(json!({"case":c["name"],"rows":rows,"elements":rows*6,"passed":true,"unambiguous_exact_IDs_checked":unambiguous,"valid_topk_witness":true,"explicit_tie_policy_matches":true,"boundary_ties":witness.boundary_ties,"minimum_selection_margin":witness.minimum_selection_margin,"max_witness_weight_error":witness.max_weight_error,"max_CUDA_weight_error":error,"CPU_vector_known_mismatch_canary":cpu_vector_known_mismatch,"CPU_vector_max_weight_difference":cpu_vector_difference,"reference_backend":"official_TorchCUDA","indices":actual.indices,"weights":actual.weights}));
    }
    ensure!(
        cpu_vector_known_mismatches == 1,
        "exactly one retained CPU vector mismatch canary required"
    );
    let mut swiglu_checks = Vec::new();
    let mut order_negative = false;
    let mut clamp_negative = false;
    let mut up_negative = false;
    for c in fixture["swiglu_cases"].as_array().unwrap() {
        let rows = c["rows"].as_u64().unwrap() as usize;
        let columns = c["columns"].as_u64().unwrap() as usize;
        let g = bs(c, "gate_bits");
        let u = bs(c, "up_bits");
        let w = (!c["weights"].is_null()).then(|| fs(c, "weights"));
        let actual = gpu_swiglu(&dev, &kernels, &stream, &g, &u, w.as_deref(), rows, columns)?;
        // Exact source goldens prevent a <=1ULP allowance from hiding premature BF16 rounding.
        let comparison = compare(&actual, &bs(c, "output_bits"), 0);
        ensure!(
            comparison["passed"] == true,
            "official elementwise BF16 source golden differs: {} {comparison}",
            c["name"]
        );
        if c["name"] == "weight_before_BF16_round" {
            order_negative = c["negative_early_BF16_mismatches"].as_u64().unwrap() > 0;
            ensure!(order_negative, "earlyBF16 negative corpus empty");
        }
        if c["name"] == "negative_gate_unclamped" {
            clamp_negative = c["negative_symmetric_gate_mismatches"].as_u64().unwrap() > 0;
            ensure!(clamp_negative, "symmetric-gate negative corpus empty");
        }
        if c["name"] == "clamp_and_signs" {
            up_negative = c["negative_missing_up_clamp_mismatches"].as_u64().unwrap() > 0;
            ensure!(up_negative, "up-clamp negative corpus empty");
        }
        swiglu_checks.push(json!({"case":c["name"],"rows":rows,"columns":columns,"weighted":w.is_some(),"passed":true,"comparison":comparison,"shared_uses_unweighted_expert":w.is_none(),"early_BF16_negative_mismatches":c["negative_early_BF16_mismatches"],"symmetric_gate_negative_mismatches":c["negative_symmetric_gate_mismatches"],"missing_up_clamp_negative_mismatches":c["negative_missing_up_clamp_mismatches"]}));
    }
    let mut width_checks = Vec::new();
    let mut results = Vec::new();
    for rows in WIDTHS {
        let columns = 2304;
        let l = (0..rows * 384)
            .map(|i| ((i * 37 + 11) % 769) as f32 * 0.0625 - 24.0)
            .collect::<Vec<_>>();
        let b = (0..384)
            .map(|i| if i < 12 { i as f32 * 8.0 + 10.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let r = gpu_route(&dev, &kernels, &stream, &l, &b, rows)?;
        let witness = moe::validate_witness(&l, &b, rows, &r)?;
        let g = (0..rows * columns)
            .map(|i| bf16::from_f32(((i * 7 + 3) % 47) as f32 * 0.75 - 17.25))
            .collect::<Vec<_>>();
        let u = (0..rows * columns)
            .map(|i| bf16::from_f32(((i * 13 + 5) % 53) as f32 * 0.625 - 16.25))
            .collect::<Vec<_>>();
        let weights = (0..rows)
            .map(|row| 0.2 + (row % 7) as f32 * 0.12345)
            .collect::<Vec<_>>();
        let actual = gpu_swiglu(
            &dev,
            &kernels,
            &stream,
            &g,
            &u,
            Some(&weights),
            rows,
            columns,
        )?;
        let comparison = compare(
            &actual,
            &moe::swiglu(&g, &u, Some(&weights), rows, columns)?,
            1,
        );
        ensure!(
            comparison["passed"] == true,
            "hidden2304 CPUreference differs"
        );
        width_checks.push(json!({"width":rows,"columns":columns,"route_elements":rows*6,"passed":true,"valid_topk_witness":true,"max_weight_error":witness.max_weight_error,"swiglu":comparison}));
        results.push((r, actual));
    }
    let widest = results.last().unwrap();
    for (i, rows) in WIDTHS.iter().enumerate() {
        let (r, a) = &results[i];
        let same = r.indices == widest.0.indices[..rows * 6]
            && r.weights
                .iter()
                .zip(&widest.0.weights[..rows * 6])
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && a.iter()
                .zip(&widest.1[..rows * 2304])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        ensure!(same, "cross-width row math differs");
        width_checks[i]["cross_width_bitwise"] = json!(same);
    }
    let mut negative = serde_json::Map::new();
    negative.insert(
        "selection_bias_weight_rejected".into(),
        json!(bias_negative),
    );
    negative.insert("early_BF16_rejected".into(), json!(order_negative));
    negative.insert(
        "symmetric_gate_clamp_rejected".into(),
        json!(clamp_negative),
    );
    negative.insert("missing_up_clamp_rejected".into(), json!(up_negative));
    for (name, nan_l) in [("NaN_logit_status", true), ("NaN_bias_status", false)] {
        let mut l = vec![0_f32; 384];
        let mut b = vec![0_f32; 384];
        if nan_l {
            l[0] = f32::NAN;
        } else {
            b[0] = f32::NAN;
        }
        let dl = dev.htod_sync_copy(&l)?;
        let db = dev.htod_sync_copy(&b)?;
        let mut ids = dev.alloc_zeros::<u32>(6)?;
        let mut weights = dev.alloc_zeros::<f32>(6)?;
        let mut flag = dev.alloc_zeros::<u32>(1)?;
        kernels.route(&stream, &dl, &db, &mut ids, &mut weights, &mut flag, 1)?;
        dev.synchronize()?;
        let code = dev.dtoh_sync_copy(&flag)?[0];
        ensure!(code & 1 != 0, "routing NaN escaped");
        negative.insert(name.into(), json!(code));
    }
    for (name, kind) in [
        ("NaN_gate_status", 0),
        ("NaN_up_status", 1),
        ("NaN_weight_status", 2),
        ("oversized_weight_status", 3),
    ] {
        let dg = dev.htod_sync_copy(&[if kind == 0 { bf16::NAN } else { bf16::ONE }])?;
        let du = dev.htod_sync_copy(&[if kind == 1 { bf16::NAN } else { bf16::ONE }])?;
        let dw = dev.htod_sync_copy(&[if kind == 2 {
            f32::NAN
        } else if kind == 3 {
            2.0
        } else {
            1.0
        }])?;
        let mut out = dev.alloc_zeros::<bf16>(1)?;
        let mut flag = dev.alloc_zeros::<u32>(1)?;
        kernels.swiglu(&stream, &dg, &du, &dw, true, &mut out, &mut flag, 1, 1)?;
        dev.synchronize()?;
        let code = dev.dtoh_sync_copy(&flag)?[0];
        ensure!(code & 1 != 0, "SwiGLU bad input escaped");
        negative.insert(name.into(), json!(code));
    }
    let dg = dev.htod_sync_copy(&[bf16::ONE])?;
    let dw = dev.htod_sync_copy(&[1_f32])?;
    let mut out = dev.alloc_zeros::<bf16>(1)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    let nonblocking = dev.fork_default_stream()?;
    ensure!(
        kernels
            .swiglu(&nonblocking, &dg, &dg, &dw, true, &mut out, &mut flag, 1, 1)
            .is_err(),
        "nonblocking stream escaped"
    );
    negative.insert("nonblocking_stream_rejected".into(), json!(true));
    ensure!(
        kernels
            .swiglu(&stream, &dg, &dg, &dw, true, &mut out, &mut flag, 129, 1)
            .is_err(),
        "shape129 escaped"
    );
    negative.insert("out_of_bound_rows_rejected".into(), json!(true));
    ensure!(began.elapsed() < Duration::from_secs(WALL), "wall limit");
    Ok(
        json!({"schema_version":1,"kind":"dsv41_moe_gpu_component_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"tensor_parallel_engine":false,"RDMA":false,"checkpoint_payload_read":false,
  "dedicated_source_id":env!("DSV41_MOE_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),"cpu_fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),"route_reference_sha256":moe::CUDA_REFERENCE_SHA,"routing_reference_backend":"official_TorchCUDA","known_CPU_vector_mismatch_cases":["tiny_score_eps"],"elapsed_seconds":began.elapsed().as_secs_f64(),
  "limits":{"wall_seconds":WALL,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND,"scope":"synthetic component arrays only; excludes CUDAcontext/moduleoverhead"},
  "constants":{"experts":384,"topk":6,"gate_temp":1.0,"route_scale":1.5,"route_eps":1e-20,"swiglu_limit":10.0},"tie_policy":"descending biased score; lowest expertID for exact ties; official tieIDs unspecified, checked by admissible top-k/unbiased-weight witness",
  "tolerance":{"route_weight_absolute":2e-6,"route_weight_relative":2e-6,"official_SwiGLU_BF16_max_ulp":0,"CPU_width_SwiGLU_BF16_max_ulp":1,"cross_width":"bitwise"},"route_checks":route_checks,"swiglu_checks":swiglu_checks,"width_checks":width_checks,"negative_controls":negative,
  "scope":"Text-only fixed384/top6 routing and BF16pre-w2 activation only; supplied dense logits/w1/w3. SharedExpert is unweighted activation, not learned sharedexpert execution. No gate/w1/w3/w2 denseGEMM, EP dispatch/accumulation, model forward, serve or speed qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_moe_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"hard wall limit"})
        );
        std::process::exit(124);
    });
    let value = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_moe_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_moe_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{value}");
    if value["all_passed"] != true {
        std::process::exit(1);
    }
}
