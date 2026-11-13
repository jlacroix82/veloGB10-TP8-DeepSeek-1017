//! Model-free1017 mHC GPU math and two-layer ordering oracle. No model or RDMA.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    gpu_hc::{HcKernels, PTX},
    hc::{self, Coefficients, Projection},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_hc_vectors.json");
const WALL_SECONDS: u64 = 60;
const ARRAY_BOUND: usize = 16 * 1024 * 1024;
const WIDTHS: [usize; 8] = [1, 2, 4, 8, 16, 32, 75, 128];
fn floats(v: &Value, key: &str) -> Vec<f32> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn bfloat(v: &Value, key: &str) -> Vec<bf16> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| bf16::from_bits(x.as_u64().unwrap() as u16))
        .collect()
}
fn base(seed: usize) -> Vec<f32> {
    (0..24)
        .map(|i| ((i + seed) % 9) as f32 * 0.125 - 0.5)
        .collect()
}
fn float_check(a: &[f32], b: &[f32]) -> Value {
    let mut mismatches = usize::from(a.len() != b.len());
    let mut max_abs = 0_f32;
    for (a, b) in a.iter().zip(b) {
        let error = (a - b).abs();
        max_abs = max_abs.max(error);
        if !a.is_finite() || !b.is_finite() || error > 2e-5 + 2e-5 * b.abs() {
            mismatches += 1;
        }
    }
    json!({"elements":a.len(),"mismatches":mismatches,"max_absolute_error":max_abs,"passed":mismatches==0})
}
fn ordered(v: bf16) -> i32 {
    let b = v.to_bits() as i32;
    if b & 0x8000 != 0 {
        0x8000 - (b & 0x7fff)
    } else {
        0x8000 + b
    }
}
fn bfloat_check(a: &[bf16], b: &[bf16], allow: u32) -> Value {
    let mut mismatches = usize::from(a.len() != b.len());
    let mut max_ulp = 0_u32;
    for (a, b) in a.iter().zip(b) {
        let ulp = ordered(*a).abs_diff(ordered(*b));
        max_ulp = max_ulp.max(ulp);
        if !a.is_finite() || !b.is_finite() || ulp > allow {
            mismatches += 1;
        }
    }
    json!({"elements":a.len(),"mismatches":mismatches,"max_bf16_ulp":max_ulp,"passed":mismatches==0})
}
fn checked(check: &Value) -> Result<()> {
    ensure!(
        check["passed"] == true,
        "component reference mismatch: {check}"
    );
    Ok(())
}
fn status(dev: &Arc<CudaDevice>, status: &cudarc::driver::CudaSlice<u32>) -> Result<()> {
    dev.synchronize()?;
    ensure!(
        dev.dtoh_sync_copy(status)? == [0],
        "nonzero sticky mHC GPU status"
    );
    Ok(())
}
struct GpuCase {
    mix: Coefficients,
    initial: Vec<f32>,
    collapsed_initial: Vec<bf16>,
    collapsed_current: Vec<bf16>,
    expanded: Vec<bf16>,
}
#[allow(clippy::too_many_arguments)]
fn gpu_case(
    dev: &Arc<CudaDevice>,
    k: &HcKernels,
    s: &CudaStream,
    h: &[bf16],
    x: &[bf16],
    p: Projection<'_>,
    tokens: usize,
    dim: usize,
) -> Result<GpuCase> {
    let dh = dev.htod_sync_copy(h)?;
    let dx = dev.htod_sync_copy(x)?;
    let dw = dev.htod_sync_copy(p.weight)?;
    let ds = dev.htod_sync_copy(p.scale)?;
    let db = dev.htod_sync_copy(p.base)?;
    let mut mixes = dev.alloc_zeros::<f32>(tokens * 24)?;
    let mut pre = dev.alloc_zeros::<f32>(tokens * 4)?;
    let mut post = dev.alloc_zeros::<f32>(tokens * 4)?;
    let mut comb = dev.alloc_zeros::<f32>(tokens * 16)?;
    let mut initial = dev.alloc_zeros::<f32>(tokens * 4)?;
    let mut collapsed = dev.alloc_zeros::<bf16>(tokens * dim)?;
    let mut expanded = dev.alloc_zeros::<bf16>(tokens * 4 * dim)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    k.project(s, &dh, &dw, &mut mixes, &mut flag, tokens, dim)?;
    k.split(
        s, &mixes, &ds, &db, &mut pre, &mut post, &mut comb, &mut flag, tokens,
    )?;
    k.initial(s, &mut initial, tokens)?;
    k.collapse(s, &dh, &initial, &mut collapsed, &mut flag, tokens, dim)?;
    status(dev, &flag)?;
    let collapsed_initial = dev.dtoh_sync_copy(&collapsed)?;
    k.collapse(s, &dh, &pre, &mut collapsed, &mut flag, tokens, dim)?;
    k.expand(
        s,
        &dx,
        &dh,
        &post,
        &comb,
        &mut expanded,
        &mut flag,
        tokens,
        dim,
    )?;
    status(dev, &flag)?;
    Ok(GpuCase {
        mix: Coefficients {
            pre: dev.dtoh_sync_copy(&pre)?,
            post: dev.dtoh_sync_copy(&post)?,
            comb: dev.dtoh_sync_copy(&comb)?,
        },
        initial: dev.dtoh_sync_copy(&initial)?,
        collapsed_initial,
        collapsed_current: dev.dtoh_sync_copy(&collapsed)?,
        expanded: dev.dtoh_sync_copy(&expanded)?,
    })
}
fn gpu_collapse(
    dev: &Arc<CudaDevice>,
    k: &HcKernels,
    s: &CudaStream,
    h: &[bf16],
    pre: &[f32],
    tokens: usize,
    dim: usize,
) -> Result<Vec<bf16>> {
    let dh = dev.htod_sync_copy(h)?;
    let dp = dev.htod_sync_copy(pre)?;
    let mut out = dev.alloc_zeros::<bf16>(tokens * dim)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    k.collapse(s, &dh, &dp, &mut out, &mut flag, tokens, dim)?;
    status(dev, &flag)?;
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
        fixture["kind"] == "official1017_mHC_CPU_goldens"
            && fixture["cases"].as_array().unwrap().len() == 4,
        "frozen official AST fixture"
    );
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let kernels = HcKernels::load(&dev, &stream)?;
    let mut fixture_checks = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        let dim = case["dim"].as_u64().unwrap() as usize;
        let h = bfloat(case, "h_bits");
        let x = bfloat(case, "sublayer_bits");
        let seed = case["weight_recipe_seed"].as_u64().unwrap() as usize;
        let weight = hc::oracle_weight_recipe(dim, seed);
        let scale = floats(case, "scale");
        let base = floats(case, "base");
        let result = gpu_case(
            &dev,
            &kernels,
            &stream,
            &h,
            &x,
            Projection {
                weight: &weight,
                scale: &scale,
                base: &base,
            },
            tokens,
            dim,
        )?;
        let mut row = json!({"case":case["name"],"tokens":tokens,"dim":dim,"passed":true,
            "pre":float_check(&result.mix.pre,&floats(case,"pre")),"post":float_check(&result.mix.post,&floats(case,"post")),"comb":float_check(&result.mix.comb,&floats(case,"comb")),
            "initial_pre_exact":result.initial==floats(case,"initial_pre"),
            "collapse_initial":bfloat_check(&result.collapsed_initial,&bfloat(case,"collapsed_initial_bits"),0),
            "collapse_current":bfloat_check(&result.collapsed_current,&bfloat(case,"collapsed_current_bits"),1),
            "expand":bfloat_check(&result.expanded,&bfloat(case,"expanded_bits"),1)});
        ensure!(row["initial_pre_exact"] == true, "initial one-hot differs");
        for key in [
            "pre",
            "post",
            "comb",
            "collapse_initial",
            "collapse_current",
            "expand",
        ] {
            checked(&row[key])?;
        }
        row["passed"] = json!(true);
        fixture_checks.push(row);
    }
    // Independent token rows, same deterministic row prefix at every width.
    let mut width_checks = Vec::new();
    let mut width_results = Vec::new();
    for width in WIDTHS {
        let dim = 5120;
        let h = (0..width * 4 * dim)
            .map(|i| bf16::from_f32(((i * 7 + 3) % 31) as f32 * 0.125 - 1.875))
            .collect::<Vec<_>>();
        let x = (0..width * dim)
            .map(|i| bf16::from_f32(((i * 5 + 11) % 23) as f32 * 0.0625 - 0.6875))
            .collect::<Vec<_>>();
        let weight = hc::oracle_weight_recipe(dim, 17);
        let scale = [0.75, -0.5, 1.25];
        let b = base(17);
        let p = Projection {
            weight: &weight,
            scale: &scale,
            base: &b,
        };
        let cpu = hc::coefficients(&h, p, width, dim)?;
        let result = gpu_case(&dev, &kernels, &stream, &h, &x, p, width, dim)?;
        let row = json!({"width":width,"dim":dim,"passed":true,"initial_pre_exact":result.initial==hc::initial_pre(width)?,"collapse_initial":bfloat_check(&result.collapsed_initial,&hc::collapse(&h,&hc::initial_pre(width)?,width,dim)?,0),"pre":float_check(&result.mix.pre,&cpu.pre),"post":float_check(&result.mix.post,&cpu.post),"comb":float_check(&result.mix.comb,&cpu.comb),
            "collapse":bfloat_check(&result.collapsed_current,&hc::collapse(&h,&cpu.pre,width,dim)?,1),
            "expand":bfloat_check(&result.expanded,&hc::expand(&x,&h,&cpu.post,&cpu.comb,width,dim)?,1)});
        ensure!(
            row["initial_pre_exact"] == true,
            "prefill one-hot initial coefficients changed"
        );
        for key in [
            "pre",
            "post",
            "comb",
            "collapse_initial",
            "collapse",
            "expand",
        ] {
            checked(&row[key])?;
        }
        width_checks.push(row);
        width_results.push(result);
    }
    let widest = width_results.last().unwrap();
    for (i, width) in WIDTHS.iter().enumerate() {
        let r = &width_results[i];
        let bitwise = r
            .mix
            .pre
            .iter()
            .zip(&widest.mix.pre)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && r.mix
                .post
                .iter()
                .zip(&widest.mix.post)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && r.mix
                .comb
                .iter()
                .zip(&widest.mix.comb)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && r.collapsed_current
                .iter()
                .zip(&widest.collapsed_current[..width * 5120])
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && r.expanded
                .iter()
                .zip(&widest.expanded[..width * 4 * 5120])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        ensure!(bitwise, "cross-width independent token math changed");
        width_checks[i]["cross_width_bitwise"] = json!(bitwise);
    }
    // Controlled affine callbacks exercise the *official* two-block ordering.
    // Attention/MoE and their RMS norms are NOT implemented by these fixtures.
    let chain = &fixture["chain"];
    let mut h = bfloat(chain, "h_bits");
    let mut incoming = hc::initial_pre(1)?;
    let mut ordering_checks = Vec::new();
    let mut old_current_pre_rejected = false;
    for layer in chain["layers"].as_array().unwrap() {
        let seed = layer["weight_recipe_seed"].as_u64().unwrap() as usize;
        let dim = 8;
        let aw = hc::oracle_weight_recipe(dim, seed);
        let ab = base(seed);
        let fw = hc::oracle_weight_recipe(dim, seed + 5);
        let fb = base(seed + 5);
        let scale = [0.75, -0.5, 1.25];
        let attention_input = gpu_collapse(&dev, &kernels, &stream, &h, &incoming, 1, dim)?;
        let attention_output = attention_input
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() * 0.5 + 0.0625))
            .collect::<Vec<_>>();
        let ar = gpu_case(
            &dev,
            &kernels,
            &stream,
            &h,
            &attention_output,
            Projection {
                weight: &aw,
                scale: &scale,
                base: &ab,
            },
            1,
            dim,
        )?;
        old_current_pre_rejected |= ar.collapsed_current != attention_input;
        let ffn_input = gpu_collapse(&dev, &kernels, &stream, &ar.expanded, &ar.mix.pre, 1, dim)?;
        let ffn_output = ffn_input
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() * -0.25 - 0.125))
            .collect::<Vec<_>>();
        let fr = gpu_case(
            &dev,
            &kernels,
            &stream,
            &ar.expanded,
            &ffn_output,
            Projection {
                weight: &fw,
                scale: &scale,
                base: &fb,
            },
            1,
            dim,
        )?;
        let collapsed = gpu_collapse(&dev, &kernels, &stream, &fr.expanded, &fr.mix.pre, 1, dim)?;
        let row = json!({"weight_recipe_seed":seed,"passed":true,"stream":bfloat_check(&fr.expanded,&bfloat(layer,"h_bits"),1),
            "next_pre":float_check(&fr.mix.pre,&floats(layer,"next_pre")),"final_collapse":bfloat_check(&collapsed,&bfloat(layer,"final_collapsed_bits"),1)});
        for key in ["stream", "next_pre", "final_collapse"] {
            checked(&row[key])?;
        }
        ordering_checks.push(row);
        h = fr.expanded;
        incoming = fr.mix.pre;
    }
    ensure!(
        old_current_pre_rejected,
        "wrong-current-pre negative did not differ"
    );
    // A source->destination cyclic permutation must produce [4,1,2,3], not its transpose.
    let mut comb = vec![0.0; 16];
    for source in 0..4 {
        comb[source * 4 + (source + 1) % 4] = 1.0;
    }
    let residual = (0..32)
        .map(|i| bf16::from_f32((i / 8 + 1) as f32))
        .collect::<Vec<_>>();
    let dx = dev.htod_sync_copy(&[bf16::ZERO; 8])?;
    let dh = dev.htod_sync_copy(&residual)?;
    let dp = dev.htod_sync_copy(&[0_f32; 4])?;
    let dc = dev.htod_sync_copy(&comb)?;
    let mut out = dev.alloc_zeros::<bf16>(32)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    kernels.expand(&stream, &dx, &dh, &dp, &dc, &mut out, &mut flag, 1, 8)?;
    status(&dev, &flag)?;
    let permutation = dev.dtoh_sync_copy(&out)?;
    let observed = (0..4)
        .map(|i| permutation[i * 8].to_f32())
        .collect::<Vec<_>>();
    ensure!(
        observed == [4.0, 1.0, 2.0, 3.0] && observed != [2.0, 3.0, 4.0, 1.0],
        "combination orientation negative failed"
    );
    drop((dx, dh, dp, dc, out, flag));
    // Real device sticky rejection controls, with fresh status per operation.
    let mut negative = serde_json::Map::new();
    for (name, huge, nan_weight, wanted) in [
        ("nan_weight_status", false, true, 1_u32),
        ("finite_rms_overflow_status", true, false, 2_u32),
    ] {
        let h = dev.htod_sync_copy(&vec![if huge { bf16::MAX } else { bf16::ONE }; 32])?;
        let mut weights = vec![0_f32; 24 * 32];
        if nan_weight {
            weights[0] = f32::NAN;
        }
        let w = dev.htod_sync_copy(&weights)?;
        let mut m = dev.alloc_zeros::<f32>(24)?;
        let mut f = dev.alloc_zeros::<u32>(1)?;
        kernels.project(&stream, &h, &w, &mut m, &mut f, 1, 8)?;
        dev.synchronize()?;
        let actual = dev.dtoh_sync_copy(&f)?[0];
        ensure!(
            actual & wanted != 0,
            "GPU rejection control escaped: {name}"
        );
        negative.insert(name.into(), json!(actual));
    }
    let mixes = dev.htod_sync_copy(&[0_f32; 24])?;
    let scale = dev.htod_sync_copy(&[f32::NAN, 1.0, 1.0])?;
    let base = dev.htod_sync_copy(&[0_f32; 24])?;
    let mut invalid_pre = dev.alloc_zeros::<f32>(4)?;
    let mut invalid_post = dev.alloc_zeros::<f32>(4)?;
    let mut invalid_comb = dev.alloc_zeros::<f32>(16)?;
    let mut invalid_flag = dev.alloc_zeros::<u32>(1)?;
    kernels.split(
        &stream,
        &mixes,
        &scale,
        &base,
        &mut invalid_pre,
        &mut invalid_post,
        &mut invalid_comb,
        &mut invalid_flag,
        1,
    )?;
    dev.synchronize()?;
    let nan_scale = dev.dtoh_sync_copy(&invalid_flag)?[0];
    ensure!(nan_scale & 1 != 0, "NaN F32 coefficient scale escaped");
    negative.insert("nan_scale_status".into(), json!(nan_scale));
    let h = dev.htod_sync_copy(&[bf16::MAX; 32])?;
    let pre = dev.htod_sync_copy(&[1_f32; 4])?;
    let mut out = dev.alloc_zeros::<bf16>(8)?;
    let mut f = dev.alloc_zeros::<u32>(1)?;
    kernels.collapse(&stream, &h, &pre, &mut out, &mut f, 1, 8)?;
    dev.synchronize()?;
    let overflow = dev.dtoh_sync_copy(&f)?[0];
    ensure!(overflow & 2 != 0, "BF16 output overflow escaped");
    negative.insert("bf16_collapse_overflow_status".into(), json!(overflow));
    let nonblocking = dev.fork_default_stream()?;
    let mut pre = dev.alloc_zeros::<f32>(4)?;
    ensure!(
        kernels.initial(&nonblocking, &mut pre, 1).is_err(),
        "nonblocking stream escaped"
    );
    negative.insert("nonblocking_stream_rejected".into(), json!(true));
    ensure!(
        kernels.initial(&stream, &mut pre, 129).is_err(),
        "shape129 escaped"
    );
    negative.insert("out_of_bound_shape_rejected".into(), json!(true));
    negative.insert(
        "wrong_current_pre_rejected".into(),
        json!(old_current_pre_rejected),
    );
    negative.insert("transposed_comb_rejected".into(), json!(true));
    ensure!(
        began.elapsed() < Duration::from_secs(WALL_SECONDS),
        "wall deadline"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_hc_gpu_component_oracle","status":"PASS","all_passed":true,
        "model_free":true,"model_execution":false,"tensor_parallel_engine":false,"RDMA":false,
        "dedicated_source_id":env!("DSV41_HC_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),
        "cpu_fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),"elapsed_seconds":began.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":WALL_SECONDS,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND,
            "scope":"explicit small synthetic component arrays, excludes CUDA context/module overhead"},
        "tolerance":{"FP32_absolute":2e-5,"FP32_relative":2e-5,"BF16_max_ulp":1,"initial_pre_max_ulp":0,"cross_width":"bitwise"},
        "constants":{"hc_mult":4,"sinkhorn_iters":20,"hc_eps":1e-6,"norm_eps":1e-20},
        "fixture_checks":fixture_checks,"width_checks":width_checks,"ordering_checks":ordering_checks,"negative_controls":negative,
        "scope":"mHC projection/split/pre/post and controlled two-layer ordering only; no checkpoint payload, attention/MoE, model inference, TP8 collective or serving/speed qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        println!(
            "{}",
            json!({"schema_version":1,"kind":"dsv41_hc_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"hard wall limit"})
        );
        std::process::exit(124);
    });
    let value = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_hc_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_hc_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{value}");
    if value["all_passed"] != true {
        std::process::exit(1);
    }
}
