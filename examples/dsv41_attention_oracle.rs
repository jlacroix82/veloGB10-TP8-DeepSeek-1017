//! Model-free sparse-attention math and lane/head-stride oracle. Zero arguments.
use anyhow::{ensure, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    attention_math,
    gpu_attention::{AttentionKernels, PTX},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

const D: usize = 512;
const MAX_BYTES: usize = 16 * 1024 * 1024;
#[derive(Clone)]
struct Host {
    q: Vec<bf16>,
    kv: Vec<bf16>,
    sink: Vec<f32>,
    indices: Vec<i32>,
    b: usize,
    m: usize,
    h: usize,
    n: usize,
    k: usize,
}
fn host(b: usize, m: usize, h: usize, n: usize, uniform: bool) -> Host {
    let mut q = vec![bf16::ZERO; b * m * h * D];
    let mut kv = vec![bf16::ZERO; b * n * D];
    let mut indices = vec![-1; b * m * n];
    for lane in 0..b {
        for row in 0..n {
            for d in 0..D {
                let value = if uniform {
                    row as f32 + lane as f32 * 16.0
                } else {
                    match d {
                        0 => 0.25 + (row % 17) as f32 * 0.125 + lane as f32 * 0.0625,
                        1 => 0.125 + (row % 7) as f32 * 0.125,
                        2 => 0.03125 * (row % 11 + 1) as f32,
                        _ => {
                            1.0 + lane as f32 * 0.125
                                + (row % 13) as f32 * 0.03125
                                + (d % 7) as f32 * 0.00390625
                        }
                    }
                };
                kv[(lane * n + row) * D + d] = bf16::from_f32(value);
            }
        }
        for query in 0..m {
            for head in 0..h {
                let base = ((lane * m + query) * h + head) * D;
                if !uniform {
                    q[base] = bf16::from_f32((head % 8 + 1) as f32 * 0.5);
                    q[base + 1] = bf16::from_f32((query + 1) as f32);
                    q[base + 2] = bf16::from_f32(if lane % 2 == 0 { 2.0 } else { -2.0 });
                }
            }
            for key in 0..n {
                // One full causal source, first tile masked, then all masked.
                if query % 3 == 0 || (query % 3 == 1 && key >= 64) {
                    indices[(lane * m + query) * n + key] = key as i32;
                }
            }
        }
    }
    Host {
        q,
        kv,
        sink: (0..h)
            .map(|i| if uniform { 0.0 } else { (i % 5) as f32 * 0.125 })
            .collect(),
        indices,
        b,
        m,
        h,
        n,
        k: n,
    }
}
fn ordered(x: bf16) -> i32 {
    let bits = x.to_bits() as i32;
    if bits & 0x8000 != 0 {
        0x8000 - (bits & 0x7fff)
    } else {
        0x8000 + bits
    }
}
fn compare(a: &[bf16], b: &[bf16]) -> Result<(usize, i32)> {
    ensure!(a.len() == b.len(), "full-output length mismatch");
    let mut bit_mismatches = 0;
    let mut maximum = 0;
    for (&x, &y) in a.iter().zip(b) {
        ensure!(
            x.is_finite() && y.is_finite(),
            "nonfinite attention comparison"
        );
        bit_mismatches += usize::from(x.to_bits() != y.to_bits());
        maximum = maximum.max((ordered(x) - ordered(y)).abs());
    }
    Ok((bit_mismatches, maximum))
}
fn gpu(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    kernels: &AttentionKernels,
    input: &Host,
    start: usize,
    width: usize,
) -> Result<(Vec<bf16>, u32)> {
    ensure!(start + width <= input.b, "lane group bound");
    let qp = input.m * input.h * D;
    let kp = input.n * D;
    let ip = input.m * input.k;
    let q = dev.htod_sync_copy(&input.q[start * qp..(start + width) * qp])?;
    let kv = dev.htod_sync_copy(&input.kv[start * kp..(start + width) * kp])?;
    let sink = dev.htod_sync_copy(&input.sink)?;
    let indices = dev.htod_sync_copy(&input.indices[start * ip..(start + width) * ip])?;
    let mut out = dev.alloc_zeros::<bf16>(width * qp)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    kernels.sparse(
        stream,
        &q,
        &kv,
        &sink,
        &indices,
        &mut out,
        &mut status,
        width,
        input.m,
        input.h,
        input.n,
        input.k,
    )?;
    dev.synchronize()?;
    Ok((dev.dtoh_sync_copy(&out)?, dev.dtoh_sync_copy(&status)?[0]))
}
fn run() -> Result<Value> {
    ensure!(std::env::args_os().len() == 1, "oracle takes no arguments");
    let began = Instant::now();
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    let kernels = AttentionKernels::load(&dev, &stream)?;
    let mut checks = Vec::new();
    for uniform in [true, false] {
        let input = host(16, 3, 8, 129, uniform);
        let reference = attention_math::sparse(
            &input.q,
            &input.kv,
            &input.sink,
            &input.indices,
            input.b,
            input.m,
            input.h,
            input.n,
            input.k,
        )?;
        let (baseline, status) = gpu(&dev, &stream, &kernels, &input, 0, 16)?;
        ensure!(status == 0, "valid C16 attention status");
        let (_, error) = compare(&baseline, &reference)?;
        ensure!(
            error <= if uniform { 0 } else { 1 },
            "attention exceeds declared analytical/one-BF16-ULP reference tolerance"
        );
        for width in [1, 2, 4, 8, 16] {
            for start in (0..16).step_by(width) {
                let (actual, status) = gpu(&dev, &stream, &kernels, &input, start, width)?;
                ensure!(status == 0, "valid attention group status");
                let pitch = input.m * input.h * D;
                let (mismatches, _) =
                    compare(&actual, &baseline[start * pitch..(start + width) * pitch])?;
                ensure!(mismatches == 0, "batch width changes attention output bits");
                checks.push(json!({"case":if uniform{"uniform"}else{"nonuniform"},"width":width,"start_lane":start,
                "heads":8,"queries":3,"topk":129,"elements":actual.len(),"CPU_max_bf16_ulp":error,
                "cross_width_bit_mismatches":mismatches,"passed":true}));
            }
        }
    }
    for heads in [1, 7, 16, 17, 32, 64] {
        let input = host(2, 3, heads, 65, false);
        let reference = attention_math::sparse(
            &input.q,
            &input.kv,
            &input.sink,
            &input.indices,
            input.b,
            input.m,
            input.h,
            input.n,
            input.k,
        )?;
        let (actual, status) = gpu(&dev, &stream, &kernels, &input, 0, 2)?;
        let (_, error) = compare(&actual, &reference)?;
        ensure!(
            status == 0 && error <= 1,
            "local-head padding/stride oracle failed"
        );
        checks.push(
            json!({"case":"local_heads","heads":heads,"width":2,"queries":3,"topk":65,
            "elements":actual.len(),"CPU_max_bf16_ulp":error,"passed":true}),
        );
    }
    let input = host(1, 1, 8, 640, false);
    let reference = attention_math::sparse(
        &input.q,
        &input.kv,
        &input.sink,
        &input.indices,
        1,
        1,
        8,
        640,
        640,
    )?;
    let (actual, status) = gpu(&dev, &stream, &kernels, &input, 0, 1)?;
    let (_, error) = compare(&actual, &reference)?;
    ensure!(status == 0 && error <= 1, "ten-tile attention failed");
    checks.push(
        json!({"case":"max_topk","heads":8,"width":1,"queries":1,"topk":640,
        "elements":actual.len(),"CPU_max_bf16_ulp":error,"passed":true}),
    );
    let mut bad = host(1, 1, 8, 65, false);
    bad.indices[0] = 65;
    let (_, bad_index) = gpu(&dev, &stream, &kernels, &bad, 0, 1)?;
    ensure!(bad_index & 1 != 0, "invalid-index control escaped");
    let mut bad = host(1, 1, 8, 65, false);
    bad.q[0] = bf16::NAN;
    let (_, bad_query) = gpu(&dev, &stream, &kernels, &bad, 0, 1)?;
    ensure!(bad_query & 2 != 0, "NaN query control escaped");
    let mut bad = host(1, 1, 8, 65, false);
    bad.sink[0] = f32::INFINITY;
    let (_, bad_sink) = gpu(&dev, &stream, &kernels, &bad, 0, 1)?;
    ensure!(bad_sink & 2 != 0, "nonfinite sink control escaped");
    ensure!(
        began.elapsed() < Duration::from_secs(60),
        "internal elapsed deadline"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_sparse_attention_component_oracle","status":"PASS","all_passed":true,
        "model_free":true,"model_execution":false,"serve_supported":false,"RDMA":false,"checks":checks,
        "source_id_hex":env!("DSV41_ATTENTION_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),
        "negative_controls":{"invalid_index":bad_index,"nonfinite_query":bad_query,"nonfinite_sink":bad_sink},
        "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":60,"gpu_bytes_upper_bound":MAX_BYTES},
        "scope":"sparse kernel with supplied BF16 queries/KV/indices; no model, RoPE/projection/index/compression or throughput qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(60));
        eprintln!("attention oracle hard wall limit");
        std::process::exit(2)
    });
    match run() {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("FAIL: {error:#}");
            std::process::exit(1)
        }
    }
}
