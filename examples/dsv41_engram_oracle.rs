//! Bounded model-free GPU Engram oracle. Parent imposes exclusive ownership,
//! 8GiB available-memory floor, thermal guards and an independent <=90s deadline.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, DeviceSlice};
use gb10_inference::dsv41::{engram_math, gpu_engram::EngramKernels};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

const PTX: &str = include_str!("../src/ptx/gpu_dsv41_engram.ptx");
const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_engram_math_vectors.json");
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];
const WALL_SECONDS: u64 = 60;
const MAX_LIVE_GPU_BYTES: usize = 16 * 1024 * 1024;

fn ordered(x: bf16) -> i32 {
    let b = x.to_bits() as i32;
    if b & 0x8000 != 0 {
        0x8000 - (b & 0x7fff)
    } else {
        0x8000 + b
    }
}

fn run() -> Result<()> {
    ensure!(std::env::args_os().len() == 1, "oracle takes no arguments");
    let started = Instant::now();
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    let kernels = EngramKernels::load(&dev, &stream)?;
    let deadline = || -> Result<()> {
        ensure!(
            started.elapsed() < Duration::from_secs(WALL_SECONDS),
            "internal wall limit"
        );
        Ok(())
    };
    let mut lookup_checks = Vec::new();
    let total = 23u64;
    for width in WIDTHS {
        let ids: Vec<u64> = (0..width * 24)
            .map(|i| ((i * 7 + 5) % total as usize) as u64)
            .collect();
        let gpu_ids = dev.htod_sync_copy(&ids)?;
        let mut combined = vec![0.0f32; ids.len() * 256];
        for rank in 0..8 {
            deadline()?;
            let first = (rank * 3).min(total as usize) as u64;
            let rows = (total - first).min(3);
            let weights: Vec<u8> = (0..rows as usize * 256)
                .map(|i| ((i * 11 + rank * 17) % 127) as u8 | if i % 3 == 0 { 128 } else { 0 })
                .collect();
            let scales: Vec<u8> = (0..rows as usize * 8)
                .map(|i| 120 + ((i + rank) % 10) as u8)
                .collect();
            let expected =
                engram_math::lookup_shard_native(&weights, &scales, &ids, first, rows, total)?;
            let gw = dev.htod_sync_copy(&weights)?;
            let gs = dev.htod_sync_copy(&scales)?;
            let mut out = dev.alloc_zeros::<f32>(ids.len() * 256)?;
            let mut status = dev.alloc_zeros::<u32>(1)?;
            kernels.lookup(
                &stream,
                &gw,
                &gs,
                &gpu_ids,
                &mut out,
                &mut status,
                first,
                rows,
                total,
            )?;
            dev.synchronize()?;
            ensure!(dev.dtoh_sync_copy(&status)? == [0], "lookup status nonzero");
            let actual = dev.dtoh_sync_copy(&out)?;
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
                .count();
            ensure!(mismatches == 0, "lookup disagrees with native CPU oracle");
            for (sum, value) in combined.iter_mut().zip(actual) {
                *sum += value;
            }
            lookup_checks.push(json!({"width":width,"rank":rank,"rows":ids.len(),
                "elements":out.len(),"mismatches":mismatches,"passed":true}));
        }
        ensure!(
            combined.iter().all(|v| v.is_finite()),
            "nonfinite combined lookup"
        );
        // This combination is CPU only. Actual RoCE reduction is qualified by a
        // separate eight-rank oracle; do not label these local probes TP8.
    }
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    ensure!(
        fixture["cases"].as_array().context("golden cases")?.len() == 4,
        "four complete gate goldens required"
    );
    let mut gate_checks = Vec::new();
    for case in fixture["cases"].as_array().context("golden cases")? {
        deadline()?;
        let values = |key: &str| -> Result<Vec<bf16>> {
            case[key]
                .as_array()
                .context("golden array")?
                .iter()
                .map(|n| {
                    Ok(bf16::from_bits(u16::try_from(
                        n.as_u64().context("BF16 bits")?,
                    )?))
                })
                .collect()
        };
        let tokens = case["tokens"].as_u64().context("tokens")? as usize;
        let dim = case["dim"].as_u64().context("dim")? as usize;
        let h = values("h_bits")?;
        let kv = values("kv_bits")?;
        let qw = values("q_weight_bits")?;
        let kw = values("k_weight_bits")?;
        let mask: Vec<u8> = case["mask"]
            .as_array()
            .context("mask")?
            .iter()
            .map(|v| Ok(u8::from(v.as_bool().context("mask bool")?)))
            .collect::<Result<_>>()?;
        let expected = values("output_bits")?;
        ensure!(
            expected.len() == h.len() && h.len() == tokens * 4 * dim,
            "golden output extent mismatch"
        );
        let gh = dev.htod_sync_copy(&h)?;
        let gkv = dev.htod_sync_copy(&kv)?;
        let gqw = dev.htod_sync_copy(&qw)?;
        let gkw = dev.htod_sync_copy(&kw)?;
        let gm = dev.htod_sync_copy(&mask)?;
        let mut out = dev.alloc_zeros::<bf16>(h.len())?;
        let mut gates = dev.alloc_zeros::<f32>(tokens * 4)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        kernels.gate(
            &stream,
            &gh,
            &gkv,
            &gqw,
            &gkw,
            &gm,
            &mut out,
            &mut gates,
            &mut status,
            tokens,
            dim,
        )?;
        dev.synchronize()?;
        ensure!(dev.dtoh_sync_copy(&status)? == [0], "gate status nonzero");
        let actual = dev.dtoh_sync_copy(&out)?;
        let gates = dev.dtoh_sync_copy(&gates)?;
        let mut max_ulp = 0;
        let mut max_abs = 0.0f32;
        let mut masked_mismatches = 0;
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            ensure!(
                a.is_finite() && b.is_finite(),
                "nonfinite golden gate result"
            );
            max_ulp = max_ulp.max((ordered(a) - ordered(b)).abs());
            max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
            if mask[i / (4 * dim)] == 0 && a.to_bits() != h[i].to_bits() {
                masked_mismatches += 1;
            }
        }
        ensure!(
            max_ulp <= 1 && masked_mismatches == 0,
            "gate exceeds predeclared one-BF16-ULP tolerance or changes masked token"
        );
        ensure!(
            gates
                .iter()
                .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "invalid gate values"
        );
        gate_checks.push(json!({"case":case["name"],"tokens":tokens,"dim":dim,"elements":actual.len(),
            "max_bf16_ulp":max_ulp,"max_absolute_error":max_abs,"masked_mismatches":masked_mismatches,"gates":gates,"passed":true}));
    }
    // Malformed global IDs and NaN source scales must trip GPU status, rather
    // than silently look up an unrelated valid row. Each uses isolated buffers.
    let gw = dev.htod_sync_copy(&vec![0x38u8; 256])?;
    let gs = dev.htod_sync_copy(&vec![127u8; 8])?;
    let badids = dev.htod_sync_copy(&[1u64])?;
    let mut out = dev.alloc_zeros::<f32>(256)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    kernels.lookup(&stream, &gw, &gs, &badids, &mut out, &mut status, 0, 1, 1)?;
    dev.synchronize()?;
    let invalid_id_status = dev.dtoh_sync_copy(&status)?[0];
    ensure!(
        invalid_id_status & 1 != 0,
        "invalid global ID negative control escaped"
    );
    let nan_scales = dev.htod_sync_copy(&vec![255u8; 8])?;
    let ids = dev.htod_sync_copy(&[0u64])?;
    let mut nan_status = dev.alloc_zeros::<u32>(1)?;
    kernels.lookup(
        &stream,
        &gw,
        &nan_scales,
        &ids,
        &mut out,
        &mut nan_status,
        0,
        1,
        1,
    )?;
    dev.synchronize()?;
    let nan_scale_status = dev.dtoh_sync_copy(&nan_status)?[0];
    ensure!(
        nan_scale_status & 2 != 0,
        "NaN scale negative control escaped"
    );
    let huge = dev.htod_sync_copy(&vec![bf16::from_f32(1e30); 4 * 32])?;
    let huge_kv = dev.htod_sync_copy(&vec![bf16::from_f32(1e30); 5 * 32])?;
    let weight = dev.htod_sync_copy(&vec![bf16::ONE; 4 * 32])?;
    let mask = dev.htod_sync_copy(&[1u8])?;
    let mut overflow_out = dev.alloc_zeros::<bf16>(4 * 32)?;
    let mut overflow_gates = dev.alloc_zeros::<f32>(4)?;
    let mut overflow_status = dev.alloc_zeros::<u32>(1)?;
    kernels.gate(
        &stream,
        &huge,
        &huge_kv,
        &weight,
        &weight,
        &mask,
        &mut overflow_out,
        &mut overflow_gates,
        &mut overflow_status,
        1,
        32,
    )?;
    dev.synchronize()?;
    let finite_input_overflow_status = dev.dtoh_sync_copy(&overflow_status)?[0];
    ensure!(
        finite_input_overflow_status & 8 != 0,
        "finite-input nonfinite-intermediate control escaped"
    );
    deadline()?;
    println!(
        "{}",
        json!({"schema_version":1,"kind":"dsv41_engram_kernel_oracle","status":"PASS","all_passed":true,
        "model_free":true,"tensor_parallel_engine":false,"RDMA":false,
        "ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),
        "goldens_sha256":format!("{:x}",Sha256::digest(FIXTURE.as_bytes())),
        "dedicated_source_id":env!("DSV41_ENGRAM_SOURCE_ID"),"elapsed_seconds":started.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":WALL_SECONDS,"gpu_bytes_upper_bound":MAX_LIVE_GPU_BYTES},
        "lookup_checks":lookup_checks,"gate_checks":gate_checks,
        "negative_controls":{"invalid_id_status":invalid_id_status,"nan_scale_status":nan_scale_status,
            "finite_input_overflow_status":finite_input_overflow_status},
        "limit":"lookup and post-projection gate only; no wkv GEMM, real table residency, network reduction, model forward or serving qualification"})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error:#}");
        std::process::exit(1);
    }
}
