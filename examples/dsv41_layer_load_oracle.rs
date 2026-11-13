//! Bounded actual-checkpoint loader gate. This reads one rank's layer0 weights;
//! it performs no model forward or token generation. Parent supplies a300s
//! deadline, exclusive GPU ownership and8GiB/80C/90C resource guards.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    config::Dsv41Config,
    gpu_weights::{self, GpuWeightBudget, LoadScope, WeightLoadLimits},
    numerics::{decode_e4m3_finite, decode_ue8m0},
    reader::{CheckpointReader, TensorSpan},
    weights::{build_rank_manifest, WoaReconstructionPolicy},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const GIB: u64 = 1024 * 1024 * 1024;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked(checks: &mut Vec<Value>, check: Value) -> Result<()> {
    println!("{}", json!({"event":"layer_load_diagnostic","check":check}));
    let passed = check["passed"] == true;
    checks.push(check);
    ensure!(
        passed,
        "layer-load diagnostic failed; preceding event retains actual/source hashes"
    );
    Ok(())
}
fn selected(reader: &mut CheckpointReader, span: &TensorSpan) -> Result<Vec<u8>> {
    ensure!(
        span.bytes <= 16 * 1024 * 1024,
        "diagnostic selection exceeds16MiB"
    );
    let mut out = Vec::with_capacity(span.bytes as usize);
    reader.stream(span, |_, bytes| {
        out.extend_from_slice(bytes);
        Ok(())
    })?;
    ensure!(out.len() as u64 == span.bytes, "selected byte extent");
    Ok(out)
}
fn run() -> Result<Value> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5 && args[1] == "--model-dir" && args[3] == "--rank",
        "usage: dsv41_layer_load_oracle --model-dir VERIFIED1017 --rank 0..7"
    );
    let dir = PathBuf::from(&args[2]);
    let rank: usize = args[4].parse()?;
    ensure!(rank < 8, "rank outsideTP8");
    let began = Instant::now();
    let config = Dsv41Config::load_dir(&dir)?;
    let mut reader = CheckpointReader::open(&dir, 4 * 1024 * 1024)?;
    let manifest = build_rank_manifest(
        &config,
        reader.inventory(),
        rank,
        WoaReconstructionPolicy::OfficialBf16Round,
    )?;
    let plan = gpu_weights::plan_scope_load(&manifest, LoadScope::Layer(0))?;
    ensure!(
        !plan.scope_contains_engram_table && plan.additional_peak_resident_bytes < 2 * GIB,
        "bounded layer0 load plan changed"
    );
    let limits = WeightLoadLimits {
        maximum_resident_bytes: 2 * GIB,
        maximum_single_allocation_bytes: 64 * 1024 * 1024,
        maximum_staging_bytes: 8 * 1024 * 1024,
        available_floor_bytes: 8 * GIB,
        reserved_nonweight_bytes: GIB,
        maximum_load_seconds: 240,
    };
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    // A budget one byte below the already known final allocation must fail
    // before touching any model payload or allocating scope-owned buffers.
    let bad_budget = GpuWeightBudget::new(WeightLoadLimits {
        maximum_resident_bytes: plan.final_resident_bytes - 1,
        ..limits.clone()
    })?;
    let before_reads = reader.statistics().payload_bytes_read;
    let mut rejected = None;
    let rejected_result = gpu_weights::load_layer(
        &dev,
        &stream,
        &mut reader,
        &manifest,
        0,
        &bad_budget,
        &mut rejected,
    );
    let rejected_as_expected = rejected_result.is_err();
    drop(rejected_result);
    println!(
        "{}",
        json!({"event":"layer_load_budget_negative","rejected":rejected_as_expected,
        "evidence":rejected,"payload_bytes_before":before_reads,
        "payload_bytes_after":reader.statistics().payload_bytes_read,"budget":bad_budget.snapshot()?})
    );
    ensure!(
        rejected_as_expected,
        "insufficient scope budget was accepted"
    );
    ensure!(
        reader.statistics().payload_bytes_read == before_reads
            && bad_budget.snapshot()?.resident_bytes == 0,
        "preflight rejection performed payloadIO/retained allocation"
    );
    let budget = GpuWeightBudget::new(limits)?;
    let mut evidence = None;
    let loaded = match gpu_weights::load_layer(
        &dev,
        &stream,
        &mut reader,
        &manifest,
        0,
        &budget,
        &mut evidence,
    ) {
        Ok(v) => v,
        Err(error) => {
            eprintln!(
                "{}",
                json!({"kind":"dsv41_layer_load_failure","evidence":evidence})
            );
            return Err(error);
        }
    };
    let evidence = evidence.context("missing successful load evidence")?;
    println!(
        "{}",
        json!({"event":"layer_load_completed","rank":rank,"load_evidence":evidence})
    );
    ensure!(evidence.status == "COMPLETE", "scope did not complete");
    let mut checks = Vec::new();
    let first_expert = rank * 48;
    for name in [
        format!("layers.0.attn.wo_b.weight"),
        format!("layers.0.ffn.experts.{first_expert}.w1.weight"),
        format!("layers.0.ffn.experts.{first_expert}.w1.scale"),
    ] {
        let actual = dev.dtoh_sync_copy(loaded.native_u8(&name)?)?;
        let expected = evidence
            .attempts
            .iter()
            .find(|a| a.span.tensor_name == name)
            .context("missing range evidence")?;
        checked(
            &mut checks,
            json!({"tensor":name,"kind":"native_bytes","bytes":actual.len(),
            "sha256":digest(&actual),"source_sha256":expected.streamed_native_sha256,
            "passed":digest(&actual)==expected.streamed_native_sha256}),
        )?;
    }
    let norm = "layers.0.attn_norm.weight";
    let values = dev.dtoh_sync_copy(loaded.bf16(norm)?)?;
    let actual: Vec<u8> = values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let attempt = evidence
        .attempts
        .iter()
        .find(|a| a.span.tensor_name == norm)
        .context("norm range evidence")?;
    checked(
        &mut checks,
        json!({"tensor":norm,"kind":"typed_BF16","elements":values.len(),
        "sha256":digest(&actual),"source_sha256":attempt.streamed_native_sha256,
        "passed":digest(&actual)==attempt.streamed_native_sha256}),
    )?;
    let hc = "layers.0.hc_attn_base";
    let values = dev.dtoh_sync_copy(loaded.f32(hc)?)?;
    let actual: Vec<u8> = values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let attempt = evidence
        .attempts
        .iter()
        .find(|a| a.span.tensor_name == hc)
        .context("HC range evidence")?;
    checked(
        &mut checks,
        json!({"tensor":hc,"kind":"typed_F32","elements":values.len(),
        "sha256":digest(&actual),"source_sha256":attempt.streamed_native_sha256,
        "passed":digest(&actual)==attempt.streamed_native_sha256}),
    )?;
    let woa = "layers.0.attn.wo_a.weight";
    let scale = "layers.0.attn.wo_a.scale";
    let wlayout = manifest
        .entries
        .iter()
        .find(|e| e.name == woa)
        .context("WO-A layout")?;
    let slayout = manifest
        .entries
        .iter()
        .find(|e| e.name == scale)
        .context("WO-A scale layout")?;
    let weight_span = wlayout.span(&reader)?;
    let scale_span = slayout.span(&reader)?;
    let w = selected(&mut reader, &weight_span)?;
    let s = selected(&mut reader, &scale_span)?;
    let actual = dev.dtoh_sync_copy(loaded.bf16(woa)?)?;
    let converted_bytes: Vec<u8> = actual
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    println!(
        "{}",
        json!({"event":"layer_load_WOA_reread","weight_bytes":w.len(),"scale_bytes":s.len(),
        "weight_sha256":digest(&w),"scale_sha256":digest(&s),"converted_elements":actual.len(),
        "converted_sha256":digest(&converted_bytes)})
    );
    ensure!(
        w.len() == 1024 * 4096 && s.len() == 32 * 128 && actual.len() == w.len(),
        "WO-A native/converted geometry"
    );
    let mut mismatches = 0;
    for (index, &code) in w.iter().enumerate() {
        let row = index / 4096;
        let col = index % 4096;
        let expected = bf16::from_f32(
            decode_e4m3_finite(code)? * decode_ue8m0(s[(row / 32) * 128 + col / 32])?,
        );
        ensure!(expected.is_finite(), "actualWO-A reference overflows");
        mismatches += usize::from(expected.to_bits() != actual[index].to_bits());
    }
    checked(
        &mut checks,
        json!({"tensor":woa,"kind":"source_BF16_reconstruction","elements":actual.len(),
        "bit_mismatches":mismatches,"passed":mismatches==0}),
    )?;
    let released = loaded.release()?;
    println!(
        "{}",
        json!({"event":"layer_load_released","budget":released})
    );
    ensure!(
        released.resident_bytes == 0 && released.active_allocations == 0,
        "scope release leaked budget ownership"
    );
    ensure!(
        began.elapsed() < Duration::from_secs(240),
        "component session deadline"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_actual_layer_load_component","status":"PASS","all_passed":true,
        "rank":rank,"world":8,"layer":0,"model_revision":manifest.model_revision,
        "manifest_sha256":manifest.manifest_sha256,"payload_identity_verified":false,
        "checkpoint_payload_read":true,"model_execution":false,"full_model_loaded":false,"serve_supported":false,
        "checks":checks,"insufficient_budget_rejection":rejected,"load_evidence":evidence,"release":released,
        "reader":reader.statistics(),"elapsed_seconds":began.elapsed().as_secs_f64(),
        "limits":{"wall_seconds":240,"scope_weight_bytes":2*GIB,"available_floor_bytes":8*GIB},
        "scope":"actual retained checkpoint layer0 upload/selection/conversion/release only; no model forward, token generation or speed claim"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(250));
        eprintln!("weightloadharddeadline");
        std::process::exit(2)
    });
    match run() {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("FAIL: {error:#}");
            std::process::exit(1)
        }
    }
}
