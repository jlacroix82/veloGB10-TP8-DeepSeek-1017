//! Actual entire rank-local text-trunk residency gate; no forward or serving.
use anyhow::{ensure, Result};
use cudarc::driver::CudaDevice;
use gb10_inference::dsv41::{
    config::Dsv41Config,
    gpu_model_weights::{plan_rank_load, RankModelWeights},
    gpu_reader::{GpuWeightBudget, WeightLoadLimits},
    reader::CheckpointReader,
    weights::{build_rank_manifest, WoaReconstructionPolicy},
};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc, time::Instant};
const GIB: u64 = 1 << 30;
const WALL: u64 = 900;
fn run() -> Result<Value> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5 && args[1] == "--model-dir" && args[3] == "--rank",
        "usage: dsv41_rank_load_oracle --model-dir VERIFIED1017 --rank0..7"
    );
    let path = PathBuf::from(&args[2]);
    let rank: usize = args[4].parse()?;
    ensure!(rank < 8, "TP8 rank bound");
    let began = Instant::now();
    let config = Dsv41Config::load_dir(&path)?;
    let mut reader = CheckpointReader::open(&path, 4 * 1024 * 1024)?;
    let manifest = Arc::new(build_rank_manifest(
        &config,
        reader.inventory(),
        rank,
        WoaReconstructionPolicy::OfficialBf16Round,
    )?);
    let plan = plan_rank_load(&manifest)?;
    ensure!(
        plan.final_resident_bytes < 72 * GIB && plan.largest_single_allocation_bytes <= 16 * GIB,
        "whole rank exceeds bounded qualification policy"
    );
    println!(
        "{}",
        json!({"event":"rank_load_preflight","rank":rank,"plan":plan,"checkpoint_payload_read":false,"model_execution":false})
    );
    let limits = WeightLoadLimits {
        maximum_resident_bytes: 72 * GIB,
        maximum_single_allocation_bytes: 16 * GIB,
        maximum_staging_bytes: 8 * 1024 * 1024,
        available_floor_bytes: 8 * GIB,
        reserved_nonweight_bytes: 4 * GIB,
        maximum_load_seconds: WALL,
    };
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let bad = GpuWeightBudget::new(WeightLoadLimits {
        maximum_resident_bytes: plan.final_resident_bytes - 1,
        ..limits.clone()
    })?;
    let before = reader.statistics().payload_bytes_read;
    let negative =
        RankModelWeights::load(&dev, &stream, &mut reader, manifest.clone(), &bad, |_| {
            anyhow::bail!("negative gate unexpectedly reached scope loading")
        });
    let rejected = negative.is_err();
    drop(negative);
    ensure!(
        rejected
            && reader.statistics().payload_bytes_read == before
            && bad.snapshot()?.resident_bytes == 0,
        "whole rank insufficient budget performed payloadIO/allocation"
    );
    println!(
        "{}",
        json!({"event":"rank_load_budget_negative","rejected":true,"payload_bytes_before":before,"payload_bytes_after":reader.statistics().payload_bytes_read,"budget":bad.snapshot()?})
    );
    let budget = GpuWeightBudget::new(limits.clone())?;
    let mut scope_count = 0usize;
    let weights =
        RankModelWeights::load(&dev, &stream, &mut reader, manifest.clone(), &budget, |e| {
            println!(
                "{}",
                json!({"event":"rank_scope_evidence","scope_index":scope_count,"load_evidence":e})
            );
            if e.status == "COMPLETE" {
                scope_count += 1;
            }
            Ok(())
        })?;
    let resident = weights.budget()?;
    ensure!(
        scope_count == 41 && resident.resident_bytes == plan.final_resident_bytes,
        "full rank scope accounting differs"
    );
    println!(
        "{}",
        json!({"event":"rank_load_resident","rank":rank,"scopes":scope_count,"plan":plan,"budget":resident,"reader":reader.statistics(),"model_execution":false})
    );
    let after = weights.release()?;
    println!("{}", json!({"event":"rank_load_release","budget":after}));
    ensure!(
        after.resident_bytes == 0
            && after.active_allocations == 0
            && after.allocations_started == after.allocations_released,
        "full rank owners not fully released"
    );
    ensure!(began.elapsed().as_secs() < WALL, "rank load deadline");
    Ok(
        json!({"schema_version":1,"kind":"dsv41_actual_rank_load_oracle","status":"PASS","all_passed":true,
    "checkpoint_payload_read":true,"model_execution":false,"full_model_loaded":false,"full_rank_text_weights_loaded_and_released":true,
    "serve_supported":false,"payload_identity_verified":false,"rank":rank,"world":8,"model_revision":manifest.model_revision,
    "manifest_sha256":manifest.manifest_sha256,"plan":plan,"completed_scopes":scope_count,"resident_budget":resident,"released_budget":after,
    "reader":reader.statistics(),"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":limits,
    "scope":"Actual rank-local native text weights including both resident Engram tables; upload/allocation/release only. No model forward, protocol, quality or performance result."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(WALL + 10));
        std::process::exit(124);
    });
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            println!(
                "{}",
                json!({"schema_version":1,"kind":"dsv41_actual_rank_load_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
            );
            std::process::exit(1);
        }
    }
}
