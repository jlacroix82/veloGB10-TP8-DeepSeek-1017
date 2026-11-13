//! CPU replay of exact original request preparation + actual native selections.
//! No CUDA device, checkpoint, network, completion constructor or execution.
use anyhow::{ensure, Context, Result};
use gb10_inference::dsv41::{
    atomic_request_admission::{
        AdmissionClock, CpuAtomicAdmission, CpuSlotBook, PreparedAtomicRequest,
        ProposedBatchDeployment,
    },
    batch_request_plan::{BatchGroupBinding, BatchPhaseCursor, BatchPhasePlan},
    gpu_collectives::Program,
    model_session_plan::native_ids_sha256,
    native_api_contract::{CpuBridge, ATOMIC_ROUTE},
    position::FrequencyBackend,
    scheduler::LaneKey,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    path::PathBuf,
    time::{Duration, Instant},
};
use uuid::Uuid;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Lane {
    slot: usize,
    generation: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    original_request: Value,
    response_id: String,
    boot: Uuid,
    lanes: Vec<Lane>,
    wall_millis: u64,
    global_checkpoint_geometry_checksum_sha256: String,
    execution_snapshot_sha256: String,
    engine_binary_sha256: String,
    first_operation_id: u64,
    first_step_id: u64,
    start_epoch: u64,
    next_batch_version: u64,
    selected_steps: Vec<Vec<u32>>,
}
fn run() -> Result<Value> {
    let mut raw = vec![];
    std::io::stdin()
        .take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    ensure!(raw.len() <= 2 * 1024 * 1024, "bounded CPU audit input");
    let input: Input = serde_json::from_slice(&raw)?;
    ensure!(
        input.wall_millis == 180000 && input.selected_steps.len() <= 320,
        "unchanged native180s/full bounded steps"
    );
    let tokenizer = std::env::var_os("DSV41_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("test-data/tokenizer.json"));
    let bridge = CpuBridge::from_verified_tokenizer(&std::fs::read(tokenizer)?)?;
    let prepared =
        PreparedAtomicRequest::prepare(&bridge, ATOMIC_ROUTE, Some("1"), &input.original_request)?;
    let now = Instant::now();
    let mut book = CpuSlotBook::new(input.boot, 16)?;
    let mut tx = CpuAtomicAdmission::new(
        &mut book,
        prepared,
        ProposedBatchDeployment {
            boot: input.boot,
            global_checkpoint_geometry_checksum_sha256: input
                .global_checkpoint_geometry_checksum_sha256,
            execution_snapshot_sha256: input.execution_snapshot_sha256,
            engine_binary_sha256: input.engine_binary_sha256,
            frequency: FrequencyBackend::CudaTorch.identity(),
        },
        input
            .lanes
            .iter()
            .map(|k| LaneKey {
                boot: input.boot,
                slot: k.slot,
                generation: k.generation,
            })
            .collect(),
        AdmissionClock::new(now, now + Duration::from_millis(input.wall_millis))?,
        input.response_id,
        now,
    )?;
    tx.commit(now)?;
    let owned = tx.take_published(now)?;
    let group = BatchGroupBinding::from_cpu(&owned, &book, now)?;
    let mut plan = BatchPhasePlan::first(
        group.clone(),
        input.first_operation_id,
        input.first_step_id,
        input.start_epoch,
        input.next_batch_version,
    )?;
    if input.selected_steps.is_empty() {
        return Ok(json!({"passed":true,"CPU_only":true,"prepare_only":true,
            "GPU_initialized":false,"network_opened":false,
            "native_prompt_tokens":group.members().iter().map(|m| m.prompt_ids.len()).collect::<Vec<_>>(),
            "group":group.as_ref()}));
    }
    let mut cursor = BatchPhaseCursor::new(plan.clone());
    let mut index = 0;
    let mut steps = vec![];
    let mut phases = vec![];
    let mut program = Program::new_batch_phase(plan.clone())?;
    loop {
        let selected = input
            .selected_steps
            .get(index)
            .context("CPU replay lacks next real selected rows")?;
        let spec = cursor.next_spec()?.clone();
        let inputs = cursor.inputs()?;
        let mut rows = vec![];
        let mut flat: Vec<u32> = vec![];
        for ((member, source), &token) in spec.members.iter().zip(&inputs).zip(selected) {
            let mut prefix = cursor.progress()[member.original_index]
                .consumed_native_ids
                .clone();
            prefix.extend(&source.native_ids);
            rows.push(
                json!({"lane":member.lane,"consumed_prefix_tokens":prefix.len(),
                "prefix_native_ids_sha256":native_ids_sha256(&prefix),"selected_native_id":token,
                "generated_output_ordinal":member.generated_output_ordinal,
                "intermediate_prefill_head_ignored":member.generated_output_ordinal.is_none()}),
            );
            flat.extend_from_slice(&source.native_ids);
        }
        let claim = cursor.observe(selected)?;
        steps.push(json!({"phase":plan.phase(),"phase_sha256":plan.digest(),"program_sha256":program.digest().iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "spec":spec,"expected_batch_version":plan.next_batch_version()+cursor.completed_steps()as u64-1,
            "input_native_ids":flat,"expected_lanes":rows}));
        index += 1;
        let Some(claim) = claim else { continue };
        phases.push(json!({"phase":plan.phase(),"phase_sha256":plan.digest(),
            "program_sha256":program.digest().iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "claim":claim,"claim_sha256":claim.digest()?,"full_reserved_next_operation_id":plan.next_operation_id(),
            "full_reserved_next_step_id":plan.next_step_id()}));
        if claim.terminal() {
            ensure!(
                index == input.selected_steps.len(),
                "CPU replay found output after actual terminal boundary"
            );
            return Ok(
                json!({"passed":true,"CPU_only":true,"GPU_initialized":false,"network_opened":false,
                "terminal_or_runtime_authority":false,"source_scope":"checked CPU logical replay only; actual GPU proof separately required",
                "group":group.as_ref(),"steps":steps,"phases":phases,"final_claim":claim,
                "cleanup_lanes":group.lanes().iter().map(|k|Ok(LaneKey{generation:k.generation.checked_add(1).context("cleanup generation overflow")?,..*k})).collect::<Result<Vec<_>>>()?}),
            );
        }
        plan = cursor.next_phase(&claim)?;
        program = Program::new_batch_phase(plan.clone())?;
        cursor = BatchPhaseCursor::new(plan.clone());
    }
}
fn main() {
    let result = run();
    let value = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            json!({"passed":false,"CPU_only":true,"error":format!("{e:#}"),"GPU_initialized":false})
        }
    };
    serde_json::to_writer(std::io::stdout(), &value).unwrap();
    std::io::stdout().write_all(b"\n").unwrap();
    if result.is_err() {
        std::process::exit(1);
    }
}
