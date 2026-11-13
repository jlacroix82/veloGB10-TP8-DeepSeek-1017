//! Model-free masked-local-sampler oracle. All eight logical vocabulary shards
//! execute on one device; this is NOT an eight-node RDMA/model qualification.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use gb10_inference::dsv41::{
    gpu_tool_constraint::ToolConstraintKernels,
    tool_constraint::{DSML, EOS, VOCAB},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
const SHARD: usize = 16160;
const ARRAY_BOUND: usize = 1024 * 1024;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn words_hash(v: &[u32]) -> String {
    sha(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
fn emit(v: &Value) {
    println!("{}", serde_json::to_string(v).unwrap());
}
fn allowed(mask: &[u32], id: usize) -> bool {
    mask[id / 32] & (1 << (id % 32)) != 0
}
fn mask(ids: &[usize]) -> Vec<u32> {
    let mut m = vec![0; VOCAB / 32];
    for &i in ids {
        m[i / 32] |= 1 << (i % 32);
    }
    m
}
fn reference(logits: &[f32], mask: &[u32], rank: usize) -> [u32; 2] {
    let mut best: Option<(f32, u32)> = None;
    for (i, &x) in logits.iter().enumerate() {
        if allowed(mask, rank * SHARD + i) {
            assert!(x.is_finite() && x > f32::MIN);
            let id = (rank * SHARD + i) as u32;
            if best.is_none_or(|(v, j)| x > v || (x == v && id < j)) {
                best = Some((x, id));
            }
        }
    }
    match best {
        Some((v, id)) => [v.to_bits(), id],
        None => [f32::MIN.to_bits(), (rank * SHARD) as u32],
    }
}
fn global(pairs: &[[u32; 2]]) -> [u32; 2] {
    let mut winner = pairs[0];
    for &p in &pairs[1..] {
        let a = f32::from_bits(p[0]);
        let b = f32::from_bits(winner[0]);
        if a > b || (a == b && p[1] < winner[1]) {
            winner = p;
        }
    }
    winner
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let done = Arc::new(AtomicBool::new(false));
    let watch = done.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(90));
        if !watch.load(Ordering::SeqCst) {
            std::process::exit(124);
        }
    });
    let dev = CudaDevice::new(0)?;
    let stream = Arc::new(gb10_inference::gpu::fork_blocking_stream(&dev));
    let helper = ToolConstraintKernels::load(&dev, stream.clone())?;
    let initial: Value = serde_json::from_str(include_str!(
        "../tests/fixtures/dsv41_forced_initial_mask.json"
    ))?;
    let initial_words: Vec<u32> = serde_json::from_value(initial["words"].clone())?;
    ensure!(
        initial_words.len() == 4040
            && words_hash(&initial_words) == initial["mask_identity"]["mask_sha256"],
        "frozen real-CpuBridge initial mask differs"
    );
    let mut status = dev.alloc_zeros::<u32>(1)?;
    let mut pair = dev.alloc_zeros::<u32>(2)?;
    let mut checks = Vec::new();
    let positives = [
        "single_last_seven_empty",
        "ties_exclude_unallowed_larger",
        "signed_zero_first_native",
        "positive_min_subnormal",
        "negative_min_subnormal",
        "actual_frozen_echo_initial_mask",
    ];
    for (family, name) in positives.iter().enumerate() {
        let allowed_ids = match family {
            0 => vec![VOCAB - 1],
            1 => vec![17, SHARD + 7, 7 * SHARD + 3],
            2 => vec![9, SHARD + 9],
            3 => vec![43, 6 * SHARD + 15],
            4 => vec![88, 7 * SHARD + 4],
            _ => Vec::new(),
        };
        let bits = if family == 5 {
            initial_words.clone()
        } else {
            mask(&allowed_ids)
        };
        let device_bits = dev.htod_sync_copy(&bits)?;
        let mut observed = Vec::new();
        let mut expected = Vec::new();
        for rank in 0..8 {
            let mut logits = (0..SHARD)
                .map(|i| ((i % 29) as f32 - 14.) * 0.125)
                .collect::<Vec<_>>();
            // Forbidden logits are intentionally huge in every shard: selecting before
            // masking would produce a different global native token.
            for i in 0..SHARD {
                if !allowed(&bits, rank * SHARD + i) {
                    logits[i] = f32::MAX;
                }
            }
            for i in 0..SHARD {
                if allowed(&bits, rank * SHARD + i) {
                    logits[i] = match family {
                        0 => -3.,
                        1 => 2.,
                        2 => {
                            if rank == 0 {
                                -0.
                            } else {
                                0.
                            }
                        }
                        3 => {
                            if rank == 0 {
                                0.
                            } else {
                                f32::from_bits(1)
                            }
                        }
                        4 => {
                            if rank == 0 {
                                -f32::from_bits(1)
                            } else {
                                -f32::from_bits(2)
                            }
                        }
                        _ => ((rank * SHARD + i) % 47) as f32 - 23.,
                    };
                }
            }
            let wanted = reference(&logits, &bits, rank);
            let device_logits = dev.htod_sync_copy(&logits)?;
            dev.memset_zeros(&mut status)?;
            dev.memset_zeros(&mut pair)?;
            helper.enqueue(
                &stream,
                &device_logits,
                &device_bits,
                &mut pair,
                &mut status,
                rank,
            )?;
            dev.synchronize()?;
            let actual = dev.dtoh_sync_copy(&pair)?;
            let code = dev.dtoh_sync_copy(&status)?[0];
            ensure!(
                actual == wanted && code == 0,
                "masked local sampler {name} rank{rank} differs"
            );
            let actual: [u32; 2] = actual.try_into().unwrap();
            observed.push(actual);
            expected.push(wanted);
            let row = json!({"name":format!("{name}/rank{rank}"),"rank":rank,"actual_pair_words":actual,"expected_pair_words":wanted,"empty_allowed_local_shard":!(rank*SHARD..(rank+1)*SHARD).any(|id|allowed(&bits,id)),"mask_sha256_u32le":words_hash(&bits),"input_logits_sha256_f32le":words_hash(&logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>()),"GPU_status":code,"passed":true});
            emit(&json!({"event":"TOOL_CONSTRAINT_CHECK","check":row}));
            checks.push(row);
            drop(device_logits);
        }
        let winner = global(&observed);
        ensure!(
            winner == global(&expected)
                && f32::from_bits(winner[0]) > f32::MIN
                && allowed(&bits, winner[1] as usize),
            "global CPU fullpair composition selected forbidden/bottom identity"
        );
        let known = match family {
            0 => Some((VOCAB - 1) as u32),
            1 => Some(17),
            2 => Some(9),
            3 => Some((6 * SHARD + 15) as u32),
            4 => Some(88),
            _ => None,
        };
        if let Some(id) = known {
            ensure!(winner[1] == id, "closed-form global winner differs");
        }
        let row = json!({"name":format!("{name}/CPU_global_composition"),"all_eight_actual_local_pairs":observed,"actual_pair_words":winner,"expected_pair_words":global(&expected),"closed_form_expected_native_id":known,"actual_GPU_local_outputs":true,"actual_RDMA":false,"passed":true});
        emit(&json!({"event":"TOOL_CONSTRAINT_CHECK","check":row}));
        checks.push(row);
    }
    // Actual CUDA negative controls preserve sticky failures; no normal sample
    // could be published from these statuses. All buffers stay live through fences.
    let bits = mask(&[7]);
    let device_bits = dev.htod_sync_copy(&bits)?;
    for (name, badid, badvalue, initial_status, expected_status) in [
        ("allowed_bottom_sentinel_rejected", 7, f32::MIN, 0u32, 2u32),
        ("allowed_NaN_rejected", 7, f32::NAN, 0, 1),
        ("unallowed_NaN_still_rejected", 8, f32::NAN, 0, 1),
        ("sticky_status_preserved", 8, 0., 8, 8),
    ] {
        let mut logits = vec![0.; SHARD];
        logits[badid] = badvalue;
        let device_logits = dev.htod_sync_copy(&logits)?;
        let st = dev.htod_sync_copy(&[initial_status])?;
        status = st;
        helper.enqueue(
            &stream,
            &device_logits,
            &device_bits,
            &mut pair,
            &mut status,
            0,
        )?;
        dev.synchronize()?;
        let code = dev.dtoh_sync_copy(&status)?[0];
        ensure!(code == expected_status, "negative {name} lost sticky error");
        let row = json!({"name":name,"GPU_status":code,"expected_status":expected_status,"normal_response_permitted":false,"passed":true});
        emit(&json!({"event":"TOOL_CONSTRAINT_CHECK","check":row}));
        checks.push(row);
    }
    let logits = dev.htod_sync_copy(&vec![0.; SHARD])?;
    for (name, rank, short) in [
        ("invalid_rank_refused_before_launch", 8usize, false),
        ("short_mask_refused_before_launch", 0, true),
    ] {
        dev.memset_zeros(&mut status)?;
        pair = dev.htod_sync_copy(&[0x12345678u32, 0x87654321])?;
        let short_mask = dev.htod_sync_copy(&vec![0u32; if short { 1 } else { 4040 }])?;
        ensure!(
            helper
                .enqueue(&stream, &logits, &short_mask, &mut pair, &mut status, rank)
                .is_err(),
            "invalid sampler shape accepted"
        );
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&pair)? == [0x12345678, 0x87654321]
                && dev.dtoh_sync_copy(&status)? == [0],
            "prelaunch refusal mutated sentinel"
        );
        let row =
            json!({"name":name,"host_refusal":true,"device_pair_unchanged":true,"passed":true});
        emit(&json!({"event":"TOOL_CONSTRAINT_CHECK","check":row}));
        checks.push(row);
    }
    let other = Arc::new(gb10_inference::gpu::fork_blocking_stream(&dev));
    dev.memset_zeros(&mut status)?;
    pair = dev.htod_sync_copy(&[0x12345678u32, 0x87654321])?;
    ensure!(
        helper
            .enqueue(&other, &logits, &device_bits, &mut pair, &mut status, 0)
            .is_err(),
        "substituted blocking stream was accepted"
    );
    dev.synchronize()?;
    ensure!(
        dev.dtoh_sync_copy(&pair)? == [0x12345678, 0x87654321]
            && dev.dtoh_sync_copy(&status)? == [0],
        "stream refusal changed owned sentinels"
    );
    let row = json!({"name":"substituted_stream_refused_before_launch","host_refusal":true,"device_pair_unchanged":true,"passed":true});
    emit(&json!({"event":"TOOL_CONSTRAINT_CHECK","check":row}));
    checks.push(row);
    ensure!(checks.len() == 61, "exact frozen check count differs");
    ensure!(
        began.elapsed().as_secs_f64() < 90.,
        "internal wall exceeded"
    );
    done.store(true, Ordering::SeqCst);
    Ok(
        json!({"schema_version":1,"kind":"native1017_masked_local_greedy_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"actual_RDMA":false,"full_model_or_tool_quality_qualified":false,"benchmark":false,
 "source_id_hex":env!("DSV41_TOOL_CONSTRAINT_SOURCE_ID"),"ptx_sha256":sha(gb10_inference::dsv41::gpu_tool_constraint::PTX.as_bytes()),"oracle_source_sha256":sha(include_str!("dsv41_tool_constraint_oracle.rs").as_bytes()),"frozen_initial_mask_fixture_sha256":sha(include_str!("../tests/fixtures/dsv41_forced_initial_mask.json").as_bytes()),
 "checks":checks,"GPU_local_shards":48,"CPU_global_fullpair_compositions":6,"negative_controls":7,"logical_vocab_shard":SHARD,"native_vocab":VOCAB,"DSML_native_id":DSML,"EOS_native_id":EOS,
 "array_bound_bytes":ARRAY_BOUND,"native_timeout_seconds":90,"elapsed_seconds":began.elapsed().as_secs_f64(),"scope":"Independent GPU local masks plus CPU full-score/native-ID composition. No checkpoint payload, actual distributed maxloc, full model execution, served tool quality or speed qualification."}),
    )
}
fn main() {
    match run() {
        Ok(v) => emit(&v),
        Err(e) => {
            emit(
                &json!({"status":"FAIL","all_passed":false,"error":format!("{e:#}"),"model_execution":false,"benchmark":false}),
            );
            std::process::exit(1);
        }
    }
}
