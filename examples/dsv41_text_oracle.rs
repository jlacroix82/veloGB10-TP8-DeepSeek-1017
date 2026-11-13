//! Synthetic text-head helper oracle. No weights, head projection or model run.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, DeviceSlice};
use gb10_inference::dsv41::gpu_text::{TextKernels, PTX, VOCAB_SHARD};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;
fn sha(v: &[u32]) -> String {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    let k = TextKernels::load(&dev, &stream)?;
    let mut checks = Vec::new();
    for rank in 0..8 {
        for case in [
            "negative_ties",
            "tail_winner",
            "cross_warp_tie",
            "signed_zero",
            "ascending",
        ] {
            let (mut input, index, score) = match case {
                "negative_ties" => (
                    (0..VOCAB_SHARD)
                        .map(|i| -((i % 239) as f32) - 2.)
                        .collect::<Vec<_>>(),
                    0,
                    -2.,
                ),
                "tail_winner" => (vec![-1.; VOCAB_SHARD], VOCAB_SHARD - 1, 3.5),
                "cross_warp_tie" => (vec![-1.; VOCAB_SHARD], 31, 2.5),
                "signed_zero" => (vec![-0.; VOCAB_SHARD], 0, -0.),
                _ => (
                    (0..VOCAB_SHARD)
                        .map(|i| (i as f32 - 8000.) / 128.)
                        .collect::<Vec<_>>(),
                    VOCAB_SHARD - 1,
                    (VOCAB_SHARD as f32 - 1. - 8000.) / 128.,
                ),
            };
            if case == "tail_winner" {
                input[index] = score;
            }
            if case == "cross_warp_tie" {
                for i in [31, 256, VOCAB_SHARD - 1] {
                    input[i] = score;
                }
            }
            if case == "signed_zero" {
                input[1] = 0.;
            }
            let input = dev.htod_sync_copy(&input)?;
            let mut pair = dev.alloc_zeros::<u32>(2)?;
            let mut status = dev.alloc_zeros::<u32>(1)?;
            k.maxloc(&stream, &input, &mut pair, &mut status, rank)?;
            dev.synchronize()?;
            let out = dev.dtoh_sync_copy(&pair)?;
            let expected = [score.to_bits(), (rank * VOCAB_SHARD + index) as u32];
            ensure!(
                dev.dtoh_sync_copy(&status)? == [0] && out == expected,
                "native maxloc {case} rank{rank}: {out:?} vs{expected:?}"
            );
            checks.push(json!({"case":case,"rank":rank,"elements":VOCAB_SHARD,"actual_pair":out,"expected_pair":expected,"pair_u32le_sha256":sha(&out),"passed":true}));
        }
    }
    for rows in [1, 2, 16, 75, 128] {
        let input = (0..rows * 5120)
            .map(|i| bf16::from_f32(((i * 19 + 3) % 127) as f32 / 16. - 63. / 16.))
            .collect::<Vec<_>>();
        let expected = input[(rows - 1) * 5120..].to_vec();
        let input = dev.htod_sync_copy(&input)?;
        let mut output = dev.alloc_zeros::<bf16>(5120)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        k.last(&stream, &input, &mut output, &mut status, rows)?;
        dev.synchronize()?;
        let out = dev.dtoh_sync_copy(&output)?;
        ensure!(
            out == expected && dev.dtoh_sync_copy(&status)? == [0],
            "last position bit copy failed"
        );
        checks.push(json!({"case":"last_position","rows":rows,"elements":5120,"bit_mismatches":0,"passed":true}));
    }
    let mut negative = serde_json::Map::new();
    for (name, value, all, bit) in [
        ("NaN_logit", f32::NAN, false, 1),
        ("Inf_logit", f32::INFINITY, false, 1),
        ("all_NaN_logits", f32::NAN, true, 3),
    ] {
        let mut input = vec![0.; VOCAB_SHARD];
        if all {
            input.fill(value)
        } else {
            input[31] = value;
        }
        let input = dev.htod_sync_copy(&input)?;
        let mut pair = dev.alloc_zeros::<u32>(2)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        k.maxloc(&stream, &input, &mut pair, &mut status, 0)?;
        dev.synchronize()?;
        let status = dev.dtoh_sync_copy(&status)?[0];
        ensure!(status & bit == bit, "missing invalid logit status");
        negative.insert(name.into(), json!(status));
    }
    let input = dev.htod_sync_copy(&vec![0.; VOCAB_SHARD])?;
    let mut pair = dev.alloc_zeros::<u32>(2)?;
    let mut status = dev.htod_sync_copy(&[4u32])?;
    k.maxloc(&stream, &input, &mut pair, &mut status, 0)?;
    dev.synchronize()?;
    ensure!(dev.dtoh_sync_copy(&status)? == [4], "sticky status cleared");
    negative.insert("upstream_status_preserved".into(), json!(true));
    let other = dev.fork_default_stream()?;
    ensure!(
        k.maxloc(&other, &input, &mut pair, &mut status, 0).is_err(),
        "another stream allowed"
    );
    negative.insert("other_stream_rejected".into(), json!(true));
    ensure!(
        k.maxloc(&stream, &input, &mut pair, &mut status, 8)
            .is_err(),
        "rank8 allowed"
    );
    negative.insert("rank8_rejected".into(), json!(true));
    let mut hidden = vec![bf16::ZERO; 5120];
    hidden[5119] = bf16::NAN;
    let hidden = dev.htod_sync_copy(&hidden)?;
    let mut out = dev.alloc_zeros::<bf16>(5120)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    k.last(&stream, &hidden, &mut out, &mut status, 1)?;
    dev.synchronize()?;
    ensure!(dev.dtoh_sync_copy(&status)? == [1], "NaN hidden allowed");
    negative.insert("NaN_hidden".into(), json!(1));
    ensure!(
        k.last(&stream, &hidden, &mut out, &mut status, 129)
            .is_err(),
        "rows129 allowed"
    );
    negative.insert("prefill129_rejected".into(), json!(true));
    ensure!(
        checks.len() == 45 && negative.len() == 8 && began.elapsed().as_secs_f64() < 60.,
        "oracle bounds"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_text_helper_gpu_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"checkpoint_payload_read":false,"serve_supported":false,"RDMA":false,"source_id_hex":env!("DSV41_TEXT_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),"checks":checks,"negative_controls":negative,"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":60,"explicit_array_allocation_upper_bound_bytes":8388608},"scope":"Local native vocabulary maximum/last hidden copy only; no real head weights, full model, network or benchmark qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(65));
        std::process::exit(124);
    });
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            println!(
                "{}",
                json!({"kind":"dsv41_text_helper_gpu_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
            );
            std::process::exit(1);
        }
    }
}
