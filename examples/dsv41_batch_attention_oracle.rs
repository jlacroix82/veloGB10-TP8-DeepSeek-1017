//! PRIVATE model-free packed-row oracle. ONE packed attention launch per case
//! versus the qualified source-order single-row attention path; no model/RDMA.
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use gb10_inference::dsv41::{
    batch_attention_layout::{AttentionPhase, LaneAttentionLayout},
    chunk_plan::{ChunkCursor, LayerChunk},
    config::Dsv41Config,
    gpu_attention::AttentionKernels,
    gpu_module::Module,
    scheduler::LaneKey,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const D: usize = 512;
const H: usize = 8;
const WALL: u64 = 120;
const BOUND: usize = 32 * 1024 * 1024;
const PACKED: &str = include_str!("../src/ptx/dsv41_batch_attention.ptx");
const LAYOUT: &str = include_str!("../src/ptx/dsv41_batch_layout.ptx");
const OLD: &str = include_str!("../src/ptx/dsv41_kv_store.ptx");
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn bits(x: &[bf16]) -> String {
    sha(&x
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>())
}
fn config() -> Result<Dsv41Config> {
    let f: Value = serde_json::from_str(include_str!("../tests/fixtures/dsv41_1017_config.json"))?;
    Dsv41Config::from_values(&f["hf_config"], &f["inference_config"])
}
fn plan(start: usize, tokens: usize, layer: usize) -> Result<LayerChunk> {
    let key = LaneKey {
        boot: Uuid::from_u128(16128),
        slot: 0,
        generation: 1,
    };
    let mut c = ChunkCursor::new(&config()?, key.boot, 1)?;
    c.reset(key)?;
    let mut at = 0;
    while at < start {
        let n = (start - at).min(128);
        let t = c.begin(key, at, n)?;
        for l in 0..40 {
            let p = c.layer(&t, l)?;
            c.commit_layer(&p)?;
        }
        c.commit(&t)?;
        at += n;
    }
    let t = c.begin(key, start, tokens)?;
    for l in 0..layer {
        let p = c.layer(&t, l)?;
        c.commit_layer(&p)?;
    }
    c.layer(&t, layer)
}
#[derive(Clone)]
struct Lane {
    slot: usize,
    start: usize,
    tokens: usize,
    phase: AttentionPhase,
}
fn cases() -> Vec<(String, Vec<Lane>)> {
    let mut out = Vec::new();
    let starts = [1, 31, 63, 75, 86, 126, 127, 128, 255, 511];
    for b in [1, 2, 4, 8, 16] {
        out.push((
            format!("decode_B{b}"),
            (0..b)
                .map(|i| Lane {
                    slot: (i * 5 + 3) % 16,
                    start: starts[i % starts.len()],
                    tokens: 1,
                    phase: AttentionPhase::Decode,
                })
                .collect(),
        ));
    }
    out.push((
        "ragged_full75_plus53".into(),
        vec![
            Lane {
                slot: 7,
                start: 0,
                tokens: 75,
                phase: AttentionPhase::Prefill {
                    full_prompt_tokens: 75,
                },
            },
            Lane {
                slot: 2,
                start: 0,
                tokens: 53,
                phase: AttentionPhase::Prefill {
                    full_prompt_tokens: 53,
                },
            },
        ],
    ));
    for (start, tokens) in [(0, 8), (16, 8), (64, 1)] {
        out.push((
            format!("prefill_B16_start{start}_T{tokens}"),
            (0..16)
                .map(|i| Lane {
                    slot: (i * 5 + 3) % 16,
                    start,
                    tokens,
                    phase: AttentionPhase::Prefill {
                        full_prompt_tokens: 75 + i,
                    },
                })
                .collect(),
        ));
    }
    out
}
fn global(slot: usize, pos: usize) -> usize {
    pos - (pos % 128 + 128 - slot) % 128
}
fn value(slot: usize, p: usize, d: usize, pattern: usize) -> bf16 {
    let tag = slot * 10003 + p * 17;
    bf16::from_f32(if d == 0 {
        ((tag % 41) as i32 - 20) as f32 * 0.25
    } else {
        (((tag + d * 7 + pattern * 13) % 61) as i32 - 30) as f32 * 0.25
    })
}
fn query(slot: usize, p: usize, pattern: usize) -> Vec<bf16> {
    let mut q = vec![bf16::ZERO; H * D];
    for h in 0..H {
        q[h * D] = bf16::from_f32((h + 1) as f32 * 0.125);
        q[h * D + 64] = bf16::from_f32(((slot + p + pattern + h) % 7) as f32 * 0.125 - 0.375);
    }
    q
}
fn copy<T>(
    s: &CudaStream,
    dst: &mut CudaSlice<T>,
    d: usize,
    src: &CudaSlice<T>,
    p: usize,
    n: usize,
) -> Result<()> {
    ensure!(
        d + n <= dst.len() && p + n <= src.len(),
        "oracle exact device copy range"
    );
    ensure!(
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *dst.device_ptr() + (d * std::mem::size_of::<T>()) as u64,
                *src.device_ptr() + (p * std::mem::size_of::<T>()) as u64,
                n * std::mem::size_of::<T>(),
                s.stream,
            )
        } == sys::CUresult::CUDA_SUCCESS,
        "oracle device copy failed"
    );
    Ok(())
}
struct Fence(Arc<CudaDevice>);
impl Drop for Fence {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}
fn source_indices(
    m: &Module,
    s: &CudaStream,
    p: &LayerChunk,
    l: &LaneAttentionLayout,
    out: &mut CudaSlice<i32>,
    status: &mut CudaSlice<u32>,
) -> Result<()> {
    let (mut o, mut e, mut start, mut tokens, mut ratio, mut w, mut c, mut lw, mut lc, mut phase) = (
        *out.device_ptr(),
        *status.device_ptr(),
        p.ticket.start as u64,
        p.ticket.tokens as u64,
        p.ratio as u64,
        p.window.snapshot_rows as u64,
        l.physical_compressed_rows as u64,
        l.logical_window_columns as u64,
        l.logical_compressed_columns as u64,
        if matches!(l.phase, AttentionPhase::Decode) {
            1u64
        } else {
            0
        },
    );
    let mut a = [
        &mut o as *mut _ as *mut c_void,
        &mut e as *mut _ as *mut c_void,
        &mut start as *mut _ as *mut c_void,
        &mut tokens as *mut _ as *mut c_void,
        &mut ratio as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void,
        &mut lw as *mut _ as *mut c_void,
        &mut lc as *mut _ as *mut c_void,
        &mut phase as *mut _ as *mut c_void,
    ];
    unsafe {
        m.launch(
            "dsv41_batch_source_indices",
            s,
            (out.len().div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut a,
        )
    }
}
fn old_indices(
    m: &Module,
    s: &CudaStream,
    lane: &Lane,
    ratio: usize,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let (full, start, tokens, w, cols) = match lane.phase {
        AttentionPhase::Prefill {
            full_prompt_tokens: n,
        } => (n, 0, n, n, n.min(128)),
        AttentionPhase::Decode => (lane.start + 1, lane.start, 1, 128, 128),
    };
    let (mut op, mut st, mut t, mut r, mut wr, mut wc, mut c) = (
        *out.device_ptr(),
        start as u64,
        tokens as u64,
        ratio as u64,
        w as u64,
        cols as u64,
        if ratio == 0 { 0 } else { full / ratio } as u64,
    );
    let mut a = [
        &mut op as *mut _ as *mut c_void,
        &mut st as *mut _ as *mut c_void,
        &mut t as *mut _ as *mut c_void,
        &mut r as *mut _ as *mut c_void,
        &mut wr as *mut _ as *mut c_void,
        &mut wc as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void,
    ];
    unsafe {
        m.launch(
            "dsv41_kv_store_indices",
            s,
            (out.len().div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut a,
        )
    }
}
fn packed(
    m: &Module,
    s: &CudaStream,
    q: &CudaSlice<bf16>,
    kv: &CudaSlice<bf16>,
    sink: &CudaSlice<f32>,
    ids: &CudaSlice<i32>,
    meta: &CudaSlice<u64>,
    out: &mut CudaSlice<bf16>,
    status: &mut CudaSlice<u32>,
    rows: usize,
    stride: usize,
) -> Result<()> {
    let (mut qp, mut kp, mut op, mut sp, mut ip, mut mp, mut k, mut n, mut h, mut scale, mut ep) = (
        *q.device_ptr(),
        *kv.device_ptr(),
        *out.device_ptr(),
        *sink.device_ptr(),
        *ids.device_ptr(),
        *meta.device_ptr(),
        stride as u64,
        (kv.len() / D) as u64,
        H as i32,
        (D as f32).sqrt().recip(),
        *status.device_ptr(),
    );
    let mut a = [
        &mut qp as *mut _ as *mut c_void,
        &mut kp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut sp as *mut _ as *mut c_void,
        &mut ip as *mut _ as *mut c_void,
        &mut mp as *mut _ as *mut c_void,
        &mut k as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut scale as *mut _ as *mut c_void,
        &mut ep as *mut _ as *mut c_void,
    ];
    unsafe {
        m.launch(
            "dsv41_packed_sparse_attention",
            s,
            (rows as u32, 1, 1),
            (256, 1, 1),
            88320,
            &mut a,
        )
    }
}
fn run() -> Result<Value> {
    let began = Instant::now();
    let dev = CudaDevice::new(0)?;
    let s = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);
    let layout = Module::load(
        &dev,
        &s,
        LAYOUT,
        &["dsv41_batch_source_indices", "dsv41_batch_pad_indices"],
        "dsv41_batch_layout_identity",
        u64::from_str_radix(env!("DSV41_BATCH_LAYOUT_SOURCE_ID"), 16)?,
        1,
    )?;
    let batch = Module::load(
        &dev,
        &s,
        PACKED,
        &["dsv41_packed_sparse_attention"],
        "dsv41_batch_attention_identity",
        u64::from_str_radix(env!("DSV41_BATCH_ATTENTION_SOURCE_ID"), 16)?,
        1,
    )?;
    batch.set_dynamic_smem("dsv41_packed_sparse_attention", 88320)?;
    let old = Module::load(
        &dev,
        &s,
        OLD,
        &["dsv41_kv_store_indices"],
        "dsv41_kv_store_identity",
        u64::from_str_radix(env!("DSV41_KV_STORE_SOURCE_ID"), 16)?,
        1,
    )?;
    let reference = AttentionKernels::load(&dev, &s)?;
    let sink = dev.htod_sync_copy(&(0..H).map(|h| h as f32 * 0.125 - 0.5).collect::<Vec<_>>())?;
    let mut checks = Vec::new();
    let mut peak = 0;
    let mut total_queries = 0;
    for (name, lanes) in cases() {
        for ratio in [0, 1, 2] {
            for pattern in 0..2 {
                ensure!(
                    began.elapsed() < Duration::from_secs(WALL),
                    "bounded oracle wall"
                );
                let layer = match ratio {
                    0 => 0,
                    1 => 20,
                    _ => 2,
                };
                let mut logical = Vec::new();
                let mut qh = Vec::new();
                let mut kvh = Vec::new();
                let mut meta = Vec::new();
                let mut expected = Vec::new();
                let mut padmeta = Vec::new();
                let mut tight_count = 0usize;
                let mut stride = 0usize;
                for lane in &lanes {
                    let p = plan(lane.start, lane.tokens, layer)?;
                    let l = LaneAttentionLayout::new(&p, lane.phase)?;
                    let first = kvh.len() / D;
                    for row in 0..p.window.snapshot_rows {
                        for d in 0..D {
                            kvh.push(value(
                                lane.slot,
                                p.window.snapshot_global_base + row,
                                d,
                                pattern,
                            ));
                        }
                    }
                    for row in 0..l.physical_compressed_rows {
                        for d in 0..D {
                            kvh.push(value(lane.slot, 10000 + row, d, pattern));
                        }
                    }
                    let ids = l.all_selected_indices()?;
                    let k = l.logical_topk();
                    stride = stride.max(k);
                    for row in 0..lane.tokens {
                        qh.extend(query(lane.slot, lane.start + row, pattern));
                        meta.extend([
                            (first * D) as u64,
                            l.physical_kv_rows() as u64,
                            k as u64,
                            lane.slot as u64,
                        ]);
                        padmeta.extend([
                            (tight_count + row * k) as u64,
                            k as u64,
                            l.physical_kv_rows() as u64,
                        ]);
                    }
                    tight_count += ids.len();
                    expected.extend(ids);
                    logical.push((p, l));
                }
                let rows = qh.len() / (H * D);
                let q = dev.htod_sync_copy(&qh)?;
                let kv = dev.htod_sync_copy(&kvh)?;
                let metadata = dev.htod_sync_copy(&meta)?;
                let pm = dev.htod_sync_copy(&padmeta)?;
                let mut ids = dev.alloc_zeros::<i32>(rows * stride)?;
                let mut tight = dev.alloc_zeros::<i32>(tight_count)?;
                let mut output = dev.alloc_zeros::<bf16>(rows * H * D)?;
                let mut status = dev.alloc_zeros::<u32>(1)?;
                dev.memset_zeros(&mut status)?;
                let _f = Fence(dev.clone());
                let mut at = 0;
                for (p, l) in &logical {
                    let mut one = dev.alloc_zeros::<i32>(p.ticket.tokens * l.logical_topk())?;
                    let _f = Fence(dev.clone());
                    source_indices(&layout, &s, p, l, &mut one, &mut status)?;
                    let n = one.len();
                    copy(&s, &mut tight, at, &one, 0, n)?;
                    dev.synchronize()?;
                    at += n;
                }
                let (mut op, mut ip, mut mp, mut ep, mut nr, mut k, mut nt) = (
                    *ids.device_ptr(),
                    *tight.device_ptr(),
                    *pm.device_ptr(),
                    *status.device_ptr(),
                    rows as u64,
                    stride as u64,
                    tight_count as u64,
                );
                let mut a = [
                    &mut op as *mut _ as *mut c_void,
                    &mut ip as *mut _ as *mut c_void,
                    &mut mp as *mut _ as *mut c_void,
                    &mut ep as *mut _ as *mut c_void,
                    &mut nr as *mut _ as *mut c_void,
                    &mut k as *mut _ as *mut c_void,
                    &mut nt as *mut _ as *mut c_void,
                ];
                unsafe {
                    layout.launch(
                        "dsv41_batch_pad_indices",
                        &s,
                        ((rows * stride).div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut a,
                    )
                }?;
                packed(
                    &batch,
                    &s,
                    &q,
                    &kv,
                    &sink,
                    &ids,
                    &metadata,
                    &mut output,
                    &mut status,
                    rows,
                    stride,
                )?;
                dev.synchronize()?;
                ensure!(
                    dev.dtoh_sync_copy(&status)? == [0],
                    "packed index/attention status failed"
                );
                let native_indices = dev.dtoh_sync_copy(&tight)?;
                ensure!(
                    native_indices == expected,
                    "full packed native indices differ from source CPU contract"
                );
                let padded_indices = dev.dtoh_sync_copy(&ids)?;
                let mut expected_padded = Vec::with_capacity(rows * stride);
                for row in 0..rows {
                    let first = padmeta[row * 3] as usize;
                    let width = padmeta[row * 3 + 1] as usize;
                    expected_padded.extend_from_slice(&expected[first..first + width]);
                    expected_padded.resize(expected_padded.len() + stride - width, -1);
                }
                ensure!(
                    padded_indices == expected_padded,
                    "complete packed index rows/common padding differ from source contract"
                );
                let actual = dev.dtoh_sync_copy(&output)?;
                let mut reference_outputs = Vec::new();
                let mut rowbase = 0;
                let base_bytes = (qh.len() + kvh.len() + rows * H * D) * 2
                    + (rows * stride + tight_count) * 4
                    + (meta.len() + padmeta.len()) * 8
                    + 4
                    + H * 4;
                for (lane, (p, l)) in lanes.iter().zip(&logical) {
                    let (full, window, source_rows) = match lane.phase {
                        AttentionPhase::Prefill {
                            full_prompt_tokens: n,
                        } => (n, n, n),
                        AttentionPhase::Decode => (lane.start + 1, 128, 1),
                    };
                    let comp = if ratio == 0 { 0 } else { full / ratio };
                    let mut kh = vec![bf16::ZERO; (window + comp) * D];
                    for j in 0..window {
                        if matches!(lane.phase, AttentionPhase::Prefill { .. })
                            || lane.start >= 127
                            || j <= lane.start
                        {
                            let position = if matches!(lane.phase, AttentionPhase::Prefill { .. }) {
                                j
                            } else if lane.start < 127 {
                                j
                            } else {
                                global(j, lane.start)
                            };
                            for d in 0..D {
                                kh[j * D + d] = value(lane.slot, position, d, pattern);
                            }
                        }
                    }
                    for j in 0..comp {
                        for d in 0..D {
                            kh[(window + j) * D + d] = value(lane.slot, 10000 + j, d, pattern);
                        }
                    }
                    let old_kv = dev.htod_sync_copy(&kh)?;
                    let mut all_ids = dev.alloc_zeros::<i32>(source_rows * l.logical_topk())?;
                    let mut one_ids = dev.alloc_zeros::<i32>(l.logical_topk())?;
                    let mut one_q = dev.alloc_zeros::<bf16>(H * D)?;
                    let mut one_out = dev.alloc_zeros::<bf16>(H * D)?;
                    let _f = Fence(dev.clone());
                    old_indices(&old, &s, lane, ratio, &mut all_ids)?;
                    let bytes = base_bytes
                        + kh.len() * 2
                        + (all_ids.len() + one_ids.len()) * 4
                        + 2 * H * D * 2;
                    peak = peak.max(bytes);
                    ensure!(bytes <= BOUND, "explicit32MiB GPU array bound");
                    for row in 0..lane.tokens {
                        let source_row = if matches!(lane.phase, AttentionPhase::Prefill { .. }) {
                            lane.start + row
                        } else {
                            0
                        };
                        copy(
                            &s,
                            &mut one_ids,
                            0,
                            &all_ids,
                            source_row * l.logical_topk(),
                            l.logical_topk(),
                        )?;
                        copy(&s, &mut one_q, 0, &q, (rowbase + row) * H * D, H * D)?;
                        reference.sparse(
                            &s,
                            &one_q,
                            &old_kv,
                            &sink,
                            &one_ids,
                            &mut one_out,
                            &mut status,
                            1,
                            1,
                            H,
                            window + comp,
                            l.logical_topk(),
                        )?;
                        dev.synchronize()?;
                        reference_outputs.extend(dev.dtoh_sync_copy(&one_out)?);
                    }
                    rowbase += lane.tokens;
                    let _ = p;
                }
                ensure!(
                    dev.dtoh_sync_copy(&status)? == [0],
                    "qualified source control status failed"
                );
                let mismatches = actual
                    .iter()
                    .zip(&reference_outputs)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(
                    actual.len() == reference_outputs.len() && mismatches == 0,
                    "packed vs qualified source attention BF16 bits differ"
                );
                checks.push(json!({"case":name,"ratio":ratio,"pattern":pattern,"lanes":lanes.len(),"query_rows":rows,"elements":actual.len(),"logical_row_strides":logical.iter().map(|(_,l)|l.logical_topk()).collect::<Vec<_>>(),"common_index_stride":stride,"packed_kernel_launches":1,"native_index_elements":expected.len(),"padded_index_elements":padded_indices.len(),"padded_indices_sha256_i32le":sha(&padded_indices.iter().flat_map(|n|n.to_le_bytes()).collect::<Vec<_>>()),"native_indices_sha256_i32le":sha(&native_indices.iter().flat_map(|n|n.to_le_bytes()).collect::<Vec<_>>()),"packed_output_sha256_bf16le":bits(&actual),"qualified_source_output_sha256_bf16le":bits(&reference_outputs),"bit_mismatches":mismatches,"passed":true}));
                total_queries += rows;
            }
        }
    }
    let mut negatives = Vec::new();
    for name in [
        "NaN_query",
        "NaN_KV",
        "index_out_of_bounds",
        "logical_topk_exceeds_stride",
        "KV_base_out_of_bounds",
        "NaN_sink",
    ] {
        let mut qh = vec![bf16::ZERO; H * D];
        let mut kh = vec![bf16::ZERO; 2 * D];
        let mut ih = vec![-1i32; 128];
        ih[0] = 0;
        ih[127] = 1;
        let mut mh = vec![0u64, 2, 128, 3];
        let mut sh = vec![0f32; H];
        let expected = match name {
            "NaN_query" => {
                qh[0] = bf16::from_bits(0x7fc0);
                2
            }
            "NaN_KV" => {
                kh[0] = bf16::from_bits(0x7fc0);
                2
            }
            "index_out_of_bounds" => {
                ih[0] = 2;
                1
            }
            "logical_topk_exceeds_stride" => {
                mh[2] = 129;
                4
            }
            "KV_base_out_of_bounds" => {
                mh[0] = 512;
                4
            }
            _ => {
                sh[0] = f32::NAN;
                2
            }
        };
        let q = dev.htod_sync_copy(&qh)?;
        let kv = dev.htod_sync_copy(&kh)?;
        let ids = dev.htod_sync_copy(&ih)?;
        let meta = dev.htod_sync_copy(&mh)?;
        let sinks = dev.htod_sync_copy(&sh)?;
        let mut output = dev.alloc_zeros::<bf16>(H * D)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut status)?;
        let _f = Fence(dev.clone());
        packed(
            &batch,
            &s,
            &q,
            &kv,
            &sinks,
            &ids,
            &meta,
            &mut output,
            &mut status,
            1,
            128,
        )?;
        dev.synchronize()?;
        let observed = dev.dtoh_sync_copy(&status)?[0];
        ensure!(
            observed == expected,
            "packed attention negative control status differs"
        );
        negatives.push(json!({"case":name,"expected_status":expected,"observed_status":observed,"passed":true}));
    }
    for (name, p, phase) in [
        (
            "invalid_decode_multi_token",
            plan(1, 2, 0)?,
            AttentionPhase::Decode,
        ),
        (
            "unimplemented_learned_selection_513",
            plan(512, 1, 20)?,
            AttentionPhase::Decode,
        ),
    ] {
        let mut logical = LaneAttentionLayout::new(
            &p,
            AttentionPhase::Prefill {
                full_prompt_tokens: p.ticket.end,
            },
        )?;
        logical.phase = phase;
        let mut ids = dev.alloc_zeros::<i32>(p.ticket.tokens * logical.logical_topk())?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut status)?;
        let _f = Fence(dev.clone());
        source_indices(&layout, &s, &p, &logical, &mut ids, &mut status)?;
        dev.synchronize()?;
        let observed = dev.dtoh_sync_copy(&status)?[0];
        ensure!(observed == 2, "packed source-index negative status differs");
        negatives.push(
            json!({"case":name,"expected_status":2,"observed_status":observed,"passed":true}),
        );
    }
    ensure!(
        checks.len() == 54 && total_queries == 2586,
        "complete B1..16/ragged/partial/phase coverage required"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_packed_attention_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"RDMA":false,"benchmark":false,"checks":checks,"negative_controls":negatives,"qualified_source_query_rows":total_queries,"peak_explicit_GPU_array_bytes":peak,"maximum_GPU_array_bytes":BOUND,"wall_seconds":WALL,"elapsed_seconds":began.elapsed().as_secs_f64(),"identities":{"batch_attention_source_id":env!("DSV41_BATCH_ATTENTION_SOURCE_ID"),"batch_attention_ptx_sha256":sha(PACKED.as_bytes()),"batch_layout_source_id":env!("DSV41_BATCH_LAYOUT_SOURCE_ID"),"batch_layout_ptx_sha256":sha(LAYOUT.as_bytes()),"qualified_attention_ptx_sha256":sha(gb10_inference::dsv41::gpu_attention::PTX.as_bytes()),"qualified_KV_indices_ptx_sha256":sha(OLD.as_bytes())},"scope":"One packed attention launch per case versus qualified source-order individual-query controls; exact BF16 bits, explicit logical tile widths. Does not run typed store/full model, real weights, network, tools or a speed benchmark."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("packed attention oracle hard deadline");
        std::process::exit(124);
    });
    let result = match run() {
        Ok(v) => v,
        Err(e) => {
            json!({"schema_version":1,"kind":"dsv41_packed_attention_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_packed_case_matrix_has_exact_independent_row_counts() {
        let c = cases();
        assert_eq!(c.len(), 9);
        assert_eq!(
            c.iter()
                .map(|(_, ls)| ls.iter().map(|l| l.tokens).sum::<usize>())
                .sum::<usize>()
                * 6,
            2586
        );
        for (_, ls) in c {
            let slots = ls
                .iter()
                .map(|l| l.slot)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(slots.len(), ls.len());
            assert!(ls.iter().map(|l| l.tokens).sum::<usize>() <= 128);
        }
    }
}
