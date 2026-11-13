//! Model-free grouped FP4 expert qualification. Unfused per-expert host dispatch
//! below is ONLY a numerical oracle; runtime GpuExperts has no route readback or
//! per-expert host GEMM loop. No checkpoint payload/serving is used.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use gb10_inference::dsv41::{
    gpu_experts::{ExpertBank, ExpertShape, ExpertWorkspace, GpuExperts, OwnedExpert, PTX},
    gpu_gemm::{GemmShape, GpuGemm},
    gpu_moe::MoeKernels,
    gpu_numerics::GpuNumerics,
};
use half::bf16;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const FIXTURE: &str = include_str!("../tests/fixtures/dsv41_experts_vectors.json");
const FIXTURE_SHA: &str = "fb2881bcc0631514b6dbef592fde4d6a4d75f070e361c7095d37ea80aee64d6f";
const WALL: u64 = 180;
const ARRAY_BOUND: usize = 512 * 1024 * 1024;
const ORDER: [usize; 16] = [7, 0, 15, 3, 12, 1, 9, 5, 14, 2, 11, 4, 13, 6, 10, 8];
#[derive(Clone, Deserialize)]
struct HostExpert {
    id: usize,
    w1: Vec<u8>,
    s1: Vec<u8>,
    w3: Vec<u8>,
    s3: Vec<u8>,
    w2: Vec<u8>,
    s2: Vec<u8>,
}
#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    dim: usize,
    inter: usize,
    rows: usize,
    input_words: Vec<u32>,
    route_ids: Vec<u32>,
    route_weights: Vec<f32>,
    logits_words: Vec<u32>,
    bias_words: Vec<u32>,
    experts: Vec<HostExpert>,
    expected_words: Vec<u32>,
    expected_sha256: String,
    negative_mismatch_counts: Value,
}
#[derive(Clone)]
struct Batch {
    dim: usize,
    rows: usize,
    input: Vec<bf16>,
    ids: Vec<u32>,
    weights: Vec<f32>,
    logits: Vec<f32>,
    bias: Vec<f32>,
}
impl Batch {
    fn from_case(c: &Case) -> Self {
        Self {
            dim: c.dim,
            rows: c.rows,
            input: c
                .input_words
                .iter()
                .map(|w| bf16::from_f32(f32::from_bits(*w)))
                .collect(),
            ids: c.route_ids.clone(),
            weights: c.route_weights.clone(),
            logits: c.logits_words.iter().map(|w| f32::from_bits(*w)).collect(),
            bias: c.bias_words.iter().map(|w| f32::from_bits(*w)).collect(),
        }
    }
    fn select(&self, indices: &[usize]) -> Self {
        let mut out = Self {
            dim: self.dim,
            rows: indices.len(),
            input: Vec::new(),
            ids: Vec::new(),
            weights: Vec::new(),
            logits: Vec::new(),
            bias: self.bias.clone(),
        };
        for &i in indices {
            out.input
                .extend_from_slice(&self.input[i * self.dim..(i + 1) * self.dim]);
            out.ids.extend_from_slice(&self.ids[i * 6..(i + 1) * 6]);
            out.weights
                .extend_from_slice(&self.weights[i * 6..(i + 1) * 6]);
            out.logits
                .extend_from_slice(&self.logits[i * 384..(i + 1) * 384]);
        }
        out
    }
}
fn fsha(x: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in x {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn mismatches(a: &[f32], b: &[f32]) -> usize {
    usize::from(a.len() != b.len())
        + a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
}
fn upload(dev: &Arc<CudaDevice>, e: &HostExpert) -> Result<OwnedExpert> {
    Ok(OwnedExpert {
        global_id: e.id,
        w1: dev.htod_sync_copy(&e.w1)?,
        s1: dev.htod_sync_copy(&e.s1)?,
        w3: dev.htod_sync_copy(&e.w3)?,
        s3: dev.htod_sync_copy(&e.s3)?,
        w2: dev.htod_sync_copy(&e.w2)?,
        s2: dev.htod_sync_copy(&e.s2)?,
    })
}
fn bank(dev: &Arc<CudaDevice>, shape: ExpertShape, experts: &[HostExpert]) -> Result<ExpertBank> {
    ExpertBank::oracle_sparse(
        dev,
        0,
        shape,
        experts
            .iter()
            .map(|e| upload(dev, e))
            .collect::<Result<_>>()?,
    )
}
fn status(dev: &Arc<CudaDevice>, s: &CudaSlice<u32>) -> Result<u32> {
    dev.synchronize()?;
    Ok(dev.dtoh_sync_copy(s)?[0])
}
fn grouped(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    k: &GpuExperts,
    moe: &MoeKernels,
    bank: &ExpertBank,
    batch: &Batch,
) -> Result<Vec<f32>> {
    let mut workspace = ExpertWorkspace::new(dev, bank.shape(), batch.rows)?;
    grouped_reusing(dev, s, k, moe, bank, batch, &mut workspace)
}
fn grouped_reusing(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    k: &GpuExperts,
    moe: &MoeKernels,
    bank: &ExpertBank,
    batch: &Batch,
    workspace: &mut ExpertWorkspace,
) -> Result<Vec<f32>> {
    let input = dev.htod_sync_copy(&batch.input)?;
    let logits = dev.htod_sync_copy(&batch.logits)?;
    let bias = dev.htod_sync_copy(&batch.bias)?;
    let mut ids = dev.alloc_zeros::<u32>(batch.rows * 6)?;
    let mut weights = dev.alloc_zeros::<f32>(batch.rows * 6)?;
    let mut upstream = dev.alloc_zeros::<u32>(1)?;
    moe.route(
        s,
        &logits,
        &bias,
        &mut ids,
        &mut weights,
        &mut upstream,
        batch.rows,
    )?;
    let pending = k.enqueue(s, bank, workspace, &input, &ids, &weights, &upstream)?;
    let out = pending.download_checked()?;
    drop(pending);
    // Oracle evidence only, AFTER actual grouped dispatch. Runtime dispatch
    // never copies routes to host or selects experts using this result.
    ensure!(status(dev, &upstream)? == 0, "upstream route failed");
    ensure!(
        dev.dtoh_sync_copy(&ids)? == batch.ids,
        "GPU route IDs differ from independent source program"
    );
    ensure!(
        mismatches(&dev.dtoh_sync_copy(&weights)?, &batch.weights) == 0,
        "GPU route weights differ"
    );
    Ok(out)
}
fn unfused(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &GpuGemm,
    n: &GpuNumerics,
    m: &MoeKernels,
    bank: &ExpertBank,
    b: &Batch,
) -> Result<Vec<f32>> {
    let mut out = vec![0_f32; b.rows * b.dim];
    let mut ids = bank.present_experts().to_vec();
    ids.sort_unstable();
    for id in ids {
        let mut rows = Vec::new();
        let mut rw = Vec::new();
        let mut x = Vec::new();
        for row in 0..b.rows {
            if let Some(top) = b.ids[row * 6..(row + 1) * 6]
                .iter()
                .position(|v| *v as usize == id)
            {
                rows.push(row);
                rw.push(b.weights[row * 6 + top]);
                x.extend_from_slice(&b.input[row * b.dim..(row + 1) * b.dim]);
            }
        }
        if rows.is_empty() {
            continue;
        }
        let w = bank.oracle_buffers(id)?;
        let input = dev.htod_sync_copy(&x)?;
        let a = n.quantize_activations_bf16(s, &input, rows.len(), b.dim)?;
        a.check()?;
        let shape = GemmShape {
            rows: rows.len(),
            outputs: bank.shape().inter(),
            inner: b.dim,
        };
        let gate = g.fp4_complete_bf16(
            s,
            unsafe { a.device_values_unchecked() },
            unsafe { a.device_scales_unchecked() },
            w[0],
            w[1],
            shape,
        )?;
        let up = g.fp4_complete_bf16(
            s,
            unsafe { a.device_values_unchecked() },
            unsafe { a.device_scales_unchecked() },
            w[2],
            w[3],
            shape,
        )?;
        gate.check()?;
        up.check()?;
        let weights = dev.htod_sync_copy(&rw)?;
        let mut silu = dev.alloc_zeros::<bf16>(rows.len() * shape.outputs)?;
        let mut flag = dev.alloc_zeros::<u32>(1)?;
        m.swiglu(
            s,
            unsafe { gate.device_values_unchecked() },
            unsafe { up.device_values_unchecked() },
            &weights,
            true,
            &mut silu,
            &mut flag,
            rows.len(),
            shape.outputs,
        )?;
        ensure!(status(dev, &flag)? == 0, "unfused SwiGLU rejected");
        let a = n.quantize_activations_bf16(s, &silu, rows.len(), shape.outputs)?;
        a.check()?;
        let result = g.fp4_complete_bf16(
            s,
            unsafe { a.device_values_unchecked() },
            unsafe { a.device_scales_unchecked() },
            w[4],
            w[5],
            GemmShape {
                rows: rows.len(),
                outputs: b.dim,
                inner: shape.outputs,
            },
        )?;
        let y = result.download_checked()?;
        for (token, &row) in rows.iter().enumerate() {
            for col in 0..b.dim {
                out[row * b.dim + col] += y[token * b.dim + col].to_f32();
            }
        }
    }
    Ok(out)
}
fn generated(expert: usize, dim: usize, inter: usize) -> HostExpert {
    fn matrix(expert: usize, branch: usize, out: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let mut w = vec![0_u8; out * k / 2];
        for row in 0..out {
            for j in 0..4 {
                let col = (row * 17 + branch * 31 + j * 997 + expert * 3) % k;
                let code = 1 + ((row + j * 3 + branch + expert) % 7) as u8
                    | if (row + j + expert) % 3 == 0 { 8 } else { 0 };
                let idx = row * (k / 2) + col / 2;
                if col & 1 == 0 {
                    w[idx] = (w[idx] & 0xf0) | code;
                } else {
                    w[idx] = (w[idx] & 15) | (code << 4);
                }
            }
        }
        let scale = (0..out * (k / 32))
            .map(|i| 123 + ((i * 7 + branch + expert) % 4) as u8)
            .collect();
        (w, scale)
    }
    let (w1, s1) = matrix(expert, 0, inter, dim);
    let (w3, s3) = matrix(expert, 1, inter, dim);
    let (w2, s2) = matrix(expert, 2, dim, inter);
    HostExpert {
        id: expert,
        w1,
        s1,
        w3,
        s3,
        w2,
        s2,
    }
}
fn native_batch(tiny: &Batch) -> Batch {
    let mut b = tiny.clone();
    b.dim = 5120;
    b.input = (0..b.rows * b.dim)
        .map(|i| bf16::from_f32(((i * 13 + 7) % 67) as f32 / 128.0 - 0.25))
        .collect();
    b
}
fn one_local_batch(dim: usize, rows: usize) -> Batch {
    let mut b = Batch {
        dim,
        rows,
        input: (0..rows * dim)
            .map(|i| bf16::from_f32(((i * 13 + 7) % 67) as f32 / 128.0 - 0.25))
            .collect(),
        ids: Vec::new(),
        weights: Vec::new(),
        logits: vec![36.; rows * 384],
        bias: vec![0.; 384],
    };
    let ids = [0_usize, 48, 49, 50, 51, 52];
    for row in 0..rows {
        for (j, &id) in ids.iter().enumerate() {
            b.logits[row * 384 + id] = ((j + 7) * (j + 7)) as f32;
        }
        b.ids.extend(ids.iter().rev().map(|v| *v as u32));
        b.weights
            .extend((7..=12).rev().map(|v| (v as f32 / 57.) * 1.5));
    }
    b
}
fn record(
    checks: &mut Vec<Value>,
    name: &str,
    rows: usize,
    dim: usize,
    out: &[f32],
    expected: &[f32],
    reference: &str,
) -> Result<()> {
    let errors = mismatches(out, expected);
    ensure!(errors == 0, "{name} differs at{errors} words");
    checks.push(json!({"case":name,"rows":rows,"dim":dim,"elements":out.len(),"bitwise_mismatches":errors,"actual_sha256":fsha(out),"reference_sha256":fsha(expected),"reference":reference,"passed":true}));
    Ok(())
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "zero-argument standalone component oracle"
    );
    let began = Instant::now();
    ensure!(
        format!("{:x}", Sha256::digest(FIXTURE.as_bytes())) == FIXTURE_SHA,
        "independent expert fixture changed"
    );
    let v: Value = serde_json::from_str(FIXTURE)?;
    let cases: Vec<Case> = serde_json::from_value(v["cases"].clone())?;
    ensure!(cases.len() == 2, "fixture coverage");
    let dev = CudaDevice::new(0)?;
    let s = gb10_inference::gpu::fork_blocking_stream(&dev);
    let k = GpuExperts::load(&dev, &s)?;
    let g = GpuGemm::load(&dev, &s)?;
    let n = GpuNumerics::load(&dev, &s)?;
    let m = MoeKernels::load(&dev, &s)?;
    let mut checks = Vec::new();
    for c in &cases {
        let shape = ExpertShape::oracle(c.dim, c.inter)?;
        let bank = bank(&dev, shape, &c.experts)?;
        let b = Batch::from_case(c);
        let expected: Vec<_> = c
            .expected_words
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();
        ensure!(
            fsha(&expected) == c.expected_sha256,
            "golden payload SHA mismatch"
        );
        let out = grouped(&dev, &s, &k, &m, &bank, &b)?;
        record(
            &mut checks,
            &format!("{}_CPU_full", c.name),
            b.rows,
            b.dim,
            &out,
            &expected,
            "independent NumPy complete CPU oracle",
        )?;
        let reference = unfused(&dev, &s, &g, &n, &m, &bank, &b)?;
        record(
            &mut checks,
            &format!("{}_unfused_full", c.name),
            b.rows,
            b.dim,
            &out,
            &reference,
            "qualified independent unfused GPU primitives",
        )?;
        for width in [1, 2, 4, 8, 16] {
            for (start, indices) in ORDER.chunks(width).enumerate() {
                let part = b.select(indices);
                let expected = indices
                    .iter()
                    .flat_map(|&row| out[row * b.dim..(row + 1) * b.dim].to_vec())
                    .collect::<Vec<_>>();
                let actual = grouped(&dev, &s, &k, &m, &bank, &part)?;
                record(
                    &mut checks,
                    &format!("{}_permute_C{width}_part{start}", c.name),
                    width,
                    b.dim,
                    &actual,
                    &expected,
                    "full CPU-checked C16 row association",
                )?;
            }
        }
    }
    let tiny = Batch::from_case(&cases[0]);
    // The same compact buffers must remain correct when local membership
    // shrinks to zero and later returns in a different request-row order.
    {
        let shape = ExpertShape::oracle(64, 32)?;
        let bank = bank(&dev, shape, &cases[0].experts)?;
        let mut workspace = ExpertWorkspace::new(&dev, shape, 16)?;
        let expected = cases[0]
            .expected_words
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect::<Vec<_>>();
        let actual = grouped_reusing(&dev, &s, &k, &m, &bank, &tiny, &mut workspace)?;
        record(
            &mut checks,
            "workspace_reuse_local",
            16,
            64,
            &actual,
            &expected,
            "independent CPU tiny golden, one reused workspace",
        )?;
        let mut remote = tiny.clone();
        remote.logits.fill(36.0);
        remote.bias.fill(0.0);
        for id in 48..54 {
            remote.bias[id] = 100.0;
        }
        remote.ids = (0..16).flat_map(|_| 48_u32..54).collect();
        remote.weights = vec![0.25; 16 * 6];
        let actual = grouped_reusing(&dev, &s, &k, &m, &bank, &remote, &mut workspace)?;
        record(
            &mut checks,
            "workspace_reuse_all_remote",
            16,
            64,
            &actual,
            &vec![0.0; 16 * 64],
            "all previous local maps/counts must be invalidated",
        )?;
        let reordered = tiny.select(&ORDER);
        let expected = ORDER
            .iter()
            .flat_map(|&row| expected[row * 64..(row + 1) * 64].to_vec())
            .collect::<Vec<_>>();
        let actual = grouped_reusing(&dev, &s, &k, &m, &bank, &reordered, &mut workspace)?;
        record(
            &mut checks,
            "workspace_reuse_reordered_local",
            16,
            64,
            &actual,
            &expected,
            "independent CPU rows after empty-to-active transition",
        )?;
    }
    let native = native_batch(&tiny);
    let shape = ExpertShape::native1017();
    let host: Vec<_> = [0, 17, 47]
        .into_iter()
        .map(|id| generated(id, 5120, 2304))
        .collect();
    let full_bank = bank(&dev, shape, &host)?;
    let baseline = grouped(&dev, &s, &k, &m, &full_bank, &native)?;
    let reference = unfused(&dev, &s, &g, &n, &m, &full_bank, &native)?;
    record(
        &mut checks,
        "native5120x2304_unfused_full",
        16,
        5120,
        &baseline,
        &reference,
        "qualified unfused GPU full native geometry,3 synthetic experts",
    )?;
    for width in [1, 2, 4, 8, 16] {
        for (start, indices) in ORDER.chunks(width).enumerate() {
            let b = native.select(indices);
            let expected = indices
                .iter()
                .flat_map(|&row| baseline[row * 5120..(row + 1) * 5120].to_vec())
                .collect::<Vec<_>>();
            let actual = grouped(&dev, &s, &k, &m, &full_bank, &b)?;
            record(
                &mut checks,
                &format!("native_permute_C{width}_part{start}"),
                width,
                5120,
                &actual,
                &expected,
                "unfused-checked native C16 row association",
            )?;
        }
    }
    let b = one_local_batch(5120, 128);
    let actual = grouped(&dev, &s, &k, &m, &full_bank, &b)?;
    let expected = unfused(&dev, &s, &g, &n, &m, &full_bank, &b)?;
    record(
        &mut checks,
        "native_prefill128_one_selected_local",
        128,
        5120,
        &actual,
        &expected,
        "qualified unfused native GEMM,128 compacted rows",
    )?;
    drop(full_bank);
    // Actual upstream GPU tie policy; all384 scores tie, lowest six IDs. This
    // uses six independently owned native buffers, not aliased dummy weights.
    let shape = ExpertShape::oracle(64, 32)?;
    let host: Vec<_> = (0..6).map(|id| generated(id, 64, 32)).collect();
    let tie_bank = bank(&dev, shape, &host)?;
    let mut tie = tiny.select(&[7, 0, 15, 3]);
    tie.logits.fill(36.);
    tie.bias.fill(0.);
    tie.ids = (0..tie.rows).flat_map(|_| 0_u32..6).collect();
    tie.weights = vec![0.25; tie.rows * 6];
    let actual = grouped(&dev, &s, &k, &m, &tie_bank, &tie)?;
    let expected = unfused(&dev, &s, &g, &n, &m, &tie_bank, &tie)?;
    record(
        &mut checks,
        "GPU_route_all384_tie_lowest_six",
        4,
        64,
        &actual,
        &expected,
        "qualified GPU routing tie policy plus unfused native math",
    )?;
    let empty = ExpertBank::oracle_sparse(&dev, 7, shape, vec![])?;
    let b = one_local_batch(64, 16);
    let actual = grouped(&dev, &s, &k, &m, &empty, &b)?;
    record(
        &mut checks,
        "rank7_empty_local_partial",
        16,
        64,
        &actual,
        &vec![0.; 16 * 64],
        "all selected IDs are remote; exact F32 positive zero",
    )?;
    // Sticky validation controls execute the real new GPU path. Only oracle
    // inputs are corrupted; each attempt starts with its own upstream flag.
    let mut negative = serde_json::Map::new();
    // Independent exact midpoint anchors: reducing routed partials to BF16
    // before this add loses the low bit that selects the next BF16 value.
    let routed = [1.00390625_f32, -1.00390625, 257.0, -257.0];
    let shared = [0.00390625_f32, -0.00390625, 1.0, -1.0].map(bf16::from_f32);
    let expected = [0x3f81_u16, 0xbf81, 0x4381, 0xc381].map(|b| bf16::from_bits(b).to_f32());
    let premature = routed
        .iter()
        .zip(shared)
        .map(|(r, s)| bf16::from_f32(bf16::from_f32(*r).to_f32() + s.to_f32()).to_f32())
        .collect::<Vec<_>>();
    ensure!(
        mismatches(&premature, &expected) == 4,
        "routed premature-BF16 midpoint control lost sensitivity"
    );
    let reduced = dev.htod_sync_copy(&routed)?;
    let shared = dev.htod_sync_copy(&shared)?;
    let mut output = dev.alloc_zeros::<bf16>(4)?;
    let mut flag = dev.alloc_zeros::<u32>(1)?;
    let combined = k.combine_reduced_shared(&s, &reduced, &shared, &mut output, &mut flag)?;
    let actual = combined
        .download_checked()?
        .iter()
        .map(|b| b.to_f32())
        .collect::<Vec<_>>();
    drop(combined);
    record(
        &mut checks,
        "reducedF32_sharedBF16_midpoints",
        1,
        4,
        &actual,
        &expected,
        "independent exact IEEE midpoint BF16 bit anchors",
    )?;
    negative.insert("premature_routed_BF16_cast".into(),json!({"bitwise_mismatches":4,"rejected":true,"scope":"independent CPU wrong-schedule control"}));
    for (name, mode, expected) in [
        ("invalid_route_id", 0, 8),
        ("duplicate_route_id", 1, 8),
        ("nonfinite_route_weight", 2, 8),
        ("upstream_route_failure", 3, 8),
        ("missing_selected_bank", 4, 16),
        ("nonfinite_input", 5, 1),
        ("invalid_native_scale", 6, 2),
        ("native_overflow", 7, 4),
    ] {
        let mut b = one_local_batch(64, 1);
        let mut host = vec![generated(0, 64, 32)];
        if mode == 0 {
            b.ids[0] = 384;
        }
        if mode == 1 {
            b.ids[0] = b.ids[1];
        }
        if mode == 2 {
            b.weights[0] = f32::NAN;
        }
        if mode == 5 {
            b.input[0] = bf16::NAN;
        }
        if mode == 6 {
            host[0].s1[0] = 255;
        }
        if mode == 7 {
            host[0].s1.fill(254);
            host[0].s3.fill(254);
            host[0].s2.fill(254);
        }
        if mode == 4 {
            host.clear();
        }
        let bank = bank(&dev, shape, &host)?;
        let input = dev.htod_sync_copy(&b.input)?;
        let ids = dev.htod_sync_copy(&b.ids)?;
        let weights = dev.htod_sync_copy(&b.weights)?;
        let upstream = dev.htod_sync_copy(&[if mode == 3 { 1_u32 } else { 0 }])?;
        let mut ws = ExpertWorkspace::new(&dev, shape, 1)?;
        let pending = k.enqueue(&s, &bank, &mut ws, &input, &ids, &weights, &upstream)?;
        let code = pending.status_code()?;
        ensure!(
            code & expected != 0 && pending.check().is_err(),
            "negative {name} escaped status0x{code:x}"
        );
        negative.insert(
            name.into(),
            json!({"status":code,"required_bit":expected,"rejected":true}),
        );
    }
    for (name, routed, shared, required) in [
        ("combine_nonfinite_routed", f32::NAN, bf16::ONE, 1_u32),
        ("combine_nonfinite_shared", 1.0_f32, bf16::NAN, 1_u32),
        ("combine_overflow", f32::MAX, bf16::MAX, 4_u32),
    ] {
        let reduced = dev.htod_sync_copy(&[routed])?;
        let shared = dev.htod_sync_copy(&[shared])?;
        let mut output = dev.alloc_zeros::<bf16>(1)?;
        let mut flag = dev.alloc_zeros::<u32>(1)?;
        let combined = k.combine_reduced_shared(&s, &reduced, &shared, &mut output, &mut flag)?;
        let code = combined.status_code()?;
        ensure!(
            code & required != 0 && combined.check().is_err(),
            "combine rejection control {name} escaped"
        );
        negative.insert(
            name.into(),
            json!({"status":code,"required_bit":required,"rejected":true}),
        );
    }
    ensure!(
        cases[0].negative_mismatch_counts["early_projection"]
            .as_u64()
            .context("negative count")?
            > 0
            && cases[0].negative_mismatch_counts["route_after_round"]
                .as_u64()
                .context("negative count")?
                > 0
            && cases[1].negative_mismatch_counts["reverse_sum"]
                .as_u64()
                .context("negative count")?
                > 0,
        "independent CPU negative controls lost sensitivity"
    );
    negative.insert("independent_CPU_wrong_schedules".into(),json!({"varied_routes":cases[0].negative_mismatch_counts,"cancellation":cases[1].negative_mismatch_counts,"source":"NumPy generator; deliberately wrong intermediate-round/route-weight/sum order","rejected":true}));
    ensure!(
        checks.len() == 105 && negative.len() == 13,
        "expert oracle coverage changed {} {}",
        checks.len(),
        negative.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "expert component wall bound"
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_experts_gpu_component_oracle","status":"PASS","all_passed":true,"model_free":true,"model_execution":false,"serve_supported":false,"tensor_parallel_engine":false,"RDMA":false,"checkpoint_payload_read":false,
        "source_id_hex":env!("DSV41_EXPERTS_SOURCE_ID"),"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),"dependencies":k.dependencies(),"fixture_sha256":FIXTURE_SHA,"elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL,"explicit_array_allocation_upper_bound_bytes":ARRAY_BOUND},"checks":checks,"negative_controls":negative,
        "scope":"Grouped local routed-expert primitive only. Synthetic native FP4 weights, official1017 geometry and explicit sparse fixture bank. Runtime has GPU route maps/no CPU route readback; separate unfused oracle intentionally uses CPU dispatch. No shared expert, distributed reduction, complete checkpoint load, graphs, serving or speed qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("grouped expert oracle hard wall limit");
        std::process::exit(124);
    });
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_experts_gpu_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_experts_gpu_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
