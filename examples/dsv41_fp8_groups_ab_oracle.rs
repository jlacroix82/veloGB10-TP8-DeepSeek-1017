//! Model-free exact parallel-K32 contribution and ordered-finish preservation gate. No model/RDMA/speed claim.
//! Parent owns exclusivity, 8GiB available RAM, 80C/90C and independent deadline.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

const BASE_PTX: &str = include_str!("../tests/fixtures/dsv41_gemm_a0_baseline.ptx.txt");
const BASE_SHA: &str = "de713c5b97c4b64b70004ee7a2e137cb47cd8186d9c4076503e855607bf1d1eb";
const BASE_ID: u64 = 0xd4562c0fe99e98bd;
const CAND_PTX: &str = gb10_inference::dsv41::gpu_gemm::PTX;
const CAND_SHA: &str = "3430ec49415b7f793dc30880ebff79ba4a01b9a7a38571e4dd161e40f55fcc02";
fn cand_sha() -> String {
    CAND_SHA.to_string()
}
const CAND_ID: u64 = 0x1f1d4fbde0b9e247;
fn cand_id() -> u64 {
    CAND_ID
}
const WALL_SECONDS: u64 = 180;
const GPU_BOUND: usize = 16 * 1024 * 1024;
const HOST_BOUND: usize = 128 * 1024 * 1024;
const GUARD_BYTES: usize = 64;
const FLOAT_GUARD: u32 = 0x7fc052a1;
const BF_GUARD: u16 = 0x7fc1;
const STATUS_GUARD: u32 = 0x5ab39cd1;
static CLEANUP_FAILURES: AtomicUsize = AtomicUsize::new(0);
const CPU_FIXTURE: &str = include_str!("../tests/fixtures/dsv41_mxf8_ab_cpu_reference.json");
const CONTRIBUTION_FIXTURE: &str =
    include_str!("../tests/fixtures/dsv41_fp8_groups_contribution_reference.json");

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn sha32(words: &[u32]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn sha16(words: &[u16]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    MixedSafe,
    Fixed(u8, u8),
    SubtileFallback(bool),
    BadScale(bool),
    NegativeZero,
    Cancellation,
    BadByte(bool, u8, usize),
    Bf16Overflow,
    OverflowThenRecovery,
    ActivationBasis,
    WeightBasis,
    ProductBasis,
    Sticky,
}
#[derive(Clone, Debug)]
struct Spec {
    name: String,
    m: usize,
    n: usize,
    k: usize,
    pattern: Pattern,
}
impl Spec {
    fn named(name: &str, m: usize, n: usize, k: usize, pattern: Pattern) -> Self {
        Self {
            name: name.into(),
            m,
            n,
            k,
            pattern,
        }
    }
}
fn specs() -> Vec<Spec> {
    let mut v = Vec::new();
    for m in [1, 2, 8, 16, 17, 31, 128] {
        for n in [16, 32, 33, 48] {
            for k in [32, 64, 5120] {
                v.push(Spec::named(
                    &format!("mixed_safe_M{m}_N{n}_K{k}"),
                    m,
                    n,
                    k,
                    Pattern::MixedSafe,
                ));
            }
        }
    }
    for (a, w) in [
        (111, 111),
        (111, 143),
        (143, 111),
        (143, 143),
        (110, 143),
        (144, 111),
        (143, 110),
        (111, 144),
        (0, 254),
        (254, 0),
        (0, 0),
        (254, 254),
    ] {
        v.push(Spec::named(
            &format!("fixed_activation_scale_{a}_weight_scale_{w}"),
            31,
            33,
            64,
            Pattern::Fixed(a, w),
        ));
    }
    for first in [false, true] {
        v.push(Spec::named(
            &format!("separate_subtile_fallback_first_{first}"),
            31,
            48,
            64,
            Pattern::SubtileFallback(first),
        ));
    }
    for activation in [false, true] {
        v.push(Spec::named(
            &format!("invalid_255_scale_activation_{activation}"),
            31,
            33,
            64,
            Pattern::BadScale(activation),
        ));
    }
    v.push(Spec::named(
        "negative_zero_inputs",
        17,
        33,
        64,
        Pattern::NegativeZero,
    ));
    v.push(Spec::named(
        "within_group_cancellation",
        31,
        48,
        5120,
        Pattern::Cancellation,
    ));
    for activation in [false, true] {
        for code in [0x7f, 0xff] {
            for byte in 0..4 {
                v.push(Spec::named(
                    &format!("NaN_activation_{activation}_code_{code}_byte_{byte}"),
                    31,
                    33,
                    64,
                    Pattern::BadByte(activation, code, byte),
                ));
            }
        }
    }
    v.push(Spec::named(
        "finite_F32_overflows_BF16",
        17,
        33,
        32,
        Pattern::Bf16Overflow,
    ));
    v.push(Spec::named(
        "F32_overflow_then_next_group_recovery",
        17,
        33,
        64,
        Pattern::OverflowThenRecovery,
    ));
    v.push(Spec::named(
        "all_254_finite_activation_codes_basis",
        128,
        33,
        64,
        Pattern::ActivationBasis,
    ));
    v.push(Spec::named(
        "all_254_finite_weight_codes_basis",
        17,
        48,
        5120,
        Pattern::WeightBasis,
    ));
    v.push(Spec::named(
        "all_finite_product_code_basis",
        31,
        33,
        5120,
        Pattern::ProductBasis,
    ));
    v.push(Spec::named(
        "sticky_status_preserved",
        17,
        33,
        64,
        Pattern::Sticky,
    ));
    v
}
#[derive(Clone)]
struct Pack {
    spec: Spec,
    a: Vec<u8>,
    sa: Vec<u8>,
    w: Vec<u8>,
    sw: Vec<u8>,
    initial_status: u32,
}
fn finite_code(i: usize) -> u8 {
    let i = i % 254;
    if i < 127 {
        i as u8
    } else {
        (i + 1) as u8
    }
}
fn pack(s: &Spec) -> Pack {
    let g = s.k / 32;
    let small = [0_u8, 0x30, 0x38, 0x3c, 0x40, 0x44];
    let mut a = (0..s.m * s.k)
        .map(|i| {
            let z = mix(i as u64 ^ 0x11a35f);
            small[z as usize % 6] | if z >> 17 & 1 != 0 { 0x80 } else { 0 }
        })
        .collect::<Vec<_>>();
    let mut w = (0..s.n * s.k)
        .map(|i| {
            let z = mix(i as u64 ^ 0x9ab35f);
            small[z as usize % 6] | if z >> 19 & 1 != 0 { 0x80 } else { 0 }
        })
        .collect::<Vec<_>>();
    let safe = [111_u8, 112, 143, 127, 140, 113, 142, 128];
    let mut sa = (0..s.m * g)
        .map(|i| {
            let r = i / g;
            let k = i % g;
            safe[(r * 5 + k * 3 + (r / 8) * 7) % 8]
        })
        .collect::<Vec<_>>();
    let mut sw = (0..s.n.div_ceil(32) * g)
        .map(|i| {
            let r = i / g;
            let k = i % g;
            safe[(r * 2 + k * 5) % 8]
        })
        .collect::<Vec<_>>();
    let mut initial_status = 0;
    match s.pattern {
        Pattern::MixedSafe => {}
        Pattern::Fixed(aa, ww) => {
            sa.fill(aa);
            sw.fill(ww);
        }
        Pattern::SubtileFallback(first) => {
            for r in 0..s.m {
                if ((r % 16) < 8) == first {
                    for k in 0..g {
                        sa[r * g + k] = if (r + k) % 2 == 0 { 110 } else { 144 };
                    }
                }
            }
        }
        Pattern::BadScale(activation) => {
            if activation {
                *sa.last_mut().unwrap() = 255
            } else {
                *sw.last_mut().unwrap() = 255
            }
        }
        Pattern::NegativeZero => {
            for (i, v) in a.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 0 } else { 0x80 };
            }
            w.fill(0x38);
            sa.fill(143);
            sw.fill(111);
        }
        Pattern::Cancellation => {
            for r in 0..s.m {
                for k in 0..s.k {
                    a[r * s.k + k] = if k % 2 == 0 { 0x38 } else { 0xb8 };
                }
            }
            w.fill(0x38);
            // A single group residual prevents a whole-matrix zero canary.
            for r in 0..s.m {
                a[r * s.k + (r % g) * 32 + 31] = 0;
            }
        }
        Pattern::BadByte(activation, code, byte) => {
            if activation {
                a[(s.m - 1) * s.k + s.k - 4 + byte] = code
            } else {
                w[(s.n - 1) * s.k + s.k - 4 + byte] = code
            }
        }
        Pattern::Bf16Overflow => {
            a.fill(0);
            w.fill(0);
            sa.fill(246);
            sw.fill(127);
            for r in 0..s.m {
                for k in 0..16 {
                    a[r * s.k + k] = 0x50;
                }
                a[r * s.k + 16] = 0x38;
            }
            for n in 0..s.n {
                for k in 0..16 {
                    w[n * s.k + k] = 0x48;
                }
                w[n * s.k + 16] = 0xb8;
            }
        }
        Pattern::OverflowThenRecovery => {
            a.fill(0);
            w.fill(0);
            sa.fill(127);
            sw.fill(127);
            for r in 0..s.m {
                sa[r * g] = 254;
                for k in 0..32 {
                    a[r * s.k + k] = 0x44;
                }
                a[r * s.k + 32] = 0x38;
            }
            for n in 0..s.n {
                for k in 0..32 {
                    w[n * s.k + k] = 0x44;
                }
                w[n * s.k + 32] = 0x38;
            }
            for n in 0..s.n.div_ceil(32) {
                sw[n * g] = 254;
            }
        }
        Pattern::ActivationBasis | Pattern::WeightBasis | Pattern::ProductBasis => {
            a.fill(0);
            w.fill(0);
            for r in 0..s.m {
                for k in 0..g {
                    a[r * s.k + k * 32] = match s.pattern {
                        Pattern::ActivationBasis => finite_code(r * g + k),
                        Pattern::ProductBasis => finite_code(r * 17 + k),
                        _ => 0x38,
                    };
                }
            }
            for n in 0..s.n {
                for k in 0..g {
                    w[n * s.k + k * 32] = match s.pattern {
                        Pattern::WeightBasis => finite_code(n * g + k),
                        Pattern::ProductBasis => finite_code(n * 31 + k * 3),
                        _ => 0x38,
                    };
                }
            }
        }
        Pattern::Sticky => initial_status = 0x20,
    }
    Pack {
        spec: s.clone(),
        a,
        sa,
        w,
        sw,
        initial_status,
    }
}
fn f8(code: u8, status: &mut u32) -> f32 {
    let mag = code & 127;
    if mag == 127 {
        *status |= 1;
        return 0.0;
    }
    let sign = if code & 128 != 0 { -1.0 } else { 1.0 };
    let exponent = mag >> 3;
    let mantissa = mag & 7;
    sign * if exponent == 0 {
        mantissa as f32 * 2_f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2_f32.powi(exponent as i32 - 7)
    }
}
fn scale(code: u8, status: &mut u32) -> f32 {
    if code == 255 {
        *status |= 2;
        1.0
    } else {
        f32::from_bits(if code == 0 {
            1 << 22
        } else {
            (code as u32) << 23
        })
    }
}
struct Cpu {
    f32_bits: Vec<u32>,
    bf16_bits: Vec<u16>,
    status_f32: u32,
    status_bf16: u32,
    contributions: Vec<u32>,
}
fn reference(p: &Pack) -> Cpu {
    let s = &p.spec;
    let groups = s.k / 32;
    let mut status = p.initial_status;
    let a = p.a.iter().map(|&x| f8(x, &mut status)).collect::<Vec<_>>();
    let w = p.w.iter().map(|&x| f8(x, &mut status)).collect::<Vec<_>>();
    let sa =
        p.sa.iter()
            .map(|&x| scale(x, &mut status))
            .collect::<Vec<_>>();
    let sw =
        p.sw.iter()
            .map(|&x| scale(x, &mut status))
            .collect::<Vec<_>>();
    let mut f32_bits = vec![0; s.m * s.n];
    let mut bf16_bits = vec![0; s.m * s.n];
    let mut bf_status = 0;
    let mut contributions = vec![0u32; s.m * s.n * groups];
    for row in 0..s.m {
        for out in 0..s.n {
            let mut accum = 0_f32;
            for group in 0..groups {
                let mut raw = 0_f32;
                for k in group * 32..(group + 1) * 32 {
                    raw += a[row * s.k + k] * w[out * s.k + k];
                }
                let corrected = (raw * sa[row * groups + group]) * sw[(out / 32) * groups + group];
                contributions[(group * s.m + row) * s.n + out] = corrected.to_bits();
                let next = accum + corrected;
                if !corrected.is_finite() || !next.is_finite() {
                    status |= 4;
                    accum = 0.0
                } else {
                    accum = next;
                }
            }
            f32_bits[row * s.n + out] = accum.to_bits();
            let rounded = bf16::from_f32(accum);
            if !rounded.is_finite() {
                bf_status |= 4;
                bf16_bits[row * s.n + out] = 0
            } else {
                bf16_bits[row * s.n + out] = rounded.to_bits();
            }
        }
    }
    Cpu {
        f32_bits,
        bf16_bits,
        status_f32: status,
        status_bf16: status | bf_status,
        contributions,
    }
}
#[derive(Clone, Copy, Debug)]
enum Mode {
    F32,
    Bf16,
    Dual,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32_only",
            Self::Bf16 => "BF16_only",
            Self::Dual => "F32_and_BF16",
        }
    }
    fn f32(self) -> bool {
        !matches!(self, Self::Bf16)
    }
    fn bf16(self) -> bool {
        !matches!(self, Self::F32)
    }
}
struct Engine {
    name: &'static str,
    kernel: CudaFunction,
    contributions: Option<CudaFunction>,
    finish: Option<CudaFunction>,
}
impl Engine {
    fn load(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        name: &'static str,
        ptx: &str,
        id: u64,
    ) -> Result<Self> {
        dev.load_ptx(
            Ptx::from_src(ptx),
            name,
            if name == "fp8_parallel_groups_candidate" {
                &[
                    "dsv41_gemm_identity",
                    "dsv41_gemm_fp8_group32",
                    "dsv41_gemm_fp8_group32_contributions",
                    "dsv41_gemm_group32_ordered_finish",
                ]
            } else {
                &["dsv41_gemm_identity", "dsv41_gemm_fp8_group32"]
            },
        )?;
        let mut identity = dev.htod_sync_copy(&[0_u64; 2])?;
        unsafe {
            dev.get_func(name, "dsv41_gemm_identity")
                .context("identity function")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut identity,),
                )
        }?;
        if let Err(error) = dev.synchronize() {
            std::mem::forget(identity);
            return Err(error.into());
        }
        ensure!(
            dev.dtoh_sync_copy(&identity)? == [id, 1],
            "actual {name} source/ABI differs"
        );
        Ok(Self {
            name,
            contributions: dev.get_func(name, "dsv41_gemm_fp8_group32_contributions"),
            finish: dev.get_func(name, "dsv41_gemm_group32_ordered_finish"),
            kernel: dev
                .get_func(name, "dsv41_gemm_fp8_group32")
                .context("FP8 function")?,
        })
    }
}
struct DeviceInputs {
    dev: Arc<CudaDevice>,
    values: Vec<CudaSlice<u8>>,
    host: Vec<Vec<u8>>,
}
impl DeviceInputs {
    fn new(dev: &Arc<CudaDevice>, p: &Pack) -> Result<Self> {
        let host = [&p.a, &p.sa, &p.w, &p.sw]
            .into_iter()
            .map(|x| {
                let mut v = vec![255; GUARD_BYTES];
                v.extend_from_slice(x);
                v.extend([255; GUARD_BYTES]);
                v
            })
            .collect::<Vec<_>>();
        let values = host
            .iter()
            .map(|x| dev.htod_sync_copy(x))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            dev: dev.clone(),
            values,
            host,
        })
    }
    fn pointer(&self, i: usize) -> u64 {
        *self.values[i].device_ptr() + GUARD_BYTES as u64
    }
    fn check_immutable(&self) -> Result<usize> {
        let mut changed = 0;
        for (d, h) in self.values.iter().zip(&self.host) {
            changed += self
                .dev
                .dtoh_sync_copy(d)?
                .iter()
                .zip(h)
                .filter(|(a, b)| a != b)
                .count();
        }
        Ok(changed)
    }
}
impl Drop for DeviceInputs {
    fn drop(&mut self) {
        if self.dev.bind_to_thread().is_err() || self.dev.synchronize().is_err() {
            CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
            for v in self.values.drain(..) {
                std::mem::forget(v);
            }
        }
    }
}
struct Output {
    dev: Arc<CudaDevice>,
    f32: Option<CudaSlice<u32>>,
    bf16: Option<CudaSlice<u16>>,
    status: Option<CudaSlice<u32>>,
    partials: Option<CudaSlice<u32>>,
}
impl Output {
    fn new(dev: &Arc<CudaDevice>, count: usize, status: u32, partials: usize) -> Result<Self> {
        Ok(Self {
            dev: dev.clone(),
            partials: if partials > 0 {
                Some(dev.htod_sync_copy(&vec![0x3df3a5c7u32; partials + 32])?)
            } else {
                None
            },
            f32: Some(dev.htod_sync_copy(&vec![FLOAT_GUARD; count + 32])?),
            bf16: Some(dev.htod_sync_copy(&vec![BF_GUARD; count + 64])?),
            status: Some(dev.htod_sync_copy(&[STATUS_GUARD, status, STATUS_GUARD])?),
        })
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        if self.dev.bind_to_thread().is_err() || self.dev.synchronize().is_err() {
            CLEANUP_FAILURES.fetch_add(1, Ordering::Relaxed);
            if let Some(v) = self.f32.take() {
                std::mem::forget(v);
            }
            if let Some(v) = self.bf16.take() {
                std::mem::forget(v);
            }
            if let Some(v) = self.status.take() {
                std::mem::forget(v);
            }
            if let Some(v) = self.partials.take() {
                std::mem::forget(v);
            }
        }
    }
}
fn canonical_contribution(word: u32) -> u32 {
    if (word & 0x7f800000) == 0x7f800000 && (word & 0x007fffff) != 0 {
        0x7fc00000
    } else {
        word
    }
}
fn differences<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    usize::from(a.len() != b.len()) + a.iter().zip(b).filter(|(a, b)| a != b).count()
}
struct Actual {
    f32: Vec<u32>,
    bf16: Vec<u16>,
    status: u32,
    guard_mismatches: usize,
    contributions: Option<Vec<u32>>,
    contributions_physical_sha256: Option<String>,
}
fn run_engine(
    dev: &Arc<CudaDevice>,
    stream: &CudaStream,
    e: &Engine,
    input: &DeviceInputs,
    s: &Spec,
    mode: Mode,
    initial_status: u32,
) -> Result<Actual> {
    let count = s.m * s.n;
    let partial_count = if e.contributions.is_some() {
        count * (s.k / 32)
    } else {
        0
    };
    let out = Output::new(dev, count, initial_status, partial_count)?;
    let fp = if mode.f32() {
        *out.f32.as_ref().unwrap().device_ptr() + 64
    } else {
        0
    };
    let bp = if mode.bf16() {
        *out.bf16.as_ref().unwrap().device_ptr() + 64
    } else {
        0
    };
    let sp = *out.status.as_ref().unwrap().device_ptr() + 4;
    if let (Some(group), Some(finish), Some(partials)) =
        (&e.contributions, &e.finish, &out.partials)
    {
        let pp = *partials.device_ptr() + 64;
        unsafe {
            group.clone().launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (
                        s.n.div_ceil(128) as u32,
                        s.m.div_ceil(16) as u32,
                        (s.k / 32) as u32,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    pp,
                    sp,
                    input.pointer(0),
                    input.pointer(1),
                    input.pointer(2),
                    input.pointer(3),
                    s.m as i32,
                    s.n as i32,
                    s.k as i32,
                ),
            )?;
            finish.clone().launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (count.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (fp, bp, sp, pp, s.m as i32, s.n as i32, s.k as i32),
            )?;
        }
    } else {
        unsafe {
            e.kernel.clone().launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (s.n.div_ceil(128) as u32, s.m.div_ceil(16) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    fp,
                    bp,
                    sp,
                    input.pointer(0),
                    input.pointer(1),
                    input.pointer(2),
                    input.pointer(3),
                    s.m as i32,
                    s.n as i32,
                    s.k as i32,
                ),
            )
        }?;
    }
    dev.synchronize()?;
    let f = dev.dtoh_sync_copy(out.f32.as_ref().unwrap())?;
    let b = dev.dtoh_sync_copy(out.bf16.as_ref().unwrap())?;
    let status = dev.dtoh_sync_copy(out.status.as_ref().unwrap())?;
    let mut partial_guard_mismatches = 0;
    let mut contributions_physical_sha256 = None;
    let contributions = if let Some(p) = &out.partials {
        let v = dev.dtoh_sync_copy(p)?;
        partial_guard_mismatches = v[..16]
            .iter()
            .chain(&v[16 + partial_count..])
            .filter(|&&x| x != 0x3df3a5c7)
            .count();
        let physical = &v[16..16 + partial_count];
        contributions_physical_sha256 = Some(sha32(physical));
        let mut logical = vec![0; partial_count];
        for group in 0..s.k / 32 {
            for row in 0..s.m {
                for column in 0..s.n {
                    logical[(group * s.m + row) * s.n + column] =
                        physical[(group * s.n + column) * s.m + row];
                }
            }
        }
        Some(logical)
    } else {
        None
    };
    let guard_mismatches = partial_guard_mismatches
        + f[..16]
            .iter()
            .chain(&f[16 + count..])
            .filter(|&&v| v != FLOAT_GUARD)
            .count()
        + b[..32]
            .iter()
            .chain(&b[32 + count..])
            .filter(|&&v| v != BF_GUARD)
            .count()
        + usize::from(status[0] != STATUS_GUARD)
        + usize::from(status[2] != STATUS_GUARD);
    Ok(Actual {
        f32: f[16..16 + count].to_vec(),
        bf16: b[32..32 + count].to_vec(),
        status: status[1],
        guard_mismatches,
        contributions,
        contributions_physical_sha256,
    })
}
fn identity() -> Value {
    json!({"baseline_source_id_hex":format!("{BASE_ID:016x}"),"baseline_ptx_sha256":BASE_SHA,
    "candidate_source_id_hex":format!("{:016x}",cand_id()),"candidate_ptx_sha256":cand_sha(),
    "candidate_cuda_source_sha256":sha(include_bytes!("../kernels/dsv41_gemm.cu")),
    "CPU_reference_fixture_sha256":sha(CPU_FIXTURE.as_bytes()),"candidate_algorithm":"parallel original K32 contributions then source-order checked F32 finish; no reordered sum","scratch_guard_bytes_each_side":64,"scratch_fill_u32":0x3df3a5c7u32,"intermediate_comparison":"full nativeF32 bits, only NaNpayload canonicalized; finalF32/BF16 exactbits/status mandatory",
    "CPU_contribution_fixture_sha256":sha(CONTRIBUTION_FIXTURE.as_bytes()),
    "module_names":["fp8_group_a0_baseline","fp8_parallel_groups_candidate"],"wall_seconds":WALL_SECONDS,
    "GPU_array_bound_bytes":GPU_BOUND,"host_array_bound_bytes":HOST_BOUND,"GPU_executed_in_metadata_mode":false,
    "model_payload_loaded":false,"actual_RDMA":false,"full_model_execution":false,"speed_qualified":false,
    "comparison":"bitwise F32/BF16/status; CPU exact diagnostic families, original ascending K32 scale/cast order"})
}
// Pure CPU expectations, not GPU evidence. This mode exits before creating a
// CUDA device, stream, module, allocation or network endpoint.
fn cpu_records() -> Vec<Value> {
    let mut records = Vec::new();
    for s in specs() {
        let p = pack(&s);
        let cpu = reference(&p);
        let count = s.m * s.n;
        for mode in [Mode::F32, Mode::Bf16, Mode::Dual] {
            let f = if mode.f32() {
                cpu.f32_bits.clone()
            } else {
                vec![FLOAT_GUARD; count]
            };
            let b = if mode.bf16() {
                cpu.bf16_bits.clone()
            } else {
                vec![BF_GUARD; count]
            };
            records.push(json!({"name":format!("{}__{}",s.name,mode.name()),"case":s.name,"mode":mode.name(),"M":s.m,"N":s.n,"K":s.k,
                "activation_sha256":sha(&p.a),"activation_scale_sha256":sha(&p.sa),"weight_sha256":sha(&p.w),"weight_scale_sha256":sha(&p.sw),
                "initial_status":p.initial_status,"expected_status":if mode.bf16(){cpu.status_bf16}else{cpu.status_f32},
                "CPU_F32_sha256":sha32(&f),"CPU_BF16_sha256":sha16(&b)}));
        }
    }
    records
}
fn run() -> Result<()> {
    ensure!(
        sha(BASE_PTX.as_bytes()) == BASE_SHA && sha(CAND_PTX.as_bytes()) == CAND_SHA,
        "frozen A/B PTX differs before load"
    );
    ensure!(
        u64::from_str_radix(env!("DSV41_GEMM_SOURCE_ID"), 16)? == cand_id(),
        "candidate source stamp differs"
    );
    let specs = specs();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--metadata-only", "true"] {
        println!(
            "{}",
            json!({"event":"FP8_GROUPS_AB_METADATA","identity":identity(),"case_count":specs.len(),"check_count":specs.len()*3,"case_names":specs.iter().map(|s|&s.name).collect::<Vec<_>>()})
        );
        return Ok(());
    }
    if args == ["--cpu-reference", "true"] {
        println!(
            "{}",
            json!({"event":"FP8_GROUPS_AB_CPU_EXPECTATIONS","identity":identity(),"GPU_executed":false,
            "case_count":specs.len(),"check_count":specs.len()*3,"expected_checks":cpu_records()})
        );
        return Ok(());
    }
    ensure!(
        args.is_empty(),
        "zero arguments, --metadata-only true or --cpu-reference true only"
    );
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL_SECONDS));
        eprintln!("parallel FP8 groups deadline");
        std::process::exit(124);
    });
    let started = Instant::now();
    let dev = CudaDevice::new(0)?;
    let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
    // Distinct named module registries and actual GPU stamp checks prevent
    // silently running the same cached function for both sides.
    let base = Engine::load(&dev, &stream, "fp8_group_a0_baseline", BASE_PTX, BASE_ID)?;
    let candidate = Engine::load(
        &dev,
        &stream,
        "fp8_parallel_groups_candidate",
        CAND_PTX,
        cand_id(),
    )?;
    ensure!(base.name != candidate.name, "module alias");
    let mut checks = Vec::new();
    for s in &specs {
        let p = pack(s);
        let cpu = reference(&p);
        let inputs = DeviceInputs::new(&dev, &p)?;
        let count = s.m * s.n;
        ensure!(
            p.a.len()
                + p.sa.len()
                + p.w.len()
                + p.sw.len()
                + 512
                + count * 12
                + 512
                + count * (s.k / 32) * 4
                + 128
                < GPU_BOUND,
            "explicit GPU array bound"
        );
        for mode in [Mode::F32, Mode::Bf16, Mode::Dual] {
            let a = run_engine(&dev, &stream, &base, &inputs, s, mode, p.initial_status)?;
            let b = run_engine(
                &dev,
                &stream,
                &candidate,
                &inputs,
                s,
                mode,
                p.initial_status,
            )?;
            let expected_f = if mode.f32() {
                cpu.f32_bits.clone()
            } else {
                vec![FLOAT_GUARD; count]
            };
            let expected_b = if mode.bf16() {
                cpu.bf16_bits.clone()
            } else {
                vec![BF_GUARD; count]
            };
            let expected_status = if mode.bf16() {
                cpu.status_bf16
            } else {
                cpu.status_f32
            };
            let f_ab = differences(&a.f32, &b.f32);
            let b_ab = differences(&a.bf16, &b.bf16);
            let f_cpu = differences(&a.f32, &expected_f) + differences(&b.f32, &expected_f);
            let b_cpu = differences(&a.bf16, &expected_b) + differences(&b.bf16, &expected_b);
            let actual_partials = b
                .contributions
                .as_ref()
                .context("candidate omitted all originalgroup contributions")?;
            let canonical_actual = actual_partials
                .iter()
                .map(|&x| canonical_contribution(x))
                .collect::<Vec<_>>();
            let canonical_cpu = cpu
                .contributions
                .iter()
                .map(|&x| canonical_contribution(x))
                .collect::<Vec<_>>();
            let partial_mismatches = differences(&canonical_actual, &canonical_cpu);
            let changed = inputs.check_immutable()?;
            let passed = f_ab
                + b_ab
                + f_cpu
                + b_cpu
                + changed
                + partial_mismatches
                + a.guard_mismatches
                + b.guard_mismatches
                == 0
                && a.status == expected_status
                && b.status == expected_status;
            let check = json!({"name":format!("{}__{}",s.name,mode.name()),"case":s.name,"mode":mode.name(),"M":s.m,"N":s.n,"K":s.k,
                "activation_sha256":sha(&p.a),"activation_scale_sha256":sha(&p.sa),"weight_sha256":sha(&p.w),"weight_scale_sha256":sha(&p.sw),
                "initial_status":p.initial_status,"expected_status":expected_status,"baseline_status":a.status,"candidate_status":b.status,
                "baseline_F32_sha256":sha32(&a.f32),"candidate_F32_sha256":sha32(&b.f32),"CPU_F32_sha256":sha32(&expected_f),
                "baseline_BF16_sha256":sha16(&a.bf16),"candidate_BF16_sha256":sha16(&b.bf16),"CPU_BF16_sha256":sha16(&expected_b),
                "F32_AB_bit_mismatches":f_ab,"BF16_AB_bit_mismatches":b_ab,"F32_CPU_bit_mismatches":f_cpu,"BF16_CPU_bit_mismatches":b_cpu,
                "candidate_contribution_count":actual_partials.len(),"candidate_contribution_physical_sha256":b.contributions_physical_sha256,"physical_contribution_layout":"group,output,row; raw logical diagnostic normalized togroup,row,output","candidate_contribution_raw_sha256":sha32(actual_partials),"candidate_contribution_canonical_sha256":sha32(&canonical_actual),"CPU_contribution_canonical_sha256":sha32(&canonical_cpu),"contribution_canonical_bit_mismatches":partial_mismatches,"immutable_input_mismatches":changed,"output_guard_mismatches":a.guard_mismatches+b.guard_mismatches,"passed":passed});
            println!("{}", json!({"event":"FP8_GROUPS_AB_CHECK","check":check}));
            checks.push(check);
        }
    }
    let cleanup = CLEANUP_FAILURES.load(Ordering::Relaxed);
    let all = checks.len() == specs.len() * 3
        && checks.iter().all(|v| v["passed"] == true)
        && cleanup == 0;
    println!(
        "{}",
        json!({"event":"FP8_GROUPS_AB_RESULT","schema_version":1,"status":if all{"PASS"}else{"FAIL"},"all_passed":all,"identity":identity(),"case_count":specs.len(),"check_count":checks.len(),"checks":checks,"cleanup_failures":cleanup,"elapsed_seconds":started.elapsed().as_secs_f64()})
    );
    ensure!(all, "parallel FP8 groups bitwise qualification failed");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        println!(
            "{}",
            json!({"event":"FP8_GROUPS_AB_FAILURE","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        );
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_original_group_contribution_matches_independent_numpy_cpu_only() {
        let fixture: Value = serde_json::from_str(CONTRIBUTION_FIXTURE).unwrap();
        let rows = fixture["records"].as_array().unwrap();
        assert_eq!(rows.len(), 124);
        for (case, wanted) in specs().iter().zip(rows) {
            assert_eq!(case.name, wanted["case"].as_str().unwrap());
            let cpu = reference(&pack(case));
            let actual = cpu
                .contributions
                .iter()
                .map(|&x| canonical_contribution(x))
                .collect::<Vec<_>>();
            assert_eq!(actual.len() as u64, wanted["count"].as_u64().unwrap());
            assert_eq!(
                sha32(&actual),
                wanted["canonical_sha256"].as_str().unwrap(),
                "{}",
                case.name
            );
        }
    }
    #[test]
    fn diagnostic_families_cover_both_subtiles_all_codes_and_error_order() {
        let all = specs();
        assert_eq!(all.len(), 124);
        assert_eq!(
            all.iter()
                .filter(|s| s.pattern == Pattern::MixedSafe)
                .count(),
            84
        );
        let p = pack(
            all.iter()
                .find(|s| s.name == "all_254_finite_activation_codes_basis")
                .unwrap(),
        );
        let codes =
            p.a.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(codes.len(), 254);
        assert!(!codes.contains(&127) && !codes.contains(&255));
        let p = pack(
            all.iter()
                .find(|s| s.name == "mixed_safe_M17_N33_K64")
                .unwrap(),
        );
        assert_ne!(&p.sa[..16], &p.sa[16..32]);
        assert_ne!(&p.sw[..2], &p.sw[2..]);
        for (name, status, bstatus, word) in [
            ("fixed_activation_scale_254_weight_scale_0", 4, 4, 0),
            ("fixed_activation_scale_0_weight_scale_254", 0, 0, 0),
            (
                "F32_overflow_then_next_group_recovery",
                4,
                4,
                1_f32.to_bits(),
            ),
            ("finite_F32_overflows_BF16", 0, 4, 511_f32.to_bits()),
        ] {
            let s = all.iter().find(|s| s.name == name).unwrap();
            let c = reference(&pack(s));
            assert_eq!(c.status_f32, status, "{name}");
            assert_eq!(c.status_bf16, bstatus, "{name}");
            if name == "F32_overflow_then_next_group_recovery" {
                assert!(c.f32_bits.iter().all(|&v| v == word));
            }
            if name == "finite_F32_overflows_BF16" {
                assert!(c
                    .f32_bits
                    .iter()
                    .all(|&v| v == (511_f32 * 2_f32.powi(119)).to_bits()));
                assert!(c.bf16_bits.iter().all(|&v| v == 0));
            }
        }
    }
    #[test]
    fn actual_frozen_PTX_identity_matches() {
        assert_eq!(sha(BASE_PTX.as_bytes()), BASE_SHA);
        assert!(CAND_PTX.contains("dsv41_gemm_fp8_group32_contributions"));
        assert!(CAND_PTX.contains("dsv41_gemm_group32_ordered_finish"));
        assert_ne!(BASE_ID, cand_id());
    }
    #[test]
    fn all_372_Rust_CPU_expectations_match_independent_NumPy_full_matrix_hashes() {
        let fixture: Value = serde_json::from_str(CPU_FIXTURE).unwrap();
        let wanted = fixture["checks"].as_array().unwrap();
        let actual = cpu_records();
        assert_eq!(wanted.len(), 372);
        assert_eq!(actual.len(), wanted.len());
        for (a, b) in actual.iter().zip(wanted) {
            for (key, value) in a.as_object().unwrap() {
                assert_eq!(value, &b[key], "{} {key}", a["name"]);
            }
        }
    }
}
