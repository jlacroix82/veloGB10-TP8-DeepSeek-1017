//! SHA-bound official1017 RoPE coefficients and isolated basic state operations.
//! Full tables come from exact CPU or independently captured CUDA Torch AST.
//! Their different coefficients are explicit backends, never equivalents.
//! Component references do not enable a model, actual prefix restore or serving.
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Read,
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::Path,
};
pub const ROTARY: usize = 64;
pub const POSITIONS: usize = 1_048_576;
pub const TABLE_BYTES: u64 = (POSITIONS * ROTARY * 4) as u64;
pub const FIXTURE_SHA: &str = "aec97f85ad55eaf6903b0df14e2c56533940acb24b26f27c4b519716a5713eea";
pub const FIXTURE: &str = include_str!("../../tests/fixtures/dsv41_position_vectors.json");
pub const CUDA_REFERENCE_SHA: &str =
    "ac02b4da5173883ee79ec9e6f7556c012b7009900a416fdd5e98761071019ded";
pub const CUDA_REFERENCE: &str =
    include_str!("../../tests/fixtures/dsv41_position_cuda_reference.json");
pub const BASIC_FIXTURE_SHA: &str =
    "40312b7abfa46ba7b507f4855b00256b5465bb3a930f767238e9dadbe00bdab3";
pub const BASIC_FIXTURE: &str =
    include_str!("../../tests/fixtures/dsv41_position_basic_vectors.json");
pub const VOCAB: usize = 129280;
pub const WORLD: usize = 8;
pub const VOCAB_ROWS: usize = VOCAB / WORLD;
pub const WINDOW: usize = 128;
pub const KV_DIM: usize = 512;
pub const MAX_BASIC_ELEMENTS: usize = 16 * 128 * 5120;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrequencyBackend {
    CpuTorch,
    CudaTorch,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrequencyProfileIdentity {
    pub backend: String,
    pub reference_manifest_sha256: String,
    pub plain_table_sha256: String,
    pub yarn_table_sha256: String,
    pub source_default_device: String,
    pub source_default_dtype: String,
}
impl FrequencyBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu-torch" => Ok(Self::CpuTorch),
            "cuda-torch" => Ok(Self::CudaTorch),
            _ => anyhow::bail!("frequency backend must be cpu-torch or cuda-torch"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::CpuTorch => "cpu-torch",
            Self::CudaTorch => "cuda-torch",
        }
    }
    pub fn table_sha(self, profile: RopeProfile) -> &'static str {
        match (self, profile) {
            (Self::CpuTorch, _) => profile.table_sha(),
            (Self::CudaTorch, RopeProfile::Plain) => {
                "4fc14ba78ee9bd20c9d63b7af974e1bfec0de279b5f5cd59fd220e6e9c9c4727"
            }
            (Self::CudaTorch, RopeProfile::Yarn) => {
                "05106ae8c428f59e11ea3e53aeda4e5fa8e31b554d2b30c652e0f03e074cd1e8"
            }
        }
    }
    /// Hashes identify the retained complete CPU manifest or actual GB10 CUDA
    /// capture, including its exact Torch build and official AST provenance.
    /// This identity does not qualify model outputs or interchangeable backends.
    pub fn identity(self) -> FrequencyProfileIdentity {
        let (manifest, device, dtype) = match self {
            Self::CpuTorch => (
                "43932db87cc6326d31567ccbf19d25f22378284ba9152e951833312c5f4628ba",
                "cpu",
                "torch.float32",
            ),
            Self::CudaTorch => (
                "2211571dfa91b54e804053f21c4ff0750314217a93669d4023abc2747f4a57c3",
                "cuda",
                "torch.bfloat16",
            ),
        };
        FrequencyProfileIdentity {
            backend: self.name().into(),
            reference_manifest_sha256: manifest.into(),
            plain_table_sha256: self.table_sha(RopeProfile::Plain).into(),
            yarn_table_sha256: self.table_sha(RopeProfile::Yarn).into(),
            source_default_device: device.into(),
            source_default_dtype: dtype.into(),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeProfile {
    Plain,
    Yarn,
}
impl RopeProfile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Yarn => "yarn",
        }
    }
    pub fn table_sha(self) -> &'static str {
        match self {
            Self::Plain => "e81ea78ffe07fb4e492f644b6ffc3d786df6e74bdf4d98e95b3c7c0596c9dba1",
            Self::Yarn => "9d678cf08499775cbab898dd9c4ba116be81d23f324edec6dc109a9db059554d",
        }
    }
}
#[derive(Clone, Debug)]
pub struct FrequencyRows {
    profile: RopeProfile,
    backend: FrequencyBackend,
    positions: Vec<u32>,
    coefficients: Vec<f32>,
    complete_table_verified: bool,
}
impl FrequencyRows {
    pub fn profile(&self) -> RopeProfile {
        self.profile
    }
    pub fn backend(&self) -> FrequencyBackend {
        self.backend
    }
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
    pub fn complete_table_verified(&self) -> bool {
        self.complete_table_verified
    }
    pub fn table_sha(&self) -> &'static str {
        self.backend.table_sha(self.profile)
    }
    pub fn slice_rows(&self, first: usize, count: usize) -> Result<Self> {
        let end = first
            .checked_add(count)
            .context("frequency row slice overflow")?;
        ensure!(
            count > 0 && end <= self.positions.len(),
            "frequency row slice extent"
        );
        Ok(Self {
            profile: self.profile,
            backend: self.backend,
            positions: self.positions[first..end].to_vec(),
            coefficients: self.coefficients[first * ROTARY..end * ROTARY].to_vec(),
            complete_table_verified: self.complete_table_verified,
        })
    }
    pub fn check_for(&self, profile: RopeProfile, positions: &[u32]) -> Result<()> {
        ensure!(
            self.profile == profile && self.positions == positions,
            "wrong RoPE profile/query-prefix positions"
        );
        Ok(())
    }
    /// Only the explicitly frozen representative positions are available without
    /// the complete table file. This never synthesizes a new long-context phase.
    pub fn from_reference_fixture(profile: RopeProfile, positions: &[u32]) -> Result<Self> {
        valid_positions(positions)?;
        ensure!(
            format!("{:x}", Sha256::digest(FIXTURE.as_bytes())) == FIXTURE_SHA,
            "RoPE reference fixture changed"
        );
        let f: serde_json::Value = serde_json::from_str(FIXTURE)?;
        let case = f["cases"]
            .as_array()
            .context("RoPE cases")?
            .iter()
            .find(|r| r["profile"] == profile.name())
            .context("missing profile")?;
        let original: Vec<u32> = serde_json::from_value(case["positions"].clone())?;
        let bits: Vec<u32> = serde_json::from_value(case["coefficients_f32_bits"].clone())?;
        ensure!(
            bits.len() == original.len() * ROTARY,
            "RoPE fixture geometry"
        );
        let mut values = Vec::with_capacity(positions.len() * ROTARY);
        for p in positions {
            let row = original.iter().position(|o| o == p).context(
                "position absent from representative fixture; verified complete table required",
            )?;
            values.extend(
                bits[row * ROTARY..(row + 1) * ROTARY]
                    .iter()
                    .map(|b| f32::from_bits(*b)),
            );
        }
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite RoPE coefficient"
        );
        Ok(Self {
            profile,
            backend: FrequencyBackend::CpuTorch,
            positions: positions.to_vec(),
            coefficients: values,
            complete_table_verified: false,
        })
    }
}
fn valid_positions(p: &[u32]) -> Result<()> {
    ensure!(
        (1..=16 * 128).contains(&p.len()) && p.iter().all(|p| (*p as usize) < POSITIONS),
        "bounded RoPE query position domain"
    );
    Ok(())
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}
impl Stamp {
    fn read(f: &File) -> Result<Self> {
        let m = f.metadata()?;
        ensure!(m.is_file(), "frequency source must be regular file");
        Ok(Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
        })
    }
}
pub struct VerifiedFrequencyTable {
    file: File,
    stamp: Stamp,
    profile: RopeProfile,
    backend: FrequencyBackend,
}
fn verify_table_identity(
    profile: RopeProfile,
    backend: FrequencyBackend,
    bytes: u64,
    sha: &str,
) -> Result<()> {
    ensure!(bytes == TABLE_BYTES, "frequency table extent differs");
    ensure!(
        sha == backend.table_sha(profile),
        "unverified frequency table bytes/profile/backend"
    );
    Ok(())
}
impl VerifiedFrequencyTable {
    /// Full bytes are hashed once with1MiB staging; selected-row reads recheck the
    /// open descriptor's device/inode/size/mtime/ctime before and after I/O.
    pub fn open(path: &Path, profile: RopeProfile) -> Result<Self> {
        Self::open_with_backend(path, profile, FrequencyBackend::CpuTorch)
    }
    pub fn open_with_backend(
        path: &Path,
        profile: RopeProfile,
        backend: FrequencyBackend,
    ) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        let stamp = Stamp::read(&file)?;
        ensure!(stamp.len == TABLE_BYTES, "frequency table extent differs");
        let mut h = Sha256::new();
        let mut block = vec![0_u8; 1024 * 1024];
        loop {
            let n = file.read(&mut block)?;
            if n == 0 {
                break;
            }
            h.update(&block[..n]);
        }
        verify_table_identity(profile, backend, stamp.len, &format!("{:x}", h.finalize()))?;
        ensure!(
            Stamp::read(&file)? == stamp,
            "frequency source changed while hashing"
        );
        Ok(Self {
            file,
            stamp,
            profile,
            backend,
        })
    }
    pub fn backend(&self) -> FrequencyBackend {
        self.backend
    }
    pub fn profile_identity(&self) -> FrequencyProfileIdentity {
        self.backend.identity()
    }
    pub fn select(&self, positions: &[u32]) -> Result<FrequencyRows> {
        valid_positions(positions)?;
        ensure!(
            Stamp::read(&self.file)? == self.stamp,
            "frequency source changed"
        );
        let mut selected = BTreeMap::new();
        let mut values = Vec::with_capacity(positions.len() * ROTARY);
        for p in positions {
            let row = selected.entry(*p).or_insert_with(|| [0_u8; ROTARY * 4]);
            if row.iter().all(|b| *b == 0) {
                self.file
                    .read_exact_at(row, *p as u64 * (ROTARY * 4) as u64)?;
            }
            values.extend(
                row.chunks_exact(4)
                    .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap()))),
            );
        }
        ensure!(
            Stamp::read(&self.file)? == self.stamp && values.iter().all(|v| v.is_finite()),
            "frequency source changed/nonfinite"
        );
        Ok(FrequencyRows {
            profile: self.profile,
            backend: self.backend,
            positions: positions.to_vec(),
            coefficients: values,
            complete_table_verified: true,
        })
    }
}
/// Adjacent F32 complex pairs rotate only the last64 BF16 elements. Positions
/// are independent per [batch,query]; forward and inverse use the same table.
pub fn rotate_in_place(
    x: &mut [bf16],
    freq: &FrequencyRows,
    profile: RopeProfile,
    positions: &[u32],
    batch: usize,
    queries: usize,
    heads: usize,
    width: usize,
    inverse: bool,
) -> Result<()> {
    freq.check_for(profile, positions)?;
    ensure!(
        (1..=16).contains(&batch)
            && (1..=128).contains(&queries)
            && (1..=64).contains(&heads)
            && (64..=512).contains(&width)
            && width % 2 == 0
            && x.len() == batch * queries * heads * width
            && positions.len() == batch * queries,
        "RoPE tensor geometry"
    );
    ensure!(x.iter().all(|v| v.is_finite()), "nonfinite RoPE input");
    let mut out = x.to_vec();
    for row in 0..batch * queries {
        for head in 0..heads {
            let base = (row * heads + head) * width + width - ROTARY;
            for pair in 0..ROTARY / 2 {
                let c = freq.coefficients[row * ROTARY + pair * 2];
                let s = freq.coefficients[row * ROTARY + pair * 2 + 1]
                    * (if inverse { -1.0 } else { 1.0 });
                let a = x[base + pair * 2].to_f32();
                let b = x[base + pair * 2 + 1].to_f32();
                // Match the actual official TorchCUDA complex multiply, including
                // its two explicit F32 contraction associations. Separate
                // products differ at observed BF16 roundtrip boundaries.
                let real = a.mul_add(c, -(b * s));
                let imag = b.mul_add(c, a * s);
                let r = bf16::from_f32(real);
                let i = bf16::from_f32(imag);
                ensure!(
                    real.is_finite() && imag.is_finite() && r.is_finite() && i.is_finite(),
                    "RoPE output overflow"
                );
                out[base + pair * 2] = r;
                out[base + pair * 2 + 1] = i;
            }
        }
    }
    x.copy_from_slice(&out);
    Ok(())
}
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingSpan {
    rank: usize,
    first: usize,
    rows: usize,
    dim: usize,
}
impl EmbeddingSpan {
    pub fn new(rank: usize, first: usize, rows: usize, dim: usize) -> Result<Self> {
        ensure!(
            rank < WORLD
                && rows > 0
                && first >= rank * VOCAB_ROWS
                && first
                    .checked_add(rows)
                    .context("embedding range overflow")?
                    <= (rank + 1) * VOCAB_ROWS
                && (8..=5120).contains(&dim)
                && dim % 8 == 0,
            "embedding span outside one vocab owner"
        );
        Ok(Self {
            rank,
            first,
            rows,
            dim,
        })
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn first(&self) -> usize {
        self.first
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
}
/// Nonowners write F32+0. An owner whose requested row was not staged FAILS;
/// missing owned payload can never masquerade as a zero nonowner partial.
pub fn embedding_partial(weights: &[bf16], ids: &[u32], span: EmbeddingSpan) -> Result<Vec<f32>> {
    ensure!(
        (1..=16 * 128).contains(&ids.len())
            && weights.len() == span.rows * span.dim
            && ids.iter().all(|i| (*i as usize) < VOCAB),
        "embedding data/token extent"
    );
    let mut out = vec![0_f32; ids.len() * span.dim];
    for (row, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id < span.rank * VOCAB_ROWS || id >= (span.rank + 1) * VOCAB_ROWS {
            continue;
        }
        ensure!(
            id >= span.first && id < span.first + span.rows,
            "owned embedding token absent from staged rows"
        );
        for col in 0..span.dim {
            let v = weights[(id - span.first) * span.dim + col].to_f32();
            ensure!(v.is_finite(), "nonfinite consumed embedding row");
            out[row * span.dim + col] = v;
        }
    }
    Ok(out)
}
pub fn cast_bf16_to_f32(x: &[bf16]) -> Result<Vec<f32>> {
    ensure!(
        !x.is_empty() && x.len() <= MAX_BASIC_ELEMENTS && x.iter().all(|v| v.is_finite()),
        "invalid BF16 cast input"
    );
    Ok(x.iter().map(|v| v.to_f32()).collect())
}
pub fn cast_f32_to_bf16(x: &[f32]) -> Result<Vec<bf16>> {
    ensure!(
        !x.is_empty() && x.len() <= MAX_BASIC_ELEMENTS && x.iter().all(|v| v.is_finite()),
        "invalid F32 cast input"
    );
    let out = x.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
    ensure!(out.iter().all(|v| v.is_finite()), "BF16 cast overflow");
    Ok(out)
}
/// Caller supplies a completed F32 embedding reduction. This only casts once
/// and repeats into4 HC copies; it neither performs nor claims a collective.
pub fn expand_embedding(reduced: &[f32], tokens: usize, dim: usize) -> Result<Vec<bf16>> {
    ensure!(
        (1..=16 * 128).contains(&tokens)
            && (8..=5120).contains(&dim)
            && dim % 8 == 0
            && reduced.len() == tokens * dim,
        "HC embedding geometry"
    );
    let row = cast_f32_to_bf16(reduced)?;
    let mut out = Vec::with_capacity(tokens * 4 * dim);
    for token in 0..tokens {
        for _ in 0..4 {
            out.extend_from_slice(&row[token * dim..(token + 1) * dim]);
        }
    }
    Ok(out)
}
#[derive(Clone, Copy, Debug)]
pub struct RingWrite {
    pub lane: usize,
    pub start_pos: u32,
    pub count: usize,
    pub input_row: usize,
}
#[derive(Clone, Debug)]
pub struct RingPlan {
    lanes: usize,
    input_rows: usize,
    writes: Vec<RingWrite>,
}
impl RingPlan {
    pub fn new(lanes: usize, input_rows: usize, writes: Vec<RingWrite>) -> Result<Self> {
        ensure!(
            (1..=16).contains(&lanes)
                && (1..=16 * 4096).contains(&input_rows)
                && !writes.is_empty()
                && writes.len() <= lanes,
            "ring plan bound"
        );
        let mut used = [false; 16];
        for w in &writes {
            ensure!(
                w.lane < lanes
                    && !used[w.lane]
                    && w.count > 0
                    && w.input_row
                        .checked_add(w.count)
                        .context("ring input overflow")?
                        <= input_rows,
                "duplicate/missing ring writer or input extent"
            );
            used[w.lane] = true;
            ensure!(
                (w.start_pos == 0 && w.count <= 4096) || (w.start_pos > 0 && w.count == 1),
                "only initial prefill or one-token append supported"
            );
            ensure!(
                (w.start_pos as usize)
                    .checked_add(w.count)
                    .context("ring position overflow")?
                    <= POSITIONS,
                "ring absolute position bound"
            );
        }
        Ok(Self {
            lanes,
            input_rows,
            writes,
        })
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn input_rows(&self) -> usize {
        self.input_rows
    }
    pub fn writes(&self) -> &[RingWrite] {
        &self.writes
    }
}
/// Initial prefill launches only the final128 writers for each lane, preventing
/// modulo-slot races. Append writes exactly one slot. Unwritten slots persist.
pub fn write_ring(cache: &mut [bf16], input: &[bf16], plan: &RingPlan) -> Result<()> {
    ensure!(
        cache.len() == plan.lanes * WINDOW * KV_DIM && input.len() == plan.input_rows * KV_DIM,
        "ring buffer shape"
    );
    let mut next = cache.to_vec();
    for w in &plan.writes {
        let first = w.count.saturating_sub(WINDOW);
        for i in first..w.count {
            let source = (w.input_row + i) * KV_DIM;
            ensure!(
                input[source..source + KV_DIM].iter().all(|v| v.is_finite()),
                "nonfinite consumed ring row"
            );
            let slot = (w.start_pos as usize + i) % WINDOW;
            let target = (w.lane * WINDOW + slot) * KV_DIM;
            next[target..target + KV_DIM].copy_from_slice(&input[source..source + KV_DIM]);
        }
    }
    cache.copy_from_slice(&next);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    const BACKEND_FIXTURE: &str =
        include_str!("../../tests/fixtures/dsv41_frequency_backends.json");
    fn backend_goldens() -> serde_json::Value {
        assert_eq!(
            format!("{:x}", Sha256::digest(BACKEND_FIXTURE.as_bytes())),
            "29dbbd334246f9dec3dfe596d536228e6aa0ef74c05d467720fe289e1e344927"
        );
        serde_json::from_str(BACKEND_FIXTURE).unwrap()
    }
    #[test]
    fn complete_table_identity_rejects_wrong_backend_profile_or_extent() {
        let f = backend_goldens();
        assert_eq!(f["CPU_and_CUDA_equivalent"], false);
        for row in f["cases"].as_array().unwrap() {
            let backend = FrequencyBackend::parse(row["backend"].as_str().unwrap()).unwrap();
            let profile = if row["profile"] == "plain" {
                RopeProfile::Plain
            } else {
                RopeProfile::Yarn
            };
            let other_backend = if backend == FrequencyBackend::CpuTorch {
                FrequencyBackend::CudaTorch
            } else {
                FrequencyBackend::CpuTorch
            };
            let other_profile = if profile == RopeProfile::Plain {
                RopeProfile::Yarn
            } else {
                RopeProfile::Plain
            };
            let digest = row["full_table_sha256"].as_str().unwrap();
            verify_table_identity(profile, backend, TABLE_BYTES, digest).unwrap();
            assert!(verify_table_identity(profile, other_backend, TABLE_BYTES, digest).is_err());
            assert!(verify_table_identity(other_profile, backend, TABLE_BYTES, digest).is_err());
            assert!(verify_table_identity(profile, backend, TABLE_BYTES - 4, digest).is_err());
            assert!(verify_table_identity(profile, backend, TABLE_BYTES, &"0".repeat(64)).is_err());
            assert_eq!(
                backend.identity().reference_manifest_sha256,
                row["reference_manifest_sha256"].as_str().unwrap()
            );
        }
        for invalid in ["", "cuda", "CPU-TORCH", "auto", "equivalent"] {
            assert!(FrequencyBackend::parse(invalid).is_err());
        }
        let temporary =
            std::env::temp_dir().join(format!("dsv41-frequency-short-{}", uuid::Uuid::new_v4()));
        std::fs::write(&temporary, [0_u8; 16]).unwrap();
        assert!(VerifiedFrequencyTable::open(&temporary, RopeProfile::Plain).is_err());
        assert!(VerifiedFrequencyTable::open_with_backend(
            &temporary,
            RopeProfile::Plain,
            FrequencyBackend::CudaTorch
        )
        .is_err());
        std::fs::remove_file(temporary).unwrap();
    }
    #[test]
    fn actual_cpu_and_cuda_coefficients_keep_distinct_identity_through_row_slices() {
        let f = backend_goldens();
        let cases = f["cases"].as_array().unwrap();
        for row in cases {
            let backend = FrequencyBackend::parse(row["backend"].as_str().unwrap()).unwrap();
            let profile = if row["profile"] == "plain" {
                RopeProfile::Plain
            } else {
                RopeProfile::Yarn
            };
            let positions: Vec<u32> = serde_json::from_value(row["positions"].clone()).unwrap();
            let bits: Vec<u32> =
                serde_json::from_value(row["coefficients_f32_bits"].clone()).unwrap();
            let rows = FrequencyRows {
                profile,
                backend,
                positions: positions.clone(),
                coefficients: bits.iter().map(|v| f32::from_bits(*v)).collect(),
                complete_table_verified: false,
            };
            let selected = rows.slice_rows(1, 2).unwrap();
            assert_eq!(selected.backend(), backend);
            assert_eq!(
                selected.table_sha(),
                row["full_table_sha256"].as_str().unwrap()
            );
            assert_eq!(selected.positions(), &positions[1..3]);
            assert_eq!(
                selected
                    .coefficients()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits[ROTARY..3 * ROTARY]
            );
            assert!(!selected.complete_table_verified());
            let other = cases
                .iter()
                .find(|v| v["profile"] == row["profile"] && v["backend"] != row["backend"])
                .unwrap();
            assert_ne!(row["coefficients_f32_bits"], other["coefficients_f32_bits"]);
            assert_ne!(row["full_table_sha256"], other["full_table_sha256"]);
        }
        assert_eq!(
            FrequencyRows::from_reference_fixture(RopeProfile::Plain, &[0])
                .unwrap()
                .backend(),
            FrequencyBackend::CpuTorch
        );
    }
    #[test]
    fn official_ast_last64_forward_inverse_and_roundtrip() {
        let f: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let p = if c["profile"] == "plain" {
                RopeProfile::Plain
            } else {
                RopeProfile::Yarn
            };
            let positions: Vec<u32> = serde_json::from_value(c["positions"].clone()).unwrap();
            let freq = FrequencyRows::from_reference_fixture(p, &positions).unwrap();
            let input: Vec<u16> = serde_json::from_value(c["input_bf16_bits"].clone()).unwrap();
            let start = input
                .iter()
                .map(|v| bf16::from_bits(*v))
                .collect::<Vec<_>>();
            for (inverse, key) in [(false, "forward_bf16_bits"), (true, "inverse_bf16_bits")] {
                let mut x = start.clone();
                rotate_in_place(
                    &mut x,
                    &freq,
                    p,
                    &positions,
                    1,
                    positions.len(),
                    3,
                    512,
                    inverse,
                )
                .unwrap();
                let cuda: serde_json::Value = serde_json::from_str(CUDA_REFERENCE).unwrap();
                let reference = cuda["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["profile"] == c["profile"])
                    .unwrap();
                let expected: Vec<u16> = serde_json::from_value(reference[key].clone()).unwrap();
                assert_eq!(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
            }
            let mut x = start.clone();
            rotate_in_place(
                &mut x,
                &freq,
                p,
                &positions,
                1,
                positions.len(),
                3,
                512,
                false,
            )
            .unwrap();
            rotate_in_place(
                &mut x,
                &freq,
                p,
                &positions,
                1,
                positions.len(),
                3,
                512,
                true,
            )
            .unwrap();
            let expected: Vec<u16> = serde_json::from_value(
                serde_json::from_str::<serde_json::Value>(CUDA_REFERENCE).unwrap()["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["profile"] == c["profile"])
                    .unwrap()["roundtrip_bf16_bits"]
                    .clone(),
            )
            .unwrap();
            assert_eq!(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
            assert!(freq.check_for(p, &[0]).is_err());
        }
        assert!(FrequencyRows::from_reference_fixture(RopeProfile::Plain, &[3]).is_err());
        assert!(
            FrequencyRows::from_reference_fixture(RopeProfile::Plain, &[POSITIONS as u32]).is_err()
        );
    }
    #[test]
    fn actual_cuda_reference_keeps_CPU_complex_and_legacy_schedule_canaries() {
        assert_eq!(
            format!("{:x}", Sha256::digest(CUDA_REFERENCE.as_bytes())),
            CUDA_REFERENCE_SHA
        );
        let fixture: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let cuda: serde_json::Value = serde_json::from_str(CUDA_REFERENCE).unwrap();
        assert_eq!(cuda["kind"], "official1017_RoPE_TorchCUDA_reference");
        assert_eq!(cuda["CPU_frequency_fixture_sha256"], FIXTURE_SHA);
        assert_eq!(cuda["model_source_sha256"], fixture["model_source_sha256"]);
        for case in fixture["cases"].as_array().unwrap() {
            let reference = cuda["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["profile"] == case["profile"])
                .unwrap();
            for key in ["forward_bf16_bits", "inverse_bf16_bits"] {
                assert_eq!(case[key], reference[key]);
            }
            let cpu: Vec<u16> =
                serde_json::from_value(case["roundtrip_bf16_bits"].clone()).unwrap();
            let expected: Vec<u16> =
                serde_json::from_value(reference["roundtrip_bf16_bits"].clone()).unwrap();
            let differences = cpu
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            assert_eq!(
                differences,
                if case["profile"] == "plain" {
                    vec![]
                } else {
                    vec![21456]
                }
            );
            let mut old: Vec<bf16> =
                serde_json::from_value::<Vec<u16>>(case["forward_bf16_bits"].clone())
                    .unwrap()
                    .into_iter()
                    .map(bf16::from_bits)
                    .collect();
            let coeff: Vec<u32> =
                serde_json::from_value(case["coefficients_f32_bits"].clone()).unwrap();
            for row in 0..15 {
                for head in 0..3 {
                    for pair in 0..32 {
                        let i = (row * 3 + head) * 512 + 448 + pair * 2;
                        let a = old[i].to_f32();
                        let b = old[i + 1].to_f32();
                        let c = f32::from_bits(coeff[row * 64 + pair * 2]);
                        let z = -f32::from_bits(coeff[row * 64 + pair * 2 + 1]);
                        let ac = std::hint::black_box(a * c);
                        let bz = std::hint::black_box(b * z);
                        let az = std::hint::black_box(a * z);
                        let bc = std::hint::black_box(b * c);
                        old[i] = bf16::from_f32(ac - bz);
                        old[i + 1] = bf16::from_f32(az + bc);
                    }
                }
            }
            let differences = old
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a.to_bits() != **b)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            assert_eq!(
                differences,
                if case["profile"] == "plain" {
                    vec![12283]
                } else {
                    vec![21456]
                }
            );
        }
    }
    #[test]
    fn one_owner_embedding_casts_expand_and_missing_rows_are_exact() {
        let dim = 8;
        let ids = (0..8)
            .map(|r| (r * VOCAB_ROWS + 1) as u32)
            .collect::<Vec<_>>();
        let mut sum = vec![0_f32; 8 * dim];
        for rank in 0..8 {
            let span = EmbeddingSpan::new(rank, rank * VOCAB_ROWS, 2, dim).unwrap();
            let weight = (0..2 * dim)
                .map(|i| bf16::from_f32((rank * 100 + i) as f32))
                .collect::<Vec<_>>();
            let part = embedding_partial(&weight, &ids, span).unwrap();
            for (a, b) in sum.iter_mut().zip(part) {
                *a += b;
            }
            assert!(embedding_partial(&weight, &[(rank * VOCAB_ROWS + 3) as u32], span).is_err());
            assert!(embedding_partial(&weight, &[VOCAB as u32], span).is_err());
        }
        let hc = expand_embedding(&sum, 8, dim).unwrap();
        for r in 0..8 {
            for copy in 0..4 {
                assert_eq!(
                    &hc[(r * 4 + copy) * dim..(r * 4 + copy + 1) * dim],
                    &sum[r * dim..(r + 1) * dim]
                        .iter()
                        .map(|v| bf16::from_f32(*v))
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(cast_f32_to_bf16(&[f32::MAX]).is_err());
        assert!(cast_bf16_to_f32(&[bf16::NAN]).is_err());
    }
    #[test]
    fn last128_prefill_and_append_have_no_duplicate_writers() {
        for count in [1, 127, 128, 129, 256, 257] {
            let plan = RingPlan::new(
                2,
                count + 1,
                vec![
                    RingWrite {
                        lane: 1,
                        start_pos: 0,
                        count,
                        input_row: 0,
                    },
                    RingWrite {
                        lane: 0,
                        start_pos: 999999,
                        count: 1,
                        input_row: count,
                    },
                ],
            )
            .unwrap();
            let input = (0..(count + 1) * KV_DIM)
                .map(|i| bf16::from_f32((i / KV_DIM) as f32))
                .collect::<Vec<_>>();
            let mut cache = vec![bf16::from_f32(-99.0); 2 * WINDOW * KV_DIM];
            write_ring(&mut cache, &input, &plan).unwrap();
            for i in count.saturating_sub(WINDOW)..count {
                assert_eq!(
                    cache[(WINDOW + i % WINDOW) * KV_DIM],
                    bf16::from_f32(i as f32)
                );
            }
            assert_eq!(
                cache[(999999 % WINDOW) * KV_DIM],
                bf16::from_f32(count as f32)
            );
        }
        assert!(RingPlan::new(
            1,
            2,
            vec![
                RingWrite {
                    lane: 0,
                    start_pos: 0,
                    count: 1,
                    input_row: 0
                },
                RingWrite {
                    lane: 0,
                    start_pos: 1,
                    count: 1,
                    input_row: 1
                }
            ]
        )
        .is_err());
        assert!(RingPlan::new(
            1,
            2,
            vec![RingWrite {
                lane: 0,
                start_pos: 1,
                count: 2,
                input_row: 0
            }]
        )
        .is_err());
    }
    #[test]
    fn official_embedding_window_ast_and_native_cast_goldens() {
        ensure_fixture_hash();
        let f: serde_json::Value = serde_json::from_str(BASIC_FIXTURE).unwrap();
        let e = &f["embedding"];
        let dim = 8;
        let ids: Vec<u32> = serde_json::from_value(e["ids"].clone()).unwrap();
        let mut sum = vec![0_f32; ids.len() * dim];
        for rank in 0..8 {
            let span = EmbeddingSpan::new(rank, rank * VOCAB_ROWS, VOCAB_ROWS, dim).unwrap();
            let weights = (0..VOCAB_ROWS * dim)
                .map(|i| {
                    let row = rank * VOCAB_ROWS + i / dim;
                    let col = i % dim;
                    bf16::from_f32((((row * 13 + col * 7) % 61 * 2) as f32 - 59.0) / 16.0)
                })
                .collect::<Vec<_>>();
            let part = embedding_partial(&weights, &ids, span).unwrap();
            let expected: Vec<u32> =
                serde_json::from_value(e["partial_f32_bits"][rank].clone()).unwrap();
            assert_eq!(
                part.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
            );
            for (a, b) in sum.iter_mut().zip(part) {
                *a += b;
            }
        }
        let expected: Vec<u32> = serde_json::from_value(e["reduced_f32_bits"].clone()).unwrap();
        assert_eq!(
            sum.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        let expanded = expand_embedding(&sum, ids.len(), dim).unwrap();
        let expected: Vec<u16> =
            serde_json::from_value(e["expanded_hc4_bf16_bits"].clone()).unwrap();
        assert_eq!(
            expanded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        for c in f["ring"].as_array().unwrap() {
            let count = c["count"].as_u64().unwrap() as usize;
            let start = c["start_pos"].as_u64().unwrap() as u32;
            let markers: Vec<u16> =
                serde_json::from_value(c["input_row_bf16_bits"].clone()).unwrap();
            let input = markers
                .iter()
                .flat_map(|v| std::iter::repeat_n(bf16::from_bits(*v), KV_DIM))
                .collect::<Vec<_>>();
            let mut cache = if c["initial_cache_row_bf16_bits"].is_array() {
                let bits: Vec<u16> =
                    serde_json::from_value(c["initial_cache_row_bf16_bits"].clone()).unwrap();
                bits.iter()
                    .flat_map(|v| std::iter::repeat_n(bf16::from_bits(*v), KV_DIM))
                    .collect::<Vec<_>>()
            } else {
                vec![bf16::from_f32(-99.0); 2 * WINDOW * KV_DIM]
            };
            let plan = RingPlan::new(
                2,
                2 * count,
                (0..2)
                    .map(|lane| RingWrite {
                        lane,
                        start_pos: start,
                        count,
                        input_row: lane * count,
                    })
                    .collect(),
            )
            .unwrap();
            write_ring(&mut cache, &input, &plan).unwrap();
            let expected: Vec<u16> =
                serde_json::from_value(c["expected_cache_row_bf16_bits"].clone()).unwrap();
            assert_eq!(
                cache
                    .chunks_exact(KV_DIM)
                    .map(|r| r[0].to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let c = &f["cast"];
        let bits: Vec<u32> = serde_json::from_value(c["input_f32_bits"].clone()).unwrap();
        let input = bits.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let cast = cast_f32_to_bf16(&input).unwrap();
        let expected: Vec<u16> = serde_json::from_value(c["expected_bf16_bits"].clone()).unwrap();
        assert_eq!(
            cast.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        let expected: Vec<u32> = serde_json::from_value(c["roundtrip_f32_bits"].clone()).unwrap();
        assert_eq!(
            cast_bf16_to_f32(&cast)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    fn ensure_fixture_hash() {
        assert_eq!(
            format!("{:x}", Sha256::digest(BASIC_FIXTURE.as_bytes())),
            BASIC_FIXTURE_SHA
        );
    }
    #[test]
    fn unverified_frequency_files_and_symlinks_are_rejected() {
        let temp = std::env::temp_dir().join(format!(
            "dsv41-position-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&temp).unwrap();
        let path = temp.join("table");
        std::fs::write(&path, [0_u8; 16]).unwrap();
        assert!(VerifiedFrequencyTable::open(&path, RopeProfile::Plain).is_err());
        let link = temp.join("link");
        std::os::unix::fs::symlink(path, &link).unwrap();
        assert!(VerifiedFrequencyTable::open(&link, RopeProfile::Plain).is_err());
        std::fs::remove_dir_all(temp).unwrap();
    }
}
