//! PRIVATE additive chunk control. Native batch projections feed the unchanged
//! qualified initial-prefill or consecutive one-token compressor kernels.
//! No old positive-start guard is relaxed. Not optimized prefill/model serving.
use super::{
    chunk_plan::{self, ChunkTicket, CompressedChunk, LayerChunk, KV_SOURCES},
    compression::Ratio,
    config::Dsv41Config,
    gpu_compression::{CompressionKernels, GpuCompressor, NormalizedCompressionGpu},
    gpu_dense::{DenseShape, GpuDense},
    gpu_numerics::GpuNumerics,
    gpu_position::PositionKernels,
    gpu_weights::{LoadScope, LoadedWeights},
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
pub const STEP_ARRAY_BOUND_BYTES: usize = 16 * 1024 * 1024;
const DIM: usize = 512;
struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: Option<T>,
}
impl<T> Queued<T> {
    fn new(dev: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: dev.clone(),
            value: Some(value),
        }
    }
    fn into_checked(mut self) -> Result<T> {
        self.dev.synchronize()?;
        Ok(self.value.take().unwrap())
    }
}
impl<T> std::ops::Deref for Queued<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().unwrap()
    }
}
impl<T> std::ops::DerefMut for Queued<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}
impl<T> Drop for Queued<T> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}

pub struct ChunkCompressionKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    dense: GpuDense,
    numerics: GpuNumerics,
    position: PositionKernels,
    compression: Arc<CompressionKernels>,
}
impl ChunkCompressionKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(
            !stream.stream.is_null(),
            "chunk compressor needs owned blocking stream"
        );
        Ok(Arc::new(Self {
            dev: dev.clone(),
            dense: GpuDense::load(dev, &stream)?,
            numerics: GpuNumerics::load(dev, &stream)?,
            position: PositionKernels::load(dev, &stream)?,
            compression: CompressionKernels::load(dev, &stream, boot)?,
            stream,
            boot,
        }))
    }
    fn input<T>(&self, v: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&v.device(), &self.dev),
            "chunk compression context differs"
        );
        Ok(())
    }
    fn copy<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        first: usize,
        dst: &mut CudaSlice<T>,
        to: usize,
        n: usize,
    ) -> Result<()> {
        self.input(src)?;
        self.input(dst)?;
        ensure!(
            n > 0
                && first.checked_add(n).is_some_and(|end| end <= src.len())
                && to.checked_add(n).is_some_and(|end| end <= dst.len()),
            "chunk projection/latent copy bounds"
        );
        self.dev.bind_to_thread()?;
        let width = std::mem::size_of::<T>();
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                *dst.device_ptr() + (to * width) as u64,
                *src.device_ptr() + (first * width) as u64,
                n * width,
                self.stream.stream,
            )
        };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "chunk compression DtoD failed {rc:?}"
        );
        Ok(())
    }
}
#[derive(Default)]
struct Validity {
    lane: Option<LaneKey>,
    version: Option<u64>,
    failed: bool,
}
#[derive(Clone)]
struct Token {
    registry: Arc<Mutex<Validity>>,
    ticket: ChunkTicket,
}
impl Token {
    fn check(&self) -> Result<()> {
        let v = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk compressor registry poisoned"))?;
        ensure!(
            !v.failed && v.lane == Some(self.ticket.lane) && v.version == Some(self.ticket.version),
            "chunk latent stale/failed lane/version"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct ChunkLatentReceipt {
    pub rank: usize,
    pub manifest_sha256: String,
    pub source_layer: usize,
    pub ticket: ChunkTicket,
    pub chunk: CompressedChunk,
    pub normalized_rows: usize,
    pub columns: usize,
    pub rope_positions: Vec<u32>,
    pub projection_dtype: &'static str,
    pub pool_tail_dtype: &'static str,
    pub RMS_input_dtype: &'static str,
    pub RMS_epsilon: f64,
    pub output_dtype: &'static str,
    pub output_unrotated: bool,
    pub all_checks_passed: bool,
    pub native_source_calls: usize,
    pub explicit_per_token_synchronize_while_queuing: bool,
    pub allocator_implicit_synchronization_excluded: bool,
    pub optimized_prefill: bool,
    pub full_model_qualified: bool,
}
pub struct NormalizedChunkLatent {
    kernels: Arc<ChunkCompressionKernels>,
    receipt: ChunkLatentReceipt,
    values: Option<CudaSlice<bf16>>,
    token: Token,
}
impl NormalizedChunkLatent {
    pub fn receipt(&self) -> Result<&ChunkLatentReceipt> {
        self.token.check()?;
        Ok(&self.receipt)
    }
    pub fn values_for_plan(
        &self,
        p: &LayerChunk,
        rank: usize,
        manifest: &str,
    ) -> Result<Option<&CudaSlice<bf16>>> {
        self.token.check()?;
        validate_latent_receipt(p, &self.receipt, rank, manifest)?;
        Ok(self.values.as_ref())
    }
    /// The indexer reads this immutable source. Outer KV rotation receives a
    /// separate completed owner, never permission to overwrite the WK source.
    pub fn rotary_copy_for_plan(
        &self,
        p: &LayerChunk,
        rank: usize,
        manifest: &str,
    ) -> Result<Option<CudaSlice<bf16>>> {
        let Some(v) = self.values_for_plan(p, rank, manifest)? else {
            return Ok(None);
        };
        let mut output = Queued::new(
            &self.kernels.dev,
            self.kernels.dev.alloc_zeros::<bf16>(v.len())?,
        );
        self.kernels.copy(v, 0, &mut output, 0, v.len())?;
        Ok(Some(output.into_checked()?))
    }
}
pub fn validate_latent_receipt(
    p: &LayerChunk,
    v: &ChunkLatentReceipt,
    rank: usize,
    manifest: &str,
) -> Result<()> {
    let chunk = p
        .compressed
        .as_ref()
        .context("compressed owner plan missing")?;
    let ratio = p.ratio;
    ensure!(
        rank < 8
            && KV_SOURCES.contains(&p.layer)
            && p.owns_kv_and_index_keys
            && p.kv_source == Some(p.layer)
            && v.rank == rank
            && v.manifest_sha256 == manifest
            && v.source_layer == p.layer
            && v.ticket == p.ticket
            && v.chunk == *chunk
            && chunk == &chunk_plan::compressed_chunk(p.ticket.start, p.ticket.tokens, ratio)?
            && v.normalized_rows == chunk.new_rows
            && v.columns == DIM
            && v.rope_positions
                == chunk
                    .groups
                    .iter()
                    .map(|g| g.rope_position as u32)
                    .collect::<Vec<_>>()
            && v.projection_dtype
                == if ratio == 2 {
                    "F32 inputs and register-promoted stored BF16 weights -> F32"
                } else {
                    "BF16 input/native BF16 Linear -> BF16"
                }
            && v.pool_tail_dtype == if ratio == 2 { "F32" } else { "none" }
            && v.RMS_input_dtype == "BF16 after source pool/projection"
            && v.RMS_epsilon == 1e-20
            && v.output_dtype == "BF16"
            && v.output_unrotated
            && v.all_checks_passed
            && v.native_source_calls
                == if p.ticket.start == 0 {
                    1
                } else {
                    p.ticket.tokens
                }
            && !v.explicit_per_token_synchronize_while_queuing
            && !v.allocator_implicit_synchronization_excluded
            && !v.full_model_qualified
            && !v.optimized_prefill,
        "chunk latent source/precision/lane/version/group receipt differs"
    );
    Ok(())
}
pub struct GpuChunkCompressor {
    kernels: Arc<ChunkCompressionKernels>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    manifest: String,
    layer: usize,
    ratio: Ratio,
    norm_f32: CudaSlice<f32>,
    old: GpuCompressor,
    registry: Arc<Mutex<Validity>>,
    prefix: usize,
    last_version: u64,
}
impl GpuChunkCompressor {
    pub fn new(
        k: Arc<ChunkCompressionKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        layer: usize,
        w: Arc<LoadedWeights>,
    ) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        m.verify_identity()?;
        ensure!(
            KV_SOURCES.contains(&layer)
                && w.scope() == &LoadScope::Layer(layer)
                && w.manifest_sha256() == m.manifest_sha256,
            "actual compressor source scope/manifest"
        );
        let p = format!("layers.{layer}.attn.compressor");
        let ratio = if layer < 20 { Ratio::Two } else { Ratio::One };
        ensure!(
            w.bf16(&format!("{p}.wkv.weight"))?.len() == 512 * 5120
                && w.bf16(&format!("{p}.norm.weight"))?.len() == 512,
            "native compressor BF16 matrix/norm shape"
        );
        if ratio == Ratio::Two {
            ensure!(
                w.bf16(&format!("{p}.wgate.weight"))?.len() == 512 * 5120,
                "ratio2 native BF16 gate storage"
            );
        }
        let mut norm = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(512)?);
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        k.position.cast_bf16_f32(
            &k.stream,
            w.bf16(&format!("{p}.norm.weight"))?,
            &mut norm,
            &mut status,
        )?;
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "compressor norm promotion"
        );
        let old = GpuCompressor::new(k.compression.clone(), &k.stream, ratio, 512, 1)?;
        let norm_f32 = norm.into_checked()?;
        Ok(Self {
            kernels: k,
            weights: w,
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            layer,
            ratio,
            norm_f32,
            old,
            registry: Arc::new(Mutex::new(Validity::default())),
            prefix: 0,
            last_version: 0,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.kernels.boot && key.slot == 0 && key.generation > 0,
            "chunk compressor C1 slot0"
        );
        let mut v = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk compressor registry poisoned"))?;
        ensure!(
            v.lane.is_none_or(|old| key.generation > old.generation),
            "chunk compressor reset generation must advance"
        );
        self.kernels.dev.synchronize()?;
        self.old.reset_lane(&self.kernels.stream, key)?;
        self.kernels.dev.synchronize()?;
        *v = Validity {
            lane: Some(key),
            version: None,
            failed: false,
        };
        self.prefix = 0;
        Ok(())
    }
    pub fn invalidate_lane(&mut self, key: LaneKey) -> Result<()> {
        let mut v = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk compressor registry poisoned"))?;
        ensure!(
            v.lane == Some(key),
            "chunk compressor invalidation generation"
        );
        v.failed = true;
        Ok(())
    }
    pub fn execute_checked(
        &mut self,
        p: &LayerChunk,
        x: &CudaSlice<bf16>,
    ) -> Result<NormalizedChunkLatent> {
        let result = self.execute_inner(p, x);
        if result.is_err() {
            if let Ok(mut v) = self.registry.lock() {
                v.failed = true;
            }
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    fn absorb(
        &self,
        p: &LayerChunk,
        value: &NormalizedCompressionGpu<'_>,
        output: &mut Option<Queued<CudaSlice<bf16>>>,
        next: &mut usize,
    ) -> Result<()> {
        let source_plan = match value {
            NormalizedCompressionGpu::Empty(v) => &v.plan,
            NormalizedCompressionGpu::Latent(v) => v.plan(),
        };
        ensure!(
            source_plan.lane == p.ticket.lane && source_plan.ratio == self.ratio,
            "consecutive native compressor source identity"
        );
        let n = source_plan.output_rows;
        let chunk = p.compressed.as_ref().unwrap();
        ensure!(
            *next + n <= chunk.new_rows
                && source_plan.positions
                    == chunk.groups[*next..*next + n]
                        .iter()
                        .map(|g| g.rope_position)
                        .collect::<Vec<_>>(),
            "source one-token group/odd-tail association"
        );
        if n > 0 {
            let v = value
                .latent()
                .context("completed source group has no normalized latent")?;
            self.kernels.copy(
                unsafe { v.unrotated_values_unchecked() },
                0,
                output.as_mut().context("chunk output missing")?,
                *next * DIM,
                n * DIM,
            )?;
        }
        *next += n;
        Ok(())
    }
    fn execute_inner(
        &mut self,
        p: &LayerChunk,
        x: &CudaSlice<bf16>,
    ) -> Result<NormalizedChunkLatent> {
        let chunk = p
            .compressed
            .as_ref()
            .context("chunk compressor requires compressed plan")?;
        ensure!(
            p.layer == self.layer
                && p.owns_kv_and_index_keys
                && p.kv_source == Some(p.layer)
                && p.ratio == self.ratio.value()
                && p.ticket.start == self.prefix
                && p.ticket.version > self.last_version
                && p.ticket.start.checked_add(p.ticket.tokens) == Some(p.ticket.end)
                && chunk
                    == &chunk_plan::compressed_chunk(p.ticket.start, p.ticket.tokens, p.ratio)?,
            "chunk compressor source/prefix/group plan differs"
        );
        self.kernels.input(x)?;
        let t = p.ticket.tokens;
        ensure!(
            x.len() == t * 5120
                && t * 5120 * 4 + t * 512 * 16 + chunk.new_rows * 512 * 12 + 2 * 1024 * 1024
                    <= STEP_ARRAY_BOUND_BYTES,
            "compressor input/temporary budget"
        );
        {
            let mut v = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk compressor registry poisoned"))?;
            ensure!(
                !v.failed && v.lane == Some(p.ticket.lane),
                "chunk compressor unbound/failed lane"
            );
            v.version = Some(p.ticket.version);
        }
        self.last_version = p.ticket.version;
        let k = self.kernels.clone();
        let s = &k.stream;
        let token = Token {
            registry: self.registry.clone(),
            ticket: p.ticket.clone(),
        };
        let mut output = if chunk.new_rows == 0 {
            None
        } else {
            Some(Queued::new(
                &k.dev,
                k.dev.alloc_zeros::<bf16>(chunk.new_rows * 512)?,
            ))
        };
        let mut next = 0;
        let source_calls = if p.ticket.start == 0 { 1 } else { t };
        let prefix = format!("layers.{}.attn.compressor", self.layer);
        if self.ratio == Ratio::One {
            let projected = Queued::new(
                &k.dev,
                k.dense.bf16_linear_bf16(
                    s,
                    x,
                    self.weights.bf16(&format!("{prefix}.wkv.weight"))?,
                    DenseShape {
                        rows: t,
                        outputs: 512,
                        inner: 5120,
                    },
                )?,
            );
            if p.ticket.start == 0 {
                let old = self.old.push_ratio1(s, p.ticket.lane, 0, t, unsafe {
                    projected.device_values_unchecked()
                })?;
                let normalized = old.normalize(s, &k.numerics, &self.norm_f32)?;
                self.absorb(p, &normalized, &mut output, &mut next)?;
                normalized.check()?;
            } else {
                let mut rows = Queued::new(&k.dev, Vec::new());
                for row in 0..t {
                    let mut v = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(512)?);
                    unsafe {
                        k.copy(
                            projected.device_values_unchecked(),
                            row * 512,
                            &mut v,
                            0,
                            512,
                        )?;
                    }
                    rows.push(v);
                }
                let mut owners = Vec::with_capacity(t);
                for (row, v) in rows.iter().enumerate() {
                    let old = self
                        .old
                        .push_ratio1(s, p.ticket.lane, p.ticket.start + row, 1, v)?;
                    let normalized = old.normalize(s, &k.numerics, &self.norm_f32)?;
                    self.absorb(p, &normalized, &mut output, &mut next)?;
                    owners.push(normalized);
                }
                for owner in &owners {
                    owner.check()?;
                }
            }
            projected.check()?;
        } else {
            let mut f32input = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(t * 5120)?);
            let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
            k.position.cast_bf16_f32(s, x, &mut f32input, &mut status)?;
            let kv = Queued::new(
                &k.dev,
                k.dense.f32_bf16_linear_f32(
                    s,
                    &f32input,
                    self.weights.bf16(&format!("{prefix}.wkv.weight"))?,
                    DenseShape {
                        rows: t,
                        outputs: 512,
                        inner: 5120,
                    },
                )?,
            );
            let gate = Queued::new(
                &k.dev,
                k.dense.f32_bf16_linear_f32(
                    s,
                    &f32input,
                    self.weights.bf16(&format!("{prefix}.wgate.weight"))?,
                    DenseShape {
                        rows: t,
                        outputs: 512,
                        inner: 5120,
                    },
                )?,
            );
            if p.ticket.start == 0 {
                let old = self.old.push_ratio2(
                    s,
                    p.ticket.lane,
                    0,
                    t,
                    unsafe { kv.device_values_unchecked() },
                    unsafe { gate.device_values_unchecked() },
                )?;
                let normalized = old.normalize(s, &k.numerics, &self.norm_f32)?;
                self.absorb(p, &normalized, &mut output, &mut next)?;
                normalized.check()?;
            } else {
                let mut rows = Queued::new(&k.dev, Vec::new());
                for row in 0..t {
                    let mut a = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(512)?);
                    let mut b = Queued::new(&k.dev, k.dev.alloc_zeros::<f32>(512)?);
                    unsafe {
                        k.copy(kv.device_values_unchecked(), row * 512, &mut a, 0, 512)?;
                        k.copy(gate.device_values_unchecked(), row * 512, &mut b, 0, 512)?;
                    }
                    // Retain each staged owner until all source operations and
                    // checks finish. No explicit per-row staging fence; CUDA
                    // allocation itself may still impose synchronization.
                    rows.push((a, b));
                }
                let mut owners = Vec::with_capacity(t);
                for (row, (a, b)) in rows.iter().enumerate() {
                    let old =
                        self.old
                            .push_ratio2(s, p.ticket.lane, p.ticket.start + row, 1, a, b)?;
                    let normalized = old.normalize(s, &k.numerics, &self.norm_f32)?;
                    self.absorb(p, &normalized, &mut output, &mut next)?;
                    owners.push(normalized);
                }
                for owner in &owners {
                    owner.check()?;
                }
            }
            kv.check()?;
            gate.check()?;
            ensure!(
                k.dev.dtoh_sync_copy(&status)? == [0],
                "compressor BF16->F32 input cast"
            );
        }
        ensure!(
            next == chunk.new_rows,
            "chunk omitted completed source groups"
        );
        let values = match output {
            Some(v) => Some(v.into_checked()?),
            None => None,
        };
        token.check()?;
        self.prefix = p.ticket.end;
        let receipt = ChunkLatentReceipt {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            source_layer: self.layer,
            ticket: p.ticket.clone(),
            chunk: chunk.clone(),
            normalized_rows: chunk.new_rows,
            columns: 512,
            rope_positions: chunk
                .groups
                .iter()
                .map(|g| g.rope_position as u32)
                .collect(),
            projection_dtype: if self.ratio == Ratio::Two {
                "F32 inputs and register-promoted stored BF16 weights -> F32"
            } else {
                "BF16 input/native BF16 Linear -> BF16"
            },
            pool_tail_dtype: if self.ratio == Ratio::Two {
                "F32"
            } else {
                "none"
            },
            RMS_input_dtype: "BF16 after source pool/projection",
            RMS_epsilon: 1e-20,
            output_dtype: "BF16",
            output_unrotated: true,
            all_checks_passed: true,
            native_source_calls: source_calls,
            explicit_per_token_synchronize_while_queuing: false,
            allocator_implicit_synchronization_excluded: false,
            optimized_prefill: false,
            full_model_qualified: false,
        };
        Ok(NormalizedChunkLatent {
            kernels: k,
            receipt,
            values,
            token,
        })
    }
}
impl Drop for GpuChunkCompressor {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn fixture() -> serde_json::Value {
        let bytes = include_bytes!("../../tests/fixtures/dsv41_chunk_compression_vectors.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "80ba0bc498319f013b60e4eefd9eaac9864614a722f0aa2802819347c8781bba"
        );
        serde_json::from_slice(bytes).unwrap()
    }
    #[test]
    fn official_CPU_AST_full_and_consecutive_chunks_prove_boundary_maps_at_real512() {
        let fixture = fixture();
        assert_eq!(
            fixture["source_sha256"],
            super::super::compression::MODEL_REFERENCE_SHA256
        );
        assert_eq!(fixture["CPU_only"], true);
        assert_eq!(fixture["CUDA_initialized"], false);
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 10);
        let mut total = 0;
        for case in cases {
            let ratio = case["ratio"].as_u64().unwrap() as usize;
            let events = case["events"].as_array().unwrap();
            assert_eq!(
                case["full_prefill_bf16_le_sha256"],
                case["one_token_bf16_le_sha256"]
            );
            assert_eq!(
                case["full_prefill_bf16_le_sha256"],
                case["chunked_bf16_le_sha256"]
            );
            let mut next = 0;
            for event in events {
                let start = event["start"].as_u64().unwrap() as usize;
                let tokens = event["tokens"].as_u64().unwrap() as usize;
                assert_eq!(start, next);
                let p = chunk_plan::compressed_chunk(start, tokens, ratio).unwrap();
                assert_eq!(p.new_rows, event["new_rows"].as_u64().unwrap() as usize);
                assert_eq!(
                    serde_json::to_value(
                        p.groups.iter().map(|g| g.rope_position).collect::<Vec<_>>()
                    )
                    .unwrap(),
                    event["positions"]
                );
                assert_eq!(
                    event["source_calls"].as_u64().unwrap() as usize,
                    if start == 0 { 1 } else { tokens }
                );
                next += tokens;
            }
            assert_eq!(next, case["total_tokens"].as_u64().unwrap() as usize);
            total += next;
        }
        assert_eq!(total, 27653);
    }
    #[test]
    fn unchanged_CPU_source_pool_survives_initial_split_and_future_append_bitwise() {
        use super::super::compression::CompressionState;
        let boot = Uuid::from_u128(27);
        let key = LaneKey {
            boot,
            slot: 0,
            generation: 1,
        };
        let dim = 8;
        let norm = vec![1f32; dim];
        let kv = (0..130 * dim)
            .map(|i| ((i * 7) % 257) as f32 / 64. - 2. + 1. / 1024.)
            .collect::<Vec<_>>();
        let score = (0..130 * dim)
            .map(|i| ((i * 11) % 17) as f32 / 4. - 2.)
            .collect::<Vec<_>>();
        for ratio in [Ratio::One, Ratio::Two] {
            let mut full = CompressionState::new(boot, 1, ratio, dim).unwrap();
            full.reset(key).unwrap();
            let mut single = CompressionState::new(boot, 1, ratio, dim).unwrap();
            single.reset(key).unwrap();
            let native = kv.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
            let first = if ratio == Ratio::One {
                full.push_ratio1(key, 0, 128, &native[..128 * dim]).unwrap()
            } else {
                full.push_ratio2(key, 0, 128, &kv[..128 * dim], &score[..128 * dim])
                    .unwrap()
            };
            let expected = first.normalized_unrotated(&norm, dim).unwrap();
            let mut actual = Vec::new();
            for position in 0..128 {
                let a = position * dim;
                let out = if ratio == Ratio::One {
                    single
                        .push_ratio1(key, position, 1, &native[a..a + dim])
                        .unwrap()
                } else {
                    single
                        .push_ratio2(key, position, 1, &kv[a..a + dim], &score[a..a + dim])
                        .unwrap()
                };
                actual.extend(out.normalized_unrotated(&norm, dim).unwrap());
            }
            assert_eq!(
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            // Whole-prefill has different unused tail slots at an even end;
            // verify that the next complete live group is still identical.
            for position in 128..130 {
                let a = position * dim;
                let mut results = Vec::new();
                for state in [&mut full, &mut single] {
                    let out = if ratio == Ratio::One {
                        state
                            .push_ratio1(key, position, 1, &native[a..a + dim])
                            .unwrap()
                    } else {
                        state
                            .push_ratio2(key, position, 1, &kv[a..a + dim], &score[a..a + dim])
                            .unwrap()
                    };
                    results.push(out.normalized_unrotated(&norm, dim).unwrap());
                }
                assert_eq!(
                    results[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    results[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn latent_receipt_cannot_cross_source_generation_or_claim_rotated_or_F32_output() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let cfg = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let boot = Uuid::from_u128(71);
        let key = LaneKey {
            boot,
            slot: 0,
            generation: 1,
        };
        let mut c = chunk_plan::ChunkCursor::new(&cfg, boot, 1).unwrap();
        c.reset(key).unwrap();
        let t = c.begin(key, 0, 127).unwrap();
        for layer in 0..40 {
            let p = c.layer(&t, layer).unwrap();
            c.commit_layer(&p).unwrap();
        }
        c.commit(&t).unwrap();
        let next = c.begin(key, 127, 128).unwrap();
        for layer in 0..2 {
            let p = c.layer(&next, layer).unwrap();
            c.commit_layer(&p).unwrap();
        }
        let p = c.layer(&next, 2).unwrap();
        let chunk = p.compressed.clone().unwrap();
        let receipt = ChunkLatentReceipt {
            rank: 3,
            manifest_sha256: "a".repeat(64),
            source_layer: 2,
            ticket: next.clone(),
            chunk: chunk.clone(),
            normalized_rows: chunk.new_rows,
            columns: 512,
            rope_positions: chunk
                .groups
                .iter()
                .map(|g| g.rope_position as u32)
                .collect(),
            projection_dtype: "F32 inputs and register-promoted stored BF16 weights -> F32",
            pool_tail_dtype: "F32",
            RMS_input_dtype: "BF16 after source pool/projection",
            RMS_epsilon: 1e-20,
            output_dtype: "BF16",
            output_unrotated: true,
            all_checks_passed: true,
            native_source_calls: 128,
            explicit_per_token_synchronize_while_queuing: false,
            allocator_implicit_synchronization_excluded: false,
            optimized_prefill: false,
            full_model_qualified: false,
        };
        validate_latent_receipt(&p, &receipt, 3, &"a".repeat(64)).unwrap();
        for control in 0..9 {
            let mut bad = receipt.clone();
            match control {
                0 => bad.source_layer = 8,
                1 => bad.ticket.lane.generation += 1,
                2 => bad.ticket.version += 1,
                3 => bad.rope_positions[0] += 1,
                4 => bad.output_unrotated = false,
                5 => bad.projection_dtype = "BF16 narrowing",
                6 => bad.output_dtype = "F32",
                7 => bad.normalized_rows += 1,
                _ => bad.RMS_epsilon = 1e-6,
            }
            assert!(validate_latent_receipt(&p, &bad, 3, &"a".repeat(64)).is_err());
        }
        let registry = Arc::new(Mutex::new(Validity {
            lane: Some(key),
            version: Some(next.version),
            failed: false,
        }));
        let token = Token {
            registry: registry.clone(),
            ticket: next,
        };
        token.check().unwrap();
        registry.lock().unwrap().failed = true;
        assert!(token.check().is_err());
    }
}
