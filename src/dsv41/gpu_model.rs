//! Bounded complete native1017 temperature-zero C1 text forward.
//! Callers explicitly load weights, verify frequency tables/token map, create the
//! immutable collective program, reset a lane, and provide exact native IDs.
//! No HTTP startup, checkpoint IO, sampling override, prefix restore or benchmark
//! claim is hidden in this executor. Context512/prefill128 are hard gates.
use super::{
    collectives::{MatrixF32, MatrixRole, RoutedLayout, RowKind},
    config::Dsv41Config,
    engram::{
        CompressedTokenMap, EngramHashPlan, EngramHashState, COMPRESSED_MAP_SHA256_U32LE,
        TOKENIZER_SHA256, TOKEN_COUNT,
    },
    gpu_attention_block::{AttentionBlockKernels, MAX_NATIVE_CONTEXT},
    gpu_block::{BlockKernels, BlockOperationIds, BlockOutputIdentity, GpuBlock},
    gpu_collectives::{GpuCollectives, Operation},
    gpu_engram_block::{EngramBlockKernels, EngramOutputIdentity, GpuEngramBlock},
    gpu_hc::HcKernels,
    gpu_kv_store::GpuKvStore,
    gpu_model_weights::RankModelWeights,
    gpu_moe_block::MoeBlockKernels,
    gpu_position::PositionKernels,
    gpu_text_head::GpuTextHead,
    position::{
        EmbeddingSpan, FrequencyProfileIdentity, RopeProfile, VerifiedFrequencyTable, VOCAB_ROWS,
    },
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Instant};
use uuid::Uuid;
const LAYERS: usize = 40;
const HIDDEN: usize = 5120;
pub const OPERATIONS_PER_FORWARD: usize = 84;
pub const MAX_PROGRAM_OPERATIONS: usize = 32768;
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelOperationIds {
    pub embedding: u64,
    pub engram: [u64; 2],
    pub blocks: Vec<BlockOperationIds>,
    pub head: u64,
}
impl ModelOperationIds {
    pub fn execution_order(&self) -> Result<Vec<u64>> {
        ensure!(
            self.blocks.len() == LAYERS,
            "model needs exactly40 block operation pairs"
        );
        let mut ids = vec![self.embedding];
        for (layer, block) in self.blocks.iter().enumerate() {
            block.validate()?;
            if layer == 1 {
                ids.push(self.engram[0]);
            }
            if layer == 14 {
                ids.push(self.engram[1]);
            }
            ids.extend([block.attention, block.routed_experts]);
        }
        ids.push(self.head);
        ensure!(
            ids.len() == OPERATIONS_PER_FORWARD
                && ids.iter().copied().collect::<BTreeSet<_>>().len() == OPERATIONS_PER_FORWARD
                && ids.windows(2).all(|w| w[0] < w[1]),
            "model collective IDs must be unique and increase in actual execution order"
        );
        Ok(ids)
    }
    /// Caller uses this BEFORE startup to freeze the exact full sequence in
    /// Program::new_model_sequence. It never mutates a live collective program.
    pub fn operations(&self, start: usize, tokens: usize) -> Result<Vec<Operation>> {
        self.execution_order()?;
        validate_extent(start, tokens)?;
        let kind = if start == 0 {
            RowKind::ContiguousPrefillRows
        } else {
            RowKind::IndependentDecodeLanes
        };
        let layout = RoutedLayout::new(tokens, kind)?;
        let mut result = vec![Operation::RoutedF32 {
            id: self.embedding,
            layout,
        }];
        for layer in 0..LAYERS {
            if layer == 1 || layer == 14 {
                result.push(Operation::MatrixF32 {
                    id: self.engram[usize::from(layer == 14)],
                    layout: MatrixF32::new(MatrixRole::EngramLookup, tokens, 6144)?,
                });
            }
            result.push(Operation::RoutedF32 {
                id: self.blocks[layer].attention,
                layout,
            });
            result.push(Operation::RoutedF32 {
                id: self.blocks[layer].routed_experts,
                layout,
            });
        }
        result.push(Operation::ShardedMaxloc {
            id: self.head,
            layout: RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?,
        });
        ensure!(
            result.len() == OPERATIONS_PER_FORWARD,
            "model operation coverage"
        );
        Ok(result)
    }
}
fn validate_extent(start: usize, tokens: usize) -> Result<usize> {
    ensure!(
        (1..=128).contains(&tokens) && (start == 0 || tokens == 1),
        "model requires initial1..128 native IDs thenone-token append"
    );
    let end = start
        .checked_add(tokens)
        .context("model native position overflow")?;
    ensure!(
        end <= MAX_NATIVE_CONTEXT,
        "complete model path is bounded to512 native tokens"
    );
    Ok(end)
}
pub fn native_ids_sha256(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
pub struct ModelKernels {
    dev: Arc<CudaDevice>,
    stream_id: usize,
    boot: Uuid,
    blocks: Arc<BlockKernels>,
    attention: Arc<AttentionBlockKernels>,
    moe: Arc<MoeBlockKernels>,
    engram: Arc<EngramBlockKernels>,
    position: PositionKernels,
    hc: HcKernels,
}
impl ModelKernels {
    pub fn load(dev: &Arc<CudaDevice>, s: &CudaStream, boot: Uuid) -> Result<Arc<Self>> {
        let me = Self {
            dev: dev.clone(),
            stream_id: s.stream as usize,
            boot,
            blocks: BlockKernels::load(dev, s)?,
            attention: AttentionBlockKernels::load(dev, s, boot)?,
            moe: MoeBlockKernels::load(dev, s)?,
            engram: EngramBlockKernels::load(dev, s, boot)?,
            position: PositionKernels::load(dev, s)?,
            hc: HcKernels::load(dev, s)?,
        };
        me.stream(s)?;
        Ok(Arc::new(me))
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            !s.stream.is_null() && s.stream as usize == self.stream_id,
            "exact complete-model owner stream required"
        );
        ensure!(
            unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) } == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "blocking complete-model stream required"
        );
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct ModelIdentity {
    pub model_revision: String,
    pub manifest_sha256: String,
    pub rank: usize,
    pub layers: usize,
    pub native_precision: String,
    pub maximum_native_context: usize,
    pub maximum_initial_prefill: usize,
    pub concurrency: usize,
    pub temperature: f32,
    pub speculation: bool,
    pub prefix_restore: bool,
    pub engram_compressed_map_sha256: String,
    pub expected_tokenizer_sha256: String,
    pub native_id_input_only: bool,
    pub frequency_profile: FrequencyProfileIdentity,
    pub model_qualification_complete: bool,
    pub serving_enabled: bool,
}
#[derive(Debug, Serialize)]
pub struct GreedyForward {
    pub identity: ModelIdentity,
    pub lane: LaneKey,
    pub start: usize,
    pub consumed_native_tokens: usize,
    pub input_native_ids: Vec<u32>,
    pub input_native_ids_sha256: String,
    pub prefix_native_ids_sha256: String,
    pub prefix_tokens: usize,
    pub selected_native_token: u32,
    pub operation_ids: ModelOperationIds,
    pub block_checks: Vec<BlockOutputIdentity>,
    pub engram_checks: Vec<EngramOutputIdentity>,
    pub all_checks_passed: bool,
    pub elapsed_seconds: f64,
    pub timing_is_benchmark: bool,
}
pub struct GpuModel {
    kernels: Arc<ModelKernels>,
    weights: RankModelWeights,
    blocks: Vec<GpuBlock>,
    engram: Vec<GpuEngramBlock>,
    head: GpuTextHead,
    kv: GpuKvStore,
    hashes: EngramHashState,
    plain: VerifiedFrequencyTable,
    yarn: VerifiedFrequencyTable,
    lane: Option<LaneKey>,
    position: usize,
    prefix: Vec<u32>,
    initialized: bool,
    failed: bool,
    completion_identity: Option<[u8; 32]>,
    last_operation: Option<u64>,
    reserved_operations: usize,
}
impl GpuModel {
    /// Consumes already-loaded rank weights. No checkpoint payload is read here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernels: Arc<ModelKernels>,
        s: &CudaStream,
        weights: RankModelWeights,
        config: Dsv41Config,
        token_map: CompressedTokenMap,
        plain: VerifiedFrequencyTable,
        yarn: VerifiedFrequencyTable,
    ) -> Result<Self> {
        kernels.stream(s)?;
        config.validate()?;
        weights.manifest.verify_identity()?;
        token_map.validate_1017()?;
        ensure!(
            weights.layers.len() == LAYERS && weights.manifest.rank < 8,
            "model must own all40 rank-local scopes"
        );
        ensure!(
            plain.select(&[0])?.profile() == RopeProfile::Plain
                && yarn.select(&[0])?.profile() == RopeProfile::Yarn
                && plain.backend() == yarn.backend(),
            "model frequency profiles swapped or backends mixed"
        );
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            blocks.push(GpuBlock::new(
                kernels.blocks.clone(),
                kernels.attention.clone(),
                kernels.moe.clone(),
                s,
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
            )?);
        }
        let mut engram = Vec::with_capacity(2);
        for layer in [1, 14] {
            engram.push(GpuEngramBlock::new(
                kernels.engram.clone(),
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
            )?);
        }
        let head = GpuTextHead::new(&kernels.dev, s, weights.global.clone(), &weights.manifest)?;
        let kv = GpuKvStore::new(&kernels.dev, s, kernels.boot, 1)?;
        let hashes = EngramHashState::new(
            EngramHashPlan::official_1017(&config)?,
            token_map,
            1,
            MAX_NATIVE_CONTEXT,
        )?;
        Ok(Self {
            kernels,
            weights,
            blocks,
            engram,
            head,
            kv,
            hashes,
            plain,
            yarn,
            lane: None,
            position: 0,
            prefix: Vec::new(),
            initialized: false,
            failed: false,
            completion_identity: None,
            last_operation: None,
            reserved_operations: 0,
        })
    }
    pub fn identity(&self) -> ModelIdentity {
        ModelIdentity{model_revision:self.weights.manifest.model_revision.clone(),manifest_sha256:self.weights.manifest.manifest_sha256.clone(),rank:self.weights.manifest.rank,layers:LAYERS,native_precision:"native checkpointFP8/UE8group32+FP4 experts; sourceBF16/F32 stage casts; BF16 compressed-QAT values".into(),maximum_native_context:MAX_NATIVE_CONTEXT,maximum_initial_prefill:128,concurrency:1,temperature:0.,speculation:false,prefix_restore:false,engram_compressed_map_sha256:COMPRESSED_MAP_SHA256_U32LE.into(),expected_tokenizer_sha256:TOKENIZER_SHA256.into(),native_id_input_only:true,frequency_profile:self.plain.profile_identity(),model_qualification_complete:false,serving_enabled:false}
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn prefix_native_ids(&self) -> &[u32] {
        &self.prefix
    }
    /// GPU prefix restore is unsupported. Reset physically clears stores and all
    /// generation-bound components; collective owner/monotonic ID history persists.
    pub fn reset_lane(&mut self, s: &CudaStream, key: LaneKey) -> Result<()> {
        self.kernels.stream(s)?;
        ensure!(
            key.boot == self.kernels.boot
                && key.slot == 0
                && key.generation > 0
                && self.lane.is_none_or(|old| key.generation > old.generation),
            "model C1 reset requires slot0/new generation/boot"
        );
        self.failed = true;
        self.kv.reset_lane(s, key)?;
        for block in &mut self.blocks {
            block.reset_lane(s, key)?;
        }
        for block in &mut self.engram {
            block.reset_lane(s, key)?;
        }
        self.hashes.reset(0)?;
        self.kernels.dev.synchronize()?;
        self.lane = Some(key);
        self.position = 0;
        self.prefix.clear();
        self.initialized = false;
        self.failed = false;
        Ok(())
    }
    fn validate_forward(
        &self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        ids: &[u32],
        operations: &ModelOperationIds,
        collectives: &GpuCollectives,
    ) -> Result<usize> {
        self.kernels.stream(s)?;
        ensure!(
            !self.failed
                && self.lane == Some(key)
                && start == self.position
                && (!self.initialized || start > 0),
            "unbound/stale/failed/noncontiguous model lane"
        );
        let end = validate_extent(start, ids.len())?;
        ensure!(
            self.hashes.position(0)? == self.position && self.prefix.len() == self.position,
            "model CPU native prefix/hash position mismatch"
        );
        ensure!(
            ids.iter().all(|id| (*id as usize) < TOKEN_COUNT),
            "invalid native token ID"
        );
        let order = operations.execution_order()?;
        ensure!(
            self.last_operation.is_none_or(|old| order[0] > old)
                && self
                    .reserved_operations
                    .checked_add(OPERATIONS_PER_FORWARD)
                    .is_some_and(|n| n <= MAX_PROGRAM_OPERATIONS),
            "model operations reused/reordered or exceed frozen32768-op budget"
        );
        ensure!(
            collectives.rank() == self.weights.manifest.rank
                && collectives.compute_stream().stream == s.stream
                && self
                    .completion_identity
                    .is_none_or(|id| id == collectives.completion_identity()),
            "wrong model collective owner/rank/stream"
        );
        Ok(end)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward_greedy(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        ids: &[u32],
        operations: ModelOperationIds,
        collectives: &mut GpuCollectives,
    ) -> Result<GreedyForward> {
        let end = self.validate_forward(s, key, start, ids, &operations, collectives)?;
        // CPU hash history is transactional: failed GPU work never masquerades as a
        // consumed native prefix. The dirty GPU lane still requires a fresh reset.
        let mut staged_hashes = self.hashes.clone();
        let rows = staged_hashes.append_at(0, start, ids, None)?;
        let mut prefix = self.prefix.clone();
        prefix.extend_from_slice(ids);
        let token = collectives.completion_identity();
        self.completion_identity = Some(token);
        // Never reuse an attempted step's IDs after a failed partial forward.
        // Prefix/hash commits remain conditional on complete numerical success.
        self.reserved_operations += OPERATIONS_PER_FORWARD;
        self.last_operation = Some(operations.head);
        let began = Instant::now();
        let outcome = self.forward_inner(s, key, start, ids, &operations, &rows, collectives);
        match outcome {
            Ok((selected, block_checks, engram_checks)) => {
                self.hashes = staged_hashes;
                self.prefix = prefix;
                self.position = end;
                self.initialized = true;
                Ok(GreedyForward {
                    identity: self.identity(),
                    lane: key,
                    start,
                    consumed_native_tokens: ids.len(),
                    input_native_ids: ids.to_vec(),
                    input_native_ids_sha256: native_ids_sha256(ids),
                    prefix_native_ids_sha256: native_ids_sha256(&self.prefix),
                    prefix_tokens: end,
                    selected_native_token: selected,
                    operation_ids: operations,
                    block_checks,
                    engram_checks,
                    all_checks_passed: true,
                    elapsed_seconds: began.elapsed().as_secs_f64(),
                    timing_is_benchmark: false,
                })
            }
            Err(error) => {
                self.failed = true;
                let _ = self.kernels.dev.synchronize();
                Err(error)
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &mut self,
        s: &CudaStream,
        key: LaneKey,
        start: usize,
        ids: &[u32],
        operations: &ModelOperationIds,
        hash_rows: &[super::engram::HashRows],
        collectives: &mut GpuCollectives,
    ) -> Result<(u32, Vec<BlockOutputIdentity>, Vec<EngramOutputIdentity>)> {
        let k = self.kernels.clone();
        let tokens = ids.len();
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();return Err(e.into());}}};}
        let positions = (start..start + tokens)
            .map(|n| n as u32)
            .collect::<Vec<_>>();
        let plain = queued!(k
            .position
            .upload_frequencies(s, &self.plain.select(&positions)?));
        let yarn = queued!(k
            .position
            .upload_frequencies(s, &self.yarn.select(&positions)?));
        let comp_progress = queued!(super::attention_state::compression_progress(
            start, tokens, 2
        ));
        let compressed_ratio2 = if comp_progress.append_rows > 0 {
            Some(queued!(k.position.upload_frequencies(
                s,
                &self.yarn.select(
                    &comp_progress
                        .new_latent_rope_positions
                        .iter()
                        .map(|p| *p as u32)
                        .collect::<Vec<_>>()
                )?
            )))
        } else {
            None
        };
        let native = queued!(k.dev.htod_sync_copy(ids));
        let mut status = queued!(k.dev.alloc_zeros::<u32>(1));
        queued!(k.dev.memset_zeros(&mut status));
        let mut partial = queued!(k.dev.alloc_zeros::<f32>(tokens * HIDDEN));
        let rank = self.weights.manifest.rank;
        queued!(k.position.embedding(
            s,
            self.weights.global.bf16("embed.weight")?,
            &native,
            &mut partial,
            &mut status,
            EmbeddingSpan::new(rank, rank * VOCAB_ROWS, VOCAB_ROWS, HIDDEN)?
        ));
        let kind = if start == 0 {
            RowKind::ContiguousPrefillRows
        } else {
            RowKind::IndependentDecodeLanes
        };
        // One vocabulary owner contributes each row. Widening those exact BF16values
        // toF32 for reduction changes no result: allother contributions are exactzero.
        queued!(collectives.enqueue_reduce(
            operations.embedding,
            RoutedLayout::new(tokens, kind)?,
            &mut partial
        ));
        let mut h = queued!(k.dev.alloc_zeros::<half::bf16>(tokens * 4 * HIDDEN));
        queued!(k
            .position
            .expand(s, &partial, &mut h, &mut status, tokens, HIDDEN));
        let mut pre = queued!(k.dev.alloc_zeros::<f32>(tokens * 4));
        queued!(k.hc.initial(s, &mut pre, tokens));
        let mut checks = Vec::with_capacity(LAYERS);
        let mut engram_checks = Vec::with_capacity(2);
        for layer in 0..LAYERS {
            if layer == 1 || layer == 14 {
                let index = usize::from(layer == 14);
                let pending = match self.engram[index].enqueue_lookup(
                    s,
                    key,
                    start,
                    &h,
                    hash_rows,
                    operations.engram[index],
                    collectives,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = k.dev.synchronize();
                        return Err(error);
                    }
                };
                let augmented = queued!(pending.finish(collectives));
                engram_checks.push(augmented.identity);
                h = augmented.h; // finish fenced every read of the former residual.
            }
            let query_freq = if layer < 2 { &plain } else { &yarn };
            let compressed_freq = match layer {
                2 | 8 | 14 => compressed_ratio2.as_ref(),
                20 => Some(&yarn),
                _ => None,
            };
            let output = queued!(self.blocks[layer].execute(
                s,
                key,
                start,
                tokens,
                &h,
                &pre,
                query_freq,
                compressed_freq,
                &mut self.kv,
                collectives,
                operations.blocks[layer]
            ));
            checks.push(output.identity);
            h = output.h;
            pre = output.next_pre;
        }
        let mut collapsed = queued!(k.dev.alloc_zeros::<half::bf16>(tokens * HIDDEN));
        queued!(k
            .hc
            .collapse(s, &h, &pre, &mut collapsed, &mut status, tokens, HIDDEN));
        let mut head = queued!(self.head.prepare(s, &collapsed, tokens));
        queued!(head.enqueue_selection(collectives, operations.head));
        queued!(head.check(collectives));
        let selected = queued!(head.selected_token());
        queued!(collectives.synchronize());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0]
                && checks.len() == LAYERS
                && engram_checks.len() == 2,
            "complete model embedding/finalcollapse/status coverage failed"
        );
        Ok((selected, checks, engram_checks))
    }
}
impl Drop for GpuModel {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn ids(first: u64) -> ModelOperationIds {
        let mut n = first;
        let embedding = n;
        n += 1;
        let mut engram = [0; 2];
        let mut blocks = Vec::new();
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                engram[usize::from(layer == 14)] = n;
                n += 1;
            }
            blocks.push(BlockOperationIds {
                attention: n,
                routed_experts: n + 1,
            });
            n += 2;
        }
        ModelOperationIds {
            embedding,
            engram,
            blocks,
            head: n,
        }
    }
    #[test]
    fn frozen_full_model_program_has84_exact_roles_and_strict_order() {
        let ids = ids(100);
        let order = ids.execution_order().unwrap();
        assert_eq!(order, (100..184).collect::<Vec<_>>());
        for (start, rows) in [(0, 75), (75, 1)] {
            let ops = ids.operations(start, rows).unwrap();
            assert_eq!(ops.len(), 84);
            assert!(matches!(ops[0], Operation::RoutedF32 { .. }));
            assert!(matches!(ops[3], Operation::MatrixF32 { .. }));
            assert!(matches!(ops[83], Operation::ShardedMaxloc { .. }));
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, Operation::MatrixF32 { .. }))
                    .count(),
                2
            );
        }
        let mut wrong = ids.clone();
        wrong.engram[0] = wrong.embedding;
        assert!(wrong.execution_order().is_err());
        wrong = ids.clone();
        wrong.blocks.swap(0, 1);
        assert!(wrong.execution_order().is_err());
        wrong = ids.clone();
        wrong.blocks.pop();
        assert!(wrong.execution_order().is_err());
    }
    #[test]
    fn strict_native_input_boundaries_and_u32le_hashes() {
        assert_eq!(validate_extent(0, 75).unwrap(), 75);
        assert_eq!(validate_extent(511, 1).unwrap(), 512);
        for (s, n) in [(0, 0), (0, 129), (10, 2), (512, 1), (usize::MAX, 1)] {
            assert!(validate_extent(s, n).is_err());
        }
        assert_eq!(
            native_ids_sha256(&[0, 1, 129279]),
            "e6975f35fedae15219b5c539423d289ee45a9902268066d6d04622201a7553dc"
        );
        assert_ne!(
            native_ids_sha256(&[1, 0, 129279]),
            native_ids_sha256(&[0, 1, 129279])
        );
    }
}
