//! PRIVATE complete C1 chunk executor. Native weights stay owned/resident.
//! Old512-context execution remains separate. This module does not create a
//! transport program, lift its32768-op cap, run HTTP, or grant terminal authority.
use super::{
    bf16_collectives::MatrixBf16Index,
    chunk_plan::{self, ChunkTicket, INDEX_SOURCES},
    collectives::{MatrixF32, MatrixRole, RoutedLayout, RowKind},
    config::{Dsv41Config, MODEL_REVISION},
    engram::{
        CompressedTokenMap, EngramHashPlan, EngramHashState, HashRows, TOKENIZER_SHA256,
        TOKEN_COUNT,
    },
    gpu_attention_chunk::{row_kind, ChunkAttentionKernels, ChunkIndexExchange},
    gpu_block_chunk::{
        ChunkBlockKernels, ChunkBlockOperationIds, ChunkBlockReceipt, GpuBlockChunk,
    },
    gpu_chunk_store::{ChunkStoreKernels, GpuChunkStore},
    gpu_collectives::{GpuCollectives, Operation},
    gpu_engram_block::EngramBlockKernels,
    gpu_engram_chunk::{ChunkEngramReceipt, GpuEngramChunk},
    gpu_hc::HcKernels,
    gpu_model_weights::RankModelWeights,
    gpu_moe_block::MoeBlockKernels,
    gpu_position::PositionKernels,
    gpu_text_head::GpuTextHead,
    model_session_plan::{
        CpuRequestPlan, ExpectedBoundary, PlannedOperation, PlannedStep, Role, StepRole,
    },
    position::{
        EmbeddingSpan, FrequencyProfileIdentity, RopeProfile, VerifiedFrequencyTable, VOCAB_ROWS,
    },
    scheduler::LaneKey,
    weights::{RankManifest, WoaReconstructionPolicy, HEADER_GEOMETRY_SHA256, INDEX_SHA256},
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaStream};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Instant};
use uuid::Uuid;
const DIM: usize = 5120;
const LAYERS: usize = 40;
const MAX_CONTEXT: usize = 8192;

fn native_digest(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
/// Shared across all eight ownership manifests. This binds exact verified
/// metadata and WO-A policy, not unverified claims about whole payload hashes.
pub fn global_checkpoint_binding(manifest: &RankManifest) -> Result<String> {
    manifest.verify_identity()?;
    official_global_checkpoint_binding(manifest.woa_policy)
}
/// CPU-only metadata helper. It reports the compiled exact1017 metadata
/// contract; it does not inspect live model files or attest their payloads.
pub fn official_global_checkpoint_binding(policy: WoaReconstructionPolicy) -> Result<String> {
    let mut h = Sha256::new();
    h.update(b"dsv41-model-global-checkpoint-metadata-v1\0");
    for field in [MODEL_REVISION, INDEX_SHA256, HEADER_GEOMETRY_SHA256] {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field.as_bytes());
    }
    let policy = serde_json::to_vec(&policy)?;
    h.update((policy.len() as u64).to_le_bytes());
    h.update(policy);
    Ok(format!("{:x}", h.finalize()))
}
/// External artifact attestations: the supervisor verifies these against the
/// actual ELF/build snapshot. This component does not hash its own executable.
#[derive(Debug, Clone, Serialize)]
pub struct ModelChunkRuntimeBinding {
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
}
impl ModelChunkRuntimeBinding {
    fn validate(&self) -> Result<()> {
        ensure!(
            sha(&self.execution_snapshot_sha256) && sha(&self.engine_binary_sha256),
            "model runtime artifact identities must be SHA256"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelChunkOperationIds {
    pub embedding: u64,
    pub engram: [u64; 2],
    pub blocks: Vec<ChunkBlockOperationIds>,
    pub head: u64,
}
impl ModelChunkOperationIds {
    /// Reconstruct the exact source schedule, checking both roles AND native
    /// matrix geometry. An opID/row count alone is insufficient association.
    pub fn from_plan(ops: &[PlannedOperation], start: usize, tokens: usize) -> Result<Self> {
        chunk_plan::window_chunk(start, tokens)?;
        let layout = RoutedLayout::new(tokens, row_kind(start, tokens)?)?;
        let mut at = 0usize;
        fn take(
            ops: &[PlannedOperation],
            at: &mut usize,
            role: Role,
            make: impl FnOnce(u64) -> Operation,
        ) -> Result<u64> {
            let op = ops.get(*at).context("missing planned source operation")?;
            let id = op.wire.id();
            ensure!(
                op.role == role && op.wire == make(id),
                "model planned role/shape/source order differs"
            );
            *at += 1;
            Ok(id)
        }
        let embedding = take(ops, &mut at, Role::Embedding, |id| Operation::RoutedF32 {
            id,
            layout,
        })?;
        let mut engram = [0; 2];
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            if layer == 1 || layer == 14 {
                let matrix = MatrixF32::new(MatrixRole::EngramLookup, tokens, 6144)?;
                engram[usize::from(layer == 14)] =
                    take(ops, &mut at, Role::Engram { layer }, |id| {
                        Operation::MatrixF32 { id, layout: matrix }
                    })?;
            }
            let index_bf16 = if INDEX_SOURCES.contains(&layer) {
                let ratio = if layer < 20 { 2 } else { 1 };
                let keys = chunk_plan::compressed_chunk(start, tokens, ratio)?.stored_rows;
                if keys > 512 {
                    let matrix = MatrixBf16Index::new(tokens, keys)?;
                    Some(take(ops, &mut at, Role::SourceBf16Index { layer }, |id| {
                        Operation::MatrixBf16 { id, layout: matrix }
                    })?)
                } else {
                    None
                }
            } else {
                None
            };
            let attention_f32 = take(ops, &mut at, Role::Attention { layer }, |id| {
                Operation::RoutedF32 { id, layout }
            })?;
            let routed_experts_f32 = take(ops, &mut at, Role::RoutedExperts { layer }, |id| {
                Operation::RoutedF32 { id, layout }
            })?;
            let block = ChunkBlockOperationIds {
                index_bf16,
                attention_f32,
                routed_experts_f32,
            };
            block.validate()?;
            blocks.push(block);
        }
        let one = RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?;
        let head = take(ops, &mut at, Role::GlobalHead, |id| {
            Operation::ShardedMaxloc { id, layout: one }
        })?;
        ensure!(
            at == ops.len()
                && ops
                    .iter()
                    .map(|p| p.wire.id())
                    .collect::<BTreeSet<_>>()
                    .len()
                    == at
                && ops.windows(2).all(|p| p[0].wire.id() < p[1].wire.id()),
            "extra/duplicate/reversed model operations"
        );
        Ok(Self {
            embedding,
            engram,
            blocks,
            head,
        })
    }
}

pub struct ModelChunkKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    blocks: Arc<ChunkBlockKernels>,
    attention: Arc<ChunkAttentionKernels>,
    moe: Arc<MoeBlockKernels>,
    engram: Arc<EngramBlockKernels>,
    store: Arc<ChunkStoreKernels>,
    position: PositionKernels,
    hc: HcKernels,
}
impl ModelChunkKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        let me = Self {
            dev: dev.clone(),
            boot,
            blocks: ChunkBlockKernels::load(dev, stream.clone())?,
            attention: ChunkAttentionKernels::load(dev, stream.clone(), boot)?,
            moe: MoeBlockKernels::load(dev, &stream)?,
            engram: EngramBlockKernels::load(dev, &stream, boot)?,
            store: ChunkStoreKernels::load(dev, stream.clone(), boot)?,
            position: PositionKernels::load(dev, &stream)?,
            hc: HcKernels::load(dev, &stream)?,
            stream,
        };
        me.check_stream()?;
        Ok(Arc::new(me))
    }
    fn check_stream(&self) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            !self.stream.stream.is_null()
                && unsafe { sys::cuStreamGetFlags(self.stream.stream, &mut flags) }
                    == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "owned blocking model-chunk stream required"
        );
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ModelChunkIdentity {
    pub model_revision: String,
    pub rank: usize,
    pub manifest_sha256: String,
    /// Verified global index/header geometry/WO-A policy binding, NOT an assertion
    /// that this executor independently verified all checkpoint payload hashes.
    pub global_checkpoint_geometry_checksum_sha256: String,
    pub full_checkpoint_payload_identity_verified_by_loader: bool,
    pub runtime: ModelChunkRuntimeBinding,
    pub runtime_artifacts_self_verified: bool,
    pub frequency_profile: FrequencyProfileIdentity,
    pub tokenizer_sha256: String,
    pub native_context: usize,
    pub maximum_chunk_tokens: usize,
    pub concurrency: usize,
    pub temperature: f32,
    pub speculation: bool,
    pub prefix_restore: bool,
    pub model_or_quality_qualified: bool,
    pub HTTP_serving_enabled: bool,
}
/// Opaque full-step receipt: no Deserialize, public constructor, or public
/// fields. The runtime session must consume actual facts from this value, not
/// accept a caller-supplied ExpectedBoundary as completion authority.
#[derive(Debug, Serialize)]
pub struct CheckedModelChunk {
    identity: ModelChunkIdentity,
    ticket: ChunkTicket,
    plan_sha256: String,
    expected_boundary: ExpectedBoundary,
    completion_owner: [u8; 32],
    model_instance_identity: [u8; 32],
    stream_identity: usize,
    actual_epoch_before: u64,
    actual_epoch_after: u64,
    actual_completed_operation_count: usize,
    input_native_ids: Vec<u32>,
    input_native_ids_sha256: String,
    consumed_native_ids: Vec<u32>,
    native_prefix_sha256: String,
    generated_native_ids: Vec<u32>,
    generated_native_ids_sha256: String,
    selected_native_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    constraint: Option<super::tool_constraint::ConstraintExecution>,
    output_ordinal: Option<usize>,
    intermediate_head_ignored: bool,
    operations: ModelChunkOperationIds,
    blocks: Vec<ChunkBlockReceipt>,
    engram: Vec<ChunkEngramReceipt>,
    all_checks_passed: bool,
    all_rank_terminal_agreement: bool,
    elapsed_seconds: f64,
    timing_is_benchmark: bool,
}
impl CheckedModelChunk {
    pub fn identity(&self) -> &ModelChunkIdentity {
        &self.identity
    }
    pub fn ticket(&self) -> &ChunkTicket {
        &self.ticket
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn expected_boundary(&self) -> &ExpectedBoundary {
        &self.expected_boundary
    }
    pub fn completion_owner(&self) -> [u8; 32] {
        self.completion_owner
    }
    pub fn model_instance_identity(&self) -> [u8; 32] {
        self.model_instance_identity
    }
    pub fn rank(&self) -> usize {
        self.identity.rank
    }
    pub fn stream_identity(&self) -> usize {
        self.stream_identity
    }
    pub fn lane(&self) -> LaneKey {
        self.ticket.lane
    }
    pub fn actual_epoch_before(&self) -> u64 {
        self.actual_epoch_before
    }
    pub fn actual_epoch_after(&self) -> u64 {
        self.actual_epoch_after
    }
    pub fn actual_completed_operation_count(&self) -> usize {
        self.actual_completed_operation_count
    }
    pub fn generated_native_ids(&self) -> &[u32] {
        &self.generated_native_ids
    }
    pub fn generated_native_ids_sha256(&self) -> &str {
        &self.generated_native_ids_sha256
    }
    pub fn native_prefix_sha256(&self) -> &str {
        &self.native_prefix_sha256
    }
    pub fn consumed_native_ids(&self) -> &[u32] {
        &self.consumed_native_ids
    }
    pub fn input_native_ids(&self) -> &[u32] {
        &self.input_native_ids
    }
    pub fn input_native_ids_sha256(&self) -> &str {
        &self.input_native_ids_sha256
    }
    pub fn forced_tool_constraint(&self) -> Option<&super::tool_constraint::ConstraintExecution> {
        self.constraint.as_ref()
    }
    pub fn selected_native_id(&self) -> u32 {
        self.selected_native_id
    }
    pub fn output_ordinal(&self) -> Option<usize> {
        self.output_ordinal
    }
    pub fn intermediate_head_ignored(&self) -> bool {
        self.intermediate_head_ignored
    }
    pub fn all_checks_passed(&self) -> bool {
        self.all_checks_passed
    }
}
/// Successful physical reset only. The session still verifies all-rank reset
/// agreement before rearming a frozen request; this is not a terminal proof.
#[derive(Debug, Serialize)]
pub struct CheckedModelReset {
    identity: ModelChunkIdentity,
    lane: LaneKey,
    completion_owner: [u8; 32],
    model_instance_identity: [u8; 32],
    actual_epoch: u64,
    queued_operation_count: usize,
    stream_identity: usize,
    all_components_reset_checked: bool,
}
impl CheckedModelReset {
    pub fn identity(&self) -> &ModelChunkIdentity {
        &self.identity
    }
    pub fn lane(&self) -> LaneKey {
        self.lane
    }
    pub fn rank(&self) -> usize {
        self.identity.rank
    }
    pub fn completion_owner(&self) -> [u8; 32] {
        self.completion_owner
    }
    pub fn model_instance_identity(&self) -> [u8; 32] {
        self.model_instance_identity
    }
    pub fn actual_epoch(&self) -> u64 {
        self.actual_epoch
    }
    pub fn queued_operation_count(&self) -> usize {
        self.queued_operation_count
    }
    pub fn stream_identity(&self) -> usize {
        self.stream_identity
    }
    pub fn all_components_reset_checked(&self) -> bool {
        self.all_components_reset_checked
    }
}

pub struct GpuModelChunk {
    kernels: Arc<ModelChunkKernels>,
    weights: RankModelWeights,
    blocks: Vec<GpuBlockChunk>,
    engram: Vec<GpuEngramChunk>,
    head: GpuTextHead,
    store: GpuChunkStore,
    exchange: ChunkIndexExchange,
    hashes: EngramHashState,
    plain: VerifiedFrequencyTable,
    yarn: VerifiedFrequencyTable,
    runtime: ModelChunkRuntimeBinding,
    owner: [u8; 32],
    instance: [u8; 32],
    global_geometry: String,
    lane: Option<LaneKey>,
    position: usize,
    prefix: Vec<u32>,
    generated: Vec<u32>,
    plan: Option<String>,
    next_step: usize,
    completed_ops: usize,
    last_attempted_operation: Option<u64>,
    active_ticket: Option<ChunkTicket>,
    failed: bool,
}
impl GpuModelChunk {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: Arc<ModelChunkKernels>,
        weights: RankModelWeights,
        config: Dsv41Config,
        token_map: CompressedTokenMap,
        plain: VerifiedFrequencyTable,
        yarn: VerifiedFrequencyTable,
        runtime: ModelChunkRuntimeBinding,
        comm: &GpuCollectives,
    ) -> Result<Self> {
        k.check_stream()?;
        chunk_plan::validate_config(&config)?;
        weights.manifest.verify_identity()?;
        token_map.validate_1017()?;
        runtime.validate()?;
        ensure!(
            weights.layers.len() == 40
                && comm.rank() == weights.manifest.rank
                && comm.compute_stream().stream == k.stream.stream,
            "complete rank scopes/collective owner stream required"
        );
        ensure!(
            plain.select(&[0])?.profile() == RopeProfile::Plain
                && yarn.select(&[0])?.profile() == RopeProfile::Yarn
                && plain.backend() == yarn.backend(),
            "complete frequency profiles/backend required"
        );
        let mut blocks = Vec::with_capacity(40);
        for layer in 0..40 {
            blocks.push(GpuBlockChunk::new(
                k.blocks.clone(),
                k.attention.clone(),
                k.moe.clone(),
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
            )?);
        }
        let mut engram = Vec::with_capacity(2);
        for layer in [1, 14] {
            engram.push(GpuEngramChunk::new(
                &k.dev,
                k.stream.clone(),
                k.boot,
                k.engram.clone(),
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
            )?);
        }
        let owner = comm.completion_identity();
        let mut instance = Sha256::new();
        instance.update(b"dsv41-resident-model-chunk-local-owner-v1\0");
        instance.update(Uuid::new_v4().as_bytes());
        instance.update(k.boot.as_bytes());
        instance.update(owner);
        instance.update(weights.manifest.manifest_sha256.as_bytes());
        let instance = instance.finalize().into();
        let head = GpuTextHead::new(&k.dev, &k.stream, weights.global.clone(), &weights.manifest)?;
        let store = GpuChunkStore::new(k.store.clone(), &config, &weights.manifest, owner)?;
        let exchange =
            ChunkIndexExchange::new(k.attention.clone(), &config, &weights.manifest, comm)?;
        let hashes = EngramHashState::new(
            EngramHashPlan::official_1017(&config)?,
            token_map,
            1,
            MAX_CONTEXT,
        )?;
        let global_geometry = global_checkpoint_binding(&weights.manifest)?;
        Ok(Self {
            kernels: k,
            weights,
            blocks,
            engram,
            head,
            store,
            exchange,
            hashes,
            plain,
            yarn,
            runtime,
            owner,
            instance,
            global_geometry,
            lane: None,
            position: 0,
            prefix: vec![],
            generated: vec![],
            plan: None,
            next_step: 0,
            completed_ops: 0,
            last_attempted_operation: None,
            active_ticket: None,
            failed: true,
        })
    }
    pub fn identity(&self) -> ModelChunkIdentity {
        ModelChunkIdentity {
            model_revision: MODEL_REVISION.into(),
            rank: self.weights.manifest.rank,
            manifest_sha256: self.weights.manifest.manifest_sha256.clone(),
            global_checkpoint_geometry_checksum_sha256: self.global_geometry.clone(),
            full_checkpoint_payload_identity_verified_by_loader: self
                .weights
                .manifest
                .payload_identity_verified,
            runtime: self.runtime.clone(),
            runtime_artifacts_self_verified: false,
            frequency_profile: self.plain.profile_identity(),
            tokenizer_sha256: TOKENIZER_SHA256.into(),
            native_context: MAX_CONTEXT,
            maximum_chunk_tokens: 128,
            concurrency: 1,
            temperature: 0.,
            speculation: false,
            prefix_restore: false,
            model_or_quality_qualified: false,
            HTTP_serving_enabled: false,
        }
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn prefix_native_ids(&self) -> &[u32] {
        &self.prefix
    }
    /// Explicit request reset. It cannot switch collective owner, reuse IDs,
    /// reload weights, restore prefixes, or rewind a global epoch.
    pub fn reset_lane(
        &mut self,
        key: LaneKey,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedModelReset> {
        self.kernels.check_stream()?;
        ensure!(
            comm.completion_identity() == self.owner
                && comm.rank() == self.weights.manifest.rank
                && comm.compute_stream().stream == self.kernels.stream.stream,
            "model reset owner/rank/stream differs"
        );
        ensure!(
            key.boot == self.kernels.boot
                && key.slot == 0
                && key.generation > 0
                && self.lane.is_none_or(|old| key.generation > old.generation),
            "model reset newer C1 generation required"
        );
        self.failed = true;
        comm.synchronize()?;
        self.store.reset_lane(key)?;
        self.exchange.reset_lane(key)?;
        for block in &mut self.blocks {
            block.reset_lane(key)?;
        }
        for block in &mut self.engram {
            block.reset_lane(key)?;
        }
        self.hashes.reset(0)?;
        self.kernels.dev.synchronize()?;
        comm.synchronize()?;
        self.lane = Some(key);
        self.position = 0;
        self.prefix.clear();
        self.generated.clear();
        self.plan = None;
        self.next_step = 0;
        self.completed_ops = 0;
        self.active_ticket = None;
        self.failed = false;
        Ok(CheckedModelReset {
            identity: self.identity(),
            lane: key,
            completion_owner: self.owner,
            model_instance_identity: self.instance,
            actual_epoch: comm.expected_epoch(),
            queued_operation_count: comm.queued_operation_count(),
            stream_identity: self.kernels.stream.stream as usize,
            all_components_reset_checked: true,
        })
    }
    fn validate_step(
        &self,
        plan: &CpuRequestPlan,
        index: usize,
        input: &[u32],
        comm: &GpuCollectives,
    ) -> Result<(PlannedStep, ModelChunkOperationIds)> {
        self.kernels.check_stream()?;
        let step = plan
            .steps()
            .get(index)
            .context("model step outside frozen request")?
            .clone();
        let b = plan.binding();
        ensure!(
            !self.failed
                && self.lane == Some(b.lane)
                && index == self.next_step
                && step.start == self.position
                && self.prefix.len() == self.position
                && self.hashes.position(0)? == self.position,
            "model chunk generation/prefix/step state differs"
        );
        ensure!(
            self.plan.as_deref().is_none_or(|old| old == plan.digest())
                && b.model_revision == MODEL_REVISION
                && b.tokenizer_sha256 == TOKENIZER_SHA256
                && b.frequency == self.plain.profile_identity()
                && b.global_checkpoint_geometry_checksum_sha256 == self.global_geometry
                && b.execution_snapshot_sha256 == self.runtime.execution_snapshot_sha256
                && b.engine_binary_sha256 == self.runtime.engine_binary_sha256,
            "frozen request/model/artifact/frequency identities differ"
        );
        ensure!(
            comm.completion_identity() == self.owner
                && comm.rank() == self.weights.manifest.rank
                && comm.compute_stream().stream == self.kernels.stream.stream
                && comm.expected_epoch() == step.epoch_before
                && comm.request_plan_digest() == Some(plan.digest())
                && comm.queued_operation_count() == step.operation_begin,
            "model collective owner/rank/stream/plan/actual epoch/count differs"
        );
        validate_native_step(plan, &step, input, &self.generated)?;
        ensure!(
            step.operation_begin == self.completed_ops
                && step.tokens == input.len()
                && input.iter().all(|id| (*id as usize) < TOKEN_COUNT),
            "native extent/completed operation count differs"
        );
        let ops = plan.step_operations(index)?;
        ensure!(
            self.last_attempted_operation
                .is_none_or(|last| ops.first().is_some_and(|op| op.wire.id() > last)),
            "attempted model operation ID reused"
        );
        let ids = ModelChunkOperationIds::from_plan(ops, step.start, step.tokens)?;
        Ok((step, ids))
    }
    pub fn execute_step(
        &mut self,
        plan: &CpuRequestPlan,
        index: usize,
        input: &[u32],
        comm: &mut GpuCollectives,
    ) -> Result<CheckedModelChunk> {
        self.execute_step_with_constraint(plan, index, input, comm, None)
    }
    pub fn execute_step_with_constraint(
        &mut self,
        plan: &CpuRequestPlan,
        index: usize,
        input: &[u32],
        comm: &mut GpuCollectives,
        mask: Option<&super::tool_constraint::AllowedNativeTokens>,
    ) -> Result<CheckedModelChunk> {
        let result = self.execute_inner(plan, index, input, comm, mask);
        if result.is_err() {
            self.failed = true;
            comm.abort_session();
            if let Some(ticket) = self.active_ticket.as_ref() {
                let _ = self.store.abort(ticket);
            }
            let _ = self.kernels.dev.synchronize();
        }
        result
    }
    fn execute_inner(
        &mut self,
        plan: &CpuRequestPlan,
        index: usize,
        input: &[u32],
        comm: &mut GpuCollectives,
        mask: Option<&super::tool_constraint::AllowedNativeTokens>,
    ) -> Result<CheckedModelChunk> {
        let (step, ops) = self.validate_step(plan, index, input, comm)?;
        if let Some(mask) = mask {
            ensure!(
                step.role != StepRole::IntermediatePrefillHeadIgnored,
                "ignored prefill head must not advance forced sampler"
            );
            mask.check_binding(
                &plan.binding().original_request_sha256,
                plan.input_ids(),
                &self.generated,
            )?;
            ensure!(
                mask.identity().tokenizer_sha256 == plan.binding().tokenizer_sha256,
                "constrained vocabulary differs from actual native model/tokenizer binding"
            );
        }
        let mut staged_hashes = self.hashes.clone();
        let hash_rows = staged_hashes.append_at(0, step.start, input, None)?;
        let mut prefix = self.prefix.clone();
        prefix.extend_from_slice(input);
        self.plan = Some(plan.digest().into());
        self.last_attempted_operation = Some(ops.head);
        let ticket = if step.role == StepRole::DecodeOutput {
            self.store
                .begin_native_decode(plan.binding().lane, step.start)?
        } else {
            self.store
                .begin(plan.binding().lane, step.start, step.tokens)?
        };
        self.active_ticket = Some(ticket.clone());
        let began = Instant::now();
        let (selected, blocks, engram) =
            self.forward_rows(&ticket, input, &hash_rows, &ops, comm, mask)?;
        ensure!(
            comm.expected_epoch() == step.epoch_after
                && comm.queued_operation_count() == step.operation_end
                && blocks.len() == 40
                && engram.len() == 2,
            "complete model chunk epoch/count/component coverage differs"
        );
        ensure!(
            engram[0].all_hash_rows_sha256_u64le == engram[1].all_hash_rows_sha256_u64le
                && engram.iter().all(|e| e.ticket == ticket
                    && e.all_checks_passed
                    && e.active_store_ticket_checked),
            "Engram layers did not consume one checked hash batch/ticket"
        );
        let mut generated = self.generated.clone();
        if let Some(ordinal) = step.output_ordinal {
            ensure!(
                generated.len() + 1 == ordinal,
                "model output ordinal differs"
            );
            generated.push(selected);
        }
        ensure!(
            generated.len() == step.expected_boundary.generated_native_count,
            "generated usage differs from planned complete boundary"
        );
        comm.synchronize()?;
        self.kernels.dev.synchronize()?;
        self.store.commit(&ticket)?;
        self.hashes = staged_hashes;
        self.prefix = prefix;
        self.position = step.end;
        self.generated = generated;
        self.next_step += 1;
        self.completed_ops = step.operation_end;
        self.active_ticket = None;
        Ok(CheckedModelChunk {
            identity: self.identity(),
            ticket,
            plan_sha256: plan.digest().into(),
            expected_boundary: step.expected_boundary.clone(),
            completion_owner: self.owner,
            model_instance_identity: self.instance,
            stream_identity: self.kernels.stream.stream as usize,
            actual_epoch_before: step.epoch_before,
            actual_epoch_after: comm.expected_epoch(),
            actual_completed_operation_count: self.completed_ops,
            input_native_ids: input.to_vec(),
            input_native_ids_sha256: native_digest(input),
            consumed_native_ids: self.prefix.clone(),
            native_prefix_sha256: native_digest(&self.prefix),
            generated_native_ids: self.generated.clone(),
            generated_native_ids_sha256: native_digest(&self.generated),
            selected_native_id: selected,
            constraint: mask.map(|m| super::gpu_tool_constraint::checked_execution(m, selected)),
            output_ordinal: step.output_ordinal,
            intermediate_head_ignored: step.role == StepRole::IntermediatePrefillHeadIgnored,
            operations: ops,
            blocks,
            engram,
            all_checks_passed: true,
            all_rank_terminal_agreement: false,
            elapsed_seconds: began.elapsed().as_secs_f64(),
            timing_is_benchmark: false,
        })
    }
    fn forward_rows(
        &mut self,
        ticket: &ChunkTicket,
        input: &[u32],
        hash_rows: &[HashRows],
        ops: &ModelChunkOperationIds,
        comm: &mut GpuCollectives,
        mask: Option<&super::tool_constraint::AllowedNativeTokens>,
    ) -> Result<(u32, Vec<ChunkBlockReceipt>, Vec<ChunkEngramReceipt>)> {
        let k = self.kernels.clone();
        let s = &k.stream;
        let t = input.len();
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{comm.abort_session();let _=k.dev.synchronize();return Err(e.into());}}};}
        let positions = (ticket.start..ticket.end)
            .map(|n| n as u32)
            .collect::<Vec<_>>();
        let plain = queued!(k
            .position
            .upload_frequencies(s, &self.plain.select(&positions)?));
        let yarn = queued!(k
            .position
            .upload_frequencies(s, &self.yarn.select(&positions)?));
        let groups = chunk_plan::compressed_chunk(ticket.start, t, 2)?;
        let ratio2 = if groups.groups.is_empty() {
            None
        } else {
            Some(queued!(k.position.upload_frequencies(
                s,
                &self.yarn.select(
                    &groups
                        .groups
                        .iter()
                        .map(|g| g.rope_position as u32)
                        .collect::<Vec<_>>()
                )?
            )))
        };
        let native = queued!(k.dev.htod_sync_copy(input));
        let mut status = queued!(k.dev.alloc_zeros::<u32>(1));
        let mut partial = queued!(k.dev.alloc_zeros::<f32>(t * DIM));
        let rank = self.weights.manifest.rank;
        queued!(k.position.embedding(
            s,
            self.weights.global.bf16("embed.weight")?,
            &native,
            &mut partial,
            &mut status,
            EmbeddingSpan::new(rank, rank * VOCAB_ROWS, VOCAB_ROWS, DIM)?
        ));
        queued!(comm.enqueue_reduce(
            ops.embedding,
            RoutedLayout::new(t, row_kind(ticket.start, t)?)?,
            &mut partial
        ));
        let mut h = queued!(k.dev.alloc_zeros::<half::bf16>(t * 4 * DIM));
        queued!(k.position.expand(s, &partial, &mut h, &mut status, t, DIM));
        let mut pre = queued!(k.dev.alloc_zeros::<f32>(t * 4));
        queued!(k.hc.initial(s, &mut pre, t));
        let mut block_checks = Vec::with_capacity(40);
        let mut engram_checks = Vec::with_capacity(2);
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                let i = usize::from(layer == 14);
                let augmented = queued!(self.engram[i].execute_checked(
                    ticket,
                    &h,
                    hash_rows,
                    ops.engram[i],
                    &self.store,
                    comm
                ));
                engram_checks.push(augmented.receipt);
                h = augmented.h;
            }
            let query = if layer < 2 { &plain } else { &yarn };
            let compressed = match layer {
                2 | 8 | 14 => ratio2.as_ref(),
                20 => Some(&yarn),
                _ => None,
            };
            let output = queued!(self.blocks[layer].execute_checked(
                ticket,
                &h,
                &pre,
                query,
                compressed,
                &mut self.exchange,
                &mut self.store,
                comm,
                ops.blocks[layer]
            ));
            ensure!(
                output.receipt.layer == layer
                    && output.receipt.all_checks_passed
                    && output.receipt.exact_cache_layer_committed
                    && output.receipt.ticket == *ticket
                    && output.receipt.engram_before_block_required == (layer == 1 || layer == 14),
                "checked model layer receipt differs"
            );
            block_checks.push(output.receipt);
            h = output.h;
            pre = output.next_pre;
            // output.snapshot drops here after its exact layer commit. Do not
            // retain40 large GPU snapshots merely to retain compact receipts.
        }
        let mut collapsed = queued!(k.dev.alloc_zeros::<half::bf16>(t * DIM));
        queued!(k
            .hc
            .collapse(s, &h, &pre, &mut collapsed, &mut status, t, DIM));
        let mut head = if let Some(mask) = mask {
            queued!(self.head.prepare_constrained(s, &collapsed, t, mask))
        } else {
            queued!(self.head.prepare(s, &collapsed, t))
        };
        queued!(head.enqueue_selection(comm, ops.head));
        queued!(head.check(comm));
        let selected = queued!(head.selected_token());
        queued!(comm.synchronize());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "complete embedding/final HC status failed"
        );
        Ok((selected, block_checks, engram_checks))
    }
}
impl Drop for GpuModelChunk {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

fn validate_native_step(
    plan: &CpuRequestPlan,
    step: &PlannedStep,
    input: &[u32],
    generated: &[u32],
) -> Result<()> {
    ensure!(
        input.len() == step.tokens && step.start.checked_add(input.len()) == Some(step.end),
        "native model chunk extent differs"
    );
    match step.role {
        StepRole::IntermediatePrefillHeadIgnored => ensure!(
            step.output_ordinal.is_none()
                && generated.is_empty()
                && step.end < plan.input_ids().len()
                && plan.input_ids().get(step.start..step.end) == Some(input),
            "intermediate prefill must consume exact prompt and ignore head"
        ),
        StepRole::FinalPrefillFirstOutput => ensure!(
            step.output_ordinal == Some(1)
                && generated.is_empty()
                && step.end == plan.input_ids().len()
                && plan.input_ids().get(step.start..step.end) == Some(input),
            "final prefill must consume exact prompt and produce output1"
        ),
        StepRole::DecodeOutput => ensure!(
            step.tokens == 1
                && generated
                    .last()
                    .is_some_and(|id| input == [*id] && *id != 1)
                && step.output_ordinal == Some(generated.len() + 1),
            "decode consumes previous generated token exactly once; never follow EOS"
        ),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        model_session_plan::{GreedySemantics, RequestBinding, RequestBudget, StopPolicy},
        position::FrequencyBackend,
        protocol::TOOLKIT_REVISION,
    };
    use super::*;
    fn plan(n: usize, output: usize) -> CpuRequestPlan {
        CpuRequestPlan::new(
            RequestBinding {
                model_revision: MODEL_REVISION.into(),
                global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
                execution_snapshot_sha256: "b".repeat(64),
                engine_binary_sha256: "c".repeat(64),
                original_request_sha256: "d".repeat(64),
                tokenizer_sha256: TOKENIZER_SHA256.into(),
                toolkit_revision: TOOLKIT_REVISION.into(),
                frequency: FrequencyBackend::CudaTorch.identity(),
                sampling: GreedySemantics {
                    temperature: 0.,
                    top_p: 1.,
                    presence_penalty: 0.,
                    frequency_penalty: 0.,
                    seed: None,
                    thinking: false,
                    effective_reasoning_score: None,
                    speculation: false,
                },
                stop: StopPolicy {
                    eos_native_id: 1,
                    user_sequences: vec![],
                },
                budget: RequestBudget {
                    context_tokens: 8192,
                    output_allowance: output,
                    wall_millis: 180000,
                },
                lane: LaneKey {
                    boot: Uuid::from_u128(8),
                    slot: 0,
                    generation: 1,
                },
            },
            (0..n).map(|i| 17 + (i % 71) as u32).collect(),
            1,
            1,
            0,
        )
        .unwrap()
    }
    #[test]
    fn compiled_global_metadata_binding_has_independent_fixed_byte_hashes() {
        // Independently generated with Python hashlib: domain NUL, three
        // length-prefixed UTF8 fields, then serde-compatible JSON policy bytes.
        assert_eq!(
            official_global_checkpoint_binding(WoaReconstructionPolicy::OfficialBf16Round).unwrap(),
            "da3626bbe7157f8fac1457023c64d571cc2568d2b2e651628a177920d3a1f9b0"
        );
        assert_eq!(
            official_global_checkpoint_binding(WoaReconstructionPolicy::RequireExact).unwrap(),
            "2b2acaabddba298dde37565455de493086b1102de602c1a02a6978ca16df1b80"
        );
    }
    #[test]
    fn complete_typed_operations_cover_short_and_long_source_thresholds() {
        for p in [plan(75, 3), plan(513, 2), plan(1025, 3), plan(4096, 4096)] {
            for step in p.steps() {
                let wire = p.step_operations(step.step_index).unwrap();
                let ids = ModelChunkOperationIds::from_plan(wire, step.start, step.tokens).unwrap();
                assert_eq!(ids.blocks.len(), 40);
                assert_eq!(ids.embedding, wire[0].wire.id());
                assert_eq!(ids.head, wire.last().unwrap().wire.id());
                let indices = ids.blocks.iter().filter(|b| b.index_bf16.is_some()).count();
                assert_eq!(
                    indices,
                    if step.end > 1025 {
                        8
                    } else if step.end > 512 {
                        5
                    } else {
                        0
                    }
                );
                assert_eq!(wire.len(), 84 + indices);
            }
        }
    }
    #[test]
    fn source_roles_bf16_dimensions_and_temporal_row_kinds_cannot_be_substituted() {
        let p = plan(640, 2);
        let step = p.steps().iter().find(|s| s.start == 512).unwrap();
        let original = p.step_operations(step.step_index).unwrap();
        assert_eq!(
            row_kind(step.start, step.tokens).unwrap(),
            RowKind::ContiguousPrefillRows
        );
        for change in 0..5 {
            let mut bad = original.to_vec();
            match change {
                0 => bad[0].role = Role::Attention { layer: 0 },
                1 => {
                    let id = bad[0].wire.id();
                    bad[0].wire = Operation::RoutedF32 {
                        id,
                        layout: RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap(),
                    };
                }
                2 => {
                    let at = bad
                        .iter()
                        .position(|v| matches!(v.role, Role::SourceBf16Index { .. }))
                        .unwrap();
                    let id = bad[at].wire.id();
                    bad[at].wire = Operation::MatrixBf16 {
                        id,
                        layout: MatrixBf16Index::new(step.tokens, step.end - 1).unwrap(),
                    };
                }
                3 => {
                    bad.swap(1, 2);
                }
                _ => {
                    bad.push(*bad.last().unwrap());
                }
            }
            assert!(ModelChunkOperationIds::from_plan(&bad, step.start, step.tokens).is_err());
        }
        let p = plan(513, 1);
        let step = p.steps().last().unwrap();
        let mut wrong_kind = p.step_operations(step.step_index).unwrap().to_vec();
        let id = wrong_kind[0].wire.id();
        wrong_kind[0].wire = Operation::RoutedF32 {
            id,
            layout: RoutedLayout::new(1, RowKind::ContiguousPrefillRows).unwrap(),
        };
        assert!(ModelChunkOperationIds::from_plan(&wrong_kind, step.start, 1).is_err());
    }
    #[test]
    fn intermediate_heads_are_ignored_and_decode_consumes_previous_native_id() {
        let p = plan(129, 3);
        let first = &p.steps()[0];
        let final_prefill = &p.steps()[1];
        let decode = &p.steps()[2];
        validate_native_step(&p, first, &p.input_ids()[..128], &[]).unwrap();
        assert!(validate_native_step(&p, first, &p.input_ids()[..128], &[42]).is_err());
        validate_native_step(&p, final_prefill, &p.input_ids()[128..129], &[]).unwrap();
        assert!(validate_native_step(&p, final_prefill, &[999], &[]).is_err());
        validate_native_step(&p, decode, &[42], &[42]).unwrap();
        assert!(validate_native_step(&p, decode, &[41], &[42]).is_err());
        assert!(validate_native_step(&p, decode, &[1], &[1]).is_err());
        let final_decode = &p.steps()[3];
        validate_native_step(&p, final_decode, &[43], &[42, 43]).unwrap();
        assert!(validate_native_step(&p, final_decode, &[43], &[43]).is_err());
    }
    #[test]
    fn exact_ratio2_rope_rows_include_retained_odd_tail() {
        for (start, tokens, expected) in [
            (0, 1, vec![]),
            (0, 128, (0..128).step_by(2).collect()),
            (127, 2, vec![126]),
            (129, 3, vec![128, 130]),
        ] {
            let c = chunk_plan::compressed_chunk(start, tokens, 2).unwrap();
            assert_eq!(
                c.groups.iter().map(|g| g.rope_position).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(c.groups.len(), c.new_rows);
        }
    }
}
