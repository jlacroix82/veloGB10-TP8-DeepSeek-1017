//! Private whole-model packed executor. All forty layers share the flattened
//! rows; each physical lane keeps independent token/KV/Engram history. This is
//! a checked compute boundary, never HTTP/terminal or benchmark authority.
use super::{
    batch_attention_layout::AttentionPhase,
    batch_compression_plan,
    batch_plan::{BatchTicket, LaneSegment, PackedKind},
    chunk_plan,
    collectives::{MatrixF32, MatrixRole, RoutedLayout, RowKind},
    config::{Dsv41Config, MODEL_REVISION},
    engram::{CompressedTokenMap, EngramHashPlan, EngramHashState, HashRows, TOKEN_COUNT},
    gpu_batch_attention_block::BatchAttentionBlockKernels,
    gpu_batch_block::{
        BatchBlockKernels, BatchBlockOperationIds, BatchBlockReceipt, DecodeHcScratchMode,
        GpuBlockBatch,
    },
    gpu_batch_engram::GpuBatchEngram,
    gpu_batch_store::{BatchStoreKernels, GpuBatchStore},
    gpu_batch_text_head::GpuBatchTextHead,
    gpu_collectives::{GpuCollectives, Operation},
    gpu_engram_block::{EngramBlockKernels, EngramOutputIdentity},
    gpu_gemm::{GemmBackend, GemmDispatchSnapshot, GemmGroupWorkspace, GemmWorkspaceBinding},
    gpu_hc::HcKernels,
    gpu_model_chunk::global_checkpoint_binding,
    gpu_model_weights::RankModelWeights,
    gpu_moe_block::MoeBlockKernels,
    gpu_position::PositionKernels,
    gpu_status_cut::{ProjectionStatusBackend, ProjectionStatusSnapshot},
    position::{
        EmbeddingSpan, FrequencyProfileIdentity, RopeProfile, VerifiedFrequencyTable, VOCAB_ROWS,
    },
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Instant};
use uuid::Uuid;
const DIM: usize = 5120;
/// Initial exact-all-keys learned scope, not the physical cache allocation limit.
pub const MAX_LEARNED_PREFIX: usize = 512;
fn ids_hash(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
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

pub struct BatchModelKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    blocks: Arc<BatchBlockKernels>,
    attention: Arc<BatchAttentionBlockKernels>,
    moe: Arc<MoeBlockKernels>,
    engram: Arc<EngramBlockKernels>,
    store: Arc<BatchStoreKernels>,
    position: PositionKernels,
    hc: HcKernels,
    gemm_workspace: Option<Arc<GemmGroupWorkspace>>,
    projection_status_backend: ProjectionStatusBackend,
}
impl BatchModelKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        Self::load_with_gemm_backend(dev, stream, boot, GemmBackend::Original)
    }
    pub fn load_with_gemm_backend(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        backend: GemmBackend,
    ) -> Result<Arc<Self>> {
        Self::load_with_backends(
            dev,
            stream,
            boot,
            backend,
            ProjectionStatusBackend::Original,
        )
    }
    pub fn load_with_backends(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        backend: GemmBackend,
        projection_status_backend: ProjectionStatusBackend,
    ) -> Result<Arc<Self>> {
        Self::load_with_decode_hc(
            dev,
            stream,
            boot,
            backend,
            projection_status_backend,
            DecodeHcScratchMode::Off,
        )
    }
    pub fn load_with_decode_hc(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        backend: GemmBackend,
        projection_status_backend: ProjectionStatusBackend,
        scratch: DecodeHcScratchMode,
    ) -> Result<Arc<Self>> {
        Self::load_with_decode_graph(
            dev,
            stream,
            boot,
            backend,
            projection_status_backend,
            scratch,
            false,
        )
    }
    pub fn load_with_decode_graph(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        backend: GemmBackend,
        projection_status_backend: ProjectionStatusBackend,
        scratch: DecodeHcScratchMode,
        graph: bool,
    ) -> Result<Arc<Self>> {
        ensure!(!boot.is_nil(), "packed model boot required");
        let workspace = match backend {
            GemmBackend::Original => None,
            GemmBackend::ParallelGroupsHotSmallM => {
                Some(GemmGroupWorkspace::new(dev, stream.clone(), boot)?)
            }
        };
        Ok(Arc::new(Self {
            dev: dev.clone(),
            boot,
            blocks: BatchBlockKernels::load_with_hc_graph(dev, stream.clone(), scratch, graph)?,
            attention: BatchAttentionBlockKernels::load_with_status_backend(
                dev,
                stream.clone(),
                boot,
                workspace.clone(),
                projection_status_backend,
            )?,
            moe: MoeBlockKernels::load_with_gemm_workspace(dev, &stream, workspace.clone())?,
            engram: EngramBlockKernels::load(dev, &stream, boot)?,
            store: BatchStoreKernels::load(dev, stream.clone(), boot)?,
            position: PositionKernels::load(dev, &stream)?,
            hc: HcKernels::load(dev, &stream)?,
            gemm_workspace: workspace,
            projection_status_backend,
            stream,
        }))
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct BatchModelOperationIds {
    pub embedding: u64,
    pub engram: [u64; 2],
    pub blocks: Vec<BatchBlockOperationIds>,
    pub head: u64,
}
impl BatchModelOperationIds {
    pub fn consecutive(first: u64) -> Result<Self> {
        ensure!(
            first.checked_add(83).is_some(),
            "packed operation IDs overflow"
        );
        let mut next = first + 1;
        let mut engram = [0; 2];
        let mut blocks = Vec::with_capacity(40);
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                engram[usize::from(layer == 14)] = next;
                next += 1;
            }
            blocks.push(BatchBlockOperationIds {
                attention_f32: next,
                routed_experts_f32: next + 1,
            });
            next += 2;
        }
        Ok(Self {
            embedding: first,
            engram,
            blocks,
            head: next,
        })
    }
    /// Actual full-program wire rows match every flattened native tensor. This
    /// first learned profile only permits <=512 compressed keys in each lane.
    pub fn operations(
        &self,
        rows: usize,
        lanes: usize,
        kind: PackedKind,
    ) -> Result<Vec<Operation>> {
        ensure!(
            self.blocks.len() == 40 && (1..=128).contains(&rows) && (1..=16).contains(&lanes),
            "packed operation geometry"
        );
        let expected = Self::consecutive(self.embedding)?;
        ensure!(
            serde_json::to_vec(self)? == serde_json::to_vec(&expected)?,
            "packed roles/IDs must retain complete source order"
        );
        let layout = RoutedLayout::new(rows, kind.collective_rows())?;
        let mut ops = vec![Operation::RoutedF32 {
            id: self.embedding,
            layout,
        }];
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                ops.push(Operation::MatrixF32 {
                    id: self.engram[usize::from(layer == 14)],
                    layout: MatrixF32::new(MatrixRole::EngramLookup, rows, 6144)?,
                });
            }
            let ids = self.blocks[layer];
            ids.validate()?;
            ops.push(Operation::RoutedF32 {
                id: ids.attention_f32,
                layout,
            });
            ops.push(Operation::RoutedF32 {
                id: ids.routed_experts_f32,
                layout,
            });
        }
        ops.push(Operation::ShardedMaxloc {
            id: self.head,
            layout: RoutedLayout::new(lanes, RowKind::IndependentDecodeLanes)?,
        });
        ensure!(ops.len() == 84, "complete packed source operation count");
        Ok(ops)
    }
}
#[derive(Clone, Debug)]
pub struct BatchLaneInput {
    pub lane: LaneKey,
    pub native_ids: Vec<u32>,
    /// Required for prefill; absent for decode. Full original prompt extent is
    /// immutable across ragged prefill segments and never inferred from a chunk.
    pub full_prompt_tokens: Option<usize>,
}
#[derive(Clone)]
struct LaneState {
    key: LaneKey,
    prefix: Vec<u32>,
    selected: Option<u32>,
    outputs: usize,
}
#[derive(Debug, Serialize)]
pub struct BatchLaneOutput {
    pub lane: LaneKey,
    pub consumed_prefix_tokens: usize,
    pub prefix_native_ids_sha256: String,
    pub selected_native_id: u32,
    pub generated_output_ordinal: Option<usize>,
    pub intermediate_prefill_head_ignored: bool,
}
#[derive(Debug, Serialize)]
pub struct CheckedBatchModel {
    ticket: BatchTicket,
    rank: usize,
    revision: String,
    manifest_sha256: String,
    global_checkpoint_geometry_binding: String,
    completion_owner: [u8; 32],
    model_instance_identity: [u8; 32],
    stream_identity: u64,
    program_sha256: String,
    operations: BatchModelOperationIds,
    epoch_before: u64,
    epoch_after: u64,
    operation_count_before: usize,
    operation_count_after: usize,
    input_native_ids: Vec<u32>,
    lanes: Vec<BatchLaneOutput>,
    blocks: Vec<BatchBlockReceipt>,
    engram: Vec<EngramOutputIdentity>,
    all_checks_passed: bool,
    all_rank_terminal_agreement: bool,
    elapsed_seconds: f64,
    timing_is_benchmark: bool,
    fp8_gemm_dispatch: Option<GemmDispatchSnapshot>,
}
impl CheckedBatchModel {
    pub fn ticket(&self) -> &BatchTicket {
        &self.ticket
    }
    pub fn lanes(&self) -> &[BatchLaneOutput] {
        &self.lanes
    }
    pub fn epoch_after(&self) -> u64 {
        self.epoch_after
    }
    pub fn all_checks_passed(&self) -> bool {
        self.all_checks_passed
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn geometry_binding(&self) -> &str {
        &self.global_checkpoint_geometry_binding
    }
    pub fn completion_owner(&self) -> [u8; 32] {
        self.completion_owner
    }
    pub fn model_instance_identity(&self) -> [u8; 32] {
        self.model_instance_identity
    }
    pub fn stream_identity(&self) -> u64 {
        self.stream_identity
    }
    pub fn program_sha256(&self) -> &str {
        &self.program_sha256
    }
    pub fn operations(&self) -> &BatchModelOperationIds {
        &self.operations
    }
    pub fn epoch_before(&self) -> u64 {
        self.epoch_before
    }
    pub fn operation_count_before(&self) -> usize {
        self.operation_count_before
    }
    pub fn operation_count_after(&self) -> usize {
        self.operation_count_after
    }
    pub fn input_native_ids(&self) -> &[u32] {
        &self.input_native_ids
    }
}
/// Caller-attested binary/snapshot identity, checked against the admitted
/// group by the session. It is not a payload checksum or completion proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchRuntimeBinding {
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchModelIdentity {
    pub rank: usize,
    pub boot: Uuid,
    pub revision: String,
    pub manifest_sha256: String,
    pub global_checkpoint_geometry_binding: String,
    pub completion_owner: [u8; 32],
    pub model_instance_identity: [u8; 32],
    pub stream_identity: u64,
    pub frequency: FrequencyProfileIdentity,
    pub max_lanes: usize,
    pub max_learned_prefix: usize,
    pub runtime: Option<BatchRuntimeBinding>,
    pub full_checkpoint_payload_checksums_verified: bool,
    pub fp8_gemm_workspace: Option<GemmWorkspaceBinding>,
    pub projection_status_backend: ProjectionStatusBackend,
}
/// Only the actual model constructs this token after whole-group validation,
/// all physical resets, a GPU fence and checked transport status/epoch. It has
/// no Deserialize/Clone/public constructor and grants no terminal authority.
#[derive(Debug, Serialize)]
pub struct CheckedBatchModelReset {
    model: BatchModelIdentity,
    lanes: Vec<LaneKey>,
    epoch: u64,
    next_batch_version: u64,
    reset_sequence: u64,
}
impl CheckedBatchModelReset {
    pub fn model(&self) -> &BatchModelIdentity {
        &self.model
    }
    pub fn lanes(&self) -> &[LaneKey] {
        &self.lanes
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn next_batch_version(&self) -> u64 {
        self.next_batch_version
    }
    pub fn reset_sequence(&self) -> u64 {
        self.reset_sequence
    }
}
pub struct GpuBatchModel {
    k: Arc<BatchModelKernels>,
    weights: RankModelWeights,
    blocks: Vec<GpuBlockBatch>,
    engram: Vec<GpuBatchEngram>,
    head: GpuBatchTextHead,
    store: GpuBatchStore,
    hashes: EngramHashState,
    plain: VerifiedFrequencyTable,
    yarn: VerifiedFrequencyTable,
    owner: [u8; 32],
    geometry: String,
    instance: [u8; 32],
    runtime: Option<BatchRuntimeBinding>,
    last_completed_batch_version: u64,
    reset_sequence: u64,
    lanes: Vec<Option<LaneState>>,
    last_attempted_head: Option<u64>,
    active: Option<BatchTicket>,
    failed: bool,
}
impl GpuBatchModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: Arc<BatchModelKernels>,
        weights: RankModelWeights,
        config: Dsv41Config,
        map: CompressedTokenMap,
        plain: VerifiedFrequencyTable,
        yarn: VerifiedFrequencyTable,
        comm: &GpuCollectives,
        max_lanes: usize,
    ) -> Result<Self> {
        chunk_plan::validate_config(&config)?;
        weights.manifest.verify_identity()?;
        map.validate_1017()?;
        ensure!(
            (1..=16).contains(&max_lanes)
                && weights.layers.len() == 40
                && comm.rank() == weights.manifest.rank
                && comm.compute_stream().stream == k.stream.stream,
            "packed model complete scopes/rank/stream/lanes differ"
        );
        ensure!(
            plain.select(&[0])?.profile() == RopeProfile::Plain
                && yarn.select(&[0])?.profile() == RopeProfile::Yarn
                && plain.backend() == yarn.backend(),
            "packed full frequency tables/backend differ"
        );
        let owner = comm.completion_identity();
        let mut blocks = Vec::with_capacity(40);
        for layer in 0..40 {
            blocks.push(GpuBlockBatch::new(
                k.blocks.clone(),
                k.attention.clone(),
                k.moe.clone(),
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
                owner,
                max_lanes,
            )?);
        }
        let mut engram = Vec::new();
        for layer in [1, 14] {
            engram.push(GpuBatchEngram::new(
                &k.dev,
                k.stream.clone(),
                k.engram.clone(),
                &config,
                &weights.manifest,
                layer,
                weights.layers[layer].clone(),
            )?);
        }
        let head =
            GpuBatchTextHead::new(&k.dev, &k.stream, weights.global.clone(), &weights.manifest)?;
        let store = GpuBatchStore::new(
            k.store.clone(),
            &config,
            &weights.manifest,
            owner,
            max_lanes,
        )?;
        let hashes = EngramHashState::new(
            EngramHashPlan::official_1017(&config)?,
            map,
            max_lanes,
            8192,
        )?;
        let geometry = global_checkpoint_binding(&weights.manifest)?;
        let mut instance = Sha256::new();
        instance.update(b"dsv41-opaque-packed-model-instance-v1\0");
        instance.update(Uuid::new_v4().as_bytes());
        instance.update(owner);
        instance.update(geometry.as_bytes());
        let instance = instance.finalize().into();
        if let Some(w) = &k.gemm_workspace {
            w.bind_model(instance, k.boot, &k.stream)?;
        }
        k.attention.bind_projection_status_owner(owner)?;
        k.blocks.bind_scratch(owner)?;
        Ok(Self {
            instance,
            runtime: None,
            last_completed_batch_version: 0,
            reset_sequence: 0,
            k,
            weights,
            blocks,
            engram,
            head,
            store,
            hashes,
            plain,
            yarn,
            owner,
            geometry,
            lanes: vec![None; max_lanes],
            last_attempted_head: None,
            active: None,
            failed: false,
        })
    }
    pub fn identity(&self) -> BatchModelIdentity {
        BatchModelIdentity {
            rank: self.weights.manifest.rank,
            boot: self.k.boot,
            revision: MODEL_REVISION.into(),
            manifest_sha256: self.weights.manifest.manifest_sha256.clone(),
            global_checkpoint_geometry_binding: self.geometry.clone(),
            completion_owner: self.owner,
            model_instance_identity: self.instance,
            stream_identity: self.k.stream.stream as usize as u64,
            frequency: self.plain.profile_identity(),
            max_lanes: self.lanes.len(),
            max_learned_prefix: MAX_LEARNED_PREFIX,
            runtime: self.runtime.clone(),
            full_checkpoint_payload_checksums_verified: false,
            fp8_gemm_workspace: self.k.gemm_workspace.as_ref().map(|w| w.binding()),
            projection_status_backend: self.k.projection_status_backend,
        }
    }
    pub fn gemm_dispatch_snapshot(&self) -> Option<GemmDispatchSnapshot> {
        self.k.gemm_workspace.as_ref().map(|w| w.snapshot())
    }
    pub fn projection_status_snapshot(&self) -> Option<ProjectionStatusSnapshot> {
        self.k.attention.projection_status_snapshot()
    }
    pub fn decode_hc_scratch_bytes(&self) -> usize {
        self.k.blocks.scratch_bytes()
    }
    pub fn bind_runtime(&mut self, binding: BatchRuntimeBinding) -> Result<()> {
        ensure!(
            !self.failed
                && self.active.is_none()
                && self.runtime.is_none()
                && self.last_attempted_head.is_none()
                && super::batch_request_plan::sha256(&binding.execution_snapshot_sha256)
                && super::batch_request_plan::sha256(&binding.engine_binary_sha256),
            "runtime binding is immutable and must precede all packed execution"
        );
        self.runtime = Some(binding);
        Ok(())
    }
    pub fn check_reset(&self, reset: &CheckedBatchModelReset, comm: &GpuCollectives) -> Result<()> {
        ensure!(
            !self.failed
                && self.active.is_none()
                && reset.model == self.identity()
                && reset.reset_sequence == self.reset_sequence
                && reset.epoch == comm.expected_epoch()
                && reset.next_batch_version
                    == self
                        .last_completed_batch_version
                        .checked_add(1)
                        .context("batch version overflow")?
                && comm.completion_identity() == self.owner
                && reset.lanes.iter().all(|key| self
                    .lanes
                    .get(key.slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|p| p.key == *key
                        && p.prefix.is_empty()
                        && p.outputs == 0
                        && p.selected.is_none())),
            "reset token is stale, crossed model/session or no longer empty"
        );
        Ok(())
    }
    pub fn poison(&mut self, comm: &mut GpuCollectives) {
        self.failed = true;
        comm.abort_session();
        self.k.blocks.poison_scratch();
        let _ = self.k.dev.synchronize();
    }
    /// Reject any invalid member before changing any physical generation. A
    /// failure after mutation poisons the complete model, never partial reuse.
    /// Read-only prevalidation used before all-rank Prepare is released. No
    /// physical reset or GPU operation is issued by this check.
    pub fn validate_group_reset(&self, lanes: &[LaneKey], comm: &GpuCollectives) -> Result<()> {
        validate_group_reset(
            lanes,
            self.k.boot,
            &self
                .lanes
                .iter()
                .map(|p| p.as_ref().map(|v| v.key.generation))
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            !self.failed
                && self.active.is_none()
                && self.runtime.is_some()
                && comm.completion_identity() == self.owner
                && comm.rank() == self.weights.manifest.rank
                && comm.compute_stream().stream == self.k.stream.stream,
            "group prevalidation requires exact healthy live model/session/stream/runtime"
        );
        Ok(())
    }
    pub fn reset_group_checked(
        &mut self,
        lanes: &[LaneKey],
        comm: &mut GpuCollectives,
    ) -> Result<CheckedBatchModelReset> {
        validate_group_reset(
            lanes,
            self.k.boot,
            &self
                .lanes
                .iter()
                .map(|p| p.as_ref().map(|v| v.key.generation))
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            !self.failed
                && self.active.is_none()
                && self.runtime.is_some()
                && comm.completion_identity() == self.owner
                && comm.rank() == self.weights.manifest.rank
                && comm.compute_stream().stream == self.k.stream.stream,
            "group reset requires this exact healthy live model/stream/transport/runtime"
        );
        let next_sequence = self
            .reset_sequence
            .checked_add(1)
            .context("group reset sequence overflow")?;
        let next_batch_version = self
            .last_completed_batch_version
            .checked_add(1)
            .context("batch version overflow")?;
        let result = (|| {
            for &key in lanes {
                self.reset_lane(key)?;
            }
            self.k.dev.synchronize()?;
            comm.synchronize()?;
            self.reset_sequence = next_sequence;
            Ok(CheckedBatchModelReset {
                model: self.identity(),
                lanes: lanes.to_vec(),
                epoch: comm.expected_epoch(),
                next_batch_version,
                reset_sequence: next_sequence,
            })
        })();
        if result.is_err() {
            self.poison(comm);
        }
        result
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            !self.failed
                && self.active.is_none()
                && key.boot == self.k.boot
                && key.slot < self.lanes.len()
                && key.generation > 0
                && self.lanes[key.slot]
                    .as_ref()
                    .is_none_or(|p| key.generation > p.key.generation),
            "packed model reset must advance a live physical generation"
        );
        let result = (|| {
            self.store.reset_lane(key)?;
            for block in &mut self.blocks {
                block.reset_lane(key)?;
            }
            for block in &mut self.engram {
                block.reset_lane(key)?;
            }
            self.hashes.reset(key.slot)?;
            self.lanes[key.slot] = Some(LaneState {
                key,
                prefix: vec![],
                selected: None,
                outputs: 0,
            });
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
            let _ = self.k.dev.synchronize();
        }
        result
    }
    pub fn execute_checked(
        &mut self,
        kind: PackedKind,
        inputs: &[BatchLaneInput],
        ops: BatchModelOperationIds,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedBatchModel> {
        let result = self.execute_inner(kind, inputs, ops, comm);
        if result.is_err() {
            self.failed = true;
            comm.abort_session();
            self.k.blocks.poison_scratch();
            if let Some(ticket) = self.active.take() {
                let _ = self.store.abort(&ticket);
            }
            let _ = self.k.dev.synchronize();
        }
        result
    }
    fn execute_inner(
        &mut self,
        kind: PackedKind,
        inputs: &[BatchLaneInput],
        ops: BatchModelOperationIds,
        comm: &mut GpuCollectives,
    ) -> Result<CheckedBatchModel> {
        if let Some(w) = &self.k.gemm_workspace {
            w.check_model(self.instance, self.k.boot, &self.k.stream)?;
        }
        ensure!(
            !self.failed
                && self.active.is_none()
                && comm.completion_identity() == self.owner
                && comm.rank() == self.weights.manifest.rank
                && comm.compute_stream().stream == self.k.stream.stream
                && self.last_attempted_head.is_none_or(|id| ops.embedding > id),
            "packed model failed/active/session/op reuse"
        );
        let mut seen = BTreeSet::new();
        let mut segments = Vec::new();
        let mut flat = Vec::new();
        let mut staged_hashes = self.hashes.clone();
        let mut hash_rows: Vec<HashRows> = Vec::new();
        let mut staged_lanes = self.lanes.clone();
        for input in inputs {
            let p = staged_lanes
                .get_mut(input.lane.slot)
                .and_then(Option::as_mut)
                .context("unreset packed input lane")?;
            ensure!(
                p.key == input.lane
                    && seen.insert(input.lane.slot)
                    && !input.native_ids.is_empty()
                    && input
                        .native_ids
                        .iter()
                        .all(|id| (*id as usize) < TOKEN_COUNT),
                "packed input generation/token/duplicate lane"
            );
            let start = p.prefix.len();
            ensure!(
                start
                    .checked_add(input.native_ids.len())
                    .is_some_and(|end| end <= MAX_LEARNED_PREFIX)
                    && input
                        .full_prompt_tokens
                        .is_none_or(|full| (1..=MAX_LEARNED_PREFIX).contains(&full)),
                "first learned packed profile requires every original/consumed prefix <=512 tokens"
            );
            match kind {
                PackedKind::PrefillSegments => ensure!(
                    input.full_prompt_tokens.is_some(),
                    "prefill full prompt extent required"
                ),
                PackedKind::DecodeLanes => ensure!(
                    input.full_prompt_tokens.is_none()
                        && input.native_ids.len() == 1
                        && p.selected == Some(input.native_ids[0])
                        && p.selected != Some(1),
                    "decode must consume last non-EOS selected native token"
                ),
            }
            segments.push(LaneSegment {
                key: p.key,
                start,
                tokens: input.native_ids.len(),
            });
            hash_rows.extend(staged_hashes.append_at(
                p.key.slot,
                start,
                &input.native_ids,
                None,
            )?);
            flat.extend_from_slice(&input.native_ids);
            p.prefix.extend_from_slice(&input.native_ids);
        }
        ops.operations(flat.len(), inputs.len(), kind)?;
        self.last_attempted_head = Some(ops.head);
        let ticket = if kind == PackedKind::PrefillSegments {
            self.store.begin_prefill_segments(
                segments
                    .into_iter()
                    .zip(inputs)
                    .map(|(s, i)| (s, i.full_prompt_tokens.unwrap()))
                    .collect(),
            )?
        } else {
            self.store.begin(kind, segments)?
        };
        self.active = Some(ticket.clone());
        let began = Instant::now();
        let before_epoch = comm.expected_epoch();
        let before_count = comm.queued_operation_count();
        let (selected, blocks, engram) = self.forward(&ticket, &flat, &hash_rows, &ops, comm)?;
        ensure!(
            selected.len() == inputs.len()
                && blocks.len() == 40
                && engram.len() == 2
                && comm.queued_operation_count() == before_count + 84,
            "complete packed model coverage/operation count differs"
        );
        let mut lanes = Vec::new();
        for (segment, token) in ticket.segments().iter().zip(selected) {
            let p = staged_lanes[segment.ticket.lane.slot].as_mut().unwrap();
            let output = match segment.phase {
                AttentionPhase::Decode => true,
                AttentionPhase::Prefill { full_prompt_tokens } => {
                    segment.ticket.end == full_prompt_tokens
                }
            };
            if output {
                p.selected = Some(token);
                p.outputs += 1;
            }
            lanes.push(BatchLaneOutput {
                lane: p.key,
                consumed_prefix_tokens: p.prefix.len(),
                prefix_native_ids_sha256: ids_hash(&p.prefix),
                selected_native_id: token,
                generated_output_ordinal: if output { Some(p.outputs) } else { None },
                intermediate_prefill_head_ignored: !output,
            });
        }
        comm.synchronize()?;
        self.k.dev.synchronize()?;
        self.store.commit(&ticket)?;
        self.hashes = staged_hashes;
        self.lanes = staged_lanes;
        self.active = None;
        self.last_completed_batch_version = ticket.version();
        Ok(CheckedBatchModel {
            ticket,
            rank: self.weights.manifest.rank,
            revision: MODEL_REVISION.into(),
            manifest_sha256: self.weights.manifest.manifest_sha256.clone(),
            global_checkpoint_geometry_binding: self.geometry.clone(),
            completion_owner: self.owner,
            model_instance_identity: self.instance,
            stream_identity: self.k.stream.stream as usize as u64,
            program_sha256: comm.program_digest_hex(),
            operations: ops,
            epoch_before: before_epoch,
            epoch_after: comm.expected_epoch(),
            operation_count_before: before_count,
            operation_count_after: comm.queued_operation_count(),
            input_native_ids: flat,
            lanes,
            blocks,
            engram,
            all_checks_passed: true,
            all_rank_terminal_agreement: false,
            elapsed_seconds: began.elapsed().as_secs_f64(),
            timing_is_benchmark: false,
            fp8_gemm_dispatch: self.gemm_dispatch_snapshot(),
        })
    }
    fn forward(
        &mut self,
        ticket: &BatchTicket,
        input: &[u32],
        hashes: &[HashRows],
        ops: &BatchModelOperationIds,
        comm: &mut GpuCollectives,
    ) -> Result<(Vec<u32>, Vec<BatchBlockReceipt>, Vec<EngramOutputIdentity>)> {
        let k = self.k.clone();
        let s = &k.stream;
        let t = input.len();
        macro_rules! queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{comm.abort_session();let _=k.dev.synchronize();return Err(e.into());}}};}
        let positions = ticket
            .segments()
            .iter()
            .flat_map(|p| p.ticket.start..p.ticket.end)
            .map(|n| n as u32)
            .collect::<Vec<_>>();
        let plain = queued!(k
            .position
            .upload_frequencies(s, &self.plain.select(&positions)?));
        let yarn = queued!(k
            .position
            .upload_frequencies(s, &self.yarn.select(&positions)?));
        let native = Queued::new(&k.dev, queued!(k.dev.htod_sync_copy(input)));
        let mut status = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<u32>(1)));
        let mut partial = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * DIM)));
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
            RoutedLayout::new(t, ticket.kind().collective_rows())?,
            &mut partial
        ));
        let mut h = Queued::new(
            &k.dev,
            queued!(k.dev.alloc_zeros::<half::bf16>(t * 4 * DIM)),
        );
        queued!(k.position.expand(s, &partial, &mut h, &mut status, t, DIM));
        let mut pre = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<f32>(t * 4)));
        queued!(k.hc.initial(s, &mut pre, t));
        let mut blocks = Vec::with_capacity(40);
        let mut engram = Vec::with_capacity(2);
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                let index = usize::from(layer == 14);
                let augmented = queued!(self.engram[index].execute_checked(
                    ticket,
                    &h,
                    hashes,
                    ops.engram[index],
                    &mut self.store,
                    comm
                ));
                engram.push(augmented.identity);
                h = Queued::new(&k.dev, augmented.h);
            }
            let compressed = if [2, 8, 14, 20].contains(&layer) {
                let plan = queued!(batch_compression_plan::build(
                    &self.store.layer(ticket, layer)?
                ));
                if plan.output_rows > 0 {
                    Some(queued!(k.position.upload_frequencies(
                        s,
                        &self.yarn.select(&plan.rope_positions)?
                    )))
                } else {
                    None
                }
            } else {
                None
            };
            let query = if layer < 2 { &plain } else { &yarn };
            let out = queued!(self.blocks[layer].execute_checked(
                ticket,
                &h,
                &pre,
                query,
                compressed.as_ref(),
                &mut self.store,
                comm,
                ops.blocks[layer]
            ));
            ensure!(
                out.receipt.layer == layer
                    && out.receipt.ticket == *ticket
                    && out.receipt.all_checks_passed
                    && out.receipt.exact_cache_layer_committed,
                "packed whole-model layer receipt differs"
            );
            blocks.push(out.receipt);
            h = Queued::new(&k.dev, out.h);
            pre = Queued::new(&k.dev, out.next_pre);
        }
        let mut collapsed = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<half::bf16>(t * DIM)));
        queued!(k
            .hc
            .collapse(s, &h, &pre, &mut collapsed, &mut status, t, DIM));
        let mut head = queued!(self.head.prepare(s, &collapsed, ticket));
        queued!(head.enqueue_selection(comm, ops.head));
        queued!(head.check(comm));
        let selected = queued!(head.selected_tokens());
        queued!(comm.synchronize());
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "packed complete embedding/final HC numeric status"
        );
        Ok((selected, blocks, engram))
    }
}
fn validate_group_reset(lanes: &[LaneKey], boot: Uuid, previous: &[Option<u64>]) -> Result<()> {
    ensure!(
        (1..=16).contains(&lanes.len()) && !boot.is_nil(),
        "bounded nonempty reset group"
    );
    let mut seen = BTreeSet::new();
    ensure!(
        lanes.iter().all(|key| key.boot == boot
            && key.generation > 0
            && key.slot < previous.len()
            && seen.insert(key.slot)
            && previous[key.slot].is_none_or(|g| key.generation > g)),
        "all reset keys must be unique, in bounds and newer BEFORE any mutation"
    );
    Ok(())
}

impl Drop for GpuBatchModel {
    fn drop(&mut self) {
        let _ = self.k.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_group_reset_keys_validate_before_any_physical_effect() {
        let boot = Uuid::from_u128(11);
        let key = |slot, generation| LaneKey {
            boot,
            slot,
            generation,
        };
        let previous = vec![Some(5); 16];
        let good: Vec<_> = (0..16).map(|s| key(s, 6)).collect();
        validate_group_reset(&good, boot, &previous).unwrap();
        for bad in [
            vec![],
            vec![key(0, 6), key(0, 7)],
            vec![key(0, 6), key(15, 5)],
            vec![key(0, 6), key(16, 6)],
            vec![LaneKey {
                boot: Uuid::from_u128(12),
                ..key(0, 6)
            }],
        ] {
            assert!(validate_group_reset(&bad, boot, &previous).is_err());
        }
        assert_eq!(previous, vec![Some(5); 16]);
    }
    #[test]
    fn packed_source_program_has_one84_operation_model_path_at_all_widths() {
        for lanes in [1, 2, 4, 8, 16] {
            let ids = BatchModelOperationIds::consecutive(170).unwrap();
            assert_eq!(ids.head, 253);
            let ops = ids
                .operations(128, lanes, PackedKind::PrefillSegments)
                .unwrap();
            assert_eq!(ops.len(), 84);
            assert!(matches!(ops[0], Operation::RoutedF32 { .. }));
            assert!(matches!(ops[3], Operation::MatrixF32 { .. }));
            let Operation::ShardedMaxloc { layout, .. } = ops.last().unwrap() else {
                panic!("missing global head")
            };
            assert_eq!(
                *layout,
                RoutedLayout::new(lanes, RowKind::IndependentDecodeLanes).unwrap()
            );
            let decode = ids
                .operations(lanes, lanes, PackedKind::DecodeLanes)
                .unwrap();
            let Operation::RoutedF32 { layout, .. } = decode[0] else {
                panic!("missing embedding")
            };
            assert_eq!(
                layout,
                RoutedLayout::new(lanes, RowKind::IndependentDecodeLanes).unwrap()
            );
        }
    }
    #[test]
    fn packed_model_program_refuses_missing_swapped_reused_roles_or_overflow() {
        let good = BatchModelOperationIds::consecutive(7).unwrap();
        let mut bad = good.clone();
        bad.head -= 1;
        assert!(bad.operations(16, 16, PackedKind::DecodeLanes).is_err());
        bad = good.clone();
        bad.blocks[2].attention_f32 += 1;
        assert!(bad.operations(16, 16, PackedKind::DecodeLanes).is_err());
        bad = good.clone();
        bad.engram.swap(0, 1);
        assert!(bad.operations(16, 16, PackedKind::DecodeLanes).is_err());
        bad = good.clone();
        bad.blocks.pop();
        assert!(bad.operations(16, 16, PackedKind::DecodeLanes).is_err());
        assert!(BatchModelOperationIds::consecutive(u64::MAX - 82).is_err());
        for (rows, lanes) in [(0, 1), (129, 16), (16, 0), (16, 17)] {
            assert!(good
                .operations(rows, lanes, PackedKind::DecodeLanes)
                .is_err());
        }
    }
}
