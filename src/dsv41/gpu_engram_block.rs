//! Owned native1017 text Engram composition for layers1/14.
//!
//! HashRows must be computed once per model step by the verified CPU
//! EngramHashState; both layers consume that same immutable result. Only the
//! small hash IDs move from CPU to GPU. Native table payloads, lookup values,
//! projection and residuals stay on GPU. This conservative executor fences
//! tiles/statuses; it does not claim full-model or serving qualification.
use super::{
    batch_plan::BatchTicket,
    collectives::{MatrixF32, MatrixRole},
    config::{Dsv41Config, MODEL_REVISION, REFERENCE_ENGRAM_SHA256},
    engram::{EngramHashPlan, HashRows, HASH_COLUMNS},
    gpu_batch_store::GpuBatchStore,
    gpu_collectives::GpuCollectives,
    gpu_engram::EngramKernels,
    gpu_gemm::{GemmShape, GpuGemm},
    gpu_numerics::GpuNumerics,
    gpu_position::PositionKernels,
    gpu_weights::{LoadScope, LoadedWeights},
    plan::row_ranges,
    scheduler::LaneKey,
    weights::{Conversion, Group, RankManifest, Selection, TensorLayout},
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

const DIM: usize = 5120;
const COPIES: usize = 4;
const LOOKUP_COLUMNS: usize = HASH_COLUMNS * 256;
const PROJECTION_OUTPUTS: usize = (COPIES + 1) * DIM;
const TILE_OUTPUTS: usize = 4096;
const MAX_TOKENS: usize = 128;
/// Temporary arrays only, excluding loaded model weights/CUDA modules/context.
pub const STEP_ARRAY_BOUND_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ProjectionTile {
    first_output: usize,
    outputs: usize,
    weight_offset_bytes: usize,
    weight_bytes: usize,
    scale_offset_bytes: usize,
    scale_bytes: usize,
}
impl ProjectionTile {
    fn new(first_output: usize, outputs: usize) -> Result<Self> {
        ensure!(
            first_output <= PROJECTION_OUTPUTS && outputs <= TILE_OUTPUTS,
            "native Engram tile bounds"
        );
        let tile = Self {
            first_output,
            outputs,
            weight_offset_bytes: first_output * LOOKUP_COLUMNS,
            weight_bytes: outputs * LOOKUP_COLUMNS,
            scale_offset_bytes: first_output / 32 * (LOOKUP_COLUMNS / 32),
            scale_bytes: outputs / 32 * (LOOKUP_COLUMNS / 32),
        };
        tile.validate()?;
        Ok(tile)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.first_output % 32 == 0
                && self.outputs > 0
                && self.outputs <= TILE_OUTPUTS
                && self.outputs % 32 == 0
                && self
                    .first_output
                    .checked_add(self.outputs)
                    .is_some_and(|n| n <= PROJECTION_OUTPUTS),
            "native Engram output-row tile outside group32 bounds"
        );
        ensure!(
            self.weight_offset_bytes == self.first_output * LOOKUP_COLUMNS
                && self.weight_bytes == self.outputs * LOOKUP_COLUMNS
                && self.scale_offset_bytes == self.first_output / 32 * (LOOKUP_COLUMNS / 32)
                && self.scale_bytes == self.outputs / 32 * (LOOKUP_COLUMNS / 32),
            "native Engram weight/scale tile row association differs"
        );
        GemmShape {
            rows: MAX_TOKENS,
            outputs: self.outputs,
            inner: LOOKUP_COLUMNS,
        }
        .validate()?;
        Ok(())
    }
}
fn projection_tiles() -> Result<Vec<ProjectionTile>> {
    (0..PROJECTION_OUTPUTS)
        .step_by(TILE_OUTPUTS)
        .map(|first| ProjectionTile::new(first, (PROJECTION_OUTPUTS - first).min(TILE_OUTPUTS)))
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct EngramLayout {
    pub layer: usize,
    pub layer_hash_index: usize,
    pub rank: usize,
    pub logical_table_rows: u64,
    pub first_row: u64,
    pub valid_rows: u64,
    pub allocated_rows: u64,
    /// Source ParallelEngramEmbedding allocates ceil rows/rank. Padding IDs are
    /// rejected against the real logical buckets before the lookup is queued.
    pub kernel_allocation_extent_rows: u64,
    pub projection_tiles: Vec<ProjectionTile>,
}
impl EngramLayout {
    fn official(config: &Dsv41Config, rank: usize, layer: usize) -> Result<Self> {
        let plan = EngramHashPlan::official_1017(config)?;
        ensure!(rank < 8, "Engram rank outside TP8");
        let layer_hash_index = plan
            .layer_ids
            .iter()
            .position(|&x| x == layer)
            .context("only official layers1/14 have Engram")?;
        let logical_table_rows = plan.num_embeddings[layer_hash_index];
        let allocated_rows = logical_table_rows.div_ceil(8);
        let ranges = row_ranges(logical_table_rows, 8)?;
        Ok(Self {
            layer,
            layer_hash_index,
            rank,
            logical_table_rows,
            first_row: ranges[rank].start,
            valid_rows: ranges[rank].count,
            allocated_rows,
            kernel_allocation_extent_rows: allocated_rows * 8,
            projection_tiles: projection_tiles()?,
        })
    }
    fn indices(&self, plan: &EngramHashPlan, rows: &[HashRows]) -> Result<Vec<u64>> {
        ensure!(
            (1..=MAX_TOKENS).contains(&rows.len()),
            "Engram initial-prefill token bound1..128"
        );
        let mut out = Vec::with_capacity(rows.len() * HASH_COLUMNS);
        for row in rows {
            for (column, &id) in row[self.layer_hash_index].iter().enumerate() {
                let first = plan.offsets[self.layer_hash_index][column];
                let end = first + plan.primes[self.layer_hash_index][column];
                ensure!(
                    id >= first && id < end && id < self.logical_table_rows,
                    "Engram hash ID is outside its exact official bucket/logical table"
                );
                out.push(id);
            }
        }
        Ok(out)
    }
}

fn step_end(start: usize, tokens: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_TOKENS).contains(&tokens) && (start == 0 || tokens == 1),
        "Engram initial1..128 tokens then one-token decode required"
    );
    start
        .checked_add(tokens)
        .filter(|&x| x <= 1_048_576)
        .context("Engram native position exceeds official context")
}
fn expected_tensor<'a>(
    manifest: &'a RankManifest,
    name: &str,
    dtype: &str,
    source: &[u64],
    selected: &[u64],
    selection: Selection,
    group: Group,
) -> Result<&'a TensorLayout> {
    let e = manifest
        .entries
        .iter()
        .find(|e| e.name == name)
        .with_context(|| format!("missing native Engram tensor{name}"))?;
    ensure!(
        e.dtype == dtype
            && e.source_shape == source
            && e.selected_shape == selected
            && e.selection == selection
            && e.group == group
            && e.conversion == Conversion::Native,
        "native Engram dtype/shape/source ownership changed:{name}"
    );
    Ok(e)
}

pub struct EngramBlockKernels {
    dev: Arc<CudaDevice>,
    stream: usize,
    boot: Uuid,
    lookup_gate: EngramKernels,
    position: PositionKernels,
    numerics: GpuNumerics,
    gemm: GpuGemm,
}
impl EngramBlockKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream, boot: Uuid) -> Result<Arc<Self>> {
        let me = Self {
            dev: dev.clone(),
            stream: stream.stream as usize,
            boot,
            lookup_gate: EngramKernels::load(dev, stream)?,
            position: PositionKernels::load(dev, stream)?,
            numerics: GpuNumerics::load(dev, stream)?,
            gemm: GpuGemm::load(dev, stream)?,
        };
        me.stream(stream)?;
        Ok(Arc::new(me))
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !s.stream.is_null() && s.stream as usize == self.stream,
            "exact Engram owner stream required"
        );
        let mut flags = 0;
        ensure!(
            unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) } == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "blocking Engram owner stream required"
        );
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct Progress {
    key: LaneKey,
    position: usize,
    initialized: bool,
    failed: bool,
}
/// Exact physical-lane progress reserved by one flattened Engram invocation.
#[derive(Clone, Copy, Debug, Serialize)]
struct ProgressExtent {
    key: LaneKey,
    start: usize,
    end: usize,
}
/// The singular source fields retain their original C1 meaning. This optional
/// identity describes every real participant when the shared row math is packed.
#[derive(Clone, Debug, Serialize)]
pub struct PackedEngramIdentity {
    pub ticket: BatchTicket,
    pub semantic_progress: Vec<(LaneKey, usize, usize)>,
    pub one_shared_lookup_projection_gate: bool,
}
pub struct GpuEngramBlock {
    kernels: Arc<EngramBlockKernels>,
    weights: Arc<LoadedWeights>,
    manifest_sha256: String,
    layout: EngramLayout,
    hash_plan: EngramHashPlan,
    lanes: [Option<Progress>; 16],
}
impl GpuEngramBlock {
    pub fn new(
        kernels: Arc<EngramBlockKernels>,
        config: &Dsv41Config,
        manifest: &RankManifest,
        layer: usize,
        weights: Arc<LoadedWeights>,
    ) -> Result<Self> {
        manifest.verify_identity()?;
        ensure!(
            manifest.model_revision == MODEL_REVISION
                && weights.scope() == &LoadScope::Layer(layer)
                && weights.manifest_sha256() == manifest.manifest_sha256,
            "Engram actual checkpoint scope/manifest/revision differs"
        );
        let layout = EngramLayout::official(config, manifest.rank, layer)?;
        let prefix = format!("layers.{layer}.engram");
        for (name, columns, dtype) in [
            ("embed.weight", 256, "F8_E4M3"),
            ("embed.scale", 8, "F8_E8M0"),
        ] {
            let name = format!("{prefix}.{name}");
            let e = expected_tensor(
                manifest,
                &name,
                dtype,
                &[layout.logical_table_rows, columns],
                &[layout.valid_rows, columns],
                Selection::Rows {
                    first: layout.first_row,
                    count: layout.valid_rows,
                    allocated_rows: layout.allocated_rows,
                },
                Group::Layer(layer),
            )?;
            ensure!(
                e.native_bytes == layout.valid_rows * columns
                    && e.allocation_padding_bytes
                        == (layout.allocated_rows - layout.valid_rows) * columns
                    && weights.native_u8(&name)?.len() as u64 == layout.allocated_rows * columns
                    && Arc::ptr_eq(&weights.native_u8(&name)?.device(), &kernels.dev),
                "Engram table allocation/zero-padding/device differs"
            );
        }
        for (name, shape, dtype) in [
            ("wkv.weight", vec![25600, 6144], "F8_E4M3"),
            ("wkv.scale", vec![800, 192], "F8_E8M0"),
            ("q_weight", vec![4, 5120], "BF16"),
            ("k_weight", vec![4, 5120], "BF16"),
        ] {
            let name = format!("{prefix}.{name}");
            let e = expected_tensor(
                manifest,
                &name,
                dtype,
                &shape,
                &shape,
                Selection::Full,
                Group::Layer(layer),
            )?;
            ensure!(
                e.allocation_padding_bytes == 0,
                "replicated Engram projection/gate has unexpected padding"
            );
            let same = if dtype == "BF16" {
                let b = weights.bf16(&name)?;
                b.len() == 20480 && Arc::ptr_eq(&b.device(), &kernels.dev)
            } else {
                let b = weights.native_u8(&name)?;
                b.len() == shape.iter().product::<u64>() as usize
                    && Arc::ptr_eq(&b.device(), &kernels.dev)
            };
            ensure!(same, "replicated native Engram buffer/device differs");
        }
        Ok(Self {
            kernels,
            weights,
            manifest_sha256: manifest.manifest_sha256.clone(),
            layout,
            hash_plan: EngramHashPlan::official_1017(config)?,
            lanes: [None; 16],
        })
    }
    pub fn layout(&self) -> &EngramLayout {
        &self.layout
    }
    pub fn reset_lane(&mut self, s: &CudaStream, key: LaneKey) -> Result<()> {
        self.kernels.stream(s)?;
        ensure!(
            key.boot == self.kernels.boot && key.slot < 16 && key.generation > 0,
            "Engram lane boot/slot/generation differs"
        );
        ensure!(
            self.lanes[key.slot].is_none_or(|old| key.generation > old.key.generation),
            "Engram reset must advance physical-slot generation"
        );
        self.lanes[key.slot] = Some(Progress {
            key,
            position: 0,
            initialized: false,
            failed: false,
        });
        Ok(())
    }
    fn poison(&mut self, key: LaneKey) {
        if let Some(p) = self.lanes.get_mut(key.slot).and_then(Option::as_mut) {
            if p.key == key {
                p.failed = true;
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_lookup<'a>(
        &'a mut self,
        s: &'a CudaStream,
        key: LaneKey,
        start: usize,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation_id: u64,
        collectives: &mut GpuCollectives,
    ) -> Result<PendingEngram<'a>> {
        self.enqueue_inner(s, key, start, h, hash_rows, None, operation_id, collectives)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_lookup_for_step<'a>(
        &'a mut self,
        s: &'a CudaStream,
        key: LaneKey,
        start: usize,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        step: u64,
        operation_id: u64,
        collectives: &mut GpuCollectives,
    ) -> Result<PendingEngram<'a>> {
        self.enqueue_inner(
            s,
            key,
            start,
            h,
            hash_rows,
            Some(step),
            operation_id,
            collectives,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_inner<'a>(
        &'a mut self,
        s: &'a CudaStream,
        key: LaneKey,
        start: usize,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        step: Option<u64>,
        operation_id: u64,
        collectives: &mut GpuCollectives,
    ) -> Result<PendingEngram<'a>> {
        // The original entrypoint and refusal tests keep their exact extent gate.
        let end = step_end(start, hash_rows.len())?;
        self.enqueue_rows(
            s,
            key,
            start,
            h,
            hash_rows,
            step,
            operation_id,
            collectives,
            end,
        )
    }
    /// Additive row entrypoint for the separately bound chunk wrapper only.
    /// This is not a public guard relaxation or a CPU history/prefix commit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_chunk_rows<'a>(
        &'a mut self,
        s: &'a CudaStream,
        ticket: &super::chunk_plan::ChunkTicket,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation_id: u64,
        collectives: &mut GpuCollectives,
    ) -> Result<PendingEngram<'a>> {
        ensure!(
            ticket.lane.slot == 0
                && ticket.lane.generation > 0
                && ticket.version > 0
                && (1..=MAX_TOKENS).contains(&ticket.tokens)
                && hash_rows.len() == ticket.tokens
                && ticket.start.checked_add(ticket.tokens) == Some(ticket.end)
                && ticket.end <= 8192,
            "Engram chunk ticket extent/generation differs"
        );
        self.enqueue_rows(
            s,
            ticket.lane,
            ticket.start,
            h,
            hash_rows,
            None,
            operation_id,
            collectives,
            ticket.end,
        )
    }
    // Shared row computation: the entire lookup/reduce/projection/gate schedule
    // below remains unchanged. Only callers choose their separately checked extent.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_rows<'a>(
        &'a mut self,
        s: &'a CudaStream,
        key: LaneKey,
        start: usize,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        step: Option<u64>,
        operation_id: u64,
        collectives: &mut GpuCollectives,
        end: usize,
    ) -> Result<PendingEngram<'a>> {
        self.kernels.stream(s)?;
        let tokens = hash_rows.len();
        ensure!(
            key.boot == self.kernels.boot && key.slot < 16,
            "Engram request boot/slot differs"
        );
        let p = self.lanes[key.slot].context("Engram lane must be explicitly reset before use")?;
        ensure!(
            p.key == key && !p.failed && p.position == start && (!p.initialized || start > 0),
            "stale/failed/noncontiguous Engram step"
        );
        self.enqueue_rows_math(
            s,
            key,
            start,
            h,
            hash_rows,
            step,
            operation_id,
            collectives,
            end,
            vec![ProgressExtent { key, start, end }],
            None,
        )
    }
    /// An additive packed entrypoint; original single-lane extent gates above
    /// remain unchanged. Every semantic lane is validated before any GPU work.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_packed_rows<'a>(
        &'a mut self,
        s: &'a CudaStream,
        ticket: &BatchTicket,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation_id: u64,
        store: &GpuBatchStore,
        collectives: &mut GpuCollectives,
    ) -> Result<PendingEngram<'a>> {
        self.kernels.stream(s)?;
        store.check_owner(
            self.layout.rank,
            &self.manifest_sha256,
            collectives.completion_identity(),
            s,
        )?;
        let plan = store.layer(ticket, self.layout.layer)?;
        ensure!(
            plan.ticket == *ticket
                && ticket.boot() == self.kernels.boot
                && ticket.total_rows() == hash_rows.len()
                && (1..=MAX_TOKENS).contains(&hash_rows.len()),
            "packed Engram active ticket/row extent differs"
        );
        let extents = self.packed_progress(ticket)?;
        let first = extents
            .first()
            .context("packed Engram has no participants")?;
        let packed = PackedEngramIdentity {
            ticket: ticket.clone(),
            semantic_progress: extents.iter().map(|e| (e.key, e.start, e.end)).collect(),
            one_shared_lookup_projection_gate: true,
        };
        self.enqueue_rows_math(
            s,
            first.key,
            first.start,
            h,
            hash_rows,
            None,
            operation_id,
            collectives,
            first.end,
            extents,
            Some(packed),
        )
    }
    fn packed_progress(&self, ticket: &BatchTicket) -> Result<Vec<ProgressExtent>> {
        packed_progress(self.kernels.boot, &self.lanes, ticket)
    }
    fn poison_extents(&mut self, extents: &[ProgressExtent]) {
        for e in extents {
            self.poison(e.key);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_rows_math<'a>(
        &'a mut self,
        s: &'a CudaStream,
        key: LaneKey,
        start: usize,
        h: &'a CudaSlice<bf16>,
        hash_rows: &[HashRows],
        step: Option<u64>,
        operation_id: u64,
        collectives: &mut GpuCollectives,
        end: usize,
        progress: Vec<ProgressExtent>,
        packed: Option<PackedEngramIdentity>,
    ) -> Result<PendingEngram<'a>> {
        let tokens = hash_rows.len();
        ensure!(
            h.len() == tokens * COPIES * DIM && Arc::ptr_eq(&h.device(), &self.kernels.dev),
            "Engram residual HC4 geometry/device differs"
        );
        ensure!(
            collectives.rank() == self.layout.rank
                && collectives.compute_stream().stream == s.stream,
            "Engram collective rank/stream differs"
        );
        let host_indices = self.layout.indices(&self.hash_plan, hash_rows)?;
        let hash_rows_sha256_u64le = format!(
            "{:x}",
            Sha256::digest(
                host_indices
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            )
        );
        let k = self.kernels.clone();
        macro_rules! queued {($e:expr)=>{match (||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();self.poison_extents(&progress);return Err(e.into());}}};}
        let indices = queued!(k.dev.htod_sync_copy(&host_indices));
        let mut values = queued!(k.dev.alloc_zeros::<f32>(tokens * LOOKUP_COLUMNS));
        let mut status = queued!(k.dev.alloc_zeros::<u32>(1));
        let prefix = format!("layers.{}.engram", self.layout.layer);
        queued!(k.lookup_gate.lookup(
            s,
            self.weights.native_u8(&format!("{prefix}.embed.weight"))?,
            self.weights.native_u8(&format!("{prefix}.embed.scale"))?,
            &indices,
            &mut values,
            &mut status,
            self.layout.first_row,
            self.layout.allocated_rows,
            self.layout.kernel_allocation_extent_rows
        ));
        // This first composition deliberately validates local lookup before
        // entering an all-rank transfer. No embedding values are copied to CPU.
        queued!(k.dev.synchronize());
        let code = queued!(k.dev.dtoh_sync_copy(&status))[0];
        queued!(if code == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Engram local lookup status0x{code:x}"))
        });
        let layout = queued!(MatrixF32::new(
            MatrixRole::EngramLookup,
            tokens,
            LOOKUP_COLUMNS
        ));
        let owner = collectives.completion_identity();
        queued!(match step {
            Some(step) =>
                collectives.enqueue_matrix_for_step(step, operation_id, layout, &mut values),
            None => collectives.enqueue_matrix(operation_id, layout, &mut values),
        });
        let epoch = collectives.expected_epoch();
        Ok(PendingEngram {
            block: self,
            s,
            key,
            start,
            end,
            progress,
            packed,
            tokens,
            h,
            _indices: indices,
            values,
            status,
            operation_id,
            step,
            completion_identity: owner,
            collective_epoch_after_enqueue: epoch,
            hash_rows_sha256_u64le,
            completed: false,
        })
    }
}
fn packed_progress(
    boot: Uuid,
    lanes: &[Option<Progress>; 16],
    ticket: &BatchTicket,
) -> Result<Vec<ProgressExtent>> {
    let mut extents = Vec::with_capacity(ticket.lanes());
    let mut row = 0;
    let mut slots = std::collections::BTreeSet::new();
    for segment in ticket.segments() {
        let t = &segment.ticket;
        ensure!(
            t.lane.boot == boot
                && t.lane.slot < 16
                && t.lane.generation > 0
                && slots.insert(t.lane.slot)
                && segment.first_row == row
                && segment.rows == t.tokens
                && t.tokens > 0
                && t.start.checked_add(t.tokens) == Some(t.end)
                && t.end <= 8192,
            "packed Engram semantic extent/row ordering differs"
        );
        let p = lanes[t.lane.slot].context("packed Engram lane not reset")?;
        ensure!(
            p.key == t.lane
                && !p.failed
                && p.position == t.start
                && (!p.initialized || t.start > 0),
            "packed Engram stale/failed/noncontiguous participant"
        );
        extents.push(ProgressExtent {
            key: t.lane,
            start: t.start,
            end: t.end,
        });
        row += segment.rows;
    }
    ensure!(
        row == ticket.total_rows() && !extents.is_empty(),
        "packed Engram rows do not cover every participant"
    );
    Ok(extents)
}
impl Drop for GpuEngramBlock {
    fn drop(&mut self) {
        let _ = self.kernels.dev.synchronize();
    }
}

#[derive(Debug, Serialize)]
pub struct EngramOutputIdentity {
    pub revision: String,
    pub manifest_sha256: String,
    pub layer: usize,
    pub rank: usize,
    pub lane: LaneKey,
    pub start: usize,
    pub tokens: usize,
    pub hash_rows_sha256_u64le: String,
    pub hash_reference_source_sha256: String,
    pub lookup_columns: usize,
    pub operation_id: u64,
    pub step: Option<u64>,
    pub completion_identity: [u8; 32],
    pub collective_epoch_after_enqueue: u64,
    pub projection_tiles: Vec<ProjectionTile>,
    pub lookup_precision: String,
    pub wkv_precision: String,
    pub text_mask_all_true: bool,
    pub temporary_array_policy_bytes: usize,
    pub all_checks_passed: bool,
    pub whole_model_qualified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packed: Option<PackedEngramIdentity>,
}
pub struct EngramOutput {
    pub h: CudaSlice<bf16>,
    pub gates: CudaSlice<f32>,
    pub identity: EngramOutputIdentity,
}
pub struct PendingEngram<'a> {
    block: &'a mut GpuEngramBlock,
    s: &'a CudaStream,
    key: LaneKey,
    start: usize,
    end: usize,
    progress: Vec<ProgressExtent>,
    packed: Option<PackedEngramIdentity>,
    tokens: usize,
    h: &'a CudaSlice<bf16>,
    _indices: CudaSlice<u64>,
    values: CudaSlice<f32>,
    status: CudaSlice<u32>,
    operation_id: u64,
    step: Option<u64>,
    completion_identity: [u8; 32],
    collective_epoch_after_enqueue: u64,
    hash_rows_sha256_u64le: String,
    completed: bool,
}

fn copy_native_range(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    source: &CudaSlice<u8>,
    offset: usize,
    dest: &mut CudaSlice<u8>,
) -> Result<()> {
    ensure!(
        Arc::ptr_eq(&source.device(), dev)
            && Arc::ptr_eq(&dest.device(), dev)
            && offset
                .checked_add(dest.len())
                .is_some_and(|n| n <= source.len()),
        "native Engram tile source range/device differs"
    );
    let pointer = (*source.device_ptr())
        .checked_add(offset as u64)
        .context("native Engram pointer overflow")?;
    let rc =
        unsafe { sys::cuMemcpyDtoDAsync_v2(*dest.device_ptr(), pointer, dest.len(), s.stream) };
    ensure!(
        rc == sys::CUresult::CUDA_SUCCESS,
        "native Engram tile GPU copy failed{rc:?}"
    );
    Ok(())
}
fn concatenate_projection(
    s: &CudaStream,
    source: &CudaSlice<bf16>,
    dest: &mut CudaSlice<bf16>,
    tokens: usize,
    tile: &ProjectionTile,
) -> Result<()> {
    tile.validate()?;
    ensure!(
        Arc::ptr_eq(&source.device(), &dest.device())
            && source.len() == tokens * tile.outputs
            && dest.len() == tokens * PROJECTION_OUTPUTS,
        "Engram projection concat extent/device differs"
    );
    let copy = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: *source.device_ptr(),
        srcArray: std::ptr::null_mut(),
        srcPitch: tile.outputs * 2,
        dstXInBytes: tile.first_output * 2,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: *dest.device_ptr(),
        dstArray: std::ptr::null_mut(),
        dstPitch: PROJECTION_OUTPUTS * 2,
        WidthInBytes: tile.outputs * 2,
        Height: tokens,
    };
    let rc = unsafe { sys::cuMemcpy2DAsync_v2(&copy, s.stream) };
    ensure!(
        rc == sys::CUresult::CUDA_SUCCESS,
        "Engram complete-BF16 tile concat failed{rc:?}"
    );
    Ok(())
}
impl PendingEngram<'_> {
    /// Completes the SAME all-rank lookup owner, then applies source group32
    /// native FP8 wkv and the HC4 gate. Returns only fenced, checked GPU owners.
    pub fn finish(mut self, collectives: &mut GpuCollectives) -> Result<EngramOutput> {
        let k = self.block.kernels.clone();
        macro_rules! queued {($e:expr)=>{match (||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();self.block.poison_extents(&self.progress);return Err(e.into());}}};}
        queued!(k.stream(self.s));
        queued!(
            if collectives.completion_identity() == self.completion_identity
                && collectives.rank() == self.block.layout.rank
                && collectives.compute_stream().stream == self.s.stream
                && collectives.expected_epoch() >= self.collective_epoch_after_enqueue
            {
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Engram completion owner/session/epoch differs"
                ))
            }
        );
        queued!(collectives.synchronize());
        let mut lookup_bf16 = queued!(k.dev.alloc_zeros::<bf16>(self.tokens * LOOKUP_COLUMNS));
        queued!(k
            .position
            .cast_f32_bf16(self.s, &self.values, &mut lookup_bf16, &mut self.status));
        let quant = queued!(k.numerics.quantize_activations_bf16(
            self.s,
            &lookup_bf16,
            self.tokens,
            LOOKUP_COLUMNS
        ));
        queued!(quant.check());
        let mut kv = queued!(k.dev.alloc_zeros::<bf16>(self.tokens * PROJECTION_OUTPUTS));
        let prefix = format!("layers.{}.engram", self.block.layout.layer);
        let tiles = self.block.layout.projection_tiles.clone();
        for tile in &tiles {
            queued!(tile.validate());
            let mut tw = queued!(k.dev.alloc_zeros::<u8>(tile.weight_bytes));
            let mut ts = queued!(k.dev.alloc_zeros::<u8>(tile.scale_bytes));
            queued!(copy_native_range(
                &k.dev,
                self.s,
                self.block
                    .weights
                    .native_u8(&format!("{prefix}.wkv.weight"))?,
                tile.weight_offset_bytes,
                &mut tw
            ));
            queued!(copy_native_range(
                &k.dev,
                self.s,
                self.block
                    .weights
                    .native_u8(&format!("{prefix}.wkv.scale"))?,
                tile.scale_offset_bytes,
                &mut ts
            ));
            let projected = queued!(unsafe {
                k.gemm.fp8_complete_bf16(
                    self.s,
                    quant.device_values_unchecked(),
                    quant.device_scales_unchecked(),
                    &tw,
                    &ts,
                    GemmShape {
                        rows: self.tokens,
                        outputs: tile.outputs,
                        inner: LOOKUP_COLUMNS,
                    },
                )
            });
            queued!(unsafe {
                concatenate_projection(
                    self.s,
                    projected.device_values_unchecked(),
                    &mut kv,
                    self.tokens,
                    tile,
                )
            });
            // Ensures both copied native tile inputs and its complete BF16
            // output remain alive until the concatenation has completed.
            queued!(projected.check());
        }
        let mask = queued!(k.dev.htod_sync_copy(&vec![1u8; self.tokens]));
        let mut h = queued!(k.dev.alloc_zeros::<bf16>(self.tokens * COPIES * DIM));
        let mut gates = queued!(k.dev.alloc_zeros::<f32>(self.tokens * COPIES));
        queued!(k.lookup_gate.gate(
            self.s,
            self.h,
            &kv,
            self.block.weights.bf16(&format!("{prefix}.q_weight"))?,
            self.block.weights.bf16(&format!("{prefix}.k_weight"))?,
            &mask,
            &mut h,
            &mut gates,
            &mut self.status,
            self.tokens,
            DIM
        ));
        queued!(k.dev.synchronize());
        let code = queued!(k.dev.dtoh_sync_copy(&self.status))[0];
        queued!(if code == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Engram lookup/cast/gate status0x{code:x}"))
        });
        // Validate ALL participants before advancing ANY lane. A late failure
        // poisons every member through the pending owner's Drop path.
        for e in &self.progress {
            let p = self.block.lanes[e.key.slot]
                .as_ref()
                .context("Engram participant disappeared")?;
            ensure!(
                p.key == e.key && !p.failed && p.position == e.start,
                "Engram completion participant generation/progress differs"
            );
        }
        for e in &self.progress {
            let p = self.block.lanes[e.key.slot]
                .as_mut()
                .expect("validated participant");
            p.position = e.end;
            p.initialized = true;
        }
        self.completed = true;
        Ok(EngramOutput {h,gates,identity:EngramOutputIdentity {
            revision:MODEL_REVISION.into(),manifest_sha256:self.block.manifest_sha256.clone(),layer:self.block.layout.layer,rank:self.block.layout.rank,lane:self.key,start:self.start,tokens:self.tokens,
            hash_rows_sha256_u64le:self.hash_rows_sha256_u64le.clone(),hash_reference_source_sha256:REFERENCE_ENGRAM_SHA256.into(),lookup_columns:LOOKUP_COLUMNS,operation_id:self.operation_id,step:self.step,
            completion_identity:self.completion_identity,collective_epoch_after_enqueue:self.collective_epoch_after_enqueue,projection_tiles:self.block.layout.projection_tiles.clone(),
            lookup_precision:"native E4M3×UE8M0 -> BF16 round on unique owner -> F32 all-rank sum -> BF16".into(),wkv_precision:"source group32 activation E4M3/UE8M0 -> unchanged native FP8 complete BF16 output tiles; original K order".into(),
            text_mask_all_true:true,temporary_array_policy_bytes:STEP_ARRAY_BOUND_BYTES,all_checks_passed:true,whole_model_qualified:false,packed:self.packed.clone(),
        }})
    }
}
impl Drop for PendingEngram<'_> {
    fn drop(&mut self) {
        let _ = self.block.kernels.dev.synchronize();
        if !self.completed {
            self.block.poison_extents(&self.progress);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Dsv41Config {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        Dsv41Config::from_values(&v["hf_config"], &v["inference_config"]).unwrap()
    }
    #[test]
    fn native_projection_tiles_preserve_exact_output_and_scale_rows() {
        assert!(PROJECTION_OUTPUTS * LOOKUP_COLUMNS > super::super::gpu_gemm::MAX_MATRIX_ELEMENTS);
        let tiles = projection_tiles().unwrap();
        assert_eq!(tiles.len(), 7);
        assert_eq!(
            tiles.iter().map(|t| t.outputs).collect::<Vec<_>>(),
            [4096, 4096, 4096, 4096, 4096, 4096, 1024]
        );
        let mut out = 0;
        let mut weights = 0;
        let mut scales = 0;
        for t in &tiles {
            t.validate().unwrap();
            assert_eq!(t.first_output, out);
            assert_eq!(t.weight_offset_bytes, weights);
            assert_eq!(t.scale_offset_bytes, scales);
            out += t.outputs;
            weights += t.weight_bytes;
            scales += t.scale_bytes;
        }
        assert_eq!(out, 25600);
        assert_eq!(weights, 25600 * 6144);
        assert_eq!(scales, 800 * 192);
        let mut wrong = tiles[1].clone();
        wrong.scale_offset_bytes += 192;
        assert!(wrong.validate().is_err());
        let mut wrong = tiles[1].clone();
        wrong.weight_offset_bytes += 32;
        assert!(wrong.validate().is_err());
        assert!(ProjectionTile::new(1, 4096).is_err());
        assert!(ProjectionTile::new(24576, 4096).is_err());
        // Includes local/reduced lookup, indices, cast/Q32 activation, output
        // KV, updated residual/gates/mask and one live native GEMM tile/status.
        let gpu_arrays = 128
            * (6144 * 4 + 24 * 8 + 6144 * 2 + 6144 + 6144 / 32 + 25600 * 2 + 20480 * 2 + 4 * 4 + 1)
            + 4096 * 6144
            + 128 * 192
            + 128 * 4096 * 2
            + 32;
        assert!(gpu_arrays < STEP_ARRAY_BOUND_BYTES);
    }
    #[test]
    fn exact_bucket_ranges_and_padded_owner_cannot_accept_padding_hash_ids() {
        let cfg = config();
        let plan = EngramHashPlan::official_1017(&cfg).unwrap();
        for layer in [1, 14] {
            for rank in 0..8 {
                let g = EngramLayout::official(&cfg, rank, layer).unwrap();
                assert!(g.first_row + g.valid_rows <= g.logical_table_rows);
                assert!(g.first_row + g.allocated_rows <= g.kernel_allocation_extent_rows);
                let mut rows = [[[0; HASH_COLUMNS]; 2]; 1];
                for l in 0..2 {
                    for c in 0..HASH_COLUMNS {
                        rows[0][l][c] = plan.offsets[l][c];
                    }
                }
                assert_eq!(g.indices(&plan, &rows).unwrap().len(), 24);
                rows[0][g.layer_hash_index][23] = g.logical_table_rows;
                assert!(g.indices(&plan, &rows).is_err());
                rows[0][g.layer_hash_index][23] = g.logical_table_rows - 1;
                rows[0][g.layer_hash_index][0] = plan.offsets[g.layer_hash_index][1];
                assert!(g.indices(&plan, &rows).is_err());
            }
        }
        assert!(EngramLayout::official(&cfg, 0, 0).is_err());
        assert!(EngramLayout::official(&cfg, 8, 1).is_err());
        for (start, n) in [(0, 0), (0, 129), (1, 2), (1_048_576, 1), (usize::MAX, 1)] {
            assert!(step_end(start, n).is_err());
        }
        assert_eq!(step_end(0, 128).unwrap(), 128);
        assert_eq!(step_end(127, 1).unwrap(), 128);
    }
    #[test]
    fn packed_engram_progress_uses_physical_lanes_not_flattened_positions() {
        use super::super::batch_plan::{BatchCursor, LaneSegment, PackedKind};
        let boot = Uuid::from_u128(1701);
        let keys = [
            LaneKey {
                boot,
                slot: 9,
                generation: 7,
            },
            LaneKey {
                boot,
                slot: 1,
                generation: 4,
            },
        ];
        let mut cursor = BatchCursor::new(&config(), boot, 16).unwrap();
        let mut lanes = [None; 16];
        for key in keys {
            cursor.reset_lane(key).unwrap();
            lanes[key.slot] = Some(Progress {
                key,
                position: 0,
                initialized: false,
                failed: false,
            });
        }
        let ticket = cursor
            .begin(
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: keys[0],
                        start: 0,
                        tokens: 3,
                    },
                    LaneSegment {
                        key: keys[1],
                        start: 0,
                        tokens: 2,
                    },
                ],
            )
            .unwrap();
        let extents = packed_progress(boot, &lanes, &ticket).unwrap();
        assert_eq!(
            extents
                .iter()
                .map(|e| (e.key.slot, e.start, e.end))
                .collect::<Vec<_>>(),
            vec![(9, 0, 3), (1, 0, 2)]
        );
        // Validation never commits a CPU/semantic prefix, even on success.
        assert_eq!(lanes[9].unwrap().position, 0);
        assert_eq!(lanes[1].unwrap().position, 0);
        let mut wrong = lanes;
        wrong[1].as_mut().unwrap().position = 1;
        assert!(packed_progress(boot, &wrong, &ticket).is_err());
        assert_eq!(wrong[9].unwrap().position, 0);
        wrong = lanes;
        wrong[1].as_mut().unwrap().failed = true;
        assert!(packed_progress(boot, &wrong, &ticket).is_err());
        wrong = lanes;
        wrong[1].as_mut().unwrap().key.generation += 1;
        assert!(packed_progress(boot, &wrong, &ticket).is_err());
        wrong = lanes;
        wrong[1] = None;
        assert!(packed_progress(boot, &wrong, &ticket).is_err());
        assert!(packed_progress(Uuid::from_u128(1702), &lanes, &ticket).is_err());
    }
    #[test]
    fn packed_engram_all16_members_cover128_rows_without_history_crossing() {
        use super::super::batch_plan::{BatchCursor, LaneSegment, PackedKind};
        let boot = Uuid::from_u128(1703);
        let mut cursor = BatchCursor::new(&config(), boot, 16).unwrap();
        let mut lanes = [None; 16];
        let mut segments = Vec::new();
        for slot in (0..16).rev() {
            let key = LaneKey {
                boot,
                slot,
                generation: 1,
            };
            cursor.reset_lane(key).unwrap();
            lanes[slot] = Some(Progress {
                key,
                position: 0,
                initialized: false,
                failed: false,
            });
            segments.push(LaneSegment {
                key,
                start: 0,
                tokens: 8,
            });
        }
        let ticket = cursor.begin(PackedKind::PrefillSegments, segments).unwrap();
        let extents = packed_progress(boot, &lanes, &ticket).unwrap();
        assert_eq!(ticket.total_rows(), 128);
        assert_eq!(extents.len(), 16);
        assert!(extents.iter().all(|e| e.start == 0 && e.end == 8));
        assert_eq!(extents[0].key.slot, 15);
        lanes[0].as_mut().unwrap().position = 8;
        assert!(packed_progress(boot, &lanes, &ticket).is_err());
        assert!(lanes[1..].iter().all(|p| p.unwrap().position == 0));
    }
}
