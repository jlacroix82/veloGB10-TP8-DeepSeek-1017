//! CPU state and selection contracts for the exact text-only 1017 attention.
//! No KV payload, GPU kernel, physical prefix restore or model serving is enabled.
//! Each forward has fresh lane/generation/chunk-stamped shared handles. Persistent
//! KV/index cache extents belong to source layers; shared handles belong to one
//! forward and must never be reused for another request or decode position.

use super::config::Dsv41Config;
use super::scheduler::LaneKey;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

pub const MAX_LANES: usize = 16;
pub const MAX_PREFILL: usize = 128;
pub const MAX_CONTEXT: usize = 131_072;
pub const MODEL_REFERENCE_SHA256: &str =
    "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LayerSharing {
    pub layer: usize,
    pub ratio: usize,
    pub kv_source: Option<usize>,
    pub index_source: Option<usize>,
    pub owns_kv_and_index_keys: bool,
    pub owns_topk: bool,
    pub produces_candidates: bool,
    pub consumes_candidates: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SharingPlan {
    pub layers: Vec<LayerSharing>,
    pub window: usize,
    pub index_topk: usize,
    pub candidate_source: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub vocab: usize,
}

impl SharingPlan {
    pub fn from_config(config: &Dsv41Config) -> Result<Self> {
        config.validate()?;
        let mut layers = Vec::new();
        for layer in 0..config.n_layers {
            let ratio = config.compress_ratios[layer] as usize;
            let source = |sources: &[usize]| sources.iter().copied().filter(|&s| s <= layer).last();
            let kv_source = if ratio > 0 {
                Some(source(&config.kv_source_layers).context("compressed layer has no KV owner")?)
            } else {
                None
            };
            let index_source = if ratio > 0 {
                Some(
                    source(&config.index_source_layers)
                        .context("compressed layer has no top-k owner")?,
                )
            } else {
                None
            };
            if let Some(owner) = kv_source {
                ensure!(
                    config.compress_ratios[owner] as usize == ratio,
                    "shared KV compression ratio differs from consumer"
                );
            }
            layers.push(LayerSharing {
                layer,
                ratio,
                kv_source,
                index_source,
                owns_kv_and_index_keys: config.kv_source_layers.contains(&layer),
                owns_topk: config.index_source_layers.contains(&layer),
                produces_candidates: layer == config.candidate_source_layer,
                consumes_candidates: config.index_source_layers.contains(&layer)
                    && layer > config.candidate_source_layer,
            });
        }
        Ok(Self {
            layers,
            window: config.window_size,
            index_topk: config.index_topk,
            candidate_source: config.candidate_source_layer,
            candidate_topk_blocks: config.candidate_topk_blocks,
            candidate_block_size: config.candidate_block_size,
            vocab: config.vocab_size,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompressionProgress {
    pub ratio: usize,
    pub prior_rows: usize,
    pub final_rows: usize,
    pub append_rows: usize,
    pub partial_tokens_before: usize,
    pub partial_tokens_after: usize,
    /// Both index keys and compressed KV rotate at the FIRST token of each group.
    pub new_latent_rope_positions: Vec<usize>,
}

pub fn compression_progress(
    start: usize,
    tokens: usize,
    ratio: usize,
) -> Result<CompressionProgress> {
    ensure!(
        (1..=MAX_PREFILL).contains(&tokens) && matches!(ratio, 1 | 2),
        "unsupported 1017 compressor shape/ratio"
    );
    ensure!(
        start == 0 || tokens == 1,
        "official positive-start compressor supports one token only"
    );
    let end = start
        .checked_add(tokens)
        .context("attention position overflow")?;
    ensure!(end <= MAX_CONTEXT, "CPU reference context bound exceeded");
    let prior_rows = start / ratio;
    let final_rows = end / ratio;
    Ok(CompressionProgress {
        ratio,
        prior_rows,
        final_rows,
        append_rows: final_rows - prior_rows,
        partial_tokens_before: start % ratio,
        partial_tokens_after: end % ratio,
        new_latent_rope_positions: (prior_rows..final_rows).map(|row| row * ratio).collect(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WindowPlan {
    pub kv_rows: usize,
    pub query_indices: Vec<Vec<i32>>,
    /// (absolute native-token position, destination ring slot). Prefill attention
    /// still reads its full input chunk; only its final window seeds the ring.
    pub ring_writes: Vec<(usize, usize)>,
}

pub fn window_plan(window: usize, start: usize, tokens: usize) -> Result<WindowPlan> {
    ensure!(
        (1..=128).contains(&window) && (1..=MAX_PREFILL).contains(&tokens),
        "unsupported window/prefill bound"
    );
    ensure!(
        start == 0 || tokens == 1,
        "official positive-start window path supports one token only"
    );
    let end = start
        .checked_add(tokens)
        .context("window position overflow")?;
    ensure!(
        end <= MAX_CONTEXT,
        "window reference context bound exceeded"
    );
    let query_indices = if start == 0 {
        (0..tokens)
            .map(|q| {
                let first = (q + 1).saturating_sub(window);
                (0..tokens.min(window))
                    .map(|j| {
                        let index = first + j;
                        if index > q {
                            -1
                        } else {
                            index as i32
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        let oldest = start % window + 1;
        vec![(oldest..window)
            .chain(0..oldest)
            .map(|slot| if slot > start { -1 } else { slot as i32 })
            .collect()]
    };
    Ok(WindowPlan {
        kv_rows: if start == 0 { tokens } else { window },
        query_indices,
        ring_writes: (if start == 0 {
            end.saturating_sub(window)
        } else {
            start
        }..end)
            .map(|position| (position, position % window))
            .collect(),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundaryTie {
    pub row: usize,
    pub stage: String,
    /// IEEE bits also represent the legitimate -Inf masked-padding frontier
    /// without turning it into JSON null or treating it as invalid raw input.
    pub cutoff_score_bits: u32,
    pub required_from_ties: usize,
    pub strictly_better: Vec<usize>,
    pub tied_indices: Vec<usize>,
}

/// Torch topk leaves tied-index stability unspecified. Ambiguous is a valid
/// input outcome, not corruption. A production executor must supply a qualified
/// choice/witness or check its result against this acceptable-set contract.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Selection<T> {
    Exact(T),
    Ambiguous(Vec<BoundaryTie>),
}

fn choose_topk(
    values: &[f32],
    count: usize,
    row: usize,
    stage: &str,
    witness: Option<&[usize]>,
    observable: impl Fn(usize) -> bool,
) -> Result<Selection<Vec<usize>>> {
    ensure!(values.iter().all(|v| !v.is_nan()), "NaN score in top-k");
    let k = count.min(values.len());
    if let Some(choice) = witness {
        let unique: BTreeSet<_> = choice.iter().copied().collect();
        ensure!(
            choice.len() == k && unique.len() == k && unique.iter().all(|&i| i < values.len()),
            "invalid top-k witness geometry"
        );
        if k > 0 {
            let floor = choice
                .iter()
                .map(|&i| values[i])
                .fold(f32::INFINITY, f32::min);
            ensure!(
                values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !unique.contains(i))
                    .all(|(_, &v)| v <= floor),
                "top-k witness omitted a strictly higher score"
            );
        }
        return Ok(Selection::Exact(choice.to_vec()));
    }
    if k == 0 {
        return Ok(Selection::Exact(Vec::new()));
    }
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
    let cutoff = values[order[k - 1]];
    let above: Vec<_> = order
        .iter()
        .copied()
        .filter(|&i| values[i] > cutoff)
        .collect();
    let tied: Vec<_> = order
        .iter()
        .copied()
        .filter(|&i| values[i] == cutoff)
        .collect();
    let needed = k - above.len();
    if tied.len() > needed && tied.iter().any(|&i| observable(i)) {
        return Ok(Selection::Ambiguous(vec![BoundaryTie {
            row,
            stage: stage.into(),
            cutoff_score_bits: cutoff.to_bits(),
            required_from_ties: needed,
            strictly_better: above,
            tied_indices: tied,
        }]));
    }
    order.truncate(k);
    Ok(Selection::Exact(order))
}

fn score_shape(scores: &[Vec<f32>], visible: &[usize]) -> Result<usize> {
    ensure!(
        !scores.is_empty() && scores.len() <= MAX_PREFILL && scores.len() == visible.len(),
        "invalid query/lens count"
    );
    let width = scores[0].len();
    ensure!(
        width <= MAX_CONTEXT
            && scores
                .iter()
                .all(|r| r.len() == width && r.iter().all(|v| v.is_finite()))
            && visible.iter().all(|&n| n <= width),
        "invalid/nonfinite score matrix or visible length"
    );
    Ok(width)
}

/// Official select_candidate_blocks: causal masking, block maxima, newest
/// reachable block pinned to +Inf, top-k blocks, then expand/clip the bool mask.
pub fn candidate_blocks(
    scores: &[Vec<f32>],
    visible: &[usize],
    topk_blocks: usize,
    block_size: usize,
    witness: Option<&[Vec<usize>]>,
) -> Result<Selection<Vec<Vec<bool>>>> {
    let width = score_shape(scores, visible)?;
    ensure!(
        topk_blocks > 0 && block_size > 0 && block_size <= 128,
        "invalid candidate geometry"
    );
    ensure!(
        witness.is_none_or(|w| w.len() == scores.len()),
        "candidate witness query count mismatch"
    );
    let blocks = width.div_ceil(block_size);
    let mut output = Vec::new();
    let mut ambiguity = Vec::new();
    for (row, values) in scores.iter().enumerate() {
        let mut maxima = vec![f32::NEG_INFINITY; blocks];
        for position in 0..visible[row] {
            maxima[position / block_size] = maxima[position / block_size].max(values[position]);
        }
        if visible[row] > 0 {
            maxima[(visible[row] - 1) / block_size] = f32::INFINITY;
        }
        match choose_topk(
            &maxima,
            topk_blocks,
            row,
            "candidate_blocks",
            witness.map(|w| w[row].as_slice()),
            |i| maxima[i] > f32::NEG_INFINITY,
        )? {
            Selection::Ambiguous(ties) => ambiguity.extend(ties),
            Selection::Exact(selected) => {
                let mut keep = vec![false; blocks];
                for block in selected {
                    keep[block] = maxima[block] > f32::NEG_INFINITY;
                }
                output.push((0..width).map(|p| keep[p / block_size]).collect());
            }
        }
    }
    if ambiguity.is_empty() {
        Ok(Selection::Exact(output))
    } else {
        Ok(Selection::Ambiguous(ambiguity))
    }
}

/// Official final top-k: mask causal/future and optional candidate positions,
/// select by score, sort chosen positions, then shift reachable indices by the
/// actual window-KV row count. Unreachable picks become -1 after sorting.
pub fn topk_positions(
    scores: &[Vec<f32>],
    visible: &[usize],
    topk: usize,
    offset: usize,
    candidates: Option<&[Vec<bool>]>,
    witness: Option<&[Vec<usize>]>,
) -> Result<Selection<Vec<Vec<i32>>>> {
    let width = score_shape(scores, visible)?;
    ensure!(
        topk > 0
            && offset <= MAX_CONTEXT
            && offset.checked_add(width).context("top-k offset overflow")? <= i32::MAX as usize,
        "invalid top-k/offset"
    );
    ensure!(
        candidates.is_none_or(|c| c.len() == scores.len() && c.iter().all(|r| r.len() == width)),
        "candidate mask shape mismatch"
    );
    ensure!(
        witness.is_none_or(|w| w.len() == scores.len()),
        "position witness query count mismatch"
    );
    let mut output = Vec::new();
    let mut ambiguity = Vec::new();
    for (row, raw) in scores.iter().enumerate() {
        let masked: Vec<_> = raw
            .iter()
            .enumerate()
            .map(|(p, &score)| {
                if p < visible[row] && candidates.is_none_or(|c| c[row][p]) {
                    score
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        match choose_topk(
            &masked,
            topk,
            row,
            "positions",
            witness.map(|w| w[row].as_slice()),
            |p| p < visible[row],
        )? {
            Selection::Ambiguous(ties) => ambiguity.extend(ties),
            Selection::Exact(mut selected) => {
                selected.sort_unstable();
                output.push(
                    selected
                        .into_iter()
                        .map(|p| {
                            if p < visible[row] {
                                (p + offset) as i32
                            } else {
                                -1
                            }
                        })
                        .collect(),
                );
            }
        }
    }
    if ambiguity.is_empty() {
        Ok(Selection::Exact(output))
    } else {
        Ok(Selection::Ambiguous(ambiguity))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChunkIdentity {
    pub lane: LaneKey,
    pub prefix_generation: u64,
    pub chunk_id: u64,
    pub start_pos: usize,
    pub tokens: usize,
    pub prefix_ids_sha256_before_u32le: String,
    pub chunk_ids_sha256_u32le: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum ResourceKind {
    WindowKv,
    CompressedKv,
    IndexKeys,
    TopkIndices,
    Candidates,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceLease {
    chunk: ChunkIdentity,
    kind: ResourceKind,
    source_layer: usize,
    width: usize,
    queries: usize,
}
impl ResourceLease {
    pub fn source_layer(&self) -> usize {
        self.source_layer
    }
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn queries(&self) -> usize {
        self.queries
    }
    pub fn chunk(&self) -> &ChunkIdentity {
        &self.chunk
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompletedPrefix {
    pub lane: LaneKey,
    pub prefix_generation: u64,
    pub native_tokens: usize,
    pub native_ids_sha256_u32le: String,
}

struct SourceCache {
    rows: usize,
    partial: usize,
}
struct Forward {
    chunk: ChunkIdentity,
    ids: Vec<u32>,
    next_layer: usize,
    shared: BTreeMap<ResourceKind, ResourceLease>,
    selections: BTreeMap<ResourceKind, Vec<Vec<i32>>>,
    candidates: Option<Vec<Vec<bool>>>,
    pending_cache: BTreeMap<usize, SourceCache>,
}
struct Lane {
    key: LaneKey,
    prefix_generation: u64,
    position: usize,
    prefix_hasher: Sha256,
    last_chunk: Option<u64>,
    cache: BTreeMap<usize, SourceCache>,
    active: Option<Forward>,
}

/// CPU ownership/lifetime bookkeeping. Published leases describe validated
/// reference stages, not allocated KV buffers or completed GPU kernels.
pub struct AttentionState {
    boot: Uuid,
    plan: SharingPlan,
    lanes: BTreeMap<usize, Lane>,
    max_lanes: usize,
    context_bound: usize,
}

impl AttentionState {
    pub fn new(
        config: &Dsv41Config,
        boot: Uuid,
        max_lanes: usize,
        context_bound: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_LANES).contains(&max_lanes) && (1..=MAX_CONTEXT).contains(&context_bound),
            "attention state bound exceeded"
        );
        Ok(Self {
            boot,
            plan: SharingPlan::from_config(config)?,
            lanes: BTreeMap::new(),
            max_lanes,
            context_bound,
        })
    }
    pub fn plan(&self) -> &SharingPlan {
        &self.plan
    }
    pub fn reset_lane(&mut self, key: LaneKey, prefix_generation: u64) -> Result<()> {
        ensure!(
            key.boot == self.boot
                && key.slot < self.max_lanes
                && key.generation > 0
                && prefix_generation > 0,
            "invalid attention lane/generation"
        );
        ensure!(
            self.lanes
                .get(&key.slot)
                .is_none_or(|l| key.generation > l.key.generation),
            "lane generation must increase on reset"
        );
        let cache = self
            .plan
            .layers
            .iter()
            .filter(|l| l.owns_kv_and_index_keys)
            .map(|l| {
                (
                    l.layer,
                    SourceCache {
                        rows: 0,
                        partial: 0,
                    },
                )
            })
            .collect();
        self.lanes.insert(
            key.slot,
            Lane {
                key,
                prefix_generation,
                position: 0,
                prefix_hasher: Sha256::new(),
                last_chunk: None,
                cache,
                active: None,
            },
        );
        Ok(())
    }
    pub fn begin_chunk(
        &mut self,
        lane: LaneKey,
        prefix_generation: u64,
        chunk_id: u64,
        start: usize,
        ids: &[u32],
    ) -> Result<ChunkIdentity> {
        let state = self
            .lanes
            .get_mut(&lane.slot)
            .context("unbound attention lane")?;
        ensure!(
            state.key == lane
                && state.prefix_generation == prefix_generation
                && state.active.is_none(),
            "wrong lane/prefix generation or forward already active"
        );
        ensure!(
            start == state.position
                && (1..=MAX_PREFILL).contains(&ids.len())
                && (start == 0 || ids.len() == 1),
            "stale/noncontiguous/unsupported attention chunk"
        );
        ensure!(
            start
                .checked_add(ids.len())
                .context("chunk position overflow")?
                <= self.context_bound
                && ids.iter().all(|&id| (id as usize) < self.plan.vocab),
            "invalid IDs or context capacity"
        );
        ensure!(
            state.last_chunk.is_none_or(|prior| chunk_id > prior),
            "chunk identity must increase"
        );
        let mut hash = Sha256::new();
        for id in ids {
            hash.update(id.to_le_bytes());
        }
        let chunk = ChunkIdentity {
            lane,
            prefix_generation,
            chunk_id,
            start_pos: start,
            tokens: ids.len(),
            prefix_ids_sha256_before_u32le: format!("{:x}", state.prefix_hasher.clone().finalize()),
            chunk_ids_sha256_u32le: format!("{:x}", hash.finalize()),
        };
        state.active = Some(Forward {
            chunk: chunk.clone(),
            ids: ids.to_vec(),
            next_layer: 0,
            shared: BTreeMap::new(),
            selections: BTreeMap::new(),
            candidates: None,
            pending_cache: BTreeMap::new(),
        });
        Ok(chunk)
    }
    fn forward(&self, chunk: &ChunkIdentity, layer: usize) -> Result<&Forward> {
        let lane = self.lanes.get(&chunk.lane.slot).context("unbound lane")?;
        ensure!(
            lane.key == chunk.lane && lane.prefix_generation == chunk.prefix_generation,
            "stale lane/prefix generation"
        );
        let active = lane.active.as_ref().context("no active attention chunk")?;
        ensure!(
            active.chunk == *chunk && active.next_layer == layer && layer < self.plan.layers.len(),
            "wrong chunk or out-of-order attention layer"
        );
        Ok(active)
    }
    fn forward_mut(&mut self, chunk: &ChunkIdentity, layer: usize) -> Result<&mut Forward> {
        self.forward(chunk, layer)?;
        Ok(self
            .lanes
            .get_mut(&chunk.lane.slot)
            .unwrap()
            .active
            .as_mut()
            .unwrap())
    }
    fn expected(
        &self,
        chunk: &ChunkIdentity,
        layer: usize,
        kind: ResourceKind,
    ) -> Result<ResourceLease> {
        self.forward(chunk, layer)?;
        let sharing = &self.plan.layers[layer];
        let compressed = if sharing.ratio > 0 {
            (chunk.start_pos + chunk.tokens) / sharing.ratio
        } else {
            0
        };
        let (source, width) = match kind {
            ResourceKind::WindowKv => (
                layer,
                if chunk.start_pos == 0 {
                    chunk.tokens
                } else {
                    self.plan.window
                },
            ),
            ResourceKind::CompressedKv | ResourceKind::IndexKeys => (
                sharing
                    .kv_source
                    .context("layer does not use compressed state")?,
                compressed,
            ),
            ResourceKind::TopkIndices => (
                sharing
                    .index_source
                    .context("layer does not use top-k state")?,
                compressed.min(self.plan.index_topk),
            ),
            ResourceKind::Candidates => {
                ensure!(
                    sharing.produces_candidates || sharing.consumes_candidates,
                    "layer does not use candidate masks"
                );
                (self.plan.candidate_source, compressed)
            }
        };
        Ok(ResourceLease {
            chunk: chunk.clone(),
            kind,
            source_layer: source,
            width,
            queries: chunk.tokens,
        })
    }
    pub fn validate_lease(
        &self,
        chunk: &ChunkIdentity,
        layer: usize,
        lease: &ResourceLease,
    ) -> Result<()> {
        let expected = self.expected(chunk, layer, lease.kind)?;
        let active = self.forward(chunk, layer)?;
        ensure!(
            *lease == expected && active.shared.get(&lease.kind) == Some(lease),
            "stale/missing/wrong-source shared attention lease"
        );
        Ok(())
    }
    pub fn read(
        &self,
        chunk: &ChunkIdentity,
        layer: usize,
        kind: ResourceKind,
    ) -> Result<ResourceLease> {
        let lease = self
            .forward(chunk, layer)?
            .shared
            .get(&kind)
            .cloned()
            .context("required shared attention resource is missing")?;
        self.validate_lease(chunk, layer, &lease)?;
        Ok(lease)
    }
    fn publish(
        &mut self,
        chunk: &ChunkIdentity,
        layer: usize,
        kind: ResourceKind,
    ) -> Result<ResourceLease> {
        let lease = self.expected(chunk, layer, kind)?;
        ensure!(
            lease.source_layer == layer,
            "only the configured source may publish this resource"
        );
        let active = self.forward_mut(chunk, layer)?;
        ensure!(
            active.shared.get(&kind) != Some(&lease),
            "resource published twice in one layer/chunk"
        );
        active.shared.insert(kind, lease.clone());
        Ok(lease)
    }
    pub fn publish_window(
        &mut self,
        chunk: &ChunkIdentity,
        layer: usize,
    ) -> Result<(ResourceLease, WindowPlan)> {
        let plan = window_plan(self.plan.window, chunk.start_pos, chunk.tokens)?;
        let lease = self.publish(chunk, layer, ResourceKind::WindowKv)?;
        Ok((lease, plan))
    }
    pub fn publish_index_keys(
        &mut self,
        chunk: &ChunkIdentity,
        layer: usize,
    ) -> Result<(ResourceLease, CompressionProgress)> {
        self.read(chunk, layer, ResourceKind::WindowKv)?;
        ensure!(
            self.plan.layers[layer].owns_kv_and_index_keys,
            "only KV sources own index keys"
        );
        let progress =
            compression_progress(chunk.start_pos, chunk.tokens, self.plan.layers[layer].ratio)?;
        let persistent = self.lanes[&chunk.lane.slot]
            .cache
            .get(&layer)
            .context("missing persistent source cache")?;
        ensure!(
            persistent.rows == progress.prior_rows
                && persistent.partial == progress.partial_tokens_before,
            "stale compressor/index cache prefix"
        );
        // When final_rows==0 the official indexer skips its kernel entirely.
        // This is an empty logical lease, not a claim of a physical key write.
        let lease = self.publish(chunk, layer, ResourceKind::IndexKeys)?;
        Ok((lease, progress))
    }
    /// Scores are the already globally reduced finite index scores. This method
    /// implements selection, not FP4 query/key arithmetic or TP score reduction.
    /// Optional witnesses resolve legitimate Torch top-k ties by acceptable-set
    /// validation; they cannot omit a strictly better position/block.
    pub fn publish_scores(
        &mut self,
        chunk: &ChunkIdentity,
        layer: usize,
        scores: &[Vec<f32>],
        candidate_witness: Option<&[Vec<usize>]>,
        position_witness: Option<&[Vec<usize>]>,
    ) -> Result<Selection<(ResourceLease, Option<ResourceLease>)>> {
        self.read(chunk, layer, ResourceKind::WindowKv)?;
        let sharing = self.plan.layers[layer].clone();
        ensure!(sharing.owns_topk, "only index sources publish top-k");
        let keys = self.read(chunk, layer, ResourceKind::IndexKeys)?;
        let visible: Vec<_> = (0..chunk.tokens)
            .map(|q| (chunk.start_pos + q + 1) / sharing.ratio)
            .collect();
        ensure!(
            scores.len() == chunk.tokens && score_shape(scores, &visible)? == keys.width,
            "scores differ from source cache/query geometry"
        );
        let candidate_mask = if sharing.produces_candidates {
            match candidate_blocks(
                scores,
                &visible,
                self.plan.candidate_topk_blocks,
                self.plan.candidate_block_size,
                candidate_witness,
            )? {
                Selection::Exact(mask) => Some(mask),
                Selection::Ambiguous(tie) => return Ok(Selection::Ambiguous(tie)),
            }
        } else if sharing.consumes_candidates {
            self.read(chunk, layer, ResourceKind::Candidates)?;
            Some(
                self.forward(chunk, layer)?
                    .candidates
                    .clone()
                    .context("candidate lease has no mask")?,
            )
        } else {
            None
        };
        let offset = if chunk.start_pos == 0 {
            chunk.tokens
        } else {
            self.plan.window
        };
        // A candidate source computes its own top-k WITHOUT its newly made
        // mask. Only later index-source layers consume that first-level mask.
        let filter = if sharing.consumes_candidates {
            candidate_mask.as_deref()
        } else {
            None
        };
        let selected = match topk_positions(
            scores,
            &visible,
            self.plan.index_topk,
            offset,
            filter,
            position_witness,
        )? {
            Selection::Exact(selected) => selected,
            Selection::Ambiguous(tie) => return Ok(Selection::Ambiguous(tie)),
        };
        let candidate_lease = if sharing.produces_candidates {
            let lease = self.publish(chunk, layer, ResourceKind::Candidates)?;
            self.forward_mut(chunk, layer)?.candidates = candidate_mask;
            Some(lease)
        } else {
            None
        };
        let topk = self.publish(chunk, layer, ResourceKind::TopkIndices)?;
        self.forward_mut(chunk, layer)?
            .selections
            .insert(ResourceKind::TopkIndices, selected);
        Ok(Selection::Exact((topk, candidate_lease)))
    }
    pub fn topk(&self, chunk: &ChunkIdentity, layer: usize) -> Result<&[Vec<i32>]> {
        self.read(chunk, layer, ResourceKind::TopkIndices)?;
        Ok(self
            .forward(chunk, layer)?
            .selections
            .get(&ResourceKind::TopkIndices)
            .context("top-k payload missing")?)
    }
    pub fn publish_compressed_kv(
        &mut self,
        chunk: &ChunkIdentity,
        layer: usize,
    ) -> Result<ResourceLease> {
        ensure!(
            self.plan
                .layers
                .get(layer)
                .is_some_and(|l| l.owns_kv_and_index_keys),
            "only KV sources publish compressed KV"
        );
        self.read(chunk, layer, ResourceKind::IndexKeys)?;
        self.read(chunk, layer, ResourceKind::TopkIndices)?;
        let progress =
            compression_progress(chunk.start_pos, chunk.tokens, self.plan.layers[layer].ratio)?;
        let lease = self.publish(chunk, layer, ResourceKind::CompressedKv)?;
        self.forward_mut(chunk, layer)?.pending_cache.insert(
            layer,
            SourceCache {
                rows: progress.final_rows,
                partial: progress.partial_tokens_after,
            },
        );
        Ok(lease)
    }
    pub fn complete_layer(&mut self, chunk: &ChunkIdentity, layer: usize) -> Result<()> {
        self.read(chunk, layer, ResourceKind::WindowKv)?;
        if self.plan.layers[layer].ratio > 0 {
            self.read(chunk, layer, ResourceKind::CompressedKv)?;
            self.read(chunk, layer, ResourceKind::TopkIndices)?;
        }
        self.forward_mut(chunk, layer)?.next_layer += 1;
        Ok(())
    }
    pub fn finish_chunk(&mut self, chunk: &ChunkIdentity) -> Result<CompletedPrefix> {
        let lane = self
            .lanes
            .get_mut(&chunk.lane.slot)
            .context("unbound attention lane")?;
        let active = lane.active.as_ref().context("no active forward")?;
        ensure!(
            lane.key == chunk.lane
                && lane.prefix_generation == chunk.prefix_generation
                && active.chunk == *chunk
                && active.next_layer == self.plan.layers.len()
                && active.pending_cache.len() == lane.cache.len(),
            "incomplete/stale attention forward"
        );
        let active = lane.active.take().unwrap();
        for id in active.ids {
            lane.prefix_hasher.update(id.to_le_bytes());
        }
        lane.position = chunk.start_pos + chunk.tokens;
        lane.last_chunk = Some(chunk.chunk_id);
        lane.cache = active.pending_cache;
        Ok(CompletedPrefix {
            lane: lane.key,
            prefix_generation: lane.prefix_generation,
            native_tokens: lane.position,
            native_ids_sha256_u32le: format!("{:x}", lane.prefix_hasher.clone().finalize()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Dsv41Config {
        let value: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        Dsv41Config::from_values(&value["hf_config"], &value["inference_config"]).unwrap()
    }
    fn exact<T>(result: Selection<T>) -> T {
        match result {
            Selection::Exact(v) => v,
            Selection::Ambiguous(_) => panic!("unexpected ambiguous fixture"),
        }
    }
    fn finish_layers(state: &mut AttentionState, chunk: &ChunkIdentity) -> Vec<ResourceLease> {
        let mut leases = Vec::new();
        for layer in 0..40 {
            state.publish_window(chunk, layer).unwrap();
            let layout = state.plan.layers[layer].clone();
            if layout.owns_kv_and_index_keys {
                leases.push(state.publish_index_keys(chunk, layer).unwrap().0);
            }
            if layout.owns_topk {
                let width = (chunk.start_pos + chunk.tokens) / layout.ratio;
                let scores: Vec<Vec<f32>> = (0..chunk.tokens)
                    .map(|q| (0..width).map(|p| p as f32 + q as f32 / 1024.0).collect())
                    .collect();
                exact(
                    state
                        .publish_scores(chunk, layer, &scores, None, None)
                        .unwrap(),
                );
            }
            if layout.owns_kv_and_index_keys {
                leases.push(state.publish_compressed_kv(chunk, layer).unwrap());
            }
            state.complete_layer(chunk, layer).unwrap();
        }
        leases
    }
    #[test]
    fn actual_1017_owners_ratios_and_compressor_boundaries() {
        let p = SharingPlan::from_config(&config()).unwrap();
        assert_eq!(
            p.layers
                .iter()
                .filter(|l| l.owns_kv_and_index_keys)
                .map(|l| l.layer)
                .collect::<Vec<_>>(),
            [2, 8, 14, 20]
        );
        assert_eq!(
            p.layers
                .iter()
                .filter(|l| l.owns_topk)
                .map(|l| l.layer)
                .collect::<Vec<_>>(),
            [2, 8, 14, 20, 24, 28, 32, 36]
        );
        assert_eq!(p.layers[19].ratio, 2);
        assert_eq!(p.layers[20].ratio, 1);
        assert_eq!(p.layers[39].kv_source, Some(20));
        assert_eq!(p.layers[35].index_source, Some(32));
        assert!(!p.layers[23].consumes_candidates);
        assert!(p.layers[24].consumes_candidates);
        assert_eq!(
            compression_progress(0, 3, 2)
                .unwrap()
                .new_latent_rope_positions,
            [0]
        );
        let next = compression_progress(3, 1, 2).unwrap();
        assert_eq!(next.new_latent_rope_positions, [2]);
        assert_eq!(next.partial_tokens_before, 1);
        assert_eq!(next.partial_tokens_after, 0);
        assert!(compression_progress(3, 2, 2).is_err());
        assert!(compression_progress(0, 8, 4).is_err());
        assert!(window_plan(128, 1, 2).is_err());
    }
    #[test]
    fn source_lifetimes_reject_wrong_forward_and_keep_lanes_isolated() {
        let boot = Uuid::new_v4();
        let mut state = AttentionState::new(&config(), boot, 2, 256).unwrap();
        let a = LaneKey {
            boot,
            slot: 0,
            generation: 1,
        };
        let b = LaneKey {
            boot,
            slot: 1,
            generation: 1,
        };
        state.reset_lane(a, 1).unwrap();
        state.reset_lane(b, 1).unwrap();
        let first = state.begin_chunk(a, 1, 1, 0, &[10, 11, 12]).unwrap();
        assert!(state.read(&first, 0, ResourceKind::WindowKv).is_err());
        assert!(state.publish_index_keys(&first, 2).is_err());
        assert!(state.finish_chunk(&first).is_err());
        let leases = finish_layers(&mut state, &first);
        let prefix = state.finish_chunk(&first).unwrap();
        assert_eq!(prefix.native_tokens, 3);
        let second = state.begin_chunk(a, 1, 2, 3, &[13]).unwrap();
        assert!(state.validate_lease(&second, 0, &leases[0]).is_err());
        let other = state.begin_chunk(b, 1, 1, 0, &[99]).unwrap();
        assert_ne!(second.chunk_ids_sha256_u32le, other.chunk_ids_sha256_u32le);
        finish_layers(&mut state, &other);
        state.finish_chunk(&other).unwrap();
        finish_layers(&mut state, &second);
        let result = state.finish_chunk(&second).unwrap();
        let mut hash = Sha256::new();
        for id in [10u32, 11, 12, 13] {
            hash.update(id.to_le_bytes());
        }
        assert_eq!(
            result.native_ids_sha256_u32le,
            format!("{:x}", hash.finalize())
        );
        assert!(state.begin_chunk(a, 1, 3, 2, &[14]).is_err());
        assert!(state.begin_chunk(a, 1, 3, 4, &[14, 15]).is_err());
        let replacement = LaneKey { generation: 2, ..a };
        state.reset_lane(replacement, 2).unwrap();
        assert!(state.begin_chunk(a, 1, 3, 4, &[14]).is_err());
        assert!(state.reset_lane(a, 1).is_err());
        let cold = state.begin_chunk(replacement, 2, 0, 0, &[42]).unwrap();
        finish_layers(&mut state, &cold);
        assert_eq!(state.finish_chunk(&cold).unwrap().native_tokens, 1);
    }
    #[test]
    fn tied_scores_are_valid_ambiguity_and_witnesses_must_be_acceptable() {
        let scores = vec![vec![0.0; 24]];
        let visible = [24];
        assert!(matches!(
            candidate_blocks(&scores, &visible, 2, 8, None).unwrap(),
            Selection::Ambiguous(_)
        ));
        let keep = exact(candidate_blocks(&scores, &visible, 2, 8, Some(&[vec![0, 2]])).unwrap());
        assert!(keep[0][0] && keep[0][16] && !keep[0][8]);
        assert!(candidate_blocks(&scores, &visible, 2, 8, Some(&[vec![0, 1]])).is_err());
        assert!(matches!(
            topk_positions(&scores, &visible, 2, 128, None, None).unwrap(),
            Selection::Ambiguous(_)
        ));
        assert_eq!(
            exact(topk_positions(&scores, &visible, 2, 128, None, Some(&[vec![5, 3]])).unwrap()),
            [vec![131, 133]]
        );
        assert!(topk_positions(&[vec![1.0, 2.0]], &[2], 1, 128, None, Some(&[vec![0]])).is_err());
        assert!(topk_positions(&[vec![f32::NAN]], &[1], 1, 128, None, None).is_err());
    }

    #[test]
    fn current_forward_cannot_read_superseded_owner_or_skip_required_stage() {
        let boot = Uuid::new_v4();
        let mut state = AttentionState::new(&config(), boot, 1, 128).unwrap();
        let lane = LaneKey {
            boot,
            slot: 0,
            generation: 1,
        };
        state.reset_lane(lane, 7).unwrap();
        let chunk = state.begin_chunk(lane, 7, 10, 0, &[10, 11]).unwrap();
        let mut owner_two = None;
        for layer in 0..8 {
            state.publish_window(&chunk, layer).unwrap();
            if layer == 2 {
                assert!(state.publish_compressed_kv(&chunk, layer).is_err());
                let lease = state.publish_index_keys(&chunk, layer).unwrap().0;
                owner_two = Some(lease);
                assert!(state.publish_compressed_kv(&chunk, layer).is_err());
                exact(
                    state
                        .publish_scores(&chunk, layer, &[vec![1.0], vec![2.0]], None, None)
                        .unwrap(),
                );
                state.publish_compressed_kv(&chunk, layer).unwrap();
            }
            state.complete_layer(&chunk, layer).unwrap();
        }
        state.publish_window(&chunk, 8).unwrap();
        let old = owner_two.unwrap();
        assert!(state.validate_lease(&chunk, 8, &old).is_err());
        assert!(state.read(&chunk, 8, ResourceKind::IndexKeys).is_err());
        assert!(state
            .publish_scores(&chunk, 8, &[vec![1.0], vec![2.0]], None, None)
            .is_err());
        let fresh = state.publish_index_keys(&chunk, 8).unwrap().0;
        assert_eq!(fresh.source_layer(), 8);
        assert_eq!(fresh.queries(), 2);
        state.validate_lease(&chunk, 8, &fresh).unwrap();
        assert!(state.validate_lease(&chunk, 8, &old).is_err());
        assert!(state.complete_layer(&chunk, 8).is_err());
        assert!(state.begin_chunk(lane, 7, 11, 0, &[99]).is_err());
        assert!(AttentionState::new(&config(), boot, 17, 128).is_err());
        assert!(AttentionState::new(&config(), boot, 1, MAX_CONTEXT + 1).is_err());
    }
    #[test]
    fn selections_and_window_indices_match_official_torch_cpu_goldens() {
        let raw = include_str!("../../tests/fixtures/dsv41_attention_state_vectors.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw.as_bytes())),
            "dd8992d793f3f464c0d294411bc7fe7e4c44925b0aa8cae43b748225ef54a0b9"
        );
        let fixture: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(fixture["model_source_sha256"], MODEL_REFERENCE_SHA256);
        let windows = fixture["window_cases"].as_array().unwrap();
        assert_eq!(windows.len(), 10);
        for case in windows {
            let result = window_plan(
                case["window"].as_u64().unwrap() as usize,
                case["start"].as_u64().unwrap() as usize,
                case["tokens"].as_u64().unwrap() as usize,
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(result.query_indices).unwrap(),
                case["indices"]
            );
        }
        let cases = fixture["selection_cases"].as_array().unwrap();
        assert_eq!(cases.len(), 7);
        for case in cases {
            let scores: Vec<Vec<f32>> = serde_json::from_value(case["scores"].clone()).unwrap();
            let visible: Vec<usize> = serde_json::from_value(case["visible"].clone()).unwrap();
            let cw: Option<Vec<Vec<usize>>> = case
                .get("candidate_witness")
                .map(|v| serde_json::from_value(v.clone()).unwrap());
            let pw: Option<Vec<Vec<usize>>> = case
                .get("position_witness")
                .map(|v| serde_json::from_value(v.clone()).unwrap());
            let mask = if let Some(expected) = case.get("candidates") {
                let m = exact(
                    candidate_blocks(
                        &scores,
                        &visible,
                        case["topk_blocks"].as_u64().unwrap() as usize,
                        case["block_size"].as_u64().unwrap() as usize,
                        cw.as_deref(),
                    )
                    .unwrap(),
                );
                assert_eq!(serde_json::to_value(&m).unwrap(), *expected);
                Some(m)
            } else {
                None
            };
            let filter = if case["apply_candidates"] == true {
                mask.as_deref()
            } else {
                None
            };
            let selected = exact(
                topk_positions(
                    &scores,
                    &visible,
                    case["topk"].as_u64().unwrap() as usize,
                    case["offset"].as_u64().unwrap() as usize,
                    filter,
                    pw.as_deref(),
                )
                .unwrap(),
            );
            assert_eq!(serde_json::to_value(selected).unwrap(), case["indices"]);
        }
    }
}
