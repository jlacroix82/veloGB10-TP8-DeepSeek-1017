//! CPU-only, additive1017 chunk contract. This is not a guard relaxation of the
//! qualified initial-prefill/one-token path, nor numerical/model qualification.
//! A later runtime must snapshot prior window rows BEFORE committing new ring
//! writes, preserve F32 compressor tails, and publish checked sources in order.
use super::{attention_state, config::Dsv41Config, scheduler::LaneKey};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use uuid::Uuid;

pub const MAX_CONTEXT: usize = 8192;
pub const MAX_CHUNK: usize = 128;
pub const WINDOW: usize = 128;
pub const LAYERS: usize = 40;
pub const KV_SOURCES: [usize; 4] = [2, 8, 14, 20];
pub const INDEX_SOURCES: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub const CANDIDATE_SOURCE: usize = 20;

pub fn validate_config(c: &Dsv41Config) -> Result<()> {
    c.validate()?;
    ensure!(
        c.n_layers == LAYERS
            && c.window_size == WINDOW
            && c.kv_source_layers == KV_SOURCES
            && c.index_source_layers == INDEX_SOURCES
            && c.candidate_source_layer == CANDIDATE_SOURCE
            && c.index_topk == 512
            && c.candidate_topk_blocks == 2048
            && c.candidate_block_size == 8,
        "chunk contract requires exact1017 sharing/index/window configuration"
    );
    Ok(())
}
fn ratio(layer: usize) -> Result<usize> {
    ensure!(layer < LAYERS, "invalid backbone layer");
    Ok(if layer < 2 {
        0
    } else if layer < 20 {
        2
    } else {
        1
    })
}
fn source(layer: usize, sources: &[usize]) -> Option<usize> {
    sources.iter().copied().filter(|&p| p <= layer).last()
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WindowChunk {
    pub start: usize,
    pub tokens: usize,
    pub snapshot_global_base: usize,
    pub previous_rows: usize,
    pub snapshot_rows: usize,
    pub query_capacity: usize,
    /// Global position -> existing physical ring slot. Copy these before writes.
    pub snapshot_prior_reads: Vec<(usize, usize)>,
    /// Rows are chronological snapshot offsets, with unreachable-1 padding.
    /// Only reachable-key order is asserted source-equivalent. Placement of
    /// unused placeholders/kernel shape is NOT a bitwise arithmetic claim.
    pub query_indices: Vec<Vec<i32>>,
    pub ring_writes_after_snapshot: Vec<(usize, usize)>,
}
pub fn window_chunk(start: usize, tokens: usize) -> Result<WindowChunk> {
    let end = extent(start, tokens)?;
    let previous_rows = start.min(WINDOW - 1);
    let base = start - previous_rows;
    let capacity = if start == 0 {
        tokens.min(WINDOW)
    } else {
        WINDOW
    };
    let mut query_indices = Vec::with_capacity(tokens);
    for q in 0..tokens {
        let p = start + q;
        let first = (p + 1).saturating_sub(WINDOW);
        ensure!(first >= base, "snapshot omitted a prior reachable key");
        let mut row = (first..=p)
            .map(|global| (global - base) as i32)
            .collect::<Vec<_>>();
        ensure!(row.len() <= capacity, "window query capacity");
        row.resize(capacity, -1);
        query_indices.push(row);
    }
    Ok(WindowChunk {
        start,
        tokens,
        snapshot_global_base: base,
        previous_rows,
        snapshot_rows: previous_rows + tokens,
        query_capacity: capacity,
        snapshot_prior_reads: (base..start).map(|p| (p, p % WINDOW)).collect(),
        query_indices,
        ring_writes_after_snapshot: (start..end).map(|p| (p, p % WINDOW)).collect(),
    })
}
fn extent(start: usize, tokens: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_CHUNK).contains(&tokens),
        "chunk must contain1..128 native tokens"
    );
    let end = start
        .checked_add(tokens)
        .context("chunk position overflow")?;
    ensure!(
        end <= MAX_CONTEXT,
        "chunk exceeds explicit8192 native context"
    );
    Ok(end)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum PoolInput {
    /// Only the earlier half of one crossing ratio2 group can predate the chunk.
    ExistingF32Tail {
        global_position: usize,
    },
    CurrentProjection {
        row: usize,
        global_position: usize,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompletedGroup {
    pub cache_row: usize,
    pub rope_position: usize,
    pub inputs: Vec<PoolInput>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompressedChunk {
    pub ratio: usize,
    pub stored_rows: usize,
    pub new_rows: usize,
    pub required_old_tail: Option<usize>,
    pub retained_new_tail: Option<usize>,
    pub groups: Vec<CompletedGroup>,
    pub visible_rows_per_query: Vec<usize>,
}
pub fn compressed_chunk(start: usize, tokens: usize, ratio: usize) -> Result<CompressedChunk> {
    let end = extent(start, tokens)?;
    ensure!(matches!(ratio, 1 | 2), "only1017 ratios1/2");
    let groups = (start / ratio..end / ratio)
        .map(|j| CompletedGroup {
            cache_row: j,
            rope_position: j * ratio,
            inputs: (j * ratio..(j + 1) * ratio)
                .map(|p| {
                    if p < start {
                        PoolInput::ExistingF32Tail { global_position: p }
                    } else {
                        PoolInput::CurrentProjection {
                            row: p - start,
                            global_position: p,
                        }
                    }
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    ensure!(
        groups.iter().flat_map(|g| &g.inputs).all(|i| match i {
            PoolInput::ExistingF32Tail { global_position: p } =>
                ratio == 2 && start % 2 == 1 && *p == start - 1,
            PoolInput::CurrentProjection {
                row,
                global_position: p,
            } => *row < tokens && *p == start + row,
        }),
        "compression crossing group is not the single prior F32tail"
    );
    Ok(CompressedChunk {
        ratio,
        stored_rows: end / ratio,
        new_rows: end / ratio - start / ratio,
        required_old_tail: (ratio == 2 && start % 2 == 1).then_some(start.saturating_sub(1)),
        retained_new_tail: (ratio == 2 && end % 2 == 1).then_some(end - 1),
        groups,
        visible_rows_per_query: (0..tokens).map(|q| (start + q + 1) / ratio).collect(),
    })
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChunkTicket {
    pub lane: LaneKey,
    /// Globally monotonic within a cursor, including failed attempts and reset.
    pub version: u64,
    pub start: usize,
    pub tokens: usize,
    pub end: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LayerChunk {
    pub ticket: ChunkTicket,
    pub layer: usize,
    pub ratio: usize,
    pub kv_source: Option<usize>,
    pub index_source: Option<usize>,
    pub candidate_source: Option<usize>,
    pub owns_kv_and_index_keys: bool,
    pub publishes_index_selection: bool,
    pub publishes_candidates: bool,
    /// Key-cache append still executes below512, even when score selection is
    /// exactly all reachable. The old short-only state cannot be reinterpreted.
    pub index_scoring_required: bool,
    pub full_candidate_selection_exact: bool,
    pub window: WindowChunk,
    pub compressed: Option<CompressedChunk>,
    pub attention_index_capacity: usize,
    pub numerical_or_model_equivalence_qualified: bool,
}
#[derive(Clone)]
struct Transaction {
    ticket: ChunkTicket,
    next_layer: usize,
    kv: [bool; LAYERS],
    index: [bool; LAYERS],
    candidates: bool,
}
struct Progress {
    lane: LaneKey,
    committed: usize,
    failed: bool,
    open: Option<Transaction>,
}
pub struct ChunkCursor {
    boot: Uuid,
    max_lanes: usize,
    next_version: u64,
    lanes: BTreeMap<usize, Progress>,
}
impl ChunkCursor {
    pub fn new(c: &Dsv41Config, boot: Uuid, max_lanes: usize) -> Result<Self> {
        validate_config(c)?;
        ensure!((1..=16).contains(&max_lanes), "chunk lane bound1..16");
        Ok(Self {
            boot,
            max_lanes,
            next_version: 1,
            lanes: BTreeMap::new(),
        })
    }
    pub fn reset(&mut self, lane: LaneKey) -> Result<()> {
        ensure!(
            lane.boot == self.boot && lane.slot < self.max_lanes && lane.generation > 0,
            "chunk lane identity"
        );
        ensure!(
            self.lanes
                .get(&lane.slot)
                .is_none_or(|p| lane.generation > p.lane.generation),
            "chunk reset generation must advance"
        );
        self.lanes.insert(
            lane.slot,
            Progress {
                lane,
                committed: 0,
                failed: false,
                open: None,
            },
        );
        Ok(())
    }
    pub fn begin(&mut self, lane: LaneKey, start: usize, tokens: usize) -> Result<ChunkTicket> {
        let end = extent(start, tokens)?;
        let p = self
            .lanes
            .get_mut(&lane.slot)
            .context("unbound chunk lane")?;
        ensure!(
            p.lane == lane && !p.failed && p.committed == start && p.open.is_none(),
            "stale/failed/noncontiguous/already-open chunk"
        );
        let version = self.next_version;
        self.next_version = version.checked_add(1).context("chunk version exhausted")?;
        let ticket = ChunkTicket {
            lane,
            version,
            start,
            tokens,
            end,
        };
        p.open = Some(Transaction {
            ticket: ticket.clone(),
            next_layer: 0,
            kv: [false; LAYERS],
            index: [false; LAYERS],
            candidates: false,
        });
        Ok(ticket)
    }
    fn transaction(&self, ticket: &ChunkTicket) -> Result<&Transaction> {
        let p = self
            .lanes
            .get(&ticket.lane.slot)
            .context("unbound chunk lane")?;
        ensure!(
            p.lane == ticket.lane && !p.failed,
            "stale/failed chunk generation"
        );
        let tx = p.open.as_ref().context("chunk is not open")?;
        ensure!(tx.ticket == *ticket, "forged/stale chunk ticket");
        Ok(tx)
    }
    pub fn layer(&self, ticket: &ChunkTicket, layer: usize) -> Result<LayerChunk> {
        let tx = self.transaction(ticket)?;
        ensure!(
            layer == tx.next_layer && layer < LAYERS,
            "layers must complete in source backbone order"
        );
        let ratio = ratio(layer)?;
        let kv = if ratio > 0 {
            source(layer, &KV_SOURCES)
        } else {
            None
        };
        let index = if ratio > 0 {
            source(layer, &INDEX_SOURCES)
        } else {
            None
        };
        if let Some(owner) = kv {
            ensure!(
                owner == layer || tx.kv[owner],
                "KV/index-key source must publish this exact chunk first"
            );
        }
        if let Some(owner) = index {
            ensure!(
                owner == layer || tx.index[owner],
                "index-selection source must publish this exact chunk first"
            );
        }
        if layer > CANDIDATE_SOURCE {
            ensure!(
                tx.candidates,
                "candidate source must publish this exact chunk first"
            );
        }
        let compressed = (ratio > 0)
            .then(|| compressed_chunk(ticket.start, ticket.tokens, ratio))
            .transpose()?;
        let rows = compressed.as_ref().map_or(0, |c| c.stored_rows);
        let window = window_chunk(ticket.start, ticket.tokens)?;
        let capacity = window.query_capacity + rows.min(512);
        Ok(LayerChunk {
            ticket: ticket.clone(),
            layer,
            ratio,
            kv_source: kv,
            index_source: index,
            candidate_source: (layer > CANDIDATE_SOURCE).then_some(CANDIDATE_SOURCE),
            owns_kv_and_index_keys: kv == Some(layer),
            publishes_index_selection: index == Some(layer),
            publishes_candidates: layer == CANDIDATE_SOURCE,
            index_scoring_required: index == Some(layer) && rows > 512,
            full_candidate_selection_exact: rows.div_ceil(8) <= 2048,
            window,
            compressed,
            attention_index_capacity: capacity,
            numerical_or_model_equivalence_qualified: false,
        })
    }
    /// Commit only after all device statuses/collectives for this layer passed.
    /// A runtime failure after any write must abort the chunk and reset generation.
    pub fn commit_layer(&mut self, plan: &LayerChunk) -> Result<()> {
        ensure!(
            self.layer(&plan.ticket, plan.layer)? == *plan,
            "forged/stale layer chunk plan"
        );
        let tx = self
            .lanes
            .get_mut(&plan.ticket.lane.slot)
            .unwrap()
            .open
            .as_mut()
            .unwrap();
        if plan.owns_kv_and_index_keys {
            tx.kv[plan.layer] = true;
        }
        if plan.publishes_index_selection {
            tx.index[plan.layer] = true;
        }
        if plan.publishes_candidates {
            tx.candidates = true;
        }
        tx.next_layer += 1;
        Ok(())
    }
    /// Prefix/history commit is atomic only after all40 layers and final head
    /// succeed. The caller must check final head before invoking this method.
    pub fn commit(&mut self, ticket: &ChunkTicket) -> Result<()> {
        ensure!(
            self.transaction(ticket)?.next_layer == LAYERS,
            "cannot commit incomplete backbone"
        );
        let p = self.lanes.get_mut(&ticket.lane.slot).unwrap();
        p.committed = ticket.end;
        p.open = None;
        Ok(())
    }
    pub fn abort(&mut self, ticket: &ChunkTicket) -> Result<()> {
        self.transaction(ticket)?;
        let p = self.lanes.get_mut(&ticket.lane.slot).unwrap();
        p.failed = true;
        p.open = None;
        Ok(())
    }
    pub fn committed_prefix(&self, lane: LaneKey) -> Result<usize> {
        let p = self.lanes.get(&lane.slot).context("unbound chunk lane")?;
        ensure!(p.lane == lane && !p.failed, "stale/failed chunk lane");
        Ok(p.committed)
    }
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct NativeCachePayload {
    pub context: usize,
    pub lanes: usize,
    pub window_bf16_bytes: u64,
    pub compressed_bf16_bytes: u64,
    pub index_key_bf16_bytes: u64,
    pub subtotal_bytes: u64,
    /// Fixed-stride simple cache allocation is separate from native payload.
    pub simple_chunk_store_allocated_bytes: u64,
    pub ratio_aware_index_store_allocated_bytes: u64,
    pub combined_draft_allocation_bytes: u64,
    pub measured_peak_or_fit_claim: bool,
}
pub fn native_cache_payload(context: usize, lanes: usize) -> Result<NativeCachePayload> {
    ensure!(
        (1..=MAX_CONTEXT).contains(&context) && (1..=16).contains(&lanes),
        "native cache accounting domain"
    );
    let rows = 3 * (context / 2) + context;
    let window = (lanes * LAYERS * WINDOW * 512 * 2) as u64;
    let compressed = (lanes * rows * 512 * 2) as u64;
    let index = (lanes * rows * 128 * 2) as u64;
    Ok(NativeCachePayload {
        context,
        lanes,
        window_bf16_bytes: window,
        compressed_bf16_bytes: compressed,
        index_key_bf16_bytes: index,
        subtotal_bytes: window + compressed + index,
        simple_chunk_store_allocated_bytes: window + (lanes * 4 * MAX_CONTEXT * 512 * 2) as u64,
        ratio_aware_index_store_allocated_bytes: (lanes
            * (3 * (MAX_CONTEXT / 2) + MAX_CONTEXT)
            * 128
            * 2) as u64,
        combined_draft_allocation_bytes: window
            + (lanes * 4 * MAX_CONTEXT * 512 * 2) as u64
            + (lanes * (3 * (MAX_CONTEXT / 2) + MAX_CONTEXT) * 128 * 2) as u64,
        measured_peak_or_fit_claim: false,
    })
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
    fn lane(boot: Uuid, slot: usize, generation: u64) -> LaneKey {
        LaneKey {
            boot,
            slot,
            generation,
        }
    }
    fn source_one_token_globals(position: usize) -> Vec<usize> {
        let p = attention_state::window_plan(WINDOW, position, 1).unwrap();
        p.query_indices[0]
            .iter()
            .filter_map(|&slot| {
                if slot < 0 {
                    None
                } else {
                    let slot = slot as usize;
                    Some(position - (position - slot) % WINDOW)
                }
            })
            .collect()
    }
    #[test]
    fn window_chunks_preserve_every_source_one_token_reachable_key_and_order() {
        for start in [
            0, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 8064, 8191,
        ] {
            for tokens in [1, 2, 3, 63, 127, 128] {
                if start + tokens > MAX_CONTEXT {
                    continue;
                }
                let p = window_chunk(start, tokens).unwrap();
                assert!(p.snapshot_rows <= 255);
                for (q, row) in p.query_indices.iter().enumerate() {
                    let globals = row
                        .iter()
                        .filter(|&&i| i >= 0)
                        .map(|&i| p.snapshot_global_base + i as usize)
                        .collect::<Vec<_>>();
                    assert_eq!(globals, source_one_token_globals(start + q));
                    assert!(row.iter().all(|&i| i < 0 || (i as usize) < p.snapshot_rows));
                }
                assert!(p
                    .snapshot_prior_reads
                    .iter()
                    .all(|&(global, slot)| global < start && slot == global % 128));
                assert_eq!(p.ring_writes_after_snapshot.len(), tokens);
            }
        }
        assert!(window_chunk(8192, 1).is_err());
        assert!(window_chunk(0, 129).is_err());
        assert!(window_chunk(usize::MAX, 1).is_err());
    }
    #[test]
    fn odd_tail_group_mapping_and_per_query_causality_equal_source_one_token_plans() {
        for start in [0, 1, 2, 127, 128, 129, 255, 256, 257, 8063] {
            for tokens in [1, 2, 3, 127, 128] {
                if start + tokens > MAX_CONTEXT {
                    continue;
                }
                for ratio in [1, 2] {
                    let p = compressed_chunk(start, tokens, ratio).unwrap();
                    let mut positions = Vec::new();
                    let mut caches = Vec::new();
                    for q in 0..tokens {
                        let one =
                            attention_state::compression_progress(start + q, 1, ratio).unwrap();
                        positions.extend(one.new_latent_rope_positions);
                        caches.extend(one.prior_rows..one.prior_rows + one.append_rows);
                        assert_eq!(p.visible_rows_per_query[q], (start + q + 1) / ratio);
                    }
                    assert_eq!(
                        p.groups.iter().map(|g| g.rope_position).collect::<Vec<_>>(),
                        positions
                    );
                    assert_eq!(
                        p.groups.iter().map(|g| g.cache_row).collect::<Vec<_>>(),
                        caches
                    );
                    assert_eq!(p.new_rows, p.groups.len());
                    assert_eq!(
                        p.required_old_tail,
                        if ratio == 2 && start % 2 == 1 {
                            Some(start - 1)
                        } else {
                            None
                        }
                    );
                    for g in &p.groups {
                        assert_eq!(g.inputs.len(), ratio);
                    }
                }
            }
        }
        assert_eq!(
            compressed_chunk(127, 128, 2).unwrap().groups[0].inputs,
            [
                PoolInput::ExistingF32Tail {
                    global_position: 126
                },
                PoolInput::CurrentProjection {
                    row: 0,
                    global_position: 127
                }
            ]
        );
        assert!(compressed_chunk(0, 1, 0).is_err());
        assert!(compressed_chunk(0, 1, 3).is_err());
    }
    #[test]
    fn source_publication_prefix_commit_and_forged_or_failed_generations_are_strict() {
        let boot = Uuid::from_u128(37);
        let a = lane(boot, 0, 1);
        let b = lane(boot, 1, 1);
        let mut c = ChunkCursor::new(&config(), boot, 2).unwrap();
        assert!(c.reset(lane(Uuid::from_u128(999), 0, 1)).is_err());
        assert!(c.reset(lane(boot, 2, 1)).is_err());
        assert!(c.reset(lane(boot, 0, 0)).is_err());
        c.reset(a).unwrap();
        c.reset(b).unwrap();
        let t = c.begin(a, 0, 127).unwrap();
        let other = c.begin(b, 0, 128).unwrap();
        assert!(other.version > t.version);
        assert!(c.begin(a, 0, 1).is_err());
        assert!(c.layer(&t, 1).is_err());
        assert!(c.commit(&t).is_err());
        let mut crossed = t.clone();
        crossed.lane = b;
        assert!(c.layer(&crossed, 0).is_err());
        for layer in 0..40 {
            let p = c.layer(&t, layer).unwrap();
            if layer == 2 {
                assert_eq!(p.compressed.as_ref().unwrap().new_rows, 63);
                assert!(!p.index_scoring_required);
            }
            let mut forged = p.clone();
            forged.window.snapshot_global_base += 1;
            assert!(c.commit_layer(&forged).is_err());
            if layer == 2 {
                let mut crossed = p.clone();
                crossed.kv_source = Some(8);
                assert!(c.commit_layer(&crossed).is_err());
                let mut future = p.clone();
                future.compressed.as_mut().unwrap().visible_rows_per_query[0] = 63;
                assert!(c.commit_layer(&future).is_err());
            }
            c.commit_layer(&p).unwrap();
            assert!(c.commit_layer(&p).is_err());
        }
        assert_eq!(c.committed_prefix(a).unwrap(), 0);
        c.commit(&t).unwrap();
        assert_eq!(c.committed_prefix(a).unwrap(), 127);
        let next = c.begin(a, 127, 128).unwrap();
        let p = c.layer(&next, 0).unwrap();
        assert_eq!(p.window.previous_rows, 127);
        c.commit_layer(&p).unwrap();
        c.abort(&next).unwrap();
        assert!(c.committed_prefix(a).is_err());
        assert!(c.begin(a, 127, 1).is_err());
        assert!(c.reset(a).is_err());
        let new = lane(boot, 0, 2);
        c.reset(new).unwrap();
        let newer = c.begin(new, 0, 1).unwrap();
        assert!(newer.version > next.version);
        assert!(c.layer(&t, 0).is_err());
        assert!(c.layer(&next, 1).is_err());
        c.abort(&other).unwrap();
        assert!(c.begin(b, 0, 1).is_err());
    }
    #[test]
    fn all8192_tokens_chunk_without_reinterpreting_short_cache_or_exceeding_source_sharing() {
        let boot = Uuid::from_u128(52);
        let key = lane(boot, 0, 1);
        let mut c = ChunkCursor::new(&config(), boot, 1).unwrap();
        c.reset(key).unwrap();
        let mut versions = Vec::new();
        for start in (0..8192).step_by(128) {
            let t = c.begin(key, start, 128).unwrap();
            versions.push(t.version);
            for layer in 0..40 {
                let p = c.layer(&t, layer).unwrap();
                assert!(p.attention_index_capacity <= 640 && p.full_candidate_selection_exact);
                if let Some(compressed) = &p.compressed {
                    assert!(compressed.stored_rows <= 8192);
                }
                assert!(!p.numerical_or_model_equivalence_qualified);
                c.commit_layer(&p).unwrap();
            }
            c.commit(&t).unwrap();
        }
        assert_eq!(c.committed_prefix(key).unwrap(), 8192);
        assert!(c.begin(key, 8192, 1).is_err());
        assert!(versions.windows(2).all(|w| w[0] < w[1]));
        let bytes = native_cache_payload(8192, 1).unwrap();
        assert_eq!(
            (
                bytes.window_bf16_bytes,
                bytes.compressed_bf16_bytes,
                bytes.index_key_bf16_bytes,
                bytes.subtotal_bytes
            ),
            (5242880, 20971520, 5242880, 31457280)
        );
        assert_eq!(
            native_cache_payload(8192, 16).unwrap().subtotal_bytes,
            503316480
        );
        assert!(!bytes.measured_peak_or_fit_claim);
        assert_eq!(bytes.simple_chunk_store_allocated_bytes, 38797312);
        assert_eq!(bytes.ratio_aware_index_store_allocated_bytes, 5242880);
        assert_eq!(bytes.combined_draft_allocation_bytes, 44040192);
        assert!(native_cache_payload(8193, 1).is_err());
    }
}
