//! PRIVATE atomic ragged-row ownership for native1017. CPU metadata only;
//! batching is proved later by combined GPU execution, never HTTP overlap.
use super::{
    batch_attention_layout::{AttentionPhase, LaneAttentionLayout},
    chunk_plan::{self, ChunkCursor, ChunkTicket, LayerChunk},
    collectives::RowKind,
    config::Dsv41Config,
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

pub const MAX_LANES: usize = 16;
pub const MAX_ROWS: usize = 128;
pub const MAX_CONTEXT: usize = 8192;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PackedKind {
    PrefillSegments,
    DecodeLanes,
}
impl PackedKind {
    pub fn collective_rows(self) -> RowKind {
        match self {
            Self::PrefillSegments => RowKind::ContiguousPrefillRows,
            Self::DecodeLanes => RowKind::IndependentDecodeLanes,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LaneSegment {
    pub key: LaneKey,
    pub start: usize,
    pub tokens: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PackedSegment {
    pub ticket: ChunkTicket,
    pub first_row: usize,
    pub rows: usize,
    pub phase: AttentionPhase,
}
/// Only BatchCursor::begin constructs this aggregate. Its native per-lane
/// tickets remain public metadata; the active cursor validates the exact copy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchTicket {
    boot: Uuid,
    version: u64,
    kind: PackedKind,
    segments: Vec<PackedSegment>,
    total_rows: usize,
    identity_sha256: String,
}
impl BatchTicket {
    pub fn boot(&self) -> Uuid {
        self.boot
    }
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn kind(&self) -> PackedKind {
        self.kind
    }
    pub fn segments(&self) -> &[PackedSegment] {
        &self.segments
    }
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
    pub fn lanes(&self) -> usize {
        self.segments.len()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BatchLayerPlan {
    pub ticket: BatchTicket,
    pub layer: usize,
    pub lanes: Vec<LayerChunk>,
    pub logical: Vec<LaneAttentionLayout>,
}
impl BatchLayerPlan {
    pub fn short_selection_exact(&self) -> bool {
        self.lanes
            .iter()
            .all(|p| p.compressed.as_ref().is_none_or(|c| c.stored_rows <= 512))
    }
    pub fn window_rows(&self) -> usize {
        self.lanes.iter().map(|p| p.window.snapshot_rows).sum()
    }
    pub fn kv_rows(&self) -> usize {
        self.lanes
            .iter()
            .map(|p| p.window.snapshot_rows + p.compressed.as_ref().map_or(0, |c| c.stored_rows))
            .sum()
    }
    pub fn maximum_topk(&self) -> usize {
        self.logical
            .iter()
            .map(|p| p.logical_topk())
            .max()
            .unwrap_or(0)
    }
}
#[derive(Clone, Debug)]
struct Active {
    ticket: BatchTicket,
    next_layer: usize,
}
pub struct BatchCursor {
    boot: Uuid,
    max_lanes: usize,
    next_batch_version: u64,
    native: ChunkCursor,
    registered: BTreeMap<usize, LaneKey>,
    prompt_extents: BTreeMap<usize, usize>,
    failed: BTreeSet<usize>,
    active: Option<Active>,
}
fn validate_segments(
    boot: Uuid,
    max_lanes: usize,
    kind: PackedKind,
    segments: &[LaneSegment],
) -> Result<usize> {
    ensure!(
        (1..=max_lanes).contains(&segments.len()),
        "packed batch lane count differs"
    );
    let mut seen = BTreeSet::new();
    let mut rows = 0usize;
    for s in segments {
        ensure!(
            s.key.boot == boot
                && s.key.slot < max_lanes
                && s.key.generation > 0
                && seen.insert(s.key.slot),
            "distinct live physical slots/sameboot required"
        );
        chunk_plan::window_chunk(s.start, s.tokens)?;
        if kind == PackedKind::DecodeLanes {
            ensure!(
                s.tokens == 1 && s.start > 0,
                "decode packs one token per initialized lane"
            );
        }
        rows = rows.checked_add(s.tokens).context("packed rows overflow")?;
    }
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "total ragged native rows exceed128"
    );
    Ok(rows)
}
impl BatchCursor {
    pub fn new(c: &Dsv41Config, boot: Uuid, max_lanes: usize) -> Result<Self> {
        chunk_plan::validate_config(c)?;
        ensure!(
            !boot.is_nil() && (1..=MAX_LANES).contains(&max_lanes),
            "nonempty boot/bounded lanes required"
        );
        Ok(Self {
            boot,
            max_lanes,
            next_batch_version: 1,
            native: ChunkCursor::new(c, boot, max_lanes)?,
            registered: BTreeMap::new(),
            prompt_extents: BTreeMap::new(),
            failed: BTreeSet::new(),
            active: None,
        })
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            self.active.is_none(),
            "cannot reset a physical slot during an active packed transaction"
        );
        ensure!(
            key.boot == self.boot
                && key.slot < self.max_lanes
                && key.generation > 0
                && self
                    .registered
                    .get(&key.slot)
                    .is_none_or(|old| key.generation > old.generation),
            "reset must advance physical lane generation"
        );
        self.native.reset(key)?;
        self.registered.insert(key.slot, key);
        self.prompt_extents.remove(&key.slot);
        self.failed.remove(&key.slot);
        Ok(())
    }
    /// Validate every participant BEFORE reserving any per-lane transaction.
    /// A later internal reservation error poisons all participants; no partial
    /// aggregate is returned and no normal GPU/model progress is implied.
    pub fn begin(&mut self, kind: PackedKind, segments: Vec<LaneSegment>) -> Result<BatchTicket> {
        let phases = segments
            .iter()
            .map(|s| match kind {
                PackedKind::PrefillSegments => {
                    ensure!(
                        s.start == 0,
                        "partial positive prefill requires explicit full-prompt extent"
                    );
                    Ok(AttentionPhase::Prefill {
                        full_prompt_tokens: s.tokens,
                    })
                }
                PackedKind::DecodeLanes => Ok(AttentionPhase::Decode),
            })
            .collect::<Result<Vec<_>>>()?;
        self.begin_with_phases(kind, segments, phases)
    }
    /// Partial packed prefill must carry the original request's full prompt
    /// extent. It is not inferred from the current segment or native position.
    pub fn begin_prefill_segments(
        &mut self,
        segments: Vec<(LaneSegment, usize)>,
    ) -> Result<BatchTicket> {
        let (segments, phases): (Vec<_>, Vec<_>) = segments
            .into_iter()
            .map(|(s, full)| {
                (
                    s,
                    AttentionPhase::Prefill {
                        full_prompt_tokens: full,
                    },
                )
            })
            .unzip();
        self.begin_with_phases(PackedKind::PrefillSegments, segments, phases)
    }
    fn begin_with_phases(
        &mut self,
        kind: PackedKind,
        segments: Vec<LaneSegment>,
        phases: Vec<AttentionPhase>,
    ) -> Result<BatchTicket> {
        ensure!(
            segments.len() == phases.len(),
            "each packed segment needs its semantic phase"
        );
        ensure!(
            self.active.is_none(),
            "one packed model transaction at a time"
        );
        let total_rows = validate_segments(self.boot, self.max_lanes, kind, &segments)?;
        for s in &segments {
            ensure!(
                self.registered.get(&s.key.slot) == Some(&s.key)
                    && !self.failed.contains(&s.key.slot)
                    && self.native.committed_prefix(s.key)? == s.start,
                "unbound/stale/failed/noncontiguous packed lane"
            );
        }
        for (s, phase) in segments.iter().zip(&phases) {
            match phase {
                AttentionPhase::Prefill { full_prompt_tokens } => {
                    ensure!(
                        kind == PackedKind::PrefillSegments
                            && (1..=MAX_CONTEXT).contains(full_prompt_tokens)
                            && s.start + s.tokens <= *full_prompt_tokens,
                        "full original prompt bound differs"
                    );
                    if s.start == 0 {
                        ensure!(
                            !self.prompt_extents.contains_key(&s.key.slot),
                            "cannot reinterpret an existing prompt"
                        );
                    } else {
                        ensure!(
                            self.prompt_extents.get(&s.key.slot) == Some(full_prompt_tokens)
                                && s.start < *full_prompt_tokens,
                            "positive prefill changed full prompt extent"
                        );
                    }
                }
                AttentionPhase::Decode => ensure!(
                    kind == PackedKind::DecodeLanes
                        && self
                            .prompt_extents
                            .get(&s.key.slot)
                            .is_some_and(|full| s.start >= *full),
                    "decode precedes complete original prompt"
                ),
            }
        }
        let version = self.next_batch_version;
        self.next_batch_version = version
            .checked_add(1)
            .context("packed batch version exhausted")?;
        let mut packed = Vec::with_capacity(segments.len());
        let mut row = 0;
        for (s, phase) in segments.iter().zip(&phases) {
            match self.native.begin(s.key, s.start, s.tokens) {
                Ok(ticket) => {
                    packed.push(PackedSegment {
                        ticket,
                        first_row: row,
                        rows: s.tokens,
                        phase: *phase,
                    });
                    row += s.tokens;
                }
                Err(e) => {
                    for p in &packed {
                        let _ = self.native.abort(&p.ticket);
                    }
                    self.failed.extend(segments.iter().map(|s| s.key.slot));
                    return Err(e);
                }
            }
        }
        let mut h = Sha256::new();
        h.update(b"dsv41-atomic-ragged-batch-ticket-v2-source-logical-slots\0");
        h.update(serde_json::to_vec(&(
            self.boot, version, kind, &packed, total_rows,
        ))?);
        let ticket = BatchTicket {
            boot: self.boot,
            version,
            kind,
            segments: packed,
            total_rows,
            identity_sha256: format!("{:x}", h.finalize()),
        };
        self.active = Some(Active {
            ticket: ticket.clone(),
            next_layer: 0,
        });
        Ok(ticket)
    }
    fn active(&self, ticket: &BatchTicket) -> Result<&Active> {
        let a = self
            .active
            .as_ref()
            .context("no active packed transaction")?;
        ensure!(a.ticket == *ticket, "forged/stale packed ticket");
        ensure!(
            ticket
                .segments
                .iter()
                .all(
                    |s| self.registered.get(&s.ticket.lane.slot) == Some(&s.ticket.lane)
                        && !self.failed.contains(&s.ticket.lane.slot)
                ),
            "packed lane generation failed/changed"
        );
        Ok(a)
    }
    pub fn layer(&self, ticket: &BatchTicket, layer: usize) -> Result<BatchLayerPlan> {
        ensure!(
            self.active(ticket)?.next_layer == layer && layer < 40,
            "packed layers must finish in source order"
        );
        let lanes = ticket
            .segments
            .iter()
            .map(|s| self.native.layer(&s.ticket, layer))
            .collect::<Result<Vec<_>>>()?;
        let logical = lanes
            .iter()
            .zip(&ticket.segments)
            .map(|(p, s)| LaneAttentionLayout::new(p, s.phase))
            .collect::<Result<Vec<_>>>()?;
        Ok(BatchLayerPlan {
            ticket: ticket.clone(),
            layer,
            lanes,
            logical,
        })
    }
    pub fn commit_layer(&mut self, plan: &BatchLayerPlan) -> Result<()> {
        ensure!(
            self.layer(&plan.ticket, plan.layer)? == *plan,
            "packed layer metadata/source receipts differ"
        );
        for lane in &plan.lanes {
            if let Err(e) = self.native.commit_layer(lane) {
                let _ = self.abort(&plan.ticket);
                return Err(e);
            }
        }
        self.active.as_mut().unwrap().next_layer += 1;
        Ok(())
    }
    /// Caller has checked every lane's head/full-step result. This does not
    /// create model authority; only native/model owners may certify that fact.
    pub fn commit(&mut self, ticket: &BatchTicket) -> Result<()> {
        ensure!(
            self.active(ticket)?.next_layer == 40,
            "all40 packed layers must complete before prefix commit"
        );
        for segment in &ticket.segments {
            if let Err(e) = self.native.commit(&segment.ticket) {
                let _ = self.abort(ticket);
                return Err(e);
            }
        }
        for s in &ticket.segments {
            if let AttentionPhase::Prefill { full_prompt_tokens } = s.phase {
                self.prompt_extents
                    .insert(s.ticket.lane.slot, full_prompt_tokens);
            }
        }
        self.active = None;
        Ok(())
    }
    pub fn abort(&mut self, ticket: &BatchTicket) -> Result<()> {
        self.active(ticket)?;
        for s in &ticket.segments {
            let _ = self.native.abort(&s.ticket);
            self.failed.insert(s.ticket.lane.slot);
        }
        self.active = None;
        Ok(())
    }
    pub fn committed_prefix(&self, key: LaneKey) -> Result<usize> {
        ensure!(
            self.registered.get(&key.slot) == Some(&key) && !self.failed.contains(&key.slot),
            "stale/failed packed lane"
        );
        self.native.committed_prefix(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Dsv41Config {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap()
    }
    fn key(slot: usize) -> LaneKey {
        LaneKey {
            boot: Uuid::from_u128(14),
            slot,
            generation: 1,
        }
    }
    fn cursor() -> BatchCursor {
        let mut c = BatchCursor::new(&config(), key(0).boot, 16).unwrap();
        for i in 0..16 {
            c.reset_lane(key(i)).unwrap();
        }
        c
    }
    fn finish(c: &mut BatchCursor, t: &BatchTicket) {
        let first = c.active(t).unwrap().next_layer;
        for l in first..40 {
            let p = c.layer(t, l).unwrap();
            c.commit_layer(&p).unwrap();
        }
        c.commit(t).unwrap();
    }
    fn advance_to(c: &mut BatchCursor, t: &BatchTicket, layer: usize) -> BatchLayerPlan {
        while c.active(t).unwrap().next_layer < layer {
            let l = c.active(t).unwrap().next_layer;
            let p = c.layer(t, l).unwrap();
            c.commit_layer(&p).unwrap();
        }
        c.layer(t, layer).unwrap()
    }
    #[test]
    fn ragged_row_ownership_is_explicit_and_decode_compaction_preserves_slots() {
        let mut c = cursor();
        let t = c
            .begin(
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: key(7),
                        start: 0,
                        tokens: 75,
                    },
                    LaneSegment {
                        key: key(2),
                        start: 0,
                        tokens: 53,
                    },
                ],
            )
            .unwrap();
        assert_eq!(t.total_rows(), 128);
        assert_eq!(t.segments[0].first_row, 0);
        assert_eq!(t.segments[1].first_row, 75);
        finish(&mut c, &t);
        let d = c
            .begin(
                PackedKind::DecodeLanes,
                vec![
                    LaneSegment {
                        key: key(2),
                        start: 53,
                        tokens: 1,
                    },
                    LaneSegment {
                        key: key(7),
                        start: 75,
                        tokens: 1,
                    },
                ],
            )
            .unwrap();
        let p = advance_to(&mut c, &d, 2);
        assert_eq!(p.lanes[0].ticket.lane.slot, 2);
        assert_eq!(
            p.lanes[0].compressed.as_ref().unwrap().required_old_tail,
            Some(52)
        );
        assert_eq!(p.lanes[1].ticket.lane.slot, 7);
        assert_eq!(
            p.lanes[1].compressed.as_ref().unwrap().required_old_tail,
            Some(74)
        );
        finish(&mut c, &d);
        assert_eq!(c.committed_prefix(key(2)).unwrap(), 54);
        assert_eq!(c.committed_prefix(key(7)).unwrap(), 76);
    }
    #[test]
    fn admission_is_all_or_none_and_complete_batch_failure_poisons_participants_only() {
        let mut c = cursor();
        assert!(c
            .begin(
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 64
                    },
                    LaneSegment {
                        key: key(1),
                        start: 1,
                        tokens: 64
                    }
                ]
            )
            .is_err());
        assert_eq!(c.committed_prefix(key(0)).unwrap(), 0);
        let t = c
            .begin(
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 64,
                    },
                    LaneSegment {
                        key: key(1),
                        start: 0,
                        tokens: 64,
                    },
                ],
            )
            .unwrap();
        assert!(c.commit(&t).is_err());
        c.abort(&t).unwrap();
        assert!(c.committed_prefix(key(0)).is_err());
        assert!(c.committed_prefix(key(1)).is_err());
        assert_eq!(c.committed_prefix(key(2)).unwrap(), 0);
        let mut fresh = key(0);
        fresh.generation = 2;
        c.reset_lane(fresh).unwrap();
        assert_eq!(c.committed_prefix(fresh).unwrap(), 0);
        assert!(c.committed_prefix(key(0)).is_err());
    }
    #[test]
    fn total_rows_unique_slots_layer_order_and_decode_posture_fail_closed() {
        let mut c = cursor();
        for (kind, s) in [
            (
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 65,
                    },
                    LaneSegment {
                        key: key(1),
                        start: 0,
                        tokens: 64,
                    },
                ],
            ),
            (
                PackedKind::PrefillSegments,
                vec![
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 1,
                    },
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 1,
                    },
                ],
            ),
            (
                PackedKind::DecodeLanes,
                vec![LaneSegment {
                    key: key(0),
                    start: 0,
                    tokens: 1,
                }],
            ),
        ] {
            assert!(c.begin(kind, s).is_err());
        }
        let t = c
            .begin(
                PackedKind::PrefillSegments,
                vec![LaneSegment {
                    key: key(0),
                    start: 0,
                    tokens: 1,
                }],
            )
            .unwrap();
        assert!(c.layer(&t, 1).is_err());
        let mut fresh = key(3);
        fresh.generation = 2;
        assert!(c.reset_lane(fresh).is_err());
        let mut forged = t.clone();
        forged.total_rows += 1;
        assert!(c.layer(&forged, 0).is_err());
        finish(&mut c, &t);
        assert!(c.layer(&t, 0).is_err());
    }
    #[test]
    fn true_b1_2_4_8_16_rows_keep_independent_prefixes_through_all_source_layers() {
        for b in [1, 2, 4, 8, 16] {
            let mut c = cursor();
            let n = 128 / b;
            let t = c
                .begin(
                    PackedKind::PrefillSegments,
                    (0..b)
                        .map(|i| LaneSegment {
                            key: key(i),
                            start: 0,
                            tokens: n,
                        })
                        .collect(),
                )
                .unwrap();
            finish(&mut c, &t);
            let d = c
                .begin(
                    PackedKind::DecodeLanes,
                    (0..b)
                        .rev()
                        .map(|i| LaneSegment {
                            key: key(i),
                            start: n,
                            tokens: 1,
                        })
                        .collect(),
                )
                .unwrap();
            assert_eq!(d.total_rows(), b);
            assert_eq!(d.kind().collective_rows(), RowKind::IndependentDecodeLanes);
            finish(&mut c, &d);
            for i in 0..b {
                assert_eq!(c.committed_prefix(key(i)).unwrap(), n + 1);
            }
        }
    }
    #[test]
    fn ragged_partial_prefill_binds_full_extent_and_refuses_early_decode_or_reinterpretation() {
        let mut c = cursor();
        let first = c
            .begin_prefill_segments(vec![
                (
                    LaneSegment {
                        key: key(0),
                        start: 0,
                        tokens: 16,
                    },
                    75,
                ),
                (
                    LaneSegment {
                        key: key(1),
                        start: 0,
                        tokens: 16,
                    },
                    53,
                ),
            ])
            .unwrap();
        let p = advance_to(&mut c, &first, 2);
        assert_eq!(p.logical[0].logical_topk(), 112);
        assert_eq!(p.logical[1].logical_topk(), 79);
        finish(&mut c, &first);
        assert!(c
            .begin(
                PackedKind::DecodeLanes,
                vec![LaneSegment {
                    key: key(0),
                    start: 16,
                    tokens: 1
                }]
            )
            .is_err());
        assert!(c
            .begin_prefill_segments(vec![(
                LaneSegment {
                    key: key(0),
                    start: 16,
                    tokens: 16
                },
                74
            )])
            .is_err());
        let second = c
            .begin_prefill_segments(vec![
                (
                    LaneSegment {
                        key: key(1),
                        start: 16,
                        tokens: 37,
                    },
                    53,
                ),
                (
                    LaneSegment {
                        key: key(0),
                        start: 16,
                        tokens: 59,
                    },
                    75,
                ),
            ])
            .unwrap();
        let p = advance_to(&mut c, &second, 2);
        assert_eq!(p.logical[0].logical_topk(), 79);
        assert_eq!(p.logical[1].logical_topk(), 112);
        finish(&mut c, &second);
        let d = c
            .begin(
                PackedKind::DecodeLanes,
                vec![
                    LaneSegment {
                        key: key(0),
                        start: 75,
                        tokens: 1,
                    },
                    LaneSegment {
                        key: key(1),
                        start: 53,
                        tokens: 1,
                    },
                ],
            )
            .unwrap();
        let p = advance_to(&mut c, &d, 2);
        assert_eq!(p.logical[0].logical_window_columns, 128);
        assert_eq!(p.logical[1].logical_window_columns, 128);
        finish(&mut c, &d);
    }
}
