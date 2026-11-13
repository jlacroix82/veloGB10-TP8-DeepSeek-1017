//! PRIVATE CPU translation from sealed batch tickets to native ratio2 groups.
//! No new tensor values: flat-row metadata preserves the source F32 odd tail.
use super::{
    batch_plan::BatchLayerPlan,
    chunk_plan::{self, PoolInput},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
pub const OLD_TAIL: u64 = u64::MAX;
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompressionMember {
    pub key: LaneKey,
    pub start: usize,
    pub end: usize,
    pub new_rows: usize,
    pub first_output_row: usize,
    pub rope_positions: Vec<u32>,
    pub required_old_tail: Option<usize>,
    pub retained_new_tail: Option<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PackedCompressionPlan {
    pub batch_sha256: String,
    pub source_layer: usize,
    pub ratio: usize,
    pub input_rows: usize,
    pub output_rows: usize,
    pub groups: Vec<[u64; 3]>,
    pub tails: Vec<[u64; 2]>,
    pub members: Vec<CompressionMember>,
    pub rope_positions: Vec<u32>,
}
pub fn build(plan: &BatchLayerPlan) -> Result<PackedCompressionPlan> {
    ensure!(
        chunk_plan::KV_SOURCES.contains(&plan.layer)
            && plan.lanes.len() == plan.ticket.segments().len(),
        "native compressed source/lane geometry required"
    );
    let ratio = if plan.layer < 20 { 2 } else { 1 };
    let mut result = PackedCompressionPlan {
        batch_sha256: plan.ticket.identity_sha256().into(),
        source_layer: plan.layer,
        ratio,
        input_rows: plan.ticket.total_rows(),
        output_rows: 0,
        groups: Vec::new(),
        tails: Vec::new(),
        members: Vec::new(),
        rope_positions: Vec::new(),
    };
    for (segment, native) in plan.ticket.segments().iter().zip(&plan.lanes) {
        let t = &native.ticket;
        ensure!(
            t == &segment.ticket
                && native.layer == plan.layer
                && native.ratio == ratio
                && segment.rows == t.tokens
                && segment.first_row + t.tokens <= result.input_rows,
            "packed compressor member differs from source ticket"
        );
        let c = native
            .compressed
            .as_ref()
            .context("source plan lacks compressed geometry")?;
        ensure!(
            *c == chunk_plan::compressed_chunk(t.start, t.tokens, ratio)?,
            "forged native group geometry"
        );
        let first = result.output_rows;
        for group in &c.groups {
            ensure!(
                group.inputs.len() == ratio && group.rope_position == group.cache_row * ratio,
                "native source group/first-token RoPE differs"
            );
            let mut projected = Vec::new();
            for input in &group.inputs {
                projected.push(match input {
                    PoolInput::ExistingF32Tail { global_position } => {
                        ensure!(
                            ratio == 2 && *global_position + 1 == t.start && t.start % 2 == 1,
                            "old tail must be the previous unrounded odd half"
                        );
                        OLD_TAIL
                    }
                    PoolInput::CurrentProjection {
                        row,
                        global_position,
                    } => {
                        ensure!(
                            *row < t.tokens && *global_position == t.start + row,
                            "group projection row differs"
                        );
                        (segment.first_row + row) as u64
                    }
                });
            }
            if ratio == 2 {
                ensure!(
                    projected[1] != OLD_TAIL,
                    "second member must be a current projection"
                );
                result
                    .groups
                    .push([t.lane.slot as u64, projected[0], projected[1]]);
            }
            result.output_rows += 1;
            result.rope_positions.push(group.rope_position as u32);
        }
        if let Some(tail) = c.retained_new_tail {
            ensure!(
                ratio == 2 && tail + 1 == t.end,
                "retained tail is final current row"
            );
            result.tails.push([
                t.lane.slot as u64,
                (segment.first_row + t.tokens - 1) as u64,
            ]);
        }
        result.members.push(CompressionMember {
            key: t.lane,
            start: t.start,
            end: t.end,
            new_rows: c.new_rows,
            first_output_row: first,
            rope_positions: c.groups.iter().map(|g| g.rope_position as u32).collect(),
            required_old_tail: c.required_old_tail,
            retained_new_tail: c.retained_new_tail,
        });
    }
    ensure!(
        result.input_rows > 0
            && result.input_rows <= 128
            && result.output_rows <= 128
            && result.tails.len() <= 16
            && result.output_rows == result.members.iter().map(|m| m.new_rows).sum::<usize>()
            && (ratio == 1 || result.groups.len() == result.output_rows),
        "bounded exact packed source output shape"
    );
    if ratio == 1 {
        ensure!(
            result.output_rows == result.input_rows
                && result.groups.is_empty()
                && result.tails.is_empty(),
            "ratio1 is source BF16 passthrough before RMS"
        );
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::{
        batch_plan::{BatchCursor, BatchTicket, LaneSegment, PackedKind},
        config::Dsv41Config,
    };
    use uuid::Uuid;
    fn config() -> Dsv41Config {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap()
    }
    fn key(i: usize) -> LaneKey {
        LaneKey {
            boot: Uuid::from_u128(16412),
            slot: i,
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
    fn layer(c: &mut BatchCursor, t: &BatchTicket, target: usize) -> BatchLayerPlan {
        for l in 0..target {
            let p = c.layer(t, l).unwrap();
            c.commit_layer(&p).unwrap();
        }
        c.layer(t, target).unwrap()
    }
    fn finish(c: &mut BatchCursor, t: &BatchTicket, first: usize) {
        for l in first..40 {
            let p = c.layer(t, l).unwrap();
            c.commit_layer(&p).unwrap();
        }
        c.commit(t).unwrap();
    }
    #[test]
    fn independent_odd_tails_complete_exact_groups_after_B1_2_4_8_16_compaction() {
        for b in [1, 2, 4, 8, 16] {
            let mut c = cursor();
            let first = c
                .begin(
                    PackedKind::PrefillSegments,
                    (0..b)
                        .map(|i| LaneSegment {
                            key: key(i),
                            start: 0,
                            tokens: 7,
                        })
                        .collect(),
                )
                .unwrap();
            let p = layer(&mut c, &first, 2);
            let g = build(&p).unwrap();
            assert_eq!(g.output_rows, b * 3);
            assert_eq!(g.tails.len(), b);
            finish(&mut c, &first, 2);
            let next = c
                .begin(
                    PackedKind::DecodeLanes,
                    (0..b)
                        .rev()
                        .map(|i| LaneSegment {
                            key: key(i),
                            start: 7,
                            tokens: 1,
                        })
                        .collect(),
                )
                .unwrap();
            let p = layer(&mut c, &next, 2);
            let g = build(&p).unwrap();
            assert_eq!(g.groups.len(), b);
            assert!(g.tails.is_empty());
            for (row, group) in g.groups.iter().enumerate() {
                assert_eq!(*group, [(b - 1 - row) as u64, OLD_TAIL, row as u64]);
                assert_eq!(g.rope_positions[row], 6);
            }
            finish(&mut c, &next, 2);
        }
    }
    #[test]
    fn a_partial_prefill_chunk_can_read_old_tail_and_install_distinct_new_tail() {
        let mut c = cursor();
        let first = c
            .begin_prefill_segments(
                (0..16)
                    .map(|i| {
                        (
                            LaneSegment {
                                key: key(i),
                                start: 0,
                                tokens: 7,
                            },
                            11,
                        )
                    })
                    .collect(),
            )
            .unwrap();
        finish(&mut c, &first, 0);
        let next = c
            .begin_prefill_segments(
                (0..16)
                    .rev()
                    .map(|i| {
                        (
                            LaneSegment {
                                key: key(i),
                                start: 7,
                                tokens: 2,
                            },
                            11,
                        )
                    })
                    .collect(),
            )
            .unwrap();
        let p = layer(&mut c, &next, 2);
        let g = build(&p).unwrap();
        assert_eq!((g.output_rows, g.tails.len()), (16, 16));
        for i in 0..16 {
            assert_eq!(g.groups[i], [(15 - i) as u64, OLD_TAIL, (i * 2) as u64]);
            assert_eq!(g.tails[i], [(15 - i) as u64, (i * 2 + 1) as u64]);
            assert_eq!(g.members[i].required_old_tail, Some(6));
            assert_eq!(g.members[i].retained_new_tail, Some(8));
        }
        let mut forged = p.clone();
        forged.lanes[0].compressed.as_mut().unwrap().groups[0].rope_position = 7;
        assert!(build(&forged).is_err());
    }
    #[test]
    fn ratio1_preserves_complete_ragged_BF16_row_order_and_native_group_positions() {
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
        let p = layer(&mut c, &t, 20);
        let g = build(&p).unwrap();
        assert_eq!((g.input_rows, g.output_rows), (128, 128));
        assert!(g.groups.is_empty() && g.tails.is_empty());
        assert_eq!(g.members[1].first_output_row, 75);
        assert_eq!(
            g.rope_positions,
            [(0..75).collect::<Vec<_>>(), (0..53).collect()].concat()
        );
    }
}
