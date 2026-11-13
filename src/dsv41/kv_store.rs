//! Exact1017 text-only short-context KV ownership and all-selected index contract.
//! Hard512 context makes compressed rows<=index_topk512 and candidate blocks<=64
//! <2048. This does not qualify longer contexts, index scoring or prefix restore.
use super::{attention_state, config::Dsv41Config, scheduler::LaneKey};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use uuid::Uuid;
pub const LAYERS: usize = 40;
pub const SOURCES: [usize; 4] = [2, 8, 14, 20];
pub const MAX_CONTEXT: usize = 512;
pub const MAX_PREFILL: usize = 128;
pub const WINDOW: usize = 128;
pub const DIM: usize = 512;
pub const MAX_LANES: usize = 16;
pub const FIXTURE_SHA: &str = "4e4178a1b2275b401ffa5ca336dc8bcb4c718d8c9f0a0401bc26b22599dd8728";
pub const FIXTURE: &str = include_str!("../../tests/fixtures/dsv41_kv_store_vectors.json");
pub fn ratio(layer: usize) -> Result<usize> {
    ensure!(layer < LAYERS, "1017 layer outside0..40");
    Ok(if layer < 2 {
        0
    } else if layer < 20 {
        2
    } else {
        1
    })
}
pub fn owner(layer: usize) -> Result<Option<usize>> {
    Ok(if ratio(layer)? == 0 {
        None
    } else {
        SOURCES.into_iter().filter(|s| *s <= layer).last()
    })
}
pub fn source_slot(layer: usize) -> Result<usize> {
    SOURCES
        .iter()
        .position(|s| *s == layer)
        .context("not a compressed KV owner")
}
pub fn validate_config(c: &Dsv41Config) -> Result<()> {
    c.validate()?;
    ensure!(
        c.n_layers == LAYERS
            && c.head_dim == DIM
            && c.window_size == WINDOW
            && c.index_topk == MAX_CONTEXT
            && c.kv_source_layers == SOURCES
            && c.candidate_source_layer == 20
            && c.candidate_block_size == 8
            && c.candidate_topk_blocks == 2048,
        "short KV store requires exact1017 source topology/selection settings"
    );
    for layer in 0..LAYERS {
        ensure!(
            c.compress_ratios[layer] as usize == ratio(layer)?,
            "compression ratio differs"
        );
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvPlan {
    pub key: LaneKey,
    pub layer: usize,
    pub start: usize,
    pub tokens: usize,
    pub end: usize,
    pub ratio: usize,
    pub compressed_owner: Option<usize>,
    pub owner_slot: Option<usize>,
    pub new_compressed_rows: usize,
    pub compressed_rows: usize,
    pub window_rows: usize,
    pub kv_rows: usize,
    pub window_topk: usize,
    pub topk: usize,
    pub supplied_compressed_rows: Option<usize>,
}
#[derive(Clone)]
struct Lane {
    key: LaneKey,
    failed: bool,
    window_ends: [Option<usize>; LAYERS],
    compressed_rows: [usize; 4],
    published: [Option<(usize, usize)>; 4],
}
pub struct KvCursor {
    boot: Uuid,
    max_lanes: usize,
    lanes: BTreeMap<usize, Lane>,
}
impl KvCursor {
    pub fn new(boot: Uuid, max_lanes: usize) -> Result<Self> {
        ensure!((1..=MAX_LANES).contains(&max_lanes), "KV lane bound");
        Ok(Self {
            boot,
            max_lanes,
            lanes: BTreeMap::new(),
        })
    }
    pub fn reset(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.boot && key.slot < self.max_lanes && key.generation > 0,
            "KV lane identity invalid"
        );
        ensure!(
            self.lanes
                .get(&key.slot)
                .is_none_or(|old| key.generation > old.key.generation),
            "KV reset generation must increase"
        );
        self.lanes.insert(
            key.slot,
            Lane {
                key,
                failed: false,
                window_ends: [None; LAYERS],
                compressed_rows: [0; 4],
                published: [None; 4],
            },
        );
        Ok(())
    }
    pub fn plan(
        &self,
        layer: usize,
        key: LaneKey,
        start: usize,
        tokens: usize,
        supplied: Option<usize>,
    ) -> Result<KvPlan> {
        let ratio = ratio(layer)?;
        let lane = self.lanes.get(&key.slot).context("unbound KV lane")?;
        ensure!(lane.key == key && !lane.failed, "stale/failed KV lane");
        match lane.window_ends[layer] {
            None => ensure!(
                start == 0 && (1..=MAX_PREFILL).contains(&tokens),
                "initial KV prefill1..128 only"
            ),
            Some(expected) => ensure!(
                start == expected && start > 0 && tokens == 1,
                "KV append must be contiguous one token"
            ),
        }
        let end = start.checked_add(tokens).context("KV position overflow")?;
        ensure!(
            end <= MAX_CONTEXT,
            "short KV path rejects context beyond512"
        );
        let source = owner(layer)?;
        let slot = source.map(source_slot).transpose()?;
        let compressed_rows = if ratio == 0 { 0 } else { end / ratio };
        let new_rows = if ratio == 0 || source != Some(layer) {
            0
        } else {
            end / ratio - start / ratio
        };
        if let Some(s) = slot {
            if source == Some(layer) {
                ensure!(
                    lane.compressed_rows[s] == start / ratio,
                    "compressed owner rows noncontiguous"
                );
                ensure!(
                    if new_rows == 0 {
                        supplied.is_none()
                    } else {
                        supplied == Some(new_rows)
                    },
                    "owner must provide exactly the NEW completed QAT rows"
                );
            } else {
                ensure!(
                    supplied.is_none(),
                    "consumer cannot append another source's compressed cache"
                );
                ensure!(
                    lane.published[s] == Some((start, tokens))
                        && lane.compressed_rows[s] == compressed_rows,
                    "compressed source must publish this exact forward before consumer"
                );
            }
        } else {
            ensure!(
                supplied.is_none(),
                "pure window layer cannot provide compressed rows"
            );
        }
        let window_rows = if start == 0 { tokens } else { WINDOW };
        let window_topk = if start == 0 {
            tokens.min(WINDOW)
        } else {
            WINDOW
        };
        ensure!(
            compressed_rows <= MAX_CONTEXT && compressed_rows.div_ceil(8) <= 2048,
            "all-selected proof outside budget"
        );
        Ok(KvPlan {
            key,
            layer,
            start,
            tokens,
            end,
            ratio,
            compressed_owner: source,
            owner_slot: slot,
            new_compressed_rows: new_rows,
            compressed_rows,
            window_rows,
            kv_rows: window_rows + compressed_rows,
            window_topk,
            topk: window_topk + compressed_rows,
            supplied_compressed_rows: supplied,
        })
    }
    pub fn commit(&mut self, p: &KvPlan) -> Result<()> {
        ensure!(
            self.plan(
                p.layer,
                p.key,
                p.start,
                p.tokens,
                p.supplied_compressed_rows
            )? == *p,
            "forged/stale KV plan"
        );
        let lane = self.lanes.get_mut(&p.key.slot).unwrap();
        lane.window_ends[p.layer] = Some(p.end);
        if p.compressed_owner == Some(p.layer) {
            let s = p.owner_slot.unwrap();
            lane.compressed_rows[s] = p.compressed_rows;
            lane.published[s] = Some((p.start, p.tokens));
        }
        Ok(())
    }
    pub fn poison(&mut self, key: LaneKey) -> Result<()> {
        let lane = self.lanes.get_mut(&key.slot).context("unbound KV lane")?;
        ensure!(lane.key == key, "stale poison");
        lane.failed = true;
        Ok(())
    }
}
pub fn indices_reference(p: &KvPlan) -> Result<Vec<i32>> {
    ensure!(
        p.end <= MAX_CONTEXT
            && p.start.checked_add(p.tokens) == Some(p.end)
            && p.ratio == ratio(p.layer)?
            && p.compressed_owner == owner(p.layer)?,
        "invalid short KV plan"
    );
    let window = attention_state::window_plan(WINDOW, p.start, p.tokens)?;
    ensure!(
        p.compressed_rows == if p.ratio == 0 { 0 } else { p.end / p.ratio }
            && window.kv_rows == p.window_rows
            && p.window_topk == window.query_indices[0].len()
            && p.kv_rows == p.window_rows + p.compressed_rows
            && p.topk == p.window_topk + p.compressed_rows,
        "KV row/selection geometry changed"
    );
    let mut out = Vec::with_capacity(p.tokens * p.topk);
    for (q, row) in window.query_indices.iter().enumerate() {
        out.extend(row);
        let visible = if p.ratio == 0 {
            0
        } else if p.start == 0 {
            (q + 1) / p.ratio
        } else {
            p.end / p.ratio
        };
        for c in 0..p.compressed_rows {
            out.push(if c < visible {
                (p.window_rows + c) as i32
            } else {
                -1
            });
        }
    }
    ensure!(out.len() == p.tokens * p.topk, "KV index geometry mismatch");
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn key(boot: Uuid, slot: usize, generation: u64) -> LaneKey {
        LaneKey {
            boot,
            slot,
            generation,
        }
    }
    #[test]
    fn every_short_selection_matches_exact_official_cpu_ast_fixture() {
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(FIXTURE.as_bytes())),
            FIXTURE_SHA
        );
        let fixture: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 35);
        for c in cases {
            let get = |n: &str| c[n].as_u64().unwrap() as usize;
            let layer = get("layer");
            let start = get("start");
            let tokens = get("tokens");
            let r = ratio(layer).unwrap();
            let p = KvPlan {
                key: key(Uuid::nil(), 0, 1),
                layer,
                start,
                tokens,
                end: start + tokens,
                ratio: r,
                compressed_owner: owner(layer).unwrap(),
                owner_slot: owner(layer).unwrap().map(|o| source_slot(o).unwrap()),
                new_compressed_rows: 0,
                compressed_rows: get("compressed_rows"),
                window_rows: get("window_rows"),
                kv_rows: get("kv_rows"),
                window_topk: if start == 0 { tokens } else { 128 },
                topk: get("topk"),
                supplied_compressed_rows: None,
            };
            assert_eq!(
                indices_reference(&p).unwrap(),
                c["indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect::<Vec<_>>()
            );
        }
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let cfg = Dsv41Config::from_values(&v["hf_config"], &v["inference_config"]).unwrap();
        validate_config(&cfg).unwrap();
    }
    #[test]
    fn source_publication_generation_contiguity_and512_gate_are_real_contracts() {
        let boot = Uuid::from_u128(7);
        let k = key(boot, 0, 1);
        let mut c = KvCursor::new(boot, 2).unwrap();
        c.reset(k).unwrap();
        assert!(c.plan(3, k, 0, 3, None).is_err());
        let p = c.plan(2, k, 0, 3, Some(1)).unwrap();
        c.commit(&p).unwrap();
        let p = c.plan(3, k, 0, 3, None).unwrap();
        assert_eq!(p.compressed_owner, Some(2));
        assert_eq!(
            indices_reference(&p).unwrap(),
            [0, -1, -1, -1, 0, 1, -1, 3, 0, 1, 2, 3]
        );
        c.commit(&p).unwrap();
        assert!(c.plan(3, k, 3, 1, None).is_err());
        let p = c.plan(2, k, 3, 1, Some(1)).unwrap();
        c.commit(&p).unwrap();
        let p = c.plan(3, k, 3, 1, None).unwrap();
        c.commit(&p).unwrap();
        let old = k;
        let newer = key(boot, 0, 2);
        c.reset(newer).unwrap();
        assert!(c.plan(2, old, 4, 1, None).is_err());
        assert!(c.plan(0, newer, 0, 129, None).is_err());
        let mut p = c.plan(0, newer, 0, 128, None).unwrap();
        c.commit(&p).unwrap();
        for start in 128..512 {
            p = c.plan(0, newer, start, 1, None).unwrap();
            c.commit(&p).unwrap();
        }
        assert!(c.plan(0, newer, 512, 1, None).is_err());
        c.reset(key(boot, 1, 1)).unwrap();
        assert!(c.plan(0, key(boot, 1, 1), 0, 1, None).is_ok());
        assert!(c
            .plan(0, key(Uuid::from_u128(9), 0, 2), 0, 1, None)
            .is_err());
    }
    #[test]
    fn all_selected_indices_match_actual_cpu_topk_and_candidates_for_every_boundary() {
        use attention_state::{candidate_blocks, topk_positions, Selection};
        for layer in [0, 2, 8, 14, 20, 24, 39] {
            for (start, tokens) in [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 127),
                (0, 128),
                (1, 1),
                (127, 1),
                (128, 1),
                (129, 1),
                (510, 1),
                (511, 1),
            ] {
                let r = ratio(layer).unwrap();
                let rows = if r == 0 { 0 } else { (start + tokens) / r };
                let wr = if start == 0 { tokens } else { 128 };
                let p = KvPlan {
                    key: key(Uuid::nil(), 0, 1),
                    layer,
                    start,
                    tokens,
                    end: start + tokens,
                    ratio: r,
                    compressed_owner: owner(layer).unwrap(),
                    owner_slot: owner(layer).unwrap().map(|o| source_slot(o).unwrap()),
                    new_compressed_rows: 0,
                    compressed_rows: rows,
                    window_rows: wr,
                    kv_rows: wr + rows,
                    window_topk: wr,
                    topk: wr + rows,
                    supplied_compressed_rows: None,
                };
                let expected = indices_reference(&p).unwrap();
                if rows == 0 {
                    continue;
                }
                let scores = (0..tokens)
                    .map(|q| {
                        (0..rows)
                            .map(|i| ((i * 17 + q * 3) % 97) as f32 - 48.)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let visible = (0..tokens)
                    .map(|q| {
                        if start == 0 {
                            (q + 1) / r
                        } else {
                            (start + tokens) / r
                        }
                    })
                    .collect::<Vec<_>>();
                let Selection::Exact(mask) =
                    candidate_blocks(&scores, &visible, 2048, 8, None).unwrap()
                else {
                    panic!("all blocks cannot have a selection boundary tie")
                };
                let Selection::Exact(actual) =
                    topk_positions(&scores, &visible, 512, wr, Some(&mask), None).unwrap()
                else {
                    panic!("selecting every position has no boundary ambiguity")
                };
                for q in 0..tokens {
                    assert_eq!(&expected[q * p.topk + wr..(q + 1) * p.topk], actual[q]);
                }
            }
        }
    }
}
