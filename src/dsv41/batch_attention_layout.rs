//! PRIVATE semantic tile layouts for true packed lanes. Logical source slots
//! are separate from the physical chronological KV snapshot and batch stride.
//! This CPU contract does not qualify a new packed attention CUDA kernel.
use super::chunk_plan::{ChunkTicket, LayerChunk};
use anyhow::{ensure, Result};
use serde::Serialize;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AttentionPhase {
    /// Full original prompt extent, even if this packed segment is much shorter.
    Prefill {
        full_prompt_tokens: usize,
    },
    Decode,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LaneAttentionLayout {
    pub ticket: ChunkTicket,
    pub phase: AttentionPhase,
    pub ratio: usize,
    pub logical_window_columns: usize,
    pub logical_compressed_columns: usize,
    pub physical_window_rows: usize,
    pub physical_compressed_rows: usize,
    pub physical_window_global_base: usize,
}
impl LaneAttentionLayout {
    pub fn new(plan: &LayerChunk, phase: AttentionPhase) -> Result<Self> {
        let t = &plan.ticket;
        ensure!(
            t.tokens > 0 && t.tokens <= 128 && t.end == t.start + t.tokens && t.end <= 8192,
            "bounded exact native segment"
        );
        let physical_compressed_rows = plan.compressed.as_ref().map_or(0, |p| p.stored_rows);
        ensure!(
            matches!(plan.ratio, 0 | 1 | 2) && (plan.ratio != 0 || physical_compressed_rows == 0),
            "native compression geometry"
        );
        let (window, compressed) = match phase {
            AttentionPhase::Prefill { full_prompt_tokens } => {
                ensure!(
                    (1..=8192).contains(&full_prompt_tokens) && t.end <= full_prompt_tokens,
                    "prefill must bind the complete original prompt extent"
                );
                (
                    full_prompt_tokens.min(128),
                    if plan.ratio == 0 {
                        0
                    } else {
                        (full_prompt_tokens / plan.ratio).min(512)
                    },
                )
            }
            AttentionPhase::Decode => {
                ensure!(
                    t.start > 0 && t.tokens == 1,
                    "decode is one token after a checked prefix"
                );
                (128, physical_compressed_rows.min(512))
            }
        };
        ensure!(
            plan.window.snapshot_rows == t.start.min(127) + t.tokens
                && plan.window.snapshot_global_base == t.start - t.start.min(127),
            "chronological source window mapping"
        );
        Ok(Self {
            ticket: t.clone(),
            phase,
            ratio: plan.ratio,
            logical_window_columns: window,
            logical_compressed_columns: compressed,
            physical_window_rows: plan.window.snapshot_rows,
            physical_compressed_rows,
            physical_window_global_base: plan.window.snapshot_global_base,
        })
    }
    pub fn logical_topk(&self) -> usize {
        self.logical_window_columns + self.logical_compressed_columns
    }
    pub fn physical_kv_rows(&self) -> usize {
        self.physical_window_rows + self.physical_compressed_rows
    }
    /// Preserves source64-key tile membership. A later packed-row kernel may
    /// add row-stride padding ONLY after these complete logical slots, and must
    /// still execute each row with its own logical_topk (extra tiles unqualified).
    pub fn window_indices(&self) -> Result<Vec<i32>> {
        let mut result = Vec::with_capacity(self.ticket.tokens * self.logical_window_columns);
        for row in 0..self.ticket.tokens {
            let p = self.ticket.start + row;
            for j in 0..self.logical_window_columns {
                let global = match self.phase {
                    AttentionPhase::Prefill { .. } => {
                        let g = (p + 1).saturating_sub(128) + j;
                        (g <= p).then_some(g)
                    }
                    AttentionPhase::Decode => {
                        // Exact unfilled ring placement, including leading−1.
                        if p < 127 {
                            let slot = (p + 1 + j) % 128;
                            (slot <= p).then_some(slot)
                        } else {
                            Some(p + 1 - 128 + j)
                        }
                    }
                };
                if let Some(global) = global {
                    ensure!(
                        global >= self.physical_window_global_base,
                        "logical key precedes immutable snapshot"
                    );
                    let local = global - self.physical_window_global_base;
                    ensure!(
                        local < self.physical_window_rows,
                        "logical key exceeds immutable snapshot"
                    );
                    result.push(local as i32);
                } else {
                    result.push(-1)
                }
            }
        }
        Ok(result)
    }
    /// Exact all-visible selection while every physically available compressed
    /// key fits512. Full-prompt future columns stay in their source logical slots
    /// as−1; this is not a learned topk substitute once stored keys exceed512.
    pub fn all_selected_indices(&self) -> Result<Vec<i32>> {
        ensure!(
            self.physical_compressed_rows <= 512,
            "learned causal selection is required beyond512 stored compressed rows"
        );
        let windows = self.window_indices()?;
        let width = self.logical_window_columns;
        let mut result = Vec::with_capacity(self.ticket.tokens * self.logical_topk());
        for q in 0..self.ticket.tokens {
            result.extend_from_slice(&windows[q * width..(q + 1) * width]);
            let visible = if self.ratio == 0 {
                0
            } else {
                (self.ticket.start + q + 1) / self.ratio
            };
            ensure!(
                visible <= self.physical_compressed_rows,
                "causal compressed key exceeds published snapshot"
            );
            for j in 0..self.logical_compressed_columns {
                result.push(if j < visible {
                    (self.physical_window_rows + j) as i32
                } else {
                    -1
                });
            }
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::super::{chunk_plan::ChunkCursor, config::Dsv41Config, scheduler::LaneKey};
    use super::*;
    use uuid::Uuid;
    fn plan(start: usize, tokens: usize, layer: usize) -> LayerChunk {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let c = Dsv41Config::from_values(&f["hf_config"], &f["inference_config"]).unwrap();
        let key = LaneKey {
            boot: Uuid::from_u128(1641),
            slot: 0,
            generation: 1,
        };
        let mut cur = ChunkCursor::new(&c, key.boot, 1).unwrap();
        cur.reset(key).unwrap();
        let mut at = 0;
        while at < start {
            let n = (start - at).min(128);
            let t = cur.begin(key, at, n).unwrap();
            for l in 0..40 {
                let p = cur.layer(&t, l).unwrap();
                cur.commit_layer(&p).unwrap();
            }
            cur.commit(&t).unwrap();
            at += n;
        }
        let t = cur.begin(key, start, tokens).unwrap();
        for l in 0..layer {
            let p = cur.layer(&t, l).unwrap();
            cur.commit_layer(&p).unwrap();
        }
        cur.layer(&t, layer).unwrap()
    }
    #[test]
    fn positive_single_token_prefill_and_decode_remain_explicitly_distinct() {
        let p = plan(16, 1, 2);
        let a = LaneAttentionLayout::new(
            &p,
            AttentionPhase::Prefill {
                full_prompt_tokens: 75,
            },
        )
        .unwrap();
        let b = LaneAttentionLayout::new(&p, AttentionPhase::Decode).unwrap();
        assert_eq!(
            (
                a.logical_window_columns,
                a.logical_compressed_columns,
                a.logical_topk()
            ),
            (75, 37, 112)
        );
        assert_eq!(
            (
                b.logical_window_columns,
                b.logical_compressed_columns,
                b.logical_topk()
            ),
            (128, 8, 136)
        );
        assert_eq!(a.window_indices().unwrap()[0], 0);
        assert_eq!(b.window_indices().unwrap()[0], -1);
        assert_eq!(a.all_selected_indices().unwrap()[75], 17);
        assert_eq!(b.all_selected_indices().unwrap()[128], 17);
        assert_ne!(a.window_indices().unwrap(), b.window_indices().unwrap());
    }
    #[test]
    fn every_official_full_prefill_ast_slice_preserves_tiles_and_physical_key_values() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_batch_prefill_layout_vectors.json"
        ))
        .unwrap();
        assert_eq!(
            f["reference_source_sha256"],
            "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
        );
        for c in f["cases"].as_array().unwrap() {
            let s = c["start"].as_u64().unwrap() as usize;
            let t = c["tokens"].as_u64().unwrap() as usize;
            let full = c["full_prompt_tokens"].as_u64().unwrap() as usize;
            let layer = c["layer"].as_u64().unwrap() as usize;
            let p = plan(s, t, layer);
            let layout = LaneAttentionLayout::new(
                &p,
                AttentionPhase::Prefill {
                    full_prompt_tokens: full,
                },
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(layout.window_indices().unwrap()).unwrap(),
                c["window_snapshot_indices"]
            );
            assert_eq!(
                layout.logical_topk(),
                c["logical_topk"].as_u64().unwrap() as usize
            );
            if c["all_selected_snapshot_indices"].is_null() {
                assert!(layout.all_selected_indices().is_err());
            } else {
                assert_eq!(
                    serde_json::to_value(layout.all_selected_indices().unwrap()).unwrap(),
                    c["all_selected_snapshot_indices"]
                );
            }
        }
    }
    #[test]
    fn false_full_prompt_extent_decode_geometry_and_unimplemented_selection_refuse() {
        let p = plan(64, 11, 20);
        assert!(LaneAttentionLayout::new(
            &p,
            AttentionPhase::Prefill {
                full_prompt_tokens: 74
            }
        )
        .is_err());
        assert!(LaneAttentionLayout::new(&p, AttentionPhase::Decode).is_err());
        let p = plan(1024, 1, 20);
        let a = LaneAttentionLayout::new(&p, AttentionPhase::Decode).unwrap();
        assert!(a.all_selected_indices().is_err());
        assert_eq!(a.logical_topk(), 640);
    }
}
