//! Official1017 compression state and local index-score reference contracts.
//! Supplied projections, norm weights, RoPE and cacheQAT stay separate components.
//! No dense projection, network reduction, candidate/topk or serving implementation.
use super::{numerics, scheduler::LaneKey};
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::Serialize;
use std::collections::BTreeMap;
use uuid::Uuid;
pub const MODEL_REFERENCE_SHA256: &str =
    "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d";
pub const MAX_LANES: usize = 16;
pub const MAX_PREFILL: usize = 128;
pub const MAX_CONTEXT: usize = 131072;
pub const LOCAL_INDEX_HEADS: usize = 4;
pub const GLOBAL_INDEX_HEADS: usize = 32;
pub const INDEX_HEAD_DIM: usize = 128;
pub const INDEX_WEIGHT_SCALE: f32 = 1.0 / 64.0;
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum Ratio {
    One,
    Two,
}
impl Ratio {
    pub fn value(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompressionPlan {
    pub lane: LaneKey,
    pub start: usize,
    pub tokens: usize,
    pub ratio: Ratio,
    pub output_rows: usize,
    pub positions: Vec<usize>,
    pub partial_before: usize,
    pub partial_after: usize,
}
#[derive(Clone)]
struct Progress {
    key: LaneKey,
    position: usize,
    initialized: bool,
    poisoned: bool,
}
pub struct CompressionCursor {
    boot: Uuid,
    max_lanes: usize,
    ratio: Ratio,
    lanes: BTreeMap<usize, Progress>,
}
impl CompressionCursor {
    pub fn new(boot: Uuid, max_lanes: usize, ratio: Ratio) -> Result<Self> {
        ensure!(
            (1..=MAX_LANES).contains(&max_lanes),
            "compression lane bound"
        );
        Ok(Self {
            boot,
            max_lanes,
            ratio,
            lanes: BTreeMap::new(),
        })
    }
    pub fn reset(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.boot && key.slot < self.max_lanes && key.generation > 0,
            "invalid compression lane key"
        );
        ensure!(
            self.lanes
                .get(&key.slot)
                .is_none_or(|p| key.generation > p.key.generation),
            "compression reset generation must increase"
        );
        self.lanes.insert(
            key.slot,
            Progress {
                key,
                position: 0,
                initialized: false,
                poisoned: false,
            },
        );
        Ok(())
    }
    pub fn plan(&self, key: LaneKey, start: usize, tokens: usize) -> Result<CompressionPlan> {
        let p = self
            .lanes
            .get(&key.slot)
            .context("unbound compression lane")?;
        ensure!(
            p.key == key && !p.poisoned && start == p.position,
            "stale/poisoned/noncontiguous compression lane"
        );
        ensure!(
            (!p.initialized && start == 0 && (1..=MAX_PREFILL).contains(&tokens))
                || (p.initialized && start > 0 && tokens == 1),
            "initialprefill<=128 thenone-token append required"
        );
        let end = start
            .checked_add(tokens)
            .context("compression position overflow")?;
        ensure!(end <= MAX_CONTEXT, "compression context bound");
        let ratio = self.ratio.value();
        let first = start / ratio;
        let last = end / ratio;
        let positions = (first..last).map(|row| row * ratio).collect::<Vec<_>>();
        Ok(CompressionPlan {
            lane: key,
            start,
            tokens,
            ratio: self.ratio,
            output_rows: last - first,
            positions,
            partial_before: start % ratio,
            partial_after: end % ratio,
        })
    }
    pub fn commit(&mut self, plan: &CompressionPlan) -> Result<()> {
        ensure!(
            self.plan(plan.lane, plan.start, plan.tokens)? == *plan,
            "forged/stale compression plan"
        );
        let p = self.lanes.get_mut(&plan.lane.slot).unwrap();
        p.position = plan.start + plan.tokens;
        p.initialized = true;
        Ok(())
    }
    pub fn poison(&mut self, key: LaneKey) -> Result<()> {
        let p = self
            .lanes
            .get_mut(&key.slot)
            .context("unbound compression lane")?;
        ensure!(p.key == key, "stale compression poison");
        p.poisoned = true;
        Ok(())
    }
    pub fn matches(&self, key: LaneKey) -> bool {
        self.lanes
            .get(&key.slot)
            .is_some_and(|p| p.key == key && !p.poisoned)
    }
}
#[derive(Debug, Clone)]
pub struct PreparedLatent {
    pub plan: CompressionPlan,
    pub pooled_f32: Option<Vec<f32>>,
    pub norm_input_bf16: Vec<bf16>,
}
impl PreparedLatent {
    /// Output stays unrotated. The indexer's WK/norm/RoPE/QAT must consume this
    /// normalized BF16 latent before outer compressed-KV rotation overwrites a copy.
    pub fn normalized_unrotated(&self, norm_weight_f32: &[f32], dim: usize) -> Result<Vec<bf16>> {
        if self.plan.output_rows == 0 {
            ensure!(self.norm_input_bf16.is_empty(), "emptylatent extent");
            return Ok(Vec::new());
        }
        numerics::rms_norm_bf16(
            &self.norm_input_bf16,
            norm_weight_f32,
            self.plan.output_rows,
            dim,
        )
    }
}
#[derive(Clone)]
struct Tail {
    kv: Vec<f32>,
    score: Vec<f32>,
}
pub struct CompressionState {
    pub cursor: CompressionCursor,
    dim: usize,
    tails: BTreeMap<usize, Tail>,
}
fn dim(dim: usize) -> Result<()> {
    ensure!(
        (1..=512).contains(&dim),
        "compressiondim1..512 boundedcomponent"
    );
    Ok(())
}
pub fn pool_pair(k0: f32, k1: f32, s0: f32, s1: f32) -> Result<f32> {
    ensure!(
        [k0, k1, s0, s1].iter().all(|v| v.is_finite()),
        "nonfinite compressor projection"
    );
    let maximum = s0.max(s1);
    let a = (s0 - maximum).exp();
    let b = (s1 - maximum).exp();
    let denominator = a + b;
    let value = k0 * (a / denominator) + k1 * (b / denominator);
    ensure!(value.is_finite(), "compressor F32pool overflow");
    Ok(value)
}
impl CompressionState {
    pub fn new(boot: Uuid, max_lanes: usize, ratio: Ratio, width: usize) -> Result<Self> {
        dim(width)?;
        Ok(Self {
            cursor: CompressionCursor::new(boot, max_lanes, ratio)?,
            dim: width,
            tails: BTreeMap::new(),
        })
    }
    pub fn reset(&mut self, key: LaneKey) -> Result<()> {
        self.cursor.reset(key)?;
        self.tails.insert(
            key.slot,
            Tail {
                kv: vec![0.; 2 * self.dim],
                score: vec![f32::NEG_INFINITY; 2 * self.dim],
            },
        );
        Ok(())
    }
    pub fn tail(&self, key: LaneKey) -> Result<(&[f32], &[f32])> {
        ensure!(self.cursor.matches(key), "stale compression tail");
        let t = &self.tails[&key.slot];
        Ok((&t.kv, &t.score))
    }
    pub fn push_ratio1(
        &mut self,
        key: LaneKey,
        start: usize,
        tokens: usize,
        projected: &[bf16],
    ) -> Result<PreparedLatent> {
        let plan = self.cursor.plan(key, start, tokens)?;
        ensure!(
            plan.ratio == Ratio::One
                && projected.len() == tokens * self.dim
                && projected.iter().all(|v| v.is_finite()),
            "ratio1 projection shape/dtype/finite contract"
        );
        let result = PreparedLatent {
            plan: plan.clone(),
            pooled_f32: None,
            norm_input_bf16: projected.to_vec(),
        };
        self.cursor.commit(&plan)?;
        Ok(result)
    }
    pub fn push_ratio2(
        &mut self,
        key: LaneKey,
        start: usize,
        tokens: usize,
        kv: &[f32],
        score: &[f32],
    ) -> Result<PreparedLatent> {
        let plan = self.cursor.plan(key, start, tokens)?;
        ensure!(
            plan.ratio == Ratio::Two
                && kv.len() == tokens * self.dim
                && score.len() == kv.len()
                && kv.iter().chain(score).all(|v| v.is_finite()),
            "ratio2 F32projection shape/finite contract"
        );
        let mut tail = self.tails[&key.slot].clone();
        let mut pooled = Vec::with_capacity(plan.output_rows * self.dim);
        if start == 0 {
            for pair in 0..tokens / 2 {
                for c in 0..self.dim {
                    let a = (pair * 2) * self.dim + c;
                    let b = a + self.dim;
                    pooled.push(pool_pair(kv[a], kv[b], score[a], score[b])?);
                }
            }
            if tokens % 2 != 0 {
                tail.kv[..self.dim]
                    .copy_from_slice(&kv[(tokens - 1) * self.dim..tokens * self.dim]);
                tail.score[..self.dim]
                    .copy_from_slice(&score[(tokens - 1) * self.dim..tokens * self.dim]);
            }
        } else {
            let slot = start % 2;
            tail.kv[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(kv);
            tail.score[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(score);
            if slot == 1 {
                for c in 0..self.dim {
                    pooled.push(pool_pair(
                        tail.kv[c],
                        tail.kv[self.dim + c],
                        tail.score[c],
                        tail.score[self.dim + c],
                    )?);
                }
            }
        }
        let norm_input: Vec<_> = pooled.iter().copied().map(bf16::from_f32).collect();
        ensure!(
            norm_input.iter().all(|v| v.is_finite()),
            "compressor BF16 norminput overflow"
        );
        self.cursor.commit(&plan)?;
        self.tails.insert(key.slot, tail);
        Ok(PreparedLatent {
            plan,
            pooled_f32: Some(pooled),
            norm_input_bf16: norm_input,
        })
    }
}
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum IndexNetworkPolicy {
    /// Source localBF16 output and sourceBF16 network dtype. Network executor external.
    SourceBf16LocalAndNetwork,
    /// Same sourceBF16 local math; widening toF32 changes distributed rounding.
    SourceBf16LocalPromotedF32NetworkAlternative,
}
#[derive(Debug, Clone, Copy, Serialize)]
pub struct IndexShape {
    pub batch: usize,
    pub queries: usize,
    pub keys: usize,
}
impl IndexShape {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=16).contains(&self.batch)
                && (1..=128).contains(&self.queries)
                && (1..=131072).contains(&self.keys),
            "index component shape bound"
        );
        ensure!(
            self.batch
                .checked_mul(self.queries)
                .and_then(|n| n.checked_mul(self.keys))
                .is_some_and(|n| n <= 4 * 1024 * 1024),
            "index score element bound"
        );
        ensure!(
            self.batch
                .checked_mul(self.keys)
                .and_then(|n| n.checked_mul(INDEX_HEAD_DIM))
                .is_some_and(|n| n <= 64 * 1024 * 1024),
            "index key element bound"
        );
        Ok(())
    }
    pub fn scores(self) -> usize {
        self.batch * self.queries * self.keys
    }
}
#[derive(Debug, Clone)]
pub struct IndexScores {
    pub scaled_weights: Vec<bf16>,
    pub dot: Vec<bf16>,
    pub relu: Vec<bf16>,
    pub product: Vec<bf16>,
    pub local_bf16: Vec<bf16>,
    pub promoted_f32: Vec<f32>,
}
/// Exact dtype boundaries of official Indexer.forward. The dot's reference
/// accumulation is FP64 diagnostic thenFP32; the synthetic oracle uses dyadic
/// data with exactFP32dots. General GEMM association still needs model qualification.
pub fn index_scores(
    q: &[bf16],
    key: &[bf16],
    projected_weights: &[bf16],
    shape: IndexShape,
) -> Result<IndexScores> {
    shape.validate()?;
    let rows = shape.batch * shape.queries;
    ensure!(
        q.len() == rows * 4 * 128
            && key.len() == shape.batch * shape.keys * 128
            && projected_weights.len() == rows * 4
            && q.iter()
                .chain(key)
                .chain(projected_weights)
                .all(|v| v.is_finite()),
        "index BF16stageinput contract"
    );
    let scaled_weights: Vec<_> = projected_weights
        .iter()
        .map(|w| bf16::from_f32(w.to_f32() * INDEX_WEIGHT_SCALE))
        .collect();
    let mut out = IndexScores {
        scaled_weights,
        dot: vec![bf16::ZERO; shape.scores() * 4],
        relu: vec![bf16::ZERO; shape.scores() * 4],
        product: vec![bf16::ZERO; shape.scores() * 4],
        local_bf16: vec![bf16::ZERO; shape.scores()],
        promoted_f32: vec![0.; shape.scores()],
    };
    for b in 0..shape.batch {
        for query in 0..shape.queries {
            for k in 0..shape.keys {
                let mut sum = 0_f32;
                for head in 0..4 {
                    let qi = ((b * shape.queries + query) * 4 + head) * 128;
                    let ki = (b * shape.keys + k) * 128;
                    let raw = (0..128)
                        .map(|d| q[qi + d].to_f32() as f64 * key[ki + d].to_f32() as f64)
                        .sum::<f64>() as f32;
                    let dot = bf16::from_f32(raw);
                    ensure!(raw.is_finite() && dot.is_finite(), "index dot overflow");
                    let relu = if dot.to_f32() < 0. { bf16::ZERO } else { dot };
                    let product = bf16::from_f32(
                        relu.to_f32()
                            * out.scaled_weights[(b * shape.queries + query) * 4 + head].to_f32(),
                    );
                    ensure!(product.is_finite(), "index weighted product overflow");
                    let index = ((b * shape.queries + query) * 4 + head) * shape.keys + k;
                    out.dot[index] = dot;
                    out.relu[index] = relu;
                    out.product[index] = product;
                    sum += product.to_f32();
                }
                let local = bf16::from_f32(sum);
                ensure!(local.is_finite(), "index BF16headsum overflow");
                let i = (b * shape.queries + query) * shape.keys + k;
                out.local_bf16[i] = local;
                out.promoted_f32[i] = local.to_f32();
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_compression_vectors.json"
        ))
        .unwrap()
    }
    fn b(v: &Value, key: &str) -> Vec<bf16> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| bf16::from_bits(v.as_u64().unwrap() as u16))
            .collect()
    }
    fn f(v: &Value, key: &str) -> Vec<f32> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect()
    }
    fn bmatch(a: &[bf16], v: &Value, key: &str) {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b(v, key).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn official_compressor_ast_pool_tail_cast_and_norm_goldens() {
        let fixture = fixture();
        assert_eq!(fixture["model_source_sha256"], MODEL_REFERENCE_SHA256);
        for case in fixture["compression_cases"].as_array().unwrap() {
            let ratio = if case["ratio"] == 1 {
                Ratio::One
            } else {
                Ratio::Two
            };
            let width = case["dim"].as_u64().unwrap() as usize;
            let boot = Uuid::new_v4();
            let key = LaneKey {
                boot,
                slot: 0,
                generation: 1,
            };
            let mut state = CompressionState::new(boot, 2, ratio, width).unwrap();
            state.reset(key).unwrap();
            let weight = b(case, "norm_weight_bf16_bits")
                .into_iter()
                .map(bf16::to_f32)
                .collect::<Vec<_>>();
            for event in case["events"].as_array().unwrap() {
                let start = event["start"].as_u64().unwrap() as usize;
                let tokens = event["tokens"].as_u64().unwrap() as usize;
                let out = if ratio == Ratio::One {
                    state.push_ratio1(key, start, tokens, &b(event, "projected_kv_bits"))
                } else {
                    state.push_ratio2(
                        key,
                        start,
                        tokens,
                        &f(event, "projected_kv_bits"),
                        &f(event, "projected_score_bits"),
                    )
                }
                .unwrap();
                assert_eq!(
                    out.plan.positions,
                    serde_json::from_value::<Vec<usize>>(event["positions"].clone()).unwrap()
                );
                bmatch(&out.norm_input_bf16, event, "norm_input_bf16_bits");
                let normalized = out.normalized_unrotated(&weight, width).unwrap();
                for (a, wanted) in normalized
                    .iter()
                    .zip(b(event, "normalized_unrotated_bf16_bits"))
                {
                    assert!(
                        a.is_finite() && a.to_bits().abs_diff(wanted.to_bits()) <= 1,
                        "{} norm {:?}",
                        case["name"],
                        out.plan
                    );
                }
                if ratio == Ratio::Two {
                    if event["output_rows"].as_u64().unwrap() > 0 {
                        for (a, wanted) in out
                            .pooled_f32
                            .as_ref()
                            .unwrap()
                            .iter()
                            .zip(f(event, "pooled_f32_bits"))
                        {
                            assert!(
                                (a - wanted).abs() <= 1e-5 + 1e-6 * wanted.abs(),
                                "F32pool reference differs"
                            );
                        }
                    }
                    let (tk, ts) = state.tail(key).unwrap();
                    assert_eq!(
                        tk.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        f(event, "tail_kv_f32_bits")
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        ts.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        f(event, "tail_score_f32_bits")
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn official_index_ast_dtype_castpoints_are_not_silently_widened() {
        let fixture = fixture();
        for c in fixture["index_cases"].as_array().unwrap() {
            let shape = IndexShape {
                batch: c["batch"].as_u64().unwrap() as usize,
                queries: c["queries"].as_u64().unwrap() as usize,
                keys: c["keys"].as_u64().unwrap() as usize,
            };
            let out = index_scores(
                &b(c, "projected_q_bf16_bits"),
                &b(c, "key_bf16_bits"),
                &b(c, "projected_weight_bf16_bits"),
                shape,
            )
            .unwrap();
            bmatch(&out.scaled_weights, c, "scaled_weight_bf16_bits");
            bmatch(&out.dot, c, "dot_bf16_bits");
            bmatch(&out.relu, c, "relu_bf16_bits");
            bmatch(&out.product, c, "product_bf16_bits");
            bmatch(&out.local_bf16, c, "local_sum_bf16_bits");
            assert_eq!(
                out.promoted_f32
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                f(c, "source_local_promoted_f32_bits")
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            assert!(c["negative_unrounded_local_mismatches"].as_u64().unwrap() > 0);
            assert!(c["stage_dtypes"]
                .as_object()
                .unwrap()
                .values()
                .all(|v| v == "torch.bfloat16"));
        }
    }
    #[test]
    fn lanes_generations_partial_failures_and_maxprefill_are_bounded() {
        let boot = Uuid::new_v4();
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
        let mut state = CompressionState::new(boot, 2, Ratio::Two, 8).unwrap();
        state.reset(a).unwrap();
        state.reset(b).unwrap();
        assert_eq!(
            state
                .push_ratio2(a, 0, 1, &[1.; 8], &[0.; 8])
                .unwrap()
                .plan
                .output_rows,
            0
        );
        assert_eq!(
            state
                .push_ratio2(b, 0, 3, &[5.; 24], &[0.; 24])
                .unwrap()
                .plan
                .positions,
            [0]
        );
        let out = state.push_ratio2(a, 1, 1, &[3.; 8], &[0.; 8]).unwrap();
        assert_eq!(out.plan.positions, [0]);
        assert!(out.norm_input_bf16.iter().all(|v| v.to_f32() == 2.));
        assert!(state
            .push_ratio2(a, 2, 1, &[f32::NAN; 8], &[0.; 8])
            .is_err());
        assert!(state.push_ratio2(a, 2, 2, &[1.; 16], &[0.; 16]).is_err());
        state.push_ratio2(a, 2, 1, &[7.; 8], &[0.; 8]).unwrap();
        assert!(state.reset(a).is_err());
        let new = LaneKey { generation: 2, ..a };
        state.reset(new).unwrap();
        assert!(state.push_ratio2(a, 3, 1, &[1.; 8], &[0.; 8]).is_err());
        assert_eq!(
            state
                .push_ratio2(new, 0, 128, &vec![1.; 128 * 8], &vec![0.; 128 * 8])
                .unwrap()
                .plan
                .output_rows,
            64
        );
        let mut cursor = CompressionCursor::new(boot, 1, Ratio::One).unwrap();
        cursor.reset(a).unwrap();
        assert!(cursor.plan(a, 0, 129).is_err());
        let mut plan = cursor.plan(a, 0, 1).unwrap();
        plan.positions[0] = 42;
        assert!(cursor.commit(&plan).is_err());
        cursor.poison(a).unwrap();
        assert!(cursor.plan(a, 0, 1).is_err());
    }
}
