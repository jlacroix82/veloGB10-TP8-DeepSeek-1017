//! Official1017 text-only MoE routing and pre-w2 SwiGLU reference primitives.
//! Dense gate/w1/w3/w2 linears, checkpoint experts and distributed accumulation
//! are intentionally outside this component. SharedExpert is Expert(weights=None).
use anyhow::{ensure, Result};
use half::bf16;
pub const EXPERTS: usize = 384;
pub const TOPK: usize = 6;
pub const ROUTE_SCALE: f32 = 1.5;
pub const ROUTE_EPS: f32 = 1e-20;
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const MAX_ROWS: usize = 128;
pub const CUDA_REFERENCE_SHA: &str =
    "6ff4aa72845bb6f9f3bb02c5f778a73128dbbef48f1328b7b7bef521586456c0";
pub const CUDA_REFERENCE: &str = include_str!("../../tests/fixtures/dsv41_moe_cuda_reference.json");

/// Actual official Gate.forward on TorchCUDA is the deployment reference.
/// The original vectorized CPU fixture remains unchanged and deliberately
/// differs at the EPS-dominant -100 subnormal canary.
pub fn official_cuda_reference() -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    ensure!(
        format!("{:x}", Sha256::digest(CUDA_REFERENCE.as_bytes())) == CUDA_REFERENCE_SHA,
        "official CUDA routing reference changed"
    );
    let value: serde_json::Value = serde_json::from_str(CUDA_REFERENCE)?;
    ensure!(
        value["kind"] == "official1017_Gate_TorchCUDA_goldens"
            && value["capture_complete"] == true
            && value["model_execution"] == false
            && value["fixture_sha256"]
                == "329867a7de2b8a55424805158bca969a9b306187338acd1f41746aeda2af8ac3"
            && value["route_cases"].as_array().map(Vec::len) == Some(6),
        "official CUDA reference scope/input identity differs"
    );
    Ok(value)
}
#[derive(Clone, Debug)]
pub struct Routing {
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}
#[derive(Clone, Debug)]
pub struct RouteWitness {
    pub boundary_ties: usize,
    pub max_weight_error: f32,
    pub minimum_selection_margin: f64,
}
fn shape(logits: &[f32], bias: &[f32], rows: usize) -> Result<()> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows) && logits.len() == rows * EXPERTS && bias.len() == EXPERTS,
        "1017 route geometry384/top6/rows1..128"
    );
    ensure!(
        logits.iter().chain(bias).all(|v| v.is_finite()),
        "nonfinite gate logits or correction bias"
    );
    Ok(())
}
/// F.softplus(beta1, threshold20), then sqrt. No bias enters this score.
pub fn unbiased_score(logit: f32) -> Result<f32> {
    ensure!(logit.is_finite(), "nonfinite gate logit");
    let value = if logit > 20.0 {
        logit
    } else {
        logit.exp().ln_1p()
    }
    .sqrt();
    ensure!(value.is_finite(), "gate score overflow");
    Ok(value)
}
/// Explicit deterministic tie policy: descending biased score, lowest ID on
/// equality. PyTorch does not specify tie IDs; validate_witness accepts legitimate alternatives.
pub fn route(logits: &[f32], bias: &[f32], rows: usize) -> Result<Routing> {
    shape(logits, bias, rows)?;
    let mut out = Routing {
        indices: Vec::with_capacity(rows * TOPK),
        weights: Vec::with_capacity(rows * TOPK),
    };
    for row in 0..rows {
        let scores = logits[row * EXPERTS..(row + 1) * EXPERTS]
            .iter()
            .map(|v| unbiased_score(*v))
            .collect::<Result<Vec<_>>>()?;
        let selected = scores
            .iter()
            .zip(bias)
            .map(|(s, b)| s + b)
            .collect::<Vec<_>>();
        ensure!(
            selected.iter().all(|v| v.is_finite()),
            "biased selection score overflow"
        );
        let mut order = (0..EXPERTS).collect::<Vec<_>>();
        order.sort_by(|a, b| selected[*b].total_cmp(&selected[*a]).then(a.cmp(b)));
        let mut sum = 0.0_f32;
        for &id in &order[..TOPK] {
            sum += scores[id];
        }
        ensure!(sum.is_finite(), "routing denominator overflow");
        for &id in &order[..TOPK] {
            out.indices.push(id as u32);
            out.weights
                .push((scores[id] / (sum + ROUTE_EPS)) * ROUTE_SCALE);
        }
    }
    Ok(out)
}
/// Checks every selected expert against every unselected expert and recomputes
/// its unbiased normalized weight. Boundary ties admit any valid top-k set;
/// this is not a claim that a different tied set yields identical model outputs.
pub fn validate_witness(
    logits: &[f32],
    bias: &[f32],
    rows: usize,
    r: &Routing,
) -> Result<RouteWitness> {
    shape(logits, bias, rows)?;
    ensure!(
        r.indices.len() == rows * TOPK && r.weights.len() == rows * TOPK,
        "route witness extent"
    );
    let mut out = RouteWitness {
        boundary_ties: 0,
        max_weight_error: 0.0,
        minimum_selection_margin: f64::INFINITY,
    };
    for row in 0..rows {
        let scores = logits[row * EXPERTS..(row + 1) * EXPERTS]
            .iter()
            .map(|v| unbiased_score(*v))
            .collect::<Result<Vec<_>>>()?;
        let selected = scores
            .iter()
            .zip(bias)
            .map(|(s, b)| s + b)
            .collect::<Vec<_>>();
        ensure!(selected.iter().all(|v| v.is_finite()), "selection overflow");
        let mut seen = [false; EXPERTS];
        let ids = &r.indices[row * TOPK..(row + 1) * TOPK];
        let mut denominator = 0.0_f32;
        for &id in ids {
            let id = id as usize;
            ensure!(id < EXPERTS && !seen[id], "duplicate/out-of-range expert");
            seen[id] = true;
            denominator += scores[id];
        }
        let mut threshold = f32::INFINITY;
        let mut previous = f32::INFINITY;
        for (rank, &id) in ids.iter().enumerate() {
            let id = id as usize;
            ensure!(selected[id] <= previous, "route not sorted by biased score");
            previous = selected[id];
            threshold = threshold.min(selected[id]);
            let expected = (scores[id] / (denominator + ROUTE_EPS)) * ROUTE_SCALE;
            let weight = r.weights[row * TOPK + rank];
            let error = (weight - expected).abs();
            ensure!(
                weight.is_finite() && weight >= 0.0 && error <= 2e-6 + 2e-6 * expected.abs(),
                "routing weight includes bias/wrong normalization/order"
            );
            out.max_weight_error = out.max_weight_error.max(error);
        }
        let other = (0..EXPERTS)
            .filter(|id| !seen[*id])
            .map(|id| selected[id])
            .fold(f32::NEG_INFINITY, f32::max);
        ensure!(
            threshold >= other,
            "selected set is not a valid top-k witness"
        );
        if threshold == other {
            out.boundary_ties += 1;
        }
        out.minimum_selection_margin = out
            .minimum_selection_margin
            .min(threshold as f64 - other as f64);
    }
    Ok(out)
}
/// BF16 projections widen to F32; gate clamps only ABOVE10, up clamps BOTH
/// signs. Route multiplication occurs in F32 before the one BF16 cast into w2.
/// None is the unweighted shared expert. This does not execute w2.
pub fn swiglu(
    gate: &[bf16],
    up: &[bf16],
    weights: Option<&[f32]>,
    rows: usize,
    columns: usize,
) -> Result<Vec<bf16>> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows)
            && (1..=2304).contains(&columns)
            && gate.len() == rows * columns
            && up.len() == gate.len(),
        "SwiGLU component geometry"
    );
    ensure!(
        gate.iter().chain(up).all(|v| v.is_finite()),
        "nonfinite SwiGLU projection"
    );
    if let Some(w) = weights {
        ensure!(
            w.len() == rows
                && w.iter()
                    .all(|v| v.is_finite() && *v >= 0.0 && *v <= ROUTE_SCALE),
            "invalid1017 route weights"
        );
    }
    let mut out = Vec::with_capacity(gate.len());
    for i in 0..gate.len() {
        let g = gate[i].to_f32().min(SWIGLU_LIMIT);
        let u = up[i].to_f32().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let silu = g / (1.0 + (-g).exp());
        let mut value = silu * u;
        if let Some(w) = weights {
            value *= w[i / columns];
        }
        let rounded = bf16::from_f32(value);
        ensure!(value.is_finite() && rounded.is_finite(), "SwiGLU overflow");
        out.push(rounded);
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn floats(v: &Value, k: &str) -> Vec<f32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn bfloat(v: &Value, k: &str) -> Vec<bf16> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|v| bf16::from_bits(v.as_u64().unwrap() as u16))
            .collect()
    }
    #[test]
    fn actual_official_cuda_gate_unambiguous_ids_and_valid_tie_witnesses() {
        let f: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_moe_vectors.json"))
                .unwrap();
        let cuda = official_cuda_reference().unwrap();
        for c in f["route_cases"].as_array().unwrap() {
            let rows = c["rows"].as_u64().unwrap() as usize;
            let logits = floats(c, "logits");
            let bias = floats(c, "bias");
            let r = route(&logits, &bias, rows).unwrap();
            validate_witness(&logits, &bias, rows, &r).unwrap();
            let reference = cuda["route_cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["name"] == c["name"])
                .unwrap();
            assert_eq!(reference["rows"], c["rows"]);
            let official = Routing {
                indices: reference["indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect(),
                weights: floats(reference, "weights"),
            };
            validate_witness(&logits, &bias, rows, &official).unwrap();
            if c["tie_policy"] == "unambiguous exact expert IDs" {
                assert_eq!(r.indices, official.indices);
                for (a, b) in r.weights.iter().zip(official.weights) {
                    assert!((a - b).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn vector_cpu_subnormal_case_remains_an_explicit_mismatch_canary() {
        let f: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_moe_vectors.json"))
                .unwrap();
        let cuda = official_cuda_reference().unwrap();
        let mut known_mismatches = 0;
        for c in f["route_cases"].as_array().unwrap() {
            let rows = c["rows"].as_u64().unwrap() as usize;
            let original = Routing {
                indices: c["indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect(),
                weights: floats(c, "weights"),
            };
            let result =
                validate_witness(&floats(c, "logits"), &floats(c, "bias"), rows, &original);
            if c["name"] == "tiny_score_eps" {
                assert!(
                    result.is_err(),
                    "known CPU vector approximation silently accepted"
                );
                let measured = cuda["route_cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|v| v["name"] == c["name"])
                    .unwrap();
                assert_eq!(
                    measured["weight_bits"],
                    serde_json::json!(vec![1020660094_u32; 6])
                );
                assert!((original.weights[0] - floats(measured, "weights")[0]).abs() > 0.0004);
                known_mismatches += 1;
            } else {
                result.unwrap();
            }
        }
        assert_eq!(known_mismatches, 1);
    }
    #[test]
    fn exact_expert_ast_clamps_shared_and_weight_before_bf16() {
        let f: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_moe_vectors.json"))
                .unwrap();
        for c in f["swiglu_cases"].as_array().unwrap() {
            let rows = c["rows"].as_u64().unwrap() as usize;
            let cols = c["columns"].as_u64().unwrap() as usize;
            let w = (!c["weights"].is_null()).then(|| floats(c, "weights"));
            let actual = swiglu(
                &bfloat(c, "gate_bits"),
                &bfloat(c, "up_bits"),
                w.as_deref(),
                rows,
                cols,
            )
            .unwrap();
            let expected = bfloat(c, "output_bits");
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(a.to_bits().abs_diff(b.to_bits()) <= 1, "{a} vs {b}");
            }
        }
    }
    #[test]
    fn bias_is_selection_only_and_invalid_inputs_never_pass() {
        let l = (0..384)
            .map(|i| i as f32 * 0.125 - 24.0)
            .collect::<Vec<_>>();
        let mut b = vec![0.0; 384];
        b[0] = 100.0;
        let r = route(&l, &b, 1).unwrap();
        assert_eq!(r.indices[0], 0);
        validate_witness(&l, &b, 1, &r).unwrap();
        let mut wrong = r.clone();
        wrong.weights[0] += 0.1;
        assert!(validate_witness(&l, &b, 1, &wrong).is_err());
        wrong = r.clone();
        wrong.indices[1] = wrong.indices[0];
        assert!(validate_witness(&l, &b, 1, &wrong).is_err());
        assert!(route(&vec![f32::NAN; 384], &b, 1).is_err());
        assert!(route(&l, &vec![f32::INFINITY; 384], 1).is_err());
        assert!(swiglu(&[bf16::NAN], &[bf16::ONE], None, 1, 1).is_err());
        assert!(swiglu(&[bf16::ONE], &[bf16::ONE], Some(&[2.0]), 1, 1).is_err());
    }
}
