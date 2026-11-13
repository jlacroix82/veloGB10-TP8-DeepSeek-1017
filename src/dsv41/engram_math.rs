//! Numerical contracts for native Engram row lookup and the post-projection gate.
//! Source: official 1017 model.py ParallelEngramEmbedding.forward/Engram.forward.
//! These CPU references neither implement wkv nor enable inference.
use super::numerics::{decode_e4m3_finite, decode_ue8m0, NORM_EPS};
use anyhow::{ensure, Context, Result};
use half::bf16;

pub const HEAD_DIM: usize = 256;
pub const HC: usize = 4;
pub const GATE_CLAMP: f32 = 1e-6;

/// Each global row has one owner. Native E4M3 values times their per-32 UE8M0
/// scale round to BF16 before the distributed sum. Nonowned entries are +0.
pub fn lookup_shard_native(
    weights: &[u8],
    scales: &[u8],
    indices: &[u64],
    first_row: u64,
    valid_rows: u64,
    total_rows: u64,
) -> Result<Vec<bf16>> {
    let end = first_row
        .checked_add(valid_rows)
        .context("Engram row extent overflow")?;
    ensure!(end <= total_rows, "Engram shard outside table");
    let rows = usize::try_from(valid_rows)?;
    ensure!(
        weights.len()
            == rows
                .checked_mul(HEAD_DIM)
                .context("Engram weight length overflow")?
            && scales.len()
                == rows
                    .checked_mul(HEAD_DIM / 32)
                    .context("Engram scale length overflow")?,
        "Engram native storage shape mismatch"
    );
    ensure!(
        indices.len() <= 128 * 24 && indices.iter().all(|&i| i < total_rows),
        "invalid/beyond-bounded Engram row IDs"
    );
    let mut output = vec![bf16::ZERO; indices.len() * HEAD_DIM];
    for (position, &row) in indices.iter().enumerate() {
        if row < first_row || row >= end {
            continue;
        }
        let local = usize::try_from(row - first_row)?;
        for column in 0..HEAD_DIM {
            let value = decode_e4m3_finite(weights[local * HEAD_DIM + column])?
                * decode_ue8m0(scales[local * (HEAD_DIM / 32) + column / 32])?;
            let rounded = bf16::from_f32(value);
            ensure!(
                value.is_finite() && rounded.is_finite(),
                "nonfinite Engram row value"
            );
            output[position * HEAD_DIM + column] = rounded;
        }
    }
    Ok(output)
}

/// h is [tokens,4,dim]; projected kv is [tokens,5*dim]. Key norms are
/// independent per HC copy. q/k weights multiply in FP32. The signed square
/// root occurs BEFORE sigmoid, then the shared value is added in FP32 once.
pub fn gated_add_bf16(
    h: &[bf16],
    kv: &[bf16],
    q_weight: &[bf16],
    k_weight: &[bf16],
    mask: &[bool],
    tokens: usize,
    dim: usize,
) -> Result<Vec<bf16>> {
    ensure!(
        (1..=128).contains(&tokens) && (32..=5120).contains(&dim) && dim % 32 == 0,
        "unsupported Engram gate shape"
    );
    ensure!(
        h.len() == tokens * HC * dim
            && kv.len() == tokens * (HC + 1) * dim
            && q_weight.len() == HC * dim
            && k_weight.len() == HC * dim
            && mask.len() == tokens,
        "Engram gate buffer shape mismatch"
    );
    ensure!(
        h.iter()
            .chain(kv)
            .chain(q_weight)
            .chain(k_weight)
            .all(|x| x.is_finite()),
        "nonfinite Engram gate input"
    );
    let mut output = h.to_vec();
    for token in 0..tokens {
        if !mask[token] {
            continue;
        }
        for copy in 0..HC {
            let base = (token * HC + copy) * dim;
            let key_base = (token * (HC + 1) + copy) * dim;
            let mut h2 = 0.0f32;
            let mut k2 = 0.0f32;
            let mut product = 0.0f32;
            for column in 0..dim {
                let stream = h[base + column].to_f32();
                let key = kv[key_base + column].to_f32();
                let weight =
                    q_weight[copy * dim + column].to_f32() * k_weight[copy * dim + column].to_f32();
                h2 += stream * stream;
                k2 += key * key;
                product += (stream * weight) * key;
            }
            ensure!(
                h2.is_finite() && k2.is_finite() && product.is_finite(),
                "nonfinite Engram reduction intermediate"
            );
            let rstd = 1.0 / (h2 / dim as f32 + NORM_EPS).sqrt()
                * (1.0 / (k2 / dim as f32 + NORM_EPS).sqrt());
            let dot = (product * rstd) * (dim as f32).powf(-0.5);
            ensure!(
                rstd.is_finite() && dot.is_finite(),
                "nonfinite normalized Engram dot"
            );
            let signed = dot.abs().max(GATE_CLAMP).sqrt().copysign(dot);
            let gate = 1.0 / (1.0 + (-signed).exp());
            ensure!(gate.is_finite(), "nonfinite Engram gate arithmetic");
            for column in 0..dim {
                let value = kv[(token * (HC + 1) + HC) * dim + column].to_f32();
                let result = bf16::from_f32(h[base + column].to_f32() + gate * value);
                ensure!(result.is_finite(), "nonfinite Engram gate output");
                output[base + column] = result;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_matches_verified_official_forward_cpu_goldens() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_engram_math_vectors.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"].as_array().unwrap().len(), 4);
        for case in fixture["cases"].as_array().unwrap() {
            let values = |key: &str| -> Vec<bf16> {
                case[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| bf16::from_bits(n.as_u64().unwrap() as u16))
                    .collect()
            };
            let expected = values("output_bits");
            let mask: Vec<_> = case["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            let actual = gated_add_bf16(
                &values("h_bits"),
                &values("kv_bits"),
                &values("q_weight_bits"),
                &values("k_weight_bits"),
                &mask,
                case["tokens"].as_u64().unwrap() as usize,
                case["dim"].as_u64().unwrap() as usize,
            )
            .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                // CPU FP32 sequential sums and Torch's vector reduction have
                // different orders. Permit one BF16 ULP, retaining every value.
                let ordered = |x: bf16| -> i32 {
                    let bits = x.to_bits() as i32;
                    if bits & 0x8000 != 0 {
                        0x8000 - (bits & 0x7fff)
                    } else {
                        0x8000 + bits
                    }
                };
                assert!(
                    (ordered(a) - ordered(b)).abs() <= 1,
                    "{} element{index}: {:?} != {:?}",
                    case["name"],
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn sharded_lookup_has_one_owner_rounds_once_and_keeps_large_global_row_offsets() {
        let start = (1u64 << 32) + 7;
        let mut weights = vec![0x38; 2 * HEAD_DIM]; // 1.0 and 2.0, different per-group scales
        weights[HEAD_DIM..].fill(0x40);
        let scales: Vec<_> = (0..16).map(|i| 124 + (i % 4) as u8).collect();
        let ids = [start - 1, start, start + 1, start + 2];
        let out = lookup_shard_native(&weights, &scales, &ids, start, 2, start + 3).unwrap();
        assert!(out[..HEAD_DIM]
            .iter()
            .chain(&out[3 * HEAD_DIM..])
            .all(|x| x.to_bits() == 0));
        for i in 0..HEAD_DIM {
            assert_eq!(
                out[HEAD_DIM + i],
                bf16::from_f32(2f32.powi((i / 32 % 4) as i32 - 3))
            );
            assert_eq!(
                out[2 * HEAD_DIM + i],
                bf16::from_f32(2f32.powi((i / 32 % 4) as i32 - 2))
            );
        }
        assert!(lookup_shard_native(&weights, &scales, &[start + 3], start, 2, start + 3).is_err());
        weights[0] = 0x7f;
        assert!(lookup_shard_native(&weights, &scales, &[start], start, 2, start + 3).is_err());
    }

    #[test]
    fn signed_dot_gate_is_copy_local_and_false_mask_is_exact_identity() {
        let dim = 32;
        let mut h = vec![bf16::ONE; 2 * HC * dim];
        h[dim..2 * dim].fill(bf16::from_f32(-1.0));
        let kv = vec![bf16::ONE; 2 * (HC + 1) * dim];
        let weight = vec![bf16::ONE; HC * dim];
        let out = gated_add_bf16(&h, &kv, &weight, &weight, &[true, false], 2, dim).unwrap();
        assert!(out[0].to_f32() > 1.9);
        assert!(out[dim].to_f32() < -0.9); // abs(dot) would incorrectly open this gate.
        assert_eq!(&out[HC * dim..], &h[HC * dim..]);
        assert!(gated_add_bf16(&h, &kv, &weight, &weight, &[], 2, dim).is_err());
        let huge = vec![bf16::from_f32(1e30); HC * dim];
        let huge_kv = vec![bf16::from_f32(1e30); (HC + 1) * dim];
        assert!(gated_add_bf16(&huge, &huge_kv, &weight, &weight, &[true], 1, dim).is_err());
    }
}
