//! Independent CPU reference for official1017 block64 sparse attention.
//! Input values are BF16; probabilities round to BF16 for P*V only. The online
//! denominator remains FP32 and the sink contributes no value vector. This is
//! not a model, indexer or pure full-precision-softmax replacement.
use anyhow::{ensure, Result};
use half::bf16;

pub const D: usize = 512;
pub const BLOCK: usize = 64;

#[allow(clippy::too_many_arguments)]
pub fn sparse(
    q: &[bf16],
    kv: &[bf16],
    sink: &[f32],
    indices: &[i32],
    b: usize,
    m: usize,
    heads: usize,
    n: usize,
    topk: usize,
) -> Result<Vec<bf16>> {
    ensure!(
        (1..=16).contains(&b)
            && (1..=128).contains(&m)
            && (1..=64).contains(&heads)
            && (1..=131200).contains(&n)
            && (1..=640).contains(&topk),
        "bounded attention reference shape"
    );
    ensure!(
        q.len() == b * m * heads * D
            && kv.len() == b * n * D
            && sink.len() == heads
            && indices.len() == b * m * topk,
        "attention reference buffer shape"
    );
    ensure!(
        q.iter().all(|v| v.is_finite()) && sink.iter().all(|v| v.is_finite()),
        "nonfinite attention reference input"
    );
    ensure!(
        indices
            .iter()
            .all(|&i| i >= -1 && (i < 0 || (i as usize) < n)),
        "attention index outside latent cache"
    );
    let scale = (D as f32).sqrt().recip();
    let mut output = vec![bf16::ZERO; q.len()];
    for lane in 0..b {
        for query in 0..m {
            let selected = &indices[(lane * m + query) * topk..(lane * m + query + 1) * topk];
            if selected.iter().all(|&i| i < 0) {
                continue;
            }
            for head in 0..heads {
                let qb = ((lane * m + query) * heads + head) * D;
                let mut maximum = -1e30f32;
                let mut denominator = 0.0f32;
                let mut acc = [0.0f32; D];
                for tile in selected.chunks(BLOCK) {
                    let mut scores = [f32::NEG_INFINITY; BLOCK];
                    for (position, &index) in tile.iter().enumerate() {
                        if index < 0 {
                            continue;
                        }
                        let kb = (lane * n + index as usize) * D;
                        ensure!(
                            kv[kb..kb + D].iter().all(|v| v.is_finite()),
                            "nonfinite consumed KV row"
                        );
                        // Independent FP64 dot diagnostic, rounded to FP32 before
                        // source scaling. The GPU oracle uses few-coordinate
                        // dyadic queries so all products/sums are exact in FP32.
                        let dot = (0..D)
                            .map(|d| q[qb + d].to_f32() as f64 * kv[kb + d].to_f32() as f64)
                            .sum::<f64>() as f32;
                        ensure!(dot.is_finite(), "attention dot overflow");
                        scores[position] = dot * scale;
                        ensure!(scores[position].is_finite(), "attention score overflow");
                    }
                    let next =
                        maximum.max(scores.iter().copied().fold(f32::NEG_INFINITY, f32::max));
                    let rescale = (maximum - next).exp();
                    let probabilities: Vec<_> = scores.iter().map(|s| (*s - next).exp()).collect();
                    let tile_sum = probabilities.iter().copied().sum::<f32>();
                    denominator = denominator * rescale + tile_sum;
                    for value in &mut acc {
                        *value *= rescale;
                    }
                    for (position, &index) in tile.iter().enumerate() {
                        if index < 0 {
                            continue;
                        }
                        let p = bf16::from_f32(probabilities[position]).to_f32();
                        let kb = (lane * n + index as usize) * D;
                        for d in 0..D {
                            acc[d] = p.mul_add(kv[kb + d].to_f32(), acc[d]);
                        }
                    }
                    maximum = next;
                }
                denominator += (sink[head] - maximum).exp();
                for d in 0..D {
                    let value = bf16::from_f32(acc[d] / denominator);
                    ensure!(value.is_finite(), "attention output nonfinite");
                    output[qb + d] = value;
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_uniform_sink_and_independent_batch_stride() {
        let b = 2;
        let m = 3;
        let heads = 8;
        let n = 129;
        let q = vec![bf16::ZERO; b * m * heads * D];
        let kv: Vec<_> = (0..b * n * D)
            .map(|i| bf16::from_f32((i / D % n) as f32 + (i / (n * D)) as f32 * 16.0))
            .collect();
        let indices: Vec<_> = (0..b * m).flat_map(|_| 0..n as i32).collect();
        let out = sparse(&q, &kv, &[0.0; 8], &indices, b, m, heads, n, n).unwrap();
        for lane in 0..b {
            let expected = bf16::from_f32((8256.0 + lane as f32 * 16.0 * 129.0) / 130.0);
            assert!(out[lane * m * heads * D..(lane + 1) * m * heads * D]
                .iter()
                .all(|&v| v == expected));
        }
    }
    #[test]
    fn first_block_masked_and_all_masked_do_not_create_nan() {
        let q = vec![bf16::ZERO; D];
        let kv = vec![bf16::ONE; D];
        let mut indices = vec![-1; 65];
        indices[64] = 0;
        let out = sparse(&q, &kv, &[0.0], &indices, 1, 1, 1, 1, 65).unwrap();
        assert!(out.iter().all(|&v| v == bf16::from_f32(0.5)));
        indices.fill(-1);
        assert!(sparse(&q, &kv, &[0.0], &indices, 1, 1, 1, 1, 65)
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0));
        indices[64] = 1;
        assert!(sparse(&q, &kv, &[0.0], &indices, 1, 1, 1, 1, 65).is_err());
    }
}
