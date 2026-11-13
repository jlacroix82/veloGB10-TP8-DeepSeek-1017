//! Official1017 four-stream hyper-connection reference math, not a serving backend.
//!
//! The coefficients computed by a sublayer collapse the *next* sublayer's
//! input. In particular the final head uses the last FFN's pre, with no learned
//! hc_head projection. Native model parameters remain F32; residuals/pre/post
//! boundaries round to BF16 exactly where model.py does.
use anyhow::{ensure, Context, Result};
use half::bf16;

pub const COPIES: usize = 4;
/// Bounded initial-prefill rows, independent of decode concurrency.
pub const MAX_TOKENS: usize = 128;
pub const MIXES: usize = 24;
pub const SINKHORN_ITERS: usize = 20;
pub const HC_EPS: f32 = 1e-6;
pub const NORM_EPS: f32 = 1e-20;

#[derive(Clone, Debug)]
pub struct Coefficients {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    /// [token, source, destination], not [destination, source].
    pub comb: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct Projection<'a> {
    /// [24, 4*dim], row-major F32, unquantized.
    pub weight: &'a [f32],
    pub scale: &'a [f32],
    pub base: &'a [f32],
}

pub fn validate_shape(tokens: usize, dim: usize) -> Result<()> {
    ensure!(
        (1..=MAX_TOKENS).contains(&tokens) && (8..=5120).contains(&dim) && dim % 8 == 0,
        "bounded mHC component requires1..128 tokens and dim8..5120 divisible by8"
    );
    Ok(())
}
fn finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}
fn bfinite(values: &[bf16]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Initial embedding copies are collapsed with the official one-hot [1,0,0,0].
pub fn initial_pre(tokens: usize) -> Result<Vec<f32>> {
    ensure!((1..=MAX_TOKENS).contains(&tokens), "mHC token bound");
    Ok((0..tokens * COPIES)
        .map(|i| if i % COPIES == 0 { 1.0 } else { 0.0 })
        .collect())
}

/// RMS over the *whole flattened4*dim* stream, then F32 projection. This is a
/// conservative CPU reference; serial F32 reduction is not a GEMM performance implementation.
pub fn project(
    h: &[bf16],
    projection: Projection<'_>,
    tokens: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    validate_shape(tokens, dim)?;
    let hidden = COPIES * dim;
    ensure!(
        h.len() == tokens * hidden
            && projection.weight.len() == MIXES * hidden
            && projection.scale.len() == 3
            && projection.base.len() == MIXES,
        "mHC projection geometry"
    );
    ensure!(
        bfinite(h)
            && finite(projection.weight)
            && finite(projection.scale)
            && finite(projection.base),
        "nonfinite mHC projection input"
    );
    let mut out = vec![0.0; tokens * MIXES];
    for token in 0..tokens {
        let x = &h[token * hidden..(token + 1) * hidden];
        let mut square = 0.0_f32;
        for v in x {
            let f = v.to_f32();
            square += f * f;
        }
        ensure!(square.is_finite(), "mHC RMS overflow");
        let inverse = 1.0 / (square / hidden as f32 + NORM_EPS).sqrt();
        for row in 0..MIXES {
            let mut dot = 0.0_f32;
            for (col, value) in x.iter().enumerate() {
                dot += value.to_f32() * projection.weight[row * hidden + col];
            }
            let value = dot * inverse;
            ensure!(value.is_finite(), "mHC projection overflow");
            out[token * MIXES + row] = value;
        }
    }
    Ok(out)
}

/// Literal1017 TileLang algorithm: stable row softmax+eps, column normalization,
/// then19 row+column rounds (20 Sinkhorn iterations total).
pub fn split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    tokens: usize,
) -> Result<Coefficients> {
    ensure!(
        (1..=MAX_TOKENS).contains(&tokens)
            && mixes.len() == tokens * MIXES
            && scale.len() == 3
            && base.len() == MIXES,
        "mHC split geometry"
    );
    ensure!(
        finite(mixes) && finite(scale) && finite(base),
        "nonfinite mHC coefficients input"
    );
    let mut result = Coefficients {
        pre: vec![0.0; tokens * COPIES],
        post: vec![0.0; tokens * COPIES],
        comb: vec![0.0; tokens * COPIES * COPIES],
    };
    for token in 0..tokens {
        let x = &mixes[token * MIXES..(token + 1) * MIXES];
        for copy in 0..COPIES {
            let pre_logit = x[copy] * scale[0] + base[copy];
            let post_logit = x[copy + COPIES] * scale[1] + base[copy + COPIES];
            ensure!(
                pre_logit.is_finite() && post_logit.is_finite(),
                "mHC pre/post logit overflow"
            );
            result.pre[token * COPIES + copy] = 1.0 / (1.0 + (-pre_logit).exp()) + HC_EPS;
            result.post[token * COPIES + copy] = 2.0 / (1.0 + (-post_logit).exp());
        }
        let comb = &mut result.comb[token * 16..(token + 1) * 16];
        for j in 0..16 {
            comb[j] = x[j + 8] * scale[2] + base[j + 8];
        }
        ensure!(finite(comb), "mHC combination logit overflow");
        for row in 0..COPIES {
            let start = row * COPIES;
            let max = comb[start..start + COPIES]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for col in 0..COPIES {
                let i = start + col;
                comb[i] = (comb[i] - max).exp();
                sum += comb[i];
            }
            for col in 0..COPIES {
                comb[start + col] = comb[start + col] / sum + HC_EPS;
            }
        }
        for iteration in 0..SINKHORN_ITERS {
            if iteration > 0 {
                for row in 0..COPIES {
                    let mut sum = 0.0_f32;
                    for col in 0..COPIES {
                        sum += comb[row * COPIES + col];
                    }
                    for col in 0..COPIES {
                        comb[row * COPIES + col] /= sum + HC_EPS;
                    }
                }
            }
            for col in 0..COPIES {
                let mut sum = 0.0_f32;
                for row in 0..COPIES {
                    sum += comb[row * COPIES + col];
                }
                for row in 0..COPIES {
                    comb[row * COPIES + col] /= sum + HC_EPS;
                }
            }
        }
        ensure!(finite(comb), "mHC Sinkhorn produced nonfinite coefficient");
    }
    Ok(result)
}

pub fn coefficients(
    h: &[bf16],
    projection: Projection<'_>,
    tokens: usize,
    dim: usize,
) -> Result<Coefficients> {
    split_sinkhorn(
        &project(h, projection, tokens, dim)?,
        projection.scale,
        projection.base,
        tokens,
    )
}

/// Four F32 products/sum and a single BF16 rounding, as Block.hc_pre.
pub fn collapse(h: &[bf16], pre: &[f32], tokens: usize, dim: usize) -> Result<Vec<bf16>> {
    validate_shape(tokens, dim)?;
    ensure!(
        h.len() == tokens * COPIES * dim && pre.len() == tokens * COPIES,
        "mHC collapse geometry"
    );
    ensure!(bfinite(h) && finite(pre), "nonfinite mHC collapse input");
    let mut out = vec![bf16::ZERO; tokens * dim];
    for token in 0..tokens {
        for col in 0..dim {
            let mut sum = 0.0_f32;
            for copy in 0..COPIES {
                sum += pre[token * COPIES + copy] * h[(token * COPIES + copy) * dim + col].to_f32();
            }
            let rounded = bf16::from_f32(sum);
            ensure!(
                sum.is_finite() && rounded.is_finite(),
                "mHC collapse overflow"
            );
            out[token * dim + col] = rounded;
        }
    }
    Ok(out)
}

/// comb[source,destination] mixes residual sources before addition of the shared
/// sublayer output. Return is BF16, not an intermediate F32 residual.
pub fn expand(
    x: &[bf16],
    residual: &[bf16],
    post: &[f32],
    comb: &[f32],
    tokens: usize,
    dim: usize,
) -> Result<Vec<bf16>> {
    validate_shape(tokens, dim)?;
    ensure!(
        x.len() == tokens * dim
            && residual.len() == tokens * COPIES * dim
            && post.len() == tokens * COPIES
            && comb.len() == tokens * 16,
        "mHC expansion geometry"
    );
    ensure!(
        bfinite(x) && bfinite(residual) && finite(post) && finite(comb),
        "nonfinite mHC expansion input"
    );
    let mut out = vec![bf16::ZERO; residual.len()];
    for token in 0..tokens {
        for destination in 0..COPIES {
            for col in 0..dim {
                let mut sum = 0.0_f32;
                for source in 0..COPIES {
                    sum += comb[token * 16 + source * COPIES + destination]
                        * residual[(token * COPIES + source) * dim + col].to_f32();
                }
                let value =
                    post[token * COPIES + destination] * x[token * dim + col].to_f32() + sum;
                let rounded = bf16::from_f32(value);
                ensure!(
                    value.is_finite() && rounded.is_finite(),
                    "mHC expansion overflow"
                );
                out[(token * COPIES + destination) * dim + col] = rounded;
            }
        }
    }
    Ok(out)
}

/// Establishes ordering only. Caller callbacks must perform their own norm and
/// actual attention/FFN; neither is supplied or claimed by this component.
#[allow(clippy::too_many_arguments)]
pub fn block_with_sublayers<A, F>(
    h: &[bf16],
    incoming_pre: &[f32],
    attention: Projection<'_>,
    ffn: Projection<'_>,
    tokens: usize,
    dim: usize,
    mut attention_fn: A,
    mut ffn_fn: F,
) -> Result<(Vec<bf16>, Vec<f32>)>
where
    A: FnMut(&[bf16]) -> Result<Vec<bf16>>,
    F: FnMut(&[bf16]) -> Result<Vec<bf16>>,
{
    let attn_mix = coefficients(h, attention, tokens, dim)?;
    let attn_input = collapse(h, incoming_pre, tokens, dim)?;
    let attn_output = attention_fn(&attn_input).context("mHC attention callback")?;
    let after_attention = expand(&attn_output, h, &attn_mix.post, &attn_mix.comb, tokens, dim)?;
    let ffn_mix = coefficients(&after_attention, ffn, tokens, dim)?;
    let ffn_input = collapse(&after_attention, &attn_mix.pre, tokens, dim)?;
    let ffn_output = ffn_fn(&ffn_input).context("mHC FFN callback")?;
    let after_ffn = expand(
        &ffn_output,
        &after_attention,
        &ffn_mix.post,
        &ffn_mix.comb,
        tokens,
        dim,
    )?;
    Ok((after_ffn, ffn_mix.pre))
}

/// Synthetic deterministic sparse F32 parameters used only by the independent
/// oracle. Returns row-major24*(4*dim), eight accumulated entries per row.
pub fn oracle_weight_recipe(dim: usize, seed: usize) -> Vec<f32> {
    let mut result = vec![0.0; MIXES * COPIES * dim];
    for output in 0..MIXES {
        for j in 0..8 {
            let col = (output * 11 + j * 17 + seed) % (COPIES * dim);
            result[output * COPIES * dim + col] +=
                ((output * 3 + j + seed) % 9) as f32 * 0.03125 - 0.125;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn values(v: &Value, key: &str) -> Vec<f32> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }
    fn bvalues(v: &Value, key: &str) -> Vec<bf16> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| bf16::from_bits(x.as_u64().unwrap() as u16))
            .collect()
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() <= 2e-5 + b.abs() * 2e-5, "{a} != {b}");
        }
    }
    fn bclose(a: &[bf16], b: &[bf16]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                a.is_finite() && b.is_finite() && a.to_bits().abs_diff(b.to_bits()) <= 1,
                "{a} != {b}"
            );
        }
    }
    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/dsv41_hc_vectors.json")).unwrap()
    }
    #[test]
    fn official_ast_coefficients_pre_post_and_hidden5120() {
        let fixture = fixture();
        assert_eq!(fixture["cases"].as_array().unwrap().len(), 4);
        for case in fixture["cases"].as_array().unwrap() {
            let tokens = case["tokens"].as_u64().unwrap() as usize;
            let dim = case["dim"].as_u64().unwrap() as usize;
            let h = bvalues(case, "h_bits");
            let x = bvalues(case, "sublayer_bits");
            let weight =
                oracle_weight_recipe(dim, case["weight_recipe_seed"].as_u64().unwrap() as usize);
            let scale = values(case, "scale");
            let base = values(case, "base");
            let mix = coefficients(
                &h,
                Projection {
                    weight: &weight,
                    scale: &scale,
                    base: &base,
                },
                tokens,
                dim,
            )
            .unwrap();
            close(&mix.pre, &values(case, "pre"));
            close(&mix.post, &values(case, "post"));
            close(&mix.comb, &values(case, "comb"));
            assert_eq!(initial_pre(tokens).unwrap(), values(case, "initial_pre"));
            assert_eq!(
                collapse(&h, &initial_pre(tokens).unwrap(), tokens, dim).unwrap(),
                bvalues(case, "collapsed_initial_bits")
            );
            bclose(
                &collapse(&h, &mix.pre, tokens, dim).unwrap(),
                &bvalues(case, "collapsed_current_bits"),
            );
            bclose(
                &expand(&x, &h, &mix.post, &mix.comb, tokens, dim).unwrap(),
                &bvalues(case, "expanded_bits"),
            );
        }
    }
    #[test]
    fn previous_pre_flows_through_two_layers_and_final_head() {
        let fixture = fixture();
        let c = &fixture["chain"];
        let dim = 8;
        let mut h = bvalues(c, "h_bits");
        let mut pre = initial_pre(1).unwrap();
        for layer in c["layers"].as_array().unwrap() {
            let seed = layer["weight_recipe_seed"].as_u64().unwrap() as usize;
            let aw = oracle_weight_recipe(dim, seed);
            let fw = oracle_weight_recipe(dim, seed + 5);
            let base = |off: usize| {
                (0..24)
                    .map(|i| ((i + off) % 9) as f32 * 0.125 - 0.5)
                    .collect::<Vec<_>>()
            };
            let ab = base(seed);
            let fb = base(seed + 5);
            let scale = [0.75, -0.5, 1.25];
            let attn = Projection {
                weight: &aw,
                scale: &scale,
                base: &ab,
            };
            let ffn = Projection {
                weight: &fw,
                scale: &scale,
                base: &fb,
            };
            let old_wrong =
                collapse(&h, &coefficients(&h, attn, 1, dim).unwrap().pre, 1, dim).unwrap();
            assert_ne!(old_wrong, collapse(&h, &pre, 1, dim).unwrap());
            let (next, np) = block_with_sublayers(
                &h,
                &pre,
                attn,
                ffn,
                1,
                dim,
                |x| {
                    Ok(x.iter()
                        .map(|x| bf16::from_f32(x.to_f32() * 0.5 + 0.0625))
                        .collect())
                },
                |x| {
                    Ok(x.iter()
                        .map(|x| bf16::from_f32(x.to_f32() * -0.25 - 0.125))
                        .collect())
                },
            )
            .unwrap();
            bclose(&next, &bvalues(layer, "h_bits"));
            close(&np, &values(layer, "next_pre"));
            bclose(
                &collapse(&next, &np, 1, dim).unwrap(),
                &bvalues(layer, "final_collapsed_bits"),
            );
            h = next;
            pre = np;
        }
    }
    #[test]
    fn bounded_prefill_rows_preserve_independent_token_math() {
        let dim = 8;
        let weight = oracle_weight_recipe(dim, 17);
        let scale = [0.75, -0.5, 1.25];
        let base = [0.0; MIXES];
        let projection = Projection {
            weight: &weight,
            scale: &scale,
            base: &base,
        };
        for tokens in [32, 75, MAX_TOKENS] {
            validate_shape(tokens, dim).unwrap();
            let h = (0..tokens * COPIES * dim)
                .map(|i| bf16::from_f32(((i * 7 + 3) % 31) as f32 * 0.125 - 1.875))
                .collect::<Vec<_>>();
            let coeffs = coefficients(&h, projection, tokens, dim).unwrap();
            let initial = initial_pre(tokens).unwrap();
            let first_copies = collapse(&h, &initial, tokens, dim).unwrap();
            for row in 0..tokens {
                assert_eq!(
                    &first_copies[row * dim..(row + 1) * dim],
                    &h[row * COPIES * dim..row * COPIES * dim + dim]
                );
                let single = coefficients(
                    &h[row * COPIES * dim..(row + 1) * COPIES * dim],
                    projection,
                    1,
                    dim,
                )
                .unwrap();
                assert_eq!(&coeffs.pre[row * COPIES..(row + 1) * COPIES], single.pre);
                assert_eq!(&coeffs.post[row * COPIES..(row + 1) * COPIES], single.post);
                assert_eq!(
                    &coeffs.comb[row * COPIES * COPIES..(row + 1) * COPIES * COPIES],
                    single.comb
                );
            }
        }
        for tokens in [0, MAX_TOKENS + 1, usize::MAX] {
            assert!(validate_shape(tokens, dim).is_err());
            assert!(initial_pre(tokens).is_err());
            assert!(split_sinkhorn(&[], &scale, &base, tokens).is_err());
        }
    }
    #[test]
    fn combination_orientation_initial_one_hot_and_finite_rejection() {
        let dim = 8;
        let h = (0..32)
            .map(|i| bf16::from_f32((i / 8 + 1) as f32))
            .collect::<Vec<_>>();
        assert_eq!(
            collapse(&h, &initial_pre(1).unwrap(), 1, dim).unwrap(),
            vec![bf16::ONE; dim]
        );
        let mut comb = vec![0.0; 16];
        for source in 0..4 {
            comb[source * 4 + (source + 1) % 4] = 1.0;
        }
        let out = expand(&vec![bf16::ZERO; dim], &h, &[0.0; 4], &comb, 1, dim).unwrap();
        assert_eq!(
            (0..4).map(|i| out[i * dim].to_f32()).collect::<Vec<_>>(),
            vec![4.0, 1.0, 2.0, 3.0]
        );
        assert!(split_sinkhorn(&[f32::NAN; 24], &[1.0; 3], &[0.0; 24], 1).is_err());
        assert!(collapse(&h, &[f32::INFINITY; 4], 1, dim).is_err());
        assert!(project(
            &vec![bf16::MAX; 32],
            Projection {
                weight: &[0.0; 768],
                scale: &[1.0; 3],
                base: &[0.0; 24]
            },
            1,
            dim
        )
        .is_err());
        assert!(initial_pre(129).is_err());
    }
}
