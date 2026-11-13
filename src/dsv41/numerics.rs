//! CPU arithmetic reference for official DeepSeek-V4.1-Flash-1017.
//!
//! Rules are from verified inference/kernel.py (group32 W8A8 activation quantization) and
//! inference/model.py RMSNorm. These helpers do not read/requantize checkpoint weights,
//! compile GPU kernels or enable model serving. GPU parity requires a separate oracle.
//!
//! Reference implementation copyright (c) 2023 DeepSeek, MIT:
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//! and associated documentation files (the "Software"), to deal in the Software without restriction,
//! including without limitation the rights to use, copy, modify, merge, publish, distribute,
//! sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//! The above copyright notice and this permission notice shall be included in all copies or
//! substantial portions of the Software.
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//! BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//! NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::Serialize;
use std::sync::OnceLock;

pub const GROUP: usize = 32;
pub const NORM_EPS: f32 = 1e-20;
pub const FP8_MAX: f32 = 448.0;
pub const KERNEL_REFERENCE_SHA256: &str =
    "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455";
pub const MODEL_REFERENCE_SHA256: &str =
    "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d";
pub const CONVERT_REFERENCE_SHA256: &str =
    "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76";

/// E4M3FN has no infinities. Codes 0x7f/0xff are NaN; the existing finite decoder is
/// exact on the remaining codes, including signed zero and FP8 subnormals.
pub fn decode_e4m3_finite(code: u8) -> Result<f32> {
    ensure!(
        code & 0x7f != 0x7f,
        "NaN E4M3 code is not finite reference data"
    );
    Ok(crate::quant::e4m3_to_f32(code))
}

/// UE8M0 byte b means 2^(b-127), including b=0's exact FP32 subnormal 2^-127.
/// Byte 255 is NaN. Do not interpret the scale byte as a numeric integer or treat 0 as zero.
pub fn decode_ue8m0(code: u8) -> Result<f32> {
    ensure!(code != 255, "NaN UE8M0 scale");
    Ok(f32::from_bits(if code == 0 {
        1 << 22
    } else {
        (code as u32) << 23
    }))
}

fn e4m3_values() -> &'static [f32; 127] {
    static VALUES: OnceLock<[f32; 127]> = OnceLock::new();
    VALUES.get_or_init(|| {
        let mut values = [0.0; 127];
        for (code, value) in values.iter_mut().enumerate() {
            *value = crate::quant::e4m3_to_f32(code as u8);
        }
        values
    })
}

/// Source act_quant explicitly clamps to +/-448 before FP8 conversion. Ties round to the
/// even E4M3 code and negative zero retains its sign. The older general encoder rounds all
/// ties toward the lower magnitude and therefore is deliberately not used here.
pub fn encode_e4m3_rne_saturating(value: f32) -> Result<u8> {
    ensure!(
        value.is_finite(),
        "nonfinite activation cannot be FP8-reference quantized"
    );
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs().min(FP8_MAX);
    let values = e4m3_values();
    let upper = values.partition_point(|&x| x < magnitude);
    let code = if upper == 0 {
        0
    } else if upper == values.len() {
        values.len() - 1
    } else {
        let lower = upper - 1;
        let dl = magnitude - values[lower];
        let du = values[upper] - magnitude;
        if dl < du || (dl == du && lower % 2 == 0) {
            lower
        } else {
            upper
        }
    };
    Ok(sign | code as u8)
}

fn matrix_elements(rows: usize, columns: usize) -> Result<usize> {
    ensure!(
        rows > 0 && columns > 0,
        "reference matrix dimensions must be positive"
    );
    rows.checked_mul(columns)
        .context("reference matrix shape overflow")
}

/// kernel.py:74-80: absmax floor1e-4, FP32 multiply by1/448, then the exact IEEE-bit
/// ceil(log2) and pow2 operation. The activation floor keeps this path in normal FP32 range.
fn activation_scale(max_abs: f32) -> Result<(u8, f32)> {
    ensure!(
        max_abs.is_finite() && max_abs >= 0.0,
        "invalid activation absmax"
    );
    let divided = max_abs.max(1e-4_f32) * (1.0_f32 / FP8_MAX);
    ensure!(
        divided.is_finite() && divided.is_normal() && divided > 0.0,
        "activation scale intermediate outside source normal-FP32 domain"
    );
    let bits = divided.to_bits();
    let exponent = ((bits >> 23) & 255) as i32 - 127 + i32::from(bits & 0x7fffff != 0);
    ensure!(
        (-126..=127).contains(&exponent),
        "activation pow2 scale exponent out of range"
    );
    let code = u8::try_from(exponent + 127)?;
    let scale = f32::from_bits((code as u32) << 23);
    ensure!(
        scale == decode_ue8m0(code)?,
        "activation scale does not round-trip UE8M0"
    );
    Ok((code, scale))
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActivationQuant {
    pub rows: usize,
    pub columns: usize,
    pub group_size: usize,
    pub e4m3_values: Vec<u8>,
    pub ue8m0_scales: Vec<u8>,
}

/// Quantize caller-provided BF16 activations, not weights. Source kernels require K%32==0.
/// Native output bytes are checked against CPU torch.float8_e4m3fn/e8m0fnu goldens.
pub fn quantize_activations_bf16(
    input: &[bf16],
    rows: usize,
    columns: usize,
) -> Result<ActivationQuant> {
    let count = matrix_elements(rows, columns)?;
    ensure!(
        columns % GROUP == 0 && input.len() == count,
        "invalid group32 activation shape"
    );
    ensure!(
        input.iter().all(|x| x.is_finite()),
        "nonfinite BF16 activation"
    );
    let groups = columns / GROUP;
    let mut output = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(rows * groups);
    for row in 0..rows {
        for group in 0..groups {
            let start = row * columns + group * GROUP;
            let block = &input[start..start + GROUP];
            let max_abs = block
                .iter()
                .map(|x| x.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let (code, scale) = activation_scale(max_abs)?;
            scales.push(code);
            for x in block {
                output.push(encode_e4m3_rne_saturating(x.to_f32() / scale)?);
            }
        }
    }
    Ok(ActivationQuant {
        rows,
        columns,
        group_size: GROUP,
        e4m3_values: output,
        ue8m0_scales: scales,
    })
}

pub fn dequantize_activations_f32(input: &ActivationQuant) -> Result<Vec<f32>> {
    let count = matrix_elements(input.rows, input.columns)?;
    ensure!(
        input.group_size == GROUP
            && input.columns % GROUP == 0
            && input.e4m3_values.len() == count
            && input.ue8m0_scales.len() == count / GROUP,
        "invalid group32 activation metadata"
    );
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let value = decode_e4m3_finite(input.e4m3_values[index])?
            * decode_ue8m0(input.ue8m0_scales[index / GROUP])?;
        ensure!(
            value.is_finite(),
            "FP32 overflow in activation dequantization"
        );
        result.push(value);
    }
    Ok(result)
}

/// Native checkpoint I8 packs low nibble=even K and high nibble=odd K. One UE8M0
/// scale covers32 K entries of each row. This preserves stored values; it does no quantization.
/// The verified convert.py FP4_TABLE canonicalizes code8's zero to +0.0, so that one
/// entry deliberately differs from the older generic decoder's negative-zero bit pattern.
pub fn dequantize_fp4_native(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    let count = matrix_elements(rows, columns)?;
    ensure!(
        columns % GROUP == 0 && packed.len() == count / 2 && scales.len() == count / GROUP,
        "invalid native packed-FP4/per32 scale shape"
    );
    let mut result = Vec::with_capacity(count);
    for row in 0..rows {
        for column in 0..columns {
            let byte = packed[row * (columns / 2) + column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            let scale = decode_ue8m0(scales[row * (columns / GROUP) + column / GROUP])?;
            let decoded = if code == 8 {
                0.0
            } else {
                crate::quant::e2m1_to_f32(code)
            };
            let value = decoded * scale;
            ensure!(
                value.is_finite(),
                "native FP4 value overflows FP32 reference"
            );
            result.push(value);
        }
    }
    Ok(result)
}

/// Exact1017 epsilon and explicit FP32 square/mean/rsqrt chain; BF16 output rounds once.
/// This CPU reduction order is a reference, not a claim of bitwise GPU tree equivalence.
pub fn rms_norm_bf16(
    input: &[bf16],
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<bf16>> {
    let count = matrix_elements(rows, columns)?;
    ensure!(
        input.len() == count && weights.len() == columns,
        "invalid RMSNorm matrix/weight shape"
    );
    ensure!(
        input.iter().all(|x| x.is_finite()) && weights.iter().all(|x| x.is_finite()),
        "nonfinite RMSNorm input/weight"
    );
    let mut output = Vec::with_capacity(count);
    for row in 0..rows {
        let start = row * columns;
        let mut sum = 0.0_f32;
        for x in &input[start..start + columns] {
            let x = x.to_f32();
            sum += x * x;
        }
        ensure!(
            sum.is_finite(),
            "RMSNorm square sum overflows FP32 reference"
        );
        let inverse = 1.0_f32 / (sum / columns as f32 + NORM_EPS).sqrt();
        ensure!(inverse.is_finite(), "nonfinite RMSNorm inverse");
        for column in 0..columns {
            let value = weights[column] * (input[start + column].to_f32() * inverse);
            ensure!(value.is_finite(), "RMSNorm output overflows FP32 reference");
            let rounded = bf16::from_f32(value);
            ensure!(
                rounded.is_finite(),
                "RMSNorm output overflows BF16 reference"
            );
            output.push(rounded);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_numerics_vectors.json"
        ))
        .unwrap()
    }
    fn usize_value(v: &Value, key: &str) -> usize {
        v[key].as_u64().unwrap() as usize
    }
    fn bf16_values(v: &Value, key: &str) -> Vec<bf16> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| bf16::from_bits(x.as_u64().unwrap() as u16))
            .collect()
    }
    fn u8_values(v: &Value, key: &str) -> Vec<u8> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect()
    }
    #[test]
    fn source_identity_and_all_fp8_cast_ties_clipping_and_signed_zero_match_torch_cpu() {
        let f = fixture();
        assert_eq!(
            f["source_identities"]["inference/kernel.py"],
            KERNEL_REFERENCE_SHA256
        );
        assert_eq!(
            f["source_identities"]["inference/model.py"],
            MODEL_REFERENCE_SHA256
        );
        assert_eq!(
            f["source_identities"]["inference/convert.py"],
            CONVERT_REFERENCE_SHA256
        );
        assert_eq!(f["oracle"]["device"], "cpu");
        for row in f["scalar_e4m3_cast_cases"].as_array().unwrap() {
            let x = f32::from_bits(row["input_f32_bits"].as_u64().unwrap() as u32);
            assert_eq!(
                encode_e4m3_rne_saturating(x).unwrap(),
                row["expected_byte"].as_u64().unwrap() as u8,
                "FP8 RNE input bits {:08x}",
                x.to_bits()
            );
        }
        assert_ne!(
            encode_e4m3_rne_saturating(4.75).unwrap(),
            crate::quant::f32_to_e4m3(4.75)
        );
        assert_eq!(encode_e4m3_rne_saturating(-0.0).unwrap(), 0x80);
    }
    #[test]
    fn per_group32_activation_dynamic_ranges_zero_floor_and_large_bf16_match_cpu() {
        let f = fixture();
        for row in f["activation_cases"].as_array().unwrap() {
            let result = quantize_activations_bf16(
                &bf16_values(row, "input_bf16_bits"),
                usize_value(row, "rows"),
                usize_value(row, "columns"),
            )
            .unwrap();
            assert_eq!(
                result.e4m3_values,
                u8_values(row, "expected_e4m3_bytes"),
                "{}",
                row["name"]
            );
            assert_eq!(
                result.ue8m0_scales,
                u8_values(row, "expected_ue8m0_scale_bytes"),
                "{}",
                row["name"]
            );
            if row["name"] == "largest_finite_bf16_group" {
                // The source quantizer's finite bytes/scales are valid, but FP8 rounding
                // can make their product exceed f32::MAX. The separate CPU dequantizer
                // intentionally refuses that overflow; it is not a quantization failure.
                assert!(dequantize_activations_f32(&result).is_err());
            } else {
                assert_eq!(
                    dequantize_activations_f32(&result).unwrap().len(),
                    result.e4m3_values.len()
                );
            }
            if row.get("old_group128_must_differ").and_then(Value::as_bool) == Some(true) {
                assert_ne!(
                    result.e4m3_values,
                    u8_values(row, "old_group128_e4m3_bytes")
                );
            }
        }
        assert_eq!(activation_scale(0.0).unwrap().0, 105);
    }
    #[test]
    fn ue8m0_tiny_scale_is_not_zero_and_native_fp4_all_codes_match_source() {
        let f = fixture();
        for row in f["ue8m0_decode_cases"].as_array().unwrap() {
            assert_eq!(
                decode_ue8m0(row["byte"].as_u64().unwrap() as u8)
                    .unwrap()
                    .to_bits(),
                row["expected_f32_bits"].as_u64().unwrap() as u32
            );
        }
        assert!(decode_ue8m0(0).unwrap() > 0.0);
        for row in f["native_fp4_cases"].as_array().unwrap() {
            let output = dequantize_fp4_native(
                &u8_values(row, "packed_bytes"),
                &u8_values(row, "ue8m0_scale_bytes"),
                usize_value(row, "rows"),
                usize_value(row, "columns"),
            )
            .unwrap();
            let expected: Vec<u32> =
                serde_json::from_value(row["expected_f32_bits"].clone()).unwrap();
            assert_eq!(
                output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected,
                "{}",
                row["name"]
            );
        }
    }
    #[test]
    fn rms_uses_exact1017_epsilon_and_rejects_old_default() {
        for row in fixture()["rms_cases"].as_array().unwrap() {
            let bits: Vec<u32> = serde_json::from_value(row["weight_f32_bits"].clone()).unwrap();
            let weights: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
            let got = rms_norm_bf16(
                &bf16_values(row, "input_bf16_bits"),
                &weights,
                usize_value(row, "rows"),
                usize_value(row, "columns"),
            )
            .unwrap();
            let expected: Vec<u16> =
                serde_json::from_value(row["expected_bf16_bits"].clone()).unwrap();
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected,
                "{}",
                row["name"]
            );
            if row["old_eps_must_differ"] == true {
                let old: Vec<u16> =
                    serde_json::from_value(row["old_eps1e6_bf16_bits"].clone()).unwrap();
                assert_ne!(expected, old);
            }
        }
    }
    #[test]
    fn unsupported_nonfinite_corrupt_shapes_scales_and_overflow_fail_closed() {
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(encode_e4m3_rne_saturating(x).is_err());
        }
        assert!(decode_e4m3_finite(0x7f).is_err());
        assert!(decode_e4m3_finite(0xff).is_err());
        assert!(decode_ue8m0(255).is_err());
        assert!(quantize_activations_bf16(&[bf16::ZERO; 31], 1, 31).is_err());
        assert!(quantize_activations_bf16(&[bf16::INFINITY; 32], 1, 32).is_err());
        assert!(quantize_activations_bf16(&[], 0, 32).is_err());
        assert!(dequantize_fp4_native(&[0; 16], &[255], 1, 32).is_err());
        assert!(dequantize_fp4_native(&[0x77; 16], &[254], 1, 32).is_err());
        assert!(dequantize_fp4_native(&[0; 15], &[127], 1, 32).is_err());
        assert!(rms_norm_bf16(&[bf16::ONE; 32], &[f32::NAN; 32], 1, 32).is_err());
        assert!(rms_norm_bf16(&[bf16::MAX; 32], &[1.0; 32], 1, 32).is_err());
        let mut q = quantize_activations_bf16(&[bf16::ZERO; 32], 1, 32).unwrap();
        q.group_size = 128;
        assert!(dequantize_activations_f32(&q).is_err());
    }
}
