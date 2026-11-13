//! Exact source arithmetic contract for 1017 FP4 activation/cache quantization.
//! Compressed KV uses group16/E4M3 scales; index Q/K uses group32/UE8M0.
//! This quantizes caller activations, never checkpoint weights. FP4 cache zero
//! retains its sign, unlike convert.py's static-weight negative-zero convention.

use super::numerics::{decode_e4m3_finite, decode_ue8m0, encode_e4m3_rne_saturating};
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::Serialize;

pub const MAX_ROWS: usize = 16 * 128 * 8;
pub const MAX_COLUMNS: usize = 5120;
pub const MAX_ELEMENTS: usize = 64 * 1024 * 1024;
pub const KERNEL_REFERENCE_SHA256: &str =
    "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455";
const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CacheFormat {
    CompressedKvE4m3Group16,
    IndexUe8m0Group32,
}
impl CacheFormat {
    pub fn group(self) -> usize {
        match self {
            Self::CompressedKvE4m3Group16 => 16,
            Self::IndexUe8m0Group32 => 32,
        }
    }
    pub fn tag(self) -> u32 {
        match self {
            Self::CompressedKvE4m3Group16 => 0,
            Self::IndexUe8m0Group32 => 1,
        }
    }
}

pub fn elements(rows: usize, columns: usize, format: CacheFormat) -> Result<usize> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows)
            && (1..=MAX_COLUMNS).contains(&columns)
            && columns % format.group() == 0,
        "unsupported cache quantization geometry"
    );
    rows.checked_mul(columns)
        .filter(|&n| n <= MAX_ELEMENTS)
        .context("cache quantization element bound exceeded")
}

pub fn decode_fp4_cache(code: u8) -> Result<f32> {
    ensure!(code < 16, "FP4 code exceeds one nibble");
    let value = VALUES[(code & 7) as usize];
    Ok(if code & 8 != 0 { -value } else { value })
}

/// RNE to the exact E2M1 table after source clamp +/-6. Negative underflow and
/// negative zero become code8, whose BF16 QAT view remains negative zero.
pub fn encode_fp4_rne(value: f32) -> Result<u8> {
    ensure!(value.is_finite(), "nonfinite FP4 activation");
    let magnitude = value.abs().min(6.0);
    let upper = VALUES.partition_point(|&x| x < magnitude);
    let code = if upper == 0 {
        0
    } else if upper == VALUES.len() {
        7
    } else {
        let lower = upper - 1;
        let dl = magnitude - VALUES[lower];
        let du = VALUES[upper] - magnitude;
        if dl < du || (dl == du && lower % 2 == 0) {
            lower
        } else {
            upper
        }
    };
    Ok(code as u8 | if value.is_sign_negative() { 8 } else { 0 })
}

pub fn scale_for_maximum(maximum: f32, format: CacheFormat) -> Result<(u8, f32)> {
    ensure!(
        maximum.is_finite() && maximum >= 0.0,
        "invalid FP4 group maximum"
    );
    match format {
        CacheFormat::CompressedKvE4m3Group16 => {
            let divided = maximum.max(6.0 * 2.0_f32.powi(-9)) / 6.0;
            // Source's CUDA E4M3 cast is rn.satfinite. CPU Torch's raw out-of-
            // range cast differs; saturation is checked separately on the GPU.
            let code = encode_e4m3_rne_saturating(divided)?;
            let scale = decode_e4m3_finite(code)?;
            ensure!(scale > 0.0, "compressed KV scale underflowed to zero");
            Ok((code, scale))
        }
        CacheFormat::IndexUe8m0Group32 => {
            let divided = maximum.max(6.0 * 2.0_f32.powi(-126)) * (1.0_f32 / 6.0);
            ensure!(
                divided.is_normal() && divided > 0.0,
                "index pow2 scale intermediate outside normal FP32"
            );
            let bits = divided.to_bits();
            let exponent = ((bits >> 23) & 255) as i32 - 127 + i32::from(bits & 0x7fffff != 0);
            ensure!(
                (-126..=127).contains(&exponent),
                "index scale exponent overflow"
            );
            let code = u8::try_from(exponent + 127)?;
            Ok((code, decode_ue8m0(code)?))
        }
    }
}

fn checked_scale(code: u8, format: CacheFormat) -> Result<f32> {
    let scale = match format {
        CacheFormat::CompressedKvE4m3Group16 => decode_e4m3_finite(code)?,
        CacheFormat::IndexUe8m0Group32 => decode_ue8m0(code)?,
    };
    ensure!(
        scale.is_finite() && scale > 0.0,
        "nonpositive/nonfinite cache scale"
    );
    Ok(scale)
}

#[derive(Clone, Debug)]
pub struct QuantizedCache {
    pub rows: usize,
    pub columns: usize,
    pub format: CacheFormat,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub qat: Vec<bf16>,
}

pub fn dequantize_bf16(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    format: CacheFormat,
) -> Result<Vec<bf16>> {
    let count = elements(rows, columns, format)?;
    ensure!(
        packed.len() == count / 2 && scales.len() == count / format.group(),
        "native cache packed/scale extent mismatch"
    );
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 2];
        let code = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
        let scale = checked_scale(scales[i / format.group()], format)?;
        let value = decode_fp4_cache(code)? * scale;
        let rounded = bf16::from_f32(value);
        ensure!(
            value.is_finite() && rounded.is_finite(),
            "cache BF16 reconstruction overflow"
        );
        result.push(rounded);
    }
    Ok(result)
}

pub fn quantize_bf16(
    input: &[bf16],
    rows: usize,
    columns: usize,
    format: CacheFormat,
) -> Result<QuantizedCache> {
    let count = elements(rows, columns, format)?;
    ensure!(
        input.len() == count && input.iter().all(|v| v.is_finite()),
        "nonfinite/wrong-shaped cache input"
    );
    let group = format.group();
    let mut packed = vec![0u8; count / 2];
    let mut scales = Vec::with_capacity(count / group);
    for (g, values) in input.chunks_exact(group).enumerate() {
        let maximum = values
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0_f32, f32::max);
        let (code, scale) = scale_for_maximum(maximum, format)?;
        scales.push(code);
        for (j, value) in values.iter().enumerate() {
            let encoded = encode_fp4_rne(value.to_f32() / scale)?;
            let i = g * group + j;
            packed[i / 2] |= encoded << if i % 2 == 0 { 0 } else { 4 };
        }
    }
    let qat = dequantize_bf16(&packed, &scales, rows, columns, format)?;
    Ok(QuantizedCache {
        rows,
        columns,
        format,
        packed,
        scales,
        qat,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rne_boundaries_and_cache_signed_zero_are_not_static_weight_rules() {
        for (value, wanted) in [
            (0.25, 0u8),
            (0.75, 2),
            (1.25, 2),
            (1.75, 4),
            (2.5, 4),
            (3.5, 6),
            (5.0, 6),
        ] {
            assert_eq!(encode_fp4_rne(value).unwrap(), wanted);
            assert_eq!(encode_fp4_rne(-value).unwrap(), wanted | 8);
        }
        assert_eq!(encode_fp4_rne(-0.0).unwrap(), 8);
        assert_eq!(decode_fp4_cache(8).unwrap().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(encode_fp4_rne(100.0).unwrap(), 7);
        assert!(encode_fp4_rne(f32::NAN).is_err());
    }
    #[test]
    fn exact_distinct_scale_floors_and_roundtrip_overflow_are_checked() {
        assert_eq!(
            scale_for_maximum(0.0, CacheFormat::CompressedKvE4m3Group16).unwrap(),
            (1, 2.0_f32.powi(-9))
        );
        assert_eq!(
            scale_for_maximum(0.0, CacheFormat::IndexUe8m0Group32).unwrap(),
            (1, 2.0_f32.powi(-126))
        );
        assert!(quantize_bf16(&[bf16::MAX; 32], 1, 32, CacheFormat::IndexUe8m0Group32).is_err());
        let large = quantize_bf16(
            &[bf16::MAX; 16],
            1,
            16,
            CacheFormat::CompressedKvE4m3Group16,
        )
        .unwrap();
        assert_eq!(large.scales, [126]);
        assert!(large.qat.iter().all(|v| v.to_f32() == 2688.0));
        for (format, scales) in [
            (CacheFormat::CompressedKvE4m3Group16, vec![127]),
            (CacheFormat::IndexUe8m0Group32, vec![255]),
        ] {
            assert!(dequantize_bf16(
                &vec![0; format.group() / 2],
                &scales,
                1,
                format.group(),
                format
            )
            .is_err());
        }
    }
    #[test]
    fn packed_and_qat_match_independent_cpu_vectors() {
        use sha2::{Digest, Sha256};
        let raw = include_str!("../../tests/fixtures/dsv41_cache_quant_vectors.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw.as_bytes())),
            "178682063f7af47a75d3a3d4c5adb0158079c08352276b194d137448d0545b27"
        );
        let fixture: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(fixture["kernel_reference_sha256"], KERNEL_REFERENCE_SHA256);
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 10);
        for case in cases {
            let format = match case["format"].as_str() {
                Some("compressed_kv") => CacheFormat::CompressedKvE4m3Group16,
                Some("index_qk") => CacheFormat::IndexUe8m0Group32,
                _ => panic!("unknown golden cache format"),
            };
            let bits: Vec<u16> = serde_json::from_value(case["input_bf16_bits"].clone()).unwrap();
            let input: Vec<_> = bits.into_iter().map(bf16::from_bits).collect();
            let result = quantize_bf16(
                &input,
                case["rows"].as_u64().unwrap() as usize,
                case["columns"].as_u64().unwrap() as usize,
                format,
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(&result.packed).unwrap(),
                case["packed_bytes"]
            );
            assert_eq!(
                serde_json::to_value(&result.scales).unwrap(),
                case["scale_bytes"]
            );
            assert_eq!(
                serde_json::to_value(result.qat.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                    .unwrap(),
                case["qat_bf16_bits"]
            );
        }
    }
}
