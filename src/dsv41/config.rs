//! Exact official DeepSeek-V4.1-Flash-1017 CPU configuration contract.
//!
//! This module validates metadata only. It does not claim a serving implementation.
//! The original `dsv4` model/configuration remains a separate architecture.
//! Canonical values below come from revision 1bae0e1088993b9b82de08f59de5be5a34e42170
//! and the two verified configuration files identified by their SHA256 constants.

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::io::Read;
use std::path::Path;

pub const MODEL_REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";
pub const HF_CONFIG_SHA256: &str =
    "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879";
pub const INFERENCE_CONFIG_SHA256: &str =
    "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809";
pub const REFERENCE_MODEL_SHA256: &str =
    "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d";
pub const REFERENCE_ENGRAM_SHA256: &str =
    "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897";
const CONFIG_BYTE_LIMIT: u64 = 64 * 1024;

// JSON object ordering is irrelevant. Missing, unknown and changed values are rejected:
// this first contract intentionally admits only the verified 1017 model, with no defaults.
const CANONICAL_HF_JSON: &str = r#"{
  "architectures": [
    "DeepseekV41ForCausalLM"
  ],
  "model_type": "deepseek_v41",
  "dtype": "bfloat16",
  "transformers_version": "5.6.0",
  "bos_token_id": 0,
  "eos_token_id": 1,
  "pad_token_id": 2,
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [
      32,
      32
    ],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": true,
    "tie_word_embeddings": false,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 16,
      "beta_fast": 32,
      "beta_slow": 1,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [
      0,
      0,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [
      2,
      8,
      14,
      20
    ],
    "index_source_layer_ids": [
      2,
      8,
      14,
      20,
      24,
      28,
      32,
      36
    ],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [
      1,
      14
    ],
    "engram_num_embeddings": [
      384006168,
      384016682
    ],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [
      37,
      38,
      39
    ],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": null
  }
}"#;
const CANONICAL_INFERENCE_JSON: &str = r#"{
    "vocab_size": 129280,
    "dim": 5120,
    "moe_inter_dim": 2304,
    "n_layers": 40,
    "n_mtp_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_n_activated_experts": 3,
    "n_heads": 64,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "n_activated_experts": 6,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 10.0,
    "q_lora_rank": 1280,
    "head_dim": 512,
    "rope_head_dim": 64,
    "norm_eps": 1e-20,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "window_size": 128,
    "kv_source_layers": [2, 8, 14, 20],
    "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
    "original_seq_len": 65536,
    "rope_theta": 10000,
    "rope_factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [1, 14],
    "engram_vocab_size": 16000000,
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_pad_id": 2,
    "engram_compressed_vocab_size": 99092,
    "dtype": "fp8",
    "expert_dtype": "fp4",
    "compress_rope_theta": 160000,
    "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "vision_n_layers": 32,
    "vision_dim": 1024,
    "vision_n_heads": 16,
    "vision_inter_dim": 2816,
    "vision_patch_size": 14,
    "vision_downsample_ratio": 3,
    "vision_max_n_token": 1024,
    "vision_min_pixels": 295936,
    "vision_max_wh_ratio": null,
    "image_token_id": 129264,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "vision_rope_theta": 10000
}"#;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dsv41Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_mtp_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_n_activated_experts: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub original_seq_len: usize,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub engram_vocab_size: usize,
    pub engram_max_ngram_size: usize,
    pub engram_pad_id: usize,
    pub engram_compressed_vocab_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub image_token_id: usize,
    pub route_scale: f64,
    pub swiglu_limit: f64,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub norm_eps: f64,
    pub hc_eps: f64,
    pub compress_rope_theta: f64,
    pub dspark_target_layer_ids: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub engram_layer_ids: Vec<usize>,
    pub compress_ratios: Vec<u32>,
    pub engram_num_embeddings: Vec<u64>,
    pub max_seq_len: usize,
    pub fp8_block_size: [usize; 2],
    pub fp4_group_size: usize,
    #[serde(skip)]
    hf_config: Value,
    #[serde(skip)]
    inference_config: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapabilitySummary {
    pub model_type: &'static str,
    pub architecture: &'static str,
    pub revision: &'static str,
    pub configuration_validated: bool,
    pub metadata_only: bool,
    pub serve_supported: bool,
    pub gpu_kernels_qualified: bool,
    pub tp8_execution_qualified: bool,
    pub vision_execution_supported: bool,
    pub reference_hf_config_sha256: &'static str,
    pub reference_inference_config_sha256: &'static str,
    pub reference_model_sha256: &'static str,
    pub reference_engram_sha256: &'static str,
    pub geometry: Dsv41Config,
}

fn mismatch(actual: &Value, expected: &Value, path: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            for key in a.keys() {
                if !e.contains_key(key) {
                    return Some(format!("{path}/{key}: unsupported field"));
                }
            }
            for (key, value) in e {
                let next = format!("{path}/{key}");
                match a.get(key) {
                    None => return Some(format!("{next}: required field missing")),
                    Some(found) => {
                        if let Some(error) = mismatch(found, value, &next) {
                            return Some(error);
                        }
                    }
                }
            }
            None
        }
        (Value::Array(a), Value::Array(e)) => {
            if a.len() != e.len() {
                return Some(format!(
                    "{path}: length {} differs from official1017 {}",
                    a.len(),
                    e.len()
                ));
            }
            a.iter()
                .zip(e)
                .enumerate()
                .find_map(|(index, (found, wanted))| {
                    mismatch(found, wanted, &format!("{path}/{index}"))
                })
        }
        _ if actual == expected => None,
        _ => Some(format!(
            "{path}: unsupported value {actual}; official1017 requires {expected}"
        )),
    }
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        value
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("{key}: required unsigned integer"))?,
    )
    .with_context(|| format!("{key}: does not fit usize"))
}
fn float_field(value: &Value, key: &str) -> Result<f64> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("{key}: required numeric value"))?;
    ensure!(number.is_finite(), "{key}: non-finite value");
    Ok(number)
}
fn usize_array(value: &Value, key: &str) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{key}: required integer array"))?
        .iter()
        .map(|x| {
            usize::try_from(
                x.as_u64()
                    .ok_or_else(|| anyhow!("{key}: noninteger entry"))?,
            )
            .with_context(|| format!("{key}: entry does not fit usize"))
        })
        .collect()
}
fn read_config(path: &Path) -> Result<Value> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open config {}", path.display()))?;
    ensure!(
        file.metadata()?.len() <= CONFIG_BYTE_LIMIT,
        "configuration {} exceeds bounded metadata size",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(CONFIG_BYTE_LIMIT + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= CONFIG_BYTE_LIMIT,
        "configuration {} grew beyond bounded metadata size",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parse config {}", path.display()))
}

impl Dsv41Config {
    /// Parse both exact reference configurations. No GPU, tensor payload, tokenizer or model load.
    pub fn from_values(hf: &Value, inference: &Value) -> Result<Self> {
        let expected_hf: Value =
            serde_json::from_str(CANONICAL_HF_JSON).context("internal canonical1017 HF config")?;
        let expected_inference: Value = serde_json::from_str(CANONICAL_INFERENCE_JSON)
            .context("internal canonical1017 inference config")?;
        if let Some(error) = mismatch(hf, &expected_hf, "hf") {
            bail!("{error}");
        }
        if let Some(error) = mismatch(inference, &expected_inference, "inference") {
            bail!("{error}");
        }
        let text = hf
            .get("text_config")
            .context("required deepseek_v41 text_config")?;
        Ok(Self {
            vocab_size: usize_field(inference, "vocab_size")?,
            dim: usize_field(inference, "dim")?,
            moe_inter_dim: usize_field(inference, "moe_inter_dim")?,
            n_layers: usize_field(inference, "n_layers")?,
            n_mtp_layers: usize_field(inference, "n_mtp_layers")?,
            dspark_block_size: usize_field(inference, "dspark_block_size")?,
            dspark_noise_token_id: usize_field(inference, "dspark_noise_token_id")?,
            dspark_markov_rank: usize_field(inference, "dspark_markov_rank")?,
            dspark_n_routed_experts: usize_field(inference, "dspark_n_routed_experts")?,
            dspark_n_activated_experts: usize_field(inference, "dspark_n_activated_experts")?,
            n_heads: usize_field(inference, "n_heads")?,
            n_routed_experts: usize_field(inference, "n_routed_experts")?,
            n_shared_experts: usize_field(inference, "n_shared_experts")?,
            n_activated_experts: usize_field(inference, "n_activated_experts")?,
            q_lora_rank: usize_field(inference, "q_lora_rank")?,
            head_dim: usize_field(inference, "head_dim")?,
            rope_head_dim: usize_field(inference, "rope_head_dim")?,
            o_groups: usize_field(inference, "o_groups")?,
            o_lora_rank: usize_field(inference, "o_lora_rank")?,
            window_size: usize_field(inference, "window_size")?,
            original_seq_len: usize_field(inference, "original_seq_len")?,
            beta_fast: usize_field(inference, "beta_fast")?,
            beta_slow: usize_field(inference, "beta_slow")?,
            index_n_heads: usize_field(inference, "index_n_heads")?,
            index_head_dim: usize_field(inference, "index_head_dim")?,
            index_topk: usize_field(inference, "index_topk")?,
            candidate_source_layer: usize_field(inference, "candidate_source_layer")?,
            candidate_topk_blocks: usize_field(inference, "candidate_topk_blocks")?,
            candidate_block_size: usize_field(inference, "candidate_block_size")?,
            hc_mult: usize_field(inference, "hc_mult")?,
            hc_sinkhorn_iters: usize_field(inference, "hc_sinkhorn_iters")?,
            engram_vocab_size: usize_field(inference, "engram_vocab_size")?,
            engram_max_ngram_size: usize_field(inference, "engram_max_ngram_size")?,
            engram_pad_id: usize_field(inference, "engram_pad_id")?,
            engram_compressed_vocab_size: usize_field(inference, "engram_compressed_vocab_size")?,
            engram_n_heads: usize_field(inference, "engram_n_heads")?,
            engram_head_dim: usize_field(inference, "engram_head_dim")?,
            image_token_id: usize_field(inference, "image_token_id")?,
            route_scale: float_field(inference, "route_scale")?,
            swiglu_limit: float_field(inference, "swiglu_limit")?,
            rope_theta: float_field(inference, "rope_theta")?,
            rope_factor: float_field(inference, "rope_factor")?,
            norm_eps: float_field(inference, "norm_eps")?,
            hc_eps: float_field(inference, "hc_eps")?,
            compress_rope_theta: float_field(inference, "compress_rope_theta")?,
            dspark_target_layer_ids: usize_array(inference, "dspark_target_layer_ids")?,
            kv_source_layers: usize_array(inference, "kv_source_layers")?,
            index_source_layers: usize_array(inference, "index_source_layers")?,
            engram_layer_ids: usize_array(inference, "engram_layer_ids")?,
            compress_ratios: usize_array(inference, "compress_ratios")?
                .into_iter()
                .map(|x| u32::try_from(x).context("compress ratio does not fit u32"))
                .collect::<Result<_>>()?,
            engram_num_embeddings: inference["engram_num_embeddings"]
                .as_array()
                .context("Engram rows")?
                .iter()
                .map(|x| x.as_u64().context("Engram row count"))
                .collect::<Result<_>>()?,
            max_seq_len: usize_field(text, "max_position_embeddings")?,
            fp8_block_size: [32, 32], // exact checked HF weight_block_size; source block32 on both operands
            fp4_group_size: 32, // inference/model.py fp4_block_size; original FP4 expert scale geometry
            hf_config: hf.clone(),
            inference_config: inference.clone(),
        })
    }

    pub fn load_dir(bundle_dir: &Path) -> Result<Self> {
        let hf = read_config(&bundle_dir.join("config.json"))?;
        let inference = read_config(&bundle_dir.join("inference/config.json"))?;
        Self::from_values(&hf, &inference)
    }

    /// Revalidate source values and all public normalized fields, catching caller mutations.
    pub fn validate(&self) -> Result<()> {
        let normalized = Self::from_values(&self.hf_config, &self.inference_config)?;
        ensure!(
            *self == normalized,
            "normalized1017 geometry was modified after validation"
        );
        Ok(())
    }

    pub fn hf_config(&self) -> &Value {
        &self.hf_config
    }
    pub fn inference_config(&self) -> &Value {
        &self.inference_config
    }

    /// A metadata contract is not a runnable serving backend.
    pub fn capabilities(&self) -> CapabilitySummary {
        CapabilitySummary {
            model_type: "deepseek_v41",
            architecture: "DeepseekV41ForCausalLM",
            revision: MODEL_REVISION,
            configuration_validated: self.validate().is_ok(),
            metadata_only: true,
            serve_supported: false,
            gpu_kernels_qualified: false,
            tp8_execution_qualified: false,
            vision_execution_supported: false,
            reference_hf_config_sha256: HF_CONFIG_SHA256,
            reference_inference_config_sha256: INFERENCE_CONFIG_SHA256,
            reference_model_sha256: REFERENCE_MODEL_SHA256,
            reference_engram_sha256: REFERENCE_ENGRAM_SHA256,
            geometry: self.clone(),
        }
    }
}

pub fn load_config(bundle_dir: &Path) -> Result<Dsv41Config> {
    Dsv41Config::load_dir(bundle_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json")).unwrap()
    }
    fn pair() -> (Value, Value) {
        let data = fixture();
        (data["hf_config"].clone(), data["inference_config"].clone())
    }
    #[test]
    fn official_fixture_identity_geometry_and_capability_are_exact() {
        let data = fixture();
        assert_eq!(data["revision"], MODEL_REVISION);
        for (field, wanted) in [
            ("hf_source_utf8", HF_CONFIG_SHA256),
            ("inference_source_utf8", INFERENCE_CONFIG_SHA256),
        ] {
            assert_eq!(
                format!(
                    "{:x}",
                    Sha256::digest(data[field].as_str().unwrap().as_bytes())
                ),
                wanted
            );
        }
        let (hf, inference) = pair();
        assert_eq!(
            serde_json::from_str::<Value>(data["hf_source_utf8"].as_str().unwrap()).unwrap(),
            hf
        );
        assert_eq!(
            serde_json::from_str::<Value>(data["inference_source_utf8"].as_str().unwrap()).unwrap(),
            inference
        );
        let cfg = Dsv41Config::from_values(&hf, &inference).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            (cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_routed_experts),
            (5120, 40, 64, 384)
        );
        assert_eq!(
            (cfg.dspark_n_routed_experts, cfg.dspark_n_activated_experts),
            (128, 3)
        );
        assert_eq!(cfg.engram_num_embeddings, vec![384006168, 384016682]);
        assert_eq!(cfg.kv_source_layers, vec![2, 8, 14, 20]);
        assert_eq!(cfg.index_source_layers, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!(cfg.norm_eps, 1e-20);
        assert_eq!(cfg.hc_eps, 1e-6);
        assert_eq!(cfg.fp8_block_size, [32, 32]);
        let caps = serde_json::to_value(cfg.capabilities()).unwrap();
        assert_eq!(caps["serve_supported"], false);
        assert_eq!(caps["tp8_execution_qualified"], false);
        assert_eq!(caps["metadata_only"], true);
    }
    #[test]
    fn different_model_families_missing_and_unknown_fields_are_rejected() {
        let (hf, inf) = pair();
        for family in ["deepseek_v4", "qwen3_5", "deepseek_v41_text"] {
            let mut changed = hf.clone();
            changed["model_type"] = Value::from(family);
            assert!(Dsv41Config::from_values(&changed, &inf).is_err());
        }
        let mut missing = inf.clone();
        missing.as_object_mut().unwrap().remove("norm_eps");
        assert!(Dsv41Config::from_values(&hf, &missing)
            .unwrap_err()
            .to_string()
            .contains("norm_eps"));
        let mut old = inf.clone();
        old["n_hash_layers"] = Value::from(3);
        assert!(Dsv41Config::from_values(&hf, &old)
            .unwrap_err()
            .to_string()
            .contains("unsupported field"));
    }
    #[test]
    fn mismatched_and_jointly_changed_geometries_do_not_get_old_defaults() {
        let (hf, inf) = pair();
        for (hkey, ikey, wrong) in [
            ("hidden_size", "dim", 4096),
            ("num_hidden_layers", "n_layers", 43),
            ("n_routed_experts", "n_routed_experts", 256),
            ("q_lora_rank", "q_lora_rank", 1024),
            ("index_n_heads", "index_n_heads", 64),
        ] {
            let mut h = hf.clone();
            let mut i = inf.clone();
            h["text_config"][hkey] = Value::from(wrong);
            assert!(Dsv41Config::from_values(&h, &i).is_err());
            i[ikey] = Value::from(wrong);
            assert!(Dsv41Config::from_values(&h, &i).is_err());
        }
        let mut cfg = Dsv41Config::from_values(&hf, &inf).unwrap();
        cfg.dim = 4096;
        assert!(cfg.validate().is_err());
        assert!(!cfg.capabilities().configuration_validated);
        assert!(!cfg.capabilities().serve_supported);
    }
    #[test]
    fn compression_source_engram_mtp_and_precision_mutations_are_rejected() {
        let (hf, inf) = pair();
        let mut variants = Vec::new();
        let mut v = inf.clone();
        v["compress_ratios"][2] = Value::from(4);
        variants.push(v);
        let mut v = inf.clone();
        v["compress_ratios"][20] = Value::from(128);
        variants.push(v);
        let mut v = inf.clone();
        v["kv_source_layers"][1] = Value::from(9);
        variants.push(v);
        let mut v = inf.clone();
        v["index_source_layers"][4] = Value::from(25);
        variants.push(v);
        let mut v = inf.clone();
        v["engram_num_embeddings"][0] = Value::from(384006169_u64);
        variants.push(v);
        let mut v = inf.clone();
        v["engram_compressed_vocab_size"] = Value::from(99093);
        variants.push(v);
        let mut v = inf.clone();
        v["dspark_n_routed_experts"] = Value::from(384);
        variants.push(v);
        let mut v = inf.clone();
        v["norm_eps"] = Value::from(1e-6);
        variants.push(v);
        for changed in variants {
            assert!(Dsv41Config::from_values(&hf, &changed).is_err());
        }
        let mut h = hf.clone();
        h["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128]);
        assert!(Dsv41Config::from_values(&h, &inf).is_err());
    }
    #[test]
    fn bounded_metadata_loader_reads_only_the_two_config_files() {
        let (hf, inf) = pair();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("dsv41-config-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(root.join("inference")).unwrap();
        std::fs::write(root.join("config.json"), serde_json::to_vec(&hf).unwrap()).unwrap();
        std::fs::write(
            root.join("inference/config.json"),
            serde_json::to_vec(&inf).unwrap(),
        )
        .unwrap();
        assert!(Dsv41Config::load_dir(&root).is_ok()); // no model/index/weights exist here
        std::fs::write(
            root.join("config.json"),
            vec![b' '; CONFIG_BYTE_LIMIT as usize + 1],
        )
        .unwrap();
        assert!(Dsv41Config::load_dir(&root)
            .unwrap_err()
            .to_string()
            .contains("bounded metadata size"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
