//! Explicit, conservative first-light contract for eight-rank serving.
//!
//! The serial FP32 transport is distinct from the optimized two/four-rank
//! pipeline. Keep unsupported postures closed until their own hardware gates.
use anyhow::{ensure, Result};

pub const WORLD: u32 = 8;

/// The general config parser has a permissive family fallback. Experimental
/// TP8 must identify its architecture explicitly rather than inheriting it.
/// Qwen3.8-27B uses these qwen3_5 architecture identifiers in its actual config.
pub fn validate_model_type(raw: &serde_json::Value) -> Result<()> {
    let model_type = raw.get("text_config").and_then(|c| c.get("model_type"))
        .or_else(|| raw.get("model_type")).and_then(serde_json::Value::as_str);
    ensure!(matches!(model_type, Some("qwen3_5" | "qwen3_5_text")),
        "experimental TP8 requires an explicit dense Qwen architecture (qwen3_5/qwen3_5_text), got {model_type:?}");
    Ok(())
}

/// Validate before discovery/shipping. A model-family check follows separately.
pub fn validate_server_options(
    world: u32,
    experimental: bool,
    mtp: &str,
    spec_source: &str,
    eager: bool,
    fp32: bool,
    kv_cache: &str,
) -> Result<()> {
    if world != WORLD {
        ensure!(!experimental, "--experimental-tp8 requires --tp 8");
        ensure!(matches!(world, 1 | 2 | 4), "serving supports TP=1/2/4 or experimental TP=8 (got {world})");
        return Ok(());
    }
    ensure!(experimental, "--tp 8 requires --experimental-tp8; TP8 is an unqualified bring-up profile");
    ensure!(matches!(mtp, "off" | "0" | "false"), "experimental TP8 requires --mtp off");
    ensure!(spec_source == "none", "experimental TP8 requires --spec-source none");
    ensure!(eager, "experimental TP8 requires --no-decode-graphs");
    ensure!(fp32, "experimental TP8 requires --tp-fp32-partials");
    ensure!(kv_cache == "bf16", "experimental TP8 requires --kv-cache bf16");
    Ok(())
}

/// The serialized config is checked independently on head and peers, before
/// model allocation or transport setup. Every required field must really ship.
pub fn validate_runtime(c: &crate::tp::TpConfig) -> Result<()> {
    validate_runtime_with_ack(c, crate::opts::on(crate::opt!("experimental-tp8")))
}

fn validate_runtime_with_ack(c: &crate::tp::TpConfig, acknowledged: bool) -> Result<()> {
    if c.world != WORLD { return Ok(()); }
    ensure!(acknowledged,
        "rank {}: experimental TP8 acknowledgement is missing from the shipped options", c.node_rank);
    ensure!(c.mode_serve && !c.exl3 && !c.dspark && !c.dflash,
        "experimental TP8 currently serves only the dense Qwen NVFP4 path");
    ensure!(c.mtp_force == Some(false) && c.spec_source == "none"
            && !c.mtp_lanes && c.ngram_draft == 0 && !c.tree_draft,
        "experimental TP8 requires plain decoding without a drafter or speculative lanes");
    ensure!(c.fp32_partials && c.no_decode_graphs && !c.graph && !c.device_loop && !c.oneshot,
        "experimental TP8 requires eager serial FP32 collectives (no graphs, device loop or one-shot)");
    ensure!(!c.shard_mtp, "experimental TP8 requires --tp-shard-mtp off");
    ensure!(c.shard_mixers, "experimental TP8 requires the default sharded mixers");
    ensure!(!c.prefix_cache, "experimental TP8 prefix reuse requires a separate qualification");
    ensure!(c.gpu_recv == Some(true) && c.reduce_fuse == Some(false),
        "experimental TP8 requires --tp-gpu-recv on --tp-reduce-fuse off");
    ensure!(!c.kv_quant && !c.kv_tq && !c.kv_tq_b3 && !c.kv_k8v4 && !c.kv_k8v8,
        "experimental TP8 requires unquantized bf16 KV");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_architectures_do_not_inherit_the_general_parser_fallback() {
        for raw in [serde_json::json!({}), serde_json::json!({"model_type":"llama"}),
            serde_json::json!({"model_type":"qwen3_5_moe"}),
            serde_json::json!({"model_type":"qwen3_5", "text_config":{"model_type":"unrecognized"}})] {
            assert!(validate_model_type(&raw).is_err());
        }
        assert!(validate_model_type(&serde_json::json!({"model_type":"qwen3_5"})).is_ok());
        assert!(validate_model_type(&serde_json::json!({"model_type":"qwen3_5", "text_config":{"model_type":"qwen3_5_text"}})).is_ok());
    }

    #[test]
    fn serialized_profile_cannot_enable_unsupported_runtime_paths() {
        let mut c = crate::tp::TpConfig::from_opts();
        c.world = WORLD; c.mode_serve = true;
        c.exl3 = false; c.dspark = false; c.dflash = false;
        c.mtp_force = Some(false); c.spec_source = "none".into();
        c.mtp_lanes = false; c.ngram_draft = 0; c.tree_draft = false;
        c.fp32_partials = true; c.no_decode_graphs = true;
        c.graph = false; c.device_loop = false; c.oneshot = false;
        c.shard_mtp = false; c.shard_mixers = true; c.prefix_cache = false;
        c.gpu_recv = Some(true); c.reduce_fuse = Some(false);
        c.kv_quant = false; c.kv_tq = false; c.kv_tq_b3 = false;
        c.kv_k8v4 = false; c.kv_k8v8 = false;
        let c: crate::tp::TpConfig = serde_json::from_slice(&serde_json::to_vec(&c).unwrap()).unwrap();
        assert!(validate_runtime_with_ack(&c, true).is_ok());
        assert!(validate_runtime_with_ack(&c, false).is_err());
        macro_rules! refused {
            ($field:ident, $value:expr) => {{
                let mut changed = c.clone(); changed.$field = $value;
                assert!(validate_runtime_with_ack(&changed, true).is_err(), stringify!($field));
            }};
        }
        refused!(mode_serve, false); refused!(exl3, true); refused!(dspark, true);
        refused!(dflash, true); refused!(mtp_force, None); refused!(spec_source, "mtp".into());
        refused!(mtp_lanes, true); refused!(ngram_draft, 1); refused!(tree_draft, true);
        refused!(fp32_partials, false); refused!(no_decode_graphs, false);
        refused!(graph, true); refused!(device_loop, true); refused!(oneshot, true);
        refused!(shard_mtp, true); refused!(shard_mixers, false); refused!(prefix_cache, true);
        refused!(gpu_recv, None); refused!(gpu_recv, Some(false));
        refused!(reduce_fuse, None); refused!(reduce_fuse, Some(true));
        refused!(kv_quant, true); refused!(kv_tq, true); refused!(kv_tq_b3, true);
        refused!(kv_k8v4, true); refused!(kv_k8v8, true);
    }

    #[test]
    fn existing_worlds_are_unchanged() {
        for world in [1, 2, 4] {
            assert!(validate_server_options(world, false, "auto", "mtp", false, false, "q8").is_ok());
        }
    }

    #[test]
    fn tp8_requires_each_first_light_property() {
        assert!(validate_server_options(8, true, "off", "none", true, true, "bf16").is_ok());
        assert!(validate_server_options(8, false, "off", "none", true, true, "bf16").is_err());
        assert!(validate_server_options(8, true, "auto", "none", true, true, "bf16").is_err());
        assert!(validate_server_options(8, true, "off", "mtp", true, true, "bf16").is_err());
        assert!(validate_server_options(8, true, "off", "none", false, true, "bf16").is_err());
        assert!(validate_server_options(8, true, "off", "none", true, false, "bf16").is_err());
        assert!(validate_server_options(8, true, "off", "none", true, true, "q8").is_err());
    }

    #[test]
    fn acknowledgement_does_not_enable_other_worlds() {
        assert!(validate_server_options(4, true, "off", "none", true, true, "bf16").is_err());
        assert!(validate_server_options(16, false, "off", "none", true, true, "bf16").is_err());
        assert!(validate_server_options(16, true, "off", "none", true, true, "bf16").is_err());
    }
}
