//! CPU-only reproduction of the retained code/prose native prompt identities.
//! Takes the verified tokenizer and private comparison-target JSON, with no API.
use anyhow::{ensure, Context, Result};
use deepseek_recipe_encoding::PromptTokenizer;
use gb10_inference::dsv41::{engram::TOKENIZER_SHA256, protocol::prepare_chat};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 3,
        "usage: dsv41_protocol_check <verified-tokenizer.json> <comparison-target.json>"
    );
    let bytes = std::fs::read(PathBuf::from(&args[1]))?;
    let tokenizer_sha = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        tokenizer_sha == TOKENIZER_SHA256,
        "unexpected tokenizer identity"
    );
    let tokenizer = tokenizers_recipe::Tokenizer::from_bytes(&bytes).map_err(anyhow::Error::msg)?;
    let encoder = PromptTokenizer::new(&tokenizer);
    let reference: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(&args[2]))?)?;
    ensure!(
        reference["model"]["revision"] == gb10_inference::dsv41::config::MODEL_REVISION,
        "wrong reference model"
    );
    let mut checks = Vec::new();
    for kind in ["code", "prose"] {
        let target = &reference["capped_speed"]["prompts_and_native_IDs"][kind];
        let text = target["text"].as_str().context("reference text")?;
        let request = json!({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":text}],
            "temperature":0,"seed":42,"max_tokens":256,"stream":false});
        let prepared = prepare_chat(request)?;
        let actual = prepared.encode_ids(&encoder)?;
        let expected: Vec<u32> = target["native_input_ids"]
            .as_array()
            .context("reference IDs")?
            .iter()
            .map(|v| Ok(u32::try_from(v.as_u64().context("native ID")?)?))
            .collect::<Result<_>>()?;
        let mut hash = Sha256::new();
        for &id in &actual {
            hash.update(id.to_le_bytes());
        }
        let digest = format!("{:x}", hash.finalize());
        let matched = actual == expected && target["native_input_ids_sha256_u32le"] == digest;
        checks.push(json!({"kind":kind,"passed":matched,"native_input_tokens":actual.len(),
            "expected_tokens":expected.len(),"native_input_ids_sha256_u32le":digest,
            "native_input_ids":actual,"rendered_prompt":prepared.rendered.prompt,
            "sampling":prepared.sampling,"max_tokens":prepared.resolve_max_tokens(actual.len() as u32,1048576)?}));
    }
    let passed = checks.iter().all(|v| v["passed"] == true);
    println!(
        "{}",
        json!({"kind":"dsv41_native_prompt_identity_check","passed":passed,
        "tokenizer_sha256":tokenizer_sha,"checks":checks,"gpu_work":false,"model_execution":false})
    );
    ensure!(
        passed,
        "native prompt IDs do not match retained DeepSeek experiment"
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
