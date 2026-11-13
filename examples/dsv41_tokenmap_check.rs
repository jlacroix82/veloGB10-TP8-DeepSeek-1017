//! CPU-only check of the real checkpoint tokenizer's Engram normalization map.
use anyhow::{ensure, Context, Result};
use gb10_inference::dsv41::engram::{CompressedTokenMap, TOKENIZER_SHA256};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 2,
        "usage: dsv41_tokenmap_check <verified tokenizer.json>"
    );
    let path = PathBuf::from(&args[1]);
    let bytes = std::fs::read(&path).context("read checkpoint tokenizer")?;
    let tokenizer_sha256 = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        tokenizer_sha256 == TOKENIZER_SHA256,
        "tokenizer bytes differ from the verified 1017 artifact"
    );
    let tokenizer = tokenizers_recipe::Tokenizer::from_bytes(&bytes).map_err(anyhow::Error::msg)?;
    let map = CompressedTokenMap::build(&tokenizer)?;
    map.validate_1017()?;
    let mut hash = Sha256::new();
    for &id in map.as_slice() {
        hash.update(id.to_le_bytes());
    }
    println!(
        "{}",
        json!({"kind":"dsv41_cpu_tokenmap_check","passed":true,
        "tokenizer_sha256":tokenizer_sha256,"tokenizer_identity_passed":true,
        "map_u32le_sha256":format!("{:x}",hash.finalize()),
        "vocabulary_entries":map.as_slice().len(),
        "compressed_entries":map.as_slice().iter().copied().max().map(|n|n+1),
        "compressed_pad_id":map.as_slice()[2],"gpu_work":false,"serving_supported":false})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
