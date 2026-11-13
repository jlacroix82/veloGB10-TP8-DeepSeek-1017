//! Exercise the compiled CLI's refusal gates with config-only fixtures. No model,
//! GPU or listener is needed: even the peer argument is deliberately unparsable.
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new(config: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "velo-tp8-profile-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("config.json"), config).unwrap();
        Self(path)
    }

    fn run(&self, world: &str, flags: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_gb10_inference"));
        command.args(["--server", "--model-dir"]).arg(&self.0)
            .args(["--tp", world, "--nodes", "NOT_A_PEER_ADDRESS", "--mem-watchdog-gb", "0"])
            .args(flags)
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::piped()).stderr(Stdio::piped());
        // Engine option environment aliases are intentionally refused by main. Inherit
        // library search paths, but never let an operator's option alias change this test.
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GB10_") || key.to_string_lossy().starts_with("RUST_INFER_") {
                command.env_remove(key);
            }
        }
        let mut child = command.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() { return child.wait_with_output().unwrap(); }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!("TP8 refusal failed to finish before discovery: {}", String::from_utf8_lossy(&output.stderr));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

const PROFILE: &[&str] = &[
    "--experimental-tp8", "--mtp", "off", "--spec-source", "none",
    "--no-decode-graphs", "--tp-fp32-partials", "--kv-cache", "bf16",
    "--tp-shard-mtp", "off", "--tp-gpu-recv", "on", "--tp-reduce-fuse", "off",
];

fn refusal(output: Output, expected: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(2), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains(expected), "expected {expected:?}; stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("building manifest") && !stderr.contains("building manifest"),
        "refusal must precede artifact shipment");
    assert!(!stdout.contains("discovering nodes") && !stderr.contains("discovering nodes"),
        "refusal must precede discovery");
    assert!(!stderr.contains("bad --nodes"), "the invalid peer argument must never be parsed");
}

#[test]
fn every_required_initial_profile_option_is_checked_before_peer_parsing() {
    let fixture = Fixture::new("{}");
    for (flag, takes_value, expected) in [
        ("--experimental-tp8", false, "--tp 8 requires --experimental-tp8"),
        ("--mtp", true, "requires --mtp off"),
        ("--spec-source", true, "requires --spec-source none"),
        ("--no-decode-graphs", false, "requires --no-decode-graphs"),
        ("--tp-fp32-partials", false, "requires --tp-fp32-partials"),
        ("--kv-cache", true, "requires --kv-cache bf16"),
    ] {
        let mut flags = PROFILE.to_vec();
        let index = flags.iter().position(|v| *v == flag).unwrap();
        flags.drain(index..index + if takes_value { 2 } else { 1 });
        refusal(fixture.run("8", &flags), expected);
    }
}

#[test]
fn tp8_acknowledgement_does_not_enable_another_world() {
    let fixture = Fixture::new("{}");
    refusal(fixture.run("4", PROFILE), "--experimental-tp8 requires --tp 8");
    refusal(fixture.run("16", &[]), "serving supports TP=1/2/4 or experimental TP=8");
}

#[test]
fn unsupported_models_refuse_before_tokenizer_or_peer_access() {
    for config in ["{}", r#"{"model_type":"llama"}"#, r#"{"model_type":"deepseek_v4"}"#] {
        let unsupported = Fixture::new(config);
        refusal(unsupported.run("8", PROFILE), "explicit dense Qwen architecture");
    }

    let bad_bands = Fixture::new(r#"{"model_type":"qwen3_5","text_config":{"intermediate_size":64}}"#);
    refusal(bad_bands.run("8", PROFILE), "does not form eight 32-aligned bands");

    let naive = Fixture::new(r#"{"model_type":"naive_n05_flash"}"#);
    refusal(naive.run("8", PROFILE), "EXL3 and Naive need separate qualification");

    let exl3 = Fixture::new("{}");
    fs::write(exl3.0.join("quantization_config.json"), r#"{"quant_method":"exl3"}"#).unwrap();
    fs::write(exl3.0.join("model.safetensors.index.json"), r#"{"weight_map":{"model.trellis":"part.safetensors"}}"#).unwrap();
    refusal(exl3.run("8", PROFILE), "EXL3 and Naive need separate qualification");
}
