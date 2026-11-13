//! Exercise the real launcher in an isolated directory with a fake engine.
//! These are command-selection checks, not model or hardware qualification.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "velo-tp8-launcher-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::create_dir(path.join("scripts")).unwrap();
        fs::create_dir_all(path.join("target/release")).unwrap();
        fs::create_dir(path.join("model")).unwrap();
        fs::write(path.join("model/config.json"), "{}").unwrap();
        fs::write(path.join("scripts/run_tp_server.sh"), include_str!("../scripts/run_tp_server.sh")).unwrap();
        let engine = path.join("target/release/gb10_inference");
        fs::write(&engine, "#!/bin/sh\necho ENGINE_EXECUTED\nexit 91\n").unwrap();
        fs::set_permissions(engine, fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }

    fn run(&self, world: usize, opts: &[&str]) -> Output {
        let peers = (1..world).map(|n| format!("192.0.2.{}:29500", n + 1)).collect::<Vec<_>>().join(",");
        Command::new("bash")
            .arg(self.0.join("scripts/run_tp_server.sh"))
            .arg("--model-dir").arg(self.0.join("model"))
            .arg("--node").arg(peers)
            .args(opts)
            .output().unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

#[test]
fn existing_two_and_four_rank_dry_runs_keep_their_world() {
    let fixture = Fixture::new();
    for world in [2, 4] {
        let output = fixture.run(world, &["--dry-run"]);
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let line = String::from_utf8(output.stdout).unwrap();
        assert!(line.contains(&format!("--tp {world}")), "{line}");
        assert!(!line.contains("--experimental-tp8") && !line.contains("ENGINE_EXECUTED"), "{line}");
    }
}

#[test]
fn seven_peers_require_explicit_experimental_opt_in_before_engine_exec() {
    let output = Fixture::new().run(8, &[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --experimental-tp8"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("ENGINE_EXECUTED"));
}

#[test]
fn opted_in_eight_rank_dry_run_forwards_all_peers_and_runtime_flag() {
    let output = Fixture::new().run(8, &["--experimental-tp8", "--dry-run", "--", "--mtp", "off", "--spec-source", "none"]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let line = String::from_utf8(output.stdout).unwrap();
    assert!(line.contains("--tp 8") && line.contains("--experimental-tp8"), "{line}");
    for last in 2..=8 { assert!(line.contains(&format!("192.0.2.{last}:29500")), "{line}"); }
    assert!(line.contains("--mtp off --spec-source none"), "{line}");
    assert!(!line.contains("ENGINE_EXECUTED"), "{line}");
}

#[test]
fn experimental_flag_does_not_allow_another_world_or_rank_count_mismatch() {
    let fixture = Fixture::new();
    for world in [2, 4, 16] {
        let output = fixture.run(world, &["--experimental-tp8"]);
        assert_eq!(output.status.code(), Some(2), "world {world}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("ENGINE_EXECUTED"));
    }
    let mismatch = fixture.run(8, &["--experimental-tp8", "--tp", "4"]);
    assert_eq!(mismatch.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("does not match"));
}

#[test]
fn incomplete_three_rank_world_is_rejected_without_executing_engine() {
    let output = Fixture::new().run(3, &["--experimental-tp8"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("power-of-two"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("ENGINE_EXECUTED"));
}
