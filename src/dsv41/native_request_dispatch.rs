//! PRIVATE bounded management-TCP admission for ONE resident TP8 request.
//! No tensor payload, per-token traffic or completion authority crosses this
//! channel. Native model/terminal proof still comes only from GPU/RDMA owners.
use super::{
    gpu_request_session::CheckedRequestCompletion,
    model_session_plan::{native_ids_sha256, CpuRequestPlan, RequestBinding},
    native_api_contract::{CpuBridge, PlannedChat, PlanningLimits},
    native_request_executor,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const MAX_FRAME: usize = 17 * 1024 * 1024;
const MAX_BODY: usize = 16 * 1024 * 1024;
const WALL: Duration = Duration::from_secs(180);
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn semantics(chat: &PlannedChat) -> Result<String> {
    Ok(sha(&serde_json::to_vec(chat.semantics())?))
}
fn deployment(b: &RequestBinding) -> Result<String> {
    Ok(sha(&serde_json::to_vec(&(
        &b.model_revision,
        &b.global_checkpoint_geometry_checksum_sha256,
        &b.execution_snapshot_sha256,
        &b.engine_binary_sha256,
        &b.tokenizer_sha256,
        &b.toolkit_revision,
        &b.frequency,
    ))?))
}
fn remaining(deadline: Instant) -> Result<Duration> {
    let d = deadline
        .checked_duration_since(Instant::now())
        .context("management dispatch deadline")?;
    ensure!(!d.is_zero(), "management dispatch deadline");
    Ok(d)
}
fn write_bytes(s: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        s.set_write_timeout(Some(remaining(deadline)?))?;
        match s.write(bytes) {
            Ok(0) => anyhow::bail!("management writeEOF"),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn read_bytes(s: &mut TcpStream, mut bytes: &mut [u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        s.set_read_timeout(Some(remaining(deadline)?))?;
        match s.read(bytes) {
            Ok(0) => anyhow::bail!("management readEOF"),
            Ok(n) => {
                let (_, tail) = bytes.split_at_mut(n);
                bytes = tail;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn send(s: &mut TcpStream, value: &Frame, deadline: Instant) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "management frame bound");
    write_bytes(s, &(bytes.len() as u32).to_le_bytes(), deadline)?;
    write_bytes(s, &bytes, deadline)
}
fn receive(s: &mut TcpStream, deadline: Instant) -> Result<Frame> {
    let mut n = [0; 4];
    read_bytes(s, &mut n, deadline)?;
    let n = u32::from_le_bytes(n) as usize;
    ensure!(n > 0 && n <= MAX_FRAME, "management frame size");
    let mut bytes = vec![0; n];
    read_bytes(s, &mut bytes, deadline)?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[derive(Clone, Debug, Serialize)]
pub struct DispatchConfig {
    pub peers: [Ipv4Addr; 8],
    pub port: u16,
    pub request_release_base: u16,
    pub maximum_requests: usize,
    pub idle_timeout_millis: u64,
    pub boot: Uuid,
}
impl DispatchConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.boot.is_nil()
                && (1..=1024).contains(&self.maximum_requests)
                && (20000..=64000).contains(&self.port)
                && (1..=3_600_000).contains(&self.idle_timeout_millis),
            "bounded private resident dispatch policy"
        );
        super::lab_network::validate_management_peers(&self.peers)?;
        let end = (self.request_release_base as usize)
            .checked_add(self.maximum_requests)
            .context("release port overflow")?;
        ensure!(
            self.request_release_base >= 20000
                && end <= 64001
                && !(self.request_release_base as usize..end).contains(&(self.port as usize)),
            "fresh bounded per-request release ports conflict"
        );
        Ok(())
    }
    pub fn release_port(&self, sequence: usize) -> Result<u16> {
        ensure!(
            (1..=self.maximum_requests).contains(&sequence),
            "resident request count bound; controlled restart required"
        );
        Ok(self.request_release_base + (sequence - 1) as u16)
    }
    fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(sha(&serde_json::to_vec(self)?))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkIdentity {
    schema: u32,
    rank: usize,
    boot: Uuid,
    config_sha256: String,
    deployment_sha256: String,
    previous_plan_sha256: String,
    previous_program_sha256: String,
    previous_generated_sha256: String,
    actual_epoch: u64,
    next_operation_id: u64,
    next_step_id: u64,
}
fn link(
    config: &DispatchConfig,
    rank: usize,
    previous: &CheckedRequestCompletion,
) -> Result<LinkIdentity> {
    ensure!(
        rank < 8
            && previous.rank() == rank
            && previous.lane().boot == config.boot
            && previous.all_rank_prefix_agreement()
            && previous.transport_lifetime_release_confirmed(),
        "actual checked bootstrap/previous request required"
    );
    Ok(LinkIdentity {
        schema: 1,
        rank,
        boot: config.boot,
        config_sha256: config.digest()?,
        deployment_sha256: deployment(previous.binding())?,
        previous_plan_sha256: previous.plan_sha256().into(),
        previous_program_sha256: previous.program_sha256().into(),
        previous_generated_sha256: previous.generated_sha256().into(),
        actual_epoch: previous.actual_epoch(),
        next_operation_id: previous.next_operation_id(),
        next_step_id: previous.next_step_id(),
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Announcement {
    identity: LinkIdentity,
    sequence: usize,
    response_id: String,
    body: Value,
    prompt_native_ids: Vec<u32>,
    semantics_sha256: String,
    plan_sha256: String,
    admission_elapsed_micros: u64,
    wall_millis: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    rank: usize,
    sequence: usize,
    response_id: String,
    plan_sha256: String,
    prompt_sha256: String,
    semantics_sha256: String,
    original_request_sha256: String,
}
fn prepared(
    rank: usize,
    sequence: usize,
    response_id: &str,
    chat: &PlannedChat,
    plan: &CpuRequestPlan,
) -> Result<Prepared> {
    Ok(Prepared {
        rank,
        sequence,
        response_id: response_id.into(),
        plan_sha256: plan.digest().into(),
        prompt_sha256: native_ids_sha256(chat.prompt_ids()),
        semantics_sha256: semantics(chat)?,
        original_request_sha256: chat.semantics().original_request_sha256.clone(),
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Release {
    prepared: Prepared,
    admission_elapsed_micros: u64,
    wall_millis: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Finished {
    rank: usize,
    sequence: usize,
    plan_sha256: String,
    program_sha256: String,
    input_sha256: String,
    generated_sha256: String,
    actual_epoch: u64,
    next_operation_id: u64,
    next_step_id: u64,
}
fn finished(rank: usize, sequence: usize, c: &CheckedRequestCompletion) -> Result<Finished> {
    ensure!(
        c.rank() == rank
            && c.all_rank_prefix_agreement()
            && c.transport_lifetime_release_confirmed(),
        "actual native completion is required before management DONE"
    );
    Ok(Finished {
        rank,
        sequence,
        plan_sha256: c.plan_sha256().into(),
        program_sha256: c.program_sha256().into(),
        input_sha256: c.input_sha256().into(),
        generated_sha256: c.generated_sha256().into(),
        actual_epoch: c.actual_epoch(),
        next_operation_id: c.next_operation_id(),
        next_step_id: c.next_step_id(),
    })
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", deny_unknown_fields)]
enum Frame {
    Hello(LinkIdentity),
    Announce(Announcement),
    Prepared(Prepared),
    Release(Release),
    Finished(Finished),
    Continue {
        sequence: usize,
        generated_sha256: String,
    },
    Stop {
        reason: String,
    },
}

/// Single native owner thread owns this coordinator and all peer connections.
/// A method failure poisons the channel; caller tears down all8 workers. No
/// deserialized frame can create a model/terminal or normal-response receipt.
pub struct HeadDispatch {
    config: DispatchConfig,
    peers: Vec<(usize, TcpStream)>,
    sequence: usize,
    active: Option<Prepared>,
    failed: bool,
}
impl HeadDispatch {
    pub fn bind(config: DispatchConfig, previous: &CheckedRequestCompletion) -> Result<Self> {
        config.validate()?;
        let expected = link(&config, 0, previous)?;
        let listener = TcpListener::bind((config.peers[0], config.port))?;
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut peers = Vec::with_capacity(7);
        let mut seen = BTreeSet::new();
        while peers.len() < 7 {
            remaining(deadline)?;
            match listener.accept() {
                Ok((mut s, address)) => {
                    let hello = match receive(&mut s, deadline)? {
                        Frame::Hello(v) => v,
                        _ => anyhow::bail!("expected resident worker HELLO"),
                    };
                    let r = hello.rank;
                    ensure!(
                        (1..8).contains(&r) && seen.insert(r) && address.ip() == config.peers[r],
                        "wrong/duplicate physical resident peer"
                    );
                    let mut wanted = expected.clone();
                    wanted.rank = r;
                    ensure!(
                        hello == wanted,
                        "resident peer bootstrap/deployment identity differs"
                    );
                    peers.push((r, s));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Err(e) => return Err(e.into()),
            }
        }
        peers.sort_by_key(|x| x.0);
        Ok(Self {
            config,
            peers,
            sequence: 0,
            active: None,
            failed: false,
        })
    }
    pub fn begin(
        &mut self,
        chat: &PlannedChat,
        plan: &CpuRequestPlan,
        previous: &CheckedRequestCompletion,
        response_id: &str,
        admitted: Instant,
    ) -> Result<usize> {
        let result = (|| {
            ensure!(
                !self.failed && self.active.is_none() && !response_id.is_empty(),
                "dispatch failed/busy or empty responseID"
            );
            let sequence = self
                .sequence
                .checked_add(1)
                .context("request sequence overflow")?;
            self.config.release_port(sequence)?;
            ensure!(
                plan.binding().budget.wall_millis == 180000,
                "full request wall policy changed"
            );
            let deadline = admitted + WALL;
            remaining(deadline)?;
            let announce = Announcement {
                identity: link(&self.config, 0, previous)?,
                sequence,
                response_id: response_id.into(),
                body: chat.original_request().clone(),
                prompt_native_ids: chat.prompt_ids().to_vec(),
                semantics_sha256: semantics(chat)?,
                plan_sha256: plan.digest().into(),
                admission_elapsed_micros: admitted.elapsed().as_micros().try_into()?,
                wall_millis: 180000,
            };
            ensure!(
                serde_json::to_vec(&announce.body)?.len() <= MAX_BODY,
                "request body bound"
            );
            for (_, s) in &mut self.peers {
                send(s, &Frame::Announce(announce.clone()), deadline)?;
            }
            let wanted = prepared(0, sequence, response_id, chat, plan)?;
            for (r, s) in &mut self.peers {
                let got = match receive(s, deadline)? {
                    Frame::Prepared(v) => v,
                    _ => anyhow::bail!("resident worker did not PREPARE exact request"),
                };
                let mut expected = wanted.clone();
                expected.rank = *r;
                ensure!(
                    got == expected,
                    "resident worker rendered/encoded/planned different request"
                );
            }
            let elapsed: u64 = admitted.elapsed().as_micros().try_into()?;
            for (r, s) in &mut self.peers {
                let mut p = wanted.clone();
                p.rank = *r;
                send(
                    s,
                    &Frame::Release(Release {
                        prepared: p,
                        admission_elapsed_micros: elapsed,
                        wall_millis: 180000,
                    }),
                    deadline,
                )?;
            }
            self.sequence = sequence;
            self.active = Some(wanted);
            Ok(sequence)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn finish(
        &mut self,
        completion: &CheckedRequestCompletion,
        admitted: Instant,
    ) -> Result<()> {
        let result = (|| {
            ensure!(!self.failed, "failed resident dispatch");
            let active = self.active.as_ref().context("no active dispatch")?;
            let deadline = admitted + WALL;
            remaining(deadline)?;
            let local = finished(0, self.sequence, completion)?;
            ensure!(
                local.plan_sha256 == active.plan_sha256,
                "actual native completion belongs to another dispatched request"
            );
            for (r, s) in &mut self.peers {
                let got = match receive(s, deadline)? {
                    Frame::Finished(v) => v,
                    _ => anyhow::bail!("resident peer completion control missing"),
                };
                let mut expected = local.clone();
                expected.rank = *r;
                ensure!(
                    got == expected,
                    "resident peer actual native completion control differs"
                );
            }
            for (_, s) in &mut self.peers {
                send(
                    s,
                    &Frame::Continue {
                        sequence: self.sequence,
                        generated_sha256: local.generated_sha256.clone(),
                    },
                    deadline,
                )?;
            }
            self.active = None;
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn release_port(&self, sequence: usize) -> Result<u16> {
        self.config.release_port(sequence)
    }
    pub fn poison(&mut self) {
        self.failed = true;
        for (_, s) in &mut self.peers {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
    }
    pub fn stop(mut self, reason: &str) -> Result<()> {
        ensure!(
            !self.failed && self.active.is_none(),
            "cannot graceful-stop unchecked native request"
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        for (_, s) in &mut self.peers {
            send(
                s,
                &Frame::Stop {
                    reason: reason.into(),
                },
                deadline,
            )?;
        }
        Ok(())
    }
}

/// Private-field admission token minted only after the real peer independently
/// prepares the exact original body, full native IDs/semantics/plan and release.
pub struct PeerRequest {
    chat: PlannedChat,
    plan: Arc<CpuRequestPlan>,
    response_id: String,
    sequence: usize,
    admitted: Instant,
}
impl PeerRequest {
    pub fn into_parts(self) -> (PlannedChat, Arc<CpuRequestPlan>, String, usize, Instant) {
        (
            self.chat,
            self.plan,
            self.response_id,
            self.sequence,
            self.admitted,
        )
    }
}
pub struct PeerDispatch {
    config: DispatchConfig,
    rank: usize,
    stream: TcpStream,
    sequence: usize,
    active: Option<Prepared>,
    failed: bool,
}
impl PeerDispatch {
    pub fn connect(
        config: DispatchConfig,
        rank: usize,
        previous: &CheckedRequestCompletion,
    ) -> Result<Self> {
        config.validate()?;
        ensure!((1..8).contains(&rank), "peer rank1..7");
        let deadline = Instant::now() + Duration::from_secs(30);
        let address = SocketAddr::from((config.peers[0], config.port));
        let mut stream = loop {
            remaining(deadline)?;
            match TcpStream::connect_timeout(&address, Duration::from_millis(500)) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(Duration::from_millis(25)),
            }
        };
        send(
            &mut stream,
            &Frame::Hello(link(&config, rank, previous)?),
            deadline,
        )?;
        Ok(Self {
            config,
            rank,
            stream,
            sequence: 0,
            active: None,
            failed: false,
        })
    }
    pub fn receive(
        &mut self,
        bridge: &CpuBridge,
        previous: &CheckedRequestCompletion,
    ) -> Result<Option<PeerRequest>> {
        let result = (|| {
            ensure!(
                !self.failed && self.active.is_none(),
                "resident peer failed/busy"
            );
            let idle = Instant::now() + Duration::from_millis(self.config.idle_timeout_millis);
            let a = match receive(&mut self.stream, idle)? {
                Frame::Announce(v) => v,
                Frame::Stop { .. } => return Ok(None),
                _ => anyhow::bail!("unexpected resident admission frame"),
            };
            ensure!(
                a.sequence == self.sequence + 1
                    && a.wall_millis == 180000
                    && !a.response_id.is_empty(),
                "resident admission sequence/policy"
            );
            self.config.release_port(a.sequence)?;
            let mut expected = link(&self.config, self.rank, previous)?;
            expected.rank = 0;
            ensure!(
                a.identity == expected,
                "head bootstrap/previous checked request differs"
            );
            ensure!(
                a.admission_elapsed_micros < 180_000_000,
                "request expired before peer preparation"
            );
            let received = Instant::now();
            let admitted = received
                .checked_sub(Duration::from_micros(a.admission_elapsed_micros))
                .context("remote admission clock underflow")?;
            let deadline = admitted + WALL;
            let limits = PlanningLimits {
                context_tokens: 8192,
                prompt_tokens: 8192,
                output_tokens: 4096,
                atomic_requests: 1,
                request_bytes: MAX_BODY,
            };
            let streaming = a.body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
            let chat = if streaming {
                bridge.prepare_stream(a.body, limits)?
            } else {
                bridge.prepare_single(a.body, limits)?
            };
            let plan = if streaming {
                native_request_executor::next_stream_plan(&chat, previous)?
            } else {
                native_request_executor::next_chat_plan(&chat, previous)?
            };
            ensure!(
                chat.prompt_ids() == a.prompt_native_ids
                    && semantics(&chat)? == a.semantics_sha256
                    && plan.digest() == a.plan_sha256,
                "peer original body/nativeIDs/semantics/plan mismatch"
            );
            let p = prepared(self.rank, a.sequence, &a.response_id, &chat, &plan)?;
            send(&mut self.stream, &Frame::Prepared(p.clone()), deadline)?;
            let release = match receive(&mut self.stream, deadline)? {
                Frame::Release(v) => v,
                _ => anyhow::bail!("missing exact resident release"),
            };
            ensure!(
                release.prepared == p
                    && release.wall_millis == 180000
                    && release.admission_elapsed_micros >= a.admission_elapsed_micros
                    && release.admission_elapsed_micros < 180_000_000,
                "resident release/request/clock mismatch"
            );
            // Monotonic Instants are process-local. Rank0's original admission
            // is the authoritative180s service clock. Peers carry its elapsed
            // duration (never restart180s); release-transit offset is not a
            // synchronized cross-host clock or benchmark measurement. Rank0's
            // own session/deadline must pass before any normal API response.
            let admitted = Instant::now()
                .checked_sub(Duration::from_micros(release.admission_elapsed_micros))
                .context("remote release clock underflow")?;
            self.sequence = a.sequence;
            self.active = Some(p);
            Ok(Some(PeerRequest {
                chat,
                plan,
                response_id: a.response_id,
                sequence: a.sequence,
                admitted,
            }))
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn finish(&mut self, c: &CheckedRequestCompletion, admitted: Instant) -> Result<()> {
        let result = (|| {
            ensure!(!self.failed, "resident peer poisoned");
            let active = self.active.as_ref().context("no active peer request")?;
            let deadline = admitted + WALL;
            remaining(deadline)?;
            let value = finished(self.rank, self.sequence, c)?;
            ensure!(
                value.plan_sha256 == active.plan_sha256,
                "peer completion is for another request"
            );
            send(&mut self.stream, &Frame::Finished(value.clone()), deadline)?;
            match receive(&mut self.stream, deadline)? {
                Frame::Continue {
                    sequence,
                    generated_sha256,
                } => ensure!(
                    sequence == self.sequence && generated_sha256 == value.generated_sha256,
                    "peer next-request release differs"
                ),
                _ => anyhow::bail!("peer release missing"),
            };
            self.active = None;
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn release_port(&self, sequence: usize) -> Result<u16> {
        self.config.release_port(sequence)
    }
    pub fn poison(&mut self) {
        self.failed = true;
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> DispatchConfig {
        DispatchConfig {
            peers: std::array::from_fn(|r| Ipv4Addr::new(172, 31, 240, 11 + r as u8)),
            port: 47430,
            request_release_base: 50000,
            maximum_requests: 1024,
            idle_timeout_millis: 900000,
            boot: Uuid::from_u128(214),
        }
    }
    #[test]
    fn bounded_exact_topology_and_distinct_request_release_ports() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(c.release_port(1).unwrap(), 50000);
        assert_eq!(c.release_port(573).unwrap(), 50572);
        assert_eq!(c.release_port(1024).unwrap(), 51023);
        assert!(c.release_port(0).is_err());
        assert!(c.release_port(1025).is_err());
        for mutate in [0, 1, 2, 3, 4, 5] {
            let mut bad = config();
            match mutate {
                0 => bad.peers[7] = bad.peers[0],
                1 => bad.peers[0] = Ipv4Addr::UNSPECIFIED,
                2 => bad.maximum_requests = 1025,
                3 => bad.request_release_base = 63500,
                4 => bad.request_release_base = 47430,
                _ => bad.boot = Uuid::nil(),
            };
            assert!(bad.validate().is_err());
        }
    }
    fn frame() -> Frame {
        Frame::Prepared(Prepared {
            rank: 1,
            sequence: 7,
            response_id: "chatcmpl-native-test".into(),
            plan_sha256: "a".repeat(64),
            prompt_sha256: "b".repeat(64),
            semantics_sha256: "c".repeat(64),
            original_request_sha256: "d".repeat(64),
        })
    }
    #[test]
    fn typed_control_frames_reject_unknowns_duplicates_and_bool_native_integer_aliases() {
        let v = serde_json::to_value(frame()).unwrap();
        let bytes = serde_json::to_vec(&v).unwrap();
        assert!(matches!(
            serde_json::from_slice::<Frame>(&bytes).unwrap(),
            Frame::Prepared(_)
        ));
        for (field, value) in [
            ("rank", serde_json::json!(true)),
            ("sequence", serde_json::json!(false)),
            ("rank", serde_json::json!(-1)),
            ("sequence", serde_json::json!(1.5)),
        ] {
            let mut bad = v.clone();
            bad["value"][field] = value;
            assert!(serde_json::from_value::<Frame>(bad).is_err());
        }
        let mut bad = v.clone();
        bad["value"]["extra"] = serde_json::json!(1);
        assert!(serde_json::from_value::<Frame>(bad).is_err());
        let mut bad = v.clone();
        bad["outside"] = serde_json::json!(1);
        assert!(serde_json::from_value::<Frame>(bad).is_err());
        assert!(serde_json::from_str::<Frame>(
            "{\"kind\":\"Prepared\",\"kind\":\"Release\",\"value\":{}}"
        )
        .is_err());
        assert!(serde_json::from_str::<Frame>(
            "{\"kind\":\"CheckedRequestCompletion\",\"value\":{\"passed\":true}}"
        )
        .is_err());
    }
    #[test]
    fn control_announcement_keeps_exact_original_body_and_native_ids_as_verification_only() {
        let body = serde_json::json!({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"café 中文"}],"tools":[{"type":"function","function":{"name":"fixture","parameters":{"type":"object","properties":{"nested":{"type":"object"},"optional":{"type":["null","string"]}}}}}],"temperature":0,"top_p":1,"max_tokens":4096,"chat_template_kwargs":{"thinking":false}});
        let a = Announcement {
            identity: LinkIdentity {
                schema: 1,
                rank: 0,
                boot: Uuid::from_u128(214),
                config_sha256: "a".repeat(64),
                deployment_sha256: "b".repeat(64),
                previous_plan_sha256: "c".repeat(64),
                previous_program_sha256: "d".repeat(64),
                previous_generated_sha256: "e".repeat(64),
                actual_epoch: 300,
                next_operation_id: 1000,
                next_step_id: 90,
            },
            sequence: 1,
            response_id: "chatcmpl-native-test".into(),
            body: body.clone(),
            prompt_native_ids: vec![17, 24, 129279],
            semantics_sha256: "f".repeat(64),
            plan_sha256: "9".repeat(64),
            admission_elapsed_micros: 9_000_000,
            wall_millis: 180000,
        };
        let value = serde_json::to_value(Frame::Announce(a)).unwrap();
        let f: Frame = serde_json::from_value(value.clone()).unwrap();
        let a = match f {
            Frame::Announce(a) => a,
            _ => panic!(),
        };
        assert_eq!(a.body, body);
        assert_eq!(a.prompt_native_ids, vec![17, 24, 129279]);
        assert_eq!(a.wall_millis, 180000);
        let mut bad = value;
        bad["value"]["prompt_native_ids"][0] = serde_json::json!(true);
        assert!(serde_json::from_value::<Frame>(bad).is_err());
    }
    #[test]
    fn elapsed_clock_never_silently_supplies_a_new_full_budget() {
        let now = Instant::now();
        let admitted = now - Duration::from_secs(37);
        let deadline = admitted + WALL;
        assert!(deadline.duration_since(now) <= Duration::from_secs(143));
        assert!(remaining(now - Duration::from_millis(1)).is_err());
        // This is process-local accounting, deliberately no claim that two
        // machines' Instant or release-transit offsets are synchronized.
    }
}
