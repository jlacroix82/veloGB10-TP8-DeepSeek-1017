//! PRIVATE bounded TCP all-rank release AFTER RDMA prefix agreement. This is
//! transport lifetime safety, not model/quality/terminal authority.
use super::request_session_state::TransportPrefixClaim;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream},
    time::{Duration, Instant},
};
const MAX_FRAME: usize = 4096;
#[derive(Clone, Debug, Serialize)]
pub struct RequestReleaseConfig {
    pub peers: [Ipv4Addr; 8],
    pub port: u16,
    pub timeout_millis: u64,
}
impl RequestReleaseConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (20000..=64000).contains(&self.port)
                && (1..=30000).contains(&self.timeout_millis)
                && self.peers.iter().copied().collect::<BTreeSet<_>>().len() == 8
                && self
                    .peers
                    .iter()
                    .all(|p| !p.is_unspecified() && !p.is_multicast() && !p.is_broadcast()),
            "private release endpoint/deadline geometry"
        );
        Ok(())
    }
    fn identity(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    kind: String,
    rank: u32,
    recipient: u32,
    world: u32,
    program_sha256: String,
    claim_sha256: String,
    endpoint_config_sha256: String,
}
fn remaining(deadline: Instant) -> Result<Duration> {
    let d = deadline
        .checked_duration_since(Instant::now())
        .context("request all-rank release deadline")?;
    ensure!(!d.is_zero(), "release deadline");
    Ok(d)
}
fn write_bytes(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        match stream.write(bytes) {
            Ok(0) => anyhow::bail!("release socket writeEOF"),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn read_bytes(stream: &mut TcpStream, mut bytes: &mut [u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        match stream.read(bytes) {
            Ok(0) => anyhow::bail!("release socket readEOF"),
            Ok(n) => {
                let (_, rest) = bytes.split_at_mut(n);
                bytes = rest;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn send(stream: &mut TcpStream, value: &Envelope, deadline: Instant) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "release frame bound");
    write_bytes(stream, &(bytes.len() as u32).to_le_bytes(), deadline)?;
    write_bytes(stream, &bytes, deadline)
}
fn receive(stream: &mut TcpStream, deadline: Instant) -> Result<Envelope> {
    let mut length = [0u8; 4];
    read_bytes(stream, &mut length, deadline)?;
    let n = u32::from_le_bytes(length) as usize;
    ensure!(n > 0 && n <= MAX_FRAME, "release frame length");
    let mut data = vec![0u8; n];
    read_bytes(stream, &mut data, deadline)?;
    Ok(serde_json::from_slice(&data)?)
}
fn check(packet: &Envelope, expected: &Envelope) -> Result<()> {
    ensure!(
        packet == expected,
        "request release identity/phase/rank/claim mismatch"
    );
    Ok(())
}
#[derive(Clone, Debug, Serialize)]
pub struct RequestReleaseReceipt {
    pub rank: usize,
    pub world: usize,
    pub program_sha256: String,
    pub claim_sha256: String,
    pub all_eight_RDMA_finish_arrivals_confirmed: bool,
    pub actual_TCP_release_completed: bool,
    pub model_execution_or_quality_proven: bool,
    pub elapsed_seconds: f64,
}
pub fn release(
    rank: usize,
    config: &RequestReleaseConfig,
    program_sha256: &str,
    claim: &TransportPrefixClaim,
) -> Result<RequestReleaseReceipt> {
    config.validate()?;
    ensure!(rank < 8, "release rank");
    let began = Instant::now();
    let deadline = began + Duration::from_millis(config.timeout_millis);
    let claim_sha = claim.digest()?;
    let base = Envelope {
        schema_version: 1,
        kind: "arrive".into(),
        rank: rank as u32,
        recipient: 0,
        world: 8,
        program_sha256: program_sha256.into(),
        claim_sha256: claim_sha.clone(),
        endpoint_config_sha256: config.identity()?,
    };
    if rank == 0 {
        let listener = TcpListener::bind(SocketAddrV4::new(config.peers[0], config.port))?;
        listener.set_nonblocking(true)?;
        let mut peers = Vec::with_capacity(7);
        let mut seen = BTreeSet::new();
        while peers.len() < 7 {
            remaining(deadline)?;
            match listener.accept() {
                Ok((mut stream, address)) => {
                    let packet = receive(&mut stream, deadline)?;
                    let source = packet.rank as usize;
                    ensure!(
                        (1..8).contains(&source) && seen.insert(source),
                        "duplicate/outside request release rank"
                    );
                    ensure!(
                        address.ip() == std::net::IpAddr::V4(config.peers[source]),
                        "release physical peerIPv4 differs"
                    );
                    let mut expected = base.clone();
                    expected.rank = source as u32;
                    check(&packet, &expected)?;
                    peers.push((source, stream));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
                Err(e) => return Err(e.into()),
            }
        }
        // Every peer sends ARRIVE only after its actual RDMA prefix agreement
        // returned. Therefore no rank still depends on that control exchange.
        for (source, mut stream) in peers {
            let mut ack = base.clone();
            ack.kind = "release".into();
            ack.recipient = source as u32;
            send(&mut stream, &ack, deadline)?;
        }
    } else {
        let address = SocketAddr::V4(SocketAddrV4::new(config.peers[0], config.port));
        let mut stream = loop {
            let left = remaining(deadline)?;
            match TcpStream::connect_timeout(&address, left.min(Duration::from_millis(250))) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        };
        send(&mut stream, &base, deadline)?;
        let ack = receive(&mut stream, deadline)?;
        let mut expected = base.clone();
        expected.kind = "release".into();
        expected.rank = 0;
        expected.recipient = rank as u32;
        check(&ack, &expected)?;
    }
    remaining(deadline)?;
    Ok(RequestReleaseReceipt {
        rank,
        world: 8,
        program_sha256: program_sha256.into(),
        claim_sha256: claim_sha,
        all_eight_RDMA_finish_arrivals_confirmed: true,
        actual_TCP_release_completed: true,
        model_execution_or_quality_proven: false,
        elapsed_seconds: began.elapsed().as_secs_f64(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_release_messages_reject_cross_phase_rank_and_hash_without_network() {
        let base = Envelope {
            schema_version: 1,
            kind: "arrive".into(),
            rank: 4,
            recipient: 0,
            world: 8,
            program_sha256: "a".repeat(64),
            claim_sha256: "b".repeat(64),
            endpoint_config_sha256: "c".repeat(64),
        };
        check(&base, &base).unwrap();
        for i in 0..6 {
            let mut bad = base.clone();
            match i {
                0 => bad.kind = "release".into(),
                1 => bad.rank = 3,
                2 => bad.recipient = 4,
                3 => bad.world = 4,
                4 => bad.claim_sha256 = "d".repeat(64),
                _ => bad.endpoint_config_sha256 = "d".repeat(64),
            };
            assert!(check(&bad, &base).is_err());
        }
        let bytes = serde_json::to_string(&base).unwrap();
        assert!(
            serde_json::from_str::<Envelope>(&bytes.replace("\"rank\":4", "\"rank\":true"))
                .is_err()
        );
        assert!(serde_json::from_str::<Envelope>(&bytes.replacen('{', "{\"rank\":1,", 1)).is_err());
    }
}
