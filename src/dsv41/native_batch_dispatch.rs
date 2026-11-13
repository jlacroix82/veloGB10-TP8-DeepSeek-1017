//! Private bounded management admission for a resident atomic TP8 group.
//! Only native tensors use the qualified RoCE link. This socket carries exact
//! original bodies/plans and final common hashes; JSON cannot mint a GPU token.
use super::{
    atomic_request_admission::{
        AdmissionClock, CpuAtomicAdmission, CpuOwnedAtomicGroup, CpuSlotBook,
        PreparedAtomicRequest, ProposedBatchDeployment,
    },
    batch_request_plan::{digest, BatchGroupBinding, BatchPhasePlan},
    gpu_batch_request_session::CheckedBatchRequestCompletion,
    native_api_contract::{CpuBridge, ATOMIC_ROUTE},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream},
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;
const MAX_FRAME: usize = 2 * 1024 * 1024;
const WALL_MILLIS: u64 = 180000;
#[derive(Clone, Debug, Serialize)]
pub struct BatchDispatchConfig {
    pub peers: [Ipv4Addr; 8],
    pub port: u16,
    pub release_base: u16,
    pub maximum_groups: usize,
    pub boot: Uuid,
    pub service_deadline: InstantMetadata,
}
/// The absolute local process deadline is never serialized as a cross-host
/// clock. This DTO records only the explicit finite idle policy.
#[derive(Clone, Debug, Serialize)]
pub struct InstantMetadata {
    pub idle_millis: u64,
}
impl BatchDispatchConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.boot.is_nil()
                && (1..=32).contains(&self.maximum_groups)
                && (1..=1_200_000).contains(&self.service_deadline.idle_millis)
                && (20000..=64000).contains(&self.port),
            "bounded private resident batch dispatch required"
        );
        super::lab_network::validate_management_peers(&self.peers)?;
        let end = self.release_base as usize + self.maximum_groups * 20;
        ensure!(
            self.release_base >= 20000
                && end <= 64001
                && !(self.release_base as usize..end).contains(&(self.port as usize)),
            "all reserved fresh atomic group phase/cleanup ports must fit without overlap"
        );
        Ok(())
    }
    pub fn release_port(&self, sequence: usize) -> Result<u16> {
        self.validate()?;
        ensure!(
            (1..=self.maximum_groups).contains(&sequence),
            "bounded group count exhausted; controlled restart required"
        );
        Ok(self.release_base + ((sequence - 1) * 20) as u16)
    }
    fn hash(&self) -> Result<String> {
        digest(b"dsv41-atomic-management-config-v1\0", self)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLane {
    slot: usize,
    generation: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommonCompletion {
    group_sha256: String,
    final_program_sha256: String,
    final_claim_sha256: String,
    epoch: u64,
    next_operation: u64,
    next_step: u64,
    next_batch_version: u64,
    cleanup_lanes: Vec<WireLane>,
    model_steps: usize,
    model_operations: usize,
}
fn common(c: &CheckedBatchRequestCompletion) -> Result<CommonCompletion> {
    ensure!(
        c.all_rank_terminal_agreement()
            && c.transport_lifetime_released()
            && c.fresh_cleanup_checked(),
        "management cannot substitute for actual opaque all8 terminal/reset"
    );
    Ok(CommonCompletion {
        group_sha256: c.binding().digest().into(),
        final_program_sha256: c.final_program_sha256().into(),
        final_claim_sha256: c.final_claim().digest()?,
        epoch: c.actual_epoch(),
        next_operation: c.next_operation_id(),
        next_step: c.next_step_id(),
        next_batch_version: c.next_batch_version(),
        cleanup_lanes: c
            .cleanup_reset()
            .lanes()
            .iter()
            .map(|k| WireLane {
                slot: k.slot,
                generation: k.generation,
            })
            .collect(),
        model_steps: c.completed_model_steps(),
        model_operations: c.completed_operations(),
    })
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "phase", deny_unknown_fields)]
enum Frame {
    Hello {
        rank: usize,
        boot: Uuid,
        config_sha256: String,
        deployment_sha256: String,
        previous: CommonCompletion,
    },
    HelloAccepted {
        rank: usize,
        config_sha256: String,
        previous: CommonCompletion,
    },
    Prepare {
        sequence: usize,
        response_id: String,
        body: Value,
        lanes: Vec<WireLane>,
        head_elapsed_millis: u64,
        wall_millis: u64,
        group_sha256: String,
        phase_sha256: String,
        program_sha256: String,
        previous: CommonCompletion,
    },
    Prepared {
        rank: usize,
        sequence: usize,
        group_sha256: String,
        phase_sha256: String,
        program_sha256: String,
    },
    Execute {
        sequence: usize,
        group_sha256: String,
    },
    Finished {
        rank: usize,
        sequence: usize,
        completion: CommonCompletion,
    },
    ReuseReleased {
        sequence: usize,
        completion: CommonCompletion,
    },
    Stop {
        previous: CommonCompletion,
    },
    Stopped {
        rank: usize,
        previous: CommonCompletion,
    },
}
fn remaining(deadline: Instant) -> Result<Duration> {
    let d = deadline
        .checked_duration_since(Instant::now())
        .context("atomic dispatch deadline")?;
    ensure!(!d.is_zero(), "atomic dispatch deadline");
    Ok(d)
}
fn write_bytes(s: &mut TcpStream, mut data: &[u8], deadline: Instant) -> Result<()> {
    while !data.is_empty() {
        s.set_write_timeout(Some(remaining(deadline)?))?;
        match s.write(data) {
            Ok(0) => anyhow::bail!("batch dispatch writeEOF"),
            Ok(n) => data = &data[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn read_bytes(s: &mut TcpStream, mut data: &mut [u8], deadline: Instant) -> Result<()> {
    while !data.is_empty() {
        s.set_read_timeout(Some(remaining(deadline)?))?;
        match s.read(data) {
            Ok(0) => anyhow::bail!("batch dispatch readEOF"),
            Ok(n) => {
                let (_, rest) = data.split_at_mut(n);
                data = rest;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn send(s: &mut TcpStream, f: &Frame, deadline: Instant) -> Result<()> {
    let data = serde_json::to_vec(f)?;
    ensure!(data.len() <= MAX_FRAME, "bounded atomic frame");
    write_bytes(s, &(data.len() as u32).to_le_bytes(), deadline)?;
    write_bytes(s, &data, deadline)
}
fn receive(s: &mut TcpStream, deadline: Instant) -> Result<Frame> {
    let mut n = [0; 4];
    read_bytes(s, &mut n, deadline)?;
    let n = u32::from_le_bytes(n) as usize;
    ensure!(n > 0 && n <= MAX_FRAME, "atomic frame size");
    let mut data = vec![0; n];
    read_bytes(s, &mut data, deadline)?;
    Ok(serde_json::from_slice(&data)?)
}
fn deployment(c: &CheckedBatchRequestCompletion) -> Result<ProposedBatchDeployment> {
    let m = c.model_identity();
    let r = m.runtime.as_ref().context("resident runtime missing")?;
    Ok(ProposedBatchDeployment {
        boot: m.boot,
        global_checkpoint_geometry_checksum_sha256: m.global_checkpoint_geometry_binding.clone(),
        execution_snapshot_sha256: r.execution_snapshot_sha256.clone(),
        engine_binary_sha256: r.engine_binary_sha256.clone(),
        frequency: m.frequency.clone(),
    })
}
fn deployment_hash(c: &CheckedBatchRequestCompletion) -> Result<String> {
    digest(b"dsv41-atomic-resident-deployment-v1\0", &deployment(c)?)
}
pub struct PreparedBatchWave {
    pub(crate) owned: CpuOwnedAtomicGroup,
    pub(crate) binding: Arc<BatchGroupBinding>,
    pub(crate) plan: Arc<BatchPhasePlan>,
    pub(crate) admitted: Instant,
    pub(crate) sequence: usize,
}
impl PreparedBatchWave {
    pub fn prepare(
        prepared: PreparedAtomicRequest,
        book: &mut CpuSlotBook,
        deployment: ProposedBatchDeployment,
        lanes: Vec<LaneKey>,
        response_id: String,
        admitted: Instant,
        sequence: usize,
        previous: &CheckedBatchRequestCompletion,
    ) -> Result<Self> {
        ensure!(
            sequence >= 2
                && prepared
                    .members()
                    .iter()
                    .all(|m| m.semantics().output_allowance <= 256),
            "original full allowance must fit the explicit256 runtime profile before slot effects"
        );
        let now = Instant::now();
        let clock = AdmissionClock::new(admitted, admitted + Duration::from_millis(WALL_MILLIS))?;
        let mut transaction =
            CpuAtomicAdmission::new(book, prepared, deployment, lanes, clock, response_id, now)?;
        transaction.commit(now)?;
        let owned = transaction.take_published(now)?;
        let result = (|| {
            let binding = BatchGroupBinding::from_cpu(&owned, book, now)?;
            let plan = BatchPhasePlan::first(
                binding.clone(),
                previous.next_operation_id(),
                previous.next_step_id(),
                previous.actual_epoch(),
                previous.next_batch_version(),
            )?;
            Ok((binding, plan))
        })();
        match result {
            Ok((binding, plan)) => Ok(Self {
                owned,
                binding,
                plan,
                admitted,
                sequence,
            }),
            Err(error) => {
                book.release_cpu_ownership(&owned)?;
                Err(error)
            }
        }
    }
    pub fn plan(&self) -> &Arc<BatchPhasePlan> {
        &self.plan
    }
    pub fn binding(&self) -> &Arc<BatchGroupBinding> {
        &self.binding
    }
}
pub struct HeadBatchDispatch {
    config: BatchDispatchConfig,
    peers: Vec<(usize, TcpStream)>,
    next: usize,
}
impl HeadBatchDispatch {
    pub fn bind(
        config: BatchDispatchConfig,
        previous: &CheckedBatchRequestCompletion,
        deadline: Instant,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(previous.model_identity().rank == 0, "head rank0 required");
        let listener = TcpListener::bind(SocketAddrV4::new(config.peers[0], config.port))?;
        listener.set_nonblocking(true)?;
        let mut peers = Vec::new();
        let mut seen = BTreeSet::new();
        while peers.len() < 7 {
            remaining(deadline)?;
            match listener.accept() {
                Ok((mut s, address)) => {
                    s.set_nodelay(true)?;
                    let Frame::Hello {
                        rank,
                        boot,
                        config_sha256,
                        deployment_sha256,
                        previous: old,
                    } = receive(&mut s, deadline)?
                    else {
                        anyhow::bail!("expected atomic peer Hello")
                    };
                    ensure!(
                        (1..8).contains(&rank)
                            && seen.insert(rank)
                            && address.ip() == std::net::IpAddr::V4(config.peers[rank])
                            && boot == config.boot
                            && config_sha256 == config.hash()?
                            && deployment_sha256 == deployment_hash(previous)?
                            && old == common(previous)?,
                        "physical atomic peer/deployment/bootstrap differs"
                    );
                    send(
                        &mut s,
                        &Frame::HelloAccepted {
                            rank,
                            config_sha256,
                            previous: old,
                        },
                        deadline,
                    )?;
                    peers.push((rank, s));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
                Err(e) => return Err(e.into()),
            }
        }
        peers.sort_by_key(|p| p.0);
        Ok(Self {
            config,
            peers,
            next: 2,
        })
    }
    pub fn begin(
        &mut self,
        wave: &PreparedBatchWave,
        previous: &CheckedBatchRequestCompletion,
    ) -> Result<()> {
        self.config.release_port(wave.sequence)?;
        ensure!(
            wave.sequence == self.next,
            "atomic management sequence reuse"
        );
        let deadline = wave.admitted + Duration::from_millis(WALL_MILLIS);
        let program = super::gpu_collectives::Program::new_batch_phase(wave.plan.clone())?;
        let program_sha = program
            .digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let lanes = wave
            .binding
            .lanes()
            .iter()
            .map(|l| WireLane {
                slot: l.slot,
                generation: l.generation,
            })
            .collect::<Vec<_>>();
        for (_, peer) in &mut self.peers {
            send(
                peer,
                &Frame::Prepare {
                    sequence: wave.sequence,
                    response_id: wave.binding.response_id().into(),
                    body: wave.owned.prepared().original_request().clone(),
                    lanes: lanes.clone(),
                    head_elapsed_millis: wave.admitted.elapsed().as_millis().try_into()?,
                    wall_millis: WALL_MILLIS,
                    group_sha256: wave.binding.digest().into(),
                    phase_sha256: wave.plan.digest().into(),
                    program_sha256: program_sha.clone(),
                    previous: common(previous)?,
                },
                deadline,
            )?;
        }
        for (rank, peer) in &mut self.peers {
            let Frame::Prepared {
                rank: actual,
                sequence,
                group_sha256,
                phase_sha256,
                program_sha256,
            } = receive(peer, deadline)?
            else {
                anyhow::bail!("peer has not independently prepared entire atomic group")
            };
            ensure!(
                actual == *rank
                    && sequence == wave.sequence
                    && group_sha256 == wave.binding.digest()
                    && phase_sha256 == wave.plan.digest()
                    && program_sha256 == program_sha,
                "rank{rank} ordered group/native IDs/full budgets/phase program differs"
            );
        }
        // No peer may reset a lane or enqueue model work before every member
        // on every rank has prepared the exact complete group successfully.
        for (_, peer) in &mut self.peers {
            send(
                peer,
                &Frame::Execute {
                    sequence: wave.sequence,
                    group_sha256: wave.binding.digest().into(),
                },
                deadline,
            )?;
        }
        Ok(())
    }
    pub fn finish(
        &mut self,
        completion: &CheckedBatchRequestCompletion,
        admitted: Instant,
    ) -> Result<()> {
        let deadline = admitted + Duration::from_millis(WALL_MILLIS);
        let expected = common(completion)?;
        for (rank, peer) in &mut self.peers {
            let Frame::Finished {
                rank: actual,
                sequence,
                completion: c,
            } = receive(peer, deadline)?
            else {
                anyhow::bail!("atomic peer actual opaque completion missing")
            };
            ensure!(
                actual == *rank && sequence == self.next && c == expected,
                "rank{rank} native outputs/stops/full-tail/reset association differs"
            );
        }
        for (_, peer) in &mut self.peers {
            send(
                peer,
                &Frame::ReuseReleased {
                    sequence: self.next,
                    completion: expected.clone(),
                },
                deadline,
            )?;
        }
        self.next += 1;
        Ok(())
    }
    pub fn stop(
        &mut self,
        previous: &CheckedBatchRequestCompletion,
        deadline: Instant,
    ) -> Result<()> {
        let expected = common(previous)?;
        for (_, peer) in &mut self.peers {
            send(
                peer,
                &Frame::Stop {
                    previous: expected.clone(),
                },
                deadline,
            )?;
        }
        for (rank, peer) in &mut self.peers {
            let Frame::Stopped {
                rank: actual,
                previous: c,
            } = receive(peer, deadline)?
            else {
                anyhow::bail!("peer stop acknowledgement missing")
            };
            ensure!(
                actual == *rank && c == expected,
                "peer idle terminal stop differs"
            );
        }
        Ok(())
    }
}
pub struct PeerBatchDispatch {
    config: BatchDispatchConfig,
    rank: usize,
    stream: TcpStream,
    next: usize,
}
impl PeerBatchDispatch {
    pub fn connect(
        config: BatchDispatchConfig,
        rank: usize,
        previous: &CheckedBatchRequestCompletion,
        deadline: Instant,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            (1..8).contains(&rank) && previous.model_identity().rank == rank,
            "peer rank differs"
        );
        let address = SocketAddrV4::new(config.peers[0], config.port).into();
        let mut stream = loop {
            remaining(deadline)?;
            match TcpStream::connect_timeout(
                &address,
                remaining(deadline)?.min(Duration::from_millis(250)),
            ) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        };
        stream.set_nodelay(true)?;
        send(
            &mut stream,
            &Frame::Hello {
                rank,
                boot: config.boot,
                config_sha256: config.hash()?,
                deployment_sha256: deployment_hash(previous)?,
                previous: common(previous)?,
            },
            deadline,
        )?;
        let Frame::HelloAccepted {
            rank: r,
            config_sha256,
            previous: c,
        } = receive(&mut stream, deadline)?
        else {
            anyhow::bail!("atomic Hello acceptance missing")
        };
        ensure!(
            r == rank && config_sha256 == config.hash()? && c == common(previous)?,
            "atomic Hello acceptance differs"
        );
        Ok(Self {
            config,
            rank,
            stream,
            next: 2,
        })
    }
    pub fn receive(
        &mut self,
        bridge: &CpuBridge,
        book: &mut CpuSlotBook,
        previous: &CheckedBatchRequestCompletion,
        model: &super::gpu_batch_model::GpuBatchModel,
        comm: &super::gpu_collectives::GpuCollectives,
        idle_deadline: Instant,
    ) -> Result<Option<PreparedBatchWave>> {
        let frame = receive(&mut self.stream, idle_deadline)?;
        if let Frame::Stop { previous: old } = &frame {
            ensure!(
                *old == common(previous)?,
                "atomic stop must bind latest idle terminal"
            );
            send(
                &mut self.stream,
                &Frame::Stopped {
                    rank: self.rank,
                    previous: old.clone(),
                },
                idle_deadline,
            )?;
            return Ok(None);
        }
        let Frame::Prepare {
            sequence,
            response_id,
            body,
            lanes,
            head_elapsed_millis,
            wall_millis,
            group_sha256,
            phase_sha256,
            program_sha256,
            previous: old,
        } = frame
        else {
            anyhow::bail!("atomic Prepare required")
        };
        self.config.release_port(sequence)?;
        ensure!(
            sequence == self.next
                && wall_millis == WALL_MILLIS
                && head_elapsed_millis < WALL_MILLIS
                && old == common(previous)?,
            "atomic sequence/deadline/prior opaque boundary differs"
        );
        let received = Instant::now();
        // Same known transit approximation as the C1 dispatcher: host Instants
        // cannot cross machines. Rank0's original admission clock remains the
        // authoritative normal-response deadline; no per-phase reset of clocks.
        let admitted = received
            .checked_sub(Duration::from_millis(head_elapsed_millis))
            .context("peer admission offset")?;
        let prepared = PreparedAtomicRequest::prepare(bridge, ATOMIC_ROUTE, Some("1"), &body)?;
        let keys = lanes
            .iter()
            .map(|l| LaneKey {
                boot: self.config.boot,
                slot: l.slot,
                generation: l.generation,
            })
            .collect();
        let wave = PreparedBatchWave::prepare(
            prepared,
            book,
            deployment(previous)?,
            keys,
            response_id,
            admitted,
            sequence,
            previous,
        )?;
        model.validate_group_reset(&wave.binding.lanes(), comm)?;
        let actual = super::gpu_collectives::Program::new_batch_phase(wave.plan.clone())?;
        let actual_sha = actual
            .digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        ensure!(
            wave.binding.digest() == group_sha256
                && wave.plan.digest() == phase_sha256
                && actual_sha == program_sha256,
            "independently prepared original atomic/native settings differ"
        );
        let deadline = admitted + Duration::from_millis(WALL_MILLIS);
        send(
            &mut self.stream,
            &Frame::Prepared {
                rank: self.rank,
                sequence,
                group_sha256: group_sha256.clone(),
                phase_sha256,
                program_sha256,
            },
            deadline,
        )?;
        let Frame::Execute {
            sequence: s,
            group_sha256: g,
        } = receive(&mut self.stream, deadline)?
        else {
            anyhow::bail!("atomic all-rank prepare release missing")
        };
        ensure!(
            s == sequence && g == group_sha256,
            "atomic execute release differs"
        );
        Ok(Some(wave))
    }
    pub fn finish(&mut self, c: &CheckedBatchRequestCompletion, admitted: Instant) -> Result<()> {
        let deadline = admitted + Duration::from_millis(WALL_MILLIS);
        let expected = common(c)?;
        send(
            &mut self.stream,
            &Frame::Finished {
                rank: self.rank,
                sequence: self.next,
                completion: expected.clone(),
            },
            deadline,
        )?;
        let Frame::ReuseReleased {
            sequence,
            completion,
        } = receive(&mut self.stream, deadline)?
        else {
            anyhow::bail!("atomic reuse release missing")
        };
        ensure!(
            sequence == self.next && completion == expected,
            "atomic reuse release native prefix/cleanup differs"
        );
        self.next += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_phase_port_reservations_are_disjoint_for32_groups_and_reject_overflow() {
        let mut c = BatchDispatchConfig {
            peers: std::array::from_fn(|i| Ipv4Addr::new(172, 31, 240, 11 + i as u8)),
            port: 47430,
            release_base: 62000,
            maximum_groups: 32,
            boot: Uuid::from_u128(214),
            service_deadline: InstantMetadata {
                idle_millis: 1200000,
            },
        };
        c.validate().unwrap();
        let mut ports = BTreeSet::new();
        for group in 1..=32 {
            for phase in 0..18 {
                assert!(ports.insert(c.release_port(group).unwrap() + phase));
            }
        }
        assert!(c.release_port(0).is_err());
        assert!(c.release_port(33).is_err());
        c.port = 62005;
        assert!(c.validate().is_err());
        c.port = 47430;
        c.release_base = 63900;
        assert!(c.validate().is_err());
    }
    #[test]
    fn native_control_schema_rejects_bool_ids_cross_phase_extra_and_duplicate_fields() {
        for raw in [
            r#"{"phase":"Execute","sequence":true,"group_sha256":"a"}"#,
            r#"{"phase":"Execute","sequence":1,"sequence":1,"group_sha256":"a"}"#,
            r#"{"phase":"Execute","sequence":1,"group_sha256":"a","terminal":true}"#,
            r#"{"phase":"Finished","rank":0,"sequence":1,"completion":{}}"#,
        ] {
            assert!(serde_json::from_str::<Frame>(raw).is_err());
        }
        let good = r#"{"phase":"Execute","sequence":1,"group_sha256":"a"}"#;
        assert!(matches!(
            serde_json::from_str::<Frame>(good).unwrap(),
            Frame::Execute { sequence: 1, .. }
        ));
    }
}
