//! Isolated, eager native-1017 TP8 collective component; no model is enabled.
//!
//! FP32 uses the existing proven serial/v2 executor without changing its kernels.
//! Maxloc has a separate receiver carrying the complete (score bits, global ID)
//! tuple for all three rounds. A finite program is agreed before proxy startup.
//! This qualification component adds finite-value guards; it is not a benchmark.
//! The owner must coordinate all ranks AFTER `finish` and BEFORE dropping it:
//! dropping aborts/joins this component's own proxy and releases its own link.

#[path = "gpu_batch_collectives.rs"]
mod batch_transport;
use super::batch_request_plan::BatchPhasePlan;
use super::bf16_collectives::MatrixBf16Index;
use super::collectives::{MatrixF32, MatrixRole, RoutedLayout, RowKind, HIDDEN, ROUNDS, WORLD};
use super::gpu_bf16_collectives::{Bf16Workspace, PendingBf16Matrix};
use super::model_session_plan::CpuRequestPlan;
use super::request_session_release::{RequestReleaseConfig, RequestReleaseReceipt};
use super::request_session_state::{RequestCursor, TransportPrefixClaim};
use crate::net::{self, HubXport, TpLink};
use anyhow::{ensure, Context, Result};
use batch_transport::BatchTransportState;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync,
    LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::ffi::c_int;
use std::sync::Arc;
use std::thread::JoinHandle;

pub const PTX: &str = include_str!("../ptx/gpu_dsv41_collectives.ptx");
pub const SOURCE_ID: &str = env!("DSV41_COLLECTIVES_SOURCE_ID");
const MODULE: &str = "dsv41_collectives_v1";
const SERIAL_MODULE: &str = "dsv41_collectives_serial_v2";
const MAX_OPERATIONS: usize = 128;
pub const MAX_MODEL_OPERATIONS: usize = 32_768;
const CONTROL_DEADLINE_NS: u64 = 5_000_000_000;
const CONTROL_MAGIC: u32 = 0x31435344;
const NAMES: [&str; 5] = [
    "kernel_build_id",
    "dsv41_context_snapshot",
    "dsv41_check_f32",
    "dsv41_check_pairs",
    "dsv41_wait_maxloc_g",
];

// Existing opaque-C-context ABI; no new native entry point or layout is introduced.
extern "C" {
    fn net_exchange_one_dl(
        c: *mut net::NetCtx,
        peer: c_int,
        nbytes: c_int,
        deadline_ns: u64,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Operation {
    RoutedF32 { id: u64, layout: RoutedLayout },
    ShardedMaxloc { id: u64, layout: RoutedLayout },
    MatrixF32 { id: u64, layout: MatrixF32 },
    MatrixBf16 { id: u64, layout: MatrixBf16Index },
}

impl Operation {
    pub fn id(self) -> u64 {
        match self {
            Self::RoutedF32 { id, .. }
            | Self::ShardedMaxloc { id, .. }
            | Self::MatrixF32 { id, .. }
            | Self::MatrixBf16 { id, .. } => id,
        }
    }
    pub fn layout(self) -> RoutedLayout {
        match self {
            Self::RoutedF32 { layout, .. } | Self::ShardedMaxloc { layout, .. } => layout,
            Self::MatrixF32 { .. } => {
                panic!("MatrixF32 has an explicit role/column layout; use matrix_layout()")
            }
            Self::MatrixBf16 { .. } => {
                panic!("MatrixBf16 has an explicit index layout; use bf16_layout()")
            }
        }
    }
    pub fn matrix_layout(self) -> Option<MatrixF32> {
        match self {
            Self::MatrixF32 { layout, .. } => Some(layout),
            _ => None,
        }
    }
    pub fn bf16_layout(self) -> Option<MatrixBf16Index> {
        match self {
            Self::MatrixBf16 { layout, .. } => Some(layout),
            _ => None,
        }
    }
    pub(crate) fn wire_epochs(self) -> usize {
        self.epochs()
    }
    fn epochs(self) -> usize {
        match self {
            Self::RoutedF32 { layout, .. } => layout.chunk_count() * ROUNDS,
            Self::ShardedMaxloc { .. } => ROUNDS,
            Self::MatrixF32 { layout, .. } => layout.chunk_count() * ROUNDS,
            Self::MatrixBf16 { layout, .. } => layout.epochs(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    operations: Vec<Operation>,
    digest: [u8; 32],
    cycle_limit: Option<u64>,
    request_plan: Option<Arc<CpuRequestPlan>>,
    batch_phase: Option<Arc<BatchPhasePlan>>,
}

impl Program {
    /// Explicit new request mode; legacy128/32768 factories and digest domains
    /// remain unchanged. This constructor alone is no GPU/model qualification.
    pub fn new_request_session(plan: Arc<CpuRequestPlan>) -> Result<Self> {
        ensure!(
            !plan.operations().is_empty()
                && plan.operations().len() <= super::model_session_plan::MAX_OPERATIONS
                && plan.steps().len() <= super::model_session_plan::MAX_STEPS
                && plan.end_epoch() <= MAX_DEVICE_EPOCH,
            "bounded request/native epoch capacity"
        );
        let mut program = Self::with_domain(
            plan.operations().iter().map(|x| x.wire).collect(),
            b"dsv41-tp8-explicit-request-session-program-v1\0",
        )?;
        let mut h = Sha256::new();
        h.update(b"dsv41-tp8-request-plan-binding-v1\0");
        h.update(program.digest);
        h.update(plan.digest().as_bytes());
        program.digest = h.finalize().into();
        program.request_plan = Some(plan);
        Ok(program)
    }
    /// Separate atomic batch phase domain, never a C1 session or fixed probe.
    pub fn new_batch_phase(plan: Arc<BatchPhasePlan>) -> Result<Self> {
        let mut program = Self::with_domain(
            plan.operations().to_vec(),
            b"dsv41-atomic-batch-wire-phase-v1\0",
        )?;
        ensure!(
            !program.operations.is_empty()
                && program.operations.len() <= MAX_MODEL_OPERATIONS
                && program.end_epoch(plan.start_epoch())? == plan.end_epoch(),
            "bounded batch phase wire geometry"
        );
        let mut hash = Sha256::new();
        hash.update(b"dsv41-atomic-batch-phase-binding-v1\0");
        hash.update(program.digest);
        hash.update(plan.digest().as_bytes());
        program.digest = hash.finalize().into();
        program.batch_phase = Some(plan);
        Ok(program)
    }
    pub fn new(operations: Vec<Operation>) -> Result<Self> {
        ensure!(
            !operations.is_empty() && operations.len() <= MAX_OPERATIONS,
            "collective program must contain 1..=128 operations"
        );
        Self::with_domain(operations, b"dsv41-tp8-serial-f32-full-maxloc-v1\0")
    }
    /// One complete, immutable model request sequence, agreed ONCE at startup.
    /// Each prefill/decode operation and shape is supplied before launch. This
    /// does not permit dynamic additions, cursor rewinds or per-step epochs.
    /// The distinct digest domain prevents confusing this request with either
    /// the legacy <=128-op component or a fixed-shape cyclic program.
    pub fn new_model_sequence(operations: Vec<Operation>) -> Result<Self> {
        ensure!(
            !operations.is_empty() && operations.len() <= MAX_MODEL_OPERATIONS,
            "model collective program must contain1..=32768 frozen operations"
        );
        Self::with_domain(operations, b"dsv41-tp8-frozen-model-sequence-v1\0")
    }
    fn with_domain(operations: Vec<Operation>, domain: &[u8]) -> Result<Self> {
        ensure!(
            operations.windows(2).all(|w| w[0].id() < w[1].id()),
            "collective operation IDs must strictly increase"
        );
        let mut hash = Sha256::new();
        hash.update(domain);
        // The startup control agreement also refuses mixed collective/transport
        // binaries, even if their logical operation lists happen to match.
        hash.update(SOURCE_ID.as_bytes());
        hash.update(Sha256::digest(PTX.as_bytes()));
        hash.update(Sha256::digest(crate::ptx_embed::GPU_BATCH.as_bytes()));
        hash.update((operations.len() as u64).to_le_bytes());
        for op in &operations {
            let (tag, id, rows, row_kind, cols) = match *op {
                Operation::RoutedF32 { id, layout } | Operation::ShardedMaxloc { id, layout } => {
                    let tag = if matches!(op, Operation::RoutedF32 { .. }) {
                        1_u64
                    } else {
                        2
                    };
                    let kind = match layout.kind() {
                        RowKind::IndependentDecodeLanes => 1,
                        RowKind::ContiguousPrefillRows => 2,
                    };
                    (tag, id, layout.rows(), kind, HIDDEN)
                }
                Operation::MatrixF32 { id, layout } => (
                    match layout.role() {
                        MatrixRole::EngramLookup => 3,
                        MatrixRole::IndexScore => 4,
                    },
                    id,
                    layout.rows(),
                    0,
                    layout.cols(),
                ),
                Operation::MatrixBf16 { id, layout } => (5, id, layout.rows(), 0, layout.cols()),
            };
            for value in [tag, id, rows as u64, row_kind, WORLD as u64, cols as u64] {
                hash.update(value.to_le_bytes());
            }
        }
        // Legacy F32/full-maxloc programs retain their EXACT digest bytes.
        // Only programs using the new datatype bind its isolated pack/guard PTX.
        if operations
            .iter()
            .any(|op| matches!(op, Operation::MatrixBf16 { .. }))
        {
            hash.update(b"dsv41-source-bf16-index-each-round-v1\0");
            hash.update(super::gpu_bf16_collectives::SOURCE_ID.as_bytes());
            hash.update(Sha256::digest(super::gpu_bf16_collectives::PTX.as_bytes()));
        }
        Ok(Self {
            operations,
            digest: hash.finalize().into(),
            cycle_limit: None,
            request_plan: None,
            batch_phase: None,
        })
    }
    /// A separately versioned startup contract. Every cycle uses the identical
    /// operation/shape list. Dynamic index widths require a separately agreed
    /// bucket; this API does not silently change the shape between steps.
    pub fn new_cyclic(operations: Vec<Operation>, max_cycles: u64) -> Result<Self> {
        ensure!(
            (1..=1_048_576).contains(&max_cycles),
            "cyclic program limit must be1..=1048576"
        );
        let mut program = Self::new(operations)?;
        let mut hash = Sha256::new();
        hash.update(b"dsv41-tp8-cyclic-program-v2\0");
        hash.update(max_cycles.to_le_bytes());
        hash.update(program.digest);
        program.digest = hash.finalize().into();
        program.cycle_limit = Some(max_cycles);
        Ok(program)
    }
    pub fn cycle_limit(&self) -> Option<u64> {
        self.cycle_limit
    }
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn end_epoch(&self, start: u64) -> Result<u64> {
        ensure!(
            start % ROUNDS as u64 == 0,
            "collective program requires an aligned starting epoch"
        );
        self.operations.iter().try_fold(start, |epoch, op| {
            epoch
                .checked_add(op.epochs() as u64)
                .context("collective epoch overflow")
        })
    }
}

const MAX_DEVICE_EPOCH: u64 = (1u64 << 44) - 1; // Native length tags reserve their low20 bits for byte count.

/// The actual CPU cursor used by cyclic GPU submission. Completion here means
/// queued to the one ordered stream, not yet confirmed GPU/network execution.
#[derive(Clone, Debug)]
pub struct CycleCursor {
    operations: Vec<Operation>,
    limit: u64,
    queued_cycles: u64,
    last_step: Option<u64>,
    active_step: Option<u64>,
    next: usize,
    expected_epoch: u64,
    step_hash: Sha256,
}
impl CycleCursor {
    pub fn new(program: &Program, start_epoch: u64) -> Result<Self> {
        let limit = program.cycle_limit.context("program is not cyclic")?;
        let one = program
            .end_epoch(start_epoch)?
            .checked_sub(start_epoch)
            .context("cycle epoch extent")?;
        let end = start_epoch
            .checked_add(
                one.checked_mul(limit)
                    .context("cyclic epoch budget overflow")?,
            )
            .context("cyclic epoch budget overflow")?;
        ensure!(
            end <= MAX_DEVICE_EPOCH,
            "cyclic epoch budget exceeds native generation-tag domain"
        );
        let mut step_hash = Sha256::new();
        step_hash.update(b"dsv41-cycle-step-ids-v1\0");
        Ok(Self {
            operations: program.operations.clone(),
            limit,
            queued_cycles: 0,
            last_step: None,
            active_step: None,
            next: 0,
            expected_epoch: start_epoch,
            step_hash,
        })
    }
    pub fn begin_step(&mut self, step: u64) -> Result<()> {
        ensure!(
            self.active_step.is_none() && self.next == 0 && self.queued_cycles < self.limit,
            "cycle is already active/incomplete or limit reached"
        );
        ensure!(
            self.last_step.is_none_or(|prior| step > prior),
            "cycle step ID must strictly increase"
        );
        self.active_step = Some(step);
        Ok(())
    }
    pub fn check_step(&self, step: u64) -> Result<()> {
        ensure!(
            self.active_step == Some(step),
            "matrix belongs to a different/inactive cycle step"
        );
        Ok(())
    }
    fn expect(&self, operation: Operation) -> Result<()> {
        ensure!(
            self.active_step.is_some() && self.operations.get(self.next) == Some(&operation),
            "cyclic operation/cursor/shape mismatch"
        );
        Ok(())
    }
    pub fn queued(&mut self, operation: Operation) -> Result<()> {
        self.expect(operation)?;
        let epoch = self
            .expected_epoch
            .checked_add(operation.epochs() as u64)
            .context("cyclic epoch overflow")?;
        ensure!(
            epoch <= MAX_DEVICE_EPOCH,
            "cyclic generation-tag extent exceeded"
        );
        self.expected_epoch = epoch;
        self.next += 1;
        Ok(())
    }
    pub fn end_step(&mut self, step: u64) -> Result<()> {
        self.check_step(step)?;
        ensure!(
            self.next == self.operations.len(),
            "cycle has missing operations"
        );
        self.queued_cycles += 1;
        self.step_hash.update(step.to_le_bytes());
        self.last_step = Some(step);
        self.active_step = None;
        self.next = 0;
        Ok(())
    }
    pub fn expected_epoch(&self) -> u64 {
        self.expected_epoch
    }
    pub fn queued_cycles(&self) -> u64 {
        self.queued_cycles
    }
    pub fn last_step(&self) -> Option<u64> {
        self.last_step
    }
    pub fn final_ready(&self) -> bool {
        self.queued_cycles > 0 && self.active_step.is_none() && self.next == 0
    }
    fn checkpoint(&self) -> CycleCompletion {
        CycleCompletion {
            cycle_limit: self.limit,
            completed_cycles: self.queued_cycles,
            last_step: self.last_step,
            step_ids_sha256: hex(&self.step_hash.clone().finalize()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CycleCompletion {
    pub cycle_limit: u64,
    pub completed_cycles: u64,
    pub last_step: Option<u64>,
    pub step_ids_sha256: String,
}

struct Control<'a> {
    link: &'a mut TpLink,
    rank: usize,
}
impl HubXport for Control<'_> {
    fn world(&self) -> usize {
        WORLD
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn send_buf(&mut self, words: usize) -> &mut [u32] {
        self.link.ctrl_send_mut(words)
    }
    fn recv_copy(&self, src: usize, words: usize) -> Vec<u32> {
        self.link.ctrl_recv(src as i32, words).to_vec()
    }
    fn exchange_one(&mut self, peer: usize, nbytes: usize) -> i32 {
        unsafe {
            net_exchange_one_dl(
                self.link.ctx_addr() as *mut net::NetCtx,
                peer as c_int,
                nbytes as c_int,
                CONTROL_DEADLINE_NS,
            )
        }
    }
}

fn agree(
    link: &mut TpLink,
    rank: usize,
    program: &Program,
    epoch: u64,
    phase: u32,
    cycles: Option<CycleCompletion>,
) -> Result<()> {
    let mut mine = vec![
        CONTROL_MAGIC,
        if cycles.is_some() { 2 } else { 1 },
        rank as u32,
        WORLD as u32,
        epoch as u32,
        (epoch >> 32) as u32,
        program.operations.len() as u32,
        phase,
    ];
    mine.extend(
        program
            .digest
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
    );
    if let Some(cycle) = cycles {
        let step = cycle.last_step.unwrap_or(0);
        mine.extend([
            cycle.cycle_limit as u32,
            (cycle.cycle_limit >> 32) as u32,
            cycle.completed_cycles as u32,
            (cycle.completed_cycles >> 32) as u32,
            u32::from(cycle.last_step.is_some()),
            step as u32,
            (step >> 32) as u32,
        ]);
        mine.extend(
            cycle
                .step_ids_sha256
                .as_bytes()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    // hub_all owns the transport's final 8-byte tag, not the hash or any metadata.
    let all = net::hub_all(&mut Control { link, rank }, &mine, mine.len() + 2)?;
    ensure!(all.len() == WORLD, "collective agreement omitted a rank");
    for (source, words) in all.iter().enumerate() {
        let mut expected = mine.clone();
        expected[2] = source as u32;
        ensure!(
            *words == expected,
            "rank {source} disagrees on collective program/source/epoch/phase"
        );
    }
    Ok(())
}

/// Low-level lifecycle result. Public construction/serialization must never be
/// accepted as a model completion proof; GpuRequestSession owns that authority.
#[derive(Clone, Debug, Serialize)]
pub struct TransportRequestPrefixCompletion {
    pub rank: usize,
    pub program_sha256: String,
    pub claim_sha256: String,
    pub actual_epoch: u64,
    pub actual_queued_operations: usize,
    pub all_rank_RDMA_prefix_agreement: bool,
    pub model_completion_authority: bool,
}
fn request_agreement_words(
    rank: usize,
    phase: u32,
    old_program: [u8; 32],
    new_program: [u8; 32],
    claim: &TransportPrefixClaim,
    epoch: u64,
    queued: usize,
) -> Result<Vec<u32>> {
    let mut mine = vec![
        0x31515244,
        3,
        rank as u32,
        WORLD as u32,
        phase,
        epoch as u32,
        (epoch >> 32) as u32,
        queued as u32,
    ];
    for data in [
        old_program.to_vec(),
        new_program.to_vec(),
        claim.digest()?.into_bytes(),
    ] {
        mine.extend(
            data.chunks_exact(4)
                .map(|x| u32::from_le_bytes(x.try_into().unwrap())),
        );
    }
    Ok(mine)
}
fn agree_request(
    link: &mut TpLink,
    rank: usize,
    phase: u32,
    old_program: [u8; 32],
    new_program: [u8; 32],
    claim: &TransportPrefixClaim,
    epoch: u64,
    queued: usize,
) -> Result<()> {
    let mine =
        request_agreement_words(rank, phase, old_program, new_program, claim, epoch, queued)?;
    let all = net::hub_all(&mut Control { link, rank }, &mine, mine.len() + 2)?;
    ensure!(
        all.len() == WORLD,
        "request agreement omitted physical rank"
    );
    for (source, words) in all.iter().enumerate() {
        let mut expected = mine.clone();
        expected[2] = source as u32;
        ensure!(
            *words == expected,
            "rank{source} request prefix/rearm agreement differs"
        );
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Completion {
    pub rank: usize,
    pub completed_operations: usize,
    pub device_epoch: u64,
    pub gpu_ready: u64,
    pub abort_status: u64,
    pub tail_fires: u64,
    pub validation_status: u32,
    pub program_sha256: String,
    pub helper_ptx_sha256: String,
    pub transport_ptx_sha256: String,
    pub helper_source_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles: Option<CycleCompletion>,
}

pub struct GpuCollectives {
    // Local completion association, independent of the wire/program digest.
    // A move preserves it; a second owner with identical rank/program does not.
    completion_identity: [u8; 32],
    dev: Arc<CudaDevice>,
    link: TpLink,
    proxy: Option<JoinHandle<()>>,
    stream: Arc<CudaStream>,
    serial: crate::tp_xport::Kernels,
    check_f32: CudaFunction,
    check_pairs: CudaFunction,
    maxloc: CudaFunction,
    status: CudaSlice<u32>,
    rank: usize,
    program: Program,
    next: usize,
    expected_epoch: u64,
    failed: bool,
    finished: bool,
    cycle: Option<CycleCursor>,
    request: Option<RequestCursor>,
    request_release: Option<RequestReleaseReceipt>,
    batch: Option<BatchTransportState>,
}

impl GpuCollectives {
    /// This is a queued CPU cursor, not evidence of completed model work.
    pub fn queued_operation_count(&self) -> usize {
        self.next
    }
    pub fn request_plan_digest(&self) -> Option<&str> {
        self.request.as_ref().map(|r| r.plan().digest())
    }
    pub fn program_digest_hex(&self) -> String {
        hex(&self.program.digest)
    }
    /// Explicit transport-only finish. It accepts declared native hashes for
    /// synthetic transport oracles; ONLY the sealed model-session layer may
    /// turn an actual CheckedModelChunk into API terminal authority.
    pub fn finish_transport_request_prefix(
        &mut self,
        claim: &TransportPrefixClaim,
    ) -> Result<TransportRequestPrefixCompletion> {
        let result = (|| {
            ensure!(
                !self.failed && !self.finished,
                "request failed/already finished"
            );
            self.request
                .as_ref()
                .context("not request mode")?
                .check_claim(claim)?;
            ensure!(
                self.next == claim.boundary.operations_queued
                    && self.expected_epoch == claim.boundary.epoch,
                "actual queued prefix differs from claimed complete boundary"
            );
            self.synchronize()?;
            agree_request(
                &mut self.link,
                self.rank,
                2,
                self.program.digest,
                self.program.digest,
                claim,
                self.expected_epoch,
                self.next,
            )?;
            self.request
                .as_mut()
                .unwrap()
                .mark_after_external_agreement(claim)?;
            self.finished = true;
            Ok(TransportRequestPrefixCompletion {
                rank: self.rank,
                program_sha256: hex(&self.program.digest),
                claim_sha256: claim.digest()?,
                actual_epoch: self.link.device_epoch(),
                actual_queued_operations: self.next,
                all_rank_RDMA_prefix_agreement: true,
                model_completion_authority: false,
            })
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }
    /// Actual all8 TCP arrival release occurs AFTER every rank's RDMA finish.
    /// The stable transport owner/link stays alive through this call.
    pub fn complete_request_release(
        &mut self,
        config: &RequestReleaseConfig,
    ) -> Result<RequestReleaseReceipt> {
        let result = (|| {
            ensure!(
                !self.failed && self.finished && self.request_release.is_none(),
                "request not awaiting release"
            );
            let claim = self
                .request
                .as_ref()
                .and_then(RequestCursor::last_claim)
                .context("request prefix not agreed")?;
            let receipt = super::request_session_release::release(
                self.rank,
                config,
                &hex(&self.program.digest),
                claim,
            )?;
            self.request_release = Some(receipt.clone());
            Ok(receipt)
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }
    /// Install a NEW fully frozen request only after prior actual prefix/release.
    /// Epochs, local owner, link/proxy, GPU status and physical buffers persist.
    /// Only the logical new-program cursor begins at0; no old IDs may be reused.
    pub fn begin_next_request(&mut self, plan: Arc<CpuRequestPlan>) -> Result<()> {
        let result = (|| {
            ensure!(
                !self.failed
                    && self.finished
                    && self
                        .request_release
                        .as_ref()
                        .is_some_and(|r| r.actual_TCP_release_completed),
                "prior request transport release incomplete"
            );
            let request = self.request.as_ref().context("not request mode")?;
            request.check_next(&plan)?;
            let old_claim = request.last_claim().context("no old prefix")?.clone();
            let new_program = Program::new_request_session(plan.clone())?;
            self.synchronize()?;
            agree_request(
                &mut self.link,
                self.rank,
                3,
                self.program.digest,
                new_program.digest,
                &old_claim,
                self.expected_epoch,
                self.next,
            )?;
            let mut next = self.request.as_ref().unwrap().clone();
            next.rearm_after_external_agreement(plan)?;
            self.program = new_program;
            self.request = Some(next);
            self.next = 0;
            self.finished = false;
            self.request_release = None;
            Ok(())
        })();
        if result.is_err() {
            self.abort_session();
        }
        result
    }

    /// Consumes a fresh, dedicated link. No model/global serving link is registered.
    pub fn new(
        dev: Arc<CudaDevice>,
        mut link: TpLink,
        rank: usize,
        program: Program,
        proxy_core: i32,
    ) -> Result<Self> {
        ensure!(
            rank < WORLD && link.world() == WORLD as i32 && !link.oneshot_on(),
            "requires a dedicated world8 recursive-doubling link"
        );
        if dev.get_func(MODULE, "dsv41_wait_maxloc_g").is_none() {
            dev.load_ptx(Ptx::from_src(PTX), MODULE, &NAMES)
                .context("load isolated 1017 collective helpers")?;
        }
        let function = |name: &str| {
            dev.get_func(MODULE, name)
                .with_context(|| format!("missing isolated kernel {name}"))
        };
        let mut stamp = dev.alloc_zeros::<u64>(1)?;
        dev.memset_zeros(&mut stamp)?;
        unsafe {
            function("kernel_build_id")?.launch(LaunchConfig::for_num_elems(1), (&stamp,))?;
        }
        dev.synchronize()?;
        let expected_id = u64::from_str_radix(SOURCE_ID, 16)?;
        ensure!(
            dev.dtoh_sync_copy(&stamp)? == [expected_id],
            "isolated helper PTX build ID mismatch"
        );
        let mut snapshot = dev.alloc_zeros::<u64>(8)?;
        dev.memset_zeros(&mut snapshot)?;
        unsafe {
            function("dsv41_context_snapshot")?.launch(
                LaunchConfig::for_num_elems(1),
                (link.ctx_device_ptr(), &snapshot),
            )?;
        }
        dev.synchronize()?;
        let info = dev.dtoh_sync_copy(&snapshot)?;
        ensure!(
            info[1..] == [rank as u64, WORLD as u64, ROUNDS as u64, 0, 1, 1, 0],
            "actual device context is not the claimed clean world8/rank: {info:?}"
        );
        let epoch = info[0];
        ensure!(
            epoch == link.gpu_ready() && epoch % ROUNDS as u64 == 0,
            "context epoch/watermark is not quiescent/aligned"
        );
        program.end_epoch(epoch)?;
        let batch = program
            .batch_phase
            .as_ref()
            .map(|plan| {
                ensure!(
                    plan.start_epoch() == epoch,
                    "initial atomic phase epoch differs from actual clean link"
                );
                Ok::<_, anyhow::Error>(BatchTransportState::new(plan.clone()))
            })
            .transpose()?;
        let request = program
            .request_plan
            .as_ref()
            .map(|p| RequestCursor::new(p.clone(), epoch))
            .transpose()?;
        let cycle = if program.cycle_limit.is_some() {
            Some(CycleCursor::new(&program, epoch)?)
        } else {
            None
        };
        link.set_payload(HIDDEN * 4, true)?;
        link.set_recv_mode(true)?;
        agree(
            &mut link,
            rank,
            &program,
            epoch,
            1,
            cycle.as_ref().map(CycleCursor::checkpoint),
        )?;
        let serial = crate::tp_xport::Kernels::load_with_recv(&dev, SERIAL_MODULE, true)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut status)?;
        dev.synchronize()?;
        let check_f32 = function("dsv41_check_f32")?;
        let check_pairs = function("dsv41_check_pairs")?;
        let maxloc = function("dsv41_wait_maxloc_g")?;
        let stream = Arc::new(crate::gpu::fork_blocking_stream(&dev));
        let proxy = Some(net::spawn_proxy_aux(link.ctx_addr(), proxy_core));
        let mut local_identity = Sha256::new();
        local_identity.update(b"dsv41-local-collective-completion-v1\0");
        local_identity.update(uuid::Uuid::new_v4().as_bytes());
        local_identity.update(program.digest());
        local_identity.update((rank as u64).to_le_bytes());
        Ok(Self {
            completion_identity: local_identity.finalize().into(),
            dev,
            link,
            proxy,
            stream,
            serial,
            check_f32,
            check_pairs,
            maxloc,
            status,
            rank,
            program,
            next: 0,
            expected_epoch: epoch,
            failed: false,
            finished: false,
            cycle,
            request,
            request_release: None,
            batch,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Identity of this exact local completion owner. It is not a cross-rank
    /// wire identity and does not replace program/source/epoch agreement.
    pub fn completion_identity(&self) -> [u8; 32] {
        self.completion_identity
    }
    /// Terminal cooperative abort of this local session. Call at the error site
    /// before a device fence/temporary teardown when work may be queued. The
    /// orchestrator still owns all-rank shutdown; no peer/TCP broadcast is made.
    pub fn abort_session(&mut self) {
        self.failed = true;
        self.link.abort();
    }

    pub fn expected_epoch(&self) -> u64 {
        self.expected_epoch
    }
    /// Producers and consumers in a future forward must use this same ordered
    /// stream (or establish their own explicit CUDA event dependencies).
    pub fn compute_stream(&self) -> &CudaStream {
        &self.stream
    }
    /// Retain the exact stream independently of a mutable collective borrow.
    /// This preserves stream ownership without manufacturing a raw CUDA alias.
    pub fn compute_stream_owner(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    /// Local checked submission boundary only. The caller's trusted SPMD work
    /// distribution must supply the same step ID to every rank; no per-step
    /// host/network agreement or GPU synchronization is inserted here.
    pub fn begin_cycle(&mut self, step: u64) -> Result<()> {
        let result = (|| {
            ensure!(
                !self.failed && !self.finished && self.next == 0,
                "component/cursor is not ready for a new cycle"
            );
            self.cycle
                .as_mut()
                .context("program is one-shot")?
                .begin_step(step)
        })();
        if result.is_err() {
            self.failed = true;
            self.link.abort();
        }
        result
    }
    pub fn end_cycle(&mut self, step: u64) -> Result<()> {
        let result = (|| {
            ensure!(
                !self.failed && !self.finished && self.next == self.program.operations.len(),
                "cycle has missing/failed operations"
            );
            self.cycle
                .as_mut()
                .context("program is one-shot")?
                .end_step(step)?;
            self.next = 0;
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
            self.link.abort();
        }
        result
    }

    fn expect(&self, operation: Operation) -> Result<()> {
        ensure!(
            !self.failed && !self.finished,
            "collective component is failed or finished"
        );
        ensure!(
            self.program.operations.get(self.next) == Some(&operation),
            "operation/source layout does not match the agreed program at {}",
            self.next
        );
        if let Some(cycle) = &self.cycle {
            cycle.expect(operation)?;
        }
        if let Some(request) = &self.request {
            request.expect(operation)?;
        }
        Ok(())
    }

    fn queued(&mut self, operation: Operation, result: Result<()>) -> Result<()> {
        if let Err(error) = result {
            self.failed = true;
            self.link.abort();
            return Err(error);
        }
        if let Some(request) = &mut self.request {
            if let Err(error) = request.queued(operation) {
                self.failed = true;
                self.link.abort();
                return Err(error);
            }
            self.expected_epoch = request.expected_epoch();
        } else if let Some(cycle) = &mut self.cycle {
            if let Err(error) = cycle.queued(operation) {
                self.failed = true;
                self.link.abort();
                return Err(error);
            }
            self.expected_epoch = cycle.expected_epoch();
        } else {
            self.expected_epoch = self
                .expected_epoch
                .checked_add(operation.epochs() as u64)
                .context("collective epoch overflow")?;
        }
        self.next += 1;
        Ok(())
    }

    fn source_checked(&mut self, operation: Operation, valid_buffer: bool) -> Result<()> {
        let result = self.expect(operation).and_then(|_| {
            ensure!(
                valid_buffer,
                "source buffer belongs to a different CUDA context/layout"
            );
            Ok(())
        });
        if result.is_err() {
            self.failed = true;
            self.link.abort();
        }
        result
    }

    /// Buffer is [row][hidden5120] FP32 on this exact CUDA context. It remains
    /// alive until synchronize; no BF16 cast occurs between the three rounds.
    pub fn enqueue_reduce(
        &mut self,
        id: u64,
        layout: RoutedLayout,
        values: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let operation = Operation::RoutedF32 { id, layout };
        self.source_checked(
            operation,
            Arc::ptr_eq(&values.device(), &self.dev) && values.len() == layout.elements(),
        )?;
        let ctx = self.link.ctx_device_ptr();
        let p = *values.device_ptr();
        let status = *self.status.device_ptr();
        let result = (|| {
            unsafe {
                self.check_f32.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig::for_num_elems(layout.elements() as u32),
                    (ctx, p, layout.elements() as i32, status),
                )?;
            }
            crate::tp_xport::reduce_serial(
                &self.serial,
                &self.stream,
                ctx,
                p,
                layout.elements(),
                ROUNDS,
            )?;
            unsafe {
                self.check_f32.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig::for_num_elems(layout.elements() as u32),
                    (ctx, p, layout.elements() as i32, status),
                )?;
            }
            Ok(())
        })();
        self.queued(operation, result)
    }

    /// Buffer is [row][score_bits, global_token_id]. Initial IDs must belong to
    /// this rank's equal vocabulary slice; incoming winners keep their source ID.
    pub fn enqueue_maxloc(
        &mut self,
        id: u64,
        layout: RoutedLayout,
        pairs: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let operation = Operation::ShardedMaxloc { id, layout };
        self.source_checked(
            operation,
            Arc::ptr_eq(&pairs.device(), &self.dev) && pairs.len() == 2 * layout.rows(),
        )?;
        let ctx = self.link.ctx_device_ptr();
        let p = *pairs.device_ptr();
        let status = *self.status.device_ptr();
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let result = (|| {
            unsafe {
                self.check_pairs.clone().launch_on_stream(
                    &self.stream,
                    config,
                    (ctx, p, layout.rows() as i32, status),
                )?;
            }
            for _ in 0..ROUNDS {
                crate::tp_xport::k1_serial(&self.serial, &self.stream, ctx, p, 2 * layout.rows())?;
                unsafe {
                    self.maxloc.clone().launch_on_stream(
                        &self.stream,
                        config,
                        (ctx, p, layout.rows() as i32, status),
                    )?;
                }
            }
            Ok(())
        })();
        self.queued(operation, result)
    }

    /// Finite row-major FP32 data; causal/candidate masking is performed later.
    /// No new wire/kernel path: this calls the same serial/v2 mode2 executor.
    pub fn enqueue_matrix(
        &mut self,
        id: u64,
        layout: MatrixF32,
        values: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(
            self.cycle.is_none(),
            "cyclic matrices require enqueue_matrix_for_step"
        );
        self.enqueue_matrix_inner(id, layout, values)
    }
    fn enqueue_matrix_inner(
        &mut self,
        id: u64,
        layout: MatrixF32,
        values: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let operation = Operation::MatrixF32 { id, layout };
        self.source_checked(
            operation,
            Arc::ptr_eq(&values.device(), &self.dev) && values.len() == layout.elements(),
        )?;
        let ctx = self.link.ctx_device_ptr();
        let pointer = *values.device_ptr();
        let status = *self.status.device_ptr();
        let result = (|| {
            unsafe {
                self.check_f32.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig::for_num_elems(layout.elements() as u32),
                    (ctx, pointer, layout.elements() as i32, status),
                )?;
            }
            crate::tp_xport::reduce_serial(
                &self.serial,
                &self.stream,
                ctx,
                pointer,
                layout.elements(),
                ROUNDS,
            )?;
            unsafe {
                self.check_f32.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig::for_num_elems(layout.elements() as u32),
                    (ctx, pointer, layout.elements() as i32, status),
                )?;
            }
            Ok(())
        })();
        self.queued(operation, result)
    }
    pub fn enqueue_matrix_for_step(
        &mut self,
        step: u64,
        id: u64,
        layout: MatrixF32,
        values: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let result = self
            .cycle
            .as_ref()
            .context("step-tagged matrix requires a cyclic program")
            .and_then(|c| c.check_step(step));
        if let Err(error) = result {
            self.failed = true;
            self.link.abort();
            return Err(error);
        }
        self.enqueue_matrix_inner(id, layout, values)
    }

    /// Source BF16 index-score collective, with no F32 network alternative.
    /// Scratch is allocated before this call. The pending owner fences values,
    /// workspace and helper kernels until exact-session completion is checked.
    pub fn enqueue_bf16_matrix<'a>(
        &mut self,
        id: u64,
        layout: MatrixBf16Index,
        values: &'a mut CudaSlice<half::bf16>,
        workspace: &'a mut Bf16Workspace,
    ) -> Result<PendingBf16Matrix<'a>> {
        if self.cycle.is_some() {
            self.failed = true;
            self.link.abort();
            anyhow::bail!("cyclic BF16 matrix requires explicit step");
        }
        self.enqueue_bf16_matrix_inner(id, layout, values, workspace)
    }
    pub fn enqueue_bf16_matrix_for_step<'a>(
        &mut self,
        step: u64,
        id: u64,
        layout: MatrixBf16Index,
        values: &'a mut CudaSlice<half::bf16>,
        workspace: &'a mut Bf16Workspace,
    ) -> Result<PendingBf16Matrix<'a>> {
        let result = self
            .cycle
            .as_ref()
            .context("BF16 step requires cyclic program")
            .and_then(|c| c.check_step(step));
        if let Err(error) = result {
            self.failed = true;
            self.link.abort();
            return Err(error);
        }
        self.enqueue_bf16_matrix_inner(id, layout, values, workspace)
    }
    fn enqueue_bf16_matrix_inner<'a>(
        &mut self,
        id: u64,
        layout: MatrixBf16Index,
        values: &'a mut CudaSlice<half::bf16>,
        workspace: &'a mut Bf16Workspace,
    ) -> Result<PendingBf16Matrix<'a>> {
        let operation = Operation::MatrixBf16 { id, layout };
        self.source_checked(
            operation,
            Arc::ptr_eq(&values.device(), &self.dev)
                && values.len() == layout.elements()
                && workspace.matches(&self.dev, &self.stream, layout),
        )?;
        let result = workspace.enqueue_async(
            &self.stream,
            &self.serial,
            self.link.ctx_device_ptr(),
            *self.status.device_ptr(),
            values,
            layout,
        );
        if let Err(error) = self.queued(operation, result) {
            // queued() cooperatively aborts BEFORE this fence, so an earlier K1
            // or K2 cannot wait for a peer after a later launch failed.
            let _ = self.dev.synchronize();
            return Err(error);
        }
        Ok(workspace.pending(
            values,
            layout,
            self.completion_identity,
            self.expected_epoch,
            self.rank,
        ))
    }

    pub fn synchronize(&mut self) -> Result<()> {
        let result = (|| {
            self.dev
                .synchronize()
                .context("synchronize 1017 collective component")?;
            let status = self.dev.dtoh_sync_copy(&self.status)?[0];
            ensure!(
                status == 0 && self.link.abort_status() == 0 && self.link.tail_fires() == 0,
                "collective validation failed: device_status={status} abort={} tail_fires={}",
                self.link.abort_status(),
                self.link.tail_fires()
            );
            ensure!(
                self.link.device_epoch() == self.expected_epoch
                    && self.link.gpu_ready() == self.expected_epoch,
                "actual collective epoch/watermark differs from agreed program"
            );
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
            self.link.abort();
        }
        result
    }

    /// All ranks must reach finish. Keep this object alive until an independent
    /// all-rank release barrier confirms nobody still needs the RDMA control path.
    pub fn finish(&mut self) -> Result<Completion> {
        ensure!(
            self.request.is_none() && self.batch.is_none(),
            "request/batch mode requires its explicit prefix finish and release"
        );
        ensure!(
            !self.finished
                && if let Some(cycle) = &self.cycle {
                    cycle.final_ready() && self.next == 0
                } else {
                    self.next == self.program.operations.len()
                },
            "collective program is incomplete or already finished"
        );
        self.synchronize()?;
        agree(
            &mut self.link,
            self.rank,
            &self.program,
            self.expected_epoch,
            2,
            self.cycle.as_ref().map(CycleCursor::checkpoint),
        )?;
        self.finished = true;
        Ok(Completion {
            rank: self.rank,
            completed_operations: if let Some(cycle) = &self.cycle {
                usize::try_from(cycle.queued_cycles())?
                    .checked_mul(self.program.operations.len())
                    .context("completed operation count overflow")?
            } else {
                self.next
            },
            device_epoch: self.link.device_epoch(),
            gpu_ready: self.link.gpu_ready(),
            abort_status: self.link.abort_status(),
            tail_fires: self.link.tail_fires(),
            validation_status: self.dev.dtoh_sync_copy(&self.status)?[0],
            program_sha256: hex(&self.program.digest),
            helper_ptx_sha256: format!("{:x}", Sha256::digest(PTX.as_bytes())),
            transport_ptx_sha256: format!(
                "{:x}",
                Sha256::digest(crate::ptx_embed::GPU_BATCH.as_bytes())
            ),
            helper_source_id: SOURCE_ID.into(),
            cycles: self.cycle.as_ref().map(CycleCursor::checkpoint),
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl Drop for GpuCollectives {
    fn drop(&mut self) {
        self.link.abort();
        // Cooperative abort precedes sync so an outstanding waiter can leave.
        let _ = self.dev.synchronize();
        if let Some(proxy) = self.proxy.take() {
            let _ = proxy.join();
        }
        // TpLink is freed only after the GPU and its proxy stopped using it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_is_a_distinct_frozen_datatype_without_changing_legacy_digest_bytes() {
        let routed = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let matrix = MatrixF32::new(MatrixRole::IndexScore, 2, 31).unwrap();
        let old = Program::new(vec![
            Operation::RoutedF32 {
                id: 1,
                layout: routed,
            },
            Operation::MatrixF32 {
                id: 2,
                layout: matrix,
            },
            Operation::ShardedMaxloc {
                id: 3,
                layout: routed,
            },
        ])
        .unwrap();
        // Frozen pre-BF16 wire digest: its source IDs and exact six-u64 records
        // still hash identically; new helper PTX is absent for these operations.
        let mut legacy = Sha256::new();
        legacy.update(b"dsv41-tp8-serial-f32-full-maxloc-v1\0");
        legacy.update(SOURCE_ID.as_bytes());
        legacy.update(Sha256::digest(PTX.as_bytes()));
        legacy.update(Sha256::digest(crate::ptx_embed::GPU_BATCH.as_bytes()));
        legacy.update(3u64.to_le_bytes());
        for record in [
            [1u64, 1, 1, 1, 8, 5120],
            [4, 2, 2, 0, 8, 31],
            [2, 3, 1, 1, 8, 5120],
        ] {
            for value in record {
                legacy.update(value.to_le_bytes());
            }
        }
        assert_eq!(old.digest(), <[u8; 32]>::from(legacy.finalize()));
        let bf16 = Operation::MatrixBf16 {
            id: 2,
            layout: MatrixBf16Index::new(2, 31).unwrap(),
        };
        let mut mixed = old.operations().to_vec();
        mixed[1] = bf16;
        let p = Program::new_cyclic(mixed.clone(), 3).unwrap();
        assert_ne!(Program::new(mixed.clone()).unwrap().digest(), old.digest());
        let mut cursor = CycleCursor::new(&p, 24).unwrap();
        for (step, end) in [(11, 33), (19, 42), (42, 51)] {
            cursor.begin_step(step).unwrap();
            cursor.queued(mixed[0]).unwrap();
            assert!(cursor
                .queued(Operation::MatrixBf16 {
                    id: 2,
                    layout: MatrixBf16Index::new(1, 62).unwrap()
                })
                .is_err());
            cursor.queued(bf16).unwrap();
            cursor.queued(mixed[2]).unwrap();
            cursor.end_step(step).unwrap();
            assert_eq!(cursor.expected_epoch(), end);
        }
        assert!(cursor.final_ready());
    }
    #[test]
    fn frozen_model_sequences_have_bounded_distinct_identity_and_no_legacy_limit_escape() {
        let d = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let ops = |n: usize| {
            (0..n)
                .map(|i| Operation::RoutedF32 {
                    id: i as u64,
                    layout: d,
                })
                .collect::<Vec<_>>()
        };
        let legacy = Program::new(ops(128)).unwrap();
        let model = Program::new_model_sequence(ops(128)).unwrap();
        assert_eq!(legacy.operations(), model.operations());
        assert_ne!(legacy.digest(), model.digest());
        assert!(model.cycle_limit().is_none());
        assert!(CycleCursor::new(&model, 0).is_err());
        assert!(Program::new(ops(129)).is_err());
        assert!(Program::new_cyclic(ops(129), 2).is_err());
        assert_eq!(
            Program::new_model_sequence(ops(129))
                .unwrap()
                .end_epoch(0)
                .unwrap(),
            387
        );
        let full = Program::new_model_sequence(ops(MAX_MODEL_OPERATIONS)).unwrap();
        assert_eq!(full.operations().len(), 32768);
        assert_eq!(full.end_epoch(24).unwrap(), 98328);
        assert!(Program::new_model_sequence(ops(MAX_MODEL_OPERATIONS + 1)).is_err());
        assert!(Program::new_model_sequence(Vec::new()).is_err());
        let mut duplicate = ops(129);
        duplicate[128] = duplicate[127];
        assert!(Program::new_model_sequence(duplicate).is_err());
        let mut request = ops(129);
        request[0] = Operation::RoutedF32 {
            id: 0,
            layout: RoutedLayout::new(75, RowKind::ContiguousPrefillRows).unwrap(),
        };
        assert_ne!(
            Program::new_model_sequence(request).unwrap().digest(),
            Program::new_model_sequence(ops(129)).unwrap().digest()
        );
    }
    #[test]
    fn cycles_reuse_one_program_without_rewinding_epochs_or_allowing_partial_steps() {
        let matrix = MatrixF32::new(MatrixRole::IndexScore, 8, 513).unwrap();
        let routed = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let ops = vec![
            Operation::MatrixF32 {
                id: 1,
                layout: matrix,
            },
            Operation::RoutedF32 {
                id: 2,
                layout: routed,
            },
            Operation::ShardedMaxloc {
                id: 3,
                layout: routed,
            },
        ];
        let one = Program::new(ops.clone()).unwrap();
        let p = Program::new_cyclic(ops.clone(), 3).unwrap();
        assert_ne!(one.digest(), p.digest());
        assert!(CycleCursor::new(&one, 0).is_err());
        let mut cursor = CycleCursor::new(&p, 24).unwrap();
        assert!(!cursor.final_ready());
        for (step, epoch) in [(11, 33), (19, 42), (42, 51)] {
            cursor.begin_step(step).unwrap();
            assert!(cursor.begin_step(step).is_err());
            assert!(cursor.end_step(step).is_err());
            assert!(cursor.queued(ops[1]).is_err());
            assert!(cursor.check_step(step + 1).is_err());
            for op in &ops {
                cursor.queued(*op).unwrap();
            }
            assert_eq!(cursor.expected_epoch(), epoch);
            assert!(cursor.queued(ops[0]).is_err());
            cursor.end_step(step).unwrap();
            assert!(cursor.end_step(step).is_err());
            assert!(cursor.final_ready());
            assert_eq!(cursor.expected_epoch(), epoch);
            assert!(cursor.begin_step(step).is_err());
        }
        assert_eq!(cursor.queued_cycles(), 3);
        assert_eq!(cursor.last_step(), Some(42));
        assert!(cursor.begin_step(43).is_err());
        assert!(CycleCursor::new(&p, MAX_DEVICE_EPOCH - 6).is_err());
        let changed = Program::new_cyclic(
            vec![Operation::MatrixF32 {
                id: 1,
                layout: MatrixF32::new(MatrixRole::IndexScore, 8, 31).unwrap(),
            }],
            3,
        )
        .unwrap();
        assert_ne!(p.digest(), changed.digest());
    }
    #[test]
    fn program_identity_covers_algorithm_order_rows_and_kind() {
        let d = RoutedLayout::new(2, RowKind::IndependentDecodeLanes).unwrap();
        let p = RoutedLayout::new(2, RowKind::ContiguousPrefillRows).unwrap();
        let one = Program::new(vec![
            Operation::RoutedF32 { id: 1, layout: d },
            Operation::ShardedMaxloc { id: 2, layout: d },
        ])
        .unwrap();
        assert_eq!(one.end_epoch(24).unwrap(), 30);
        for variant in [
            vec![
                Operation::RoutedF32 { id: 1, layout: p },
                Operation::ShardedMaxloc { id: 2, layout: d },
            ],
            vec![
                Operation::ShardedMaxloc { id: 1, layout: d },
                Operation::RoutedF32 { id: 2, layout: d },
            ],
            vec![
                Operation::RoutedF32 { id: 1, layout: d },
                Operation::ShardedMaxloc { id: 3, layout: d },
            ],
        ] {
            assert_ne!(one.digest(), Program::new(variant).unwrap().digest());
        }
        let wide = RoutedLayout::new(128, RowKind::ContiguousPrefillRows).unwrap();
        assert_eq!(
            Program::new(vec![Operation::RoutedF32 {
                id: 0,
                layout: wide
            }])
            .unwrap()
            .end_epoch(0)
            .unwrap(),
            9
        );
        assert!(Program::new(Vec::new()).is_err());
        assert!(Program::new(vec![
            Operation::RoutedF32 { id: 2, layout: d },
            Operation::ShardedMaxloc { id: 1, layout: d }
        ])
        .is_err());
        assert!(one.end_epoch(1).is_err());
        assert!(one.end_epoch(u64::MAX).is_err());
    }
}
#[cfg(test)]
mod request_tests {
    use super::super::{
        model_session_plan::native_ids_sha256,
        request_session_state::{tests::plan, TransportStop},
    };
    use super::*;
    #[test]
    fn explicit_request_program_accepts_full_allowance_without_enlarging_legacy_constructor() {
        let p = plan(4096, 4096, 1, 1, 0, 1);
        let wire = p.operations().iter().map(|x| x.wire).collect::<Vec<_>>();
        assert_eq!(MAX_MODEL_OPERATIONS, 32768);
        assert!(Program::new_model_sequence(wire).is_err());
        let request = Program::new_request_session(p.clone()).unwrap();
        assert_eq!(request.operations.len(), 379640);
        assert_eq!(request.end_epoch(0).unwrap(), p.end_epoch());
        assert!(request.request_plan.is_some());
        assert!(request.cycle_limit.is_none());
        let short = plan(75, 1, 1, 1, 0, 1);
        let legacy =
            Program::new_model_sequence(short.operations().iter().map(|x| x.wire).collect())
                .unwrap();
        let request = Program::new_request_session(short).unwrap();
        assert_eq!(legacy.operations, request.operations);
        assert_ne!(legacy.digest, request.digest);
        assert!(legacy.request_plan.is_none());
    }
    #[test]
    fn actual_agreement_frame_binds_phase_native_hashes_and_next_program() {
        let p = plan(75, 2, 1, 1, 0, 1);
        let claim = TransportPrefixClaim {
            boundary: p.steps()[0].expected_boundary.clone(),
            stop: TransportStop::NativeEos,
            selected_native_id: Some(1),
            consumed_native_ids_sha256: p.input_digest().into(),
            generated_native_ids_sha256: native_ids_sha256(&[1]),
        };
        let words = request_agreement_words(0, 2, [3; 32], [4; 32], &claim, 501, 84).unwrap();
        for rank in 0..8 {
            let mut expected = words.clone();
            expected[2] = rank as u32;
            assert_eq!(
                request_agreement_words(rank, 2, [3; 32], [4; 32], &claim, 501, 84).unwrap(),
                expected
            );
        }
        let mut bad = claim.clone();
        bad.generated_native_ids_sha256 = "0".repeat(64);
        assert_ne!(
            request_agreement_words(0, 2, [3; 32], [4; 32], &bad, 501, 84).unwrap(),
            words
        );
        assert_ne!(
            request_agreement_words(0, 3, [3; 32], [4; 32], &claim, 501, 84).unwrap(),
            words
        );
        assert_ne!(
            request_agreement_words(0, 2, [3; 32], [5; 32], &claim, 501, 84).unwrap(),
            words
        );
        assert_ne!(
            request_agreement_words(0, 2, [3; 32], [4; 32], &claim, 504, 84).unwrap(),
            words
        );
    }
}
