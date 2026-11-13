//! PRIVATE CPU-only immutable C1 request schedule. This is not a GPU Program,
//! completion receipt, EOS authority, serving backend or persistent session.
//! Existing Program's32768-operation limit remains unchanged.
use super::{
    bf16_collectives::MatrixBf16Index,
    chunk_plan::{self, INDEX_SOURCES},
    collectives::{MatrixF32, MatrixRole, RoutedLayout, RowKind, ROUNDS, VOCAB},
    gpu_collectives::Operation,
    native_api_contract::TOKENIZER_SHA256,
    position::{FrequencyBackend, FrequencyProfileIdentity},
    protocol::{MODEL_REVISION, TOOLKIT_REVISION},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
const DOMAIN: &[u8] = b"dsv41-tp8-c1-frozen-request-plan-v1\0";
const BOUNDARY_DOMAIN: &[u8] = b"dsv41-tp8-c1-expected-complete-step-boundary-v1\0";
pub const MAX_CONTEXT: usize = 8192;
pub const MAX_OUTPUT: usize = 4096;
pub const MAX_STEPS: usize = 8192;
pub const MAX_OPERATIONS: usize = 1_000_000;
pub const PREFILL_CHUNK: usize = 128;
pub const EOS_ID: u32 = 1;
fn hex(value: &str, n: usize) -> bool {
    value.len() == n
        && value
            .bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}
pub fn native_ids_sha256(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GreedySemantics {
    pub temperature: f64,
    pub top_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub seed: Option<u64>,
    pub thinking: bool,
    pub effective_reasoning_score: Option<u8>,
    pub speculation: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StopPolicy {
    pub eos_native_id: u32,
    /// Exact ordered parser strings, included in the digest, never tokenized or
    /// silently clipped here. Actual official-parser authority is deferred.
    pub user_sequences: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RequestBudget {
    pub context_tokens: usize,
    pub output_allowance: usize,
    pub wall_millis: u64,
}
/// All identity fields here must agree across ranks. In particular the global
/// geometry/checksum identity must NOT be one rank's different ownership plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestBinding {
    pub model_revision: String,
    pub global_checkpoint_geometry_checksum_sha256: String,
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
    pub original_request_sha256: String,
    pub tokenizer_sha256: String,
    pub toolkit_revision: String,
    pub frequency: FrequencyProfileIdentity,
    pub sampling: GreedySemantics,
    pub stop: StopPolicy,
    pub budget: RequestBudget,
    pub lane: LaneKey,
}
impl RequestBinding {
    fn validate(&self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            self.model_revision == MODEL_REVISION
                && self.tokenizer_sha256 == TOKENIZER_SHA256
                && self.toolkit_revision == TOOLKIT_REVISION,
            "exact1017/checkpoint tokenizer/toolkit identity required"
        );
        for hash in [
            &self.global_checkpoint_geometry_checksum_sha256,
            &self.execution_snapshot_sha256,
            &self.engine_binary_sha256,
            &self.original_request_sha256,
        ] {
            ensure!(
                hex(hash, 64),
                "global snapshot/binary/request identity must be SHA256"
            );
        }
        ensure!(
            self.frequency == FrequencyBackend::parse(&self.frequency.backend)?.identity(),
            "frequency backend/table/reference profile differs"
        );
        let s = &self.sampling;
        ensure!(s.temperature.is_finite()&&s.temperature==0.0 && s.top_p==1.0
            &&s.presence_penalty==0.0&&s.frequency_penalty==0.0&&!s.speculation,
            "CPU contract supports explicit greedy temperature0/top_p1/zero penalties/speculationoff only");
        ensure!(
            s.effective_reasoning_score
                .is_none_or(|n| (1..=100).contains(&n)),
            "reasoning score1..100"
        );
        ensure!(
            self.stop.eos_native_id == EOS_ID
                && self.stop.user_sequences.iter().all(|s| !s.is_empty()),
            "nativeEOS1/ordered nonempty user stop sequences required"
        );
        ensure!(
            !self.lane.boot.is_nil() && self.lane.slot == 0 && self.lane.generation > 0,
            "C1 slot0 lane identity"
        );
        let b = &self.budget;
        ensure!(
            (1..=MAX_CONTEXT).contains(&prompt_tokens)
                && (1..=MAX_CONTEXT).contains(&b.context_tokens)
                && (1..=MAX_OUTPUT).contains(&b.output_allowance)
                && b.wall_millis > 0,
            "explicit prompt/context/output/wall bounds; no clipping"
        );
        ensure!(
            prompt_tokens
                .checked_add(b.output_allowance)
                .is_some_and(|n| n <= b.context_tokens),
            "reserve complete requested output allowance in context; do not shorten it"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Role {
    Embedding,
    Engram { layer: usize },
    SourceBf16Index { layer: usize },
    Attention { layer: usize },
    RoutedExperts { layer: usize },
    GlobalHead,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PlannedOperation {
    pub role: Role,
    pub wire: Operation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum StepRole {
    IntermediatePrefillHeadIgnored,
    FinalPrefillFirstOutput,
    DecodeOutput,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpectedBoundary {
    pub plan_sha256: String,
    pub boundary_sha256: String,
    pub step_index: usize,
    pub step_id: u64,
    pub operations_queued: usize,
    pub epoch: u64,
    pub consumed_native_context: usize,
    pub generated_native_count: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PlannedStep {
    pub step_index: usize,
    pub step_id: u64,
    pub role: StepRole,
    pub start: usize,
    pub tokens: usize,
    pub end: usize,
    pub operation_begin: usize,
    pub operation_end: usize,
    pub epoch_before: u64,
    pub epoch_after: u64,
    /// None means an intermediate prefill head is ignored and MUST NOT enter
    /// generated IDs, parser, token usage or a future decode input.
    pub output_ordinal: Option<usize>,
    pub expected_boundary: ExpectedBoundary,
}
#[derive(Debug)]
pub struct CpuRequestPlan {
    binding: RequestBinding,
    input_ids: Vec<u32>,
    input_sha256: String,
    first_operation_id: u64,
    next_operation_id: u64,
    first_step_id: u64,
    start_epoch: u64,
    end_epoch: u64,
    operations: Vec<PlannedOperation>,
    steps: Vec<PlannedStep>,
    sha256: String,
}
fn row_kind(start: usize, tokens: usize) -> RowKind {
    if start == 0 || tokens > 1 {
        RowKind::ContiguousPrefillRows
    } else {
        RowKind::IndependentDecodeLanes
    }
}
fn epochs(op: Operation) -> usize {
    match op {
        Operation::RoutedF32 { layout, .. } => layout.chunk_count() * ROUNDS,
        Operation::MatrixF32 { layout, .. } => layout.chunk_count() * ROUNDS,
        Operation::MatrixBf16 { layout, .. } => layout.epochs(),
        Operation::ShardedMaxloc { .. } => ROUNDS,
    }
}
fn step_fields(h: &mut Sha256, step: &PlannedStep) {
    for x in [
        step.step_index as u64,
        step.step_id,
        match step.role {
            StepRole::IntermediatePrefillHeadIgnored => 0,
            StepRole::FinalPrefillFirstOutput => 1,
            StepRole::DecodeOutput => 2,
        },
        step.start as u64,
        step.tokens as u64,
        step.end as u64,
        step.operation_begin as u64,
        step.operation_end as u64,
        step.epoch_before,
        step.epoch_after,
        step.output_ordinal.unwrap_or(0) as u64,
    ] {
        h.update(x.to_le_bytes());
    }
}
impl CpuRequestPlan {
    pub fn new(
        binding: RequestBinding,
        input_ids: Vec<u32>,
        first_operation_id: u64,
        first_step_id: u64,
        start_epoch: u64,
    ) -> Result<Self> {
        binding.validate(input_ids.len())?;
        ensure!(
            input_ids.iter().all(|x| *x < VOCAB),
            "input native token ID outside checkpoint vocabulary"
        );
        ensure!(
            first_operation_id > 0 && first_step_id > 0 && start_epoch % ROUNDS as u64 == 0,
            "positive monotonic IDs and three-round-aligned starting global epoch required"
        );
        let n = input_ids.len();
        let prefill_steps = n.div_ceil(PREFILL_CHUNK);
        let count = prefill_steps
            .checked_add(binding.budget.output_allowance - 1)
            .context("step countoverflow")?;
        ensure!(
            count <= MAX_STEPS && count.checked_mul(92).is_some_and(|n| n <= MAX_OPERATIONS),
            "CPU request schedule bound"
        );
        first_step_id
            .checked_add(count as u64)
            .context("global step ID exhausted")?;
        let mut plan = Self {
            input_sha256: native_ids_sha256(&input_ids),
            binding,
            input_ids,
            first_operation_id,
            next_operation_id: first_operation_id,
            first_step_id,
            start_epoch,
            end_epoch: start_epoch,
            operations: Vec::with_capacity(count * 92),
            steps: Vec::with_capacity(count),
            sha256: String::new(),
        };
        let mut at = 0;
        while at < n {
            let tokens = (n - at).min(PREFILL_CHUNK);
            let final_chunk = at + tokens == n;
            plan.push_step(
                at,
                tokens,
                if final_chunk {
                    StepRole::FinalPrefillFirstOutput
                } else {
                    StepRole::IntermediatePrefillHeadIgnored
                },
                final_chunk.then_some(1),
            )?;
            at += tokens;
        }
        for ordinal in 2..=plan.binding.budget.output_allowance {
            plan.push_step(n + ordinal - 2, 1, StepRole::DecodeOutput, Some(ordinal))?;
        }
        ensure!(
            plan.steps.len() == count && plan.operations.len() <= MAX_OPERATIONS,
            "full frozen schedule coverage"
        );
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        let encoded = serde_json::to_vec(&plan.binding)?;
        hash.update((encoded.len() as u64).to_le_bytes());
        hash.update(encoded);
        hash.update(plan.input_sha256.as_bytes());
        for x in [
            first_operation_id,
            first_step_id,
            start_epoch,
            count as u64,
            plan.operations.len() as u64,
            PREFILL_CHUNK as u64,
        ] {
            hash.update(x.to_le_bytes());
        }
        for step in &plan.steps {
            step_fields(&mut hash, step);
            for op in &plan.operations[step.operation_begin..step.operation_end] {
                let v = serde_json::to_vec(op)?;
                hash.update((v.len() as u64).to_le_bytes());
                hash.update(v);
                hash.update((epochs(op.wire) as u64).to_le_bytes());
            }
        }
        plan.sha256 = format!("{:x}", hash.finalize());
        for step in &mut plan.steps {
            let mut h = Sha256::new();
            h.update(BOUNDARY_DOMAIN);
            h.update(plan.sha256.as_bytes());
            step_fields(&mut h, step);
            step.expected_boundary = ExpectedBoundary {
                plan_sha256: plan.sha256.clone(),
                boundary_sha256: format!("{:x}", h.finalize()),
                step_index: step.step_index,
                step_id: step.step_id,
                operations_queued: step.operation_end,
                epoch: step.epoch_after,
                consumed_native_context: step.end,
                generated_native_count: step.output_ordinal.unwrap_or(0),
            };
        }
        Ok(plan)
    }
    fn push(&mut self, role: Role, make: impl FnOnce(u64) -> Operation) -> Result<()> {
        ensure!(
            self.operations.len() < MAX_OPERATIONS,
            "one-million-operation CPU bound"
        );
        let id = self.next_operation_id;
        self.next_operation_id = id.checked_add(1).context("global operation ID exhausted")?;
        let op = make(id);
        self.end_epoch = self
            .end_epoch
            .checked_add(epochs(op) as u64)
            .context("global GPU epoch would overflow")?;
        self.operations.push(PlannedOperation { role, wire: op });
        Ok(())
    }
    fn push_step(
        &mut self,
        start: usize,
        tokens: usize,
        role: StepRole,
        output_ordinal: Option<usize>,
    ) -> Result<()> {
        ensure!(self.steps.len() < MAX_STEPS, "8192-step CPU bound");
        let end = start.checked_add(tokens).context("step extentoverflow")?;
        chunk_plan::window_chunk(start, tokens)?;
        let layout = RoutedLayout::new(tokens, row_kind(start, tokens))?;
        let begin = self.operations.len();
        let epoch_before = self.end_epoch;
        self.push(Role::Embedding, |id| Operation::RoutedF32 { id, layout })?;
        for layer in 0..40 {
            if layer == 1 || layer == 14 {
                let layout = MatrixF32::new(MatrixRole::EngramLookup, tokens, 6144)?;
                self.push(Role::Engram { layer }, |id| Operation::MatrixF32 {
                    id,
                    layout,
                })?;
            }
            if INDEX_SOURCES.contains(&layer) {
                let ratio = if layer < 20 { 2 } else { 1 };
                let keys = chunk_plan::compressed_chunk(start, tokens, ratio)?.stored_rows;
                if keys > 512 {
                    let layout = MatrixBf16Index::new(tokens, keys)?;
                    self.push(Role::SourceBf16Index { layer }, |id| {
                        Operation::MatrixBf16 { id, layout }
                    })?;
                }
            }
            self.push(Role::Attention { layer }, |id| Operation::RoutedF32 {
                id,
                layout,
            })?;
            self.push(Role::RoutedExperts { layer }, |id| Operation::RoutedF32 {
                id,
                layout,
            })?;
        }
        let head = RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?;
        self.push(Role::GlobalHead, |id| Operation::ShardedMaxloc {
            id,
            layout: head,
        })?;
        let index = self.steps.len();
        let step_id = self
            .first_step_id
            .checked_add(index as u64)
            .context("step IDoverflow")?;
        self.steps.push(PlannedStep {
            step_index: index,
            step_id,
            role,
            start,
            tokens,
            end,
            operation_begin: begin,
            operation_end: self.operations.len(),
            epoch_before,
            epoch_after: self.end_epoch,
            output_ordinal,
            expected_boundary: ExpectedBoundary {
                plan_sha256: String::new(),
                boundary_sha256: String::new(),
                step_index: index,
                step_id,
                operations_queued: self.operations.len(),
                epoch: self.end_epoch,
                consumed_native_context: end,
                generated_native_count: output_ordinal.unwrap_or(0),
            },
        });
        Ok(())
    }
    pub fn digest(&self) -> &str {
        &self.sha256
    }
    pub fn binding(&self) -> &RequestBinding {
        &self.binding
    }
    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }
    pub fn input_digest(&self) -> &str {
        &self.input_sha256
    }
    pub fn steps(&self) -> &[PlannedStep] {
        &self.steps
    }
    pub fn operations(&self) -> &[PlannedOperation] {
        &self.operations
    }
    pub fn step_operations(&self, index: usize) -> Result<&[PlannedOperation]> {
        let s = self
            .steps
            .get(index)
            .context("planned step outside frozen schedule")?;
        Ok(&self.operations[s.operation_begin..s.operation_end])
    }
    pub fn start_epoch(&self) -> u64 {
        self.start_epoch
    }
    pub fn end_epoch(&self) -> u64 {
        self.end_epoch
    }
    pub fn first_operation_id(&self) -> u64 {
        self.first_operation_id
    }
    pub fn next_operation_id(&self) -> u64 {
        self.next_operation_id
    }
    /// Shape/identity compatibility ONLY. A caller may freely construct a CPU
    /// proposal, so success here proves NO device execution, completion, parser
    /// terminal, all-rank agreement, correct answer or successful agent task.
    pub fn check_stop_proposal(&self, proposal: &CpuStopProposal) -> Result<CpuStopCompatibility> {
        let expected = self
            .steps
            .get(proposal.boundary.step_index)
            .context("stop step outside plan")?;
        ensure!(
            proposal.boundary == expected.expected_boundary,
            "stale/partial/wrong source or complete-step boundary"
        );
        ensure!(
            proposal.generated_native_ids.len() == expected.output_ordinal.unwrap_or(0)
                && proposal.generated_native_ids.iter().all(|x| *x < VOCAB),
            "intermediate heads ignored; generated native count/vocabulary differs"
        );
        let ids = &proposal.generated_native_ids;
        let eos = ids.iter().filter(|&&x| x == EOS_ID).count();
        ensure!(
            eos == 0 || (eos == 1 && ids.last() == Some(&EOS_ID)),
            "no generated native IDs may follow EOS"
        );
        let output_count = ids.len();
        let b = &self.binding.budget;
        let (label, natural_candidate) = match &proposal.reason {
            ProposedStop::NativeEos => {
                ensure!(
                    output_count > 0 && ids.last() == Some(&EOS_ID) && eos == 1,
                    "EOS exactly once at final generated position"
                );
                ("proposed_native_EOS", true)
            }
            ProposedStop::OutputAllowance => {
                ensure!(
                    output_count == b.output_allowance && eos == 0,
                    "cap consumes full allowance without EOS; EOS takes precedence at cap"
                );
                ("proposed_output_allowance", false)
            }
            ProposedStop::UserSequence { sequence } => {
                ensure!(
                    output_count > 0
                        && eos == 0
                        && self.binding.stop.user_sequences.contains(sequence),
                    "stop sequence not bound or EOS precedence differs"
                );
                ("proposed_user_stop_sequence_parser_unverified", true)
            }
            ProposedStop::WallTimeBudget => {
                ensure!(
                    proposal.reported_elapsed_millis >= b.wall_millis,
                    "wall deadline not reached"
                );
                ("proposed_wall_time_budget_failure", false)
            }
        };
        if !matches!(&proposal.reason, ProposedStop::WallTimeBudget) {
            ensure!(
                proposal.reported_elapsed_millis <= b.wall_millis,
                "completion claimed beyond declared wall budget"
            );
        }
        Ok(CpuStopCompatibility {
            plan_sha256: self.sha256.clone(),
            boundary_sha256: proposal.boundary.boundary_sha256.clone(),
            terminal_label: label,
            generated_native_count: output_count,
            generated_native_ids_sha256: native_ids_sha256(ids),
            natural_stop_proposed: natural_candidate,
            CPU_compatibility_only: true,
            runtime_completion_authority: false,
            all_rank_prefix_agreement_proven: false,
            model_or_task_correctness_proven: false,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum ProposedStop {
    NativeEos,
    OutputAllowance,
    UserSequence { sequence: String },
    WallTimeBudget,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CpuStopProposal {
    pub boundary: ExpectedBoundary,
    pub generated_native_ids: Vec<u32>,
    pub reason: ProposedStop,
    pub reported_elapsed_millis: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct CpuStopCompatibility {
    pub plan_sha256: String,
    pub boundary_sha256: String,
    pub terminal_label: &'static str,
    pub generated_native_count: usize,
    pub generated_native_ids_sha256: String,
    pub natural_stop_proposed: bool,
    pub CPU_compatibility_only: bool,
    pub runtime_completion_authority: bool,
    pub all_rank_prefix_agreement_proven: bool,
    pub model_or_task_correctness_proven: bool,
}
#[cfg(test)]
mod tests {
    use super::super::{
        gpu_block::BlockOperationIds, gpu_collectives::Program, gpu_model::ModelOperationIds,
    };
    use super::*;
    use uuid::Uuid;
    fn binding(output: usize) -> RequestBinding {
        RequestBinding {
            model_revision: MODEL_REVISION.into(),
            global_checkpoint_geometry_checksum_sha256: "a".repeat(64),
            execution_snapshot_sha256: "b".repeat(64),
            engine_binary_sha256: "c".repeat(64),
            original_request_sha256: "d".repeat(64),
            tokenizer_sha256: TOKENIZER_SHA256.into(),
            toolkit_revision: TOOLKIT_REVISION.into(),
            frequency: FrequencyBackend::CudaTorch.identity(),
            sampling: GreedySemantics {
                temperature: 0.,
                top_p: 1.,
                presence_penalty: 0.,
                frequency_penalty: 0.,
                seed: Some(42),
                thinking: false,
                effective_reasoning_score: None,
                speculation: false,
            },
            stop: StopPolicy {
                eos_native_id: EOS_ID,
                user_sequences: vec!["STOP".into()],
            },
            budget: RequestBudget {
                context_tokens: 8192,
                output_allowance: output,
                wall_millis: 180_000,
            },
            lane: LaneKey {
                boot: Uuid::from_u128(417),
                slot: 0,
                generation: 1,
            },
        }
    }
    fn plan(input: usize, output: usize) -> CpuRequestPlan {
        CpuRequestPlan::new(binding(output), vec![17; input], 1, 1, 0).unwrap()
    }
    fn independent_geometry_reference(p: &CpuRequestPlan, name: &str) {
        let raw = include_bytes!("../../tests/fixtures/dsv41_model_session_plan_vectors.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            "52a5705dac09a0ae4911debfcc4b3453c4a61032c281898edee3fa4dd997b152"
        );
        let v: serde_json::Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(v["CPU_only"], true);
        assert_eq!(v["GPU_imports"], false);
        assert_eq!(
            v["F32_epoch_elements"].as_u64().unwrap() as usize,
            crate::tp_xport::EPOCH_FLOATS
        );
        let case = v["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        assert_eq!(case["steps"].as_u64().unwrap() as usize, p.steps.len());
        assert_eq!(
            case["operations"].as_u64().unwrap() as usize,
            p.operations.len()
        );
        assert_eq!(case["end_epoch"].as_u64().unwrap(), p.end_epoch);
        let mut operations = Sha256::new();
        operations.update(b"dsv41-session-cpu-operation-reference-v1\0");
        for op in &p.operations {
            let (role, layer) = match op.role {
                Role::Embedding => (1, 40),
                Role::Engram { layer } => (2, layer),
                Role::SourceBf16Index { layer } => (3, layer),
                Role::Attention { layer } => (4, layer),
                Role::RoutedExperts { layer } => (5, layer),
                Role::GlobalHead => (6, 40),
            };
            let (rows, cols, kind) = match op.wire {
                Operation::RoutedF32 { layout, .. } => (
                    layout.rows(),
                    5120,
                    if layout.kind() == RowKind::IndependentDecodeLanes {
                        1
                    } else {
                        2
                    },
                ),
                Operation::ShardedMaxloc { layout, .. } => (layout.rows(), 2, 1),
                Operation::MatrixF32 { layout, .. } => (layout.rows(), layout.cols(), 0),
                Operation::MatrixBf16 { layout, .. } => (layout.rows(), layout.cols(), 0),
            };
            for word in [
                role as u64,
                layer as u64,
                op.wire.id(),
                rows as u64,
                cols as u64,
                kind as u64,
                epochs(op.wire) as u64,
            ] {
                operations.update(word.to_le_bytes());
            }
        }
        assert_eq!(
            format!("{:x}", operations.finalize()),
            case["operation_geometry_sha256"].as_str().unwrap()
        );
        let mut steps = Sha256::new();
        steps.update(b"dsv41-session-cpu-step-reference-v1\0");
        for step in &p.steps {
            for word in [
                step.step_index as u64,
                step.start as u64,
                step.tokens as u64,
                step.end as u64,
                step.output_ordinal.unwrap_or(0) as u64,
                step.operation_begin as u64,
                step.operation_end as u64,
                step.epoch_before,
                step.epoch_after,
            ] {
                steps.update(word.to_le_bytes());
            }
        }
        assert_eq!(
            format!("{:x}", steps.finalize()),
            case["step_geometry_sha256"].as_str().unwrap()
        );
    }
    #[test]
    fn old_short_75_plus32_uses_the_exact84_operation_order_and_layouts() {
        let p = plan(75, 32);
        independent_geometry_reference(&p, "old75_output32");
        assert_eq!(p.steps.len(), 32);
        assert_eq!(p.operations.len(), 32 * 84);
        for step in &p.steps {
            let wire = p
                .step_operations(step.step_index)
                .unwrap()
                .iter()
                .map(|x| x.wire)
                .collect::<Vec<_>>();
            let mut ids = ModelOperationIds {
                embedding: 0,
                engram: [0; 2],
                blocks: vec![
                    BlockOperationIds {
                        attention: 0,
                        routed_experts: 0
                    };
                    40
                ],
                head: 0,
            };
            for op in p.step_operations(step.step_index).unwrap() {
                let id = op.wire.id();
                match op.role {
                    Role::Embedding => ids.embedding = id,
                    Role::Engram { layer } => ids.engram[usize::from(layer == 14)] = id,
                    Role::Attention { layer } => ids.blocks[layer].attention = id,
                    Role::RoutedExperts { layer } => ids.blocks[layer].routed_experts = id,
                    Role::GlobalHead => ids.head = id,
                    Role::SourceBf16Index { .. } => panic!("short path unexpectedly indexed"),
                }
            }
            assert_eq!(wire, ids.operations(step.start, step.tokens).unwrap());
        }
        assert!(Program::new_model_sequence(p.operations.iter().map(|x| x.wire).collect()).is_ok());
    }
    #[test]
    fn full4096_allowance_prefill_heads_ignored_and_legacy_program_cap_unchanged() {
        let p = plan(4096, 4096);
        independent_geometry_reference(&p, "full4096_output4096");
        assert_eq!(p.steps.len(), 4127);
        assert!(p.operations.len() > 32768 && p.operations.len() < 400000);
        assert_eq!(
            p.steps
                .iter()
                .filter(|s| s.output_ordinal.is_none())
                .count(),
            31
        );
        assert_eq!(
            p.steps
                .iter()
                .filter(|s| s.output_ordinal.is_some())
                .count(),
            4096
        );
        assert_eq!(p.steps.last().unwrap().output_ordinal, Some(4096));
        assert_eq!(p.steps.last().unwrap().end, 8191);
        assert_eq!(super::super::gpu_collectives::MAX_MODEL_OPERATIONS, 32768);
        assert!(
            Program::new_model_sequence(p.operations.iter().map(|x| x.wire).collect()).is_err()
        );
        assert!(p
            .operations
            .windows(2)
            .all(|w| w[0].wire.id() + 1 == w[1].wire.id()));
        assert!(p
            .steps
            .windows(2)
            .all(|w| w[0].epoch_after == w[1].epoch_before));
    }
    #[test]
    fn optional_source_BF16_keys_grow_at_exact513_and_odd1025_boundaries() {
        let p = plan(1025, 3);
        independent_geometry_reference(&p, "odd1025_output3");
        let last = p
            .steps
            .iter()
            .find(|s| s.role == StepRole::FinalPrefillFirstOutput)
            .unwrap();
        assert_eq!((last.start, last.tokens), (1024, 1));
        for (step, expected) in [(last.step_index, 5usize), (last.step_index + 1, 8)] {
            let ops = p.step_operations(step).unwrap();
            let index = ops
                .iter()
                .filter(|x| matches!(x.role, Role::SourceBf16Index { .. }))
                .collect::<Vec<_>>();
            assert_eq!(index.len(), expected);
            for item in index {
                let layer = match item.role {
                    Role::SourceBf16Index { layer } => layer,
                    _ => unreachable!(),
                };
                let matrix = item.wire.bf16_layout().unwrap();
                assert_eq!(matrix.rows(), 1);
                assert_eq!(
                    matrix.cols(),
                    p.steps[step].end / if layer < 20 { 2 } else { 1 }
                );
                assert_eq!(matrix.epochs(), matrix.chunk_count() * 3);
                let at = ops
                    .iter()
                    .position(|x| x.wire.id() == item.wire.id())
                    .unwrap();
                assert_eq!(ops[at + 1].role, Role::Attention { layer });
            }
        }
        let p = plan(513, 1);
        independent_geometry_reference(&p, "index513_output1");
        let last = p.steps.last().unwrap();
        assert_eq!(last.operation_end - last.operation_begin, 89);
        let p = plan(640, 1);
        let last = p.steps.last().unwrap();
        assert_eq!(last.tokens, 128);
        assert!(p
            .step_operations(last.step_index)
            .unwrap()
            .iter()
            .filter_map(|x| x.wire.bf16_layout())
            .all(|l| l.rows() == 128 && l.cols() == 640));
    }
    #[test]
    fn EOS_cap_parser_timeout_and_stale_boundary_are_distinct_CPU_proposals() {
        let p = plan(129, 2);
        independent_geometry_reference(&p, "prefill129_output2");
        let last = p.steps.last().unwrap();
        let mut s = CpuStopProposal {
            boundary: last.expected_boundary.clone(),
            generated_native_ids: vec![2, EOS_ID],
            reason: ProposedStop::NativeEos,
            reported_elapsed_millis: 100,
        };
        let c = p.check_stop_proposal(&s).unwrap();
        assert!(
            c.CPU_compatibility_only
                && !c.runtime_completion_authority
                && !c.all_rank_prefix_agreement_proven
                && !c.model_or_task_correctness_proven
        );
        s.reason = ProposedStop::OutputAllowance;
        assert!(p.check_stop_proposal(&s).is_err());
        s.generated_native_ids = vec![2, 3];
        p.check_stop_proposal(&s).unwrap();
        s.reason = ProposedStop::UserSequence {
            sequence: "STOP".into(),
        };
        p.check_stop_proposal(&s).unwrap();
        s.reason = ProposedStop::UserSequence {
            sequence: "wrong".into(),
        };
        assert!(p.check_stop_proposal(&s).is_err());
        s.reason = ProposedStop::WallTimeBudget;
        assert!(p.check_stop_proposal(&s).is_err());
        s.reported_elapsed_millis = 180000;
        p.check_stop_proposal(&s).unwrap();
        let valid = s.clone();
        for field in 0..6 {
            s = valid.clone();
            match field {
                0 => s.boundary.epoch += 3,
                1 => s.boundary.operations_queued -= 1,
                2 => s.boundary.boundary_sha256 = "0".repeat(64),
                3 => s.boundary.plan_sha256 = "0".repeat(64),
                4 => s.boundary.step_id += 1,
                _ => s.boundary.generated_native_count -= 1,
            };
            assert!(p.check_stop_proposal(&s).is_err());
        }
        let mut newer_binding = binding(2);
        newer_binding.lane.generation += 1;
        let newer = CpuRequestPlan::new(newer_binding, vec![17; 129], 1, 1, 0).unwrap();
        assert!(newer.check_stop_proposal(&valid).is_err());
        s = valid;
        s.boundary = p.steps[0].expected_boundary.clone();
        s.generated_native_ids = vec![EOS_ID];
        s.reason = ProposedStop::NativeEos;
        assert!(p.check_stop_proposal(&s).is_err());
    }
    #[test]
    fn exact_index_posture_and_rows_match_real_ChunkCursor_not_only_threshold_prose() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let config =
            super::super::config::Dsv41Config::from_values(&f["hf_config"], &f["inference_config"])
                .unwrap();
        let p = plan(1025, 3);
        let key = p.binding.lane;
        let mut cursor = chunk_plan::ChunkCursor::new(&config, key.boot, 1).unwrap();
        cursor.reset(key).unwrap();
        for step in &p.steps {
            let ticket = cursor.begin(key, step.start, step.tokens).unwrap();
            let ops = p.step_operations(step.step_index).unwrap();
            for layer in 0..40 {
                let actual = cursor.layer(&ticket, layer).unwrap();
                let found = ops
                    .iter()
                    .find(|x| x.role == Role::SourceBf16Index { layer });
                assert_eq!(found.is_some(), actual.index_scoring_required);
                if let Some(op) = found {
                    let l = op.wire.bf16_layout().unwrap();
                    assert_eq!(l.rows(), step.tokens);
                    assert_eq!(l.cols(), actual.compressed.as_ref().unwrap().stored_rows);
                }
                cursor.commit_layer(&actual).unwrap();
            }
            cursor.commit(&ticket).unwrap();
        }
    }
    #[test]
    fn full_budget_identity_sampling_and_monotonic_counter_overflow_fail_closed() {
        assert!(CpuRequestPlan::new(binding(4096), vec![0; 4097], 1, 1, 0).is_err());
        assert!(CpuRequestPlan::new(binding(4097), vec![0], 1, 1, 0).is_err());
        assert!(CpuRequestPlan::new(binding(1), vec![], 1, 1, 0).is_err());
        assert!(CpuRequestPlan::new(binding(1), vec![VOCAB], 1, 1, 0).is_err());
        for (id, step, epoch) in [
            (0, 1, 0),
            (1, 0, 0),
            (u64::MAX - 3, 1, 0),
            (1, u64::MAX, 0),
            (1, 1, u64::MAX - 3),
            (1, 1, 1),
        ] {
            assert!(CpuRequestPlan::new(binding(1), vec![0], id, step, epoch).is_err());
        }
        let original = plan(75, 2);
        for field in 0..11 {
            let mut b = binding(2);
            match field {
                0 => b.global_checkpoint_geometry_checksum_sha256 = "e".repeat(64),
                1 => b.execution_snapshot_sha256 = "e".repeat(64),
                2 => b.engine_binary_sha256 = "e".repeat(64),
                3 => b.original_request_sha256 = "e".repeat(64),
                4 => b.frequency = FrequencyBackend::CpuTorch.identity(),
                5 => b.sampling.thinking = true,
                6 => b.sampling.seed = Some(43),
                7 => b.stop.user_sequences.push("END".into()),
                8 => b.lane.generation += 1,
                9 => b.budget.output_allowance = 3,
                _ => b.budget.wall_millis += 1,
            };
            assert_ne!(
                CpuRequestPlan::new(b, vec![17; 75], 1, 1, 0)
                    .unwrap()
                    .digest(),
                original.digest()
            );
        }
        let mut b = binding(1);
        b.sampling.temperature = 1e-50;
        assert!(CpuRequestPlan::new(b, vec![0], 1, 1, 0).is_err());
        let mut b = binding(1);
        b.frequency.yarn_table_sha256 = "0".repeat(64);
        assert!(CpuRequestPlan::new(b, vec![0], 1, 1, 0).is_err());
        assert_ne!(
            CpuRequestPlan::new(binding(2), vec![18; 75], 1, 1, 0)
                .unwrap()
                .digest(),
            original.digest()
        );
        let next = CpuRequestPlan::new(
            binding(2),
            vec![17; 75],
            original.next_operation_id(),
            33,
            original.end_epoch(),
        )
        .unwrap();
        assert!(next.start_epoch() > 0);
        assert_ne!(next.digest(), original.digest());
    }
}
