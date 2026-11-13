//! PRIVATE additive C1 chunk Engram control. The original Engram entrypoint
//! still refuses positive-start multiple rows. This wrapper binds an exact
//! 8192-context ChunkTicket before sharing the unchanged checked row math.
//! The caller computes HashRows once with the official transactional CPU hash
//! state; this component never commits that history or the full cache prefix.
use super::{
    chunk_plan::{self, ChunkTicket},
    config::Dsv41Config,
    engram::{EngramHashPlan, HashRows, HASH_COLUMNS},
    gpu_chunk_store::GpuChunkStore,
    gpu_collectives::GpuCollectives,
    gpu_engram_block::{EngramBlockKernels, EngramOutputIdentity, GpuEngramBlock},
    gpu_weights::LoadedWeights,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

const DIM: usize = 5120;
const COPIES: usize = 4;
const MAX_CONTEXT: usize = 8192;

fn extent(ticket: &ChunkTicket, hash_rows: usize) -> Result<usize> {
    ensure!(
        ticket.lane.slot == 0
            && ticket.lane.generation > 0
            && ticket.version > 0
            && (1..=128).contains(&ticket.tokens)
            && ticket.tokens == hash_rows
            && ticket.start.checked_add(ticket.tokens) == Some(ticket.end)
            && ticket.end <= MAX_CONTEXT,
        "chunk Engram ticket extent/generation/version differs"
    );
    Ok(ticket.end)
}

/// Hashes BOTH layers in token/layer/column order. Each single-layer receipt
/// still retains its own existing hash. Matching this digest associates both
/// blocks with one immutable HashRows batch; it is not a tokenizer/BPE proof.
fn rows_digest(plan: &EngramHashPlan, rows: &[HashRows]) -> Result<String> {
    ensure!((1..=128).contains(&rows.len()), "chunk hash row bound");
    let mut hash = Sha256::new();
    for row in rows {
        for (layer, ids) in row.iter().enumerate() {
            for (column, &id) in ids.iter().enumerate() {
                let first = plan.offsets[layer][column];
                let end = first
                    .checked_add(plan.primes[layer][column])
                    .context("chunk hash bucket overflow")?;
                ensure!(
                    id >= first && id < end && id < plan.num_embeddings[layer],
                    "chunk Engram hash ID outside its source bucket"
                );
                hash.update(id.to_le_bytes());
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone, Debug)]
struct Progress {
    boot: Uuid,
    lane: Option<LaneKey>,
    position: usize,
    last_attempted_version: Option<u64>,
    last_attempted_operation: Option<u64>,
    owner: Option<[u8; 32]>,
    failed: bool,
}
impl Progress {
    fn new(boot: Uuid) -> Self {
        Self {
            boot,
            lane: None,
            position: 0,
            last_attempted_version: None,
            last_attempted_operation: None,
            owner: None,
            failed: true,
        }
    }
    fn validate_reset(&self, lane: LaneKey) -> Result<()> {
        ensure!(
            lane.boot == self.boot
                && lane.slot == 0
                && lane.generation > 0
                && self.lane.is_none_or(|old| lane.generation > old.generation),
            "chunk Engram reset must advance C1 owner generation"
        );
        Ok(())
    }
    fn reset_after_inner_checked(&mut self, lane: LaneKey) {
        self.lane = Some(lane);
        self.position = 0;
        self.failed = false;
        // Completion owner, attempted versions and opIDs persist across reset.
        // Reset is never an epoch reset or permission to reuse an attempted ID.
    }
    fn validate(
        &self,
        ticket: &ChunkTicket,
        count: usize,
        owner: [u8; 32],
        operation: u64,
    ) -> Result<()> {
        extent(ticket, count)?;
        ensure!(
            !self.failed
                && self.lane == Some(ticket.lane)
                && ticket.start == self.position
                && self
                    .last_attempted_version
                    .is_none_or(|v| ticket.version > v)
                && self
                    .last_attempted_operation
                    .is_none_or(|id| operation > id)
                && self.owner.is_none_or(|old| old == owner),
            "stale/failed/noncontiguous chunk Engram ticket/session/opID"
        );
        Ok(())
    }
    fn reserve(&mut self, ticket: &ChunkTicket, owner: [u8; 32], operation: u64) {
        self.last_attempted_version = Some(ticket.version);
        self.last_attempted_operation = Some(operation);
        self.owner = Some(owner);
    }
}

#[derive(Debug, Serialize)]
pub struct ChunkEngramReceipt {
    pub ticket: ChunkTicket,
    pub rank: usize,
    pub layer: usize,
    pub manifest_sha256: String,
    pub completion_owner: [u8; 32],
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub operation_id: u64,
    pub all_hash_rows_sha256_u64le: String,
    pub hash_rows_source: String,
    pub source: EngramOutputIdentity,
    pub all_checks_passed: bool,
    pub active_store_ticket_checked: bool,
    pub component_position_advanced: bool,
    pub CPU_hash_or_native_history_committed: bool,
    pub full_cache_prefix_committed: bool,
    pub original_reference_guard_preserved: bool,
    pub full_model_or_quality_qualified: bool,
    pub optimized_execution: bool,
}
pub struct CheckedChunkEngram {
    pub h: CudaSlice<bf16>,
    pub gates: CudaSlice<f32>,
    pub receipt: ChunkEngramReceipt,
}

pub struct GpuEngramChunk {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    rank: usize,
    layer: usize,
    manifest: String,
    hash_plan: EngramHashPlan,
    inner: GpuEngramBlock,
    progress: Progress,
}
impl GpuEngramChunk {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        boot: Uuid,
        kernels: Arc<EngramBlockKernels>,
        config: &Dsv41Config,
        manifest: &RankManifest,
        layer: usize,
        weights: Arc<LoadedWeights>,
    ) -> Result<Self> {
        chunk_plan::validate_config(config)?;
        let inner = GpuEngramBlock::new(kernels, config, manifest, layer, weights)?;
        let me = Self {
            dev: dev.clone(),
            stream,
            rank: manifest.rank,
            layer,
            manifest: manifest.manifest_sha256.clone(),
            hash_plan: EngramHashPlan::official_1017(config)?,
            inner,
            progress: Progress::new(boot),
        };
        me.check_stream()?;
        Ok(me)
    }
    fn check_stream(&self) -> Result<()> {
        self.dev.bind_to_thread()?;
        let mut flags = 0;
        ensure!(
            !self.stream.stream.is_null()
                && unsafe { sys::cuStreamGetFlags(self.stream.stream, &mut flags) }
                    == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "owned blocking chunk Engram stream required"
        );
        Ok(())
    }
    pub fn reset_lane(&mut self, lane: LaneKey) -> Result<()> {
        self.check_stream()?;
        self.progress.validate_reset(lane)?;
        self.progress.failed = true;
        self.inner.reset_lane(&self.stream, lane)?;
        self.progress.reset_after_inner_checked(lane);
        Ok(())
    }
    /// Source HashRows are caller-computed once using a staged official
    /// EngramHashState append. Do not commit that CPU state until the whole
    /// model chunk (including final head where declared) succeeds.
    pub fn execute_checked(
        &mut self,
        ticket: &ChunkTicket,
        h: &CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation: u64,
        store: &GpuChunkStore,
        collectives: &mut GpuCollectives,
    ) -> Result<CheckedChunkEngram> {
        let result = self.execute_inner(ticket, h, hash_rows, operation, store, collectives);
        if result.is_err() {
            self.progress.failed = true;
            collectives.abort_session();
            // Existing row primitives may fence during inner failure/Drop.
            // This ordering only promises abort before THIS explicit fence.
            let _ = self.dev.synchronize();
        }
        result
    }
    fn execute_inner(
        &mut self,
        ticket: &ChunkTicket,
        h: &CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation: u64,
        store: &GpuChunkStore,
        collectives: &mut GpuCollectives,
    ) -> Result<CheckedChunkEngram> {
        self.check_stream()?;
        let owner = collectives.completion_identity();
        store.check_owner(self.rank, &self.manifest, owner, &self.stream)?;
        let layer_plan = store.layer(ticket, self.layer)?;
        ensure!(
            layer_plan.ticket == *ticket && layer_plan.layer == self.layer,
            "chunk Engram must precede its actual next cache layer"
        );
        self.progress
            .validate(ticket, hash_rows.len(), owner, operation)?;
        ensure!(
            collectives.rank() == self.rank
                && collectives.compute_stream().stream == self.stream.stream
                && h.len() == ticket.tokens * COPIES * DIM
                && Arc::ptr_eq(&h.device(), &self.dev),
            "chunk Engram residual/collective rank/stream/context differs"
        );
        let digest = rows_digest(&self.hash_plan, hash_rows)?;
        let epoch_before = collectives.expected_epoch();
        self.progress.reserve(ticket, owner, operation);
        let pending = self.inner.enqueue_chunk_rows(
            &self.stream,
            ticket,
            h,
            hash_rows,
            operation,
            collectives,
        )?;
        let output = pending.finish(collectives)?;
        ensure!(
            output.identity.all_checks_passed
                && output.identity.lane == ticket.lane
                && output.identity.start == ticket.start
                && output.identity.tokens == ticket.tokens
                && output.identity.rank == self.rank
                && output.identity.layer == self.layer
                && output.identity.manifest_sha256 == self.manifest
                && output.identity.operation_id == operation
                && output.identity.completion_identity == owner
                && output.identity.step.is_none(),
            "checked chunk Engram source receipt differs"
        );
        self.progress.position = ticket.end;
        Ok(CheckedChunkEngram {
            h: output.h,
            gates: output.gates,
            receipt: ChunkEngramReceipt {
                ticket: ticket.clone(),
                rank: self.rank,
                layer: self.layer,
                manifest_sha256: self.manifest.clone(),
                completion_owner: owner,
                epoch_before,
                epoch_after: collectives.expected_epoch(),
                operation_id: operation,
                all_hash_rows_sha256_u64le: digest,
                hash_rows_source: "caller-staged official EngramHashState; both layers token/layer/24-column order".into(),
                source: output.identity,
                all_checks_passed: true,
                active_store_ticket_checked: true,
                component_position_advanced: true,
                CPU_hash_or_native_history_committed: false,
                full_cache_prefix_committed: false,
                original_reference_guard_preserved: true,
                full_model_or_quality_qualified: false,
                optimized_execution: false,
            },
        })
    }
}
impl Drop for GpuEngramChunk {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ticket(start: usize, tokens: usize, version: u64) -> ChunkTicket {
        ChunkTicket {
            lane: LaneKey {
                boot: Uuid::from_u128(7),
                slot: 0,
                generation: 1,
            },
            version,
            start,
            tokens,
            end: start + tokens,
        }
    }
    fn plan() -> EngramHashPlan {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        let c = Dsv41Config::from_values(&v["hf_config"], &v["inference_config"]).unwrap();
        EngramHashPlan::official_1017(&c).unwrap()
    }
    #[test]
    fn arbitrary_positive_chunks_reach_exact8192_and_reject_bad_tickets() {
        for (start, tokens) in [(0, 1), (0, 128), (1, 2), (127, 128), (8064, 128), (8191, 1)] {
            assert_eq!(
                extent(&ticket(start, tokens, 1), tokens).unwrap(),
                start + tokens
            );
        }
        for (start, tokens) in [(0, 0), (0, 129), (8065, 128), (8192, 1)] {
            assert!(extent(&ticket(start, tokens, 1), tokens).is_err());
        }
        let mut bad = ticket(1, 2, 1);
        bad.end += 1;
        assert!(extent(&bad, 2).is_err());
        bad = ticket(1, 2, 0);
        assert!(extent(&bad, 2).is_err());
        bad = ticket(1, 2, 1);
        bad.lane.slot = 1;
        assert!(extent(&bad, 2).is_err());
        assert!(extent(&ticket(1, 2, 1), 1).is_err());
    }
    #[test]
    fn transaction_version_owner_and_attempted_ids_survive_reset() {
        let first = ticket(0, 128, 1);
        let owner = [9; 32];
        let mut p = Progress::new(first.lane.boot);
        p.validate_reset(first.lane).unwrap();
        p.reset_after_inner_checked(first.lane);
        p.validate(&first, 128, owner, 7).unwrap();
        p.reserve(&first, owner, 7);
        p.position = first.end;
        let next = ticket(128, 128, 2);
        p.validate(&next, 128, owner, 8).unwrap();
        assert!(p.validate(&next, 128, [8; 32], 8).is_err());
        assert!(p.validate(&next, 128, owner, 7).is_err());
        let mut stale = next.clone();
        stale.version = 1;
        assert!(p.validate(&stale, 128, owner, 8).is_err());
        p.failed = true;
        assert!(p.validate(&next, 128, owner, 8).is_err());
        let mut new_lane = first.lane;
        new_lane.generation = 2;
        p.validate_reset(new_lane).unwrap();
        p.reset_after_inner_checked(new_lane);
        let mut reset = ticket(0, 128, 3);
        reset.lane = new_lane;
        p.validate(&reset, 128, owner, 8).unwrap();
        assert!(p.validate(&reset, 128, [8; 32], 8).is_err());
        assert!(p.validate(&reset, 128, owner, 7).is_err());
        reset.version = 1;
        assert!(p.validate(&reset, 128, owner, 8).is_err());
    }
    #[test]
    fn full_rows_digest_binds_both_layers_order_and_buckets() {
        let plan = plan();
        let row = plan.offsets;
        let base = rows_digest(&plan, &[row, row]).unwrap();
        let mut changed = row;
        changed[1][HASH_COLUMNS - 1] += 1;
        assert_ne!(base, rows_digest(&plan, &[row, changed]).unwrap());
        assert_ne!(
            rows_digest(&plan, &[row, changed]).unwrap(),
            rows_digest(&plan, &[changed, row]).unwrap()
        );
        let mut bad = row;
        bad[0][0] = plan.offsets[0][0] + plan.primes[0][0];
        assert!(rows_digest(&plan, &[bad]).is_err());
        assert!(rows_digest(&plan, &[]).is_err());
    }
    #[test]
    fn chunk_metadata_matches_consecutive_native_position_progress() {
        for chunks in [
            vec![128; 64],
            vec![1; 8192],
            vec![75, 128, 1, 127, 65, 128, 3],
        ] {
            let mut p = Progress::new(Uuid::from_u128(7));
            let lane = ticket(0, 1, 1).lane;
            p.reset_after_inner_checked(lane);
            let mut reference = 0;
            for (i, n) in chunks.into_iter().enumerate() {
                let t = ticket(reference, n, i as u64 + 1);
                p.validate(&t, n, [1; 32], i as u64).unwrap();
                p.reserve(&t, [1; 32], i as u64);
                for _ in 0..n {
                    reference += 1;
                }
                p.position = t.end;
                assert_eq!(p.position, reference);
            }
        }
    }
}
