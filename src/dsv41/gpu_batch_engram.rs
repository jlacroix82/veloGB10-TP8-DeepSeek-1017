//! One shared native Engram lookup/projection/gate for a real packed model step.
//! CPU hash rows are staged separately per physical lane by the full-model
//! owner; they are never derived by concatenating different users' histories.
use super::{
    batch_plan::BatchTicket,
    config::Dsv41Config,
    engram::HashRows,
    gpu_batch_store::GpuBatchStore,
    gpu_collectives::GpuCollectives,
    gpu_engram_block::{EngramBlockKernels, EngramOutput, GpuEngramBlock},
    gpu_weights::LoadedWeights,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use half::bf16;
use std::sync::Arc;

pub struct GpuBatchEngram {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    inner: GpuEngramBlock,
}
impl GpuBatchEngram {
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: Arc<CudaStream>,
        kernels: Arc<EngramBlockKernels>,
        config: &Dsv41Config,
        manifest: &RankManifest,
        layer: usize,
        weights: Arc<LoadedWeights>,
    ) -> Result<Self> {
        Ok(Self {
            dev: dev.clone(),
            stream,
            inner: GpuEngramBlock::new(kernels, config, manifest, layer, weights)?,
        })
    }
    pub fn reset_lane(&mut self, lane: LaneKey) -> Result<()> {
        self.inner.reset_lane(&self.stream, lane)
    }
    /// The active BatchStore ticket and native collective owner must match.
    /// No per-lane tensor copy, separate projection, CPU embedding readback or
    /// implicit CPU-history/cache-prefix commit occurs in this wrapper.
    pub fn execute_checked(
        &mut self,
        ticket: &BatchTicket,
        h: &CudaSlice<bf16>,
        hash_rows: &[HashRows],
        operation: u64,
        store: &mut GpuBatchStore,
        comm: &mut GpuCollectives,
    ) -> Result<EngramOutput> {
        let result = (|| {
            let epoch = comm.expected_epoch();
            let owner = comm.completion_identity();
            let pending = self.inner.enqueue_packed_rows(
                &self.stream,
                ticket,
                h,
                hash_rows,
                operation,
                store,
                comm,
            )?;
            let out = pending.finish(comm)?;
            let packed = out
                .identity
                .packed
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("packed Engram authority missing"))?;
            ensure!(
                packed.ticket == *ticket
                    && packed.one_shared_lookup_projection_gate
                    && packed.semantic_progress.len() == ticket.lanes()
                    && out.identity.tokens == ticket.total_rows()
                    && out.identity.operation_id == operation
                    && out.identity.completion_identity == owner
                    && out.identity.collective_epoch_after_enqueue > epoch
                    && out.identity.all_checks_passed,
                "packed Engram checked receipt differs"
            );
            Ok(out)
        })();
        if result.is_err() {
            comm.abort_session();
            let _ = store.abort(ticket);
            let _ = self.dev.synchronize();
        }
        result
    }
}
impl Drop for GpuBatchEngram {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
