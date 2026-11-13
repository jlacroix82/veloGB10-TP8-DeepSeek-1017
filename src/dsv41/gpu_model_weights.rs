//! Explicit rank-local whole text-trunk load. Requires a caller-supplied shared
//! allocation budget, immutable checkpoint reader and independently supervised
//! lifetime. This is not automatic model startup or a measured memory-fit claim.
use super::{
    gpu_reader::{GpuWeightBudget, WeightBudgetSnapshot},
    gpu_weights::{self, LoadEvidence, LoadScope, LoadedWeights, ScopeLoadPlan},
    reader::CheckpointReader,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct RankLoadPlan {
    pub manifest_sha256: String,
    pub rank: usize,
    pub scopes: Vec<ScopeLoadPlan>,
    pub native_bytes_to_stream: u64,
    pub final_resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub largest_single_allocation_bytes: u64,
    pub tensors: usize,
    pub engram_tables_resident: bool,
    pub full_model_execution: bool,
}
pub fn plan_rank_load(manifest: &RankManifest) -> Result<RankLoadPlan> {
    manifest.verify_identity()?;
    let mut scopes = vec![gpu_weights::plan_scope_load(manifest, LoadScope::Global)?];
    for layer in 0..40 {
        scopes.push(gpu_weights::plan_scope_load(
            manifest,
            LoadScope::Layer(layer),
        )?);
    }
    let (mut native, mut resident, mut peak, mut largest, mut tensors) =
        (0u64, 0u64, 0u64, 0u64, 0usize);
    for scope in &scopes {
        native = native
            .checked_add(scope.native_bytes_to_stream)
            .context("rank native load overflow")?;
        peak = peak.max(
            resident
                .checked_add(scope.additional_peak_resident_bytes)
                .context("rank peak overflow")?,
        );
        resident = resident
            .checked_add(scope.final_resident_bytes)
            .context("rank resident overflow")?;
        largest = largest.max(scope.largest_single_allocation_bytes);
        tensors = tensors
            .checked_add(scope.tensors)
            .context("rank tensor count overflow")?;
    }
    ensure!(
        scopes.len() == 41
            && scopes
                .iter()
                .filter(|p| p.scope_contains_engram_table)
                .count()
                == 2,
        "rank text-trunk/Engram scope coverage"
    );
    Ok(RankLoadPlan {
        manifest_sha256: manifest.manifest_sha256.clone(),
        rank: manifest.rank,
        scopes,
        native_bytes_to_stream: native,
        final_resident_bytes: resident,
        peak_resident_bytes: peak,
        largest_single_allocation_bytes: largest,
        tensors,
        engram_tables_resident: true,
        full_model_execution: false,
    })
}
pub struct RankModelWeights {
    dev: Arc<CudaDevice>,
    pub manifest: Arc<RankManifest>,
    pub global: Arc<LoadedWeights>,
    pub layers: Vec<Arc<LoadedWeights>>,
    budget: GpuWeightBudget,
}
impl Drop for RankModelWeights {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
impl RankModelWeights {
    /// All41 scope and cumulative memory limits are checked before payload IO.
    /// The callback receives each success/failure's range evidence immediately.
    /// GPU allocation failures roll back this load; previously unrelated owners
    /// remain intact under the same budget.
    pub fn load<F>(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        reader: &mut CheckpointReader,
        manifest: Arc<RankManifest>,
        budget: &GpuWeightBudget,
        mut evidence: F,
    ) -> Result<Self>
    where
        F: FnMut(&LoadEvidence) -> Result<()>,
    {
        let plan = plan_rank_load(&manifest)?;
        let before = budget.snapshot()?;
        let limits = budget.limits();
        ensure!(before.resident_bytes.checked_add(plan.peak_resident_bytes).context("combined rank peak overflow")?<=limits.maximum_resident_bytes
            &&plan.largest_single_allocation_bytes<=limits.maximum_single_allocation_bytes,
            "whole-rank scope exceeds declared resident/individual allocation budget before payload IO");
        let mut loaded = Vec::with_capacity(41);
        for scope in &plan.scopes {
            let mut record = None;
            let outcome = gpu_weights::load_scope(
                dev,
                stream,
                reader,
                &manifest,
                scope.scope.clone(),
                budget,
                &mut record,
            );
            let record = record.context("scope loader omitted evidence")?;
            // Evidence can be persisted even if the scope failed. A callback
            // error still drops all current owners through their GPU fences.
            let persistence = evidence(&record);
            let scope_weights = outcome?;
            persistence?;
            ensure!(
                record.status == "COMPLETE"
                    && scope_weights.manifest_sha256() == plan.manifest_sha256,
                "whole-rank scope identity/result mismatch"
            );
            loaded.push(Arc::new(scope_weights));
        }
        dev.synchronize()?;
        let after = budget.snapshot()?;
        ensure!(
            after.resident_bytes == before.resident_bytes + plan.final_resident_bytes
                && loaded.len() == 41,
            "whole-rank retained budget/count differs from preflight"
        );
        let global = loaded.remove(0);
        Ok(Self {
            dev: dev.clone(),
            manifest,
            global,
            layers: loaded,
            budget: budget.clone(),
        })
    }
    pub fn budget(&self) -> Result<WeightBudgetSnapshot> {
        self.budget.snapshot()
    }
    /// Call only after all model execution borrowers have dropped. External Arc
    /// clones can retain tensors; returned accounting reports that honestly.
    pub fn release(mut self) -> Result<WeightBudgetSnapshot> {
        self.dev.synchronize()?;
        self.layers.clear();
        // Swap global into a temporary container so its last strong owner drops
        // before the returned shared-budget snapshot; never silently clone it.
        let global = Arc::get_mut(&mut self.global)
            .context("global weights still borrowed by an external owner")?;
        let _ = global;
        let budget = self.budget.clone();
        let dev = self.dev.clone();
        drop(self);
        dev.synchronize()?;
        budget.snapshot()
    }
}
