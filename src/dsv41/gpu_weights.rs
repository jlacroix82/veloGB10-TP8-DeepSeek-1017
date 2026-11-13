//! Bounded rank-local native1017 loader for one declared scope at a time.
//! No automatic whole-model load, host mutation, checkpoint conversion or serving route.
//! The supervisor must supply an independent hard deadline/thermal guard.
pub use super::gpu_reader::{GpuWeightBudget, WeightLoadLimits};
use super::{
    gpu_dense::{DenseOutput, DequantCast, GpuDense},
    gpu_reader::{
        TypedLoadAttempt, TypedNativeGpuTensor, WeightAllocationLease, WeightBudgetSnapshot,
    },
    reader::{CheckpointReader, ReadStatistics},
    weights::{
        header_geometry_sha256, Conversion, Group, RankManifest, TensorLayout,
        WoaReconstructionPolicy,
    },
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream};
use half::bf16;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum LoadScope {
    Global,
    Layer(usize),
}
#[derive(Debug, Clone, Serialize)]
pub struct ScopeLoadPlan {
    pub scope: LoadScope,
    pub tensors: usize,
    pub native_bytes_to_stream: u64,
    pub final_resident_bytes: u64,
    pub additional_peak_resident_bytes: u64,
    pub largest_single_allocation_bytes: u64,
    pub scope_contains_engram_table: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct WoaConversionRecord {
    pub tensor_name: String,
    pub scale_tensor_name: String,
    pub source_native_sha256: String,
    pub scale_native_sha256: String,
    pub policy: WoaReconstructionPolicy,
    pub BF16_elements: usize,
    pub output_allocation_bytes: u64,
    pub status: u32,
    pub dense_source_id_hex: String,
    pub dense_ptx_sha256: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct LoadEvidence {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub rank: usize,
    pub scope: LoadScope,
    pub started_unix_ms: u128,
    pub elapsed_seconds: f64,
    pub status: String,
    pub limits: WeightLoadLimits,
    pub plan: Option<ScopeLoadPlan>,
    pub budget_before: WeightBudgetSnapshot,
    pub budget_after: WeightBudgetSnapshot,
    pub reader_before: ReadStatistics,
    pub reader_after: ReadStatistics,
    pub attempts: Vec<TypedLoadAttempt>,
    pub woa_conversions: Vec<WoaConversionRecord>,
    pub failure: Option<String>,
    pub rollback_synchronized: bool,
    pub rollback_resident_matches_initial: Option<bool>,
    pub payload_identity_verified: bool,
    pub serve_supported: bool,
    pub load_serialization: &'static str,
}
enum LoadedTensor {
    Native(TypedNativeGpuTensor),
    WoaBf16 {
        values: DenseOutput<bf16>,
        _lease: WeightAllocationLease,
    },
}
pub struct LoadedWeights {
    dev: Arc<CudaDevice>,
    manifest_sha256: String,
    scope: LoadScope,
    tensors: BTreeMap<String, LoadedTensor>,
    budget: GpuWeightBudget,
}
impl Drop for LoadedWeights {
    fn drop(&mut self) {
        // Runtime borrowers must finish before the owned buffers can be freed.
        let _ = self.dev.synchronize();
    }
}
impl LoadedWeights {
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn scope(&self) -> &LoadScope {
        &self.scope
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn native_u8(&self, name: &str) -> Result<&CudaSlice<u8>> {
        match self
            .tensors
            .get(name)
            .with_context(|| format!("unloaded tensor{name}"))?
        {
            LoadedTensor::Native(v) => v.as_u8(),
            _ => anyhow::bail!("tensor{name} is reconstructedBF16"),
        }
    }
    pub fn bf16(&self, name: &str) -> Result<&CudaSlice<bf16>> {
        match self
            .tensors
            .get(name)
            .with_context(|| format!("unloaded tensor{name}"))?
        {
            LoadedTensor::Native(v) => v.as_bf16(),
            // Construction checks this output status and synchronizes before publication.
            LoadedTensor::WoaBf16 { values, .. } => Ok(unsafe { values.device_values_unchecked() }),
        }
    }
    pub fn f32(&self, name: &str) -> Result<&CudaSlice<f32>> {
        match self
            .tensors
            .get(name)
            .with_context(|| format!("unloaded tensor{name}"))?
        {
            LoadedTensor::Native(v) => v.as_f32(),
            _ => anyhow::bail!("tensor{name} is reconstructedBF16"),
        }
    }
    /// Explicit release waits for pending work, drops this scope and waits for
    /// deallocations. The cumulative budget follows owners that remain alive.
    pub fn release(mut self) -> Result<WeightBudgetSnapshot> {
        self.dev.synchronize()?;
        drop(std::mem::take(&mut self.tensors));
        self.dev.synchronize()?;
        self.budget.snapshot()
    }
}
fn layouts<'a>(manifest: &'a RankManifest, scope: &LoadScope) -> Result<Vec<&'a TensorLayout>> {
    let entries = match scope {
        LoadScope::Global => manifest.global(),
        LoadScope::Layer(layer) => manifest.layer(*layer)?,
    };
    ensure!(!entries.is_empty(), "empty requested model-load scope");
    Ok(entries)
}
fn calculate_plan(scope: LoadScope, entries: &[&TensorLayout]) -> Result<ScopeLoadPlan> {
    let mut current = 0_u64;
    let mut peak = 0_u64;
    let mut largest = 0_u64;
    let mut native = 0_u64;
    let mut contains_engram = false;
    for entry in entries {
        let allocation = entry
            .native_bytes
            .checked_add(entry.allocation_padding_bytes)
            .context("native allocation overflow")?;
        native = native
            .checked_add(entry.native_bytes)
            .context("scope native sum overflow")?;
        largest = largest.max(allocation);
        let during_native = current
            .checked_add(allocation)
            .context("scope native peak overflow")?;
        peak = peak.max(during_native);
        if matches!(entry.conversion, Conversion::WoaFp8ToBf16 { .. }) {
            // BF16 output plus DenseOutput's mandatory sticky status. The native
            // FP8 weight remains alive until conversion completion is checked.
            let target = entry
                .resident_bytes
                .checked_add(4)
                .context("conversion allocation overflow")?;
            largest = largest.max(target);
            peak = peak.max(
                during_native
                    .checked_add(target)
                    .context("WO-A overlap peak overflow")?,
            );
            current = current
                .checked_add(target)
                .context("scope resident overflow")?;
        } else {
            current = during_native;
        }
        contains_engram |= entry.name.contains(".engram.embed.");
    }
    Ok(ScopeLoadPlan {
        scope,
        tensors: entries.len(),
        native_bytes_to_stream: native,
        final_resident_bytes: current,
        additional_peak_resident_bytes: peak,
        largest_single_allocation_bytes: largest,
        scope_contains_engram_table: contains_engram,
    })
}
pub fn plan_scope_load(manifest: &RankManifest, scope: LoadScope) -> Result<ScopeLoadPlan> {
    manifest.verify_identity()?;
    calculate_plan(scope.clone(), &layouts(manifest, &scope)?)
}
/// Header validation and budget preflight run before any GPU allocation or payload
/// read. Caller owns saving evidence even on Err. Failed scopes release every new
/// native/converted owner; previous successfully loaded scopes remain untouched.
pub fn load_scope(
    dev: &Arc<CudaDevice>,
    stream: &CudaStream,
    reader: &mut CheckpointReader,
    manifest: &RankManifest,
    scope: LoadScope,
    budget: &GpuWeightBudget,
    evidence: &mut Option<LoadEvidence>,
) -> Result<LoadedWeights> {
    let began = Instant::now();
    let mut before = budget.snapshot()?;
    let mut record = LoadEvidence {
        schema_version: 1,
        manifest_sha256: manifest.manifest_sha256.clone(),
        rank: manifest.rank,
        scope: scope.clone(),
        started_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        elapsed_seconds: 0.,
        status: "PREFLIGHT".into(),
        limits: budget.limits().clone(),
        plan: None,
        budget_before: before.clone(),
        budget_after: before.clone(),
        reader_before: reader.statistics().clone(),
        reader_after: reader.statistics().clone(),
        attempts: Vec::new(),
        woa_conversions: Vec::new(),
        failure: None,
        rollback_synchronized: false,
        rollback_resident_matches_initial: None,
        payload_identity_verified: false,
        serve_supported: false,
        load_serialization:
            "one whole transaction per shared physical budget; waiting consumes session deadline",
    };
    let mut tensors = BTreeMap::new();
    // Retain the shared physical-load lock through success or complete rollback.
    // Inner tensor loads borrow this token rather than reacquiring it.
    let mut transaction = None;
    let result = (|| {
        transaction = Some(budget.begin_transaction()?);
        let transaction = transaction.as_ref().unwrap();
        before = budget.snapshot()?;
        record.budget_before = before.clone();
        manifest.verify_identity()?;
        ensure!(
            reader.inventory().index_sha256 == manifest.source_index_sha256
                && header_geometry_sha256(reader.inventory())?
                    == manifest.source_header_geometry_sha256,
            "reader is not the manifest's exact checkpoint header identity"
        );
        dev.bind_to_thread()?;
        let mut flags = 0_u32;
        ensure!(
            !stream.stream.is_null()
                && unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) }
                    == sys::CUresult::CUDA_SUCCESS
                && flags == 0,
            "native model load needs caller-owned blocking stream"
        );
        let entries = layouts(manifest, &scope)?;
        let plan = calculate_plan(scope.clone(), &entries)?;
        ensure!(
            reader.chunk_bytes() <= budget.limits().maximum_staging_bytes,
            "reader staging exceeds declared scope limit"
        );
        ensure!(
            plan.largest_single_allocation_bytes <= budget.limits().maximum_single_allocation_bytes
                && before
                    .resident_bytes
                    .checked_add(plan.additional_peak_resident_bytes)
                    .is_some_and(|n| n <= budget.limits().maximum_resident_bytes),
            "requested scope exceeds cumulative/single native-load budget before payload reads"
        );
        // Construct/validate every source selection before beginning allocation.
        for entry in &entries {
            entry.span(reader)?;
        }
        record.plan = Some(plan);
        record.status = "LOADING".into();
        for entry in entries {
            budget.check_memory()?;
            let span = entry.span(reader)?;
            record.attempts.push(TypedLoadAttempt::planned(
                &span,
                entry.allocation_padding_bytes,
            )?);
            let attempt = record.attempts.last_mut().unwrap();
            let native = TypedNativeGpuTensor::load_in_transaction(
                dev,
                reader,
                &span,
                entry.allocation_padding_bytes,
                transaction,
                attempt,
            )
            .with_context(|| format!("load native tensor{}", entry.name))?;
            let loaded = match &entry.conversion {
                Conversion::Native => LoadedTensor::Native(native),
                Conversion::WoaFp8ToBf16 {
                    scale_tensor,
                    policy,
                } => {
                    let scale = match tensors
                        .get(scale_tensor)
                        .context("WO-A scales must be loaded before nativeweight")?
                    {
                        LoadedTensor::Native(v) => v,
                        _ => anyhow::bail!("WO-A scales unexpectedly converted"),
                    };
                    ensure!(
                        entry.selected_shape.len() == 2,
                        "WO-A selected matrix expected"
                    );
                    let output_bytes = entry
                        .resident_bytes
                        .checked_add(4)
                        .context("WO-A runtime allocation overflow")?;
                    let lease = transaction.reserve(output_bytes)?;
                    let dense = GpuDense::load(dev, stream)?;
                    budget.check_memory()?;
                    let values = dense.fp8_weights_to_bf16(
                        stream,
                        native.as_u8()?,
                        scale.as_u8()?,
                        usize::try_from(entry.selected_shape[0])?,
                        usize::try_from(entry.selected_shape[1])?,
                        match policy {
                            WoaReconstructionPolicy::RequireExact => DequantCast::RequireExact,
                            WoaReconstructionPolicy::OfficialBf16Round => {
                                DequantCast::OfficialBf16Round
                            }
                        },
                    )?;
                    let status = values.status_code()?;
                    record.woa_conversions.push(WoaConversionRecord {
                        tensor_name: entry.name.clone(),
                        scale_tensor_name: scale_tensor.clone(),
                        source_native_sha256: native.record().streamed_native_sha256.clone(),
                        scale_native_sha256: scale.record().streamed_native_sha256.clone(),
                        policy: *policy,
                        BF16_elements: usize::try_from(entry.resident_bytes / 2)?,
                        output_allocation_bytes: output_bytes,
                        status,
                        dense_source_id_hex: format!("{:016x}", dense.source_id()),
                        dense_ptx_sha256: dense.ptx_sha256(),
                    });
                    values.check()?;
                    budget.check_memory()?;
                    drop(native);
                    dev.synchronize()?;
                    LoadedTensor::WoaBf16 {
                        values,
                        _lease: lease,
                    }
                }
            };
            ensure!(
                tensors.insert(entry.name.clone(), loaded).is_none(),
                "duplicate native load target"
            );
        }
        budget.check_memory()?;
        dev.synchronize()?;
        Ok(())
    })();
    record.elapsed_seconds = began.elapsed().as_secs_f64();
    record.reader_after = reader.statistics().clone();
    match result {
        Ok(()) => {
            record.status = "COMPLETE".into();
            record.budget_after = budget.snapshot()?;
            *evidence = Some(record);
            Ok(LoadedWeights {
                dev: dev.clone(),
                manifest_sha256: manifest.manifest_sha256.clone(),
                scope,
                tensors,
                budget: budget.clone(),
            })
        }
        Err(error) => {
            record.status = "FAILED_RETAINED".into();
            record.failure = Some(format!("{error:#}"));
            drop(tensors);
            match dev.synchronize() {
                Ok(()) => record.rollback_synchronized = true,
                Err(e) => record
                    .failure
                    .as_mut()
                    .unwrap()
                    .push_str(&format!("; rollbackCUDAwaitfailed:{e}")),
            }
            record.budget_after = budget.snapshot()?;
            record.rollback_resident_matches_initial = Some(
                record.budget_after.resident_bytes == before.resident_bytes
                    && record.budget_after.active_allocations == before.active_allocations,
            );
            *evidence = Some(record);
            Err(error)
        }
    }
}
pub fn load_layer(
    dev: &Arc<CudaDevice>,
    stream: &CudaStream,
    reader: &mut CheckpointReader,
    manifest: &RankManifest,
    layer: usize,
    budget: &GpuWeightBudget,
    evidence: &mut Option<LoadEvidence>,
) -> Result<LoadedWeights> {
    load_scope(
        dev,
        stream,
        reader,
        manifest,
        LoadScope::Layer(layer),
        budget,
        evidence,
    )
}
#[cfg(test)]
mod tests {
    use super::super::weights::Selection;
    use super::*;
    fn entry(name: &str, native: u64, resident: u64, conversion: Conversion) -> TensorLayout {
        TensorLayout {
            name: name.into(),
            dtype: "F8_E4M3".into(),
            source_shape: vec![32, 32],
            selected_shape: vec![32, 32],
            selection: Selection::Full,
            group: Group::Layer(0),
            runtime_use: "CPUtest".into(),
            native_bytes: native,
            allocation_padding_bytes: 0,
            resident_bytes: resident,
            coupled_scale: None,
            conversion,
        }
    }
    #[test]
    fn preflight_counts_native_and_converted_overlap_and_status_without_gpu() {
        let a = entry("a", 10, 10, Conversion::Native);
        let w = entry(
            "w",
            20,
            40,
            Conversion::WoaFp8ToBf16 {
                scale_tensor: "s".into(),
                policy: WoaReconstructionPolicy::RequireExact,
            },
        );
        let z = entry("z", 30, 30, Conversion::Native);
        let p = calculate_plan(LoadScope::Layer(0), &[&a, &w, &z]).unwrap();
        assert_eq!(p.native_bytes_to_stream, 60);
        assert_eq!(p.final_resident_bytes, 84);
        assert_eq!(p.additional_peak_resident_bytes, 84);
        assert_eq!(p.largest_single_allocation_bytes, 44);
        let p = calculate_plan(LoadScope::Layer(0), &[&a, &z, &w]).unwrap();
        assert_eq!(p.final_resident_bytes, 84);
        assert_eq!(p.additional_peak_resident_bytes, 104);
        let mut huge = w.clone();
        huge.resident_bytes = u64::MAX;
        assert!(calculate_plan(LoadScope::Layer(0), &[&huge]).is_err());
    }
}
