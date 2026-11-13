//! Native-byte streaming into a bounded device allocation. This component does
//! not select model tensors, dequantize them, establish their payload identity,
//! or qualify a model memory budget. The process supervisor still owns thermal
//! limits and an independent deadline.
use super::reader::{CheckpointReader, TensorSpan};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

const MIN_AVAILABLE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct NativeLoadRecord {
    pub tensor_name: String,
    pub dtype: String,
    pub selected_shape: Vec<u64>,
    pub first_row: u64,
    pub valid_rows: u64,
    pub native_bytes: u64,
    pub streamed_bytes_sha256: String,
    pub staging_limit_bytes: usize,
    pub original_checkpoint_payload_identity_verified: bool,
    pub conversion: &'static str,
}
// Cumulative ownership budget for native model allocations. These are declared
// experiment limits, not a vendor capacity guarantee; KV/workspaces are separate.
#[derive(Debug, Clone, Serialize)]
pub struct WeightLoadLimits {
    pub maximum_resident_bytes: u64,
    pub maximum_single_allocation_bytes: u64,
    pub maximum_staging_bytes: usize,
    pub available_floor_bytes: u64,
    pub reserved_nonweight_bytes: u64,
    /// Cooperative whole-session deadline; supervisor must bound blocked device/disk calls.
    pub maximum_load_seconds: u64,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct WeightBudgetSnapshot {
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub active_allocations: u64,
    pub allocations_started: u64,
    pub allocations_released: u64,
    pub minimum_available_bytes: Option<u64>,
    pub elapsed_seconds: f64,
}
#[derive(Clone)]
pub struct GpuWeightBudget {
    limits: WeightLoadLimits,
    state: Arc<std::sync::Mutex<WeightBudgetSnapshot>>,
    began: std::time::Instant,
    transactions: Arc<std::sync::Mutex<()>>,
}
/// A borrowed token serializes one complete load transaction, including CUDA
/// allocation/transfer/rollback, across every clone of this physical budget.
pub(crate) struct WeightLoadTransaction<'a> {
    budget: &'a GpuWeightBudget,
    _guard: std::sync::MutexGuard<'a, ()>,
}
impl WeightLoadTransaction<'_> {
    pub(crate) fn budget(&self) -> &GpuWeightBudget {
        self.budget
    }
    pub(crate) fn reserve(&self, bytes: u64) -> Result<WeightAllocationLease> {
        self.budget
            .reserve_available(bytes, self.budget.check_memory()?)
    }
}
pub(crate) struct WeightAllocationLease {
    budget: GpuWeightBudget,
    bytes: u64,
}
impl Drop for WeightAllocationLease {
    fn drop(&mut self) {
        if let Ok(mut state) = self.budget.state.lock() {
            state.resident_bytes = state.resident_bytes.saturating_sub(self.bytes);
            state.active_allocations = state.active_allocations.saturating_sub(1);
            state.allocations_released = state.allocations_released.saturating_add(1);
        }
    }
}
impl GpuWeightBudget {
    pub fn new(limits: WeightLoadLimits) -> Result<Self> {
        ensure!(
            limits.maximum_resident_bytes > 0
                && limits.maximum_single_allocation_bytes > 0
                && limits.maximum_single_allocation_bytes <= limits.maximum_resident_bytes
                && (1..=super::reader::MAX_CHUNK_BYTES).contains(&limits.maximum_staging_bytes)
                && limits.available_floor_bytes >= MIN_AVAILABLE_BYTES
                && (1..=3600).contains(&limits.maximum_load_seconds),
            "invalid declared weight-load limits"
        );
        limits
            .available_floor_bytes
            .checked_add(limits.reserved_nonweight_bytes)
            .and_then(|n| n.checked_add(limits.maximum_staging_bytes as u64))
            .context("weight floor/headroom overflow")?;
        Ok(Self {
            limits,
            state: Arc::new(std::sync::Mutex::new(WeightBudgetSnapshot::default())),
            began: std::time::Instant::now(),
            transactions: Arc::new(std::sync::Mutex::new(())),
        })
    }
    pub fn limits(&self) -> &WeightLoadLimits {
        &self.limits
    }
    pub fn snapshot(&self) -> Result<WeightBudgetSnapshot> {
        let mut snapshot = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("weight budget poisoned"))?
            .clone();
        snapshot.elapsed_seconds = self.began.elapsed().as_secs_f64();
        Ok(snapshot)
    }
    pub fn check_memory(&self) -> Result<u64> {
        ensure!(
            self.began.elapsed().as_secs_f64() <= self.limits.maximum_load_seconds as f64,
            "weight load session deadline exceeded"
        );
        let available = mem_available()?;
        self.observe(available)?;
        ensure!(
            available
                >= self
                    .limits
                    .available_floor_bytes
                    .checked_add(self.limits.reserved_nonweight_bytes)
                    .context("weight runtime headroom overflow")?,
            "weight-loader available-memory floor/nonweight headroom crossed"
        );
        Ok(available)
    }
    fn observe(&self, available: u64) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("weight budget poisoned"))?;
        state.minimum_available_bytes = Some(
            state
                .minimum_available_bytes
                .unwrap_or(available)
                .min(available),
        );
        Ok(())
    }
    pub(crate) fn begin_transaction(&self) -> Result<WeightLoadTransaction<'_>> {
        loop {
            ensure!(
                self.began.elapsed().as_secs_f64() < self.limits.maximum_load_seconds as f64,
                "weight load session deadline while waiting for transaction"
            );
            match self.transactions.try_lock() {
                Ok(guard) => {
                    return Ok(WeightLoadTransaction {
                        budget: self,
                        _guard: guard,
                    })
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    anyhow::bail!("weight load transaction poisoned")
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(2))
                }
            }
        }
    }
    fn reserve_available(&self, bytes: u64, available: u64) -> Result<WeightAllocationLease> {
        ensure!(
            bytes > 0 && bytes <= self.limits.maximum_single_allocation_bytes,
            "native weight allocation exceeds declared single-allocation limit"
        );
        let headroom = self
            .limits
            .available_floor_bytes
            .checked_add(self.limits.reserved_nonweight_bytes)
            .and_then(|n| n.checked_add(self.limits.maximum_staging_bytes as u64))
            .context("weight headroom overflow")?;
        ensure!(
            available
                .checked_sub(bytes)
                .is_some_and(|left| left >= headroom),
            "native allocation would consume floor/staging/nonweight headroom"
        );
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("weight budget poisoned"))?;
        let resident = state
            .resident_bytes
            .checked_add(bytes)
            .context("weight resident overflow")?;
        ensure!(
            resident <= self.limits.maximum_resident_bytes,
            "cumulative model-weight budget exceeded"
        );
        state.resident_bytes = resident;
        state.peak_resident_bytes = state.peak_resident_bytes.max(resident);
        state.active_allocations += 1;
        state.allocations_started += 1;
        drop(state);
        Ok(WeightAllocationLease {
            budget: self.clone(),
            bytes,
        })
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct TypedLoadAttempt {
    pub span: TensorSpan,
    pub allocation_bytes: u64,
    pub zero_padding_bytes: u64,
    pub streamed_native_bytes: u64,
    pub streamed_native_sha256: String,
    pub complete: bool,
    pub error: Option<String>,
    pub budget: WeightBudgetSnapshot,
}
impl TypedLoadAttempt {
    pub fn planned(span: &TensorSpan, padding: u64) -> Result<Self> {
        Ok(Self {
            span: span.clone(),
            allocation_bytes: span
                .bytes
                .checked_add(padding)
                .context("allocation bytes overflow")?,
            zero_padding_bytes: padding,
            streamed_native_bytes: 0,
            streamed_native_sha256: format!("{:x}", Sha256::digest([])),
            complete: false,
            error: None,
            budget: WeightBudgetSnapshot::default(),
        })
    }
}
enum TypedStorage {
    Bytes(CudaSlice<u8>),
    Bf16(CudaSlice<bf16>),
    F32(CudaSlice<f32>),
}
/// Typed owners remain typed for their entire allocation lifetime. The temporary
/// byte view used for HtoD never takes ownership and cannot escape an upload.
pub struct TypedNativeGpuTensor {
    storage: TypedStorage,
    record: TypedLoadAttempt,
    _lease: WeightAllocationLease,
}
impl TypedNativeGpuTensor {
    pub fn record(&self) -> &TypedLoadAttempt {
        &self.record
    }
    pub fn as_u8(&self) -> Result<&CudaSlice<u8>> {
        match &self.storage {
            TypedStorage::Bytes(v) => Ok(v),
            _ => anyhow::bail!("tensor is not an owned nativebyte dtype"),
        }
    }
    pub fn as_bf16(&self) -> Result<&CudaSlice<bf16>> {
        match &self.storage {
            TypedStorage::Bf16(v) => Ok(v),
            _ => anyhow::bail!("tensor is not owned BF16"),
        }
    }
    pub fn as_f32(&self) -> Result<&CudaSlice<f32>> {
        match &self.storage {
            TypedStorage::F32(v) => Ok(v),
            _ => anyhow::bail!("tensor is not owned F32"),
        }
    }
    pub fn load(
        dev: &Arc<CudaDevice>,
        reader: &mut CheckpointReader,
        span: &TensorSpan,
        padding: u64,
        budget: &GpuWeightBudget,
        attempt: &mut TypedLoadAttempt,
    ) -> Result<Self> {
        *attempt = TypedLoadAttempt::planned(span, padding)?;
        let transaction = match budget.begin_transaction() {
            Ok(transaction) => transaction,
            Err(error) => {
                attempt.error = Some(format!("{error:#}"));
                attempt.budget = budget.snapshot()?;
                return Err(error);
            }
        };
        Self::load_in_transaction(dev, reader, span, padding, &transaction, attempt)
    }
    pub(crate) fn load_in_transaction(
        dev: &Arc<CudaDevice>,
        reader: &mut CheckpointReader,
        span: &TensorSpan,
        padding: u64,
        transaction: &WeightLoadTransaction<'_>,
        attempt: &mut TypedLoadAttempt,
    ) -> Result<Self> {
        let budget = transaction.budget();
        ensure!(
            cfg!(target_endian = "little"),
            "native typed checkpoint requires little endian"
        );
        ensure!(
            reader.chunk_bytes() <= budget.limits.maximum_staging_bytes,
            "reader staging exceeds declared model-load limit"
        );
        *attempt = TypedLoadAttempt::planned(span, padding)?;
        let result = (|| {
            let allocation = attempt.allocation_bytes;
            let lease = transaction.reserve(allocation)?;
            let storage = match span.dtype.as_str() {
                "BF16" => TypedStorage::Bf16(stream_typed::<bf16>(
                    dev, reader, span, allocation, budget, attempt,
                )?),
                "F32" => TypedStorage::F32(stream_typed::<f32>(
                    dev, reader, span, allocation, budget, attempt,
                )?),
                "I8" | "F8_E4M3" | "F8_E8M0" => TypedStorage::Bytes(stream_typed::<u8>(
                    dev, reader, span, allocation, budget, attempt,
                )?),
                _ => anyhow::bail!("unsupported exact1017 native dtype{}", span.dtype),
            };
            attempt.complete = true;
            attempt.budget = budget.snapshot()?;
            Ok(Self {
                storage,
                record: attempt.clone(),
                _lease: lease,
            })
        })();
        if let Err(error) = &result {
            attempt.error = Some(format!("{error:#}"));
            attempt.budget = budget.snapshot()?;
        }
        result
    }
}
fn stream_typed<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &Arc<CudaDevice>,
    reader: &mut CheckpointReader,
    span: &TensorSpan,
    allocation: u64,
    budget: &GpuWeightBudget,
    attempt: &mut TypedLoadAttempt,
) -> Result<CudaSlice<T>> {
    let item = std::mem::size_of::<T>() as u64;
    ensure!(
        allocation % item == 0 && span.bytes % item == 0,
        "typed native byte alignment"
    );
    let mut values = dev.alloc_zeros::<T>(usize::try_from(allocation / item)?)?;
    dev.synchronize()?;
    budget.check_memory()?;
    let mut digest = Sha256::new();
    let result = reader.stream(span, |offset, chunk| {
        budget.check_memory()?;
        let start = usize::try_from(offset)?;
        let end = start
            .checked_add(chunk.len())
            .context("typed upload extent overflow")?;
        // Owner stays CudaSlice<T>. Only this short-lived mutable byte view is
        // reinterpreted, after header dtype/extent validation, for raw HtoD.
        let mut bytes = unsafe { values.transmute_mut::<u8>(usize::try_from(allocation)?) }
            .context("typed upload byte view")?;
        let mut target = bytes
            .try_slice_mut(start..end)
            .context("typed device upload range")?;
        dev.htod_sync_copy_into(chunk, &mut target)?;
        digest.update(chunk);
        attempt.streamed_native_bytes = offset + chunk.len() as u64;
        budget.check_memory()?;
        Ok(())
    });
    attempt.streamed_native_sha256 = format!("{:x}", digest.finalize());
    result?;
    ensure!(
        attempt.streamed_native_bytes == span.bytes,
        "typed native stream incomplete"
    );
    dev.synchronize()?;
    budget.check_memory()?;
    Ok(values)
}
#[cfg(test)]
mod typed_budget_tests {
    use super::*;
    fn limits() -> WeightLoadLimits {
        WeightLoadLimits {
            maximum_resident_bytes: 100,
            maximum_single_allocation_bytes: 80,
            maximum_staging_bytes: 4,
            available_floor_bytes: MIN_AVAILABLE_BYTES,
            reserved_nonweight_bytes: 8,
            maximum_load_seconds: 60,
        }
    }
    #[test]
    fn cumulative_limits_and_rollback_follow_owner_lifetimes_without_gpu() {
        let budget = GpuWeightBudget::new(limits()).unwrap();
        let available = MIN_AVAILABLE_BYTES + 1000;
        let first = budget.reserve_available(60, available).unwrap();
        assert!(budget.reserve_available(41, available).is_err());
        {
            let second = budget.reserve_available(40, available).unwrap();
            assert_eq!(budget.snapshot().unwrap().resident_bytes, 100);
            drop(second);
        }
        assert_eq!(budget.snapshot().unwrap().resident_bytes, 60);
        drop(first);
        let state = budget.snapshot().unwrap();
        assert_eq!(state.resident_bytes, 0);
        assert_eq!(state.peak_resident_bytes, 100);
        assert_eq!(state.active_allocations, 0);
        assert_eq!(state.allocations_started, state.allocations_released);
        assert!(budget.reserve_available(81, available).is_err());
        assert!(budget
            .reserve_available(1, MIN_AVAILABLE_BYTES + 12)
            .is_err());
        assert!(budget.reserve_available(u64::MAX, available).is_err());
    }
    #[test]
    fn floor_staging_and_nonweight_headroom_are_predeclared() {
        let mut bad = limits();
        bad.available_floor_bytes = MIN_AVAILABLE_BYTES - 1;
        assert!(GpuWeightBudget::new(bad).is_err());
        let mut bad = limits();
        bad.maximum_staging_bytes = 0;
        assert!(GpuWeightBudget::new(bad).is_err());
        let mut bad = limits();
        bad.reserved_nonweight_bytes = u64::MAX;
        assert!(GpuWeightBudget::new(bad).is_err());
        let budget = GpuWeightBudget::new(limits()).unwrap();
        assert!(budget
            .reserve_available(1, MIN_AVAILABLE_BYTES + 4 + 8)
            .is_err());
        assert_eq!(budget.snapshot().unwrap().resident_bytes, 0);
    }
    #[test]
    fn cloned_budget_serializes_whole_transactions_and_releases_failed_owners() {
        use std::sync::mpsc;
        use std::time::Duration;
        let budget = GpuWeightBudget::new(limits()).unwrap();
        let held = budget.begin_transaction().unwrap();
        let clone = budget.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let _transaction = clone.begin_transaction().unwrap();
            acquired_tx.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(acquired_rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(held);
        acquired_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        let result: Result<()> = (|| {
            let transaction = budget.begin_transaction()?;
            let _owner = transaction
                .budget
                .reserve_available(60, MIN_AVAILABLE_BYTES + 1000)?;
            anyhow::bail!("forced CPU-only transaction failure")
        })();
        assert!(result.is_err());
        let _next = budget.begin_transaction().unwrap();
        let state = budget.snapshot().unwrap();
        assert_eq!(state.resident_bytes, 0);
        assert_eq!(state.active_allocations, 0);
        assert_eq!(state.allocations_started, state.allocations_released);
    }
    #[test]
    fn transaction_wait_is_included_in_the_declared_session_deadline() {
        use std::time::Duration;
        let mut declared = limits();
        declared.maximum_load_seconds = 1;
        let budget = GpuWeightBudget::new(declared).unwrap();
        let held = budget.begin_transaction().unwrap();
        let clone = budget.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let error = clone.begin_transaction().err().map(|e| e.to_string());
            tx.send(error).unwrap();
        });
        let error = rx.recv_timeout(Duration::from_secs(3)).unwrap().unwrap();
        assert!(error.contains("deadline"));
        drop(held);
        worker.join().unwrap();
        assert_eq!(budget.snapshot().unwrap().resident_bytes, 0);
    }
}

/// The streamed-byte digest identifies this selected range, not the full model.
pub struct NativeGpuTensor {
    bytes: CudaSlice<u8>,
    record: NativeLoadRecord,
}

fn mem_available() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .context("MemAvailable is missing")?;
    let mut fields = value.split_whitespace();
    let amount = fields
        .next()
        .context("MemAvailable amount")?
        .parse::<u64>()?;
    ensure!(
        fields.next() == Some("kB") && fields.next().is_none(),
        "unexpected MemAvailable units"
    );
    amount.checked_mul(1024).context("MemAvailable overflow")
}

impl NativeGpuTensor {
    /// The caller must predeclare a maximum single allocation and a memory
    /// floor at least8GiB. Available memory is checked before allocating and
    /// after every synchronous transfer. There is no hidden full-shard buffer.
    pub fn load(
        dev: &Arc<CudaDevice>,
        reader: &mut CheckpointReader,
        span: &TensorSpan,
        max_allocation_bytes: u64,
        available_floor_bytes: u64,
    ) -> Result<Self> {
        ensure!(
            cfg!(target_endian = "little"),
            "native1017 byte loader requires little endian"
        );
        ensure!(
            available_floor_bytes >= MIN_AVAILABLE_BYTES,
            "available-memory floor cannot be below8GiB"
        );
        ensure!(
            span.bytes > 0 && span.bytes <= max_allocation_bytes,
            "selected tensor exceeds declared allocation bound"
        );
        let available = mem_available()?;
        ensure!(
            available
                .checked_sub(span.bytes)
                .is_some_and(|left| left >= available_floor_bytes),
            "native allocation would consume the available-memory floor"
        );
        let mut bytes = dev.alloc_zeros::<u8>(usize::try_from(span.bytes)?)?;
        dev.synchronize()?;
        ensure!(
            mem_available()? >= available_floor_bytes,
            "memory floor after device allocation"
        );
        let mut digest = Sha256::new();
        reader.stream(span, |offset, chunk| {
            ensure!(
                mem_available()? >= available_floor_bytes,
                "memory floor before native transfer"
            );
            let start = usize::try_from(offset)?;
            let end = start
                .checked_add(chunk.len())
                .context("device subrange overflow")?;
            let mut target = bytes
                .try_slice_mut(start..end)
                .context("native device subrange out of bounds")?;
            dev.htod_sync_copy_into(chunk, &mut target)?;
            digest.update(chunk);
            ensure!(
                mem_available()? >= available_floor_bytes,
                "memory floor after native transfer"
            );
            Ok(())
        })?;
        let record = NativeLoadRecord {
            tensor_name: span.tensor_name.clone(),
            dtype: span.dtype.clone(),
            selected_shape: span.shape.clone(),
            first_row: span.first_row,
            valid_rows: span.row_count,
            native_bytes: span.bytes,
            streamed_bytes_sha256: format!("{:x}", digest.finalize()),
            staging_limit_bytes: reader.statistics().maximum_staging_bytes,
            original_checkpoint_payload_identity_verified: false,
            conversion: "none; exact stored bytes",
        };
        Ok(Self { bytes, record })
    }

    pub fn record(&self) -> &NativeLoadRecord {
        &self.record
    }
    pub fn native_bytes(&self) -> &CudaSlice<u8> {
        &self.bytes
    }
    pub fn bf16_view(&self) -> Result<CudaView<'_, bf16>> {
        ensure!(
            self.record.dtype == "BF16" && self.bytes.len() % 2 == 0,
            "tensor is not native BF16"
        );
        // CUDA allocation is suitably aligned, dtype was header-validated, and
        // the view borrows the unchanged byte owner for its whole lifetime.
        unsafe { self.bytes.transmute::<bf16>(self.bytes.len() / 2) }.context("BF16 view extent")
    }
    pub fn f32_view(&self) -> Result<CudaView<'_, f32>> {
        ensure!(
            self.record.dtype == "F32" && self.bytes.len() % 4 == 0,
            "tensor is not native F32"
        );
        unsafe { self.bytes.transmute::<f32>(self.bytes.len() / 4) }.context("F32 view extent")
    }
}
