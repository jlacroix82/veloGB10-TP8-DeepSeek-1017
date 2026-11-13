//! Native1017 short-context GPU KV store. Hardcontext512; no score/index-network
//! approximation is used: every compressed position fits the exact source topk.
//! Safe prepare waits for input copies; unsafe prepare_async requires producer
//! owners to remain alive until snapshot completion. Snapshots never borrow store.
use super::{
    gpu_module::Module,
    kv_store::{self, KvCursor, KvPlan, DIM, LAYERS, MAX_CONTEXT, MAX_LANES, WINDOW},
    scheduler::LaneKey,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};
use uuid::Uuid;
pub const PTX: &str = include_str!("../ptx/dsv41_kv_store.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_kv_store.cu");
const NAMES: &[&str] = &[
    "dsv41_kv_store_reset",
    "dsv41_kv_store_window",
    "dsv41_kv_store_compressed",
    "dsv41_kv_store_snapshot",
    "dsv41_kv_store_indices",
];
struct Registry {
    lanes: BTreeMap<usize, (LaneKey, bool)>,
}
#[derive(Clone)]
struct Token {
    dev: Arc<CudaDevice>,
    registry: Arc<Mutex<Registry>>,
    key: LaneKey,
    status: Arc<CudaSlice<u32>>,
}
impl Token {
    fn valid(&self) -> Result<()> {
        let r = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("KV registry poisoned"))?;
        ensure!(
            r.lanes.get(&self.key.slot) == Some(&(self.key, false)),
            "stale/failed KV snapshot generation"
        );
        Ok(())
    }
    fn poison(&self) {
        if let Ok(mut r) = self.registry.lock() {
            if let Some((key, failed)) = r.lanes.get_mut(&self.key.slot) {
                if *key == self.key {
                    *failed = true;
                }
            }
        }
    }
    fn check(&self) -> Result<()> {
        self.valid()?;
        self.dev.bind_to_thread()?;
        self.dev.synchronize()?;
        let status = self.dev.dtoh_sync_copy(self.status.as_ref())?[0];
        self.valid()?;
        if status != 0 {
            self.poison();
            anyhow::bail!("KV store nonfinite input/cache status{status}");
        }
        Ok(())
    }
}
pub struct GpuKvStore {
    dev: Arc<CudaDevice>,
    module: Module,
    stream_id: usize,
    source_id: u64,
    max_lanes: usize,
    cursor: KvCursor,
    windows: CudaSlice<bf16>,
    compressed: CudaSlice<bf16>,
    statuses: Vec<Arc<CudaSlice<u32>>>,
    registry: Arc<Mutex<Registry>>,
    next_snapshot: u64,
}
impl Drop for GpuKvStore {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
fn ptr<T>(v: &CudaSlice<T>) -> u64 {
    *v.device_ptr()
}
impl GpuKvStore {
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        boot: Uuid,
        max_lanes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_LANES).contains(&max_lanes),
            "short KV lane bound1..16"
        );
        let source_id = u64::from_str_radix(
            env!("DSV41_KV_STORE_SOURCE_ID").trim_start_matches("0x"),
            16,
        )?;
        let module = Module::load(
            dev,
            stream,
            PTX,
            NAMES,
            "dsv41_kv_store_identity",
            source_id,
            1,
        )?;
        let windows = dev.alloc_zeros::<bf16>(max_lanes * LAYERS * WINDOW * DIM)?;
        let compressed = dev.alloc_zeros::<bf16>(max_lanes * 4 * MAX_CONTEXT * DIM)?;
        let mut statuses = Vec::new();
        for _ in 0..max_lanes {
            let mut status = dev.alloc_zeros::<u32>(1)?;
            dev.memset_zeros(&mut status)?;
            statuses.push(Arc::new(status));
        }
        Ok(Self {
            dev: dev.clone(),
            module,
            stream_id: stream.stream as usize,
            source_id,
            max_lanes,
            cursor: KvCursor::new(boot, max_lanes)?,
            windows,
            compressed,
            statuses,
            registry: Arc::new(Mutex::new(Registry {
                lanes: BTreeMap::new(),
            })),
            next_snapshot: 0,
        })
    }
    pub fn source_id(&self) -> u64 {
        self.source_id
    }
    pub fn ptx_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(PTX.as_bytes()))
    }
    pub fn cuda_source_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(CUDA_SOURCE))
    }
    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }
    fn stream(&self, s: &CudaStream) -> Result<()> {
        ensure!(
            s.stream as usize == self.stream_id,
            "exact owner KV blocking stream required"
        );
        self.module.check_stream(s)
    }
    fn launch(&self, name: &str, s: &CudaStream, count: usize, args: &mut [u64]) -> Result<()> {
        ensure!(
            count > 0 && count <= MAX_LANES * LAYERS * WINDOW * DIM,
            "KV launch bound"
        );
        let mut params: Vec<*mut c_void> = args
            .iter_mut()
            .map(|v| v as *mut u64 as *mut c_void)
            .collect();
        unsafe {
            self.module.launch(
                name,
                s,
                (count.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut params,
            )
        }
    }
    fn token(&self, key: LaneKey) -> Result<Token> {
        ensure!(key.slot < self.max_lanes, "KV slot bound");
        let t = Token {
            dev: self.dev.clone(),
            registry: self.registry.clone(),
            key,
            status: self.statuses[key.slot].clone(),
        };
        t.valid()?;
        Ok(t)
    }
    pub fn reset_lane(&mut self, s: &CudaStream, key: LaneKey) -> Result<()> {
        self.stream(s)?;
        self.cursor.reset(key)?;
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("KV registry poisoned"))?
            .lanes
            .insert(key.slot, (key, false));
        let result = self.launch(
            "dsv41_kv_store_reset",
            s,
            LAYERS * WINDOW * DIM,
            &mut [
                ptr(&self.windows),
                ptr(&self.compressed),
                ptr(self.statuses[key.slot].as_ref()),
                key.slot as u64,
            ],
        );
        if result.is_err() {
            self.token(key)?.poison();
            self.cursor.poison(key)?;
        }
        result
    }
    /// Waits for all source reads/copies before return so input owners may be
    /// released safely. No cache tensor or per-head data is copied to the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        s: &CudaStream,
        layer: usize,
        key: LaneKey,
        start: usize,
        tokens: usize,
        current_window_qat: &CudaSlice<bf16>,
        new_compressed_qat: Option<&CudaSlice<bf16>>,
    ) -> Result<OwnedKvSnapshot> {
        let snapshot = unsafe {
            self.prepare_async(
                s,
                layer,
                key,
                start,
                tokens,
                current_window_qat,
                new_compressed_qat,
            )
        }?;
        if let Err(e) = self.dev.synchronize() {
            snapshot.token.poison();
            self.cursor.poison(key)?;
            return Err(e.into());
        }
        Ok(snapshot)
    }
    /// Queues source copies/assembly on the EXACT store stream. Caller must
    /// retain both supplied producer owners until snapshot.check/completion,
    /// and check their producer statuses separately before publication. The
    /// returned snapshot owns all outputs/status and never borrows this store.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare_async(
        &mut self,
        s: &CudaStream,
        layer: usize,
        key: LaneKey,
        start: usize,
        tokens: usize,
        current_window_qat: &CudaSlice<bf16>,
        new_compressed_qat: Option<&CudaSlice<bf16>>,
    ) -> Result<OwnedKvSnapshot> {
        self.stream(s)?;
        let token = self.token(key)?;
        self.module.check_buffer(current_window_qat)?;
        ensure!(
            current_window_qat.len()
                == tokens
                    .checked_mul(DIM)
                    .context("window input extent overflow")?,
            "window QAT input shape"
        );
        let supplied = match new_compressed_qat {
            Some(v) => {
                self.module.check_buffer(v)?;
                ensure!(v.len() % DIM == 0, "compressed QAT row extent");
                Some(v.len() / DIM)
            }
            None => None,
        };
        let plan = self.cursor.plan(layer, key, start, tokens, supplied)?;
        let serial = self
            .next_snapshot
            .checked_add(1)
            .context("KV snapshot counter overflow")?;
        let kv = self.dev.alloc_zeros::<bf16>(plan.kv_rows * DIM)?;
        let indices = self.dev.alloc_zeros::<i32>(tokens * plan.topk)?;
        let status = ptr(token.status.as_ref());
        let lane = key.slot as u64;
        let l = layer as u64;
        let result = (|| {
            self.launch(
                "dsv41_kv_store_window",
                s,
                tokens * DIM,
                &mut [
                    ptr(current_window_qat),
                    ptr(&self.windows),
                    status,
                    lane,
                    l,
                    start as u64,
                    tokens as u64,
                ],
            )?;
            if plan.compressed_owner == Some(layer) && plan.new_compressed_rows > 0 {
                self.launch(
                    "dsv41_kv_store_compressed",
                    s,
                    plan.new_compressed_rows * DIM,
                    &mut [
                        ptr(new_compressed_qat.unwrap()),
                        ptr(&self.compressed),
                        status,
                        lane,
                        plan.owner_slot.unwrap() as u64,
                        (start / plan.ratio) as u64,
                        plan.new_compressed_rows as u64,
                    ],
                )?;
            }
            self.launch(
                "dsv41_kv_store_snapshot",
                s,
                plan.kv_rows * DIM,
                &mut [
                    ptr(&kv),
                    ptr(current_window_qat),
                    ptr(&self.windows),
                    ptr(&self.compressed),
                    status,
                    lane,
                    l,
                    start as u64,
                    plan.window_rows as u64,
                    plan.compressed_rows as u64,
                    plan.owner_slot.unwrap_or(0) as u64,
                ],
            )?;
            self.launch(
                "dsv41_kv_store_indices",
                s,
                tokens * plan.topk,
                &mut [
                    ptr(&indices),
                    start as u64,
                    tokens as u64,
                    plan.ratio as u64,
                    plan.window_rows as u64,
                    plan.window_topk as u64,
                    plan.compressed_rows as u64,
                ],
            )?;
            Ok(())
        })();
        if let Err(e) = result {
            token.poison();
            self.cursor.poison(key)?;
            let _ = self.dev.synchronize();
            return Err(e);
        }
        self.cursor.commit(&plan)?;
        self.next_snapshot = serial;
        Ok(OwnedKvSnapshot {
            plan,
            serial,
            kv,
            indices,
            token,
        })
    }
}
pub struct OwnedKvSnapshot {
    plan: KvPlan,
    serial: u64,
    kv: CudaSlice<bf16>,
    indices: CudaSlice<i32>,
    token: Token,
}
impl OwnedKvSnapshot {
    pub fn plan(&self) -> &KvPlan {
        &self.plan
    }
    pub fn serial(&self) -> u64 {
        self.serial
    }
    pub fn kv_rows(&self) -> usize {
        self.plan.kv_rows
    }
    pub fn topk(&self) -> usize {
        self.plan.topk
    }
    pub fn tokens(&self) -> usize {
        self.plan.tokens
    }
    pub fn layer(&self) -> usize {
        self.plan.layer
    }
    pub fn key(&self) -> LaneKey {
        self.plan.key
    }
    pub fn start(&self) -> usize {
        self.plan.start
    }
    pub fn compressed_owner(&self) -> Option<usize> {
        self.plan.compressed_owner
    }
    pub fn check(&self) -> Result<()> {
        self.token.check()
    }
    pub fn status_code(&self) -> Result<u32> {
        self.token.dev.bind_to_thread()?;
        self.token.dev.synchronize()?;
        Ok(self.token.dev.dtoh_sync_copy(self.token.status.as_ref())?[0])
    }
    pub fn download_checked(&self) -> Result<(Vec<bf16>, Vec<i32>)> {
        self.check()?;
        Ok((
            self.token.dev.dtoh_sync_copy(&self.kv)?,
            self.token.dev.dtoh_sync_copy(&self.indices)?,
        ))
    }
    /// Same ordered stream; caller checks producer and snapshot statuses before
    /// consuming/publishing output and retains this owner for dependent kernels.
    pub unsafe fn device_kv_unchecked(&self) -> &CudaSlice<bf16> {
        &self.kv
    }
    pub unsafe fn device_indices_unchecked(&self) -> &CudaSlice<i32> {
        &self.indices
    }
}
impl Drop for OwnedKvSnapshot {
    fn drop(&mut self) {
        let _ = self.token.dev.synchronize();
    }
}
