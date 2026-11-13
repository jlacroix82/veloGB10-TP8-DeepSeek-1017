//! PRIVATE UNREGISTERED8192-context C1 cache composition draft.
//! Inputs are already checked source RoPE/QAT BF16. No model qualification,
//! projection readback, checkpoint IO or automatic serving is implied.
use super::{
    chunk_plan::{
        self, ChunkCursor, ChunkTicket, LayerChunk, KV_SOURCES, LAYERS, MAX_CONTEXT, WINDOW,
    },
    config::Dsv41Config,
    gpu_index_block::{GpuIndexOutput, IndexOutputIdentity},
    gpu_module::Module,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};
use uuid::Uuid;
const D: usize = 512;
const WINDOW_ELEMENTS: usize = LAYERS * WINDOW * D;
const COMPRESSED_ELEMENTS: usize = 4 * MAX_CONTEXT * D;
pub const CACHE_PAYLOAD_BYTES: usize = (WINDOW_ELEMENTS + COMPRESSED_ELEMENTS) * 2;
pub const SNAPSHOT_ARRAY_BOUND: usize = 16 * 1024 * 1024;
pub const PTX: &str = include_str!("../ptx/dsv41_chunk_store.ptx");
pub const SOURCE_ID: &str = env!("DSV41_CHUNK_STORE_SOURCE_ID");

/// Borrowed owner metadata: no device work or allocation on the validation path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OwnerRef<'a> {
    rank: usize,
    manifest: &'a str,
    completion: [u8; 32],
    stream_id: usize,
}
fn check_owner_metadata(actual: OwnerRef<'_>, requested: OwnerRef<'_>) -> Result<()> {
    ensure!(
        actual.rank < 8
            && actual.stream_id != 0
            && actual.manifest.len() == 64
            && actual
                .manifest
                .bytes()
                .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
            && actual == requested,
        "chunk-store rank/manifest/completion/stream owner differs"
    );
    Ok(())
}

struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: T,
}
impl<T> Queued<T> {
    fn new(dev: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: dev.clone(),
            value,
        }
    }
}
impl<T> std::ops::Deref for Queued<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl<T> std::ops::DerefMut for Queued<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
impl<T> Drop for Queued<T> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}

pub struct ChunkStoreKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    module: Module,
    source_decode_indices: super::gpu_chunk_source_indices::SourceDecodeIndices,
}
impl ChunkStoreKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        let module = Module::load(
            dev,
            &stream,
            PTX,
            &[
                "dsv41_chunk_store_reset",
                "dsv41_chunk_store_window_snapshot",
                "dsv41_chunk_store_window_write",
                "dsv41_chunk_store_compressed_append",
                "dsv41_chunk_store_combined_snapshot",
                "dsv41_chunk_store_indices",
            ],
            "dsv41_chunk_store_identity",
            u64::from_str_radix(SOURCE_ID.trim_start_matches("0x"), 16)?,
            1,
        )?;
        let source_decode_indices =
            super::gpu_chunk_source_indices::SourceDecodeIndices::load(dev, stream.clone())?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            stream,
            boot,
            module,
            source_decode_indices,
        }))
    }
    fn buffer<T>(&self, value: &CudaSlice<T>) -> Result<()> {
        self.module.check_buffer(value)
    }
    unsafe fn launch(&self, name: &str, n: usize, args: &mut [*mut c_void]) -> Result<()> {
        ensure!(
            n > 0 && n <= COMPRESSED_ELEMENTS,
            "chunk kernel launch bound"
        );
        self.module.launch(
            name,
            &self.stream,
            ((n as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            args,
        )
    }
}
#[derive(Default)]
struct Validity {
    lane: Option<LaneKey>,
    failed: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Reservation {
    ticket: ChunkTicket,
    layer: usize,
    nonce: u64,
}
#[derive(Default)]
struct StageState {
    reserved: Option<Reservation>,
    next_nonce: u64,
}
impl StageState {
    fn reserve(&mut self, ticket: &ChunkTicket, layer: usize) -> Result<Reservation> {
        ensure!(
            self.reserved.is_none() && layer < LAYERS,
            "chunk layer already staged/invalid"
        );
        let nonce = self
            .next_nonce
            .checked_add(1)
            .context("chunk snapshot nonce exhausted")?;
        let r = Reservation {
            ticket: ticket.clone(),
            layer,
            nonce,
        };
        self.next_nonce = nonce;
        self.reserved = Some(r.clone());
        Ok(r)
    }
    fn confirm(&self, r: &Reservation) -> Result<()> {
        ensure!(
            self.reserved.as_ref() == Some(r),
            "cache snapshot is not the exact reserved ticket/layer/nonce"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Publication {
    ticket: ChunkTicket,
    owner: usize,
    rows: usize,
}
#[derive(Debug, Clone, Serialize)]
pub struct ChunkAllocation {
    pub cache_payload_bf16_bytes: usize,
    pub cache_status_bytes: usize,
    pub total_cache_array_bytes: usize,
    pub fixed_compressed_stride_rows: usize,
    pub native_payload_is_smaller_than_fixed_stride: bool,
    pub whole_model_memory_fit_qualified: bool,
}
pub struct GpuChunkStore {
    k: Arc<ChunkStoreKernels>,
    rank: usize,
    manifest: String,
    completion: [u8; 32],
    cursor: ChunkCursor,
    windows: Queued<CudaSlice<bf16>>,
    compressed: Queued<CudaSlice<bf16>>,
    reset_status: Queued<CudaSlice<u32>>,
    validity: Arc<Mutex<Validity>>,
    stage: StageState,
    publications: [Option<Publication>; 4],
    source_ends: [usize; 4],
    // Explicit semantic phase, never inferred for generic positive prefill.
    native_decode_ticket: Option<ChunkTicket>,
}
impl GpuChunkStore {
    pub fn new(
        k: Arc<ChunkStoreKernels>,
        config: &Dsv41Config,
        manifest: &RankManifest,
        completion_owner: [u8; 32],
    ) -> Result<Self> {
        manifest.verify_identity()?;
        chunk_plan::validate_config(config)?;
        ensure!(manifest.rank < 8, "chunk rankoutsideTP8");
        let windows = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(WINDOW_ELEMENTS)?);
        let compressed = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(COMPRESSED_ELEMENTS)?);
        let reset_status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        Ok(Self {
            rank: manifest.rank,
            manifest: manifest.manifest_sha256.clone(),
            completion: completion_owner,
            cursor: ChunkCursor::new(config, k.boot, 1)?,
            k,
            windows,
            compressed,
            reset_status,
            validity: Arc::new(Mutex::new(Validity::default())),
            stage: StageState::default(),
            publications: std::array::from_fn(|_| None),
            source_ends: [0; 4],
            native_decode_ticket: None,
        })
    }
    pub fn allocation(&self) -> ChunkAllocation {
        ChunkAllocation {
            cache_payload_bf16_bytes: CACHE_PAYLOAD_BYTES,
            cache_status_bytes: 4,
            total_cache_array_bytes: CACHE_PAYLOAD_BYTES + 4,
            fixed_compressed_stride_rows: MAX_CONTEXT,
            native_payload_is_smaller_than_fixed_stride: true,
            whole_model_memory_fit_qualified: false,
        }
    }
    /// Validate the caller's exact serving owner before projections or cache
    /// publication, including ratio0 layers that carry no typed index receipt.
    pub fn check_owner(
        &self,
        rank: usize,
        manifest_sha256: &str,
        completion_owner: [u8; 32],
        stream: &CudaStream,
    ) -> Result<()> {
        check_owner_metadata(
            OwnerRef {
                rank: self.rank,
                manifest: &self.manifest,
                completion: self.completion,
                stream_id: self.k.stream.stream as usize,
            },
            OwnerRef {
                rank,
                manifest: manifest_sha256,
                completion: completion_owner,
                stream_id: stream.stream as usize,
            },
        )?;
        self.k.module.check_stream(stream)
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        ensure!(
            key.boot == self.k.boot && key.slot == 0 && key.generation > 0,
            "C1slot0 chunk resetidentity"
        );
        self.k.dev.synchronize()?;
        self.cursor.reset(key)?;
        {
            let mut v = self
                .validity
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk validity poisoned"))?;
            v.lane = Some(key);
            v.failed = true;
        }
        self.stage.reserved = None;
        self.publications = std::array::from_fn(|_| None);
        self.source_ends = [0; 4];
        let result = (|| {
            self.k.dev.memset_zeros(&mut *self.reset_status)?;
            let (mut w, mut c, mut st, mut lane, mut lanes) = (
                *self.windows.device_ptr(),
                *self.compressed.device_ptr(),
                *self.reset_status.device_ptr(),
                0u64,
                1u64,
            );
            let mut args = [
                &mut w as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut st as *mut _ as *mut c_void,
                &mut lane as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
            ];
            unsafe {
                self.k
                    .launch("dsv41_chunk_store_reset", COMPRESSED_ELEMENTS, &mut args)?;
            }
            self.k.dev.synchronize()?;
            ensure!(
                self.k.dev.dtoh_sync_copy(&self.reset_status)? == [0],
                "chunk reset status"
            );
            Ok(())
        })();
        if result.is_ok() {
            self.validity
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk validity poisoned"))?
                .failed = false;
        }
        result
    }
    fn valid(&self, key: LaneKey) -> Result<()> {
        let v = self
            .validity
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk validity poisoned"))?;
        ensure!(
            !v.failed && v.lane == Some(key),
            "stale/failed chunk generation"
        );
        Ok(())
    }
    fn poison(&mut self, ticket: &ChunkTicket) {
        let _ = self.cursor.abort(ticket);
        if let Ok(mut v) = self.validity.lock() {
            v.failed = true;
        }
        self.stage.reserved = None;
    }
    pub fn begin(&mut self, key: LaneKey, start: usize, tokens: usize) -> Result<ChunkTicket> {
        self.valid(key)?;
        ensure!(
            self.stage.reserved.is_none(),
            "cannot begin during reserved cache stage"
        );
        let ticket = self.cursor.begin(key, start, tokens)?;
        self.native_decode_ticket = None;
        Ok(ticket)
    }
    /// Explicit one-token native DECODE. Generic/prefill begin retains its
    /// predecessor layout, including the separately retained472 raw goldens.
    pub fn begin_native_decode(&mut self, key: LaneKey, start: usize) -> Result<ChunkTicket> {
        ensure!(start > 0, "native decode needs an existing prefix");
        let ticket = self.begin(key, start, 1)?;
        self.native_decode_ticket = Some(ticket.clone());
        Ok(ticket)
    }
    pub fn layer(&self, ticket: &ChunkTicket, layer: usize) -> Result<LayerChunk> {
        self.valid(ticket.lane)?;
        self.cursor.layer(ticket, layer)
    }
    pub fn committed_prefix(&self, key: LaneKey) -> Result<usize> {
        self.valid(key)?;
        self.cursor.committed_prefix(key)
    }
    /// This initial correctness path fences every input copy before return. The
    /// caller may release producer tensors afterwards; tensor payloads stay onGPU.
    pub fn prepare_checked(
        &mut self,
        ticket: &ChunkTicket,
        layer: usize,
        window_qat: &CudaSlice<bf16>,
        new_compressed_qat: Option<&CudaSlice<bf16>>,
        index: Option<Arc<GpuIndexOutput>>,
    ) -> Result<OwnedChunkSnapshot> {
        let plan = self.layer(ticket, layer)?;
        ensure!(
            self.stage.reserved.is_none(),
            "duplicate cache prepare before layer commit"
        );
        self.k.module.check_stream(&self.k.stream)?;
        self.k.buffer(window_qat)?;
        ensure!(
            window_qat.len() == ticket.tokens * D,
            "current window QAT extent"
        );
        let comp = plan.compressed.as_ref();
        let compressed_rows = comp.map_or(0, |x| x.stored_rows);
        let new_rows = comp.map_or(0, |x| x.new_rows);
        let owner_slot = plan
            .kv_source
            .map(|source| {
                KV_SOURCES
                    .iter()
                    .position(|x| *x == source)
                    .context("unsupported compressed owner")
            })
            .transpose()?;
        let selection = if plan.ratio == 0 {
            ensure!(
                index.is_none() && new_compressed_qat.is_none(),
                "uncompressed layer must not supply index/compressor"
            );
            None
        } else {
            index
                .as_ref()
                .context("compressed layer requires typed checked index source")?
                .indices_for_plan(&plan, self.rank, &self.manifest, self.completion)?
        };
        if let Some(s) = selection {
            self.k.buffer(s)?;
            ensure!(
                s.len() == ticket.tokens * compressed_rows.min(512),
                "source offset0 index extent"
            );
        }
        if plan.owns_kv_and_index_keys {
            let slot = owner_slot.context("compressed owner slot")?;
            ensure!(
                self.source_ends[slot] == ticket.start,
                "compressed source stale/noncontiguous"
            );
            if new_rows > 0 {
                let x = new_compressed_qat
                    .context("owner requires new completed compressed QAT rows")?;
                self.k.buffer(x)?;
                ensure!(x.len() == new_rows * D, "new compressed QAT extent");
            } else {
                ensure!(
                    new_compressed_qat.is_none(),
                    "incomplete compression group cannot append rows"
                );
            }
        } else if let Some(slot) = owner_slot {
            ensure!(
                new_compressed_qat.is_none()
                    && self.publications[slot].as_ref()
                        == Some(&Publication {
                            ticket: ticket.clone(),
                            owner: plan.kv_source.unwrap(),
                            rows: compressed_rows
                        }),
                "compressed consumer missing exact source ticket/publication"
            );
        }
        let receipt = index.as_ref().map(|x| x.receipt().cloned()).transpose()?;
        let window_rows = plan.window.snapshot_rows;
        let kv_rows = window_rows
            .checked_add(compressed_rows)
            .context("snapshot extent overflow")?;
        let array_bytes = (window_rows + kv_rows)
            .checked_mul(D * 2)
            .and_then(|x| x.checked_add(ticket.tokens * plan.attention_index_capacity * 4 + 4))
            .context("snapshot bytesoverflow")?;
        ensure!(
            array_bytes <= SNAPSHOT_ARRAY_BOUND,
            "chunk snapshot/scratch exceeds16MiB"
        );
        // Reserve BEFORE any mutable cache write. A failed partial stage cannot be
        // retried against overwritten ring rows under the same generation.
        let reservation = self.stage.reserve(ticket, layer)?;
        let result = self.prepare_reserved(
            &plan,
            &reservation,
            window_qat,
            new_compressed_qat,
            selection,
            owner_slot,
            compressed_rows,
            window_rows,
            kv_rows,
            array_bytes,
            receipt,
            index.clone(),
        );
        match result {
            Ok(snapshot) => {
                if plan.owns_kv_and_index_keys {
                    let slot = owner_slot.unwrap();
                    self.source_ends[slot] = ticket.end;
                    self.publications[slot] = Some(Publication {
                        ticket: ticket.clone(),
                        owner: layer,
                        rows: compressed_rows,
                    });
                }
                Ok(snapshot)
            }
            Err(error) => {
                self.poison(ticket);
                Err(error)
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_reserved(
        &self,
        plan: &LayerChunk,
        reservation: &Reservation,
        window: &CudaSlice<bf16>,
        new_compressed: Option<&CudaSlice<bf16>>,
        selection: Option<&CudaSlice<i32>>,
        owner_slot: Option<usize>,
        compressed_rows: usize,
        window_rows: usize,
        kv_rows: usize,
        array_bytes: usize,
        index_receipt: Option<IndexOutputIdentity>,
        index_owner: Option<Arc<GpuIndexOutput>>,
    ) -> Result<OwnedChunkSnapshot> {
        let k = self.k.clone();
        let t = &plan.ticket;
        let prior = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(window_rows * D)?);
        let kv = Queued::new(&k.dev, k.dev.alloc_zeros::<bf16>(kv_rows * D)?);
        let mut indices = Queued::new(
            &k.dev,
            k.dev
                .alloc_zeros::<i32>(t.tokens * plan.attention_index_capacity)?,
        );
        let mut status = Queued::new(&k.dev, k.dev.alloc_zeros::<u32>(1)?);
        k.dev.memset_zeros(&mut *status)?;
        let (
            mut out,
            mut current,
            mut cache,
            mut st,
            mut lane,
            mut lanes,
            mut layer,
            mut start,
            mut tokens,
        ) = (
            *prior.device_ptr(),
            *window.device_ptr(),
            *self.windows.device_ptr(),
            *status.device_ptr(),
            0u64,
            1u64,
            plan.layer as u64,
            t.start as u64,
            t.tokens as u64,
        );
        let mut args = [
            &mut out as *mut _ as *mut c_void,
            &mut current as *mut _ as *mut c_void,
            &mut cache as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut lane as *mut _ as *mut c_void,
            &mut lanes as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut start as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
        ];
        unsafe {
            k.launch(
                "dsv41_chunk_store_window_snapshot",
                window_rows * D,
                &mut args,
            )?;
        }
        let mut args = [
            &mut current as *mut _ as *mut c_void,
            &mut cache as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut lane as *mut _ as *mut c_void,
            &mut lanes as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut start as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
        ];
        unsafe {
            k.launch("dsv41_chunk_store_window_write", t.tokens * D, &mut args)?;
        }
        let mut owner = owner_slot.unwrap_or(0) as u64;
        if let Some(source) = new_compressed {
            let (mut source, mut cache, mut first, mut rows) = (
                *source.device_ptr(),
                *self.compressed.device_ptr(),
                (t.start / plan.ratio) as u64,
                plan.compressed.as_ref().unwrap().new_rows as u64,
            );
            let mut args = [
                &mut source as *mut _ as *mut c_void,
                &mut cache as *mut _ as *mut c_void,
                &mut st as *mut _ as *mut c_void,
                &mut lane as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
                &mut owner as *mut _ as *mut c_void,
                &mut first as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
            ];
            unsafe {
                k.launch(
                    "dsv41_chunk_store_compressed_append",
                    rows as usize * D,
                    &mut args,
                )?;
            }
        }
        let (mut output, mut old, mut cache, mut wrows, mut crows) = (
            *kv.device_ptr(),
            *prior.device_ptr(),
            *self.compressed.device_ptr(),
            window_rows as u64,
            compressed_rows as u64,
        );
        let mut args = [
            &mut output as *mut _ as *mut c_void,
            &mut old as *mut _ as *mut c_void,
            &mut cache as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut lane as *mut _ as *mut c_void,
            &mut lanes as *mut _ as *mut c_void,
            &mut owner as *mut _ as *mut c_void,
            &mut wrows as *mut _ as *mut c_void,
            &mut crows as *mut _ as *mut c_void,
        ];
        unsafe {
            k.launch(
                "dsv41_chunk_store_combined_snapshot",
                kv_rows * D,
                &mut args,
            )?;
        }
        let (mut out, mut selected, mut ratio) = (
            *indices.device_ptr(),
            selection.map(|s| *s.device_ptr()).unwrap_or(0),
            plan.ratio as u64,
        );
        let mut args = [
            &mut out as *mut _ as *mut c_void,
            &mut selected as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut start as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
            &mut ratio as *mut _ as *mut c_void,
            &mut crows as *mut _ as *mut c_void,
        ];
        let source_decode_order = self.native_decode_ticket.as_ref() == Some(t)
            && super::gpu_chunk_source_indices::applicable(plan);
        if source_decode_order {
            // Logical source128 window positions, including leading masked
            // slots, are preserved. Physical chronological KV offsets differ
            // from the old ring but the gathered BF16 tile values do not.
            k.source_decode_indices
                .enqueue(&k.stream, plan, &mut indices)?;
        } else {
            unsafe {
                k.launch(
                    "dsv41_chunk_store_indices",
                    t.tokens * plan.attention_index_capacity,
                    &mut args,
                )?;
            }
        }
        k.dev.synchronize()?;
        ensure!(
            k.dev.dtoh_sync_copy(&status)? == [0],
            "chunk cache finite/extent/selectedindex status"
        );
        Ok(OwnedChunkSnapshot {
            k,
            kv,
            indices,
            status,
            _prior: prior,
            _index_owner: index_owner,
            index_receipt,
            plan: plan.clone(),
            reservation: reservation.clone(),
            validity: self.validity.clone(),
            rank: self.rank,
            manifest: self.manifest.clone(),
            completion: self.completion,
            kv_rows,
            array_bytes,
            source_decode_order,
        })
    }
    /// Caller must have checked the ATTENTION result using this exact snapshot.
    /// This method itself certifies only the checked cache and ordered transaction.
    pub fn commit_layer(
        &mut self,
        ticket: &ChunkTicket,
        snapshot: &OwnedChunkSnapshot,
    ) -> Result<()> {
        snapshot.check()?;
        self.stage.confirm(&snapshot.reservation)?;
        ensure!(
            &snapshot.plan.ticket == ticket
                && snapshot.rank == self.rank
                && snapshot.manifest == self.manifest
                && snapshot.completion == self.completion
                && Arc::ptr_eq(&snapshot.validity, &self.validity),
            "cache commit source/ticket/owner mismatch"
        );
        self.cursor.commit_layer(&snapshot.plan)?;
        self.stage.reserved = None;
        Ok(())
    }
    /// Caller checks the final model head BEFORE committing all40 layer results.
    pub fn commit(&mut self, ticket: &ChunkTicket) -> Result<()> {
        self.valid(ticket.lane)?;
        ensure!(
            self.stage.reserved.is_none(),
            "cannotcommit an uncheckedreserved layer"
        );
        self.cursor.commit(ticket)
    }
    pub fn abort(&mut self, ticket: &ChunkTicket) -> Result<()> {
        self.k.dev.synchronize()?;
        self.cursor.abort(ticket)?;
        if let Ok(mut v) = self.validity.lock() {
            v.failed = true;
        }
        self.stage.reserved = None;
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ChunkSnapshotIdentity {
    pub rank: usize,
    pub manifest_sha256: String,
    pub ticket: ChunkTicket,
    pub layer: usize,
    pub kv_rows: usize,
    pub window_rows: usize,
    pub compressed_rows: usize,
    pub topk: usize,
    pub owned_snapshot_array_bytes: usize,
    pub source_id_hex: &'static str,
    pub copied_before_ring_write: bool,
    pub selection_offset_added_once: bool,
    pub source_decode_window_order: bool,
    pub full_model_or_quality_qualified: bool,
}
pub struct OwnedChunkSnapshot {
    k: Arc<ChunkStoreKernels>,
    kv: Queued<CudaSlice<bf16>>,
    indices: Queued<CudaSlice<i32>>,
    status: Queued<CudaSlice<u32>>,
    _prior: Queued<CudaSlice<bf16>>,
    _index_owner: Option<Arc<GpuIndexOutput>>,
    index_receipt: Option<IndexOutputIdentity>,
    plan: LayerChunk,
    reservation: Reservation,
    validity: Arc<Mutex<Validity>>,
    rank: usize,
    manifest: String,
    completion: [u8; 32],
    kv_rows: usize,
    array_bytes: usize,
    source_decode_order: bool,
}
impl OwnedChunkSnapshot {
    pub fn check(&self) -> Result<()> {
        let v = self
            .validity
            .lock()
            .map_err(|_| anyhow::anyhow!("chunk validity poisoned"))?;
        ensure!(
            !v.failed && v.lane == Some(self.plan.ticket.lane),
            "immutablecache snapshot stale/failed generation"
        );
        drop(v);
        self.k.dev.synchronize()?;
        ensure!(
            self.k.dev.dtoh_sync_copy(&self.status)? == [0],
            "chunk snapshot device status"
        );
        Ok(())
    }
    pub fn identity(&self) -> ChunkSnapshotIdentity {
        ChunkSnapshotIdentity {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            ticket: self.plan.ticket.clone(),
            layer: self.plan.layer,
            kv_rows: self.kv_rows,
            window_rows: self.plan.window.snapshot_rows,
            compressed_rows: self.plan.compressed.as_ref().map_or(0, |x| x.stored_rows),
            topk: self.plan.attention_index_capacity,
            owned_snapshot_array_bytes: self.array_bytes,
            source_id_hex: SOURCE_ID,
            copied_before_ring_write: true,
            selection_offset_added_once: true,
            source_decode_window_order: self.source_decode_order,
            full_model_or_quality_qualified: false,
        }
    }
    pub fn source_decode_window_order(&self) -> bool {
        self.source_decode_order
    }
    pub fn plan(&self) -> &LayerChunk {
        &self.plan
    }
    pub fn kv_rows(&self) -> usize {
        self.kv_rows
    }
    pub fn topk(&self) -> usize {
        self.plan.attention_index_capacity
    }
    pub fn index_receipt(&self) -> Option<&IndexOutputIdentity> {
        self.index_receipt.as_ref()
    }
    /// Exact retained stream; caller retains snapshot until attention completes and
    /// checks this cache plus every upstream/attention status before publication.
    pub unsafe fn device_kv_unchecked(&self) -> &CudaSlice<bf16> {
        &self.kv
    }
    pub unsafe fn device_indices_unchecked(&self) -> &CudaSlice<i32> {
        &self.indices
    }
}
impl Drop for GpuChunkStore {
    fn drop(&mut self) {
        let _ = self.k.dev.synchronize();
    }
}
impl Drop for OwnedChunkSnapshot {
    fn drop(&mut self) {
        let _ = self.k.dev.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn ticket() -> ChunkTicket {
        let boot = Uuid::new_v4();
        ChunkTicket {
            lane: LaneKey {
                boot,
                slot: 0,
                generation: 1,
            },
            version: 7,
            start: 128,
            tokens: 128,
            end: 256,
        }
    }
    #[test]
    fn reservation_cannot_restage_overwritten_ring_or_commit_a_forged_snapshot() {
        let t = ticket();
        let mut s = StageState::default();
        let a = s.reserve(&t, 2).unwrap();
        assert!(s.reserve(&t, 2).is_err());
        s.confirm(&a).unwrap();
        let mut forged = a.clone();
        forged.ticket.version += 1;
        assert!(s.confirm(&forged).is_err());
        forged = a.clone();
        forged.layer += 1;
        assert!(s.confirm(&forged).is_err());
        s.reserved = None;
        let b = s.reserve(&t, 3).unwrap();
        assert!(b.nonce > a.nonce);
        assert!(s.confirm(&a).is_err());
    }
    #[test]
    fn owner_binding_rejects_cross_rank_manifest_completion_stream_and_invalid_metadata() {
        let manifest = "a".repeat(64);
        let actual = OwnerRef {
            rank: 3,
            manifest: &manifest,
            completion: [7; 32],
            stream_id: 0x1000,
        };
        check_owner_metadata(actual, actual).unwrap();
        let other_manifest = "b".repeat(64);
        for bad in [
            OwnerRef { rank: 4, ..actual },
            OwnerRef {
                manifest: &other_manifest,
                ..actual
            },
            OwnerRef {
                completion: [8; 32],
                ..actual
            },
            OwnerRef {
                stream_id: 0x2000,
                ..actual
            },
            OwnerRef {
                stream_id: 0,
                ..actual
            },
        ] {
            assert!(check_owner_metadata(actual, bad).is_err());
        }
        for bad in [
            OwnerRef { rank: 8, ..actual },
            OwnerRef {
                manifest: "invalid",
                ..actual
            },
            OwnerRef {
                stream_id: 0,
                ..actual
            },
        ] {
            assert!(check_owner_metadata(bad, bad).is_err());
        }
    }
    #[test]
    fn actual_fixed_allocations_and_largest_owned_snapshot_are_honest() {
        assert_eq!(CACHE_PAYLOAD_BYTES, 37 * 1024 * 1024);
        let max_snapshot = (255 + 255 + 8192) * 512 * 2 + 128 * (128 + 512) * 4 + 4;
        assert!(max_snapshot < 16 * 1024 * 1024);
        assert_ne!(CACHE_PAYLOAD_BYTES, 25 * 1024 * 1024);
    }
}
