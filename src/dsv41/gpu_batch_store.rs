//! PRIVATE true multi-slot cache ownership and one packed KV snapshot. This
//! does not execute16 serial C1 models. Combined projections/attention follow.
//! Initial safe entry refuses learned top-k demand (>512 compressed rows).
use super::{
    batch_attention_layout::AttentionPhase,
    batch_plan::{BatchCursor, BatchLayerPlan, BatchTicket, MAX_CONTEXT, MAX_LANES},
    chunk_plan::KV_SOURCES,
    config::Dsv41Config,
    gpu_module::Module,
    scheduler::LaneKey,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};
use uuid::Uuid;
const D: usize = 512;
const WINDOW_PER_LANE: usize = 40 * 128 * D;
const COMPRESSED_PER_LANE: usize = 4 * MAX_CONTEXT * D;
const KEY_D: usize = 128;
const KEY_ROWS_PER_LANE: usize = 3 * (MAX_CONTEXT / 2) + MAX_CONTEXT;
const KEY_ELEMENTS_PER_LANE: usize = KEY_ROWS_PER_LANE * KEY_D;
pub const PAYLOAD_BYTES_PER_LANE: usize =
    (WINDOW_PER_LANE + COMPRESSED_PER_LANE + KEY_ELEMENTS_PER_LANE) * 2;
pub const PACKED_SNAPSHOT_BOUND_BYTES: usize = 160 * 1024 * 1024;
pub const RAW_PTX: &str = include_str!("../ptx/dsv41_chunk_store.ptx");
pub const LAYOUT_PTX: &str = include_str!("../ptx/dsv41_batch_layout.ptx");

struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: T,
}
impl<T> Queued<T> {
    fn new(d: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: d.clone(),
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
fn offset<T>(value: &CudaSlice<T>, first: usize, count: usize) -> Result<u64> {
    ensure!(
        first
            .checked_add(count)
            .is_some_and(|end| end <= value.len()),
        "packed tensor range outside owner"
    );
    (*value.device_ptr())
        .checked_add(
            first
                .checked_mul(std::mem::size_of::<T>())
                .context("packed byte offset overflow")? as u64,
        )
        .context("packed device pointer overflow")
}
fn key_base_rows(lane: usize, owner: usize) -> Result<usize> {
    ensure!(
        lane < MAX_LANES && owner < 4,
        "packed key source/slot outside bounds"
    );
    Ok(lane * KEY_ROWS_PER_LANE + owner * (MAX_CONTEXT / 2))
}
pub struct BatchStoreKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    raw: Module,
    layout: Module,
}
impl BatchStoreKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        let raw = Module::load(
            dev,
            &stream,
            RAW_PTX,
            &[
                "dsv41_chunk_store_reset",
                "dsv41_chunk_store_window_snapshot",
                "dsv41_chunk_store_window_write",
                "dsv41_chunk_store_compressed_append",
                "dsv41_chunk_store_combined_snapshot",
                "dsv41_chunk_store_indices",
            ],
            "dsv41_chunk_store_identity",
            u64::from_str_radix(
                env!("DSV41_CHUNK_STORE_SOURCE_ID").trim_start_matches("0x"),
                16,
            )?,
            1,
        )?;
        let layout = Module::load(
            dev,
            &stream,
            LAYOUT_PTX,
            &[
                "dsv41_batch_pad_indices",
                "dsv41_batch_gather_bf16",
                "dsv41_batch_source_indices",
            ],
            "dsv41_batch_layout_identity",
            u64::from_str_radix(
                env!("DSV41_BATCH_LAYOUT_SOURCE_ID").trim_start_matches("0x"),
                16,
            )?,
            1,
        )?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            stream,
            boot,
            raw,
            layout,
        }))
    }
    fn launch(
        &self,
        layout: bool,
        name: &str,
        elements: usize,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        ensure!(
            elements > 0 && elements <= u32::MAX as usize,
            "bounded nonempty packed launch"
        );
        let module = if layout { &self.layout } else { &self.raw };
        unsafe {
            module.launch(
                name,
                &self.stream,
                (elements.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                args,
            )
        }
    }
}
#[derive(Default)]
struct Validity {
    lanes: BTreeMap<usize, (LaneKey, bool)>,
}
fn valid(registry: &Arc<Mutex<Validity>>, ticket: &BatchTicket) -> Result<()> {
    let r = registry
        .lock()
        .map_err(|_| anyhow::anyhow!("packed cache registry poisoned"))?;
    ensure!(
        ticket
            .segments()
            .iter()
            .all(|s| r.lanes.get(&s.ticket.lane.slot) == Some(&(s.ticket.lane, false))),
        "stale/failed packed cache lane generation"
    );
    Ok(())
}
fn poison(registry: &Arc<Mutex<Validity>>, ticket: &BatchTicket) {
    if let Ok(mut r) = registry.lock() {
        for s in ticket.segments() {
            if let Some((key, failed)) = r.lanes.get_mut(&s.ticket.lane.slot) {
                if *key == s.ticket.lane {
                    *failed = true;
                }
            }
        }
    }
}
#[derive(Clone)]
struct Publication {
    ticket: super::chunk_plan::ChunkTicket,
    rows: usize,
}
#[derive(Clone)]
struct Stage {
    identity: String,
    layer: usize,
    nonce: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct PackedQueryRow {
    pub key: LaneKey,
    pub native_position: usize,
    pub query_row: usize,
    pub kv_first_row: usize,
    pub kv_rows: usize,
    pub actual_topk: usize,
}

pub struct GpuBatchStore {
    k: Arc<BatchStoreKernels>,
    rank: usize,
    manifest: String,
    completion: [u8; 32],
    instance: [u8; 32],
    max_lanes: usize,
    windows: Queued<CudaSlice<bf16>>,
    compressed: Queued<CudaSlice<bf16>>,
    index_keys: Queued<CudaSlice<bf16>>,
    key_copy_rows: Queued<CudaSlice<u32>>,
    cursor: BatchCursor,
    registry: Arc<Mutex<Validity>>,
    publications: Vec<[Option<Publication>; 4]>,
    ends: Vec<[usize; 4]>,
    key_publications: Vec<[Option<Publication>; 4]>,
    key_ends: Vec<[usize; 4]>,
    stage: Option<Stage>,
    next_nonce: u64,
}
impl GpuBatchStore {
    pub fn new(
        k: Arc<BatchStoreKernels>,
        c: &Dsv41Config,
        m: &RankManifest,
        completion: [u8; 32],
        max_lanes: usize,
    ) -> Result<Self> {
        m.verify_identity()?;
        ensure!(
            (1..=MAX_LANES).contains(&max_lanes) && completion != [0; 32],
            "packed lane bound/completion owner"
        );
        let windows = Queued::new(
            &k.dev,
            k.dev.alloc_zeros::<bf16>(max_lanes * WINDOW_PER_LANE)?,
        );
        let compressed = Queued::new(
            &k.dev,
            k.dev.alloc_zeros::<bf16>(max_lanes * COMPRESSED_PER_LANE)?,
        );
        let index_keys = Queued::new(
            &k.dev,
            k.dev
                .alloc_zeros::<bf16>(max_lanes * KEY_ELEMENTS_PER_LANE)?,
        );
        let key_copy_rows = Queued::new(
            &k.dev,
            k.dev.htod_sync_copy(&(0u32..128).collect::<Vec<_>>())?,
        );
        let mut h = Sha256::new();
        h.update(b"dsv41-packed-cache-local-owner-v1\0");
        h.update(Uuid::new_v4().as_bytes());
        h.update(completion);
        h.update(m.manifest_sha256.as_bytes());
        Ok(Self {
            cursor: BatchCursor::new(c, k.boot, max_lanes)?,
            k,
            rank: m.rank,
            manifest: m.manifest_sha256.clone(),
            completion,
            instance: h.finalize().into(),
            max_lanes,
            windows,
            compressed,
            index_keys,
            key_copy_rows,
            registry: Arc::new(Mutex::new(Validity::default())),
            publications: (0..max_lanes)
                .map(|_| std::array::from_fn(|_| None))
                .collect(),
            ends: vec![[0; 4]; max_lanes],
            key_publications: (0..max_lanes)
                .map(|_| std::array::from_fn(|_| None))
                .collect(),
            key_ends: vec![[0; 4]; max_lanes],
            stage: None,
            next_nonce: 1,
        })
    }
    pub fn allocation_bytes(&self) -> usize {
        self.max_lanes * PAYLOAD_BYTES_PER_LANE + 128 * 4
    }
    pub fn check_owner(
        &self,
        rank: usize,
        manifest: &str,
        completion: [u8; 32],
        stream: &CudaStream,
    ) -> Result<()> {
        self.k.raw.check_stream(stream)?;
        ensure!(
            rank == self.rank
                && manifest == self.manifest
                && completion == self.completion
                && stream.stream == self.k.stream.stream,
            "packed store rank/manifest/session/stream differs"
        );
        Ok(())
    }
    pub fn reset_lane(&mut self, key: LaneKey) -> Result<()> {
        self.k.raw.check_stream(&self.k.stream)?;
        self.cursor.reset_lane(key)?;
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?
            .lanes
            .insert(key.slot, (key, false));
        let result = (|| {
            let mut status = Queued::new(&self.k.dev, self.k.dev.alloc_zeros::<u32>(1)?);
            self.k.dev.memset_zeros(&mut *status)?;
            let (mut wp, mut cp, mut ep, mut lane, mut lanes) = (
                *self.windows.device_ptr(),
                *self.compressed.device_ptr(),
                *status.device_ptr(),
                key.slot as u64,
                self.max_lanes as u64,
            );
            let mut args = [
                &mut wp as *mut _ as *mut c_void,
                &mut cp as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut lane as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
            ];
            self.k.launch(
                false,
                "dsv41_chunk_store_reset",
                COMPRESSED_PER_LANE,
                &mut args,
            )?;
            let keys = offset(
                &self.index_keys,
                key.slot * KEY_ELEMENTS_PER_LANE,
                KEY_ELEMENTS_PER_LANE,
            )?;
            ensure!(
                unsafe {
                    sys::cuMemsetD8Async(keys, 0, KEY_ELEMENTS_PER_LANE * 2, self.k.stream.stream)
                } == sys::CUresult::CUDA_SUCCESS,
                "packed index-key slot reset failed"
            );
            self.k.dev.synchronize()?;
            ensure!(
                self.k.dev.dtoh_sync_copy(&status)? == [0],
                "packed physical slot reset failed"
            );
            Ok(())
        })();
        if result.is_err() {
            if let Ok(mut r) = self.registry.lock() {
                r.lanes.insert(key.slot, (key, true));
            }
        } else {
            self.ends[key.slot] = [0; 4];
            self.publications[key.slot] = std::array::from_fn(|_| None);
            self.key_ends[key.slot] = [0; 4];
            self.key_publications[key.slot] = std::array::from_fn(|_| None);
        }
        result
    }
    pub fn begin(
        &mut self,
        kind: super::batch_plan::PackedKind,
        segments: Vec<super::batch_plan::LaneSegment>,
    ) -> Result<BatchTicket> {
        {
            let r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
            ensure!(
                segments
                    .iter()
                    .all(|s| r.lanes.get(&s.key.slot) == Some(&(s.key, false))),
                "unreset/failed packed slots"
            );
        }
        self.cursor.begin(kind, segments)
    }
    pub fn begin_prefill_segments(
        &mut self,
        segments: Vec<(super::batch_plan::LaneSegment, usize)>,
    ) -> Result<BatchTicket> {
        {
            let r = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
            ensure!(
                segments
                    .iter()
                    .all(|(s, _)| r.lanes.get(&s.key.slot) == Some(&(s.key, false))),
                "unreset/failed packed slots"
            );
        }
        self.cursor.begin_prefill_segments(segments)
    }
    pub fn layer(&self, ticket: &BatchTicket, layer: usize) -> Result<BatchLayerPlan> {
        valid(&self.registry, ticket)?;
        self.cursor.layer(ticket, layer)
    }
    pub fn abort(&mut self, ticket: &BatchTicket) -> Result<()> {
        self.cursor.abort(ticket)?;
        poison(&self.registry, ticket);
        self.stage = None;
        Ok(())
    }
    pub fn committed_prefix(&self, key: LaneKey) -> Result<usize> {
        let r = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("packed registry poisoned"))?;
        ensure!(
            r.lanes.get(&key.slot) == Some(&(key, false)),
            "unreset/failed packed lane"
        );
        self.cursor.committed_prefix(key)
    }
    /// EARLY source WK publication. Inputs are already checked native index
    /// RMS/RoPE/FP4-QAT BF16 keys from the normalized UNROTATED512 latent.
    /// This must precede outer KV rotation and this layer's selection/snapshot.
    pub fn publish_index_keys_checked(
        &mut self,
        ticket: &BatchTicket,
        layer: usize,
        new_keys: Option<&CudaSlice<bf16>>,
    ) -> Result<()> {
        let result = self.publish_keys_inner(ticket, layer, new_keys);
        if result.is_err() {
            let _ = self.abort(ticket);
            let _ = self.k.dev.synchronize();
        }
        result
    }
    fn publish_keys_inner(
        &mut self,
        ticket: &BatchTicket,
        layer: usize,
        new_keys: Option<&CudaSlice<bf16>>,
    ) -> Result<()> {
        self.k.raw.check_stream(&self.k.stream)?;
        valid(&self.registry, ticket)?;
        let plan = self.cursor.layer(ticket, layer)?;
        let owner = KV_SOURCES
            .iter()
            .position(|l| *l == layer)
            .context("only4 native source layers publish index keys")?;
        ensure!(
            self.stage.is_none(),
            "index keys must publish before KV snapshot reservation"
        );
        let total = plan
            .lanes
            .iter()
            .map(|p| p.compressed.as_ref().unwrap().new_rows)
            .sum::<usize>();
        ensure!(
            new_keys.is_some() == (total > 0),
            "packed index-key new-row presence differs"
        );
        if let Some(keys) = new_keys {
            self.k.raw.check_buffer(keys)?;
            ensure!(
                keys.len() == total * KEY_D,
                "packed index keys wrong native128 geometry"
            );
        }
        for p in &plan.lanes {
            let slot = p.ticket.lane.slot;
            ensure!(
                self.key_ends[slot][owner] == p.ticket.start / p.ratio
                    && self.key_publications[slot][owner]
                        .as_ref()
                        .is_none_or(|old| old.ticket != p.ticket),
                "stale/repeated packed source key publication"
            );
        }
        let k = self.k.clone();
        macro_rules!queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();return Err(e.into());}}};}
        let status = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<u32>(1)));
        let mut consumed = 0usize;
        for p in &plan.lanes {
            let c = p.compressed.as_ref().unwrap();
            if c.new_rows == 0 {
                continue;
            }
            let mut src = queued!(offset(
                new_keys.unwrap(),
                consumed * KEY_D,
                c.new_rows * KEY_D
            ));
            let first = key_base_rows(p.ticket.lane.slot, owner)? + p.ticket.start / p.ratio;
            let mut dst = queued!(offset(&self.index_keys, first * KEY_D, c.new_rows * KEY_D));
            let (mut selections, mut ep, mut input_rows, mut output_rows, mut width) = (
                *self.key_copy_rows.device_ptr(),
                *status.device_ptr(),
                c.new_rows as u64,
                c.new_rows as u64,
                KEY_D as u64,
            );
            let mut args = [
                &mut dst as *mut _ as *mut c_void,
                &mut src as *mut _ as *mut c_void,
                &mut selections as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut input_rows as *mut _ as *mut c_void,
                &mut output_rows as *mut _ as *mut c_void,
                &mut width as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                true,
                "dsv41_batch_gather_bf16",
                c.new_rows * KEY_D,
                &mut args
            ));
            consumed += c.new_rows;
        }
        ensure!(consumed == total, "packed key source group offsets differ");
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(&status)) == [0],
            "packed key publication GPU status failed"
        );
        valid(&self.registry, ticket)?;
        for p in &plan.lanes {
            let rows = p.compressed.as_ref().unwrap().stored_rows;
            let slot = p.ticket.lane.slot;
            self.key_ends[slot][owner] = rows;
            self.key_publications[slot][owner] = Some(Publication {
                ticket: p.ticket.clone(),
                rows,
            });
        }
        Ok(())
    }
    /// One ragged, immutable index-key snapshot shared by all queries belonging
    /// to each physical lane. No per-query KV duplication or host key readback.
    pub fn index_key_snapshot_checked(
        &self,
        ticket: &BatchTicket,
        layer: usize,
    ) -> Result<Option<PackedIndexKeys>> {
        self.k.raw.check_stream(&self.k.stream)?;
        valid(&self.registry, ticket)?;
        let plan = self.cursor.layer(ticket, layer)?;
        let mut rows = 0usize;
        let mut ranges = Vec::with_capacity(plan.lanes.len());
        for p in &plan.lanes {
            let source = p
                .kv_source
                .context("uncompressed layers have no index keys")?;
            let owner = KV_SOURCES.iter().position(|l| *l == source).unwrap();
            let slot = p.ticket.lane.slot;
            let count = p.compressed.as_ref().unwrap().stored_rows;
            let publication = self.key_publications[slot][owner]
                .as_ref()
                .context("source index keys unpublished")?;
            ensure!(
                publication.ticket == p.ticket
                    && publication.rows == count
                    && self.key_ends[slot][owner] == count,
                "stale/crosslane/chunk index-key source"
            );
            ranges.push(PackedKeyRange {
                key: p.ticket.lane,
                source_layer: source,
                first_row: rows,
                rows: count,
            });
            rows += count;
        }
        ensure!(
            rows * KEY_D * 2 <= 64 * 1024 * 1024,
            "packed key snapshot bound"
        );
        if rows == 0 {
            return Ok(None);
        }
        let values = Queued::new(&self.k.dev, self.k.dev.alloc_zeros::<bf16>(rows * KEY_D)?);
        // CudaSlice retains a real typed owner even for0 completed ratio2 rows.
        for range in &ranges {
            if range.rows > 0 {
                let owner = KV_SOURCES
                    .iter()
                    .position(|l| *l == range.source_layer)
                    .unwrap();
                let source = offset(
                    &self.index_keys,
                    key_base_rows(range.key.slot, owner)? * KEY_D,
                    range.rows * KEY_D,
                )?;
                let dst = offset(&values, range.first_row * KEY_D, range.rows * KEY_D)?;
                ensure!(
                    unsafe {
                        sys::cuMemcpyDtoDAsync_v2(
                            dst,
                            source,
                            range.rows * KEY_D * 2,
                            self.k.stream.stream,
                        )
                    } == sys::CUresult::CUDA_SUCCESS,
                    "packed index-key immutable copy failed"
                );
            }
        }
        self.k.dev.synchronize()?;
        valid(&self.registry, ticket)?;
        Ok(Some(PackedIndexKeys {
            k: self.k.clone(),
            registry: self.registry.clone(),
            ticket: ticket.clone(),
            layer,
            values,
            ranges,
        }))
    }
    /// Inputs are source-checked BF16 window/compressed-QAT values, in packed
    /// segment order. Safe first control requires all compressed positions fit
    /// native top512; larger contexts are explicitly refused pending batch index.
    pub fn prepare_short_checked(
        &mut self,
        ticket: &BatchTicket,
        layer: usize,
        window: &CudaSlice<bf16>,
        new_compressed: Option<&CudaSlice<bf16>>,
    ) -> Result<PackedKvSnapshot> {
        let result = self.prepare_short_inner(ticket, layer, window, new_compressed);
        if result.is_err() {
            let _ = self.abort(ticket);
            let _ = self.k.dev.synchronize();
        }
        result
    }
    fn prepare_short_inner(
        &mut self,
        ticket: &BatchTicket,
        layer: usize,
        window: &CudaSlice<bf16>,
        new_compressed: Option<&CudaSlice<bf16>>,
    ) -> Result<PackedKvSnapshot> {
        self.k.raw.check_stream(&self.k.stream)?;
        valid(&self.registry, ticket)?;
        let plan = self.cursor.layer(ticket, layer)?;
        ensure!(
            self.stage.is_none() && plan.short_selection_exact(),
            "duplicate layer preparation or unimplemented packed learned index (>512keys)"
        );
        self.k.raw.check_buffer(window)?;
        ensure!(
            window.len() == ticket.total_rows() * D,
            "packed window rows differ"
        );
        let owns = KV_SOURCES.contains(&layer);
        let new_rows = if owns {
            plan.lanes
                .iter()
                .map(|p| p.compressed.as_ref().map_or(0, |c| c.new_rows))
                .sum()
        } else {
            0usize
        };
        ensure!(
            new_compressed.is_some() == (new_rows > 0),
            "only actual KV source receives its nonempty packed new-QAT rows"
        );
        if let Some(c) = new_compressed {
            self.k.raw.check_buffer(c)?;
            ensure!(
                c.len() == new_rows * D,
                "packed compressed row association differs"
            );
        }
        for p in &plan.lanes {
            if let Some(source) = p.kv_source {
                let owner = KV_SOURCES.iter().position(|l| *l == source).unwrap();
                let slot = p.ticket.lane.slot;
                let c = p.compressed.as_ref().unwrap();
                let keys = self.key_publications[slot][owner]
                    .as_ref()
                    .context("source key publication must precede KV snapshot")?;
                ensure!(
                    keys.ticket == p.ticket
                        && keys.rows == c.stored_rows
                        && self.key_ends[slot][owner] == c.stored_rows,
                    "packed source keys and KV chunk differ"
                );
                if source == layer {
                    ensure!(
                        self.ends[slot][owner] == p.ticket.start / p.ratio,
                        "packed source cache append position differs"
                    );
                } else {
                    let pubd = self.publications[slot][owner]
                        .as_ref()
                        .context("packed KV source unpublished")?;
                    ensure!(
                        pubd.ticket == p.ticket
                            && pubd.rows == c.stored_rows
                            && self.ends[slot][owner] == c.stored_rows,
                        "cross-lane/chunk KV source publication differs"
                    );
                }
            }
        }
        let kv_rows = plan.kv_rows();
        let window_rows = plan.window_rows();
        let max_topk = plan.maximum_topk();
        let tight_count = plan
            .logical
            .iter()
            .map(|p| p.ticket.tokens * p.logical_topk())
            .sum::<usize>();
        let bytes = (kv_rows + window_rows) * D * 2
            + (tight_count + ticket.total_rows() * max_topk) * 4
            + ticket.total_rows() * 7 * 8
            + 4;
        ensure!(
            bytes <= PACKED_SNAPSHOT_BOUND_BYTES,
            "packed immutable snapshot array bound exceeded"
        );
        let nonce = self.next_nonce;
        self.next_nonce = nonce
            .checked_add(1)
            .context("packed snapshot nonce exhausted")?;
        self.stage = Some(Stage {
            identity: ticket.identity_sha256().into(),
            layer,
            nonce,
        });
        let k = self.k.clone();
        macro_rules!queued{($e:expr)=>{match(||$e)(){Ok(v)=>v,Err(e)=>{let _=k.dev.synchronize();return Err(e.into());}}};}
        let status = Arc::new(queued!(k.dev.alloc_zeros::<u32>(1)));
        let scratch = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(window_rows * D)));
        let kv = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<bf16>(kv_rows * D)));
        let tight = Queued::new(&k.dev, queued!(k.dev.alloc_zeros::<i32>(tight_count)));
        let indices = Queued::new(
            &k.dev,
            queued!(k.dev.alloc_zeros::<i32>(ticket.total_rows() * max_topk)),
        );
        let mut descriptors = Vec::with_capacity(ticket.total_rows());
        let mut pad_meta = Vec::with_capacity(ticket.total_rows() * 3);
        let mut row_meta = Vec::with_capacity(ticket.total_rows() * 4);
        let (mut kv_at, mut win_at, mut tight_at, mut comp_at) = (0usize, 0usize, 0usize, 0usize);
        for ((segment, p), logical) in ticket.segments().iter().zip(&plan.lanes).zip(&plan.logical)
        {
            let (mut lane, mut lanes, mut native_layer, mut start, mut tokens) = (
                p.ticket.lane.slot as u64,
                self.max_lanes as u64,
                layer as u64,
                p.ticket.start as u64,
                p.ticket.tokens as u64,
            );
            let mut wp = *self.windows.device_ptr();
            let mut cp = *self.compressed.device_ptr();
            let mut ep = *status.device_ptr();
            let mut input = queued!(offset(window, segment.first_row * D, segment.rows * D));
            let mut scratch_ptr = queued!(offset(&scratch, win_at * D, p.window.snapshot_rows * D));
            let mut args = [
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut input as *mut _ as *mut c_void,
                &mut wp as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut lane as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
                &mut native_layer as *mut _ as *mut c_void,
                &mut start as *mut _ as *mut c_void,
                &mut tokens as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                false,
                "dsv41_chunk_store_window_snapshot",
                p.window.snapshot_rows * D,
                &mut args
            ));
            let mut args = [
                &mut input as *mut _ as *mut c_void,
                &mut wp as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut lane as *mut _ as *mut c_void,
                &mut lanes as *mut _ as *mut c_void,
                &mut native_layer as *mut _ as *mut c_void,
                &mut start as *mut _ as *mut c_void,
                &mut tokens as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                false,
                "dsv41_chunk_store_window_write",
                p.ticket.tokens * D,
                &mut args
            ));
            let mut kv_ptr = queued!(offset(
                &kv,
                kv_at * D,
                (p.window.snapshot_rows + p.compressed.as_ref().map_or(0, |c| c.stored_rows)) * D
            ));
            if let Some(source) = p.kv_source {
                let mut owner = KV_SOURCES.iter().position(|l| *l == source).unwrap() as u64;
                let c = p.compressed.as_ref().unwrap();
                if source == layer && c.new_rows > 0 {
                    let mut src =
                        queued!(offset(new_compressed.unwrap(), comp_at * D, c.new_rows * D));
                    let mut first = (p.ticket.start / p.ratio) as u64;
                    let mut rows = c.new_rows as u64;
                    let mut args = [
                        &mut src as *mut _ as *mut c_void,
                        &mut cp as *mut _ as *mut c_void,
                        &mut ep as *mut _ as *mut c_void,
                        &mut lane as *mut _ as *mut c_void,
                        &mut lanes as *mut _ as *mut c_void,
                        &mut owner as *mut _ as *mut c_void,
                        &mut first as *mut _ as *mut c_void,
                        &mut rows as *mut _ as *mut c_void,
                    ];
                    queued!(k.launch(
                        false,
                        "dsv41_chunk_store_compressed_append",
                        c.new_rows * D,
                        &mut args
                    ));
                    comp_at += c.new_rows;
                }
                let mut wr = p.window.snapshot_rows as u64;
                let mut cr = c.stored_rows as u64;
                let mut args = [
                    &mut kv_ptr as *mut _ as *mut c_void,
                    &mut scratch_ptr as *mut _ as *mut c_void,
                    &mut cp as *mut _ as *mut c_void,
                    &mut ep as *mut _ as *mut c_void,
                    &mut lane as *mut _ as *mut c_void,
                    &mut lanes as *mut _ as *mut c_void,
                    &mut owner as *mut _ as *mut c_void,
                    &mut wr as *mut _ as *mut c_void,
                    &mut cr as *mut _ as *mut c_void,
                ];
                queued!(k.launch(
                    false,
                    "dsv41_chunk_store_combined_snapshot",
                    (p.window.snapshot_rows + c.stored_rows) * D,
                    &mut args
                ));
            } else {
                queued!(if unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        kv_ptr,
                        scratch_ptr,
                        p.window.snapshot_rows * D * 2,
                        k.stream.stream,
                    )
                } == sys::CUresult::CUDA_SUCCESS
                {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("packed ratio0 KV copy failed"))
                });
            }
            let logical_topk = logical.logical_topk();
            let mut idx = queued!(offset(&tight, tight_at, p.ticket.tokens * logical_topk));
            let mut ratio = p.ratio as u64;
            let mut compressed = p.compressed.as_ref().map_or(0, |c| c.stored_rows) as u64;
            let mut physical_window = p.window.snapshot_rows as u64;
            let mut logical_window = logical.logical_window_columns as u64;
            let mut logical_compressed = logical.logical_compressed_columns as u64;
            let mut phase = match logical.phase {
                AttentionPhase::Prefill { .. } => 0u64,
                AttentionPhase::Decode => 1u64,
            };
            let mut args = [
                &mut idx as *mut _ as *mut c_void,
                &mut ep as *mut _ as *mut c_void,
                &mut start as *mut _ as *mut c_void,
                &mut tokens as *mut _ as *mut c_void,
                &mut ratio as *mut _ as *mut c_void,
                &mut physical_window as *mut _ as *mut c_void,
                &mut compressed as *mut _ as *mut c_void,
                &mut logical_window as *mut _ as *mut c_void,
                &mut logical_compressed as *mut _ as *mut c_void,
                &mut phase as *mut _ as *mut c_void,
            ];
            queued!(k.launch(
                true,
                "dsv41_batch_source_indices",
                p.ticket.tokens * logical_topk,
                &mut args
            ));
            let n = p.window.snapshot_rows + p.compressed.as_ref().map_or(0, |c| c.stored_rows);
            for q in 0..p.ticket.tokens {
                pad_meta.extend([
                    (tight_at + q * logical_topk) as u64,
                    logical_topk as u64,
                    n as u64,
                ]);
                row_meta.extend([
                    (kv_at * D) as u64,
                    n as u64,
                    logical_topk as u64,
                    p.ticket.lane.slot as u64,
                ]);
                descriptors.push(PackedQueryRow {
                    key: p.ticket.lane,
                    native_position: p.ticket.start + q,
                    query_row: segment.first_row + q,
                    kv_first_row: kv_at,
                    kv_rows: n,
                    actual_topk: logical_topk,
                });
            }
            kv_at += n;
            win_at += p.window.snapshot_rows;
            tight_at += p.ticket.tokens * logical_topk;
        }
        ensure!(
            comp_at == new_rows
                && kv_at == kv_rows
                && win_at == window_rows
                && tight_at == tight_count,
            "packed range accounting differs"
        );
        let pad_metadata = Queued::new(&k.dev, queued!(k.dev.htod_sync_copy(&pad_meta)));
        let metadata = Queued::new(&k.dev, queued!(k.dev.htod_sync_copy(&row_meta)));
        let (mut op, mut ip, mut mp, mut ep, mut rows, mut topk, mut count) = (
            *indices.device_ptr(),
            *tight.device_ptr(),
            *pad_metadata.device_ptr(),
            *status.device_ptr(),
            ticket.total_rows() as u64,
            max_topk as u64,
            tight_count as u64,
        );
        let mut args = [
            &mut op as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
            &mut rows as *mut _ as *mut c_void,
            &mut topk as *mut _ as *mut c_void,
            &mut count as *mut _ as *mut c_void,
        ];
        queued!(k.launch(
            true,
            "dsv41_batch_pad_indices",
            ticket.total_rows() * max_topk,
            &mut args
        ));
        queued!(k.dev.synchronize());
        ensure!(
            queued!(k.dev.dtoh_sync_copy(status.as_ref())) == [0],
            "packed cache/layout GPU status failed"
        );
        valid(&self.registry, ticket)?;
        if owns {
            let owner = KV_SOURCES.iter().position(|l| *l == layer).unwrap();
            for p in &plan.lanes {
                let rows = p.compressed.as_ref().unwrap().stored_rows;
                self.ends[p.ticket.lane.slot][owner] = rows;
                self.publications[p.ticket.lane.slot][owner] = Some(Publication {
                    ticket: p.ticket.clone(),
                    rows,
                });
            }
        }
        Ok(PackedKvSnapshot {
            k,
            registry: self.registry.clone(),
            plan,
            instance: self.instance,
            completion: self.completion,
            nonce,
            rank: self.rank,
            manifest: self.manifest.clone(),
            max_topk,
            kv,
            indices,
            metadata,
            status,
            descriptors,
            array_bytes: bytes,
        })
    }
    /// Outer packed block checks all HC/attention/MoE lanes BEFORE this commit.
    pub fn commit_layer(
        &mut self,
        ticket: &BatchTicket,
        snapshot: &PackedKvSnapshot,
    ) -> Result<()> {
        let stage = self.stage.as_ref().context("packed layer not reserved")?;
        ensure!(
            snapshot.instance == self.instance
                && snapshot.completion == self.completion
                && snapshot.plan.ticket == *ticket
                && stage.identity == ticket.identity_sha256()
                && stage.layer == snapshot.plan.layer
                && stage.nonce == snapshot.nonce,
            "wrong store/ticket/reservation snapshot"
        );
        snapshot.check()?;
        self.cursor.commit_layer(&snapshot.plan)?;
        self.stage = None;
        Ok(())
    }
    pub fn commit(&mut self, ticket: &BatchTicket) -> Result<()> {
        ensure!(self.stage.is_none(), "unchecked prepared packed layer");
        valid(&self.registry, ticket)?;
        self.cursor.commit(ticket)
    }
}

/// One immutable concatenated KV owner for the WHOLE packed step. Row metadata
/// selects exact per-lane KV base/length/actualtopk without duplicating KV per
/// query; later ragged attention uses actualtopk to preserve C1 reduction order.
pub struct PackedKvSnapshot {
    k: Arc<BatchStoreKernels>,
    registry: Arc<Mutex<Validity>>,
    plan: BatchLayerPlan,
    instance: [u8; 32],
    completion: [u8; 32],
    nonce: u64,
    rank: usize,
    manifest: String,
    max_topk: usize,
    kv: Queued<CudaSlice<bf16>>,
    indices: Queued<CudaSlice<i32>>,
    metadata: Queued<CudaSlice<u64>>,
    status: Arc<CudaSlice<u32>>,
    descriptors: Vec<PackedQueryRow>,
    array_bytes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct PackedKeyRange {
    pub key: LaneKey,
    pub source_layer: usize,
    pub first_row: usize,
    pub rows: usize,
}
pub struct PackedIndexKeys {
    k: Arc<BatchStoreKernels>,
    registry: Arc<Mutex<Validity>>,
    ticket: BatchTicket,
    layer: usize,
    values: Queued<CudaSlice<bf16>>,
    ranges: Vec<PackedKeyRange>,
}
impl PackedIndexKeys {
    pub fn check(&self) -> Result<()> {
        self.k.raw.check_stream(&self.k.stream)?;
        valid(&self.registry, &self.ticket)?;
        self.k.dev.synchronize()?;
        valid(&self.registry, &self.ticket)
    }
    pub fn ticket(&self) -> &BatchTicket {
        &self.ticket
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn ranges(&self) -> &[PackedKeyRange] {
        &self.ranges
    }
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<bf16> {
        &self.values
    }
    pub fn download_checked(&self) -> Result<Vec<bf16>> {
        self.check()?;
        Ok(self.k.dev.dtoh_sync_copy(&self.values)?)
    }
}
impl PackedKvSnapshot {
    pub fn check_owner(
        &self,
        rank: usize,
        manifest: &str,
        completion: [u8; 32],
        stream: &CudaStream,
    ) -> Result<()> {
        ensure!(
            self.rank == rank
                && self.manifest == manifest
                && self.completion == completion
                && self.k.stream.stream == stream.stream,
            "packed snapshot rank/manifest/session/stream differs"
        );
        valid(&self.registry, &self.plan.ticket)
    }
    pub(crate) fn poison_generation(&self) {
        poison(&self.registry, &self.plan.ticket);
    }
    pub fn check(&self) -> Result<()> {
        self.k.raw.check_stream(&self.k.stream)?;
        valid(&self.registry, &self.plan.ticket)?;
        self.k.dev.synchronize()?;
        let code = self.k.dev.dtoh_sync_copy(self.status.as_ref())?[0];
        if code != 0 {
            poison(&self.registry, &self.plan.ticket);
            anyhow::bail!("packed snapshot GPU status{code}");
        }
        valid(&self.registry, &self.plan.ticket)
    }
    pub fn plan(&self) -> &BatchLayerPlan {
        &self.plan
    }
    pub fn descriptors(&self) -> &[PackedQueryRow] {
        &self.descriptors
    }
    pub fn max_topk(&self) -> usize {
        self.max_topk
    }
    pub fn array_bytes(&self) -> usize {
        self.array_bytes
    }
    pub unsafe fn device_kv_unchecked(&self) -> &CudaSlice<bf16> {
        &self.kv
    }
    pub unsafe fn device_indices_unchecked(&self) -> &CudaSlice<i32> {
        &self.indices
    }
    pub unsafe fn device_row_metadata_unchecked(&self) -> &CudaSlice<u64> {
        &self.metadata
    }
    /// Untimed oracle evidence only; production composition performs no payload
    /// readback. Raw bit preservation is checked before subsequent qualification.
    pub fn download_checked(&self) -> Result<(Vec<bf16>, Vec<i32>, Vec<u64>)> {
        self.check()?;
        Ok((
            self.k.dev.dtoh_sync_copy(&self.kv)?,
            self.k.dev.dtoh_sync_copy(&self.indices)?,
            self.k.dev.dtoh_sync_copy(&self.metadata)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_lane_and_ratio_aware_native_key_ranges_never_alias() {
        let mut last = 0;
        for slot in 0..16 {
            for owner in 0..4 {
                let first = key_base_rows(slot, owner).unwrap();
                let rows = if owner < 3 {
                    MAX_CONTEXT / 2
                } else {
                    MAX_CONTEXT
                };
                assert_eq!(first, last);
                last = first + rows;
            }
        }
        assert_eq!(last, 16 * KEY_ROWS_PER_LANE);
        assert!(key_base_rows(16, 0).is_err());
        assert!(key_base_rows(0, 4).is_err());
        assert_eq!(PAYLOAD_BYTES_PER_LANE, 42 * 1024 * 1024);
        assert_eq!(16 * PAYLOAD_BYTES_PER_LANE, 672 * 1024 * 1024);
    }
    #[test]
    fn worst_ragged_snapshot_counts_kv_once_per_lane_not_once_per_query() {
        let lanes = 16;
        let queries = 128;
        let windows = lanes * 127 + queries;
        let kv = windows + lanes * 8192;
        let topk = 640;
        let tight = queries * topk;
        let all = (kv + windows) * 512 * 2 + (tight + queries * topk) * 4 + queries * 7 * 8 + 4;
        assert_eq!(kv, 133232);
        assert!(all < PACKED_SNAPSHOT_BOUND_BYTES);
        assert!(queries * (8192 + 255) * 512 * 2 > PACKED_SNAPSHOT_BOUND_BYTES);
        assert_eq!(lanes * 8192 * KEY_D * 2, 32 * 1024 * 1024);
    }
}
