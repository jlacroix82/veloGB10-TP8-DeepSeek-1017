//! PRIVATE one-launch sparse attention over packed independent/ragged rows.
//! Per-row native logical widths preserve source64-key tile membership; common
//! index row stride is only an address stride. No full-model authority here.
use super::{
    gpu_batch_store::{PackedKvSnapshot, PackedQueryRow},
    gpu_module::Module,
};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use std::{ffi::c_void, sync::Arc};
use uuid::Uuid;
const D: usize = 512;
const H: usize = 8;
const SHARED: usize = 88320;
pub const PTX: &str = include_str!("../ptx/dsv41_batch_attention.ptx");
pub struct PackedAttentionKernels {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    module: Module,
}
impl PackedAttentionKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(!boot.is_nil(), "packed attention boot required");
        let module = Module::load(
            dev,
            &stream,
            PTX,
            &["dsv41_packed_sparse_attention"],
            "dsv41_batch_attention_identity",
            u64::from_str_radix(
                env!("DSV41_BATCH_ATTENTION_SOURCE_ID").trim_start_matches("0x"),
                16,
            )?,
            1,
        )?;
        module.set_dynamic_smem("dsv41_packed_sparse_attention", SHARED)?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            stream,
            boot,
            module,
        }))
    }
    pub fn enqueue<'a>(
        self: &Arc<Self>,
        rank: usize,
        manifest: &str,
        completion: [u8; 32],
        snapshot: &'a PackedKvSnapshot,
        q: &'a CudaSlice<bf16>,
        sink: &'a CudaSlice<f32>,
    ) -> Result<PendingPackedSparse<'a>> {
        let t = &snapshot.plan().ticket;
        let rows = t.total_rows();
        ensure!(
            t.boot() == self.boot
                && (1..=128).contains(&rows)
                && snapshot.descriptors().len() == rows,
            "packed query boot/row associations differ"
        );
        snapshot.check_owner(rank, manifest, completion, &self.stream)?;
        snapshot.check()?;
        self.module.check_stream(&self.stream)?;
        self.module.check_buffer(q)?;
        self.module.check_buffer(sink)?;
        ensure!(
            q.len() == rows * H * D && sink.len() == H,
            "nativeTP8 packed Q/sink geometry differs"
        );
        let kv = unsafe { snapshot.device_kv_unchecked() };
        let indices = unsafe { snapshot.device_indices_unchecked() };
        let metadata = unsafe { snapshot.device_row_metadata_unchecked() };
        for (actual, expected) in [
            (q.len(), rows * H * D),
            (indices.len(), rows * snapshot.max_topk()),
            (metadata.len(), rows * 4),
        ] {
            ensure!(actual == expected, "packed sparse array extent differs");
        }
        ensure!(
            kv.len() % D == 0 && kv.len() / D == snapshot.plan().kv_rows(),
            "packed KV owner extent differs"
        );
        self.module.check_buffer(kv)?;
        self.module.check_buffer(indices)?;
        self.module.check_buffer(metadata)?;
        let output = self.dev.alloc_zeros::<bf16>(rows * H * D)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        let mut pending = PendingPackedSparse {
            k: self.clone(),
            snapshot,
            q,
            sink,
            output: Some(output),
            status,
            rank,
            manifest: manifest.into(),
            completion,
            checked: false,
        };
        let (
            mut qp,
            mut kp,
            mut op,
            mut sp,
            mut ip,
            mut mp,
            mut stride,
            mut total,
            mut heads,
            mut scale,
            mut ep,
        ) = (
            *q.device_ptr(),
            *kv.device_ptr(),
            *pending.output.as_ref().unwrap().device_ptr(),
            *sink.device_ptr(),
            *indices.device_ptr(),
            *metadata.device_ptr(),
            snapshot.max_topk() as u64,
            (kv.len() / D) as u64,
            H as i32,
            (D as f32).sqrt().recip(),
            *pending.status.device_ptr(),
        );
        let mut args = [
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void,
            &mut stride as *mut _ as *mut c_void,
            &mut total as *mut _ as *mut c_void,
            &mut heads as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module.launch(
                "dsv41_packed_sparse_attention",
                &self.stream,
                (rows as u32, 1, 1),
                (256, 1, 1),
                SHARED as u32,
                &mut args,
            )
        }?;
        // ONE packed launch, independent of logical lane count. Primitive timing
        // and actual model batching still require the guarded GPU oracle.
        Ok(pending)
    }
}
pub struct PendingPackedSparse<'a> {
    k: Arc<PackedAttentionKernels>,
    snapshot: &'a PackedKvSnapshot,
    q: &'a CudaSlice<bf16>,
    sink: &'a CudaSlice<f32>,
    output: Option<CudaSlice<bf16>>,
    status: CudaSlice<u32>,
    rank: usize,
    manifest: String,
    completion: [u8; 32],
    checked: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct PackedSparseIdentity {
    pub rank: usize,
    pub manifest_sha256: String,
    pub completion_owner: [u8; 32],
    pub batch_ticket_sha256: String,
    pub layer: usize,
    pub rows: Vec<PackedQueryRow>,
    pub shared_index_stride: usize,
    pub per_row_logical_topk_preserved: bool,
    pub one_packed_kernel_launch: bool,
    pub all_checks_passed: bool,
    pub full_model_or_quality_qualified: bool,
}
pub struct PackedSparseOutput {
    output: CudaSlice<bf16>,
    identity: PackedSparseIdentity,
}
impl PackedSparseOutput {
    pub fn values(&self) -> &CudaSlice<bf16> {
        &self.output
    }
    pub fn identity(&self) -> &PackedSparseIdentity {
        &self.identity
    }
}
impl<'a> PendingPackedSparse<'a> {
    pub fn finish(mut self) -> Result<PackedSparseOutput> {
        self.snapshot
            .check_owner(self.rank, &self.manifest, self.completion, &self.k.stream)?;
        self.k.dev.synchronize()?;
        ensure!(
            self.k.dev.dtoh_sync_copy(&self.status)? == [0],
            "packed attention GPU sticky status failed"
        );
        self.snapshot.check()?;
        // Borrowed Q/sink and immutable snapshot have remained live throughout
        // the complete launch/status fence; no per-head payload readback occurs.
        let _ = (self.q, self.sink);
        let identity = PackedSparseIdentity {
            rank: self.rank,
            manifest_sha256: self.manifest.clone(),
            completion_owner: self.completion,
            batch_ticket_sha256: self.snapshot.plan().ticket.identity_sha256().into(),
            layer: self.snapshot.plan().layer,
            rows: self.snapshot.descriptors().to_vec(),
            shared_index_stride: self.snapshot.max_topk(),
            per_row_logical_topk_preserved: true,
            one_packed_kernel_launch: true,
            all_checks_passed: true,
            full_model_or_quality_qualified: false,
        };
        self.checked = true;
        Ok(PackedSparseOutput {
            output: self.output.take().unwrap(),
            identity,
        })
    }
}
impl Drop for PendingPackedSparse<'_> {
    fn drop(&mut self) {
        if !self.checked {
            self.snapshot.poison_generation();
        }
        let _ = self.k.dev.synchronize();
    }
}
