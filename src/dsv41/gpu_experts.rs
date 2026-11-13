//! Isolated native1017 grouped routed FP4 experts. No shared expert, all-reduce,
//! checkpoint mutation, graphs or full-model/serving qualification is implied.
//! Workspace allocation and pointer-table upload happen before enqueue. Runtime
//! routing stays on GPU; each enqueue launches a fixed bounded kernel sequence.
use super::{
    gpu_gemm::GpuGemm,
    gpu_module::Module,
    gpu_moe::MoeKernels,
    gpu_numerics::GpuNumerics,
    gpu_weights::{LoadScope, LoadedWeights},
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, ffi::c_void, sync::Arc};
pub const PTX: &str = include_str!("../ptx/dsv41_experts.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_experts.cu");
pub const LOCAL_EXPERTS: usize = 48;
pub const MAX_ROWS: usize = 128;
pub const ABI: u64 = 1;
pub const MAX_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;
/// Source-order CPU reference, independent of the GPU execution/owner API.
pub fn combine_reduced_shared_reference(reduced: &[f32], shared: &[bf16]) -> Result<Vec<bf16>> {
    ensure!(
        !reduced.is_empty() && reduced.len() <= MAX_ROWS * 5120 && reduced.len() == shared.len(),
        "reduced/shared CPU combination extent"
    );
    reduced
        .iter()
        .zip(shared)
        .map(|(r, s)| {
            ensure!(
                r.is_finite() && s.is_finite(),
                "nonfinite reduced/shared CPU input"
            );
            let sum = *r + s.to_f32();
            let out = bf16::from_f32(sum);
            ensure!(
                sum.is_finite() && out.is_finite(),
                "reduced/shared CPU output overflow"
            );
            Ok(out)
        })
        .collect()
}
const NAMES: &[&str] = &[
    "dsv41_experts_gemm_dependency",
    "dsv41_experts_reset",
    "dsv41_experts_validate_routes",
    "dsv41_experts_prepare",
    "dsv41_experts_input_quant",
    "dsv41_experts_w13",
    "dsv41_experts_silu_quant",
    "dsv41_experts_w2",
    "dsv41_experts_sum",
    "dsv41_experts_combine",
];
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertShape {
    dim: usize,
    inter: usize,
}
impl ExpertShape {
    pub fn native1017() -> Self {
        Self {
            dim: 5120,
            inter: 2304,
        }
    }
    /// Explicit component-oracle geometry; this does not qualify other models.
    pub fn oracle(dim: usize, inter: usize) -> Result<Self> {
        let shape = Self { dim, inter };
        shape.validate()?;
        Ok(shape)
    }
    pub fn dim(self) -> usize {
        self.dim
    }
    pub fn inter(self) -> usize {
        self.inter
    }
    fn validate(self) -> Result<()> {
        ensure!(
            self.dim > 0
                && self.dim <= 5120
                && self.inter > 0
                && self.inter <= 2304
                && self.dim % 32 == 0
                && self.inter % 32 == 0,
            "bounded expert geometry, positive group32 dimensions required"
        );
        Ok(())
    }
    fn lengths(self) -> [usize; 6] {
        [
            self.inter * self.dim / 2,
            self.inter * (self.dim / 32),
            self.inter * self.dim / 2,
            self.inter * (self.dim / 32),
            self.dim * self.inter / 2,
            self.dim * (self.inter / 32),
        ]
    }
    pub fn workspace_bytes(self, rows: usize) -> Result<usize> {
        self.validate()?;
        ensure!((1..=MAX_ROWS).contains(&rows), "expert rows1..128 required");
        // input E4/scales, compact BF16 SwiGLU/E4/scales, BF16 w2,
        // F32 output, maps+inverse+route weights, counts and sticky status.
        let bytes = rows * self.dim
            + rows * self.dim / 32
            + LOCAL_EXPERTS * rows * self.inter * 3
            + LOCAL_EXPERTS * rows * self.inter / 32
            + LOCAL_EXPERTS * rows * self.dim * 2
            + rows * self.dim * 4
            + LOCAL_EXPERTS * rows * 12
            + LOCAL_EXPERTS * 4
            + 4;
        ensure!(
            bytes <= MAX_WORKSPACE_BYTES,
            "bounded expert workspace exceeded"
        );
        Ok(bytes)
    }
}
/// Each buffer owns exact native bytes; low-even/high-odd FP4 packing and
/// per-output/per-K32 UE8 scales are preserved without requantization.
pub struct OwnedExpert {
    pub global_id: usize,
    pub w1: CudaSlice<u8>,
    pub s1: CudaSlice<u8>,
    pub w3: CudaSlice<u8>,
    pub s3: CudaSlice<u8>,
    pub w2: CudaSlice<u8>,
    pub s2: CudaSlice<u8>,
}
impl OwnedExpert {
    fn buffers(&self) -> [&CudaSlice<u8>; 6] {
        [&self.w1, &self.s1, &self.w3, &self.s3, &self.w2, &self.s2]
    }
}
enum BankOwner {
    Loaded(Arc<LoadedWeights>),
    Oracle(Vec<OwnedExpert>),
}
pub struct ExpertBank {
    dev: Arc<CudaDevice>,
    rank: usize,
    shape: ExpertShape,
    pointers: CudaSlice<u64>,
    present: Vec<usize>,
    owner: BankOwner,
    manifest_sha256: Option<String>,
}
impl Drop for ExpertBank {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
fn validate_ids(rank: usize, ids: &[usize], complete: bool) -> Result<()> {
    ensure!(
        rank < 8 && ids.len() <= LOCAL_EXPERTS,
        "expert bank rank/cardinality"
    );
    let mut seen = BTreeSet::new();
    for &id in ids {
        ensure!(
            id / LOCAL_EXPERTS == rank && seen.insert(id),
            "duplicate or foreign expert bank entry"
        );
    }
    ensure!(
        !complete || ids.len() == LOCAL_EXPERTS,
        "native bank must own all48 experts"
    );
    Ok(())
}
fn checked_pointers(
    dev: &Arc<CudaDevice>,
    shape: ExpertShape,
    entries: &[(usize, [&CudaSlice<u8>; 6])],
) -> Result<Vec<u64>> {
    let mut table = vec![0_u64; LOCAL_EXPERTS * 6];
    let mut ranges = Vec::new();
    for (global, buffers) in entries {
        for (i, (buffer, expected)) in buffers.iter().zip(shape.lengths()).enumerate() {
            ensure!(
                Arc::ptr_eq(&buffer.device(), dev) && buffer.len() == expected,
                "expert native dtype extent/device mismatch"
            );
            let start = *buffer.device_ptr();
            let end = start
                .checked_add(expected as u64)
                .context("expert native pointer overflow")?;
            ensure!(start != 0, "null native expert allocation");
            ranges.push((start, end));
            table[(global % LOCAL_EXPERTS) * 6 + i] = start;
        }
    }
    ranges.sort_unstable();
    ensure!(
        ranges.windows(2).all(|w| w[0].1 <= w[1].0),
        "native expert owners alias/overlap"
    );
    Ok(table)
}
impl ExpertBank {
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn shape(&self) -> ExpertShape {
        self.shape
    }
    pub fn complete_native_bank(&self) -> bool {
        self.manifest_sha256.is_some() && self.present.len() == LOCAL_EXPERTS
    }
    pub fn manifest_sha256(&self) -> Option<&str> {
        self.manifest_sha256.as_deref()
    }
    pub fn present_experts(&self) -> &[usize] {
        &self.present
    }
    /// Oracle-only borrowed native inputs for the separately qualified,
    /// unfused GEMM reference. Runtime dispatch never iterates this API.
    pub fn oracle_buffers(&self, global_id: usize) -> Result<[&CudaSlice<u8>; 6]> {
        match &self.owner {
            BankOwner::Oracle(entries) => entries
                .iter()
                .find(|e| e.global_id == global_id)
                .map(OwnedExpert::buffers)
                .context("missing oracle native expert"),
            BankOwner::Loaded(_) => {
                anyhow::bail!("unfused oracle access is not a production dispatch path")
            }
        }
    }
    pub fn from_loaded(
        dev: &Arc<CudaDevice>,
        manifest: &RankManifest,
        layer: usize,
        owner: Arc<LoadedWeights>,
    ) -> Result<Self> {
        manifest.verify_identity()?;
        ensure!(
            layer < 40
                && owner.scope() == &LoadScope::Layer(layer)
                && owner.manifest_sha256() == manifest.manifest_sha256,
            "expert bank source scope/manifest differs"
        );
        let shape = ExpertShape::native1017();
        let ids: Vec<_> =
            (manifest.rank * LOCAL_EXPERTS..(manifest.rank + 1) * LOCAL_EXPERTS).collect();
        validate_ids(manifest.rank, &ids, true)?;
        let mut entries = Vec::new();
        for &id in &ids {
            let prefix = format!("layers.{layer}.ffn.experts.{id}");
            let names = [
                format!("{prefix}.w1.weight"),
                format!("{prefix}.w1.scale"),
                format!("{prefix}.w3.weight"),
                format!("{prefix}.w3.scale"),
                format!("{prefix}.w2.weight"),
                format!("{prefix}.w2.scale"),
            ];
            let buffers = [
                owner.native_u8(&names[0])?,
                owner.native_u8(&names[1])?,
                owner.native_u8(&names[2])?,
                owner.native_u8(&names[3])?,
                owner.native_u8(&names[4])?,
                owner.native_u8(&names[5])?,
            ];
            entries.push((id, buffers));
        }
        let table = checked_pointers(dev, shape, &entries)?;
        drop(entries);
        Ok(Self {
            dev: dev.clone(),
            rank: manifest.rank,
            shape,
            pointers: dev.htod_sync_copy(&table)?,
            present: ids,
            owner: BankOwner::Loaded(owner),
            manifest_sha256: Some(manifest.manifest_sha256.clone()),
        })
    }
    /// Explicit model-free sparse fixture bank. Unselected missing experts are
    /// skipped; selecting one sets BAD_BANK on GPU. Never a full-model bank.
    pub fn oracle_sparse(
        dev: &Arc<CudaDevice>,
        rank: usize,
        shape: ExpertShape,
        owner: Vec<OwnedExpert>,
    ) -> Result<Self> {
        shape.validate()?;
        let ids: Vec<_> = owner.iter().map(|e| e.global_id).collect();
        validate_ids(rank, &ids, false)?;
        let entries: Vec<_> = owner.iter().map(|e| (e.global_id, e.buffers())).collect();
        let table = checked_pointers(dev, shape, &entries)?;
        drop(entries);
        Ok(Self {
            dev: dev.clone(),
            rank,
            shape,
            pointers: dev.htod_sync_copy(&table)?,
            present: ids,
            owner: BankOwner::Oracle(owner),
            manifest_sha256: None,
        })
    }
}
pub struct ExpertWorkspace {
    dev: Arc<CudaDevice>,
    shape: ExpertShape,
    rows: usize,
    maps: CudaSlice<u32>,
    inverse: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    selected: CudaSlice<f32>,
    input_values: CudaSlice<u8>,
    input_scales: CudaSlice<u8>,
    silu: CudaSlice<bf16>,
    silu_values: CudaSlice<u8>,
    silu_scales: CudaSlice<u8>,
    partial: CudaSlice<bf16>,
    output: CudaSlice<f32>,
    status: CudaSlice<u32>,
}
impl Drop for ExpertWorkspace {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
impl ExpertWorkspace {
    pub fn new(dev: &Arc<CudaDevice>, shape: ExpertShape, rows: usize) -> Result<Self> {
        shape.workspace_bytes(rows)?;
        let compact = LOCAL_EXPERTS * rows;
        Ok(Self {
            dev: dev.clone(),
            shape,
            rows,
            maps: dev.alloc_zeros(compact)?,
            inverse: dev.alloc_zeros(compact)?,
            counts: dev.alloc_zeros(LOCAL_EXPERTS)?,
            selected: dev.alloc_zeros(compact)?,
            input_values: dev.alloc_zeros(rows * shape.dim)?,
            input_scales: dev.alloc_zeros(rows * shape.dim / 32)?,
            silu: dev.alloc_zeros(compact * shape.inter)?,
            silu_values: dev.alloc_zeros(compact * shape.inter)?,
            silu_scales: dev.alloc_zeros(compact * shape.inter / 32)?,
            partial: dev.alloc_zeros(compact * shape.dim)?,
            output: dev.alloc_zeros(rows * shape.dim)?,
            status: dev.alloc_zeros(1)?,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn shape(&self) -> ExpertShape {
        self.shape
    }
}
#[derive(Debug, Serialize)]
pub struct ExpertDependencies {
    pub experts_source_id: String,
    pub experts_ptx_sha256: String,
    pub experts_cuda_sha256: String,
    pub gemm_source_id: String,
    pub gemm_ptx_sha256: String,
    pub gemm_cuda_sha256: String,
    pub numerics_source_id: String,
    pub numerics_ptx_sha256: String,
    pub numerics_cuda_sha256: String,
    pub moe_source_id: String,
    pub moe_ptx_sha256: String,
    pub moe_cuda_sha256: String,
}
pub struct GpuExperts {
    dev: Arc<CudaDevice>,
    module: Module,
    source_id: u64,
    _gemm: GpuGemm,
    _numerics: GpuNumerics,
    _moe: MoeKernels,
}
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
impl GpuExperts {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let source_id =
            u64::from_str_radix(env!("DSV41_EXPERTS_SOURCE_ID").trim_start_matches("0x"), 16)?;
        let gemm = GpuGemm::load(dev, stream)?;
        let numerics = GpuNumerics::load(dev, stream)?;
        let moe = MoeKernels::load(dev, stream)?;
        let module = Module::load(
            dev,
            stream,
            PTX,
            NAMES,
            "dsv41_experts_identity",
            source_id,
            ABI,
        )?;
        let me = Self {
            dev: dev.clone(),
            module,
            source_id,
            _gemm: gemm,
            _numerics: numerics,
            _moe: moe,
        };
        let check = dev.alloc_zeros::<u64>(1)?;
        let mut args = [ptr(&check)];
        me.launch(
            "dsv41_experts_gemm_dependency",
            stream,
            (1, 1, 1),
            (1, 1, 1),
            &mut args,
        )?;
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&check)? == [me._gemm.source_id()],
            "grouped experts compiled against another GEMM source ID"
        );
        Ok(me)
    }
    pub fn dependencies(&self) -> ExpertDependencies {
        fn sha(x: &[u8]) -> String {
            format!("{:x}", Sha256::digest(x))
        }
        ExpertDependencies {
            experts_source_id: format!("{:016x}", self.source_id),
            experts_ptx_sha256: sha(PTX.as_bytes()),
            experts_cuda_sha256: sha(CUDA_SOURCE),
            gemm_source_id: format!("{:016x}", self._gemm.source_id()),
            gemm_ptx_sha256: self._gemm.ptx_sha256(),
            gemm_cuda_sha256: sha(super::gpu_gemm::CUDA_SOURCE),
            numerics_source_id: format!("{:016x}", self._numerics.source_id()),
            numerics_ptx_sha256: self._numerics.ptx_sha256(),
            numerics_cuda_sha256: sha(super::gpu_numerics::CUDA_SOURCE),
            moe_source_id: env!("DSV41_MOE_SOURCE_ID").into(),
            moe_ptx_sha256: sha(super::gpu_moe::PTX.as_bytes()),
            moe_cuda_sha256: sha(include_bytes!("../../kernels/dsv41_moe.cu")),
        }
    }
    fn launch(
        &self,
        name: &str,
        stream: &CudaStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [u64],
    ) -> Result<()> {
        let mut params: Vec<*mut c_void> = args
            .iter_mut()
            .map(|p| p as *mut u64 as *mut c_void)
            .collect();
        unsafe {
            self.module
                .launch(name, stream, grid, block, 0, &mut params)
        }
    }
    /// One F32 add followed by one BF16 cast, matching MoE.forward after the
    /// routed partial has completed its independent network reduction. This
    /// helper neither runs the shared expert nor performs/attests a collective.
    pub fn combine_reduced_shared<'a>(
        &'a self,
        stream: &'a CudaStream,
        reduced: &'a CudaSlice<f32>,
        shared: &'a CudaSlice<bf16>,
        output: &'a mut CudaSlice<bf16>,
        status: &'a mut CudaSlice<u32>,
    ) -> Result<PendingCombined<'a>> {
        self.module.check_stream(stream)?;
        self.module.check_buffer(reduced)?;
        self.module.check_buffer(shared)?;
        self.module.check_buffer(output)?;
        self.module.check_buffer(status)?;
        ensure!(
            !reduced.is_empty()
                && reduced.len() <= MAX_ROWS * 5120
                && reduced.len() == shared.len()
                && reduced.len() == output.len()
                && status.len() == 1,
            "reduced/shared combination shape outside bounded component domain"
        );
        let result = (|| {
            self.launch(
                "dsv41_experts_reset",
                stream,
                (1, 1, 1),
                (1, 1, 1),
                &mut [ptr(status)],
            )?;
            self.launch(
                "dsv41_experts_combine",
                stream,
                (reduced.len().div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    ptr(output),
                    ptr(status),
                    ptr(reduced),
                    ptr(shared),
                    reduced.len() as u64,
                ],
            )
        })();
        if let Err(error) = result {
            let _ = self.dev.synchronize();
            return Err(error);
        }
        Ok(PendingCombined {
            component: self,
            _stream: stream,
            _reduced: reduced,
            _shared: shared,
            output,
            status,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue<'a>(
        &'a self,
        stream: &'a CudaStream,
        bank: &'a ExpertBank,
        workspace: &'a mut ExpertWorkspace,
        input: &'a CudaSlice<bf16>,
        route_ids: &'a CudaSlice<u32>,
        route_weights: &'a CudaSlice<f32>,
        route_status: &'a CudaSlice<u32>,
    ) -> Result<PendingExperts<'a>> {
        self.module.check_stream(stream)?;
        ensure!(
            Arc::ptr_eq(&self.dev, &bank.dev)
                && Arc::ptr_eq(&self.dev, &workspace.dev)
                && bank.shape == workspace.shape,
            "expert bank/workspace device or geometry differs"
        );
        for valid in [
            Arc::ptr_eq(&input.device(), &self.dev),
            Arc::ptr_eq(&route_ids.device(), &self.dev),
            Arc::ptr_eq(&route_weights.device(), &self.dev),
            Arc::ptr_eq(&route_status.device(), &self.dev),
        ] {
            ensure!(valid, "expert source context mismatch");
        }
        let rows = workspace.rows;
        let shape = workspace.shape;
        shape.workspace_bytes(rows)?;
        ensure!(
            input.len() == rows * shape.dim
                && route_ids.len() == rows * 6
                && route_weights.len() == rows * 6
                && route_status.len() == 1,
            "expert input/routing extents changed"
        );
        let w = &*workspace;
        let r = rows as u64;
        let d = shape.dim as u64;
        let h = shape.inter as u64;
        let rank = bank.rank as u64;
        let groups = rows * shape.dim / 32;
        let silu_groups = LOCAL_EXPERTS * rows * shape.inter / 32;
        let result = (|| {
            self.launch(
                "dsv41_experts_reset",
                stream,
                (1, 1, 1),
                (1, 1, 1),
                &mut [ptr(&w.status)],
            )?;
            self.launch(
                "dsv41_experts_validate_routes",
                stream,
                (rows.div_ceil(128) as u32, 1, 1),
                (128, 1, 1),
                &mut [
                    ptr(route_ids),
                    ptr(route_weights),
                    ptr(route_status),
                    ptr(&w.status),
                    r,
                ],
            )?;
            self.launch(
                "dsv41_experts_prepare",
                stream,
                (LOCAL_EXPERTS as u32, 1, 1),
                (1, 1, 1),
                &mut [
                    ptr(&w.maps),
                    ptr(&w.inverse),
                    ptr(&w.counts),
                    ptr(&w.selected),
                    ptr(route_ids),
                    ptr(route_weights),
                    ptr(&bank.pointers),
                    ptr(&w.status),
                    r,
                    rank,
                ],
            )?;
            self.launch(
                "dsv41_experts_input_quant",
                stream,
                (groups.div_ceil(4) as u32, 1, 1),
                (128, 1, 1),
                &mut [
                    ptr(&w.input_values),
                    ptr(&w.input_scales),
                    ptr(&w.status),
                    ptr(input),
                    groups as u64,
                ],
            )?;
            self.launch(
                "dsv41_experts_w13",
                stream,
                (
                    shape.inter.div_ceil(128) as u32,
                    rows.div_ceil(16) as u32,
                    LOCAL_EXPERTS as u32,
                ),
                (256, 1, 1),
                &mut [
                    ptr(&w.silu),
                    ptr(&w.status),
                    ptr(&w.input_values),
                    ptr(&w.input_scales),
                    ptr(&bank.pointers),
                    ptr(&w.maps),
                    ptr(&w.counts),
                    ptr(&w.selected),
                    r,
                    d,
                    h,
                ],
            )?;
            self.launch(
                "dsv41_experts_silu_quant",
                stream,
                (silu_groups.div_ceil(4) as u32, 1, 1),
                (128, 1, 1),
                &mut [
                    ptr(&w.silu_values),
                    ptr(&w.silu_scales),
                    ptr(&w.status),
                    ptr(&w.silu),
                    ptr(&w.counts),
                    r,
                    h,
                ],
            )?;
            self.launch(
                "dsv41_experts_w2",
                stream,
                (
                    shape.dim.div_ceil(128) as u32,
                    rows.div_ceil(16) as u32,
                    LOCAL_EXPERTS as u32,
                ),
                (256, 1, 1),
                &mut [
                    ptr(&w.partial),
                    ptr(&w.status),
                    ptr(&w.silu_values),
                    ptr(&w.silu_scales),
                    ptr(&bank.pointers),
                    ptr(&w.counts),
                    r,
                    d,
                    h,
                ],
            )?;
            self.launch(
                "dsv41_experts_sum",
                stream,
                ((rows * shape.dim).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    ptr(&w.output),
                    ptr(&w.status),
                    ptr(&w.partial),
                    ptr(&w.inverse),
                    r,
                    d,
                ],
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = self.dev.synchronize();
            return Err(error);
        }
        Ok(PendingExperts {
            component: self,
            workspace,
            _bank: bank,
            _stream: stream,
            _input: input,
            _ids: route_ids,
            _weights: route_weights,
            _route_status: route_status,
        })
    }
}
/// Borrows every input, bank owner and workspace until queued work completes.
/// Drop also waits, so callers cannot free/reuse a pointer while kernels use it.
pub struct PendingExperts<'a> {
    component: &'a GpuExperts,
    workspace: &'a mut ExpertWorkspace,
    _bank: &'a ExpertBank,
    _stream: &'a CudaStream,
    _input: &'a CudaSlice<bf16>,
    _ids: &'a CudaSlice<u32>,
    _weights: &'a CudaSlice<f32>,
    _route_status: &'a CudaSlice<u32>,
}
pub struct PendingCombined<'a> {
    component: &'a GpuExperts,
    _stream: &'a CudaStream,
    _reduced: &'a CudaSlice<f32>,
    _shared: &'a CudaSlice<bf16>,
    output: &'a mut CudaSlice<bf16>,
    status: &'a mut CudaSlice<u32>,
}
impl PendingCombined<'_> {
    pub fn status_code(&self) -> Result<u32> {
        self.component.dev.synchronize()?;
        Ok(self.component.dev.dtoh_sync_copy(self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(
            status == 0,
            "reduced/shared combination GPU status0x{status:x}"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<bf16>> {
        self.check()?;
        Ok(self.component.dev.dtoh_sync_copy(self.output)?)
    }
    /// Caller preserves this owner and checks status before GPU consumption.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<bf16> {
        self.output
    }
}
impl Drop for PendingCombined<'_> {
    fn drop(&mut self) {
        let _ = self.component.dev.synchronize();
    }
}
impl PendingExperts<'_> {
    pub fn status_code(&self) -> Result<u32> {
        self.component.dev.synchronize()?;
        Ok(self.component.dev.dtoh_sync_copy(&self.workspace.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let code = self.status_code()?;
        ensure!(
            code == 0,
            "grouped experts GPU status0x{code:x} (1=input,2=scale,4=overflow,8=route,16=bank)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<f32>> {
        self.check()?;
        Ok(self.component.dev.dtoh_sync_copy(&self.workspace.output)?)
    }
    /// Caller must propagate/check status and preserve this pending owner while
    /// any downstream kernel reads its FP32 local partials on the same stream.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<f32> {
        &self.workspace.output
    }
}
impl Drop for PendingExperts<'_> {
    fn drop(&mut self) {
        let _ = self.component.dev.synchronize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduced_shared_midpoint_goldens_reject_premature_routed_cast_without_gpu() {
        let reduced = [1.00390625_f32, -1.00390625, 257.0, -257.0];
        let shared = [0.00390625_f32, -0.00390625, 1.0, -1.0].map(bf16::from_f32);
        let out = combine_reduced_shared_reference(&reduced, &shared).unwrap();
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0x3f81, 0xbf81, 0x4381, 0xc381]
        );
        let prematurely_rounded = reduced.map(|v| bf16::from_f32(v).to_f32());
        assert_eq!(
            combine_reduced_shared_reference(&prematurely_rounded, &shared)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x3f80, 0xbf80, 0x4380, 0xc380]
        );
        assert!(combine_reduced_shared_reference(&[f32::NAN], &[bf16::ONE]).is_err());
        assert!(combine_reduced_shared_reference(&[1.0], &[bf16::NAN]).is_err());
        assert!(combine_reduced_shared_reference(&[f32::MAX], &[bf16::MAX]).is_err());
        assert!(combine_reduced_shared_reference(&[1.0], &[]).is_err());
    }
    #[test]
    fn geometry_workspace_and_bank_ownership_fail_closed_without_gpu() {
        let shape = ExpertShape::native1017();
        assert_eq!(
            shape.lengths(),
            [5898240, 368640, 5898240, 368640, 5898240, 368640]
        );
        assert!(shape.workspace_bytes(128).unwrap() < 128 * 1024 * 1024);
        for rows in [0, 129, usize::MAX] {
            assert!(shape.workspace_bytes(rows).is_err());
        }
        for (dim, inter) in [(0, 32), (33, 32), (5121, 32), (32, 2305), (usize::MAX, 32)] {
            assert!(ExpertShape::oracle(dim, inter).is_err());
        }
        let ids: Vec<_> = (144..192).collect();
        assert!(validate_ids(3, &ids, true).is_ok());
        assert!(validate_ids(3, &ids[..47], true).is_err());
        assert!(validate_ids(3, &ids[..2], false).is_ok());
        assert!(validate_ids(3, &[144, 144], false).is_err());
        assert!(validate_ids(3, &[143], false).is_err());
        assert!(validate_ids(8, &[], false).is_err());
    }
}
