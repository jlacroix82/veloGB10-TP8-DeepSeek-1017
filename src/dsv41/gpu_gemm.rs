//! Isolated faithful1017 native group32 W8A8 and FP8-activation/native-FP4 GEMM.
//! Native checkpoint bytes/scales remain row-major. No conversion/requantization is performed.
//! Separate FP32 partial and BF16 projection outputs are available. Official1017
//! RowParallelLinear rounds each local Linear output toBF16 before FP32 reduction;
//! choosing FP32 partials is an explicit alternative precision path, not bitwise
//! equivalence to that model schedule. WO-A also uses a separate BF16 einsum path.
//! This component is not a model loader or serving implementation.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice,
    LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub const ABI_VERSION: u64 = 1;
pub const PTX: &str = include_str!("../ptx/dsv41_gemm.ptx");
pub const CUDA_SOURCE: &[u8] = include_bytes!("../../kernels/dsv41_gemm.cu");
const MODULE: &str = "dsv41_gemm_isolated";
const NAMES: &[&str] = &[
    "dsv41_gemm_identity",
    "dsv41_gemm_fp8_group32",
    "dsv41_gemm_fp4_group32",
    "dsv41_gemm_fp8_group32_contributions",
    "dsv41_gemm_group32_ordered_finish",
];
// Bounded component policy, not a model/GPU capacity claim.
pub const MAX_MATRIX_ELEMENTS: usize = 64 * 1024 * 1024;
pub const MAX_ROWS: usize = 128;
/// Explicit opt-in scratch policy; this is separate from ordinary GEMM bounds.
pub const MAX_GROUP_CONTRIBUTION_BYTES: usize = 256 * 1024 * 1024;
pub const HOT_GROUP_WORKSPACE_ELEMENTS: usize = 16 * 2304 * (5120 / 32);

/// Explicit model policy. It never changes FP4, weight representation or the
/// order of the original K32 additions. Large/ragged prefill shapes fall back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GemmBackend {
    Original,
    ParallelGroupsHotSmallM,
}
impl GemmBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "original" => Ok(Self::Original),
            "parallel-groups-hot-small-m" => Ok(Self::ParallelGroupsHotSmallM),
            _ => anyhow::bail!("unknown FP8 GEMM backend {value}"),
        }
    }
    /// This is the actual dispatch selector, also exercised by CPU controls.
    pub fn selected_hot_shape(self, shape: GemmShape, fp4: bool) -> Option<usize> {
        if self != Self::ParallelGroupsHotSmallM
            || fp4
            || !(1..=16).contains(&shape.rows)
            || shape.inner != 5120
        {
            return None;
        }
        [512, 1280, 2304].iter().position(|n| *n == shape.outputs)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GemmWorkspaceBinding {
    pub backend: GemmBackend,
    pub owner: Uuid,
    pub boot: Uuid,
    pub model_instance_identity: [u8; 32],
    pub stream_identity: u64,
    pub contribution_capacity_bytes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct GemmDispatchSnapshot {
    pub binding: GemmWorkspaceBinding,
    pub eligible_outputs: [usize; 3],
    pub selected_queued_by_shape: [u64; 3],
    pub original_fallback_queued: u64,
    pub poisoned: bool,
}
/// The lock serializes host enqueue of each producer+finish pair. Reuse is
/// ordered on ONE exact owned stream, so a later producer cannot overwrite
/// scratch before the earlier finish consumes it. Outputs/status stay owned
/// separately. A queued output retains this Arc; final owner drop fences once.
pub struct GemmGroupWorkspace {
    dev: Arc<CudaDevice>,
    // A selected output can outlive its model; retain the actual stream owner
    // alongside its allocation, rather than relying on a raw handle lifetime.
    _stream_owner: Arc<CudaStream>,
    owner: Uuid,
    boot: Uuid,
    stream_identity: u64,
    state: Mutex<GroupWorkspaceState>,
}
struct GroupWorkspaceState {
    scratch: CudaSlice<f32>,
    control: WorkspaceControl,
}
#[derive(Default)]
struct WorkspaceControl {
    model: Option<[u8; 32]>,
    poisoned: bool,
    selected: [u64; 3],
    fallback: u64,
}
fn check_owner(actual_boot: Uuid, actual_stream: u64, boot: Uuid, stream: u64) -> Result<()> {
    ensure!(
        !actual_boot.is_nil()
            && actual_stream != 0
            && boot == actual_boot
            && stream == actual_stream,
        "GEMM scratch cross-boot/stream use"
    );
    Ok(())
}
impl WorkspaceControl {
    fn bind(&mut self, model: [u8; 32]) -> Result<()> {
        ensure!(
            !self.poisoned && self.model.is_none() && model != [0; 32],
            "GEMM scratch binds exactly one nonzero model instance"
        );
        self.model = Some(model);
        Ok(())
    }
    fn check(&self, model: [u8; 32]) -> Result<()> {
        ensure!(
            !self.poisoned && self.model == Some(model),
            "GEMM scratch cross-model/poisoned use"
        );
        Ok(())
    }
    fn can_queue(&self, selected: Option<usize>) -> Result<()> {
        ensure!(
            !self.poisoned && self.model.is_some(),
            "GEMM scratch unbound/poisoned enqueue"
        );
        let count = match selected {
            Some(i) => *self.selected.get(i).context("unknown hot GEMM shape")?,
            None => self.fallback,
        };
        ensure!(count < u64::MAX, "GEMM dispatch counter overflow");
        Ok(())
    }
    fn queued(&mut self, selected: Option<usize>) -> Result<()> {
        self.can_queue(selected)?;
        let count = match selected {
            Some(i) => self.selected.get_mut(i).context("unknown hot GEMM shape")?,
            None => &mut self.fallback,
        };
        *count = count
            .checked_add(1)
            .context("GEMM dispatch counter overflow")?;
        Ok(())
    }
}
impl GemmGroupWorkspace {
    pub fn new(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Arc<Self>> {
        ensure!(
            !boot.is_nil() && !stream.stream.is_null(),
            "GEMM scratch needs actual boot/stream"
        );
        let bytes = HOT_GROUP_WORKSPACE_ELEMENTS
            .checked_mul(4)
            .context("scratch bytes overflow")?;
        ensure!(
            bytes <= MAX_GROUP_CONTRIBUTION_BYTES,
            "contribution scratch exceeds256MiB"
        );
        // This one22.5MiB arena is shared by the attention and shared-expert callers.
        Ok(Arc::new(Self {
            dev: dev.clone(),
            _stream_owner: stream.clone(),
            owner: Uuid::new_v4(),
            boot,
            stream_identity: stream.stream as usize as u64,
            state: Mutex::new(GroupWorkspaceState {
                scratch: dev.alloc_zeros::<f32>(HOT_GROUP_WORKSPACE_ELEMENTS)?,
                control: WorkspaceControl::default(),
            }),
        }))
    }
    fn check_context(&self, dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, dev) && self.stream_identity == stream.stream as usize as u64,
            "GEMM scratch cross-device/stream use"
        );
        Ok(())
    }
    pub fn bind_model(&self, model: [u8; 32], boot: Uuid, stream: &CudaStream) -> Result<()> {
        check_owner(
            self.boot,
            self.stream_identity,
            boot,
            stream.stream as usize as u64,
        )?;
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("GEMM scratch lock poisoned"))?
            .control
            .bind(model)
    }
    pub fn check_model(&self, model: [u8; 32], boot: Uuid, stream: &CudaStream) -> Result<()> {
        check_owner(
            self.boot,
            self.stream_identity,
            boot,
            stream.stream as usize as u64,
        )?;
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("GEMM scratch lock poisoned"))?
            .control
            .check(model)
    }
    pub fn binding(&self) -> GemmWorkspaceBinding {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        GemmWorkspaceBinding {
            backend: GemmBackend::ParallelGroupsHotSmallM,
            owner: self.owner,
            boot: self.boot,
            model_instance_identity: state.control.model.unwrap_or([0; 32]),
            stream_identity: self.stream_identity,
            contribution_capacity_bytes: HOT_GROUP_WORKSPACE_ELEMENTS * 4,
        }
    }
    pub fn snapshot(&self) -> GemmDispatchSnapshot {
        let binding = self.binding();
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        GemmDispatchSnapshot {
            binding,
            eligible_outputs: [512, 1280, 2304],
            selected_queued_by_shape: state.control.selected,
            original_fallback_queued: state.control.fallback,
            poisoned: state.control.poisoned || self.state.is_poisoned(),
        }
    }
}
impl Drop for GemmGroupWorkspace {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
#[derive(Debug, Clone, Copy)]
pub struct GemmShape {
    pub rows: usize,
    pub outputs: usize,
    pub inner: usize,
}
impl GemmShape {
    pub fn group_contribution_elements(self) -> Result<usize> {
        self.validate()?;
        let elements = self
            .rows
            .checked_mul(self.outputs)
            .and_then(|x| x.checked_mul(self.inner / 32))
            .context("parallel group scratch size overflow")?;
        ensure!(
            elements
                .checked_mul(4)
                .is_some_and(|x| x <= MAX_GROUP_CONTRIBUTION_BYTES),
            "parallel group scratch exceeds separate256MiB component policy"
        );
        // CUDA's grid Z is bounded independently of allocation arithmetic.
        ensure!(
            self.inner / 32 <= 65535,
            "parallel group gridZ exceeds65535"
        );
        Ok(elements)
    }
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.rows > 0 && self.rows <= MAX_ROWS,
            "GEMM row bound1..128"
        );
        ensure!(
            self.outputs > 0 && self.inner > 0 && self.inner % 32 == 0,
            "positive GEMM shape and K divisible by32 required"
        );
        for (a, b) in [
            (self.rows, self.inner),
            (self.outputs, self.inner),
            (self.rows, self.outputs),
        ] {
            ensure!(
                a.checked_mul(b).is_some_and(|x| x <= MAX_MATRIX_ELEMENTS),
                "GEMM component allocation policy exceeded"
            );
        }
        i32::try_from(self.rows)?;
        i32::try_from(self.outputs)?;
        i32::try_from(self.inner)?;
        Ok(())
    }
}
fn ptr<T>(slice: &CudaSlice<T>) -> u64 {
    *slice.device_ptr()
}
pub struct GemmOutput<T> {
    dev: Arc<CudaDevice>,
    values: CudaSlice<T>,
    status: CudaSlice<u32>,
    // Keep first-stage producers alive through ordered finish and output use.
    // Existing dispatch leaves this None and executes the original kernel.
    _group_contributions: Option<GroupStorage>,
    pub shape: GemmShape,
}
#[allow(dead_code)] // Ownership, not a host read: both variants retain queued producers.
enum GroupStorage {
    Owned(CudaSlice<f32>),
    Shared(Arc<GemmGroupWorkspace>),
}
impl<T: DeviceRepr> GemmOutput<T> {
    pub(crate) fn status_device_word(&self) -> &CudaSlice<u32> {
        &self.status
    }
    pub fn status_code(&self) -> Result<u32> {
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&self.status)?[0])
    }
    pub fn check(&self) -> Result<()> {
        let status = self.status_code()?;
        ensure!(
            status == 0,
            "dsv41 GEMM status0x{status:x} (1=NaNcode,2=NaNscale,4=overflow)"
        );
        Ok(())
    }
    pub fn download_checked(&self) -> Result<Vec<T>> {
        self.check()?;
        Ok(self.dev.dtoh_sync_copy(&self.values)?)
    }
    /// Caller must establish completion and check status before consuming these pending values.
    pub unsafe fn device_values_unchecked(&self) -> &CudaSlice<T> {
        &self.values
    }
}
pub struct GpuGemm {
    dev: Arc<CudaDevice>,
    source_id: u64,
    workspace: Option<Arc<GemmGroupWorkspace>>,
}
impl GpuGemm {
    fn stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit caller-owned blocking stream required"
        );
        let mut flags = 0_u32;
        let result = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            result == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "dsv41 GEMM requires valid blocking stream flags0; got{flags} {result:?}"
        );
        Ok(())
    }
    fn input<T>(&self, input: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &input.device()),
            "GEMM input device/context mismatch"
        );
        Ok(())
    }
    fn function(&self, name: &'static str) -> Result<CudaFunction> {
        self.dev
            .get_func(MODULE, name)
            .with_context(|| format!("missing isolated GEMM kernel{name}"))
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let source_id =
            u64::from_str_radix(env!("DSV41_GEMM_SOURCE_ID").trim_start_matches("0x"), 16)
                .context("invalid dedicatedDSV41_GEMM_SOURCE_ID")?;
        let this = Self {
            dev: dev.clone(),
            source_id,
            workspace: None,
        };
        this.stream(stream)?;
        if dev.get_func(MODULE, "dsv41_gemm_identity").is_none() {
            dev.load_ptx(Ptx::from_src(PTX), MODULE, NAMES)
                .context("load isolated group32 GEMM PTX")?;
        }
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        dev.memset_zeros(&mut identity)?;
        unsafe {
            this.function("dsv41_gemm_identity")?.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (ptr(&identity),),
            )?;
        }
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&identity)? == [source_id, ABI_VERSION],
            "isolated GEMM source/ABI mismatch"
        );
        Ok(this)
    }
    pub fn load_with_workspace(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        workspace: Option<Arc<GemmGroupWorkspace>>,
    ) -> Result<Self> {
        if let Some(w) = &workspace {
            w.check_context(dev, stream)?;
        }
        let mut this = Self::load(dev, stream)?;
        this.workspace = workspace;
        Ok(this)
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
    fn launch<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
        fp4: bool,
        bf16_output: bool,
        explicit_parallel: bool,
    ) -> Result<GemmOutput<T>> {
        self.stream(stream)?;
        shape.validate()?;
        for input in [a, a_scale, w, w_scale] {
            self.input(input)?;
        }
        let groups = shape.inner / 32;
        ensure!(
            a.len() == shape.rows * shape.inner && a_scale.len() == shape.rows * groups,
            "GEMM activation/scales shape mismatch"
        );
        ensure!(
            w.len() == shape.outputs * shape.inner / (if fp4 { 2 } else { 1 }),
            "GEMM native weight shape mismatch"
        );
        let weight_scale_rows = if fp4 {
            shape.outputs
        } else {
            shape.outputs.div_ceil(32)
        };
        ensure!(
            w_scale.len() == weight_scale_rows * groups,
            "GEMM native group32 weight scales shape mismatch"
        );
        ensure!(
            !explicit_parallel || !fp4,
            "parallel backend supports originalFP8 only"
        );
        ensure!(
            !explicit_parallel || self.workspace.is_none(),
            "raw parallel API and model-owned dispatch cannot be mixed"
        );
        let selected = if self.workspace.is_some() {
            GemmBackend::ParallelGroupsHotSmallM.selected_hot_shape(shape, fp4)
        } else {
            None
        };
        let parallel = explicit_parallel || selected.is_some();
        let contribution_elements = if parallel {
            Some(shape.group_contribution_elements()?)
        } else {
            None
        };
        // Validate binding and reserve the whole enqueue pair BEFORE launching.
        // The same lock also serializes fallback enqueues on this owned stream.
        let mut shared = if let Some(workspace) = &self.workspace {
            workspace.check_context(&self.dev, stream)?;
            let guard = workspace
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("GEMM scratch lock poisoned"))?;
            guard.control.can_queue(selected)?;
            if selected.is_some() {
                ensure!(
                    contribution_elements.unwrap() <= guard.scratch.len(),
                    "GEMM scratch capacity exceeded"
                );
            }
            Some(guard)
        } else {
            None
        };
        let values = self.dev.alloc_zeros::<T>(shape.rows * shape.outputs)?;
        let mut status = self.dev.alloc_zeros::<u32>(1)?;
        self.dev.memset_zeros(&mut status)?;
        let output = ptr(&values);
        let owned = if explicit_parallel {
            Some(
                self.dev
                    .alloc_zeros::<f32>(contribution_elements.unwrap())?,
            )
        } else {
            None
        };
        let queued = (|| -> Result<()> {
            unsafe {
                if parallel {
                    let partial = if selected.is_some() {
                        ptr(&shared.as_ref().unwrap().scratch)
                    } else {
                        ptr(owned.as_ref().unwrap())
                    };
                    self.function("dsv41_gemm_fp8_group32_contributions")?
                        .launch_on_stream(
                            stream,
                            LaunchConfig {
                                grid_dim: (
                                    u32::try_from(shape.outputs.div_ceil(128))?,
                                    u32::try_from(shape.rows.div_ceil(16))?,
                                    u32::try_from(groups)?,
                                ),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                partial,
                                ptr(&status),
                                ptr(a),
                                ptr(a_scale),
                                ptr(w),
                                ptr(w_scale),
                                shape.rows as i32,
                                shape.outputs as i32,
                                shape.inner as i32,
                            ),
                        )?;
                    self.function("dsv41_gemm_group32_ordered_finish")?
                        .launch_on_stream(
                            stream,
                            LaunchConfig {
                                grid_dim: (
                                    u32::try_from((shape.rows * shape.outputs).div_ceil(256))?,
                                    1,
                                    1,
                                ),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                if bf16_output { 0_u64 } else { output },
                                if bf16_output { output } else { 0_u64 },
                                ptr(&status),
                                partial,
                                shape.rows as i32,
                                shape.outputs as i32,
                                shape.inner as i32,
                            ),
                        )?;
                } else {
                    self.function(if fp4 {
                        "dsv41_gemm_fp4_group32"
                    } else {
                        "dsv41_gemm_fp8_group32"
                    })?
                    .launch_on_stream(
                        stream,
                        LaunchConfig {
                            grid_dim: (
                                u32::try_from(shape.outputs.div_ceil(128))?,
                                u32::try_from(shape.rows.div_ceil(16))?,
                                1,
                            ),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            if bf16_output { 0_u64 } else { output },
                            if bf16_output { output } else { 0_u64 },
                            ptr(&status),
                            ptr(a),
                            ptr(a_scale),
                            ptr(w),
                            ptr(w_scale),
                            shape.rows as i32,
                            shape.outputs as i32,
                            shape.inner as i32,
                        ),
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(error) = queued {
            if let Some(g) = &mut shared {
                g.control.poisoned = true;
            }
            // New pending producers/outputs stay alive through this error fence.
            let _ = self.dev.synchronize();
            return Err(error);
        }
        if let Some(g) = &mut shared {
            g.control.queued(selected)?;
        }
        let storage = if selected.is_some() {
            Some(GroupStorage::Shared(
                self.workspace.as_ref().unwrap().clone(),
            ))
        } else {
            owned.map(GroupStorage::Owned)
        };
        Ok(GemmOutput {
            dev: self.dev.clone(),
            values,
            status,
            _group_contributions: storage,
            shape,
        })
    }
    pub fn fp8_partial_f32(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<f32>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, false, false, false)
    }
    pub fn fp8_complete_bf16(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<bf16>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, false, true, false)
    }
    pub fn fp4_partial_f32(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<f32>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, true, false, false)
    }
    pub fn fp4_complete_bf16(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<bf16>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, true, true, false)
    }

    /// Experimental explicit backend. No model/default dispatch selects this.
    pub fn fp8_parallel_groups_partial_f32(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<f32>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, false, false, true)
    }
    pub fn fp8_parallel_groups_complete_bf16(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<u8>,
        shape: GemmShape,
    ) -> Result<GemmOutput<bf16>> {
        self.launch(stream, a, a_scale, w, w_scale, shape, false, true, true)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_model_selector_is_explicit_hot_small_m_only() {
        let original = GemmBackend::parse("original").unwrap();
        let parallel = GemmBackend::parse("parallel-groups-hot-small-m").unwrap();
        assert!(GemmBackend::parse("parallel").is_err());
        for rows in 1..=16 {
            for (index, outputs) in [512, 1280, 2304].into_iter().enumerate() {
                let shape = GemmShape {
                    rows,
                    outputs,
                    inner: 5120,
                };
                assert_eq!(parallel.selected_hot_shape(shape, false), Some(index));
                assert_eq!(parallel.selected_hot_shape(shape, true), None);
                assert_eq!(original.selected_hot_shape(shape, false), None);
                assert!(
                    shape.group_contribution_elements().unwrap() <= HOT_GROUP_WORKSPACE_ELEMENTS
                );
            }
        }
        for shape in [
            GemmShape {
                rows: 0,
                outputs: 512,
                inner: 5120,
            },
            GemmShape {
                rows: 17,
                outputs: 512,
                inner: 5120,
            },
            GemmShape {
                rows: 128,
                outputs: 1280,
                inner: 5120,
            },
            GemmShape {
                rows: 16,
                outputs: 4096,
                inner: 1280,
            },
            GemmShape {
                rows: 16,
                outputs: 5120,
                inner: 2304,
            },
            GemmShape {
                rows: 16,
                outputs: 512,
                inner: 2560,
            },
        ] {
            assert_eq!(parallel.selected_hot_shape(shape, false), None);
        }
        assert_eq!(HOT_GROUP_WORKSPACE_ELEMENTS * 4, 23592960);
        assert!(HOT_GROUP_WORKSPACE_ELEMENTS * 4 < MAX_GROUP_CONTRIBUTION_BYTES);
    }
    #[test]
    fn actual_workspace_control_rejects_rebind_stale_model_poison_and_counter_wrap() {
        let mut c = WorkspaceControl::default();
        assert!(c.can_queue(Some(0)).is_err());
        assert!(c.bind([0; 32]).is_err());
        c.bind([7; 32]).unwrap();
        assert!(c.bind([7; 32]).is_err());
        assert!(c.bind([8; 32]).is_err());
        c.check([7; 32]).unwrap();
        assert!(c.check([8; 32]).is_err());
        for i in 0..3 {
            c.can_queue(Some(i)).unwrap();
            c.queued(Some(i)).unwrap();
        }
        c.queued(None).unwrap();
        assert_eq!(c.selected, [1, 1, 1]);
        assert_eq!(c.fallback, 1);
        assert!(c.can_queue(Some(3)).is_err());
        c.selected[0] = u64::MAX;
        assert!(c.can_queue(Some(0)).is_err());
        assert!(c.queued(Some(0)).is_err());
        assert_eq!(c.selected[0], u64::MAX);
        c.fallback = u64::MAX;
        assert!(c.can_queue(None).is_err());
        c.poisoned = true;
        assert!(c.check([7; 32]).is_err());
        assert!(c.can_queue(Some(1)).is_err());
    }
    #[test]
    fn actual_owner_validation_rejects_cross_boot_stream_and_zero_identity() {
        let b = Uuid::from_u128(214);
        check_owner(b, 123, b, 123).unwrap();
        assert!(check_owner(b, 123, Uuid::from_u128(215), 123).is_err());
        assert!(check_owner(b, 123, b, 124).is_err());
        assert!(check_owner(Uuid::nil(), 123, Uuid::nil(), 123).is_err());
        assert!(check_owner(b, 0, b, 0).is_err());
    }
    #[test]
    fn parallel_group_scratch_is_checked_before_allocation_cpu_only() {
        for (rows, outputs, inner, elements) in [
            (1, 5120, 5120, 819200),
            (16, 5120, 5120, 13107200),
            (16, 10240, 5120, 26214400),
            (128, 48, 5120, 983040),
            (2, 33, 96, 198),
        ] {
            assert_eq!(
                GemmShape {
                    rows,
                    outputs,
                    inner
                }
                .group_contribution_elements()
                .unwrap(),
                elements
            );
        }
        // Exactly256MiB is allowed; neither the ordinary matrix bounds nor
        // an unchecked rows*outputs*groups product can evade this policy.
        assert_eq!(
            GemmShape {
                rows: 128,
                outputs: 4096,
                inner: 4096
            }
            .group_contribution_elements()
            .unwrap()
                * 4,
            MAX_GROUP_CONTRIBUTION_BYTES
        );
        for shape in [
            GemmShape {
                rows: 128,
                outputs: 4097,
                inner: 4096,
            },
            GemmShape {
                rows: 16,
                outputs: 10240,
                inner: 16384,
            },
            GemmShape {
                rows: 1,
                outputs: 32,
                inner: 32 * 65536,
            },
            GemmShape {
                rows: 0,
                outputs: 32,
                inner: 32,
            },
            GemmShape {
                rows: 1,
                outputs: usize::MAX,
                inner: 32,
            },
            GemmShape {
                rows: usize::MAX,
                outputs: 1,
                inner: 32,
            },
        ] {
            assert!(shape.group_contribution_elements().is_err(), "{shape:?}");
        }
    }
    #[test]
    fn native_group32_shapes_and_real1017_geometry_cpu_only() {
        for (outputs, inner) in [
            (35, 96),
            (1280, 5120),
            (4096, 1280),
            (1024, 4096),
            (5120, 1024),
            (3200, 6144),
            (5120, 2304),
            (2304, 5120),
        ] {
            for rows in [1, 2, 4, 8, 16, 128] {
                GemmShape {
                    rows,
                    outputs,
                    inner,
                }
                .validate()
                .unwrap();
            }
        }
        assert!(GemmShape {
            rows: 0,
            outputs: 32,
            inner: 32
        }
        .validate()
        .is_err());
        assert!(GemmShape {
            rows: 129,
            outputs: 32,
            inner: 32
        }
        .validate()
        .is_err());
        assert!(GemmShape {
            rows: 1,
            outputs: 32,
            inner: 33
        }
        .validate()
        .is_err());
        assert!(GemmShape {
            rows: 1,
            outputs: usize::MAX,
            inner: 32
        }
        .validate()
        .is_err());
    }
}
