//! Graph for the packed HC attention-input triplet only. Inputs and weights
//! vary per layer/forward, so bind them on every launch. Scratch addresses
//! and geometry stay fixed per width; all original status checks stay eager.
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaStream};
use std::{collections::BTreeMap, ffi::c_void, mem::MaybeUninit, ptr};

const DIM: u32 = 5120;

pub(super) struct HcArguments {
    pub h: u64,
    pub weight: u64,
    pub mixes: u64,
    pub status: u64,
    pub scale: u64,
    pub base: u64,
    pub pre: u64,
    pub post: u64,
    pub comb: u64,
    pub incoming: u64,
    pub collapsed: u64,
}

struct Executable {
    original: sys::CUgraph,
    exec: sys::CUgraphExec,
    nodes: [sys::CUgraphNode; 3],
    params: [sys::CUDA_KERNEL_NODE_PARAMS; 3],
}
// Access is serialized by BatchBlockKernels' model-owned scratch lock. All
// CUDA calls are on the exact bound compute stream with its context current.
unsafe impl Send for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                sys::cuGraphExecDestroy(self.exec);
            }
            if !self.original.is_null() {
                sys::cuGraphDestroy(self.original);
            }
        }
    }
}

fn check(rc: sys::CUresult, stage: &str) -> Result<()> {
    ensure!(
        rc == sys::CUresult::CUDA_SUCCESS,
        "HC graph {stage}: {rc:?}"
    );
    Ok(())
}

impl Executable {
    fn capture(stream: &CudaStream, enqueue: impl FnOnce() -> Result<()>) -> Result<Self> {
        check(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    stream.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "begin",
        )?;
        let body = enqueue();
        let mut graph = ptr::null_mut();
        let ended = unsafe { sys::cuStreamEndCapture(stream.stream, &mut graph) };
        let mut out = Self {
            original: graph,
            exec: ptr::null_mut(),
            nodes: [ptr::null_mut(); 3],
            params: std::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() }),
        };
        body?;
        check(ended, "end")?;
        ensure!(!graph.is_null(), "missing captured graph");
        let mut n = 0;
        check(
            unsafe { sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut n) },
            "count",
        )?;
        ensure!(n == 3, "HC capture must contain exactly three kernels");
        let mut count = 1;
        check(
            unsafe { sys::cuGraphGetRootNodes(graph, out.nodes.as_mut_ptr(), &mut count) },
            "root",
        )?;
        ensure!(count == 1, "HC graph needs one root");
        for i in 0..3 {
            let node = out.nodes[i];
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            check(
                unsafe { sys::cuGraphNodeGetType(node, &mut ty) },
                "node type",
            )?;
            ensure!(
                ty == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL,
                "HC graph contains a nonkernel node"
            );
            check(
                unsafe { sys::cuGraphKernelNodeGetParams(node, &mut out.params[i]) },
                "kernel metadata",
            )?;
            let mut count = 0;
            check(
                unsafe { sys::cuGraphNodeGetDependentNodes(node, ptr::null_mut(), &mut count) },
                "edge count",
            )?;
            ensure!(
                count == usize::from(i < 2),
                "HC graph is not the expected single kernel chain"
            );
            if i < 2 {
                check(
                    unsafe {
                        sys::cuGraphNodeGetDependentNodes(node, &mut out.nodes[i + 1], &mut count)
                    },
                    "edge",
                )?;
            }
        }
        ensure!(
            out.params[0].gridDimY == 24
                && out.params[1].gridDimY == 1
                && out.params[2].gridDimY == 1
                && out.params[0].func != out.params[1].func
                && out.params[1].func != out.params[2].func
                && out.params[0].func != out.params[2].func,
            "HC node identity/order changed"
        );
        check(
            unsafe {
                sys::cuGraphInstantiate_v2(
                    &mut out.exec,
                    graph,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                )
            },
            "instantiate",
        )?;
        Ok(out)
    }

    fn run(&self, stream: &CudaStream, args: HcArguments, width: u32) -> Result<()> {
        macro_rules! params { ($($x:expr),+ $(,)?) => { [ $((&$x as *const _ as *mut c_void)),+ ] } }
        let dim = DIM;
        let mut project = params![args.h, args.weight, args.mixes, args.status, dim];
        let mut split = params![
            args.mixes,
            args.scale,
            args.base,
            args.pre,
            args.post,
            args.comb,
            args.status,
            width
        ];
        let mut collapse = params![
            args.h,
            args.incoming,
            args.collapsed,
            args.status,
            width,
            dim
        ];
        let kernel_params = [
            project.as_mut_ptr(),
            split.as_mut_ptr(),
            collapse.as_mut_ptr(),
        ];
        for (i, mut spec) in self.params.iter().copied().enumerate() {
            spec.kernelParams = kernel_params[i];
            spec.extra = ptr::null_mut();
            check(
                unsafe { sys::cuGraphExecKernelNodeSetParams(self.exec, self.nodes[i], &spec) },
                "update live inputs",
            )?;
        }
        check(
            unsafe { sys::cuGraphLaunch(self.exec, stream.stream) },
            "launch",
        )
    }
}

#[derive(Default)]
pub(super) struct HcGraphs(BTreeMap<usize, Executable>);
impl HcGraphs {
    /// First use is allowed only in an untimed ramp for that width. A failed
    /// capture is an error; it cannot silently turn the candidate eager.
    pub fn enqueue(
        &mut self,
        width: usize,
        stream: &CudaStream,
        args: HcArguments,
        enqueue: impl FnOnce() -> Result<()>,
    ) -> Result<bool> {
        let mut captured = false;
        let exec = match self.0.entry(width) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                check(
                    unsafe { sys::cuStreamSynchronize(stream.stream) },
                    "settle before capture",
                )?;
                captured = true;
                e.insert(Executable::capture(stream, enqueue)?)
            }
        };
        exec.run(stream, args, width as u32)?;
        Ok(captured)
    }
}

/// Small offline device oracle: alternate the real eager and graph path
/// with new inputs and layer weights, including a poisoned sticky status.
#[doc(hidden)]
pub fn check_device_replay() -> Result<Vec<(usize, usize)>> {
    use super::gpu_hc::HcKernels;
    use cudarc::driver::CudaDevice;
    use half::bf16;
    let dev = CudaDevice::new(0)?;
    let stream = crate::gpu::fork_blocking_stream(&dev);
    let kernels = HcKernels::load(&dev, &stream)?;
    let mut graphs = HcGraphs::default();
    let mut widths = Vec::new();
    let dim = DIM as usize;
    for (case, t) in [8, 16, 1, 2, 16, 8, 1, 2, 8, 16].into_iter().enumerate() {
        let xs: Vec<bf16> = (0..t * 4 * dim)
            .map(|i| bf16::from_f32((((i + case * 7) % 29) as f32 - 14.0) / 16.0))
            .collect();
        let ws: Vec<f32> = (0..24 * 4 * dim)
            .map(|i| (((i * 13 + case * 7) % 31) as f32 - 15.0) / 2048.0)
            .collect();
        let h = dev.htod_sync_copy(&xs)?;
        let w = dev.htod_sync_copy(&ws)?;
        let scale = dev.htod_sync_copy(&[0.2f32, 0.3, 0.4])?;
        let base = dev.htod_sync_copy(&[0.02f32; 24])?;
        let incoming = dev.htod_sync_copy(&vec![0.25f32; t * 4])?;
        let result = |enable, graphs: &mut HcGraphs| -> Result<_> {
            let mut mixes = dev.alloc_zeros::<f32>(t * 24)?;
            let mut pre = dev.alloc_zeros::<f32>(t * 4)?;
            let mut post = dev.alloc_zeros::<f32>(t * 4)?;
            let mut comb = dev.alloc_zeros::<f32>(t * 16)?;
            let mut out = dev.alloc_zeros::<bf16>(t * dim)?;
            let mut status = dev.htod_sync_copy(&[u32::from(case >= 8) * 4])?;
            let checked = kernels.check_attention_input(
                &stream,
                &h,
                &w,
                &mut mixes,
                &mut status,
                &scale,
                &base,
                &mut pre,
                &mut post,
                &mut comb,
                &incoming,
                &mut out,
                t,
                dim,
            )?;
            let capture = if enable {
                let args = checked.arguments();
                graphs.enqueue(t, &stream, args, || checked.enqueue())?
            } else {
                checked.enqueue()?;
                false
            };
            check(
                unsafe { sys::cuStreamSynchronize(stream.stream) },
                "oracle sync",
            )?;
            let status = dev.dtoh_sync_copy(&status)?[0];
            let mut values = Vec::new();
            for ptr in [&mixes, &pre, &post, &comb] {
                values.extend(dev.dtoh_sync_copy(ptr)?.iter().map(|x| x.to_bits()));
            }
            values.extend(dev.dtoh_sync_copy(&out)?.iter().map(|x| x.to_bits() as u32));
            Ok((capture, status, values))
        };
        let (_, eager_status, eager) = result(false, &mut graphs)?;
        let (new, graph_status, captured) = result(true, &mut graphs)?;
        ensure!(new == (case < 4), "graph reuse/capture history differs");
        ensure!(
            eager_status == u32::from(case >= 8) * 4 && graph_status == eager_status,
            "graph sticky status differs"
        );
        ensure!(
            eager == captured,
            "graph mHC output differs at width {t} case {case}"
        );
        widths.push((t, eager.len()));
    }
    Ok(widths)
}
