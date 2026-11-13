//! Isolated sparse-attention primitive for explicit local heads and independent
//! lanes. Caller supplies already rotated/quantized KV and checked index state.
//! This does not implement projections, compression, index scoring or a model.
use super::gpu_module::Module;
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

pub const PTX: &str = include_str!("../ptx/dsv41_attention.ptx");
pub const D: usize = 512;
pub const SHARED_BYTES: usize = 88320;
pub struct AttentionKernels {
    module: Module,
}

impl AttentionKernels {
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let module = Module::load(
            dev,
            stream,
            PTX,
            &["dsv41_sparse_attention"],
            "dsv41_attention_identity",
            u64::from_str_radix(env!("DSV41_ATTENTION_SOURCE_ID"), 16)?,
            1,
        )?;
        module.set_dynamic_smem("dsv41_sparse_attention", SHARED_BYTES)?;
        Ok(Self { module })
    }

    /// `b` independent lanes each have `m` queries and `n` latent KV rows. Index
    /// rows contain exactly topk slots (-1 masked), independently for every query.
    /// Buffers are caller-owned; status is sticky and must be checked before use.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse(
        &self,
        stream: &CudaStream,
        q: &CudaSlice<bf16>,
        kv: &CudaSlice<bf16>,
        sink: &CudaSlice<f32>,
        indices: &CudaSlice<i32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        b: usize,
        m: usize,
        heads: usize,
        n: usize,
        topk: usize,
    ) -> Result<()> {
        ensure!(
            (1..=16).contains(&b)
                && (1..=128).contains(&m)
                && (1..=64).contains(&heads)
                && (1..=131200).contains(&n)
                && (1..=640).contains(&topk),
            "bounded sparse attention dimensions"
        );
        ensure!(
            q.len() == b * m * heads * D
                && kv.len() == b * n * D
                && sink.len() == heads
                && indices.len() == b * m * topk
                && out.len() == q.len()
                && status.len() == 1,
            "sparse attention buffer extent"
        );
        self.module.check_buffer(q)?;
        self.module.check_buffer(kv)?;
        self.module.check_buffer(sink)?;
        self.module.check_buffer(indices)?;
        self.module.check_buffer(out)?;
        self.module.check_buffer(status)?;
        let mut qp = *q.device_ptr();
        let mut kp = *kv.device_ptr();
        let mut op = *out.device_ptr();
        let mut sp = *sink.device_ptr();
        let mut ip = *indices.device_ptr();
        let mut ep = *status.device_ptr();
        let mut tk = topk as i32;
        let mut ni = n as i32;
        let mut nh = heads as i32;
        let mut scale = (D as f32).sqrt().recip();
        let mut args: [*mut c_void; 10] = [
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut tk as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        unsafe {
            self.module.launch(
                "dsv41_sparse_attention",
                stream,
                (m as u32, b as u32, heads.div_ceil(16) as u32),
                (256, 1, 1),
                SHARED_BYTES as u32,
                &mut args,
            )
        }
    }
}
