//! Standalone1017 route/SwiGLU only. Dense linears and expert dispatch are absent.
//! Caller clears status, waits for stream completion and requires0 before use.
use super::moe;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;
pub const PTX: &str = include_str!("../ptx/dsv41_moe.ptx");
const MODULE: &str = "dsv41_moe_v1";
pub struct MoeKernels {
    dev: Arc<CudaDevice>,
}
impl MoeKernels {
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(!s.stream.is_null(), "explicit blocking stream required");
        let mut f = 0;
        let rc = unsafe { sys::cuStreamGetFlags(s.stream, &mut f) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && f == 0,
            "valid blocking stream required"
        );
        Ok(())
    }
    fn input<T>(&self, s: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &s.device()),
            "MoE buffer from another device/context handle"
        );
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let me = Self { dev: dev.clone() };
        me.stream(stream)?;
        dev.load_ptx(
            Ptx::from_src(PTX),
            MODULE,
            &["dsv41_moe_identity", "dsv41_moe_route", "dsv41_moe_swiglu"],
        )?;
        let mut id = dev.alloc_zeros::<u64>(2)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_moe_identity")
                .context("MoE identity kernel")?
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), (&mut id,))?;
        }
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&id)? == [u64::from_str_radix(env!("DSV41_MOE_SOURCE_ID"), 16)?, 1],
            "stale MoE PTX/source/ABI identity"
        );
        Ok(me)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        s: &CudaStream,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        ids: &mut CudaSlice<u32>,
        weights: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        rows: usize,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(logits)?;
        self.input(bias)?;
        self.input(ids)?;
        self.input(weights)?;
        self.input(status)?;
        ensure!(
            (1..=moe::MAX_ROWS).contains(&rows)
                && logits.len() == rows * 384
                && bias.len() == 384
                && ids.len() == rows * 6
                && weights.len() == rows * 6
                && status.len() == 1,
            "route384/top6 geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_moe_route")
                .context("route kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (logits, bias, ids, weights, status),
                )?;
        }
        Ok(())
    }
    /// Shared Expert uses weighted=false; route-weight buffer remains an explicit
    /// rows-sized allocation but is not read by that kernel branch.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(
        &self,
        s: &CudaStream,
        gate: &CudaSlice<bf16>,
        up: &CudaSlice<bf16>,
        weights: &CudaSlice<f32>,
        weighted: bool,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(gate)?;
        self.input(up)?;
        self.input(weights)?;
        self.input(out)?;
        self.input(status)?;
        ensure!(
            (1..=moe::MAX_ROWS).contains(&rows)
                && (1..=2304).contains(&columns)
                && gate.len() == rows * columns
                && up.len() == gate.len()
                && out.len() == gate.len()
                && weights.len() == rows
                && status.len() == 1,
            "SwiGLU geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_moe_swiglu")
                .context("SwiGLU kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems((rows * columns) as u32),
                    (
                        gate,
                        up,
                        weights,
                        out,
                        status,
                        rows as u32,
                        columns as u32,
                        u32::from(weighted),
                    ),
                )?;
        }
        Ok(())
    }
}
