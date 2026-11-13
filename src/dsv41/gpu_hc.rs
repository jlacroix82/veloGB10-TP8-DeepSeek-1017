//! Bounded standalone1017 mHC kernels; no whole-model/serving implementation.
//! Caller clears the sticky status before a group, waits for stream completion,
//! and requires status0 before consuming any output (1=nonfinite input,
//! 2=nonfinite intermediate/BF16 output). Every operation checks device handles.
use super::{gpu_hc_graph::HcArguments, hc};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;

pub const PTX: &str = include_str!("../ptx/dsv41_hc.ptx");
const MODULE: &str = "dsv41_hc_v1";
pub struct HcKernels {
    dev: Arc<CudaDevice>,
}

/// The checks run before capture; only launch operations may follow.
pub(super) struct CheckedAttentionInput<'a> {
    k: &'a HcKernels,
    stream: &'a CudaStream,
    h: &'a CudaSlice<bf16>,
    weight: &'a CudaSlice<f32>,
    mixes: &'a mut CudaSlice<f32>,
    status: &'a mut CudaSlice<u32>,
    scale: &'a CudaSlice<f32>,
    base: &'a CudaSlice<f32>,
    pre: &'a mut CudaSlice<f32>,
    post: &'a mut CudaSlice<f32>,
    comb: &'a mut CudaSlice<f32>,
    incoming: &'a CudaSlice<f32>,
    collapsed: &'a mut CudaSlice<bf16>,
    tokens: usize,
    dim: usize,
}

impl CheckedAttentionInput<'_> {
    pub(super) fn arguments(&self) -> HcArguments {
        HcArguments {
            h: *self.h.device_ptr(),
            weight: *self.weight.device_ptr(),
            mixes: *self.mixes.device_ptr(),
            status: *self.status.device_ptr(),
            scale: *self.scale.device_ptr(),
            base: *self.base.device_ptr(),
            pre: *self.pre.device_ptr(),
            post: *self.post.device_ptr(),
            comb: *self.comb.device_ptr(),
            incoming: *self.incoming.device_ptr(),
            collapsed: *self.collapsed.device_ptr(),
        }
    }

    pub(super) fn enqueue(self) -> Result<()> {
        let Self {
            k,
            stream,
            h,
            weight,
            mixes,
            status,
            scale,
            base,
            pre,
            post,
            comb,
            incoming,
            collapsed,
            tokens,
            dim,
        } = self;
        k.launch_project(stream, h, weight, mixes, status, tokens, dim)?;
        k.launch_split(stream, mixes, scale, base, pre, post, comb, status, tokens)?;
        k.launch_collapse(stream, h, incoming, collapsed, status, tokens, dim)
    }
}

impl HcKernels {
    fn stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit blocking mHC stream required"
        );
        let mut flags = 0;
        let rc = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "valid blocking mHC stream required; got flags {flags}, {rc:?}"
        );
        Ok(())
    }
    fn input<T>(&self, value: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &value.device()),
            "mHC buffer from different CUDA device/context handle"
        );
        Ok(())
    }
    fn status(&self, status: &CudaSlice<u32>) -> Result<()> {
        self.input(status)?;
        ensure!(status.len() == 1, "mHC status buffer geometry");
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let this = Self { dev: dev.clone() };
        this.stream(stream)?;
        dev.load_ptx(
            Ptx::from_src(PTX),
            MODULE,
            &[
                "dsv41_hc_identity",
                "dsv41_hc_project",
                "dsv41_hc_split",
                "dsv41_hc_initial",
                "dsv41_hc_pre",
                "dsv41_hc_post",
            ],
        )?;
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_hc_identity")
                .context("mHC identity kernel")?
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), (&mut identity,))?;
        }
        dev.synchronize()?;
        let actual = dev.dtoh_sync_copy(&identity)?;
        ensure!(
            actual == [u64::from_str_radix(env!("DSV41_HC_SOURCE_ID"), 16)?, 1],
            "stale mHC PTX/source/ABI identity"
        );
        Ok(this)
    }
    pub fn initial(
        &self,
        stream: &CudaStream,
        pre: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        self.input(pre)?;
        ensure!(
            (1..=hc::MAX_TOKENS).contains(&tokens) && pre.len() == tokens * 4,
            "initial pre geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_hc_initial")
                .context("initial kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems((tokens * 4) as u32),
                    (pre, tokens as u32),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<bf16>,
        weight: &CudaSlice<f32>,
        mixes: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        self.check_project(h, weight, mixes, status, tokens, dim)?;
        self.launch_project(stream, h, weight, mixes, status, tokens, dim)
    }
    fn check_project(
        &self,
        h: &CudaSlice<bf16>,
        weight: &CudaSlice<f32>,
        mixes: &CudaSlice<f32>,
        status: &CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        hc::validate_shape(tokens, dim)?;
        self.input(h)?;
        self.input(weight)?;
        self.input(mixes)?;
        self.status(status)?;
        ensure!(
            h.len() == tokens * 4 * dim
                && weight.len() == 24 * 4 * dim
                && mixes.len() == tokens * 24,
            "mHC projection geometry"
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn launch_project(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<bf16>,
        weight: &CudaSlice<f32>,
        mixes: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_hc_project")
                .context("project kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (tokens as u32, 24, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (h, weight, mixes, status, dim as u32),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &self,
        stream: &CudaStream,
        mixes: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        pre: &mut CudaSlice<f32>,
        post: &mut CudaSlice<f32>,
        comb: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        self.check_split(mixes, scale, base, pre, post, comb, status, tokens)?;
        self.launch_split(stream, mixes, scale, base, pre, post, comb, status, tokens)
    }
    #[allow(clippy::too_many_arguments)]
    fn check_split(
        &self,
        mixes: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        pre: &CudaSlice<f32>,
        post: &CudaSlice<f32>,
        comb: &CudaSlice<f32>,
        status: &CudaSlice<u32>,
        tokens: usize,
    ) -> Result<()> {
        self.input(mixes)?;
        self.input(scale)?;
        self.input(base)?;
        self.input(pre)?;
        self.input(post)?;
        self.input(comb)?;
        self.status(status)?;
        ensure!(
            (1..=hc::MAX_TOKENS).contains(&tokens)
                && mixes.len() == tokens * 24
                && scale.len() == 3
                && base.len() == 24
                && pre.len() == tokens * 4
                && post.len() == tokens * 4
                && comb.len() == tokens * 16,
            "mHC split geometry"
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn launch_split(
        &self,
        stream: &CudaStream,
        mixes: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        pre: &mut CudaSlice<f32>,
        post: &mut CudaSlice<f32>,
        comb: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
    ) -> Result<()> {
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_hc_split")
                .context("split kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems(tokens as u32),
                    (mixes, scale, base, pre, post, comb, status, tokens as u32),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn collapse(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<bf16>,
        pre: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        self.check_collapse(h, pre, out, status, tokens, dim)?;
        self.launch_collapse(stream, h, pre, out, status, tokens, dim)
    }
    fn check_collapse(
        &self,
        h: &CudaSlice<bf16>,
        pre: &CudaSlice<f32>,
        out: &CudaSlice<bf16>,
        status: &CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        hc::validate_shape(tokens, dim)?;
        self.input(h)?;
        self.input(pre)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            h.len() == tokens * 4 * dim && pre.len() == tokens * 4 && out.len() == tokens * dim,
            "mHC collapse geometry"
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn launch_collapse(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<bf16>,
        pre: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_hc_pre")
                .context("pre kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems((tokens * dim) as u32),
                    (h, pre, out, status, tokens as u32, dim as u32),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_attention_input<'a>(
        &'a self,
        stream: &'a CudaStream,
        h: &'a CudaSlice<bf16>,
        weight: &'a CudaSlice<f32>,
        mixes: &'a mut CudaSlice<f32>,
        status: &'a mut CudaSlice<u32>,
        scale: &'a CudaSlice<f32>,
        base: &'a CudaSlice<f32>,
        pre: &'a mut CudaSlice<f32>,
        post: &'a mut CudaSlice<f32>,
        comb: &'a mut CudaSlice<f32>,
        incoming: &'a CudaSlice<f32>,
        collapsed: &'a mut CudaSlice<bf16>,
        tokens: usize,
        dim: usize,
    ) -> Result<CheckedAttentionInput<'a>> {
        self.stream(stream)?;
        self.check_project(h, weight, mixes, status, tokens, dim)?;
        self.check_split(mixes, scale, base, pre, post, comb, status, tokens)?;
        self.check_collapse(h, incoming, collapsed, status, tokens, dim)?;
        Ok(CheckedAttentionInput {
            k: self,
            stream,
            h,
            weight,
            mixes,
            status,
            scale,
            base,
            pre,
            post,
            comb,
            incoming,
            collapsed,
            tokens,
            dim,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn expand(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        residual: &CudaSlice<bf16>,
        post: &CudaSlice<f32>,
        comb: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        hc::validate_shape(tokens, dim)?;
        self.input(x)?;
        self.input(residual)?;
        self.input(post)?;
        self.input(comb)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            x.len() == tokens * dim
                && residual.len() == tokens * 4 * dim
                && post.len() == tokens * 4
                && comb.len() == tokens * 16
                && out.len() == residual.len(),
            "mHC expansion geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_hc_post")
                .context("post kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems((tokens * 4 * dim) as u32),
                    (
                        x,
                        residual,
                        post,
                        comb,
                        out,
                        status,
                        tokens as u32,
                        dim as u32,
                    ),
                )?;
        }
        Ok(())
    }
}
