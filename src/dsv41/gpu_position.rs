//! Standalone SHA-bound coefficient RoPE, embedding partials, casts and ring writes.
//! All inputs/plans must remain alive until stream completion and status0 is checked.
use super::position::{
    self, EmbeddingSpan, FrequencyBackend, FrequencyRows, RingPlan, RopeProfile,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    sys, CudaDevice, CudaSlice, CudaStream, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;
pub const PTX: &str = include_str!("../ptx/dsv41_position.ptx");
const MODULE: &str = "dsv41_position_v1";
pub struct GpuFrequencyRows {
    profile: RopeProfile,
    backend: FrequencyBackend,
    positions: Vec<u32>,
    coeff: CudaSlice<f32>,
    table_sha: &'static str,
    complete_table_verified: bool,
}
impl GpuFrequencyRows {
    pub fn profile(&self) -> RopeProfile {
        self.profile
    }
    pub fn backend(&self) -> FrequencyBackend {
        self.backend
    }
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }
    pub fn table_sha(&self) -> &'static str {
        self.table_sha
    }
    pub fn complete_table_verified(&self) -> bool {
        self.complete_table_verified
    }
}
pub struct GpuRingPlan {
    plan: RingPlan,
    descriptors: CudaSlice<u64>,
    max_count: usize,
}
pub struct PositionKernels {
    dev: Arc<CudaDevice>,
}
impl PositionKernels {
    fn stream(&self, s: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(!s.stream.is_null(), "explicit blocking stream required");
        let mut flags = 0;
        let rc = unsafe { sys::cuStreamGetFlags(s.stream, &mut flags) };
        ensure!(
            rc == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "valid blocking stream required"
        );
        Ok(())
    }
    fn input<T>(&self, x: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &x.device()),
            "position buffer device/context differs"
        );
        Ok(())
    }
    fn status(&self, x: &CudaSlice<u32>) -> Result<()> {
        self.input(x)?;
        ensure!(x.len() == 1, "position status extent");
        Ok(())
    }
    pub fn load(dev: &Arc<CudaDevice>, s: &CudaStream) -> Result<Self> {
        let me = Self { dev: dev.clone() };
        me.stream(s)?;
        dev.load_ptx(
            Ptx::from_src(PTX),
            MODULE,
            &[
                "dsv41_position_identity",
                "dsv41_position_rope",
                "dsv41_position_embedding",
                "dsv41_position_bf16_f32",
                "dsv41_position_f32_bf16",
                "dsv41_position_expand",
                "dsv41_position_ring",
            ],
        )?;
        let mut id = dev.alloc_zeros::<u64>(2)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_position_identity")
                .context("position identity")?
                .launch_on_stream(s, LaunchConfig::for_num_elems(1), (&mut id,))?;
        }
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&id)?
                == [
                    u64::from_str_radix(env!("DSV41_POSITION_SOURCE_ID"), 16)?,
                    1
                ],
            "position source/ABI mismatch"
        );
        Ok(me)
    }
    pub fn upload_frequencies(
        &self,
        s: &CudaStream,
        f: &FrequencyRows,
    ) -> Result<GpuFrequencyRows> {
        self.stream(s)?;
        Ok(GpuFrequencyRows {
            profile: f.profile(),
            backend: f.backend(),
            positions: f.positions().to_vec(),
            coeff: self.dev.htod_sync_copy(f.coefficients())?,
            table_sha: f.table_sha(),
            complete_table_verified: f.complete_table_verified(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rotate(
        &self,
        s: &CudaStream,
        x: &mut CudaSlice<bf16>,
        f: &GpuFrequencyRows,
        profile: RopeProfile,
        positions: &[u32],
        status: &mut CudaSlice<u32>,
        b: usize,
        m: usize,
        heads: usize,
        width: usize,
        inverse: bool,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(x)?;
        self.input(&f.coeff)?;
        self.status(status)?;
        ensure!(
            f.profile == profile && f.positions == positions,
            "wrong table/profile/query-prefix position list"
        );
        ensure!(
            (1..=16).contains(&b)
                && (1..=128).contains(&m)
                && (1..=64).contains(&heads)
                && (64..=512).contains(&width)
                && width % 2 == 0
                && positions.len() == b * m
                && f.coeff.len() == positions.len() * 64
                && x.len() == b * m * heads * width,
            "position RoPE shape"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_rope")
                .context("RoPE kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems((b * m * heads * 32) as u32),
                    (
                        x,
                        &f.coeff,
                        status,
                        (b * m) as u32,
                        heads as u32,
                        width as u32,
                        u32::from(inverse),
                    ),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn embedding(
        &self,
        s: &CudaStream,
        w: &CudaSlice<bf16>,
        ids: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        span: EmbeddingSpan,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(w)?;
        self.input(ids)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            (1..=16 * 128).contains(&ids.len())
                && w.len() == span.rows() * span.dim()
                && out.len() == ids.len() * span.dim(),
            "embedding buffer extent"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_embedding")
                .context("embedding kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems(out.len() as u32),
                    (
                        w,
                        ids,
                        out,
                        status,
                        ids.len() as u32,
                        span.dim() as u32,
                        span.rank() as u32,
                        span.first() as u32,
                        span.rows() as u32,
                    ),
                )?;
        }
        Ok(())
    }
    pub fn cast_bf16_f32(
        &self,
        s: &CudaStream,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(x)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            !x.is_empty() && x.len() <= position::MAX_BASIC_ELEMENTS && out.len() == x.len(),
            "cast extent"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_bf16_f32")
                .context("cast kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems(x.len() as u32),
                    (x, out, status, x.len() as u32),
                )?;
        }
        Ok(())
    }
    pub fn cast_f32_bf16(
        &self,
        s: &CudaStream,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(x)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            !x.is_empty() && x.len() <= position::MAX_BASIC_ELEMENTS && out.len() == x.len(),
            "cast extent"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_f32_bf16")
                .context("cast kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems(x.len() as u32),
                    (x, out, status, x.len() as u32),
                )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn expand(
        &self,
        s: &CudaStream,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(x)?;
        self.input(out)?;
        self.status(status)?;
        ensure!(
            (1..=16 * 128).contains(&tokens)
                && (8..=5120).contains(&dim)
                && dim % 8 == 0
                && x.len() == tokens * dim
                && out.len() == tokens * 4 * dim,
            "embedding HC4 expansion extent"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_expand")
                .context("HC4 expand kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig::for_num_elems(out.len() as u32),
                    (x, out, status, tokens as u32, dim as u32),
                )?;
        }
        Ok(())
    }
    pub fn upload_ring_plan(&self, s: &CudaStream, p: &RingPlan) -> Result<GpuRingPlan> {
        self.stream(s)?;
        let data = p
            .writes()
            .iter()
            .flat_map(|w| {
                [
                    w.lane as u64,
                    w.start_pos as u64,
                    w.count as u64,
                    w.input_row as u64,
                ]
            })
            .collect::<Vec<_>>();
        Ok(GpuRingPlan {
            plan: p.clone(),
            descriptors: self.dev.htod_sync_copy(&data)?,
            max_count: p.writes().iter().map(|w| w.count).max().unwrap(),
        })
    }
    pub fn write_ring(
        &self,
        s: &CudaStream,
        input: &CudaSlice<bf16>,
        cache: &mut CudaSlice<bf16>,
        p: &GpuRingPlan,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        self.stream(s)?;
        self.input(input)?;
        self.input(cache)?;
        self.input(&p.descriptors)?;
        self.status(status)?;
        ensure!(
            input.len() == p.plan.input_rows() * 512 && cache.len() == p.plan.lanes() * 128 * 512,
            "ring storage extent"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_position_ring")
                .context("ring write kernel")?
                .launch_on_stream(
                    s,
                    LaunchConfig {
                        grid_dim: (p.max_count as u32, p.plan.writes().len() as u32, 1),
                        block_dim: (512, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        input,
                        cache,
                        &p.descriptors,
                        status,
                        p.plan.lanes() as u32,
                        p.plan.input_rows() as u32,
                    ),
                )?;
        }
        Ok(())
    }
}
