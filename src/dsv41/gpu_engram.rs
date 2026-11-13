//! Model-free native Engram lookup/gate kernels. wkv and model integration remain
//! separate. Caller owns an explicitly blocking compute stream and all buffers.
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaStream;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;

const PTX: &str = include_str!("../ptx/gpu_dsv41_engram.ptx");
const MODULE: &str = "dsv41_engram_v1";

pub struct EngramKernels {
    dev: Arc<CudaDevice>,
}

impl EngramKernels {
    fn stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit blocking stream required"
        );
        let mut flags = 0;
        let result = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            result == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "valid blocking stream required"
        );
        Ok(())
    }

    fn input<T>(&self, input: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &input.device()),
            "Engram buffer has a different CUDA device/context handle"
        );
        Ok(())
    }

    pub fn load(dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<Self> {
        let me = Self { dev: dev.clone() };
        me.stream(stream)?;
        dev.load_ptx(
            Ptx::from_src(PTX),
            MODULE,
            &[
                "dsv41_engram_identity",
                "dsv41_engram_lookup",
                "dsv41_engram_gate",
            ],
        )?;
        let mut identity = dev.alloc_zeros::<u64>(2)?;
        unsafe {
            dev.get_func(MODULE, "dsv41_engram_identity")
                .context("identity kernel")?
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), (&mut identity,))?;
        }
        dev.synchronize()?;
        let actual = dev.dtoh_sync_copy(&identity)?;
        let expected = u64::from_str_radix(env!("DSV41_ENGRAM_SOURCE_ID"), 16)?;
        ensure!(actual == [expected, 1], "stale Engram kernel/ABI");
        Ok(me)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lookup(
        &self,
        stream: &CudaStream,
        weights: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        indices: &CudaSlice<u64>,
        out: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        first: u64,
        rows: u64,
        total: u64,
    ) -> Result<()> {
        self.stream(stream)?;
        self.input(weights)?;
        self.input(scales)?;
        self.input(indices)?;
        self.input(out)?;
        self.input(status)?;
        ensure!(
            first.checked_add(rows).context("row range overflow")? <= total,
            "row range outside table"
        );
        let rows_usize = usize::try_from(rows)?;
        ensure!(
            weights.len()
                == rows_usize
                    .checked_mul(256)
                    .context("weight extent overflow")?
                && scales.len() == rows_usize.checked_mul(8).context("scale extent overflow")?,
            "native table geometry"
        );
        let count = indices.len();
        ensure!(
            (1..=128 * 24).contains(&count) && out.len() == count * 256 && status.len() == 1,
            "lookup buffer geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_engram_lookup")
                .context("lookup kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig::for_num_elems((count * 256) as u32),
                    (
                        weights,
                        scales,
                        indices,
                        out,
                        status,
                        first,
                        rows,
                        total,
                        count as u32,
                    ),
                )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gate(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<bf16>,
        kv: &CudaSlice<bf16>,
        qw: &CudaSlice<bf16>,
        kw: &CudaSlice<bf16>,
        mask: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
        gates: &mut CudaSlice<f32>,
        status: &mut CudaSlice<u32>,
        tokens: usize,
        dim: usize,
    ) -> Result<()> {
        self.stream(stream)?;
        self.input(h)?;
        self.input(kv)?;
        self.input(qw)?;
        self.input(kw)?;
        self.input(mask)?;
        self.input(out)?;
        self.input(gates)?;
        self.input(status)?;
        ensure!(
            (1..=128).contains(&tokens) && (32..=5120).contains(&dim) && dim % 32 == 0,
            "gate dimensions"
        );
        ensure!(
            h.len() == tokens * 4 * dim
                && kv.len() == tokens * 5 * dim
                && qw.len() == 4 * dim
                && kw.len() == 4 * dim
                && mask.len() == tokens
                && out.len() == h.len()
                && gates.len() == tokens * 4
                && status.len() == 1,
            "gate buffer geometry"
        );
        unsafe {
            self.dev
                .get_func(MODULE, "dsv41_engram_gate")
                .context("gate kernel")?
                .launch_on_stream(
                    stream,
                    LaunchConfig {
                        grid_dim: (tokens as u32, 4, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (h, kv, qw, kw, mask, out, gates, status, dim as u32),
                )?;
        }
        Ok(())
    }
}
