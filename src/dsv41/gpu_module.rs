//! Raw CUDA module owner for isolated1017 kernels needing dynamic shared memory.
//! Every module has its own source/ABI handshake and explicit blocking stream.
use anyhow::{anyhow, ensure, Context, Result};
use cudarc::driver::{result, sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::Arc;

pub struct Module {
    dev: Arc<CudaDevice>,
    module: sys::CUmodule,
    functions: BTreeMap<&'static str, sys::CUfunction>,
}

impl Module {
    pub fn load(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        ptx: &str,
        names: &[&'static str],
        stamp: &'static str,
        source_id: u64,
        abi: u64,
    ) -> Result<Self> {
        dev.bind_to_thread()?;
        let text = CString::new(ptx)?;
        let module = unsafe { result::module::load_data(text.as_ptr() as *const _) }?;
        let mut me = Self {
            dev: dev.clone(),
            module,
            functions: BTreeMap::new(),
        };
        me.check_stream(stream)?;
        for name in names.iter().copied().chain(std::iter::once(stamp)) {
            let function = unsafe { result::module::get_function(module, CString::new(name)?) }?;
            me.functions.insert(name, function);
        }
        let mut output = dev.alloc_zeros::<u64>(2)?;
        let mut pointer = *output.device_ptr();
        let mut params = [&mut pointer as *mut _ as *mut c_void];
        unsafe { me.launch(stamp, stream, (1, 1, 1), (1, 1, 1), 0, &mut params) }?;
        dev.synchronize()?;
        ensure!(
            dev.dtoh_sync_copy(&output)? == [source_id, abi],
            "isolated CUDA source/ABI mismatch"
        );
        // Keep the owner alive until after the synchronous identity read.
        let _ = &mut output;
        Ok(me)
    }

    pub fn check_stream(&self, stream: &CudaStream) -> Result<()> {
        self.dev.bind_to_thread()?;
        ensure!(
            !stream.stream.is_null(),
            "explicit blocking CUDA stream required"
        );
        let mut flags = 0;
        let status = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            status == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "invalid/nonblocking CUDA stream"
        );
        Ok(())
    }

    pub fn check_buffer<T>(&self, buffer: &CudaSlice<T>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.dev, &buffer.device()),
            "buffer device/context differs"
        );
        Ok(())
    }

    pub fn set_dynamic_smem(&self, name: &str, bytes: usize) -> Result<()> {
        self.dev.bind_to_thread()?;
        let function = *self
            .functions
            .get(name)
            .context("missing dynamic-smem kernel")?;
        let bytes = i32::try_from(bytes)?;
        let status = unsafe {
            sys::cuFuncSetAttribute(
                function,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        };
        ensure!(
            status == sys::CUresult::CUDA_SUCCESS,
            "dynamic shared-memory opt-in failed: {status:?}"
        );
        Ok(())
    }

    /// Caller validates dimensions, buffer extents and parameter types/lifetimes.
    pub unsafe fn launch(
        &self,
        name: &str,
        stream: &CudaStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.check_stream(stream)?;
        ensure!(
            grid.0 > 0 && grid.1 > 0 && grid.2 > 0 && block.0 > 0 && block.1 > 0 && block.2 > 0,
            "zero CUDA launch geometry"
        );
        let function = *self
            .functions
            .get(name)
            .with_context(|| format!("missing kernel{name}"))?;
        result::launch_kernel(function, grid, block, smem, stream.stream, params)
            .map_err(|e| anyhow!("isolated {name} launch: {e}"))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        let _ = self.dev.bind_to_thread();
        // Owner must complete outstanding work before dropping buffers/modules.
        let _ = self.dev.synchronize();
        let _ = unsafe { result::module::unload(self.module) };
    }
}
