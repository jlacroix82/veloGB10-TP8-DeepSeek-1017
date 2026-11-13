//! One checked readback boundary for the packed attention projection chain.
//! No status is combined, skipped, cleared, or allowed past a failed cut.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use serde::Serialize;
use std::{
    ffi::c_void,
    mem::size_of,
    ptr::NonNull,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

const WORDS: usize = 10;
const LABELS: [&str; WORDS] = [
    "input quantization",
    "query A",
    "query RMS",
    "query quantization",
    "query B",
    "window projection",
    "window RMS",
    "window quantization",
    "window reconstruction",
    "query/window rotation",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectionStatusBackend {
    #[default]
    Original,
    PinnedSingleFence,
}

impl ProjectionStatusBackend {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "original" => Ok(Self::Original),
            "pinned-single-fence" => Ok(Self::PinnedSingleFence),
            _ => anyhow::bail!("unsupported packed projection status backend {name}"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ProjectionStatusSnapshot {
    pub successful_cuts: u64,
    pub checked_individual_words: u64,
    pub poisoned: bool,
    pub pinned_host_bytes: usize,
}

struct HostWords(NonNull<u32>);
// The storage is accessed only under the cut's mutex. The thread binds the
// exact CUDA context first, and never reads the destination until its owned
// stream has completed the ordered copies.
unsafe impl Send for HostWords {}

struct State {
    words: HostWords,
    owner: Option<[u8; 32]>,
    successful_cuts: u64,
    poisoned: bool,
}

pub(crate) struct ProjectionStatusCut {
    dev: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    boot: Uuid,
    state: Mutex<State>,
}

fn checked_words(values: &[u32; WORDS]) -> Result<()> {
    for (label, value) in LABELS.iter().zip(values) {
        ensure!(
            *value == 0,
            "packed projection {label} GPU status0x{value:x}"
        );
    }
    Ok(())
}

impl ProjectionStatusCut {
    pub(crate) fn new(dev: &Arc<CudaDevice>, stream: Arc<CudaStream>, boot: Uuid) -> Result<Self> {
        ensure!(
            !boot.is_nil() && !stream.stream.is_null(),
            "projection status cut needs the actual boot and stream"
        );
        dev.bind_to_thread()?;
        let mut flags = 0;
        let flag_result = unsafe { sys::cuStreamGetFlags(stream.stream, &mut flags) };
        ensure!(
            flag_result == sys::CUresult::CUDA_SUCCESS && flags == 0,
            "projection status cut requires the original blocking stream"
        );
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { sys::cuMemHostAlloc(&mut ptr, WORDS * size_of::<u32>(), 0) };
        ensure!(
            result == sys::CUresult::CUDA_SUCCESS,
            "projection pinned status allocation failed: {result:?}"
        );
        let words = HostWords(NonNull::new(ptr.cast()).context("null pinned status allocation")?);
        Ok(Self {
            dev: dev.clone(),
            stream,
            boot,
            state: Mutex::new(State {
                words,
                owner: None,
                successful_cuts: 0,
                poisoned: false,
            }),
        })
    }

    pub(crate) fn bind_owner(&self, owner: [u8; 32]) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("projection status lock poisoned"))?;
        ensure!(
            owner != [0; 32] && state.owner.is_none() && !state.poisoned,
            "projection status cut owner can only bind once"
        );
        state.owner = Some(owner);
        Ok(())
    }

    pub(crate) fn check(
        &self,
        owner: [u8; 32],
        boot: Uuid,
        stream: &Arc<CudaStream>,
        sources: [&CudaSlice<u32>; WORDS],
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("projection status lock poisoned"))?;
        ensure!(
            !state.poisoned
                && state.owner == Some(owner)
                && boot == self.boot
                && Arc::ptr_eq(stream, &self.stream),
            "projection status cut is poisoned or crossed owner/boot/stream"
        );
        state.poisoned = true;
        let next = state
            .successful_cuts
            .checked_add(1)
            .context("projection status cut counter exhausted")?;
        next.checked_mul(WORDS as u64)
            .context("projection status word counter exhausted")?;
        let mut pointers = [0; WORDS];
        for (slot, source) in sources.iter().enumerate() {
            let address = *source.device_ptr();
            ensure!(
                source.len() == 1
                    && Arc::ptr_eq(&source.device(), &self.dev)
                    && address != 0
                    && !pointers[..slot].contains(&address),
                "projection status source must be a distinct owned one-word allocation"
            );
            pointers[slot] = address;
        }
        self.dev.bind_to_thread()?;
        for (slot, source) in sources.iter().enumerate() {
            // No host read, clear or free of either allocation occurs until
            // the exact producer stream is fenced. The first nine sources
            // and the rotation status are still owned by the caller.
            let result = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    state.words.0.as_ptr().add(slot).cast(),
                    *source.device_ptr(),
                    size_of::<u32>(),
                    self.stream.stream,
                )
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                let fence = unsafe { sys::cuStreamSynchronize(self.stream.stream) };
                anyhow::bail!(
                    "projection status enqueue failed at {}: {result:?}; fence={fence:?}",
                    LABELS[slot]
                );
            }
        }
        let result = unsafe { sys::cuStreamSynchronize(self.stream.stream) };
        ensure!(
            result == sys::CUresult::CUDA_SUCCESS,
            "projection status stream fence failed: {result:?}"
        );
        let values = unsafe { state.words.0.cast::<[u32; WORDS]>().as_ptr().read() };
        checked_words(&values)?;
        state.successful_cuts = next;
        state.poisoned = false;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> ProjectionStatusSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        ProjectionStatusSnapshot {
            successful_cuts: state.successful_cuts,
            checked_individual_words: state.successful_cuts * WORDS as u64,
            poisoned: state.poisoned || self.state.is_poisoned(),
            pinned_host_bytes: WORDS * size_of::<u32>(),
        }
    }
}

impl Drop for ProjectionStatusCut {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        // A failed driver fence cannot prove DMA retired. Preserve the tiny
        // allocation until process teardown rather than free a live target.
        if self.dev.bind_to_thread().is_ok()
            && unsafe { sys::cuStreamSynchronize(self.stream.stream) }
                == sys::CUresult::CUDA_SUCCESS
        {
            let _ = unsafe { sys::cuMemFreeHost(state.words.0.as_ptr().cast()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_word_stays_separate_and_fail_closed() {
        assert_eq!(
            ProjectionStatusBackend::default(),
            ProjectionStatusBackend::Original
        );
        assert!(ProjectionStatusBackend::parse("pinned").is_err());
        assert!(checked_words(&[0; WORDS]).is_ok());
        for (slot, label) in LABELS.iter().enumerate() {
            for code in [1, 2, 4, 8, 0x8000_0000] {
                let mut values = [0; WORDS];
                values[slot] = code;
                let error = checked_words(&values).unwrap_err().to_string();
                assert!(error.contains(label));
                assert!(error.contains(&format!("0x{code:x}")));
            }
        }
    }

    #[test]
    #[ignore = "model-free CUDA oracle; run only in a reviewed exclusive guarded GPU window"]
    fn device_copy_order_poison_and_owner_scope() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let stream = Arc::new(crate::gpu::fork_blocking_stream(&dev));
        let boot = Uuid::new_v4();
        let owner = [3; 32];
        let cut = ProjectionStatusCut::new(&dev, stream.clone(), boot)?;
        cut.bind_owner(owner)?;
        assert!(cut.bind_owner([4; 32]).is_err());
        let clean = (0..WORDS)
            .map(|_| dev.alloc_zeros::<u32>(1))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let words = std::array::from_fn(|index| &clean[index]);
        assert!(cut.check([4; 32], boot, &stream, words).is_err());
        assert!(cut.check(owner, Uuid::new_v4(), &stream, words).is_err());
        assert!(cut
            .check(
                owner,
                boot,
                &Arc::new(crate::gpu::fork_blocking_stream(&dev)),
                words
            )
            .is_err());
        cut.check(owner, boot, &stream, words)?;
        cut.check(owner, boot, &stream, words)?;
        let receipt = cut.snapshot();
        assert_eq!(receipt.successful_cuts, 2);
        assert_eq!(receipt.checked_individual_words, 20);
        assert!(!receipt.poisoned);
        let bad = dev.htod_sync_copy(&[4_u32])?;
        let mut marked = words;
        marked[8] = &bad;
        assert!(cut
            .check(owner, boot, &stream, marked)
            .unwrap_err()
            .to_string()
            .contains("window reconstruction GPU status0x4"));
        assert!(cut.snapshot().poisoned);
        assert_eq!(cut.snapshot().successful_cuts, 2);
        assert!(cut.check(owner, boot, &stream, words).is_err());
        Ok(())
    }
}
