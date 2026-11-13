//! Owned native1017 text head for the initial temperature-zero execution gate.
//! It consumes the final collapsed HC stream, normalizes the last position,
//! computes local F32 logits from BF16 checkpoint weights promoted to F32, and
//! requires the qualified TP8 full-score/token-ID collective before publication.
use super::{
    collectives::{RoutedLayout, RowKind},
    gpu_collectives::GpuCollectives,
    gpu_dense::{DenseOutput, DenseShape, GpuDense},
    gpu_numerics::{DeviceOutput, GpuNumerics},
    gpu_position::PositionKernels,
    gpu_text::{TextKernels, VOCAB_SHARD},
    gpu_tool_constraint::{checked_final_membership, ToolConstraintKernels},
    gpu_weights::{LoadScope, LoadedWeights},
    tool_constraint::AllowedNativeTokens,
    weights::RankManifest,
};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use half::bf16;
use std::sync::Arc;

pub struct GpuTextHead {
    dev: Arc<CudaDevice>,
    weights: Arc<LoadedWeights>,
    rank: usize,
    stream: usize,
    numerics: GpuNumerics,
    dense: GpuDense,
    position: PositionKernels,
    text: TextKernels,
    norm_f32: CudaSlice<f32>,
    init_status: CudaSlice<u32>,
    constrained: std::cell::RefCell<Option<ToolConstraintKernels>>,
}
impl GpuTextHead {
    pub fn new(
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        weights: Arc<LoadedWeights>,
        manifest: &RankManifest,
    ) -> Result<Self> {
        manifest.verify_identity()?;
        ensure!(
            weights.scope() == &LoadScope::Global
                && weights.manifest_sha256() == manifest.manifest_sha256
                && manifest.rank < 8,
            "text head source/global rank manifest mismatch"
        );
        ensure!(
            weights.bf16("head.weight")?.len() == VOCAB_SHARD * 5120
                && weights.bf16("norm.weight")?.len() == 5120,
            "text head native BF16 geometry"
        );
        let position = PositionKernels::load(dev, stream)?;
        let mut norm_f32 = dev.alloc_zeros::<f32>(5120)?;
        let mut init_status = dev.alloc_zeros::<u32>(1)?;
        // Fence the conversion before either error teardown or publication.
        let result = position.cast_bf16_f32(
            stream,
            weights.bf16("norm.weight")?,
            &mut norm_f32,
            &mut init_status,
        );
        dev.synchronize()?;
        result?;
        ensure!(
            dev.dtoh_sync_copy(&init_status)? == [0],
            "text head norm conversion failed"
        );
        Ok(Self {
            dev: dev.clone(),
            weights,
            rank: manifest.rank,
            stream: stream.stream as usize,
            numerics: GpuNumerics::load(dev, stream)?,
            dense: GpuDense::load(dev, stream)?,
            position,
            text: TextKernels::load(dev, stream)?,
            norm_f32,
            init_status,
            constrained: std::cell::RefCell::new(None),
        })
    }
    pub fn prepare<'a>(
        &'a self,
        stream: &'a CudaStream,
        hidden: &CudaSlice<bf16>,
        rows: usize,
    ) -> Result<PendingTextHead<'a>> {
        self.prepare_inner(stream, hidden, rows, None, None)
    }
    pub fn prepare_constrained<'a>(
        &'a self,
        stream: &'a Arc<CudaStream>,
        hidden: &CudaSlice<bf16>,
        rows: usize,
        mask: &'a AllowedNativeTokens,
    ) -> Result<PendingTextHead<'a>> {
        self.prepare_inner(stream, hidden, rows, Some(mask), Some(stream.clone()))
    }
    fn prepare_inner<'a>(
        &'a self,
        stream: &'a CudaStream,
        hidden: &CudaSlice<bf16>,
        rows: usize,
        mask: Option<&'a AllowedNativeTokens>,
        constraint_stream: Option<Arc<CudaStream>>,
    ) -> Result<PendingTextHead<'a>> {
        ensure!(
            stream.stream as usize == self.stream
                && Arc::ptr_eq(&self.dev, &hidden.device())
                && (1..=128).contains(&rows)
                && hidden.len() == rows * 5120,
            "text head input context/stream/shape"
        );
        // Every error fences while previous local GPU owners remain alive,
        // including errors from nested fallible weight accessors.
        macro_rules! queued {
            ($expression:expr) => {
                match (|| $expression)() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.dev.synchronize();
                        return Err(e.into());
                    }
                }
            };
        }
        let mut last = queued!(self.dev.alloc_zeros::<bf16>(5120));
        let mut status = queued!(self.dev.alloc_zeros::<u32>(1));
        queued!(self.text.last(stream, hidden, &mut last, &mut status, rows));
        let normalized =
            queued!(self
                .numerics
                .rms_norm_1017(stream, &last, &self.norm_f32, 1, 5120));
        let mut promoted = queued!(self.dev.alloc_zeros::<f32>(5120));
        queued!(self.position.cast_bf16_f32(
            stream,
            unsafe { normalized.device_values_unchecked() },
            &mut promoted,
            &mut status
        ));
        let logits = queued!(self.dense.f32_bf16_linear_f32(
            stream,
            &promoted,
            self.weights.bf16("head.weight")?,
            DenseShape {
                rows: 1,
                outputs: VOCAB_SHARD,
                inner: 5120
            }
        ));
        let mut pair = queued!(self.dev.alloc_zeros::<u32>(2));
        let native_mask = if let Some(mask) = mask {
            let gpu = queued!(self.dev.htod_sync_copy(mask.words()));
            // Lazy module loading keeps default/unconstrained constructors and
            // their old CUDA identities/launches unchanged.
            let mut module = self.constrained.borrow_mut();
            if module.is_none() {
                *module = Some(queued!(ToolConstraintKernels::load(
                    &self.dev,
                    constraint_stream
                        .context("constrained sampler requires retained stream owner")?
                )));
            }
            queued!(module.as_ref().unwrap().enqueue(
                stream,
                unsafe { logits.device_values_unchecked() },
                &gpu,
                &mut pair,
                &mut status,
                self.rank
            ));
            Some(gpu)
        } else {
            queued!(self.text.maxloc(
                stream,
                unsafe { logits.device_values_unchecked() },
                &mut pair,
                &mut status,
                self.rank
            ));
            None
        };
        // Input is not retained, so its last-row copy must finish before this
        // method returns. The remaining owned pipeline also completes here.
        queued!(self.dev.synchronize());
        queued!(normalized.check());
        queued!(logits.check());
        ensure!(
            queued!(self.dev.dtoh_sync_copy(&status)) == [0],
            "text head numeric status failed"
        );
        Ok(PendingTextHead {
            head: self,
            stream,
            pair,
            status,
            normalized,
            logits,
            _last: last,
            _promoted: promoted,
            enqueued: false,
            checked: false,
            completion_identity: None,
            constraint: mask,
            _native_mask: native_mask,
        })
    }
}
pub struct PendingTextHead<'a> {
    head: &'a GpuTextHead,
    stream: &'a CudaStream,
    pair: CudaSlice<u32>,
    status: CudaSlice<u32>,
    normalized: DeviceOutput<bf16>,
    logits: DenseOutput<f32>,
    _last: CudaSlice<bf16>,
    _promoted: CudaSlice<f32>,
    enqueued: bool,
    checked: bool,
    completion_identity: Option<[u8; 32]>,
    constraint: Option<&'a AllowedNativeTokens>,
    _native_mask: Option<CudaSlice<u32>>,
}
impl PendingTextHead<'_> {
    pub fn enqueue_selection(
        &mut self,
        collectives: &mut GpuCollectives,
        operation: u64,
    ) -> Result<()> {
        ensure!(
            !self.enqueued
                && collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.stream.stream,
            "text head selection session mismatch"
        );
        collectives.enqueue_maxloc(
            operation,
            RoutedLayout::new(1, RowKind::IndependentDecodeLanes)?,
            &mut self.pair,
        )?;
        self.enqueued = true;
        self.completion_identity = Some(collectives.completion_identity());
        Ok(())
    }
    pub fn check(&mut self, collectives: &mut GpuCollectives) -> Result<()> {
        self.checked = false;
        ensure!(
            self.enqueued
                && self.completion_identity == Some(collectives.completion_identity())
                && collectives.rank() == self.head.rank
                && collectives.compute_stream().stream == self.stream.stream,
            "text head selection incomplete/wrong session"
        );
        collectives.synchronize()?;
        self.normalized.check()?;
        self.logits.check()?;
        ensure!(
            self.head.dev.dtoh_sync_copy(&self.status)? == [0]
                && self.head.dev.dtoh_sync_copy(&self.head.init_status)? == [0],
            "text head status failure"
        );
        self.checked = true;
        Ok(())
    }
    /// Only the checked, globally selected native token reaches the CPU.
    pub fn selected_token(&self) -> Result<u32> {
        ensure!(self.checked, "global text head output was not checked");
        let pair = self.head.dev.dtoh_sync_copy(&self.pair)?;
        ensure!(
            f32::from_bits(pair[0]).is_finite() && pair[1] < 129280,
            "invalid global logit/token pair"
        );
        if let Some(mask) = self.constraint {
            checked_final_membership(mask, &pair)
        } else {
            Ok(pair[1])
        }
    }
}
impl Drop for PendingTextHead<'_> {
    fn drop(&mut self) {
        let _ = self.head.dev.synchronize();
    }
}
