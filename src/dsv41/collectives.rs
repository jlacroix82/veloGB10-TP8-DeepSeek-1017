//! CPU reference and reference-frame ABI for native 1017 routed FP32 partials.
//!
//! This module performs no GPU work and enables no serving backend. The checked
//! envelope is for reference artifacts/control validation; the existing native
//! RoCE executor still carries raw FP32 payloads and its own generation tags.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const WORLD: usize = 8;
pub const HIDDEN: usize = 5120;
pub const EXPERTS: u64 = 384;
pub const VOCAB: u32 = 129_280;
pub const ROUNDS: usize = 3;
pub const MAX_DECODE_LANES: usize = 16;
pub const MAX_REFERENCE_ROWS: usize = 128;
const MAGIC: &[u8; 8] = b"DS41F32\0";
const VERSION: u32 = 1;
const PREFIX_BYTES: usize = 56;
const HEADER_BYTES: usize = PREFIX_BYTES + 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RowKind {
    IndependentDecodeLanes,
    ContiguousPrefillRows,
}

impl RowKind {
    fn tag(self) -> u32 {
        match self {
            Self::IndependentDecodeLanes => 1,
            Self::ContiguousPrefillRows => 2,
        }
    }
    fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            1 => Ok(Self::IndependentDecodeLanes),
            2 => Ok(Self::ContiguousPrefillRows),
            _ => anyhow::bail!("unknown routed row kind {tag}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RoutedLayout {
    rows: usize,
    kind: RowKind,
}

impl RoutedLayout {
    pub fn new(rows: usize, kind: RowKind) -> Result<Self> {
        Self::from_geometry(rows, HIDDEN, WORLD, kind)
    }
    pub fn from_geometry(rows: usize, hidden: usize, world: usize, kind: RowKind) -> Result<Self> {
        ensure!(
            world == WORLD && hidden == HIDDEN,
            "1017 reference requires world8/hidden5120"
        );
        ensure!(
            rows > 0 && rows <= MAX_REFERENCE_ROWS,
            "reference rows must be in 1..=128"
        );
        if kind == RowKind::IndependentDecodeLanes {
            ensure!(
                rows <= MAX_DECODE_LANES,
                "independent decode reference is bounded to 16 lanes"
            );
        }
        rows.checked_mul(HIDDEN)
            .and_then(|n| n.checked_mul(4))
            .context("routed payload size overflow")?;
        Ok(Self { rows, kind })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn kind(self) -> RowKind {
        self.kind
    }
    pub fn elements(self) -> usize {
        self.rows * HIDDEN
    }
    pub fn payload_bytes(self) -> usize {
        self.elements() * 4
    }
    pub fn chunk_count(self) -> usize {
        self.elements().div_ceil(crate::tp_xport::EPOCH_FLOATS)
    }
    pub fn expert_owner(self, expert: u64) -> Result<usize> {
        super::plan::expert_owner(EXPERTS, WORLD, expert)
    }
}

#[derive(Clone, Debug)]
pub struct RoutedFrame {
    layout: RoutedLayout,
    rank: usize,
    operation: u64,
    values: Vec<f32>,
}

fn finite(values: &[f32], label: &str) -> Result<()> {
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite/corrupt FP32 values in {label}"
    );
    Ok(())
}

impl RoutedFrame {
    pub fn new(
        layout: RoutedLayout,
        rank: usize,
        operation: u64,
        values: Vec<f32>,
    ) -> Result<Self> {
        ensure!(rank < WORLD, "source rank {rank} outside world8");
        ensure!(
            values.len() == layout.elements(),
            "partial length differs from routed layout"
        );
        finite(&values, "source frame")?;
        Ok(Self {
            layout,
            rank,
            operation,
            values,
        })
    }
    pub fn layout(&self) -> RoutedLayout {
        self.layout
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn operation(&self) -> u64 {
        self.operation
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Versioned, little-endian CPU artifact. The digest covers the metadata
    /// prefix and all raw float bits, including signed zero. This envelope is
    /// deliberately not injected into the native collective's payload ring.
    pub fn encode(&self) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(PREFIX_BYTES);
        prefix.extend_from_slice(MAGIC);
        for value in [
            VERSION,
            WORLD as u32,
            self.rank as u32,
            self.layout.kind.tag(),
        ] {
            prefix.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            self.layout.rows as u64,
            HIDDEN as u64,
            self.operation,
            self.layout.payload_bytes() as u64,
        ] {
            prefix.extend_from_slice(&value.to_le_bytes());
        }
        debug_assert_eq!(prefix.len(), PREFIX_BYTES);
        let mut payload = Vec::with_capacity(self.layout.payload_bytes());
        for value in &self.values {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let mut digest = Sha256::new();
        digest.update(&prefix);
        digest.update(&payload);
        let mut bytes = prefix;
        bytes.extend_from_slice(&digest.finalize());
        bytes.extend_from_slice(&payload);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= HEADER_BYTES && &bytes[..8] == MAGIC,
            "invalid routed reference frame magic/length"
        );
        let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        ensure!(u32_at(8) == VERSION, "unsupported routed frame version");
        let kind = RowKind::from_tag(u32_at(20))?;
        let layout = RoutedLayout::from_geometry(
            usize::try_from(u64_at(24))?,
            usize::try_from(u64_at(32))?,
            u32_at(12) as usize,
            kind,
        )?;
        ensure!(
            u64_at(48) == layout.payload_bytes() as u64
                && bytes.len() == HEADER_BYTES + layout.payload_bytes(),
            "routed frame payload length mismatch"
        );
        let payload = &bytes[HEADER_BYTES..];
        let mut digest = Sha256::new();
        digest.update(&bytes[..PREFIX_BYTES]);
        digest.update(payload);
        ensure!(
            digest.finalize().as_slice() == &bytes[PREFIX_BYTES..HEADER_BYTES],
            "routed reference frame checksum mismatch"
        );
        let values = payload
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .collect();
        Self::new(layout, u32_at(16) as usize, u64_at(40), values)
    }
}

/// Native K1 increments the epoch before choosing epoch%3's XOR partner. At an
/// aligned start the actual order is 1,2,0, rather than assuming 0,1,2.
pub fn round_order(start_epoch: u64) -> Result<[usize; ROUNDS]> {
    let mut order = [0; ROUNDS];
    for (step, round) in order.iter_mut().enumerate() {
        *round = (start_epoch
            .checked_add(step as u64 + 1)
            .context("collective epoch overflow")?
            % ROUNDS as u64) as usize;
    }
    Ok(order)
}

fn aligned_epoch(start_epoch: u64) -> Result<()> {
    ensure!(
        start_epoch % ROUNDS as u64 == 0,
        "logical collective must start at epoch%3==0"
    );
    Ok(())
}

#[derive(Debug)]
pub struct RoutedReference {
    pub layout: RoutedLayout,
    pub operation: u64,
    pub start_epoch: u64,
    pub end_epoch: u64,
    pub chunk_count: usize,
    pub ranks: Vec<Vec<f32>>,
}

/// Consume the production serial schedule, using each epoch's real XOR bit and
/// canonical lower-group+upper-group FP32 order. Every intermediate stays FP32;
/// arithmetic overflow is refused rather than becoming plausible output data.
pub fn reduce_reference(frames: &[RoutedFrame], start_epoch: u64) -> Result<RoutedReference> {
    aligned_epoch(start_epoch)?;
    ensure!(
        frames.len() == WORLD,
        "a routed reduce requires all 8 rank frames"
    );
    let first = &frames[0];
    let layout = first.layout;
    let operation = first.operation;
    let mut state: Vec<Vec<f32>> = vec![Vec::new(); WORLD];
    let mut seen = [false; WORLD];
    for frame in frames {
        ensure!(
            frame.rank < WORLD
                && frame.layout == layout
                && frame.operation == operation
                && !seen[frame.rank],
            "invalid rank or mixed operation/layout in routed frames"
        );
        finite(&frame.values, "rank partial")?;
        seen[frame.rank] = true;
        state[frame.rank] = frame.values.clone();
    }
    let schedule = crate::tp_xport::serial_plan(layout.elements(), ROUNDS);
    let end_epoch = start_epoch
        .checked_add(schedule.len() as u64)
        .context("collective epoch overflow")?;
    let mut epoch = start_epoch;
    for pair in schedule {
        epoch += 1;
        let bit = 1_usize << (epoch % ROUNDS as u64);
        let end = pair.off + pair.len;
        let mut stage = Vec::with_capacity(WORLD);
        for rank in 0..WORLD {
            let lower = rank & !bit;
            let upper = lower | bit;
            let row: Vec<f32> = (pair.off..end)
                .map(|i| state[lower][i] + state[upper][i])
                .collect();
            finite(&row, "FP32 intermediate")?;
            stage.push(row);
        }
        for rank in 0..WORLD {
            state[rank][pair.off..end].copy_from_slice(&stage[rank]);
        }
    }
    debug_assert_eq!(epoch, end_epoch);
    ensure!(
        end_epoch % ROUNDS as u64 == 0,
        "serial reduction lost its epoch phase"
    );
    Ok(RoutedReference {
        layout,
        operation,
        start_epoch,
        end_epoch,
        chunk_count: layout.chunk_count(),
        ranks: state,
    })
}

pub const MAX_MATRIX_ELEMENTS: usize = 4 * 1024 * 1024;
pub const MAX_INDEX_COLUMNS: usize = 131_072;
const MATRIX_MAGIC: &[u8; 8] = b"DS41MAT\0";
const MATRIX_PREFIX: usize = 64;
const MATRIX_HEADER: usize = MATRIX_PREFIX + 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MatrixRole {
    EngramLookup,
    IndexScore,
}
impl MatrixRole {
    pub fn tag(self) -> u32 {
        match self {
            Self::EngramLookup => 1,
            Self::IndexScore => 2,
        }
    }
    fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            1 => Ok(Self::EngramLookup),
            2 => Ok(Self::IndexScore),
            _ => anyhow::bail!("unsupported FP32 matrix role {tag}"),
        }
    }
}

/// Explicit non-routed FP32 payload. Engram rows contain all24*256 lookup
/// columns; index-score width is a fixed, pre-agreed bucket. Padding must remain
/// finite (+0 is suitable); causal/candidate -Inf masks are applied AFTER reduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MatrixF32 {
    role: MatrixRole,
    rows: usize,
    cols: usize,
}
impl MatrixF32 {
    pub fn new(role: MatrixRole, rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_REFERENCE_ROWS).contains(&rows),
            "FP32 matrix row bound is 1..=128"
        );
        match role {
            MatrixRole::EngramLookup => {
                ensure!(cols == 24 * 256, "native1017 Engram lookup has6144 columns")
            }
            MatrixRole::IndexScore => ensure!(
                (1..=MAX_INDEX_COLUMNS).contains(&cols),
                "index-score width outside bound"
            ),
        }
        ensure!(
            rows.checked_mul(cols)
                .is_some_and(|n| n <= MAX_MATRIX_ELEMENTS),
            "FP32 matrix element bound exceeded"
        );
        Ok(Self { role, rows, cols })
    }
    pub fn role(self) -> MatrixRole {
        self.role
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn cols(self) -> usize {
        self.cols
    }
    pub fn elements(self) -> usize {
        self.rows * self.cols
    }
    pub fn payload_bytes(self) -> usize {
        self.elements() * 4
    }
    pub fn chunk_count(self) -> usize {
        self.elements().div_ceil(crate::tp_xport::EPOCH_FLOATS)
    }
}

#[derive(Clone, Debug)]
pub struct MatrixFrame {
    layout: MatrixF32,
    rank: usize,
    step: u64,
    operation: u64,
    values: Vec<f32>,
}
impl MatrixFrame {
    pub fn new(
        layout: MatrixF32,
        rank: usize,
        step: u64,
        operation: u64,
        values: Vec<f32>,
    ) -> Result<Self> {
        ensure!(
            rank < WORLD && values.len() == layout.elements(),
            "FP32 matrix source rank/extent mismatch"
        );
        finite(&values, "matrix partial")?;
        Ok(Self {
            layout,
            rank,
            step,
            operation,
            values,
        })
    }
    pub fn layout(&self) -> MatrixF32 {
        self.layout
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn step(&self) -> u64 {
        self.step
    }
    pub fn operation(&self) -> u64 {
        self.operation
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    /// CPU artifact envelope only; native RoCE still sends raw FP32 + its epoch tail.
    pub fn encode(&self) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(MATRIX_PREFIX);
        prefix.extend_from_slice(MATRIX_MAGIC);
        for x in [1, WORLD as u32, self.rank as u32, self.layout.role.tag()] {
            prefix.extend_from_slice(&x.to_le_bytes());
        }
        for x in [
            self.layout.rows as u64,
            self.layout.cols as u64,
            self.operation,
            self.step,
            self.layout.payload_bytes() as u64,
        ] {
            prefix.extend_from_slice(&x.to_le_bytes());
        }
        debug_assert_eq!(prefix.len(), MATRIX_PREFIX);
        let mut body = Vec::with_capacity(self.layout.payload_bytes());
        for value in &self.values {
            body.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let mut hash = Sha256::new();
        hash.update(&prefix);
        hash.update(&body);
        prefix.extend_from_slice(&hash.finalize());
        prefix.extend_from_slice(&body);
        prefix
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= MATRIX_HEADER && &bytes[..8] == MATRIX_MAGIC,
            "invalid matrix artifact envelope"
        );
        let u32_at = |p| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        let u64_at = |p| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
        ensure!(
            u32_at(8) == 1 && u32_at(12) == WORLD as u32,
            "matrix artifact version/world mismatch"
        );
        let layout = MatrixF32::new(
            MatrixRole::from_tag(u32_at(20))?,
            usize::try_from(u64_at(24))?,
            usize::try_from(u64_at(32))?,
        )?;
        ensure!(
            u64_at(56) == layout.payload_bytes() as u64
                && bytes.len() == MATRIX_HEADER + layout.payload_bytes(),
            "matrix artifact payload geometry mismatch"
        );
        let mut hash = Sha256::new();
        hash.update(&bytes[..MATRIX_PREFIX]);
        hash.update(&bytes[MATRIX_HEADER..]);
        ensure!(
            hash.finalize().as_slice() == &bytes[MATRIX_PREFIX..MATRIX_HEADER],
            "matrix artifact metadata/payload corruption"
        );
        let values = bytes[MATRIX_HEADER..]
            .chunks_exact(4)
            .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
            .collect();
        Self::new(layout, u32_at(16) as usize, u64_at(48), u64_at(40), values)
    }
}

#[derive(Debug)]
pub struct MatrixReference {
    pub layout: MatrixF32,
    pub step: u64,
    pub operation: u64,
    pub start_epoch: u64,
    pub end_epoch: u64,
    pub chunk_count: usize,
    pub ranks: Vec<Vec<f32>>,
}

pub fn reduce_matrix_reference(
    frames: &[MatrixFrame],
    start_epoch: u64,
) -> Result<MatrixReference> {
    aligned_epoch(start_epoch)?;
    ensure!(
        frames.len() == WORLD,
        "matrix reduce requires eight source frames"
    );
    let first = &frames[0];
    let mut state = vec![Vec::new(); WORLD];
    let mut seen = [false; WORLD];
    for frame in frames {
        ensure!(
            frame.rank < WORLD
                && !seen[frame.rank]
                && frame.layout == first.layout
                && frame.step == first.step
                && frame.operation == first.operation,
            "duplicate/wrong-step/wrong-layout matrix source"
        );
        finite(&frame.values, "matrix source")?;
        state[frame.rank] = frame.values.clone();
        seen[frame.rank] = true;
    }
    let schedule = crate::tp_xport::serial_plan(first.layout.elements(), ROUNDS);
    let end_epoch = start_epoch
        .checked_add(schedule.len() as u64)
        .context("matrix epoch overflow")?;
    let mut epoch = start_epoch;
    for pair in schedule {
        epoch += 1;
        let bit = 1usize << (epoch % ROUNDS as u64);
        let end = pair.off + pair.len;
        let mut next = Vec::with_capacity(WORLD);
        for rank in 0..WORLD {
            let lower = rank & !bit;
            let upper = lower | bit;
            let values: Vec<_> = (pair.off..end)
                .map(|i| state[lower][i] + state[upper][i])
                .collect();
            finite(&values, "matrix FP32 intermediate")?;
            next.push(values);
        }
        for rank in 0..WORLD {
            state[rank][pair.off..end].copy_from_slice(&next[rank]);
        }
    }
    ensure!(
        epoch == end_epoch && epoch % ROUNDS as u64 == 0,
        "matrix reduction lost its phase"
    );
    Ok(MatrixReference {
        layout: first.layout,
        step: first.step,
        operation: first.operation,
        start_epoch,
        end_epoch,
        chunk_count: first.layout.chunk_count(),
        ranks: state,
    })
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Maxloc {
    value: f32,
    token_id: u32,
}

impl Maxloc {
    pub fn new(value: f32, token_id: u32) -> Result<Self> {
        ensure!(
            value.is_finite() && token_id < VOCAB,
            "nonfinite value or invalid global token ID in maxloc"
        );
        Ok(Self { value, token_id })
    }
    pub fn value(self) -> f32 {
        self.value
    }
    pub fn token_id(self) -> u32 {
        self.token_id
    }
    pub fn encode_pair(self) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&self.value.to_bits().to_le_bytes());
        bytes[4..].copy_from_slice(&self.token_id.to_le_bytes());
        bytes
    }
    pub fn decode_pair(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == 8,
            "maxloc pair must contain both value and index (8 bytes)"
        );
        Self::new(
            f32::from_bits(u32::from_le_bytes(bytes[..4].try_into().unwrap())),
            u32::from_le_bytes(bytes[4..].try_into().unwrap()),
        )
    }
}

/// Lexicographic winner: higher score, then lower global ID. Return the complete
/// winning source pair, including its original float bits, at every stage.
pub fn maxloc_combine(lower: Maxloc, upper: Maxloc) -> Result<Maxloc> {
    Maxloc::new(lower.value, lower.token_id)?;
    Maxloc::new(upper.value, upper.token_id)?;
    if lower.value > upper.value || (lower.value == upper.value && lower.token_id <= upper.token_id)
    {
        Ok(lower)
    } else {
        Ok(upper)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaxlocFrame {
    pub rank: usize,
    pub operation: u64,
    pub pair: Maxloc,
}

#[derive(Debug)]
pub struct MaxlocReference {
    pub operation: u64,
    pub start_epoch: u64,
    pub end_epoch: u64,
    pub ranks: [Maxloc; WORLD],
}

pub fn maxloc_reference(frames: &[MaxlocFrame], start_epoch: u64) -> Result<MaxlocReference> {
    aligned_epoch(start_epoch)?;
    ensure!(frames.len() == WORLD, "maxloc requires all 8 source frames");
    let operation = frames[0].operation;
    let mut state = [Maxloc::new(0.0, 0)?; WORLD];
    let mut seen = [false; WORLD];
    for frame in frames {
        ensure!(
            frame.rank < WORLD && !seen[frame.rank] && frame.operation == operation,
            "invalid rank or mixed operation in maxloc"
        );
        state[frame.rank] = Maxloc::new(frame.pair.value, frame.pair.token_id)?;
        seen[frame.rank] = true;
    }
    for round in round_order(start_epoch)? {
        let bit = 1 << round;
        let mut next = state;
        for rank in 0..WORLD {
            let lower = rank & !bit;
            next[rank] = maxloc_combine(state[lower], state[lower | bit])?;
        }
        state = next;
    }
    Ok(MaxlocReference {
        operation,
        start_epoch,
        end_epoch: start_epoch + ROUNDS as u64,
        ranks: state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_roles_odd_widths_and_finite_padding_use_the_actual_serial_schedule() {
        for rows in [1, 8, 16, 128] {
            for (role, cols) in [
                (MatrixRole::EngramLookup, 6144),
                (MatrixRole::IndexScore, 31),
                (MatrixRole::IndexScore, 513),
            ] {
                let layout = MatrixF32::new(role, rows, cols).unwrap();
                let frames: Vec<_> = (0..WORLD)
                    .map(|rank| {
                        MatrixFrame::new(
                            layout,
                            rank,
                            42,
                            9,
                            (0..layout.elements())
                                .map(|i| {
                                    if role == MatrixRole::IndexScore && i % cols >= cols - 3 {
                                        0.0
                                    } else {
                                        rank as f32 + 42.0 / 256.0 + (i % 17) as f32 / 1024.0
                                    }
                                })
                                .collect(),
                        )
                        .unwrap()
                    })
                    .collect();
                let result = reduce_matrix_reference(&frames, 24).unwrap();
                assert_eq!(result.end_epoch, 24 + (layout.chunk_count() * 3) as u64);
                for values in &result.ranks {
                    for (i, &value) in values.iter().enumerate() {
                        let expected = if role == MatrixRole::IndexScore && i % cols >= cols - 3 {
                            0.0
                        } else {
                            28.0 + 8.0 * (42.0 / 256.0 + (i % 17) as f32 / 1024.0)
                        };
                        assert_eq!(value.to_bits(), expected.to_bits());
                    }
                }
            }
        }
        assert_eq!(
            MatrixF32::new(MatrixRole::EngramLookup, 128, 6144)
                .unwrap()
                .chunk_count(),
            4
        );
        assert!(MatrixF32::new(MatrixRole::EngramLookup, 1, 5120).is_err());
        assert!(MatrixF32::new(MatrixRole::IndexScore, 0, 31).is_err());
        assert!(MatrixF32::new(MatrixRole::IndexScore, 128, MAX_INDEX_COLUMNS).is_err());
        assert!(MatrixF32::new(MatrixRole::IndexScore, 1, MAX_INDEX_COLUMNS + 1).is_err());
    }

    #[test]
    fn matrix_envelope_and_sources_reject_corruption_shape_aliases_and_wrong_steps() {
        let layout = MatrixF32::new(MatrixRole::EngramLookup, 1, 6144).unwrap();
        let mut values = vec![1.25; layout.elements()];
        values[0] = -0.0;
        let good = MatrixFrame::new(layout, 7, 123, 8, values).unwrap();
        let bytes = good.encode();
        let read = MatrixFrame::decode(&bytes).unwrap();
        assert_eq!(read.encode(), bytes);
        assert_eq!(read.values()[0].to_bits(), (-0.0_f32).to_bits());
        for index in [16, 20, 24, 32, 40, 48, 56, 64, MATRIX_HEADER + 4] {
            let mut bad = bytes.clone();
            bad[index] ^= 1;
            assert!(MatrixFrame::decode(&bad).is_err());
        }
        let mut nonfinite = good.clone();
        nonfinite.values[0] = f32::NEG_INFINITY;
        assert!(MatrixFrame::decode(&nonfinite.encode()).is_err());
        let mut frames: Vec<_> = (0..WORLD)
            .map(|rank| {
                MatrixFrame::new(layout, rank, 123, 8, vec![rank as f32; layout.elements()])
                    .unwrap()
            })
            .collect();
        assert!(reduce_matrix_reference(&frames, 0).is_ok());
        frames[7].step = 124;
        assert!(reduce_matrix_reference(&frames, 0).is_err());
        frames[7].step = 123;
        frames[7].layout = MatrixF32::new(MatrixRole::IndexScore, 1, 6144).unwrap();
        assert!(reduce_matrix_reference(&frames, 0).is_err());
        frames[7] = frames[0].clone();
        assert!(reduce_matrix_reference(&frames, 0).is_err());
        assert!(
            MatrixFrame::new(layout, 0, 0, 0, vec![f32::NEG_INFINITY; layout.elements()]).is_err()
        );
    }

    fn frames(layout: RoutedLayout) -> Vec<RoutedFrame> {
        (0..WORLD)
            .map(|rank| {
                RoutedFrame::new(
                    layout,
                    rank,
                    17,
                    (0..layout.elements())
                        .map(|i| rank as f32 + (i % 97) as f32 / 1024.0)
                        .collect(),
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn supported_rows_follow_actual_three_round_chunk_schedule() {
        for rows in [1, 2, 4, 8, 16, 52, 128] {
            let kind = if rows <= 16 {
                RowKind::IndependentDecodeLanes
            } else {
                RowKind::ContiguousPrefillRows
            };
            let layout = RoutedLayout::new(rows, kind).unwrap();
            let source = frames(layout);
            let reduced = reduce_reference(&source, 24).unwrap();
            assert_eq!(reduced.end_epoch, 24 + (layout.chunk_count() * 3) as u64);
            assert_eq!(reduced.ranks.len(), WORLD);
            for rank in 0..WORLD {
                for (i, value) in reduced.ranks[rank].iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        (28.0 + 8.0 * (i % 97) as f32 / 1024.0).to_bits()
                    );
                }
            }
        }
        assert_eq!(
            RoutedLayout::new(52, RowKind::ContiguousPrefillRows)
                .unwrap()
                .chunk_count(),
            2
        );
        assert_eq!(round_order(0).unwrap(), [1, 2, 0]);
        assert_eq!(round_order(1).unwrap(), [2, 0, 1]);
        assert_eq!(round_order(2).unwrap(), [0, 1, 2]);
        assert!(RoutedLayout::from_geometry(1, 4096, 8, RowKind::IndependentDecodeLanes).is_err());
        assert!(RoutedLayout::from_geometry(1, 5120, 4, RowKind::IndependentDecodeLanes).is_err());
        assert!(RoutedLayout::new(52, RowKind::IndependentDecodeLanes).is_err());
        assert!(RoutedLayout::new(0, RowKind::ContiguousPrefillRows).is_err());
        assert!(RoutedLayout::new(129, RowKind::ContiguousPrefillRows).is_err());
    }

    #[test]
    fn native_epoch_order_is_numerically_material() {
        let layout = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let values = [
            16_777_216.0_f32,
            -16_777_216.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let mut source: Vec<_> = values
            .into_iter()
            .enumerate()
            .map(|(rank, value)| RoutedFrame::new(layout, rank, 7, vec![value; HIDDEN]).unwrap())
            .collect();
        let correct = reduce_reference(&source, 0).unwrap();
        assert!(correct.ranks.iter().flatten().all(|&x| x == 1.0));
        // Rank frames may arrive in any order; native epoch order must not change.
        source.reverse();
        let reordered = reduce_reference(&source, 0).unwrap();
        assert_eq!(reordered.ranks, correct.ranks);
        // A superficially plausible 0,1,2 XOR order changes FP32 rounding.
        let mut wrong = values;
        for round in [0, 1, 2] {
            let bit = 1 << round;
            let old = wrong;
            for rank in 0..WORLD {
                let lower = rank & !bit;
                wrong[rank] = old[lower] + old[lower | bit];
            }
        }
        assert_eq!(wrong, [2.0; WORLD]);
        assert!(reduce_reference(&source, u64::MAX).is_err());
    }

    #[test]
    fn reference_frame_preserves_bits_and_detects_metadata_or_payload_corruption() {
        let layout = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let mut values = vec![1.25; HIDDEN];
        values[0] = -0.0;
        let frame = RoutedFrame::new(layout, 7, 999, values).unwrap();
        let bytes = frame.encode();
        let decoded = RoutedFrame::decode(&bytes).unwrap();
        assert_eq!(decoded.rank(), 7);
        assert_eq!(decoded.operation(), 999);
        assert_eq!(decoded.values()[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(decoded.encode(), bytes);
        for index in [16, 40, 56, HEADER_BYTES + 5] {
            let mut bad = bytes.clone();
            bad[index] ^= 1;
            assert!(RoutedFrame::decode(&bad).is_err());
        }
        assert!(RoutedFrame::decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(RoutedFrame::new(layout, 0, 1, vec![f32::NAN; HIDDEN]).is_err());
        assert!(RoutedFrame::new(layout, 0, 1, vec![f32::INFINITY; HIDDEN]).is_err());
    }

    #[test]
    fn expert_boundaries_and_empty_rank_partials_keep_exact_ownership() {
        let layout = RoutedLayout::new(2, RowKind::IndependentDecodeLanes).unwrap();
        let routes = [[0, 47, 48, 95, 96, 143], [191, 192, 239, 240, 335, 383]];
        let mut local = vec![vec![0.0; layout.elements()]; WORLD];
        for (row, experts) in routes.iter().enumerate() {
            for (slot, &expert) in experts.iter().enumerate() {
                let owner = layout.expert_owner(expert).unwrap();
                for d in 0..HIDDEN {
                    local[owner][row * HIDDEN + d] += (slot + 1) as f32;
                }
            }
        }
        assert!(local[7][..HIDDEN].iter().all(|&x| x == 0.0));
        assert_eq!(layout.expert_owner(47).unwrap(), 0);
        assert_eq!(layout.expert_owner(48).unwrap(), 1);
        assert_eq!(layout.expert_owner(383).unwrap(), 7);
        assert!(layout.expert_owner(384).is_err());
        let source: Vec<_> = local
            .into_iter()
            .enumerate()
            .map(|(rank, values)| RoutedFrame::new(layout, rank, 0, values).unwrap())
            .collect();
        let sum = reduce_reference(&source, 0).unwrap();
        assert!(sum.ranks.iter().flatten().all(|&x| x == 21.0));
        let mut bad = source.clone();
        bad[7] = source[0].clone();
        assert!(reduce_reference(&bad, 0).is_err());
        let mut bad = source.clone();
        bad[7].operation += 1;
        assert!(reduce_reference(&bad, 0).is_err());
        assert!(reduce_reference(&source[..7], 0).is_err());
        assert!(reduce_reference(&source, 1).is_err());
    }

    #[test]
    fn early_bf16_rounding_is_a_real_negative_control() {
        let layout = RoutedLayout::new(1, RowKind::IndependentDecodeLanes).unwrap();
        let mut source: Vec<_> = (0..WORLD)
            .map(|rank| RoutedFrame::new(layout, rank, 0, vec![0.0; HIDDEN]).unwrap())
            .collect();
        source[0].values.fill(1.003_906_25);
        source[1].values.fill(-1.0);
        let correct = reduce_reference(&source, 0).unwrap();
        assert_eq!(correct.ranks[0][0], 0.003_906_25);
        let mut old: Vec<f32> = source.iter().map(|f| f.values[0]).collect();
        for round in round_order(0).unwrap() {
            let bit = 1 << round;
            let mut next = old.clone();
            for rank in 0..WORLD {
                let low = rank & !bit;
                next[rank] = half::bf16::from_f32(old[low] + old[low | bit]).to_f32();
            }
            old = next;
        }
        assert_eq!(old[0], 0.0);
        assert_ne!(old[0].to_bits(), correct.ranks[0][0].to_bits());
        for frame in &mut source {
            frame.values.fill(f32::MAX);
        }
        assert!(reduce_reference(&source, 0).is_err());
    }

    #[test]
    fn maxloc_retains_value_and_lowest_global_id_through_all_rounds() {
        let pairs = [
            (5.0, 10),
            (7.0, 5),
            (9.0, 20),
            (1.0, 30),
            (2.0, 40),
            (3.0, 50),
            (4.0, 60),
            (0.0, 70),
        ];
        let frames: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(rank, &(value, id))| MaxlocFrame {
                rank,
                operation: 3,
                pair: Maxloc::new(value, id).unwrap(),
            })
            .collect();
        let correct = maxloc_reference(&frames, 0).unwrap();
        assert!(correct
            .ranks
            .iter()
            .all(|p| p.token_id() == 20 && p.value() == 9.0));
        // Reproduce an index-only carry: later rounds retain the rank's old
        // value while replacing just the index. It elects the wrong token.
        let mut old: Vec<_> = frames.iter().map(|f| f.pair).collect();
        for round in round_order(0).unwrap() {
            let bit = 1 << round;
            let mut next = old.clone();
            for rank in 0..WORLD {
                let low = rank & !bit;
                let winner = maxloc_combine(old[low], old[low | bit]).unwrap();
                next[rank] = Maxloc::new(old[rank].value(), winner.token_id()).unwrap();
            }
            old = next;
        }
        assert_ne!(old[0].token_id(), correct.ranks[0].token_id());
        let tie: Vec<_> = [50, 20, 40, 10, 70, 60, 90, 80]
            .into_iter()
            .enumerate()
            .map(|(rank, id)| MaxlocFrame {
                rank,
                operation: 1,
                pair: Maxloc::new(if rank == 3 { -0.0 } else { 0.0 }, id).unwrap(),
            })
            .collect();
        let winners = maxloc_reference(&tie, 0).unwrap();
        for pair in winners.ranks {
            assert_eq!(pair.token_id(), 10);
            assert_eq!(pair.value().to_bits(), (-0.0_f32).to_bits());
            assert_eq!(
                Maxloc::decode_pair(&pair.encode_pair())
                    .unwrap()
                    .encode_pair(),
                pair.encode_pair()
            );
        }
        assert!(Maxloc::new(f32::NAN, 0).is_err());
        assert!(Maxloc::new(f32::NEG_INFINITY, 0).is_err());
        assert!(Maxloc::new(1.0, VOCAB).is_err());
        assert!(Maxloc::decode_pair(&[0; 4]).is_err());
        let mut duplicate = frames.clone();
        duplicate[7].rank = 0;
        assert!(maxloc_reference(&duplicate, 0).is_err());
        let mut mixed = frames.clone();
        mixed[7].operation += 1;
        assert!(maxloc_reference(&mixed, 0).is_err());
        let mut bad_rank = frames.clone();
        bad_rank[7].rank = WORLD;
        assert!(maxloc_reference(&bad_rank, 0).is_err());
        assert!(maxloc_reference(&frames[..7], 0).is_err());
        assert!(maxloc_reference(&frames, u64::MAX).is_err());
    }
}
