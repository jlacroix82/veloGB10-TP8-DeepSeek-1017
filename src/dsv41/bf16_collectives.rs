//! Source-BF16 index-score matrix ABI and exact serial TP8 reference.
//! Each XOR exchange sums widened BF16 operands in F32 and rounds BACK to BF16.
//! This explicitly fixes our tree/epoch association, not NCCL bit equivalence.
//! Causal/candidate masking follows reduction; network inputs must be finite.
use super::collectives::{ROUNDS, WORLD};
use anyhow::{ensure, Context, Result};
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const MAX_ROWS: usize = 128;
pub const MAX_COLS: usize = 131_072;
pub const MAX_ELEMENTS: usize = 4 * 1024 * 1024;
pub const EPOCH_BF16: usize = crate::tp_xport::EPOCH_FLOATS * 2;
const MAGIC: &[u8; 8] = b"DS41B16\0";
const PREFIX: usize = 56;
const HEADER: usize = PREFIX + 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct MatrixBf16Index {
    rows: usize,
    cols: usize,
}
impl MatrixBf16Index {
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        let n = rows
            .checked_mul(cols)
            .context("BF16 index matrix extent overflow")?;
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && (1..=MAX_COLS).contains(&cols) && n <= MAX_ELEMENTS,
            "BF16 index matrix exceeds bounded rows/columns/elements"
        );
        Ok(Self { rows, cols })
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
    /// Only the flattened matrix tail is padded; logical row stride stays cols.
    /// Four BF16 words give8-byte wire alignment and avoid K1's4-byte copy tail.
    pub fn padded_elements(self) -> usize {
        self.elements().div_ceil(4) * 4
    }
    pub fn padding_elements(self) -> usize {
        self.padded_elements() - self.elements()
    }
    pub fn chunk_count(self) -> usize {
        self.padded_elements().div_ceil(EPOCH_BF16)
    }
    pub fn epochs(self) -> usize {
        self.chunk_count() * ROUNDS
    }
    pub fn chunks(self) -> Vec<(usize, usize)> {
        (0..self.padded_elements())
            .step_by(EPOCH_BF16)
            .map(|off| (off, (self.padded_elements() - off).min(EPOCH_BF16)))
            .collect()
    }
    pub fn pack(self, values: &[bf16]) -> Result<Vec<bf16>> {
        ensure!(
            values.len() == self.elements() && values.iter().all(|x| x.is_finite()),
            "BF16 index source shape/nonfinite"
        );
        let mut padded = values.to_vec();
        padded.resize(self.padded_elements(), bf16::ZERO);
        Ok(padded)
    }
}

#[derive(Clone, Debug)]
pub struct Bf16IndexFrame {
    pub layout: MatrixBf16Index,
    pub rank: usize,
    pub operation: u64,
    pub step: u64,
    pub values: Vec<bf16>,
}
impl Bf16IndexFrame {
    pub fn new(
        layout: MatrixBf16Index,
        rank: usize,
        operation: u64,
        step: u64,
        values: Vec<bf16>,
    ) -> Result<Self> {
        ensure!(rank < WORLD, "BF16 index source rank outsideworld8");
        layout.pack(&values)?;
        Ok(Self {
            layout,
            rank,
            operation,
            step,
            values,
        })
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        ensure!(self.rank < WORLD, "invalid BF16 index rank");
        let padded = self.layout.pack(&self.values)?;
        let mut prefix = MAGIC.to_vec();
        prefix.extend_from_slice(&1u32.to_le_bytes());
        prefix.extend_from_slice(&(self.rank as u32).to_le_bytes());
        for x in [
            self.operation,
            self.step,
            self.layout.rows() as u64,
            self.layout.cols() as u64,
            self.layout.padded_elements() as u64,
        ] {
            prefix.extend_from_slice(&x.to_le_bytes());
        }
        let payload = padded
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut h = Sha256::new();
        h.update(&prefix);
        h.update(&payload);
        let mut result = prefix;
        result.extend_from_slice(&h.finalize());
        result.extend_from_slice(&payload);
        Ok(result)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= HEADER && &bytes[..8] == MAGIC,
            "BF16 frame magic/truncation"
        );
        let u32at = |p| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        let u64at = |p| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
        ensure!(u32at(8) == 1, "BF16 frame version");
        let rank = u32at(12) as usize;
        let layout =
            MatrixBf16Index::new(usize::try_from(u64at(32))?, usize::try_from(u64at(40))?)?;
        ensure!(
            rank < WORLD
                && u64at(48) == layout.padded_elements() as u64
                && bytes.len() == HEADER + layout.padded_elements() * 2,
            "BF16 frame rank/shape/padding extent"
        );
        let mut h = Sha256::new();
        h.update(&bytes[..PREFIX]);
        h.update(&bytes[HEADER..]);
        ensure!(
            h.finalize().as_slice() == &bytes[PREFIX..HEADER],
            "BF16 frame checksum"
        );
        let values = bytes[HEADER..]
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect::<Vec<_>>();
        ensure!(
            values[..layout.elements()].iter().all(|x| x.is_finite())
                && values[layout.elements()..].iter().all(|x| x.to_bits() == 0),
            "BF16 frame nonfinite/nonzero padding"
        );
        Self::new(
            layout,
            rank,
            u64at(16),
            u64at(24),
            values[..layout.elements()].to_vec(),
        )
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Bf16StageProof {
    pub epoch: u64,
    pub xor_bit: usize,
    pub chunk_offset: usize,
    pub chunk_elements: usize,
    pub rank_output_sha256_u16le: Vec<String>,
}
pub struct Bf16Reduction {
    pub outputs: Vec<Vec<bf16>>,
    pub stages: Vec<Bf16StageProof>,
    pub end_epoch: u64,
}
pub fn values_sha256(values: &[bf16]) -> String {
    let mut h = Sha256::new();
    for x in values {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
/// Actual K1 preincrements the epoch, so a quiescent start0 exchanges XOR bits
///1,2,0, not0,1,2. Every chunk consumes3epochs and keeps this phase invariant.
pub fn serial_bf16_reference(frames: &[Bf16IndexFrame], start_epoch: u64) -> Result<Bf16Reduction> {
    ensure!(
        frames.len() == WORLD && start_epoch % ROUNDS as u64 == 0,
        "BF16 reference needs8sources/alignedepoch"
    );
    let first = &frames[0];
    let layout = first.layout;
    let mut ordered = vec![None; WORLD];
    for frame in frames {
        ensure!(
            frame.rank < WORLD
                && ordered[frame.rank].is_none()
                && frame.layout == layout
                && frame.operation == first.operation
                && frame.step == first.step,
            "BF16 rank/operation/step/layout disagreement"
        );
        ordered[frame.rank] = Some(layout.pack(&frame.values)?);
    }
    let mut values = ordered
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .context("unowned BF16 source rank")?;
    let mut epoch = start_epoch;
    let mut stages = Vec::new();
    for (offset, count) in layout.chunks() {
        for _ in 0..ROUNDS {
            epoch = epoch
                .checked_add(1)
                .context("BF16 reference epochoverflow")?;
            let bit = (epoch % ROUNDS as u64) as usize;
            let mask = 1 << bit;
            let mut next = values.clone();
            for rank in 0..WORLD {
                let lower = rank & !mask;
                let upper = lower | mask;
                for i in offset..offset + count {
                    let sum = values[lower][i].to_f32() + values[upper][i].to_f32();
                    let rounded = bf16::from_f32(sum);
                    ensure!(
                        sum.is_finite() && rounded.is_finite(),
                        "BF16 reference intermediate overflow/corruption"
                    );
                    next[rank][i] = rounded;
                }
            }
            values = next;
            stages.push(Bf16StageProof {
                epoch,
                xor_bit: bit,
                chunk_offset: offset,
                chunk_elements: count,
                rank_output_sha256_u16le: values
                    .iter()
                    .map(|x| values_sha256(&x[offset..offset + count]))
                    .collect(),
            });
        }
    }
    for rank in 1..WORLD {
        ensure!(
            values[rank]
                .iter()
                .map(|x| x.to_bits())
                .eq(values[0].iter().map(|x| x.to_bits())),
            "BF16 ranks disagree"
        );
    }
    ensure!(
        values[0][layout.elements()..]
            .iter()
            .all(|x| x.to_bits() == 0),
        "BF16 padding changed"
    );
    Ok(Bf16Reduction {
        outputs: values
            .into_iter()
            .map(|x| x[..layout.elements()].to_vec())
            .collect(),
        stages,
        end_epoch: epoch,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn frames(layout: MatrixBf16Index, values: [f32; 8]) -> Vec<Bf16IndexFrame> {
        (0..8)
            .map(|rank| {
                Bf16IndexFrame::new(
                    layout,
                    rank,
                    73,
                    19,
                    vec![bf16::from_f32(values[rank]); layout.elements()],
                )
                .unwrap()
            })
            .collect()
    }
    #[test]
    fn exact_shape_padding_and_source_frame_corruption_are_rejected() {
        for rows in [1, 2, 4, 8, 16, 52, 128] {
            for cols in [31, 513] {
                let layout = MatrixBf16Index::new(rows, cols).unwrap();
                let f = frames(layout, [1.; 8]);
                let encoded = f[7].encode().unwrap();
                let decoded = Bf16IndexFrame::decode(&encoded).unwrap();
                assert_eq!(decoded.values, f[7].values);
                assert_eq!(decoded.layout, layout);
                assert_eq!(layout.padded_elements() % 4, 0);
                assert!(layout.padding_elements() < 4);
                let mut corrupt = encoded.clone();
                corrupt[HEADER + 1] ^= 1;
                assert!(Bf16IndexFrame::decode(&corrupt).is_err());
                let mut badrank = encoded.clone();
                badrank[12] = 8;
                assert!(Bf16IndexFrame::decode(&badrank).is_err());
                assert!(Bf16IndexFrame::decode(&encoded[..encoded.len() - 1]).is_err());
            }
        }
        let layout = MatrixBf16Index::new(1, 31).unwrap();
        let mut bytes = frames(layout, [0.; 8])[0].encode().unwrap();
        // Recomputed valid checksum cannot hide corrupted logical padding.
        *bytes.last_mut().unwrap() = 1;
        let mut h = Sha256::new();
        h.update(&bytes[..PREFIX]);
        h.update(&bytes[HEADER..]);
        bytes[PREFIX..HEADER].copy_from_slice(&h.finalize());
        assert!(Bf16IndexFrame::decode(&bytes).is_err());
        for shape in [
            (0, 31),
            (129, 1),
            (1, 0),
            (1, 131073),
            (128, 131072),
            (usize::MAX, 2),
        ] {
            assert!(MatrixBf16Index::new(shape.0, shape.1).is_err());
        }
        assert!(Bf16IndexFrame::new(layout, 0, 1, 1, vec![bf16::NAN; 31]).is_err());
    }
    #[test]
    fn three_stage_bf16_rounding_exposes_f32_alternative_and_signed_zero() {
        let layout = MatrixBf16Index::new(1, 31).unwrap();
        let f = frames(layout, [256., -256., 1., 1., 0., 0., 0., 0.]);
        let r = serial_bf16_reference(&f, 0).unwrap();
        assert_eq!(
            r.stages.iter().map(|x| x.xor_bit).collect::<Vec<_>>(),
            [1, 2, 0]
        );
        assert_eq!(r.end_epoch, 3);
        assert!(r.outputs[0].iter().all(|x| x.to_f32() == 1.));
        let stage_one = [256., -255., 256., -255., 0., 0., 0., 0.];
        for (rank, value) in stage_one.into_iter().enumerate() {
            let mut words = vec![bf16::from_f32(value); 31];
            words.push(bf16::ZERO);
            assert_eq!(
                r.stages[0].rank_output_sha256_u16le[rank],
                values_sha256(&words)
            );
        }
        let f32_alternative = bf16::from_f32([256f32, -256., 1., 1., 0., 0., 0., 0.].iter().sum());
        assert_eq!(f32_alternative.to_f32(), 2.);
        assert_ne!(r.outputs[0][0], f32_alternative);
        let zero = serial_bf16_reference(&frames(layout, [-0.; 8]), 3).unwrap();
        assert!(zero.outputs[0].iter().all(|x| x.to_bits() == 0x8000));
        let tiny =
            serial_bf16_reference(&frames(layout, [bf16::from_bits(1).to_f32(); 8]), 6).unwrap();
        assert_eq!(tiny.outputs[0][0].to_bits(), 8);
        let mut duplicate = f.clone();
        duplicate[7].rank = 0;
        assert!(serial_bf16_reference(&duplicate, 0).is_err());
        let mut stale = f.clone();
        stale[7].step += 1;
        assert!(serial_bf16_reference(&stale, 0).is_err());
        assert!(serial_bf16_reference(&f, 1).is_err());
        assert!(serial_bf16_reference(&frames(layout, [bf16::MAX.to_f32(); 8]), 0).is_err());
    }
    #[test]
    fn multichunk_three_round_schedule_keeps_monotonic_shared_epoch_phase() {
        let layout = MatrixBf16Index::new(128, 4097).unwrap();
        assert_eq!(layout.chunk_count(), 2);
        assert_eq!(layout.epochs(), 6);
        let result =
            serial_bf16_reference(&frames(layout, [1., 2., 3., 4., 5., 6., 7., 8.]), 24).unwrap();
        assert_eq!(result.end_epoch, 30);
        assert_eq!(
            result.stages.iter().map(|x| x.xor_bit).collect::<Vec<_>>(),
            [1, 2, 0, 1, 2, 0]
        );
        assert!(result
            .outputs
            .iter()
            .all(|v| v.iter().all(|x| x.to_f32() == 36.)));
    }
}
