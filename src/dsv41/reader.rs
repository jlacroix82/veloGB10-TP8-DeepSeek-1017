//! Bounded, read-only native checkpoint I/O. No conversions, GPU allocations or
//! payload-identity claims. Callers can stream a large owned tensor to a device
//! without first materializing its whole safetensors shard in host memory.
use super::inventory::{dtype_size_bytes, read_inventory, TensorInventory, TensorMetadata};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

pub const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl FileStamp {
    fn of(file: &File) -> Result<Self> {
        let m = file.metadata()?;
        ensure!(m.is_file(), "checkpoint shard must be a regular file");
        Ok(Self {
            device: m.dev(),
            inode: m.ino(),
            bytes: m.len(),
            modified: (m.mtime(), m.mtime_nsec()),
            changed: (m.ctime(), m.ctime_nsec()),
        })
    }
}

struct OpenShard {
    file: File,
    stamp: FileStamp,
    data_start: u64,
}

/// A span can only be constructed by its reader. Offsets are relative to the
/// tensor's first element; the private identity binds spans to the same unchanged
/// source snapshot. Independently opened readers of that snapshot may share spans.
#[derive(Debug, Clone, Serialize)]
pub struct TensorSpan {
    pub tensor_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub first_row: u64,
    pub row_count: u64,
    /// Present only for a packed rectangular slice of a two-dimensional tensor.
    pub first_column: Option<u64>,
    pub column_count: Option<u64>,
    pub bytes: u64,
    shard: String,
    file_offset: u64,
    source_row_bytes: u64,
    selected_row_bytes: u64,
    reader_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadStatistics {
    pub payload_bytes_read: u64,
    pub payload_read_calls: u64,
    pub largest_read_bytes: usize,
    pub maximum_staging_bytes: usize,
    pub payload_checksums_verified: bool,
}

pub struct CheckpointReader {
    root: PathBuf,
    inventory: TensorInventory,
    tensor_indices: BTreeMap<String, usize>,
    shards: BTreeMap<String, OpenShard>,
    chunk_bytes: usize,
    reader_id: String,
    statistics: ReadStatistics,
}

fn basename(name: &str) -> Result<()> {
    let mut parts = Path::new(name).components();
    ensure!(
        matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
        "shard must be a basename"
    );
    Ok(())
}

impl CheckpointReader {
    /// Opens regular files without following shard symlinks. Verifies all header
    /// identities a second time against the open descriptors, then retains those
    /// descriptors. Concurrent file replacement cannot redirect an existing read.
    pub fn open(root: &Path, chunk_bytes: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CHUNK_BYTES).contains(&chunk_bytes),
            "chunk size must be 1..=64 MiB"
        );
        let root = root
            .canonicalize()
            .context("canonicalize checkpoint root")?;
        let inventory = read_inventory(&root)?;
        let mut shards = BTreeMap::new();
        let mut identity = Sha256::new();
        identity.update(inventory.index_sha256.as_bytes());
        for metadata in &inventory.shards {
            basename(&metadata.name)?;
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(root.join(&metadata.name))
                .with_context(|| format!("open shard {}", metadata.name))?;
            let stamp = FileStamp::of(&file)?;
            ensure!(
                stamp.bytes == metadata.file_bytes,
                "shard length changed since inventory"
            );
            let mut prefix = [0u8; 8];
            file.read_exact_at(&mut prefix, 0)?;
            let header_bytes = u64::from_le_bytes(prefix);
            ensure!(
                header_bytes == metadata.header_bytes,
                "shard header length changed"
            );
            let mut header = vec![0u8; usize::try_from(header_bytes)?];
            file.read_exact_at(&mut header, 8)?;
            ensure!(
                format!("{:x}", Sha256::digest(&header)) == metadata.header_sha256,
                "shard header changed since inventory"
            );
            ensure!(
                FileStamp::of(&file)? == stamp,
                "shard changed while verifying header"
            );
            identity.update(metadata.name.as_bytes());
            identity.update(metadata.header_sha256.as_bytes());
            identity.update(stamp.device.to_le_bytes());
            identity.update(stamp.inode.to_le_bytes());
            identity.update(stamp.bytes.to_le_bytes());
            identity.update(stamp.modified.0.to_le_bytes());
            identity.update(stamp.modified.1.to_le_bytes());
            identity.update(stamp.changed.0.to_le_bytes());
            identity.update(stamp.changed.1.to_le_bytes());
            shards.insert(
                metadata.name.clone(),
                OpenShard {
                    file,
                    stamp,
                    data_start: metadata.data_start_bytes,
                },
            );
        }
        let tensor_indices = inventory
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            root,
            inventory,
            tensor_indices,
            shards,
            chunk_bytes,
            reader_id: format!("{:x}", identity.finalize()),
            statistics: ReadStatistics {
                payload_bytes_read: 0,
                payload_read_calls: 0,
                largest_read_bytes: 0,
                maximum_staging_bytes: 0,
                payload_checksums_verified: false,
            },
        })
    }

    pub fn inventory(&self) -> &TensorInventory {
        &self.inventory
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn statistics(&self) -> &ReadStatistics {
        &self.statistics
    }
    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorMetadata> {
        Ok(&self.inventory.tensors[*self
            .tensor_indices
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?])
    }

    pub fn full_span(&self, name: &str) -> Result<TensorSpan> {
        let tensor = self.tensor(name)?;
        self.row_span(name, 0, tensor.shape.first().copied().unwrap_or(1))
    }

    /// Native first-dimension sharding, suitable for each expert tensor and both
    /// Engram weight/scale row ranges. Zero valid rows produces an empty span;
    /// checkpoint bytes never include optional allocation padding.
    pub fn row_span(&self, name: &str, first_row: u64, row_count: u64) -> Result<TensorSpan> {
        let tensor = self.tensor(name)?;
        let rows = tensor.shape.first().copied().unwrap_or(1);
        let end = first_row
            .checked_add(row_count)
            .context("row span overflow")?;
        ensure!(end <= rows, "row span is outside tensor");
        let element_bytes = dtype_size_bytes(&tensor.dtype)?;
        let row_bytes = tensor
            .shape
            .iter()
            .skip(1)
            .try_fold(element_bytes, |n, &d| {
                n.checked_mul(d).context("row size overflow")
            })?;
        let relative = first_row
            .checked_mul(row_bytes)
            .context("row offset overflow")?;
        let bytes = row_count
            .checked_mul(row_bytes)
            .context("row byte count overflow")?;
        ensure!(
            relative
                .checked_add(bytes)
                .context("span extent overflow")?
                <= tensor.stored_bytes,
            "span exceeds stored tensor"
        );
        let shard = self
            .shards
            .get(&tensor.shard)
            .context("missing opened shard")?;
        let file_offset = shard
            .data_start
            .checked_add(tensor.data_offsets[0])
            .and_then(|x| x.checked_add(relative))
            .context("file offset overflow")?;
        let mut shape = tensor.shape.clone();
        if let Some(first) = shape.first_mut() {
            *first = row_count;
        }
        Ok(TensorSpan {
            tensor_name: name.to_owned(),
            dtype: tensor.dtype.clone(),
            shape,
            first_row,
            row_count,
            first_column: None,
            column_count: None,
            bytes,
            shard: tensor.shard.clone(),
            file_offset,
            source_row_bytes: row_bytes,
            selected_row_bytes: row_bytes,
            reader_id: self.reader_id.clone(),
        })
    }

    /// Select a two-dimensional native rectangle without reading intervening
    /// columns. The callback receives its rows packed consecutively. This is
    /// used for matching WO-B weight and group32 scale column shards.
    pub fn rectangle_span(
        &self,
        name: &str,
        first_row: u64,
        row_count: u64,
        first_column: u64,
        column_count: u64,
    ) -> Result<TensorSpan> {
        let tensor = self.tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "rectangular native slice requires a matrix"
        );
        ensure!(
            first_column
                .checked_add(column_count)
                .is_some_and(|n| n <= tensor.shape[1]),
            "rectangle columns outside tensor"
        );
        let mut span = self.row_span(name, first_row, row_count)?;
        let item = dtype_size_bytes(&tensor.dtype)?;
        let column_offset = first_column
            .checked_mul(item)
            .context("column offset overflow")?;
        span.file_offset = span
            .file_offset
            .checked_add(column_offset)
            .context("rectangle offset overflow")?;
        span.selected_row_bytes = column_count
            .checked_mul(item)
            .context("rectangle row bytes overflow")?;
        span.bytes = row_count
            .checked_mul(span.selected_row_bytes)
            .context("rectangle byte count overflow")?;
        span.first_column = Some(first_column);
        span.column_count = Some(column_count);
        span.shape = vec![row_count, column_count];
        Ok(span)
    }

    /// The callback sees at most chunk_bytes of borrowed staging data and the
    /// offset within the selected span. An error stops I/O immediately. No
    /// conversion, whole-shard allocation, or persistent payload cache occurs.
    pub fn stream<F>(&mut self, span: &TensorSpan, mut consume: F) -> Result<()>
    where
        F: FnMut(u64, &[u8]) -> Result<()>,
    {
        ensure!(
            span.reader_id == self.reader_id,
            "span belongs to a different checkpoint reader"
        );
        let expected = match (span.first_column, span.column_count) {
            (None, None) => self.row_span(&span.tensor_name, span.first_row, span.row_count)?,
            (Some(first), Some(count)) => self.rectangle_span(
                &span.tensor_name,
                span.first_row,
                span.row_count,
                first,
                count,
            )?,
            _ => anyhow::bail!("incomplete rectangular span"),
        };
        ensure!(
            span.shard == expected.shard
                && span.file_offset == expected.file_offset
                && span.bytes == expected.bytes
                && span.dtype == expected.dtype
                && span.shape == expected.shape,
            "span metadata changed"
        );
        ensure!(
            span.source_row_bytes == expected.source_row_bytes
                && span.selected_row_bytes == expected.selected_row_bytes,
            "span row stride changed"
        );
        let shard = self
            .shards
            .get(&span.shard)
            .context("span shard unavailable")?;
        ensure!(
            FileStamp::of(&shard.file)? == shard.stamp,
            "shard changed before payload read"
        );
        let capacity = usize::try_from(span.bytes.min(self.chunk_bytes as u64).min(
            if span.first_column.is_some() {
                span.selected_row_bytes
            } else {
                u64::MAX
            },
        ))?;
        let mut buffer = vec![0u8; capacity];
        self.statistics.maximum_staging_bytes = self.statistics.maximum_staging_bytes.max(capacity);
        let mut offset = 0u64;
        while offset < span.bytes {
            let (file_offset, row_remaining) = if span.first_column.is_some() {
                let row = offset / span.selected_row_bytes;
                let column = offset % span.selected_row_bytes;
                (
                    span.file_offset
                        .checked_add(
                            row.checked_mul(span.source_row_bytes)
                                .context("rectangle read stride overflow")?,
                        )
                        .and_then(|x| x.checked_add(column))
                        .context("rectangle read offset overflow")?,
                    span.selected_row_bytes - column,
                )
            } else {
                (
                    span.file_offset
                        .checked_add(offset)
                        .context("read offset overflow")?,
                    u64::MAX,
                )
            };
            let bytes = usize::try_from(
                (span.bytes - offset)
                    .min(capacity as u64)
                    .min(row_remaining),
            )?;
            shard
                .file
                .read_exact_at(&mut buffer[..bytes], file_offset)?;
            self.statistics.payload_bytes_read = self
                .statistics
                .payload_bytes_read
                .checked_add(bytes as u64)
                .context("read byte counter overflow")?;
            self.statistics.payload_read_calls = self
                .statistics
                .payload_read_calls
                .checked_add(1)
                .context("read call counter overflow")?;
            self.statistics.largest_read_bytes = self.statistics.largest_read_bytes.max(bytes);
            ensure!(
                FileStamp::of(&shard.file)? == shard.stamp,
                "shard changed during payload read"
            );
            consume(offset, &buffer[..bytes])?;
            offset += bytes as u64;
        }
        ensure!(
            FileStamp::of(&shard.file)? == shard.stamp,
            "shard changed before stream completion"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn fixture() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "dsv41-reader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let header = serde_json::to_vec(&serde_json::json!({
            "large": {"dtype":"I8","shape":[1u64<<30,8],"data_offsets":[0,8u64<<30]},
            "scales": {"dtype":"F8_E8M0","shape":[4,2],"data_offsets":[8u64<<30,(8u64<<30)+8]}
        }))
        .unwrap();
        let mut shard = File::create(root.join("part.safetensors")).unwrap();
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        shard.write_all(&header).unwrap();
        shard
            .set_len(8 + header.len() as u64 + (8u64 << 30) + 8)
            .unwrap();
        shard
            .seek(SeekFrom::Start(8 + header.len() as u64 + 8 * 123))
            .unwrap();
        shard
            .write_all(&[
                10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27,
            ])
            .unwrap();
        shard
            .seek(SeekFrom::Start(8 + header.len() as u64 + (8u64 << 30)))
            .unwrap();
        shard
            .write_all(&[120, 121, 122, 123, 124, 125, 126, 127])
            .unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata":{"total_size":(8u64<<30)+8},
                "weight_map":{"large":"part.safetensors","scales":"part.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        root
    }

    #[test]
    fn sparse_multigigabyte_source_reads_only_requested_native_rows_with_bounded_staging() {
        let root = fixture();
        let mut reader = CheckpointReader::open(&root, 5).unwrap();
        assert!(reader.inventory().metadata_bytes_read < 4096);
        assert_eq!(reader.statistics().payload_bytes_read, 0);
        let span = reader.row_span("large", 123, 2).unwrap();
        let mut result = Vec::new();
        let mut offsets = Vec::new();
        reader
            .stream(&span, |offset, bytes| {
                offsets.push(offset);
                result.extend_from_slice(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            result,
            [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27]
        );
        assert_eq!(offsets, [0, 5, 10, 15]);
        assert_eq!(reader.statistics().payload_bytes_read, 16);
        assert_eq!(reader.statistics().largest_read_bytes, 5);
        assert_eq!(reader.statistics().maximum_staging_bytes, 5);
        assert!(!reader.statistics().payload_checksums_verified);
        let scale_span = reader.row_span("scales", 2, 1).unwrap();
        let mut scales = Vec::new();
        reader
            .stream(&scale_span, |_, data| {
                scales.extend_from_slice(data);
                Ok(())
            })
            .unwrap();
        assert_eq!(scales, [124, 125]);
        assert_eq!(reader.statistics().payload_bytes_read, 18);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sparse_bigfile_column_shards_are_packed_without_reading_gaps() {
        let root = fixture();
        let mut reader = CheckpointReader::open(&root, 2).unwrap();
        let span = reader.rectangle_span("large", 123, 2, 2, 3).unwrap();
        assert_eq!(span.shape, [2, 3]);
        assert_eq!(span.bytes, 6);
        let mut got = Vec::new();
        let mut offsets = Vec::new();
        reader
            .stream(&span, |offset, bytes| {
                offsets.push(offset);
                got.extend_from_slice(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(got, [12, 13, 14, 22, 23, 24]);
        assert_eq!(offsets, [0, 2, 3, 5]);
        assert_eq!(reader.statistics().payload_bytes_read, 6);
        assert_eq!(reader.statistics().maximum_staging_bytes, 2);
        let scales = reader.rectangle_span("scales", 0, 4, 1, 1).unwrap();
        let mut got = Vec::new();
        reader
            .stream(&scales, |_, bytes| {
                got.extend_from_slice(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(got, [121, 123, 125, 127]);
        assert_eq!(reader.statistics().payload_bytes_read, 10);
        assert!(reader.rectangle_span("large", 0, 1, 7, 2).is_err());
        assert!(reader.rectangle_span("large", 0, 1, u64::MAX, 1).is_err());
        let mut forged = span.clone();
        forged.first_column = Some(1);
        assert!(reader.stream(&forged, |_, _| Ok(())).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn out_of_bounds_forged_spans_changed_files_and_callback_failures_are_rejected() {
        let root = fixture();
        let mut reader = CheckpointReader::open(&root, 4).unwrap();
        assert!(reader.row_span("large", u64::MAX, 1).is_err());
        assert!(reader.row_span("large", (1 << 30) - 1, 2).is_err());
        assert!(reader.row_span("missing", 0, 1).is_err());
        let mut forged = reader.row_span("large", 123, 2).unwrap();
        forged.bytes += 1;
        assert!(reader.stream(&forged, |_, _| Ok(())).is_err());
        let span = reader.row_span("large", 123, 2).unwrap();
        let error = reader
            .stream(&span, |_, _| anyhow::bail!("consumer stopped"))
            .unwrap_err();
        assert!(error.to_string().contains("consumer stopped"));
        assert_eq!(reader.statistics().payload_bytes_read, 4);
        let file = OpenOptions::new()
            .write(true)
            .open(root.join("part.safetensors"))
            .unwrap();
        file.set_len(20).unwrap();
        assert!(reader.stream(&span, |_, _| Ok(())).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn symlink_shards_and_invalid_chunk_sizes_are_rejected_without_payload_reads() {
        let root = fixture();
        assert!(CheckpointReader::open(&root, 0).is_err());
        assert!(CheckpointReader::open(&root, MAX_CHUNK_BYTES + 1).is_err());
        std::fs::rename(
            root.join("part.safetensors"),
            root.join("original.safetensors"),
        )
        .unwrap();
        std::os::unix::fs::symlink("original.safetensors", root.join("part.safetensors")).unwrap();
        assert!(CheckpointReader::open(&root, 4).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
