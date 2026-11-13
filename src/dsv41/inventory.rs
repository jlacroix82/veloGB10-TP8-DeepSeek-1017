//! Read checkpoint headers for ownership planning, without reading tensor payloads or using CUDA.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

// serde_json::Value and BTreeMap ordinarily keep the last duplicate key. That
// would let two ambiguous tensor/index descriptions collapse into one valid
// geometry. Parse recursively with uniqueness first, including __metadata__.
struct UniqueJson(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> serde::de::Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                v: bool,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(n.into()))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                v: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = seq.next_element::<UniqueJson>()? {
                    values.push(value);
                }
                Ok(UniqueJson(values.into()))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, UniqueJson(value))) = map.next_entry::<String, UniqueJson>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON object key {key:?}"
                        )));
                    }
                }
                Ok(UniqueJson(values.into()))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn unique_json(bytes: &[u8]) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice::<UniqueJson>(bytes)?.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte offsets relative to the shard's data section, as stored in safetensors.
    pub data_offsets: [u64; 2],
    pub stored_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub data_start_bytes: u64,
    pub header_sha256: String,
    pub payload_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventory {
    pub index_sha256: String,
    pub index_declared_payload_bytes: Option<u64>,
    pub shards: Vec<ShardMetadata>,
    pub tensors: Vec<TensorMetadata>,
    pub total_payload_bytes: u64,
    pub total_file_bytes: u64,
    /// Logical application reads, excluding any operating-system read-ahead.
    pub metadata_bytes_read: u64,
    pub logical_payload_bytes_read: u64,
    pub payload_checksums_verified: bool,
}

pub fn dtype_size_bytes(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" | "F8_E8M0" | "F8_E8M0FNU" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => bail!("unsupported checkpoint dtype {dtype:?}; no guessed storage width"),
    })
}

fn shard_basename(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_)))
            && components.next().is_none()
            && !name.contains('\\')
            && name.ends_with(".safetensors"),
        "index shard must be one safetensors basename: {name:?}"
    );
    Ok(())
}

fn read_shard(path: &Path, name: &str) -> Result<(ShardMetadata, Vec<TensorMetadata>)> {
    let mut file = File::open(path).with_context(|| format!("open shard {name}"))?;
    ensure!(
        file.metadata()?.is_file(),
        "shard {name} is not a regular file"
    );
    let file_bytes = file.metadata()?.len();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("read header length: {name}"))?;
    let header_bytes = u64::from_le_bytes(prefix);
    ensure!(
        (2..=MAX_HEADER_BYTES).contains(&header_bytes),
        "invalid or oversized header in {name}"
    );
    let data_start = 8_u64
        .checked_add(header_bytes)
        .context("header length overflow")?;
    ensure!(
        data_start <= file_bytes,
        "header extends past end of {name}"
    );
    let mut header = vec![0; usize::try_from(header_bytes)?];
    file.read_exact(&mut header)?;
    let object =
        match unique_json(&header).with_context(|| format!("parse safetensors header {name}"))? {
            serde_json::Value::Object(value) => value,
            _ => bail!("safetensors header {name} is not an object"),
        };
    let payload_bytes = file_bytes - data_start;
    let mut tensors = Vec::new();
    for (tensor_name, value) in object {
        if tensor_name == "__metadata__" {
            continue;
        }
        ensure!(!tensor_name.is_empty(), "empty tensor name in {name}");
        #[derive(Deserialize)]
        struct HeaderTensor {
            dtype: String,
            shape: Vec<u64>,
            data_offsets: [u64; 2],
        }
        let tensor: HeaderTensor = serde_json::from_value(value)
            .with_context(|| format!("invalid tensor header {tensor_name} in {name}"))?;
        let elements = tensor
            .shape
            .iter()
            .try_fold(1_u64, |size, &dim| size.checked_mul(dim))
            .with_context(|| format!("shape product overflow: {tensor_name}"))?;
        let stored_bytes = elements
            .checked_mul(dtype_size_bytes(&tensor.dtype)?)
            .with_context(|| format!("byte length overflow: {tensor_name}"))?;
        let [begin, end] = tensor.data_offsets;
        ensure!(
            begin <= end && end <= payload_bytes,
            "out-of-bounds offsets for {tensor_name} in {name}"
        );
        ensure!(
            end - begin == stored_bytes,
            "shape/dtype byte length differs from offsets: {tensor_name}"
        );
        tensors.push(TensorMetadata {
            name: tensor_name,
            shard: name.to_owned(),
            dtype: tensor.dtype,
            shape: tensor.shape,
            data_offsets: tensor.data_offsets,
            stored_bytes,
        });
    }
    ensure!(!tensors.is_empty(), "shard {name} has no tensors");
    let mut intervals: Vec<_> = tensors.iter().filter(|t| t.stored_bytes > 0).collect();
    intervals.sort_by_key(|t| t.data_offsets[0]);
    let mut cursor = 0;
    for tensor in intervals {
        ensure!(
            tensor.data_offsets[0] == cursor,
            "overlap or uncovered payload before {} in {name}",
            tensor.name
        );
        cursor = tensor.data_offsets[1];
    }
    ensure!(
        cursor == payload_bytes,
        "uncovered trailing payload in {name}"
    );
    Ok((
        ShardMetadata {
            name: name.to_owned(),
            file_bytes,
            header_bytes,
            data_start_bytes: data_start,
            header_sha256: format!("{:x}", Sha256::digest(&header)),
            payload_bytes,
        },
        tensors,
    ))
}

pub fn read_inventory(model_dir: &Path) -> Result<TensorInventory> {
    let index_bytes = std::fs::read(model_dir.join("model.safetensors.index.json"))?;
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
        #[serde(default)]
        metadata: serde_json::Value,
    }
    let index: Index = serde_json::from_value(
        unique_json(&index_bytes).context("parse checkpoint index with unique keys")?,
    )
    .context("parse checkpoint index")?;
    ensure!(
        !index.weight_map.is_empty(),
        "checkpoint index has no tensors"
    );
    let declared = index
        .metadata
        .get("total_size")
        .map(|value| {
            value
                .as_u64()
                .context("index metadata.total_size must be an unsigned integer")
        })
        .transpose()?;
    let names: BTreeSet<_> = index.weight_map.values().cloned().collect();
    for name in &names {
        shard_basename(name)?;
    }
    let mut result = TensorInventory {
        index_sha256: format!("{:x}", Sha256::digest(&index_bytes)),
        index_declared_payload_bytes: declared,
        shards: Vec::new(),
        tensors: Vec::new(),
        total_payload_bytes: 0,
        total_file_bytes: 0,
        metadata_bytes_read: index_bytes.len() as u64,
        logical_payload_bytes_read: 0,
        payload_checksums_verified: false,
    };
    let mut seen = BTreeSet::new();
    for name in names {
        let (shard, tensors) = read_shard(&model_dir.join(&name), &name)?;
        for tensor in &tensors {
            ensure!(
                seen.insert(tensor.name.clone()),
                "duplicate tensor {} across shards",
                tensor.name
            );
            ensure!(
                index.weight_map.get(&tensor.name) == Some(&name),
                "header tensor {} is missing from, or disagrees with, the index",
                tensor.name
            );
        }
        result.total_payload_bytes = result
            .total_payload_bytes
            .checked_add(shard.payload_bytes)
            .context("payload total overflow")?;
        result.total_file_bytes = result
            .total_file_bytes
            .checked_add(shard.file_bytes)
            .context("file total overflow")?;
        result.metadata_bytes_read = result
            .metadata_bytes_read
            .checked_add(shard.data_start_bytes)
            .context("metadata total overflow")?;
        result.shards.push(shard);
        result.tensors.extend(tensors);
    }
    ensure!(
        seen.len() == index.weight_map.len(),
        "index names missing tensors from shard headers"
    );
    if let Some(size) = declared {
        ensure!(
            size == result.total_payload_bytes,
            "index total_size differs from exact stored payload"
        );
    }
    result.tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "dsv41-inventory-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn shard(&self, header: serde_json::Value, payload: u64) {
            let bytes = serde_json::to_vec(&header).unwrap();
            self.shard_raw(&bytes, payload);
        }
        fn shard_raw(&self, bytes: &[u8], payload: u64) {
            let mut file = File::create(self.0.join("model-01.safetensors")).unwrap();
            file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.set_len(8 + bytes.len() as u64 + payload).unwrap();
        }
        fn index(&self, map: serde_json::Value, size: u64) {
            std::fs::write(
                self.0.join("model.safetensors.index.json"),
                serde_json::to_vec(
                    &serde_json::json!({"metadata":{"total_size":size},"weight_map":map}),
                )
                .unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn header_only_large_sparse_checkpoint_preserves_exact_storage_types() {
        let f = Fixture::new();
        let count = 2_u64.pow(33);
        f.shard(
            serde_json::json!({
                "table":{"dtype":"I8","shape":[count,1],"data_offsets":[0,count]},
                "scale":{"dtype":"F8_E8M0","shape":[2],"data_offsets":[count,count+2]},
                "scalar":{"dtype":"F32","shape":[],"data_offsets":[count+2,count+6]}
            }),
            count + 6,
        );
        f.index(serde_json::json!({"table":"model-01.safetensors","scale":"model-01.safetensors","scalar":"model-01.safetensors"}),count+6);
        let inventory = read_inventory(&f.0).unwrap();
        assert_eq!(inventory.total_payload_bytes, count + 6);
        assert!(inventory.metadata_bytes_read < 4096);
        assert_eq!(inventory.logical_payload_bytes_read, 0);
        assert!(!inventory.payload_checksums_verified);
        assert_eq!(
            inventory
                .tensors
                .iter()
                .find(|t| t.name == "scale")
                .unwrap()
                .dtype,
            "F8_E8M0"
        );
    }

    #[test]
    fn corrupt_offsets_index_types_and_shape_overflow_are_refused() {
        for (header, map, payload, declared) in [
            (
                serde_json::json!({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,3]}}),
                serde_json::json!({"x":"model-01.safetensors"}),
                3,
                3,
            ),
            (
                serde_json::json!({"x":{"dtype":"I8","shape":[2],"data_offsets":[1,3]}}),
                serde_json::json!({"x":"model-01.safetensors"}),
                3,
                3,
            ),
            (
                serde_json::json!({"x":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,1]}}),
                serde_json::json!({"x":"model-01.safetensors"}),
                1,
                1,
            ),
            (
                serde_json::json!({"x":{"dtype":"I8","shape":[u64::MAX,2],"data_offsets":[0,1]}}),
                serde_json::json!({"x":"model-01.safetensors"}),
                1,
                1,
            ),
            (
                serde_json::json!({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
                serde_json::json!({"y":"model-01.safetensors"}),
                1,
                1,
            ),
            (
                serde_json::json!({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
                serde_json::json!({"x":"model-01.safetensors"}),
                1,
                2,
            ),
        ] {
            let f = Fixture::new();
            f.shard(header, payload);
            f.index(map, declared);
            assert!(read_inventory(&f.0).is_err());
        }
        let f = Fixture::new();
        f.index(serde_json::json!({"x":"../private.safetensors"}), 1);
        assert!(read_inventory(&f.0).is_err());
    }

    #[test]
    fn duplicate_header_and_index_keys_are_rejected_before_collapse() {
        let valid_header = br#"{"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        let valid_index =
            br#"{"metadata":{"total_size":1},"weight_map":{"x":"model-01.safetensors"}}"#;
        // Use raw bytes: constructing serde_json::Value first would erase the
        // very duplicates the actual file reader must reject.
        let cases: Vec<(&[u8], &[u8])> = vec![
            (br#"{"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#, valid_index),
            (br#"{"x":{"dtype":"I8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#, valid_index),
            (br#"{"__metadata__":{"tag":"one","tag":"one"},"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#, valid_index),
            (valid_header, br#"{"metadata":{"total_size":1},"weight_map":{"x":"model-01.safetensors","x":"model-01.safetensors"}}"#),
            (valid_header, br#"{"metadata":{"total_size":1},"weight_map":{"x":"model-01.safetensors"},"weight_map":{"x":"model-01.safetensors"}}"#),
            (valid_header, br#"{"metadata":{"total_size":1,"total_size":1},"weight_map":{"x":"model-01.safetensors"}}"#),
        ];
        for (header, index) in cases {
            let f = Fixture::new();
            f.shard_raw(header, 1);
            std::fs::write(f.0.join("model.safetensors.index.json"), index).unwrap();
            let error = read_inventory(&f.0).unwrap_err();
            assert!(
                format!("{error:#}").contains("duplicate JSON object key"),
                "wrong refusal: {error:#}"
            );
        }
        let f = Fixture::new();
        f.shard_raw(valid_header, 1);
        std::fs::write(f.0.join("model.safetensors.index.json"), valid_index).unwrap();
        assert_eq!(read_inventory(&f.0).unwrap().total_payload_bytes, 1);
    }
}
