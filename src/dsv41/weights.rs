//! Exact1017 text-trunk TP8 tensor ownership and native layout manifest.
//! Header-only. This does not read payloads, allocate GPU memory or enable serving.
//! Source contracts: verified model.py/convert.py, official snapshot1bae0e1088993b9b82de08f59de5be5a34e42170.
use super::{
    config::Dsv41Config,
    inventory::{dtype_size_bytes, TensorInventory, TensorMetadata},
    plan::row_ranges,
    reader::{CheckpointReader, TensorSpan},
};
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
pub const WORLD: usize = 8;
pub const REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";
pub const INDEX_SHA256: &str = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8";
pub const HEADER_GEOMETRY_SHA256: &str =
    "6dd44cb0abedd3b97863db19c46dffa4883715217f71914ba77ee895e81ce6e9";
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum WoaReconstructionPolicy {
    RequireExact,
    OfficialBf16Round,
}
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum Group {
    Global,
    Layer(usize),
}
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum Selection {
    Full,
    Rows {
        first: u64,
        count: u64,
        allocated_rows: u64,
    },
    Rectangle2d {
        first_row: u64,
        row_count: u64,
        first_column: u64,
        column_count: u64,
    },
}
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum Conversion {
    Native,
    WoaFp8ToBf16 {
        scale_tensor: String,
        policy: WoaReconstructionPolicy,
    },
}
#[derive(Debug, Clone, Serialize)]
pub struct TensorLayout {
    pub name: String,
    pub dtype: String,
    pub source_shape: Vec<u64>,
    pub selected_shape: Vec<u64>,
    pub selection: Selection,
    pub group: Group,
    pub runtime_use: String,
    pub native_bytes: u64,
    pub allocation_padding_bytes: u64,
    pub resident_bytes: u64,
    pub coupled_scale: Option<String>,
    pub conversion: Conversion,
}
impl TensorLayout {
    pub fn span(&self, reader: &CheckpointReader) -> Result<TensorSpan> {
        let t = reader.tensor(&self.name)?;
        ensure!(
            t.dtype == self.dtype && t.shape == self.source_shape,
            "layout/source header changed"
        );
        let span = match self.selection {
            Selection::Full => reader.full_span(&self.name)?,
            Selection::Rows { first, count, .. } => reader.row_span(&self.name, first, count)?,
            Selection::Rectangle2d {
                first_row,
                row_count,
                first_column,
                column_count,
            } => reader.rectangle_span(
                &self.name,
                first_row,
                row_count,
                first_column,
                column_count,
            )?,
        };
        ensure!(
            span.shape == self.selected_shape && span.bytes == self.native_bytes,
            "layout selected native shape/bytes changed"
        );
        Ok(span)
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct RankManifest {
    pub schema_version: u32,
    pub model_revision: String,
    pub rank: usize,
    pub world: usize,
    pub source_index_sha256: String,
    pub source_header_geometry_sha256: String,
    pub header_geometry_identity_verified: bool,
    pub payload_identity_verified: bool,
    pub woa_policy: WoaReconstructionPolicy,
    pub entries: Vec<TensorLayout>,
    pub ignored_tensor_count: usize,
    pub native_bytes: u64,
    pub resident_bytes: u64,
    pub allocation_padding_bytes: u64,
    pub manifest_sha256: String,
    pub serve_supported: bool,
}
impl RankManifest {
    pub fn verify_identity(&self) -> Result<()> {
        ensure!(
            self.header_geometry_identity_verified
                && !self.serve_supported
                && self.rank < WORLD
                && self.world == WORLD
                && self.model_revision == REVISION
                && self.source_index_sha256 == INDEX_SHA256
                && self.source_header_geometry_sha256 == HEADER_GEOMETRY_SHA256,
            "manifest is not the exact1017 TP8 header-qualified layout"
        );
        ensure!(
            self.manifest_sha256 == manifest_digest(self)?,
            "manifest changed since construction"
        );
        Ok(())
    }
    pub fn layer(&self, layer: usize) -> Result<Vec<&TensorLayout>> {
        ensure!(layer < 40, "1017 trunk layer outside0..40");
        Ok(self
            .entries
            .iter()
            .filter(|e| e.group == Group::Layer(layer))
            .collect())
    }
    pub fn global(&self) -> Vec<&TensorLayout> {
        self.entries
            .iter()
            .filter(|e| e.group == Group::Global)
            .collect()
    }
}
fn manifest_digest(m: &RankManifest) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            m.schema_version,
            &m.model_revision,
            m.rank,
            m.world,
            &m.source_index_sha256,
            &m.source_header_geometry_sha256,
            m.header_geometry_identity_verified,
            m.payload_identity_verified,
            m.woa_policy,
            &m.entries,
            m.ignored_tensor_count,
            m.native_bytes,
            m.resident_bytes,
            m.allocation_padding_bytes,
            m.serve_supported
        ))?)
    ))
}
pub fn header_geometry_sha256(inventory: &TensorInventory) -> Result<String> {
    let mut tensors: Vec<_> = inventory.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let mut hash = Sha256::new();
    hash.update(b"dsv41-native-header-geometry-v1\0");
    let mut seen = BTreeSet::new();
    for t in tensors {
        ensure!(
            seen.insert(t.name.as_str()),
            "duplicate tensor in manifest inventory"
        );
        validate_tensor(t)?;
        for text in [&t.name, &t.dtype] {
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.update((t.shape.len() as u64).to_le_bytes());
        for d in &t.shape {
            hash.update(d.to_le_bytes());
        }
        hash.update(t.stored_bytes.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn validate_tensor(t: &TensorMetadata) -> Result<()> {
    ensure!(!t.name.is_empty()&&t.name.split('.').all(|p|!p.is_empty()&&p.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_')),
        "invalid/traversal tensor name");
    ensure!(
        !t.shard.is_empty()
            && !t.shard.contains('/')
            && !t.shard.contains('\\')
            && t.shard.ends_with(".safetensors")
            && t.shard != ".safetensors",
        "invalid/traversal shard name"
    );
    let elements = t.shape.iter().try_fold(1_u64, |n, &d| {
        ensure!(d > 0, "zero tensor extent");
        n.checked_mul(d).context("shape overflow")
    })?;
    ensure!(
        elements.checked_mul(dtype_size_bytes(&t.dtype)?) == Some(t.stored_bytes)
            && t.data_offsets[1].checked_sub(t.data_offsets[0]) == Some(t.stored_bytes),
        "native tensor shape/byte extent mismatch"
    );
    Ok(())
}
#[derive(Clone, Copy)]
enum Rule {
    Full,
    RowsEqual,
    ColumnsEqual,
    Expert(usize),
    Engram(u64),
    Ignored,
}
#[derive(Clone)]
struct Spec {
    dtype: &'static str,
    shape: Vec<u64>,
    group: Group,
    usage: String,
    rule: Rule,
    coupled_scale: Option<String>,
    conversion: Conversion,
}
fn insert(
    map: &mut BTreeMap<String, Spec>,
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    group: Group,
    usage: &str,
    rule: Rule,
) -> Result<()> {
    ensure!(
        map.insert(
            name,
            Spec {
                dtype,
                shape,
                group,
                usage: usage.to_owned(),
                rule,
                coupled_scale: None,
                conversion: Conversion::Native
            }
        )
        .is_none(),
        "duplicate expected tensor"
    );
    Ok(())
}
fn quant(
    map: &mut BTreeMap<String, Spec>,
    prefix: &str,
    rows: u64,
    columns: u64,
    fp4: bool,
    group: Group,
    usage: &str,
    rule: Rule,
    woa: Option<WoaReconstructionPolicy>,
) -> Result<()> {
    ensure!(rows % 32 == 0 && columns % 32 == 0, "native quant geometry");
    let weight = format!("{prefix}.weight");
    let scale = format!("{prefix}.scale");
    insert(
        map,
        weight.clone(),
        if fp4 { "I8" } else { "F8_E4M3" },
        vec![rows, columns / if fp4 { 2 } else { 1 }],
        group.clone(),
        usage,
        rule.clone(),
    )?;
    let scale_rule = match rule {
        Rule::RowsEqual => Rule::RowsEqual,
        Rule::ColumnsEqual => Rule::ColumnsEqual,
        other => other,
    };
    insert(
        map,
        scale.clone(),
        "F8_E8M0",
        vec![if fp4 { rows } else { rows / 32 }, columns / 32],
        group,
        usage,
        scale_rule,
    )?;
    let w = map.get_mut(&weight).unwrap();
    w.coupled_scale = Some(scale.clone());
    if let Some(policy) = woa {
        w.conversion = Conversion::WoaFp8ToBf16 {
            scale_tensor: scale,
            policy,
        };
    }
    Ok(())
}
fn expected(
    config: &Dsv41Config,
    policy: WoaReconstructionPolicy,
) -> Result<BTreeMap<String, Spec>> {
    config.validate()?;
    let mut map = BTreeMap::new();
    let dim = config.dim as u64;
    let inter = config.moe_inter_dim as u64;
    insert(
        &mut map,
        "embed.weight".into(),
        "BF16",
        vec![config.vocab_size as u64, dim],
        Group::Global,
        "vocabularyembedding",
        Rule::RowsEqual,
    )?;
    insert(
        &mut map,
        "head.weight".into(),
        "BF16",
        vec![config.vocab_size as u64, dim],
        Group::Global,
        "F32head via exact registerpromotion",
        Rule::RowsEqual,
    )?;
    insert(
        &mut map,
        "norm.weight".into(),
        "BF16",
        vec![dim],
        Group::Global,
        "finalRMS",
        Rule::Full,
    )?;
    for layer in 0..config.n_layers {
        let group = Group::Layer(layer);
        let p = format!("layers.{layer}");
        for stage in ["attn", "ffn"] {
            insert(
                &mut map,
                format!("{p}.hc_{stage}_fn"),
                "F32",
                vec![24, 4 * dim],
                group.clone(),
                "mHCnativeF32",
                Rule::Full,
            )?;
            insert(
                &mut map,
                format!("{p}.hc_{stage}_base"),
                "F32",
                vec![24],
                group.clone(),
                "mHCnativeF32",
                Rule::Full,
            )?;
            insert(
                &mut map,
                format!("{p}.hc_{stage}_scale"),
                "F32",
                vec![3],
                group.clone(),
                "mHCnativeF32",
                Rule::Full,
            )?;
        }
        insert(
            &mut map,
            format!("{p}.attn.attn_sink"),
            "F32",
            vec![64],
            group.clone(),
            "local8attentionheads",
            Rule::RowsEqual,
        )?;
        for (suffix, width) in [
            ("attn.q_norm.weight", 1280),
            ("attn.kv_norm.weight", 512),
            ("attn_norm.weight", dim),
            ("ffn_norm.weight", dim),
        ] {
            insert(
                &mut map,
                format!("{p}.{suffix}"),
                "BF16",
                vec![width],
                group.clone(),
                "RMSnativeweights",
                Rule::Full,
            )?;
        }
        insert(
            &mut map,
            format!("{p}.ffn.gate.weight"),
            "BF16",
            vec![384, dim],
            group.clone(),
            "F32router via exact registerpromotion",
            Rule::Full,
        )?;
        insert(
            &mut map,
            format!("{p}.ffn.gate.bias"),
            "F32",
            vec![384],
            group.clone(),
            "text routing correction",
            Rule::Full,
        )?;
        insert(
            &mut map,
            format!("{p}.ffn.gate.bias_vl"),
            "F32",
            vec![384],
            group.clone(),
            "excluded vision routing correction",
            Rule::Ignored,
        )?;
        for (name, rows, cols, rule, woa) in [
            ("wq_a", 1280, dim, Rule::Full, None),
            ("wq_b", 32768, 1280, Rule::RowsEqual, None),
            ("wkv", 512, dim, Rule::Full, None),
            ("wo_a", 8192, 4096, Rule::RowsEqual, Some(policy)),
            ("wo_b", dim, 8192, Rule::ColumnsEqual, None),
        ] {
            quant(
                &mut map,
                &format!("{p}.attn.{name}"),
                rows,
                cols,
                false,
                group.clone(),
                if name == "wo_a" {
                    "WO-A BF16 blockdiagonal"
                } else {
                    "attentionnativeFP8"
                },
                rule,
                woa,
            )?;
        }
        if config.kv_source_layers.contains(&layer) {
            for suffix in ["norm.weight", "wkv.weight"] {
                insert(
                    &mut map,
                    format!("{p}.attn.compressor.{suffix}"),
                    "BF16",
                    if suffix == "norm.weight" {
                        vec![512]
                    } else {
                        vec![512, dim]
                    },
                    group.clone(),
                    if config.compress_ratios[layer] > 1 {
                        "F32compressor via exact registerpromotion"
                    } else {
                        "BF16compressor"
                    },
                    Rule::Full,
                )?;
            }
            if config.compress_ratios[layer] > 1 {
                insert(
                    &mut map,
                    format!("{p}.attn.compressor.wgate.weight"),
                    "BF16",
                    vec![512, dim],
                    group.clone(),
                    "F32compressorgate via exact registerpromotion",
                    Rule::Full,
                )?;
            }
        }
        if config.index_source_layers.contains(&layer) {
            quant(
                &mut map,
                &format!("{p}.attn.indexer.wq_b"),
                4096,
                1280,
                false,
                group.clone(),
                "local4indexheads",
                Rule::RowsEqual,
                None,
            )?;
            insert(
                &mut map,
                format!("{p}.attn.indexer.weights_proj.weight"),
                "BF16",
                vec![32, dim],
                group.clone(),
                "local4indexweights",
                Rule::RowsEqual,
            )?;
            if config.kv_source_layers.contains(&layer) {
                insert(
                    &mut map,
                    format!("{p}.attn.indexer.wk.weight"),
                    "BF16",
                    vec![128, 512],
                    group.clone(),
                    "sharedindexkey",
                    Rule::Full,
                )?;
                insert(
                    &mut map,
                    format!("{p}.attn.indexer.k_norm.weight"),
                    "BF16",
                    vec![128],
                    group.clone(),
                    "sharedindexkeyRMS",
                    Rule::Full,
                )?;
            }
        }
        for expert in 0..config.n_routed_experts {
            for (name, rows, cols) in [("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)] {
                quant(
                    &mut map,
                    &format!("{p}.ffn.experts.{expert}.{name}"),
                    rows,
                    cols,
                    true,
                    group.clone(),
                    "ownednativeFP4expert",
                    Rule::Expert(expert),
                    None,
                )?;
            }
        }
        for (name, rows, cols) in [("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)] {
            quant(
                &mut map,
                &format!("{p}.ffn.shared_experts.{name}"),
                rows,
                cols,
                false,
                group.clone(),
                "replicatednativeFP8sharedexpert",
                Rule::Full,
                None,
            )?;
        }
    }
    for (&layer, &rows) in config
        .engram_layer_ids
        .iter()
        .zip(&config.engram_num_embeddings)
    {
        let group = Group::Layer(layer);
        let p = format!("layers.{layer}.engram");
        insert(
            &mut map,
            format!("{p}.embed.weight"),
            "F8_E4M3",
            vec![rows, 256],
            group.clone(),
            "Engramceilrows",
            Rule::Engram(rows),
        )?;
        insert(
            &mut map,
            format!("{p}.embed.scale"),
            "F8_E8M0",
            vec![rows, 8],
            group.clone(),
            "Engramceilrows",
            Rule::Engram(rows),
        )?;
        map.get_mut(&format!("{p}.embed.weight"))
            .unwrap()
            .coupled_scale = Some(format!("{p}.embed.scale"));
        for name in ["q_weight", "k_weight"] {
            insert(
                &mut map,
                format!("{p}.{name}"),
                "BF16",
                vec![4, dim],
                group.clone(),
                "replicatedEngramgate",
                Rule::Full,
            )?;
        }
        quant(
            &mut map,
            &format!("{p}.wkv"),
            25600,
            6144,
            false,
            group,
            "replicatedEngramprojection",
            Rule::Full,
            None,
        )?;
    }
    ensure!(
        map.len() == 93418,
        "unexpected official1017 requiredheader count"
    );
    Ok(map)
}
fn layout(name: &str, spec: &Spec, rank: usize) -> Result<Option<TensorLayout>> {
    let selection = match spec.rule {
        Rule::Ignored => return Ok(None),
        Rule::Expert(expert) => {
            if expert / 48 != rank {
                return Ok(None);
            }
            Selection::Full
        }
        Rule::Full => Selection::Full,
        Rule::RowsEqual => {
            ensure!(spec.shape[0] % 8 == 0, "uneven equal row shard");
            let count = spec.shape[0] / 8;
            Selection::Rows {
                first: rank as u64 * count,
                count,
                allocated_rows: count,
            }
        }
        Rule::ColumnsEqual => {
            ensure!(
                spec.shape.len() == 2 && spec.shape[1] % 8 == 0,
                "uneven equal column shard"
            );
            let count = spec.shape[1] / 8;
            Selection::Rectangle2d {
                first_row: 0,
                row_count: spec.shape[0],
                first_column: rank as u64 * count,
                column_count: count,
            }
        }
        Rule::Engram(rows) => {
            let range = row_ranges(rows, WORLD)?[rank];
            let allocated = rows.div_ceil(8);
            Selection::Rows {
                first: range.start,
                count: range.count,
                allocated_rows: allocated,
            }
        }
    };
    let mut selected = spec.shape.clone();
    let mut padding = 0_u64;
    match selection {
        Selection::Full => {}
        Selection::Rows {
            count,
            allocated_rows,
            ..
        } => {
            selected[0] = count;
            let row = spec
                .shape
                .iter()
                .skip(1)
                .try_fold(dtype_size_bytes(spec.dtype)?, |n, &d| {
                    n.checked_mul(d).context("padding row overflow")
                })?;
            padding = (allocated_rows - count)
                .checked_mul(row)
                .context("padding bytes overflow")?;
        }
        Selection::Rectangle2d {
            row_count,
            column_count,
            ..
        } => selected = vec![row_count, column_count],
    }
    let bytes = selected
        .iter()
        .try_fold(dtype_size_bytes(spec.dtype)?, |n, &d| {
            n.checked_mul(d).context("selected byte overflow")
        })?;
    let resident = if matches!(spec.conversion, Conversion::WoaFp8ToBf16 { .. }) {
        bytes.checked_mul(2).context("WO-A BF16 bytes overflow")?
    } else {
        bytes
    };
    Ok(Some(TensorLayout {
        name: name.into(),
        dtype: spec.dtype.into(),
        source_shape: spec.shape.clone(),
        selected_shape: selected,
        selection,
        group: spec.group.clone(),
        runtime_use: spec.usage.clone(),
        native_bytes: bytes,
        allocation_padding_bytes: padding,
        resident_bytes: resident
            .checked_add(padding)
            .context("resident byte overflow")?,
        coupled_scale: spec.coupled_scale.clone(),
        conversion: spec.conversion.clone(),
    }))
}
fn build_impl(
    config: &Dsv41Config,
    inventory: &TensorInventory,
    rank: usize,
    policy: WoaReconstructionPolicy,
    require_official_identity: bool,
) -> Result<RankManifest> {
    ensure!(rank < WORLD, "TP8 rank outside0..8");
    let expected = expected(config, policy)?;
    let geometry = header_geometry_sha256(inventory)?;
    if require_official_identity {
        ensure!(
            inventory.index_sha256 == INDEX_SHA256
                && geometry == HEADER_GEOMETRY_SHA256
                && inventory.tensors.len() == 96085
                && inventory.shards.len() == 48
                && inventory.total_payload_bytes == 510286023000,
            "inventory is not exact official1017 header identity"
        );
    }
    let mut found = BTreeSet::new();
    let mut ignored = 0;
    let mut source_bytes = 0_u64;
    for tensor in &inventory.tensors {
        source_bytes = source_bytes
            .checked_add(tensor.stored_bytes)
            .context("source bytes overflow")?;
        if let Some(spec) = expected.get(&tensor.name) {
            ensure!(
                tensor.dtype == spec.dtype && tensor.shape == spec.shape,
                "unknown/mismatched1017 geometry for{}",
                tensor.name
            );
            found.insert(tensor.name.as_str());
            if matches!(spec.rule, Rule::Ignored) {
                ignored += 1;
            }
        } else {
            ensure!(
                !tensor.name.starts_with("layers."),
                "unknown1017 trunk tensor{}",
                tensor.name
            );
            ensure!(
                tensor.name.starts_with("mtp.")
                    || tensor.name.starts_with("vision.")
                    || tensor.name.starts_with("aligner.")
                    || matches!(
                        tensor.name.as_str(),
                        "image_start" | "image_end" | "image_newline"
                    ),
                "unknown1017 tensor{}",
                tensor.name
            );
            ignored += 1;
        }
    }
    ensure!(
        source_bytes == inventory.total_payload_bytes,
        "inventory byte sum changed"
    );
    ensure!(
        found.len() == expected.len(),
        "missing1017 tensors (found{} expected{})",
        found.len(),
        expected.len()
    );
    let mut entries = Vec::new();
    let mut native = 0_u64;
    let mut resident = 0_u64;
    let mut padding = 0_u64;
    for (name, spec) in &expected {
        if let Some(entry) = layout(name, spec, rank)? {
            native = native
                .checked_add(entry.native_bytes)
                .context("rank native sum overflow")?;
            resident = resident
                .checked_add(entry.resident_bytes)
                .context("rank resident sum overflow")?;
            padding = padding
                .checked_add(entry.allocation_padding_bytes)
                .context("rank padding sum overflow")?;
            entries.push(entry);
        }
    }
    ensure!(
        entries.len() == 12738,
        "unexpected rank native tensor count"
    );
    let mut out = RankManifest {
        schema_version: 1,
        model_revision: REVISION.into(),
        rank,
        world: WORLD,
        source_index_sha256: inventory.index_sha256.clone(),
        source_header_geometry_sha256: geometry,
        header_geometry_identity_verified: require_official_identity,
        payload_identity_verified: false,
        woa_policy: policy,
        entries,
        ignored_tensor_count: ignored,
        native_bytes: native,
        resident_bytes: resident,
        allocation_padding_bytes: padding,
        manifest_sha256: String::new(),
        serve_supported: false,
    };
    out.manifest_sha256 = manifest_digest(&out)?;
    Ok(out)
}
/// Full40layer manifest, canonical header identity required before any runtime loader can use it.
/// Sourcepayload identity remains independently recorded by the retained fullmodelverification.
pub fn build_rank_manifest(
    config: &Dsv41Config,
    inventory: &TensorInventory,
    rank: usize,
    policy: WoaReconstructionPolicy,
) -> Result<RankManifest> {
    build_impl(config, inventory, rank, policy, true)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Dsv41Config {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dsv41_1017_config.json"))
                .unwrap();
        Dsv41Config::from_values(&v["hf_config"], &v["inference_config"]).unwrap()
    }
    fn fixture() -> TensorInventory {
        let specs = expected(&config(), WoaReconstructionPolicy::OfficialBf16Round).unwrap();
        let mut offset = 0;
        let tensors = specs
            .into_iter()
            .map(|(name, s)| {
                let bytes = s.shape.iter().product::<u64>() * dtype_size_bytes(s.dtype).unwrap();
                let t = TensorMetadata {
                    name,
                    shard: "part.safetensors".into(),
                    dtype: s.dtype.into(),
                    shape: s.shape,
                    data_offsets: [offset, offset + bytes],
                    stored_bytes: bytes,
                };
                offset += bytes;
                t
            })
            .collect();
        TensorInventory {
            index_sha256: "syntheticCPUfixture".into(),
            index_declared_payload_bytes: Some(offset),
            shards: vec![],
            tensors,
            total_payload_bytes: offset,
            total_file_bytes: offset,
            metadata_bytes_read: 0,
            logical_payload_bytes_read: 0,
            payload_checksums_verified: false,
        }
    }
    fn entry<'a>(m: &'a RankManifest, name: &str) -> &'a TensorLayout {
        m.entries.iter().find(|e| e.name == name).unwrap()
    }
    #[test]
    fn full40layer_tp8_heads_columns_experts_vocab_and_engram_are_exact() {
        let inv = fixture();
        let cfg = config();
        for rank in 0..8 {
            let m = build_impl(
                &cfg,
                &inv,
                rank,
                WoaReconstructionPolicy::OfficialBf16Round,
                false,
            )
            .unwrap();
            assert!(!m.header_geometry_identity_verified);
            assert!(m.verify_identity().is_err());
            assert_eq!(m.entries.len(), 12738);
            assert_eq!(
                entry(&m, "layers.0.attn.wq_b.weight").selected_shape,
                [4096, 1280]
            );
            assert_eq!(
                entry(&m, "layers.0.attn.wq_b.scale").selected_shape,
                [128, 40]
            );
            assert_eq!(
                entry(&m, "layers.0.attn.wo_a.weight").selected_shape,
                [1024, 4096]
            );
            assert_eq!(
                entry(&m, "layers.0.attn.wo_b.weight").selection,
                Selection::Rectangle2d {
                    first_row: 0,
                    row_count: 5120,
                    first_column: (rank * 1024) as u64,
                    column_count: 1024
                }
            );
            assert_eq!(
                entry(&m, "layers.0.attn.wo_b.scale").selection,
                Selection::Rectangle2d {
                    first_row: 0,
                    row_count: 160,
                    first_column: (rank * 32) as u64,
                    column_count: 32
                }
            );
            assert_eq!(
                entry(&m, "layers.24.attn.indexer.wq_b.weight").selected_shape,
                [512, 1280]
            );
            assert_eq!(
                entry(&m, "layers.24.attn.indexer.weights_proj.weight").selected_shape,
                [4, 5120]
            );
            assert_eq!(entry(&m, "head.weight").selected_shape, [16160, 5120]);
            assert_eq!(entry(&m, "embed.weight").selected_shape, [16160, 5120]);
            let experts = m
                .entries
                .iter()
                .filter(|e| e.name.starts_with("layers.0.ffn.experts."))
                .count();
            assert_eq!(experts, 48 * 6);
            assert!(m
                .entries
                .iter()
                .any(|e| e.name == format!("layers.39.ffn.experts.{}.w2.weight", rank * 48)));
            assert!(!m
                .entries
                .iter()
                .any(|e| e.name
                    == format!("layers.0.ffn.experts.{}.w1.weight", (rank * 48 + 48) % 384)));
            let engram = entry(&m, "layers.14.engram.embed.weight");
            assert_eq!(
                engram.selected_shape[0],
                if rank == 7 { 48002080 } else { 48002086 }
            );
            assert_eq!(
                engram.allocation_padding_bytes,
                if rank == 7 { 6 * 256 } else { 0 }
            );
            assert_eq!(
                entry(&m, "layers.14.engram.wkv.weight").selected_shape,
                [25600, 6144]
            );
            assert!(m.layer(40).is_err());
        }
    }
    #[test]
    fn missing_unknown_mismatched_unpinned_and_traversal_metadata_fail_closed() {
        let cfg = config();
        let mut inv = fixture();
        assert!(build_rank_manifest(&cfg, &inv, 0, WoaReconstructionPolicy::RequireExact).is_err());
        let removed = inv.tensors.pop().unwrap();
        inv.total_payload_bytes -= removed.stored_bytes;
        assert!(build_impl(&cfg, &inv, 0, WoaReconstructionPolicy::RequireExact, false).is_err());
        inv.tensors.push(removed);
        inv.total_payload_bytes = inv.tensors.iter().map(|t| t.stored_bytes).sum();
        inv.tensors[0].name = "layers.0.unknown".into();
        assert!(build_impl(&cfg, &inv, 0, WoaReconstructionPolicy::RequireExact, false).is_err());
        inv.tensors[0].name = "../escape".into();
        assert!(header_geometry_sha256(&inv).is_err());
        let mut inv = fixture();
        inv.tensors[0].shard = "../escape.safetensors".into();
        assert!(header_geometry_sha256(&inv).is_err());
        let mut inv = fixture();
        inv.tensors[0].dtype = "F32".into();
        assert!(build_impl(&cfg, &inv, 0, WoaReconstructionPolicy::RequireExact, false).is_err());
    }
}
