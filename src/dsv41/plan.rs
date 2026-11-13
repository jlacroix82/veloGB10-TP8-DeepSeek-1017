//! Header-only ownership/accounting for the native 1017 checkpoint. No payload is
//! read or converted, and these storage numbers are not a runtime memory budget.

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use super::config::Dsv41Config;
use super::inventory::{TensorInventory, TensorMetadata};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TensorBytes {
    pub data_bytes: u64,
    pub scale_bytes: u64,
    pub total_bytes: u64,
}

impl TensorBytes {
    fn add(&mut self, bytes: u64, scale: bool) -> Result<()> {
        let target = if scale {
            &mut self.scale_bytes
        } else {
            &mut self.data_bytes
        };
        *target = target
            .checked_add(bytes)
            .context("tensor byte count overflow")?;
        self.total_bytes = self
            .total_bytes
            .checked_add(bytes)
            .context("total byte count overflow")?;
        Ok(())
    }

    fn plus(self, other: Self) -> Result<Self> {
        let mut result = self;
        result.add(other.data_bytes, false)?;
        result.add(other.scale_bytes, true)?;
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IndexRange {
    pub start: u64,
    pub end: u64,
    pub count: u64,
}

fn check_world(world: usize) -> Result<u64> {
    ensure!(
        world > 0 && world.is_power_of_two() && world <= 16,
        "ownership world must be a power of two in 1..=16, got {world}"
    );
    Ok(world as u64)
}

/// Official Engram ownership: rank r starts at r*ceil(rows/world), with the
/// final valid range clipped to the table. Allocation padding is reported
/// separately by the table plan; it is never counted as checkpoint payload.
pub fn row_ranges(rows: u64, world: usize) -> Result<Vec<IndexRange>> {
    let w = check_world(world)?;
    let width = rows / w + u64::from(rows % w != 0);
    let mut ranges = Vec::with_capacity(world);
    for rank in 0..world {
        let start = (rank as u64)
            .checked_mul(width)
            .context("row start overflow")?
            .min(rows);
        let count = (rows - start).min(width);
        let end = start.checked_add(count).context("row range overflow")?;
        ranges.push(IndexRange { start, end, count });
    }
    ensure!(
        ranges.last().map(|r| r.end) == Some(rows),
        "row partition did not cover its source"
    );
    Ok(ranges)
}

pub fn row_owner(rows: u64, world: usize, row: u64) -> Result<usize> {
    let w = check_world(world)?;
    ensure!(row < rows, "row {row} outside table with {rows} rows");
    let width = rows / w + u64::from(rows % w != 0);
    Ok((row / width) as usize)
}

pub fn expert_ranges(experts: u64, world: usize) -> Result<Vec<IndexRange>> {
    let w = check_world(world)?;
    ensure!(
        experts > 0 && experts % w == 0,
        "expert count {experts} must be positive and divisible by world {world}"
    );
    row_ranges(experts, world)
}

pub fn expert_owner(experts: u64, world: usize, expert: u64) -> Result<usize> {
    let w = check_world(world)?;
    ensure!(
        experts > 0 && experts % w == 0 && expert < experts,
        "expert {expert} / count {experts} is invalid at world {world}"
    );
    Ok((expert / (experts / w)) as usize)
}

#[derive(Clone, Debug, Serialize)]
pub struct EngramRankSlice {
    pub rank: usize,
    pub rows: IndexRange,
    /// Offsets are relative to the safetensors DATA SECTION, not the file start.
    /// They do not include the eight-byte header length or JSON header itself.
    pub weight_data_offsets: [u64; 2],
    pub scale_data_offsets: [u64; 2],
    pub bytes: TensorBytes,
    pub proposed_allocated_rows: u64,
    pub padding_rows: u64,
    pub proposed_allocation_bytes: TensorBytes,
    pub padding_bytes: TensorBytes,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngramTablePlan {
    pub layer: usize,
    pub rows: u64,
    pub weight_name: String,
    pub weight_shard: String,
    pub scale_name: String,
    pub scale_shard: String,
    pub data_bytes_per_row: u64,
    pub scale_bytes_per_row: u64,
    pub native_bytes: TensorBytes,
    pub row_shards: Vec<EngramRankSlice>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RankWeightPlan {
    pub rank: usize,
    pub trunk_experts: IndexRange,
    pub owned_expert_tensor_count: usize,
    pub expert_bytes: TensorBytes,
    pub replicated_trunk_bytes: TensorBytes,
    pub engram_table_bytes: TensorBytes,
    pub native_stored_bytes: TensorBytes,
    pub engram_padding_bytes: TensorBytes,
    pub proposed_storage_allocation_bytes: TensorBytes,
}

#[derive(Clone, Debug, Serialize)]
pub struct StorageScenario {
    pub engram_placement: &'static str,
    pub ranks: Vec<RankWeightPlan>,
    pub cluster_native_stored_bytes: TensorBytes,
    pub cluster_proposed_storage_allocation_bytes: TensorBytes,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeMemoryBudget {
    pub activation_bytes: Option<u64>,
    pub kv_and_index_cache_bytes: Option<u64>,
    pub workspace_and_prefill_bytes: Option<u64>,
    pub cuda_context_and_kernel_bytes: Option<u64>,
    pub staging_and_conversion_peak_bytes: Option<u64>,
    pub operating_system_and_page_cache_bytes: Option<u64>,
    pub fits_available_memory: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OwnershipPlan {
    pub schema_version: u32,
    pub kind: &'static str,
    pub world: usize,
    pub serving_supported: bool,
    pub source_tensor_count: usize,
    pub source_payload_bytes: u64,
    pub selected_plain_text_bytes: TensorBytes,
    pub excluded_bytes: BTreeMap<String, TensorBytes>,
    pub replicated_trunk_tensor_count: usize,
    pub replicated_trunk_bytes: TensorBytes,
    pub trunk_expert_bytes: TensorBytes,
    pub engram_tables: Vec<EngramTablePlan>,
    pub scenarios: Vec<StorageScenario>,
    pub runtime_memory_budget: RuntimeMemoryBudget,
    pub notes: Vec<&'static str>,
}

struct Geometry {
    dim: u64,
    inter: u64,
    layers: usize,
    experts: u64,
    mtp_layers: usize,
    mtp_experts: u64,
    table_width: u64,
    tables: BTreeMap<usize, u64>,
}

/// The public plan is deliberately TP8-only. The generic partition helpers above
/// also test the older 256-expert geometry without calling it a supported model.
pub fn build_plan(
    config: &Dsv41Config,
    inventory: &TensorInventory,
    world: usize,
) -> Result<OwnershipPlan> {
    config.validate()?;
    ensure!(
        world == 8,
        "the 1017 ownership plan currently supports world8 only"
    );
    ensure!(
        config.engram_layer_ids.len() == config.engram_num_embeddings.len(),
        "Engram layer/row count mismatch"
    );
    let mut tables = BTreeMap::new();
    for (&layer, &rows) in config
        .engram_layer_ids
        .iter()
        .zip(&config.engram_num_embeddings)
    {
        ensure!(
            tables.insert(layer, rows).is_none(),
            "duplicate Engram layer {layer}"
        );
    }
    let geometry = Geometry {
        dim: config.dim as u64,
        inter: config.moe_inter_dim as u64,
        layers: config.n_layers,
        experts: config.n_routed_experts as u64,
        mtp_layers: config.n_mtp_layers,
        mtp_experts: config.dspark_n_routed_experts as u64,
        table_width: config.engram_head_dim as u64,
        tables,
    };
    plan_native(
        &geometry,
        &inventory.tensors,
        inventory.total_payload_bytes,
        world,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertKey {
    mtp: bool,
    layer: usize,
    expert: u64,
    matrix: u8,
    scale: bool,
}

fn expert_key(name: &str) -> Result<Option<ExpertKey>> {
    if !name.contains(".ffn.experts.") {
        return Ok(None);
    }
    let p: Vec<_> = name.split('.').collect();
    ensure!(
        p.len() == 7 && matches!(p[0], "layers" | "mtp") && p[2] == "ffn" && p[3] == "experts",
        "unsupported expert key layout {name}"
    );
    let matrix = match p[5] {
        "w1" => 1,
        "w2" => 2,
        "w3" => 3,
        _ => bail!("unsupported expert matrix in {name}"),
    };
    let scale = match p[6] {
        "weight" => false,
        "scale" => true,
        _ => bail!("unsupported expert payload format in {name}"),
    };
    Ok(Some(ExpertKey {
        mtp: p[0] == "mtp",
        layer: p[1].parse().context("expert layer index")?,
        expert: p[4].parse().context("expert index")?,
        matrix,
        scale,
    }))
}

fn validate_tensor_bytes(t: &TensorMetadata) -> Result<()> {
    let item = match t.dtype.as_str() {
        "BF16" => 2_u64,
        "F32" => 4,
        "I8" | "F8_E4M3" | "F8_E8M0" => 1,
        _ => bail!(
            "unsupported native1017 tensor dtype {} for {}",
            t.dtype,
            t.name
        ),
    };
    let elements = t.shape.iter().try_fold(1_u64, |n, &d| {
        ensure!(d > 0, "zero tensor extent in {}", t.name);
        n.checked_mul(d).context("tensor shape overflow")
    })?;
    ensure!(
        elements.checked_mul(item) == Some(t.stored_bytes),
        "stored byte/shape mismatch for {}",
        t.name
    );
    ensure!(
        t.data_offsets[1].checked_sub(t.data_offsets[0]) == Some(t.stored_bytes),
        "stored data offsets mismatch for {}",
        t.name
    );
    Ok(())
}

fn validate_expert(t: &TensorMetadata, key: ExpertKey, g: &Geometry) -> Result<()> {
    let (layers, experts) = if key.mtp {
        (g.mtp_layers, g.mtp_experts)
    } else {
        (g.layers, g.experts)
    };
    ensure!(
        key.layer < layers && key.expert < experts,
        "expert identity outside declared geometry: {}",
        t.name
    );
    ensure!(
        g.dim % 32 == 0 && g.inter % 32 == 0,
        "FP4 expert dimensions must divide native group32"
    );
    let (rows, columns) = if key.matrix == 2 {
        (g.dim, g.inter)
    } else {
        (g.inter, g.dim)
    };
    let (dtype, shape) = if key.scale {
        ("F8_E8M0", vec![rows, columns / 32])
    } else {
        ("I8", vec![rows, columns / 2])
    };
    ensure!(
        t.dtype == dtype && t.shape == shape,
        "unsupported packed-FP4/per32 scale layout for {}: {} {:?}, expected {dtype} {shape:?}",
        t.name,
        t.dtype,
        t.shape
    );
    Ok(())
}

fn table_tensor<'a>(
    map: &'a BTreeMap<&str, &'a TensorMetadata>,
    layer: usize,
    suffix: &str,
) -> Result<&'a TensorMetadata> {
    let name = format!("layers.{layer}.engram.embed.{suffix}");
    map.get(name.as_str())
        .copied()
        .with_context(|| format!("missing declared Engram table tensor {name}"))
}

fn slice_offsets(t: &TensorMetadata, rows: IndexRange, row_bytes: u64) -> Result<[u64; 2]> {
    let start = rows
        .start
        .checked_mul(row_bytes)
        .and_then(|v| t.data_offsets[0].checked_add(v))
        .context("row offset overflow")?;
    let end = rows
        .end
        .checked_mul(row_bytes)
        .and_then(|v| t.data_offsets[0].checked_add(v))
        .context("row offset overflow")?;
    ensure!(
        start <= end && end <= t.data_offsets[1],
        "row slice outside tensor {}",
        t.name
    );
    Ok([start, end])
}

fn plan_native(
    g: &Geometry,
    tensors: &[TensorMetadata],
    source_total: u64,
    world: usize,
) -> Result<OwnershipPlan> {
    let ranges = expert_ranges(g.experts, world)?;
    let mut map = BTreeMap::new();
    let mut sum = 0_u64;
    for t in tensors {
        validate_tensor_bytes(t)?;
        ensure!(
            map.insert(t.name.as_str(), t).is_none(),
            "duplicate tensor {}",
            t.name
        );
        sum = sum
            .checked_add(t.stored_bytes)
            .context("source payload total overflow")?;
    }
    ensure!(
        sum == source_total,
        "inventory payload total {source_total} differs from tensor sum {sum}"
    );
    for name in ["embed.weight", "head.weight", "norm.weight"] {
        ensure!(map.contains_key(name), "missing trunk-top tensor {name}");
    }
    let mut table_names = BTreeSet::new();
    let mut tables = Vec::new();
    let mut table_bytes = TensorBytes::default();
    for (&layer, &rows) in &g.tables {
        ensure!(
            layer < g.layers && rows > 0 && g.table_width > 0 && g.table_width % 32 == 0,
            "invalid Engram table geometry"
        );
        let (w, s) = (
            table_tensor(&map, layer, "weight")?,
            table_tensor(&map, layer, "scale")?,
        );
        let scale_width = g.table_width / 32;
        ensure!(
            w.dtype == "F8_E4M3" && w.shape == vec![rows, g.table_width],
            "unsupported Engram row data layout {}",
            w.name
        );
        ensure!(
            s.dtype == "F8_E8M0" && s.shape == vec![rows, scale_width],
            "unsupported Engram per32 row scale layout {}",
            s.name
        );
        table_names.insert(w.name.as_str());
        table_names.insert(s.name.as_str());
        let mut native = TensorBytes::default();
        native.add(w.stored_bytes, false)?;
        native.add(s.stored_bytes, true)?;
        table_bytes = table_bytes.plus(native)?;
        let mut slices = Vec::new();
        let proposed_allocated_rows = rows / world as u64 + u64::from(rows % world as u64 != 0);
        for (rank, range) in row_ranges(rows, world)?.into_iter().enumerate() {
            let mut bytes = TensorBytes::default();
            bytes.add(
                range
                    .count
                    .checked_mul(g.table_width)
                    .context("Engram data bytes overflow")?,
                false,
            )?;
            bytes.add(
                range
                    .count
                    .checked_mul(scale_width)
                    .context("Engram scale bytes overflow")?,
                true,
            )?;
            let padding_rows = proposed_allocated_rows - range.count;
            let mut padding_bytes = TensorBytes::default();
            padding_bytes.add(
                padding_rows
                    .checked_mul(g.table_width)
                    .context("Engram data padding overflow")?,
                false,
            )?;
            padding_bytes.add(
                padding_rows
                    .checked_mul(scale_width)
                    .context("Engram scale padding overflow")?,
                true,
            )?;
            slices.push(EngramRankSlice {
                rank,
                rows: range,
                weight_data_offsets: slice_offsets(w, range, g.table_width)?,
                scale_data_offsets: slice_offsets(s, range, scale_width)?,
                bytes,
                proposed_allocated_rows,
                padding_rows,
                proposed_allocation_bytes: bytes.plus(padding_bytes)?,
                padding_bytes,
            });
        }
        tables.push(EngramTablePlan {
            layer,
            rows,
            weight_name: w.name.clone(),
            weight_shard: w.shard.clone(),
            scale_name: s.name.clone(),
            scale_shard: s.shard.clone(),
            data_bytes_per_row: g.table_width,
            scale_bytes_per_row: scale_width,
            native_bytes: native,
            row_shards: slices,
        });
    }
    let mut expert_seen = BTreeSet::new();
    let mut owned = vec![TensorBytes::default(); world];
    let mut owned_count = vec![0_usize; world];
    let mut experts = TensorBytes::default();
    let mut replicated = TensorBytes::default();
    let mut replicated_count = 0;
    let mut excluded: BTreeMap<String, TensorBytes> = BTreeMap::new();
    for t in tensors {
        let key = expert_key(&t.name)?;
        if let Some(key) = key {
            validate_expert(t, key, g)?;
        }
        // Every FP8 weight keeps its native block32 scale partner. The Engram
        // table is row-scaled instead and was validated above.
        if t.dtype == "F8_E4M3" && !table_names.contains(t.name.as_str()) {
            let base = t
                .name
                .strip_suffix(".weight")
                .with_context(|| format!("unsupported FP8 weight name {}", t.name))?;
            let scale = map
                .get(format!("{base}.scale").as_str())
                .with_context(|| format!("missing FP8 scale for {}", t.name))?;
            ensure!(
                t.shape.len() == 2
                    && scale.dtype == "F8_E8M0"
                    && scale.shape == vec![t.shape[0].div_ceil(32), t.shape[1].div_ceil(32)],
                "unsupported native block32 FP8 scale layout for {}",
                t.name
            );
        }
        if t.dtype == "F8_E8M0" && key.is_none() && !table_names.contains(t.name.as_str()) {
            let base = t
                .name
                .strip_suffix(".scale")
                .with_context(|| format!("unsupported FP8 scale name {}", t.name))?;
            let weight = map
                .get(format!("{base}.weight").as_str())
                .with_context(|| format!("orphan FP8 scale {}", t.name))?;
            ensure!(
                weight.dtype == "F8_E4M3",
                "FP8 scale {} has unsupported weight dtype",
                t.name
            );
        }
        ensure!(
            t.dtype != "I8" || key.is_some(),
            "I8 tensor {} is not a supported packed expert",
            t.name
        );
        if t.name.starts_with("vision.")
            || t.name.starts_with("aligner.")
            || matches!(
                t.name.as_str(),
                "image_start" | "image_end" | "image_newline"
            )
        {
            excluded
                .entry("vision_not_selected_by_plain_text_plan".to_string())
                .or_default()
                .add(t.stored_bytes, t.dtype == "F8_E8M0")?;
        } else if t.name.starts_with("mtp.") {
            excluded
                .entry("mtp_not_selected_by_plain_decode_plan".to_string())
                .or_default()
                .add(t.stored_bytes, t.dtype == "F8_E8M0")?;
        } else if let Some(key) = key {
            ensure!(
                !key.mtp && expert_seen.insert(key),
                "duplicate trunk expert tensor {}",
                t.name
            );
            let rank = expert_owner(g.experts, world, key.expert)?;
            owned[rank].add(t.stored_bytes, key.scale)?;
            owned_count[rank] += 1;
            experts.add(t.stored_bytes, key.scale)?;
        } else if table_names.contains(t.name.as_str()) {
            // Accounted in row ownership, never silently replicated here.
        } else {
            ensure!(
                !t.name.contains(".engram.embed."),
                "undeclared Engram table tensor {}",
                t.name
            );
            replicated.add(t.stored_bytes, t.dtype == "F8_E8M0")?;
            replicated_count += 1;
        }
    }
    let expected = (g.layers as u64)
        .checked_mul(g.experts)
        .and_then(|v| v.checked_mul(6))
        .context("expert tensor count overflow")?;
    ensure!(
        expert_seen.len() as u64 == expected,
        "incomplete trunk expert bank: {} tensors, expected {expected}",
        expert_seen.len()
    );
    // Each validated key lies within the declared layer/expert/matrix/kind
    // product. Uniqueness plus exact cardinality proves complete ownership.
    let selected = replicated.plus(experts)?.plus(table_bytes)?;
    let excluded_total = excluded.values().try_fold(0_u64, |n, b| {
        n.checked_add(b.total_bytes)
            .context("excluded total overflow")
    })?;
    ensure!(
        selected.total_bytes.checked_add(excluded_total) == Some(source_total),
        "source ownership accounting does not close"
    );
    let mut scenarios = Vec::new();
    for row_sharded in [true, false] {
        let mut ranks = Vec::new();
        let mut cluster = TensorBytes::default();
        let mut cluster_allocation = TensorBytes::default();
        for rank in 0..world {
            let engram = if row_sharded {
                tables.iter().try_fold(TensorBytes::default(), |a, t| {
                    a.plus(t.row_shards[rank].bytes)
                })?
            } else {
                table_bytes
            };
            let total = owned[rank].plus(replicated)?.plus(engram)?;
            let padding = if row_sharded {
                tables.iter().try_fold(TensorBytes::default(), |a, t| {
                    a.plus(t.row_shards[rank].padding_bytes)
                })?
            } else {
                TensorBytes::default()
            };
            let proposed_allocation = total.plus(padding)?;
            cluster = cluster.plus(total)?;
            cluster_allocation = cluster_allocation.plus(proposed_allocation)?;
            ranks.push(RankWeightPlan {
                rank,
                trunk_experts: ranges[rank],
                owned_expert_tensor_count: owned_count[rank],
                expert_bytes: owned[rank],
                replicated_trunk_bytes: replicated,
                engram_table_bytes: engram,
                native_stored_bytes: total,
                engram_padding_bytes: padding,
                proposed_storage_allocation_bytes: proposed_allocation,
            });
        }
        scenarios.push(StorageScenario {
            engram_placement: if row_sharded {
                "official_ceil_row_shards"
            } else {
                "replicated_tables"
            },
            ranks,
            cluster_native_stored_bytes: cluster,
            cluster_proposed_storage_allocation_bytes: cluster_allocation,
        });
    }
    Ok(OwnershipPlan {
        schema_version: 1, kind: "deepseek_1017_native_header_ownership_plan", world, serving_supported: false,
        source_tensor_count: tensors.len(), source_payload_bytes: source_total, selected_plain_text_bytes: selected,
        excluded_bytes: excluded, replicated_trunk_tensor_count: replicated_count, replicated_trunk_bytes: replicated,
        trunk_expert_bytes: experts, engram_tables: tables, scenarios,
        runtime_memory_budget: RuntimeMemoryBudget { activation_bytes: None, kv_and_index_cache_bytes: None,
            workspace_and_prefill_bytes: None, cuda_context_and_kernel_bytes: None, staging_and_conversion_peak_bytes: None,
            operating_system_and_page_cache_bytes: None, fits_available_memory: None },
        notes: vec!["Exact native stored tensor bytes only; no payload read, quantization, conversion or allocation performed.",
            "Scale bytes are separately accounted and are included in every total, not added twice.",
            "All nonexpert/non-table text tensors, including Engram projections and the full head, are proposed replicated.",
            "MTP and vision payloads remain part of source verification but are excluded explicitly from this plain text planning selection.",
            "Engram row data and scales share official ceil-row owner ranges; valid payload and proposed padding allocation bytes are separate, and offsets are relative to each shard data section.",
            "Resident representations, conversion expansion, load copies and activation/KV/workspace budgets remain unknown; no memory-fit claim.",
            "Header compatibility is not payload integrity or numerical qualification; serving remains unsupported."],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_bands_cover_256_and_384_without_aliases() {
        for experts in [256, 384] {
            let ranges = expert_ranges(experts, 8).unwrap();
            let mut seen = BTreeSet::new();
            for (rank, r) in ranges.iter().enumerate() {
                assert_eq!(r.count, experts / 8);
                for e in r.start..r.end {
                    assert!(seen.insert(e));
                    assert_eq!(expert_owner(experts, 8, e).unwrap(), rank);
                }
            }
            assert_eq!(seen.len() as u64, experts);
        }
        assert!(expert_ranges(385, 8).is_err());
        assert!(expert_owner(384, 8, 384).is_err());
    }

    #[test]
    fn row_remainders_and_small_empty_shards_preserve_every_row() {
        for rows in [0, 1, 7, 8, 9, 17, 384_016_682] {
            let ranges = row_ranges(rows, 8).unwrap();
            let mut end = 0;
            for (rank, r) in ranges.iter().enumerate() {
                assert_eq!(r.start, end);
                assert_eq!(r.end - r.start, r.count);
                if r.count > 0 {
                    assert_eq!(row_owner(rows, 8, r.start).unwrap(), rank);
                    assert_eq!(row_owner(rows, 8, r.end - 1).unwrap(), rank);
                }
                end = r.end;
            }
            assert_eq!(end, rows);
            assert!(row_owner(rows, 8, rows).is_err());
            if rows < 100 {
                for row in 0..rows {
                    let owner = row_owner(rows, 8, row).unwrap();
                    assert!(ranges[owner].start <= row && row < ranges[owner].end);
                }
            }
        }
        let r = row_ranges(384_016_682, 8).unwrap();
        assert_eq!(r[0].count, 48_002_086);
        assert_eq!(r[1].count, 48_002_086);
        assert_eq!(r[2].count, 48_002_086);
        assert_eq!(r[7].count, 48_002_080);
        assert_eq!(row_owner(u64::MAX, 1, u64::MAX - 1).unwrap(), 0);
        assert_eq!(row_ranges(u64::MAX, 1).unwrap()[0].end, u64::MAX);
    }

    fn fixture() -> (Geometry, Vec<TensorMetadata>, u64) {
        let g = Geometry {
            dim: 32,
            inter: 32,
            layers: 1,
            experts: 384,
            mtp_layers: 0,
            mtp_experts: 128,
            table_width: 256,
            tables: BTreeMap::from([(0, 17)]),
        };
        let mut tensors = Vec::new();
        let mut offset = 0_u64;
        let mut push = |name: String, dtype: &str, shape: Vec<u64>| {
            let item = if dtype == "BF16" { 2 } else { 1 };
            let bytes = shape.iter().product::<u64>() * item;
            tensors.push(TensorMetadata {
                name,
                shard: "fixture.safetensors".to_string(),
                dtype: dtype.to_string(),
                shape,
                data_offsets: [offset, offset + bytes],
                stored_bytes: bytes,
            });
            offset += bytes;
        };
        for n in ["embed.weight", "head.weight", "norm.weight"] {
            push(n.to_string(), "BF16", vec![1]);
        }
        for e in 0..384 {
            for m in 1..=3 {
                push(
                    format!("layers.0.ffn.experts.{e}.w{m}.weight"),
                    "I8",
                    vec![32, 16],
                );
                push(
                    format!("layers.0.ffn.experts.{e}.w{m}.scale"),
                    "F8_E8M0",
                    vec![32, 1],
                );
            }
        }
        push(
            "layers.0.engram.embed.weight".to_string(),
            "F8_E4M3",
            vec![17, 256],
        );
        push(
            "layers.0.engram.embed.scale".to_string(),
            "F8_E8M0",
            vec![17, 8],
        );
        push("vision.fixture.weight".to_string(), "BF16", vec![2]);
        (g, tensors, offset)
    }

    #[test]
    fn actual_plan_closes_bytes_and_paired_engram_offsets() {
        let (g, tensors, total) = fixture();
        let p = plan_native(&g, &tensors, total, 8).unwrap();
        assert!(!p.serving_supported);
        assert_eq!(p.runtime_memory_budget.fits_available_memory, None);
        assert_eq!(
            p.selected_plain_text_bytes.total_bytes
                + p.excluded_bytes
                    .values()
                    .map(|x| x.total_bytes)
                    .sum::<u64>(),
            total
        );
        let table = &p.engram_tables[0];
        assert!(table
            .row_shards
            .iter()
            .all(|s| s.proposed_allocated_rows == 3));
        assert_eq!(
            table.row_shards.iter().map(|s| s.rows.count).sum::<u64>(),
            17
        );
        assert_eq!(
            table.row_shards.iter().map(|s| s.padding_rows).sum::<u64>(),
            7
        );
        let mut w = table.row_shards[0].weight_data_offsets[0];
        let mut s = table.row_shards[0].scale_data_offsets[0];
        for slice in &table.row_shards {
            assert_eq!(slice.weight_data_offsets[0], w);
            assert_eq!(slice.scale_data_offsets[0], s);
            assert_eq!(slice.weight_data_offsets[1] - w, slice.rows.count * 256);
            assert_eq!(slice.scale_data_offsets[1] - s, slice.rows.count * 8);
            w = slice.weight_data_offsets[1];
            s = slice.scale_data_offsets[1];
        }
        assert_eq!(w - table.row_shards[0].weight_data_offsets[0], 17 * 256);
        assert_eq!(s - table.row_shards[0].scale_data_offsets[0], 17 * 8);
        for rank in &p.scenarios[0].ranks {
            assert_eq!(
                rank.expert_bytes,
                TensorBytes {
                    data_bytes: 48 * 3 * 512,
                    scale_bytes: 48 * 3 * 32,
                    total_bytes: 48 * 3 * 544
                }
            );
            assert_eq!(
                rank.native_stored_bytes,
                rank.expert_bytes
                    .plus(p.replicated_trunk_bytes)
                    .unwrap()
                    .plus(table.row_shards[rank.rank].bytes)
                    .unwrap()
            );
        }
        for rank in &p.scenarios[1].ranks {
            assert_eq!(rank.engram_table_bytes.total_bytes, 17 * 264);
            assert_eq!(rank.engram_padding_bytes.total_bytes, 0);
        }
        assert_eq!(
            p.scenarios[0]
                .cluster_proposed_storage_allocation_bytes
                .total_bytes
                - p.scenarios[0].cluster_native_stored_bytes.total_bytes,
            7 * 264
        );
        assert_eq!(
            p.scenarios[1].cluster_proposed_storage_allocation_bytes,
            p.scenarios[1].cluster_native_stored_bytes
        );
    }

    #[test]
    fn unsupported_missing_and_misaligned_layouts_fail_closed() {
        let (g, tensors, total) = fixture();
        let mut bad = tensors.clone();
        bad[0].dtype = "U16".to_string();
        assert!(plan_native(&g, &bad, total, 8).is_err());
        let mut bad = tensors.clone();
        let missing = bad.remove(6);
        assert!(plan_native(&g, &bad, total - missing.stored_bytes, 8).is_err());
        let mut bad = tensors.clone();
        bad[4].shape = vec![16, 2];
        assert!(plan_native(&g, &bad, total, 8).is_err());
        let mut bad = tensors.clone();
        let scale = bad
            .iter_mut()
            .find(|t| t.name.ends_with("engram.embed.scale"))
            .unwrap();
        scale.shape = vec![8, 17];
        assert!(plan_native(&g, &bad, total, 8).is_err());
        let mut bad = tensors.clone();
        bad.push(tensors[0].clone());
        assert!(plan_native(&g, &bad, total + tensors[0].stored_bytes, 8).is_err());
        assert!(plan_native(&g, &tensors, total + 1, 8).is_err());
    }
}
