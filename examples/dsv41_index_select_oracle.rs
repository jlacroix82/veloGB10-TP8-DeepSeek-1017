//! Model-free source BF16 candidate/top512 oracle. Stable ascending-index ties
//! are explicit admissible witnesses, never a claim about Torch tie stability.
use anyhow::{bail, ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use gb10_inference::dsv41::{
    attention_state::{candidate_blocks, topk_positions, Selection, MODEL_REFERENCE_SHA256},
    gpu_index_select::{validate_shape, IndexSelection, CUDA_SOURCE},
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

const WALL: u64 = 120;
const ARRAY_BOUND: usize = 64 * 1024 * 1024;
const WIDTHS: [usize; 6] = [1, 2, 4, 8, 16, 128];
const FAMILY: &str = "deterministic ascending-index admissible top-k witness";
#[derive(Clone, Copy)]
enum Pattern {
    Zero,
    Increasing,
    Negative,
    SignedZero,
    Subnormals,
    Extremes,
    LatestLow,
    Mixed,
}
struct Case {
    name: String,
    rows: usize,
    columns: usize,
    offset: usize,
    pattern: Pattern,
    visible: Vec<usize>,
    lanes: Vec<usize>,
    mask: bool,
    max_closed_form: bool,
    family: Option<&'static str>,
}
fn record(checks: &mut Vec<Value>, name: impl Into<String>, passed: bool, details: Value) {
    checks.push(json!({"name":name.into(),"passed":passed,"details":details}));
}
fn hash_bytes(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}
fn hash_bf16(v: &[bf16]) -> String {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn hash_i32(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn mismatch<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    usize::from(a.len() != b.len()) + a.iter().zip(b).filter(|(a, b)| a != b).count()
}
fn exact<T>(v: Selection<T>) -> Result<T> {
    match v {
        Selection::Exact(x) => Ok(x),
        Selection::Ambiguous(_) => bail!("explicit stable witness remained ambiguous"),
    }
}
fn order(values: &[f32], count: usize) -> Vec<usize> {
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
    ids.truncate(count.min(values.len()));
    ids
}
fn value(pattern: Pattern, lane: usize, p: usize, n: usize) -> bf16 {
    match pattern {
        Pattern::Zero => bf16::ZERO,
        Pattern::Increasing => bf16::from_f32((p % 257) as f32 - 128.),
        Pattern::Negative => bf16::from_f32(-1. - ((p * 31 + lane * 17) % 1024) as f32 / 16.),
        Pattern::SignedZero => bf16::from_bits(if (p + lane) % 2 == 0 { 0x8000 } else { 0 }),
        Pattern::Subnormals => bf16::from_bits([1, 0x8001, 0, 0x8000][(p / 8 + lane) % 4]),
        Pattern::Extremes => {
            bf16::from_bits([0xff7f, 0x8001, 0x8000, 0, 1, 0x7f7f][(p + lane) % 6])
        }
        Pattern::LatestLow => bf16::from_f32(if p + 1 == n {
            -1000.
        } else {
            (p / 8 % 257) as f32
        }),
        Pattern::Mixed => bf16::from_bits(match (p + lane * 7) % 13 {
            0 => 0x0001,
            1 => 0x8001,
            2 => 0x8000,
            3 => 0,
            _ => bf16::from_f32(((p * 13 + lane * 97 + p / 8) % 193) as f32 / 8. - 16.).to_bits(),
        }),
    }
}
fn row_reference(
    raw: &[f32],
    visible: usize,
    offset: usize,
    use_mask: bool,
) -> Result<(Vec<u8>, Vec<i32>, usize)> {
    let n = raw.len();
    // This reference sorts numeric scores and native positions; it does not
    // reuse the GPU integer-key/radix implementation.
    let mut maxima = vec![f32::NEG_INFINITY; n.div_ceil(8)];
    for p in 0..visible {
        maxima[p / 8] = maxima[p / 8].max(raw[p]);
    }
    if visible > 0 {
        maxima[(visible - 1) / 8] = f32::INFINITY;
    }
    let block_choice = order(&maxima, 2048);
    let scores = vec![raw.to_vec()];
    let mask = exact(candidate_blocks(
        &scores,
        &[visible],
        2048,
        8,
        Some(&[block_choice]),
    )?)?
    .remove(0);
    let masked = raw
        .iter()
        .enumerate()
        .map(|(p, &v)| {
            if p < visible && (!use_mask || mask[p]) {
                v
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect::<Vec<_>>();
    let choice = order(&masked, 512);
    let indices = exact(topk_positions(
        &scores,
        &[visible],
        512,
        offset,
        use_mask.then_some(std::slice::from_ref(&mask)),
        Some(&[choice]),
    )?)?
    .remove(0);
    let observable_ties = usize::from(matches!(
        candidate_blocks(&scores, &[visible], 2048, 8, None)?,
        Selection::Ambiguous(_)
    )) + usize::from(matches!(
        topk_positions(
            &scores,
            &[visible],
            512,
            offset,
            use_mask.then_some(std::slice::from_ref(&mask)),
            None
        )?,
        Selection::Ambiguous(_)
    ));
    Ok((
        mask.into_iter().map(u8::from).collect(),
        indices,
        observable_ties,
    ))
}
fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut add = |name: &str, n, visible, pattern| {
        v.push(Case {
            name: name.into(),
            rows: 1,
            columns: n,
            offset: 128,
            pattern,
            visible: vec![visible],
            lanes: vec![0],
            mask: true,
            max_closed_form: false,
            family: None,
        })
    };
    add("n1_reachable", 1, 1, Pattern::Zero);
    add("n1_zero_visible", 1, 0, Pattern::Zero);
    add("n511_increasing", 511, 511, Pattern::Increasing);
    add("n512_negative", 512, 512, Pattern::Negative);
    add("n513_signed_zero", 513, 513, Pattern::SignedZero);
    add("n513_bf16_extremes", 513, 513, Pattern::Extremes);
    add(
        "n1025_subnormal_position_cutoff",
        1025,
        1025,
        Pattern::Subnormals,
    );
    add(
        "n16385_latest_block_forced",
        16385,
        16385,
        Pattern::LatestLow,
    );
    add("n16385_causal_partial_block", 16385, 16377, Pattern::Mixed);
    add("n16385_zero_visible", 16385, 0, Pattern::Zero);
    for rows in WIDTHS {
        let lanes = (0..rows).map(|i| (i * 37 + 11) % 128).collect::<Vec<_>>();
        let visible = lanes
            .iter()
            .map(|&i| [0, 1, 7, 8, 9, 511, 512, 513][i % 8])
            .collect();
        v.push(Case {
            name: format!("permuted_short_width{rows}"),
            rows,
            columns: 513,
            offset: 17,
            pattern: Pattern::Mixed,
            visible,
            lanes,
            mask: true,
            max_closed_form: false,
            family: Some("short"),
        });
    }
    for rows in [1, 2, 4, 8, 16] {
        let lanes = (0..rows).map(|i| (i * 7 + 3) % 16).collect::<Vec<_>>();
        let visible = lanes
            .iter()
            .map(|&i| [16385, 16377, 12001, 512][i % 4])
            .collect();
        v.push(Case {
            name: format!("permuted_long_width{rows}"),
            rows,
            columns: 16385,
            offset: 128,
            pattern: Pattern::Mixed,
            visible,
            lanes,
            mask: true,
            max_closed_form: false,
            family: Some("long"),
        });
    }
    v.push(Case {
        name: "n131072_bf16_extremes".into(),
        rows: 1,
        columns: 131072,
        offset: 0,
        pattern: Pattern::Extremes,
        visible: vec![131072],
        lanes: vec![0],
        mask: true,
        max_closed_form: false,
        family: None,
    });
    add_subnormal_block_case(&mut v);
    let lanes = (0..16).map(|i| (i * 7 + 3) % 16).collect::<Vec<_>>();
    let visible = lanes.iter().map(|&i| 131072 - i * 7).collect();
    v.push(Case {
        name: "n131072_permuted_width16".into(),
        rows: 16,
        columns: 131072,
        offset: 128,
        pattern: Pattern::Mixed,
        visible,
        lanes,
        mask: true,
        max_closed_form: false,
        family: None,
    });
    v.push(Case {
        name: "maximum_matrix128x131072_all_zero".into(),
        rows: 128,
        columns: 131072,
        offset: 128,
        pattern: Pattern::Zero,
        visible: vec![131072; 128],
        lanes: (0..128).collect(),
        mask: true,
        max_closed_form: true,
        family: None,
    });
    v.push(Case {
        name: "n16385_without_candidate_mask".into(),
        rows: 4,
        columns: 16385,
        offset: 0,
        pattern: Pattern::Mixed,
        visible: vec![0, 511, 16377, 16385],
        lanes: vec![7, 2, 13, 0],
        mask: false,
        max_closed_form: false,
        family: None,
    });
    v
}
fn add_subnormal_block_case(v: &mut Vec<Case>) {
    v.push(Case {
        name: "n131072_subnormal_block_cutoff".into(),
        rows: 1,
        columns: 131072,
        offset: 0,
        pattern: Pattern::Subnormals,
        visible: vec![131072],
        lanes: vec![0],
        mask: true,
        max_closed_form: false,
        family: None,
    });
}
fn positive(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &IndexSelection,
    c: &Case,
    checks: &mut Vec<Value>,
    prior: &mut BTreeMap<(String, usize), (String, String)>,
    peak: &mut usize,
) -> Result<()> {
    let elements = c.rows * c.columns;
    let out_elements = c.rows * c.columns.min(512);
    let allocated = elements * 3 + out_elements * 4 + c.rows * 4 + 4;
    ensure!(allocated <= ARRAY_BOUND, "explicit GPU array bound");
    *peak = (*peak).max(allocated);
    let data = c
        .lanes
        .iter()
        .flat_map(|&lane| (0..c.columns).map(move |p| value(c.pattern, lane, p, c.columns)))
        .collect::<Vec<_>>();
    let mut expected_mask = Vec::with_capacity(elements);
    let mut expected_indices = Vec::with_capacity(out_elements);
    let mut ambiguous = 0;
    if c.max_closed_form {
        // Independent closed form:2047 lowest complete blocks plus forced last
        // block. Stable score tie selects positions0..511 after source sorting.
        let one_mask = (0..c.columns)
            .map(|p| u8::from(p < 2047 * 8 || p / 8 == (c.columns - 1) / 8))
            .collect::<Vec<_>>();
        let one_indices = (0..512).map(|p| (p + c.offset) as i32).collect::<Vec<_>>();
        let raw = vec![0.; c.columns];
        let (cpu_m, cpu_i, ties) = row_reference(&raw, c.columns, c.offset, true)?;
        ensure!(
            cpu_m == one_mask && cpu_i == one_indices,
            "maximum closed form disagrees with independent CPU witnesses"
        );
        ambiguous = ties * c.rows;
        for _ in 0..c.rows {
            expected_mask.extend_from_slice(&one_mask);
            expected_indices.extend_from_slice(&one_indices);
        }
    } else {
        for row in 0..c.rows {
            let raw = data[row * c.columns..(row + 1) * c.columns]
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>();
            let (m, i, t) = row_reference(&raw, c.visible[row], c.offset, c.mask)?;
            expected_mask.extend(m);
            expected_indices.extend(i);
            ambiguous += t;
        }
    }
    let d = dev.htod_sync_copy(&data)?;
    let visible = dev.htod_sync_copy(&c.visible.iter().map(|&x| x as u32).collect::<Vec<_>>())?;
    let mut mask = dev.alloc_zeros::<u8>(elements)?;
    let mut indices = dev.alloc_zeros::<i32>(out_elements)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    let actual_mask = g
        .candidates(s, &d, &visible, &mut mask, &mut status, c.rows, c.columns)?
        .download_checked()?;
    let actual_indices = g
        .positions(
            s,
            &d,
            &visible,
            c.mask.then_some(&mask),
            &mut indices,
            &mut status,
            c.rows,
            c.columns,
            c.offset,
        )?
        .download_checked()?;
    let status_bits = dev.dtoh_sync_copy(&status)?[0];
    let mask_bad = mismatch(&actual_mask, &expected_mask);
    let index_bad = mismatch(&actual_indices, &expected_indices);
    let mut cross_checked = 0;
    let mut cross_bad = 0;
    if let Some(family) = c.family {
        for (row, &lane) in c.lanes.iter().enumerate() {
            let m = hash_bytes(&actual_mask[row * c.columns..(row + 1) * c.columns]);
            let k = c.columns.min(512);
            let i = hash_i32(&actual_indices[row * k..(row + 1) * k]);
            if let Some(old) = prior.insert((family.into(), lane), (m.clone(), i.clone())) {
                cross_checked += 1;
                cross_bad += usize::from(old != (m, i));
            }
        }
    }
    record(
        checks,
        &c.name,
        mask_bad + index_bad + cross_bad == 0 && status_bits == 0,
        json!({
            "queries":c.rows,"columns":c.columns,"offset":c.offset,"logical_lanes":c.lanes,"visible":c.visible,"candidate_mask_used":c.mask,
            "input_bf16_le_sha256":hash_bf16(&data),"candidate_mask_sha256":hash_bytes(&actual_mask),"expected_candidate_mask_sha256":hash_bytes(&expected_mask),
            "indices_i32_le_sha256":hash_i32(&actual_indices),"expected_indices_i32_le_sha256":hash_i32(&expected_indices),
            "full_mask_elements":elements,"full_index_elements":out_elements,"mask_mismatches":mask_bad,"index_mismatches":index_bad,"status_bits":status_bits,
            "stable_low_index_witness_validated":true,"observable_ambiguous_boundaries":ambiguous,"Torch_stable_tie_order_assumed":false,
            "cross_width_prior_rows_checked":cross_checked,"cross_width_mismatches":cross_bad,"explicit_gpu_array_bytes":allocated,
            "reference":if c.max_closed_form{"full closed-form output plus independent full-length CPU witness"}else{"attention_state CPU score-sort refs with explicit witnesses"},
            "candidate_blocks":2048,"block_size":8,"topk":512,"tolerance":"exact mask and native integer indices"
        }),
    );
    Ok(())
}
fn controls(
    dev: &Arc<CudaDevice>,
    s: &CudaStream,
    g: &IndexSelection,
    checks: &mut Vec<Value>,
) -> Result<()> {
    for (name, bits, vis, candidate_expected, positions_expected) in [
        // A visible invalid value also leaves too few eligible top-k keys:
        // both numeric(1) and inconsistent-count(4) must remain observable.
        ("visible_nan", 0x7fc1, 8, 1, 5),
        ("positive_inf", 0x7f80, 8, 1, 5),
        ("negative_inf", 0xff80, 8, 1, 5),
        ("future_nan", 0x7fc1, 0, 1, 1),
        ("visible_out_of_bounds", 0, 9, 2, 2),
    ] {
        let mut host = vec![bf16::ZERO; 8];
        host[7] = bf16::from_bits(bits);
        let data = dev.htod_sync_copy(&host)?;
        let visible = dev.htod_sync_copy(&[vis as u32])?;
        let mut mask = dev.alloc_zeros::<u8>(8)?;
        let mut output = dev.alloc_zeros::<i32>(8)?;
        let mut status = dev.alloc_zeros::<u32>(1)?;
        let candidate_rejected = g
            .candidates(s, &data, &visible, &mut mask, &mut status, 1, 8)?
            .check()
            .is_err();
        let candidate_status = dev.dtoh_sync_copy(&status)?[0];
        dev.memset_zeros(&mut status)?;
        let positions_rejected = g
            .positions(s, &data, &visible, None, &mut output, &mut status, 1, 8, 0)?
            .check()
            .is_err();
        let positions_status = dev.dtoh_sync_copy(&status)?[0];
        record(
            checks,
            format!("negative_{name}"),
            candidate_rejected
                && positions_rejected
                && candidate_status == candidate_expected
                && positions_status == positions_expected,
            json!({"candidate_rejected":candidate_rejected,"positions_rejected":positions_rejected,"candidate_status_bits":candidate_status,"positions_status_bits":positions_status,"expected_candidate_status_bits":candidate_expected,"expected_positions_status_bits":positions_expected}),
        );
    }
    let data = dev.htod_sync_copy(&vec![bf16::ZERO; 513])?;
    let visible = dev.htod_sync_copy(&[513u32])?;
    let false_mask = dev.alloc_zeros::<u8>(513)?;
    let mut output = dev.alloc_zeros::<i32>(512)?;
    let mut status = dev.alloc_zeros::<u32>(1)?;
    let rejected = g
        .positions(
            s,
            &data,
            &visible,
            Some(&false_mask),
            &mut output,
            &mut status,
            1,
            513,
            0,
        )?
        .check()
        .is_err();
    let bits = dev.dtoh_sync_copy(&status)?[0];
    record(
        checks,
        "negative_inconsistent_candidate_count",
        rejected && bits == 4,
        json!({"rejected":rejected,"status_bits":bits,"expected_status_bits":4}),
    );
    let mut sticky = dev.htod_sync_copy(&[8u32])?;
    let mut mask = dev.alloc_zeros::<u8>(513)?;
    let candidate_rejected = g
        .candidates(s, &data, &visible, &mut mask, &mut sticky, 1, 513)?
        .check()
        .is_err();
    let positions_rejected = g
        .positions(
            s,
            &data,
            &visible,
            None,
            &mut output,
            &mut sticky,
            1,
            513,
            0,
        )?
        .check()
        .is_err();
    let bits = dev.dtoh_sync_copy(&sticky)?[0];
    record(
        checks,
        "negative_sticky_failure",
        candidate_rejected && positions_rejected && bits == 8,
        json!({"candidate_rejected":candidate_rejected,"positions_rejected":positions_rejected,"status_bits":bits,"expected_status_bits":8}),
    );
    let other = gb10_inference::gpu::fork_blocking_stream(dev);
    dev.memset_zeros(&mut status)?;
    let candidate_rejected = g
        .candidates(&other, &data, &visible, &mut mask, &mut status, 1, 513)
        .is_err();
    let positions_rejected = g
        .positions(
            &other,
            &data,
            &visible,
            None,
            &mut output,
            &mut status,
            1,
            513,
            0,
        )
        .is_err();
    record(
        checks,
        "negative_wrong_stream",
        candidate_rejected && positions_rejected,
        json!({"candidate_rejected_before_launch":candidate_rejected,"positions_rejected_before_launch":positions_rejected}),
    );
    for (name, q, n, o) in [
        ("queries0", 0, 1, 0),
        ("queries129", 129, 1, 0),
        ("columns0", 1, 0, 0),
        ("columns131073", 1, 131073, 0),
        ("offset_overflow", 1, 1, usize::MAX),
    ] {
        let rejected = validate_shape(q, n, o).is_err();
        record(
            checks,
            format!("negative_{name}"),
            rejected,
            json!({"rejected_before_allocation":rejected}),
        );
    }
    let mut bad_output = dev.alloc_zeros::<i32>(511)?;
    let rejected = g
        .positions(
            s,
            &data,
            &visible,
            None,
            &mut bad_output,
            &mut status,
            1,
            513,
            0,
        )
        .is_err();
    record(
        checks,
        "negative_output_extent",
        rejected,
        json!({"rejected_before_launch":rejected}),
    );
    let mut bad_mask = dev.alloc_zeros::<u8>(512)?;
    let candidate_rejected = g
        .candidates(s, &data, &visible, &mut bad_mask, &mut status, 1, 513)
        .is_err();
    let positions_rejected = g
        .positions(
            s,
            &data,
            &visible,
            Some(&bad_mask),
            &mut output,
            &mut status,
            1,
            513,
            0,
        )
        .is_err();
    record(
        checks,
        "negative_mask_extent",
        candidate_rejected && positions_rejected,
        json!({"candidate_rejected_before_launch":candidate_rejected,"positions_rejected_before_launch":positions_rejected}),
    );
    Ok(())
}
fn run() -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "index-selection oracle accepts zero arguments"
    );
    let began = Instant::now();
    let dev = CudaDevice::new(0)?;
    let s = gb10_inference::gpu::fork_blocking_stream(&dev);
    let g = IndexSelection::load(&dev, &s)?;
    let mut checks = Vec::new();
    let mut prior = BTreeMap::new();
    let mut peak = 0;
    let corpus = cases();
    ensure!(corpus.len() == 26, "positive case count changed");
    for case in &corpus {
        positive(&dev, &s, &g, case, &mut checks, &mut prior, &mut peak)?;
    }
    controls(&dev, &s, &g, &mut checks)?;
    ensure!(
        checks.len() == 41,
        "index-selection check count changed:{}",
        checks.len()
    );
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "index-selection wall deadline"
    );
    let passed = checks.iter().all(|v| v["passed"] == true);
    Ok(
        json!({"schema_version":1,"kind":"dsv41_index_select_component_oracle","status":if passed{"PASS"}else{"FAIL"},"all_passed":passed,
        "source_id_hex":g.source_id_hex(),"ptx_sha256":g.ptx_sha256(),"cuda_source_sha256":hash_bytes(CUDA_SOURCE),"official_model_reference_sha256":MODEL_REFERENCE_SHA256,
        "model_free":true,"model_execution":false,"checkpoint_payload_read":false,"RDMA":false,"physical_cluster_execution":false,
        "elapsed_seconds":began.elapsed().as_secs_f64(),"limits":{"wall_seconds":WALL,"explicit_gpu_array_bound_bytes":ARRAY_BOUND,"observed_peak_explicit_gpu_array_bytes":peak},
        "positive_case_count":26,"negative_control_count":15,"checks":checks,"tie_policy":FAMILY,
        "scope":"Source BF16 finite-score candidate2048×8 and position-sorted top512 primitive only. Full masks/indices compared exactly. No score projection, index collective, cache ownership, long-context model quality, throughput or serving qualification."}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("index-selection component hard deadline");
        std::process::exit(124);
    });
    let out = match std::panic::catch_unwind(run) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_index_select_component_oracle","status":"FAIL","all_passed":false,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_index_select_component_oracle","status":"FAIL","all_passed":false,"error":"panic"})
        }
    };
    println!("{out}");
    if out["all_passed"] != true {
        std::process::exit(1);
    }
}
