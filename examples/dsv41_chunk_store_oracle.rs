//! PRIVATE model-free raw-kernel cache oracle. Synthetic post-QAT BF16 payloads
//! and source selections do not forge learned IndexOutput or qualify its wrapper.
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr};
use gb10_inference::dsv41::{
    chunk_plan::{compressed_chunk, window_chunk, MAX_CONTEXT},
    gpu_chunk_store::{PTX, SOURCE_ID},
    gpu_module::Module,
};
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};
const D: usize = 512;
const W: usize = 128;
const L: usize = 40;
const S: usize = 4;
const LANES: usize = 2; // Raw-stride qualification only. Production wrapper is C1.
const WE: usize = LANES * L * W * D;
const CE: usize = LANES * S * MAX_CONTEXT * D;
const GUARD: usize = 16;
const BGUARD: u16 = 0x3eab;
const IGUARD: i32 = -123456789;
const SGUARD: u32 = 0x2ad30b17;
const WALL: u64 = 180;
const DEVICE_BOUND: usize = 128 * 1024 * 1024;
const HOST_BOUND: usize = 256 * 1024 * 1024;
const ACTIVE: [usize; 6] = [0, 2, 8, 14, 20, 39];
const OWNERS: [usize; 4] = [2, 8, 14, 20];
const SOURCE_CUDA_SHA: &str = "77807b96076d01a35effda010618ae26f94753c8510d2c9341cf759e4c57886d";
struct Queued<T> {
    dev: Arc<CudaDevice>,
    value: T,
}
impl<T> Queued<T> {
    fn new(dev: &Arc<CudaDevice>, value: T) -> Self {
        Self {
            dev: dev.clone(),
            value,
        }
    }
}
impl<T> std::ops::Deref for Queued<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl<T> std::ops::DerefMut for Queued<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
impl<T> Drop for Queued<T> {
    fn drop(&mut self) {
        let _ = self.dev.synchronize();
    }
}
fn ptr<T>(x: &CudaSlice<T>) -> u64 {
    *x.device_ptr()
}
fn sha_b(x: &[bf16]) -> String {
    let mut h = Sha256::new();
    for v in x {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn sha_i(x: &[i32]) -> String {
    let mut h = Sha256::new();
    for v in x {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn diff_b(a: &[bf16], b: &[bf16]) -> usize {
    usize::from(a.len() != b.len())
        + a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
}
fn diff_i(a: &[i32], b: &[i32]) -> usize {
    usize::from(a.len() != b.len()) + a.iter().zip(b).filter(|(a, b)| a != b).count()
}
fn guard_b(n: usize) -> Vec<bf16> {
    let mut v = vec![bf16::ZERO; n + GUARD];
    v[n..].fill(bf16::from_bits(BGUARD));
    v
}
fn guard_i(n: usize) -> Vec<i32> {
    vec![IGUARD; n + GUARD]
}
fn record(
    checks: &mut Vec<Value>,
    name: impl Into<String>,
    passed: bool,
    details: Value,
) -> Result<()> {
    let c = json!({"name":name.into(),"passed":passed,"details":details});
    println!("{}", json!({"event":"chunk_store_check","check":c}));
    checks.push(c);
    ensure!(
        passed,
        "raw chunk-cache check failed: {}",
        checks.last().unwrap()["name"]
    );
    Ok(())
}
fn data(lane: usize, layer: usize, row: usize, col: usize, compressed: bool) -> bf16 {
    // Raw BF16 bits preserve high native positions; no f32->BF16 narrowing can
    // wash out row/layer/lane fingerprints. Explicit finite signed-zero/tiny/max.
    let sign = ((row + layer + lane) & 1) as u16 * 0x8000;
    let bits = match col {
        0 => sign,
        1 => sign | 1,
        2 => sign | 0x7f7f,
        _ => {
            0x2000
                + ((row * 37 + col * 3 + layer * 101 + lane * 701 + usize::from(compressed) * 503)
                    % 0x3fff) as u16
        }
    };
    bf16::from_bits(bits)
}
fn payload(lane: usize, layer: usize, first: usize, rows: usize, compressed: bool) -> Vec<bf16> {
    (0..rows * D)
        .map(|i| data(lane, layer, first + i / D, i % D, compressed))
        .collect()
}
fn owner(layer: usize) -> Option<usize> {
    match layer {
        0 => None,
        2 => Some(0),
        8 => Some(1),
        14 => Some(2),
        20 | 39 => Some(3),
        _ => unreachable!(),
    }
}
fn ratio(layer: usize) -> usize {
    if layer == 0 {
        0
    } else if layer < 20 {
        2
    } else {
        1
    }
}
struct Raw {
    dev: Arc<CudaDevice>,
    stream: CudaStream,
    module: Module,
    windows: Queued<CudaSlice<bf16>>,
    compressed: Queued<CudaSlice<bf16>>,
    status: Queued<CudaSlice<u32>>,
}
impl Raw {
    fn new(dev: Arc<CudaDevice>) -> Result<Self> {
        let stream = gb10_inference::gpu::fork_blocking_stream(&dev);
        let module = Module::load(
            &dev,
            &stream,
            PTX,
            &[
                "dsv41_chunk_store_reset",
                "dsv41_chunk_store_window_snapshot",
                "dsv41_chunk_store_window_write",
                "dsv41_chunk_store_compressed_append",
                "dsv41_chunk_store_combined_snapshot",
                "dsv41_chunk_store_indices",
            ],
            "dsv41_chunk_store_identity",
            u64::from_str_radix(SOURCE_ID.trim_start_matches("0x"), 16)?,
            1,
        )?;
        let windows = Queued::new(&dev, dev.htod_sync_copy(&guard_b(WE))?);
        let compressed = Queued::new(&dev, dev.htod_sync_copy(&guard_b(CE))?);
        let mut status_data = vec![SGUARD; GUARD + 1];
        status_data[0] = 0;
        let status = Queued::new(&dev, dev.htod_sync_copy(&status_data)?);
        Ok(Self {
            dev,
            stream,
            module,
            windows,
            compressed,
            status,
        })
    }
    fn launch(&self, name: &str, n: usize, values: &[u64]) -> Result<()> {
        ensure!(n > 0 && n <= S * MAX_CONTEXT * D, "raw launch extent");
        let mut v = values.to_vec();
        let mut args = v
            .iter_mut()
            .map(|x| x as *mut u64 as *mut c_void)
            .collect::<Vec<_>>();
        unsafe {
            self.module.launch(
                name,
                &self.stream,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )?;
        }
        Ok(())
    }
    fn clear_status(&mut self) -> Result<()> {
        let mut v = vec![SGUARD; GUARD + 1];
        v[0] = 0;
        self.dev.htod_sync_copy_into(&v, &mut *self.status)?;
        Ok(())
    }
    fn status(&self) -> Result<u32> {
        self.dev.synchronize()?;
        let v = self.dev.dtoh_sync_copy(&self.status)?;
        ensure!(
            v.len() == GUARD + 1 && v[1..].iter().all(|x| *x == SGUARD),
            "status OOB guard changed"
        );
        Ok(v[0])
    }
    fn reset(&self, lane: usize) -> Result<()> {
        self.launch(
            "dsv41_chunk_store_reset",
            S * MAX_CONTEXT * D,
            &[
                ptr(&self.windows),
                ptr(&self.compressed),
                ptr(&self.status),
                lane as u64,
                LANES as u64,
            ],
        )
    }
    fn reset_all(&mut self) -> Result<()> {
        self.clear_status()?;
        for lane in 0..LANES {
            self.reset(lane)?;
        }
        ensure!(self.status()? == 0, "reset failed");
        Ok(())
    }
    fn write(
        &self,
        lane: usize,
        layer: usize,
        start: usize,
        tokens: usize,
        input: &CudaSlice<bf16>,
    ) -> Result<()> {
        self.launch(
            "dsv41_chunk_store_window_write",
            tokens * D,
            &[
                ptr(input),
                ptr(&self.windows),
                ptr(&self.status),
                lane as u64,
                LANES as u64,
                layer as u64,
                start as u64,
                tokens as u64,
            ],
        )
    }
    fn append(
        &self,
        lane: usize,
        slot: usize,
        first: usize,
        rows: usize,
        input: &CudaSlice<bf16>,
    ) -> Result<()> {
        self.launch(
            "dsv41_chunk_store_compressed_append",
            rows * D,
            &[
                ptr(input),
                ptr(&self.compressed),
                ptr(&self.status),
                lane as u64,
                LANES as u64,
                slot as u64,
                first as u64,
                rows as u64,
            ],
        )
    }
    fn expected_cache(&self, w: &[bf16], c: &[bf16]) -> Result<Value> {
        ensure!(
            w.len() == WE + GUARD && c.len() == CE + GUARD,
            "CPU fullcache geometry"
        );
        let aw = self.dev.dtoh_sync_copy(&self.windows)?;
        let ac = self.dev.dtoh_sync_copy(&self.compressed)?;
        Ok(
            json!({"window_elements_including_guard":aw.len(),"compressed_elements_including_guard":ac.len(),"window_bit_mismatches":diff_b(&aw,w),"compressed_bit_mismatches":diff_b(&ac,c),"window_actual_sha256":sha_b(&aw),"window_expected_sha256":sha_b(w),"compressed_actual_sha256":sha_b(&ac),"compressed_expected_sha256":sha_b(c),"status":self.status()?}),
        )
    }
}
fn cache_ok(v: &Value) -> bool {
    v["window_bit_mismatches"] == 0 && v["compressed_bit_mismatches"] == 0 && v["status"] == 0
}
fn set_window(expected: &mut [bf16], lane: usize, layer: usize, start: usize, tokens: usize) {
    for row in start..start + tokens {
        let at = ((lane * L + layer) * W + row % W) * D;
        for col in 0..D {
            expected[at + col] = data(lane, layer, row, col, false);
        }
    }
}
fn set_compressed(expected: &mut [bf16], lane: usize, slot: usize, first: usize, rows: usize) {
    let layer = OWNERS[slot];
    let at = ((lane * S + slot) * MAX_CONTEXT + first) * D;
    for i in 0..rows * D {
        expected[at + i] = data(lane, layer, first + i / D, i % D, true);
    }
}
fn strides(raw: &mut Raw, checks: &mut Vec<Value>) -> Result<()> {
    raw.reset_all()?;
    let mut w = guard_b(WE);
    let mut c = guard_b(CE);
    let v = raw.expected_cache(&w, &c)?;
    record(checks, "all_cache_initial_zero", cache_ok(&v), v)?;
    for lane in 0..LANES {
        for layer in 0..L {
            let input = Queued::new(
                &raw.dev,
                raw.dev
                    .htod_sync_copy(&payload(lane, layer, 0, 128, false))?,
            );
            raw.write(lane, layer, 0, 128, &input)?;
            set_window(&mut w, lane, layer, 0, 128);
        }
        for slot in 0..S {
            let input = Queued::new(
                &raw.dev,
                raw.dev
                    .htod_sync_copy(&payload(lane, OWNERS[slot], 0, 128, true))?,
            );
            raw.append(lane, slot, 0, 128, &input)?;
            set_compressed(&mut c, lane, slot, 0, 128);
        }
    }
    let v = raw.expected_cache(&w, &c)?;
    record(
        checks,
        "all40_windows_four_sources_two_lanes",
        cache_ok(&v),
        v,
    )?;
    raw.reset(1)?;
    w[L * W * D..WE].fill(bf16::ZERO);
    c[S * MAX_CONTEXT * D..CE].fill(bf16::ZERO);
    let v = raw.expected_cache(&w, &c)?;
    record(checks, "lane1_reset_preserves_lane0", cache_ok(&v), v)?;
    Ok(())
}
fn source_selection(start: usize, tokens: usize, r: usize, compressed: usize) -> Vec<i32> {
    let k = compressed.min(512);
    let mut out = Vec::with_capacity(tokens * k);
    for q in 0..tokens {
        let visible = (start + q + 1) / r;
        let keep = visible.min(k);
        out.extend((visible - keep..visible).map(|j| j as i32));
        out.resize(out.len() + k - keep, -1);
    }
    out
}
fn expected_indices(
    start: usize,
    tokens: usize,
    r: usize,
    compressed: usize,
) -> Result<(Vec<i32>, usize)> {
    let p = window_chunk(start, tokens)?;
    let k = compressed.min(512);
    let selected = if r > 0 {
        source_selection(start, tokens, r, compressed)
    } else {
        vec![]
    };
    let mut out = Vec::new();
    for q in 0..tokens {
        // Independent one-token history: last128 original tokens, irrespective of
        // the chunk and physical ring; convert logical keys into tight offsets.
        let at = start + q;
        let first = (at + 1).saturating_sub(W);
        let mut row = (first..=at)
            .map(|g| (g - p.snapshot_global_base) as i32)
            .collect::<Vec<_>>();
        row.resize(p.query_capacity, -1);
        ensure!(
            row == p.query_indices[q],
            "ChunkPlan differs from one-token window history"
        );
        out.extend(row);
        if r > 0 {
            for &s in &selected[q * k..(q + 1) * k] {
                out.push(if s < 0 {
                    -1
                } else {
                    p.snapshot_rows as i32 + s
                });
            }
        }
    }
    Ok((out, p.query_capacity + k))
}
fn stage(
    raw: &mut Raw,
    lane: usize,
    layer: usize,
    start: usize,
    tokens: usize,
    label: &str,
    checks: &mut Vec<Value>,
) -> Result<usize> {
    let p = window_chunk(start, tokens)?;
    let r = ratio(layer);
    let cp = if r > 0 {
        Some(compressed_chunk(start, tokens, r)?)
    } else {
        None
    };
    let cr = cp.as_ref().map_or(0, |x| x.stored_rows);
    let nr = cp.as_ref().map_or(0, |x| x.new_rows);
    let slot = owner(layer).unwrap_or(0);
    let kvrows = p.snapshot_rows + cr;
    let (expected_i, topk) = expected_indices(start, tokens, r, cr)?;
    let device_peak = (WE + CE + 2 * GUARD) * 2
        + (GUARD + 1) * 4
        + (p.snapshot_rows + kvrows + tokens + nr) * D * 2
        + (tokens * topk + tokens * cr.min(512) + 2 * GUARD) * 4
        + 2 * GUARD * 2;
    ensure!(device_peak <= DEVICE_BOUND, "stage device-array budget");
    let current = Queued::new(
        &raw.dev,
        raw.dev
            .htod_sync_copy(&payload(lane, layer, start, tokens, false))?,
    );
    let old = Queued::new(
        &raw.dev,
        raw.dev.htod_sync_copy(&guard_b(p.snapshot_rows * D))?,
    );
    let kv = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&guard_b(kvrows * D))?);
    let indices = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&guard_i(tokens * topk))?);
    let new = if OWNERS.contains(&layer) && nr > 0 {
        Some(Queued::new(
            &raw.dev,
            raw.dev
                .htod_sync_copy(&payload(lane, layer, start / r, nr, true))?,
        ))
    } else {
        None
    };
    let selected = if cr > 512 {
        Some(Queued::new(
            &raw.dev,
            raw.dev
                .htod_sync_copy(&source_selection(start, tokens, r, cr))?,
        ))
    } else {
        None
    };
    raw.launch(
        "dsv41_chunk_store_window_snapshot",
        p.snapshot_rows * D,
        &[
            ptr(&old),
            ptr(&current),
            ptr(&raw.windows),
            ptr(&raw.status),
            lane as u64,
            LANES as u64,
            layer as u64,
            start as u64,
            tokens as u64,
        ],
    )?;
    raw.write(lane, layer, start, tokens, &current)?;
    if let Some(n) = &new {
        raw.append(lane, slot, start / r, nr, n)?;
    }
    raw.launch(
        "dsv41_chunk_store_combined_snapshot",
        kvrows * D,
        &[
            ptr(&kv),
            ptr(&old),
            ptr(&raw.compressed),
            ptr(&raw.status),
            lane as u64,
            LANES as u64,
            slot as u64,
            p.snapshot_rows as u64,
            cr as u64,
        ],
    )?;
    raw.launch(
        "dsv41_chunk_store_indices",
        tokens * topk,
        &[
            ptr(&indices),
            selected.as_ref().map(|x| ptr(x)).unwrap_or(0),
            ptr(&raw.status),
            start as u64,
            tokens as u64,
            r as u64,
            cr as u64,
        ],
    )?;
    let status = raw.status()?;
    let actual_old = raw.dev.dtoh_sync_copy(&old)?;
    let actual_kv = raw.dev.dtoh_sync_copy(&kv)?;
    let actual_i = raw.dev.dtoh_sync_copy(&indices)?;
    let mut expected_old = payload(lane, layer, p.snapshot_global_base, p.snapshot_rows, false);
    expected_old.extend(vec![bf16::from_bits(BGUARD); GUARD]);
    let mut expected_kv = expected_old[..p.snapshot_rows * D].to_vec();
    if cr > 0 {
        expected_kv.extend(payload(lane, OWNERS[slot], 0, cr, true));
    }
    expected_kv.extend(vec![bf16::from_bits(BGUARD); GUARD]);
    let mut wanted_i = expected_i.clone();
    wanted_i.extend(vec![IGUARD; GUARD]);
    let mo = diff_b(&actual_old, &expected_old);
    let mk = diff_b(&actual_kv, &expected_kv);
    let mi = diff_i(&actual_i, &wanted_i);
    // This wrong alternative must change at least one nonnegative compressed
    // offset. It protects the single-offset ABI without altering actual outputs.
    let mut wrong = expected_i.clone();
    for row in wrong.chunks_mut(topk) {
        for i in &mut row[p.query_capacity..] {
            if *i >= 0 {
                *i += p.snapshot_rows as i32;
            }
        }
    }
    let double_offset = diff_i(&expected_i, &wrong);
    record(
        checks,
        format!("{label}_start_{start}_layer_{layer}"),
        status == 0 && mo + mk + mi == 0,
        json!({"lane":lane,"layer":layer,"ratio":r,"start":start,"tokens":tokens,"window_rows":p.snapshot_rows,"compressed_rows":cr,"new_compressed_rows":nr,"kv_rows":kvrows,"topk":topk,"snapshot_bit_mismatches":mo,"KV_bit_mismatches":mk,"index_mismatches":mi,"status":status,"snapshot_sha256":sha_b(&actual_old),"KV_sha256":sha_b(&actual_kv),"indices_sha256":sha_i(&actual_i),"expected_snapshot_sha256":sha_b(&expected_old),"expected_KV_sha256":sha_b(&expected_kv),"expected_indices_sha256":sha_i(&wanted_i),"one_token_reference":"last128 logical history; monotonic source-key scores select latest512 visible groups, position-sorted; future fillers=-1","output_guard_elements":GUARD,"explicit_device_array_peak_bytes":device_peak,"wrong_double_offset_mismatches":double_offset}),
    )?;
    Ok(double_offset)
}
fn recipe(
    raw: &mut Raw,
    label: &str,
    chunks: &[usize],
    checks: &mut Vec<Value>,
    began: Instant,
) -> Result<usize> {
    raw.reset_all()?;
    let mut w = guard_b(WE);
    let mut c = guard_b(CE);
    let mut start = 0;
    let mut canary = 0;
    for &tokens in chunks {
        ensure!(
            began.elapsed() < Duration::from_secs(WALL),
            "oracle deadline"
        );
        for layer in ACTIVE {
            canary += stage(raw, 0, layer, start, tokens, label, checks)?;
            set_window(&mut w, 0, layer, start, tokens);
            if let Some(slot) = owner(layer) {
                if OWNERS[slot] == layer {
                    let r = ratio(layer);
                    set_compressed(&mut c, 0, slot, start / r, (start + tokens) / r - start / r);
                }
            }
        }
        start += tokens;
    }
    let v = raw.expected_cache(&w, &c)?;
    record(
        checks,
        format!("{label}_full_cache"),
        cache_ok(&v),
        json!({"native_prefix":start,"full_arrays":v,"untouched_raw_lane1_zero":true}),
    )?;
    Ok(canary)
}
fn expected_status(
    raw: &mut Raw,
    checks: &mut Vec<Value>,
    name: &str,
    expected: u32,
    call: impl FnOnce(&Raw) -> Result<()>,
) -> Result<()> {
    raw.clear_status()?;
    call(raw)?;
    let actual = raw.status()?;
    record(
        checks,
        name,
        actual == expected,
        json!({"status":actual,"expected_status":expected,"status_guard_intact":true}),
    )
}
fn controls(raw: &mut Raw, checks: &mut Vec<Value>) -> Result<()> {
    let before_window_sha = sha_b(&raw.dev.dtoh_sync_copy(&raw.windows)?);
    let before_compressed_sha = sha_b(&raw.dev.dtoh_sync_copy(&raw.compressed)?);
    let input = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&vec![bf16::ONE; 129 * D])?);
    let output = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&guard_b(256 * D))?);
    let indices = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&guard_i(128 + 512))?);
    for (name, lane, lanes) in [
        ("negative_lane_outside_raw_allocation", 2, 2),
        ("negative_zero_lane_count", 0, 0),
    ] {
        expected_status(raw, checks, name, 2, |x| {
            x.launch(
                "dsv41_chunk_store_reset",
                1,
                &[
                    ptr(&x.windows),
                    ptr(&x.compressed),
                    ptr(&x.status),
                    lane,
                    lanes,
                ],
            )
        })?;
    }
    for (name, layer, start, tokens) in [
        ("negative_layer40", 40, 0, 1),
        ("negative_chunk129", 0, 0, 129),
        ("negative_context8193", 0, 8192, 1),
    ] {
        expected_status(raw, checks, name, 2, |x| {
            x.launch(
                "dsv41_chunk_store_window_write",
                1,
                &[
                    ptr(&input),
                    ptr(&x.windows),
                    ptr(&x.status),
                    0,
                    2,
                    layer,
                    start,
                    tokens,
                ],
            )
        })?;
    }
    for (name, slot, first, rows) in [
        ("negative_owner4", 4, 0, 1),
        ("negative_compressed_overflow", 0, 8192, 1),
        ("negative_compressed_chunk129", 0, 0, 129),
    ] {
        expected_status(raw, checks, name, 2, |x| {
            x.launch(
                "dsv41_chunk_store_compressed_append",
                1,
                &[
                    ptr(&input),
                    ptr(&x.compressed),
                    ptr(&x.status),
                    0,
                    2,
                    slot,
                    first,
                    rows,
                ],
            )
        })?;
    }
    expected_status(raw, checks, "negative_snapshot_window256", 2, |x| {
        x.launch(
            "dsv41_chunk_store_combined_snapshot",
            1,
            &[
                ptr(&output),
                ptr(&input),
                ptr(&x.compressed),
                ptr(&x.status),
                0,
                2,
                0,
                256,
                0,
            ],
        )
    })?;
    for (name, start, r, cr) in [
        ("negative_ratio3", 0, 3, 0),
        ("negative_ratio0_compressed", 0, 0, 1),
        ("negative_missing_selection513", 512, 1, 513),
    ] {
        expected_status(raw, checks, name, 2, |x| {
            x.launch(
                "dsv41_chunk_store_indices",
                1,
                &[ptr(&indices), 0, ptr(&x.status), start, 1, r, cr],
            )
        })?;
    }
    let unchanged = before_window_sha == sha_b(&raw.dev.dtoh_sync_copy(&raw.windows)?)
        && before_compressed_sha == sha_b(&raw.dev.dtoh_sync_copy(&raw.compressed)?)
        && diff_b(&raw.dev.dtoh_sync_copy(&output)?, &guard_b(256 * D)) == 0
        && raw.dev.dtoh_sync_copy(&indices)? == guard_i(128 + 512);
    record(
        checks,
        "negative_geometry_preserves_full_cache_and_outputs",
        unchanged,
        json!({"all_window_and_compressed_arrays_unchanged":unchanged,"rejected_output_and_index_arrays_unchanged":unchanged}),
    )?;
    for (name, bad) in [
        ("negative_future_selected_key", 513),
        ("negative_selected_below_minus1", -2),
    ] {
        let selected = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&vec![bad; 512])?);
        expected_status(raw, checks, name, 4, |x| {
            x.launch(
                "dsv41_chunk_store_indices",
                640,
                &[
                    ptr(&indices),
                    ptr(&selected),
                    ptr(&x.status),
                    512,
                    1,
                    1,
                    513,
                ],
            )
        })?;
        let got = raw.dev.dtoh_sync_copy(&indices)?;
        ensure!(
            got[640..].iter().all(|x| *x == IGUARD),
            "negative indices OOB"
        );
    }
    for (name, bits, compressed) in [
        ("negative_window_NaN", 0x7fc1, false),
        ("negative_compressed_Inf", 0x7f80, true),
    ] {
        let mut v = vec![bf16::ONE; D];
        v[9] = bf16::from_bits(bits);
        let bad = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&v)?);
        expected_status(raw, checks, name, 1, |x| {
            if compressed {
                x.append(1, 0, 0, 1, &bad)
            } else {
                x.write(1, 0, 0, 1, &bad)
            }
        })?;
    }
    // Corruption already in a cache must also be detected and sanitized by
    // combined_snapshot, independently of input-side append checks.
    let mut corrupt = raw.dev.dtoh_sync_copy(&raw.compressed)?;
    corrupt[S * MAX_CONTEXT * D + 9] = bf16::from_bits(0x7fc1);
    raw.dev
        .htod_sync_copy_into(&corrupt, &mut *raw.compressed)?;
    expected_status(raw, checks, "negative_combined_cached_NaN", 1, |x| {
        x.launch(
            "dsv41_chunk_store_combined_snapshot",
            2 * D,
            &[
                ptr(&output),
                ptr(&input),
                ptr(&x.compressed),
                ptr(&x.status),
                1,
                2,
                0,
                1,
                1,
            ],
        )
    })?;
    let combined = raw.dev.dtoh_sync_copy(&output)?;
    ensure!(
        combined[D + 9].to_bits() == 0,
        "nonfinite combined cache not sanitized"
    );
    // Last Inf failure stays set across subsequent valid writes. Reset does not
    // silently clear status, so explicit owner action is required for recovery.
    raw.write(1, 0, 1, 1, &input)?;
    raw.reset(1)?;
    record(
        checks,
        "negative_sticky_status_survives_valid_write_and_reset",
        raw.status()? == 1,
        json!({"status":raw.status()?,"expected_status":1}),
    )?;
    let actual = raw.dev.dtoh_sync_copy(&raw.windows)?;
    let compressed = raw.dev.dtoh_sync_copy(&raw.compressed)?;
    record(
        checks,
        "all_negative_output_extent_guards",
        actual[WE..].iter().all(|x| x.to_bits() == BGUARD)
            && compressed[CE..].iter().all(|x| x.to_bits() == BGUARD),
        json!({"window_guard_elements":GUARD,"compressed_guard_elements":GUARD,"status_guard_intact":true}),
    )?;
    // Actual deliberately wrong launch order, isolated in raw lane1. Earliest
    // queries need old positions0..126, overwritten by the new127..254 chunk.
    raw.reset_all()?;
    let old = Queued::new(
        &raw.dev,
        raw.dev.htod_sync_copy(&payload(1, 0, 0, 127, false))?,
    );
    raw.write(1, 0, 0, 127, &old)?;
    let current = Queued::new(
        &raw.dev,
        raw.dev.htod_sync_copy(&payload(1, 0, 127, 128, false))?,
    );
    let snap = Queued::new(&raw.dev, raw.dev.htod_sync_copy(&guard_b(255 * D))?);
    raw.write(1, 0, 127, 128, &current)?;
    raw.launch(
        "dsv41_chunk_store_window_snapshot",
        255 * D,
        &[
            ptr(&snap),
            ptr(&current),
            ptr(&raw.windows),
            ptr(&raw.status),
            1,
            2,
            0,
            127,
            128,
        ],
    )?;
    let a = raw.dev.dtoh_sync_copy(&snap)?;
    let mut wanted = payload(1, 0, 0, 255, false);
    wanted.extend(vec![bf16::from_bits(BGUARD); GUARD]);
    let mismatches = diff_b(&a, &wanted);
    record(
        checks,
        "negative_snapshot_after_ring_write_detected",
        mismatches == 127 * (D - 3)
            && raw.status()? == 0
            && a[255 * D..].iter().all(|x| x.to_bits() == BGUARD),
        json!({"bit_mismatches":mismatches,"expected_wrong_order_mismatches":127*(D-3),"actual_wrong_order_sha256":sha_b(&a),"correct_expected_sha256":sha_b(&wanted),"wrong_order_deliberately_executed":true,"does_not_qualify_production_wrapper_ordering":true}),
    )?;
    Ok(())
}
fn run(checks: &mut Vec<Value>) -> Result<Value> {
    ensure!(
        std::env::args_os().len() == 1,
        "chunk oracle takes no arguments"
    );
    let began = Instant::now();
    let peak = (WE + CE + 2 * GUARD) * 2 + (GUARD + 1) * 4 + 16 * 1024 * 1024;
    ensure!(
        peak < DEVICE_BOUND && 4 * peak < HOST_BOUND * 2,
        "oracle memory bound"
    );
    let dev = CudaDevice::new(0)?;
    let mut raw = Raw::new(dev)?;
    strides(&mut raw, checks)?;
    let mut canary = recipe(&mut raw, "even_chunks", &[128, 128, 1], checks, began)?;
    canary += recipe(
        &mut raw,
        "odd_crossing",
        &[1, 126, 128, 128, 2, 127],
        checks,
        began,
    )?;
    let mut long = vec![127];
    long.extend(vec![128; 63]);
    long.push(1);
    ensure!(long.iter().sum::<usize>() == 8192, "full native extent");
    canary += recipe(&mut raw, "full8192", &long, checks, began)?;
    record(
        checks,
        "negative_double_offset_detected",
        canary > 0,
        json!({"wrong_alternative_index_mismatches":canary,"actual_kernel_offset_count":1}),
    )?;
    controls(&mut raw, checks)?;
    ensure!(
        began.elapsed() < Duration::from_secs(WALL),
        "oracle deadline"
    );
    let expected = 3 + (3 + 6 + 65) * 6 + 3 + 1 + 21;
    ensure!(
        checks.len() == expected,
        "oracle coverage changed: {} vs{expected}",
        checks.len()
    );
    Ok(
        json!({"schema_version":1,"kind":"dsv41_chunk_store_raw_gpu_component_oracle","status":"PASS","all_passed":true,"checks":checks,"source_id_hex":SOURCE_ID,"cuda_source_sha256":SOURCE_CUDA_SHA,"ptx_sha256":format!("{:x}",Sha256::digest(PTX.as_bytes())),"elapsed_seconds":began.elapsed().as_secs_f64(),"model_free":true,"model_execution":false,"serve_supported":false,"RDMA":false,"raw_lanes":LANES,"production_wrapper_lanes":1,"limits":{"wall_seconds":WALL,"explicit_device_array_bound_bytes":DEVICE_BOUND,"explicit_host_array_bound_bytes":HOST_BOUND,"persistent_device_array_bytes":(WE+CE+2*GUARD)*2+(GUARD+1)*4},"scope":"raw cache/index-copy numerical ABI only; synthetic already-QAT BF16 data and supplied source selections; no forged IndexOutput, learned compression/index/projection, production wrapper transaction/receipt/lifetime, attention numerical, model, B2 serving, quality or speed qualification"}),
    )
}
fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(WALL));
        eprintln!("raw chunk store oracle hard deadline");
        std::process::exit(124);
    });
    let mut checks = Vec::new();
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut checks))) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            json!({"schema_version":1,"kind":"dsv41_chunk_store_raw_gpu_component_oracle","status":"FAIL","all_passed":false,"checks":checks,"error":format!("{e:#}")})
        }
        Err(_) => {
            json!({"schema_version":1,"kind":"dsv41_chunk_store_raw_gpu_component_oracle","status":"FAIL","all_passed":false,"checks":checks,"error":"panic"})
        }
    };
    println!("{result}");
    if result["all_passed"] != true {
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_one_token_reference_has_exact_chunkplan_indices_at_every_boundary() {
        let mut recipes = vec![vec![128, 128, 1], vec![1, 126, 128, 128, 2, 127]];
        let mut long = vec![127];
        long.extend(vec![128; 63]);
        long.push(1);
        recipes.push(long);
        for chunks in recipes {
            let mut start = 0;
            for tokens in chunks {
                for r in [0, 1, 2] {
                    let cr = if r == 0 { 0 } else { (start + tokens) / r };
                    let (indices, topk) = expected_indices(start, tokens, r, cr).unwrap();
                    let p = window_chunk(start, tokens).unwrap();
                    assert_eq!(indices.len(), tokens * topk);
                    for q in 0..tokens {
                        let at = start + q;
                        let row = &indices[q * topk..(q + 1) * topk];
                        let logical = row[..p.query_capacity]
                            .iter()
                            .filter(|x| **x >= 0)
                            .map(|x| p.snapshot_global_base + *x as usize)
                            .collect::<Vec<_>>();
                        assert_eq!(
                            logical,
                            ((at + 1).saturating_sub(128)..=at).collect::<Vec<_>>()
                        );
                        if r > 0 {
                            let visible = (at + 1) / r;
                            let compressed = row[p.query_capacity..]
                                .iter()
                                .filter(|x| **x >= 0)
                                .map(|x| *x as usize - p.snapshot_rows)
                                .collect::<Vec<_>>();
                            assert_eq!(
                                compressed,
                                (visible.saturating_sub(512)..visible).collect::<Vec<_>>()
                            );
                        }
                    }
                }
                start += tokens;
            }
        }
    }
    #[test]
    fn raw_BF16_row_fingerprints_expose_wrong_order_even_at_high_native_positions() {
        for start in [127usize, 128, 1023, 4095, 8064] {
            let old_start = start.saturating_sub(127);
            let mut ring = vec![bf16::ZERO; 128 * D];
            set_window(&mut ring, 0, 0, old_start, start - old_start);
            set_window(&mut ring, 0, 0, start, 128);
            let mut mismatches = 0;
            for position in old_start..start {
                for col in 0..D {
                    mismatches += usize::from(
                        ring[(position % 128) * D + col].to_bits()
                            != data(0, 0, position, col, false).to_bits(),
                    );
                }
            }
            assert_eq!(mismatches, 127 * (D - 3));
            assert_ne!(
                data(0, 0, start, 41, false).to_bits(),
                data(1, 0, start, 41, false).to_bits()
            );
            assert_ne!(
                data(0, 0, start, 41, false).to_bits(),
                data(0, 39, start, 41, false).to_bits()
            );
        }
    }
}
