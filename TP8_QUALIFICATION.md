# Experimental TP8 qualification — 2026-11-07

This local development branch qualifies one bounded **dense Qwen** profile on
eight GB10 systems. It does not add qualified TP8 support for DeepSeek, EXL3,
Naive or MoE models. Production promotion, graphs, speculation, prefix reuse,
long-context serving, restart injection and a sustained mixed-work soak remain
outside these results.

## Tested identities

| Component | Identity |
|---|---|
| Upstream base | `438176b68436295a0a9bbda11355e10dbf92be01` |
| Tested serving source | `b624f8828ffe531259076124f3447b91803a3380` |
| Native release executable SHA256 | `9f11af7615fa7d4961228ad4690b3b6f2ce2e17404eb3fc44013361ace1aac8f` |
| Embedded source / kernel build IDs | `32831bf8a9437232` / `eb9dcb68e4f1c950` |
| Compiler | Rust 1.98.1; CUDA 13.0.88; native AArch64 release build |
| Runtime | Native executable, no serving container image; driver 580.178.04 |
| Platform | Kernel `7.0.0-1024-nvidia`, 4 KiB pages; eight separate physical GPUs |
| Model label | Local `qwen3.8-27b-nvfp4`, mixed compressed-tensors NVFP4/FP8 preview, `qwen3_5` / `qwen3_5_text` |
| Local model metadata revision | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| Main weight file SHA256 | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` |
| Full model manifest SHA256 | `86fc9cfec20e818cefa6b316099f44b1e466c9bcc60489088bb591fef15f47ae` |
| Client harness SHA256 | `68fd2d6d2a7c9db4057fa962e92890f36c71343ed0fb4ee7c926055b24334edb` |
| Shared workload SHA256 | `41259d5bdf12ad233ee99ca0a75cca190e52ad0908c4e84a2143175501a651f6` |

The exact public quantized-model repository name was not recovered; the local
label is not a claim of official weight provenance. Model, tokenizer and encoding
files were identified by the private checksum manifest and reused unchanged.
All twelve staged model files were verified on each of the seven workers; source
metadata and the unused MTP weights remain retained. All 21 PTX artifacts stayed
byte-identical during the attention dispatch correction. Later documentation
changes do not change the identity of the tested executable above.

The test used a rank-zero head, seven workers, and a separate controller/client.
Every node retained `min_free_kbytes=2097152`. The existing GPU clock policy was
unchanged; observed busy clocks were approximately 2 GHz. No kernel, module,
network, global service or clock changes were made during this campaign.

## Profile and measurements

FFN/GDN weights shard eight ways. Attention replicates when its KV head count
cannot split eight ways; the output head replicates. Reductions use three serial
recursive-doubling rounds with FP32 partials and GPU-validated receive tails.

```text
--tp 8 --experimental-tp8
--mtp off --spec-source none --no-decode-graphs --tp-fp32-partials
--kv-cache bf16 --tp-shard-mtp off --tp-gpu-recv on --tp-reduce-fuse off
--prefix-cache off --max-seq-len 4096 --tp-batch-trace
--max-batch 8
```

Device/GID, peer addresses, model directory and private API/control ports were
recorded per run. `--max-batch` was set to 2, 4 or 8 for the corresponding cell.
Trace logging was enabled in every measured profile. A stock
served-model ID was discovered and supplied explicitly to every request.

Eight distinct short prompts (four code, four prose) ran serially and then in
C-wide groups at each concurrency. Every request used the same native input IDs,
greedy sampling, neutral penalties, exact 128/256 native-token output caps and
`ignore_eos=true`. Warmups preceded timing. KV prefix reuse was disabled. The
client coordinated arrival; this is not atomic server admission.

| Concurrent requests on TP8 | 128-token cap, aggregate tok/s | 256-token cap, aggregate tok/s | Twelve natural tasks, concurrent |
|---|---:|---:|---:|
| C2 | 37.85 | 39.25 | 101.58 s, 12/12 pass |
| C4 | 64.17 | 68.02 | 80.45 s, 12/12 pass |
| C8 | 97.77 | 107.96 | 71.93 s, 12/12 pass |

Single-request serial baselines on the same TP8 service measured 20.75–20.85
tok/s at 128 tokens and 21.24–21.32 at 256. These are server-native output tokens
divided by first client send to last completion, including prefill and queueing;
they are not decode-only rates or SSE-event counts. Weighted group rates use
sum(tokens)/sum(windows). Non-streaming requests do not measure generated TTFT.
There was one bounded screen per capacity, not a repeated sustained-load campaign.

All sixteen timed concurrent outputs per cell matched their serial counterparts
in visible bytes and native counts. Exact generated token IDs were unavailable.
All cross-cell serial baselines matched too. Head traces proved the exact request
SID sets in every timed batch: C2 reached 126/254 full-width decode steps per
128/256-token group; C4 reached 124/252; C8 reached 120/248, with narrower
ramp/drain steps retained. These are head dispatch records, not independent
per-rank step attestations.

The separate natural-completion pack contained twelve synthetic variants: two
replicas of three Python function tasks and three stateful tool tasks. It tested
Unicode/BMP boundaries, nested/null arguments, discovery, error recovery, result
association and exactly-once writes. Every attempt reset its conversation and
fixture. The fixed budgets were 90 seconds, six model turns, eight tool calls and
1,024 output tokens per turn. Evaluators stayed outside the model's writable area.

All 72 attempts across isolated/concurrent C2/C4/C8 phases passed naturally,
without caps or timeouts. Each phase generated 1,936 native output tokens.
Independent replay reexecuted 312 predeclared code assertions and checked 120 tool
calls against 36 final fixture states. Serial pack time was 140.29–140.54 seconds.
Observed successful-task rates were 425.30 / 537.00 / 600.58 tasks/hour at C2/C4/C8,
respectively, using complete phase wall times. These are observations on twelve
synthetic variants, not a repository benchmark, a quality margin or a stable
production service-rate estimate.

## Defect found and retained failures

The first C2 screen failed visible-output equality in 14/16 cases. An inherited
BF16 dispatch ran tensor-core attention and then scalar split-K into the same
partial buffer, followed by the tensor-core reducer. Some keys were counted twice.
The correction makes producers exclusive and uses rebased batch-1 tensor-core
calls for independent lanes, preserving each request's arithmetic.

The standalone `examples/gqa_bf16_oracle.rs` embeds the unchanged PTX. On all eight
GPUs, both legacy variants returned **54.75 instead of 49.5** on a known-answer
case. Pure tensor-core/scalar positives passed; both tensor-core variants passed
bitwise C1/C2/C4/C8 comparisons with distinct queries, physical slots, positions
and padded query pitches through a 4K kernel context. There were 480 individual
case checks over the eight independent GPU probes. CPU-reference tolerances were
predeclared: half a BF16 ULP plus `1e-5` for uniform cases, and `0.025 + 0.01*abs(reference)`
for nonuniform softmax; cross-width comparisons required bitwise equality.
This validates kernels and
the launch plan; model-serving qualification came from the separate runs above.

The original natural pack also scored 10/12 in both phases: two counter answers
used `final_value` where the strict checker expected `value`, although all tool
effects were correct. V3 explicitly named literal JSON keys before fresh tests;
scorers, budgets, speed bodies and fixture checks were unchanged. Old failures,
streams, outputs and audits remain preserved; they were not rescored as passes.

All eight primary RoCE interfaces carried model-serving traffic. Secondary rails
stayed idle for this profile. Drops, receive errors, retransmissions and ACK
timeouts had zero measured increments; small out-of-sequence counter increments
were retained. Across corrected model runs, minimum available RAM was 58.04 GiB
and maximum sampled GPU temperature was 60°C. Independent deadlines and an
8 GiB available-memory / 80°C GPU / 90°C board guard bounded every window.

Private evidence remains in the operator's experiment archive: complete per-rank runtime identities,
raw responses, native prompt IDs, batch traces, resource/fabric records, failed
attempts, model manifests and independent audits. Final inspection verified all
eight GPUs idle with the requested 2 GiB reserve. This profile is a qualified
experimental baseline for further work, not a production promotion.
