# Experimental DeepSeek-V4.1-Flash-1017 TP8 port

The full native model now runs independent lanes through B16 on eight GB10s.
A bounded atomic HTTP service passes short B1/2/4/8/16 controls, and an upgraded
service passes natural-EOS and mixed-length B2 checks. **A matched speed win
against the previous deployment and the full completed-work quality pilot have
not been demonstrated.** The dense-Qwen results do not qualify this model.

## Comparison target

The official checkpoint revision is
`1bae0e1088993b9b82de08f59de5be5a34e42170`, with native FP8/group-32 scales and
packed FP4 experts. This work introduces no adapters, modified weights or
additional weight quantization. Header inventory identifies 96,085 tensors in
48 shards and 510,286,023,000 payload bytes; it does not certify whole-checkpoint
payload checksums. Vision and native1017 DSpark execution remain unimplemented.

The retained deployment's code medians were **560.3908 native tok/s at C8** and
**730.1430 at C16**, measured as generated native tokens divided by the whole
client wave, including prefill, queueing and intentional arrival delays. The
outputs were capped at256 tokens, temperature0/seed42, with thinking and DSpark
enabled. These figures are targets, not results of this port. The best C16
reference used two eight-request groups separated by90ms; the first native
screen uses one atomic16-request body and must be labelled accordingly.

Completed coding/tool tasks have separate executable evaluators. Capped and
timed-out responses never count as completed correct tasks. No proposed quality
margin has been accepted as a concession.

## Checked milestones

| Evidence | Result and limit |
| --- | --- |
| Native C1 model, `5939619` and resident `9dffee5` | All eight ranks execute all40 layers and both Engram layers. Code/prose32 controls match retained native IDs; resident requests reuse one loaded model and transport with checked resets. One continuation copied across ranks is one request. |
| Attention ordering correction | 378 official-AST index checks and72 old/fixed GPU attention comparisons per GPU pass with zero BF16-bit differences. The incorrect predecessor changes17 cases per GPU and remains retained. |
| Prompt semantics | 573 frozen requests match the pinned official encoder's rendered text and complete native IDs. The speed screen uses the retained75-token code and68-token prose sequences. |
| Native streaming protocol | Four of the planned20 cases pass, across six real HTTP requests with private parser/native-token evidence. This does not qualify the full protocol pack. |
| Completed coding/tool attempts | Two of the planned60 pilot attempts pass to natural completion: an interval-code fix with tests and a stateful stock-tool task. Original180-second task and4096-token-per-turn budgets remain unchanged. Their whole service window fails the strict fabric gate on one retained retransmission; it is not promoted. The preceding timeout/poisoned-service failure is also retained. |
| Packed word/G2 model, `6236dca` | All eight ranks pass208 checked forwards,17,472 operations and1,024 unique generated outputs across B1 code/prose and mixed B2/4/8/16. Rank duplicates are never added to throughput. Outputs, per-lane history, actual batches, resets, epochs, fabric and cleanup pass. |
| First atomic HTTP service, `b6b4ff0` | Six groups at B1 code/prose and B2/4/8/16 pass the full eight-rank audit with eight-output caps. One loaded model serves the groups. This is a short correctness gate. |
| Upgraded atomic HTTP service, `d9ba3fe` | Four common128-budget, thinking-off controls pass on all eight ranks: isolated short/code and both mixed B2 orders. The short response reaches real EOS; actual decode membership shrinks2→1, and each member's IDs/text/stop/count match its isolated control. Full phase/terminal agreement, unused reservation burn, fresh reset, resources, fabric and cleanup pass. The long response can reach its cap; this is not a completed coding task. |

The current source branch includes the qualified atomic service and word/G2
kernels. Its actual serving candidate was built from `d9ba3fe`; later source or
documentation commits do not rename that measured ELF. Every experiment keeps
its actual binary, source, image, CUDA/PTX and configuration identities privately.
Compiled-module counts are recorded separately from modules used by a caller.
Seventeen legacy PTX declarations are compiled; older21-file retained archives
also contain four unused predecessor artifacts and are not21 compiled modules.

## Placement, service behavior and bounds

The native model holds complete per-rank Engram table/scale shards in CUDA
allocations: about **23.604GiB per rank** in the shared128GB physical pool.
Per-token lookup uses those allocations. Model staging over NFS is separate
from token-time table access; tensor collectives have actual RoCE evidence.

The current atomic profile accepts one active text group of at most16 members,
with at most128 flattened prefill rows, **512 total prompt-plus-output tokens
per member**, and a256-token output limit. It preserves native EOS and the
request's full allowance. It explicitly refuses unsupported tool, sampling and
context requirements. It does not shorten original512-output protocol requests
or4096-output quality tasks to fit this profile.

The separate C1 resident path has an8192-token source envelope and preserves
4096-token output allowances and180-second task budgets. A formal learned
context ladder and the complete coding/tool campaign remain unqualified.
Packed learned-index integration has separate source/component/network gates;
its live512-token bound is unchanged until real checkpoint/model gates pass.

The HTTP tests use bounded private service windows and retain loaded weights
between requests. They do not establish reusable conversation KV prefixes,
continuous admission, a permanent production route or a realistic mixed-work
soak. Checked disconnect recovery is a separate candidate: its first test was
rejected before admission by an unsupported streaming-client flag, so it proves
neither successful cancellation nor failed GPU recovery. A corrected client
retains the original budgets and awaits hardware testing.

## Performance evidence and next experiments

The lossless FP4 register decoder and aligned native FP8 word loads preserve
original input/scale handling. The G2 finite guard samples the full-width abort
value afresh per block/launch while retaining numeric checks and final live
abort/ownership checks. Raw high-bit, nonfinite and queued-abort controls pass;
the combined packed-model gate above passes. One diagnostic window reduces
internal B8/B16 step medians from about316/445ms to271/341ms. These are checked
forward timings, not client token rates or completed-work throughput.

A separate native MXF8 candidate passes all1,024 packed outputs on all eight
ranks, including exact agreement with the retained deployment's first32 code
and prose tokens. Its one-window internal B16 median is319ms versus341ms for
word/G2. This is a modest diagnostic lead, with uncontrolled window variation;
it does not establish a matched HTTP speed improvement. It remains separate
from the current atomic candidate.

A fresh word/G2 profile identifies small TP8 FP8 shards as a remaining cost:
several hot projections launch only4/10/18 blocks on a48-SM GPU. The exact
parallel-K32 contribution candidate preserves original per-group MMA/scaling
and ascending F32 accumulation, with an explicit scratch bound. Its CPU and
independent numerical fixtures pass; GPU correctness and warm timings remain
pending. Original/default GEMM dispatch is unchanged.

The same trace observes approximately6,100 synchronization calls and2,000
four-byte status readbacks per decode step. Their API durations overlap GPU
work and cannot be added as removable CPU time. Receive/poll kernels include
waiting for peer computation and do not measure pure fabric latency. These
findings motivate persistent scratch, consolidated checks and separately
qualified collective scheduling. Actual pool Used/Reserved/threshold values
were not captured; no allocator-retention gain is established.

Further candidates include exact BF16 operand caches with a separately
qualified cuBLAS numerical backend, atomic Auto-tool parsing, reusable prefixes,
graph capture and native1017 DSpark. None is a performance result or a generic
flag replacement. DSpark requires target verification and exact tentative-state
commit/rollback; graphs require capture-safe workspace and transport ownership.

## Remaining gates

1. Preserve the completed atomic natural-EOS result; finish the original256-token
   C8/C16 screen with100-second client deadlines and report failures as failures.
2. Qualify checked cancellation, complete the unchanged protocol pack, and add
   actual Auto-tool fixture effects to batches without shrinking task budgets.
3. Run the genuine learned-index model/context comparisons before expanding the
   packed capacity limit; keep C1 implementation controls separate from independent
   model truth and useful long-context task quality.
4. Complete the frozen coding/tool pilot and holdout with real tests and side
   effects, retaining failures, timeouts, tokens and all consumed time.
5. Measure one-factor performance candidates, combine only retained changes,
   then repeat quality/interference, mixed work and restart/soak gates before
   promotion or routing changes.

Rank zero is head; eight GB10 systems form the test cluster. Each retains a 2 GiB kernel reserve and
the recorded clock cap. No platform, kernel or network change is part of these
source milestones. Tests require8GiB available memory per node, GPU80°C and
board90°C cutoffs, independent deadlines and exclusive engine ownership.
Source, raw receipts, resource traces, failure logs and audits remain private;
no new deployment result is inferred from code compilation.

Derived official numerical/hash/Engram code is covered by
[DeepSeek's MIT license](src/dsv41/DEEPSEEK_LICENSE). The vendored protocol
toolkit carries its upstream license and provenance manifest.
