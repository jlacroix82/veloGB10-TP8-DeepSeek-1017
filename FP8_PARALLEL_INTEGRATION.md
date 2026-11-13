# Native1017 FP8 parallel contribution integration

This is an opt-in implementation candidate, with no new model or speed qualification.
The default remains `--fp8-gemm-backend original`. The atomic resident service
and packed model probe accept `--fp8-gemm-backend parallel-groups-hot-small-m`.

The actual selector uses only FP8 matrices with 1–16 rows, K=5120 and
N=512/1280/2304 (attention KV/query-A and shared-expert gate/up). Other shapes,
including larger prefill and all FP4 experts, keep the original entrypoint.
The parallel kernel and ordered finish are ported unchanged from qualified raw
source 7bc20f646c25e876dc2d36f8d861c8cbba70937d. Native bytes/scales, one MMA
per K32, corrected multiply order, ascending F32 addition/reset and BF16 output
conversion remain unchanged. It is not a tree reduction or a new quantization.

One 23,592,960-byte (22.5 MiB) contribution arena is shared between attention
and MoE kernels for one actual model/boot/stream. Shape arithmetic remains
bounded separately by 256 MiB. The arena binds once to the actual opaque model
instance, and is checked before each forward. A host mutex seals each complete
producer/finish enqueue pair on the exact same owned stream. Stream order
allows later scratch reuse only after the previous finish consumes it. Returned
output and status allocations remain independently owned; each selected output
retains the arena owner and its actual `Arc<CudaStream>`. Enqueue failure poisons the arena and fences while all
producers are alive. Final arena-owner destruction fences once. No per-launch
workspace fence was added. Existing correctness/status fences remain in place.

The immutable model identity includes backend, arena UUID, model/boot/stream
binding and capacity. Each checked forward and terminal service record expose
actual queued counts for all three eligible output widths and original fallback
calls. These counters are separate from reset/session identity comparisons.

The existing official1017 model, prompt/tokenizer/encoding, output/EOS semantics,
TP8 RoCE program, Engram, KV, tool policy and external 8 GiB RAM / 80°C GPU /
90°C board safety controls are unchanged. Full learned-model native output,
batch invariance, resource, protocol and completed-task gates remain required
before any matched speed claim or promotion. No GPU execution is part of the
source preparation.

The inherited packed-probe constants referred to an older77f/4766 collective
helper while this main tree and the qualified atomic ELF use d412/8da. The
new probe uses independently recomputed7823/3e67 program identities for the
exact same operation bytes and budgets. The original13-pass/2-fail CPU attempt
is retained privately. A separate fixture pins both complete program hashes,
counts/epochs and actual compiled helper/transport identities; it does not
change any native output reference or numerical acceptance rule.
