# DeepSeek 1017 TP8: buffer reuse and CUDA graphs

On November 13, 2026, eight GB10 systems ran the official 1017 checkpoint
at 2 GHz CPU and GPU. Each comparison used the same binary in its control
and candidate arms. The workload was two untimed ramps, then C8/C16/C16/C8
with 75 input tokens and a 32-token output allowance per request. Rates
divide all native output tokens by the full client-observed wave time,
including prefill and the buffered response. There were two timed repeats
per width per arm. The speed requests are separate from the naturally
completed coding and tool checks.

| Change | C8 median tok/s | C16 median tok/s | Slowest-rank median decode step at C8 / C16 |
| --- | ---: | ---: | ---: |
| GPU scratch off → reuse | 20.57 → 20.49 (−0.42%) | 27.09 → 26.91 (−0.66%) | 238.67 → 238.82 / 318.51 → 318.47 ms |
| Three-kernel graph off → on, reuse fixed on | 20.38 → 20.46 (+0.38%) | 26.69 → 26.95 (+0.97%) | 236.60 → 236.52 / 321.95 → 318.19 ms |

The scratch pool reserves 8,405,952 bytes per rank and reuses 11
layer-local buffers per eligible decode width. It leaves KV, Engram
and collective buffers outside the pool. It did not speed up decode.
The graph captures the three attention input kernels and leaves checks,
collectives and cache updates outside capture. All eight ranks captured
C8 and C16 before timing. The two C16 graph repeats improved by 1.94%
and 0.04%; the result does not establish a dependable gain.

Both pairs matched prompt token IDs, effective sampling and thinking
settings, generated tokens, stop reasons and Engram histories. The same
two Python fixes passed protected tests; the tool task read 37, set 38
once, and returned 38. These are correctness fixtures, not a full
quality pilot or a measurement of completed tasks per hour.

All eight RoCE ports carried traffic, with zero strict counter increases
during requests. One adaptive retransmit after the requests disqualified
the reuse arm. Two after the requests disqualified the corrected graph
control. The graph-on arm itself closed cleanly. The original graph
candidate failed before timing because the driver rejected a stream flags
query inside capture. Moving the validation before capture preserved the
launches; the corrected model-free GB10 check compared 403,104 output
values bit for bit. The corrected graph pair peaked at 50°C GPU and
52.5°C board, with at least 33.78 GiB available. No serving route changed.

The [wave data](../benchmarks/capped-tp8-hc-2026-11-13.csv) contains the
actual token counts, elapsed times and decode brackets. [Run identities](../benchmarks/capped-tp8-hc-2026-11-13.json)
record the measured source and binary digests, evidence digests, and
verification outcomes. This repository is a clean source export; it does
not include model data, private launcher state, or raw cluster logs.
