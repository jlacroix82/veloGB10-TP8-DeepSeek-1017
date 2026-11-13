# Packed attention status cut experiment

The pinned status backend is opt-in on the packed atomic caller. It changes
only the first projection cut of each packed attention layer. The original
path remains the default, and the CUDA kernels, weights, sample policy,
projection arithmetic, status writers, and collective schedule stay the same.

The candidate copies all ten individual status words to one reusable 40-byte
page-locked host allocation on the exact model compute stream, then fences that
stream once. Every source allocation remains owned through the fence. It checks
the ten values before the same next dependent stage as the original path.
The cut binds once to the model's transport owner and boot. A CUDA failure or
nonzero status poisons the cut, and the existing outer error path aborts the
session and fences. No status is reset or hidden. The new process-local host
allocation is released only when the CUDA stream fence succeeds.

Before model timing, build native artifacts and run the focused CPU policy
test, then run the explicitly ignored model-free CUDA oracle under the usual
exclusive worker guards. Compare original and candidate full native IDs/text,
rank ownership, launch identities, exact ten-word receipts, fabric and clean
reset on the same frozen packed workload. Profile the affected host fences and
actual whole-wave time separately. The removal of repeated synchronizations
does not predict a speedup and cannot qualify the long-context or natural task
requirements. No CUDA graph or whole-model dispatch change is made here.
