// Isolated native-1017 collective helpers. No existing model kernel is replaced.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "tp_doorbell.h"

#ifndef DSV41_COLLECTIVES_SOURCE_ID
#error "DSV41_COLLECTIVES_SOURCE_ID must identify the isolated build inputs"
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) {
    *out = DSV41_COLLECTIVES_SOURCE_ID;
}

namespace {
constexpr unsigned WORLD = 8;
constexpr unsigned ROUNDS = 3;
constexpr unsigned VOCAB = 129280;
constexpr unsigned VOCAB_SHARD = VOCAB / WORLD;
constexpr unsigned MAX_ROWS = 128;
static_assert(VOCAB % WORLD == 0, "1017 vocabulary must divide into eight exact source slices");

__device__ __forceinline__ unsigned long long ld(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long timer() {
    unsigned long long v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}
__device__ __forceinline__ unsigned long long* flag(tp_dev_ctx* c, unsigned offset) {
    return (unsigned long long*)((char*)c->flags + offset);
}
__device__ __forceinline__ bool valid(tp_dev_ctx* c) {
    return c->world == WORLD && c->rounds == ROUNDS && c->rank >= 0 && c->rank < WORLD
        && c->oneshot == 0 && c->nway_flags && c->nway_recv;
}
__device__ __forceinline__ void fail(tp_dev_ctx* c, unsigned* status, unsigned bit) {
    // The atomic targets ordinary device scratch, never a C2C polling flag (I6).
    atomicOr(status, bit);
    st(flag(c, TP_F_ABORT), 13);
}
__device__ __forceinline__ void stamp(tp_dev_ctx* c, unsigned long long e, unsigned s) {
    if (c->gpu_ts) c->gpu_ts[(e % TP_GTS_EPOCHS) * TP_GTS_STRIDE + s] = timer();
}
}

// Startup-only snapshot: never infer the actual rank/world from a caller's claim.
extern "C" __global__ void dsv41_context_snapshot(tp_dev_ctx* c, unsigned long long* out) {
    if (blockIdx.x || threadIdx.x) return;
    out[0] = c->epoch;
    out[1] = (unsigned long long)c->rank;
    out[2] = c->world;
    out[3] = c->rounds;
    out[4] = c->oneshot;
    out[5] = c->nway_flags != nullptr;
    out[6] = c->nway_recv != nullptr;
    out[7] = ld(flag(c, TP_F_ABORT));
}

// Qualification guards. Their cost is explicit; this component is not a speed claim.
extern "C" __global__ void dsv41_check_f32(tp_dev_ctx* c, const float* x, int n,
                                            unsigned* status) {
    // These six geometry/pointer fields are immutable for the lifetime of c.
    // Sample these fields and the full-width cooperative abort flag afresh for
    // every block/launch. This read-only finite checker has no polling loop;
    // the owner must still check the live abort after its device fence. All
    // K1/K2 polling paths remain live. Every thread reaches the barrier before
    // any possible return; the flag is never cleared or cached across launches.
    __shared__ int context_valid;
    __shared__ unsigned long long block_abort;
    if (threadIdx.x == 0) {
        block_abort = ld(flag(c, TP_F_ABORT));
        context_valid = valid(c);
    }
    __syncthreads();
    if (block_abort) return;
    if (!context_valid) { if (!blockIdx.x && !threadIdx.x) fail(c, status, 4); return; }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(x[i])) fail(c, status, 1);
}

extern "C" __global__ void dsv41_check_pairs(tp_dev_ctx* c, const unsigned* pairs, int rows,
                                              unsigned* status) {
    if (ld(flag(c, TP_F_ABORT))) return;
    if (!valid(c) || rows < 1 || rows > MAX_ROWS) {
        if (!blockIdx.x && !threadIdx.x) fail(c, status, 4);
        return;
    }
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        unsigned id = pairs[2 * row + 1];
        if (!isfinite(__uint_as_float(pairs[2 * row]))) fail(c, status, 1);
        // Only the INITIAL source is rank-owned. Later round winners may originate anywhere.
        if (id / VOCAB_SHARD != (unsigned)c->rank || id >= VOCAB) fail(c, status, 2);
    }
}

// Full-pair recursive-doubling K2. K1 is the existing tp_gate_copy_signal, shipping
// rows*8 bytes. Every round writes BOTH score bits and global ID back to the same
// array before the next K1. This must never call the old index-only maxloc kernel.
extern "C" __global__ void dsv41_wait_maxloc_g(tp_dev_ctx* c, unsigned* pairs, int rows,
                                               unsigned* status) {
    __shared__ int stopped;
    if (threadIdx.x == 0) {
        stopped = ld(flag(c, TP_F_ABORT)) != 0;
        if (!valid(c) || rows < 1 || rows > MAX_ROWS) { fail(c, status, 4); stopped = 1; }
    }
    __syncthreads();
    if (stopped) return;
    const unsigned long long e = c->epoch;
    const unsigned round = e % ROUNDS;
    const unsigned peer_rank = (unsigned)c->rank ^ (1u << round);
    const unsigned slot = e & (TP_RING_SLOTS - 1);
    const unsigned char* peer_bytes = c->nway_recv
        + ((unsigned long long)round * TP_RING_SLOTS + slot) * c->slot_stride;
    const unsigned long long* hint = (const unsigned long long*)((const char*)c->nway_flags
        + TP_NWAY_PEER_OFF + peer_rank * TP_CL);
    const unsigned long long* tail = (const unsigned long long*)(peer_bytes + rows * 8u);
    stamp(c, e, TP_GTS_K2_IN);
    asm volatile("griddepcontrol.launch_dependents;");

    if (threadIdx.x == 0) {
        // The hint only chooses the deadline. Exact NIC-owned tail equality is
        // the placement proof. Never clear or otherwise write a receive tag.
        const unsigned long long* abort = flag(c, TP_F_ABORT);
        unsigned long long deadline = timer() + TP_HINT_WAIT_NS;
        unsigned ns = 64;
        bool hinted = false;
        for (;;) {
            if (ld(abort)) { stopped = 1; break; }
            if (ld(hint) >= e) { hinted = true; break; }
            if (timer() >= deadline) break;
            __nanosleep(ns);
            if (ns < 512) ns <<= 1;
        }
        deadline = timer() + (hinted ? TP_TAIL_WAIT_NS : TP_TAIL_LONG_WAIT_NS);
        ns = 64;
        while (!stopped && ld(tail) != e) {
            if (ld(abort)) { stopped = 1; break; }
            if (timer() >= deadline) {
                atomicOr(status, 8u);
                st(flag(c, TP_F_ABORT), 11);
                printf("[DS41-maxloc-tail-timeout] rank=%d epoch=%llu rows=%d hinted=%d\n",
                       c->rank, e, rows, (int)hinted);
                stopped = 1;
                break;
            }
            __nanosleep(ns);
            if (ns < 512) ns <<= 1;
        }
        if (!stopped) asm volatile("fence.acquire.sys;" ::: "memory");
    }
    __syncthreads();
    if (stopped) return;
    stamp(c, e, TP_GTS_K2_GO);
    if (threadIdx.x == 0) {
        const unsigned* peer = (const unsigned*)peer_bytes;
        // Check the entire incoming array before publishing any winner.
        for (int row = 0; row < rows; ++row) {
            if (!isfinite(__uint_as_float(pairs[2 * row]))
                || !isfinite(__uint_as_float(peer[2 * row]))) {
                fail(c, status, 1); stopped = 1;
            }
            if (pairs[2 * row + 1] >= VOCAB || peer[2 * row + 1] >= VOCAB) {
                fail(c, status, 2); stopped = 1;
            }
        }
        if (!stopped) {
            const bool local_lower = ((unsigned)c->rank & (1u << round)) == 0;
            for (int row = 0; row < rows; ++row) {
                const unsigned* lower = (local_lower ? pairs : peer) + 2 * row;
                const unsigned* upper = (local_lower ? peer : pairs) + 2 * row;
                const float lv = __uint_as_float(lower[0]), uv = __uint_as_float(upper[0]);
                const bool lower_wins = lv > uv || (lv == uv && lower[1] <= upper[1]);
                const unsigned value = lower_wins ? lower[0] : upper[0];
                const unsigned index = lower_wins ? lower[1] : upper[1];
                pairs[2 * row] = value;
                pairs[2 * row + 1] = index;
            }
            st(flag(c, TP_F_RX_DONE), e);
        }
    }
}
