// Adapted from veloGB10 gpu_dsv4.cu (MIT), preserving block64 online
// softmax and BF16 probability rounding. Explicit local heads and lane stride.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_ATTENTION_SOURCE_ID
#error Dedicated attention source identity required
#endif
extern "C" __global__ void dsv41_attention_identity(uint64_t* id) {
    if(blockIdx.x==0&&threadIdx.x==0) {id[0]=DSV41_ATTENTION_SOURCE_ID;id[1]=1;}
}
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) {return __bfloat162float(x);}
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {return __float2bfloat16_rn(x);}
#define DSV41_FULL_MASK 0xffffffffu
#define DSV41_GA_HB 16
#define DSV41_GA_KB 64
#define DSV41_GA_D 512
extern "C" __global__ void __launch_bounds__(256, 1)
dsv41_sparse_attention(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                 __nv_bfloat16* __restrict__ o, const float* __restrict__ attn_sink,
                 const int* __restrict__ topk_idxs, int topk, int n, int heads, float scale, uint32_t* status) {
    extern __shared__ __align__(16) uint8_t dsmem[];
    __nv_bfloat16* q_s = (__nv_bfloat16*)dsmem;                                   // [HB][D]
    __nv_bfloat16* kv_s = (__nv_bfloat16*)(dsmem + DSV41_GA_HB * DSV41_GA_D * 2);   // [KB][D]
    float* sc_s = (float*)(kv_s + DSV41_GA_KB * DSV41_GA_D);                        // [HB][KB]
    __nv_bfloat16* pc_s = (__nv_bfloat16*)(sc_s + DSV41_GA_HB * DSV41_GA_KB);       // [HB][KB]
    int* idx_s = (int*)(pc_s + DSV41_GA_HB * DSV41_GA_KB);                          // [KB]

    const int bx = blockIdx.x;              // query row within batch
    const int by = blockIdx.y;              // batch
    const int hb = blockIdx.z;              // head block (16 heads)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m_queries = gridDim.x;
    __shared__ int has_valid;
    if(tid==0) has_valid=0;
    __syncthreads();

    // load q [HB][D] (contiguous)
    const size_t q_base = (((size_t)by * m_queries + bx) * heads + hb * DSV41_GA_HB) * DSV41_GA_D;
    for (int i = tid; i < DSV41_GA_HB * DSV41_GA_D; i += 256) {
        q_s[i] = (hb*DSV41_GA_HB+i/DSV41_GA_D<heads) ? q[q_base+i] : to_bf16(0.0f);
        if(!isfinite(to_f32(q_s[i]))) atomicOr(status,2u);
    }

    const int h0 = 2 * warp, h1 = 2 * warp + 1;  // this warp's heads (local)
    float m_run0 = -1e30f, m_run1 = -1e30f;
    float l0 = 0.0f, l1 = 0.0f;
    float acc0[16], acc1[16];  // lane owns dims lane*16 .. lane*16+15 of each of its heads
#pragma unroll
    for (int dd = 0; dd < 16; ++dd) { acc0[dd] = 0.0f; acc1[dd] = 0.0f; }

    const int n_tiles = (topk + DSV41_GA_KB - 1) / DSV41_GA_KB;
    const int* idx_row = topk_idxs + ((size_t)by * m_queries + bx) * (size_t)topk;

    for (int t = 0; t < n_tiles; ++t) {
        __syncthreads();  // protect smem overwrite vs previous iteration's reads
        if (tid < DSV41_GA_KB) {
            int g = t * DSV41_GA_KB + tid;
            int selected=(g < topk) ? idx_row[g] : -1;
            if(selected < -1 || selected >= n) {atomicOr(status,1u);selected=-1;}
            idx_s[tid]=selected;
            if(selected>=0) atomicExch(&has_valid,1);
        }
        __syncthreads();
        // gather kv tile: each row read once, all 16 heads reuse from smem
        for (int i = tid; i < DSV41_GA_KB * DSV41_GA_D; i += 256) {
            int row = i >> 9, col = i & 511;
            int idx = idx_s[row];
            kv_s[i] = (idx >= 0) ? kv[((size_t)by * n + idx) * DSV41_GA_D + col]
                                 : __nv_bfloat16(0.0f);
            if(!isfinite(to_f32(kv_s[i]))) atomicOr(status,2u);
        }
        __syncthreads();

        // scores: warp-dot per (head, row) cell; masked rows -> -inf
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV41_GA_KB; ++j) {
                float part = 0.0f;
#pragma unroll
                for (int dd = 0; dd < 16; ++dd)
                    part = fmaf(to_f32(q_s[h * DSV41_GA_D + lane * 16 + dd]),
                                to_f32(kv_s[j * DSV41_GA_D + lane * 16 + dd]), part);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    part += __shfl_xor_sync(DSV41_FULL_MASK, part, off);
                float sc = part * scale;
                if (idx_s[j] < 0) sc = -INFINITY;
                else if(!isfinite(sc)) {atomicOr(status,2u);sc=-INFINITY;}
                if (lane == 0) sc_s[h * DSV41_GA_KB + j] = sc;
            }
        }
        __syncthreads();

        // per head: tile max, online-softmax rescale (applied to acc below), p -> bf16,
        // fp32 sum of UNROUNDED p (reference reduces the fp32 fragment)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            float m_run = hh == 0 ? m_run0 : m_run1;
            float m_tile = fmaxf(sc_s[h * DSV41_GA_KB + lane], sc_s[h * DSV41_GA_KB + lane + 32]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(DSV41_FULL_MASK, m_tile, off));
            float m_new = fmaxf(m_run, m_tile);
            float rescale = expf(m_run - m_new);
            float p_a = expf(sc_s[h * DSV41_GA_KB + lane] - m_new);
            float p_b = expf(sc_s[h * DSV41_GA_KB + lane + 32] - m_new);
            pc_s[h * DSV41_GA_KB + lane] = to_bf16(p_a);
            pc_s[h * DSV41_GA_KB + lane + 32] = to_bf16(p_b);
            float tsum = p_a + p_b;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                tsum += __shfl_xor_sync(DSV41_FULL_MASK, tsum, off);
            // acc *= rescale (compile-time array selection: hh is unrolled)
#pragma unroll
            for (int dd = 0; dd < 16; ++dd) {
                if (hh == 0) acc0[dd] *= rescale; else acc1[dd] *= rescale;
            }
            if (hh == 0) { m_run0 = m_new; l0 = l0 * rescale + tsum; }
            else         { m_run1 = m_new; l1 = l1 * rescale + tsum; }
        }
        __syncthreads();

        // acc += P_bf16 · V (ascending-j accumulation, fp32)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV41_GA_KB; ++j) {
                float p = to_f32(pc_s[h * DSV41_GA_KB + j]);
#pragma unroll
                for (int dd = 0; dd < 16; ++dd) {
                    float kv = to_f32(kv_s[j * DSV41_GA_D + lane * 16 + dd]);
                    if (hh == 0) acc0[dd] = fmaf(p, kv, acc0[dd]);
                    else         acc1[dd] = fmaf(p, kv, acc1[dd]);
                }
            }
        }
    }

    // epilogue: denominator-only sink, o = acc / sum_exp -> bf16
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int hg = hb * DSV41_GA_HB + (hh == 0 ? h0 : h1);
        float m = hh == 0 ? m_run0 : m_run1;
        float l = (hh == 0 ? l0 : l1) + expf((hg<heads ? attn_sink[hg] : 0.0f) - m);
        const size_t o_base = (((size_t)by * m_queries + bx) * heads + hg) * DSV41_GA_D;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            float a = hh == 0 ? acc0[dd] : acc1[dd];
            if(hg<heads) {
                if(!isfinite(attn_sink[hg])) atomicOr(status,2u);
                const float value=has_valid ? a/l : 0.0f;
                const __nv_bfloat16 rounded=to_bf16(value);
                if(!isfinite(value)||!isfinite(to_f32(rounded))) atomicOr(status,2u);
                o[o_base+lane*16+dd]=rounded;
            }
        }
    }
}
