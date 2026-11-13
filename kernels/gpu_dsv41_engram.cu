// Derived from DeepSeek official1017 inference/model.py, MIT license.
// Isolated model-free Engram components; no projection GEMM or inference backend.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_ENGRAM_SOURCE_ID
#error Dedicated Engram source identity must be supplied by build.rs
#endif

extern "C" __global__ void dsv41_engram_identity(uint64_t* id) {
    if(blockIdx.x==0 && threadIdx.x==0) {id[0] = DSV41_ENGRAM_SOURCE_ID; id[1] = 1;}
}

__device__ float e4m3(uint8_t code) {
    const unsigned magnitude=code&127;
    const unsigned exponent=magnitude>>3, mantissa=magnitude&7;
    float value=exponent ? ldexpf(1.0f+float(mantissa)*0.125f,int(exponent)-7)
                         : float(mantissa)*0.001953125f;
    return code&128 ? -value : value;
}

__device__ float ue8(uint8_t code) {
    return __uint_as_float(code ? unsigned(code)<<23 : 1u<<22);
}

// Output is BF16-rounded then widened to F32 for a lossless one-owner collective.
// A rank with no ownership writes +0; invalid global IDs/NaNs set a sticky status.
extern "C" __global__ void dsv41_engram_lookup(
    const uint8_t* weights, const uint8_t* scales, const uint64_t* indices,
    float* out, uint32_t* status, uint64_t first, uint64_t rows,
    uint64_t total, unsigned count) {
    const unsigned flat=blockIdx.x*blockDim.x+threadIdx.x;
    if(flat>=count*256u) return;
    const uint64_t row=indices[flat/256u];
    const unsigned col=flat%256u;
    if(row>=total) { atomicOr(status,1u); out[flat]=0.0f; return; }
    if(row<first || row-first>=rows) { out[flat]=0.0f; return; }
    const uint64_t local=row-first;
    const uint8_t w=weights[local*256u+col], s=scales[local*8u+col/32u];
    if((w&127u)==127u || s==255u) { atomicOr(status,2u); out[flat]=0.0f; return; }
    const float value=e4m3(w)*ue8(s);
    const float rounded=__bfloat162float(__float2bfloat16_rn(value));
    if(!isfinite(value) || !isfinite(rounded)) atomicOr(status,4u);
    out[flat]=rounded;
}

__device__ float block_sum(float x, float* shared) {
    const unsigned tid=threadIdx.x;
    shared[tid]=x; __syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride/=2) {
        if(tid<stride) shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    return shared[0];
}

// Per-token/per-copy statistics, with exact source operation boundaries (-fmad=false).
extern "C" __global__ void dsv41_engram_gate(
    const __nv_bfloat16* h, const __nv_bfloat16* kv,
    const __nv_bfloat16* qw, const __nv_bfloat16* kw,
    const uint8_t* mask, __nv_bfloat16* out, float* gates,
    uint32_t* status, unsigned dim) {
    const unsigned token=blockIdx.x, copy=blockIdx.y, tid=threadIdx.x;
    const unsigned hbase=(token*4u+copy)*dim, kbase=(token*5u+copy)*dim;
    if(!mask[token]) {
        for(unsigned col=tid;col<dim;col+=blockDim.x) out[hbase+col]=h[hbase+col];
        if(tid==0) gates[token*4u+copy]=0.0f;
        return;
    }
    __shared__ float scratch[256];
    __shared__ float norm_h, norm_k, gate;
    __shared__ unsigned invalid_stats;
    if(tid==0) invalid_stats=0;
    __syncthreads();
    float h2=0.0f,k2=0.0f,dot=0.0f;
    for(unsigned col=tid;col<dim;col+=blockDim.x) {
        const float stream=__bfloat162float(h[hbase+col]);
        const float key=__bfloat162float(kv[kbase+col]);
        const float weight=__bfloat162float(qw[copy*dim+col])*__bfloat162float(kw[copy*dim+col]);
        if(!isfinite(stream)||!isfinite(key)||!isfinite(weight)) atomicOr(status,8u);
        h2+=stream*stream; k2+=key*key; dot+=(stream*weight)*key;
    }
    const float all_h2=block_sum(h2,scratch);
    if(tid==0) {
        if(!isfinite(all_h2)) invalid_stats=1;
        norm_h=rsqrtf(all_h2/float(dim)+1e-20f);
    }
    __syncthreads();
    const float all_k2=block_sum(k2,scratch);
    if(tid==0) {
        if(!isfinite(all_k2)) invalid_stats=1;
        norm_k=rsqrtf(all_k2/float(dim)+1e-20f);
    }
    __syncthreads();
    const float all_dot=block_sum(dot,scratch);
    if(tid==0) {
        const float normalized=(all_dot*(norm_h*norm_k))*rsqrtf(float(dim));
        if(invalid_stats || !isfinite(all_dot) || !isfinite(norm_h*norm_k) || !isfinite(normalized)) {
            atomicOr(status,8u); gate=NAN;
        } else {
            const float signed_root=copysignf(sqrtf(fmaxf(fabsf(normalized),1e-6f)),normalized);
            gate=1.0f/(1.0f+expf(-signed_root));
        }
        if(!isfinite(gate)) atomicOr(status,8u);
        gates[token*4u+copy]=gate;
    }
    __syncthreads();
    for(unsigned col=tid;col<dim;col+=blockDim.x) {
        const float value=__bfloat162float(kv[(token*5u+4u)*dim+col]);
        const float updated=__bfloat162float(h[hbase+col])+gate*value;
        const __nv_bfloat16 rounded=__float2bfloat16_rn(updated);
        if(!isfinite(value)||!isfinite(__bfloat162float(rounded))) atomicOr(status,8u);
        out[hbase+col]=rounded;
    }
}
