// Exact1017 short-context BF16 cache/snapshot operations, no model/index scores.
// Values have ALREADY passed source RoPE+windowFP8/compressedFP4 QAT.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_KV_STORE_SOURCE_ID
#error Dedicated DSV41_KV_STORE_SOURCE_ID required
#endif
namespace {
constexpr uint64_t D=512,W=128,L=40,S=4,C=512;
__device__ __forceinline__ __nv_bfloat16 checked(__nv_bfloat16 v,unsigned*status){
    if(!isfinite(__bfloat162float(v))){atomicOr(status,1u);return __float2bfloat16_rn(0.f);}return v;
}
}
extern "C" __global__ void dsv41_kv_store_identity(uint64_t*out){if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_KV_STORE_SOURCE_ID;out[1]=1;}}
extern "C" __global__ void dsv41_kv_store_reset(__nv_bfloat16*windows,__nv_bfloat16*compressed,unsigned*status,uint64_t lane){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<L*W*D)windows[lane*L*W*D+i]=__float2bfloat16_rn(0.f);
    if(i<S*C*D)compressed[lane*S*C*D+i]=__float2bfloat16_rn(0.f);
    if(!i)*status=0;
}
extern "C" __global__ void dsv41_kv_store_window(
    const __nv_bfloat16*input,__nv_bfloat16*windows,unsigned*status,uint64_t lane,uint64_t layer,uint64_t start,uint64_t tokens){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*D)return;
    uint64_t row=i/D,col=i%D,slot=(start+row)%W;
    windows[((lane*L+layer)*W+slot)*D+col]=checked(input[i],status);
}
extern "C" __global__ void dsv41_kv_store_compressed(
    const __nv_bfloat16*input,__nv_bfloat16*cache,unsigned*status,uint64_t lane,uint64_t owner_slot,uint64_t first,uint64_t rows){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*D)return;
    cache[((lane*S+owner_slot)*C+first)*D+i]=checked(input[i],status);
}
extern "C" __global__ void dsv41_kv_store_snapshot(
    __nv_bfloat16*out,const __nv_bfloat16*current,const __nv_bfloat16*windows,const __nv_bfloat16*compressed,
    unsigned*status,uint64_t lane,uint64_t layer,uint64_t start,uint64_t window_rows,uint64_t compressed_rows,uint64_t owner_slot){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(window_rows+compressed_rows)*D)return;
    uint64_t row=i/D,col=i%D;
    __nv_bfloat16 value;
    if(row<window_rows)value=start==0?current[i]:windows[((lane*L+layer)*W+row)*D+col];
    else value=compressed[((lane*S+owner_slot)*C+row-window_rows)*D+col];
    out[i]=checked(value,status);
}
extern "C" __global__ void dsv41_kv_store_indices(
    int*out,uint64_t start,uint64_t tokens,uint64_t ratio,uint64_t window_rows,uint64_t window_topk,uint64_t compressed_rows){
    uint64_t topk=window_topk+compressed_rows,i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*topk)return;
    uint64_t query=i/topk,j=i%topk;
    if(j<window_topk){
        if(start==0)out[i]=j<=query?(int)j:-1;
        else{uint64_t slot=(start%W+1+j)%W;out[i]=slot<=start?(int)slot:-1;}
    }else{
        uint64_t c=j-window_topk,visible=start==0?(query+1)/ratio:(start+tokens)/ratio;
        out[i]=c<visible?(int)(window_rows+c):-1;
    }
}
