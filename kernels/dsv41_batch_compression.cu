// PRIVATE vectorized native1017 per-channel ratio2 pool. Dense projections,
// BF16-before-RMS and normalization use separately qualified components.
// Pool scalar arithmetic is copied unchanged from dsv41_compression.cu (MIT).
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_BATCH_COMPRESSION_SOURCE_ID
#error Dedicated DSV41_BATCH_COMPRESSION_SOURCE_ID required
#endif
namespace{
__device__ float finite_value(float x,unsigned*status){if(!isfinite(x)){atomicOr(status,1u);return 0.f;}return x;}
__device__ __nv_bfloat16 rounded(float x,unsigned*status){__nv_bfloat16 b=__float2bfloat16_rn(x);if(!isfinite(x)||!isfinite(__bfloat162float(b))){atomicOr(status,2u);return __float2bfloat16_rn(0.f);}return b;}
__device__ float pool(float a,float b,float sa,float sb,unsigned*status){
 a=finite_value(a,status);b=finite_value(b,status);sa=finite_value(sa,status);sb=finite_value(sb,status);
 float maximum=fmaxf(sa,sb);float pa=expf(__fsub_rn(sa,maximum)),pb=expf(__fsub_rn(sb,maximum));
 float denominator=__fadd_rn(pa,pb);
 float value=__fadd_rn(__fmul_rn(a,__fdiv_rn(pa,denominator)),__fmul_rn(b,__fdiv_rn(pb,denominator)));
 if(!isfinite(value)){atomicOr(status,2u);return 0.f;}return value;
}
}
extern "C" __global__ void dsv41_batch_compression_identity(uint64_t*out){if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_BATCH_COMPRESSION_SOURCE_ID;out[1]=1;}}
extern "C" __global__ void dsv41_batch_compression_reset(float*kv,float*score,unsigned*status,uint64_t lane,uint64_t lanes){
 if(!lanes||lanes>16||lane>=lanes){if(!threadIdx.x&&!blockIdx.x)atomicOr(status,4u);return;}
 uint64_t d=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(d>=512)return;
 kv[lane*512+d]=0.f;score[lane*512+d]=-INFINITY;
}
// descriptors[group]={physical_lane, first_flat_row_or_UINT64_MAX, second_row}.
// An old tail is read ONLY; a subsequent ordered launch installs new tails.
extern "C" __global__ void dsv41_batch_compression_pool2(
 const float*kv,const float*score,const float*tail_kv,const float*tail_score,
 const uint64_t*descriptors,__nv_bfloat16*norm_input,unsigned*status,
 uint64_t input_rows,uint64_t groups,uint64_t lanes){
 if(!input_rows||input_rows>128||!groups||groups>128||!lanes||lanes>16){if(!threadIdx.x&&!blockIdx.x)atomicOr(status,4u);return;}
 uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*512)return;
 uint64_t group=i/512,d=i%512,lane=descriptors[group*3],a=descriptors[group*3+1],b=descriptors[group*3+2];
 if(lane>=lanes||b>=input_rows||(a!=UINT64_MAX&&a>=input_rows)){atomicOr(status,4u);norm_input[i]=__float2bfloat16_rn(0.f);return;}
 float av=a==UINT64_MAX?tail_kv[lane*512+d]:kv[a*512+d];
 float as=a==UINT64_MAX?tail_score[lane*512+d]:score[a*512+d];
 norm_input[i]=rounded(pool(av,kv[b*512+d],as,score[b*512+d],status),status);
}
// descriptors[tail]={physical_lane,last_flat_row}. Must be queued AFTER pool2
// on the same retained stream, including a chunk that both reads and writes
// its lane's tail. No cross-CTA old-tail overwrite race is permitted.
extern "C" __global__ void dsv41_batch_compression_tail2(
 const float*kv,const float*score,float*tail_kv,float*tail_score,
 const uint64_t*descriptors,unsigned*status,uint64_t input_rows,uint64_t tails,uint64_t lanes){
 if(!input_rows||input_rows>128||!tails||tails>16||!lanes||lanes>16){if(!threadIdx.x&&!blockIdx.x)atomicOr(status,4u);return;}
 uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=tails*512)return;
 uint64_t t=i/512,d=i%512,lane=descriptors[t*2],row=descriptors[t*2+1];
 if(lane>=lanes||row>=input_rows){atomicOr(status,4u);return;}
 tail_kv[lane*512+d]=finite_value(kv[row*512+d],status);tail_score[lane*512+d]=finite_value(score[row*512+d],status);
}
