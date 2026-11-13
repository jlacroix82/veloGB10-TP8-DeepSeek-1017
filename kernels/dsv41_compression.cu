// Isolated1017 Compressor/Indexer arithmetic; no dense projection or network.
// Exact source model.py SHAe5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d.
// DeepSeek MIT attribution: src/dsv41/numerics.rs.
// sm_121 --ftz=false --fmad=false, dedicated source stamp.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_COMPRESSION_SOURCE_ID
#error DedicatedDSV41_COMPRESSION_SOURCE_ID required
#endif
namespace{
constexpr unsigned BAD_INPUT=1,OVERFLOW=2;
__device__ void bad(unsigned*status,unsigned flag){atomicOr(status,flag);}
__device__ float finite(float x,unsigned*status){
    if(!isfinite(x)){bad(status,BAD_INPUT);return 0.f;}return x;
}
__device__ __nv_bfloat16 rounded(float x,unsigned*status){
    __nv_bfloat16 b=__float2bfloat16_rn(x);
    if(!isfinite(x)||!isfinite(__bfloat162float(b))){bad(status,OVERFLOW);return __float2bfloat16_rn(0.f);}return b;
}
__device__ float pool(float a,float b,float sa,float sb,unsigned*status){
    a=finite(a,status);b=finite(b,status);sa=finite(sa,status);sb=finite(sb,status);
    float maximum=fmaxf(sa,sb);
    float pa=expf(__fsub_rn(sa,maximum)),pb=expf(__fsub_rn(sb,maximum));
    float denominator=__fadd_rn(pa,pb);
    float value=__fadd_rn(__fmul_rn(a,__fdiv_rn(pa,denominator)),__fmul_rn(b,__fdiv_rn(pb,denominator)));
    if(!isfinite(value)){bad(status,OVERFLOW);return 0.f;}return value;
}
}
extern "C" __global__ void dsv41_compression_identity(unsigned long long*out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_COMPRESSION_SOURCE_ID;out[1]=1;}
}
extern "C" __global__ void dsv41_compression_reset(float*kv,float*score,unsigned*status,unsigned dim,unsigned lane){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<2u*dim){kv[lane*2u*dim+i]=0.f;score[lane*2u*dim+i]=-INFINITY;}
    if(i==0)status[0]=0;
}
extern "C" __global__ void dsv41_compression_copy_bf16(
    const __nv_bfloat16*input,__nv_bfloat16*out,unsigned*status,uint64_t count){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float x=__bfloat162float(input[i]);out[i]=__float2bfloat16_rn(finite(x,status));
}
// Initial prefill source: consecutive pairs pool independently; odd final row
// is retained unrounded in F32slot0. Norm gets exactly one BF16 cast AFTERpool.
extern "C" __global__ void dsv41_compression_prefill2(
    const float*kv,const float*score,float*state_kv,float*state_score,
    float*pooled,__nv_bfloat16*norm_input,unsigned*status,unsigned tokens,unsigned dim,unsigned lane){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,rows=tokens/2;
    if(i<rows*dim){
        unsigned row=i/dim,c=i%dim,a=row*2u*dim+c,b=a+dim;
        float value=pool(kv[a],kv[b],score[a],score[b],status);
        pooled[i]=value;norm_input[i]=rounded(value,status);
    }
    if((tokens&1u)&&i<dim){
        unsigned source=(tokens-1u)*dim+i,target=lane*2u*dim+i;
        state_kv[target]=finite(kv[source],status);state_score[target]=finite(score[source],status);
    }
}
// Positive start has exactly one projected row. Slot1 completes a pair and uses
// its new value directly, so no cross-thread/block state-write barrier is needed.
extern "C" __global__ void dsv41_compression_append2(
    const float*kv,const float*score,float*state_kv,float*state_score,
    float*pooled,__nv_bfloat16*norm_input,unsigned*status,unsigned dim,unsigned lane,unsigned slot){
    unsigned c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=dim)return;
    float value=finite(kv[c],status),s=finite(score[c],status);
    unsigned base=lane*2u*dim;state_kv[base+slot*dim+c]=value;state_score[base+slot*dim+c]=s;
    if(slot==1){
        float answer=pool(state_kv[base+c],value,state_score[base+c],s,status);
        pooled[c]=answer;norm_input[c]=rounded(answer,status);
    }
}
// Supplied q/key are already RoPE'd and cacheQAT'd BF16. This preserves EVERY
// official local dtype boundary: BF16dot, ReLU, scaledBF16weights, BF16product,
// F32headsum followed byBF16. Widened F32output is a copy of that BF16local,
// not an unrounded alternative local computation. Network dtype is caller policy.
extern "C" __global__ void dsv41_index_scores_source(
    const __nv_bfloat16*q,const __nv_bfloat16*key,const __nv_bfloat16*projected_weights,
    __nv_bfloat16*dot_out,__nv_bfloat16*relu_out,__nv_bfloat16*product_out,
    __nv_bfloat16*scaled_weights,__nv_bfloat16*local_bf16,float*local_promoted_f32,
    unsigned*status,unsigned queries,unsigned keys){
    unsigned index=blockIdx.x,k=index%keys,query=(index/keys)%queries,b=index/(keys*queries);
    unsigned lane=threadIdx.x&31u,head=threadIdx.x>>5;
    __shared__ float products[4];
    float sum=0.f;
    for(unsigned d=lane;d<128;d+=32){
        float a=finite(__bfloat162float(q[((uint64_t)b*queries+query)*4u*128u+head*128u+d]),status);
        float w=finite(__bfloat162float(key[((uint64_t)b*keys+k)*128u+d]),status);
        sum=__fadd_rn(sum,__fmul_rn(a,w));
    }
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,delta));
    if(lane==0){
        __nv_bfloat16 dot=rounded(sum,status);
        float r=__bfloat162float(dot); // Preserve ReLU's negativezero when value==0.
        __nv_bfloat16 relu=r<0.f?__float2bfloat16_rn(0.f):dot;
        unsigned wi=(b*queries+query)*4u+head;
        float wp=finite(__bfloat162float(projected_weights[wi]),status);
        __nv_bfloat16 scaled=rounded(__fmul_rn(wp,0x1p-6f),status);
        __nv_bfloat16 product=rounded(__fmul_rn(__bfloat162float(relu),__bfloat162float(scaled)),status);
        uint64_t out=((uint64_t)b*queries+query)*4u*keys+head*keys+k;
        dot_out[out]=dot;relu_out[out]=relu;product_out[out]=product;
        if(k==0)scaled_weights[wi]=scaled;
        products[head]=__bfloat162float(product);
    }
    __syncthreads();
    if(threadIdx.x==0){
        float total=0.f;
        #pragma unroll
        for(unsigned h=0;h<4;h++)total=__fadd_rn(total,products[h]);
        __nv_bfloat16 local=rounded(total,status);
        local_bf16[index]=local;local_promoted_f32[index]=__bfloat162float(local);
    }
}

