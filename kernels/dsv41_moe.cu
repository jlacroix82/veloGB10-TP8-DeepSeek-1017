// Isolated official1017 Gate and pre-w2 SwiGLU, no dense/full-expert/model path.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_MOE_SOURCE_ID
#error Dedicated DSV41_MOE_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_moe_identity(uint64_t* out){if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_MOE_SOURCE_ID;out[1]=1;}}
// Fixed384/top6, temperature1, sqrtsoftplus; explicit lowest-ID tie policy.
extern "C" __global__ void dsv41_moe_route(const float* logits,const float* bias,uint32_t* ids,float* weights,uint32_t* status){
 unsigned row=blockIdx.x,t=threadIdx.x;
 __shared__ float score[384],biased[384];
 for(unsigned i=t;i<384;i+=blockDim.x){float x=logits[row*384u+i],b=bias[i];
  if(!isfinite(x)||!isfinite(b)){atomicOr(status,1u);x=0.0f;b=0.0f;}
  float s=sqrtf(x>20.0f?x:log1pf(expf(x)));float key=s+b;
  if(!isfinite(s)||!isfinite(key)){atomicOr(status,2u);s=0.0f;key=0.0f;}
  score[i]=s;biased[i]=key;
 }
 __syncthreads();
 if(t==0){unsigned chosen[6];float total=0.0f;
  for(unsigned n=0;n<6;n++){float best=-INFINITY;unsigned id=0;
   for(unsigned i=0;i<384;i++){bool used=false;for(unsigned j=0;j<n;j++)used|=chosen[j]==i;
    if(!used&&biased[i]>best){best=biased[i];id=i;}}
   chosen[n]=id;total+=score[id];ids[row*6u+n]=id;
  }
  if(!isfinite(total))atomicOr(status,2u);
  for(unsigned n=0;n<6;n++){float w=(score[chosen[n]]/(total+1e-20f))*1.5f;
   if(!isfinite(w))atomicOr(status,2u);weights[row*6u+n]=w;}
 }
}
// Input projections are BF16 Linear outputs. F32 clamp/silu/product/route-weight
// precede the single BF16 cast passed to w2. Shared expert sets weighted=0.
extern "C" __global__ void dsv41_moe_swiglu(const __nv_bfloat16* gate,const __nv_bfloat16* up,const float* weights,__nv_bfloat16* out,uint32_t* status,unsigned rows,unsigned columns,unsigned weighted){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*columns)return;
 float g=__bfloat162float(gate[i]),u=__bfloat162float(up[i]);float w=weighted?weights[i/columns]:1.0f;
 if(!isfinite(g)||!isfinite(u)||!isfinite(w)||w<0.0f||w>1.5f){atomicOr(status,1u);out[i]=__float2bfloat16_rn(0.0f);return;}
 g=fminf(g,10.0f);u=fminf(10.0f,fmaxf(-10.0f,u));
 float silu=g/(1.0f+expf(-g));float value=silu*u;if(weighted)value*=w;
 __nv_bfloat16 result=__float2bfloat16_rn(value);if(!isfinite(value)||!isfinite(__bfloat162float(result)))atomicOr(status,2u);out[i]=result;
}
