// New BF16 packing/validation helpers only. The reduction is the unchanged
// gpu_batch.cu tp_wait_add_g(mode0), with BF16 RNE after every XOR exchange.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include "tp_doorbell.h"
#ifndef DSV41_BF16_COLLECTIVES_SOURCE_ID
#error Dedicated DSV41_BF16_COLLECTIVES_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_bf16_collectives_identity(uint64_t*out){
 if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_BF16_COLLECTIVES_SOURCE_ID;out[1]=1;}
}
namespace {
__device__ __forceinline__ unsigned long long* abort_flag(tp_dev_ctx*c){return(unsigned long long*)((char*)c->flags+TP_F_ABORT);}
__device__ __forceinline__ bool stopped(tp_dev_ctx*c){unsigned long long v;asm volatile("ld.relaxed.sys.b64 %0,[%1];":"=l"(v):"l"(abort_flag(c)):"memory");return v!=0;}
__device__ __forceinline__ void fail(tp_dev_ctx*c,unsigned*local,unsigned*global,unsigned bit){
 atomicOr(local,bit);atomicOr(global,bit);asm volatile("st.release.sys.b64 [%0],%1;"::"l"(abort_flag(c)),"l"(13ull):"memory");
}
__device__ __forceinline__ bool geometry(tp_dev_ctx*c,unsigned logical,unsigned padded){
 return c->world==8&&c->rounds==3&&c->rank>=0&&c->rank<8&&c->oneshot==0&&c->nway_flags&&c->nway_recv
  &&logical>0&&logical<=4194304u&&padded==((logical+3u)/4u)*4u;
}
}
extern "C" __global__ void dsv41_bf16_pack(tp_dev_ctx*c,const __nv_bfloat16*input,__nv_bfloat16*scratch,
 unsigned logical,unsigned padded,unsigned*status,unsigned*global){
 if(stopped(c))return;
 if(!geometry(c,logical,padded)){if(!blockIdx.x&&!threadIdx.x)fail(c,status,global,4u);return;}
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded)return;
 __nv_bfloat16 value=__float2bfloat16_rn(0.f);
 if(i<logical){value=input[i];if(!isfinite(__bfloat162float(value))){fail(c,status,global,1u);value=__float2bfloat16_rn(0.f);}}
 scratch[i]=value;
}
extern "C" __global__ void dsv41_bf16_check(tp_dev_ctx*c,const __nv_bfloat16*scratch,
 unsigned logical,unsigned padded,unsigned first,unsigned count,unsigned*status,unsigned*global){
 if(stopped(c))return;
 if(!geometry(c,logical,padded)||first>padded||count>padded-first){if(!blockIdx.x&&!threadIdx.x)fail(c,status,global,4u);return;}
 unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;unsigned i=first+j;
 __nv_bfloat16 value=scratch[i];if(!isfinite(__bfloat162float(value)))fail(c,status,global,1u);
 if(i>=logical&&__bfloat16_as_ushort(value)!=0)fail(c,status,global,2u);
}
extern "C" __global__ void dsv41_bf16_unpack(tp_dev_ctx*c,const __nv_bfloat16*scratch,__nv_bfloat16*output,
 unsigned logical,unsigned padded,unsigned*status,unsigned*global){
 if(stopped(c))return;
 if(!geometry(c,logical,padded)){if(!blockIdx.x&&!threadIdx.x)fail(c,status,global,4u);return;}
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<logical)output[i]=scratch[i];
}
