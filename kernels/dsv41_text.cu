// Bounded native1017 text-head helpers. No model, sampler, quantization or serving.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_TEXT_SOURCE_ID
#error Dedicated DSV41_TEXT_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_text_identity(uint64_t* output) {
    if(!blockIdx.x&&!threadIdx.x){output[0]=DSV41_TEXT_SOURCE_ID;output[1]=1;}
}
extern "C" __global__ void dsv41_text_last(
    const __nv_bfloat16* input,__nv_bfloat16* output,unsigned* status,unsigned rows,unsigned dim) {
    unsigned col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=dim)return;
    __nv_bfloat16 value=input[(uint64_t)(rows-1)*dim+col];
    if(!isfinite(__bfloat162float(value))){atomicOr(status,1u);value=__float2bfloat16_rn(0.f);}
    output[col]=value;
}
extern "C" __global__ void dsv41_text_maxloc(
    const float* logits,unsigned* pair,unsigned* status,unsigned count,unsigned first) {
    __shared__ float scores[256];
    __shared__ unsigned ids[256];
    unsigned tid=threadIdx.x;
    float best=-INFINITY;unsigned id=0xffffffffu;
    for(unsigned i=tid;i<count;i+=256){
        float value=logits[i];
        if(!isfinite(value)){atomicOr(status,1u);continue;}
        unsigned candidate=first+i;
        if(value>best||(value==best&&candidate<id)){best=value;id=candidate;}
    }
    scores[tid]=best;ids[tid]=id;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1){
        if(tid<stride){
            float value=scores[tid+stride];unsigned candidate=ids[tid+stride];
            if(value>scores[tid]||(value==scores[tid]&&candidate<ids[tid])){
                scores[tid]=value;ids[tid]=candidate;
            }
        }
        __syncthreads();
    }
    if(!tid){
        if(ids[0]==0xffffffffu)atomicOr(status,2u);
        pair[0]=__float_as_uint(scores[0]);pair[1]=ids[0];
    }
}
