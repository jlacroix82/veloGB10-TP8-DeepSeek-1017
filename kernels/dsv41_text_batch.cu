// Native1017 independent decode-lane greedy selection. No whole-model/sampler.
// Each CTA uses the qualified C1 reduction tree unchanged for its own row.
#include <stdint.h>
#include <math.h>
#ifndef DSV41_TEXT_BATCH_SOURCE_ID
#error Dedicated DSV41_TEXT_BATCH_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_text_batch_identity(uint64_t*out){
 if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_TEXT_BATCH_SOURCE_ID;out[1]=1;}
}
extern "C" __global__ void dsv41_text_batch_maxloc(
 const float*logits,unsigned*pair,unsigned*status,unsigned count,unsigned first){
 __shared__ float scores[256];__shared__ unsigned ids[256];
 unsigned tid=threadIdx.x,row=blockIdx.x;
 logits+=(uint64_t)row*count;pair+=(uint64_t)row*2;
 float best=-INFINITY;unsigned id=0xffffffffu;
 for(unsigned i=tid;i<count;i+=256){
  float value=logits[i];if(!isfinite(value)){atomicOr(status,1u);continue;}
  unsigned candidate=first+i;
  if(value>best||(value==best&&candidate<id)){best=value;id=candidate;}
 }
 scores[tid]=best;ids[tid]=id;__syncthreads();
 for(unsigned stride=128;stride;stride>>=1){
  if(tid<stride){
   float value=scores[tid+stride];unsigned candidate=ids[tid+stride];
   if(value>scores[tid]||(value==scores[tid]&&candidate<ids[tid])){scores[tid]=value;ids[tid]=candidate;}
  }__syncthreads();
 }
 if(!tid){if(ids[0]==0xffffffffu)atomicOr(status,2u);pair[0]=__float_as_uint(scores[0]);pair[1]=ids[0];}
}

