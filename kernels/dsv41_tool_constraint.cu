// Additive constrained-greedy local maxloc. Legacy text/PTX/collective unchanged.
#include <stdint.h>
#include <float.h>
#include <math.h>
#ifndef DSV41_TOOL_CONSTRAINT_SOURCE_ID
#error Dedicated DSV41_TOOL_CONSTRAINT_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_tool_constraint_identity(uint64_t*out){if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_TOOL_CONSTRAINT_SOURCE_ID;out[1]=1;}}
extern "C" __global__ void dsv41_tool_masked_maxloc(
 const float*logits,const uint32_t*allowed,uint32_t*pair,uint32_t*status,
 unsigned count,unsigned first,unsigned words){
 __shared__ float scores[256];__shared__ unsigned ids[256];
 const unsigned t=threadIdx.x;
 if(blockDim.x!=256||gridDim.x!=1||count!=16160||first%16160||first>=129280||words!=4040){if(!t)atomicOr(status,4u);return;}
 // Empty local shards participate with a FINITE bottom identity. Every allowed
 // score on every rank is checked strictly above it, so this identity cannot
 // win. No model-bound assumption or -Inf is hidden in the old finite transport.
 float best=-FLT_MAX;unsigned id=first;
 for(unsigned i=t;i<count;i+=256){float x=logits[i];unsigned global=first+i;
  if(!isfinite(x)){atomicOr(status,1u);continue;}
  if(!(allowed[global/32]&(1u<<(global%32))))continue;
  if(!(x>-FLT_MAX)){atomicOr(status,2u);continue;}
  if(x>best||(x==best&&global<id)){best=x;id=global;}
 }
 scores[t]=best;ids[t]=id;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s){float x=scores[t+s];unsigned candidate=ids[t+s];if(x>scores[t]||(x==scores[t]&&candidate<ids[t])){scores[t]=x;ids[t]=candidate;}}__syncthreads();}
 if(!t){pair[0]=__float_as_uint(scores[0]);pair[1]=ids[0];}
}
