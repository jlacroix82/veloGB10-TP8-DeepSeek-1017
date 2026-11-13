// Isolated1017 adjacent-pair RoPE and basic BF16 state operations; no model.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_POSITION_SOURCE_ID
#error Dedicated DSV41_POSITION_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_position_identity(uint64_t* out){if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_POSITION_SOURCE_ID;out[1]=1;}}
extern "C" __global__ void dsv41_position_rope(__nv_bfloat16* x,const float* coeff,uint32_t* status,unsigned rows,unsigned heads,unsigned width,unsigned inverse){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*32u)return;
 unsigned pair=i%32u,head=(i/32u)%heads,row=i/(heads*32u),base=(row*heads+head)*width;
 for(unsigned col=pair;col<width-64u;col+=32u)if(!isfinite(__bfloat162float(x[base+col])))atomicOr(status,1u);
 float c=coeff[row*64u+2u*pair],s=coeff[row*64u+2u*pair+1u];if(inverse)s=-s;
 unsigned p=base+width-64u+2u*pair;float a=__bfloat162float(x[p]),b=__bfloat162float(x[p+1]);
 if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(s))atomicOr(status,1u);
 // Official TorchCUDA complex multiply: explicit contractions are semantic,
 // even when the dedicated target uses --fmad=false for all other expressions.
 float real=__fmaf_rn(a,c,-__fmul_rn(b,s)),imag=__fmaf_rn(b,c,__fmul_rn(a,s));__nv_bfloat16 r=__float2bfloat16_rn(real),j=__float2bfloat16_rn(imag);
 if(!isfinite(real)||!isfinite(imag)||!isfinite(__bfloat162float(r))||!isfinite(__bfloat162float(j)))atomicOr(status,2u);x[p]=r;x[p+1]=j;
}
extern "C" __global__ void dsv41_position_embedding(const __nv_bfloat16* weights,const uint32_t* ids,float* out,uint32_t* status,unsigned tokens,unsigned dim,unsigned rank,unsigned first,unsigned loaded_rows){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*dim)return;unsigned id=ids[i/dim],col=i%dim;
 if(id>=129280u){atomicOr(status,1u);out[i]=0.0f;return;}
 if(id<rank*16160u||id>=(rank+1u)*16160u){out[i]=0.0f;return;}
 if(id<first||id-first>=loaded_rows){atomicOr(status,4u);out[i]=0.0f;return;}
 float value=__bfloat162float(weights[(id-first)*dim+col]);if(!isfinite(value))atomicOr(status,1u);out[i]=value;
}
extern "C" __global__ void dsv41_position_bf16_f32(const __nv_bfloat16* input,float* out,uint32_t* status,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=__bfloat162float(input[i]);if(!isfinite(v))atomicOr(status,1u);out[i]=v;}}
extern "C" __global__ void dsv41_position_f32_bf16(const float* input,__nv_bfloat16* out,uint32_t* status,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=input[i];__nv_bfloat16 r=__float2bfloat16_rn(v);if(!isfinite(v))atomicOr(status,1u);if(!isfinite(__bfloat162float(r)))atomicOr(status,2u);out[i]=r;}}
extern "C" __global__ void dsv41_position_expand(const float* input,__nv_bfloat16* out,uint32_t* status,unsigned tokens,unsigned dim){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<tokens*4u*dim){unsigned row=i/(4u*dim),col=i%dim;float v=input[row*dim+col];__nv_bfloat16 r=__float2bfloat16_rn(v);if(!isfinite(v))atomicOr(status,1u);if(!isfinite(__bfloat162float(r)))atomicOr(status,2u);out[i]=r;}}
// Descriptor[lane,start,count,inputrow]. Exactly one descriptor per cache lane;
// filtering to the last128 prefill rows prevents modulo-slot duplicate writers.
extern "C" __global__ void dsv41_position_ring(const __nv_bfloat16* input,__nv_bfloat16* cache,const uint64_t* descriptors,uint32_t* status,unsigned lanes,unsigned input_rows){
 unsigned j=blockIdx.x,w=blockIdx.y,col=threadIdx.x;const uint64_t* d=descriptors+4u*w;
 uint64_t lane=d[0],start=d[1],count=d[2],source=d[3];
 if(lane>=lanes||!count||count>4096u||source+count>input_rows||start+count>1048576u||(start&&count!=1u)){if(col==0)atomicOr(status,4u);return;}
 if(j>=count||(count>128u&&j<count-128u))return;
 unsigned slot=unsigned((start+j)%128u);float value=__bfloat162float(input[(source+j)*512u+col]);
 if(!isfinite(value))atomicOr(status,1u);cache[(lane*128u+slot)*512u+col]=__float2bfloat16_rn(value);
}
