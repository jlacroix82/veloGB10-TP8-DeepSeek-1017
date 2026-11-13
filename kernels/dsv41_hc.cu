// Isolated official1017 four-stream mHC components; MIT reference model.py/kernel.py.
// No model, attention, MoE, quantization, collectives, or serving implementation.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_HC_SOURCE_ID
#error Dedicated DSV41_HC_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_hc_identity(uint64_t* out) {
    if(blockIdx.x==0 && threadIdx.x==0) {out[0]=DSV41_HC_SOURCE_ID; out[1]=1;}
}
__device__ float hc_sum(float x,float* scratch) {
    unsigned t=threadIdx.x;scratch[t]=x;__syncthreads();
    for(unsigned n=blockDim.x/2;n;n/=2) {if(t<n)scratch[t]+=scratch[t+n];__syncthreads();}
    return scratch[0];
}
// RMS covers4*dim together; projection weights and accumulators stay F32.
extern "C" __global__ void dsv41_hc_project(
    const __nv_bfloat16* h,const float* weight,float* mixes,uint32_t* status,unsigned dim) {
    unsigned token=blockIdx.x,row=blockIdx.y,t=threadIdx.x,n=4u*dim;
    __shared__ float scratch[256];__shared__ float inverse;
    float square=0.0f,dot=0.0f;
    for(unsigned c=t;c<n;c+=blockDim.x) {
        float x=__bfloat162float(h[token*n+c]),w=weight[row*n+c];
        if(!isfinite(x)||!isfinite(w))atomicOr(status,1u);
        square+=x*x;dot+=x*w;
    }
    float all_square=hc_sum(square,scratch);
    if(t==0) {
        if(!isfinite(all_square))atomicOr(status,2u);
        inverse=1.0f/sqrtf(all_square/float(n)+1e-20f);
    }
    __syncthreads();
    float all_dot=hc_sum(dot,scratch);
    if(t==0) {
        float value=all_dot*inverse;
        if(!isfinite(value))atomicOr(status,2u);
        mixes[token*24u+row]=value;
    }
}
// One independent thread per token. Literal stable softmax +20 Sinkhorn rounds.
extern "C" __global__ void dsv41_hc_split(
    const float* mixes,const float* scale,const float* base,float* pre,float* post,float* comb,
    uint32_t* status,unsigned tokens) {
    unsigned token=blockIdx.x*blockDim.x+threadIdx.x;if(token>=tokens)return;
    float m[24];
    for(unsigned i=0;i<24;i++) {
        float x=mixes[token*24u+i];
        float s=scale[i<4?0:(i<8?1:2)],b=base[i];
        if(!isfinite(x)||!isfinite(s)||!isfinite(b))atomicOr(status,1u);
        m[i]=x*s+b;if(!isfinite(m[i]))atomicOr(status,2u);
    }
    for(unsigned i=0;i<4;i++) {
        pre[token*4u+i]=1.0f/(1.0f+expf(-m[i]))+1e-6f;
        post[token*4u+i]=2.0f/(1.0f+expf(-m[i+4]));
    }
    float c[16];
    for(unsigned row=0;row<4;row++) {
        float mx=m[8+row*4];for(unsigned col=1;col<4;col++)mx=fmaxf(mx,m[8+row*4+col]);
        float sum=0.0f;
        for(unsigned col=0;col<4;col++){unsigned i=row*4+col;c[i]=expf(m[8+i]-mx);sum+=c[i];}
        for(unsigned col=0;col<4;col++)c[row*4+col]=c[row*4+col]/sum+1e-6f;
    }
    for(unsigned iteration=0;iteration<20;iteration++) {
        if(iteration)for(unsigned row=0;row<4;row++) {
            float sum=0.0f;for(unsigned col=0;col<4;col++)sum+=c[row*4+col];
            for(unsigned col=0;col<4;col++)c[row*4+col]/=sum+1e-6f;
        }
        for(unsigned col=0;col<4;col++) {
            float sum=0.0f;for(unsigned row=0;row<4;row++)sum+=c[row*4+col];
            for(unsigned row=0;row<4;row++)c[row*4+col]/=sum+1e-6f;
        }
    }
    for(unsigned i=0;i<16;i++){if(!isfinite(c[i]))atomicOr(status,2u);comb[token*16u+i]=c[i];}
}
extern "C" __global__ void dsv41_hc_initial(float* pre,unsigned tokens) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<tokens*4u)pre[i]=(i%4u==0u)?1.0f:0.0f;
}
// BF16 only at the official hc_pre output boundary, never between products.
extern "C" __global__ void dsv41_hc_pre(
    const __nv_bfloat16* h,const float* pre,__nv_bfloat16* out,uint32_t* status,unsigned tokens,unsigned dim) {
    unsigned flat=blockIdx.x*blockDim.x+threadIdx.x;if(flat>=tokens*dim)return;
    unsigned token=flat/dim,col=flat%dim;float sum=0.0f;
    for(unsigned copy=0;copy<4;copy++) {
        float v=__bfloat162float(h[(token*4u+copy)*dim+col]),p=pre[token*4u+copy];
        if(!isfinite(v)||!isfinite(p))atomicOr(status,1u);sum+=p*v;
    }
    __nv_bfloat16 rounded=__float2bfloat16_rn(sum);
    if(!isfinite(sum)||!isfinite(__bfloat162float(rounded)))atomicOr(status,2u);out[flat]=rounded;
}
// comb[source,destination] and sum over source is the exact official orientation.
extern "C" __global__ void dsv41_hc_post(
    const __nv_bfloat16* x,const __nv_bfloat16* residual,const float* post,const float* comb,
    __nv_bfloat16* out,uint32_t* status,unsigned tokens,unsigned dim) {
    unsigned flat=blockIdx.x*blockDim.x+threadIdx.x;if(flat>=tokens*4u*dim)return;
    unsigned col=flat%dim,destination=(flat/dim)%4u,token=flat/(dim*4u);float sum=0.0f;
    for(unsigned source=0;source<4;source++) {
        float c=comb[token*16u+source*4u+destination],v=__bfloat162float(residual[(token*4u+source)*dim+col]);
        if(!isfinite(c)||!isfinite(v))atomicOr(status,1u);sum+=c*v;
    }
    float p=post[token*4u+destination],v=__bfloat162float(x[token*dim+col]);
    if(!isfinite(p)||!isfinite(v))atomicOr(status,1u);
    float value=p*v+sum;__nv_bfloat16 rounded=__float2bfloat16_rn(value);
    if(!isfinite(value)||!isfinite(__bfloat162float(rounded)))atomicOr(status,2u);out[flat]=rounded;
}
