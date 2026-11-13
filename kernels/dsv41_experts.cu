// Native1017 grouped routed experts, text trunk only; no shared expert/reduction/model.
// Qualified native FP4/E4M3 MMA helpers are included unchanged. Dedicated build must
// hash this file plus the GEMM/Numerics/MoE source dependencies and exact flags.
#include "dsv41_gemm.cu"
#include <cuda_fp8.h>
#ifndef DSV41_EXPERTS_SOURCE_ID
#error "Dedicated DSV41_EXPERTS_SOURCE_ID required"
#endif
namespace {
constexpr unsigned EXPERT_ABI = 1, LOCAL_EXPERTS = 48, TOPK = 6;
constexpr unsigned BAD_ROUTE = 8, BAD_BANK = 16;
__device__ __forceinline__ float expert_ue8(unsigned char code) {
    return __uint_as_float(code == 0 ? (1u << 22) : ((unsigned)code << 23));
}
__device__ __forceinline__ unsigned char expert_fp8(float value) {
    return __nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,value)),__NV_SATFINITE,__NV_E4M3);
}
// Exact group32 body from qualified dsv41_numerics.cu. Each warp consumes one
// disjoint group. Compacted expert rows beyond counts are never read/stored.
__device__ __forceinline__ void expert_quant(
    unsigned char* out,unsigned char* scales,unsigned* status,const __nv_bfloat16* input,
    uint64_t group) {
    unsigned lane=threadIdx.x&31u;
    float value=__bfloat162float(input[group*32+lane]);
    if(!isfinite(value)){bad(status,BAD_INPUT);value=0.f;}
    float maximum=fabsf(value);
    #pragma unroll
    for(int delta=16;delta>0;delta>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,delta));
    float divided=__fmul_rn(fmaxf(maximum,1e-4f),1.f/448.f);
    unsigned bits=__float_as_uint(divided);
    int exponent=(int)((bits>>23)&255u)-127+((bits&0x7fffffu)!=0);
    unsigned code=(unsigned)(exponent+127);
    if(!(divided>0.f)||!isfinite(divided)||code==0||code>=255){
        bad(status,BAD_SCALE);if(lane==0)scales[group]=127;out[group*32+lane]=0;return;
    }
    float s=__uint_as_float(code<<23);
    if(lane==0)scales[group]=(unsigned char)code;
    out[group*32+lane]=expert_fp8(__fdiv_rn(value,s));
}
__device__ __forceinline__ unsigned gather_activation_word(
    const unsigned char* input,const unsigned* maps,int expert,int token,int count,int capacity,
    int inner,int column,unsigned* status) {
    if(token>=count)return 0;
    int row=(int)maps[(uint64_t)expert*capacity+token];
    return activation_word(input,row,capacity,inner,column,status);
}
} // namespace
extern "C" __global__ void dsv41_experts_identity(uint64_t* out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_EXPERTS_SOURCE_ID;out[1]=EXPERT_ABI;}
}
extern "C" __global__ void dsv41_experts_gemm_dependency(uint64_t* out){
    if(!blockIdx.x&&!threadIdx.x)out[0]=DSV41_GEMM_SOURCE_ID;
}
extern "C" __global__ void dsv41_experts_reset(unsigned* status){
    if(!blockIdx.x&&!threadIdx.x)*status=0;
}
extern "C" __global__ void dsv41_experts_validate_routes(
    const unsigned* ids,const float* route_weights,const unsigned* upstream,unsigned* status,uint64_t rows){
    unsigned row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    if(*upstream){bad(status,BAD_ROUTE);return;}
    for(unsigned k=0;k<TOPK;++k){
        unsigned id=ids[(uint64_t)row*TOPK+k];float w=route_weights[(uint64_t)row*TOPK+k];
        if(id>=384||!isfinite(w)||w<0.f||w>1.5f)bad(status,BAD_ROUTE);
        for(unsigned j=0;j<k;++j)if(ids[(uint64_t)row*TOPK+j]==id)bad(status,BAD_ROUTE);
    }
}
extern "C" __global__ void dsv41_experts_prepare(
    unsigned* maps,unsigned* inverse,unsigned* counts,float* selected_weights,
    const unsigned* ids,const float* route_weights,const uint64_t* bank,unsigned* status,
    uint64_t rows,uint64_t rank){
    unsigned e=blockIdx.x;if(e>=LOCAL_EXPERTS||threadIdx.x)return;
    unsigned count=0;
    for(unsigned row=0;row<rows;++row)inverse[(uint64_t)e*rows+row]=0xffffffffu;
    if(!atomicOr(status,0u)){
        unsigned global=(unsigned)rank*LOCAL_EXPERTS+e;
        for(unsigned row=0;row<rows;++row)for(unsigned k=0;k<TOPK;++k){
            if(ids[(uint64_t)row*TOPK+k]==global){
                maps[(uint64_t)e*rows+count]=row;
                inverse[(uint64_t)e*rows+row]=count;
                selected_weights[(uint64_t)e*rows+count]=route_weights[(uint64_t)row*TOPK+k];++count;
            }
        }
        if(count){for(unsigned i=0;i<6;++i)if(!bank[e*6+i]){bad(status,BAD_BANK);count=0;}}
    }
    counts[e]=count;
}
extern "C" __global__ void dsv41_experts_input_quant(
    unsigned char* out,unsigned char* scales,unsigned* status,const __nv_bfloat16* input,uint64_t groups){
    uint64_t group=(uint64_t)blockIdx.x*4+(threadIdx.x>>5);
    if(group<groups)expert_quant(out,scales,status,input,group);
}
// Fused w1/w3 shares the gathered E4 activation fragments, preserving each
// projection's independent K32 raw accumulator and complete-BF16 boundary.
extern "C" __global__ void dsv41_experts_w13(
    __nv_bfloat16* silu,unsigned* status,const unsigned char* activations,const unsigned char* as,
    const uint64_t* bank,const unsigned* maps,const unsigned* counts,const float* weights,
    uint64_t capacity64,uint64_t dim64,uint64_t inter64){
    int capacity=(int)capacity64,inner=(int)dim64,outputs=(int)inter64,e=(int)blockIdx.z;
    int count=(int)counts[e],token0=(int)blockIdx.y*16;
    // One CTA snapshot prevents a concurrently reported failure from making
    // only part of a warp leave before its collective MMA/shuffle instructions.
    __shared__ unsigned blocked;if(!threadIdx.x)blocked=atomicOr(status,0u);__syncthreads();
    if(token0>=count||blocked)return; // skip before any expert weight access
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
    int output0=((int)blockIdx.x*8+warp)*16;if(output0>=outputs)return;
    const unsigned char* w1=(const unsigned char*)bank[e*6];
    const unsigned char* s1=(const unsigned char*)bank[e*6+1];
    const unsigned char* w3=(const unsigned char*)bank[e*6+2];
    const unsigned char* s3=(const unsigned char*)bank[e*6+3];
    float acc1[2][4]={{0,0,0,0},{0,0,0,0}},acc3[2][4]={{0,0,0,0},{0,0,0,0}};
    int groups=inner/32;
    for(int k=0;k<groups;++k){
        int column=k*32+4*t;
        unsigned a1[4]={weight_word<true>(w1,output0+g,outputs,inner,column,status),
            weight_word<true>(w1,output0+g+8,outputs,inner,column,status),
            weight_word<true>(w1,output0+g,outputs,inner,column+16,status),
            weight_word<true>(w1,output0+g+8,outputs,inner,column+16,status)};
        unsigned a3[4]={weight_word<true>(w3,output0+g,outputs,inner,column,status),
            weight_word<true>(w3,output0+g+8,outputs,inner,column,status),
            weight_word<true>(w3,output0+g,outputs,inner,column+16,status),
            weight_word<true>(w3,output0+g+8,outputs,inner,column+16,status)};
        float raw1[2][4]={{0,0,0,0},{0,0,0,0}},raw3[2][4]={{0,0,0,0},{0,0,0,0}};
        #pragma unroll
        for(int sub=0;sub<2;++sub){
            unsigned b[2]={gather_activation_word(activations,maps,e,token0+sub*8+g,count,capacity,inner,column,status),
                gather_activation_word(activations,maps,e,token0+sub*8+g,count,capacity,inner,column+16,status)};
            mma(raw1[sub],a1,b);mma(raw3[sub],a3,b);
        }
        #pragma unroll
        for(int sub=0;sub<2;++sub)
            #pragma unroll
            for(int i=0;i<4;++i){
                int output=output0+g+8*(i>=2),token=token0+sub*8+2*t+(i&1);
                if(output<outputs&&token<count){
                    unsigned row=maps[(uint64_t)e*capacity+token];
                    float a_scale=scale(as[(uint64_t)row*groups+k],status);
                    float c1=__fmul_rn(__fmul_rn(raw1[sub][i],a_scale),scale(s1[(uint64_t)output*groups+k],status));
                    float c3=__fmul_rn(__fmul_rn(raw3[sub][i],a_scale),scale(s3[(uint64_t)output*groups+k],status));
                    float n1=__fadd_rn(acc1[sub][i],c1),n3=__fadd_rn(acc3[sub][i],c3);
                    if(!isfinite(c1)||!isfinite(c3)||!isfinite(n1)||!isfinite(n3)){bad(status,OVERFLOW);n1=n3=0.f;}
                    acc1[sub][i]=n1;acc3[sub][i]=n3;
                }
            }
    }
    #pragma unroll
    for(int sub=0;sub<2;++sub)
        #pragma unroll
        for(int i=0;i<4;++i){
            int output=output0+g+8*(i>=2),token=token0+sub*8+2*t+(i&1);
            if(output<outputs&&token<count){
                // Exactly the qualified MoE pre-w2 order. Never cast a partial
                // or unweighted SwiGLU result to BF16.
                float gate=__bfloat162float(__float2bfloat16_rn(acc1[sub][i]));
                float up=__bfloat162float(__float2bfloat16_rn(acc3[sub][i]));
                if(!isfinite(gate)||!isfinite(up)){bad(status,OVERFLOW);gate=up=0.f;}
                gate=fminf(gate,10.f);up=fminf(10.f,fmaxf(-10.f,up));
                float value=(gate/(1.f+expf(-gate)))*up;
                value*=weights[(uint64_t)e*capacity+token];
                __nv_bfloat16 result=__float2bfloat16_rn(value);
                if(!isfinite(value)||!isfinite(__bfloat162float(result))){bad(status,OVERFLOW);result=__float2bfloat16_rn(0.f);}
                silu[((uint64_t)e*capacity+token)*outputs+output]=result;
            }
        }
}
extern "C" __global__ void dsv41_experts_silu_quant(
    unsigned char* out,unsigned char* scales,unsigned* status,const __nv_bfloat16* input,
    const unsigned* counts,uint64_t capacity,uint64_t inter){
    __shared__ unsigned blocked;if(!threadIdx.x)blocked=atomicOr(status,0u);__syncthreads();
    uint64_t group=(uint64_t)blockIdx.x*4+(threadIdx.x>>5);
    uint64_t per_expert=capacity*(inter/32),e=group/per_expert;
    if(e>=LOCAL_EXPERTS||blocked)return;
    uint64_t token=(group%per_expert)/(inter/32);
    if(token<counts[e])expert_quant(out,scales,status,input,group);
}
extern "C" __global__ void dsv41_experts_w2(
    __nv_bfloat16* out,unsigned* status,const unsigned char* a,const unsigned char* as,
    const uint64_t* bank,const unsigned* counts,uint64_t capacity,uint64_t dim,uint64_t inter){
    __shared__ unsigned blocked;if(!threadIdx.x)blocked=atomicOr(status,0u);__syncthreads();
    unsigned e=blockIdx.z,count=counts[e];if(blockIdx.y*16>=count||blocked)return;
    gemm_body<true>(nullptr,out+(uint64_t)e*capacity*dim,status,
        a+(uint64_t)e*capacity*inter,as+(uint64_t)e*capacity*(inter/32),
        (const unsigned char*)bank[e*6+4],(const unsigned char*)bank[e*6+5],
        (int)count,(int)dim,(int)inter);
}
extern "C" __global__ void dsv41_experts_sum(
    float* out,unsigned* status,const __nv_bfloat16* partial,const unsigned* inverse,
    uint64_t rows,uint64_t dim){
    __shared__ unsigned blocked;if(!threadIdx.x)blocked=atomicOr(status,0u);__syncthreads();
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*dim)return;
    unsigned row=(unsigned)(i/dim);uint64_t column=i%dim;float total=0.f;
    if(!blocked)for(unsigned e=0;e<LOCAL_EXPERTS;++e){
        unsigned token=inverse[(uint64_t)e*rows+row];
        if(token!=0xffffffffu)total=__fadd_rn(total,__bfloat162float(partial[((uint64_t)e*rows+token)*dim+column]));
    }
    if(!isfinite(total)){bad(status,OVERFLOW);total=0.f;}out[i]=total;
}
// MoE.forward combines the network-reduced routed F32 partial with the shared
// expert's complete BF16 output BEFORE the single final BF16 cast.
extern "C" __global__ void dsv41_experts_combine(
    __nv_bfloat16* out,unsigned* status,const float* reduced,const __nv_bfloat16* shared,uint64_t count){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float routed=reduced[i],s=__bfloat162float(shared[i]);
    if(!isfinite(routed)||!isfinite(s)){bad(status,BAD_INPUT);out[i]=__float2bfloat16_rn(0.f);return;}
    float sum=__fadd_rn(routed,s);__nv_bfloat16 value=__float2bfloat16_rn(sum);
    if(!isfinite(sum)||!isfinite(__bfloat162float(value))){bad(status,OVERFLOW);value=__float2bfloat16_rn(0.f);}
    out[i]=value;
}
