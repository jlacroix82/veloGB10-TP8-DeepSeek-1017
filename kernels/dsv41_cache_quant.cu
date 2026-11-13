// Isolated official1017 FP4 cache/activation quantization. No checkpoint conversion.
// Source kernel.py1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455.
// DeepSeek MIT attribution: src/dsv41/numerics.rs. sm_121, --ftz=false --fmad=false.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_CACHE_QUANT_SOURCE_ID
#error Dedicated DSV41_CACHE_QUANT_SOURCE_ID required
#endif
namespace {
constexpr unsigned BAD_INPUT=1,BAD_SCALE=2,OVERFLOW=4;
__device__ void bad(unsigned* status,unsigned bit){atomicOr(status,bit);}
__device__ float fp4(unsigned code){
    const float values[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    const float magnitude=values[code&7u];return code&8u?-magnitude:magnitude;
}
__device__ float ue8(unsigned code){return __uint_as_float(code?code<<23:1u<<22);}
__device__ float e4(unsigned code){
    const unsigned e=(code>>3)&15u,m=code&7u;
    float value=e?__uint_as_float(((e+120u)<<23)|(m<<20)):float(m)*0x1p-9f;
    return code&128u?-value:value;
}
__device__ float decode_scale(unsigned code,unsigned mode,unsigned* status){
    if((mode==0&&(code&127u)==127u)||(mode==1&&code==255u)) {bad(status,BAD_SCALE);return 1.f;}
    float value=mode?ue8(code):e4(code);
    if(!(value>0.f)||!isfinite(value)){bad(status,BAD_SCALE);return 1.f;}return value;
}
template<int GROUP,bool E4>
__device__ void quant(unsigned char* packed,unsigned char* scales,__nv_bfloat16* qat,
    unsigned* status,const __nv_bfloat16* input,uint64_t groups){
    const unsigned lane=threadIdx.x&31u;
    const uint64_t group=(uint64_t)blockIdx.x*4u+(threadIdx.x>>5);
    if(group>=groups)return;
    float value=lane<GROUP?__bfloat162float(input[group*GROUP+lane]):0.f;
    if(!isfinite(value)){bad(status,BAD_INPUT);value=0.f;}
    float maximum=fabsf(value);
    #pragma unroll
    for(unsigned offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    unsigned code;float scale;
    if(E4){
        float divided=__fdiv_rn(fmaxf(maximum,0x1.8p-7f),6.f); //6*2^-9
        code=__nv_cvt_float_to_fp8(divided,__NV_SATFINITE,__NV_E4M3);
        scale=e4(code);
    }else{
        float divided=__fmul_rn(fmaxf(maximum,0x1.8p-124f),1.f/6.f); //6*2^-126
        unsigned bits=__float_as_uint(divided);
        int exponent=int((bits>>23)&255u)-127+int((bits&0x7fffffu)!=0);
        code=unsigned(exponent+127);scale=__uint_as_float(code<<23);
    }
    if(!(scale>0.f)||!isfinite(scale)||code>=255u){
        bad(status,BAD_SCALE);scale=1.f;code=E4?0x38u:127u;
    }
    if(lane==0)scales[group]=(unsigned char)code;
    const float normalized=__fdiv_rn(value,scale);
    const float clamped=fminf(6.f,fmaxf(-6.f,normalized));
    unsigned nibble=__nv_cvt_float_to_fp4(clamped,__NV_E2M1,cudaRoundNearest)&15u;
    unsigned next=__shfl_down_sync(0xffffffffu,nibble,1);
    if(lane<GROUP){
        if(!(lane&1u))packed[group*(GROUP/2)+lane/2]=(unsigned char)(nibble|(next<<4));
        float restored=__fmul_rn(fp4(nibble),scale);
        __nv_bfloat16 rounded=__float2bfloat16_rn(restored);
        if(!isfinite(normalized)||!isfinite(restored)||!isfinite(__bfloat162float(rounded))){bad(status,OVERFLOW);rounded=__float2bfloat16_rn(0.f);}
        qat[group*GROUP+lane]=rounded;
    }
}
}
extern "C" __global__ void dsv41_cache_quant_identity(unsigned long long* out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_CACHE_QUANT_SOURCE_ID;out[1]=1;}
}
extern "C" __global__ void dsv41_cache_quant_e4_group16(unsigned char* packed,unsigned char* scales,
    __nv_bfloat16* qat,unsigned* status,const __nv_bfloat16* input,uint64_t groups){quant<16,true>(packed,scales,qat,status,input,groups);}
extern "C" __global__ void dsv41_cache_quant_ue8_group32(unsigned char* packed,unsigned char* scales,
    __nv_bfloat16* qat,unsigned* status,const __nv_bfloat16* input,uint64_t groups){quant<32,false>(packed,scales,qat,status,input,groups);}
extern "C" __global__ void dsv41_cache_dequant_bf16(__nv_bfloat16* out,unsigned* status,
    const unsigned char* packed,const unsigned char* scales,uint64_t count,unsigned mode){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned byte=packed[i/2],code=(i&1u)?byte>>4:byte&15u;
    float scale=decode_scale(scales[i/(mode?32u:16u)],mode,status);
    float value=__fmul_rn(fp4(code),scale);__nv_bfloat16 rounded=__float2bfloat16_rn(value);
    if(!isfinite(value)||!isfinite(__bfloat162float(rounded))){bad(status,OVERFLOW);rounded=__float2bfloat16_rn(0.f);}
    out[i]=rounded;
}
// Direct native cast probe, independent of the CPU RNE-table implementation.
extern "C" __global__ void dsv41_cache_fp4_encode(unsigned char* out,unsigned* status,const float* input,uint64_t count){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float x=input[i];if(!isfinite(x)){bad(status,BAD_INPUT);x=0.f;}
    out[i]=__nv_cvt_float_to_fp4(fminf(6.f,fmaxf(-6.f,x)),__NV_E2M1,cudaRoundNearest)&15u;
}
