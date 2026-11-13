// Isolated official1017 dense BF16/F32 components; no model/serving route.
// References model.py e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d
// and convert.py035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76.
// DeepSeek copyright(c)2023, MIT; full notice in dsv41/numerics.rs.
// Native BF16 MMA fragment mapping follows gpu_batch.cu. Compile separately:
// sm_121 -O3 --ftz=false --fmad=false; no TF32 or FP16 arithmetic.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_DENSE_SOURCE_ID
#error DedicatedDSV41_DENSE_SOURCE_ID required
#endif
namespace {
constexpr unsigned BAD_INPUT=1, BAD_SCALE=2, OVERFLOW=4, LOSSY_CAST=8;
__device__ __forceinline__ void bad(unsigned* status,unsigned bit){atomicOr(status,bit);}
__device__ __forceinline__ float get_bf16(const __nv_bfloat16* data,uint64_t i,unsigned* status){
    float v=__bfloat162float(data[i]);
    if(!isfinite(v)){bad(status,BAD_INPUT);return 0.f;}return v;
}
__device__ __forceinline__ float get_f32(const float* data,uint64_t i,unsigned* status){
    float v=data[i];if(!isfinite(v)){bad(status,BAD_INPUT);return 0.f;}return v;
}
__device__ __forceinline__ __nv_bfloat16 round_complete(float x,unsigned* status){
    __nv_bfloat16 b=__float2bfloat16_rn(x);
    if(!isfinite(x)||!isfinite(__bfloat162float(b))){bad(status,OVERFLOW);return __float2bfloat16_rn(0.f);}
    return b;
}
__device__ __forceinline__ unsigned pair(const __nv_bfloat16* data,int row,int rows,
    int inner,int col,unsigned* status){
    float lo=(row<rows&&col<inner)?get_bf16(data,(uint64_t)row*inner+col,status):0.f;
    float hi=(row<rows&&col+1<inner)?get_bf16(data,(uint64_t)row*inner+col+1,status):0.f;
    __nv_bfloat162 p=__floats2bfloat162_rn(lo,hi);
    return *reinterpret_cast<unsigned*>(&p);
}
__device__ __forceinline__ void mma(float* d,const unsigned* a,const unsigned* b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void store(float* out_f32,__nv_bfloat16* out_bf16,
    uint64_t index,float value,unsigned* status){
    if(!isfinite(value)){bad(status,OVERFLOW);value=0.f;}
    if(out_f32)out_f32[index]=value;
    if(out_bf16)out_bf16[index]=round_complete(value,status);
}
template<bool BF16_WEIGHTS>
__device__ __forceinline__ void f32_body(float* out_f32,__nv_bfloat16* out_bf16,
    unsigned* status,const float* input,const void* weights,int rows,int outputs,int inner){
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int out=(int)blockIdx.x*8+warp,token0=(int)blockIdx.y*16;
    if(out>=outputs)return; // entire warp
    float accum[16];
    #pragma unroll
    for(int row=0;row<16;row++)accum[row]=0.f;
    // Fixed per-lane ascending K stripe and fixed warp tree at every batch width.
    // StoredBF16 weights are promoted directly toFP32 registers, exactly as weight.float().
    for(int k=lane;k<inner;k+=32){
        float w=BF16_WEIGHTS?get_bf16((const __nv_bfloat16*)weights,(uint64_t)out*inner+k,status):
            get_f32((const float*)weights,(uint64_t)out*inner+k,status);
        #pragma unroll
        for(int row=0;row<16;row++){
            float a=token0+row<rows?get_f32(input,(uint64_t)(token0+row)*inner+k,status):0.f;
            accum[row]=__fadd_rn(accum[row],__fmul_rn(a,w));
        }
    }
    #pragma unroll
    for(int row=0;row<16;row++){
        float x=accum[row];
        #pragma unroll
        for(int delta=16;delta;delta>>=1)
            x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,delta));
        if(lane==0&&token0+row<rows)
            store(out_f32,out_bf16,(uint64_t)(token0+row)*outputs+out,x,status);
    }
}
__device__ __forceinline__ float fp8_value(unsigned char code,unsigned* status){
    if((code&127u)==127u){bad(status,BAD_INPUT);return 0.f;}
    unsigned e=(code>>3)&15u,m=code&7u;
    float value=e?__uint_as_float(((e+120u)<<23)|(m<<20)):(float)m*0x1p-9f;
    return code&128u?-value:value;
}
}
extern "C" __global__ void dsv41_dense_identity(unsigned long long* out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_DENSE_SOURCE_ID;out[1]=1;}
}
extern "C" __global__ void dsv41_dense_bf16_mma(float* out_f32,__nv_bfloat16* out_bf16,
    unsigned* status,const __nv_bfloat16* input,const __nv_bfloat16* weights,
    int rows,int outputs,int inner){
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
    const int output0=((int)blockIdx.x*8+warp)*16,token0=(int)blockIdx.y*16;
    if(output0>=outputs)return;
    float accum[2][4]={{0,0,0,0},{0,0,0,0}};
    for(int k=0;k<inner;k+=16){
        unsigned a[4]={pair(weights,output0+g,outputs,inner,k+2*t,status),
            pair(weights,output0+g+8,outputs,inner,k+2*t,status),
            pair(weights,output0+g,outputs,inner,k+2*t+8,status),
            pair(weights,output0+g+8,outputs,inner,k+2*t+8,status)};
        #pragma unroll
        for(int sub=0;sub<2;sub++){
            unsigned b[2]={pair(input,token0+sub*8+g,rows,inner,k+2*t,status),
                pair(input,token0+sub*8+g,rows,inner,k+2*t+8,status)};
            mma(accum[sub],a,b);
        }
    }
    #pragma unroll
    for(int sub=0;sub<2;sub++)
        #pragma unroll
        for(int i=0;i<4;i++){
            int out=output0+g+8*(i>=2),row=token0+sub*8+2*t+(i&1);
            if(out<outputs&&row<rows)
                store(out_f32,out_bf16,(uint64_t)row*outputs+out,accum[sub][i],status);
        }
}
extern "C" __global__ void dsv41_dense_f32_bf16(float* out_f32,__nv_bfloat16* out_bf16,
    unsigned* status,const float* input,const __nv_bfloat16* weights,int rows,int outputs,int inner){
    f32_body<true>(out_f32,out_bf16,status,input,weights,rows,outputs,inner);
}
extern "C" __global__ void dsv41_dense_f32_f32(float* out_f32,__nv_bfloat16* out_bf16,
    unsigned* status,const float* input,const float* weights,int rows,int outputs,int inner){
    f32_body<false>(out_f32,out_bf16,status,input,weights,rows,outputs,inner);
}
// Mode0: originalFP8weight [N,K], scales[ceil(N/32),K/32].
// Mode1: quantizedactivation [M,K], scales[M,K/32] from officialact_quant.
// Strict exact flag is a qualification guard: reject any non-lossless BF16 cast.
extern "C" __global__ void dsv41_dense_fp8_to_bf16(__nv_bfloat16* output,unsigned* status,
    const unsigned char* codes,const unsigned char* scales,uint64_t count,unsigned inner,
    unsigned activation_mode,unsigned strict_exact){
    uint64_t index=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    unsigned row=index/inner,col=index%inner;
    unsigned sr=activation_mode?row:row/32;
    unsigned char code=scales[(uint64_t)sr*(inner/32)+col/32];
    if(code==255){bad(status,BAD_SCALE);output[index]=__float2bfloat16_rn(0.f);return;}
    float s=__uint_as_float(code==0?(1u<<22):((unsigned)code<<23));
    float x=__fmul_rn(fp8_value(codes[index],status),s);
    __nv_bfloat16 rounded=round_complete(x,status);
    if(strict_exact&&isfinite(x)&&__bfloat162float(rounded)!=x)bad(status,LOSSY_CAST);
    output[index]=rounded;
}
