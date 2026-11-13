// Additive 8192-context chunk-cache primitives. Not registered or qualified.
// Inputs already passed source RoPE and window/compressed cache QAT. The caller
// snapshots the old chronological window BEFORE enqueueing any ring writes.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_CHUNK_STORE_SOURCE_ID
#error Dedicated DSV41_CHUNK_STORE_SOURCE_ID required
#endif
namespace {
constexpr uint64_t D=512, W=128, L=40, S=4, C=8192;
__device__ bool lane_ok(uint64_t lane,uint64_t lanes,unsigned*status){
    if(!lanes||lanes>16||lane>=lanes){atomicOr(status,2u);return false;}return true;
}
__device__ bool chunk_ok(uint64_t start,uint64_t tokens,unsigned*status){
    if(!tokens||tokens>128||start>C||tokens>C-start){atomicOr(status,2u);return false;}return true;
}
__device__ __nv_bfloat16 finite_value(__nv_bfloat16 v,unsigned*status){
    if(!isfinite(__bfloat162float(v))){atomicOr(status,1u);return __float2bfloat16_rn(0.f);}return v;
}
}
extern "C" __global__ void dsv41_chunk_store_identity(uint64_t*out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_CHUNK_STORE_SOURCE_ID;out[1]=1;}
}
extern "C" __global__ void dsv41_chunk_store_reset(
    __nv_bfloat16*windows,__nv_bfloat16*compressed,unsigned*status,uint64_t lane,uint64_t lanes){
    if(!lane_ok(lane,lanes,status))return;
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<L*W*D)windows[lane*L*W*D+i]=__float2bfloat16_rn(0.f);
    if(i<S*C*D)compressed[lane*S*C*D+i]=__float2bfloat16_rn(0.f);
    // Status is sticky. Its explicit owner initializes/clears it before reset.
}
extern "C" __global__ void dsv41_chunk_store_window_snapshot(
    __nv_bfloat16*out,const __nv_bfloat16*current,const __nv_bfloat16*windows,
    unsigned*status,uint64_t lane,uint64_t lanes,uint64_t layer,uint64_t start,uint64_t tokens){
    if(!lane_ok(lane,lanes,status)||!chunk_ok(start,tokens,status))return;
    if(layer>=L){atomicOr(status,2u);return;}
    uint64_t previous=min(start,W-1),base=start-previous;
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(previous+tokens)*D)return;
    uint64_t row=i/D,col=i%D;
    __nv_bfloat16 value=row<previous
        ?windows[((lane*L+layer)*W+(base+row)%W)*D+col]
        :current[(row-previous)*D+col];
    out[i]=finite_value(value,status);
}
extern "C" __global__ void dsv41_chunk_store_window_write(
    const __nv_bfloat16*current,__nv_bfloat16*windows,unsigned*status,
    uint64_t lane,uint64_t lanes,uint64_t layer,uint64_t start,uint64_t tokens){
    if(!lane_ok(lane,lanes,status)||!chunk_ok(start,tokens,status))return;
    if(layer>=L){atomicOr(status,2u);return;}
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*D)return;
    uint64_t row=i/D,col=i%D;
    windows[((lane*L+layer)*W+(start+row)%W)*D+col]=finite_value(current[i],status);
}
extern "C" __global__ void dsv41_chunk_store_compressed_append(
    const __nv_bfloat16*current,__nv_bfloat16*compressed,unsigned*status,
    uint64_t lane,uint64_t lanes,uint64_t owner,uint64_t first,uint64_t rows){
    if(!lane_ok(lane,lanes,status))return;
    if(owner>=S||first>C||rows>128||rows>C-first){atomicOr(status,2u);return;}
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*D)return;
    compressed[((lane*S+owner)*C+first)*D+i]=finite_value(current[i],status);
}
extern "C" __global__ void dsv41_chunk_store_combined_snapshot(
    __nv_bfloat16*out,const __nv_bfloat16*window_snapshot,const __nv_bfloat16*compressed,
    unsigned*status,uint64_t lane,uint64_t lanes,uint64_t owner,uint64_t window_rows,uint64_t compressed_rows){
    if(!lane_ok(lane,lanes,status))return;
    if(owner>=S||!window_rows||window_rows>255||compressed_rows>C){atomicOr(status,2u);return;}
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(window_rows+compressed_rows)*D)return;
    uint64_t row=i/D,col=i%D;
    out[i]=finite_value(row<window_rows?window_snapshot[i]
        :compressed[((lane*S+owner)*C+row-window_rows)*D+col],status);
}
// selected positions are source position-sorted top512, before combined offset.
// A null selection is exact ONLY when all compressed keys fit top512.
extern "C" __global__ void dsv41_chunk_store_indices(
    int*out,const int*selection,unsigned*status,uint64_t start,uint64_t tokens,
    uint64_t ratio,uint64_t compressed_rows){
    if(!chunk_ok(start,tokens,status))return;
    if((ratio!=0&&ratio!=1&&ratio!=2)||compressed_rows>C
        ||(ratio==0?compressed_rows!=0:compressed_rows!=(start+tokens)/ratio)
        ||(compressed_rows>512&&!selection)){atomicOr(status,2u);return;}
    uint64_t previous=min(start,W-1),base=start-previous,window_rows=previous+tokens;
    uint64_t window_capacity=start?W:min(tokens,W),k=min(compressed_rows,(uint64_t)512);
    uint64_t topk=window_capacity+k,i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*topk)return;
    uint64_t q=i/topk,j=i%topk,p=start+q;
    if(j<window_capacity){
        uint64_t first=p+1>W?p+1-W:0,reachable=p-first+1;
        out[i]=j<reachable?(int)(first+j-base):-1;
    }else{
        uint64_t col=j-window_capacity,visible=(p+1)/ratio;
        int position=selection?selection[q*k+col]:(col<visible?(int)col:-1);
        if(position < -1 || (position>=0&&(uint64_t)position>=visible)){
            atomicOr(status,4u);out[i]=-1;
        }else out[i]=position<0?-1:(int)(window_rows+position);
    }
}
