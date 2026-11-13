// Official1017 candidate-block/top512 selection over source BF16 scores.
// Equal-score choices use ascending position, a deterministic valid topk witness.
// No score, cache, projection, collective or model execution is performed here.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_INDEX_SELECT_SOURCE_ID
#error Dedicated DSV41_INDEX_SELECT_SOURCE_ID required
#endif
namespace {
constexpr unsigned BAD_NUMERIC=1, BAD_EXTENT=2, BAD_CANDIDATES=4;
__device__ unsigned ordered(__nv_bfloat16 value){
    unsigned bits=*reinterpret_cast<unsigned short*>(&value);
    if((bits&0x7fffu)==0)bits=0; // source numeric equality treats both zero signs alike.
    return bits&0x8000u?(~bits&0xffffu):(bits^0x8000u);
}
struct SharedKeys{
    const unsigned short* values;
    __device__ unsigned key(unsigned i)const{return values[i];}
};
struct PositionKeys{
    const __nv_bfloat16* values;const unsigned char* candidates;unsigned visible;
    __device__ unsigned key(unsigned i)const{
        if(i>=visible||(candidates&&!candidates[i]))return 0;
        auto v=values[i];return isfinite(__bfloat162float(v))?ordered(v):0;
    }
};
// Two radix passes over the complete BF16 numeric domain find a score cutoff.
// The final stable tie index is computed independently of atomic scheduling.
template<class Keys>
__device__ void cutoff(Keys keys,unsigned count,unsigned wanted,unsigned* histogram,
    unsigned* prefix,unsigned* remaining,unsigned* tie_index){
    const unsigned tid=threadIdx.x;
    if(!tid){*prefix=0;*remaining=min(wanted,count);*tie_index=0xffffffffu;}
    __syncthreads();
    for(int shift=8;shift>=0;shift-=8){
        if(tid<256)histogram[tid]=0;
        __syncthreads();
        for(unsigned i=tid;i<count;i+=blockDim.x){
            unsigned k=keys.key(i);
            if((k>>(shift+8))==(*prefix>>(shift+8)))atomicAdd(histogram+((k>>shift)&255u),1u);
        }
        __syncthreads();
        if(!tid){for(int bucket=255;bucket>=0;--bucket){
            unsigned n=histogram[bucket];
            if(*remaining>n)*remaining-=n;else{*prefix|=(unsigned)bucket<<shift;break;}
        }}
        __syncthreads();
    }
    if(!tid){unsigned seen=0;for(unsigned i=0;i<count;++i)if(keys.key(i)==*prefix){
        if(++seen==*remaining){*tie_index=i;break;}
    }}
    __syncthreads();
}
__device__ bool selected(unsigned key,unsigned index,unsigned boundary,unsigned tie){
    return key&&(key>boundary||(key==boundary&&index<=tie));
}
}
extern "C" __global__ void dsv41_index_select_identity(uint64_t* output){
    if(!blockIdx.x&&!threadIdx.x){output[0]=DSV41_INDEX_SELECT_SOURCE_ID;output[1]=1;}
}
// Exactly source select_candidate_blocks: causal block maxima, latest pinned,
// top2048 blocks, drop unreachable choices, expand each selected block by8.
extern "C" __global__ void dsv41_candidate_blocks(
    const __nv_bfloat16* scores,const unsigned* visible_counts,unsigned char* output,
    unsigned* status,unsigned columns){
    const unsigned row=blockIdx.x,tid=threadIdx.x,blocks=(columns+7)/8;
    const __nv_bfloat16* input=scores+(uint64_t)row*columns;
    unsigned char* mask=output+(uint64_t)row*columns;
    unsigned visible=visible_counts[row];
    if(visible>columns){if(!tid)atomicOr(status,BAD_EXTENT);visible=columns;}
    __shared__ unsigned short keys[16384];
    __shared__ unsigned hist[256],boundary,remaining,tie;
    for(unsigned i=tid;i<columns;i+=blockDim.x)if(!isfinite(__bfloat162float(input[i])))atomicOr(status,BAD_NUMERIC);
    for(unsigned block=tid;block<blocks;block+=blockDim.x){
        unsigned maximum=0;
        for(unsigned j=0;j<8;++j){unsigned i=block*8+j;
            if(i<visible&&isfinite(__bfloat162float(input[i])))maximum=max(maximum,ordered(input[i]));
        }
        if(visible&&block==(visible-1)/8)keys[block]=0xff80u; // numeric +Inf sentinel.
        else keys[block]=(unsigned short)maximum;
    }
    __syncthreads();
    cutoff(SharedKeys{keys},blocks,2048,hist,&boundary,&remaining,&tie);
    for(unsigned i=tid;i<columns;i+=blockDim.x)mask[i]=selected(keys[i/8],i/8,boundary,tie);
}
// Source final topk is score-selected then POSITION-sorted; unreachable choices
// become -1 after sorting. The fixed2048x8 candidate budget guarantees at least
// min(512,visible) eligible positions; inconsistent supplied masks fail closed.
extern "C" __global__ void dsv41_topk_positions(
    const __nv_bfloat16* scores,const unsigned* visible_counts,const unsigned char* candidates,
    int* output,unsigned* status,unsigned columns,unsigned offset){
    const unsigned row=blockIdx.x,tid=threadIdx.x,wanted=min(columns,512u);
    const __nv_bfloat16* input=scores+(uint64_t)row*columns;
    unsigned visible=visible_counts[row];
    if(visible>columns){if(!tid)atomicOr(status,BAD_EXTENT);visible=columns;}
    PositionKeys keys{input,candidates?candidates+(uint64_t)row*columns:nullptr,visible};
    __shared__ unsigned hist[256],boundary,remaining,tie,count;
    __shared__ unsigned positions[512];
    if(!tid)count=0;if(tid<512)positions[tid]=0xffffffffu;
    for(unsigned i=tid;i<columns;i+=blockDim.x)if(!isfinite(__bfloat162float(input[i])))atomicOr(status,BAD_NUMERIC);
    __syncthreads();
    cutoff(keys,columns,wanted,hist,&boundary,&remaining,&tie);
    for(unsigned i=tid;i<columns;i+=blockDim.x)if(selected(keys.key(i),i,boundary,tie)){
        unsigned slot=atomicAdd(&count,1u);if(slot<512)positions[slot]=i;else atomicOr(status,BAD_EXTENT);
    }
    __syncthreads();
    if(!tid&&count<min(wanted,visible))atomicOr(status,BAD_CANDIDATES);
    // Atomic collection order is intentionally erased by an integer-only sort.
    for(unsigned width=2;width<=512;width<<=1)for(unsigned distance=width>>1;distance;distance>>=1){
        if(tid<512){unsigned other=tid^distance;if(other>tid){
            bool ascending=(tid&width)==0;
            if((positions[tid]>positions[other])==ascending){unsigned value=positions[tid];positions[tid]=positions[other];positions[other]=value;}
        }}
        __syncthreads();
    }
    if(tid<wanted)output[(uint64_t)row*wanted+tid]=positions[tid]==0xffffffffu?-1:(int)(positions[tid]+offset);
}
