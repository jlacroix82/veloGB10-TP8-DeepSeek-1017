// PRIVATE additive row-layout helpers only. No floating reduction, precision
// conversion, source projection or attention approximation is performed here.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#ifndef DSV41_BATCH_LAYOUT_SOURCE_ID
#error Dedicated DSV41_BATCH_LAYOUT_SOURCE_ID required
#endif
extern "C" __global__ void dsv41_batch_layout_identity(uint64_t*out){
    if(!blockIdx.x&&!threadIdx.x){out[0]=DSV41_BATCH_LAYOUT_SOURCE_ID;out[1]=1;}
}
// metadata[row] = {tight-index offset, actualtopk, laneKV rows}. Indices remain
// LOCAL to their lane's immutable KV slice. Packed attention chooses the KV
// base once from separate row metadata; no second compressed/window shift.
extern "C" __global__ void dsv41_batch_pad_indices(
    int*out,const int*tight,const uint64_t*metadata,unsigned*status,
    uint64_t rows,uint64_t max_topk,uint64_t tight_count){
    if(!rows||rows>128||!max_topk||max_topk>640||!tight_count||tight_count>128*640){atomicOr(status,2u);return;}
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*max_topk)return;
    uint64_t row=i/max_topk,j=i%max_topk;
    uint64_t first=metadata[row*3],k=metadata[row*3+1],kvrows=metadata[row*3+2];
    if(!k||k>max_topk||first>tight_count||k>tight_count-first||!kvrows||kvrows>8447){atomicOr(status,2u);out[i]=-1;return;}
    int value=j<k?tight[first+j]:-1;
    if(value < -1 || (value>=0&&(uint64_t)value>=kvrows)){atomicOr(status,4u);value=-1;}
    out[i]=value;
}
// Select final temporal rows for a single shared B-lane head projection.
// outrow order is the caller's explicit physical-lane association, not slot ID.
extern "C" __global__ void dsv41_batch_gather_bf16(
    __nv_bfloat16*out,const __nv_bfloat16*input,const uint32_t*selected_rows,
    unsigned*status,uint64_t input_rows,uint64_t output_rows,uint64_t width){
    if(!input_rows||input_rows>128||!output_rows||output_rows>128||!width||width>32768){atomicOr(status,2u);return;}
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=output_rows*width)return;
    uint64_t source=selected_rows[i/width];
    if(source>=input_rows){atomicOr(status,4u);out[i]=__float2bfloat16_rn(0.f);return;}
    __nv_bfloat16 value=input[source*width+i%width];
    if(!isfinite(__bfloat162float(value))){atomicOr(status,1u);value=__float2bfloat16_rn(0.f);}
    out[i]=value;
}
// Source semantic slots, not inferred from the current segment length. phase0
// is an explicitly full-prompt-bound PREFILL; phase1 is one-token DECODE.
// This initial exact path selects all physically available compressed keys;
// learned selection above512 stored rows is refused instead of approximated.
extern "C" __global__ void dsv41_batch_source_indices(
    int*out,unsigned*status,uint64_t start,uint64_t tokens,uint64_t ratio,
    uint64_t window_rows,uint64_t compressed_rows,uint64_t logical_window,
    uint64_t logical_compressed,uint64_t phase){
    uint64_t previous=min(start,(uint64_t)127),base=start-previous;
    if(!tokens||tokens>128||start>=8192||tokens>8192-start
       ||window_rows!=previous+tokens||!logical_window||logical_window>128
       ||logical_compressed>512||compressed_rows>512||phase>1
       ||(ratio!=0&&ratio!=1&&ratio!=2)
       ||(ratio==0?(compressed_rows||logical_compressed):compressed_rows!=(start+tokens)/ratio)
       ||(phase==1&&(tokens!=1||!start||logical_window!=128))){atomicOr(status,2u);return;}
    uint64_t width=logical_window+logical_compressed;
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*width)return;
    uint64_t q=i/width,j=i%width,p=start+q;
    if(j<logical_window){
        uint64_t global;bool valid=true;
        if(phase==1&&p<127){global=(p+1+j)%128;valid=global<=p;}
        else{global=(p+1>128?p+1-128:0)+j;valid=global<=p;}
        if(!valid){out[i]=-1;return;}
        if(global<base||global-base>=window_rows){atomicOr(status,4u);out[i]=-1;return;}
        out[i]=(int)(global-base);
    }else{
        uint64_t col=j-logical_window,visible=(p+1)/ratio;
        if(visible>compressed_rows){atomicOr(status,4u);out[i]=-1;return;}
        out[i]=col<visible?(int)(window_rows+col):-1;
    }
}
