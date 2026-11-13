#!/usr/bin/env python3
"""CPU-only 1017 FP4 cache vectors; explicit RNE table, Torch scale/BF16 casts.

CPU Torch does not implement native FP4 casts. These vectors therefore expose
their table-based FP4 provenance; the separate oracle must test real GPU casts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
LEVELS=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)


def encode(values):
    flat=np.array(values,dtype=np.float32).reshape(-1)
    assert np.isfinite(flat).all()
    distance=np.abs(np.minimum(np.abs(flat),np.float32(6))[:,None]-LEVELS[None,:])
    chosen=np.argmin(distance,axis=1)
    for i,code in enumerate(chosen):
        if code%2 and code<7 and distance[i,code]==distance[i,code+1]: chosen[i]+=1
    return chosen.astype(np.uint8)|(np.signbit(flat).astype(np.uint8)<<3)


def case(name,format,rows,columns,values=None,bits=None):
    if bits is not None:
        tensor=torch.tensor(bits,dtype=torch.uint16).view(torch.bfloat16).reshape(rows,columns)
    else:
        tensor=torch.tensor(values,dtype=torch.float32,device='cpu').reshape(rows,columns).to(torch.bfloat16)
    raw=tensor.float().numpy();group=16 if format=='compressed_kv' else 32
    shaped=raw.reshape(-1,group);maximum=np.max(np.abs(shaped),axis=1)
    if format=='compressed_kv':
        divided=np.maximum(maximum,np.float32(6*2**-9))/np.float32(6)
        # CUDA source emits cvt.rn.satfinite.e4m3. Raw CPU Torch overflow produces
        # NaN instead; pre-clamp reproduces that documented saturation boundary.
        scale_tensor=torch.from_numpy(np.minimum(divided,np.float32(448))).to(torch.float8_e4m3fn)
        scale_bytes=scale_tensor.view(torch.uint8).tolist();scale=scale_tensor.float().numpy()
    else:
        divided=np.maximum(maximum,np.float32(6*2**-126))*np.float32(1/6)
        rawbits=divided.view(np.uint32)
        exponent=((rawbits>>23)&255).astype(np.int32)-127+((rawbits&0x7fffff)!=0)
        code=(exponent+127).astype(np.uint32)
        scale=(code<<23).view(np.float32)
        scale_tensor=torch.from_numpy(scale.copy()).to(torch.float8_e8m0fnu)
        scale_bytes=scale_tensor.view(torch.uint8).tolist()
        assert scale_bytes==code.tolist()
    normalized=(shaped/scale[:,None]).reshape(-1)
    codes=encode(normalized)
    packed=(codes[0::2]|(codes[1::2]<<4)).astype(np.uint8)
    restored=LEVELS[codes&7].copy()
    restored=np.copysign(restored,np.where(codes&8,np.float32(-1),np.float32(1))).astype(np.float32)
    restored=(restored.reshape(-1,group)*scale[:,None]).reshape(rows,columns)
    qat=torch.from_numpy(restored).to(torch.bfloat16)
    assert torch.isfinite(qat).all()
    return {'name':name,'format':format,'rows':rows,'columns':columns,
        'input_bf16_bits':tensor.view(torch.uint16).reshape(-1).tolist(),
        'packed_bytes':packed.tolist(),'scale_bytes':scale_bytes,'qat_bf16_bits':qat.view(torch.uint16).reshape(-1).tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kernel-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert hashlib.sha256(a.kernel_source.read_bytes()).hexdigest()==KERNEL_SHA
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(False)
    assert not torch.cuda.is_initialized()
    cases=[]
    zeros=[0x8000 if i%2 else 0 for i in range(64)]
    cases.append(case('compressed_zero_signs','compressed_kv',2,32,bits=zeros))
    base=np.array([0,-0.,.25,-.25,.75,-.75,1.25,-1.25,1.75,-1.75,2.5,-2.5,3.5,-3.5,5,6],dtype=np.float32)
    mixed=np.concatenate([base*np.float32(2.**((g%18)-9)) for g in range(3*512//16)])
    cases.append(case('compressed_hidden512_dynamic_groups','compressed_kv',3,512,values=mixed))
    tie=np.concatenate([np.linspace(-m,m,16,dtype=np.float32)for m in [6.375,7.125,6.34375,6.40625]])
    cases.append(case('compressed_scale_RNE_boundaries','compressed_kv',1,64,values=tie))
    cases.append(case('compressed_saturating_scale','compressed_kv',1,16,bits=[0x7f7f,0xff7f]+[0x8000,0]*7))
    cases.append(case('compressed_bf16_subnormals','compressed_kv',1,32,bits=[0,0x8000,1,0x8001,127,0x807f,128,0x8080]*4))
    cases.append(case('index_zero_signs','index_qk',2,32,bits=zeros))
    index=np.concatenate([np.tile(base,2)*np.float32(2.**((g%18)-9))for g in range(3*128//32)])
    cases.append(case('index_head128_dynamic_groups','index_qk',3,128,values=index))
    cases.append(case('index_scale_floor_subnormals','index_qk',1,64,bits=[0,0x8000,1,0x8001,127,128,256,257]*8))
    threshold=np.concatenate([np.linspace(-m,m,32,dtype=np.float32)for m in [6.,6.03125,12.,11.9375]])
    cases.append(case('index_pow2_thresholds','index_qk',1,128,values=threshold))
    cases.append(case('index_large_safe_scale','index_qk',1,32,values=np.tile(base,2)*np.float32(2.**120)))
    scalars=[np.float32(0),np.float32(-0.),np.float32(8),np.float32(-8)]
    for mid in [.25,.75,1.25,1.75,2.5,3.5,5.]:
        for sign in [1,-1]:
            value=np.float32(sign*mid)
            scalars.extend([np.nextafter(value,np.float32(-np.inf)),value,np.nextafter(value,np.float32(np.inf))])
    scalars=np.array(scalars,dtype=np.float32)
    report={'schema_version':1,'kind':'official1017_FP4_cache_CPU_vectors','kernel_reference_sha256':KERNEL_SHA,
        'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu','gpu_initialized':False,
        'FP4_reference':'explicit E2M1 levels and RNE tie parity; CPU Torch FP4 casts unavailable; separate GPU native-cast oracle required',
        'scale_reference':'Torch CPU E4M3/UE8M0 casts; E4M3 preclamp448 explicitly models CUDA rn.satfinite; FP32 source floors/divide/pow2',
        'BF16_reference':'Torch CPU BF16 rounding of decoded native FP4*stored scale; cache negative zero preserved',
        'cases':cases,'scalar_cast':{'input_f32_bits':scalars.view(np.uint32).tolist(),'expected_nibbles':encode(scalars).tolist()}}
    assert not torch.cuda.is_initialized()
    data=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();a.output.write_bytes(data)
    print(json.dumps({'path':str(a.output),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'cases':len(cases),'scalar_cases':len(scalars),'GPU_work':False}))


if __name__=='__main__':main()
