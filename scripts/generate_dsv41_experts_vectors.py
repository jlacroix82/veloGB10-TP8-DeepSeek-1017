#!/usr/bin/env python3
"""Independent CPU-only NumPy reference for native1017 grouped routed experts.
No checkpoint payload, CUDA import, device allocation, or serving is performed.
Native FP4+per32 UE8 scales and E4M3/UE8 activation quantization are explicit.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FP4=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
E4=[]
for c in range(127):
    exp=(c>>3)&15;mant=c&7
    E4.append(np.float32(mant*2**-9 if exp==0 else (1+mant/8)*2**(exp-7)))
E4=np.array(E4,dtype=np.float32)
def f32(x):return np.asarray(x,dtype=np.float32)
def b16(x):
    v=f32(x);bits=v.view(np.uint32);return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def words(x):return f32(x).view(np.uint32).reshape(-1).tolist()
def quant(x):
    x=f32(x);rows,k=x.shape;g=x.reshape(rows,k//32,32)
    amax=np.maximum(np.max(np.abs(g),axis=2),np.float32(1e-4))
    divided=(amax*np.float32(1/448)).astype(np.float32)
    bits=divided.view(np.uint32);codes=((bits>>23)+((bits&0x7fffff)!=0)).astype(np.uint8)
    scales=(codes.astype(np.uint32)<<23).view(np.float32)
    normalized=np.clip((g/scales[:,:,None]).astype(np.float32),-448,448)
    absolute=np.abs(normalized);hi=np.searchsorted(E4,absolute);hi=np.minimum(hi,126);lo=np.maximum(hi-1,0)
    dlo=absolute-E4[lo];dhi=E4[hi]-absolute
    out=np.where((dlo<dhi)|((dlo==dhi)&((lo&1)==0)),lo,hi).astype(np.uint8)
    out |= (np.signbit(normalized).astype(np.uint8)<<7)
    decoded=E4[out&127]*np.where(out&128,np.float32(-1),np.float32(1))
    return decoded.reshape(rows,k),scales,codes,out.reshape(rows,k)
def unpack(w,shape):
    b=np.array(w,dtype=np.uint8);out=np.empty(b.size*2,dtype=np.uint8);out[0::2]=b&15;out[1::2]=b>>4
    d=FP4[out&7]*np.where(((out&8)!=0)&((out&7)!=0),np.float32(-1),np.float32(1))
    return d.reshape(shape)
def linear(x,w,s,out,early=False):
    rows,k=x.shape;a,asc,_,_=quant(x);weight=unpack(w,(out,k));scale=(np.array(s,dtype=np.uint32)<<23).view(np.float32).reshape(out,k//32)
    acc=np.zeros((rows,out),dtype=np.float32)
    for group in range(k//32):
        # Products and raw sums in this dyadic family are exactly representable.
        raw=(a[:,None,group*32:(group+1)*32]*weight[None,:,group*32:(group+1)*32]).sum(axis=2,dtype=np.float32)
        exact=(a[:,None,group*32:(group+1)*32].astype(np.float64)*weight[None,:,group*32:(group+1)*32].astype(np.float64)).sum(axis=2)
        assert np.array_equal(raw.astype(np.float64),exact), 'fixture raw dot is not exact; cannot assume MMA reduction-order independence'
        corrected=((raw*asc[:,group,None]).astype(np.float32)*scale[None,:,group]).astype(np.float32)
        acc=(acc+corrected).astype(np.float32)
        if early:acc=b16(acc)
    return b16(acc)
def weight_pack(expert,branch,out,k,cancel=False):
    if cancel:expert=0
    n=np.arange(out*k,dtype=np.uint64)
    codes=((n*17+(n//k)*7+expert*13+branch*19+(n%5)*3)%16).astype(np.uint8)
    packed=(codes[0::2]|(codes[1::2]<<4)).tolist()
    n=np.arange(out*(k//32),dtype=np.uint64)
    scales=(122+(n*7+expert+branch)%5).astype(np.uint8)
    return packed,scales.tolist()
def run(case,mode=None):
    dim,inter=case['dim'],case['inter'];x=np.array(case['input_words'],dtype=np.uint32).view(np.float32).reshape(-1,dim)
    ids=np.array(case['route_ids'],dtype=np.uint32).reshape(-1,6);rw=f32(case['route_weights']).reshape(-1,6)
    y=np.zeros_like(x,dtype=np.float32)
    experts=case['experts'];order=sorted(experts,key=lambda e:e['id'],reverse=mode=='reverse_sum')
    for e in order:
        rows,top=np.where(ids==e['id'])
        if not len(rows):continue
        gate=linear(x[rows],e['w1'],e['s1'],inter,early=mode=='early_projection')
        up=linear(x[rows],e['w3'],e['s3'],inter,early=mode=='early_projection')
        g=np.minimum(gate,np.float32(10));u=np.clip(up,-10,10)
        with np.errstate(over='ignore',under='ignore'):
            h=((g/(np.float32(1)+np.exp(-g,dtype=np.float32))).astype(np.float32)*u).astype(np.float32)
        if mode=='route_after_round':h=b16(h)
        h=(h*rw[rows,top,None]).astype(np.float32)
        projected=linear(b16(h),e['w2'],e['s2'],dim,early=mode=='early_projection')
        y[rows]=(y[rows]+projected).astype(np.float32)
    return y
patterns=([0,17,47,48,100,333],[17,48,100,200,300,333],[0,47,96,150,250,350],[48,96,150,200,250,350])
def case(cancel):
    dim,inter,rows=64,32,16
    x=b16((np.arange(rows*dim,dtype=np.float32).reshape(rows,dim)%37-18)/64)
    ids=[];weights=[];logits=np.full((rows,384),36,dtype=np.float32);bias=np.zeros(384,dtype=np.float32)
    for row in range(rows):
        selected=patterns[0] if cancel else patterns[row%4]
        for j,idx in enumerate(selected):logits[row,idx]=36 if cancel else (j+7)**2
        ordered=sorted(selected) if cancel else list(reversed(selected))
        ids.extend(ordered)
        if cancel:weights.extend([.25]*6)
        else:weights.extend([np.float32(np.float32(j/57)*np.float32(1.5)).item()for j in range(12,6,-1)])
    if cancel:bias[list(patterns[0])]=100
    experts=[]
    for expert in (0,17,47):
        e={'id':expert}
        for branch,(name,out,k) in enumerate([('1',inter,dim),('3',inter,dim),('2',dim,inter)]):
            w,s=weight_pack(expert,branch,out,k,cancel)
            if cancel and name=='2' and expert in (0,17):
                s=[v+30 for v in s]
                if expert==17:w=[v^0x88 for v in w]
            e['w'+name]=w;e['s'+name]=s
        experts.append(e)
    c={'name':'cancellation' if cancel else 'varied_routes','dim':dim,'inter':inter,'rows':rows,
       'input_words':words(x),'route_ids':ids,'route_weights':weights,'logits_words':words(logits),'bias_words':words(bias),'experts':experts}
    ref=run(c);c['expected_words']=words(ref);c['expected_sha256']=hashlib.sha256(ref.tobytes()).hexdigest()
    negatives={}
    for mode in ('early_projection','route_after_round','reverse_sum'):
        bad=run(c,mode);negatives[mode]=int(np.count_nonzero(ref.view(np.uint32)!=bad.view(np.uint32)))
    c['negative_mismatch_counts']=negatives
    return c
cases=[case(False),case(True)]
assert cases[0]['negative_mismatch_counts']['early_projection']>0
assert cases[0]['negative_mismatch_counts']['route_after_round']>0
assert cases[1]['negative_mismatch_counts']['reverse_sum']>0
fixture={'schema_version':1,'kind':'independent_CPU_native1017_grouped_experts','model_free':True,'CUDA_used':False,
    'reference_model_sha256':'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d',
    'reference_kernel_sha256':'1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'schedule':'native FP4 w1/w3 complete BF16; F32 clamp/SwiGLU/route then BF16; group32 E4 activation quant; native FP4 w2 complete BF16; ascending expert F32 sum',
    'precision':'dyadic native raw products are exact; explicit NumPy FP32 K32 scaling/add, CPU exp then complete BF16; actual unfused qualified CUDA path independently compared',
    'cases':cases}
path=ROOT/'tests/fixtures/dsv41_experts_vectors.json';path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'negative_controls':{c['name']:c['negative_mismatch_counts']for c in cases},'GPU_used':False}))
