"""CPU-only1017 mHC goldens: exact Block methods + TileLang source algorithm.

Usage: python generate_dsv41_hc_vectors.py verified/model.py verified/kernel.py NEW.json
No GPU, model, weights, imports of the official GPU inference module or network.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import torch
import torch.nn.functional as F

MODEL_SHA = 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
KERNEL_SHA = '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
model = Path(sys.argv[1]).read_bytes()
kernel = Path(sys.argv[2]).read_bytes()
assert hashlib.sha256(model).hexdigest() == MODEL_SHA
# Bind the actual kernel source from the verified retained source manifest.
manifest = json.loads((Path(sys.argv[1]).parent.parent / 'source-manifest.json').read_text())
assert manifest['revision'] == '1bae0e1088993b9b82de08f59de5be5a34e42170'
assert manifest['files']['inference/model.py']['sha256'] == MODEL_SHA
assert manifest['files']['inference/kernel.py']['sha256'] == KERNEL_SHA
assert hashlib.sha256(kernel).hexdigest() == KERNEL_SHA
assert b'for _ in T.serial(sinkhorn_iters - 1)' in kernel
assert b'comb_frag[j, k] / (col_sum[k] + eps)' in kernel
assert b'comb_frag[j, k] / row_sum[j] + eps' in kernel

torch.set_default_device('cpu')
torch.set_num_threads(1)
torch.set_flush_denormal(False)

def split(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Literal FP32 tensor translation of hc_split_sinkhorn_kernel, not TileLang execution."""
    assert hc_mult == 4 and sinkhorn_iters == 20 and eps == 1e-6
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb

namespace = {'torch':torch, 'F':F, 'hc_split_sinkhorn':split}
tree = ast.parse(model)
block = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Block')
methods = [n for n in block.body if isinstance(n,ast.FunctionDef) and n.name in ['hc_mixes','hc_pre','hc_post','forward']]
identity = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_identity_pre_mix')
exec(compile(ast.fix_missing_locations(ast.Module(body=methods+[identity],type_ignores=[])), 'verified_1017_Block_methods','exec'),namespace)

def bits(t): return t.detach().contiguous().view(torch.uint16).reshape(-1).tolist()
def floats(t): return t.detach().contiguous().reshape(-1).tolist()
def shaped_values(n,seed,scale):
    return (((torch.arange(n,dtype=torch.int64)*7+seed)%31-15).float()*scale).to(torch.bfloat16)
def make_weights(dim,seed):
    # Eight exactly represented nonzero values per output; no serialized model data.
    result=torch.zeros(24,4*dim,dtype=torch.float32)
    for o in range(24):
        for j in range(8):
            c=(o*11+j*17+seed)%(4*dim)
            result[o,c] += ((o*3+j+seed)%9-4)*0.03125
    return result

def fake(dim,seed):
    obj=types.SimpleNamespace(norm_eps=1e-20,hc_mult=4,hc_sinkhorn_iters=20,hc_eps=1e-6)
    for name in ['hc_mixes','hc_pre','hc_post']:
        setattr(obj,name,types.MethodType(namespace[name],obj))
    for name,off in [('attn',seed),('ffn',seed+5)]:
        setattr(obj,'hc_'+name+'_fn',make_weights(dim,off))
        setattr(obj,'hc_'+name+'_scale',torch.tensor([0.75,-0.5,1.25],dtype=torch.float32))
        setattr(obj,'hc_'+name+'_base',((torch.arange(24,dtype=torch.float32)+off)%9-4)*0.125)
    obj.attn_norm=lambda x:x
    obj.ffn_norm=lambda x:x
    obj.attn=lambda x,*args: (x.float()*0.5+0.0625).to(torch.bfloat16)
    obj.ffn=lambda x,*args: (x.float()*-0.25-0.125).to(torch.bfloat16)
    return obj

cases=[]
for name,tokens,dim,seed,tiny in [('signed_distinct_copies',1,8,3,False),('multi_token',2,32,7,False),('epsilon_floor',1,8,11,True),('hidden5120',1,5120,13,False)]:
    h=shaped_values(tokens*4*dim,seed,2**-40 if tiny else 0.125).view(tokens,1,4,dim)
    x=shaped_values(tokens*dim,seed+2,0.0625).view(tokens,1,dim)
    f=fake(dim,seed)
    pre,post,comb=f.hc_mixes(h,f.hc_attn_fn,f.hc_attn_scale,f.hc_attn_base)
    initial=namespace['make_identity_pre_mix'](h,4)
    cases.append(dict(name=name,tokens=tokens,dim=dim,weight_recipe_seed=seed,
        h_bits=bits(h),sublayer_bits=bits(x),scale=floats(f.hc_attn_scale),base=floats(f.hc_attn_base),
        pre=floats(pre),post=floats(post),comb=floats(comb),initial_pre=floats(initial),
        collapsed_initial_bits=bits(f.hc_pre(h,initial)),collapsed_current_bits=bits(f.hc_pre(h,pre)),
        expanded_bits=bits(f.hc_post(x,h,post,comb))))

# Exact official forward ordering for two distinct layers. Attention/FFN are
# controlled affine fixtures, so this is not a model/attention/MoE qualification.
dim=8; h=shaped_values(4*dim,23,0.125).view(1,1,4,dim)
premix=namespace['make_identity_pre_mix'](h,4)
chain=[]; original=h.clone()
for seed in [19,29]:
    f=fake(dim,seed)
    h,premix=namespace['forward'](f,h,0,premix,None)
    chain.append(dict(weight_recipe_seed=seed,h_bits=bits(h),next_pre=floats(premix),final_collapsed_bits=bits(f.hc_pre(h,premix))))
record=dict(kind='official1017_mHC_CPU_goldens',schema_version=1,
    model_source_sha256=MODEL_SHA,kernel_source_sha256=KERNEL_SHA,torch_version=torch.__version__,device='cpu',
    source_methods=['Block.hc_mixes','Block.hc_pre','Block.hc_post','Block.forward','make_identity_pre_mix'],
    coefficient_oracle='FP32 PyTorch CPU translation of verified TileLang hc_split_sinkhorn_kernel; actual TileLang not executed',
    constants=dict(hc_mult=4,sinkhorn_iters=20,hc_eps=1e-6,norm_eps=1e-20),
    parameter_recipe='8 sparse F32 entries per output, deterministic make_weights; synthetic, no model weights',
    cases=cases,chain=dict(tokens=1,dim=dim,h_bits=bits(original),layers=chain,
        sublayer_fixtures='attention BF16(x*0.5+0.0625); FFN BF16(x*-0.25-0.125); norms identity'))
target=Path(sys.argv[3])
with target.open('x') as handle: json.dump(record,handle,indent=2);handle.write('\n')
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),cases=len(cases),layers=2,GPU_work=False)))
