"""CPU-only1017 Gate.forward/Expert.forward AST goldens with supplied projections.

Usage: python generate_dsv41_moe_vectors.py VERIFIED/model.py NEW.json
Dense linears, learned experts and the model are not executed or qualified.
"""
import ast, hashlib, json, sys, types
from pathlib import Path
import torch
import torch.nn.functional as F
SOURCE_SHA='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
source=Path(sys.argv[1]).read_bytes();assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
torch.set_default_device('cpu');torch.set_num_threads(1);torch.set_flush_denormal(False)
tree=ast.parse(source);space={'torch':torch,'F':F,'linear':lambda x,w:x}
for cls in ['Gate','Expert']:
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 f.name=cls+'_forward'
 exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'verified1017_'+cls+'.forward','exec'),space)
def fs(x):return x.detach().contiguous().reshape(-1).tolist()
def bs(x):return fs(x.view(torch.uint16))
route=[]
for name,rows in [('unbiased',1),('selection_bias',2),('softplus_thresholds',1),('all_underflow',1),('tiny_score_eps',1),('ties',2)]:
 i=torch.arange(rows*384,dtype=torch.int64).reshape(rows,384)
 logits=(((i*37+11)%769-384).float()*0.0625)
 bias=torch.zeros(384,dtype=torch.float32)
 if name=='selection_bias':bias[:12]=torch.arange(12,dtype=torch.float32)*8+10
 if name=='softplus_thresholds':
  logits[:]=-40;logits[0,:12]=torch.tensor([-100,-20,-1,0,1,19.9,20,20.1,30,40,60,80])
 if name=='all_underflow':logits.fill_(-200)
 if name=='tiny_score_eps':logits.fill_(-100)
 if name=='ties':logits.fill_(0);bias[:8]=1
 fake=types.SimpleNamespace(weight=torch.empty(1),gate_temp=1.,score_func='sqrtsoftplus',bias=bias,bias_vl=None,topk=6,norm_topk_prob=True,route_scale=1.5)
 weights,ids=space['Gate_forward'](fake,logits)
 scores=F.softplus(logits).sqrt();selection=scores+bias
 route.append(dict(name=name,rows=rows,experts=384,topk=6,logits=fs(logits),bias=fs(bias),weights=fs(weights),indices=fs(ids),scores=fs(scores),
  tie_policy='admissible top-k witness; exact Torch IDs unspecified' if name in ['ties','all_underflow','tiny_score_eps'] else 'unambiguous exact expert IDs',
  selected_min=fs(selection.gather(1,ids).min(dim=1).values)))
swiglu=[]
for name,rows,columns,weighted in [('clamp_and_signs',1,16,True),('weight_before_BF16_round',2,32,True),('shared_unweighted',2,32,False),('negative_gate_unclamped',1,8,True),('hidden2304',2,2304,True)]:
 i=torch.arange(rows*columns,dtype=torch.int64)
 gate=(((i*7+3)%47-23).float()*0.75).to(torch.bfloat16).reshape(rows,columns)
 up=(((i*13+5)%53-26).float()*0.625).to(torch.bfloat16).reshape(rows,columns)
 if name=='negative_gate_unclamped':gate.fill_(-20);up.fill_(12)
 weight=torch.tensor([0.33333334+row*0.12123457 for row in range(rows)],dtype=torch.float32).reshape(rows,1) if weighted else None
 fake=types.SimpleNamespace(w1=lambda x:gate,w3=lambda x:up,w2=lambda x:x,swiglu_limit=10.)
 output=space['Expert_forward'](fake,torch.zeros(rows,columns,dtype=torch.bfloat16),weight)
 current=F.silu(gate.float().clamp(max=10))*up.float().clamp(-10,10)
 wrong_early=(current.to(torch.bfloat16).float()*(weight if weight is not None else 1)).to(torch.bfloat16)
 wrong_symmetric=(F.silu(gate.float().clamp(-10,10))*up.float().clamp(-10,10)*(weight if weight is not None else 1)).to(torch.bfloat16)
 wrong_no_up=(F.silu(gate.float().clamp(max=10))*up.float()*(weight if weight is not None else 1)).to(torch.bfloat16)
 swiglu.append(dict(name=name,rows=rows,columns=columns,gate_bits=bs(gate),up_bits=bs(up),weights=fs(weight) if weighted else None,output_bits=bs(output),
   negative_early_BF16_mismatches=int((output.view(torch.uint16)!=wrong_early.view(torch.uint16)).sum()),
   negative_symmetric_gate_mismatches=int((output.view(torch.uint16)!=wrong_symmetric.view(torch.uint16)).sum()),
   negative_missing_up_clamp_mismatches=int((output.view(torch.uint16)!=wrong_no_up.view(torch.uint16)).sum())))
assert next(c for c in swiglu if c['name']=='weight_before_BF16_round')['negative_early_BF16_mismatches']>0
assert next(c for c in swiglu if c['name']=='negative_gate_unclamped')['negative_symmetric_gate_mismatches']>0
record=dict(schema_version=1,kind='official1017_MoE_CPU_AST_goldens',model_source_sha256=SOURCE_SHA,torch_version=torch.__version__,device='cpu',
 constants=dict(experts=384,topk=6,gate_temp=1.,norm_eps=1e-20,route_scale=1.5,swiglu_limit=10.),
 scope='Gate supplied denseF32logits; Expert supplied BF16w1/w3 and identityw2; shared Expert(weights=None); no learned parameters or denseGEMMs',route_cases=route,swiglu_cases=swiglu)
p=Path(sys.argv[2]);
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,route_cases=len(route),swiglu_cases=len(swiglu),GPU_work=False)))
