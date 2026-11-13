"""CPU-only official1017 embedding and window-ring AST goldens.
Usage: python generate_dsv41_basicops_vectors.py VERIFIED/model.py NEW.json
Rope/quantization/projections are supplied identity fixtures, separately qualified.
"""
import ast,hashlib,json,sys,types
from pathlib import Path
import torch
import torch.nn.functional as F
SOURCE='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
raw=Path(sys.argv[1]).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE
torch.set_default_device('cpu');torch.set_num_threads(1);torch.set_flush_denormal(False)
space={'torch':torch,'F':F,'world_size':8,'dist':types.SimpleNamespace(all_reduce=lambda x:None),
 'apply_rotary_emb':lambda x,f:x,'act_quant':lambda x,*a:x,'fp8_block_size':32,'scale_fmt':None,'scale_dtype':None,
 'get_window_topk_idxs':lambda *a:None}
tree=ast.parse(raw)
for cls,name,alias in [('ParallelEmbedding','forward','embed'),('Attention','_window_kv','ring')]:
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 method=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name);method.name=alias
 exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'verified1017_'+cls+'.'+name,'exec'),space)
def bits(t):return t.detach().contiguous().view(torch.uint16).reshape(-1).tolist()
def fbits(t):return t.detach().contiguous().view(torch.uint32).reshape(-1).tolist()
ids=[r*16160+j for r in range(8) for j in [0,1,16159]]+[0,16159,16160,129279];dim=8;partials=[]
for rank in range(8):
 row=torch.arange(rank*16160,(rank+1)*16160,dtype=torch.int64).view(-1,1);col=torch.arange(dim,dtype=torch.int64).view(1,-1)
 weights=(((row*13+col*7)%61*2-59).float()/16).to(torch.bfloat16)
 fake=types.SimpleNamespace(weight=weights,vocab_start_idx=rank*16160,vocab_end_idx=(rank+1)*16160)
 partial=space['embed'](fake,torch.tensor(ids,dtype=torch.int64));partials.append(partial.float())
reduced=torch.stack(partials).sum(dim=0);expanded=reduced.to(torch.bfloat16).unsqueeze(1).repeat(1,4,1)
embedding=dict(vocab=129280,world=8,dim=dim,ids=ids,weight_recipe='BF16((((globalrow*13+col*7)%61)*2-59)/16)',partial_f32_bits=[fbits(p) for p in partials],reduced_f32_bits=fbits(reduced),expanded_hc4_bf16_bits=bits(expanded))
ring=[]
for count in [1,127,128,129,256,257]:
 lanes=2;markers=(((torch.arange(lanes*count,dtype=torch.int64)*7)%101-50).float()/16).to(torch.bfloat16).view(lanes,count)
 x=markers.unsqueeze(-1).repeat(1,1,512);cache=torch.full((lanes,128,512),-99.,dtype=torch.bfloat16)
 fake=types.SimpleNamespace(window_size=128,wkv=lambda x:x,kv_norm=lambda x:x,rope_head_dim=64,window_kv_cache=cache)
 space['ring'](fake,x,None,0)
 ring.append(dict(name='prefill'+str(count),lanes=lanes,count=count,start_pos=0,input_row_bf16_bits=bits(markers),initial_marker=-99.,expected_cache_row_bf16_bits=bits(cache[:,:,0])))
for start in [1,127,128,129,257,999999]:
 lanes=2;cache=((torch.arange(lanes*128,dtype=torch.int64)%127-63).float()/8).to(torch.bfloat16).view(lanes,128,1).repeat(1,1,512)
 initial=bits(cache[:,:,0]);x=torch.tensor([[[1.125]],[[2.375]]],dtype=torch.bfloat16).repeat(1,1,512)
 fake=types.SimpleNamespace(window_size=128,wkv=lambda x:x,kv_norm=lambda x:x,rope_head_dim=64,window_kv_cache=cache)
 space['ring'](fake,x,None,start)
 ring.append(dict(name='append'+str(start),lanes=lanes,count=1,start_pos=start,input_row_bf16_bits=bits(x[:,:,0]),initial_cache_row_bf16_bits=initial,expected_cache_row_bf16_bits=bits(cache[:,:,0])))
values=torch.tensor([0.,-0.,1.,-1.,1+2**-8,1+3*2**-8,-1-2**-8,2**-126,2**-127,2**-133,2**-134,3*2**-134,-2**-133,3.25,-123.5],dtype=torch.float32)
cast=dict(input_f32_bits=fbits(values),expected_bf16_bits=bits(values.to(torch.bfloat16)),roundtrip_f32_bits=fbits(values.to(torch.bfloat16).float()))
record=dict(schema_version=1,kind='official1017_basicops_CPU_AST_goldens',source_sha256=SOURCE,torch_version=torch.__version__,device='cpu',embedding=embedding,ring=ring,cast=cast,
 scope='Exact embedding/Attention._window_kv AST with identity projection/norm/Rope/quantization; distributed sum supplied F32single-owner reference; no model, actual collective or fullattention qualification.')
p=Path(sys.argv[2]);
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,ring_cases=len(ring),GPU_work=False)))
