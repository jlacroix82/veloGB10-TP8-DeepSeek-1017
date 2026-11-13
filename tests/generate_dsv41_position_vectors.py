"""CPU-only exact official1017 frequency tables + apply_rotary_emb AST goldens.

Usage: python generate_dsv41_position_vectors.py VERIFIED/model.py NEW_TABLE_DIR NEW_FIXTURE.json
Complete complex64 tables are private artifacts, never model/checkpoint data.
"""
import ast, functools, hashlib, json, math, os, sys, time
from pathlib import Path
import torch
SOURCE_SHA='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
POSITIONS=[0,1,2,63,64,127,128,255,256,65535,65536,99999,100000,1048574,1048575]
LENGTH=1048576
os.umask(0o077);began=time.monotonic();source=Path(sys.argv[1]).read_bytes();assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False,mode=0o700)
torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(2);torch.set_flush_denormal(False)
namespace={'torch':torch,'math':math,'lru_cache':functools.lru_cache};tree=ast.parse(source)
methods=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['precompute_freqs_cis','apply_rotary_emb']]
assert len(methods)==2
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'verified1017_position_methods','exec'),namespace)
def bits16(t):return t.detach().contiguous().view(torch.uint16).reshape(-1).tolist()
def bits32(t):return t.detach().contiguous().view(torch.uint32).reshape(-1).tolist()
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()
profiles=[];cases=[]
for name,base,original in [('plain',10000.,0),('yarn',160000.,65536)]:
 table=namespace['precompute_freqs_cis'](64,LENGTH,original,base,16.,32,1)
 assert table.dtype==torch.complex64 and table.device.type=='cpu' and table.shape==(LENGTH,32)
 p=root/(name+'.complex64-le');array=table.detach().contiguous().numpy();assert array.dtype.byteorder in ('=','<') and sys.byteorder=='little'
 array.tofile(p);assert p.stat().st_size==LENGTH*32*8
 profile=dict(name=name,base=base,original_seq_len=original,factor=16.,beta_fast=32,beta_slow=1,positions=LENGTH,pairs=32,bytes=p.stat().st_size,storage='little-endian interleaved real/imag float32',file=p.name,sha256=sha_file(p))
 selected=table[torch.tensor(POSITIONS,dtype=torch.int64)].clone()
 profiles.append(profile)
 count=len(POSITIONS);heads=3;width=512
 full=(((torch.arange(count*heads*width,dtype=torch.int64)*7+13)%67-33).float()*0.0625).to(torch.bfloat16).view(1,count,heads,width)
 forward=full.clone();namespace['apply_rotary_emb'](forward[...,-64:],selected,False)
 inverse=full.clone();namespace['apply_rotary_emb'](inverse[...,-64:],selected,True)
 roundtrip=forward.clone();namespace['apply_rotary_emb'](roundtrip[...,-64:],selected,True)
 cases.append(dict(profile=name,positions=POSITIONS,heads=heads,width=width,rotary_dim=64,coefficients_f32_bits=bits32(torch.view_as_real(selected)),
                   input_bf16_bits=bits16(full),forward_bf16_bits=bits16(forward),inverse_bf16_bits=bits16(inverse),roundtrip_bf16_bits=bits16(roundtrip)))
 del table,array,selected;namespace['precompute_freqs_cis'].cache_clear()
manifest=dict(schema_version=1,kind='official1017_complete_CPU_RoPE_frequency_tables',model_source_sha256=SOURCE_SHA,
 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,device='cpu',gpu_initialized=torch.cuda.is_initialized(),
 logical_positions=LENGTH,rotary_dim=64,profiles=profiles,representative_positions=POSITIONS,
 scope='Exact official CPUtorch frequency/complex arithmetic artifacts; no Rust/CUDA pow/sincos phase approximation, model or contextquality qualification.')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
fixture=dict(schema_version=1,kind='official1017_RoPE_CPU_AST_goldens',model_source_sha256=SOURCE_SHA,torch_version=torch.__version__,device='cpu',
 profiles=profiles,manifest_sha256=sha_file(root/'manifest.json'),logical_positions=LENGTH,cases=cases,
 scope='Exact official precompute_freqs_cis/apply_rotary_emb CPU AST; last64 adjacent pairs on BF16 fullwidth512. Complete private tables are separately SHA bound.')
out=Path(sys.argv[3]);
with out.open('x') as f:json.dump(fixture,f,indent=2);f.write('\n')
print(json.dumps(dict(table_directory=str(root),manifest_sha256=sha_file(root/'manifest.json'),fixture=str(out),fixture_sha256=sha_file(out),profiles=profiles,
 elapsed_seconds=time.monotonic()-began,GPU_work=False)))
