#!/usr/bin/env python3
"""CPU-only exact-source AST proof of the native1017 <=512 all-selected path."""
import ast,hashlib,json,functools
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[1]/'private/velogb10-tp8/deepseek-1017-reference/inference/model.py'
EXPECTED='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'
raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
module=ast.parse(raw);names={'get_window_topk_idxs','select_candidate_blocks'}
functions=[n for n in module.body if isinstance(n,ast.FunctionDef)and n.name in names]
assert len(functions)==2
scope={'torch':torch,'F':torch.nn.functional,'lru_cache':functools.lru_cache}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),scope)
cases=[]
for layer,ratio in [(0,0),(2,2),(20,1)]:
    shapes=[(0,n)for n in ([1,2,127,128]if ratio==0 else[1,2,3,127,128])]
    shapes +=[(s,1)for s in ([1,2,127,128,511]if ratio==0 else[1,2,3,127,128,129,510,511])]
    for start,tokens in shapes:
        end=start+tokens;assert end<=512
        window=scope['get_window_topk_idxs'](128,1,tokens,start)[0]
        wr=tokens if start==0 else 128;cr=0 if ratio==0 else end//ratio
        compressed=torch.empty((tokens,0),dtype=torch.int32);same=True
        if cr:
            scores=torch.tensor([[(i*17+q*3)%97-48 for i in range(cr)]for q in range(tokens)],dtype=torch.float32)
            visible=torch.arange(1,tokens+1).unsqueeze(-1)//ratio if start==0 else end//ratio
            if start==0:scores.masked_fill_(torch.arange(cr)>=visible,-torch.inf)
            candidates=scope['select_candidate_blocks'](scores,visible,2048,8)
            def select(x):
                idx=x.topk(min(512,cr),dim=-1,sorted=False).indices.sort(dim=-1).values
                return torch.where(idx<visible,idx+wr,-1).int()
            compressed=select(scores);filtered=select(scores.masked_fill(~candidates,-torch.inf))
            same=torch.equal(compressed,filtered);assert same
            expected=torch.tensor([[wr+i if i<((q+1)//ratio if start==0 else end//ratio)else-1 for i in range(cr)]for q in range(tokens)],dtype=torch.int32)
            assert torch.equal(compressed,expected)
        indices=torch.cat([window,compressed],dim=-1)
        cases.append({'layer':layer,'ratio':ratio,'start':start,'tokens':tokens,'window_rows':wr,'compressed_rows':cr,'kv_rows':wr+cr,'topk':indices.shape[-1],'indices':indices.flatten().tolist(),'candidate_mask_changes_selection':False,'sorted_all_compressed_exact':True})
fixture={'schema_version':1,'kind':'official1017_CPU_AST_shortKV_all_selected','model_source_sha256':EXPECTED,'torch_version':torch.__version__,'GPU_used':False,'checkpoint_payload_read':False,'hard_context':512,'index_topk':512,'candidate_topk_blocks':2048,'candidate_block_size':8,'cases':cases}
p=ROOT/'tests/fixtures/dsv41_kv_store_vectors.json';p.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':len(cases),'GPU_used':False}))
