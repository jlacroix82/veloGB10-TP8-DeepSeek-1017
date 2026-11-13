"""CPU-only oracle using the exact verified official Engram.forward AST.

Usage: python generate_dsv41_engram_vectors.py /verified/inference/model.py output.json
Projection output is supplied as a fixture: this does not qualify wkv or a model.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import torch

SOURCE_SHA = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
tree = ast.parse(source)
engram = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engram")
forward = next(n for n in engram.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
namespace = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])), "verified_Engram.forward", "exec"), namespace)
torch.set_default_device("cpu")
torch.set_flush_denormal(False)

def bits(t):
    return t.detach().contiguous().view(torch.uint16).reshape(-1).tolist()

cases = []
for name, tokens, dim, tiny in [("signed_and_masked", 2, 32, False), ("different_copy_norms", 3, 128, False), ("real_hidden_width", 1, 5120, False), ("tiny_norm_epsilon", 2, 32, True)]:
    def values(count, seed, scale):
        i = torch.arange(count, dtype=torch.int64, device="cpu")
        return (((i * 7 + seed) % 31 - 15).float() * scale).to(torch.bfloat16)
    h = values(tokens*4*dim, 3, 2**-40 if tiny else 0.03125).view(tokens, 1, 4, dim)
    kv = values(tokens*5*dim, 5, 2**-39 if tiny else 0.0625).view(tokens, 1, 5*dim)
    qw = values(4*dim, 7, 0.125).view(4, dim)
    kw = values(4*dim, 11, 0.125).view(4, dim)
    # Different copies must not accidentally share a norm, and a negative dot
    # must keep its sign through sqrt. All scaling values round exactly to BF16.
    for copy in range(4):
        h[:, :, copy] *= (1 if copy%2 == 0 else -1) * 2**copy
    mask = torch.tensor([i%3 != 1 for i in range(tokens)], dtype=torch.bool, device="cpu").view(tokens, 1)
    fake = types.SimpleNamespace(dim=dim, hc_mult=4, eps=1e-20, clamp_value=1e-6,
        q_weight=qw, k_weight=kw,
        embed=lambda ids: torch.zeros(tokens, 1, 24, 256, dtype=torch.bfloat16, device="cpu"),
        wkv=lambda unused: kv)
    out = namespace["forward"](fake, h, torch.zeros(tokens, 1, 24, dtype=torch.int64, device="cpu"), mask)
    cases.append(dict(name=name, tokens=tokens, dim=dim, h_bits=bits(h), kv_bits=bits(kv),
        q_weight_bits=bits(qw), k_weight_bits=bits(kw), mask=mask.reshape(-1).tolist(), output_bits=bits(out)))

record = dict(kind="official1017_Engram_forward_CPU_goldens", model_source_sha256=SOURCE_SHA,
    torch_version=torch.__version__, device="cpu", projection="supplied BF16 kv; wkv not evaluated",
    cases=cases)
target = Path(sys.argv[2])
with target.open("x") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
print(json.dumps(dict(path=str(target), bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(), cases=len(cases), gpu_work=False)))
