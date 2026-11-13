#!/usr/bin/env python3
"""Generate CPU-only vectors by executing the verified official selection AST.

No model is loaded. No CUDA operation is permitted. The final Indexer.forward
selection statements are extracted verbatim after globally reduced scores.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn.functional as F
from functools import lru_cache

MODEL_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"


def load_reference(path):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == MODEL_SHA256, "unqualified model.py identity"
    tree = ast.parse(data)
    names = {"get_window_topk_idxs", "select_candidate_blocks"}
    functions = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    forward = next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    # The first top-level `if start_pos == 0` is the causal-mask/selection tail.
    begin = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.If)
                 and ast.dump(n.test) == ast.dump(ast.parse("start_pos == 0", mode="eval").body))
    body = ast.parse("end_pos = start_pos + seqlen\nx = index_score\n").body
    body += copy.deepcopy(forward.body[begin:])
    signature = ast.parse("def official_selection(self, index_score, start_pos, seqlen, ratio, offset):\n    pass\n").body[0]
    signature.body = body
    functions.append(signature)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"torch": torch, "F": F, "lru_cache": lru_cache,
                 "shared_attn": SimpleNamespace(candidates=None)}
    exec(compile(module, str(path) + "[selection-only]", "exec"), namespace)
    return namespace, hashlib.sha256(ast.dump(module).encode()).hexdigest()


def make_case(ref, name, start, tokens, ratio, topk, raw, *, source=False,
              consumer=False, blocks=2048, block_size=8, prior_candidates=None,
              candidate_witness=False, position_witness=False):
    scores = torch.tensor(raw, dtype=torch.float32, device="cpu")
    assert list(scores.shape) == [tokens, (start + tokens) // ratio]
    settings = SimpleNamespace(is_candidate_source=source, uses_candidates=consumer,
        candidate_topk_blocks=blocks, candidate_block_size=block_size, index_topk=topk)
    ref["shared_attn"].candidates = None if prior_candidates is None else prior_candidates.unsqueeze(0)
    indices = ref["official_selection"](settings, scores.clone().unsqueeze(0), start, tokens, ratio,
        tokens if start == 0 else 128).squeeze(0)
    mask = ref["shared_attn"].candidates
    mask = None if mask is None else mask.squeeze(0).clone()
    visible = [((start + q + 1) // ratio) for q in range(tokens)]
    result = {"name": name, "start": start, "tokens": tokens, "ratio": ratio,
        "scores": scores.tolist(), "visible": visible, "topk": topk,
        "offset": tokens if start == 0 else 128, "apply_candidates": consumer,
        "indices": indices.tolist()}
    if source:
        result.update(candidates=mask.tolist(), topk_blocks=blocks, block_size=block_size)
    if consumer:
        # Record a first-level source score matrix so the Rust checker also
        # independently re-creates the exact mask before applying it.
        assert prior_candidates is not None
    if candidate_witness:
        result["candidate_witness"] = [[block for block in range((scores.shape[1]+block_size-1)//block_size)
            if bool(mask[row, block*block_size])] for row in range(tokens)]
        assert all(len(row) == min(blocks, (scores.shape[1]+block_size-1)//block_size)
                   for row in result["candidate_witness"])
        result["expect_candidate_ambiguity"] = True
    if position_witness:
        offset = result["offset"]
        result["position_witness"] = [[int(i)-offset for i in row] for row in indices.tolist()]
        assert all(i >= 0 for row in result["position_witness"] for i in row)
        result["expect_position_ambiguity"] = True
    return result, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    assert not torch.cuda.is_initialized()
    ref, extracted_sha = load_reference(args.model_source)
    windows = []
    for window, start, tokens in [(128,0,1),(128,0,2),(128,0,127),(128,0,128),
            (128,1,1),(128,127,1),(128,128,1),(128,255,1),(4,0,6),(4,6,1)]:
        tensor = ref["get_window_topk_idxs"](window, 1, tokens, start)
        windows.append({"window":window,"start":start,"tokens":tokens,
                        "indices":tensor.squeeze(0).tolist()})
    cases = []
    add = lambda *a, **kw: make_case(ref, *a, **kw)
    c, _ = add("ratio2_prefill_causal",0,8,2,512,
        [[float(p)+q/1024.0 for p in range(4)] for q in range(8)])
    cases.append(c)
    c, mask = add("source20_topk_does_not_filter_itself",24,1,1,4,
        [[float(25-p) for p in range(25)]],source=True,blocks=1)
    cases.append(c)
    # Same finite source scores reproduce that source's candidate mask, then
    # exercise consumer filtering with a one-position top-k (no cutoff tie).
    c, _ = add("consumer_filters_source_mask",24,1,1,1,
        [[float(25-p) for p in range(25)]],consumer=True,prior_candidates=mask)
    c.update(candidates=mask.tolist(),topk_blocks=1,block_size=8)
    cases.append(c)
    width=16393
    raw=[[float(width-p) for p in range(width)]]
    c, mask = add("actual2048x8_newest_partial_block",width-1,1,1,512,raw,
        source=True,blocks=2048,block_size=8)
    cases.append(c)
    c, _ = add("actual512_consumer_positions",width-1,1,1,512,raw,
        consumer=True,prior_candidates=mask)
    c.update(candidates=mask.tolist(),topk_blocks=2048,block_size=8)
    cases.append(c)
    c, _ = add("legitimate_block_and_position_ties",23,1,1,2,[[0.0]*24],
        source=True,blocks=2,block_size=8,candidate_witness=True,position_witness=True)
    cases.append(c)
    c, _ = add("legitimate_position_ties",5,1,1,2,[[0.0]*6],position_witness=True)
    cases.append(c)
    assert not torch.cuda.is_initialized()
    report={"kind":"official1017_attention_selection_CPU_goldens","schema_version":1,
        "model_source_sha256":MODEL_SHA256,"extracted_selection_ast_sha256":extracted_sha,
        "torch_version":torch.__version__,"device":"cpu","gpu_initialized":False,
        "scope":"official window and Indexer.forward selection tail given finite globally reduced scores; no projection, attention kernel, model payload or serving",
        "tie_policy":"unspecified Torch ties are retained as actual CPU choices and validated using acceptable-set witnesses; no portable tied-index order claimed",
        "window_cases":windows,"selection_cases":cases}
    data=(json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
    args.output.write_bytes(data)
    print(json.dumps({"path":str(args.output),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
        "window_cases":len(windows),"selection_cases":len(cases),"gpu_work":False}))


if __name__ == "__main__":
    main()
