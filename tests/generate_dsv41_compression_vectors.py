#!/usr/bin/env python3
"""CPU-only goldens from the pinned official1017 Compressor/Indexer methods.
Dense projections, RoPE and FP4 cacheQAT are supplied fixtures here; no weights,
CUDA initialization, network calls or model execution occur.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

SOURCE_SHA = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"

def bits(value):
    value = value.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).flatten().tolist()
    if value.dtype == torch.float32:
        return value.view(torch.uint32).flatten().tolist()
    raise ValueError("unexpected golden dtype " + str(value.dtype))

def function(tree, class_name, transform=None):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    if transform:
        node = transform.visit(node)
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))

class PoolCapture(ast.NodeTransformer):
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (any(isinstance(t, ast.Name) and t.id == "kv" for t in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "sum"):
            log = ast.Expr(ast.Call(func=ast.Name("_record_pool", ast.Load()),
                                   args=[ast.Name("kv", ast.Load())], keywords=[]))
            return [node, log]
        return node

class Projection:
    def __init__(self):
        self.value = None
        self.input_dtypes = []
    def __call__(self, x):
        self.input_dtypes.append(str(x.dtype))
        return self.value.clone()

class Norm:
    def __init__(self, forward, dim):
        self.forward = forward
        self.eps = 1e-20
        self.weight = ((torch.arange(dim, dtype=torch.float32) % 7) / 8 + 0.625).to(torch.bfloat16)
        self.inputs = []
    def __call__(self, value):
        self.inputs.append(value.clone())
        return self.forward(self, value)

def compression_cases(tree):
    rms_ns = {"torch": torch}
    exec(compile(function(tree, "RMSNorm"), "<official-RMSNorm>", "exec"), rms_ns)
    rows = []
    cases = [
        ("ratio1_prefill_append", 1, 8, [(0, 3), (3, 1)]),
        ("ratio2_even_prefill", 2, 8, [(0, 4)]),
        ("ratio2_odd_prefill_tail", 2, 8, [(0, 3), (3, 1)]),
        ("ratio2_one_token_tail", 2, 8, [(0, 1), (1, 1)]),
        ("ratio2_append_incomplete_complete", 2, 8, [(0, 2), (2, 1), (3, 1)]),
        ("ratio2_stable_large_negative_scores", 2, 8, [(0, 3), (3, 1)]),
        ("ratio1_real512_prefill4", 1, 512, [(0, 4)]),
        ("ratio2_real512_prefill7", 2, 512, [(0, 7)]),
    ]
    for name, ratio, dim, events in cases:
        pools = []
        namespace = {"torch": torch, "_record_pool": lambda value: pools.append(value.clone())}
        exec(compile(function(tree, "Compressor", PoolCapture()), "<official-Compressor>", "exec"), namespace)
        obj = SimpleNamespace(compress_ratio=ratio, wkv=Projection(),
                              norm=Norm(rms_ns["forward"], dim))
        if ratio == 2:
            obj.wgate = Projection()
            obj.kv_state = torch.zeros(1, 2, dim, dtype=torch.float32)
            obj.score_state = torch.full((1, 2, dim), -torch.inf, dtype=torch.float32)
        result = []
        for event_no, (start, tokens) in enumerate(events):
            ids = torch.arange(tokens * dim, dtype=torch.float32).reshape(1, tokens, dim)
            if ratio == 1:
                kv = (((ids * 7 + 3 + start) % 31 - 15) / 8).to(torch.bfloat16)
                score = None
            else:
                kv = ((ids * 7 + 3 + start) % 257 - 128) / 64 + 1 / 1024
                score = ((ids * 11 + 5 + start) % 13 - 6) / 4
                if "large_negative" in name:
                    score = score - 1000
                obj.wgate.value = score
            obj.wkv.value = kv
            before_norm = len(obj.norm.inputs)
            before_pool = len(pools)
            output = namespace["forward"](obj, torch.zeros(1, tokens, dim, dtype=torch.bfloat16), start)
            output_rows = (tokens if ratio == 1 else
                           tokens // 2 if start == 0 else int((start + 1) % 2 == 0))
            positions = (list(range(start, start + tokens)) if ratio == 1 else
                         [2 * i for i in range(tokens // 2)] if start == 0 else
                         [start - 1] if output_rows else [])
            pool = pools[-1] if len(pools) != before_pool else None
            norm_input = obj.norm.inputs[-1] if len(obj.norm.inputs) != before_norm else None
            early_bf16_difference = 0
            if ratio == 2 and start == 0 and tokens >= 2:
                cutoff = tokens - tokens % 2
                kk = kv[:, :cutoff].unflatten(1, (-1, 2))
                ss = score[:, :cutoff].unflatten(1, (-1, 2)).softmax(dim=2)
                wrong = (kk.to(torch.bfloat16).float() * ss).sum(dim=2).to(torch.bfloat16)
                early_bf16_difference = int((wrong.view(torch.uint16) != norm_input.view(torch.uint16)).sum())
            result.append({
                "start": start, "tokens": tokens, "output_rows": output_rows,
                "positions": positions, "projected_kv_dtype": str(kv.dtype),
                "projected_kv_bits": bits(kv), "projected_score_bits": bits(score) if score is not None else None,
                "pooled_f32_bits": bits(pool) if pool is not None else None,
                "pool_dtype": str(pool.dtype) if pool is not None else None,
                "norm_input_bf16_bits": bits(norm_input) if norm_input is not None else [],
                "norm_input_dtype": str(norm_input.dtype) if norm_input is not None else None,
                "normalized_unrotated_bf16_bits": bits(output) if output is not None else [],
                "normalized_dtype": str(output.dtype) if output is not None else None,
                "tail_kv_f32_bits": bits(obj.kv_state) if ratio == 2 else None,
                # -Inf denotes unwritten slots; it is state initialization, not finite projection data.
                "tail_score_f32_bits": bits(obj.score_state) if ratio == 2 else None,
                "negative_early_BF16_pool_input_mismatches": early_bf16_difference,
            })
        rows.append({"name": name, "ratio": ratio, "dim": dim,
                     "norm_weight_bf16_bits": bits(obj.norm.weight), "events": result})
    return rows

class TorchProxy:
    def __init__(self, capture):
        self.capture = capture
    def __getattr__(self, key):
        return getattr(torch, key)
    def einsum(self, expression, *values):
        answer = torch.einsum(expression, *values)
        self.capture["dot"] = answer.clone()
        return answer

def index_cases(tree):
    results = []
    for name, b, queries, keys in [
        ("local4_heads_signed", 1, 4, 7),
        ("local4_heads_repeated_keys", 2, 3, 5),
        ("local4_heads_rounding_boundaries", 1, 2, 31),
        ("local4_heads_prefill16", 1, 16, 16),
    ]:
        q = (((torch.arange(b * queries * 4 * 128, dtype=torch.float32) * 7 + 3) % 31 - 15) / 8).reshape(b, queries, 4, 128).to(torch.bfloat16)
        key = (((torch.arange(b * keys * 128, dtype=torch.float32) * 11 + 5) % 29 - 14) / 8).reshape(b, keys, 128).to(torch.bfloat16)
        projected = (((torch.arange(b * queries * 4, dtype=torch.float32) * 13 + 17) % 37 - 18) / 16 + 1 / 128).reshape(b, queries, 4).to(torch.bfloat16)
        capture = {}
        proxy = TorchProxy(capture)
        def all_reduce(value):
            capture["reduce_input"] = value.clone()
        namespace = {"torch": proxy, "fp4_act_quant": lambda *a, **k: None,
                     "apply_rotary_emb": lambda *a, **k: None, "world_size": 8,
                     "dist": SimpleNamespace(all_reduce=all_reduce),
                     "shared_attn": SimpleNamespace(index_k=key), "fp4_block_size":32}
        exec(compile(function(tree, "Indexer"), "<official-Indexer>", "exec"), namespace)
        q_proj = Projection(); q_proj.value = q.flatten(-2)
        weight_proj = Projection(); weight_proj.value = projected
        obj = SimpleNamespace(freqs_cis=torch.ones(queries, 32, dtype=torch.complex64),
                              compress_ratio=1, rope_head_dim=64, owns_k=False,
                              wq_b=q_proj, n_local_heads=4, index_head_dim=128,
                              weights_proj=weight_proj, softmax_scale=128 ** -0.5,
                              n_heads=32, index_topk=512, is_candidate_source=False,
                              uses_candidates=False)
        # Positive start keeps supplied fixed key extent independent of the query count.
        # The source allows arbitrary seqlen in Indexer; the component state API separately
        # restricts actual positive-start compressor appends to one token.
        start = keys - queries
        if start < 0:
            start = 0
            # For start0 the index_key cache is sliced toqueries entries. Provide
            # enough physical keys and later retain only that source-visible slice.
            if keys < queries:
                key = key.repeat(1, (queries + keys - 1) // keys, 1)[:, :queries].contiguous()
                keys = queries
                namespace["shared_attn"].index_k = key
        obj.freqs_cis = torch.ones(max(start + queries, queries), 32, dtype=torch.complex64)
        namespace["forward"](obj, torch.zeros(b, queries, 8, dtype=torch.bfloat16),
                             torch.zeros(b, queries, 8, dtype=torch.bfloat16), None, start, 128)
        factor = 128 ** -0.5 * 32 ** -0.5
        scaled = projected * factor
        dot = capture["dot"]
        relu = dot.relu()
        product = relu * scaled.unsqueeze(-1)
        local = product.sum(dim=2)
        assert torch.equal(local, capture["reduce_input"])
        alternate = (dot.float().relu() * scaled.float().unsqueeze(-1)).sum(dim=2)
        results.append({
            "name": name, "batch": b, "queries": queries, "keys": keys,
            "local_heads": 4, "global_heads": 32, "head_dim": 128,
            "projected_q_bf16_bits": bits(q), "key_bf16_bits": bits(key),
            "projected_weight_bf16_bits": bits(projected),
            "scaled_weight_bf16_bits": bits(scaled), "dot_bf16_bits": bits(dot),
            "relu_bf16_bits": bits(relu), "product_bf16_bits": bits(product),
            "local_sum_bf16_bits": bits(local),
            "source_local_promoted_f32_bits": bits(local.float()),
            "negative_unrounded_product_or_sum_f32_bits": bits(alternate),
            "negative_unrounded_local_mismatches": int((alternate != local.float()).sum()),
            "stage_dtypes": {key: str(value.dtype) for key, value in
                             [("q", q), ("key", key), ("projected_weights", projected),
                              ("scaled_weights", scaled), ("einsum", dot), ("relu", relu),
                              ("product", product), ("head_sum", local),
                              ("official_allreduce_input", capture["reduce_input"])]},
            "factor_python_f64": factor,
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_device("cpu")
    torch.set_flush_denormal(False)
    raw = args.reference_model.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    tree = ast.parse(raw.decode())
    fixture = {"schema_version": 1, "kind": "official1017_compression_index_CPU_AST_goldens",
               "model_source_sha256": SOURCE_SHA, "torch": torch.__version__,
               "device": "cpu", "CUDA_initialized": torch.cuda.is_initialized(),
               "model_execution": False, "checkpoint_payload_read": False,
               "projection_and_rotation_scope": "Dense projections, RoPE and cacheQAT are supplied stageinputs; original Compressor/Indexer/RMSNorm arithmetic executed.",
               "compression_cases": compression_cases(tree), "index_cases": index_cases(tree),
               "index_network_policy": "Official source allreduces BF16 locals directly. Promoting these locals for MatrixF32 network reduction is an explicit alternative rounding path."}
    assert fixture["CUDA_initialized"] is False
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")

if __name__ == "__main__":
    main()

