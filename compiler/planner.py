#!/usr/bin/env python3
"""Inspect checkpoint shapes and circuit size without loading tensor payloads."""

import argparse
import json
from pathlib import Path
from safetensors import safe_open

FAMILIES = ("gpt2", "llama", "qwen2", "mistral")


def inspect_checkpoint(directory, context=64):
    directory = Path(directory).resolve()
    cfg = json.loads((directory / "config.json").read_text())
    index = directory / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    files = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    tensors = {}
    for name in files:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("checkpoint shard escapes model directory")
        with safe_open(path, framework="np") as file:
            for key in file.keys():
                if key in tensors or (
                    weight_map is not None and weight_map.get(key) != name
                ):
                    raise ValueError(f"invalid shard index for {key}")
                view = file.get_slice(key)
                shape = view.get_shape()
                elements = 1
                for dim in shape:
                    elements *= dim
                tensors[key] = dict(
                    shape=shape, elements=elements, dtype=view.get_dtype(), shard=name
                )
    if weight_map is not None and set(tensors) != set(weight_map):
        raise ValueError("shard index and tensors disagree")
    family = cfg.get("model_type")
    supported = family in FAMILIES
    result = dict(
        model_type=family,
        decoder_adapter=supported,
        tensor_count=len(tensors),
        stored_parameters=sum(t["elements"] for t in tensors.values()),
        tensors=tensors,
    )
    if supported:
        d = cfg["n_embd"] if family == "gpt2" else cfg["hidden_size"]
        h = cfg["n_head"] if family == "gpt2" else cfg["num_attention_heads"]
        l = cfg["n_layer"] if family == "gpt2" else cfg["num_hidden_layers"]
        v = cfg["vocab_size"]
        kh = cfg.get("num_key_value_heads", h)
        limit = (
            cfg["n_positions"]
            if family == "gpt2"
            else cfg.get("max_position_embeddings", 2048)
        )
        if not 2 <= context <= min(limit, 1024):
            raise ValueError("context must be in 2..min(model limit,1024)")
        # Normalisation affine terms are folded into the projections. Tied token
        # embeddings and output projection are separate functions in the circuit.
        if family == "gpt2":
            inner = cfg.get("n_inner") or 4 * d
            coefficients = 2 * v * d + context * d + l * (4 * d * d + 2 * d * inner)
        else:
            inner = cfg["intermediate_size"]
            kd = kh * (d // h)
            coefficients = 2 * v * d + l * (2 * d * d + 2 * d * kd + 3 * d * inner)
        result.update(
            width=d,
            layers=l,
            query_heads=h,
            kv_heads=kh,
            vocab=v,
            context=context,
            fixed_coefficients=coefficients,
            kv_cache_bits=2 * l * context * kh * (d // h) * 8,
            note="Coefficient count is not cell area. Detailed configuration is validated during lowering.",
        )
    else:
        result["note"] = (
            "No full decoder adapter. Rank-2 tensors can still be compiled with --matrices-only; this does not produce complete inference."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--context", type=int, default=64)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.model_dir, args.context)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"inspect: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
