#!/usr/bin/env python3
"""Pretrained safetensors checkpoint -> fixed-weight SystemVerilog."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from compiler.bundle import save_bundle
from compiler.checkpoint import load_checkpoint
from compiler.emit import emit_linear
from compiler.gpt2 import fold_gpt2
from compiler.reference import calibrate, quantise_model
from compiler.llama import fold_llama
from compiler.planner import inspect_checkpoint
from compiler.model_rtl import emit_model
from compiler.quant import quant_weight


def llama_matrices(cfg, state, bits):
    """Llama projections only; this adapter does not emit a Llama token engine."""
    if cfg.get("attention_bias", False) or cfg.get("mlp_bias", False):
        raise ValueError("Llama matrix adapter does not support projection biases")
    if "lm_head.weight" not in state and not cfg.get("tie_word_embeddings", False):
        raise ValueError("untied checkpoint is missing lm_head.weight")
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for block, names in [
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ]:
            for name in names:
                tensor = prefix + block + "." + name + ".weight"
                weight = state[tensor]
                gamma = None
                if name in ("q_proj", "k_proj", "v_proj"):
                    gamma = state[prefix + "input_layernorm.weight"]
                if name in ("gate_proj", "up_proj"):
                    gamma = state[prefix + "post_attention_layernorm.weight"]
                if gamma is not None:
                    weight = weight.astype(np.float64) * gamma[None, :]
                w, scale = quant_weight(weight, bits)
                yield f"layer{layer}_{name}", w, scale, tensor
    tensor = (
        "lm_head.weight" if "lm_head.weight" in state else "model.embed_tokens.weight"
    )
    w, scale = quant_weight(
        state[tensor].astype(np.float64) * state["model.norm.weight"][None, :],
        bits,
        False,
    )
    yield "lm_head", w, scale, tensor


def read_calibration(path, vocab, context):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("calibration must be a nonempty list of token-ID sequences")
    windows = []
    for row in data:
        if (
            not isinstance(row, list)
            or not row
            or any(type(t) is not int or not 0 <= t < vocab for t in row)
        ):
            raise ValueError("invalid calibration token ID")
        windows.append(row[:context])
    return windows


def compile_checkpoint(
    model_dir,
    out,
    bits=8,
    context=8,
    calibration=None,
    matrices_only=False,
    only=None,
    max_coefficients=2_000_000,
    activation_bits=None,
):
    out = Path(out)
    if out.exists():
        raise ValueError(f"output already exists: {out}; choose a new output directory")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4 or 8")
    if max_coefficients < 0:
        raise ValueError("max_coefficients must be nonnegative")
    if not matrices_only:
        plan = inspect_checkpoint(model_dir, context)
        if not plan["decoder_adapter"]:
            raise ValueError(
                f"no decoder adapter for {plan['model_type']!r}; supported: gpt2, llama, qwen2, mistral"
            )
        if max_coefficients and plan["fixed_coefficients"] > max_coefficients:
            raise ValueError(
                f"{plan['fixed_coefficients']:,} coefficients exceed the emission limit {max_coefficients:,}; use --max-coefficients 0 to allow a larger circuit"
            )
    cfg, state, hashes = load_checkpoint(model_dir)
    provenance_path = Path(model_dir) / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    manifest = dict(
        schema_version=1,
        model_type=cfg.get("model_type"),
        checkpoint_sha256=hashes,
        source=provenance,
        bits=bits,
        weight_storage="constant_logic",
    )
    prepared = None
    selected = None
    if (
        cfg.get("model_type") in ("gpt2", "llama", "qwen2", "mistral")
        and not matrices_only
    ):
        if only:
            raise ValueError("--only requires --matrices-only")
        f = (fold_gpt2 if cfg["model_type"] == "gpt2" else fold_llama)(
            cfg, state, context
        )
        if calibration is None:
            calibration = Path(model_dir) / "calibration.json"
        windows = read_calibration(calibration, f["V"], context)
        stats = calibrate(f, windows)
        if activation_bits is None:
            embedding_rms = max(float(np.sqrt(np.mean(f["tok_emb"] ** 2))), 1e-9)
            activation_bits = (
                16 if stats["ln"] > 7.5 or stats["h"] / embedding_rms > 1000 else 8
            )
            manifest["activation_precision_selection"] = "calibration ranges"
        else:
            manifest["activation_precision_selection"] = "explicit"
        prepared = quantise_model(f, stats, bits, activation_bits)
        prepared["stats"] = stats
        if not 0 <= prepared["eps_var"] < 2 ** (63 if activation_bits == 16 else 31):
            raise ValueError(
                "normalisation epsilon is outside the supported integer range"
            )
        size = (
            f["tok_emb"].size
            + (f["pos_emb"].size if cfg["model_type"] == "gpt2" else 0)
            + sum(
                layer[m][0].size
                for layer in f["layers"]
                for m in ("qkv", "proj", "ffwd1", "ffwd2")
            )
            + f["lm_head"][0].size
        )
        manifest["calibration_sha256"] = hashlib.sha256(
            Path(calibration).read_bytes()
        ).hexdigest()
        manifest["calibration_tokens"] = windows
    elif cfg.get("model_type") == "llama" and matrices_only:
        selected = []
        for name, w, scale, tensor in llama_matrices(cfg, state, bits):
            if only is None or name == only:
                selected.append((name, w, scale, tensor))
        if not selected:
            raise ValueError(f"no matrix named {only!r}")
        size = sum(w.size for _, w, _, _ in selected)
    elif matrices_only:
        selected = []
        names = set()
        for tensor, value in sorted(state.items()):
            if value.ndim != 2 or (only is not None and tensor != only):
                continue
            name = "tensor_" + re.sub(r"[^A-Za-z0-9_]", "_", tensor)
            if name in names:
                raise ValueError(f"RTL name collision for {tensor}")
            names.add(name)
            w, scale = quant_weight(value, bits)
            selected.append((name, w, scale, tensor))
        if not selected:
            raise ValueError(f"no rank-2 tensor named {only!r}")
        size = sum(w.size for _, w, _, _ in selected)
    else:
        raise ValueError("unsupported decoder configuration")
    if max_coefficients and size > max_coefficients:
        raise ValueError(
            f"{size:,} coefficients exceed the emission limit {max_coefficients:,}; "
            "use --max-coefficients 0 to explicitly allow a larger circuit"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".llm-to-rtl-", dir=out.parent))
    try:
        if prepared is not None:
            manifest.update(emit_model(staging, prepared))
            save_bundle(staging, prepared)
            manifest["scope"] = "full_model"
        else:
            rtl = staging / "rtl"
            rtl.mkdir()
            manifest["scope"] = "matrices_only"
            manifest["matrix_modules"] = []
            refs = {}
            for name, w, scale, tensor in selected:
                entry = emit_linear(rtl / f"{name}.sv", name, w)
                entry["source_tensor"] = tensor
                manifest["matrix_modules"].append(entry)
                refs[name] = {"w": w, "scale": scale}
            save_bundle(staging, refs)
        manifest["emitted_coefficients"] = int(size)
        manifest["rtl_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((staging / "rtl").glob("*.sv"))
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(8, 4, 2), default=8)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument(
        "--activation-bits",
        type=int,
        choices=(8, 16),
        default=None,
        help="default: select from calibration ranges; 16 uses INT32 residuals",
    )
    parser.add_argument(
        "--calibration", type=Path, help="JSON token-ID sequences; no training"
    )
    parser.add_argument("--matrices-only", action="store_true")
    parser.add_argument("--only", help="exact matrix name; requires --matrices-only")
    parser.add_argument(
        "--max-coefficients",
        type=int,
        default=2_000_000,
        help="emission size guard; 0 disables",
    )
    args = parser.parse_args()
    try:
        result = compile_checkpoint(**vars(args))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"compile: {exc}\n")
    print(
        f"{result['scope']}: {result['emitted_coefficients']:,} fixed coefficients -> {args.out}"
    )


if __name__ == "__main__":
    main()
