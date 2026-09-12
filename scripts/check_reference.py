#!/usr/bin/env python3
"""Compare the compiler's float decoder with the upstream Transformers model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.checkpoint import load_checkpoint
from compiler.compile import read_calibration
from compiler.gpt2 import fold_gpt2
from compiler.llama import fold_llama
from compiler.reference import FloatDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg, state, hashes = load_checkpoint(args.model_dir)
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = (
        AutoModelForCausalLM.from_config(config, trust_remote_code=False).float().eval()
    )
    tensors = {
        k: torch.from_numpy(v)
        for k, v in state.items()
        if not k.endswith(".attn.masked_bias")
    }
    if config.tie_word_embeddings and "lm_head.weight" not in tensors:
        key = (
            "transformer.wte.weight"
            if cfg["model_type"] == "gpt2"
            else "model.embed_tokens.weight"
        )
        tensors["lm_head.weight"] = tensors[key]
    model.load_state_dict(tensors, strict=True)
    folded = (fold_gpt2 if cfg["model_type"] == "gpt2" else fold_llama)(
        cfg, state, args.context
    )
    reference = FloatDecoder(folded)
    sequences = read_calibration(
        args.model_dir / "evaluation.json", cfg["vocab_size"], args.context
    )
    maximum = 0.0
    total = 0
    for tokens in sequences:
        with torch.no_grad():
            expected = model(torch.tensor([tokens])).logits[0].float().numpy()
        actual = reference.forward(np.array(tokens))
        # The folded reference uses float64; upstream accumulates in float32.
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=2e-4)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        total += actual.size
    report = dict(
        passed=True,
        upstream_dtype="float32",
        absolute_tolerance=5e-4,
        relative_tolerance=2e-4,
        sequences=len(sequences),
        logits_compared=total,
        max_abs_error=maximum,
        checkpoint_sha256=hashes,
        transformers=__import__("transformers").__version__,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
