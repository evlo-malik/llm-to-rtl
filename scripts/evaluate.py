#!/usr/bin/env python3
"""Measure quantisation error on token sequences that were not used for calibration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.bundle import load_bundle
from compiler.checkpoint import load_checkpoint
from compiler.compile import read_calibration
from compiler.gpt2 import fold_gpt2, FloatGPT, IntGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=Path, help="JSON evaluation token sequences")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    cfg, state, hashes = load_checkpoint(args.model_dir)
    if hashes != manifest["checkpoint_sha256"]:
        parser.error("checkpoint differs from the compiled source")
    sequences = read_calibration(
        args.tokens or args.model_dir / "evaluation.json",
        manifest["vocab"],
        manifest["context"],
    )
    if any(s in manifest["calibration_tokens"] for s in sequences):
        parser.error("evaluation sequence was used for calibration")
    q = load_bundle(args.output)
    floating = FloatGPT(fold_gpt2(cfg, state, manifest["context"]))
    integer = IntGPT(q)
    agree = total = 0
    max_error = 0.0
    abs_error = 0.0
    elements = 0
    for tokens in sequences:
        expected = floating.forward(np.array(tokens))
        actual = integer.forward(tokens) * q["lm_head"]["s_out"][None, :]
        agree += int(np.count_nonzero(actual.argmax(-1) == expected.argmax(-1)))
        total += len(tokens)
        error = np.abs(actual - expected)
        max_error = max(max_error, float(error.max()))
        abs_error += float(error.sum())
        elements += error.size
    report = dict(
        sequences=len(sequences),
        positions=total,
        argmax_agreement=agree / total,
        max_abs_logit_error=max_error,
        mean_abs_logit_error=abs_error / elements,
        evaluation_tokens=sequences,
        checkpoint_sha256=hashes,
    )
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("evaluation_tokens", "checkpoint_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
