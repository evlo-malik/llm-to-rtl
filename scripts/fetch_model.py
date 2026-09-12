#!/usr/bin/env python3
"""Fetch a pinned Hugging Face safetensors checkpoint and prepare calibration tokens."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument(
        "--revision",
        required=True,
        help="commit or ref; resolved to a commit before downloading",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / "examples/calibration.txt"
    )
    parser.add_argument(
        "--evaluation", type=Path, default=ROOT / "examples/evaluation.txt"
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new model directory")
    info = HfApi().model_info(args.model, revision=args.revision)
    names = [entry.rfilename for entry in info.siblings if "/" not in entry.rfilename]
    files = [
        name
        for name in names
        if name
        in (
            "config.json",
            "README.md",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        )
        or name.startswith("tokenizer")
        or name.endswith(".model")
        or (
            name.startswith("model")
            and (
                name.endswith(".safetensors")
                or name.endswith(".safetensors.index.json")
            )
        )
    ]
    if "config.json" not in files or not any(
        name.endswith(".safetensors") for name in files
    ):
        parser.error(
            "repository does not contain a supported root-level safetensors checkpoint"
        )
    snapshot_download(
        args.model, revision=info.sha, local_dir=args.out, allow_patterns=files
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.out, local_files_only=True, trust_remote_code=False
    )
    for name, path in [
        ("calibration", args.calibration),
        ("evaluation", args.evaluation),
    ]:
        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not texts:
            parser.error(f"{name} text file is empty")
        (args.out / f"{name}.json").write_text(
            json.dumps([tokenizer.encode(text) for text in texts], indent=2) + "\n"
        )
    (args.out / "provenance.json").write_text(
        json.dumps(dict(model=args.model, revision=info.sha), indent=2) + "\n"
    )
    print(f"{args.model}@{info.sha} -> {args.out}")


if __name__ == "__main__":
    main()
