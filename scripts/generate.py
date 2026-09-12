#!/usr/bin/env python3
"""Generate text from the RTL simulator, checking every output against the reference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--sim", choices=["icarus", "verilator"], default="verilator")
    args = parser.parse_args()
    if (args.model_dir / "tokenizer.model").exists():
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor(
            model_file=str(args.model_dir / "tokenizer.model")
        )
        prompt = [tokenizer.bos_id()] + tokenizer.encode(args.prompt)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        prompt = tokenizer.encode(args.prompt)
    manifest = json.loads((args.output / "manifest.json").read_text())
    steps = len(prompt) + args.new_tokens - 1
    if not prompt or args.new_tokens < 1 or steps > manifest["context"]:
        parser.error(
            "prompt and requested output exceed the compiled context; recompile with a longer context"
        )
    log = args.output / f"generation_{args.sim}.log"
    with log.open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify.py"),
                str(args.output),
                "--sim",
                args.sim,
                "--tokens",
                ",".join(map(str, prompt)),
                "--steps",
                str(steps),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verification = json.loads(
        (args.output / f"verification_{args.sim}.json").read_text()
    )
    tokens = verification["tokens"] + [verification["next_token"]]
    text = tokenizer.decode(tokens)
    record = dict(
        prompt=args.prompt,
        text=text,
        tokens=tokens,
        simulator=args.sim,
        source="RTL simulation",
        rtl_sha256=manifest["rtl_sha256"],
        verification=verification,
    )
    (args.output / f"generation_{args.sim}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(text)
    print(f"\n{verification['logits_compared']:,} RTL logits matched; log: {log}")


if __name__ == "__main__":
    main()
