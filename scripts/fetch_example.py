#!/usr/bin/env python3
"""Download a pinned GPT-2 checkpoint and convert its tensors to safetensors."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from safetensors.torch import save_file

MODEL = "sshleifer/tiny-gpt2"
REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
FILES = ("config.json", "pytorch_model.bin", "vocab.json", "merges.txt",
         "tokenizer_config.json", "special_tokens_map.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/tiny-gpt2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        with urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120) as response:
            data = response.read()
        (args.out / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    state = torch.load(args.out / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # GPT-2 ties the output projection to the token embedding. Clone aliases for safetensors.
    state = {k: v.contiguous().clone() for k, v in state.items() if v.is_floating_point()}
    save_file(state, str(args.out / "model.safetensors"))
    provenance = {"model": MODEL, "revision": REVISION, "sha256": hashes}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    from transformers import GPT2TokenizerFast
    tokenizer=GPT2TokenizerFast.from_pretrained(args.out,local_files_only=True)
    texts=["Hello world. This is a small hardware test.", "The quick brown fox jumps over the lazy dog.",
           "Memory and computation are connected by wires.", "A model predicts the next token in a sentence.",
           "One two three four five six seven eight.", "London is a city. Paris is another city."]
    (args.out / "calibration.json").write_text(json.dumps([tokenizer.encode(t) for t in texts],indent=2)+"\n")
    print(f"{MODEL}@{REVISION} -> {args.out}")


if __name__ == "__main__":
    main()
