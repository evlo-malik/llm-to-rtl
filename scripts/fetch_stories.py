#!/usr/bin/env python3
"""Import the pretrained Stories260K checkpoint into Hugging Face tensor layout."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
import sentencepiece as spm
from safetensors.torch import save_file

MODEL = "karpathy/tinyllamas"
REVISION = "0bd21da7698eaf29a0d7de3992de8a46ef624add"


def convert(checkpoint):
    args = checkpoint["model_args"]
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    d, h, l, v = (args[k] for k in ("dim", "n_heads", "n_layers", "vocab_size"))
    kh = args.get("n_kv_heads") or h
    inner = state["layers.0.feed_forward.w1.weight"].shape[0]
    cfg = dict(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        hidden_size=d,
        intermediate_size=inner,
        num_hidden_layers=l,
        num_attention_heads=h,
        num_key_value_heads=kh,
        vocab_size=v,
        max_position_embeddings=args["max_seq_len"],
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    result = {
        "model.embed_tokens.weight": state["tok_embeddings.weight"],
        "model.norm.weight": state["norm.weight"],
    }

    def rotary_order(w, heads):
        # llama2.c rotates adjacent coordinates; HF rotates the two head halves.
        return w.reshape(heads, d // h // 2, 2, d).transpose(1, 2).reshape(w.shape)

    for layer in range(l):
        src = f"layers.{layer}."
        dst = f"model.layers.{layer}."
        for a, b in [
            ("attention.wq", "self_attn.q_proj"),
            ("attention.wk", "self_attn.k_proj"),
            ("attention.wv", "self_attn.v_proj"),
            ("attention.wo", "self_attn.o_proj"),
            ("feed_forward.w1", "mlp.gate_proj"),
            ("feed_forward.w2", "mlp.down_proj"),
            ("feed_forward.w3", "mlp.up_proj"),
            ("attention_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),
        ]:
            w = state[src + a + ".weight"]
            if a == "attention.wq":
                w = rotary_order(w, h)
            if a == "attention.wk":
                w = rotary_order(w, kh)
            result[dst + b + ".weight"] = w
    if not torch.equal(state["output.weight"], state["tok_embeddings.weight"]):
        cfg["tie_word_embeddings"] = False
        result["lm_head.weight"] = state["output.weight"]
    return cfg, {k: v.float().contiguous().clone() for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/stories260k"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for remote, local in [
        ("stories260K.pt", "source.pt"),
        ("tok512.model", "tokenizer.model"),
    ]:
        with urlopen(
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/stories260K/{remote}",
            timeout=120,
        ) as response:
            data = response.read()
        hashes[remote] = hashlib.sha256(data).hexdigest()
        (args.out / local).write_bytes(data)
    cfg, state = convert(
        torch.load(args.out / "source.pt", weights_only=True, map_location="cpu")
    )
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(state, str(args.out / "model.safetensors"))
    (args.out / "provenance.json").write_text(
        json.dumps(
            dict(
                model=MODEL, revision=REVISION, subfolder="stories260K", sha256=hashes
            ),
            indent=2,
        )
        + "\n"
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.out / "tokenizer.model"))
    calibration = [
        "Once upon a time, there was a little girl named Lily. She loved to play outside with her friends.",
        "One day, a small boy found a red ball in the garden. He picked it up and ran home to show his mum.",
        "The dog was hungry. It went into the house and looked for some food. Then it saw a big bowl.",
        "A bird sat in a tree and sang a happy song. The sun was warm and the sky was blue.",
        "Tom and his sister went to the park. They played on the swings and had a lot of fun together.",
        "There was a big bear in the forest. He wanted to find a friend who would play with him.",
        "Mum gave the little boy a cookie. He said thank you and smiled because it was very good.",
        "The cat saw a mouse near the door. It ran after the mouse, but the mouse was too fast.",
    ]
    evaluation = [
        "Once upon a time, a little rabbit lived in a green forest. She liked to eat carrots and play with her brother.",
        "Lucy had a yellow toy boat. She took it to the pond and put it in the water.",
        "A child found a lost puppy on the road. They brought it home and gave it water.",
        "The old man had a beautiful garden. Every morning, he watered the flowers and watched the butterflies.",
    ]
    for name, texts in [("calibration", calibration), ("evaluation", evaluation)]:
        (args.out / f"{name}.json").write_text(
            json.dumps([[1] + tokenizer.encode(x) for x in texts], indent=2) + "\n"
        )
    print(f"{MODEL}/stories260K@{REVISION} -> {args.out}")


if __name__ == "__main__":
    main()
