"""Compile pretrained Llama matrix weights into fixed-coefficient RTL."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from emit import const_array_sv, const_wrapper_sv
from quant import quant_weight


def matrices_llama(cfg, st, bits):
    L = cfg["num_hidden_layers"]
    out = []

    def add(name, w_out_in, gamma=None, per_channel=True):
        w = w_out_in.astype(np.float64)
        if gamma is not None:                       # RMSNorm weight folded into the consumer
            w = w * gamma[None, :].astype(np.float64)
        wq, sw = quant_weight(w, bits, per_channel=per_channel)
        out.append(dict(name=name, w=wq.T.copy(), b=None, sw=sw, src=name))

    for l in range(L):
        p = f"model.layers.{l}."
        g_in = st[p + "input_layernorm.weight"]
        g_post = st[p + "post_attention_layernorm.weight"]
        for m in ("q_proj", "k_proj", "v_proj"):
            add(f"layer{l}_{m}", st[p + f"self_attn.{m}.weight"], g_in)
        add(f"layer{l}_o_proj", st[p + "self_attn.o_proj.weight"])
        for m in ("gate_proj", "up_proj"):
            add(f"layer{l}_{m}", st[p + f"mlp.{m}.weight"], g_post)
        add(f"layer{l}_down_proj", st[p + "mlp.down_proj.weight"])
    lm = st["lm_head.weight"] if "lm_head.weight" in st else st["model.embed_tokens.weight"]   # tied
    add("lm_head", lm, st["model.norm.weight"], per_channel=False)
    return out, None



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", type=int, choices=(8, 4, 2), default=8)
    args = ap.parse_args()
    cfg = json.loads((args.model_dir / "config.json").read_text())
    if cfg.get("model_type") != "llama":
        ap.error("supported matrix adapter: llama")
    tensors = {k: v.float().numpy() for k, v in load_file(str(args.model_dir / "model.safetensors")).items()}
    matrices, _ = matrices_llama(cfg, tensors, args.bits)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"weight_storage": "constant_logic", "bits": args.bits, "matrices": []}
    for m in matrices:
        name = m["name"]
        core = "const_" + name
        (args.out / (core + ".sv")).write_text(const_array_sv(core, m["w"]))
        (args.out / (name + ".sv")).write_text(const_wrapper_sv(name, core, *m["w"].shape, m["b"]))
        manifest["matrices"].append({"name": name, "shape": list(m["w"].shape)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
