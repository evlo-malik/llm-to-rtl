"""Lower Hugging Face GPT-2 tensors into the common decoder representation."""

import numpy as np


def fold_gpt2(cfg, st, context):
    d, h, layers = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    if not 1 <= layers <= 65535 or cfg["vocab_size"] < 2:
        raise ValueError("invalid layer count or vocabulary size")
    if not 2 <= d <= 4096 or not 1 <= h <= d or d % h:
        raise ValueError("GPT-2 width must be in 2..4096 and divisible by n_head")
    if not 2 <= context <= min(cfg["n_positions"], 1024):
        raise ValueError("context must be between 2 and min(n_positions, 1024)")
    if cfg.get("activation_function", "gelu_new") != "gelu_new":
        raise ValueError("only GPT-2 gelu_new is supported")
    for option, expected in [
        ("scale_attn_weights", True),
        ("scale_attn_by_inverse_layer_idx", False),
        ("reorder_and_upcast_attn", False),
        ("add_cross_attention", False),
    ]:
        if cfg.get(option, expected) != expected:
            raise ValueError(f"unsupported GPT-2 option: {option}")

    def tensor(name, shape):
        if name not in st or st[name].shape != shape:
            raise ValueError(f"missing or invalid tensor {name}: expected {shape}")
        x = st[name].astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite tensor: {name}")
        return x

    vocab = cfg["vocab_size"]
    f = dict(
        D=d,
        H=h,
        L=layers,
        T=context,
        V=vocab,
        eps=cfg.get("layer_norm_epsilon", 1e-5),
        tok_emb=tensor("transformer.wte.weight", (vocab, d)),
        pos_emb=tensor("transformer.wpe.weight", (cfg["n_positions"], d))[:context],
        layers=[],
    )
    inner = cfg.get("n_inner") or 4 * d
    for l in range(layers):
        prefix = f"transformer.h.{l}."
        gamma = tensor(prefix + "ln_1.weight", (d,))
        beta = tensor(prefix + "ln_1.bias", (d,))
        qkv = fold_ln(
            tensor(prefix + "attn.c_attn.weight", (d, 3 * d)).T,
            tensor(prefix + "attn.c_attn.bias", (3 * d,)),
            gamma,
            beta,
        )
        gamma = tensor(prefix + "ln_2.weight", (d,))
        beta = tensor(prefix + "ln_2.bias", (d,))
        ff1 = fold_ln(
            tensor(prefix + "mlp.c_fc.weight", (d, inner)).T,
            tensor(prefix + "mlp.c_fc.bias", (inner,)),
            gamma,
            beta,
        )
        f["layers"].append(
            dict(
                qkv=qkv,
                ffwd1=ff1,
                proj=(
                    tensor(prefix + "attn.c_proj.weight", (d, d)).T,
                    tensor(prefix + "attn.c_proj.bias", (d,)),
                ),
                ffwd2=(
                    tensor(prefix + "mlp.c_proj.weight", (inner, d)).T,
                    tensor(prefix + "mlp.c_proj.bias", (d,)),
                ),
            )
        )
    head = (
        tensor("lm_head.weight", (vocab, d)) if "lm_head.weight" in st else f["tok_emb"]
    )
    if "lm_head.weight" not in st and not cfg.get("tie_word_embeddings", True):
        raise ValueError("untied GPT-2 checkpoint is missing lm_head.weight")
    f["lm_head"] = fold_ln(
        head,
        None,
        tensor("transformer.ln_f.weight", (d,)),
        tensor("transformer.ln_f.bias", (d,)),
    )
    return f


def fold_ln(w, b, gamma, beta):
    """LN(x)*gamma+beta followed by x@W.T+b  ==  LNnorm(x) @ (W*gamma).T + (b + W@beta)."""
    w2 = w * gamma[None, :]
    b2 = (np.zeros(w.shape[0]) if b is None else b) + w @ beta
    return w2, b2
