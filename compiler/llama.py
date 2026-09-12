"""Lower dense rotary decoder checkpoints to the common transformer representation."""

import numpy as np


def fold_llama(cfg, state, context):
    family = cfg["model_type"]
    if family not in ("llama", "qwen2", "mistral"):
        raise ValueError(f"unsupported rotary decoder: {family}")
    d, h, l, v = (
        cfg[k]
        for k in (
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
        )
    )
    kh = cfg.get("num_key_value_heads", h)
    hd = cfg.get("head_dim") or d // h
    if not 2 <= d <= 4096 or d != h * hd or hd % 2 or not 1 <= kh <= h or h % kh:
        raise ValueError(
            "require width 2..4096, even head dimension, and query heads divisible by KV heads"
        )
    if not 1 <= l <= 4096 or v < 2:
        raise ValueError("invalid layer count or vocabulary")
    if not 2 <= context <= min(cfg.get("max_position_embeddings", 2048), 1024):
        raise ValueError("context must be in 2..min(max_position_embeddings, 1024)")
    if cfg.get("hidden_act", "silu") != "silu":
        raise ValueError("rotary decoder requires silu")
    if cfg.get("quantization_config"):
        raise ValueError(
            "packed quantised checkpoints need an explicit dequantisation adapter"
        )
    rope = cfg.get("rope_scaling") or {}
    rt = rope.get("rope_type", rope.get("type", "default"))
    if rt not in ("default", "linear", "llama3"):
        raise ValueError(f"unsupported RoPE scaling: {rt}")
    theta = float(cfg.get("rope_theta", 10000.0))
    inv = 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    if rt == "linear":
        inv /= float(rope["factor"])
    elif rt == "llama3":
        factor, low, high, original = (
            float(rope[k])
            for k in (
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            )
        )
        wavelength = 2 * np.pi / inv
        smooth = (original / wavelength - low) / (high - low)
        inv = np.where(
            wavelength > original / low,
            inv / factor,
            np.where(
                wavelength < original / high,
                inv,
                (1 - smooth) * inv / factor + smooth * inv,
            ),
        )
    if not np.isfinite(inv).all():
        raise ValueError("invalid rotary frequencies")
    # A fixed context shorter than the sliding window has ordinary causal masking.
    window = (
        cfg.get("sliding_window")
        if family == "mistral" or cfg.get("use_sliding_window", False)
        else None
    )
    window = min(context, int(window)) if window else context
    if window < 1:
        raise ValueError("invalid sliding window")

    def tensor(name, shape):
        if name not in state or state[name].shape != shape:
            raise ValueError(f"missing or invalid tensor {name}: expected {shape}")
        return state[name].astype(np.float64)

    def bias(name, size, required):
        return tensor(name, (size,)) if required else np.zeros(size)

    kd = kh * hd
    f = dict(
        D=d,
        H=h,
        KH=kh,
        L=l,
        V=v,
        T=context,
        architecture=family,
        norm="rms",
        gated=True,
        eps=float(cfg.get("rms_norm_eps", 1e-6)),
        tok_emb=tensor("model.embed_tokens.weight", (v, d)),
        pos_emb=np.zeros((context, d)),
        rope_angles=np.arange(context)[:, None] * inv[None, :],
        layers=[],
    )
    inner = cfg["intermediate_size"]
    if inner < 1:
        raise ValueError("invalid intermediate size")
    layer_types = cfg.get("layer_types")
    if layer_types is None:
        layer_types = [
            "sliding_attention"
            if window < context
            and (family != "qwen2" or i >= cfg.get("max_window_layers", l))
            else "full_attention"
            for i in range(l)
        ]
    if len(layer_types) != l or any(
        x not in ("sliding_attention", "full_attention") for x in layer_types
    ):
        raise ValueError("unsupported attention layer types")
    for layer in range(l):
        p = f"model.layers.{layer}."
        g1 = tensor(p + "input_layernorm.weight", (d,))
        g2 = tensor(p + "post_attention_layernorm.weight", (d,))
        qbias = cfg.get("attention_bias", family == "qwen2")
        qkv_w, qkv_b = [], []
        for name, size in [("q_proj", d), ("k_proj", kd), ("v_proj", kd)]:
            key = p + "self_attn." + name
            qkv_w.append(tensor(key + ".weight", (size, d)) * g1[None, :])
            qkv_b.append(bias(key + ".bias", size, qbias))
        first_w, first_b = [], []
        for name in ("gate_proj", "up_proj"):
            key = p + "mlp." + name
            first_w.append(tensor(key + ".weight", (inner, d)) * g2[None, :])
            first_b.append(bias(key + ".bias", inner, cfg.get("mlp_bias", False)))
        # Qwen2 only biases Q/K/V; Llama attention_bias also covers O.
        output_bias = cfg.get("attention_bias", False) if family != "qwen2" else False
        f["layers"].append(
            dict(
                qkv=(np.concatenate(qkv_w), np.concatenate(qkv_b)),
                proj=(
                    tensor(p + "self_attn.o_proj.weight", (d, d)),
                    bias(p + "self_attn.o_proj.bias", d, output_bias),
                ),
                ffwd1=(np.concatenate(first_w), np.concatenate(first_b)),
                ffwd2=(
                    tensor(p + "mlp.down_proj.weight", (d, inner)),
                    bias(p + "mlp.down_proj.bias", d, cfg.get("mlp_bias", False)),
                ),
                window=window if layer_types[layer] == "sliding_attention" else context,
            )
        )
    head = (
        "model.embed_tokens.weight"
        if cfg.get("tie_word_embeddings", False)
        else "lm_head.weight"
    )
    f["lm_head"] = (
        tensor(head, (v, d)) * tensor("model.norm.weight", (d,))[None, :],
        np.zeros(v),
    )
    return f
