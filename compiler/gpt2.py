"""GPT-2 inference references and fixed-point lowering; no training code."""

import numpy as np
from .quant import (
    requant,
    requant_params,
    sat,
    layernorm_int,
    exp_lut,
    softmax_int,
    quant_weight,
    LN_SHIFT,
    P_BITS,
)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


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


# ---------------------------------------------------------------- float reference


def ln_norm(x, eps=1e-5, rms=False):
    mu = 0 if rms else x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class FloatGPT:
    def __init__(self, f):
        self.f = f

    def forward(self, tokens, stats=None):
        f = self.f
        D, H = f["D"], f["H"]
        hd = D // H
        T = len(tokens)
        h = f["tok_emb"][tokens] + f["pos_emb"][:T]
        rec = (
            (lambda k, v: stats.setdefault(k, []).append(np.abs(v).max()))
            if stats is not None
            else (lambda k, v: None)
        )
        rec("h", h)
        mask = np.tril(np.ones((T, T), dtype=bool))
        for l, lay in enumerate(f["layers"]):
            x = ln_norm(h, f["eps"], f.get("norm") == "rms")
            rec("ln", x)
            w, b = lay["qkv"]
            qkv = x @ w.T + b
            kd = f.get("KH", H) * hd
            q, k, v = qkv[:, :D], qkv[:, D : D + kd], qkv[:, D + kd :]
            if "rope_angles" in f:
                q = rotary_float(q, f["rope_angles"][:T], hd)
                k = rotary_float(k, f["rope_angles"][:T], hd)
            rec(f"q{l}", q)
            rec(f"k{l}", k)
            rec(f"v{l}", v)
            o = np.zeros((T, D))
            for hh in range(H):
                sl = slice(hh * hd, (hh + 1) * hd)
                ks = slice(
                    (hh // (H // f.get("KH", H))) * hd,
                    (hh // (H // f.get("KH", H)) + 1) * hd,
                )
                s = q[:, sl] @ k[:, ks].T * hd**-0.5
                window = lay.get("window", T)
                local_mask = mask & (
                    np.arange(T)[None, :] > np.arange(T)[:, None] - window
                )
                s = np.where(local_mask, s, -np.inf)
                s = s - s.max(-1, keepdims=True)
                p = np.exp(s)
                p = p / p.sum(-1, keepdims=True)
                o[:, sl] = p @ v[:, ks]
            rec(f"o{l}", o)
            w, b = lay["proj"]
            h = h + o @ w.T + b
            rec("h", h)
            x = ln_norm(h, f["eps"], f.get("norm") == "rms")
            rec("ln", x)
            w, b = lay["ffwd1"]
            pre = x @ w.T + b
            rec(f"pre{l}", pre)
            if f.get("gated"):
                gate, up = np.split(pre, 2, axis=-1)
                rec(f"gate{l}", gate)
                rec(f"up{l}", up)
                rec(f"silu{l}", silu(gate))
                ff = silu(gate) * up
            else:
                ff = gelu(pre)
            rec(f"f{l}", ff)
            w, b = lay["ffwd2"]
            h = h + ff @ w.T + b
            rec("h", h)
        x = ln_norm(h, f["eps"], f.get("norm") == "rms")
        rec("ln", x)
        w, b = f["lm_head"]
        logits = x @ w.T + b
        rec("logits", logits)
        return logits


# ---------------------------------------------------------------- quantisation


def silu(x):
    return x / (1 + np.exp(-np.clip(x, -700, 700)))


def rotary_float(x, angles, hd):
    shape = x.shape
    z = x.reshape(len(x), -1, hd)
    a, b = np.split(z, 2, axis=-1)
    c, s = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    return np.concatenate((a * c - b * s, b * c + a * s), axis=-1).reshape(shape)


def rotary_int(x, cos, sin, hd):
    z = np.asarray(x).reshape(-1, hd)
    a, b = np.split(z, 2, axis=-1)
    return sat(
        (np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1) + 8192) >> 14,
        8,
    ).reshape(-1)


def calibrate(f, token_windows):
    stats = {}
    m = FloatGPT(f)
    for toks in token_windows:
        m.forward(np.asarray(toks), stats)
    return {k: float(np.max(v)) for k, v in stats.items()}


def quantise_model(f, stats, bits):
    """Folded float params + calibration ranges -> everything the RTL and IntGPT need."""
    D, H, L = f["D"], f["H"], f["L"]
    hd = D // H
    s_h = (
        max(
            stats["h"],
            float(np.abs(f["tok_emb"]).max()),
            float(np.abs(f["pos_emb"]).max()),
            1e-6,
        )
        * 1.05
        / 32767.0
    )
    s_ln = 2.0**-LN_SHIFT
    q = dict(
        D=D,
        H=H,
        L=L,
        T=f["T"],
        V=f["V"],
        bits=bits,
        s_h=s_h,
        s_ln=s_ln,
        tok_emb=sat(np.rint(f["tok_emb"] / s_h), 16),
        pos_emb=sat(np.rint(f["pos_emb"] / s_h), 16),
        eps_var=int(round(f["eps"] / s_h**2)),
        exp_lut=exp_lut(),
        layers=[],
    )
    q.update(
        architecture=f.get("architecture", "gpt2"),
        norm=f.get("norm", "layer"),
        gated=f.get("gated", False),
        KH=f.get("KH", H),
    )
    if "rope_angles" in f:
        q["rope_cos"] = np.rint(np.cos(f["rope_angles"]) * 16384).astype(np.int64)
        q["rope_sin"] = np.rint(np.sin(f["rope_angles"]) * 16384).astype(np.int64)
    for l, lay in enumerate(f["layers"]):
        ql = {"window": lay.get("window", f["T"])}
        # qkv: per-channel weights, per-tensor outputs so the dot products are consistent
        w, b = lay["qkv"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        s_q, s_k, s_v = (
            max(stats[f"q{l}"], 1e-9) / 127,
            max(stats[f"k{l}"], 1e-9) / 127,
            max(stats[f"v{l}"], 1e-9) / 127,
        )
        s_out = np.concatenate(
            [np.full(D, s_q), np.full(q["KH"] * hd, s_k), np.full(q["KH"] * hd, s_v)]
        )
        ql["qkv"] = matvec_q(wq, sw, b, s_ln, s_out, out_bits=8)
        # attention: scores -> exp index, PV -> o
        s_z = s_q * s_k * hd**-0.5
        ql["attn"] = dict(
            u=requant_params(s_z * 16),
            o=requant_params(2.0**-P_BITS * s_v / (max(stats[f"o{l}"], 1e-9) / 127)),
            s_o=max(stats[f"o{l}"], 1e-9) / 127,
        )
        w, b = lay["proj"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        ql["proj"] = matvec_q(
            wq, sw, b, ql["attn"]["s_o"], np.full(D, s_h), out_bits=16
        )
        w, b = lay["ffwd1"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        s_f = max(stats[f"f{l}"], 1e-9) / 127
        s_pre = max(stats[f"pre{l}"], 1e-9) / 127
        if q["gated"]:
            sg = max(stats[f"gate{l}"], 1e-9) / 127
            su = max(stats[f"up{l}"], 1e-9) / 127
            sa = max(stats[f"silu{l}"], 1e-9) / 127
            ql["gelu"] = sat(np.rint(silu(np.arange(-128, 128) * sg) / sa), 8)
            ql["gate_scale"] = requant_params(sa * su / s_f)
            scales = np.concatenate(
                (np.full(w.shape[0] // 2, sg), np.full(w.shape[0] // 2, su))
            )
        else:
            ql["gelu"] = sat(np.rint(gelu(np.arange(-128, 128) * s_pre) / s_f), 8)
            scales = np.full(w.shape[0], s_pre)
        ql["ffwd1"] = matvec_q(wq, sw, b, s_ln, scales, out_bits=8)
        w, b = lay["ffwd2"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        ql["ffwd2"] = matvec_q(wq, sw, b, s_f, np.full(D, s_h), out_bits=16)
        q["layers"].append(ql)
    # lm_head: one weight scale so every logit shares a scale and argmax is meaningful
    w, b = f["lm_head"]
    wq, sw = quant_weight(w, bits, per_channel=False)
    q["lm_head"] = matvec_q(wq, np.full(w.shape[0], sw), b, s_ln, None, out_bits=32)
    return q


def matvec_q(wq, sw, b, s_x, s_out, out_bits, relu=False):
    """One Linear as the hardware runs it: int weights [out,in], int32 bias at scale
    s_x*sw[j], and per-column requant (M0, n) to s_out (None = raw int32 out)."""
    out = wq.shape[0]
    bq = np.rint(b / (s_x * sw)).astype(np.int64)
    assert np.abs(bq).max() < (1 << 31)
    d = dict(w=wq, sw=sw, b=bq, s_x=s_x, out_bits=out_bits, relu=relu)
    if s_out is not None:
        mn = [requant_params(s_x * sw[j] / s_out[j]) for j in range(out)]
        d["m0"] = np.array([m for m, n in mn], dtype=np.int64)
        d["n"] = np.array([n for m, n in mn], dtype=np.int64)
        d["s_out"] = np.asarray(s_out, dtype=np.float64)
    else:
        d["s_out"] = s_x * sw
    return d


# ---------------------------------------------------------------- integer reference


class IntGPT:
    """Token-by-token forward with a KV cache, all integer. forward() returns the
    int32 logits for every position, exactly what gpt_top.sv streams out."""

    def __init__(self, q, trace=None):
        self.q = q
        self.trace = trace  # dict to fill with intermediate vectors, for debugging RTL
        self.reset()

    def reset(self):
        q = self.q
        self.t = 0
        self.kc = np.zeros(
            (q["L"], q["T"], q.get("KH", q["H"]) * (q["D"] // q["H"])), dtype=np.int64
        )
        self.vc = np.zeros(
            (q["L"], q["T"], q.get("KH", q["H"]) * (q["D"] // q["H"])), dtype=np.int64
        )

    def linear(self, mv, x):
        acc = np.asarray(x, dtype=np.int64) @ mv["w"].T + mv["b"]
        assert np.abs(acc).max() < (1 << 31), "int32 accumulator overflow"
        if mv["out_bits"] == 32:
            return acc
        return requant(acc, mv["m0"], mv["n"], mv["out_bits"], relu=mv["relu"])

    def step(self, token):
        q = self.q
        D, H = q["D"], q["H"]
        hd = D // H
        t = self.t
        assert t < q["T"], "context full; reset()"
        tr = self.trace if self.trace is not None else {}
        tr.setdefault("h_emb", [])
        tr.setdefault("x_ln1", [])
        tr.setdefault("qkv", [])
        tr.setdefault("o", [])
        tr.setdefault("h_attn", [])
        tr.setdefault("x_ln2", [])
        tr.setdefault("f", [])
        tr.setdefault("h_ffwd", [])
        tr.setdefault("x_lnf", [])
        h = sat(q["tok_emb"][token] + q["pos_emb"][t], 16)
        tr["h_emb"].append(h.copy())
        for l, lay in enumerate(q["layers"]):
            x = layernorm_int(h, q["eps_var"], rms=q.get("norm") == "rms")
            tr["x_ln1"].append(x.copy())
            qkv = self.linear(lay["qkv"], x)
            tr["qkv"].append(qkv.copy())
            kd = q.get("KH", H) * hd
            qv, kv, vv = qkv[:D], qkv[D : D + kd], qkv[D + kd :]
            if "rope_cos" in q:
                qv = rotary_int(qv, q["rope_cos"][t], q["rope_sin"][t], hd)
                kv = rotary_int(kv, q["rope_cos"][t], q["rope_sin"][t], hd)
            self.kc[l, t] = kv
            self.vc[l, t] = vv
            o = np.zeros(D, dtype=np.int64)
            m0u, nu = lay["attn"]["u"]
            m0o, no = lay["attn"]["o"]
            for hh in range(H):
                sl = slice(hh * hd, (hh + 1) * hd)
                ks = slice(
                    (hh // (H // q.get("KH", H))) * hd,
                    (hh // (H // q.get("KH", H)) + 1) * hd,
                )
                begin = max(0, t + 1 - lay.get("window", q["T"]))
                scores = self.kc[l, begin : t + 1, ks] @ qv[sl]  # int32
                p = softmax_int(scores, m0u, nu, q["exp_lut"])  # uint16
                acc = p @ self.vc[l, begin : t + 1, ks]  # < 2^28
                o[sl] = requant(acc, m0o, no, 8)
            tr["o"].append(o.copy())
            h = sat(h + self.linear(lay["proj"], o), 16)
            tr["h_attn"].append(h.copy())
            x = layernorm_int(h, q["eps_var"], rms=q.get("norm") == "rms")
            tr["x_ln2"].append(x.copy())
            pre = self.linear(lay["ffwd1"], x)
            if q.get("gated"):
                gate, up = np.split(pre, 2)
                ff = requant(lay["gelu"][gate + 128] * up, *lay["gate_scale"], 8)
            else:
                ff = lay["gelu"][pre + 128]
            tr["f"].append(ff.copy())
            h = sat(h + self.linear(lay["ffwd2"], ff), 16)
            tr["h_ffwd"].append(h.copy())
        x = layernorm_int(h, q["eps_var"], rms=q.get("norm") == "rms")
        tr["x_lnf"].append(x.copy())
        logits = self.linear(q["lm_head"], x)
        self.t += 1
        return logits

    def forward(self, tokens):
        self.reset()
        return np.stack([self.step(int(tk)) for tk in tokens])

    def generate(self, prompt, n_new):
        """Greedy. Context is limited to T tokens total, like the hardware."""
        self.reset()
        toks = list(prompt)
        logits = None
        for tk in toks:
            logits = self.step(tk)
        for _ in range(n_new):
            if self.t >= self.q["T"]:
                break
            nxt = int(np.argmax(logits))
            toks.append(nxt)
            logits = self.step(nxt)
        return toks
