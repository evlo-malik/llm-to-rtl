"""Shared floating-point and integer decoder references."""

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


# ---------------------------------------------------------------- float reference


def ln_norm(x, eps=1e-5, rms=False):
    mu = 0 if rms else x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class FloatDecoder:
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
    m = FloatDecoder(f)
    for toks in token_windows:
        m.forward(np.asarray(toks), stats)
    return {k: float(np.max(v)) for k, v in stats.items()}


def quantise_model(f, stats, bits, activation_bits=8):
    """Folded float params + calibration ranges -> everything the RTL and IntDecoder need."""
    if activation_bits not in (8, 16):
        raise ValueError("activation_bits must be 8 or 16")
    residual_bits = 32 if activation_bits == 16 else 16
    ln_shift = 8 if activation_bits == 16 else LN_SHIFT
    activation_max = (1 << (activation_bits - 1)) - 1
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
        / float((1 << (residual_bits - 1)) - 1)
    )
    s_ln = 2.0**-ln_shift
    q = dict(
        D=D,
        H=H,
        L=L,
        T=f["T"],
        V=f["V"],
        bits=bits,
        activation_bits=activation_bits,
        residual_bits=residual_bits,
        ln_shift=ln_shift,
        s_h=s_h,
        s_ln=s_ln,
        tok_emb=sat(np.rint(f["tok_emb"] / s_h), residual_bits),
        pos_emb=sat(np.rint(f["pos_emb"] / s_h), residual_bits),
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
            wq, sw, b, ql["attn"]["s_o"], np.full(D, s_h), out_bits=residual_bits
        )
        w, b = lay["ffwd1"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        s_f = max(stats[f"f{l}"], 1e-9) / activation_max
        s_pre = max(stats[f"pre{l}"], 1e-9) / activation_max
        lut_shift = 4 if activation_bits == 16 else 0
        grid = np.arange(
            -(1 << (activation_bits - 1)), 1 << (activation_bits - 1), 1 << lut_shift
        )
        ql["lut_shift"] = lut_shift
        if q["gated"]:
            sg = max(stats[f"gate{l}"], 1e-9) / activation_max
            su = max(stats[f"up{l}"], 1e-9) / activation_max
            sa = max(stats[f"silu{l}"], 1e-9) / activation_max
            ql["gelu"] = sat(np.rint(silu(grid * sg) / sa), activation_bits)
            ql["gate_scale"] = requant_params(sa * su / s_f)
            scales = np.concatenate(
                (np.full(w.shape[0] // 2, sg), np.full(w.shape[0] // 2, su))
            )
        else:
            ql["gelu"] = sat(np.rint(gelu(grid * s_pre) / s_f), activation_bits)
            scales = np.full(w.shape[0], s_pre)
        ql["ffwd1"] = matvec_q(wq, sw, b, s_ln, scales, out_bits=activation_bits)
        w, b = lay["ffwd2"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        ql["ffwd2"] = matvec_q(wq, sw, b, s_f, np.full(D, s_h), out_bits=residual_bits)
        for name in ("qkv", "ffwd1", "ffwd2"):
            ql[name]["in_bits"] = activation_bits
        ql["proj"]["in_bits"] = 8
        q["layers"].append(ql)
    # lm_head: one weight scale so every logit shares a scale and argmax is meaningful
    w, b = f["lm_head"]
    wq, sw = quant_weight(w, bits, per_channel=False)
    q["lm_head"] = matvec_q(wq, np.full(w.shape[0], sw), b, s_ln, None, out_bits=32)
    q["lm_head"]["in_bits"] = activation_bits
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


def lut_index(x, bits, shift):
    half = (1 << (shift - 1)) if shift else 0
    return np.clip(
        (x + (1 << (bits - 1)) + half) >> shift, 0, (1 << (bits - shift)) - 1
    )


class IntDecoder:
    """Token-by-token forward with a KV cache, all integer. forward() returns the
    int32 logits for every position, exactly what model_top.sv streams out."""

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
        if "m0" not in mv:
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
        rb = q.get("residual_bits", 16)
        ab = q.get("activation_bits", 8)
        h = sat(q["tok_emb"][token] + q["pos_emb"][t], rb)
        tr["h_emb"].append(h.copy())
        for l, lay in enumerate(q["layers"]):
            x = layernorm_int(
                h,
                q["eps_var"],
                rms=q.get("norm") == "rms",
                out_bits=ab,
                shift=q.get("ln_shift", LN_SHIFT),
                precision=48 if rb == 32 else 24,
            )
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
            h = sat(h + self.linear(lay["proj"], o), rb)
            tr["h_attn"].append(h.copy())
            x = layernorm_int(
                h,
                q["eps_var"],
                rms=q.get("norm") == "rms",
                out_bits=ab,
                shift=q.get("ln_shift", LN_SHIFT),
                precision=48 if rb == 32 else 24,
            )
            tr["x_ln2"].append(x.copy())
            pre = self.linear(lay["ffwd1"], x)
            if q.get("gated"):
                gate, up = np.split(pre, 2)
                idx = lut_index(gate, ab, lay.get("lut_shift", 0))
                ff = requant(lay["gelu"][idx] * up, *lay["gate_scale"], ab)
            else:
                ff = lay["gelu"][lut_index(pre, ab, lay.get("lut_shift", 0))]
            tr["f"].append(ff.copy())
            h = sat(h + self.linear(lay["ffwd2"], ff), rb)
            tr["h_ffwd"].append(h.copy())
        x = layernorm_int(
            h,
            q["eps_var"],
            rms=q.get("norm") == "rms",
            out_bits=ab,
            shift=q.get("ln_shift", LN_SHIFT),
            precision=48 if rb == 32 else 24,
        )
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
