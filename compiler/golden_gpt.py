# The tiny GPT the way the hardware sees it, in two forms.
#
# FloatGPT: LayerNorm affine folded into the Linear that follows it, everything else
# as Karpathy wrote it. Used to check the folding against the trained torch model
# and to collect activation ranges for calibration.
#
# IntGPT: the same dataflow in integers, token by token with a KV cache, using only
# the operations the RTL has (int8 matvec, requant, integer LayerNorm, LUT softmax,
# saturating adds). This is the bit-exact reference for gpt_top.sv. Nothing in here
# is approximate with respect to the hardware; it is approximate with respect to
# the float model, and that gap is what model/report numbers measure.
import json
import os

import numpy as np

from quant import (requant, requant_params, requant_u8, sat, layernorm_int, exp_lut,
                   softmax_int, quant_weight, LN_SHIFT, P_BITS)


# ---------------------------------------------------------------- loading + folding

def load_karpathy(model_dir):
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    st = {k: v.astype(np.float64) for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
    return cfg, st


def fold_ln(w, b, gamma, beta):
    """LN(x)*gamma+beta followed by x@W.T+b  ==  LNnorm(x) @ (W*gamma).T + (b + W@beta)."""
    w2 = w * gamma[None, :]
    b2 = (np.zeros(w.shape[0]) if b is None else b) + w @ beta
    return w2, b2


def fold_karpathy(cfg, st):
    """Returns the folded float parameters in the layout the hardware uses."""
    D, H, L = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    layers = []
    for l in range(L):
        p = f"blocks.{l}."
        wq = np.concatenate([st[p + f"sa.heads.{h}.query.weight"] for h in range(H)])
        wk = np.concatenate([st[p + f"sa.heads.{h}.key.weight"] for h in range(H)])
        wv = np.concatenate([st[p + f"sa.heads.{h}.value.weight"] for h in range(H)])
        wqkv = np.concatenate([wq, wk, wv])                          # [3D, D]
        wqkv, bqkv = fold_ln(wqkv, None, st[p + "ln1.weight"], st[p + "ln1.bias"])
        w1, b1 = fold_ln(st[p + "ffwd.net.0.weight"], st[p + "ffwd.net.0.bias"],
                         st[p + "ln2.weight"], st[p + "ln2.bias"])
        layers.append(dict(
            qkv=(wqkv, bqkv),
            proj=(st[p + "sa.proj.weight"], st[p + "sa.proj.bias"]),
            ffwd1=(w1, b1),
            ffwd2=(st[p + "ffwd.net.2.weight"], st[p + "ffwd.net.2.bias"]),
        ))
    wlm, blm = fold_ln(st["lm_head.weight"], st["lm_head.bias"], st["ln_f.weight"], st["ln_f.bias"])
    return dict(tok_emb=st["token_embedding_table.weight"], pos_emb=st["position_embedding_table.weight"],
                layers=layers, lm_head=(wlm, blm), D=D, H=H, L=L, T=cfg["block_size"], V=cfg["vocab_size"])


# ---------------------------------------------------------------- float reference

def ln_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
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
        rec = (lambda k, v: stats.setdefault(k, []).append(np.abs(v).max())) if stats is not None else (lambda k, v: None)
        rec("h", h)
        mask = np.tril(np.ones((T, T), dtype=bool))
        for l, lay in enumerate(f["layers"]):
            x = ln_norm(h)
            rec("ln", x)
            w, b = lay["qkv"]
            qkv = x @ w.T + b
            q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
            rec(f"q{l}", q); rec(f"k{l}", k); rec(f"v{l}", v)
            o = np.zeros((T, D))
            for hh in range(H):
                sl = slice(hh * hd, (hh + 1) * hd)
                s = q[:, sl] @ k[:, sl].T * hd ** -0.5
                s = np.where(mask, s, -np.inf)
                s = s - s.max(-1, keepdims=True)
                p = np.exp(s)
                p = p / p.sum(-1, keepdims=True)
                o[:, sl] = p @ v[:, sl]
            rec(f"o{l}", o)
            w, b = lay["proj"]
            h = h + o @ w.T + b
            rec("h", h)
            x = ln_norm(h)
            rec("ln", x)
            w, b = lay["ffwd1"]
            ff = np.maximum(x @ w.T + b, 0)
            rec(f"f{l}", ff)
            w, b = lay["ffwd2"]
            h = h + ff @ w.T + b
            rec("h", h)
        x = ln_norm(h)
        rec("ln", x)
        w, b = f["lm_head"]
        logits = x @ w.T + b
        rec("logits", logits)
        return logits


# ---------------------------------------------------------------- quantisation

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
    s_h = stats["h"] / 32767.0
    s_ln = 2.0 ** -LN_SHIFT
    q = dict(D=D, H=H, L=L, T=f["T"], V=f["V"], bits=bits, s_h=s_h, s_ln=s_ln,
             tok_emb=sat(np.rint(f["tok_emb"] / s_h), 16),
             pos_emb=sat(np.rint(f["pos_emb"] / s_h), 16),
             exp_lut=exp_lut(), layers=[])
    for l, lay in enumerate(f["layers"]):
        ql = {}
        # qkv: per-channel weights, per-tensor outputs so the dot products are consistent
        w, b = lay["qkv"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        s_q, s_k, s_v = stats[f"q{l}"] / 127, stats[f"k{l}"] / 127, stats[f"v{l}"] / 127
        s_out = np.concatenate([np.full(D, s_q), np.full(D, s_k), np.full(D, s_v)])
        ql["qkv"] = matvec_q(wq, sw, b, s_ln, s_out, out_bits=8)
        # attention: scores -> exp index, PV -> o
        s_z = s_q * s_k * hd ** -0.5
        ql["attn"] = dict(u=requant_params(s_z * 16), o=requant_params(2.0 ** -P_BITS * s_v / (stats[f"o{l}"] / 127)),
                          s_o=stats[f"o{l}"] / 127)
        w, b = lay["proj"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        ql["proj"] = matvec_q(wq, sw, b, ql["attn"]["s_o"], np.full(D, s_h), out_bits=16)
        w, b = lay["ffwd1"]
        wq, sw = quant_weight(w, bits, per_channel=True)
        s_f = stats[f"f{l}"] / 127
        ql["ffwd1"] = matvec_q(wq, sw, b, s_ln, np.full(w.shape[0], s_f), out_bits=8, relu=True)
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
        self.trace = trace          # dict to fill with intermediate vectors, for debugging RTL
        self.reset()

    def reset(self):
        q = self.q
        self.t = 0
        self.kc = np.zeros((q["L"], q["T"], q["D"]), dtype=np.int64)
        self.vc = np.zeros((q["L"], q["T"], q["D"]), dtype=np.int64)

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
        tr.setdefault("h_emb", []); tr.setdefault("x_ln1", []); tr.setdefault("qkv", []); tr.setdefault("o", [])
        tr.setdefault("h_attn", []); tr.setdefault("x_ln2", []); tr.setdefault("f", []); tr.setdefault("h_ffwd", []); tr.setdefault("x_lnf", [])
        h = sat(q["tok_emb"][token] + q["pos_emb"][t], 16)
        tr["h_emb"].append(h.copy())
        for l, lay in enumerate(q["layers"]):
            x = layernorm_int(h)
            tr["x_ln1"].append(x.copy())
            qkv = self.linear(lay["qkv"], x)
            tr["qkv"].append(qkv.copy())
            qv, kv, vv = qkv[:D], qkv[D:2 * D], qkv[2 * D:]
            self.kc[l, t] = kv
            self.vc[l, t] = vv
            o = np.zeros(D, dtype=np.int64)
            m0u, nu = lay["attn"]["u"]
            m0o, no = lay["attn"]["o"]
            for hh in range(H):
                sl = slice(hh * hd, (hh + 1) * hd)
                scores = self.kc[l, :t + 1, sl] @ qv[sl]                       # int32
                p = softmax_int(scores, m0u, nu, q["exp_lut"])                # uint16
                acc = p @ self.vc[l, :t + 1, sl]                              # < 2^28
                o[sl] = requant(acc, m0o, no, 8)
            tr["o"].append(o.copy())
            h = sat(h + self.linear(lay["proj"], o), 16)
            tr["h_attn"].append(h.copy())
            x = layernorm_int(h)
            tr["x_ln2"].append(x.copy())
            ff = self.linear(lay["ffwd1"], x)
            tr["f"].append(ff.copy())
            h = sat(h + self.linear(lay["ffwd2"], ff), 16)
            tr["h_ffwd"].append(h.copy())
        x = layernorm_int(h)
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
