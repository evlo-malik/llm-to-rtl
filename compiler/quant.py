# Quantisation and the fixed-point requantisation used everywhere downstream.
# Symmetric, no zero points: real = scale * q. Per-output-channel weights unless a
# consumer needs one scale across channels (q and k for the dot product, lm_head for
# the argmax). Requantisation is Jacob et al. 2018 style: q_y = (acc * M0) >> n with
# M0 normally has 16 significant bits. Large expansion ratios use an integer M0.
import numpy as np
import math


def qrange(bits):
    if bits == 2:
        return -1, 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def quant_weight(w, bits, per_channel=True):
    """w [out, in] float -> (wq int64 [out, in], scale [out] or scalar).
    8/4 bit: absmax per row. Ternary: BitNet-style absmean per row, round, clip."""
    w = np.asarray(w, dtype=np.float64)
    lo, hi = qrange(bits)
    if bits == 2:
        s = np.mean(np.abs(w), axis=1) if per_channel else np.mean(np.abs(w))
    else:
        s = np.max(np.abs(w), axis=1) / hi if per_channel else np.max(np.abs(w)) / hi
    s = np.where(s == 0, 1.0, s) if per_channel else (s if s != 0 else 1.0)
    wq = np.clip(np.rint(w / (s[:, None] if per_channel else s)), lo, hi).astype(
        np.int64
    )
    return wq, s


def requant_params(r):
    """Positive scale -> integer multiplier and nonnegative shift; product fits INT64."""
    r = float(r)
    if r <= 0:
        return 0, 0
    n = 15 - int(np.floor(np.log2(r)))
    m0 = int(np.rint(r * 2.0**n))
    if n < 0:
        if not np.isfinite(r) or r >= 2**31:
            raise ValueError("requant multiplier exceeds INT31")
        return int(round(r)), 0
    if m0 >= 1 << 16:
        m0 >>= 1
        n -= 1
    assert 0 <= n < 64, f"requant shift {n} out of range for r={r}"
    assert (1 << 15) <= m0 < (1 << 16)
    return m0, n


def requant(acc, m0, n, bits, relu=False):
    """Integer requantise: multiply by M0, add half,
    arithmetic shift right n, clip to `bits`. acc, m0, n may be arrays."""
    acc = np.asarray(acc, dtype=np.int64)
    m0 = np.asarray(m0, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    prod = acc * m0
    half = np.where(n > 0, np.int64(1) << np.maximum(n - 1, 0), 0)
    y = (prod + half) >> n
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    y = np.clip(y, lo, hi)
    if relu:
        y = np.maximum(y, 0)
    return y


def requant_u8(z, m0, n):
    """Unsigned variant for the softmax index: z >= 0, result clipped to 0..255."""
    z = np.asarray(z, dtype=np.int64)
    half = (1 << (n - 1)) if n > 0 else 0
    return np.clip((z * m0 + half) >> n, 0, 255)


def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


LN_SHIFT = 4  # LayerNorm output is int8 with scale 2^-4: 16 units per sigma
LN_P = 24  # reciprocal precision: inv = 2^24 // std
P_BITS = 15  # softmax probabilities are uint16 with scale 2^-15
EXP_BITS = 16  # exp LUT entries are 2^16 * exp(-u/16), 17 bits wide
LUT_STEP = 16  # index u = 16 * (logit gap); u = 255 is a gap of 15.9 nats
RECIP_Q = 36  # softmax reciprocal: R = 2^36 // sum


def layernorm_int(h, eps_var=0, rms=False, out_bits=8, shift=LN_SHIFT, precision=LN_P):
    """int16 vector -> int8 vector, normalised, scale 2^-4, no affine (folded away).
    Mirrors rtl/layernorm.sv step for step."""
    h = np.asarray(h, dtype=np.int64)
    d = len(h)
    if not 2 <= d <= 4096:
        raise ValueError("LayerNorm width must be in 2..4096")
    mean = 0 if rms else int(h.sum()) // d
    c = h - mean
    var = sum(int(x) * int(x) for x in c) // d
    std = math.isqrt(var + int(eps_var))
    if std == 0:
        std = 1
    inv = (1 << precision) // std
    y = (c * inv + (1 << (precision - shift - 1))) >> (precision - shift)
    return sat(y, out_bits)


def exp_lut():
    u = np.arange(256)
    return np.rint((1 << EXP_BITS) * np.exp(-u / LUT_STEP)).astype(np.int64)


def softmax_int(scores, m0u, nu, lut):
    """int32 scores (length t+1) -> uint16 probabilities, scale 2^-15.
    Mirrors the softmax phase of rtl/attention.sv."""
    s = np.asarray(scores, dtype=np.int64)
    z = int(s.max()) - s
    u = requant_u8(z, m0u, nu)
    e = lut[u]
    total = int(e.sum())
    recip = (1 << RECIP_Q) // total
    p = (e * recip) >> (RECIP_Q - P_BITS)
    assert p.max() <= (1 << P_BITS)
    return p
