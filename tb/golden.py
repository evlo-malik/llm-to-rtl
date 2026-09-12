# Reference maths and bit-packing shared by the testbenches. NumPy int64 is the oracle;
# nothing in here knows about clocks.
import numpy as np


def latency(n):
    """Cycles from x_valid to y_valid in top.sv."""
    return 2 * n - 1


def rand_i8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int64)


def matmul(a, w):
    return np.asarray(a, dtype=np.int64) @ np.asarray(w, dtype=np.int64)


def pack(vals, bits):
    """Little-endian lane packing: element i lands in bits [bits*i +: bits]."""
    mask = (1 << bits) - 1
    word = 0
    for i, v in enumerate(vals):
        word |= (int(v) & mask) << (bits * i)
    return word


def unpack(word, n, bits, signed=True):
    mask = (1 << bits) - 1
    out = []
    for i in range(n):
        v = (int(word) >> (bits * i)) & mask
        if signed and v >> (bits - 1):
            v -= 1 << bits
        out.append(v)
    return np.array(out, dtype=np.int64)
