# Shared driver for anything with the serial matvec interface (matvec_rom.sv and the
# generated mv_* modules): push K int8 values, collect N int32 values, compare with
# torch's integer linear. Used by test_matvec_rom.py and by every generated test.
import numpy as np
import torch
import torch.nn.functional as F
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_pull(dut, x, n, gap=0, timeout=None):
    """Send vector x (int8 list), return the n int32 outputs. gap idle cycles between
    inputs. Also checks nothing comes out early and busy drops at the end."""
    k = len(x)
    got = []
    for i, v in enumerate(x):
        dut.in_valid.value = 1
        dut.in_data.value = int(v) & 0xFF
        await RisingEdge(dut.clk)
        for _ in range(gap):
            dut.in_valid.value = 0
            dut.in_data.value = 0x5A          # junk while idle
            await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.in_data.value = 0
    cycles = 0
    limit = timeout or (50 * (k + n) + 200 * k * n // 4 + 10000)
    while len(got) < n:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        cycles += 1
        if dut.out_valid.value == 1:
            got.append(dut.out_data.value.to_signed())
        assert cycles < limit, f"only {len(got)}/{n} outputs after {cycles} cycles"
    # busy must drop within a few cycles of the last output, with nothing more emitted
    for _ in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert dut.out_valid.value == 0, "extra output after N values"
        if dut.busy.value == 0:
            break
    assert dut.busy.value == 0, "busy still high after the last output"
    return np.array(got, dtype=np.int64), cycles


def expected(x, w_kn, bias):
    """torch does the reference in int64: linear(x, W^T, b) = x @ W + b."""
    xt = torch.tensor(np.asarray(x, dtype=np.int64))
    wt = torch.tensor(np.asarray(w_kn, dtype=np.int64).T.copy())
    bt = None if bias is None else torch.tensor(np.asarray(bias, dtype=np.int64))
    return F.linear(xt, wt, bt).numpy()


def edge_vectors(k, rng):
    lo, hi = -128, 127
    yield np.zeros(k, dtype=np.int64)
    yield np.full(k, lo, dtype=np.int64)
    yield np.full(k, hi, dtype=np.int64)
    e = np.zeros(k, dtype=np.int64); e[0] = 1; yield e
    e = np.zeros(k, dtype=np.int64); e[-1] = lo; yield e
    yield rng.integers(lo, hi + 1, size=k, dtype=np.int64)


async def check_module(dut, w_kn, bias, trials=4, log=None):
    k, n = np.asarray(w_kn).shape
    rng = np.random.default_rng(k * 1000 + n)
    await reset(dut)
    total = 0
    vecs = list(edge_vectors(k, rng)) + [rng.integers(-128, 128, size=k, dtype=np.int64) for _ in range(trials)]
    for i, x in enumerate(vecs):
        got, cycles = await push_pull(dut, x, n, gap=(i % 3 == 1))
        exp = expected(x, w_kn, bias)
        assert np.array_equal(got, exp), (
            f"vector {i}: first mismatch at column {int(np.argmax(got != exp))}: "
            f"hardware {got[got != exp][:4]}, torch {exp[got != exp][:4]}")
        total += cycles
    if log:
        log(f"{k}x{n}: {len(vecs)} vectors bit-exact, {total // len(vecs)} cycles per vector")
    return total // len(vecs)
