"""Normalisation extremes, including values whose sum of squares exceeds 64 bits."""

import os
import sys
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.environ["SOURCE_ROOT"])
from compiler.quant import layernorm_int


@cocotb.test()
async def wide_normalisation(dut):
    d = int(os.environ["NORM_D"])
    rms = bool(int(os.environ["NORM_RMS"]))
    eps = 12345678901
    rng = np.random.default_rng(291)
    rows = [
        np.zeros(d, dtype=np.int64),
        np.ones(d, dtype=np.int64),
        np.full(d, -(2**31), dtype=np.int64),
        np.full(d, 2**31 - 1, dtype=np.int64),
        np.array([-(2**31) if i % 2 else 2**31 - 1 for i in range(d)], dtype=np.int64),
    ]
    rows += [rng.integers(-(2**31), 2**31, d, dtype=np.int64) for _ in range(3)]
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0

    async def tick():
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

    await tick()
    await tick()
    dut.rst_n.value = 1
    for x in rows:
        assert not dut.busy.value
        for i, v in enumerate(x):
            dut.in_data.value = int(v) & 0xFFFFFFFF
            dut.in_valid.value = 1
            await tick()
            dut.in_valid.value = 0
            if i < d - 1 and i % 3 == 0:
                await tick()
        got = []
        for _ in range(4 * d + 200):
            await tick()
            if dut.out_valid.value:
                got.append(dut.out_data.value.to_signed())
            if len(got) == d:
                break
        expected = layernorm_int(x, eps, rms, out_bits=16, shift=8, precision=48)
        np.testing.assert_array_equal(got, expected)
        await tick()
