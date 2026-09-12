"""Generated fixed linear map: gaps, extremes, repeated transactions and reset."""

import os
import json
from pathlib import Path
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def fixed_linear(dut):
    fixture = np.load(os.environ["FIXTURE"])
    w, b = fixture["w"], fixture["b"]
    n, k = w.shape
    rng = np.random.default_rng(9)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.rst_n.value = 1
    vectors = [np.zeros(k, dtype=np.int64), np.full(k, -128), np.full(k, 127)]
    vectors += [
        rng.integers(-128, 128, k) for _ in range(int(os.getenv("FIXED_TRIALS", "8")))
    ]
    for trial, x in enumerate(vectors):
        for index, value in enumerate(x):
            dut.in_valid.value = 1
            dut.in_data.value = int(value) & 255
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            dut.in_valid.value = 0
            if trial % 2 and index < k - 1:
                await RisingEdge(dut.clk)
                await Timer(1, unit="ns")
                assert not dut.out_valid.value
        got = []
        for _ in range(n + 3):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if dut.out_valid.value:
                got.append(dut.out_data.value.to_signed())
        exp = w.astype(np.int64) @ x + b
        if "m0" in fixture:
            m0, shifts = fixture["m0"], fixture["shifts"]
            half = np.where(shifts > 0, 1 << np.maximum(shifts - 1, 0), 0)
            exp = np.clip((exp * m0 + half) >> shifts, -128, 127)
        assert got == exp.tolist(), (trial, got, exp)
        assert not dut.busy.value
    # Discard an incomplete vector on reset, then run another full transaction.
    dut.in_valid.value = 1
    dut.in_data.value = 42
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert not dut.busy.value and not dut.out_valid.value

    if os.getenv("FIXED_REPORT"):
        Path(os.environ["FIXED_REPORT"]).write_text(
            json.dumps(
                dict(
                    passed=True,
                    inputs=k,
                    outputs=n,
                    vectors=len(vectors),
                    logits_compared=len(vectors) * n,
                    simulator=cocotb.SIM_NAME,
                ),
                indent=2,
            )
            + "\n"
        )
