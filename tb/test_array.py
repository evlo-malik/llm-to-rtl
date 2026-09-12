# array.sv on its own, with the testbench doing the skew that top.sv does in hardware.
# Row i of vector m goes in at cycle m+i; column j of vector m comes out at cycle m+N+j.
# Passing here means the grid wiring and the weight shift-in are right; the timing
# formula is checked because the testbench reads each column on exactly that cycle.
import os
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from golden import rand_i8, matmul

N = int(os.getenv("N", "4"))


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.load_w.value = 0
    for i in range(N):
        dut.x_in[i].value = 0
        dut.w_in[i].value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1


async def step(dut):
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def load(dut, w):
    # last row first: each load cycle pushes the column down one PE
    for r in range(N - 1, -1, -1):
        for j in range(N):
            dut.w_in[j].value = int(w[r, j])
        dut.load_w.value = 1
        await step(dut)
    dut.load_w.value = 0
    for j in range(N):
        dut.w_in[j].value = 0


async def stream(dut, a, w):
    """Push every row of a through the array back to back and check every output on
    the cycle it is due."""
    m = a.shape[0]
    y = matmul(a, w)
    for tau in range(m + 2 * N):
        for i in range(N):
            k = tau - i
            dut.x_in[i].value = int(a[k, i]) if 0 <= k < m else 0
        await step(dut)
        # registers now hold cycle tau+1
        for j in range(N):
            k = (tau + 1) - N - j
            if 0 <= k < m:
                got = dut.result[j].value.to_signed()
                assert got == y[k, j], (
                    f"N={N} vector {k} column {j}: array {got}, numpy {y[k, j]}")


@cocotb.test()
async def hand_picked(dut):
    await reset(dut)
    eye = np.eye(N, dtype=np.int64)
    a = rand_i8(np.random.default_rng(1), (8, N))
    await load(dut, eye)
    await stream(dut, a, eye)                      # identity: output copies input
    ones = np.ones((N, N), dtype=np.int64)
    await load(dut, ones)
    await stream(dut, a, ones)                     # every column is the row sum
    ext = np.full((N, N), -128, dtype=np.int64)
    await load(dut, ext)
    await stream(dut, np.full((4, N), -128, dtype=np.int64), ext)   # N * 16384, the sign trap
    await stream(dut, np.full((4, N), 127, dtype=np.int64), ext)


@cocotb.test()
async def random_matrices(dut):
    await reset(dut)
    rng = np.random.default_rng(2)
    for trial in range(30):
        w = rand_i8(rng, (N, N))
        a = rand_i8(rng, (int(rng.integers(1, 40)), N))
        await load(dut, w)
        await stream(dut, a, w)
    dut._log.info(f"N={N}: 30 random weight loads, every output matched numpy on its cycle")


def run():
    here = Path(__file__).parent
    rtl = here.parent / "rtl"
    runner = get_runner(os.getenv("SIM", "verilator"))
    build = here.parent / "sim_build" / f"array_N{N}"
    runner.build(sources=[rtl / "pe.sv", rtl / "array.sv"], hdl_toplevel="array",
                 parameters={"N": N}, waves=True, always=True,
                 timescale=("1ns", "1ps"), build_dir=build)
    runner.test(hdl_toplevel="array", test_module="test_array", waves=True, build_dir=build)


if __name__ == "__main__":
    run()
