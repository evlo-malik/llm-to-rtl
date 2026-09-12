# top.sv: the array with skew, deskew and valid. Random INT8 matrices at N = 4, 8, 16,
# checked bit-exact against numpy, on whichever simulator SIM names.
#
#   N=8 python3 tb/test_top.py
#   N=16 SIM=icarus python3 tb/test_top.py
import os
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from golden import rand_i8, matmul, pack, unpack, latency

N = int(os.getenv("N", "4"))
LAT = latency(N)


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.w_load.value = 0
    dut.w_row.value = 0
    dut.x_valid.value = 0
    dut.x_vec.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1


async def step(dut):
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def load(dut, w):
    assert dut.busy.value == 0, "loading weights over a wave in flight"
    for r in range(N - 1, -1, -1):
        dut.w_row.value = pack(w[r], 8)
        dut.w_load.value = 1
        await step(dut)
    dut.w_load.value = 0
    dut.w_row.value = 0


async def stream(dut, a, w, gaps=None):
    """Feed the rows of a, one per cycle (or with idle gaps), collect every y_valid,
    compare with a @ w. Also checks y_valid lands exactly LAT cycles after x_valid."""
    m = a.shape[0]
    y = matmul(a, w)
    sent = []          # cycle each row went in
    got = []
    tau = 0
    k = 0
    while k < m or (sent and tau - sent[-1] <= LAT):
        fire = k < m and (gaps is None or not gaps[k])
        if fire:
            dut.x_vec.value = pack(a[k], 8)
            dut.x_valid.value = 1
            sent.append(tau)
            k += 1
        else:
            dut.x_vec.value = pack(rand_i8(np.random.default_rng(tau), N), 8)   # junk must be ignored
            dut.x_valid.value = 0
            if gaps is not None and k < m:
                gaps[k] = False
        await step(dut)
        tau += 1
        if dut.y_valid.value == 1:
            got.append((tau, unpack(int(dut.y_vec.value), N, 32)))
    dut.x_valid.value = 0
    assert len(got) == m, f"N={N}: sent {m} vectors, got {len(got)} y_valid"
    for i, (t, row) in enumerate(got):
        assert t - sent[i] == LAT, f"N={N}: vector {i} latency {t - sent[i]}, expected {LAT}"
        assert np.array_equal(row, y[i]), f"N={N} vector {i}: top {row}, numpy {y[i]}"


@cocotb.test()
async def directed(dut):
    await reset(dut)
    rng = np.random.default_rng(3)
    a = rand_i8(rng, (6, N))
    await load(dut, np.eye(N, dtype=np.int64))
    await stream(dut, a, np.eye(N, dtype=np.int64))
    w = np.full((N, N), -128, dtype=np.int64)
    await load(dut, w)
    await stream(dut, np.full((3, N), -128, dtype=np.int64), w)
    await stream(dut, np.full((3, N), 127, dtype=np.int64), w)


@cocotb.test()
async def streamed(dut):
    await reset(dut)
    rng = np.random.default_rng(4)
    for trial in range(20):
        w = rand_i8(rng, (N, N))
        a = rand_i8(rng, (64, N))
        await load(dut, w)
        await stream(dut, a, w)


@cocotb.test()
async def with_gaps(dut):
    await reset(dut)
    rng = np.random.default_rng(5)
    for trial in range(10):
        w = rand_i8(rng, (N, N))
        a = rand_i8(rng, (32, N))
        gaps = list(rng.random(32) < 0.4)
        await load(dut, w)
        await stream(dut, a, w, gaps)


@cocotb.test()
async def reload_between_vectors(dut):
    await reset(dut)
    rng = np.random.default_rng(6)
    for trial in range(40):
        w = rand_i8(rng, (N, N))
        a = rand_i8(rng, (int(rng.integers(1, 6)), N))
        await load(dut, w)
        await stream(dut, a, w)
    dut._log.info(f"N={N}: latency {LAT} held on every vector; 70 weight loads bit-exact")


def run():
    here = Path(__file__).parent
    rtl = here.parent / "rtl"
    runner = get_runner(os.getenv("SIM", "verilator"))
    build = here.parent / "sim_build" / f"top_N{N}"
    runner.build(sources=[rtl / "pe.sv", rtl / "array.sv", rtl / "skew.sv", rtl / "top.sv"],
                 hdl_toplevel="top", parameters={"N": N}, waves=True, always=True,
                 timescale=("1ns", "1ps"), build_dir=build)
    runner.test(hdl_toplevel="top", test_module="test_top", waves=True, build_dir=build)


if __name__ == "__main__":
    run()
