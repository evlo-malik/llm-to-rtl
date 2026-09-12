# layernorm.sv against quant.layernorm_int on int16 vectors: random, constant (zero
# variance), tiny variance, and full-range alternating values.
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "compiler"))
from quant import layernorm_int              # noqa: E402

D = int(os.getenv("D", "64"))
EPS_VAR = int(os.getenv("EPS_VAR", "0"))
BUILD = HERE.parent / "sim_build" / f"layernorm_D{D}"


def vectors(rng):
    yield np.zeros(D, dtype=np.int64)
    yield np.full(D, 1234, dtype=np.int64)
    v = np.zeros(D, dtype=np.int64); v[0] = 1; yield v
    v = np.zeros(D, dtype=np.int64); v[min(3, D-1)] = -1; yield v
    yield np.where(np.arange(D) % 2 == 0, 32767, -32768).astype(np.int64)
    yield np.full(D, -32768, dtype=np.int64)
    yield rng.integers(-5, 6, size=D, dtype=np.int64)
    for _ in range(40):
        scale = int(rng.choice([8, 100, 1000, 30000]))
        yield np.clip(rng.normal(rng.integers(-2000, 2000), scale, size=D).round(), -32768, 32767).astype(np.int64)
    for _ in range(20):
        yield rng.integers(-32768, 32768, size=D, dtype=np.int64)


@cocotb.test()
async def matches_python(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    rng = np.random.default_rng(21)
    n_done = 0
    for h in vectors(rng):
        exp = layernorm_int(h, EPS_VAR)
        for i, v in enumerate(h):
            dut.in_valid.value = 1
            dut.in_data.value = int(v) & 0xFFFFFFFF
            await RisingEdge(dut.clk)
            if i % 7 == 3:                     # a bubble now and then
                dut.in_valid.value = 0
                await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        dut.in_data.value = 0x7FFF7FFF
        got = []
        cycles = 0
        while len(got) < D:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            cycles += 1
            if dut.out_valid.value == 1:
                got.append(dut.out_data.value.to_signed())
            assert cycles < 3 * D + 200, f"only {len(got)} outputs after {cycles} cycles"
        while dut.busy.value == 1:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            cycles += 1
        got = np.array(got)
        assert np.array_equal(got, exp), (
            f"vector {n_done}: mismatch at {np.flatnonzero(got != exp)[:5]}: hw {got[got != exp][:5]} py {exp[got != exp][:5]}; "
            f"h={h[:8]}...")
        n_done += 1
    dut._log.info(f"{n_done} LayerNorms bit-exact, {cycles} cycles from last input to last output")


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    rtl = HERE.parent / "rtl"
    runner = get_runner(os.getenv("SIM", "verilator"))
    runner.build(sources=[rtl / "isqrt.sv", rtl / "udiv.sv", rtl / "layernorm.sv"], hdl_toplevel="layernorm",
                 parameters={"D": D, "EPS_VAR": EPS_VAR}, waves=True, always=True, timescale=("1ns", "1ps"), build_dir=BUILD)
    runner.test(hdl_toplevel="layernorm", test_module="test_layernorm", waves=True, build_dir=BUILD)


if __name__ == "__main__":
    run()
