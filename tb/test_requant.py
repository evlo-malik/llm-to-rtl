# requant.sv against quant.requant: random int32 accumulators through random
# (bias, M0, n) parameter sets from a memh, all three output widths, with and without ReLU.
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
from quant import requant                    # noqa: E402
from emit import write_memh                  # noqa: E402

NPARAM = 256
BW = 18
BUILD = HERE.parent / "sim_build" / "requant"


def make_params():
    rng = np.random.default_rng(11)
    m0 = rng.integers(1 << 15, 1 << 16, size=NPARAM, dtype=np.int64)
    n = rng.integers(0, 41, size=NPARAM, dtype=np.int64)
    bias = rng.integers(-(1 << 17), 1 << 17, size=NPARAM, dtype=np.int64)
    n[0] = 0; n[1] = 1; n[2] = 40; n[3] = 15; m0[3] = 1 << 15; m0[4] = (1 << 16) - 1
    bias[0] = -(1 << 17); bias[1] = (1 << 17) - 1; bias[3] = 0
    # entries 8..15 are identity requants (M0 = 2^15, n = 15) so the int32 mode is a bias add
    m0[8:16] = 1 << 15; n[8:16] = 15
    return m0, n, bias


@cocotb.test()
async def matches_python(dut):
    m0, n, bias = make_params()
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_first.value = 0
    dut.in_data.value = 0
    dut.base.value = 0
    dut.relu.value = 0
    dut.width.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    rng = np.random.default_rng(12)
    checked = 0
    for trial in range(80):
        length = int(rng.integers(1, 40))
        base = int(rng.integers(0, NPARAM - length))
        width = int(rng.integers(0, 3))
        relu = int(rng.integers(0, 2))
        if trial % 4 == 3:                            # exercise the identity entries in int32 mode
            base, width, length, relu = 8, 2, 8, 0
        acc = rng.integers(-(1 << 31), 1 << 31, size=length, dtype=np.int64)
        if trial < 4:
            acc[0] = -(1 << 31); acc[-1] = (1 << 31) - 1
        bits = {0: 8, 1: 16, 2: 32}[width]
        exp = requant(acc + bias[base:base + length], m0[base:base + length], n[base:base + length], bits, relu=bool(relu))
        if width == 2 and trial % 4 == 3:
            plain = np.clip(acc + bias[base:base + length], -(1 << 31), (1 << 31) - 1)
            assert np.array_equal(exp, plain), "identity entries must be a plain (int32-clipped) bias add"
        dut.base.value = base
        dut.width.value = width
        dut.relu.value = relu
        got = []
        i = 0
        bubbles = list(rng.random(length) < 0.2)
        cycles = 0
        while len(got) < length:
            if i < length and not (bubbles[i] and cycles % 2):
                dut.in_valid.value = 1
                dut.in_first.value = int(i == 0)
                dut.in_data.value = int(acc[i]) & 0xFFFFFFFF
                i += 1
            else:
                dut.in_valid.value = 0
                dut.in_first.value = 0
                dut.in_data.value = 0xDEADBEEF
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            cycles += 1
            if dut.out_valid.value == 1:
                got.append(dut.out_data.value.to_signed())
            assert cycles < 4 * length + 20
        dut.in_valid.value = 0
        got = np.array(got)
        assert np.array_equal(got, exp), (
            f"trial {trial} base={base} width={width} relu={relu}: first mismatch at {int(np.argmax(got != exp))}: "
            f"hw {got[got != exp][:3]} py {exp[got != exp][:3]} acc {acc[got != exp][:3]}")
        checked += length
    dut._log.info(f"{checked} requantisations bit-exact across 80 vectors, three widths")


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    m0, n, bias = make_params()
    write_memh(BUILD / "params.memh",
               [((int(b) & ((1 << BW) - 1)) << 22) | (int(nn) << 16) | int(mm) for mm, nn, b in zip(m0, n, bias)], 22 + BW)
    runner = get_runner(os.getenv("SIM", "verilator"))
    runner.build(sources=[HERE.parent / "rtl" / "requant.sv"], hdl_toplevel="requant",
                 parameters={"NPARAM": NPARAM, "BW": BW, "PARAM_FILE": f'"{BUILD / "params.memh"}"'},
                 waves=True, always=True, timescale=("1ns", "1ps"), build_dir=BUILD)
    runner.test(hdl_toplevel="requant", test_module="test_requant", waves=True, build_dir=BUILD)


if __name__ == "__main__":
    run()
