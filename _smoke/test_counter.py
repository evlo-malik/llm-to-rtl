import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner


@cocotb.test()
async def test_counts(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.en.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.count.value) == 0, "reset did not clear the counter"

    dut.rst_n.value = 1
    dut.en.value = 1
    for i in range(1, 300):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        got = int(dut.count.value)
        assert got == i % 256, f"tick {i}: expected {i % 256}, got {got}"
    dut._log.info("counter smoke test passed (incl. 8-bit wraparound)")


def test_runner():
    sim = os.getenv("SIM", "verilator")
    here = Path(__file__).parent
    runner = get_runner(sim)
    runner.build(sources=[here / "counter.sv"], hdl_toplevel="counter",
                 waves=True, always=True, timescale=("1ns", "1ps"))
    runner.test(hdl_toplevel="counter", test_module="test_counter", waves=True)


if __name__ == "__main__":
    test_runner()
