# ---- imports: things this file uses ----

import os                          # to read the SIM environment variable (which simulator to use)
import random                      # to make random test numbers
from pathlib import Path           # to build file paths that work on any OS
import cocotb                      # the library that lets Python drive a Verilog simulation
from cocotb.clock import Clock     # a helper that toggles the clock pin for us
from cocotb.triggers import RisingEdge, Timer   # things we can "await" to let simulated time pass
from cocotb_tools.runner import get_runner      # compiles the Verilog and launches the test


# "@cocotb.test()" marks this function as a test. cocotb will run it inside the
# simulator. "async" because the function pauses (awaits) while simulated time moves.
# "dut" = Device Under Test = your mul circuit. dut.a is the pin called a.
@cocotb.test()
async def directed(dut):

    # Start the clock: flip dut.clk every 5 ns, so one full tick every 10 ns,
    # forever, in the background. Without this the circuit never does anything.
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Press reset (rst_n = 0) and put 0 on the inputs so nothing is undefined.
    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0

    # Let two clock ticks happen while reset is pressed. "await RisingEdge(dut.clk)"
    # means: pause this Python function until the clock next goes 0->1.
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Release reset. From now on the circuit multiplies.
    dut.rst_n.value = 1

    # Hand-picked pairs. Each one is there to catch a specific bug:
    # (3,4) basic. (-3,4) one negative. (-1,-1) two negatives, must give +1.
    # (127,127) biggest positive. (-128,-128) biggest product. (0,99) zero.
    cases = [(3, 4), (-3, 4), (-1, -1), (127, 127), (-128, -128), (0, 99)]

    for a, b in cases:
        # Put the numbers on the input pins. Negative ints are fine; cocotb
        # converts -3 into the 8-bit pattern 11111101 for you.
        dut.a.value = a
        dut.b.value = b

        # Wait for one clock tick. At this edge the always_ff block runs
        # and p becomes a*b.
        await RisingEdge(dut.clk)

        # Wait 1 ns more. At the exact instant of the edge, p is still
        # changing. 1 ns later it has settled. Without this you read the OLD p.
        await Timer(1, unit="ns")

        # Read the output pin as a signed integer.
        got = dut.p.value.to_signed()

        # Python computes the same product on its own. If the two disagree,
        # the test fails and prints the message. This is verification:
        # two independent calculators must agree.
        assert got == a * b, f"{a} * {b}: hardware said {got}, expected {a * b}"

        # Print a line so you can see it happening. :5d = pad to 5 characters.
        dut._log.info(f"{a:5d} * {b:5d} = {got:7d}  OK")


# Second test: same thing but 200 random pairs instead of 6 hand-picked ones.
@cocotb.test()
async def randomised(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())   # clock again (each test starts fresh)
    dut.rst_n.value = 0                                          # press reset
    await RisingEdge(dut.clk)                                    # one tick
    dut.rst_n.value = 1                                          # release

    for _ in range(200):                                         # 200 times:
        a = random.randint(-128, 127)                            #   random signed 8-bit number
        b = random.randint(-128, 127)                            #   another
        dut.a.value = a                                          #   put them on the pins
        dut.b.value = b
        await RisingEdge(dut.clk)                                #   one tick: multiply happens
        await Timer(1, unit="ns")                                #   settle
        got = dut.p.value.to_signed()                            #   read result
        assert got == a * b, f"{a} * {b}: hardware said {got}, expected {a * b}"   # compare


# ---- boilerplate that compiles and runs. Copy this block into every test file. ----
def run():
    here = Path(__file__).parent                       # the tb/ folder this file is in
    runner = get_runner(os.getenv("SIM", "verilator")) # which simulator: SIM=icarus or default verilator
    runner.build(
        sources=[here.parent / "rtl" / "mul.sv"],      # the Verilog file(s) to compile
        hdl_toplevel="mul",                            # the module name to simulate
        waves=True,                                    # record every signal to sim_build/dump.vcd for Surfer
        always=True,                                   # recompile every run (so edits are picked up)
        timescale=("1ns", "1ps"),                      # time units; Icarus refuses to run without this
    )
    runner.test(
        hdl_toplevel="mul",                            # same module name
        test_module="test_mul",                        # this Python file's name, without .py
        waves=True,
    )


# Python runs this only when you execute the file directly
# (python3 tb/test_mul.py), not when something imports it.
if __name__ == "__main__":
    run()