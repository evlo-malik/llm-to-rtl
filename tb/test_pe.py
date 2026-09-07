import os
import random
from pathlib import Path
import torch
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner


# ---------- helpers ----------

async def reset(dut):
    """Start the clock, hold reset two ticks, release, zero every input."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.load_w.value = 0
    dut.w_in.value = 0
    dut.x_in.value = 0
    dut.psum_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1


async def load_weight(dut, w):
    """Put w on w_in, raise load_w for exactly one tick, drop it."""
    dut.w_in.value = w
    dut.load_w.value = 1
    await RisingEdge(dut.clk)          # the box captures w on this edge
    dut.load_w.value = 0
    dut.w_in.value = 0                 # trash w_in on purpose: if the weight
                                       # follows it, it was not stationary


async def mac(dut, x, psum_in):
    """One MAC: drive x_in and psum_in, wait a tick, return (psum_out, x_out)."""
    dut.x_in.value = x
    dut.psum_in.value = psum_in
    await RisingEdge(dut.clk)          # the box computes on this edge
    await Timer(1, unit="ns")          # let the output registers settle
    return dut.psum_out.value.to_signed(), dut.x_out.value.to_signed()


def ref_mac(x, w, psum_in):
    """PyTorch does the same maths independently, in int32."""
    a = torch.tensor(x, dtype=torch.int32)
    b = torch.tensor(w, dtype=torch.int32)
    p = torch.tensor(psum_in, dtype=torch.int32)
    return int(p + a * b)


# ---------- test 1: hand-picked cases, each aimed at one bug ----------

@cocotb.test()
async def directed(dut):
    await reset(dut)

    # (weight, x, incoming partial sum)
    cases = [
        (0,     55,   0),          # weight 0: output must equal psum_in
        (3,     4,    0),          # basic
        (3,     4,    100),        # psum_in is really added
        (-3,    4,    0),          # one negative
        (-1,    -1,   0),          # two negatives -> +1
        (127,   127,  0),          # biggest positive product
        (-128,  -128, 0),          # biggest product, the signedness trap
        (-128,  127,  -2000000),   # big negative psum_in, negative product
        (5,     0,    12345),      # x = 0: output equals psum_in
    ]

    for w, x, psum in cases:
        await load_weight(dut, w)
        got_psum, got_x = await mac(dut, x, psum)
        exp = ref_mac(x, w, psum)
        assert got_psum == exp, f"w={w} x={x} psum_in={psum}: hardware {got_psum}, torch {exp}"
        assert got_x == x,      f"x_out should copy x_in, got {got_x} for {x}"
        dut._log.info(f"w={w:5d} x={x:5d} psum_in={psum:9d} -> {got_psum:9d}  OK")


# ---------- test 2: the weight must STAY ----------

@cocotb.test()
async def weight_is_stationary(dut):
    await reset(dut)
    await load_weight(dut, 7)                        # load once

    for x in [1, 2, 3, -4, 100, -128]:               # stream six x values past it
        dut.w_in.value = random.randint(-128, 127)   # junk on w_in with load_w = 0
        got_psum, _ = await mac(dut, x, 0)           # must be ignored
        assert got_psum == x * 7, f"weight drifted: x={x} gave {got_psum}, expected {x * 7}"
    dut._log.info("weight held at 7 through six MACs and six junk w_in values  OK")


# ---------- test 3: chain MACs like a column of the grid ----------

@cocotb.test()
async def one_pe_many_macs(dut):
    """Feed psum_out back into psum_in so one PE acts like a whole column
    with the same weight. Result = dot(xs, [w, w, w, ...])."""
    await reset(dut)
    for trial in range(20):
        w  = random.randint(-128, 127)
        xs = [random.randint(-128, 127) for _ in range(random.randint(1, 64))]
        await load_weight(dut, w)

        psum = 0
        for x in xs:
            psum, _ = await mac(dut, x, psum)

        a = torch.tensor(xs, dtype=torch.int32)
        exp = int(torch.dot(a, torch.full_like(a, w)))
        assert psum == exp, f"trial {trial}: w={w} xs={xs}: hardware {psum}, torch {exp}"
    dut._log.info("20 chained sequences matched torch.dot  OK")


# ---------- test 4: 1000 random single MACs ----------

@cocotb.test()
async def randomised(dut):
    await reset(dut)
    for _ in range(1000):
        w    = random.randint(-128, 127)
        x    = random.randint(-128, 127)
        psum = random.randint(-2**30, 2**30)
        await load_weight(dut, w)
        got, _ = await mac(dut, x, psum)
        exp = ref_mac(x, w, psum)
        assert got == exp, f"w={w} x={x} psum_in={psum}: hardware {got}, torch {exp}"


# ---------- boilerplate ----------

def run():
    here = Path(__file__).parent
    runner = get_runner(os.getenv("SIM", "verilator"))
    runner.build(sources=[here.parent / "rtl" / "pe.sv"], hdl_toplevel="pe",
                 waves=True, always=True, timescale=("1ns", "1ps"))
    runner.test(hdl_toplevel="pe", test_module="test_pe", waves=True)


if __name__ == "__main__":
    run()