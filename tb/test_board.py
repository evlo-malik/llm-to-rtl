# tang_nano_20k.sv with a UART model on both ends: send a prompt plus Enter, read back
# what the board prints, compare with IntGPT's greedy continuation of the same prompt.
# The UART divider is shrunk (CLK_HZ / BAUD = 10 cycles per bit) to keep the run short.
import json
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent
REPO = HERE.parent
GEN = Path(os.getenv("GEN", "gen/tinygpt"))
GEN = GEN if GEN.is_absolute() else (REPO / GEN).resolve()      # the sim runs from the build dir
sys.path.insert(0, str(REPO / "compiler"))
from compile import load_bundle       # noqa: E402
from golden_gpt import IntGPT         # noqa: E402

CLK_HZ, BAUD = 1_000_000, 100_000
BIT = CLK_HZ // BAUD
PROMPT = os.getenv("PROMPT", "ROMEO:")
BUILD = REPO / "sim_build" / "board" / "build"


async def uart_send(dut, byte):
    bits = [0] + [(byte >> i) & 1 for i in range(8)] + [1]
    for b in bits:
        dut.rx.value = b
        for _ in range(BIT):
            await RisingEdge(dut.clk)


async def uart_recv(dut, timeout_cycles):
    """Wait for a start bit, sample mid-bit, return the byte (or None on timeout)."""
    n = 0
    while dut.tx.value == 1:
        await RisingEdge(dut.clk)
        n += 1
        if n > timeout_cycles:
            return None
    for _ in range(BIT // 2):
        await RisingEdge(dut.clk)
    val = 0
    for i in range(8):
        for _ in range(BIT):
            await RisingEdge(dut.clk)
        val |= int(dut.tx.value) << i
    for _ in range(BIT):
        await RisingEdge(dut.clk)
    return val


@cocotb.test()
async def prompt_in_text_out(dut):
    manifest = json.load(open(GEN / "manifest.json"))
    chars = json.load(open(Path(manifest["model_dir"]) / "tokenizer.json"))["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    q = load_bundle(GEN / "model_q")
    T = q["T"]
    ref = IntGPT(q)
    toks = ref.generate([stoi[c] for c in PROMPT], T)
    expect = "".join(chars[i] for i in toks[len(PROMPT):]) + "\r\n"

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_i.value = 0
    dut.rx.value = 1
    for _ in range(8):
        await RisingEdge(dut.clk)
    dut.rst_i.value = 1
    for _ in range(8):
        await RisingEdge(dut.clk)

    for c in PROMPT + "\n":
        await uart_send(dut, ord(c))
    got = ""
    while not got.endswith("\r\n"):
        b = await uart_recv(dut, 3_000_000)
        assert b is not None, f"board went quiet after {got!r}"
        got += chr(b)
    dut._log.info(f"prompt {PROMPT!r} -> board printed {got!r}")
    assert got == expect, f"board {got!r} vs golden {expect!r}"
    assert int(dut.u_gpt.pos.value) == 0, "position not cleared after the end marker"
    (GEN / "board_sample.txt").write_text(PROMPT + got)


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    rtl = REPO / "rtl"
    srcs = [rtl / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv", "matvec_rom.sv", "requant.sv", "isqrt.sv",
                              "udiv.sv", "layernorm.sv", "attention.sv", "embed.sv", "sequencer.sv",
                              "board/uart_rx.sv", "board/uart_tx.sv", "board/tang_nano_20k.sv")]
    srcs += sorted((GEN / "rtl").glob("mv_*.sv")) + sorted((GEN / "rtl").glob("const_*.sv")) + [GEN / "rtl" / "gpt_top.sv"]
    runner = get_runner(os.getenv("SIM", "verilator"))
    runner.build(sources=srcs, hdl_toplevel="tang_nano_20k", always=True, timescale=("1ns", "1ps"), build_dir=BUILD,
                 parameters={"CLK_HZ": CLK_HZ, "BAUD": BAUD, "STOI_FILE": f'"{GEN / "mem" / "stoi.memh"}"',
                             "ITOS_FILE": f'"{GEN / "mem" / "itos.memh"}"'},
                 build_args=["-Wno-fatal"])
    runner.test(hdl_toplevel="tang_nano_20k", test_module="test_board", build_dir=BUILD)


if __name__ == "__main__":
    run()
