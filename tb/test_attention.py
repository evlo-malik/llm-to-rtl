# attention.sv against the same integer maths IntGPT uses: random int8 q/k/v for every
# position of two layers, parameters from the compiled tiny GPT plus a few synthetic ones.
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
sys.path.insert(0, str(HERE.parent / "compiler"))
from quant import softmax_int, requant, exp_lut, requant_params   # noqa: E402
from emit import write_memh                                       # noqa: E402

D, H, T, L = 64, 4, 32, 2
HD = D // H
BUILD = HERE.parent / "sim_build" / "attention" / "build"   # own parent dir: cocotb shares ../verilator.o


def params():
    """Layer 0: the real tiny GPT layer-0 numbers. Layer 1: a synthetic set with a
    coarser softmax step so more of the LUT range is exercised."""
    q = json.load(open(HERE.parent / "gen" / "tinygpt" / "model_q.json"))
    p0 = (tuple(q["layers.0.attn.u"]), tuple(q["layers.0.attn.o"]))
    p1 = (requant_params(0.05), requant_params(2.0 ** -15 * 1.2))
    return [p0, p1]


def ref(q, kc, vc, t, prm, lut):
    (m0u, nu), (m0o, no) = prm
    o = np.zeros(D, dtype=np.int64)
    for hh in range(H):
        sl = slice(hh * HD, (hh + 1) * HD)
        s = kc[:t + 1, sl] @ q[sl]
        p = softmax_int(s, m0u, nu, lut)
        acc = p @ vc[:t + 1, sl]
        o[sl] = requant(acc, m0o, no, 8)
    return o


@cocotb.test()
async def matches_python(dut):
    prm = params()
    lut = exp_lut()
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.layer.value = 0
    dut.pos.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    rng = np.random.default_rng(31)
    total_cycles = 0
    for layer in range(L):
        kc = np.zeros((T, D), dtype=np.int64)
        vc = np.zeros((T, D), dtype=np.int64)
        for t in range(T):
            spread = 128 if t % 3 else 20            # small values keep scores close, big ones saturate u
            q = rng.integers(-spread, spread, size=D, dtype=np.int64)
            k = rng.integers(-spread, spread, size=D, dtype=np.int64)
            v = rng.integers(-128, 128, size=D, dtype=np.int64)
            if t == 5:
                q[:] = 127; k[:] = -128                 # extreme dot products
            kc[t] = k
            vc[t] = v
            exp = ref(q, kc, vc, t, prm[layer], lut)
            dut.layer.value = layer
            dut.pos.value = t
            for val in np.concatenate([q, k, v]):
                dut.in_valid.value = 1
                dut.in_data.value = int(val) & 0xFFFFFFFF
                await RisingEdge(dut.clk)
            dut.in_valid.value = 0
            dut.in_data.value = 0x55
            got = []
            cycles = 0
            while len(got) < D:
                await RisingEdge(dut.clk)
                await Timer(1, unit="ns")
                cycles += 1
                if dut.out_valid.value == 1:
                    got.append(dut.out_data.value.to_signed())
                assert cycles < 20000, f"layer {layer} pos {t}: only {len(got)} outputs after {cycles} cycles"
            got = np.array(got)
            assert np.array_equal(got, exp), (
                f"layer {layer} pos {t}: mismatch at {np.flatnonzero(got != exp)[:6]}: hw {got[got != exp][:6]} "
                f"py {exp[got != exp][:6]}")
            # busy must drop
            for _ in range(6):
                await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            assert dut.busy.value == 0
            total_cycles += cycles
        dut._log.info(f"layer {layer}: all {T} positions bit-exact")
    dut._log.info(f"{total_cycles / (L * T):.0f} cycles per token on average (4 heads)")


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    prm = params()
    words = [(no << 38) | (m0o << 22) | (nu << 16) | m0u for (m0u, nu), (m0o, no) in prm]
    write_memh(BUILD / "attn_params.memh", words, 44)
    write_memh(BUILD / "exp_lut.memh", [int(v) for v in exp_lut()], 20)
    rtl = HERE.parent / "rtl"
    runner = get_runner(os.getenv("SIM", "verilator"))
    runner.build(sources=[rtl / "udiv.sv", rtl / "attention.sv"], hdl_toplevel="attention",
                 parameters={"D": D, "H": H, "T": T, "L": L,
                             "PARAM_FILE": f'"{BUILD / "attn_params.memh"}"', "LUT_FILE": f'"{BUILD / "exp_lut.memh"}"'},
                 waves=bool(int(os.getenv("WAVES", "0"))), always=True, timescale=("1ns", "1ps"), build_dir=BUILD)
    runner.test(hdl_toplevel="attention", test_module="test_attention", waves=bool(int(os.getenv("WAVES", "0"))), build_dir=BUILD)


if __name__ == "__main__":
    run()
