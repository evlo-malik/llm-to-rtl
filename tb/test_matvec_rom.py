# matvec_rom.sv with a random weight matrix written to a memh the way compile.py
# writes them. Shape and array size come from the environment:
#   K=13 N=7 T=4 WBITS=4 BIAS=1 python3 tb/test_matvec_rom.py
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "compiler"))
from emit import write_rom, write_bias      # noqa: E402
from mvtest import check_module             # noqa: E402

K = int(os.getenv("K", "8"))
N = int(os.getenv("N", "8"))
T = int(os.getenv("T", "4"))
WBITS = int(os.getenv("WBITS", "8"))
BIAS = int(os.getenv("BIAS", "1"))
TAG = f"K{K}_N{N}_T{T}_W{WBITS}_B{BIAS}"
BUILD = HERE.parent / "sim_build" / f"matvec_{TAG}"


def make_weights():
    rng = np.random.default_rng(K * 7 + N * 13 + T + WBITS)
    lo, hi = -(1 << (WBITS - 1)), (1 << (WBITS - 1)) - 1
    if WBITS == 2:
        lo, hi = -1, 1
    w = rng.integers(lo, hi + 1, size=(K, N), dtype=np.int64)
    w[0, 0] = lo                      # corners get the extremes
    w[-1, -1] = hi
    b = rng.integers(-(1 << 20), 1 << 20, size=N, dtype=np.int64) if BIAS else None
    return w, b


@cocotb.test()
async def matches_torch(dut):
    w = np.load(BUILD / "w.npy")
    b = np.load(BUILD / "b.npy") if BIAS else None
    cyc = await check_module(dut, w, b, trials=4, log=dut._log.info)
    kt, nt = (K + T - 1) // T, (N + T - 1) // T
    dut._log.info(f"{kt*nt} tiles of {T}x{T}; {cyc} cycles/vector, "
                  f"{K*N/cyc:.2f} MACs/cycle from {T*T} multipliers")


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    w, b = make_weights()
    np.save(BUILD / "w.npy", w)
    write_rom(BUILD / "w.memh", w, T, WBITS)
    if BIAS:
        np.save(BUILD / "b.npy", b)
        write_bias(BUILD / "b.memh", b, N, T)
    rtl = HERE.parent / "rtl"
    runner = get_runner(os.getenv("SIM", "verilator"))
    params = {"K": K, "N": N, "T": T, "WBITS": WBITS, "HAS_BIAS": BIAS,
              "ROM_FILE": f'"{BUILD / "w.memh"}"', "BIAS_FILE": f'"{BUILD / "b.memh"}"'}
    runner.build(sources=[rtl / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv", "matvec_rom.sv")],
                 hdl_toplevel="matvec_rom", parameters=params, waves=True, always=True,
                 timescale=("1ns", "1ps"), build_dir=BUILD)
    runner.test(hdl_toplevel="matvec_rom", test_module="test_matvec_rom", waves=True, build_dir=BUILD)


if __name__ == "__main__":
    run()
