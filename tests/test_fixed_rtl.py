from pathlib import Path
import numpy as np
import pytest
from cocotb_tools.runner import get_runner
from compiler.emit import emit_linear

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sim", ["icarus", "verilator"])
@pytest.mark.parametrize(
    "shape,scaled", [((1, 1), False), ((7, 5), False), ((35, 9), True)]
)
def test_fixed_rtl(tmp_path, monkeypatch, sim, shape, scaled):
    rng = np.random.default_rng(17)
    w = rng.integers(-128, 128, shape, dtype=np.int64)
    if shape[0] > 1:
        w[0] = 0
    b = rng.integers(-500, 500, shape[0], dtype=np.int64)
    params = {}
    if scaled:
        params = {"m0": np.full(shape[0], 32768), "shifts": np.full(shape[0], 23)}
    rtl = tmp_path / "fixed.sv"
    emit_linear(rtl, "fixed", w, b, **params, out_bits=8 if scaled else 32)
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, w=w, b=b, **params)
    runner = get_runner(sim)
    build = tmp_path / "build"
    runner.build(
        sources=[rtl],
        hdl_toplevel="fixed",
        build_dir=build,
        timescale=("1ns", "1ps"),
        build_args=["-Wno-fatal"] if sim == "verilator" else [],
    )
    runner.test(
        hdl_toplevel="fixed",
        test_module="test_fixed",
        test_dir=ROOT / "tb",
        build_dir=build,
        extra_env={"FIXTURE": str(fixture)},
    )
