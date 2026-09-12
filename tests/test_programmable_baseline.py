"""The area-study baseline must implement the same matrix operation."""

from pathlib import Path
import numpy as np
import pytest
from cocotb_tools.runner import get_runner
from syn.compare import programmable

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_programmable(tmp_path, bits):
    w = np.random.default_rng(55).integers(
        -1 if bits == 2 else -(1 << (bits - 1)),
        2 if bits == 2 else 1 << (bits - 1),
        (5, 7),
        dtype=np.int64,
    )
    rtl = tmp_path / "linear.sv"
    programmable(rtl, 5, 7, bits)
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, w=w, b=np.zeros(5, dtype=np.int64))
    runner = get_runner("icarus")
    build = tmp_path / "build"
    runner.build(
        sources=[rtl], hdl_toplevel="linear", build_dir=build, timescale=("1ns", "1ps")
    )
    runner.test(
        hdl_toplevel="linear",
        test_module="test_fixed",
        build_dir=build,
        test_dir=ROOT / "tb",
        extra_env={"FIXTURE": str(fixture)},
    )
