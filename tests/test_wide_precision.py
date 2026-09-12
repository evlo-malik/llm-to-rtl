"""High-range residuals must match the same integer definition in both simulators."""

import subprocess
import sys
from pathlib import Path

import pytest

from compiler.compile import compile_checkpoint
from tests.test_rotary_models import fixture
from tests.test_model_rtl import fixture_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "family,sim", [("llama", "icarus"), ("qwen2", "verilator"), ("gpt2", "icarus")]
)
def test_wide_decoder(tmp_path, family, sim):
    model = tmp_path / "model"
    if family == "gpt2":
        fixture_checkpoint(model)
    else:
        fixture(model, family)
    compile_checkpoint(model, tmp_path / "rtl", context=8, activation_bits=16)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify.py"),
            str(tmp_path / "rtl"),
            "--tokens",
            "1,2,3,4,5,6,7,8",
            "--sim",
            sim,
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-6500:] + run.stderr[-2500:]
    if family == "llama":
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_matrix.py"),
                str(tmp_path / "rtl"),
                "--module",
                "layer0_ffwd2",
                "--sim",
                "icarus",
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


@pytest.mark.parametrize("d,rms", [(7, 0), (7, 1), (64, 0), (64, 1)])
def test_normaliser_extremes(tmp_path, d, rms):
    from cocotb_tools.runner import get_runner

    runner = get_runner("icarus")
    runner.build(
        sources=[
            ROOT / "rtl/normalise_wide.sv",
            ROOT / "rtl/isqrt_wide.sv",
            ROOT / "rtl/udiv.sv",
        ],
        hdl_toplevel="normalise_wide",
        build_dir=tmp_path / "build",
        timescale=("1ns", "1ps"),
        parameters={"D": d, "RMS": rms, "EPS_VAR": 12345678901},
    )
    runner.test(
        hdl_toplevel="normalise_wide",
        test_module="test_wide_norm",
        test_dir=ROOT / "tb",
        build_dir=tmp_path / "build",
        extra_env={"SOURCE_ROOT": str(ROOT), "NORM_D": str(d), "NORM_RMS": str(rms)},
    )


@pytest.mark.parametrize("sim", ["icarus", "verilator"])
def test_wide_fixed_arithmetic(tmp_path, sim):
    import numpy as np
    from compiler.emit import emit_linear
    from cocotb_tools.runner import get_runner

    w = np.random.default_rng(31).integers(-128, 128, (7, 5), dtype=np.int64)
    rtl = tmp_path / "fixed.sv"
    emit_linear(rtl, "fixed", w, in_bits=16)
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, w=w, b=np.zeros(7, dtype=np.int64))
    runner = get_runner(sim)
    runner.build(
        sources=[rtl],
        hdl_toplevel="fixed",
        build_dir=tmp_path / "build",
        timescale=("1ns", "1ps"),
        build_args=["-Wno-fatal"] if sim == "verilator" else [],
    )
    runner.test(
        hdl_toplevel="fixed",
        test_module="test_fixed",
        test_dir=ROOT / "tb",
        build_dir=tmp_path / "build",
        extra_env={"FIXTURE": str(fixture)},
    )
