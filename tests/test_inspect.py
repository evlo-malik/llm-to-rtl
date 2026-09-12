import json
import numpy as np
import pytest
from safetensors.numpy import save_file
from compiler.planner import inspect_checkpoint
from compiler.compile import compile_checkpoint
from tests.test_model_rtl import fixture_checkpoint


def test_header_plan_matches_emission(tmp_path):
    fixture_checkpoint(tmp_path / "model")
    plan = inspect_checkpoint(tmp_path / "model", 8)
    manifest = compile_checkpoint(tmp_path / "model", tmp_path / "rtl", context=8)
    assert plan["fixed_coefficients"] == manifest["emitted_coefficients"]


def test_size_guard_runs_before_payload_loader(tmp_path, monkeypatch):
    fixture_checkpoint(tmp_path / "model")

    def forbidden(*args):
        raise AssertionError("payload was loaded")

    monkeypatch.setattr("compiler.compile.load_checkpoint", forbidden)
    with pytest.raises(ValueError, match="emission limit"):
        compile_checkpoint(
            tmp_path / "model", tmp_path / "rtl", context=8, max_coefficients=1
        )


def test_unknown_architecture_can_emit_explicit_matrix(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "new_architecture"}))
    save_file(
        {"block.weight": np.arange(12, dtype=np.float32).reshape(3, 4)},
        str(model / "model.safetensors"),
    )
    assert not inspect_checkpoint(model)["decoder_adapter"]
    with pytest.raises(ValueError, match="no decoder adapter"):
        compile_checkpoint(model, tmp_path / "full")
    m = compile_checkpoint(
        model, tmp_path / "matrix", matrices_only=True, only="block.weight"
    )
    assert m["scope"] == "matrices_only" and m["emitted_coefficients"] == 12


def test_compile_command(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    fixture_checkpoint(tmp_path / "model")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "compiler/compile.py"),
            str(tmp_path / "model"),
            "--out",
            str(tmp_path / "out"),
        ],
        check=True,
        capture_output=True,
    )
