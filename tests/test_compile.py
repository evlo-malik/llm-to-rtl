import json
import pytest
from safetensors.numpy import load_file, save_file
from compiler.compile import compile_checkpoint
from tests.test_model_rtl import fixture_checkpoint


def test_deterministic_output_and_changed_weights(tmp_path):
    model = tmp_path / "model"
    fixture_checkpoint(model)
    first = compile_checkpoint(model, tmp_path / "first")
    second = compile_checkpoint(model, tmp_path / "second")
    assert first["rtl_sha256"] == second["rtl_sha256"]
    state = load_file(str(model / "model.safetensors"))
    state["transformer.h.0.attn.c_proj.weight"] *= -1
    save_file(state, str(model / "model.safetensors"))
    changed = compile_checkpoint(model, tmp_path / "changed")
    assert (
        changed["rtl_sha256"]["layer0_proj.sv"] != first["rtl_sha256"]["layer0_proj.sv"]
    )


def test_output_guard_and_emission_limit(tmp_path):
    model = tmp_path / "model"
    fixture_checkpoint(model)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(ValueError, match="already exists"):
        compile_checkpoint(model, existing)
    assert sentinel.read_text() == "existing output"
    with pytest.raises(ValueError, match="emission limit"):
        compile_checkpoint(model, tmp_path / "too_big", max_coefficients=1)
    assert not (tmp_path / "too_big").exists()
    assert not list(tmp_path.glob(".llm-to-rtl-*"))


def test_invalid_calibration_and_unsupported_activation(tmp_path):
    model = tmp_path / "model"
    fixture_checkpoint(model)
    (model / "calibration.json").write_text("[[true]]")
    with pytest.raises(ValueError, match="token"):
        compile_checkpoint(model, tmp_path / "bad")
    cfg = json.loads((model / "config.json").read_text())
    cfg["activation_function"] = "relu"
    (model / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="gelu_new"):
        compile_checkpoint(model, tmp_path / "bad")
