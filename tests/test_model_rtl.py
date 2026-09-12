"""A wider synthetic checkpoint exercises general datapath dimensions; no training."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file
from compiler.compile import compile_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def fixture_checkpoint(directory):
    directory.mkdir()
    cfg = dict(
        model_type="gpt2",
        n_embd=8,
        n_head=2,
        n_layer=2,
        n_positions=8,
        vocab_size=17,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
    )
    (directory / "config.json").write_text(json.dumps(cfg))
    rng = np.random.default_rng(333)
    state = {}

    def weight(name, shape, scale=0.2):
        state[name] = rng.normal(0, scale, shape).astype(np.float32)

    weight("transformer.wte.weight", (17, 8))
    weight("transformer.wpe.weight", (8, 8))
    for l in range(2):
        p = f"transformer.h.{l}."
        for ln in ("ln_1", "ln_2"):
            state[p + ln + ".weight"] = rng.uniform(0.5, 1.5, 8).astype(np.float32)
            weight(p + ln + ".bias", (8,), 0.1)
        for name, shape in [
            ("attn.c_attn", (8, 24)),
            ("attn.c_proj", (8, 8)),
            ("mlp.c_fc", (8, 32)),
            ("mlp.c_proj", (32, 8)),
        ]:
            weight(p + name + ".weight", shape)
            weight(p + name + ".bias", (shape[1],), 0.05)
    state["transformer.ln_f.weight"] = np.ones(8, dtype=np.float32)
    state["transformer.ln_f.bias"] = np.zeros(8, dtype=np.float32)
    save_file(state, str(directory / "model.safetensors"))
    (directory / "calibration.json").write_text(
        json.dumps([list(range(8)), list(range(8, 16))])
    )


@pytest.mark.parametrize("sim,bits", [("icarus", 8), ("verilator", 4), ("icarus", 2)])
def test_complete_model(tmp_path, sim, bits):
    model = tmp_path / "checkpoint"
    fixture_checkpoint(model)
    out = tmp_path / "compiled"
    manifest = compile_checkpoint(model, out, bits=bits, context=8)
    assert manifest["scope"] == "full_model"
    for source in (out / "rtl").glob("*.sv"):
        text = source.read_text()
        assert "$readmem" not in text
        assert "load_w" not in text
    # Move the artifact before verification: no generator-machine paths may remain.
    moved = tmp_path / "relocated"
    out.rename(moved)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify.py"),
            str(moved),
            "--sim",
            sim,
            "--tokens",
            "1,2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((moved / f"verification_{sim}.json").read_text())
    assert result["passed"] and result["logits_compared"] == 17 * 8
