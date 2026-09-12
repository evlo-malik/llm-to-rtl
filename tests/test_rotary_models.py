"""Architecture adapters are checked against upstream models before RTL comparison."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from compiler.llama import fold_llama
from compiler.reference import FloatDecoder
from compiler.checkpoint import load_checkpoint
from compiler.compile import compile_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def fixture(directory, family, rope=None):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )

    types = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
    }
    config, model = types[family]
    kwargs = dict(
        hidden_size=12,
        intermediate_size=20,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=19,
        max_position_embeddings=16,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        sliding_window=3,
    )
    if family == "qwen2":
        kwargs.update(use_sliding_window=True, max_window_layers=1)
    if rope:
        kwargs["rope_scaling"] = rope
    cfg = config(**kwargs)
    cfg._attn_implementation = "eager"
    torch.manual_seed(781)
    ref = model(cfg).eval()
    # Nonuniform affine terms catch incorrect RMSNorm folding.
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.copy_(torch.linspace(0.7, 1.3, p.numel()).reshape(p.shape))
            if name.endswith("bias"):
                p.copy_(torch.linspace(-0.1, 0.1, p.numel()))
    directory.mkdir()
    (directory / "config.json").write_text(cfg.to_json_string())
    save_file(
        {k: v.contiguous().clone() for k, v in ref.state_dict().items()},
        str(directory / "model.safetensors"),
    )
    (directory / "calibration.json").write_text(
        json.dumps([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
    )
    return ref


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
def test_float_adapter(tmp_path, family):
    ref = fixture(tmp_path / "model", family)
    cfg, st, _ = load_checkpoint(tmp_path / "model")
    f = fold_llama(cfg, st, 8)
    tokens = [1, 5, 7, 2, 11, 3, 15, 8]
    with torch.no_grad():
        expected = ref(torch.tensor([tokens])).logits[0].numpy()
    np.testing.assert_allclose(
        FloatDecoder(f).forward(tokens), expected, rtol=2e-4, atol=2e-6
    )


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_type": "linear", "factor": 2.0},
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8,
        },
    ],
)
def test_scaled_rope(tmp_path, rope):
    ref = fixture(tmp_path / "model", "llama", rope)
    cfg, st, _ = load_checkpoint(tmp_path / "model")
    tokens = [1, 5, 7, 2, 11, 3, 15, 8]
    with torch.no_grad():
        expected = ref(torch.tensor([tokens])).logits[0].numpy()
    np.testing.assert_allclose(
        FloatDecoder(fold_llama(cfg, st, 8)).forward(tokens),
        expected,
        rtol=2e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "family,sim,bits",
    [("llama", "icarus", 8), ("qwen2", "verilator", 4), ("mistral", "icarus", 2)],
)
def test_full_rotary_rtl(tmp_path, family, sim, bits):
    fixture(tmp_path / "model", family)
    out = tmp_path / "circuit"
    compile_checkpoint(tmp_path / "model", out, context=8, bits=bits)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify.py"),
            str(out),
            "--sim",
            sim,
            "--tokens",
            "1,5,7,2,11,3,15,8",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-3000:]
    assert json.loads((out / f"verification_{sim}.json").read_text())["passed"]


def test_reject_unsupported_rope(tmp_path):
    fixture(tmp_path / "model", "llama")
    cfg, st, _ = load_checkpoint(tmp_path / "model")
    cfg["rope_scaling"] = {"rope_type": "unknown"}
    with pytest.raises(ValueError, match="RoPE"):
        fold_llama(cfg, st, 8)
