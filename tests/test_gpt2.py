"""Independent comparison with the upstream GPT-2 implementation."""

from pathlib import Path
import numpy as np
import pytest
import torch
from compiler.checkpoint import load_checkpoint
from compiler.gpt2 import fold_gpt2, FloatGPT, calibrate, quantise_model, IntGPT

MODEL = Path(__file__).resolve().parents[1] / "models/tiny-gpt2"


@pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists(),
    reason="download example checkpoint first",
)
def test_pretrained_gpt2_against_transformers():
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg, st, _ = load_checkpoint(MODEL)
    config = GPT2Config(**cfg)
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    state = {
        k: torch.from_numpy(v)
        for k, v in st.items()
        if not k.endswith(".attn.masked_bias")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    folded = fold_gpt2(cfg, st, 8)
    tokens = [15496, 995, 0, 314, 716, 257, 703, 13]
    with torch.no_grad():
        expected = model(torch.tensor([tokens])).logits[0].numpy()
    actual = FloatGPT(folded).forward(tokens)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    q = quantise_model(folded, calibrate(folded, [tokens]), 8)
    actual_int = IntGPT(q).forward(tokens)
    agreement = np.mean(actual_int.argmax(-1) == expected.argmax(-1))
    assert agreement >= 0.75
    print(
        "float max error:",
        np.abs(actual - expected).max(),
        "INT8 argmax agreement:",
        agreement,
    )


def test_wider_float_adapter_against_transformers(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel
    from tests.test_model_rtl import fixture_checkpoint

    directory = tmp_path / "checkpoint"
    fixture_checkpoint(directory)
    cfg, state, _ = load_checkpoint(directory)
    config = GPT2Config(**cfg)
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    tensors = {k: torch.from_numpy(v) for k, v in state.items()}
    tensors["lm_head.weight"] = tensors["transformer.wte.weight"]
    model.load_state_dict(tensors, strict=True)
    model.eval()
    tokens = [1, 6, 4, 9, 16, 0, 2, 8]
    with torch.no_grad():
        expected = model(torch.tensor([tokens])).logits[0].numpy()
    actual = FloatGPT(fold_gpt2(cfg, state, 8)).forward(np.array(tokens))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
