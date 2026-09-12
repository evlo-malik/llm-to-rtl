"""Pretrained Llama fixture, independent comparison against Transformers."""

from pathlib import Path
import numpy as np
import pytest
import torch
from compiler.checkpoint import load_checkpoint
from compiler.llama import fold_llama
from compiler.reference import FloatDecoder

MODEL = Path(__file__).resolve().parents[1] / "models/stories260k"


@pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists(), reason="run scripts/fetch_stories.py"
)
def test_pretrained_llama_against_upstream():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg, state, _ = load_checkpoint(MODEL)
    config = LlamaConfig(**cfg)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    tensors = {k: torch.from_numpy(v) for k, v in state.items()}
    if cfg["tie_word_embeddings"]:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    model.load_state_dict(tensors, strict=True)
    tokens = [1, 403, 407, 261, 265, 411, 272, 315]
    with torch.no_grad():
        expected = model(torch.tensor([tokens])).logits[0].numpy()
    np.testing.assert_allclose(
        FloatDecoder(fold_llama(cfg, state, 8)).forward(tokens),
        expected,
        rtol=2e-4,
        atol=2e-5,
    )
