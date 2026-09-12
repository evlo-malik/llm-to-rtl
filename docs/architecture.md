# Compiler structure

Checkpoint arrays and configuration → architecture adapter → common decoder
representation → calibration and quantisation → fixed arithmetic and controller →
RTL simulation / logic synthesis.

| File | Responsibility |
|---|---|
| `compiler/planner.py` | Read tensor headers; estimate coefficient count and KV storage before loading weights |
| `compiler/checkpoint.py` | Validate and load safetensors, shard indexes and source hashes |
| `compiler/gpt2.py`, `compiler/llama.py` | Interpret architecture and tensor layouts; fold normalisation affine terms |
| `compiler/reference.py`, `compiler/quant.py` | Floating-point inference, calibration and integer arithmetic specification |
| `compiler/emit.py` | Emit constant matrix arithmetic with shared products and balanced addition trees |
| `compiler/model_rtl.py`, `compiler/rotary_rtl.py` | Emit embeddings, positional/SwiGLU units and the fixed operation schedule |
| `rtl/model_control.sv` | Move activations between units and stream logits |
| `rtl/attention.sv`, `rtl/layernorm.sv`, `rtl/normalise_wide.sv` | Input-dependent attention and normalisation |
| `syn/synth.py` | Lower tables to logic and audit weight storage |
| `syn/compare.py` | Map a checkpoint matrix and a programmable baseline into the same cell library |

## Adapter contract

Adapters produce embeddings, layer projections, normalisation type, attention-head
counts, position functions and attention windows. Matrices have shape `[output,input]`.
Affine normalisation parameters are folded into the following projection before
quantisation. Shared runtime units receive activations and fixed scheduling fields;
there is no weight-loading instruction or external weight interface.

Adding an architecture requires its adapter, validation against the upstream
floating-point implementation, and complete RTL comparisons. Renaming tensor keys
alone does not support a new computation. Mixture-of-experts routing, state-space
models, multimodal inputs and encoder-decoder cross-attention need additional
operations and are rejected by the full-model compiler.

## Supported configuration bounds

- Hidden width 2–4096; head dimensions must divide it. Rotary heads must be even.
- Compile-time context 2–1024, within the checkpoint's position limit.
- GPT-2 uses `gelu_new` and standard attention scaling.
- Llama/Qwen2/Mistral use SiLU and RMSNorm. Grouped KV heads must divide query heads.
- RoPE supports default, linear scaling and Llama-3 frequency scaling. Dynamic,
  YaRN and other scaling rules need an adapter extension.
- Sliding attention is causal with a fixed per-layer window. The sequence cache
  does not wrap: clear the sequence when the compiled context fills.
- Input checkpoints contain floating-point tensors. Packed GPTQ/AWQ checkpoints
  require dequantisation before full-model compilation.

The matrix-only Llama adapter folds the preceding RMSNorm affine weight into its
projection. Other matrix-only inputs are raw rank-2 tensors interpreted as `[out,in]`.
Neither matrix-only mode implements a full model.

## Precision selection

The default uses calibration ranges to select between two paths:

| Path | Normalised / feed-forward values | Residuals | Q/K/V and KV cache |
|---|---|---|---|
| Compact | INT8 | INT16 | INT8 |
| Wide | INT16 | INT32 | INT8 |

Use --activation-bits 8 or 16 to override it. The wide normaliser accumulates
squares in 80 bits and uses a 64-bit square root with a 48-bit reciprocal.
Its SiLU/GELU lookup uses 4,096 nearest-grid entries. Matrix accumulators remain
INT32; compilation rejects a dot product whose worst-case bound would overflow.

This is post-training quantisation, not a guarantee of unchanged model quality.
Evaluate held-out token sequences before using a new checkpoint. Source hashes,
precision selection and scales are retained with each compiled artifact.
