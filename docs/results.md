# Measured results

Recorded 12 September 2026. [Reports and hashes](results.json).
[Clean Ubuntu CI: 54 tests passed](https://github.com/evlo-malik/llm-to-rtl/actions/runs/34705279353).

| Check | Stories260K | SmolLM2-135M |
|---|---|---|
| Complete circuit emitted | 292,096 fixed coefficients; context 64 | 162,791,424 fixed coefficients; context 8 |
| Emitted coefficient audit | All 21 matrices and embeddings passed | All 121 matrices and embeddings passed |
| RTL simulation | 32,768 logits / 64 positions in Verilator; 4,096 / 8 in Icarus | First QKV projection: 6 vectors, 5,760 outputs in Verilator |
| Synthesis and weight-storage audit | Complete model passed | First QKV projection passed |

SmolLM2 emission includes all 30 layers and the full 49,152-token vocabulary.
The coefficient audit reconstructs constants from emitted arithmetic and checks
embedding literals. It does not replace full-model simulation. No full SmolLM2
RTL simulation, full-model cell mapping or physical implementation is claimed.
Fixed coefficient counts include separate embedding and output-head circuits,
even when the checkpoint ties their weights.

Stories260K generated this text in RTL simulation:

> Once upon a time, there was a little girl named Lily. She loved to play outside
> in the park. One day, she went to the park to play. She saw a big, red ball.
> The ball was very scared

Every output score matched the integer reference. The full-context run also
checked context exhaustion and clearing state both during and after inference.
Non-power-of-two vocabulary fixtures test invalid token IDs. Simulation wall time
is not hardware throughput; no clock frequency or tokens/second is claimed.

## Numerical accuracy

The floating-point adapters passed comparisons with upstream Transformers for
both pretrained checkpoints. Separate held-out checks compare the quantised
reference with floating-point inference:

| Model | Positions | Same next-token choice | Float perplexity | Integer perplexity |
|---|---:|---:|---:|---:|
| Stories260K | 179 | 169 / 179 | 4.36 | 4.48 |
| SmolLM2-135M | 31 | 25 / 31 | 25.07 | 35.46 |

These are four short sequences per model, not language-quality benchmarks.
Perplexity measures prediction error; lower is better. SmolLM2 loses measurable
accuracy under this quantisation scheme. It uses 16-bit normalised/feedforward
activations and 32-bit residuals; Stories uses 8-bit and 16-bit respectively.
RTL matching the integer reference does not mean matching the original model's
floating-point outputs.

## Area

The [64×64 pretrained matrix study](area.md) measured programmable/fixed mapped
cell-area ratios of 6.22× at INT8, 6.20× at INT4 and 6.68× for ternary weights.
Both circuits use the same parallelism and input/output protocol. This is a
cell-library mapping result for one matrix, not full-model area or a comparison
against an optimised time-shared accelerator.

## Reproduce

Install the dependencies in [setup](setup.md). Output directories must be new.
Downloaded weights and generated RTL stay outside Git.

```sh
python scripts/fetch_stories.py
python compiler/compile.py models/stories260k --out gen/stories --context 64
python scripts/generate.py gen/stories --model-dir models/stories260k \
  --prompt '' --new-tokens 64
python scripts/verify.py gen/stories --sim icarus --steps 8
python scripts/audit_coefficients.py gen/stories
python scripts/evaluate.py models/stories260k gen/stories
python syn/synth.py gen/stories
```

The larger example emits about 2.9 GiB of RTL:

```sh
python scripts/fetch_model.py HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --out models/smollm2-135m
python compiler/compile.py models/smollm2-135m --out gen/smol \
  --context 8 --max-coefficients 0
python scripts/audit_coefficients.py gen/smol
python scripts/verify_matrix.py gen/smol --module layer0_qkv --sim verilator
python syn/synth.py gen/smol --module layer0_qkv
python scripts/evaluate.py models/smollm2-135m gen/smol
```

The regression also checks GPT-2, Qwen2 and Mistral fixtures, rotary position
variants, grouped-query and sliding-window attention, arithmetic extremes,
checkpoint validation and malformed coefficient rejection:

```sh
python -m pytest tests -q
```
