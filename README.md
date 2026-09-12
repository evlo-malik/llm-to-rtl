# llm-to-rtl

[![Verification](https://github.com/evlo-malik/llm-to-rtl/actions/workflows/verify.yml/badge.svg)](https://github.com/evlo-malik/llm-to-rtl/actions/workflows/verify.yml)

Compile a **pretrained transformer checkpoint into a model-specific RTL circuit**.
Matrix weights become constant shift/add networks. Embeddings become fixed logic.
The circuit accepts token IDs and produces next-token scores. No training and no
weight fetches into a shared compute array during inference.

This explores the model-specific hardware direction described by
[Taalas](https://taalas.com/the-path-to-ubiquitous-ai/) and
[Lamb Labs](https://lamb-labs.com/). It is an independent RTL compiler; those
companies' physical implementations and performance results are not reproduced here.

## Demonstrated result

The pretrained Stories260K circuit generated this in Verilator:

> Once upon a time, there was a little girl named Lily. She loved to play outside
> in the park. One day, she went to the park to play.

All 32,768 logits over 64 token positions matched the integer reference. Icarus
independently checked eight positions. The complete circuit passed synthesis and
the weight-storage audit. [Results and reproduction commands](docs/results.md).

The larger SmolLM2-135M checkpoint produced a complete circuit description with
162.8 million fixed coefficients. Every emitted coefficient was audited; one
projection was simulated and synthesis-audited. Full-model SmolLM2 RTL simulation
and physical implementation have not been completed.

A matched 64×64 matrix study measured **6.22× lower mapped cell area at INT8**
than a programmable parallel datapath. See [area measurements](docs/area.md) for
the baseline and limits. The compiler and RTL regression has 54 passing tests.

## Run a pretrained model

Install Python 3.13 and the [EDA tools](docs/setup.md), then:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_stories.py
python compiler/compile.py models/stories260k --out gen/stories --context 64
python scripts/generate.py gen/stories --model-dir models/stories260k \
  --prompt '' --new-tokens 64
```

This downloads a pinned **Stories260K** checkpoint: five Llama-style layers,
width 64, eight query heads, four KV heads, and the complete 512-token vocabulary.
The generation command runs RTL simulation and checks every logit against the
integer reference. The first build takes several minutes.

The compiler also accepts your own local `config.json` and single or sharded
safetensors checkpoint. Supply representative token sequences in `calibration.json`;
calibration chooses numerical scales without changing the trained weights.

## Supported computation

| Decoder | Implemented operations | Verification |
|---|---|---|
| GPT-2 | LayerNorm, learned positions, GELU, multi-head attention | Pretrained tiny GPT-2 and wider fixtures |
| Llama | RMSNorm, RoPE, SwiGLU, grouped-query attention | Pretrained Stories260K and fixtures |
| Qwen2 / Qwen2.5 | Q/K/V biases, grouped-query and mixed-window attention | Fixtures compared with Transformers and RTL |
| Mistral | Grouped-query, sliding-window attention | Fixtures compared with Transformers and RTL |

Matrix weights support INT8, INT4 and ternary. Calibration selects 8-bit or 16-bit
activation paths; the latter uses 32-bit residuals to preserve large dynamic ranges.
Configuration limits are listed in [architecture support](docs/architecture.md).

**A file format does not specify a computation.** Safetensors contains named arrays;
`config.json` selects the model architecture. A new architecture needs an adapter
and any missing operations. `--matrices-only` can emit individual rank-2 tensors
from other architectures, but does not produce complete model inference.

## Output and checks

`gen/stories/rtl/` is the generated hardware. `model_top.sv` connects the fixed
projections, normalisation, positional operations, attention, activation buffers
and controller. `manifest.json` records source hashes, numerical settings and the
operation schedule. Software reference files are used only for verification.

```sh
python -m pytest tests -q
python scripts/verify.py gen/stories --sim icarus --steps 8
python scripts/evaluate.py models/stories260k gen/stories
python syn/synth.py gen/stories
python compiler/planner.py models/stories260k --context 64
```

See [measured results](docs/results.md), [the host interface](docs/interface.md),
and [the compiler structure](docs/architecture.md).

## Hardware scope

All fixed linear maps exist spatially; input-dependent activations and the KV cache
use writable memory. A larger model requires more logic and routing. The planner
reports coefficient count before compilation; the default emission guard is two
million coefficients. `--max-coefficients 0` removes that guard, not physical limits.

The repository verifies RTL and supports logic synthesis. FPGA integration,
place-and-route, timing closure and power measurement remain hardware work.
The original programmable systolic array is retained as a reference; emitted
models do not instantiate it. Downloaded models, generated circuits and
`open-source/` are ignored by Git.

MIT. See [LICENSE](LICENSE).
