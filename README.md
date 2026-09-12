# llm-to-rtl

Pretrained checkpoint → fixed-weight SystemVerilog.

Each linear map becomes constant shift/add logic. Weights are never loaded into a
shared array. Embeddings become fixed decoders. Only activations, counters and the
KV cache change during inference.

## Supported

| Input | Output |
|---|---|
| GPT-2 safetensors + config + calibration token IDs | Complete token-to-logits circuit |
| Llama-family safetensors + config, `--matrices-only` | Individual fixed linear maps |

INT8, INT4 and ternary matrix weights; INT8 linear inputs, INT16 residuals, INT32
accumulators. GPT-2 uses integer LayerNorm, a GELU lookup and integer attention.
Changing a checkpoint requires recompiling the circuit. No training step.

## Run

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
source env.sh
python scripts/fetch_example.py
python compiler/compile.py models/tiny-gpt2 --out gen/gpt2 --context 8
python scripts/verify.py gen/gpt2 --sim icarus
```

The example is `sshleifer/tiny-gpt2`, pinned to a checkpoint revision. It retains
all 50,257 vocabulary entries and both layers. Its hidden width is only 2; use it
for compiler verification, not language-quality claims. Checkpoints and generated
files are ignored by Git.

Calibration reads token IDs to choose numerical scales; it does not update weights.
The output directory must be new. Use `--max-coefficients 0` to lift the default
2-million-coefficient emission guard. Larger models produce larger circuits;
there is no weight-memory fallback.

## Verify

```sh
python -m pytest tests -q
python scripts/verify.py gen/gpt2 --sim verilator
python syn/synth.py gen/gpt2
```

Simulation requires Icarus or Verilator. Synthesis requires Yosys with `read_slang`.
The synthesis flow maps read-only tables to gates and audits fixed linear maps for
weight memories and runtime multipliers. Mutable activation memories are allowed.

No FPGA fit, physical timing, power or manufactured-silicon claim. The original
programmable systolic array remains in `rtl/` as a reference; compiled models do
not instantiate it.

## Layout

| Path | Contents |
|---|---|
| `compiler/` | Checkpoint loader, GPT-2 lowering, integer reference, RTL emitters |
| `rtl/` | Dynamic attention, normalisation, controller, reference array |
| `scripts/` | Pinned example download and model verification |
| `tests/`, `tb/` | Compiler tests and RTL comparisons |
| `syn/` | Logic lowering and structural audit |

MIT. See `LICENSE`.
