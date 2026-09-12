# Verification

Recorded 12 September 2026; [machine-readable results](results.json).

| Check | Result |
|---|---|
| Compiler and RTL regression | 25 passed |
| Complete pretrained GPT-2, INT8, context 8 | 402,056 logits exact in each simulator |
| GPT-2 fixed-matrix synthesis audit | 9 modules passed |
| SmolLM2 key projection, 192 × 576, INT4 | 6 vectors / 1,152 outputs exact; audit passed |
| Held-out GPT-2 float vs integer comparison | Same argmax at 31/31 positions; maximum logit error 0.00422 |

The tiny GPT-2 checkpoint is a verification fixture. These checks do not establish
language quality. Its RTL took 101,200–101,312 cycles per token, including serial
logit readout. No clock frequency or tokens/second is claimed.

Commands below use the pinned example fetched by `scripts/fetch_example.py`.
The checked-in JSON records checkpoint and generated-RTL hashes. Build products
and downloaded weights stay outside Git.

```sh
source env.sh
python -m pytest tests -q
python compiler/compile.py models/tiny-gpt2 --out gen/gpt2 --context 8
python scripts/verify.py gen/gpt2 --sim icarus
python scripts/verify.py gen/gpt2 --sim verilator
python scripts/evaluate.py models/tiny-gpt2 gen/gpt2
python syn/synth.py gen/gpt2
```

The full-model test compares every logit against an integer reference over eight
positions, then checks invalid inputs, context exhaustion and reset during work.
Compiler tests also exercise INT4 and ternary complete circuits, checkpoint
validation, deterministic generation and agreement with upstream GPT-2 software.

A separate pretrained SmolLM2-135M key projection exercises a larger matrix:

```sh
python compiler/compile.py models/smollm2-135m --out gen/key \
  --matrices-only --only layer0_k_proj --bits 4
python scripts/verify_matrix.py gen/key --sim verilator
python syn/synth.py gen/key
```

This is one matrix, not a complete SmolLM2 inference test. The synthesis audit
checks the lowered circuit for read-only memories and runtime multipliers in
fixed linear maps. It is not cell-library mapping or physical implementation.
