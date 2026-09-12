# llm-to-rtl

Compile pretrained model weights into fixed-coefficient SystemVerilog.
Weights become constant arithmetic. Generated matrix circuits have no weight-loading
port and do not fetch weights during inference.

## Status

- Fixed-coefficient matrix emitter; INT8, INT4 and ternary weights.
- Original programmable systolic array retained in `rtl/` as a reference.
- Pretrained full-model integration is being rebuilt around fixed-weight circuits.
- No training workflow or FPGA fit claim.

The previous ROM-fed prototype and its results remain in Git history. They do not
validate this backend.

## Layout

| Path | Purpose |
|---|---|
| `compiler/` | Model adapters, quantisation, RTL generation |
| `rtl/` | Reference array and numerical building blocks |
| `tb/` | Simulation tests |
| `models/`, `gen/` | Local checkpoints and generated output; ignored by Git |

MIT. See `LICENSE`.
