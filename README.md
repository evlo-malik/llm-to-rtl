# rtl-systolic-array

A weight-stationary INT8 systolic array in SystemVerilog, verified against PyTorch with cocotb,
synthesised to sky130 with Yosys and OpenSTA.

**Status: work in progress.** Part 1 of a larger project: a compiler from Hugging Face
safetensors to Verilog with the model's weights hardwired into the arrays, ending with a
self-trained GPT running on an FPGA. This repo is the array itself.

## What exists today

| File | What | Verified |
|---|---|---|
| `rtl/mul.sv` | Registered signed 8×8→16 multiplier | 6 directed + 200 random cases, Verilator and Icarus |
| `rtl/pe.sv` | Processing element: weight register, `psum_out = psum_in + x*weight` | directed edge cases, weight-stationarity, chained MACs vs `torch.dot`, 1000 random |
| `rtl/array.sv` | N×N grid of PEs (in progress) | not yet |

Everything below is planned, not done: skew registers, load/compute/drain controller,
randomised matrix tests at N = 4, 8, 16, synthesis sweep of area and Fmax vs N.

## Design

- Inputs `x` flow left to right, partial sums flow top to bottom, each PE holds one weight.
- Signed INT8 inputs, signed INT32 partial sums.
- Weights shift in from the top edge during a load phase; the array is fed only at its edges,
  so pin count is linear in N.

## Reproduce

Apple Silicon, Homebrew. Everything is free and runs offline.

```bash
brew install verilator icarus-verilog yosys surfer
python3.13 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
source env.sh                       # every session: fixes PATH so arm64 Python wins
bash doctor.sh                      # 16 toolchain checks incl. an end-to-end smoke test
PYTHONPATH=tb python3 tb/test_pe.py            # Verilator
SIM=icarus PYTHONPATH=tb python3 tb/test_pe.py # second simulator
surfer sim_build/dump.vcd                      # waveforms
```

For synthesis timing, OpenSTA is built from source (see `START-HERE.md`); `env.sh` adds it
to PATH if present. The sky130 liberty file is not committed; `doctor.sh` tells you where to
put it.

## Layout

```
rtl/      SystemVerilog, one module per file
tb/       cocotb tests, PyTorch as the reference
syn/      Yosys and OpenSTA scripts, reports
docs/     spec, diagrams, report
_smoke/   throwaway counter that proves the toolchain; not part of the design
```

`LESSONS.md` is the build order. `START-HERE.md` is the environment setup and its traps.

## Licence

MIT.
