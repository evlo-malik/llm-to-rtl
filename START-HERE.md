# Toolchain setup

The recorded runs used Apple Silicon, Python 3.13, Homebrew Verilator and Icarus,
and the OSS CAD Suite for Yosys with the slang frontend and the Gowin tools.
See [the report](docs/report.md) for versions and results.

```bash
brew install verilator icarus-verilog yosys surfer
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
source env.sh
```

Run these commands from the repository root. Source `env.sh` in each new shell.
It puts the virtual environment and Homebrew ahead of Anaconda, and clears
`PYTHONHOME` and `PYTHONPATH`. Mixing Intel Python libraries with arm64 Verilator
previously caused import errors and linker failures.

## Simulation

```bash
N=8 PYTHONPATH=tb python3 tb/test_top.py
SIM=icarus N=8 PYTHONPATH=tb python3 tb/test_top.py
```

The [README](README.md#reproduce) has the compiler and whole-model commands.
The tiny checkpoint is included. Download the corpus before compiling it, because
activation calibration reads `model/data/input.txt` even when training is skipped.
Generated files use absolute paths. Recompile after moving the checkout.

## Synthesis and FPGA tools

Install the OSS CAD Suite under `~/Developer/eda/oss-cad-suite`; the synthesis
scripts use its Yosys with `read_slang`. `env.sh` also checks
`~/Developer/eda/OpenSTA/build` for `sta`.

Supply these technology libraries locally; they are excluded from Git:

- `syn/lib/sky130_fd_sc_hd__tt_025C_1v80.lib`
- `syn/lib/Nangate45_typ.lib` (used by the toolchain health check)

```bash
bash doctor.sh
python3 syn/sweep.py 2 4 8 16
python3 syn/bitwidth.py
```

`doctor.sh` checks the complete local toolchain, including optional synthesis tools.
Its smoke tests clear `sim_build/`. It is specific to the recorded Apple Silicon
setup; missing synthesis tools do not prevent the standalone simulation tests.

The Tang Nano commands are in the README. The current flow passes
`--timing-allow-fail` to nextpnr; successful packing does not establish timing
closure or physical board operation. Read the board section of the report before
using the bitstream.

## Reading the design

Start with `pe.sv`, `array.sv`, `skew.sv` and `top.sv`, then `matvec_rom.sv`.
The remaining datapath units feed `sequencer.sv`. Read each module alongside
[the interface specification](docs/spec.md) and its testbench.
[LESSONS.md](LESSONS.md) records the implementation stages and verification results.
