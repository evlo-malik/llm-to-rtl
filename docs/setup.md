# Setup

Tested with Python 3.13, Icarus 13, Verilator 5.050, and Yosys 0.68 with the slang
frontend. Python packages are pinned in `requirements.txt`.

Install the [OSS CAD Suite release dated 2026-09-12](https://github.com/YosysHQ/oss-cad-suite-build/releases/tag/2026-09-12)
for your operating system. Activate its environment as described in that release,
then create and activate this repository's Python virtual environment. Keep the
virtual environment ahead of other Python installations on `PATH`.

Check the tools before running the examples:

```sh
python --version
iverilog -V
verilator --version
yosys -m slang -p 'help read_slang'
```

`env.sh` is an optional helper for the original macOS workspace. The GitHub workflow
installs the pinned EDA release on a clean Ubuntu runner and runs the regression
suite plus synthesis of the pretrained Stories circuit.

Compilation refuses an existing output directory. Use a new directory after
changing weights, calibration, precision or context. Keep the generated manifest
with the RTL: verification checks its hashes before running.

For the area comparison, provide a Sky130 HD Liberty file explicitly. It is a
separate PDK dependency and is not redistributed here:

```sh
python syn/compare.py models/stories260k --out gen/area \
  --liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
```

The comparison records the library hash. Cell area depends on the library and
mapping flow; it does not measure routed die area or power.
