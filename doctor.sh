#!/usr/bin/env bash
# One-command health check for the toolchain. Run: bash doctor.sh
set -u
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1
pass=0; fail=0
chk() { printf "  %-34s" "$1"; if eval "$2" >/dev/null 2>&1; then echo "OK"; pass=$((pass+1)); else echo "FAIL"; fail=$((fail+1)); fi; }
echo "== toolchain =="
chk "verilator"                  "verilator --version"
chk "icarus verilog"             "command -v iverilog"
chk "yosys"                      "yosys -V"
chk "surfer (waveform viewer)"   "command -v surfer"
chk "python is arm64"            "file \$(command -v python3) | grep -q arm64"
chk "cocotb importable"          "python3 -c 'import cocotb'"
chk "numpy importable"           "python3 -c 'import numpy'"
chk "cocotb vpi lib is arm64"    "file .venv/lib/python3.13/site-packages/cocotb/libs/libcocotbvpi_verilator.so | grep -q arm64"
chk "sky130 liberty present"     "test -s syn/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
chk "nangate45 liberty present"  "test -s syn/lib/Nangate45_typ.lib"
chk "opensta (Fmax)"             "command -v sta"
chk "no spaces in repo path"     "! pwd | grep -q ' '"
chk "oss-cad-suite yosys + slang"    "$HOME/Developer/eda/oss-cad-suite/bin/yosys -m slang -p 'help read_slang' | grep -q read_slang"
chk "nextpnr-himbaechel (gowin)"     "$HOME/Developer/eda/oss-cad-suite/bin/nextpnr-himbaechel --help 2>&1 | grep -q device"
chk "gowin_pack"                     "$HOME/Developer/eda/oss-cad-suite/bin/gowin_pack --help"
chk "safetensors + matplotlib"       "python3 -c 'import safetensors, matplotlib'"
echo "== end-to-end smoke =="
chk "verilator + cocotb sim"     "rm -rf sim_build && SIM=verilator PYTHONPATH=_smoke python3 _smoke/test_counter.py"
chk "icarus + cocotb sim"        "rm -rf sim_build && SIM=icarus PYTHONPATH=_smoke python3 _smoke/test_counter.py"
chk "yosys synth to sky130"      "yosys -q _smoke/synth_smoke.ys"
chk "opensta timing report"      "sta -no_splash -exit _smoke/sta_smoke.tcl | grep -q slack"
echo
echo "  $pass passed, $fail failed"
exit $((fail > 0))
