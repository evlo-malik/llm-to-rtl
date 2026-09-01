# Source this before doing anything: `source env.sh`
# Fixes the two traps found on 2026-09-01:
#   1. Anaconda's python3 is x86_64 and breaks Verilator's internal scripts
#      ("SRE module mismatch") and links Intel cocotb libs against arm64 Verilator.
#   2. Homebrew EDA tools must be ahead of Anaconda on PATH.
REPO="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export PATH="$REPO/.venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
[ -d "$HOME/Developer/eda/oss-cad-suite/bin" ] && export PATH="$PATH:$HOME/Developer/eda/oss-cad-suite/bin"
export REPO
unset PYTHONHOME
unset PYTHONPATH
echo "rtl-systolic-array env ready"
echo "  python:    $(command -v python3)  ($(python3 --version 2>&1))"
echo "  verilator: $(verilator --version 2>/dev/null | head -1)"
echo "  yosys:     $(yosys -V 2>/dev/null | cut -d' ' -f1-2)"
