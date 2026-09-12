# What a hardwired weight costs, by bit width, against the generic array.
#
# Takes a KxK slice of a real weight matrix from the trained tiny GPT, quantises it to
# 8, 4 and 2 bits (ternary) with compiler/quant.py, emits the hardwired array with
# compiler/emit.py (CSD shift-adds, no multipliers) and synthesises it twice: to sky130
# with yosys+abc for area, and with synth_gowin for LUT4-equivalents (LUT + ALU cells).
# The generic KxK array (top.sv) goes through the same two flows. Writes
# syn/reports/bitwidth.json and docs/area_per_weight.png.
#
#   python3 syn/bitwidth.py            # K = 8 and 16
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "compiler"))
from quant import quant_weight          # noqa: E402
from emit import const_array_sv, csd    # noqa: E402
from golden_gpt import load_karpathy    # noqa: E402

LIB = REPO / "syn" / "lib" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
REPORTS = REPO / "syn" / "reports"
YOSYS = Path.home() / "Developer" / "eda" / "oss-cad-suite" / "bin" / "yosys"
WORK = REPORTS / "bitwidth"


def yosys(script, log):
    with open(log, "w") as f:
        subprocess.run([str(YOSYS), "-q", "-m", "slang", "-p", script], check=True, stdout=f, stderr=subprocess.STDOUT)


def stat_json(path):
    d = json.load(open(path))
    mods = d.get("modules", {})
    return next(iter(mods.values())) if mods else d["design"]


def sky130(sources, top, tag, gparams=""):
    out = WORK / f"{tag}.sky130.json"
    yosys(f"""
read_slang {gparams} --top {top} {' '.join(str(s) for s in sources)}
hierarchy -top {top}
synth -top {top} -flatten
dfflibmap -liberty {LIB}
abc -liberty {LIB}
opt_clean
tee -q -o {out} stat -liberty {LIB} -json
""", WORK / f"{tag}.sky130.log")
    return stat_json(out)


def gowin(sources, top, tag, gparams=""):
    out = WORK / f"{tag}.gowin.json"
    yosys(f"""
read_slang {gparams} --top {top} {' '.join(str(s) for s in sources)}
hierarchy -top {top}
synth_gowin -top {top} -nodsp
tee -q -o {out} stat -json
""", WORK / f"{tag}.gowin.log")
    st = stat_json(out)
    by = st["num_cells_by_type"]
    luts = sum(v for k, v in by.items() if k.startswith("LUT"))
    alus = sum(v for k, v in by.items() if k == "ALU")
    ffs = sum(v for k, v in by.items() if k.startswith("DFF"))
    return dict(luts=luts, alus=alus, lut_equiv=luts + alus, ffs=ffs, cells=st["num_cells"])


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    cfg, st = load_karpathy(REPO / "model" / "tinygpt")
    w_full = st["blocks.0.sa.proj.weight"]          # [out, in], a dense matrix with no LN fold
    rtl = REPO / "rtl"
    rows = []
    for k in (8, 16):
        # generic array of the same size
        srcs = [rtl / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv")]
        sk = sky130(srcs, "top", f"generic_{k}", f"-G N={k}")
        gw = gowin(srcs, "top", f"generic_{k}", f"-G N={k}")
        rows.append(dict(kind="generic INT8 (top.sv)", bits=8, K=k, weights=k * k, nonzero=k * k, csd_digits=None,
                         area_um2=sk["area"], area_per_weight=sk["area"] / (k * k), sky_cells=sk["num_cells"],
                         lut_equiv=gw["lut_equiv"], lut_per_weight=gw["lut_equiv"] / (k * k), ffs=gw["ffs"]))
        print(rows[-1], flush=True)
        for bits in (8, 4, 2):
            wq, _ = quant_weight(w_full[:k, :k], bits, per_channel=True)
            w_kn = wq.T                                     # x @ W
            name = f"const_proj{k}_b{bits}"
            sv = WORK / f"{name}.sv"
            sv.write_text(const_array_sv(name, w_kn))
            digits = int(sum(len(csd(v)) for v in w_kn.reshape(-1)))
            srcs = [rtl / "skew.sv", sv]
            sk = sky130(srcs, name, name)
            gw = gowin(srcs, name, name)
            rows.append(dict(kind=f"hardwired INT{bits}" if bits > 2 else "hardwired ternary", bits=bits, K=k,
                             weights=k * k, nonzero=int((w_kn != 0).sum()), csd_digits=digits,
                             area_um2=sk["area"], area_per_weight=sk["area"] / (k * k), sky_cells=sk["num_cells"],
                             lut_equiv=gw["lut_equiv"], lut_per_weight=gw["lut_equiv"] / (k * k), ffs=gw["ffs"]))
            print(rows[-1], flush=True)
    json.dump(rows, open(REPORTS / "bitwidth.json", "w"), indent=2)
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = ["generic INT8 (top.sv)", "hardwired INT8", "hardwired INT4", "hardwired ternary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in ((axes[0], "area_per_weight", "sky130 area per weight, um$^2$"),
                           (axes[1], "lut_per_weight", "Gowin LUT4 + ALU cells per weight")):
        for ki, kind in enumerate(kinds):
            for kk, mark in ((8, "o"), (16, "s")):
                r = [x for x in rows if x["kind"] == kind and x["K"] == kk]
                if r:
                    ax.bar(ki + (0.2 if kk == 16 else -0.2), r[0][key], width=0.38,
                           color=f"C{ki}", alpha=0.6 if kk == 16 else 1.0, label=f"{kk}x{kk}" if ki == 0 else None)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(["generic\nINT8", "hardwired\nINT8", "hardwired\nINT4", "hardwired\nternary"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="array")
    fig.suptitle("cost of one weight: registers + multiplier vs CSD shift-add constants (layer0 proj slice)")
    fig.tight_layout()
    fig.savefig(REPO / "docs" / "area_per_weight.png", dpi=150)
    print("wrote docs/area_per_weight.png")


if __name__ == "__main__":
    main()
