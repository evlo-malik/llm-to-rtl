# Synthesis sweep: top.sv at several N through yosys-slang -> sky130 -> OpenSTA.
# Writes syn/reports/top_N<n>.{json,v,sta.txt}, syn/reports/sweep.json and
# docs/area_fmax_vs_N.png.
#
#   python3 syn/sweep.py            # N = 2 4 8 16
#   python3 syn/sweep.py 4 8        # just those
#
# Yosys must be the OSS CAD Suite build (has the slang frontend; the Homebrew one
# cannot read unpacked-array ports). env.sh puts it on PATH after Homebrew, so the
# path is resolved explicitly here.
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "syn" / "lib" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
REPORTS = REPO / "syn" / "reports"
OSS = Path.home() / "Developer" / "eda" / "oss-cad-suite" / "bin"
YOSYS = OSS / "yosys"
STA = Path.home() / "Developer" / "eda" / "OpenSTA" / "build" / "sta"
RTL = [REPO / "rtl" / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv")]
PERIOD_NS = 5.0   # any value works; Fmax comes from period - slack
ABC_CONSTR = REPORTS / "abc.constr"   # lets abc buffer high-fanout nets and size gates


def synth(n, top="top", extra_flags=""):
    REPORTS.mkdir(exist_ok=True)
    ABC_CONSTR.write_text("set_driving_cell sky130_fd_sc_hd__buf_2\nset_load 0.02\n")
    tag = f"{top}_N{n}"
    netlist = REPORTS / f"{tag}_netlist.v"
    stat = REPORTS / f"{tag}.json"
    script = f"""
read_slang -G N={n} --top {top} {' '.join(str(f) for f in RTL)}
hierarchy -top {top}
synth -top {top} -flatten {extra_flags}
dfflibmap -liberty {LIB}
abc -liberty {LIB} -constr {ABC_CONSTR} -D 2500
opt_clean
tee -q -o {stat} stat -liberty {LIB} -json
write_verilog -noattr {netlist}
"""
    log = REPORTS / f"{tag}.yosys.log"
    with open(log, "w") as f:
        subprocess.run([str(YOSYS), "-q", "-m", "slang", "-p", script], check=True, stdout=f, stderr=subprocess.STDOUT)
    d = json.load(open(stat))
    # after -flatten there is one module; its key carries the parameter suffix
    mods = d.get("modules", {})
    m = mods[top] if top in mods else (next(iter(mods.values())) if mods else d["design"])
    return netlist, m


def timing(n, netlist, top="top"):
    tag = f"{top}_N{n}"
    tcl = REPORTS / f"{tag}.sta.tcl"
    out = REPORTS / f"{tag}.sta.txt"
    tcl.write_text(f"""read_liberty {LIB}
read_verilog {netlist}
link_design {top}
create_clock -name clk -period {PERIOD_NS} [get_ports clk]
set_input_delay 0 -clock clk [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay 0 -clock clk [all_outputs]
puts "== reg2reg =="
report_checks -path_delay max -digits 4 -from [all_registers -clock_pins] -to [all_registers -data_pins]
puts "== from_inputs =="
report_checks -path_delay max -digits 4 -from [delete_from_list [all_inputs] [get_ports clk]]
""")
    txt = subprocess.run([str(STA), "-no_splash", "-exit", str(tcl)], capture_output=True, text=True).stdout
    out.write_text(txt)
    res = {}
    for key in ("reg2reg", "from_inputs"):
        sec = txt.split(f"== {key} ==")[1].split("== ")[0]
        m = re.search(r"([-\d.]+)\s+slack \((MET|VIOLATED)\)", sec)
        slack = float(m.group(1))
        tmin = PERIOD_NS - slack
        start = re.search(r"Startpoint: (\S+)", sec)
        end = re.search(r"Endpoint: (\S+)", sec)
        res[key] = (tmin, 1000.0 / tmin, start.group(1) if start else "?", end.group(1) if end else "?")
    return res


def main():
    ns = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16]
    rows = []
    for n in ns:
        netlist, st = synth(n)
        tm = timing(n, netlist)
        (tmin, fmax, sp, ep), (itmin, ifmax, isp, iep) = tm["reg2reg"], tm["from_inputs"]
        cells = st["num_cells"]
        area = st["area"]
        row = dict(N=n, pes=n * n, cells=cells, area_um2=area, area_per_pe_um2=area / (n * n),
                   tmin_ns=tmin, fmax_mhz=fmax, crit_start=sp, crit_end=ep,
                   input_tmin_ns=itmin, input_fmax_mhz=ifmax, input_crit_start=isp, input_crit_end=iep,
                   dffs=sum(v for k, v in st["num_cells_by_type"].items() if "df" in k),
                   bufs=sum(v for k, v in st["num_cells_by_type"].items() if "buf" in k))
        rows.append(row)
        print(f"N={n:3d}  cells={cells:7d}  area={area:10.0f} um2  per PE={area/(n*n):7.0f}  "
              f"reg2reg Tmin={tmin:.3f} ns Fmax={fmax:.0f} MHz ({sp} -> {ep})   "
              f"input paths Tmin={itmin:.3f} ns ({isp} -> {iep})", flush=True)
    # merge with earlier runs so N=16 can be run on its own
    old = json.load(open(REPORTS / "sweep.json")) if (REPORTS / "sweep.json").exists() else []
    done = {r["N"] for r in rows}
    rows = sorted([r for r in old if r["N"] not in done] + rows, key=lambda r: r["N"])
    (REPORTS / "sweep.json").write_text(json.dumps(rows, indent=2))
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["N"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ns, [r["area_um2"] / 1e6 for r in rows], "o-", color="tab:blue")
    ax1.set_xlabel("N (array is N x N)")
    ax1.set_ylabel("area, mm$^2$ (sky130_fd_sc_hd)", color="tab:blue")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax2 = ax1.twinx()
    ax2.plot(ns, [r["fmax_mhz"] for r in rows], "s--", color="tab:red")
    ax2.set_ylabel("Fmax, MHz (OpenSTA, tt 25C 1.8V)", color="tab:red")
    ax2.set_ylim(0, max(r["fmax_mhz"] for r in rows) * 1.2)
    for r in rows:
        ax1.annotate(f"{r['area_per_pe_um2']:.0f} um$^2$/PE", (r["N"], r["area_um2"] / 1e6),
                     textcoords="offset points", xytext=(6, -12), fontsize=7, color="tab:blue")
    ax1.set_title("top.sv: area and register-to-register Fmax vs N (yosys + abc)")
    fig.tight_layout()
    (REPO / "docs").mkdir(exist_ok=True)
    fig.savefig(REPO / "docs" / "area_fmax_vs_N.png", dpi=150)
    print("wrote docs/area_fmax_vs_N.png")


if __name__ == "__main__":
    main()
