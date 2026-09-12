#!/usr/bin/env python3
# Tang Nano 20K flow for a compiled model: yosys synth_gowin -> nextpnr-himbaechel ->
# gowin_pack. Same commands as apicula's examples/Makefile, with the slang frontend so
# the SystemVerilog reads as-is.
#
#   python3 syn/gowin/build.py gen/tinygpt            # full flow, writes gen/tinygpt/gowin/
#   python3 syn/gowin/build.py gen/tinygpt --synth    # synthesis only, prints utilisation
#   python3 syn/gowin/build.py gen/tinygpt --top gpt_top --synth   # core without the UART
#   python3 syn/gowin/build.py gen/tinygpt_t2_int4 --nodsp          # the build that fits; see docs/report.md
#
# Flash with: openFPGALoader -b tangnano20k gen/tinygpt/gowin/tang_nano_20k.fs
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OSS = Path.home() / "Developer" / "eda" / "oss-cad-suite" / "bin"
CORE = ["pe.sv", "array.sv", "skew.sv", "top.sv", "matvec_rom.sv", "requant.sv", "isqrt.sv", "udiv.sv",
        "layernorm.sv", "attention.sv", "embed.sv", "sequencer.sv"]
BOARD = ["board/uart_rx.sv", "board/uart_tx.sv", "board/tang_nano_20k.sv"]
DEVICE = "GW2AR-LV18QN88C8/I7"
FAMILY = "GW2A-18C"


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(str(c) for c in cmd)}\nsee {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--top", default="tang_nano_20k")
    ap.add_argument("--synth", action="store_true", help="stop after synthesis")
    ap.add_argument("--nodsp", action="store_true", help="multipliers in LUTs; nextpnr could not place 147 MULT9X9")
    a = ap.parse_args()
    gen = Path(a.gen_dir).resolve()
    out = gen / "gowin"
    out.mkdir(exist_ok=True)
    mem = gen / "mem"
    srcs = [REPO / "rtl" / f for f in CORE] + sorted((gen / "rtl").glob("mv_*.sv")) + sorted((gen / "rtl").glob("const_*.sv")) + [gen / "rtl" / "gpt_top.sv"]
    gparams = ""
    if a.top == "tang_nano_20k":
        srcs += [REPO / "rtl" / f for f in BOARD]
        gparams = f'-G STOI_FILE="{mem / "stoi.memh"}" -G ITOS_FILE="{mem / "itos.memh"}"'
    synth_json = out / f"{a.top}_synth.json"
    script = (f"read_slang {gparams} --top {a.top} " + " ".join(str(s) for s in srcs) +
              f"; hierarchy -top {a.top}; synth_gowin -top {a.top} -family gw2a {'-nodsp' if a.nodsp else ''} -json {synth_json}; "
              f"tee -q -o {out / (a.top + '_stat.json')} stat -json")
    run([str(OSS / "yosys"), "-m", "slang", "-p", script], out / f"{a.top}_synth.log")
    st = json.load(open(out / f"{a.top}_stat.json"))
    m = st.get("modules", {})
    m = next(iter(m.values())) if m else st["design"]
    by = m["num_cells_by_type"]
    util = dict(
        lut=sum(v for k, v in by.items() if k.startswith("LUT")),
        alu=sum(v for k, v in by.items() if k == "ALU"),
        dff=sum(v for k, v in by.items() if k.startswith("DFF")),
        bsram=sum(v for k, v in by.items() if k in ("SP", "SPX9", "DP", "DPX9", "DPB", "DPX9B", "SDP", "SDPB", "SDPX9B", "pROM", "pROMX9")),
        lutram=sum(v for k, v in by.items() if k.startswith("RAM16")),
        mult9=by.get("MULT9X9", 0), mult18=by.get("MULT18X18", 0),
        cells_by_type=by,
    )
    print(f"synth: {util['lut']} LUT + {util['alu']} ALU (of 20736 LUT4), {util['dff']} DFF (of 15552), "
          f"{util['bsram']} BSRAM blocks (of 46), {util['lutram']} LUTRAM, {util['mult9']} MULT9X9 + {util['mult18']} MULT18X18 "
          f"(48 DSP macros = 192 MULT9X9)", flush=True)
    json.dump(util, open(out / f"{a.top}_util.json", "w"), indent=2)
    if a.synth:
        return
    pnr_json = out / f"{a.top}_pnr.json"
    run([str(OSS / "nextpnr-himbaechel"), "--json", str(synth_json), "--write", str(pnr_json), "--device", DEVICE,
         "--vopt", f"family={FAMILY}", "--vopt", f"cst={REPO / 'syn' / 'gowin' / 'tangnano20k.cst'}",
         "--freq", "27", "--timing-allow-fail", "--report", str(out / f"{a.top}_report.json")], out / f"{a.top}_pnr.log")
    # --timing-allow-fail: nextpnr's Gowin timing model reports hold "violations" of ~0.1 ns on
    # flop -> BSRAM address paths and exits nonzero; the routed design is still written
    log = (out / f"{a.top}_pnr.log").read_text()
    fmax = re.findall(r"Max frequency for clock\s+'\S+': ([\d.]+) MHz", log)
    holds = len(re.findall(r"Hold/min time violation", log))
    print(f"place and route done; max frequency {fmax[-1] if fmax else '?'} MHz (27 MHz crystal); "
          f"{holds} hold reports from nextpnr's Gowin timing model")
    run([str(OSS / "gowin_pack"), "-c", "-d", FAMILY, "-o", str(out / f"{a.top}.fs"), str(pnr_json)], out / f"{a.top}_pack.log")
    print(f"bitstream: {out / (a.top + '.fs')}")


if __name__ == "__main__":
    main()
