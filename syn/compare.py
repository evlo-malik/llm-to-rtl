#!/usr/bin/env python3
"""Compare mapped cell area of fixed and programmable copies of one checkpoint matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.checkpoint import load_checkpoint
from compiler.emit import emit_linear, balanced_sum
from compiler.quant import quant_weight


def programmable(path, rows, cols, bits):
    # Same buffering, parallel dot products and output protocol as the fixed map.
    emit_linear(path, "linear", np.ones((rows, cols), dtype=np.int64))
    s = path.read_text().replace(
        "input logic [7:0] in_data,",
        f"input logic [7:0] in_data, input logic load_w, input logic [{rows * cols * bits - 1}:0] weights_in,",
    )
    start = s.index("    wire signed [31:0] x0 =")
    end = s.index("    wire [31:0] bank_result", start)
    lines = [
        f"    logic [{rows * cols * bits - 1}:0] weights;",
        "    always_ff @(posedge clk) if(!rst_n) weights <= '0; else if(load_w) weights <= weights_in;",
    ]
    for j in range(rows):
        terms = []
        for i in range(cols):
            name = f"p{j}_{i}"
            coefficient = f"weights[{(j * cols + i) * bits} +:{bits}]"
            if bits == 2:
                expression = f"{coefficient} == 2'b01 ? 32'($signed(x[{i}])) : {coefficient} == 2'b11 ? -32'($signed(x[{i}])) : 32'sd0"
            else:
                expression = f"32'($signed(x[{i}])) * 32'($signed({coefficient}))"
            lines.append(f"    wire signed [31:0] {name} = {expression};")
            terms.append(name)
        lines.append(f"    wire signed [31:0] y{j} = {balanced_sum(terms)};")
    path.write_text(s[:start] + "\n".join(lines) + "\n" + s[end:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--liberty", type=Path, required=True)
    p.add_argument(
        "--yosys",
        default=shutil.which("yosys")
        or str(Path.home() / "Developer/eda/oss-cad-suite/bin/yosys"),
    )
    p.add_argument("--rows", type=int, default=0)
    p.add_argument("--cols", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, state, hashes = load_checkpoint(args.model_dir)
    w = state[args.tensor]
    if w.ndim != 2:
        p.error("select a rank-2 tensor")
    w = w[: args.rows or len(w), : args.cols or w.shape[1]]
    lib = args.liberty.resolve()
    report = dict(
        mapping_script="strash; &get -n; &nf; &put",
        yosys=subprocess.check_output([args.yosys, "-V"], text=True).strip(),
        baseline="parallel programmable dot products; ternary uses sign/zero selection",
        tensor=args.tensor,
        shape=list(w.shape),
        checkpoint_sha256=hashes,
        liberty_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        liberty=lib.name,
        results=[],
    )
    for bits in (8, 4, 2):
        qw, _ = quant_weight(w, bits)
        for kind in ("fixed", "programmable"):
            dest = args.out / f"{kind}_{bits}"
            dest.mkdir()
            rtl = dest / "linear.sv"
            if kind == "fixed":
                emit_linear(rtl, "linear", qw)
            else:
                programmable(rtl, *w.shape, bits)
            script = f'''read_slang --top linear linear.sv
synth -top linear -noabc
dfflibmap -liberty "{lib}"
abc -script "+strash;&get,-n;&nf;&put" -liberty "{lib}"
clean
tee -o stats.json stat -json -liberty "{lib}"
write_json mapped.json
'''
            (dest / "map.ys").write_text(script)
            with (dest / "yosys.log").open("w") as log:
                subprocess.run(
                    [args.yosys, "-m", "slang", "-s", "map.ys"],
                    cwd=dest,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            # Yosys stat can append human-readable hierarchy text after JSON.
            raw = (dest / "stats.json").read_text()
            stats = json.JSONDecoder().raw_decode(raw[raw.index("{") :])[0]
            module = stats["modules"]["\\linear"]
            if any(name.startswith("$") for name in module["num_cells_by_type"]):
                raise RuntimeError("unmapped internal cells remain")

            row = dict(
                bits=bits,
                kind=kind,
                cells=module.get("num_cells"),
                cell_area=module.get("area"),
                rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
            )
            if row["cell_area"] is None:
                raise RuntimeError("technology mapping did not report cell area")
            report["results"].append(row)
            (args.out / "comparison.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
    print(args.out / "comparison.json")


if __name__ == "__main__":
    main()
