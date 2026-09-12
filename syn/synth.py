#!/usr/bin/env python3
"""Lower RTL, map every read-only table to gates, and audit coefficient storage."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def number(value):
    return int(value, 2) if isinstance(value, str) else int(value)


def audit(netlist, matrix_names):
    modules = netlist["modules"]
    memories = []
    violations = []
    matrix_cells = {}
    seen = set()
    for name, module in modules.items():
        base = name.split("$", 1)[0]
        if base in matrix_names:
            seen.add(base)
            matrix_cells[name] = len(module.get("cells", {}))
        for cell_name, cell in module.get("cells", {}).items():
            kind = cell["type"]
            if kind.startswith("$mem"):
                params = cell["parameters"]
                write_ports = number(params.get("WR_PORTS", 0))
                entry = dict(
                    module=name,
                    cell=cell_name,
                    write_ports=write_ports,
                    bits=number(params.get("WIDTH", 0)) * number(params.get("SIZE", 0)),
                )
                memories.append(entry)
                if write_ports == 0 or (base in matrix_names and cell_name != "x"):
                    violations.append(
                        f"{name}.{cell_name}: coefficient/read-only memory remains"
                    )
            if base in matrix_names and kind in ("$mul", "$macc", "$macc_v2"):
                violations.append(
                    f"{name}.{cell_name}: runtime multiplier in fixed linear map"
                )
    missing = sorted(set(matrix_names) - seen)
    violations += [f"missing matrix module: {name}" for name in missing]
    return dict(
        passed=not violations,
        violations=violations,
        matrix_cells=matrix_cells,
        mutable_memories=memories,
        mutable_memory_bits=sum(m["bits"] for m in memories),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--module", help="audit one fixed matrix from a full-model artifact"
    )
    parser.add_argument("--yosys", help="Yosys with the slang plugin")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    target = out / "synthesis"
    if args.module:
        target = target / args.module
    target.mkdir(parents=True, exist_ok=True)
    default = Path.home() / "Developer/eda/oss-cad-suite/bin/yosys"
    yosys = args.yosys or (str(default) if default.exists() else shutil.which("yosys"))
    if not yosys:
        parser.error("Yosys was not found")
    names = [m["module"] for m in manifest["matrix_modules"]]
    if args.module:
        if args.module not in names:
            parser.error("module is not a fixed matrix in this manifest")
        names = [args.module]
    if not args.module and manifest["scope"] != "full_model" and len(names) != 1:
        parser.error("synthesise a full model or a single selected matrix")
    target.mkdir(parents=True, exist_ok=True)
    top = args.module or manifest.get("top", names[0])
    hashes = {
        k: v
        for k, v in manifest["rtl_sha256"].items()
        if not args.module or k == args.module + ".sv"
    }
    for name, digest in hashes.items():
        if hashlib.sha256((out / "rtl" / name).read_bytes()).hexdigest() != digest:
            parser.error(f"generated RTL changed: {name}")
    # Run from the output directory so no absolute paths enter the scripts.
    files = " ".join("rtl/" + name for name in sorted(hashes))
    relative = target.relative_to(out)
    script = f"""read_slang --keep-hierarchy --top {top} {files}
hierarchy -check -top {top}
proc
opt -fast
memory_collect
memory_map -rom-only
opt_clean
check -assert
write_json {relative}/netlist.json
stat
"""
    (target / "lower.ys").write_text(script)
    with (target / "yosys.log").open("w") as log:
        subprocess.run(
            [yosys, "-m", "slang", "-s", str(relative / "lower.ys")],
            cwd=out,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report = audit(json.loads((target / "netlist.json").read_text()), names)
    report["yosys"] = subprocess.check_output([yosys, "-V"], text=True).strip()
    report["rtl_sha256"] = hashes
    report["scope"] = "selected_matrix" if args.module else manifest["scope"]
    report["top"] = top
    (target / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
