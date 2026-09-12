#!/usr/bin/env python3
"""Compare one emitted fixed linear map with its integer checkpoint weights."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.bundle import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--module", help="select a fixed matrix from a full model")
    parser.add_argument("--sim", choices=("icarus", "verilator"), default="verilator")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    modules = manifest["matrix_modules"]
    names = [m["module"] for m in modules]
    if args.module:
        if args.module not in names:
            parser.error("module is not a fixed matrix in this manifest")
        name = args.module
    elif manifest["scope"] == "matrices_only" and len(names) == 1:
        name = names[0]
    else:
        parser.error("select a matrix with --module NAME")
    source = out / "rtl" / f"{name}.sv"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != manifest["rtl_sha256"][source.name]
    ):
        parser.error("generated RTL differs from manifest")
    bundle = load_bundle(out)
    if manifest["scope"] == "full_model":
        if name == "lm_head":
            q = bundle[name]
        else:
            match = re.fullmatch(r"layer(\d+)_(qkv|proj|ffwd1|ffwd2)", name)
            q = bundle["layers"][int(match[1])][match[2]]
    else:
        q = bundle[name]
    workspace = out / "sim_build" / name
    fixture = workspace / "fixture.npz"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    data = dict(w=q["w"], b=q.get("b", np.zeros(q["w"].shape[0], dtype=np.int64)))
    if "m0" in q:
        data.update(m0=q["m0"], shifts=q["n"], out_bits=q["out_bits"])
    np.savez(fixture, **data)
    suffix = name + "_" if args.module else ""
    report = out / f"verification_{suffix}{args.sim}.json"
    report.unlink(missing_ok=True)
    build = workspace / args.sim / "build"
    runner = get_runner(args.sim)
    runner.build(
        sources=[source],
        hdl_toplevel=name,
        build_dir=build,
        timescale=("1ns", "1ps"),
        build_args=[
            "-Wno-fatal",
            "--no-public-flat-rw",
            "--public-depth",
            "1",
            "--output-split",
            "10000",
        ]
        if args.sim == "verilator"
        else [],
    )
    runner.test(
        hdl_toplevel=name,
        test_module="test_fixed",
        build_dir=build,
        test_dir=ROOT / "tb",
        extra_env={
            "FIXTURE": str(fixture),
            "FIXED_TRIALS": "3",
            "FIXED_REPORT": str(report),
        },
    )
    if not report.exists():
        raise SystemExit("verification did not complete")
    print(report.read_text())


if __name__ == "__main__":
    main()
