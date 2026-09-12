#!/usr/bin/env python3
"""Compare one emitted fixed linear map with its integer checkpoint weights."""

import argparse
import hashlib
import json
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
    parser.add_argument("--sim", choices=("icarus", "verilator"), default="verilator")
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    modules = manifest["matrix_modules"]
    if manifest["scope"] != "matrices_only" or len(modules) != 1:
        parser.error("compile one matrix with --matrices-only --only NAME")
    name = modules[0]["module"]
    source = out / "rtl" / f"{name}.sv"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != manifest["rtl_sha256"][source.name]
    ):
        parser.error("generated RTL differs from manifest")
    q = load_bundle(out)[name]
    fixture = out / "sim_build" / "fixture.npz"
    fixture.parent.mkdir(exist_ok=True)
    np.savez(fixture, w=q["w"], b=np.zeros(q["w"].shape[0], dtype=np.int64))
    report = out / f"verification_{args.sim}.json"
    report.unlink(missing_ok=True)
    build = out / "sim_build" / args.sim / "build"
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
