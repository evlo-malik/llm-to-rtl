#!/usr/bin/env python3
"""Simulate a compiled model and compare every logit with its integer reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sim", choices=("icarus", "verilator"), default="verilator")
    parser.add_argument(
        "--tokens", default="15496,995", help="comma-separated prompt token IDs"
    )
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["scope"] != "full_model":
        parser.error("output is not a complete model")
    tokens = [int(t) for t in args.tokens.split(",")]
    steps = args.steps or manifest["context"]
    if (
        not tokens
        or not len(tokens) <= steps <= manifest["context"]
        or any(not 0 <= t < manifest["vocab"] for t in tokens)
    ):
        parser.error("invalid token IDs, prompt length or step count")
    for name, digest in manifest["rtl_sha256"].items():
        if hashlib.sha256((out / "rtl" / name).read_bytes()).hexdigest() != digest:
            parser.error(
                f"generated RTL changed: {name}; recompile to refresh the reference"
            )
    build = out / "sim_build" / args.sim / "build"
    report = out / f"verification_{args.sim}.json"
    report.unlink(missing_ok=True)
    if args.sim == "verilator" and sys.platform == "darwin":
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != resource.RLIM_INFINITY and soft < 32 * 1024 * 1024:
            # macOS requires raising the stack before exec, not inside Python.
            os.execv(
                "/bin/bash",
                [
                    "/bin/bash",
                    "-c",
                    'ulimit -s 32768 && exec "$@"',
                    "verify",
                    sys.executable,
                    *sys.argv,
                ],
            )
    runner = get_runner(args.sim)
    runner.build(
        sources=sorted((out / "rtl").glob("*.sv")),
        hdl_toplevel="model_top",
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
        hdl_toplevel="model_top",
        test_module="test_model",
        build_dir=build,
        test_dir=ROOT / "tb",
        extra_env={
            "SOURCE_ROOT": str(ROOT),
            "MODEL_OUT": str(out),
            "VERIFY_TOKENS": json.dumps(tokens),
            "VERIFY_STEPS": str(steps),
            "VERIFY_REPORT": str(report),
        },
    )
    if not report.exists() or not json.loads(report.read_text())["passed"]:
        raise SystemExit("verification did not complete")
    print(report.read_text())


if __name__ == "__main__":
    main()
