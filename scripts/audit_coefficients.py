#!/usr/bin/env python3
"""Reconstruct emitted linear coefficients and compare every row with the quantised model."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.bundle import load_bundle

PRODUCT = re.compile(r"wire signed \[31:0\] (p\d+_[np]\d+) = (.*);")
ROW = re.compile(r"wire signed \[31:0\] a(\d+) = (.*);")
TERM = re.compile(r"p\d+_[np]\d+|-?32'sd\d+")
SHIFT = re.compile(r"([+-])(?:\(x(\d+)<<<(\d+)\)|x(\d+))")


def validate_addition(expression):
    depth = 0
    operand = True
    for char in expression:
        if char == "(" and operand:
            depth += 1
        elif char == "t" and operand:
            operand = False
        elif char == "+" and not operand:
            operand = True
        elif char == ")" and not operand and depth:
            depth -= 1
        else:
            raise ValueError("unsupported or malformed row expression")
    if depth or operand:
        raise ValueError("incomplete row expression")


def audit_linear(path, matrix):
    products = {}
    rows = set()
    count = 0
    w = matrix["w"]
    bias = matrix.get("b", np.zeros(len(w), dtype=np.int64))
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            match = PRODUCT.fullmatch(line)
            if match:
                name, expression = match.groups()
                expression = expression.replace(" ", "")
                if not expression.startswith("32'sd0"):
                    raise ValueError("invalid constant product")
                tail = expression[len("32'sd0") :]
                terms = list(SHIFT.finditer(tail))
                if "".join(x[0] for x in terms) != tail:
                    raise ValueError("unsupported product arithmetic")
                columns = {int(x[2] if x[2] is not None else x[4]) for x in terms}
                if len(columns) != 1:
                    raise ValueError("product does not use exactly one input")
                coefficient = sum(
                    (1 if x[1] == "+" else -1) * (1 << int(x[3] or 0)) for x in terms
                )
                if name in products:
                    raise ValueError("duplicate product")
                products[name] = (columns.pop(), coefficient)
            match = ROW.fullmatch(line)
            if match:
                row, expression = int(match[1]), match[2].replace(" ", "")
                if row in rows or row >= len(w):
                    raise ValueError("duplicate or invalid row")
                tokens = TERM.findall(expression)
                validate_addition(TERM.sub("t", expression))
                columns = []
                values = []
                constant = 0
                for token in tokens:
                    if token.startswith("p"):
                        column, value = products[token]
                        columns.append(column)
                        values.append(value)
                    else:
                        constant += int(token.replace("32'sd", ""))
                if len(columns) != len(set(columns)):
                    raise ValueError("input repeated in a dot product")
                actual = np.zeros(w.shape[1], dtype=np.int64)
                actual[columns] = values
                if not np.array_equal(actual, w[row]) or constant != bias[row]:
                    raise ValueError(
                        f"{path.name}: row {row} differs from quantised checkpoint"
                    )
                rows.add(row)
                count += len(actual)
    if rows != set(range(len(w))):
        raise ValueError("missing output rows")
    return dict(rows=len(rows), coefficients=count, products=len(products), passed=True)


def decode_literal(expression):
    pieces = re.findall(r"(\d+)'h([0-9a-f]+)", expression)
    remainder = re.sub(r"\d+'h[0-9a-f]+", "", expression)
    if remainder.strip("{} ,\n\t") or not pieces:
        raise ValueError("invalid embedding literal")
    word = width = 0
    for size, value in pieces:
        size = int(size)
        value = int(value, 16)
        if value.bit_length() > size:
            raise ValueError("oversized literal")
        word = (word << size) | value
        width += size
    return word.to_bytes((width + 7) // 8, "little"), width


def audit_embedding(path, q):
    found = {"token_value": set(), "position_value": set()}
    values = {"token_value": q["tok_emb"], "position_value": q["pos_emb"]}
    bits = q.get("residual_bits", 16)
    with path.open() as stream:
        for line in stream:
            match = re.match(
                r"\d+'d(\d+): (token_value|position_value)=(.*)", line.strip()
            )
            if not match:
                continue
            index, name, expression = int(match[1]), match[2], match[3]
            while ";" not in expression:
                expression += next(stream)
            expression = expression.split(";")[0]
            actual, width = decode_literal(expression)
            if index in found[name] or index >= len(values[name]):
                raise ValueError("invalid embedding row")
            expected = np.asarray(
                values[name][index], dtype="<i4" if bits == 32 else "<i2"
            ).tobytes()
            if width != q["D"] * bits or actual != expected:
                raise ValueError("embedding differs from quantised checkpoint")
            found[name].add(index)
    if any(found[name] != set(range(len(value))) for name, value in values.items()):
        raise ValueError("missing embedding rows")
    return dict(
        token_rows=len(found["token_value"]),
        position_rows=len(found["position_value"]),
        passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    q = load_bundle(out)
    report = dict(
        scope="emitted coefficient arithmetic, not a full simulation",
        passed=True,
        matrices={},
    )
    for entry in manifest["matrix_modules"]:
        name = entry["module"]
        if manifest["scope"] == "full_model":
            if name == "lm_head":
                matrix = q[name]
            else:
                match = re.fullmatch(r"layer(\d+)_(qkv|proj|ffwd1|ffwd2)", name)
                matrix = q["layers"][int(match[1])][match[2]]
        else:
            matrix = q[name]
        path = out / "rtl" / f"{name}.sv"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["rtl_sha256"][path.name]:
            raise ValueError("RTL hash differs from manifest")
        report["matrices"][name] = audit_linear(path, matrix)
        print(f"{name}: {matrix['w'].size:,} coefficients checked", flush=True)
    if manifest["scope"] == "full_model":
        report["embeddings"] = audit_embedding(out / "rtl/fixed_embedding.sv", q)
    report["matrix_coefficients"] = sum(
        x["coefficients"] for x in report["matrices"].values()
    )
    report["rtl_sha256"] = manifest["rtl_sha256"]
    (out / "coefficient_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
