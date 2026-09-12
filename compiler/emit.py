"""Emit fixed-coefficient datapaths. Selection happens after arithmetic."""

from pathlib import Path
import re

import numpy as np


def csd(value):
    """Signed powers of two; e.g. 7 = 8 - 1."""
    value = int(value)
    sign = -1 if value < 0 else 1
    value = abs(value)
    terms = []
    shift = 0
    while value:
        if value & 1:
            digit = 2 - (value & 3)
            terms.append((sign * digit, shift))
            value -= digit
        value >>= 1
        shift += 1
    return terms


def literal(value, width=32):
    value = int(value)
    return ("-" if value < 0 else "") + f"{width}'sd{abs(value)}"


def multiply_constant(signal, value, width=32):
    terms = []
    for sign, shift in csd(value):
        term = f"({signal} <<< {shift})" if shift else signal
        terms.append((" + " if sign > 0 else " - ") + term)
    return f"{width}'sd0" + "".join(terms)


def balanced_sum(terms):
    if not terms:
        return "32'sd0"
    while len(terms) > 1:
        terms = [
            f"({terms[i]} + {terms[i + 1]})" if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


def identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"invalid RTL identifier: {name!r}")
    return name


def emit_linear(
    path, name, weights, bias=None, m0=None, shifts=None, out_bits=32, in_bits=8
):
    """weights[out,in], signed INT8/INT16 input; ordered signed INT32 output stream.

    All output dot products exist spatially. A mux serialises their results.
    There is no coefficient address, read port, array or runtime multiplier.
    Repeated products of the same input and constant share a wire.
    """
    identifier(name)
    w = np.asarray(weights)
    if w.ndim != 2 or min(w.shape) < 1 or w.dtype.kind not in "iu":
        raise ValueError("weights must be a nonempty integer matrix")
    if w.min() < -128 or w.max() > 127:
        raise ValueError("weights must fit signed INT8")
    if in_bits not in (8, 16):
        raise ValueError("in_bits must be 8 or 16")
    n, k = w.shape
    b = (
        np.zeros(n, dtype=np.int64)
        if bias is None
        else np.asarray(bias, dtype=np.int64)
    )
    if b.shape != (n,):
        raise ValueError("bias shape differs from output size")
    if np.any(
        np.abs(b) + np.abs(w.astype(np.int64)).sum(axis=1) * (1 << (in_bits - 1))
        > 2**31 - 1
    ):
        raise ValueError("dot product may overflow INT32")
    if out_bits not in (8, 16, 32):
        raise ValueError("out_bits must be 8, 16 or 32")
    if (m0 is None) != (shifts is None):
        raise ValueError("requantisation needs both multipliers and shifts")
    if m0 is not None:
        m0, shifts = np.asarray(m0), np.asarray(shifts)
        if (
            m0.shape != (n,)
            or shifts.shape != (n,)
            or np.any(m0 < 0)
            or np.any(m0 >= 2**31)
            or np.any(shifts < 0)
            or np.any(shifts > 63)
        ):
            raise ValueError("invalid requantisation parameters")
    cw = max(1, (max(k, n) - 1).bit_length())
    lines = [
        f"// Fixed {n}x{k} linear map. Coefficients are gates, not stored words.",
        f"module {name} (input logic clk, rst_n, in_valid, input logic [{in_bits - 1}:0] in_data,",
        "    output logic out_valid, output logic [31:0] out_data, output logic busy);",
        f"    logic signed [{in_bits - 1}:0] x [0:{k - 1}];",
        f"    logic [{cw - 1}:0] count;",
        "    logic draining;",
        "    assign busy = draining || (count != 0);",
    ]
    products = {}
    for i in range(k):
        lines += [
            f"    wire signed [31:0] x{i} = {{{{{32 - in_bits}{{x[{i}][{in_bits - 1}]}}}}, x[{i}]}};"
        ]
        for value in sorted(set(int(v) for v in w[:, i]) - {0}):
            p = f"p{i}_{'n' if value < 0 else 'p'}{abs(value)}"
            products[i, value] = p
            lines += [
                f"    wire signed [31:0] {p} = {multiply_constant(f'x{i}', value)};"
            ]
    for j in range(n):
        terms = [products[i, int(v)] for i, v in enumerate(w[j]) if v]
        if b[j]:
            terms.append(literal(b[j]))
        lines += [f"    wire signed [31:0] a{j} = {balanced_sum(terms)};"]
        if m0 is not None:
            shift = int(shifts[j])
            half = (1 << (shift - 1)) if shift else 0
            lines += [
                f"    wire signed [63:0] e{j} = {{{{32{{a{j}[31]}}}}, a{j}}};",
                f"    wire signed [63:0] r{j} = ({multiply_constant(f'e{j}', int(m0[j]), 64)} + {literal(half, 64)}) >>> {shift};",
                f"    wire signed [31:0] y{j} = r{j} < {literal(-(1 << (out_bits - 1)), 64)} ? {literal(-(1 << (out_bits - 1)))} :",
                f"        r{j} > {literal((1 << (out_bits - 1)) - 1, 64)} ? {literal((1 << (out_bits - 1)) - 1)} : r{j}[31:0];",
            ]
        else:
            lines += [f"    wire signed [31:0] y{j} = a{j};"]
    # Bound mux fan-in for large vocabularies. These are computed results;
    # all coefficient arithmetic is upstream of the selector.
    bank_size = 256
    banks = (n + bank_size - 1) // bank_size
    lines += [f"    wire [31:0] bank_result [0:{banks - 1}];"]
    for bank in range(banks):
        length = min(bank_size, n - bank * bank_size)
        lines += [f"    wire [31:0] results_{bank} [0:{length - 1}];"]
        for offset in range(length):
            lines += [
                f"    assign results_{bank}[{offset}] = y{bank * bank_size + offset};"
            ]
        index = "(count & 32'd255)" if banks > 1 else "count"
        lines += [f"    assign bank_result[{bank}] = results_{bank}[{index}];"]
    selected = "bank_result[count >> 8]" if banks > 1 else "bank_result[0]"
    lines += [
        f"    wire [31:0] selected = {selected};",
        "    always_ff @(posedge clk) begin",
        "        if (!rst_n) begin",
        "            count <= 0; draining <= 0; out_valid <= 0; out_data <= 0;",
        f"            for (integer i=0; i<{k}; i=i+1) x[i] <= 0;",
        "        end else begin",
        "            out_valid <= 0;",
        "            if (draining) begin",
        "                out_valid <= 1; out_data <= selected;",
        f"                if (count == {n - 1}) begin count <= 0; draining <= 0; end",
        "                else count <= count + 1'b1;",
        "            end else if (in_valid) begin",
        "                x[count] <= in_data;",
        f"                if (count == {k - 1}) begin count <= 0; draining <= 1; end",
        "                else count <= count + 1'b1;",
        "            end",
        "        end",
        "    end",
        "endmodule",
        "",
    ]
    Path(path).write_text("\n".join(lines))
    return {
        "module": name,
        "inputs": k,
        "outputs": n,
        "coefficients": int(w.size),
        "nonzero": int(np.count_nonzero(w)),
        "constant_products": len(products),
        "csd_add_sub_terms": sum(len(csd(v)) for _, v in products),
        "weight_storage": "constant_logic",
        "output_bits": out_bits,
        "input_bits": in_bits,
    }
