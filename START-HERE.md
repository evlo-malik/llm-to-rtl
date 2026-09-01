# START HERE — day 1

Toolchain installed and verified end to end on 2026-09-01. You write the RTL. Nothing in
`rtl/` was written for you, and nothing should be: the whole value of this project is that
you can defend every line.

## Every session begins with these two commands

```bash
cd ~/Desktop/ME/iron-man/Projects/rtl-systolic-array
source env.sh
```

`env.sh` is not optional. It puts arm64 Python and Homebrew ahead of Anaconda on your PATH.
Without it Verilator dies with `SRE module mismatch` and cocotb links Intel libraries into an
arm64 simulator. Both were hit and fixed during install; `env.sh` is the fix.

Health check any time something feels broken:

```bash
bash doctor.sh
```

## What is already proven working

- Verilator 5.050 + cocotb 2.1.0 — running a real testbench, VCD written
- Icarus Verilog 13.0 — same testbench, independent cross-check, both pass
- Yosys 0.68 — synthesised to sky130 standard cells, real area in µm²
- Surfer — waveform viewer, native arm64

The proof lives in `_smoke/`. It is a throwaway 8-bit counter. **Delete `_smoke/` once `pe.sv`
passes its own tests** — it exists only so that when your first real test fails, you know the
failure is in your design and not in the toolchain.

## Today, in order

### 1. Read EXPLAINER §7 with a pen (60–90 min)

`../RTL Systolic Array Accelerator/EXPLAINER.md`, section 7. Hand-copy the 2×2 cycle-by-cycle
table onto paper. Do not skim it. On day 3 you will be debugging a waveform against that table,
and if you have not internalised it now you will be guessing then. Guessing is how ten days
becomes twenty.

### 2. Write the spec before the RTL (60 min) → `docs/spec.md`

Half a page. It must answer, in writing:

- Every port on `pe.sv` and on `top.sv`: name, width, direction, signed or unsigned.
- Reset: synchronous or asynchronous, active high or low. Pick one, write it down, never deviate.
- What `valid` means precisely. Which cycle is data sampled?
- Numerics: INT8 in, INT32 accumulate. What happens on overflow — wrap or saturate? **Decide now.**
  This is a question an interviewer will ask, and "I never thought about it" is the wrong answer.
- The timing identity: at what cycle does the last output appear? EXPLAINER gives t_last = 2N+K−3.
  Write out why, in your own words.

You will be tempted to skip this because it is not typing code. Skipping it is the single most
expensive thing you can do this week.

### 3. Write `rtl/pe.sv` (60 min)

One processing element. Roughly fifteen lines. Registered `a` out, registered `b` out,
accumulator in place. Output-stationary.

The trap, and it is worth stating plainly: **`logic [7:0]` is unsigned in SystemVerilog.**
If you multiply two of those you get the wrong answer for negative operands and the design will
still look fine on positive test vectors. Use `logic signed [7:0]`, and make the accumulator
`logic signed [31:0]`. This is EXPLAINER §8. It is the bug that eats a day if it reaches the
4×4 array, and fifteen minutes if it is caught in the PE.

### 4. Write `tb/test_pe.py` (60–90 min)

Copy `_smoke/test_counter.py` as your structural template — the runner boilerplate at the bottom
is correct and non-obvious, particularly `waves=True` and `timescale=("1ns","1ps")` (Icarus
refuses to run without the latter).

The test must include, before you write a single random case:

- accumulate zeros → 0
- one positive MAC by hand, checked against a number you computed yourself
- **all operands `-128`** — this is the case that catches the signedness bug
- `127 × 127` accumulated repeatedly, to prove INT32 headroom is real

Then randomised: 1000 sequences of signed MACs, checked against NumPy.

### Gate for today

`test_pe.py` passes 1000 random signed sequences, on **both** simulators:

```bash
SIM=verilator python3 tb/test_pe.py
SIM=icarus    python3 tb/test_pe.py
```

Two simulators disagreeing is a gift — it means one of them found a bug the other missed.
Do not move to the array until both are green.

### Then commit

```bash
git add -A && git commit -m "PE: signed INT8 MAC, verified against NumPy on 1000 random cases"
```

The commit history is part of the deliverable. Ten days of daily commits reads very differently
from one commit on day ten.

## Waveforms

```bash
surfer sim_build/dump.vcd
```

## Working rules for this project

- **Do not let me write your RTL.** Ask me to explain a concept, review your code, review your
  testbench, hunt a bug with you, or tell you your parameterisation is fake. If I write `pe.sv`
  you cannot defend it in an interview and the project is worth nothing.
- **A failing test is progress. A skipped test is debt.**
- If a tool resists for more than an hour, say so and we drop to the fallback. The deliverable
  is a verified design, not a perfect toolchain.
