# Lessons — Part 1, the systolic array

Each lesson ends with a gate. Do not start the next lesson until the gate is green.
Every session: `cd ~/Desktop/ME/iron-man/Projects/rtl-systolic-array && source env.sh`

| # | Lesson | You build | You learn | Gate |
|---|---|---|---|---|
| 1 | The loop | `rtl/mul.sv`, `tb/test_mul.py` | module, ports, signed, clock, reset, `<=`, cocotb, Surfer | 200 random signed products match Python, both simulators |
| 2 | The PE | `rtl/pe.sv`, `tb/test_pe.py` | weight register, load vs compute, partial sums, directed edge cases | 1000 random MACs match NumPy |
| 3 | The grid | `rtl/array.sv` (2×2) | `generate`, wiring boxes, tracing a waveform against paper | 2×2 matmul matches NumPy on hand-picked matrices |
| 4 | Skew + control | `rtl/skew.sv`, `rtl/top.sv` | shift-register delay lines, load→compute→drain FSM, timing formula | 4×4 matmul, any matrices |
| 5 | Verification | `tb/golden.py`, `tb/test_top.py` | randomised testing, N=4/8/16, assertions, both simulators, coverage argument | 300 random matrices per N, all green |
| 6 | Synthesis | `syn/synth.ys`, `syn/sta.tcl`, `syn/sweep.py` | Yosys→sky130, area, OpenSTA critical path, Fmax, sweep N, plot | area/Fmax vs N plot with a written explanation |
| 7 | Ship | `README.md`, `docs/report.md` | writing claims that are numbers | clone → one command → green |

Reading, on demand only: DDCA lecture 18 before lesson 3. Nothing before lesson 1.
