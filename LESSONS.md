# Lessons

The order this was built in. Each step has a gate; nothing after it started before the
gate was green. Every session: `source env.sh`.

## Part 1, the array

| # | Built | Gate | Result |
|---|---|---|---|
| 1 | `rtl/mul.sv`, `tb/test_mul.py` | 200 random signed products, both simulators | pass |
| 2 | `rtl/pe.sv`, `tb/test_pe.py` | 1000 random MACs vs torch, weight stationary, -128 x -128 | pass |
| 3 | `rtl/array.sv`, `tb/test_array.py` | grid + testbench-side skew, outputs read on the cycle the formula says | pass, N = 2, 4 |
| 4 | `rtl/skew.sv`, `rtl/top.sv`, `tb/test_top.py` | random matrices streamed back to back, gaps, reloads; latency 2N-1 on every vector | pass, N = 4, 8, 16, Verilator and Icarus |
| 5 | `syn/sweep.py` | area and reg-to-reg Fmax vs N on sky130, with an explanation | `docs/area_fmax_vs_N.png`, `docs/report.md` |

What the array taught: the load order (last row first) falls out of the column shift
register; the whole timing story is one formula, `c + N + j` for column j, and the
testbench can check it on every cycle; a lone PE synthesises bigger than a PE inside a
flattened array because the top rows never need 32 accumulator bits.

## Part 2, the compiler

| # | Built | Gate | Result |
|---|---|---|---|
| 6 | `rtl/matvec_rom.sv`, `tb/test_matvec_rom.py`, `compiler/emit.py` | random K, N, T incl. padding, bias, 8/4/2-bit ROMs, bit-exact vs torch | pass, 7 shapes |
| 7 | `compiler/compile.py` (adapters, quantiser, tests, runner) | every emitted module of the tiny GPT passes | 9 / 9 |
| 8 | same, SmolLM2-135M | every emitted module passes, lm_head included | 211 / 211 |
| 9 | `syn/bitwidth.py` | area and LUTs per weight, hardwired 8 / 4 / ternary vs generic | `docs/area_per_weight.png` |

What the compiler taught: the weight-stationary array does useful work one cycle in
3T when it gets one vector per tile; the fix is a second weight register per PE, not a
bigger array. A hardwired weight is not free even when it is zero, because the systolic
registers are still there.

## Part 3, the model

| # | Built | Gate | Result |
|---|---|---|---|
| 10 | `model/train.py` | Karpathy's model, 110k params, saved as safetensors | val loss 1.71 |
| 11 | `compiler/golden_gpt.py`, `check_golden.py` | folded float model = torch to 1e-5; int8 model within 0.01 nats | 1.2e-5; +0.004 nats, 95% argmax agreement |
| 12 | `requant.sv`, `layernorm.sv` (+ `isqrt`, `udiv`), `attention.sv`, `embed.sv` | each bit-exact vs `quant.py` on random and edge inputs | pass, both simulators for requant and layernorm |
| 13 | `sequencer.sv`, generated `gpt_top.sv` | every logit of 32 positions bit-exact vs `IntGPT`; greedy text | pass at 8, 4 and 2 bits; 51k cycles per token |
| 14 | `rtl/board/`, `syn/gowin/` | UART prompt in, text out, in simulation; Gowin synth + PnR numbers | see `docs/report.md` |
| 15 | flash a Tang Nano 20K, film it | text on a terminal from the board, tokens/s and watts | **not done, needs the board** |

What the model taught: every narrowing step is a design decision (where the residual
stream saturates, where the softmax index clips, why lm_head needs one weight scale),
and the golden model has to make the same decision in the same order or nothing is
bit-exact. The test that found the most bugs was the one that compared the second
vector, not the first: units that returned to idle a cycle late dropped an input.

Reading, on demand only: DDCA L18 before lesson 3; ACA L2-L6 when the 3T-cycle tile
number makes you ask why the array starves.
