# Spec

Interfaces and numerics for every block. If the RTL and this file disagree, one of them
is wrong; the testbenches decide which.

## Conventions

- Clock `clk`, rising edge. Reset `rst_n`, synchronous, active low, everywhere.
- Signed two's complement throughout. `logic [7:0]` ports carry int8; the sign is
  applied where the value is used (`signed'(...)`), never assumed from the port type.
- Packed vectors: element i is bits `[W*i +: W]`.
- INT8 x INT8 products accumulate in INT32. Overflow is impossible in the arrays
  (`N * 127 * 128 < 2^31` for any N under 131072); everything that narrows a value
  saturates, and every narrowing step is listed below.

## pe.sv (Malik's)

| port | dir | width | meaning |
|---|---|---|---|
| load_w | in | 1 | capture `w_in` into the weight register on this edge |
| w_in / w_out | in / out | 8 | weight shifts down the column; `w_out` is the register, combinational |
| x_in / x_out | in / out | 8 | activation flows right, one register per PE |
| psum_in / psum_out | in / out | 32 | partial sum flows down, one register per PE |

`psum_out <= psum_in + weight * x_in` every edge, whether or not the data is valid.

## array.sv (Malik's)

N x N grid of `pe`. `x_in[i]` feeds row i, `w_in[j]` feeds column j from the top,
`result[j]` is `psum_out` of the bottom PE of column j. Weights load as a column shift
register: `load_w` high for N edges with rows presented **last row first**.

## top.sv

| port | dir | width | meaning |
|---|---|---|---|
| w_load, w_row | in | 1, 8N | one row of W per cycle for N cycles, W[N-1] first |
| x_valid, x_vec | in | 1, 8N | one activation vector |
| y_valid, y_vec | out | 1, 32N | `x_vec @ W`, exactly **2N-1** cycles after `x_valid` |
| busy | out | 1 | a vector is in flight |

Row i is delayed i cycles before the array, column j is delayed N-1-j cycles after it.
Vectors may be presented every cycle. `w_load` must not be raised while `busy` or in
the same cycle as `x_valid`. Nothing else is required of the caller; garbage on `x_vec`
while `x_valid` is low is harmless because `y_valid` only follows `x_valid`.

Why 2N-1: row i reaches PE(i,j) at cycle c+i+j, the bottom PE of column j registers
its sum at c+N+j, and the deskew adds N-1-j.

## matvec_rom.sv, and the generated mv_* modules

Serial interface shared by every matrix module and by the hardwired variant:

| port | dir | width | meaning |
|---|---|---|---|
| in_valid, in_data | in | 1, 8 | the K int8 inputs, in order, gaps allowed |
| out_valid, out_data | out | 1, 32 | the N int32 outputs, in order, `x @ W + b` |
| busy | out | 1 | do not send the next vector while high |

ROM word `(nt*KT + kt)*T + r` holds row r of tile (kt, nt): `W[kt*T + r][nt*T + c]` at
bits `[WBITS*c +: WBITS]`, zero past K and N. WBITS in {8, 4, 2}; a 2-bit weight is
ternary two's complement (`01` = +1, `11` = -1). Bias ROM word nt holds the T int32
biases of column tile nt. Per tile: T load cycles, one drain cycle, one fire cycle,
2T-1 wait cycles.

Cycles per vector: `NT * (KT * (3T + 2) + T)` plus K for the input. Measured:

| matrix | K x N | T | cycles | MACs/cycle |
|---|---|---|---|---|
| tiny GPT qkv | 64 x 192 | 8 | 4991 | 2.5 of 64 |
| tiny GPT ffwd1 | 64 x 256 | 8 | 6655 | 2.5 of 64 |
| SmolLM2 q_proj | 576 x 576 | 16 | 64k | 5.1 of 256 |
| SmolLM2 lm_head | 576 x 49152 | 16 | 5.47M | 5.2 of 256 |

## requant.sv

`y = clip( (acc * M0 + 2^(n-1)) >>> n )`, then ReLU if asked, `M0` 16-bit in
`[2^15, 2^16)`, `n` 0..63, per column from a ROM at `base + column`. `in_first` marks
column 0. Clip to int8 or int16 (`wide`). Three cycles of latency, one element per
cycle, output sign-extended to 32 bits. `compiler/quant.py: requant` is the reference.

## layernorm.sv

D int16 in, D int8 out at scale 2^-4, no affine (folded into the next Linear):

    mean = sum >> log2(D)          c = h - mean          var = (sum c^2) >> log2(D)
    std  = isqrt(var), min 1       inv = 2^24 / std       y = clip8((c * inv + 2^19) >> 20)

Shifts are arithmetic (floor). `isqrt` is the bit-serial restoring root (16 cycles),
`udiv` the restoring divider (25 cycles). No epsilon: at the residual scale it would be
below one unit. D must be a power of two. Latency about D + 50 cycles after the last
input; `busy` says when the next vector may start.

## attention.sv

Inputs per token: q, k, v (3D int8). k and v go to the cache at `[layer][pos]`. Per head:

    s[i]  = q_h . k_h[i]                              i = 0..pos, int32
    u[i]  = clip255( ((max s - s[i]) * M0u + 2^(nu-1)) >> nu )      one unit = 1/16 nat
    e[i]  = LUT[u[i]]           LUT[u] = round(2^16 * exp(-u/16)), 17 bits
    R     = 2^36 / sum e        restoring divider, 37 cycles
    p[i]  = (e[i] * R) >> 21    uint16, 2^-15 = 1.0
    o_h[c] = clip8( (sum_i p[i] * v_h[i][c]) * M0o + 2^(no-1) >> no )

The 1/sqrt(head_dim) and both activation scales live in `M0u`. Causal masking is the
loop bound. One multiply per cycle; measured about 1250 cycles per head at pos 31.

## sequencer.sv and gpt_top.sv

The generated top runs one program per token from `prog.memh`; see
`gen/tinygpt/program.txt` for the listing and the memory map. Ops move whole vectors
between a 32-bit scratchpad and a unit; `ADD16` (saturating residual add) and `OUT`
(logit stream and argmax, first index wins ties) run in the sequencer itself.

Numerics of the whole model, in order, for one token: embeddings are int16 at the
residual scale `s_h` (calibrated absmax / 32767); LayerNorm out is int8 at 1/16;
q, k, v are int8 at per-layer per-tensor scales; attention probabilities are uint16;
o is int8; the projection and ffwd2 outputs are requantised straight to int16 at `s_h`
and added with saturation; ffwd1 out is int8 after ReLU; logits are the raw int32
accumulator of lm_head, which uses one weight scale so argmax is meaningful.
`compiler/golden_gpt.py: IntGPT` is this list as code, and `gpt_top.sv` must match it
on every logit of every position.
