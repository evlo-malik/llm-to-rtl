# Circuit interface

`model_top` accepts one token ID and emits the next-token logits. Tokenisation
and choosing/sending the next token run on the host.

| Signal | Meaning |
|---|---|
| `clk` | Rising-edge clock |
| `rst_n` | Synchronous active-low reset |
| `clear` | Synchronous reset of the sequence and all compute units |
| `tok_ready` | A token can be accepted |
| `tok_valid`, `tok_in` | Present a token for one clock while ready |
| `busy` | A token is being processed; new tokens are ignored |
| `logit_valid`, `logit_data` | Signed INT32 logits, vocabulary order, one per valid cycle |
| `done`, `argmax` | One-cycle completion pulse and highest-scoring token ID |
| `pos` | Number of tokens processed in the current sequence |
| `error` | One-cycle pulse for an invalid token or exhausted context |

There is no output backpressure. Capture every valid logit. `clear` also aborts
an in-flight token. Clear before starting an unrelated prompt. Context length is
fixed at compile time; the cache does not slide when it fills.

## Data path

Token and position → fixed embedding decoder → transformer layers → final
normalisation → fixed vocabulary projection → logits.

Each layer applies normalisation, fixed Q/K/V projections, causal attention,
a fixed output projection and a residual add, followed by normalisation, fixed feed-forward projections with GELU (GPT-2) or
SwiGLU (rotary decoders), and another residual add.
Rotary decoders rotate queries and keys according to token position. Grouped
attention shares each stored KV head across several query heads. Attention stores
previous keys and values; these depend on the input tokens.

The compiler emits a separate constant circuit for every weight matrix. For
example, multiplying an input by a weight of 5 becomes `(input << 2) + input`.
Products can be shared when the same input has the same coefficient. The circuit
buffers its input vector, computes the outputs in parallel and serialises them.
The controller sequences these circuits; it never supplies weights to them.

Embeddings and nonlinear lookup tables select fixed values. Yosys maps these
read-only tables into logic. Activation buffers, the scratchpad and KV cache
remain writable memory. Removing weight traffic does not remove activation
traffic, wiring, or the area required to represent the model.

## Numbers and scope

Matrix weights use INT8, INT4 or ternary quantisation. The compact path uses INT8
linear inputs and INT16 residuals. The wide path uses INT16 normalised and
feed-forward values with INT32 residuals. Attention Q/K/V remain INT8; dot products
use INT32 with a checked overflow bound. LayerNorm/RMSNorm, GELU/SiLU and softmax use integer
approximations. Calibration selects scales without training the model.
RTL is checked exactly against the integer reference. Quantisation error against
the floating-point checkpoint is measured separately.

The output is synthesizable RTL, not a fabricated chip. An FPGA implements these
constants in configurable logic; an ASIC flow must map, place and route them
before they become permanent silicon. This repository does not establish that
a large model fits a particular device or meets timing or power targets.
