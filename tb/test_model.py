"""All logits, greedy token selection, context bounds and clear/reset."""

import json
import os
from pathlib import Path
import sys
import time

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.environ["SOURCE_ROOT"])
from compiler.bundle import load_bundle
from compiler.reference import IntDecoder


async def tick(dut):
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def token(dut, value, vocab):
    assert dut.tok_ready.value
    dut.tok_in.value = value
    dut.tok_valid.value = 1
    await tick(dut)
    dut.tok_valid.value = 0
    logits = []
    for cycles in range(1, 10_000_000):
        await tick(dut)
        assert not dut.error.value
        if dut.logit_valid.value:
            logits.append(dut.logit_data.value.to_signed())
        if dut.done.value:
            assert len(logits) == vocab, (len(logits), vocab)
            return np.array(logits, dtype=np.int64), int(dut.argmax.value), cycles
    raise AssertionError("token timed out")


@cocotb.test()
async def pretrained_model(dut):
    out = Path(os.environ["MODEL_OUT"])
    q = load_bundle(out)
    ref = IntDecoder(q)
    prompt = json.loads(os.environ["VERIFY_TOKENS"])
    steps = int(os.environ["VERIFY_STEPS"])
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.tok_valid.value = 0
    dut.tok_in.value = 0
    for _ in range(3):
        await tick(dut)
    dut.rst_n.value = 1
    await tick(dut)
    tokens = []
    times = []
    next_token = prompt[0]
    start = time.monotonic()
    for position in range(steps):
        value = prompt[position] if position < len(prompt) else next_token
        got, next_token, cycles = await token(dut, value, q["V"])
        expected = ref.step(value)
        np.testing.assert_array_equal(
            got, expected, err_msg=f"position {position}, token {value}"
        )
        assert next_token == int(expected.argmax())
        assert int(dut.pos.value) == position + 1
        tokens.append(value)
        times.append(cycles)
        dut._log.info(
            "position %d: %d logits match, %d cycles, next token %d",
            position,
            q["V"],
            cycles,
            next_token,
        )
    if steps == q["T"]:
        dut.tok_valid.value = 1
        await tick(dut)
        dut.tok_valid.value = 0
        assert dut.error.value and not dut.busy.value and not dut.tok_ready.value
    # Clear must isolate the next sequence from the previous KV cache.
    dut.clear.value = 1
    await tick(dut)
    dut.clear.value = 0
    await tick(dut)
    ref.reset()
    got, _, _ = await token(dut, prompt[0], q["V"])
    np.testing.assert_array_equal(got, ref.step(prompt[0]))
    # Abort an in-flight token, then verify a complete fresh token.
    dut.tok_valid.value = 1
    dut.tok_in.value = prompt[0]
    await tick(dut)
    dut.tok_valid.value = 0
    for _ in range(7):
        await tick(dut)
    dut.clear.value = 1
    await tick(dut)
    dut.clear.value = 0
    await tick(dut)
    ref.reset()
    got, _, _ = await token(dut, prompt[-1], q["V"])
    np.testing.assert_array_equal(got, ref.step(prompt[-1]))
    # Reject a representable token ID outside the vocabulary.
    invalid = (1 << len(dut.tok_in)) - 1
    if invalid >= q["V"]:
        dut.tok_in.value = invalid
        dut.tok_valid.value = 1
        await tick(dut)
        dut.tok_valid.value = 0
        assert dut.error.value and not dut.busy.value
    report = dict(
        simulator=cocotb.SIM_NAME,
        positions=steps,
        logits_compared=steps * q["V"],
        tokens=tokens,
        next_token=next_token,
        cycles_per_token=times,
        clear_after_sequence=True,
        clear_during_token=True,
        invalid_token_checked=invalid >= q["V"],
        context_limit_checked=steps == q["T"],
        seconds=round(time.monotonic() - start, 3),
        passed=True,
    )
    Path(os.environ["VERIFY_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
