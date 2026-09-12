# Sanity checks on the golden models, run before trusting anything downstream:
#   1. FloatGPT (LayerNorm folded) reproduces the trained torch model's logits.
#   2. IntGPT agrees with the float model most of the time on held-out text and
#      its loss is close. Prints the numbers; they go in docs/report.md.
#   python3 compiler/check_golden.py model/tinygpt [--bits 8]
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_gpt import load_karpathy, fold_karpathy, FloatGPT, IntGPT, calibrate, quantise_model  # noqa: E402


def torch_logits(model_dir, tokens):
    sys.path.insert(0, os.path.join(os.path.dirname(model_dir.rstrip("/"))))
    os.environ["TINYGPT_NO_TRAIN"] = "1"
    import importlib
    train = importlib.import_module("train")
    m = train.GPTLanguageModel()
    m.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    m.eval()
    with torch.no_grad():
        lg, _ = m(torch.tensor([tokens]))
    return lg[0].double().numpy()


def windows(text_tokens, T, n, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(text_tokens) - T - 1, size=n)
    return [text_tokens[s:s + T] for s in starts], [text_tokens[s + 1:s + T + 1] for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=64)
    a = ap.parse_args()

    cfg, st = load_karpathy(a.model_dir)
    f = fold_karpathy(cfg, st)
    chars = json.load(open(os.path.join(a.model_dir, "tokenizer.json")))["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    text = open(os.path.join(os.path.dirname(a.model_dir.rstrip("/")), "data", "input.txt")).read()
    toks = np.array([stoi[c] for c in text])
    n = int(0.9 * len(toks))
    train_toks, val_toks = toks[:n], toks[n:]
    T = cfg["block_size"]

    # 1. folding
    x = train_toks[:T]
    ref = torch_logits(a.model_dir, list(x))
    got = FloatGPT(f).forward(x)
    err = np.abs(ref - got).max()
    print(f"fold check: max |logit diff| vs torch = {err:.2e}  ({'OK' if err < 1e-3 else 'BAD'})")

    # 2. quantise on training windows, evaluate on validation windows
    cal, _ = windows(train_toks, T, a.calib, seed=1)
    stats = calibrate(f, cal)
    print("calibration absmax:", {k: round(v, 3) for k, v in sorted(stats.items())})
    q = quantise_model(f, stats, a.bits)
    ev, tg = windows(val_toks, T, a.eval, seed=2)
    fl = FloatGPT(f)
    it = IntGPT(q)
    agree = tot = 0
    ce_f = ce_i = 0.0
    for xw, yw in zip(ev, tg):
        lf = fl.forward(xw)
        li = it.forward(xw) * q["lm_head"]["s_out"][None, :]
        agree += int((lf.argmax(-1) == li.argmax(-1)).sum())
        tot += len(xw)
        def ce(lg):
            lg = lg - lg.max(-1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
            return -lp[np.arange(len(yw)), yw].sum()
        ce_f += ce(lf)
        ce_i += ce(li)
    print(f"int{a.bits} vs float on {tot} validation positions: argmax agreement {agree/tot*100:.1f}%, "
          f"cross-entropy float {ce_f/tot:.4f}  int {ce_i/tot:.4f}")
    json.dump(dict(bits=a.bits, fold_err=err, positions=tot, agreement=agree / tot, ce_float=ce_f / tot, ce_int=ce_i / tot,
                   stats=stats), open(os.path.join(a.model_dir, f"golden_check_int{a.bits}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
