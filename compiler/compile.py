#!/usr/bin/env python3
# safetensors + config.json -> one Verilog module per weight matrix, weights baked in,
# plus a cocotb test per module. For the Karpathy GPT it also writes the quantised
# model bundle that gpt_top.sv and the integer golden model are built from.
#
#   python3 compiler/compile.py model/tinygpt      --bits 8 --out gen/tinygpt --tile 8
#   python3 compiler/compile.py models/smollm2-135m --bits 8 --out gen/smollm2 --tile 16
#   python3 gen/tinygpt/run_tests.py -j 6
#
# A matrix with at most --const-max weights becomes a hardwired array (every weight a
# CSD shift-add in its own cell); anything bigger becomes a ROM beside one TxT array.
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from emit import write_rom, write_bias, write_memh, rom_wrapper_sv, const_array_sv, const_wrapper_sv   # noqa: E402
from quant import quant_weight                                                            # noqa: E402


def load_tensors(model_dir):
    """bf16 checkpoints need torch to decode; everything comes back float32 numpy."""
    from safetensors.torch import load_file
    st = load_file(str(Path(model_dir) / "model.safetensors"))
    return {k: v.float().numpy() for k, v in st.items()}


# ------------------------------------------------------------------ adapters
# Each returns a list of dicts: name, w [K, N] int (x @ w), b int32 [N] or None,
# plus what it came from. The hardware convention is x @ W, so torch's [out, in]
# gets transposed here and nowhere else.

def matrices_llama(cfg, st, bits):
    L = cfg["num_hidden_layers"]
    out = []

    def add(name, w_out_in, gamma=None, per_channel=True):
        w = w_out_in.astype(np.float64)
        if gamma is not None:                       # RMSNorm weight folded into the consumer
            w = w * gamma[None, :].astype(np.float64)
        wq, sw = quant_weight(w, bits, per_channel=per_channel)
        out.append(dict(name=name, w=wq.T.copy(), b=None, sw=sw, src=name))

    for l in range(L):
        p = f"model.layers.{l}."
        g_in = st[p + "input_layernorm.weight"]
        g_post = st[p + "post_attention_layernorm.weight"]
        for m in ("q_proj", "k_proj", "v_proj"):
            add(f"layer{l}_{m}", st[p + f"self_attn.{m}.weight"], g_in)
        add(f"layer{l}_o_proj", st[p + "self_attn.o_proj.weight"])
        for m in ("gate_proj", "up_proj"):
            add(f"layer{l}_{m}", st[p + f"mlp.{m}.weight"], g_post)
        add(f"layer{l}_down_proj", st[p + "mlp.down_proj.weight"])
    lm = st["lm_head.weight"] if "lm_head.weight" in st else st["model.embed_tokens.weight"]   # tied
    add("lm_head", lm, st["model.norm.weight"], per_channel=False)
    return out, None


def matrices_karpathy(cfg, st, bits, model_dir, calib_windows=64):
    from golden_gpt import fold_karpathy, calibrate, quantise_model
    f = fold_karpathy(cfg, st)
    chars = json.load(open(Path(model_dir) / "tokenizer.json"))["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    text = (Path(model_dir).parent / "data" / "input.txt").read_text()
    toks = np.array([stoi[c] for c in text])[: int(0.9 * len(text))]
    rng = np.random.default_rng(1)
    T = cfg["block_size"]
    windows = [toks[s:s + T] for s in rng.integers(0, len(toks) - T - 1, size=calib_windows)]
    stats = calibrate(f, windows)
    q = quantise_model(f, stats, bits)
    q["stats"] = stats
    out = []
    for l, lay in enumerate(q["layers"]):
        for m in ("qkv", "proj", "ffwd1", "ffwd2"):
            mv = lay[m]
            # the bias is applied in the requant stage, so the module is the plain matmul
            out.append(dict(name=f"layer{l}_{m}", w=mv["w"].T.copy(), b=None, sw=mv["sw"], src=f"blocks.{l}.{m}"))
    mv = q["lm_head"]
    out.append(dict(name="lm_head", w=mv["w"].T.copy(), b=None, sw=mv["sw"], src="lm_head"))
    return out, q


# ------------------------------------------------------------------ emission

def emit_matrix(m, out_dir, tile, bits, const_max):
    k, n = m["w"].shape
    name = "mv_" + m["name"]
    rtl = out_dir / "rtl"
    mem = out_dir / "mem"
    np.save(mem / f"{m['name']}.w.npy", m["w"])
    if m["b"] is not None:
        np.save(mem / f"{m['name']}.b.npy", m["b"])
    if k * n <= const_max:
        core = "const_" + m["name"]
        (rtl / f"{core}.sv").write_text(const_array_sv(core, m["w"]))
        (rtl / f"{name}.sv").write_text(const_wrapper_sv(name, core, k, n, m["b"]))
        kind, files = "const", [f"{core}.sv", f"{name}.sv"]
    else:
        wf = mem / f"{m['name']}.w.memh"
        write_rom(wf, m["w"], tile, bits)
        bf = None
        if m["b"] is not None:
            bf = mem / f"{m['name']}.b.memh"
            write_bias(bf, m["b"], n, tile)
        (rtl / f"{name}.sv").write_text(rom_wrapper_sv(name, k, n, tile, bits, wf.resolve(), bf.resolve() if bf else None))
        kind, files = "rom", [f"{name}.sv"]
    return dict(name=m["name"], module=name, kind=kind, K=k, N=n, weights=k * n, nonzero=int((m["w"] != 0).sum()),
                bias=m["b"] is not None, tile=tile if kind == "rom" else None, bits=bits, rtl=files, src=m["src"],
                sw_min=float(np.min(m["sw"])), sw_max=float(np.max(m["sw"])))


TEST_TEMPLATE = '''# generated by compiler/compile.py for {module}: {K}x{N}, {bits}-bit, {kind}
import os
import sys
from pathlib import Path
import numpy as np
import cocotb
from cocotb_tools.runner import get_runner

REPO = Path("{repo}")
OUT = Path("{out}")
sys.path.insert(0, str(REPO / "tb"))
from mvtest import check_module   # noqa: E402


@cocotb.test()
async def matches_torch(dut):
    w = np.load(OUT / "mem" / "{name}.w.npy")
    b = np.load(OUT / "mem" / "{name}.b.npy") if {has_bias} else None
    await check_module(dut, w, b, trials={trials}, log=dut._log.info)


def run():
    build = OUT / "sim_build" / "{name}"
    rtl = REPO / "rtl"
    srcs = [rtl / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv", "matvec_rom.sv")]
    srcs += [OUT / "rtl" / f for f in {rtl_files}]
    runner = get_runner(os.getenv("SIM", "verilator"))
    waves = bool(int(os.getenv("WAVES", "0")))
    runner.build(sources=srcs, hdl_toplevel="{module}", waves=waves, always=True,
                 timescale=("1ns", "1ps"), build_dir=build)
    runner.test(hdl_toplevel="{module}", test_module="test_{module}", waves=waves, build_dir=build)


if __name__ == "__main__":
    run()
'''

RUNNER_TEMPLATE = '''#!/usr/bin/env python3
# Runs every generated module test, in parallel with -j, and writes test_results.json
# and test_results.md next to this file. Exit code is the number of failures.
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path("{repo}")
manifest = json.load(open(OUT / "manifest.json"))


def run_one(m):
    log = OUT / "logs" / (m["module"] + ".log")
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(OUT / "tb") + ":" + str(REPO / "tb"))
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, str(OUT / "tb" / ("test_" + m["module"] + ".py"))],
                            stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    txt = log.read_text()
    ok = rc == 0 and "FAIL=0" in txt and "PASS=1" in txt
    cyc = re.search(r"bit-exact, (\\d+) cycles per vector", txt)
    return dict(name=m["name"], K=m["K"], N=m["N"], kind=m["kind"], ok=ok, seconds=round(time.time() - t0, 1),
                cycles_per_vector=int(cyc.group(1)) if cyc else None, log=str(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--only", default=None, help="regex on matrix name")
    a = ap.parse_args()
    todo = [m for m in manifest["matrices"] if a.only is None or re.search(a.only, m["name"])]
    with ThreadPoolExecutor(a.j) as ex:
        results = list(ex.map(run_one, todo))
    for r in results:
        print(f"{{'PASS' if r['ok'] else 'FAIL'}}  {{r['name']:28s}} {{r['K']:6d}}x{{r['N']:<6d}} {{r['kind']:5s}} "
              f"{{str(r['cycles_per_vector']):>8s}} cyc/vec  {{r['seconds']:6.1f}}s", flush=True)
    fails = [r for r in results if not r["ok"]]
    json.dump(results, open(OUT / "test_results.json", "w"), indent=2)
    with open(OUT / "test_results.md", "w") as f:
        f.write("| matrix | K x N | kind | cycles/vector | MACs/cycle | result |\\n|---|---|---|---|---|---|\\n")
        for r in results:
            macs = f"{{r['K']*r['N']/r['cycles_per_vector']:.1f}}" if r["cycles_per_vector"] else "-"
            f.write(f"| {{r['name']}} | {{r['K']}} x {{r['N']}} | {{r['kind']}} | {{r['cycles_per_vector']}} | {{macs}} | {{'pass' if r['ok'] else 'FAIL'}} |\\n")
    print(f"{{len(results) - len(fails)}} passed, {{len(fails)}} failed")
    sys.exit(len(fails))


if __name__ == "__main__":
    main()
'''


def save_bundle(q, path):
    """Quantised model -> npz + json (numpy arrays in the npz, scalars in the json)."""
    arrays, meta = {}, {}

    def put(prefix, obj):
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                put(key + ".", v)
            elif isinstance(v, np.ndarray):
                arrays[key] = v
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    put(f"{key}.{i}.", item)
            else:
                meta[key] = v.tolist() if hasattr(v, "tolist") else v
    put("", q)
    np.savez_compressed(str(path) + ".npz", **arrays)
    json.dump(meta, open(str(path) + ".json", "w"), indent=1)


def load_bundle(path):
    """Inverse of save_bundle: back to the nested dict quantise_model produced."""
    arrays = dict(np.load(str(path) + ".npz"))
    meta = json.load(open(str(path) + ".json"))
    q = {}
    for k, v in list(arrays.items()) + list(meta.items()):
        parts = k.split(".")
        d = q
        for p in parts[:-1]:
            if p.isdigit():
                p = int(p)
                if not isinstance(d, list):
                    raise ValueError(k)
                while len(d) <= p:
                    d.append({})
                d = d[p]
            else:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    nxt_is_idx = parts[parts.index(p) + 1].isdigit() if parts.index(p) + 1 < len(parts) else False
                    d[p] = [] if nxt_is_idx else {}
                    d = d[p]
        d[parts[-1]] = v
    for lay in q["layers"]:
        for m in ("qkv", "proj", "ffwd1", "ffwd2"):
            lay[m]["relu"] = bool(lay[m]["relu"])
        lay["attn"]["u"] = tuple(lay["attn"]["u"])
        lay["attn"]["o"] = tuple(lay["attn"]["o"])
    q["lm_head"]["relu"] = bool(q["lm_head"]["relu"])
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 4, 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=8, help="array size T for ROM-fed matrices")
    ap.add_argument("--const-max", type=int, default=0, help="hardwire matrices with this many weights or fewer")
    ap.add_argument("--only", default=None, help="regex: only emit matching matrices")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    for d in ("rtl", "mem", "tb"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(Path(a.model_dir) / "config.json"))
    st = load_tensors(a.model_dir)
    kind = cfg.get("model_type")
    if kind == "karpathy_gpt":
        mats, q = matrices_karpathy(cfg, st, a.bits, a.model_dir)
    elif kind == "llama":
        mats, q = matrices_llama(cfg, st, a.bits)
    else:
        sys.exit(f"no adapter for model_type {kind!r}")
    if a.only:
        mats = [m for m in mats if re.search(a.only, m["name"])]

    manifest = dict(model_dir=str(Path(a.model_dir).resolve()), model_type=kind, bits=a.bits, tile=a.tile,
                    const_max=a.const_max, matrices=[])
    total = 0
    for m in mats:
        e = emit_matrix(m, out, a.tile, a.bits, a.const_max)
        trials = 2 if e["weights"] > 4_000_000 else 4
        (out / "tb" / f"test_{e['module']}.py").write_text(TEST_TEMPLATE.format(
            module=e["module"], name=m["name"], K=e["K"], N=e["N"], bits=a.bits, kind=e["kind"], repo=REPO, out=out,
            has_bias=e["bias"], trials=trials, rtl_files=e["rtl"]))
        manifest["matrices"].append(e)
        total += e["weights"]
        print(f"{e['module']:28s} {e['K']:6d} x {e['N']:<6d} {e['kind']:5s} {e['weights']:>10,d} weights", flush=True)
    manifest["total_weights"] = total
    if q is not None:
        save_bundle(q, out / "model_q")
        manifest["bundle"] = "model_q"
        if a.only is None:
            emit_gpt_top(q, out, manifest, a.tile)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    (out / "run_tests.py").write_text(RUNNER_TEMPLATE.format(repo=REPO))
    os.chmod(out / "run_tests.py", 0o755)
    print(f"{len(mats)} matrices, {total:,d} weights at {a.bits} bit -> {out}")



# ------------------------------------------------------------------ whole model

OP = dict(END=0, EMBED=1, LN=2, MATVEC=3, REQUANT=4, ATTN=5, ADD16=6, OUT=7)


def op_word(opcode, unit=0, src=0, dst=0, len_in=0, len_out=0, param=0, flags=0):
    assert src < 4096 and dst < 4096 and len_in < 4096 and len_out < 4096 and param < 65536
    return (flags << 76) | (param << 60) | (len_out << 48) | (len_in << 36) | (dst << 24) | (src << 12) | (unit << 4) | opcode


def emit_gpt_top(q, out, manifest, tile):
    """Program, parameter tables and gpt_top.sv for a Karpathy GPT bundle."""
    D, H, L, T, V = q["D"], q["H"], q["L"], q["T"], q["V"]
    F = q["layers"][0]["ffwd1"]["w"].shape[0]
    mem = out / "mem"
    rtl = out / "rtl"
    mv_index = {m["name"]: i for i, m in enumerate(manifest["matrices"])}

    # memory map, in 32-bit words
    A = {}
    cur = 0
    for name, size in (("H", D), ("X", D), ("QKVACC", 3 * D), ("QKV", 3 * D), ("O", D), ("ACC", max(F, D)),
                       ("TMP", D), ("F", F), ("LOGITS_ACC", V), ("LOGITS", V)):
        A[name] = cur
        cur += size
    mem_words = 1 << (cur - 1).bit_length()

    # requant parameters, one ROM, base offsets per matrix
    # {bias, n, M0} per column; lm_head gets M0 = 2^15, n = 15 (identity) so it is a bias add
    entries, rq_base = [], {}
    for l, lay in enumerate(q["layers"]):
        for m in ("qkv", "proj", "ffwd1", "ffwd2"):
            rq_base[f"layer{l}_{m}"] = len(entries)
            entries += list(zip(lay[m]["b"], lay[m]["n"], lay[m]["m0"]))
    rq_base["lm_head"] = len(entries)
    entries += [(b, 15, 1 << 15) for b in q["lm_head"]["b"]]
    bmax = max(abs(int(b)) for b, _, _ in entries)
    bw = 18 if bmax < (1 << 17) else 34            # 22 + BW must be whole hex digits
    rq_words = [((int(b) & ((1 << bw) - 1)) << 22) | (int(n) << 16) | int(m0) for b, n, m0 in entries]
    nparam = 1 << (len(entries) - 1).bit_length()
    write_memh(mem / "requant.memh", rq_words, 22 + bw)

    # attention parameters and the exp table
    attn_words = []
    for lay in q["layers"]:
        (m0u, nu), (m0o, no) = lay["attn"]["u"], lay["attn"]["o"]
        attn_words.append((int(no) << 38) | (int(m0o) << 22) | (int(nu) << 16) | int(m0u))
    write_memh(mem / "attn_params.memh", attn_words, 44)
    write_memh(mem / "exp_lut.memh", [int(v) for v in q["exp_lut"]], 20)
    write_memh(mem / "tok_emb.memh", [int(v) & 0xFFFF for v in q["tok_emb"].reshape(-1)], 16)
    # character tables for the board: ASCII -> token id (0xff = unknown) and back
    chars = json.load(open(Path(manifest["model_dir"]) / "tokenizer.json"))["chars"]
    stoi = [0xFF] * 256
    for i, c in enumerate(chars):
        stoi[ord(c)] = i
    write_memh(mem / "stoi.memh", stoi, 8)
    write_memh(mem / "itos.memh", [ord(c) for c in chars], 8)
    write_memh(mem / "pos_emb.memh", [int(v) & 0xFFFF for v in q["pos_emb"].reshape(-1)], 16)

    # the program
    prog, listing = [], []

    def emit(name, **kw):
        prog.append(op_word(OP[name], **kw))
        listing.append(f"{len(prog)-1:3d}  {name:8s} " + " ".join(f"{k}={v}" for k, v in kw.items()))

    emit("EMBED", dst=A["H"], len_out=D)
    for l in range(L):
        emit("LN", src=A["H"], dst=A["X"], len_in=D, len_out=D)
        emit("MATVEC", unit=mv_index[f"layer{l}_qkv"], src=A["X"], dst=A["QKVACC"], len_in=D, len_out=3 * D)
        emit("REQUANT", src=A["QKVACC"], dst=A["QKV"], len_in=3 * D, len_out=3 * D, param=rq_base[f"layer{l}_qkv"])
        emit("ATTN", src=A["QKV"], dst=A["O"], len_in=3 * D, len_out=D, param=l)
        emit("MATVEC", unit=mv_index[f"layer{l}_proj"], src=A["O"], dst=A["ACC"], len_in=D, len_out=D)
        emit("REQUANT", src=A["ACC"], dst=A["TMP"], len_in=D, len_out=D, param=rq_base[f"layer{l}_proj"], flags=2)   # int16
        emit("ADD16", src=A["TMP"], dst=A["H"], len_in=D)
        emit("LN", src=A["H"], dst=A["X"], len_in=D, len_out=D)
        emit("MATVEC", unit=mv_index[f"layer{l}_ffwd1"], src=A["X"], dst=A["ACC"], len_in=D, len_out=F)
        emit("REQUANT", src=A["ACC"], dst=A["F"], len_in=F, len_out=F, param=rq_base[f"layer{l}_ffwd1"], flags=1)
        emit("MATVEC", unit=mv_index[f"layer{l}_ffwd2"], src=A["F"], dst=A["ACC"], len_in=F, len_out=D)
        emit("REQUANT", src=A["ACC"], dst=A["TMP"], len_in=D, len_out=D, param=rq_base[f"layer{l}_ffwd2"], flags=2)
        emit("ADD16", src=A["TMP"], dst=A["H"], len_in=D)
    emit("LN", src=A["H"], dst=A["X"], len_in=D, len_out=D)
    emit("MATVEC", unit=mv_index["lm_head"], src=A["X"], dst=A["LOGITS_ACC"], len_in=D, len_out=V)
    emit("REQUANT", src=A["LOGITS_ACC"], dst=A["LOGITS"], len_in=V, len_out=V, param=rq_base["lm_head"], flags=4)   # int32: bias only
    emit("OUT", src=A["LOGITS"], len_in=V)
    emit("END")
    prog_len = 1 << (len(prog) - 1).bit_length()
    write_memh(mem / "prog.memh", prog + [0] * (prog_len - len(prog)), 96)
    (out / "program.txt").write_text("\n".join(listing) + "\n\nmemory map (words): " +
                                    ", ".join(f"{k}={v}" for k, v in A.items()) + "\n")

    nmv = len(manifest["matrices"])
    inst = []
    for m in manifest["matrices"]:
        i = mv_index[m["name"]]
        inst.append(f"    {m['module']} u_mv{i} (.clk(clk), .rst_n(rst_n), .in_valid(mv_in_valid[{i}]), .in_data(unit_in_data[7:0]),\n"
                    f"        .out_valid(mv_out_valid[{i}]), .out_data(mv_out_data[{i}]), .busy(mv_busy[{i}]));   // {m['K']}x{m['N']}")
    sv = f"""// generated by compiler/compile.py from {manifest['model_dir']}
// {L} layers, D={D}, {H} heads, context {T}, vocab {V}, weights {manifest['total_weights']:,d} at {manifest['bits']} bit,
// {tile}x{tile} arrays. Program and memory map in program.txt.
//
// Feed one token at a time: tok_valid with tok_in, wait for done. The {V} int32 logits
// for that position stream out on logit_valid/logit_data during the OUT op and argmax
// holds the greedy next token when done pulses. clear resets the position to 0; the
// context is {T} tokens and there is no sliding window.
module gpt_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        tok_valid,
    input  logic [{(V-1).bit_length()-1}:0]  tok_in,
    input  logic        clear,
    output logic        done,
    output logic        busy,
    output logic [{T.bit_length()-1}:0]  pos,
    output logic        logit_valid,
    output logic [31:0] logit_data,
    output logic [{(V-1).bit_length()-1}:0]  argmax
);
    logic [31:0] unit_in_data;
    logic        emb_start, emb_out_valid, emb_busy;
    logic [31:0] emb_out_data;
    logic [{(V-1).bit_length()-1}:0]  emb_token;
    logic        ln_in_valid, ln_out_valid, ln_busy;
    logic [31:0] ln_out_data;
    logic        rq_in_valid, rq_in_first, rq_relu, rq_out_valid;
    logic [1:0]  rq_width;
    logic [{nparam.bit_length()-2}:0] rq_base;
    logic [31:0] rq_out_data;
    logic        at_in_valid, at_out_valid, at_busy;
    logic [3:0]  at_layer;
    logic [31:0] at_out_data;
    logic        mv_in_valid [{nmv}];
    logic        mv_out_valid [{nmv}];
    logic [31:0] mv_out_data [{nmv}];
    logic        mv_busy [{nmv}];

    sequencer #(.NMV({nmv}), .MEM_WORDS({mem_words}), .PROG_LEN({prog_len}), .VOCAB({V}), .CTX({T}), .NPARAM({nparam}),
                .PROG_FILE("{(mem / 'prog.memh').resolve()}")) u_seq (
        .clk(clk), .rst_n(rst_n), .tok_valid(tok_valid), .tok_in(tok_in), .clear(clear), .done(done), .busy(busy), .pos(pos),
        .logit_valid(logit_valid), .logit_data(logit_data), .argmax(argmax),
        .emb_start(emb_start), .emb_token(emb_token), .emb_out_valid(emb_out_valid), .emb_out_data(emb_out_data), .emb_busy(emb_busy),
        .ln_in_valid(ln_in_valid), .ln_out_valid(ln_out_valid), .ln_out_data(ln_out_data), .ln_busy(ln_busy),
        .rq_in_valid(rq_in_valid), .rq_in_first(rq_in_first), .rq_base(rq_base), .rq_relu(rq_relu), .rq_width(rq_width),
        .rq_out_valid(rq_out_valid), .rq_out_data(rq_out_data),
        .at_in_valid(at_in_valid), .at_layer(at_layer), .at_out_valid(at_out_valid), .at_out_data(at_out_data), .at_busy(at_busy),
        .mv_in_valid(mv_in_valid), .mv_out_valid(mv_out_valid), .mv_out_data(mv_out_data), .mv_busy(mv_busy),
        .unit_in_data(unit_in_data));

    embed #(.D({D}), .V({V}), .T({T}), .TOK_FILE("{(mem / 'tok_emb.memh').resolve()}"), .POS_FILE("{(mem / 'pos_emb.memh').resolve()}")) u_emb (
        .clk(clk), .rst_n(rst_n), .start(emb_start), .token(emb_token), .pos(pos[{(T-1).bit_length()-1}:0]),
        .out_valid(emb_out_valid), .out_data(emb_out_data), .busy(emb_busy));

    layernorm #(.D({D})) u_ln (.clk(clk), .rst_n(rst_n), .in_valid(ln_in_valid), .in_data(unit_in_data),
        .out_valid(ln_out_valid), .out_data(ln_out_data), .busy(ln_busy));

    requant #(.NPARAM({nparam}), .BW({bw}), .PARAM_FILE("{(mem / 'requant.memh').resolve()}")) u_rq (
        .clk(clk), .rst_n(rst_n), .in_valid(rq_in_valid), .in_first(rq_in_first), .in_data(unit_in_data),
        .base(rq_base), .relu(rq_relu), .width(rq_width), .out_valid(rq_out_valid), .out_data(rq_out_data));

    attention #(.D({D}), .H({H}), .T({T}), .L({L}), .PARAM_FILE("{(mem / 'attn_params.memh').resolve()}"),
                .LUT_FILE("{(mem / 'exp_lut.memh').resolve()}")) u_attn (
        .clk(clk), .rst_n(rst_n), .layer(at_layer[{(L-1).bit_length()-1 if L > 1 else 0}:0]), .pos(pos[{(T-1).bit_length()-1}:0]), .in_valid(at_in_valid), .in_data(unit_in_data),
        .out_valid(at_out_valid), .out_data(at_out_data), .busy(at_busy));

{chr(10).join(inst)}
endmodule
"""
    (rtl / "gpt_top.sv").write_text(sv)
    (out / "tb" / "test_gpt_top.py").write_text(GPT_TEST_TEMPLATE.format(repo=REPO, out=out, V=V, T=T))
    manifest["gpt_top"] = dict(mem_words=mem_words, nparam=nparam, bias_bits=bw, prog_len=len(prog), memory_map=A)


GPT_TEST_TEMPLATE = '''# generated by compiler/compile.py: gpt_top.sv against IntGPT, token by token.
# Every logit of every position must match; then the hardware generates text greedily.
#   python3 gen/tinygpt/tb/test_gpt_top.py            (PROMPT="..." to change the prompt)
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

REPO = Path("{repo}")
OUT = Path("{out}")
sys.path.insert(0, str(REPO / "compiler"))
from compile import load_bundle          # noqa: E402
from golden_gpt import IntGPT            # noqa: E402

V, T = {V}, {T}
manifest = json.load(open(OUT / "manifest.json"))
chars = json.load(open(Path(manifest["model_dir"]) / "tokenizer.json"))["chars"]
stoi = {{c: i for i, c in enumerate(chars)}}
PROMPT = os.getenv("PROMPT", "ROMEO:")


async def run_token(dut, tok):
    dut.tok_in.value = tok
    dut.tok_valid.value = 1
    await RisingEdge(dut.clk)
    dut.tok_valid.value = 0
    logits = []
    cycles = 0
    while True:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        cycles += 1
        if dut.logit_valid.value == 1:
            logits.append(dut.logit_data.value.to_signed())
        if dut.done.value == 1:
            break
        assert cycles < 2_000_000, "token never finished"
    assert len(logits) == V, f"{{len(logits)}} logits, expected {{V}}"
    return np.array(logits, dtype=np.int64), int(dut.argmax.value), cycles


@cocotb.test()
async def generates_like_the_golden_model(dut):
    q = load_bundle(OUT / "model_q")
    ref = IntGPT(q)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.tok_valid.value = 0
    dut.tok_in.value = 0
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    prompt = [stoi[c] for c in PROMPT]
    toks = list(prompt)
    total_cycles = 0
    t0 = time.time()
    logits = None
    for t in range(T):
        tok = toks[t] if t < len(toks) else int(np.argmax(logits))
        if t >= len(toks):
            toks.append(tok)
        got, hw_argmax, cycles = await run_token(dut, tok)
        exp = ref.step(tok)
        assert np.array_equal(got, exp), (
            f"position {{t}} token {{tok!r}}: logits differ at {{np.flatnonzero(got != exp)[:5]}}: "
            f"hw {{got[got != exp][:5]}} golden {{exp[got != exp][:5]}}")
        assert hw_argmax == int(np.argmax(exp)), f"position {{t}}: argmax {{hw_argmax}} vs {{int(np.argmax(exp))}}"
        assert int(dut.pos.value) == t + 1
        logits = got
        total_cycles += cycles
    text = "".join(chars[i] for i in toks)
    dut._log.info(f"all {{T}} positions bit-exact against IntGPT; {{total_cycles // T}} cycles per token; "
                  f"{{time.time() - t0:.1f}} s wall")
    dut._log.info("prompt " + repr(PROMPT) + " -> hardware wrote: " + repr(text))
    (OUT / "hardware_sample.txt").write_text(text + "\\n")
    (OUT / "hardware_cycles.json").write_text(json.dumps(dict(cycles_per_token=total_cycles // T, tokens=T)))


def run():
    build = OUT / "sim_build" / "gpt_top" / "build"      # own parent dir: cocotb shares ../verilator.o
    rtl = REPO / "rtl"
    srcs = [rtl / f for f in ("pe.sv", "array.sv", "skew.sv", "top.sv", "matvec_rom.sv", "requant.sv", "isqrt.sv",
                              "udiv.sv", "layernorm.sv", "attention.sv", "embed.sv", "sequencer.sv")]
    srcs += sorted((OUT / "rtl").glob("mv_*.sv")) + sorted((OUT / "rtl").glob("const_*.sv")) + [OUT / "rtl" / "gpt_top.sv"]
    runner = get_runner(os.getenv("SIM", "verilator"))
    waves = bool(int(os.getenv("WAVES", "0")))
    runner.build(sources=srcs, hdl_toplevel="gpt_top", waves=waves, always=True, timescale=("1ns", "1ps"),
                 build_dir=build, build_args=["-Wno-fatal"] if os.getenv("SIM", "verilator") == "verilator" else [])
    runner.test(hdl_toplevel="gpt_top", test_module="test_gpt_top", waves=waves, build_dir=build)


if __name__ == "__main__":
    run()
'''


if __name__ == "__main__":
    main()
