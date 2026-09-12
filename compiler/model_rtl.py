"""Connect fixed linear maps, fixed embedding decoders and dynamic attention."""

from pathlib import Path
import shutil

from .emit import emit_linear, literal

ROOT = Path(__file__).resolve().parents[1]


def packed(values, width):
    word = 0
    for i, value in enumerate(values):
        word |= (int(value) & ((1 << width) - 1)) << (width * i)
    return f"{len(values) * width}'h{word:x}"


def emit_embedding(path, q):
    d, v, t = q["D"], q["V"], q["T"]
    vw = max(1, (v - 1).bit_length())
    tw = max(1, (t - 1).bit_length())
    lines = [
        f"module fixed_embedding(input logic clk,rst_n,start, input logic [{vw - 1}:0] token,",
        f"input logic [{tw - 1}:0] pos, output logic out_valid,busy, output logic [31:0] out_data);",
        f"logic [{16 * d - 1}:0] token_value, position_value, held_token, held_position;",
        "integer column;",
        "always_comb begin",
        "token_value='0;",
        "case(token)",
    ]
    # Literal decode, not an initialised RAM. The physical flow maps this truth table
    # to gates; no coefficient word is read into a multiplier.
    for i, row in enumerate(q["tok_emb"]):
        lines.append(f"{vw}'d{i}: token_value={packed(row, 16)};")
    lines += ["default: ;", "endcase", "position_value='0;", "case(pos)"]
    for i, row in enumerate(q["pos_emb"]):
        lines.append(f"{tw}'d{i}: position_value={packed(row, 16)};")
    lines += [
        "default: ;",
        "endcase",
        "end",
        "wire signed [16:0] total=17'($signed(held_token[16*column +:16])) + 17'($signed(held_position[16*column +:16]));",
        "always_ff @(posedge clk) begin",
        "if(!rst_n) begin busy<=0; out_valid<=0; out_data<=0; column<=0; held_token<=0; held_position<=0; end",
        "else begin out_valid<=0;",
        "if(start && !busy) begin held_token<=token_value; held_position<=position_value; column<=0; busy<=1; end",
        "else if(busy) begin",
        "out_valid<=1; out_data<=total < -17'sd32768 ? -32'sd32768 : total > 17'sd32767 ? 32'sd32767 : 32'(total);",
        f"if(column=={d - 1}) busy<=0; else column<=column+1;",
        "end end end endmodule",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def emit_gelu(path, name, table):
    lines = [
        f"module {name}(input logic clk,rst_n,in_valid, input logic [7:0] in_data,",
        "output logic out_valid,busy, output logic [31:0] out_data);",
        "assign busy=1'b0;",
        "logic [31:0] value;",
        "always_comb begin",
        "value=32'd0;",
        "case(in_data)",
    ]
    for i, value in enumerate(table):
        lines.append(f"8'd{(i - 128) & 255}: value={literal(value)};")
    lines += [
        "default: ;",
        "endcase",
        "end",
        "always_ff @(posedge clk) begin",
        "if(!rst_n) begin out_valid<=0; out_data<=0; end",
        "else begin out_valid<=in_valid; out_data<=value; end",
        "end endmodule",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def emit_model(out, q):
    from .rotary_rtl import emit_rotary, emit_gated

    out = Path(out)
    rtl = out / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)
    units = []
    index = {}
    stats = []
    for l, layer in enumerate(q["layers"]):
        for kind in ("qkv", "proj", "ffwd1", "ffwd2"):
            name = f"layer{l}_{kind}"
            m = layer[kind]
            index[name] = len(units)
            units.append(name)
            kwargs = {} if m["out_bits"] == 32 else dict(m0=m["m0"], shifts=m["n"])
            stats.append(
                emit_linear(
                    rtl / f"{name}.sv",
                    name,
                    m["w"],
                    m["b"],
                    out_bits=m["out_bits"],
                    **kwargs,
                )
            )
        name = f"layer{l}_gelu"
        index[name] = len(units)
        units.append(name)
        if q.get("gated"):
            emit_gated(
                rtl / f"{name}.sv",
                name,
                layer["ffwd2"]["w"].shape[1],
                layer["gelu"],
                layer["gate_scale"],
            )
        else:
            emit_gelu(rtl / f"{name}.sv", name, layer["gelu"])
    if "rope_cos" in q:
        index["fixed_rotary"] = len(units)
        units.append("fixed_rotary")
        emit_rotary(rtl / "fixed_rotary.sv", q)
    index["lm_head"] = len(units)
    units.append("lm_head")
    m = q["lm_head"]
    stats.append(emit_linear(rtl / "lm_head.sv", "lm_head", m["w"], m["b"]))
    emit_embedding(rtl / "fixed_embedding.sv", q)
    for name in (
        "model_control.sv",
        "layernorm.sv",
        "isqrt.sv",
        "udiv.sv",
        "attention.sv",
    ):
        shutil.copyfile(ROOT / "rtl" / name, rtl / name)
    d, v, t, layers = q["D"], q["V"], q["T"], q["L"]
    inner = q["layers"][0]["ffwd2"]["w"].shape[1]
    first = inner * (2 if q.get("gated") else 1)
    qkv_size = d + 2 * q.get("KH", q["H"]) * (d // q["H"])
    sizes = {
        "H": d,
        "X": d,
        "QKV": qkv_size,
        "ROT": qkv_size,
        "O": d,
        "TMP": d,
        "F": first,
        "G": inner,
        "LOGITS": v,
    }
    addresses = {}
    cursor = 0
    for name, size in sizes.items():
        addresses[name] = cursor
        cursor += size
    mem_words = 1 << (cursor - 1).bit_length()
    program = []
    listing = []

    def op(code, unit=0, src=None, dst=None, ni=0, no=0, param=0):
        si = addresses[src] if src else 0
        di = addresses[dst] if dst else 0
        program.append(
            code | unit << 4 | si << 20 | di << 52 | ni << 84 | no << 116 | param << 148
        )
        listing.append(
            dict(
                opcode=code,
                unit=unit,
                src=src,
                dst=dst,
                inputs=ni,
                outputs=no,
                param=param,
            )
        )

    def linear(name, src, dst, ni, no):
        op(3, index[name], src, dst, ni, no)

    op(1, dst="H", no=d)
    for l in range(layers):
        op(2, src="H", dst="X", ni=d, no=d)
        linear(f"layer{l}_qkv", "X", "QKV", d, qkv_size)
        if "rope_cos" in q:
            linear("fixed_rotary", "QKV", "ROT", qkv_size, qkv_size)
        op(
            4,
            src="ROT" if "rope_cos" in q else "QKV",
            dst="O",
            ni=qkv_size,
            no=d,
            param=l,
        )
        linear(f"layer{l}_proj", "O", "TMP", d, d)
        op(5, src="TMP", dst="H", ni=d)
        op(2, src="H", dst="X", ni=d, no=d)
        linear(f"layer{l}_ffwd1", "X", "F", d, first)
        linear(f"layer{l}_gelu", "F", "G", first, inner)
        linear(f"layer{l}_ffwd2", "G", "TMP", inner, d)
        op(5, src="TMP", dst="H", ni=d)
    op(2, src="H", dst="X", ni=d, no=d)
    linear("lm_head", "X", "LOGITS", d, v)
    op(6, src="LOGITS", ni=v)
    op(0)
    nprog = 1 << (len(program) - 1).bit_length()
    program += [0] * (nprog - len(program))
    attention = []
    for layer in q["layers"]:
        m0u, nu = layer["attn"]["u"]
        m0o, no = layer["attn"]["o"]
        attention.append(m0u | nu << 16 | m0o << 22 | no << 38)
    nmv = len(units)
    vw = (v - 1).bit_length()
    pw = t.bit_length()
    tw = (t - 1).bit_length()
    lw = max(1, (layers - 1).bit_length())
    lines = [
        f"// {q.get('architecture', 'gpt2')}, {layers} layers, width {d}, context {t}. All checkpoint tensors are constants.",
        f"module model_top(input logic clk,rst_n,clear,tok_valid, input logic [{vw - 1}:0] tok_in,",
        f"output logic done,busy,error,tok_ready, output logic [{pw - 1}:0] pos,",
        f"output logic logit_valid, output logic [31:0] logit_data, output logic [{vw - 1}:0] argmax);",
        "wire core_rst_n=rst_n && !clear;",
        "assign tok_ready=core_rst_n && !busy && pos<" + str(t) + ";",
        "wire [31:0] unit_in_data,emb_out_data,ln_out_data,at_out_data;",
        "wire emb_start,emb_out_valid,emb_busy,ln_in_valid,ln_out_valid,ln_busy,at_in_valid,at_out_valid,at_busy;",
        f"wire [{vw - 1}:0] emb_token;",
        "wire [15:0] at_layer;",
        f"wire [{nmv - 1}:0] mv_in_valid,mv_out_valid,mv_busy;",
        f"wire [{32 * nmv - 1}:0] mv_out_data;",
        f"model_control #(.NMV({nmv}),.MEM_WORDS({mem_words}),.PROG_LEN({nprog}),.VOCAB({v}),.CTX({t}),",
        f".PROGRAM({packed(program, 164)})) control(",
        ".clk(clk),.rst_n(core_rst_n),.tok_valid(tok_valid && !clear),.tok_in(tok_in),.done(done),.busy(busy),.error(error),.pos(pos),",
        ".logit_valid(logit_valid),.logit_data(logit_data),.argmax(argmax),",
        ".emb_start(emb_start),.emb_token(emb_token),.emb_out_valid(emb_out_valid),.emb_busy(emb_busy),.emb_out_data(emb_out_data),",
        ".ln_in_valid(ln_in_valid),.ln_out_valid(ln_out_valid),.ln_busy(ln_busy),.ln_out_data(ln_out_data),",
        ".at_in_valid(at_in_valid),.at_layer(at_layer),.at_out_valid(at_out_valid),.at_busy(at_busy),.at_out_data(at_out_data),",
        ".mv_in_valid(mv_in_valid),.mv_out_valid(mv_out_valid),.mv_busy(mv_busy),.mv_out_data(mv_out_data),.unit_in_data(unit_in_data));",
        f"fixed_embedding embed(.clk(clk),.rst_n(core_rst_n),.start(emb_start),.token(emb_token),.pos(pos[{tw - 1}:0]),",
        ".out_valid(emb_out_valid),.out_data(emb_out_data),.busy(emb_busy));",
        f"layernorm #(.D({d}),.EPS_VAR({q['eps_var']}),.RMS({int(q.get('norm') == 'rms')})) norm(.clk(clk),.rst_n(core_rst_n),.in_valid(ln_in_valid),.in_data(unit_in_data),",
        ".out_valid(ln_out_valid),.out_data(ln_out_data),.busy(ln_busy));",
        f"attention #(.D({d}),.H({q['H']}),.KH({q.get('KH', q['H'])}),.WINDOWS({packed([x.get('window', t) for x in q['layers']], 11)}),.T({t}),.L({layers}),.PARAMS({packed(attention, 44)}),.EXP_TABLE({packed(q['exp_lut'], 17)})) attn(",
        f".clk(clk),.rst_n(core_rst_n),.layer(at_layer[{lw - 1}:0]),.pos(pos[{tw - 1}:0]),.in_valid(at_in_valid),.in_data(unit_in_data),",
        ".out_valid(at_out_valid),.out_data(at_out_data),.busy(at_busy));",
    ]
    for i, name in enumerate(units):
        lines += [
            f"{name} u_{name}("
            + (f".pos(pos[{tw - 1}:0])," if name == "fixed_rotary" else "")
            + f".clk(clk),.rst_n(core_rst_n),.in_valid(mv_in_valid[{i}]),.in_data(unit_in_data[7:0]),",
            f".out_valid(mv_out_valid[{i}]),.out_data(mv_out_data[{32 * i} +:32]),.busy(mv_busy[{i}]));",
        ]
    lines += ["endmodule", ""]
    (rtl / "model_top.sv").write_text("\n".join(lines))
    return dict(
        top="model_top",
        matrix_modules=stats,
        program=listing,
        scratch_words=mem_words,
        weight_storage="constant_logic",
        embedding_storage="constant_decoder",
        context=t,
        vocab=v,
        width=d,
        layers=layers,
    )
