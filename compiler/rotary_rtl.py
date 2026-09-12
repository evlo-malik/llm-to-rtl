"""Emit positional rotation and SwiGLU units; their buffers hold activations."""

from .model_rtl import packed
from .emit import literal, multiply_constant


def emit_rotary(path, q):
    d, hd = q["D"], q["D"] // q["H"]
    kd = q["KH"] * hd
    size, tw = d + 2 * kd, (q["T"] - 1).bit_length()
    text = f"""module fixed_rotary(input logic clk,rst_n,in_valid, input logic [7:0] in_data,
    input logic [{tw - 1}:0] pos,
    output logic out_valid,busy, output logic [31:0] out_data);
    localparam logic [{q["T"] * hd // 2 * 16 - 1}:0] COS={packed(q["rope_cos"].reshape(-1), 16)};
    localparam logic [{q["T"] * hd // 2 * 16 - 1}:0] SIN={packed(q["rope_sin"].reshape(-1), 16)};
    logic signed [7:0] x [0:{size - 1}];
    integer index;
    logic draining;
    assign busy=draining || index!=0;
    wire upper=(index % {hd}) >= {hd // 2};
    wire signed [31:0] a=32'($signed(x[index]));
    wire signed [31:0] b=32'($signed(x[upper ? index-{hd // 2} : index+{hd // 2}]));
    wire signed [15:0] c=$signed(COS[16*(pos*{hd // 2}+index%{hd // 2}) +:16]);
    wire signed [15:0] s=$signed(SIN[16*(pos*{hd // 2}+index%{hd // 2}) +:16]);
    wire signed [31:0] rotated=(a*c + (upper ? b*s : -(b*s)) + 32'sd8192) >>> 14;
    wire signed [31:0] value=index >= {d + kd} ? a : rotated;
    always_ff @(posedge clk) begin
        if(!rst_n) begin index<=0; draining<=0; out_valid<=0; out_data<=0; end
        else begin
            out_valid<=0;
            if(draining) begin
                out_valid<=1; out_data<=value < -128 ? -32'sd128 : value > 127 ? 32'sd127 : value;
                if(index=={size - 1}) begin index<=0; draining<=0; end
                else index<=index+1;
            end else if(in_valid) begin
                x[index]<=in_data;
                if(index=={size - 1}) begin index<=0; draining<=1; end
                else index<=index+1;
            end
        end
    end
endmodule
"""
    path.write_text(text)


def emit_gated(path, name, inner, table, scale, bits=8, lut_shift=0):
    m, shift = map(int, scale)
    half = (1 << (shift - 1)) if shift else 0
    path.write_text(f"""module {name}(input logic clk,rst_n,in_valid, input logic [{bits - 1}:0] in_data,
    output logic out_valid,busy, output logic [31:0] out_data);
    localparam logic [{len(table) * bits - 1}:0] SILU={packed(table, bits)};
    logic signed [{bits - 1}:0] x[0:{2 * inner - 1}];
    integer index;
    logic draining;
    assign busy=draining || index!=0;
    wire signed [{bits + 1}:0] biased={bits + 2}'($signed(x[index]))+{bits + 2}'sd{(1 << (bits - 1)) + ((1 << (lut_shift - 1)) if lut_shift else 0)};
    wire [{bits - lut_shift - 1}:0] address=biased >= {bits + 2}'sd{1 << bits} ? {bits - lut_shift}'d{len(table) - 1} : biased >> {lut_shift};
    wire signed [{bits - 1}:0] gate=$signed(SILU[{bits}*address +:{bits}]);
    wire signed [31:0] product=32'(gate)*32'($signed(x[index+{inner}]));
    wire signed [63:0] wide=64'(product);
    wire signed [63:0] value=({multiply_constant("wide", m, 64)} + {literal(half, 64)}) >>> {shift};
    always_ff @(posedge clk) begin
        if(!rst_n) begin index<=0; draining<=0; out_valid<=0; out_data<=0; end
        else begin
            out_valid<=0;
            if(draining) begin
                out_valid<=1; out_data<=value < {literal(-(1 << (bits - 1)), 64)} ? {literal(-(1 << (bits - 1)))} : value > {literal((1 << (bits - 1)) - 1, 64)} ? {literal((1 << (bits - 1)) - 1)} : value[31:0];
                if(index=={inner - 1}) begin index<=0; draining<=0; end
                else index<=index+1;
            end else if(in_valid) begin
                x[index]<=in_data;
                if(index=={2 * inner - 1}) begin index<=0; draining<=1; end
                else index<=index+1;
            end
        end
    end
endmodule
""")
