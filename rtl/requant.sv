// Streaming requantiser: int32 accumulator in, int8 / int16 / int32 out, one element per
// cycle, three cycles of latency.
//     y = clip( ((acc + bias) * M0 + 2^(n-1)) >>> n ),   then ReLU if asked
// M0 is 16-bit in [2^15, 2^16), n is 0..63, bias is BW bits signed. All three come from a
// ROM indexed by base + column, so one instance serves every matrix in the model;
// compile.py writes the ROM as {bias[BW-1:0], n[5:0], M0[15:0]} per column. The bias
// lives here rather than in the matvec modules because a per-matrix bias ROM costs a
// whole block RAM on an FPGA for a few hundred bits.
//
// in_first marks column 0 of a vector. base, relu and width must hold for the whole
// vector. width: 0 int8, 1 int16, 2 int32 (no clip; with M0 = 2^15, n = 15 that is a
// plain bias add, used for the logits). Output is sign-extended to 32 bits.
module requant #(
    parameter int unsigned NPARAM     = 1024,
    parameter int unsigned BW         = 18,     // 22 + BW is a whole number of hex digits: 18 or 34
    parameter              PARAM_FILE = ""
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic        in_first,
    input  logic [31:0] in_data,
    input  logic [$clog2(NPARAM)-1:0] base,
    input  logic        relu,
    input  logic [1:0]  width,
    output logic        out_valid,
    output logic [31:0] out_data
);
    localparam int unsigned PW = $clog2(NPARAM);
    localparam int unsigned WW = 22 + BW;

    logic [WW-1:0] params [NPARAM];
    initial $readmemh(PARAM_FILE, params);

    // stage 0: column counter and ROM address
    logic [PW-1:0] col;
    logic [PW-1:0] addr;
    assign addr = base + (in_first ? PW'(0) : col);

    // stage 1: parameter word and the accumulator
    logic          v1;
    logic [WW-1:0] p1;
    logic signed [31:0] a1;
    // stage 2: (acc + bias) * M0 plus rounding, and the shift amount
    logic          v2;
    logic signed [49:0] prod2;
    logic [5:0]    n2;

    always_ff @(posedge clk) begin
        p1 <= params[addr];
        if (!rst_n) begin
            col <= '0; v1 <= 1'b0; v2 <= 1'b0; out_valid <= 1'b0;
            a1 <= '0; prod2 <= '0; n2 <= '0; out_data <= '0;
        end else begin
            if (in_valid) col <= (in_first ? PW'(1) : col + 1'b1);
            v1 <= in_valid;
            a1 <= signed'(in_data);

            v2 <= v1;
            n2 <= p1[21:16];
            prod2 <= (50'(a1) + 50'(signed'(p1[WW-1:22]))) * 50'(signed'({1'b0, p1[15:0]}))
                     + (p1[21:16] == 0 ? 50'sd0 : (50'sd1 <<< (p1[21:16] - 1)));

            out_valid <= v2;
            if (v2) out_data <= clip(prod2 >>> n2, width, relu);
        end
    end

    function automatic logic [31:0] clip(input logic signed [49:0] x, input logic [1:0] w, input logic r);
        logic signed [49:0] lo, hi, y;
        lo = (w == 2'd0) ? -50'sd128 : (w == 2'd1) ? -50'sd32768 : -50'sd2147483648;
        hi = (w == 2'd0) ?  50'sd127 : (w == 2'd1) ?  50'sd32767 :  50'sd2147483647;
        y = (x < lo) ? lo : (x > hi) ? hi : x;
        if (r && y < 0) y = 0;
        return y[31:0];
    endfunction
endmodule
