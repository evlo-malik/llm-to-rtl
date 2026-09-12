// Integer LayerNorm over a D-element int16 vector, no affine (gamma and beta are folded
// into the Linear that follows, see compiler/gpt2.py). Output int8 with scale
// 2^-4, i.e. 16 units per standard deviation. Step for step this is
// quant.layernorm_int:
//     mean = sum >> log2 D        c = h - mean        var = (sum c^2) >> log2 D
//     stdv  = isqrt(var), at least 1        inv = 2^24 / stdv
//     y    = clip8( (c * inv + 2^19) >> 20 )
// D int16 values stream in (low 16 bits of in_data), D int8 values stream out
// sign-extended, roughly D + 50 cycles after the last input.
module layernorm #(parameter int unsigned D = 64, parameter int unsigned EPS_VAR = 0) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0] in_data,      // int16 in the low bits
    /* verilator lint_on UNUSEDSIGNAL */
    output logic        out_valid,
    output logic [31:0] out_data,
    output logic        busy            // inputs are ignored while high
);
    localparam int unsigned LG = $clog2(D);
    localparam int unsigned AW = LG > 0 ? LG : 1;
    initial if ((1 << LG) != D) $error("layernorm: D must be a power of two");

    typedef enum logic [2:0] {FILL, MEAN, SS, SQRT, DIV, OUT} state_t;
    state_t state;

    logic signed [15:0] buf_ [D];
    logic [AW-1:0]      wr_i, rd_i;
    logic signed [15:0] rd_q;
    logic               rd_en, rd_en_d;      // address issued / data valid

    logic signed [23:0] sum;
    logic signed [16:0] mean;
    logic signed [17:0] c;
    logic        [39:0] ss;
    logic        [31:0] var_;
    logic        [15:0] stdv;
    logic        [24:0] inv;
    logic               sq_start, sq_done, dv_start, dv_done;
    logic        [15:0] sq_q;
    logic        [24:0] dv_q;
    logic signed [42:0] prod;
    logic               pv;

    isqrt u_sqrt (.clk(clk), .rst_n(rst_n), .start(sq_start), .v(var_), .done(sq_done), .q(sq_q));
    udiv #(.NW(25), .DW(17)) u_div (.clk(clk), .rst_n(rst_n), .start(dv_start),
                                    .num(25'h1000000), .den({1'b0, stdv}), .done(dv_done), .quo(dv_q));

    // buffer: written as the vector arrives, read twice afterwards
    always_ff @(posedge clk) begin
        if (in_valid && state == FILL) buf_[wr_i] <= in_data[15:0];
        rd_q <= buf_[rd_i];
    end

    assign c = 18'(rd_q) - 18'(mean);
    assign busy = (state != FILL) || (wr_i != 0);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= FILL; wr_i <= '0; rd_i <= '0; rd_en <= 1'b0; rd_en_d <= 1'b0;
            sum <= '0; mean <= '0; ss <= '0; var_ <= '0; stdv <= '0; inv <= '0;
            sq_start <= 1'b0; dv_start <= 1'b0; prod <= '0; pv <= 1'b0;
            out_valid <= 1'b0; out_data <= '0;
        end else begin
            sq_start <= 1'b0; dv_start <= 1'b0;
            rd_en_d <= rd_en;
            // the read pointer sweeps 0..D-1 whenever rd_en is up
            if (rd_en) begin
                rd_i <= rd_i + 1'b1;
                if (rd_i == AW'(D - 1)) rd_en <= 1'b0;
            end
            // output pipeline: c*inv one cycle after rd_q, clip the cycle after that
            pv <= rd_en_d && state == OUT;
            prod <= c * 43'(signed'({1'b0, inv}));
            out_valid <= pv;
            out_data  <= clip8((prod + 43'sd524288) >>> 20);

            case (state)
                FILL: if (in_valid) begin
                    sum <= sum + 24'(signed'(in_data[15:0]));
                    if (wr_i == AW'(D - 1)) begin wr_i <= '0; state <= MEAN; end
                    else wr_i <= wr_i + 1'b1;
                end
                MEAN: begin
                    mean <= 17'(sum >>> LG);
                    ss <= '0; rd_i <= '0; rd_en <= 1'b1;
                    state <= SS;
                end
                SS: begin
                    if (rd_en_d) ss <= ss + 40'(c * c);
                    if (!rd_en && !rd_en_d) begin
                        var_ <= 32'(ss >> LG) + EPS_VAR;
                        sq_start <= 1'b1;
                        state <= SQRT;
                    end
                end
                SQRT: if (sq_done) begin
                    stdv <= (sq_q == 0) ? 16'd1 : sq_q;
                    dv_start <= 1'b1;
                    state <= DIV;
                end
                DIV: if (dv_done) begin
                    inv <= dv_q;
                    rd_i <= '0; rd_en <= 1'b1; sum <= '0;
                    state <= OUT;
                end
                OUT: if (!rd_en && !rd_en_d && !pv) state <= FILL;
                default: state <= FILL;
            endcase
        end
    end

    function automatic logic [31:0] clip8(input logic signed [42:0] x);
        logic signed [42:0] y;
        y = (x < -43'sd128) ? -43'sd128 : (x > 43'sd127) ? 43'sd127 : x;
        return y[31:0];
    endfunction
endmodule
