// h = tok_emb[token] + pos_emb[pos], int16 with saturation, streamed out as D values.
// Both tables are int16 at the residual-stream scale, written by compile.py.
module embed #(
    parameter int unsigned D = 64,
    parameter int unsigned V = 65,
    parameter int unsigned T = 32,
    parameter              TOK_FILE = "",
    parameter              POS_FILE = ""
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [$clog2(V)-1:0] token,
    input  logic [$clog2(T)-1:0] pos,
    output logic        out_valid,
    output logic [31:0] out_data,
    output logic        busy
);
    localparam int unsigned DW = $clog2(D);
    logic [15:0] tok_rom [V * D];
    logic [15:0] pos_rom [T * D];
    initial begin
        $readmemh(TOK_FILE, tok_rom);
        $readmemh(POS_FILE, pos_rom);
    end

    logic [DW-1:0] d;
    logic          run, run_q;
    logic signed [15:0] t_q, p_q;

    always_ff @(posedge clk) begin
        t_q <= tok_rom[32'(token) * D + 32'(d)];
        p_q <= pos_rom[32'(pos) * D + 32'(d)];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            d <= '0; run <= 1'b0; run_q <= 1'b0; out_valid <= 1'b0; out_data <= '0;
        end else begin
            run_q <= run;
            if (start && !run) begin
                run <= 1'b1; d <= '0;
            end else if (run) begin
                d <= d + 1'b1;
                if (d == DW'(D - 1)) run <= 1'b0;
            end
            out_valid <= run_q;
            out_data  <= sat16(17'(t_q) + 17'(p_q));
        end
    end
    assign busy = run || run_q;

    function automatic logic [31:0] sat16(input logic signed [16:0] x);
        logic signed [16:0] y;
        y = (x < -17'sd32768) ? -17'sd32768 : (x > 17'sd32767) ? 17'sd32767 : x;
        return 32'(y);
    endfunction
endmodule
