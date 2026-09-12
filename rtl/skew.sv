// DELAY-cycle delay line. DELAY = 0 is a wire. Used three ways in top.sv: to skew
// row i of the input by i cycles, to deskew column j of the output by N-1-j cycles,
// and to carry the valid bit alongside the data.
module skew #(
    parameter int unsigned W     = 8,
    parameter int unsigned DELAY = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic         clk,      // unused when DELAY == 0
    input  logic         rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [W-1:0] d,
    output logic [W-1:0] q
);
    generate
        if (DELAY == 0) begin : g_wire
            assign q = d;
        end else begin : g_regs
            logic [W-1:0] r [DELAY];
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int k = 0; k < DELAY; k++) r[k] <= '0;
                end else begin
                    r[0] <= d;
                    for (int k = 1; k < DELAY; k++) r[k] <= r[k-1];
                end
            end
            assign q = r[DELAY-1];
        end
    endgenerate
endmodule
