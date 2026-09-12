// Unsigned restoring divider, one quotient bit per cycle. Sized by parameter so the
// same module gives LayerNorm its 2^24/std and the softmax its 2^36/sum.
module udiv #(
    parameter int unsigned NW = 25,   // dividend width
    parameter int unsigned DW = 17    // divisor width
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          start,
    input  logic [NW-1:0] num,
    input  logic [DW-1:0] den,
    output logic          done,
    output logic [NW-1:0] quo
);
    logic [NW-1:0] n, q;
    logic [DW:0]   rem;
    logic [DW-1:0] d;
    logic          busy;
    logic [$clog2(NW+1)-1:0] cnt;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0; done <= 1'b0; n <= '0; q <= '0; rem <= '0; d <= '0; cnt <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1; n <= num; d <= den; rem <= '0; q <= '0; cnt <= '0;
            end else if (busy) begin
                // bring down the next dividend bit, msb first
                logic [DW:0] r;
                r = {rem[DW-1:0], n[NW-1]};
                n <= n << 1;
                if (r >= {1'b0, d}) begin
                    rem <= r - {1'b0, d};
                    q <= {q[NW-2:0], 1'b1};
                end else begin
                    rem <= r;
                    q <= {q[NW-2:0], 1'b0};
                end
                cnt <= cnt + 1'b1;
                if (32'(cnt) == NW - 1) begin
                    busy <= 1'b0; done <= 1'b1;
                end
            end
        end
    end
    // q is complete the cycle after the last iteration, which is when done is high
    assign quo = q;
endmodule
