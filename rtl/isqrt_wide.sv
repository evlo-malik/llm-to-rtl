// Floor square root of a 64-bit unsigned number, one result bit per cycle, 32 cycles.
// Same digit-by-digit loop as quant.isqrt in the compiler. Pulse start when idle;
// done pulses with the answer on q and q holds until the next start.
module isqrt_wide (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [63:0] v,
    output logic        done,
    output logic [31:0] q
);
    logic [63:0] rem, root, bit_;
    logic        busy;
    logic [4:0]  cnt;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0; done <= 1'b0; rem <= '0; root <= '0; bit_ <= '0; cnt <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1; rem <= v; root <= '0; bit_ <= 64'h4000_0000_0000_0000; cnt <= '0;
            end else if (busy) begin
                if (rem >= root + bit_) begin
                    rem  <= rem - (root + bit_);
                    root <= (root >> 1) + bit_;
                end else begin
                    root <= root >> 1;
                end
                bit_ <= bit_ >> 2;
                cnt  <= cnt + 1'b1;
                if (cnt == 5'd31) begin
                    busy <= 1'b0; done <= 1'b1;
                end
            end
        end
    end
    assign q = root[31:0];
endmodule
