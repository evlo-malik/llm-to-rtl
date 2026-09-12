// 8N1 receiver, 16x oversampled by counting clock cycles. Samples mid-bit. data_valid
// pulses for one cycle per byte; no FIFO here.
module uart_rx #(
    parameter int unsigned CLK_HZ = 27_000_000,
    parameter int unsigned BAUD   = 115_200
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       rx,
    output logic       data_valid,
    output logic [7:0] data
);
    localparam int unsigned DIV = CLK_HZ / BAUD;
    logic [$clog2(DIV)-1:0] cnt;
    logic [3:0] bit_i;
    logic [7:0] sh;
    logic       run;
    logic [1:0] sync;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sync <= 2'b11; run <= 1'b0; cnt <= '0; bit_i <= '0; sh <= '0; data_valid <= 1'b0; data <= '0;
        end else begin
            sync <= {sync[0], rx};
            data_valid <= 1'b0;
            if (!run) begin
                if (sync[1] == 1'b0) begin           // start bit edge: sample half a bit later
                    run <= 1'b1; cnt <= ($clog2(DIV))'(DIV / 2); bit_i <= '0;
                end
            end else if (cnt == 0) begin
                cnt <= ($clog2(DIV))'(DIV - 1);
                if (bit_i == 0) begin
                    if (sync[1] != 1'b0) run <= 1'b0;   // false start
                    bit_i <= 4'd1;
                end else if (bit_i <= 4'd8) begin
                    sh <= {sync[1], sh[7:1]};
                    bit_i <= bit_i + 1'b1;
                end else begin                           // stop bit
                    run <= 1'b0;
                    if (sync[1]) begin data <= sh; data_valid <= 1'b1; end
                end
            end else cnt <= cnt - 1'b1;
        end
    end
endmodule
