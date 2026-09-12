// 8N1 transmitter. Raise send with data while ready; ready drops until the stop bit
// is out.
module uart_tx #(
    parameter int unsigned CLK_HZ = 27_000_000,
    parameter int unsigned BAUD   = 115_200
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       send,
    input  logic [7:0] data,
    output logic       ready,
    output logic       tx
);
    localparam int unsigned DIV = CLK_HZ / BAUD;
    logic [$clog2(DIV)-1:0] cnt;
    logic [3:0] bit_i;
    logic [9:0] sh;      // start, 8 data, stop

    assign ready = (bit_i == 0);
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cnt <= '0; bit_i <= '0; sh <= '1; tx <= 1'b1;
        end else if (bit_i == 0) begin
            tx <= 1'b1;
            if (send) begin
                sh <= {1'b1, data, 1'b0}; bit_i <= 4'd10; cnt <= ($clog2(DIV))'(DIV - 1);
                tx <= 1'b0;
            end
        end else if (cnt == 0) begin
            cnt <= ($clog2(DIV))'(DIV - 1);
            sh <= {1'b1, sh[9:1]};
            tx <= sh[1];
            bit_i <= bit_i - 1'b1;
        end else cnt <= cnt - 1'b1;
    end
endmodule
