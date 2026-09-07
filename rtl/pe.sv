module pe(
    input logic clk,
    input logic rst_n, 
    input logic load_w, 
    input logic signed [7:0] w_in, 
    input logic signed [7:0] x_in, 
    input logic signed [31:0] psum_in,
    output logic signed [31:0] psum_out,
    output logic signed [7:0] x_out
); 
    logic signed [7:0] weight;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight <= 0;
            psum_out <= 0;
            x_out <= 0;
        end
        else begin
            if (load_w) weight <= w_in;
            x_out <= x_in;
            psum_out <= psum_in + weight * x_in;
        end
    end
endmodule
