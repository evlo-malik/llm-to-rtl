module mul (
    input logic clk,
    input logic rst_n,
    input logic signed [7:0] a,
    input logic signed [7:0] b,
    output logic signed [15:0] p
);

always_ff @(posedge clk) begin
    if (!rst_n) p <=0;
    else p <= a * b;
end

endmodule

