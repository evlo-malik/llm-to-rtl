module array #(parameter int unsigned N = 3) (
    input logic clk,
    input logic rst_n,
    input logic signed [7:0] x_in [N],
    input logic signed [7:0] w_in [N],
    output logic signed [31:0] result [N]
);
wire signed [7:0] x_wire [N][N];
wire signed [7:0] w_wire [N][N];
wire signed [31:0] result_wire [N][N];