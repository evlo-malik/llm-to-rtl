module array #(parameter int unsigned N = 3) (
    input logic clk,
    input logic rst_n,
    input logic load_w,
    input logic signed [7:0] x_in [N],
    input logic signed [7:0] w_in [N],
    output logic signed [31:0] result [N]
);
wire signed [7:0] x_wire [N][N];
wire signed [7:0] w_wire [N][N];
wire signed [31:0] result_wire [N][N];

genvar i, j;
generate
    for (i = 0; i<N; i++) begin : row
        for (j = 0; j<N; j++) begin : col
            pe unit_pe (
                .clk(clk),
                .rst_n(rst_n),
                .load_w(load_w),
                .x_in(j == 0 ? x_in[i] : x_wire[i][j-1]),
                .w_in(i == 0 ? w_in[j] : w_wire[i-1][j]),
                .psum_in( i == 0 ? 0 : result_wire[i-1][j]),
                .x_out(x_wire[i][j]),
                .w_out(w_wire[i][j]),
                .psum_out(result_wire[i][j])
            );
        end
    end
endgenerate

generate
    for (j = 0; j<N; j++) begin : out
        assign result[j] = result_wire[N-1][j];
    end
endgenerate

endmodule
