// N x N weight-stationary matmul unit: array.sv plus input skew, output deskew and a
// valid pipeline, behind packed ports so the whole thing synthesises and cocotb can
// drive it as plain integers.
//
// Weights: hold w_load high for N cycles with one row of W on w_row per cycle, LAST
// row first (W[N-1] ... W[0]). Rows shift down the columns and land in place on the
// Nth cycle.
//
// Activations: x_vec with x_valid. y_vec = x_vec @ W (int32 per column) comes out
// 2N-1 cycles later with y_valid. Vectors can go in back to back, one per cycle.
//
// Timing, from pe.sv: row i reaches PE(i,j) at cycle c+i+j and PE(N-1,j) registers
// column j at c+N+j, so column j is deskewed by N-1-j cycles. Everything lines up at
// c+2N-1.
//
// Never raise w_load while busy is high or in the same cycle as x_valid: the wave in
// flight would see weights moving underneath it.
module top #(parameter int unsigned N = 4) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            w_load,
    input  logic [N*8-1:0]  w_row,     // element j in bits [8*j +: 8], signed
    input  logic            x_valid,
    input  logic [N*8-1:0]  x_vec,     // element i in bits [8*i +: 8], signed
    output logic            y_valid,
    output logic [N*32-1:0] y_vec,     // column j in bits [32*j +: 32], signed
    output logic            busy
);
    localparam int unsigned LAT = 2*N - 1;

    logic signed [7:0]  w_in  [N];
    logic signed [7:0]  x_row [N];   // unpacked, unskewed
    logic signed [7:0]  x_skw [N];   // row i delayed i cycles
    logic signed [31:0] col   [N];   // array bottom row
    logic signed [31:0] y     [N];   // deskewed

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : g_io
            assign w_in[i]  = w_row[8*i +: 8];
            assign x_row[i] = x_vec[8*i +: 8];
            skew #(.W(8),  .DELAY(i))     u_in  (.clk(clk), .rst_n(rst_n), .d(x_row[i]), .q(x_skw[i]));
            skew #(.W(32), .DELAY(N-1-i)) u_out (.clk(clk), .rst_n(rst_n), .d(col[i]),   .q(y[i]));
            assign y_vec[32*i +: 32] = y[i];
        end
    endgenerate

    array #(.N(N)) u_array (
        .clk    (clk),
        .rst_n  (rst_n),
        .load_w (w_load),
        .x_in   (x_skw),
        .w_in   (w_in),
        .result (col)
    );

    logic [LAT-1:0] vpipe;
    always_ff @(posedge clk) begin
        if (!rst_n) vpipe <= '0;
        else begin
            vpipe[0] <= x_valid;
            for (int k = 1; k < LAT; k++) vpipe[k] <= vpipe[k-1];
        end
    end
    assign y_valid = vpipe[LAT-1];
    assign busy    = |vpipe;
endmodule
