// High-dynamic-range normalisation: INT32 residuals -> INT16 at scale 2^-8.
// Affine terms are folded into the following fixed projection.
module normalise_wide #(
    parameter integer D=64,
    parameter bit RMS=1,
    parameter logic [63:0] EPS_VAR=0
)(
    input logic clk,rst_n,in_valid,
    input logic [31:0] in_data,
    output logic out_valid,busy,
    output logic [31:0] out_data
);
    logic signed [31:0] x [0:D-1];
    integer index;
    logic signed [63:0] sum;
    logic signed [32:0] mean;
    logic [79:0] ss;
    logic [31:0] standard_deviation;
    logic [48:0] inverse;
    typedef enum logic [3:0] {FILL, MEAN, SQUARES, SQ_START, SQ_WAIT, DIV_START, DIV_WAIT, OUTPUT} state_t;
    state_t state;
    wire signed [32:0] centered=33'(x[index])-mean;
    wire [63:0] variance=64'(ss/D)+EPS_VAR;
    wire signed [82:0] product=83'(centered)*83'($signed({1'b0,inverse}));
    wire signed [82:0] value=(product+83'sd549755813888) >>> 40;
    logic sq_start,sq_done,div_start,div_done;
    logic [31:0] root;
    logic [48:0] quotient;
    isqrt_wide square_root(.clk(clk),.rst_n(rst_n),.start(sq_start),.v(variance),.done(sq_done),.q(root));
    udiv #(.NW(49),.DW(33)) divide(.clk(clk),.rst_n(rst_n),.start(div_start),
        .num(49'h1000000000000),.den({1'b0,standard_deviation}),.done(div_done),.quo(quotient));
    assign busy=state!=FILL || index!=0;
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            index<=0;sum<=0;mean<=0;ss<=0;standard_deviation<=0;inverse<=0;
            state<=FILL;out_valid<=0;out_data<=0;sq_start<=0;div_start<=0;
        end else begin
            out_valid<=0;sq_start<=0;div_start<=0;
            case(state)
                FILL: if(in_valid) begin
                    x[index]<=in_data;sum<=sum+64'($signed(in_data));
                    if(index==D-1) begin index<=0;state<=MEAN;end
                    else index<=index+1;
                end
                MEAN: begin
                    mean<=RMS ? 33'sd0 : 33'(sum>=0 ? sum/D : -(((-sum)+D-1)/D));
                    ss<=0;state<=SQUARES;
                end
                SQUARES: begin
                    ss<=ss+80'(centered*centered);
                    if(index==D-1) begin index<=0;state<=SQ_START;end
                    else index<=index+1;
                end
                SQ_START: begin sq_start<=1;state<=SQ_WAIT;end
                SQ_WAIT: if(sq_done) begin standard_deviation<=root==0 ? 32'd1 : root;state<=DIV_START;end
                DIV_START: begin div_start<=1;state<=DIV_WAIT;end
                DIV_WAIT: if(div_done) begin inverse<=quotient;state<=OUTPUT;end
                OUTPUT: begin
                    out_valid<=1;out_data<=value < -32768 ? -32'sd32768 : value > 32767 ? 32'sd32767 : value[31:0];
                    if(index==D-1) begin index<=0;sum<=0;state<=FILL;end
                    else index<=index+1;
                end
                default: state<=FILL;
            endcase
        end
    end
endmodule
