// Throwaway module — exists only to prove the toolchain works. Delete once pe.sv is real.
module counter #(parameter W = 8) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         en,
    output logic [W-1:0] count
);
    always_ff @(posedge clk) begin
        if (!rst_n)   count <= '0;
        else if (en)  count <= count + 1'b1;
    end
endmodule
