// Fixed operation schedule. The scratchpad stores activations only.
module model_control #(
    parameter integer NMV=1, MEM_WORDS=1024, PROG_LEN=32, VOCAB=256, CTX=8,
    parameter logic [164*PROG_LEN-1:0] PROGRAM='0
)(
    input logic clk, rst_n, tok_valid,
    input logic [$clog2(VOCAB)-1:0] tok_in,
    output logic done, busy, error,
    output logic [$clog2(CTX+1)-1:0] pos,
    output logic logit_valid,
    output logic [31:0] logit_data,
    output logic [$clog2(VOCAB)-1:0] argmax,
    output logic emb_start,
    output logic [$clog2(VOCAB)-1:0] emb_token,
    input logic emb_out_valid, emb_busy,
    input logic [31:0] emb_out_data,
    output logic ln_in_valid,
    input logic ln_out_valid, ln_busy,
    input logic [31:0] ln_out_data,
    output logic at_in_valid,
    output logic [15:0] at_layer,
    input logic at_out_valid, at_busy,
    input logic [31:0] at_out_data,
    output logic [NMV-1:0] mv_in_valid,
    input logic [NMV-1:0] mv_out_valid, mv_busy,
    input logic [32*NMV-1:0] mv_out_data,
    output logic [31:0] unit_in_data
);
    localparam integer AW=$clog2(MEM_WORDS), PW=$clog2(PROG_LEN);
    logic [31:0] scratch [0:MEM_WORDS-1];
    logic [AW-1:0] rd_addr, wr_addr;
    logic [31:0] rd_data, wr_data;
    logic wr_en;
    always_ff @(posedge clk) begin
        if (wr_en) scratch[wr_addr] <= wr_data;
        rd_data <= scratch[rd_addr];
    end
    assign unit_in_data=rd_data;
    logic [PW-1:0] pc;
    wire [163:0] op=PROGRAM[164*pc +: 164];
    wire [3:0] opcode=op[3:0];
    wire [15:0] unit_id=op[19:4];
    wire [31:0] src=op[51:20], dst=op[83:52];
    wire [31:0] len_in=op[115:84], len_out=op[147:116];
    wire [15:0] param_id=op[163:148];
    typedef enum logic [2:0] {IDLE, SETUP, RUN, ADD, OUTPUT, FINISH} state_t;
    state_t state;
    logic [31:0] in_count, out_count, add_index;
    logic issued;
    logic [1:0] add_phase;
    logic signed [31:0] add_a, best;
    logic [$clog2(VOCAB)-1:0] token;
    logic valid_result, unit_busy;
    logic [31:0] result;
    always_comb begin
        valid_result=0; result=0; unit_busy=0;
        case(opcode)
            1: begin valid_result=emb_out_valid; result=emb_out_data; unit_busy=emb_busy; end
            2: begin valid_result=ln_out_valid; result=ln_out_data; unit_busy=ln_busy; end
            3: begin valid_result=mv_out_valid[unit_id]; result=mv_out_data[32*unit_id +:32]; unit_busy=mv_busy[unit_id]; end
            4: begin valid_result=at_out_valid; result=at_out_data; unit_busy=at_busy; end
            default: ;
        endcase
    end
    wire feed=(state==RUN) && issued;
    assign ln_in_valid=feed && opcode==2;
    assign at_in_valid=feed && opcode==4;
    assign at_layer=param_id;
    assign emb_token=token;
    for(genvar i=0;i<NMV;i=i+1) begin:g_input
        assign mv_in_valid[i]=feed && opcode==3 && unit_id==i;
    end
    assign busy=state!=IDLE;
    always_comb begin
        rd_addr='0;
        case(state)
            RUN, OUTPUT: rd_addr=AW'(src+in_count);
            ADD: rd_addr=add_phase==0 ? AW'(src+add_index) : AW'(dst+add_index);
            default: ;
        endcase
    end
    function automatic [31:0] sat16(input logic signed [32:0] x);
        sat16=x < -33'sd32768 ? -32'sd32768 : x > 33'sd32767 ? 32'sd32767 : x[31:0];
    endfunction
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE; pc<=0; pos<=0; token<=0; done<=0; error<=0;
            emb_start<=0; in_count<=0; out_count<=0; issued<=0;
            wr_en<=0; wr_addr<=0; wr_data<=0; add_index<=0; add_phase<=0;
            add_a<=0; best<=0; argmax<=0; logit_valid<=0; logit_data<=0;
        end else begin
            done<=0; error<=0; emb_start<=0; wr_en<=0; logit_valid<=0; issued<=0;
            case(state)
                IDLE: if(tok_valid) begin
                    if(tok_in>=VOCAB || pos>=CTX) error<=1;
                    else begin token<=tok_in; pc<=0; state<=SETUP; end
                end
                SETUP: begin
                    in_count<=0; out_count<=0;
                    case(opcode)
                        0: state<=FINISH;
                        1: begin emb_start<=1; state<=RUN; end
                        5: begin add_index<=0; add_phase<=0; state<=ADD; end
                        6: begin best<=32'sh80000000; argmax<=0; state<=OUTPUT; end
                        default: state<=RUN;
                    endcase
                end
                RUN: begin
                    if(in_count<len_in) begin in_count<=in_count+1; issued<=1; end
                    if(valid_result) begin
                        wr_en<=1; wr_addr<=AW'(dst+out_count); wr_data<=result;
                        out_count<=out_count+1;
                    end
                    if(out_count==len_out && in_count==len_in && !issued && !valid_result && !unit_busy) begin
                        pc<=pc+1'b1; state<=SETUP;
                    end
                end
                ADD: case(add_phase)
                    0: add_phase<=1;
                    1: begin add_a<=$signed(rd_data); add_phase<=2; end
                    2: begin
                        wr_en<=1; wr_addr<=AW'(dst+add_index);
                        wr_data<=sat16(33'(add_a)+33'($signed(rd_data)));
                        add_phase<=0;
                        if(add_index==len_in-1) begin pc<=pc+1'b1; state<=SETUP; end
                        else add_index<=add_index+1;
                    end
                    default: add_phase<=0;
                endcase
                OUTPUT: begin
                    if(in_count<len_in) begin in_count<=in_count+1; issued<=1; end
                    if(issued) begin
                        logit_valid<=1; logit_data<=rd_data;
                        if($signed(rd_data)>best) begin best<=$signed(rd_data); argmax<=out_count[$clog2(VOCAB)-1:0]; end
                        out_count<=out_count+1;
                        if(out_count==len_in-1) begin pc<=pc+1'b1; state<=SETUP; end
                    end
                end
                FINISH: begin pos<=pos+1'b1; done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
