// Runs the model program for one token: a list of ops from PROG_FILE, each one
// "stream len_in words from scratch[src] into a unit, write its len_out outputs to
// scratch[dst]". The scratchpad is 32-bit words, one read port and one write port,
// so a unit can be fed and drained in the same cycles. Units are the fixed ones
// (embed, layernorm, requant, attention) plus NMV matvec modules on an array of
// ports; gpt_top.sv (generated) wires them up.
//
// Op word, 96 bits:
//   [3:0] opcode  0 END  1 EMBED  2 LN  3 MATVEC  4 REQUANT  5 ATTN  6 ADD16  7 OUT
//   [11:4] unit   matvec index for MATVEC
//   [23:12] src   [35:24] dst   [47:36] len_in   [59:48] len_out
//   [75:60] param requant base / attention layer
//   [79:76] flags bit0 relu, bits[2:1] output width for REQUANT (0 int8, 1 int16, 2 int32)
//
// ADD16 is done here: dst[i] = sat16(dst[i] + src[i]). OUT streams the logits to
// the top-level port and tracks the argmax (first index wins ties, like numpy).
module sequencer #(
    parameter int unsigned NMV       = 9,
    parameter int unsigned MEM_WORDS = 2048,
    parameter int unsigned PROG_LEN  = 64,
    parameter int unsigned VOCAB     = 65,
    parameter int unsigned CTX       = 32,
    parameter int unsigned NPARAM    = 1024,
    parameter              PROG_FILE = ""
) (
    input  logic        clk,
    input  logic        rst_n,
    // token interface
    input  logic        tok_valid,
    input  logic [$clog2(VOCAB)-1:0] tok_in,
    input  logic        clear,             // back to position 0
    output logic        done,              // one-cycle pulse when the token's program ends
    output logic        busy,
    output logic [$clog2(CTX+1)-1:0] pos,      // counts to CTX, so one bit wider than an index
    output logic        logit_valid,
    output logic [31:0] logit_data,
    output logic [$clog2(VOCAB)-1:0] argmax,
    // embed
    output logic        emb_start,
    output logic [$clog2(VOCAB)-1:0] emb_token,
    input  logic        emb_out_valid,
    input  logic [31:0] emb_out_data,
    input  logic        emb_busy,
    // layernorm
    output logic        ln_in_valid,
    input  logic        ln_out_valid,
    input  logic [31:0] ln_out_data,
    input  logic        ln_busy,
    // requant
    output logic        rq_in_valid,
    output logic        rq_in_first,
    output logic [$clog2(NPARAM)-1:0] rq_base,
    output logic        rq_relu,
    output logic [1:0]  rq_width,
    input  logic        rq_out_valid,
    input  logic [31:0] rq_out_data,
    // attention
    output logic        at_in_valid,
    output logic [3:0]  at_layer,
    input  logic        at_out_valid,
    input  logic [31:0] at_out_data,
    input  logic        at_busy,
    // matvecs
    output logic        mv_in_valid [NMV],
    input  logic        mv_out_valid [NMV],
    input  logic [31:0] mv_out_data [NMV],
    input  logic        mv_busy [NMV],
    // shared data to every unit's input
    output logic [31:0] unit_in_data
);
    localparam int unsigned AW = $clog2(MEM_WORDS);
    localparam int unsigned PW = $clog2(PROG_LEN);
    localparam int unsigned UW = $clog2(NMV > 1 ? NMV : 2);

    // ---- scratchpad and program ----
    logic [31:0] mem [MEM_WORDS];
    (* ram_style = "logic" *) logic [95:0] prog [PROG_LEN];
    initial $readmemh(PROG_FILE, prog);

    logic [AW-1:0] rd_addr, wr_addr;
    logic [31:0]   rd_data, wr_data;
    logic          wr_en;
    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
    assign unit_in_data = rd_data;

    // ---- current op ----
    logic [PW-1:0] pc;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [95:0]   op;
    /* verilator lint_on UNUSEDSIGNAL */
    assign op = prog[pc];
    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0]  opcode;  logic [7:0] unit;  logic [11:0] src, dst, len_in, len_out;  logic [15:0] param;  logic [3:0] flags;
    /* verilator lint_on UNUSEDSIGNAL */
    assign {flags, param, len_out, len_in, dst, src, unit, opcode} = op[79:0];
    logic [UW-1:0] ui;
    assign ui = unit[UW-1:0];

    typedef enum logic [2:0] {IDLE, FETCH, RUN, ADD, OUTP, FINISH} state_t;
    state_t state;

    logic [11:0] in_cnt, out_cnt;      // words issued / captured
    logic        in_v;                 // a read was issued last cycle -> rd_data valid now
    logic        in_first;
    logic [11:0] add_i;
    logic [1:0]  add_ph;
    logic signed [31:0] add_a;
    logic signed [31:0] best;
    logic        u_out_valid;
    logic [31:0] u_out_data;
    logic        u_busy;
    logic [$clog2(VOCAB)-1:0] token;

    // unit output mux
    always_comb begin
        u_out_valid = 1'b0; u_out_data = '0; u_busy = 1'b0;
        case (opcode)
            4'd1: begin u_out_valid = emb_out_valid; u_out_data = emb_out_data; u_busy = emb_busy; end
            4'd2: begin u_out_valid = ln_out_valid;  u_out_data = ln_out_data;  u_busy = ln_busy; end
            4'd3: begin u_out_valid = mv_out_valid[ui]; u_out_data = mv_out_data[ui]; u_busy = mv_busy[ui]; end
            4'd4: begin u_out_valid = rq_out_valid;  u_out_data = rq_out_data;  end
            4'd5: begin u_out_valid = at_out_valid;  u_out_data = at_out_data;  u_busy = at_busy; end
            default: ;
        endcase
    end

    // unit input strobes: whichever unit the op names sees the read stream
    logic feed;
    assign feed = (state == RUN) && in_v;
    assign ln_in_valid = feed && opcode == 4'd2;
    assign rq_in_valid = feed && opcode == 4'd4;
    assign rq_in_first = rq_in_valid && in_first;
    assign at_in_valid = feed && opcode == 4'd5;
    assign rq_base = param[$clog2(NPARAM)-1:0];
    assign rq_relu = flags[0];
    assign rq_width = flags[2:1];
    assign at_layer = param[3:0];
    assign emb_token = token;
    genvar g;
    generate
        for (g = 0; g < NMV; g++) begin : g_mv
            assign mv_in_valid[g] = feed && opcode == 4'd3 && ui == UW'(g);
        end
    endgenerate

    assign busy = (state != IDLE);

    // read address: streaming reads during RUN, the two operands during ADD, logits during OUTP
    always_comb begin
        rd_addr = '0;
        case (state)
            RUN:  rd_addr = AW'(src + in_cnt);
            ADD:  rd_addr = (add_ph == 2'd0) ? AW'(src + add_i) : AW'(dst + add_i);
            OUTP: rd_addr = AW'(src + in_cnt);
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; pc <= '0; pos <= '0; token <= '0; done <= 1'b0; emb_start <= 1'b0;
            in_cnt <= '0; out_cnt <= '0; in_v <= 1'b0; in_first <= 1'b0; wr_en <= 1'b0; wr_addr <= '0; wr_data <= '0;
            add_i <= '0; add_ph <= '0; add_a <= '0; best <= '0; argmax <= '0; logit_valid <= 1'b0; logit_data <= '0;
        end else begin
            done <= 1'b0; emb_start <= 1'b0; wr_en <= 1'b0; logit_valid <= 1'b0;
            in_v <= 1'b0;
            case (state)
                IDLE: begin
                    if (clear) pos <= '0;
                    if (tok_valid) begin
                        token <= tok_in; pc <= '0; state <= FETCH;
                    end
                end
                FETCH: begin
                    // op is combinational from pc; set up counters for it
                    in_cnt <= '0; out_cnt <= '0; in_first <= 1'b1;
                    case (opcode)
                        4'd0: state <= FINISH;
                        4'd1: begin emb_start <= 1'b1; state <= RUN; end
                        4'd6: begin add_i <= '0; add_ph <= '0; state <= ADD; end
                        4'd7: begin best <= 32'sh8000_0000; argmax <= '0; state <= OUTP; end
                        default: state <= RUN;
                    endcase
                end
                RUN: begin
                    // issue reads one per cycle until len_in are out
                    if (in_cnt < len_in) begin
                        in_cnt <= in_cnt + 1'b1;
                        in_v <= 1'b1;
                    end
                    if (in_v) in_first <= 1'b0;
                    // capture outputs
                    if (u_out_valid) begin
                        wr_en <= 1'b1; wr_addr <= AW'(dst + out_cnt); wr_data <= u_out_data;
                        out_cnt <= out_cnt + 1'b1;
                    end
                    if (out_cnt == len_out && in_cnt == len_in && !in_v && !u_out_valid && !u_busy) begin
                        pc <= pc + 1'b1; state <= FETCH;
                    end
                end
                ADD: begin
                    // phase 0: read src[i]; 1: read dst[i], src data arrives; 2: dst data arrives, write
                    case (add_ph)
                        2'd0: add_ph <= 2'd1;
                        2'd1: begin add_a <= signed'(rd_data); add_ph <= 2'd2; end
                        2'd2: begin
                            wr_en <= 1'b1; wr_addr <= AW'(dst + add_i);
                            wr_data <= sat16(33'(add_a) + 33'(signed'(rd_data)));
                            add_ph <= 2'd0;
                            if (add_i == len_in - 1) begin pc <= pc + 1'b1; state <= FETCH; end
                            else add_i <= add_i + 1'b1;
                        end
                        default: add_ph <= 2'd0;
                    endcase
                end
                OUTP: begin
                    if (in_cnt < len_in) begin in_cnt <= in_cnt + 1'b1; in_v <= 1'b1; end
                    if (in_v) begin
                        logit_valid <= 1'b1; logit_data <= rd_data;
                        if (signed'(rd_data) > best) begin
                            best <= signed'(rd_data);
                            argmax <= out_cnt[$clog2(VOCAB)-1:0];
                        end
                        out_cnt <= out_cnt + 1'b1;
                        if (out_cnt == len_in - 1) begin pc <= pc + 1'b1; state <= FETCH; end
                    end
                end
                FINISH: begin
                    pos <= pos + 1'b1; done <= 1'b1; state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    function automatic logic [31:0] sat16(input logic signed [32:0] x);
        logic signed [32:0] y;
        y = (x < -33'sd32768) ? -33'sd32768 : (x > 33'sd32767) ? 33'sd32767 : x;
        return y[31:0];
    endfunction
endmodule
