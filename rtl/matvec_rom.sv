// y = x @ W (+ b) with W in a ROM, pushed tile by tile through one T x T
// weight-stationary array (top.sv). x arrives serially as K int8 values; y leaves
// serially as N int32 values, in NT bursts of T with out_valid low on the padding.
//
// ROM layout, written by compiler/emit.py: one tile row per address,
//     addr = (nt*KT + kt)*T + r   holds   W[kt*T + r][nt*T + c]   at bits [WBITS*c +: WBITS]
// zero-padded past K and N. WBITS is 8, 4 or 2 (ternary), two's complement, sign-extended
// to 8 bits on the way into the array.
//
// Per tile: T cycles to shift the rows in (the T inputs for the tile are read from
// x_buf in the same cycles), one cycle for the load pipeline to drain, one cycle to
// fire the vector, 2T-1 cycles for it to come out the bottom. 3T+2 cycles for T*T
// MACs, so the array does useful work one cycle in 3T. That is what a single weight
// register per PE and one vector per tile costs; see docs/report.md.
module matvec_rom #(
    parameter int unsigned K        = 64,
    parameter int unsigned N        = 64,
    parameter int unsigned T        = 8,
    parameter int unsigned WBITS    = 8,
    parameter bit          HAS_BIAS = 1'b0,
    parameter              ROM_FILE  = "",
    parameter              BIAS_FILE = ""
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [7:0]  in_data,
    output logic        out_valid,
    output logic [31:0] out_data,
    output logic        busy
);
    localparam int unsigned KT    = (K + T - 1) / T;
    localparam int unsigned NT    = (N + T - 1) / T;
    localparam int unsigned DEPTH = KT * NT * T;
    localparam int unsigned AW    = DEPTH > 1 ? $clog2(DEPTH) : 1;
    localparam int unsigned XW    = K > 1 ? $clog2(K) : 1;
    localparam int unsigned CW    = T > 1 ? $clog2(T) : 1;
    localparam int unsigned KTW   = KT > 1 ? $clog2(KT) : 1;
    localparam int unsigned NTW   = NT > 1 ? $clog2(NT) : 1;

    // ---- weight ROM ----
    logic [T*WBITS-1:0] rom [DEPTH];
    initial $readmemh(ROM_FILE, rom);
    logic [AW-1:0]      rom_addr;
    logic [T*WBITS-1:0] rom_q;
    always_ff @(posedge clk) rom_q <= rom[rom_addr];

    logic [T*8-1:0] w_row;
    genvar c;
    generate
        for (c = 0; c < T; c++) begin : g_sext
            if (WBITS == 8) begin : g_w8
                assign w_row[8*c +: 8] = rom_q[8*c +: 8];
            end else begin : g_wn
                assign w_row[8*c +: 8] = {{(8-WBITS){rom_q[WBITS*c + WBITS-1]}}, rom_q[WBITS*c +: WBITS]};
            end
        end
    endgenerate

    // ---- control state, declared first because the ROM read below indexes on nt ----
    typedef enum logic [2:0] {IDLE, LOAD, LAST, FIRE, WAIT, EMIT} state_t;
    state_t state;
    logic [NTW-1:0] nt;
    logic [KTW-1:0] kt;
    logic [CW-1:0]  cnt, cnt_d;
    logic           load_d, pad, pad_d;
    logic signed [31:0] acc [T];

    // ---- bias ROM, one T-lane word per column tile ----
    logic [T*32-1:0] bias_q;
    generate
        if (HAS_BIAS) begin : g_bias
            logic [T*32-1:0] bias_rom [NT];
            initial $readmemh(BIAS_FILE, bias_rom);
            always_ff @(posedge clk) bias_q <= bias_rom[nt];
        end else begin : g_nobias
            assign bias_q = '0;
        end
    endgenerate

    // ---- input buffer ----
    (* ram_style = "distributed" *) logic [7:0] x_buf [K];
    logic [XW-1:0] k_in;
    logic [XW-1:0] x_addr;
    logic [7:0]    x_q;
    always_ff @(posedge clk) begin
        if (in_valid) x_buf[k_in] <= in_data;
        x_q <= x_buf[x_addr];
    end

    // ---- the array ----
    logic           w_load, x_valid, y_valid;
    /* verilator lint_off UNUSEDSIGNAL */
    logic           arr_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [T*8-1:0] x_sr;
    logic [T*32-1:0] y_vec;
    top #(.N(T)) u_arr (
        .clk(clk), .rst_n(rst_n),
        .w_load(w_load), .w_row(w_row),
        .x_valid(x_valid), .x_vec(x_sr),
        .y_valid(y_valid), .y_vec(y_vec), .busy(arr_busy)
    );


    // tile row address and input address, both combinational from the counters
    logic [31:0] tile_base, x_idx;
    assign tile_base = (32'(nt) * KT + 32'(kt)) * T;
    assign rom_addr  = AW'(tile_base + T - 1 - 32'(cnt));
    assign x_idx     = 32'(kt) * T + 32'(cnt);
    assign pad    = (x_idx >= K);
    assign x_addr = pad ? '0 : XW'(x_idx);

    assign w_load  = load_d;
    assign x_valid = (state == FIRE);
    assign busy    = (state != IDLE) || (k_in != 0);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; k_in <= '0; nt <= '0; kt <= '0; cnt <= '0;
            load_d <= 1'b0; cnt_d <= '0; pad_d <= 1'b0; x_sr <= '0;
            out_valid <= 1'b0; out_data <= '0;
            for (int i = 0; i < T; i++) acc[i] <= '0;
        end else begin
            // load pipeline: rom_q / x_q trail the counters by one cycle
            load_d <= (state == LOAD);
            cnt_d  <= cnt;
            pad_d  <= pad;
            if (load_d) x_sr[8*cnt_d +: 8] <= pad_d ? 8'd0 : x_q;
            out_valid <= 1'b0;

            case (state)
                IDLE: if (in_valid) begin
                    if (k_in == XW'(K - 1)) begin
                        k_in <= '0; nt <= '0; kt <= '0; cnt <= '0;
                        state <= LOAD;
                    end else k_in <= k_in + 1'b1;
                end
                LOAD: begin
                    if (cnt == CW'(T - 1)) begin cnt <= '0; state <= LAST; end
                    else cnt <= cnt + 1'b1;
                end
                LAST: state <= FIRE;
                FIRE: state <= WAIT;
                WAIT: if (y_valid) begin
                    for (int i = 0; i < T; i++)
                        acc[i] <= (kt == 0 ? signed'(bias_q[32*i +: 32]) : acc[i]) + signed'(y_vec[32*i +: 32]);
                    if (kt == KTW'(KT - 1)) begin cnt <= '0; state <= EMIT; end
                    else begin kt <= kt + 1'b1; state <= LOAD; end
                end
                EMIT: begin
                    // one lane per cycle; the last tile stops at column N-1, so no
                    // padding lanes are ever emitted
                    out_valid <= 1'b1;
                    out_data  <= acc[cnt];
                    if (32'(nt) * T + 32'(cnt) == N - 1) begin
                        cnt <= '0; nt <= '0; state <= IDLE;
                    end else if (cnt == CW'(T - 1)) begin
                        cnt <= '0; nt <= nt + 1'b1; kt <= '0; state <= LOAD;
                    end else cnt <= cnt + 1'b1;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
