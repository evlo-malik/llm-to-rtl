// Dynamic Q/K/V arithmetic. All numerical parameters are compile-time constants.
module attention #(
    parameter int unsigned D = 64,
    parameter int unsigned H = 4,
    parameter int unsigned T = 32,
    parameter int unsigned L = 2,
    parameter logic [44*L-1:0] PARAMS = '0,
    parameter logic [17*256-1:0] EXP_TABLE = '0
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [$clog2(L > 1 ? L : 2)-1:0] layer,
    input  logic [$clog2(T)-1:0] pos,
    input  logic        in_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0] in_data,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic        out_valid,
    output logic [31:0] out_data,
    output logic        busy
);
    localparam int unsigned HD  = D / H;
    localparam int unsigned TW  = $clog2(T);
    localparam int unsigned DW  = $clog2(D);
    localparam int unsigned HW  = $clog2(H > 1 ? H : 2);
    localparam int unsigned CW  = $clog2(HD > 1 ? HD : 2);
    localparam int unsigned KW  = $clog2(L * T * D);
    localparam int unsigned IW  = $clog2(3 * D);

    logic [43:0] prm;
    logic [15:0] m0u, m0o;
    logic [5:0]  nu, no;
    assign prm = PARAMS[44*layer +: 44];
    assign m0u = prm[15:0];
    assign nu  = prm[21:16];
    assign m0o = prm[37:22];
    assign no  = prm[43:38];

    // ---- storage ----
    logic signed [7:0]  q_buf [D];
    logic signed [7:0]  kc [L * T * D];
    logic signed [7:0]  vc [L * T * D];
    logic signed [31:0] s_buf [T];
    logic        [16:0] e_buf [T];
    logic        [15:0] p_buf [T];

    typedef enum logic [2:0] {FILL, SCORE, SMAX, RECIP, PROB, PV, DONE} state_t;
    state_t state;

    logic [IW-1:0] fill_i;
    logic [HW-1:0] hh;
    logic [TW-1:0] i;          // position counter for the current sweep
    logic [CW-1:0] c;          // channel within the head
    logic          iss;        // an address is being issued this cycle
    logic          iss_last;   // ... and it is the last of the sweep

    // stage-1 flags (data from the RAMs is valid)
    logic          d_v, d_last, d_lastc, d_lasti;
    logic [TW-1:0] d_i;

    logic signed [7:0]  q_q, k_q, v_q;
    logic signed [31:0] s_q;
    logic        [16:0] e_q;
    logic        [15:0] p_q;
    logic        [16:0] lut_q;

    logic signed [31:0] mac;
    logic signed [31:0] smax;
    logic               first_s;
    localparam integer EW = $clog2(T+1)+17;
    logic [EW-1:0] esum;
    /* verilator lint_off UNUSEDSIGNAL */
    logic        [36:0] recip;      // only the low 21 bits can be nonzero: sum >= 2^16
    /* verilator lint_on UNUSEDSIGNAL */
    logic               dv_start, dv_done;
    logic        [36:0] dv_q;

    // softmax pipeline registers
    logic        u_v, l_v;
    logic [7:0]  u_r;
    logic [TW-1:0] u_i, l_i;
    // output pipeline registers
    logic        r1_v, r2_v;
    logic signed [31:0] oacc;
    logic signed [48:0] oprod;

    udiv #(.NW(37), .DW(EW)) u_div (.clk(clk), .rst_n(rst_n), .start(dv_start),
                                    .num(37'h10_0000_0000), .den(esum), .done(dv_done), .quo(dv_q));

    // ---- RAM ports: one read per RAM per cycle, addresses from the sweep counters ----
    logic [KW-1:0] kv_addr;
    assign kv_addr = KW'((32'(layer) * T + 32'(i)) * D + 32'(hh) * HD + 32'(c));

    always_ff @(posedge clk) begin
        // writes during FILL
        if (in_valid && state == FILL) begin
            if (fill_i < IW'(D))
                q_buf[fill_i[DW-1:0]] <= in_data[7:0];
            else if (fill_i < IW'(2 * D))
                kc[KW'((32'(layer) * T + 32'(pos)) * D) + KW'(fill_i - IW'(D))] <= in_data[7:0];
            else
                vc[KW'((32'(layer) * T + 32'(pos)) * D) + KW'(fill_i - IW'(2 * D))] <= in_data[7:0];
        end
        // reads
        q_q <= q_buf[DW'(32'(hh) * HD + 32'(c))];
        k_q <= kc[kv_addr];
        v_q <= vc[kv_addr];
        s_q <= s_buf[i];
        e_q <= e_buf[i];
        p_q <= p_buf[i];
        lut_q <= EXP_TABLE[17*u_r +: 17];
        // small buffers written from the pipelines
        if (d_v && d_lastc && state == SCORE) s_buf[d_i] <= mac + 32'(q_q) * 32'(k_q);
        if (l_v) e_buf[l_i] <= lut_q;
        if (d_v && state == PROB) p_buf[d_i] <= 16'((38'(e_q) * 38'(recip[20:0])) >> 21);
    end

    assign busy = (state != FILL) || (fill_i != 0);
    assign iss_last = (state == SCORE) ? (i == pos && c == CW'(HD - 1)) :
                      (state == PV)    ? (i == pos && c == CW'(HD - 1)) :
                                         (i == pos);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= FILL; fill_i <= '0; hh <= '0; i <= '0; c <= '0; iss <= 1'b0;
            d_v <= 1'b0; d_last <= 1'b0; d_lastc <= 1'b0; d_lasti <= 1'b0; d_i <= '0;
            mac <= '0; smax <= '0; first_s <= 1'b1; esum <= '0; recip <= '0; dv_start <= 1'b0;
            u_v <= 1'b0; l_v <= 1'b0; u_r <= '0; u_i <= '0; l_i <= '0;
            r1_v <= 1'b0; r2_v <= 1'b0; oacc <= '0; oprod <= '0; out_valid <= 1'b0; out_data <= '0;
        end else begin
            dv_start <= 1'b0;
            // stage-1 flags follow the issue counters by one cycle
            d_v     <= iss;
            d_last  <= iss && iss_last;
            d_lastc <= (c == CW'(HD - 1));
            d_lasti <= (i == pos);
            d_i     <= i;

            // sweep counters. SCORE and PV walk (i, c) with c fastest for SCORE and
            // i fastest for PV; SMAX and PROB walk i only.
            if (iss) begin
                case (state)
                    SCORE: begin
                        if (c == CW'(HD - 1)) begin c <= '0; i <= i + 1'b1; end
                        else c <= c + 1'b1;
                    end
                    PV: begin
                        if (i == pos) begin i <= '0; c <= c + 1'b1; end
                        else i <= i + 1'b1;
                    end
                    default: i <= i + 1'b1;
                endcase
                if (iss_last) iss <= 1'b0;
            end

            // softmax pipeline: s_q -> u_r -> lut_q -> e_buf/esum
            u_v <= d_v && state == SMAX;
            u_i <= d_i;
            if (d_v && state == SMAX) u_r <= u_of(smax - s_q);
            l_v <= u_v;
            l_i <= u_i;
            if (l_v) esum <= esum + EW'(lut_q);

            // output pipeline: mac -> oacc -> oprod -> out
            r1_v <= d_v && d_lasti && state == PV;
            if (d_v && d_lasti && state == PV) oacc <= mac + 32'(p_q) * 32'(v_q);
            r2_v <= r1_v;
            oprod <= oacc * signed'({1'b0, m0o}) + (no == 0 ? 49'sd0 : (49'sd1 <<< (no - 1)));
            out_valid <= r2_v;
            if (r2_v) out_data <= clip8(oprod >>> no);

            case (state)
                FILL: if (in_valid) begin
                    if (fill_i == IW'(3 * D - 1)) begin
                        fill_i <= '0; hh <= '0; i <= '0; c <= '0; iss <= 1'b1;
                        mac <= '0; first_s <= 1'b1; state <= SCORE;
                    end else fill_i <= fill_i + 1'b1;
                end
                SCORE: begin
                    if (d_v) begin
                        if (d_lastc) begin
                            mac <= '0;
                            smax <= (first_s || (mac + 32'(q_q) * 32'(k_q)) > smax) ? mac + 32'(q_q) * 32'(k_q) : smax;
                            first_s <= 1'b0;
                        end else mac <= mac + 32'(q_q) * 32'(k_q);
                    end
                    if (d_last) begin
                        // smax lands this edge; start the softmax sweep next cycle
                        i <= '0; c <= '0; iss <= 1'b1; esum <= '0; state <= SMAX;
                    end
                end
                SMAX: begin
                    // wait for the last e to be summed: l_v with l_i == pos
                    if (l_v && l_i == pos) begin
                        dv_start <= 1'b1; state <= RECIP;
                    end
                end
                RECIP: if (dv_done) begin
                    recip <= dv_q;
                    i <= '0; iss <= 1'b1; state <= PROB;
                end
                PROB: if (d_v && d_last) begin
                    i <= '0; c <= '0; iss <= 1'b1; mac <= '0; state <= PV;
                end
                PV: begin
                    if (d_v) mac <= d_lasti ? '0 : mac + 32'(p_q) * 32'(v_q);
                    if (d_last) begin
                        if (hh == HW'(H - 1)) state <= DONE;
                        else begin
                            hh <= hh + 1'b1; i <= '0; c <= '0; iss <= 1'b1; mac <= '0; first_s <= 1'b1;
                            state <= SCORE;
                        end
                    end
                end
                DONE: if (!r1_v && !r2_v && !out_valid) state <= FILL;   // let the last output leave
                default: state <= FILL;
            endcase
        end
    end

    function automatic logic [7:0] u_of(input logic signed [31:0] z);
        logic signed [48:0] t;
        t = (z * signed'({1'b0, m0u}) + (nu == 0 ? 49'sd0 : (49'sd1 <<< (nu - 1)))) >>> nu;
        return (t < 0) ? 8'd0 : (t > 49'sd255) ? 8'd255 : t[7:0];
    endfunction
    function automatic logic [31:0] clip8(input logic signed [48:0] x);
        logic signed [48:0] y;
        y = (x < -49'sd128) ? -49'sd128 : (x > 49'sd127) ? 49'sd127 : x;
        return y[31:0];
    endfunction
endmodule
