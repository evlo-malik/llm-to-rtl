// Tang Nano 20K top: the generated gpt_top behind a UART. Type a prompt into a serial
// terminal (115200 8N1 on the board's USB-C port) and end it with Enter; the board
// feeds the prompt through the model one character per token, then generates
// greedily until the 32-token context is full, printing each character as it is
// produced, then "\r\n", then waits for the next prompt. Characters the model has
// never seen are dropped. The 27 MHz crystal is the only clock.
//
// LEDs are active low on this board: led[0] model busy, led[1] generating, led[2]
// heartbeat.
module tang_nano_20k #(
    parameter int unsigned CLK_HZ = 27_000_000,
    parameter int unsigned BAUD   = 115_200,
    parameter int unsigned VOCAB  = 65,
    parameter int unsigned CTX    = 32,
    parameter              STOI_FILE = "",     // 256 x 8: ASCII -> token id, 0xff = unknown
    parameter              ITOS_FILE = ""      // VOCAB x 8: token id -> ASCII
) (
    input  logic       clk,
    input  logic       rst_i,      // S1 button, active low
    input  logic       rx,
    output logic       tx,
    output logic [5:0] led
);
    localparam int unsigned VW = $clog2(VOCAB);
    localparam int unsigned PW = $clog2(CTX + 1);

    // reset synchroniser
    logic [2:0] rst_sync;
    logic       rst_n;
    always_ff @(posedge clk) rst_sync <= {rst_sync[1:0], rst_i};
    assign rst_n = rst_sync[2];

    // uart
    logic       rx_valid, tx_send, tx_ready;
    logic [7:0] rx_data, tx_data;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (.clk(clk), .rst_n(rst_n), .rx(rx), .data_valid(rx_valid), .data(rx_data));
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (.clk(clk), .rst_n(rst_n), .send(tx_send), .data(tx_data), .ready(tx_ready), .tx(tx));

    // character tables
    logic [7:0] stoi [256];
    (* ram_style = "logic" *) logic [7:0] itos [VOCAB];
    initial begin
        $readmemh(STOI_FILE, stoi);
        $readmemh(ITOS_FILE, itos);
    end

    // prompt fifo: 64 bytes is enough for a full context
    (* ram_style = "distributed" *) logic [7:0] fifo [64];
    logic [5:0] wp, rp;
    logic       fifo_empty;
    assign fifo_empty = (wp == rp);
    always_ff @(posedge clk) if (rx_valid && rx_data != 8'h0A && rx_data != 8'h0D) fifo[wp] <= rx_data;

    // the model
    logic          tok_valid, clear, done, busy;
    logic [VW-1:0] tok_in, argmax;
    logic [PW-1:0] pos;
    gpt_top u_gpt (.clk(clk), .rst_n(rst_n), .tok_valid(tok_valid), .tok_in(tok_in), .clear(clear), .done(done),
                   .busy(busy), .pos(pos), .logit_valid(), .logit_data(), .argmax(argmax));

    typedef enum logic [3:0] {PROMPT, FEED, WAIT, GEN, EMIT, CR, LF, CLEAR} state_t;
    state_t state;
    logic       go;               // Enter seen
    logic [7:0] ch;
    logic       generating;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= PROMPT; wp <= '0; rp <= '0; go <= 1'b0; tok_valid <= 1'b0; clear <= 1'b1; tok_in <= '0;
            tx_send <= 1'b0; tx_data <= '0; ch <= '0; generating <= 1'b0;
        end else begin
            tok_valid <= 1'b0; clear <= 1'b0; tx_send <= 1'b0;
            if (rx_valid) begin
                if (rx_data == 8'h0A || rx_data == 8'h0D) go <= 1'b1;
                else wp <= wp + 1'b1;
            end
            case (state)
                PROMPT: begin
                    generating <= 1'b0;
                    if (!fifo_empty && !busy) begin
                        ch <= fifo[rp]; rp <= rp + 1'b1; state <= FEED;
                    end else if (go && fifo_empty && !busy) begin
                        go <= 1'b0;
                        if (pos == 0) state <= PROMPT;          // nothing to continue from
                        else state <= GEN;
                    end
                end
                FEED: begin
                    // ch -> token; skip unknown characters; stop feeding when the context is full
                    if (stoi[ch] == 8'hFF || pos >= PW'(CTX)) state <= PROMPT;
                    else begin tok_in <= stoi[ch][VW-1:0]; tok_valid <= 1'b1; state <= WAIT; end
                end
                WAIT: if (done) state <= (generating ? EMIT : PROMPT);
                GEN: begin
                    generating <= 1'b1;
                    if (pos >= PW'(CTX)) state <= CR;
                    else begin tok_in <= argmax; tok_valid <= 1'b1; state <= WAIT; end
                end
                EMIT: if (tx_ready) begin
                    // print the token that was just fed: that is the generated character
                    tx_data <= itos[tok_in]; tx_send <= 1'b1;
                    state <= GEN;
                end
                // end of text is "\r\n": the model can generate "\n" but never "\r"
                CR: if (tx_ready && !tx_send) begin
                    tx_data <= 8'h0D; tx_send <= 1'b1; state <= LF;
                end
                LF: if (tx_ready && !tx_send) begin
                    tx_data <= 8'h0A; tx_send <= 1'b1; state <= CLEAR;
                end
                CLEAR: begin clear <= 1'b1; generating <= 1'b0; state <= PROMPT; end
                default: state <= PROMPT;
            endcase
        end
    end

    // heartbeat
    logic [24:0] beat;
    always_ff @(posedge clk) beat <= beat + 1'b1;
    assign led = ~{3'b000, beat[24], generating, busy};
endmodule
