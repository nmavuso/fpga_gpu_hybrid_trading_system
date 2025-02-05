// streaming_interface.v
// Receives market data packets from an external interface (Ethernet, feed handler, etc.)
// and streams them into the order_book module.

`timescale 1ns/1ps

module streaming_interface(
    input  wire        clk,
    input  wire        reset_n,
    // External feed input
    input  wire [127:0] packet_in,
    input  wire        packet_in_valid,
    // Output to order_book
    output wire [31:0] market_data_out,
    output wire        market_data_valid
);

    // Simple placeholder: take lower 32 bits as "market_data_out"
    assign market_data_out   = packet_in[31:0];
    assign market_data_valid = packet_in_valid;

endmodule
