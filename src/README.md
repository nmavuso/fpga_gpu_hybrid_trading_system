# FPGA Module (Advanced)

Contains advanced Verilog modules for:
- `order_book.v`: Real-time order book with sub-microsecond updates.
- `dma_controller.v`: A simplified DMA engine to transfer data to/from system memory via PCIe.
- `streaming_interface.v`: Streams data from the exchange feed into the order book logic.

## Synthesis Steps
1. Open Vivado/Quartus.
2. Import these Verilog files.
3. Configure:
   - Clock constraints
   - Pin assignments (PCIe lanes, memory interfaces, etc.)
   - Timing constraints to ensure sub-microsecond latency.
4. Generate the bitstream (.bit or .sof).
5. Program the FPGA hardware.

## Customizing
- Modify `order_book.v` to handle your exchange protocols, multi-level order book, etc.
- Extend `dma_controller.v` for batch or burst transfers, or a vendor-provided IP block.
- `streaming_interface.v` can be replaced with your logic for capturing external data packets (e.g., 10GbE, 25GbE).
