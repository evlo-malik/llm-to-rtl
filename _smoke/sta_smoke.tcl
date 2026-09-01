read_liberty syn/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog syn/reports/counter_netlist.v
link_design counter
create_clock -name clk -period 10 [get_ports clk]
set_input_delay  1 -clock clk [get_ports {rst_n en}]
set_output_delay 1 -clock clk [all_outputs]
report_checks -path_delay max -digits 4
report_wns
report_tns
