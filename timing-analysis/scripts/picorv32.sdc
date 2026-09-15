# Clock definition
# PicoRV32 typically runs at 50-100 MHz on sky130
# 50 MHz = 20ns period - start conservative
create_clock -name clk -period 20.000 [get_ports clk]

# Clock uncertainty (models jitter and skew)
set_clock_uncertainty -setup 0.5 [get_clocks clk]
set_clock_uncertainty -hold  0.2 [get_clocks clk]

# Input delays - assume inputs arrive 2ns after clock edge
set_input_delay -max 2.0 -clock clk [get_ports {resetn}]
set_input_delay -max 2.0 -clock clk [get_ports {mem_ready}]
set_input_delay -max 2.0 -clock clk [get_ports {mem_rdata}]

# Output delays - outputs must be stable 2ns before next clock
set_output_delay -max 2.0 -clock clk [get_ports {trap}]
set_output_delay -max 2.0 -clock clk [get_ports {mem_valid}]
set_output_delay -max 2.0 -clock clk [get_ports {mem_addr}]

# False paths on reset (reset is async, not timed)
set_false_path -from [get_ports resetn]
