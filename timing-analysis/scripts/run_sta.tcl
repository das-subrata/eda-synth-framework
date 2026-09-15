# Read Liberty file
read_liberty /home/subrata/projects/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

# Read gate-level netlist
read_verilog /home/subrata/projects/eda-portfolio/synth-flow/outputs/picorv32_netlist.v

# Link design
link_design picorv32

# Read timing constraints
read_sdc /home/subrata/projects/eda-portfolio/timing-analysis/scripts/picorv32.sdc

# Report WNS and TNS
report_wns
report_tns

# Report 10 worst paths
report_checks -path_delay max \
              -fields {slew cap input nets fanout} \
              -group_count 10 \
              > /home/subrata/projects/eda-portfolio/timing-analysis/outputs/timing_report.txt

# Report to screen - 5 worst paths
report_checks -path_delay max \
              -fields {slew cap input nets fanout} \
              -group_count 5

# Also check hold timing
report_checks -path_delay min \
              -group_count 3

exit
