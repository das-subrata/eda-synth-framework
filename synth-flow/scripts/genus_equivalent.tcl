# =============================================================
# Genus Equivalent of synthesize.ys
# =============================================================
# This file shows the Cadence Genus commands that correspond
# to each step in the Yosys synthesis script (synthesize.ys).
# Both scripts produce the same result: a gate-level netlist
# mapped to a standard cell library.
# =============================================================

# -------------------------------------------------------------
# STEP 1: Load standard cell library
# -------------------------------------------------------------
# Yosys:  read_liberty -ignore_miss_func sky130.lib
# Genus:
set_db init_lib_search_path /path/to/libs
set_db library {sky130_fd_sc_hd__ss_n40C_1v28.lib \
                sky130_fd_sc_hd__tt_025C_1v80.lib \
                sky130_fd_sc_hd__ff_100C_1v95.lib}
# Note: Genus takes multiple corners at once (MMMC)
# Yosys uses one Liberty file at a time

# -------------------------------------------------------------
# STEP 2: Read RTL design
# -------------------------------------------------------------
# Yosys:  read_verilog /path/to/picorv32.v
# Genus:
read_hdl -sv /path/to/picorv32.v
# Note: Genus uses read_hdl; -sv flag enables SystemVerilog
# Yosys uses read_verilog; -sv flag for SystemVerilog

# -------------------------------------------------------------
# STEP 3: Elaborate (build design hierarchy)
# -------------------------------------------------------------
# Yosys:  hierarchy -check -top picorv32
# Genus:
elaborate picorv32
# Note: Both commands resolve module hierarchy and
# identify the top-level module

# -------------------------------------------------------------
# STEP 4: Read timing constraints
# -------------------------------------------------------------
# Yosys:  (SDC read happens in OpenSTA separately)
# Genus:
read_sdc /path/to/picorv32.sdc
# Note: Genus reads SDC during synthesis for
# timing-driven optimization. Yosys does not use
# SDC during synthesis - timing is checked separately
# in OpenSTA (equivalent to Tempus)

# -------------------------------------------------------------
# STEP 5: Generic synthesis (technology independent)
# -------------------------------------------------------------
# Yosys:  synth -top picorv32  (includes all steps below)
# Genus:
set_db syn_generic_effort high
syn_generic
# Note: Maps RTL to technology-independent logic gates
# (AND, OR, NOT, MUX, FF) without using the Liberty library

# -------------------------------------------------------------
# STEP 6: Technology mapping (map to standard cells)
# -------------------------------------------------------------
# Yosys:  dfflibmap + abc  (two separate commands)
# Genus:
set_db syn_map_effort high
syn_map
# Note: Maps the generic gates to actual standard cells
# from the Liberty library. Genus does this in one command.
# Yosys separates flip-flop mapping (dfflibmap) from
# combinational mapping (abc)

# -------------------------------------------------------------
# STEP 7: Optimization
# -------------------------------------------------------------
# Yosys:  (basic optimization included in synth command)
# Genus:
set_db syn_opt_effort high
set_db lp_insert_clock_gating true     # Enable clock gating
set_db lp_multi_vt_optimization_effort high  # Multi-Vt
syn_opt
# Note: Genus has a dedicated optimization pass with
# advanced features (clock gating, multi-Vt cell swapping)
# that Yosys does not support natively

# -------------------------------------------------------------
# STEP 8: Reports
# -------------------------------------------------------------
# Yosys:  stat -liberty sky130.lib
# Genus:
report_timing > timing_report.rpt
report_area   > area_report.rpt
report_power  > power_report.rpt
report_qor    > qor_report.rpt
# Note: Genus has separate, detailed reports for each metric
# Yosys stat gives a combined summary

# -------------------------------------------------------------
# STEP 9: Write gate-level netlist
# -------------------------------------------------------------
# Yosys:  write_verilog picorv32_netlist.v
# Genus:
write_hdl > picorv32_netlist.v
write_sdc > picorv32_synth.sdc
# Note: Genus also writes a mapped SDC with updated
# timing constraints for the P&R tool (Innovus)

# =============================================================
# KEY DIFFERENCES SUMMARY
# =============================================================
# 1. Genus is timing-driven during synthesis (reads SDC early)
#    Yosys optimizes without timing awareness
#
# 2. Genus supports MMMC (multiple corners simultaneously)
#    Yosys uses one Liberty file at a time
#
# 3. Genus has advanced low-power features (clock gating,
#    multi-Vt) built in. Yosys requires manual steps.
#
# 4. Genus integrates with Innovus (P&R) via the .db format
#    Yosys uses standard Verilog netlist handoff
#
# 5. Both produce functionally equivalent gate-level netlists
#    mapped to the same standard cell library
# =============================================================
