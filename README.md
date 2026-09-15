# EDA Synthesis Flow Automation

Automated synthesis flow for digital designs using open-source EDA tools.

## Tools
- **Yosys 0.52** — RTL synthesis (maps Verilog to standard cells)
- **OpenSTA 2.0.17** — Static timing analysis  
- **SkyWater 130nm PDK** — Open-source foundry library (422 standard cells)

## What This Does
Takes a Verilog RTL design and synthesizes it to a gate-level netlist mapped
to SkyWater 130nm standard cells, producing area and cell count statistics.

## Result: PicoRV32 RISC-V Processor
| Metric | Value |
|--------|-------|
| Total cells | 5,774 |
| Chip area | 77,250 µm² |
| Sequential elements | 57.97% |
| Runtime | 5.2 seconds |

## How to Run
```bash
yosys synth-flow/scripts/synthesize.ys
```

## Skills Demonstrated
- EDA tool scripting (Yosys synthesis flow)
- Liberty file integration (SkyWater 130nm PDK)  
- Gate-level netlist generation
- Technology mapping (RTL → standard cells)
