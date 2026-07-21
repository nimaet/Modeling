# 2025-01-17 Weekly Meeting

Source: `Old_sources/Energy harvesting 20250117.pptx`

Slides: [slides.pptx](slides.pptx)

## Summary

- Investigated why ROM and full FEM simulations disagreed.
- Focused on the linear regime to test the original ROM ansatz.
- Compared natural-frequency estimates from the full model and the ROM.
- Identified that assuming the beam vibrates in the ordinary first bending mode introduces frequency error because the piezoelectric harvester modifies the mode shape.
- Updated the ROM using an FEM-simulated mode shape that includes the harvester effect.
- The corrected ROM improved agreement with full beam simulations.

## Follow-Up

- Use the FEM-informed mode consistently in subsequent ROM calculations.
- Recompute branch thresholds and power-flow metrics with the corrected ROM.
