# 2024-12-20 Weekly Meeting

Source: `Old_sources/Energy harvesting 20241220.pptx`

Slides: [slides.pptx](slides.pptx)

## Summary

- Focused on FEM simulations for the beam harvester to test whether the ROM predictions hold in a continuous model.
- Used 40 beam elements with Newmark-beta time integration and Newton-Raphson iterations for the nonlinear algebraic system.
- Found that FEM and ROM did not always agree on whether the system reaches the lower branch, higher branch, or a branch transition.
- The continuous model still showed transition behavior, but the nonlinear threshold differed from the ROM prediction.
- A working hypothesis was that the ROM overestimates the relevant first electromechanical mode frequency.

## Follow-Up

- Remove nonlinearity first to debug FEM validity.
- Study how nonlinearity affects effective damping and modal frequency.
- Revisit the ROM mode-shape assumption.
