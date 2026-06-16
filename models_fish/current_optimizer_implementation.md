# Current Piezo Optimizer Implementation Before Simplification

This note records the optimizer structure before stripping the multi-mode
normalization and composite traveling-wave score.

## Core Optimizer

- `PiezoPatchOptimizer` optimizes patch geometry with `random`, `powell`,
  `random_powell`, or `differential_evolution`.
- The outer design vector is:
  `z = [L1, gap12, L2, gap23, ..., L_Np]`.
- Each valid design builds a `PiezoBeamFE`, then delegates to an inner objective.
- The outer SciPy objective returns `-score` for valid designs and a large
  penalty for invalid geometry or failed FE/objective evaluation.

## Standing-Wave Objectives

- `single_mode` evaluates one natural mode from `standing_wave_settings`.
- `multi_mode` evaluates a tuple of mode numbers with the same geometry.
- Each mode gets its own phase optimization.
- Binary phase mode brute-forces every `+/-` sign pattern.
- Continuous phase mode uses analytic phase alignment for tip output and
  multi-start `scipy.optimize.minimize` for spatial metrics.
- Multi-mode currently combines scores using optional weights and optional
  score normalizers, returning raw, normalized, and weighted score arrays.

## Traveling-Wave Objective

- `traveling_wave` currently evaluates one fixed frequency.
- The frequency is either `frequency_hz` or an interpolation between a
  `mode_pair` using `frequency_fraction`.
- Phase optimization is binary brute force or continuous multi-start optimize.
- Traveling quality is a composite score:
  traveling-index term times amplitude score times envelope score times
  direction score.
- Metrics include traveling index, RMS amplitude, envelope variation, phase
  slope, direction score, and full/windowed complex shapes.

## Post-Processing

- `postprocess.py` provides dense metric FRF sweeps and traveling-wave metric
  sweeps for plotting.
- `plotting.py` visualizes layout, FRFs, phase patterns, mode shapes, multi-mode
  summaries, and traveling-wave shapes.
- `runner.py` provides optional Cartesian sweep utilities for notebooks.
