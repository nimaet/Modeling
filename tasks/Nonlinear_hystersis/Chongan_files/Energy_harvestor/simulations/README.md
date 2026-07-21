# MATLAB Simulation Archive

This folder contains the cleaned working archive of MATLAB files copied from
`Old_sources/`. The historical archive remains untouched.

## Main Line

- `fem/continuous_harvester.m` is the continuous beam FEM time simulation.
- `fem/eigenanalysis_corrected_single_mode.m` computes the FEM-informed first
  electromechanical mode used to correct the ROM ansatz.
- `fem/FEM_modes_norm.mat` stores the corrected scalar mode data used by the
  main ROM scripts.
- `rom/corrected_fem_mode_rom.m` is the main corrected reduced-order model.
- `rom/corrected_hysteresis_with_damping.m` is the corrected ROM variant with
  structural damping in the hysteresis calculation.

The important correction is not the old analytic first bending mode. The
corrected ROM loads `FEM_modes_norm.mat` and uses:

```matlab
omega1 = omega1_FEM;
phi_r_prime_l_2 = - phi_r_prime_l_2_FEM;
phi_r_prime_l = - phi_r_prime_l_FEM;
int_0_l_phi_r = - int_0_l_phi_r_FEM;
```

This matches the January 17 presentation story: the earlier ROM disagreed with
the full beam simulation because the assumed mode shape ignored the harvester's
effect on the mode.

## Supporting Branches

- `robustness/second_harmonics/` contains the nonlinear and linear second
  harmonic robustness sweeps used around the February 14 presentation.
- `legacy_reference/` keeps the older analytic-mode ROM and linear comparison
  scripts as references only. Do not use them as the main corrected model.

## Source Map

| New file | Source |
| --- | --- |
| `fem/continuous_harvester.m` | `Old_sources/Continuous model/Continuous_harvester.m` |
| `fem/eigenanalysis_corrected_single_mode.m` | `Old_sources/Continuous model/Eigenanalysis.m` |
| `fem/FEM_modes_norm.mat` | `Old_sources/Continuous model/FEM_modes_norm.mat` |
| `rom/corrected_fem_mode_rom.m` | `Old_sources/Continuous model/Energy_harvesting_FEM_mode.m` |
| `rom/corrected_hysteresis_with_damping.m` | `Old_sources/Continuous model/Hysteresis_wt_damping.m` |
| `robustness/second_harmonics/nonlinear_second_harmonic_sweep.m` | `Old_sources/Robustness_study/Second_harmonics/Energy_harvesting_FEM_robustness_of_secondary.m` |
| `robustness/second_harmonics/linear_second_harmonic_sweep.m` | `Old_sources/Robustness_study/Second_harmonics/Energy_harvesting_FEM_robustness_of_secondary_linear.m` |
| `legacy_reference/original_analytic_mode_rom.m` | `Old_sources/Energy_harvesting.m` |
| `legacy_reference/original_linear_comparison_rom.m` | `Old_sources/Energy_harvesting_linear.m` |

## Notes

- `Old_sources/Continuous model/Energy_harvesting_FEM_sent_to_Mohammad.m` was
  not selected as the main ROM because it references `varphi_Lambda_1_num`
  after the local calculation of that variable is commented out.
- `Old_sources/Multi_DOF/` appears to be a later multi-mode exploration. Its
  `FEM_modes_norm.mat` stores vector mode data, while the main corrected ROM
  scripts expect scalar first-mode data. Keep it separate until the ROM is
  explicitly refactored for multiple modes.
