"""Minimal usage example for the multimode piezo patch optimizer."""

from beam_properties import PiezoBeamParams
from Modeling.models_fish.piezo_patch_optimizer_multimode import (
    build_region_types_from_params,
    GeometrySettings,
    ObjectiveSettings,
    CircuitSettings,
    OptimizerSettings,
    PiezoPatchOptimizer,
)

base_params = PiezoBeamParams()
L = 0.3185  # replace with your desired total beam length if different
region_types = build_region_types_from_params(base_params, h_patch=1e-3, h_gap=1e-3)

geom_settings = GeometrySettings(
    Np=3,
    patch_length_bounds=(10e-3, 80e-3),
    gap_bounds=(3e-3, 40e-3),
    tip_substrate_bounds=(0.0, L),
)

# Single-mode example
objective_settings = ObjectiveSettings(
    objective="single_mode",
    single_mode_number=1,
    phase_mode="continuous",
    output="tip",
    voltage_amplitude=1.0,
    final_sweep_range_hz=(0.1, 40.0),
    final_sweep_n_freq=1000,
)

# Multi-mode example: uncomment this block instead of the single-mode block above.
# objective_settings = ObjectiveSettings(
#     objective="multi_mode",
#     multi_mode_numbers=(1, 2, 3),
#     multi_mode_weights=(1.0, 1.0, 1.0),
#     multi_mode_reduction="weighted_sum",
#     phase_mode="continuous",
#     output="tip",
#     voltage_amplitude=1.0,
#     final_sweep_range_hz=(0.1, 40.0),
#     final_sweep_n_freq=1000,
# )

optimizer = PiezoPatchOptimizer(
    L=L,
    region_types=region_types,
    base_params=base_params,
    geometry_settings=geom_settings,
    objective_settings=objective_settings,
    circuit_settings=CircuitSettings(R_c=1e3, K_p=0.02, K_i=0.0, K_c=0.0),
    optimizer_settings=OptimizerSettings(method="random", n_random_samples=50, seed=2),
)

result = optimizer.run()
best = optimizer.inspect_result(result)

print("Best score:", best["inner"]["score"])
print("Objective:", best["inner"]["objective"])
print("Frequencies [Hz]:", best["inner"]["freq_hz"])
print("Best design [mm]:", 1e3 * result.x)
print("Patch xL [mm]:", 1e3 * best["layout"]["xL"])
print("Patch xR [mm]:", 1e3 * best["layout"]["xR"])
print("Phase [deg]:", best["inner"]["phase_deg"])
