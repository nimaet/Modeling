"""Minimal example driver for the refactored piezo patch optimizer."""

from pathlib import Path
import sys
import numpy as np

# Edit this to the folder containing Modeling/ if needed.
PROJECT_ROOT = Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from Modeling.models.beam_properties import PiezoBeamParams, compute_EI_and_rhoA
from Modeling.models.piezo_patch_optimizer import (
    GeometrySettings,
    ObjectiveSettings,
    CircuitSettings,
    OptimizerSettings,
    PiezoPatchOptimizer,
)

# ---------------------------------------------------------------------
# Replace these section parameters with the values from your notebook.
# ---------------------------------------------------------------------
L = 0.30
b_p = 10e-3
h_s = 0.51e-3
h_p = 0.252e-3
E_s = 70e9
E_p = 1.0 / 1.5873e-11
rho_s = 2700.0
rho_p = 7600.0

base_params = PiezoBeamParams(
    b=b_p,
    hp=h_p,
    hs=h_s,
    rho_p=rho_p,
    rho_s=rho_s,
    E_s=E_s,
    s11=1.0 / E_p,
    d31=-1.48e-10,
    eps_r=1700.0,
    omega_p=2*np.pi*1.0,
    omega_q=2*np.pi*100.0,
)
base_params.zeta_p = 0.05
base_params.zeta_q = 0.20

rhoA_patch = b_p * (rho_s*h_s + 2*rho_p*h_p)
rhoA_gap = b_p * rho_s*h_s
region_types = {
    "piezo": {"EI": base_params.YI, "rhoA": rhoA_patch, "h": 10e-3},
    "substrate": {"EI": base_params.YI_s, "rhoA": rhoA_gap, "h": 10e-3},
}

geom_settings = GeometrySettings(
    Np=3,
    patch_length_bounds=(10e-3, 40e-3),
    gap_bounds=(4e-3, 80e-3),
    tip_substrate_bounds=(0.0, 150e-3),
)
mode_settings = ObjectiveSettings(
    target_mode_number=1,
    voltage_amplitude=1.0,
    phase_mode="binary",
    final_sweep_range_hz=(0.1, 10.0),
    final_sweep_n_freq=1000,
)
circuit_settings = CircuitSettings(R_c=1e3, K_p=0.02, K_i=0.0, K_c=0.0)
opt_settings = OptimizerSettings(method="random", n_random_samples=2, seed=1, show_progress=False)

optimizer = PiezoPatchOptimizer(
    L=L,
    region_types=region_types,
    base_params=base_params,
    geometry_settings=geom_settings,
    mode_settings=mode_settings,
    circuit_settings=circuit_settings,
    optimizer_settings=opt_settings,
)

result = optimizer.run()
best = optimizer.inspect_result(result)

print("Best objective:", result.fun)
print("Best score [tip displacement/V]:", best["inner"]["score"])
print("Best natural frequency [Hz]:", best["inner"]["freq_hz"])
print("Best design z [mm]:", 1e3 * result.x)
print("Patch xL [mm]:", 1e3 * best["layout"]["xL"])
print("Patch xR [mm]:", 1e3 * best["layout"]["xR"])
print("Phase mode:", best["inner"]["phase_mode"])
print("Phase [deg]:", best["inner"]["phase_deg"])
