"""Editable set of per-simulation reducers for the reduced-npz pipeline.

reduce_nD_sweep_results.py loads each raw sim_*.npz's `data` dict (t, u_dot,
v, v_exc, ...) and calls every function in REDUCTION_SPEC on it, keeping only
the (small) scalar/vector outputs. This file is the only thing you should
need to edit to change *what* gets extracted -- the loop, parallelism, and
file I/O in reduce_nD_sweep_results.py stay the same.

Each reducer has the signature:

    fn(data: dict, params: dict, sweep_entry: dict) -> float | np.ndarray

- `data` is the sim's raw `data` dict, e.g. {"t", "u_dot", "v", "v_exc"} as
  written by generic_nD_sweep_SLURMarray.py's OUTPUT_SPEC. `u_dot` has shape
  (n_steps+1, N_mech) (mechanical DOFs); `v` has shape (n_steps+1, N_elec)
  (electrical/piezo-interface DOFs); `t` has shape (n_steps+1,).
- `params` is the sim's resolved solver params (K_c, K_i, K_p, R_c, amp, f0,
  f1, ...). For a single-tone frequency sweep (the "freq" SweepParam in
  generic_nD_sweep_SLURMarray.py), params["f0"] == params["f1"] == the swept
  frequency.
- `sweep_entry` is the raw swept value(s) for this sim (e.g. {"freq": 1500.0}).

Which DOFs get reduced is controlled explicitly by MECH_DOFS / ELEC_DOFS
below -- there is no automatic "most active DOF" guessing. Set them to the
column indices you actually care about (e.g. the drive point, a tip node, a
specific piezo interface).

Starting example, built for a frequency sweep + bifurcation-diagram workflow:

- `response_amplitude`: array of shape (len(MECH_DOFS),) -- steady-state
  peak-to-peak/2 of u_dot at each selected mechanical DOF. Plot against
  frequency for a standard FRF/backbone curve.
- `poincare_samples`: array of shape (len(MECH_DOFS), n_periods) -- a
  stroboscopic (Poincare) section of u_dot at each selected mechanical DOF,
  sampled once per forcing period over the tail of the steady-state
  response. For a period-1 response every sample lands on the same value;
  period-doubling or chaos shows up as 2, 4, ... or a scattered band of
  distinct values at that frequency. Scatter `excitation_freq` against every
  value in one DOF's row (see plot_bifurcation_from_reduced.py) to get the
  bifurcation diagram.
- `voltage_amplitude`: array of shape (len(ELEC_DOFS),) -- electrical analog
  of `response_amplitude`, steady-state peak-to-peak/2 of `v` at each
  selected electrical DOF.
- `excitation_freq`: scalar -- the x-axis for the plots above.

Keep reducers cheap and side-effect free -- they run once per sim, often
under joblib multiprocessing.
"""
from __future__ import annotations

import numpy as np

# Mechanical DOF indices to keep -- columns of u_dot / u / u_ddot. These are
# NOT raw indices into the ODE state vector `x`/`x_dot`, and NOT the same
# index space as `j_exc` (that indexes electrical/piezo channels, unrelated).
# FE_helpers.solve_newmark does `x_dot[:, :N_mech:2]` before ever handing
# `data["u_dot"]` to this module -- the `::2` stride already discards every
# rotational DOF (each free node has a [w, theta] pair; step 2 keeps only w).
# So index k here means "the k-th free node's transverse velocity/
# displacement channel" (0-indexed, after the clamped node's DOFs were
# dropped by _apply_bc) -- rotations are gone by the time this file sees it.
MECH_DOFS = [30]

# Electrical DOF indices to keep -- columns of v / q. Also not the original
# piezo-interface numbering: build_ode_system removes the `j_exc` (excited)
# channels before assembling the ODE, so v's columns are indexed over the
# remaining `idx_f` (free/sensing) piezos only, in their post-removal order.
# Index k here means "the k-th free piezo channel", which only coincides
# with the original piezo index when every excited channel's original index
# is higher than k.
ELEC_DOFS = [0]

# Fraction of the tail of the time trace treated as "steady state" (skips
# the startup transient).
STEADY_STATE_FRACTION = 0.1

# Number of forcing periods to keep in the Poincare section, counted
# backwards from the end of the simulation (i.e. always deep into steady
# state, regardless of how many periods the sim ran for). Kept small on
# purpose: this array is what gets downloaded, so it should stay orders of
# magnitude smaller than the raw time series.
N_POINCARE_PERIODS = 5000


def _steady_state(t: np.ndarray, x: np.ndarray):
    """Tail slice of `x` (and matching `t`) along axis 0, skipping the transient."""
    n = len(t)
    start = int(n * (1.0 - STEADY_STATE_FRACTION))
    return t[start:], x[start:]


def _poincare_section(t: np.ndarray, x: np.ndarray, period: float, n_periods: int) -> np.ndarray:
    """Stroboscopic samples of `x` at multiples of `period`, ending at t[-1].

    `x` may be 1D (n_steps,) or 2D (n_steps, n_dofs). Returns shape
    (n_valid,) for 1D input, or (n_dofs, n_valid) for 2D input -- one row per
    DOF/column, consistent with how MECH_DOFS selections are used elsewhere.
    `n_valid <= n_periods`: sample times that fall before the start of the
    trace are dropped rather than padded, so asking for more periods than
    the run actually contains just shrinks the result instead of erroring.
    """
    if not np.isfinite(period) or period <= 0:
        return np.array([])

    t_end = t[-1]
    sample_times = t_end - period * np.arange(n_periods)[::-1]
    sample_times = sample_times[sample_times >= t[0]]
    idx = np.clip(np.searchsorted(t, sample_times), 0, len(t) - 1)

    samples = x[idx]
    return samples.T if samples.ndim == 2 else samples


def response_amplitude(data, params, sweep_entry) -> np.ndarray:
    """Steady-state peak-to-peak/2 of u_dot at each MECH_DOFS entry (FRF/backbone amplitude)."""
    _, u_dot_ss = _steady_state(data["t"], data["u_dot"][:, MECH_DOFS])
    return 0.5 * (np.max(u_dot_ss, axis=0) - np.min(u_dot_ss, axis=0))



def poincare_samples_velocity(data, params, sweep_entry) -> np.ndarray:
    """Stroboscopic (Poincare) section of u_dot at each MECH_DOFS entry, for the bifurcation diagram."""
    f0 = float(params.get("f0", np.nan))
    period = 1.0 / f0 if f0 else np.nan
    return _poincare_section(data["t"], data["u_dot"][:, MECH_DOFS], period, N_POINCARE_PERIODS)


def poincare_samples_disp(data, params, sweep_entry) -> np.ndarray:
    """Stroboscopic (Poincare) section of u at each MECH_DOFS entry, for the bifurcation diagram."""
    f0 = float(params.get("f0", np.nan))
    period = 1.0 / f0 if f0 else np.nan
    return _poincare_section(data["t"], data["u"][:, MECH_DOFS], period, N_POINCARE_PERIODS)

def poincare_samples_voltage(data, params, sweep_entry) -> np.ndarray:
    """Stroboscopic (Poincare) section of v at each ELEC_DOFS entry, for the bifurcation diagram."""
    f0 = float(params.get("f0", np.nan))
    period = 1.0 / f0 if f0 else np.nan
    return _poincare_section(data["t"], data["v"][:, ELEC_DOFS], period, N_POINCARE_PERIODS)

def poincare_samples_flux(data, params, sweep_entry) -> np.ndarray:
    """Stroboscopic (Poincare) section of z at each ELEC_DOFS entry, for the bifurcation diagram."""
    f0 = float(params.get("f0", np.nan))
    period = 1.0 / f0 if f0 else np.nan
    return _poincare_section(data["t"], data["z"][:, ELEC_DOFS], period, N_POINCARE_PERIODS)

def voltage_amplitude(data, params, sweep_entry) -> np.ndarray:
    """Steady-state peak-to-peak/2 of v (voltage) at each ELEC_DOFS entry."""
    if "v" not in data or data["v"] is None:
        return np.full(len(ELEC_DOFS), np.nan)
    _, v_ss = _steady_state(data["t"], data["v"][:, ELEC_DOFS])
    return 0.5 * (np.max(v_ss, axis=0) - np.min(v_ss, axis=0))


def excitation_freq(data, params, sweep_entry) -> float:
    """Excitation frequency actually used by the solver -- x-axis for the plots above."""
    return float(params.get("f0", np.nan))


# Name -> reducer function. Add/remove/rename entries freely -- each run of
# reduce_nD_sweep_results.py gets its own sequential version number
# (reduced_results__v001.pkl, v002.pkl, ...), so changing this spec (or
# MECH_DOFS/ELEC_DOFS/STEADY_STATE_FRACTION/N_POINCARE_PERIODS above) and
# re-running never overwrites an older reduction of the same run.
# reduction_manifest.json records which reducers/config each version used.
REDUCTION_SPEC = {
    "response_amplitude": response_amplitude,
    "poincare_samples_velocity": poincare_samples_velocity,
    "poincare_samples_disp": poincare_samples_disp,
    "poincare_samples_voltage": poincare_samples_voltage,
    "poincare_samples_flux": poincare_samples_flux,
    "voltage_amplitude": voltage_amplitude,
    "excitation_freq": excitation_freq,
}
