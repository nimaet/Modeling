"""
piezo_patch_optimizer_starter.py

Starter optimizer for piezo patch placement using the FE3.py / FE_helpers.py framework.

Core idea
---------
For each candidate geometry:
  1. Build a piecewise Euler-Bernoulli FE model using FE3.build_geometry_arbitrary_piezos.
  2. Assemble K, M, C, Gamma using PiezoBeamFE.
  3. For each target mode:
       - sweep a mode-specific frequency window,
       - compute each patch's complex contribution to the output,
       - choose the best mode-specific phase/sign pattern,
       - keep the best amplitude over that frequency window.
  4. Combine modal amplitudes into a multi-mode objective.

This starter uses a direct mechanical FRF:
    (K + i*w*C - w^2*M) u_hat = Gamma * v_hat

That is the fastest version for voltage-actuated linear FRFs.
If you later need electrical shunts/free piezo DOFs, replace the direct solve
with FE.build_ode_system(...) + FE_helpers.frequency_response_linear(...).

Author notes
------------
- Patch number is handled by an outer loop.
- Patch lengths and gaps are encoded using a feasible gap/length parameterization.
- Each mode gets its own optimal actuation frequency and phase/sign vector.
- Add your system-specific constraints in `user_geometry_penalty`.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult


# -------------------------------------------------------------------------
# Imports: adjust these to match your project structure if needed.
# -------------------------------------------------------------------------
try:
    from Modeling.models.FE3 import PiezoBeamFE, build_geometry_arbitrary_piezos
    from Modeling.models.beam_properties import PiezoBeamParams
except ImportError:
    # If this script lives in the same folder as FE3.py and beam_properties.py.
    from FE3 import PiezoBeamFE, build_geometry_arbitrary_piezos
    from beam_properties import PiezoBeamParams


# -------------------------------------------------------------------------
# User-editable configuration
# -------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """
    Main user-facing settings.

    Geometry is parameterized as:
        root_gap -> patch_1 -> gap_1 -> patch_2 -> ... -> patch_N -> tip_gap

    The optimizer directly controls patch lengths. The remaining beam length is
    distributed among root/internal/tip gaps using gap allocation variables.
    """

    # -----------------------------
    # Global beam / mesh settings
    # -----------------------------
    L: Optional[float] = None              # Beam length [m]. If None, uses params.L_b.
    h_patch: float = 1.0e-3               # FE element size in patched regions [m]
    h_gap: float = 1.0e-3                 # FE element size in gap regions [m]

    # -----------------------------
    # Patch-count settings
    # -----------------------------
    allowed_n_patches: tuple[int, ...] = (1, 2, 3, 4)
    # You can set this to (4,) if you want exactly four patches.

    # -----------------------------
    # Patch length constraints
    # -----------------------------
    length_min: float = 5.0e-3            # minimum patch length [m]
    length_max: float = 70.0e-3           # maximum patch length [m]

    # Optional per-patch length bounds.
    # Example for Np=4:
    # length_bounds_by_patch = [(10e-3, 40e-3), (10e-3, 40e-3), ...]
    length_bounds_by_patch: Optional[list[tuple[float, float]]] = None

    # Optional fixed lengths by zero-based patch index.
    # Example: {0: 20e-3, 3: 10e-3}
    fixed_lengths: dict[int, float] = field(default_factory=dict)

    # -----------------------------
    # Gap constraints
    # -----------------------------
    gap_min: float = 2.0e-3               # minimum internal gap between patches [m]
    root_gap_min: float = 0.0             # minimum gap from clamp to first patch [m]
    tip_gap_min: float = 0.0              # minimum gap from last patch to tip [m]

    # Optional total patch material limit.
    # If None, only length/gap constraints are used.
    max_total_patch_length: Optional[float] = None

    # -----------------------------
    # Modal objective settings
    # -----------------------------
    # Each tuple is (f_min, f_max) in Hz.
    # These are mode-specific frequency windows.
    mode_windows_hz: tuple[tuple[float, float], ...] = (
        (0.5, 3.0),
        (2.0, 6.0),
        (5.0, 10.0),
    )

    mode_weights: tuple[float, ...] = (1.0, 1.0, 1.0)

    # Frequency grid points per mode window.
    n_freq_per_mode: int = 151

    # Normalize each modal amplitude before combining.
    # Useful if mode 1 is naturally much larger than mode 2/3.
    # Example: baseline_modal_amplitudes=(A1_bare, A2_bare, A3_bare)
    baseline_modal_amplitudes: Optional[tuple[float, ...]] = None

    # -----------------------------
    # Actuation settings
    # -----------------------------
    voltage_amp: float = 1.0              # Use 1 V during optimization; scale later.

    # phase_mode:
    #   "continuous" : arbitrary phase per patch, chosen analytically
    #   "binary"     : each patch is +V or -V, brute-force for small Np
    #   "fixed"      : use user-provided fixed_phase_patterns
    phase_mode: str = "binary"

    # For binary phase mode, brute force up to this many patches.
    # For larger Np, a simple greedy sign optimizer is used.
    max_binary_bruteforce_patches: int = 12

    # Fixed phase/sign patterns for phase_mode="fixed".
    # Dict key = mode index 0,1,2,...
    # Value can be:
    #   signs, e.g. np.array([1, -1, -1, 1])
    #   or complex phases, e.g. np.exp(1j*np.array([0, np.pi, np.pi, 0]))
    fixed_phase_patterns: Optional[dict[int, np.ndarray]] = None

    # -----------------------------
    # Output metric
    # -----------------------------
    # Currently robustly implemented:
    #   "tip_displacement"
    #   "tip_velocity"
    # You can add line-average/RMS later in get_output_contributions().
    output_metric: str = "tip_displacement"

    # -----------------------------
    # Damping model
    # -----------------------------
    # "modal"               : use fe.C_red only
    # "modal_plus_rayleigh" : use fe.C_red + c_alpha*M + c_beta*K, matching FE3.build_ode_system logic
    # "rayleigh"            : use c_alpha*M + c_beta*K only
    damping_model: str = "modal_plus_rayleigh"

    # -----------------------------
    # Optimizer settings
    # -----------------------------
    differential_evolution_maxiter: int = 40
    differential_evolution_popsize: int = 12
    differential_evolution_tol: float = 1e-3
    seed: Optional[int] = 1

    # Large penalty value for invalid geometries.
    invalid_penalty: float = 1e30


# -------------------------------------------------------------------------
# Constraint hook: add project-specific constraints here
# -------------------------------------------------------------------------

def user_geometry_penalty(
    xL: np.ndarray,
    xR: np.ndarray,
    cfg: OptimizerConfig,
    params: PiezoBeamParams,
) -> float:
    """
    Add custom penalties here.

    Return 0.0 if the design is acceptable.
    Return a positive penalty if the design violates a soft constraint.

    Examples you can add:
      - keep patch 1 away from root,
      - force patch 3 to have a specific length,
      - penalize patch centers outside certain regions,
      - penalize excessive mass,
      - penalize natural frequencies outside target bands.
    """
    penalty = 0.0

    # Example 1: penalize total patch length above a limit.
    lengths = xR - xL
    if cfg.max_total_patch_length is not None:
        excess = np.sum(lengths) - cfg.max_total_patch_length
        if excess > 0:
            penalty += 1e8 * excess**2

    # Example 2: reserve a forbidden region, disabled by default.
    # forbidden = (0.12, 0.15)
    # overlap = np.maximum(0.0, np.minimum(xR, forbidden[1]) - np.maximum(xL, forbidden[0]))
    # penalty += 1e8 * np.sum(overlap**2)

    return penalty


# -------------------------------------------------------------------------
# Geometry encoding / decoding
# -------------------------------------------------------------------------

def patch_length_bounds(cfg: OptimizerConfig, patch_index: int) -> tuple[float, float]:
    """Return length bounds for a given patch index."""
    if patch_index in cfg.fixed_lengths:
        val = cfg.fixed_lengths[patch_index]
        return val, val

    if cfg.length_bounds_by_patch is not None:
        return cfg.length_bounds_by_patch[patch_index]

    return cfg.length_min, cfg.length_max


def design_vector_bounds(n_patches: int, cfg: OptimizerConfig) -> list[tuple[float, float]]:
    """
    Bounds for scipy.differential_evolution.

    First n_patches variables:
        u_len[j] in [0, 1] mapped to length bounds.

    Last n_patches + 1 variables:
        gap allocation weights in [0, 1].
    """
    return [(0.0, 1.0)] * (n_patches + n_patches + 1)


def decode_design_vector(
    z: np.ndarray,
    n_patches: int,
    cfg: OptimizerConfig,
    params: PiezoBeamParams,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert normalized optimization variables into xL/xR.

    Returns
    -------
    xL, xR, info

    This function hard-enforces:
      - patch length bounds,
      - minimum internal gaps,
      - root/tip margins,
      - ordering of patches.

    If the chosen lengths cannot fit into the beam, info["valid"] = False.
    """
    z = np.asarray(z, dtype=float)

    L = params.L_b if cfg.L is None else cfg.L

    # ---- patch lengths ----
    u_len = z[:n_patches]
    lengths = np.zeros(n_patches)

    for j in range(n_patches):
        lo, hi = patch_length_bounds(cfg, j)
        if hi < lo:
            raise ValueError(f"Patch {j}: upper length bound is below lower bound.")
        if np.isclose(lo, hi):
            lengths[j] = lo
        else:
            lengths[j] = lo + u_len[j] * (hi - lo)

    # ---- minimum required length ----
    min_required = (
        np.sum(lengths)
        + cfg.root_gap_min
        + cfg.tip_gap_min
        + max(0, n_patches - 1) * cfg.gap_min
    )

    remaining = L - min_required
    if remaining < 0:
        return np.array([]), np.array([]), {
            "valid": False,
            "reason": "lengths_and_min_gaps_exceed_beam_length",
            "remaining": remaining,
            "lengths": lengths,
        }

    # ---- distribute remaining length among root/internal/tip gaps ----
    gap_weights = np.asarray(z[n_patches:], dtype=float)
    if len(gap_weights) != n_patches + 1:
        raise ValueError("Incorrect design vector length.")

    # Add epsilon so all gaps are allowed to receive some slack.
    gap_weights = gap_weights + 1e-12
    gap_weights = gap_weights / np.sum(gap_weights)

    extra_gaps = remaining * gap_weights

    root_gap = cfg.root_gap_min + extra_gaps[0]
    tip_gap = cfg.tip_gap_min + extra_gaps[-1]
    internal_gaps = np.full(max(0, n_patches - 1), cfg.gap_min)

    if n_patches > 1:
        internal_gaps += extra_gaps[1:-1]

    # ---- construct xL/xR ----
    xL = np.zeros(n_patches)
    xR = np.zeros(n_patches)

    x = root_gap
    for j in range(n_patches):
        xL[j] = x
        xR[j] = x + lengths[j]
        x = xR[j]
        if j < n_patches - 1:
            x += internal_gaps[j]

    # tip gap is implicit: L - x
    actual_tip_gap = L - x

    info = {
        "valid": True,
        "lengths": lengths,
        "root_gap": root_gap,
        "internal_gaps": internal_gaps,
        "tip_gap": actual_tip_gap,
        "allocated_tip_gap": tip_gap,
        "remaining": remaining,
    }

    return xL, xR, info


# -------------------------------------------------------------------------
# FE model construction
# -------------------------------------------------------------------------

def make_params_with_geometry(
    base_params: PiezoBeamParams,
    xL: np.ndarray,
    xR: np.ndarray,
    cfg: OptimizerConfig,
) -> PiezoBeamParams:
    """
    Create a copy of PiezoBeamParams with arbitrary geometry attached.

    PiezoBeamFE checks `params.geometry` first; if present, it uses it instead
    of the default repeated-patch layout.
    """
    params = copy.deepcopy(base_params)
    L = params.L_b if cfg.L is None else cfg.L

    rhoA_patch = params.b * (params.rho_s * params.hs + 2.0 * params.rho_p * params.hp)
    rhoA_gap = params.b * params.rho_s * params.hs

    EI_patch = params.YI
    EI_gap = params.YI_s

    geom = build_geometry_arbitrary_piezos(
        L=L,
        xL=np.asarray(xL),
        xR=np.asarray(xR),
        EI_patch=EI_patch,
        rhoA_patch=rhoA_patch,
        EI_gap=EI_gap,
        rhoA_gap=rhoA_gap,
        h_patch=cfg.h_patch,
        h_gap=cfg.h_gap,
    )

    params.geometry = geom
    params.xL = np.asarray(xL)
    params.xR = np.asarray(xR)
    params.n_patches = len(xL)
    params.S = len(xL)
    params.Q = len(xL)

    # If your Cp differs with patch length, update params.Cp here.
    # In the current FE3 direct mechanical FRF, Cp is not used.
    return params


def build_fe_model(
    base_params: PiezoBeamParams,
    xL: np.ndarray,
    xR: np.ndarray,
    cfg: OptimizerConfig,
) -> PiezoBeamFE:
    params = make_params_with_geometry(base_params, xL, xR, cfg)
    return PiezoBeamFE(params, n_el_patch=1, n_el_gap=1)


def mechanical_damping_matrix(fe: PiezoBeamFE, cfg: OptimizerConfig) -> np.ndarray:
    """Return mechanical damping matrix for the direct mechanical FRF."""
    if cfg.damping_model == "modal":
        return fe.C_red

    if cfg.damping_model == "rayleigh":
        return fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red

    if cfg.damping_model == "modal_plus_rayleigh":
        return fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red

    raise ValueError(f"Unknown damping_model: {cfg.damping_model}")


# -------------------------------------------------------------------------
# Direct FRF evaluator
# -------------------------------------------------------------------------

def tip_displacement_reduced_index(fe: PiezoBeamFE) -> int:
    """Return reduced DOF index corresponding to full tip displacement w(L)."""
    n_nodes = len(fe.geom.x_nodes)
    tip_full_dof = 2 * (n_nodes - 1)
    matches = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(matches) != 1:
        raise RuntimeError("Could not locate tip displacement DOF in reduced coordinates.")
    return int(matches[0])


def patch_output_contributions(
    fe: PiezoBeamFE,
    omega: float,
    cfg: OptimizerConfig,
) -> np.ndarray:
    """
    Compute complex output contribution h_j from each patch j at frequency omega.

    Returns
    -------
    h : ndarray, shape (n_patches,)
        h[j] = output response due to 1 V on patch j.
    """
    M = fe.M_red
    K = fe.K_red
    C = mechanical_damping_matrix(fe, cfg)
    Gamma = fe.Gamma_red  # shape: (N_mech_dof, n_patches)

    Z = K + 1j * omega * C - (omega**2) * M

    # Solve for all unit-patch RHS columns at once:
    # U[:, j] = response to unit voltage on patch j.
    U = np.linalg.solve(Z, Gamma)

    if cfg.output_metric == "tip_displacement":
        out_idx = tip_displacement_reduced_index(fe)
        return U[out_idx, :]

    if cfg.output_metric == "tip_velocity":
        out_idx = tip_displacement_reduced_index(fe)
        return 1j * omega * U[out_idx, :]

    raise NotImplementedError(
        f"output_metric={cfg.output_metric!r} is not implemented. "
        "Start with tip_displacement or tip_velocity."
    )


# -------------------------------------------------------------------------
# Actuation optimization for a fixed geometry and fixed frequency
# -------------------------------------------------------------------------

def best_binary_signs_bruteforce(h: np.ndarray, voltage_amp: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Brute-force best signs s_j in {-1,+1} for scalar complex output h @ (V*s)."""
    n = len(h)
    best_amp = -np.inf
    best_signs = None
    best_v = None

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        signs_arr = np.asarray(signs)
        v = voltage_amp * signs_arr
        amp = abs(np.dot(h, v))
        if amp > best_amp:
            best_amp = amp
            best_signs = signs_arr
            best_v = v.astype(complex)

    return float(best_amp), best_v, best_signs


def best_binary_signs_greedy(h: np.ndarray, voltage_amp: float, n_passes: int = 5) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Greedy sign-flip optimizer for larger patch counts.

    This is not guaranteed global-optimal, but is fast.
    """
    n = len(h)

    # Good initial guess: align with real part of h.
    signs = np.where(np.real(h) >= 0.0, 1.0, -1.0)

    def amp_for(s):
        return abs(np.dot(h, voltage_amp * s))

    best_amp = amp_for(signs)

    for _ in range(n_passes):
        improved = False
        for j in range(n):
            candidate = signs.copy()
            candidate[j] *= -1.0
            cand_amp = amp_for(candidate)
            if cand_amp > best_amp:
                signs = candidate
                best_amp = cand_amp
                improved = True
        if not improved:
            break

    return float(best_amp), voltage_amp * signs.astype(complex), signs


def best_actuation_at_frequency(
    h: np.ndarray,
    cfg: OptimizerConfig,
    mode_index: int,
) -> dict:
    """
    Given patch contribution vector h, choose voltage vector v_hat.

    The scalar output is:
        y_hat = h @ v_hat
    """
    h = np.asarray(h, dtype=complex)
    n = len(h)

    if cfg.phase_mode == "continuous":
        # h_j * exp(-i angle(h_j)) = |h_j|
        v = cfg.voltage_amp * np.exp(-1j * np.angle(h))
        y = np.dot(h, v)
        return {
            "amplitude": float(abs(y)),
            "voltage_vector": v,
            "phase_rad": np.angle(v),
            "signs": None,
            "mode": "continuous",
        }

    if cfg.phase_mode == "binary":
        if n <= cfg.max_binary_bruteforce_patches:
            amp, v, signs = best_binary_signs_bruteforce(h, cfg.voltage_amp)
        else:
            amp, v, signs = best_binary_signs_greedy(h, cfg.voltage_amp)

        return {
            "amplitude": amp,
            "voltage_vector": v,
            "phase_rad": np.angle(v),
            "signs": signs,
            "mode": "binary",
        }

    if cfg.phase_mode == "fixed":
        if cfg.fixed_phase_patterns is None or mode_index not in cfg.fixed_phase_patterns:
            raise ValueError(f"Missing fixed phase/sign pattern for mode index {mode_index}.")

        pattern = np.asarray(cfg.fixed_phase_patterns[mode_index])
        if len(pattern) != n:
            raise ValueError("Fixed phase pattern length does not match number of patches.")

        if np.iscomplexobj(pattern):
            v = cfg.voltage_amp * pattern / np.maximum(np.abs(pattern), 1e-15)
        else:
            # Treat as signs or phase angles.
            unique_vals = set(np.unique(pattern).tolist())
            if unique_vals.issubset({-1, 1, -1.0, 1.0}):
                v = cfg.voltage_amp * pattern.astype(float)
            else:
                v = cfg.voltage_amp * np.exp(1j * pattern.astype(float))

        y = np.dot(h, v)
        return {
            "amplitude": float(abs(y)),
            "voltage_vector": v,
            "phase_rad": np.angle(v),
            "signs": np.sign(np.real(v)) if np.allclose(np.imag(v), 0.0) else None,
            "mode": "fixed",
        }

    raise ValueError(f"Unknown phase_mode: {cfg.phase_mode}")


# -------------------------------------------------------------------------
# Mode-specific and multi-mode scoring
# -------------------------------------------------------------------------

def evaluate_mode(
    fe: PiezoBeamFE,
    cfg: OptimizerConfig,
    mode_index: int,
    window_hz: tuple[float, float],
) -> dict:
    """
    For one mode, sweep frequency and choose best phase/sign configuration.
    """
    fmin, fmax = window_hz
    freq_vec = np.linspace(fmin, fmax, cfg.n_freq_per_mode)
    omega_vec = 2.0 * np.pi * freq_vec

    best = {
        "amplitude": -np.inf,
        "freq_hz": None,
        "omega": None,
        "voltage_vector": None,
        "phase_rad": None,
        "signs": None,
    }

    for freq_hz, omega in zip(freq_vec, omega_vec):
        h = patch_output_contributions(fe, omega, cfg)
        act = best_actuation_at_frequency(h, cfg, mode_index)
        if act["amplitude"] > best["amplitude"]:
            best.update({
                "amplitude": act["amplitude"],
                "freq_hz": float(freq_hz),
                "omega": float(omega),
                "voltage_vector": act["voltage_vector"],
                "phase_rad": act["phase_rad"],
                "signs": act["signs"],
            })

    return best


def evaluate_geometry(
    xL: np.ndarray,
    xR: np.ndarray,
    base_params: PiezoBeamParams,
    cfg: OptimizerConfig,
    active_mode_indices: Optional[Sequence[int]] = None,
) -> dict:
    """
    Evaluate a candidate geometry.

    active_mode_indices:
        None     -> use all configured modes
        [0]      -> optimize/evaluate mode 1 only
        [1]      -> optimize/evaluate mode 2 only
        [0,1,2]  -> multi-mode objective
    """
    xL = np.asarray(xL, dtype=float)
    xR = np.asarray(xR, dtype=float)

    # Basic hard constraints.
    if len(xL) != len(xR) or len(xL) == 0:
        return {"valid": False, "score": -cfg.invalid_penalty, "reason": "bad_patch_arrays"}

    if not np.all(xL < xR):
        return {"valid": False, "score": -cfg.invalid_penalty, "reason": "nonpositive_patch_length"}

    L = base_params.L_b if cfg.L is None else cfg.L
    if np.min(xL) < -1e-12 or np.max(xR) > L + 1e-12:
        return {"valid": False, "score": -cfg.invalid_penalty, "reason": "outside_beam"}

    if len(xL) > 1 and np.any(xL[1:] - xR[:-1] < cfg.gap_min - 1e-12):
        return {"valid": False, "score": -cfg.invalid_penalty, "reason": "gap_too_small"}

    penalty = user_geometry_penalty(xL, xR, cfg, base_params)

    try:
        fe = build_fe_model(base_params, xL, xR, cfg)
    except Exception as exc:
        return {
            "valid": False,
            "score": -cfg.invalid_penalty,
            "reason": f"fe_build_failed: {exc}",
        }

    if active_mode_indices is None:
        active_mode_indices = list(range(len(cfg.mode_windows_hz)))

    mode_results = []
    raw_modal_amps = []
    normalized_modal_amps = []

    for mi in active_mode_indices:
        mode_result = evaluate_mode(
            fe=fe,
            cfg=cfg,
            mode_index=mi,
            window_hz=cfg.mode_windows_hz[mi],
        )
        mode_results.append(mode_result)

        amp = mode_result["amplitude"]
        raw_modal_amps.append(amp)

        if cfg.baseline_modal_amplitudes is not None:
            denom = cfg.baseline_modal_amplitudes[mi]
            amp_norm = amp / max(abs(denom), 1e-30)
        else:
            amp_norm = amp

        normalized_modal_amps.append(amp_norm)

    weights = np.asarray([cfg.mode_weights[mi] for mi in active_mode_indices], dtype=float)
    weighted_score = float(np.dot(weights, np.asarray(normalized_modal_amps)))

    # Convert penalty to same "maximize score" convention.
    score = weighted_score - penalty

    return {
        "valid": True,
        "score": score,
        "weighted_score_before_penalty": weighted_score,
        "penalty": penalty,
        "xL": xL,
        "xR": xR,
        "lengths": xR - xL,
        "natural_freq_hz": fe.freq,
        "mode_results": mode_results,
        "active_mode_indices": list(active_mode_indices),
    }


# -------------------------------------------------------------------------
# Objective wrappers for scipy.optimize
# -------------------------------------------------------------------------

def objective_from_design_vector(
    z: np.ndarray,
    n_patches: int,
    base_params: PiezoBeamParams,
    cfg: OptimizerConfig,
    active_mode_indices: Optional[Sequence[int]] = None,
) -> float:
    """
    scipy.optimize minimizes, so return negative score.
    """
    xL, xR, info = decode_design_vector(z, n_patches, cfg, base_params)
    if not info["valid"]:
        return cfg.invalid_penalty + 1e12 * abs(info.get("remaining", 0.0))

    result = evaluate_geometry(
        xL=xL,
        xR=xR,
        base_params=base_params,
        cfg=cfg,
        active_mode_indices=active_mode_indices,
    )

    if not result["valid"]:
        return cfg.invalid_penalty

    return -result["score"]


def optimize_for_patch_count(
    n_patches: int,
    base_params: PiezoBeamParams,
    cfg: OptimizerConfig,
    active_mode_indices: Optional[Sequence[int]] = None,
    polish: bool = True,
) -> dict:
    """
    Optimize geometry for a fixed number of patches.
    """
    bounds = design_vector_bounds(n_patches, cfg)

    opt: OptimizeResult = differential_evolution(
        func=lambda z: objective_from_design_vector(
            z=z,
            n_patches=n_patches,
            base_params=base_params,
            cfg=cfg,
            active_mode_indices=active_mode_indices,
        ),
        bounds=bounds,
        maxiter=cfg.differential_evolution_maxiter,
        popsize=cfg.differential_evolution_popsize,
        tol=cfg.differential_evolution_tol,
        seed=cfg.seed,
        polish=polish,
        updating="immediate",
        workers=1,  # change to -1 if your model/build is multiprocessing-safe
    )

    xL, xR, decode_info = decode_design_vector(opt.x, n_patches, cfg, base_params)
    eval_result = evaluate_geometry(
        xL=xL,
        xR=xR,
        base_params=base_params,
        cfg=cfg,
        active_mode_indices=active_mode_indices,
    )

    return {
        "n_patches": n_patches,
        "optimizer_result": opt,
        "decode_info": decode_info,
        "evaluation": eval_result,
    }


def run_patch_count_sweep(
    base_params: PiezoBeamParams,
    cfg: OptimizerConfig,
    active_mode_indices: Optional[Sequence[int]] = None,
) -> list[dict]:
    """
    Run one optimization for each allowed patch count.
    """
    all_results = []
    for n_patches in cfg.allowed_n_patches:
        print(f"\n=== Optimizing Np = {n_patches}, active modes = {active_mode_indices} ===")
        res = optimize_for_patch_count(
            n_patches=n_patches,
            base_params=base_params,
            cfg=cfg,
            active_mode_indices=active_mode_indices,
        )
        all_results.append(res)

        ev = res["evaluation"]
        print(f"Best score: {ev['score']:.6e}")
        print(f"xL [mm]: {1e3 * ev['xL']}")
        print(f"xR [mm]: {1e3 * ev['xR']}")
        print(f"lengths [mm]: {1e3 * ev['lengths']}")
        for k, mr in zip(ev["active_mode_indices"], ev["mode_results"]):
            print(
                f"  Mode {k+1}: amp={mr['amplitude']:.6e}, "
                f"f={mr['freq_hz']:.3f} Hz, signs={mr['signs']}"
            )

    return all_results


def best_result(results: list[dict]) -> dict:
    """Return the result dictionary with the highest evaluated score."""
    return max(results, key=lambda r: r["evaluation"]["score"])


# -------------------------------------------------------------------------
# Recommended usage
# -------------------------------------------------------------------------

def main():
    # 1) Load baseline parameters.
    params = PiezoBeamParams()

    # 2) Configure optimizer.
    cfg = OptimizerConfig(
        # Use params.L_b by default. Override here if your physical beam length differs:
        # L=0.3185,

        allowed_n_patches=(1, 2, 3, 4),

        # Example constraints:
        length_min=10e-3,
        length_max=70e-3,
        gap_min=3e-3,
        root_gap_min=0.0,
        tip_gap_min=0.0,

        # Optional total active material constraint:
        # max_total_patch_length=160e-3,

        # Mode windows should be adapted after checking FE natural frequencies.
        mode_windows_hz=(
            (0.5, 3.0),
            (2.0, 6.0),
            (5.0, 10.0),
        ),
        mode_weights=(1.0, 1.0, 1.0),
        n_freq_per_mode=101,

        # Start with binary signs to match +V/-V COMSOL studies.
        phase_mode="binary",
        voltage_amp=1.0,

        differential_evolution_maxiter=25,
        differential_evolution_popsize=10,
        seed=2,
    )

    # 3) Optional: single-mode searches for insight / initialization.
    # These are diagnostic: they reveal what each mode "wants" geometrically.
    single_mode_results = {}
    for mode_idx in range(len(cfg.mode_windows_hz)):
        single_mode_results[mode_idx] = run_patch_count_sweep(
            base_params=params,
            cfg=cfg,
            active_mode_indices=[mode_idx],
        )

    # 4) Multi-mode optimization: this is the design you should ultimately use.
    multi_results = run_patch_count_sweep(
        base_params=params,
        cfg=cfg,
        active_mode_indices=None,
    )

    best_multi = best_result(multi_results)
    ev = best_multi["evaluation"]

    print("\n=== BEST MULTI-MODE DESIGN ===")
    print(f"Np = {best_multi['n_patches']}")
    print(f"score = {ev['score']:.6e}")
    print(f"xL [mm] = {1e3 * ev['xL']}")
    print(f"xR [mm] = {1e3 * ev['xR']}")
    print(f"lengths [mm] = {1e3 * ev['lengths']}")

    for k, mr in zip(ev["active_mode_indices"], ev["mode_results"]):
        print(f"\nMode {k+1}")
        print(f"  best amplitude = {mr['amplitude']:.6e}")
        print(f"  best frequency = {mr['freq_hz']:.4f} Hz")
        print(f"  best signs     = {mr['signs']}")
        print(f"  phase [deg]    = {np.rad2deg(mr['phase_rad'])}")


if __name__ == "__main__":
    main()
