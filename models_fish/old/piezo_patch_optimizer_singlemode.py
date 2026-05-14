"""Patch-placement optimizer for 1D piezoelectric beam FE models.

Version 2 additions:
- cleaner experiment-loop support via external runner helpers
- arbitrary patch counts through GeometrySettings.Np
- binary and continuous phase optimization
- output metrics: tip, line-average displacement, RMS displacement

The optimizer intentionally keeps notebooks thin: configure settings, call
``PiezoPatchOptimizer.run()``, then plot/post-process results.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from tqdm.auto import tqdm

try:  # package imports
    from Modeling.models_fish.beam_properties_fish import PiezoBeamParams
    import Modeling.models_fish.FE_fish as FE_module
except Exception:  # local / notebook fallback
    try:
        from beam_properties import PiezoBeamParams
    except Exception:
        from beam_properties_refactored import PiezoBeamParams
    try:
        import FE3_refactored as FE_module
    except Exception:
        import FE3 as FE_module


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

@dataclass
class GeometrySettings:
    """Outer geometry design settings.

    The design vector is always
    ``z = [L1, g12, L2, g23, ..., g(n-1,n), Ln]`` and has length ``2*Np - 1``.
    The final substrate length is the remaining beam length.
    """

    Np: int = 3
    patch_length_bounds: Tuple[float, float] = (10e-3, 40e-3)
    gap_bounds: Tuple[float, float] = (4e-3, 80e-3)
    tip_substrate_bounds: Tuple[float, float] = (0.0, 150e-3)
    total_patch_length_bounds: Optional[Tuple[float, float]] = None
    fixed_patch_lengths: Dict[int, float] = field(default_factory=dict)  # patch index j = 0..Np-1
    fixed_gaps: Dict[int, float] = field(default_factory=dict)           # gap index j = 0..Np-2
    invalid_penalty: float = 1e12


@dataclass
class SingleModeSettings:
    target_mode_number: int = 1
    voltage_amplitude: float = 1.0
    phase_mode: str = "binary"  # "binary" or "continuous"

    # Output metric used in the objective.
    #   "tip" / "tip_disp": tip displacement magnitude.
    #   "mean_abs" / "line_average": integral average of |w(x)| over beam length.
    #   "rms" / "rms_displacement": RMS of |w(x)| over beam length.
    output: str = "tip"

    # Dense final sweep settings for post-processing.
    final_sweep_range_hz: Tuple[float, float] = (0.1, 10.0)
    final_sweep_n_freq: int = 1000

    # Only used when phase_mode="continuous" and output is not scalar tip.
    # For tip output, continuous phase has an analytic solution.
    continuous_phase_n_starts: int = 8
    continuous_phase_seed: Optional[int] = 1
    continuous_phase_method: str = "L-BFGS-B"


@dataclass
class CircuitSettings:
    R_c: float = 1e3
    K_p: float = 0.02
    K_i: float = 0.0
    K_c: float = 0.0


@dataclass
class OptimizerSettings:
    method: str = "differential_evolution"  # differential_evolution, random, powell, random_powell
    maxiter: int = 15
    popsize: int = 8
    seed: Optional[int] = 1
    polish: bool = False
    workers: int = 1
    n_random_samples: int = 300
    powell_maxiter: int = 80
    powell_xtol: float = 1e-4
    powell_ftol: float = 1e-4
    show_progress: bool = True


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def sign_label(signs) -> str:
    return "".join("+" if s > 0 else "-" for s in np.asarray(signs))


def make_region_sequence(n_patches: int) -> List[str]:
    """Return ['piezo','substrate', ...] for N patch/substrate pairs."""
    if n_patches < 1:
        raise ValueError("n_patches must be at least 1")
    seq: List[str] = []
    for _ in range(int(n_patches)):
        seq.extend(["piezo", "substrate"])
    return seq


def wrap_phase_rad(phase_rad: np.ndarray) -> np.ndarray:
    """Wrap phases to [0, 2*pi)."""
    return np.mod(np.asarray(phase_rad, dtype=float), 2 * np.pi)


def relative_phase_rad(phase_rad: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """Return phases relative to one patch, wrapped to [0, 2*pi)."""
    phase_rad = np.asarray(phase_rad, dtype=float)
    return wrap_phase_rad(phase_rad - phase_rad[reference_index])


# -----------------------------------------------------------------------------
# Output metrics
# -----------------------------------------------------------------------------

def tip_reduced_index(fe) -> int:
    """Reduced DOF index for tip transverse displacement."""
    tip_full_dof = 2 * (len(fe.geom.x_nodes) - 1)
    idx = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(idx) != 1:
        raise RuntimeError("Could not find tip displacement DOF in reduced system")
    return int(idx[0])


def transverse_reduced_indices(fe) -> np.ndarray:
    """Reduced indices corresponding to transverse displacement DOFs w, excluding fixed DOFs."""
    return np.where((np.asarray(fe.free_dofs) % 2) == 0)[0]


def reduced_to_full_displacement_nodes(fe, u_red: np.ndarray) -> np.ndarray:
    """Convert reduced mechanical response vector to nodal transverse displacement array."""
    u_red = np.asarray(u_red, dtype=complex)
    full = np.zeros(fe.Ndof, dtype=complex)
    full[fe.free_dofs] = u_red
    return full[0::2]


def trapezoid_node_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Integration weights for nodal values on a nonuniform 1D mesh."""
    x = np.asarray(x_nodes, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x_nodes must be a 1D array with at least two nodes")
    weights = np.zeros_like(x)
    dx = np.diff(x)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return weights


def canonical_output_name(output: str) -> str:
    output = output.lower().strip()
    aliases = {
        "tip": "tip",
        "tip_disp": "tip",
        "tip_displacement": "tip",
        "line_average": "mean_abs",
        "avg": "mean_abs",
        "average": "mean_abs",
        "average_displacement": "mean_abs",
        "mean": "mean_abs",
        "mean_abs": "mean_abs",
        "mean_abs_displacement": "mean_abs",
        "rms": "rms",
        "rms_displacement": "rms",
    }
    if output not in aliases:
        raise ValueError(
            "Unknown output metric. Use 'tip', 'mean_abs'/'line_average', or 'rms'."
        )
    return aliases[output]


def metric_label(output: str) -> str:
    output = canonical_output_name(output)
    if output == "tip":
        return "Tip displacement magnitude [m/V]"
    if output == "mean_abs":
        return "Line-average displacement magnitude [m/V]"
    if output == "rms":
        return "RMS beam displacement magnitude [m/V]"
    return "Output metric"


def evaluate_output_metric(fe, u_red: np.ndarray, output: str = "tip") -> float:
    """Evaluate scalar output metric from a reduced displacement response vector.

    For harmonic complex response u_hat, this evaluates the amplitude metric.
    """
    output = canonical_output_name(output)

    if output == "tip":
        return float(abs(u_red[tip_reduced_index(fe)]))

    w_nodes = reduced_to_full_displacement_nodes(fe, u_red)
    weights = trapezoid_node_weights(fe.geom.x_nodes)
    L = float(fe.geom.x_nodes[-1] - fe.geom.x_nodes[0])
    if L <= 0:
        raise ValueError("Beam length must be positive")

    if output == "mean_abs":
        return float(np.sum(weights * np.abs(w_nodes)) / L)

    if output == "rms":
        return float(np.sqrt(np.sum(weights * np.abs(w_nodes) ** 2) / L))

    raise ValueError(f"Unhandled output metric {output}")


def response_summary(fe, u_red: np.ndarray, output: str) -> dict:
    """Return common response metrics for a reduced displacement vector."""
    return {
        "tip": evaluate_output_metric(fe, u_red, "tip"),
        "mean_abs": evaluate_output_metric(fe, u_red, "mean_abs"),
        "rms": evaluate_output_metric(fe, u_red, "rms"),
        "selected": evaluate_output_metric(fe, u_red, output),
        "output": canonical_output_name(output),
    }


# -----------------------------------------------------------------------------
# Inner phase optimizers
# -----------------------------------------------------------------------------

def optimize_binary_phases_general(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
) -> dict:
    """Brute-force binary signs for a full reduced-response matrix.

    U_cols[:, j] is the reduced displacement response per unit voltage on patch j.
    """
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    output = canonical_output_name(output)

    best_score = -np.inf
    best_signs = None
    best_response_red = None
    all_results = []

    for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
        signs = np.asarray(signs_tuple, dtype=float)
        voltage_vector = voltage_amplitude * signs.astype(complex)
        u_red = U_cols @ voltage_vector
        score = evaluate_output_metric(fe, u_red, output)
        record = {
            "signs": signs,
            "phase_rad": np.where(signs > 0, 0.0, np.pi),
            "phase_deg": np.where(signs > 0, 0.0, 180.0),
            "voltage_vector": voltage_vector,
            "response_red": u_red,
            "score": float(score),
            "label": sign_label(signs),
            **response_summary(fe, u_red, output),
        }
        all_results.append(record)
        if score > best_score:
            best_score = score
            best_signs = signs
            best_response_red = u_red

    best_phase_rad = np.where(best_signs > 0, 0.0, np.pi)
    best_voltage_vector = voltage_amplitude * best_signs.astype(complex)
    return {
        "phase_mode": "binary",
        "score": float(best_score),
        "response": best_response_red[tip_reduced_index(fe)],  # backward-compatible scalar tip response
        "response_red": best_response_red,
        "response_metrics": response_summary(fe, best_response_red, output),
        "signs": best_signs,
        "phase_rad": best_phase_rad,
        "phase_deg": np.rad2deg(best_phase_rad),
        "relative_phase_rad": relative_phase_rad(best_phase_rad),
        "relative_phase_deg": np.rad2deg(relative_phase_rad(best_phase_rad)),
        "voltage_vector": best_voltage_vector,
        "all_phase_results": all_results,
    }


def optimize_continuous_phases_tip(
    fe,
    U_cols: np.ndarray,
    voltage_amplitude: float = 1.0,
) -> dict:
    """Analytic continuous phase alignment for scalar tip displacement."""
    h = np.asarray(U_cols[tip_reduced_index(fe), :], dtype=complex)
    phase_rad = -np.angle(h)
    # Normalize to relative phases. Global phase does not change displacement magnitudes.
    phase_rad = relative_phase_rad(phase_rad, reference_index=0)
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    u_red = U_cols @ voltage_vector
    return {
        "phase_mode": "continuous",
        "phase_optimizer": "analytic_tip_alignment",
        "score": evaluate_output_metric(fe, u_red, "tip"),
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "response_metrics": response_summary(fe, u_red, "tip"),
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": phase_rad,
        "relative_phase_deg": np.rad2deg(phase_rad),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
    }


def optimize_continuous_phases_metric(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
    *,
    n_starts: int = 8,
    seed: Optional[int] = 1,
    method: str = "L-BFGS-B",
) -> dict:
    """Numerically optimize continuous patch phases for a non-scalar output metric.

    The first patch phase is fixed to zero because global phase does not affect
    metrics based on displacement magnitude. This reduces the search dimension
    from Np to Np-1.
    """
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    output = canonical_output_name(output)

    if n == 1:
        phase_rad = np.array([0.0])
        voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
        u_red = U_cols @ voltage_vector
        return {
            "phase_mode": "continuous",
            "phase_optimizer": "single_patch",
            "score": evaluate_output_metric(fe, u_red, output),
            "response": u_red[tip_reduced_index(fe)],
            "response_red": u_red,
            "response_metrics": response_summary(fe, u_red, output),
            "signs": np.sign(np.real(voltage_vector)),
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": phase_rad,
            "relative_phase_deg": np.rad2deg(phase_rad),
            "voltage_vector": voltage_vector,
            "all_phase_results": None,
        }

    rng = np.random.default_rng(seed)

    def make_voltage(alpha_free: np.ndarray) -> np.ndarray:
        phase_rad = np.concatenate([[0.0], np.asarray(alpha_free, dtype=float)])
        return voltage_amplitude * np.exp(1j * phase_rad)

    def neg_score(alpha_free: np.ndarray) -> float:
        u_red = U_cols @ make_voltage(alpha_free)
        return -evaluate_output_metric(fe, u_red, output)

    starts = [np.zeros(n - 1)]

    # Include binary-like initial guesses; often good for beam modes.
    for signs_tuple in itertools.product([0.0, np.pi], repeat=n - 1):
        starts.append(np.asarray(signs_tuple, dtype=float))
        if len(starts) >= max(2, min(n_starts, 2 ** (n - 1) + 1)):
            break

    while len(starts) < n_starts:
        starts.append(rng.uniform(0.0, 2 * np.pi, size=n - 1))

    bounds = [(0.0, 2 * np.pi)] * (n - 1)
    best_res = None
    for x0 in starts:
        res = minimize(neg_score, x0, method=method, bounds=bounds)
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    phase_rad = wrap_phase_rad(np.concatenate([[0.0], best_res.x]))
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    u_red = U_cols @ voltage_vector
    return {
        "phase_mode": "continuous",
        "phase_optimizer": f"numeric_{method}",
        "score": evaluate_output_metric(fe, u_red, output),
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "response_metrics": response_summary(fe, u_red, output),
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": phase_rad,
        "relative_phase_deg": np.rad2deg(phase_rad),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
        "inner_opt_result": best_res,
    }


# -----------------------------------------------------------------------------
# Main optimizer
# -----------------------------------------------------------------------------

class PiezoPatchOptimizer:
    """Outer geometry optimizer with an inner phase optimization layer."""

    def __init__(
        self,
        L: float,
        region_types: dict,
        base_params: PiezoBeamParams,
        geometry_settings: GeometrySettings,
        mode_settings: SingleModeSettings,
        circuit_settings: Optional[CircuitSettings] = None,
        optimizer_settings: Optional[OptimizerSettings] = None,
        *,
        fe_module=FE_module,
        default_h: float = 1e-3,
    ):
        self.L = float(L)
        self.region_types = region_types
        self.base_params = base_params
        self.geometry_settings = geometry_settings
        self.mode_settings = mode_settings
        self.circuit_settings = circuit_settings or CircuitSettings()
        self.optimizer_settings = optimizer_settings or OptimizerSettings()
        self.fe_module = fe_module
        self.default_h = default_h
        self.evaluation_history: list[dict] = []

    @property
    def n_patches(self) -> int:
        return int(self.geometry_settings.Np)

    @property
    def n_design_variables(self) -> int:
        return 2 * self.n_patches - 1

    def make_region_sequence(self) -> List[str]:
        return make_region_sequence(self.n_patches)

    def make_bounds(self) -> List[Tuple[float, float]]:
        gs = self.geometry_settings
        bounds: List[Tuple[float, float]] = []
        for j in range(gs.Np):
            patch_bounds = gs.patch_length_bounds
            if j in gs.fixed_patch_lengths:
                v = float(gs.fixed_patch_lengths[j])
                patch_bounds = (v, v)
            bounds.append(patch_bounds)

            if j < gs.Np - 1:
                gap_bounds = gs.gap_bounds
                if j in gs.fixed_gaps:
                    v = float(gs.fixed_gaps[j])
                    gap_bounds = (v, v)
                bounds.append(gap_bounds)
        return bounds

    def decode_design(self, z: np.ndarray) -> dict:
        """Decode z into patch lengths, gaps, starts, active xL/xR, and tip length."""
        z = np.asarray(z, dtype=float)
        if z.size != self.n_design_variables:
            raise ValueError(f"Expected design vector length {self.n_design_variables}, got {z.size}")

        patch_lengths = z[0::2]
        gaps = z[1::2]
        x_starts = [0.0]
        x = 0.0
        for j in range(self.n_patches):
            x += patch_lengths[j]
            x_starts.append(x)  # start of substrate after patch j
            if j < self.n_patches - 1:
                x += gaps[j]
                x_starts.append(x)  # start of next patch

        x_starts = np.asarray(x_starts, dtype=float)
        xL = x_starts[0::2]
        xR = x_starts[1::2]
        tip_substrate = self.L - x_starts[-1]

        return {
            "z": z,
            "patch_lengths": patch_lengths,
            "gaps": gaps,
            "x_starts": x_starts,
            "xL": xL,
            "xR": xR,
            "tip_substrate": float(tip_substrate),
            "total_patch_length": float(np.sum(patch_lengths)),
            "region_sequence": self.make_region_sequence(),
        }

    def geometry_penalty(self, layout: dict) -> float:
        gs = self.geometry_settings
        penalty = 0.0

        if np.any(layout["patch_lengths"] <= 0) or np.any(layout["gaps"] < 0):
            penalty += gs.invalid_penalty
        if np.any(np.diff(layout["x_starts"]) < -1e-12):
            penalty += gs.invalid_penalty
        if layout["x_starts"][-1] > self.L + 1e-12:
            penalty += gs.invalid_penalty

        tip_min, tip_max = gs.tip_substrate_bounds
        tip = layout["tip_substrate"]
        if tip < tip_min:
            penalty += gs.invalid_penalty + gs.invalid_penalty * (tip_min - tip) ** 2
        if tip > tip_max:
            penalty += gs.invalid_penalty + gs.invalid_penalty * (tip - tip_max) ** 2

        if gs.total_patch_length_bounds is not None:
            low, high = gs.total_patch_length_bounds
            total = layout["total_patch_length"]
            if total < low:
                penalty += gs.invalid_penalty + gs.invalid_penalty * (low - total) ** 2
            if total > high:
                penalty += gs.invalid_penalty + gs.invalid_penalty * (total - high) ** 2

        return float(penalty)

    def build_fe_for_design(self, z: np.ndarray):
        """Return (fe, layout, penalty) for a candidate design."""
        layout = self.decode_design(z)
        penalty = self.geometry_penalty(layout)
        if penalty > 0:
            return None, layout, penalty

        try:
            geom = self.fe_module.build_geometry_from_types(
                L=self.L,
                region_types=self.region_types,
                region_sequence=layout["region_sequence"],
                x_starts=layout["x_starts"],
                default_h=self.default_h,
            )
            params = copy.copy(self.base_params)
            params.geometry = geom
            if hasattr(params, "sync_patch_count"):
                params.sync_patch_count(len(geom.piezos))
            else:  # backward compatibility with older parameter class
                params.n_patches = len(geom.piezos)
                params.S = len(geom.piezos)
                params.Q = len(geom.piezos)
                params.Cp = params.Cp_scalar * np.ones(params.S)
            fe = self.fe_module.PiezoBeamFE(params)
            return fe, layout, 0.0
        except Exception as exc:
            layout["error"] = repr(exc)
            return None, layout, self.geometry_settings.invalid_penalty

    def response_columns(self, fe, omega: float) -> np.ndarray:
        """Return reduced displacement columns per unit patch voltage at omega."""
        D = fe.effective_damping_matrix() if hasattr(fe, "effective_damping_matrix") else fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red
        Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
        return np.linalg.solve(Z, fe.Gamma_red)

    def unit_patch_response(self, fe, omega: float, output: Optional[str] = None) -> np.ndarray:
        """Backward-compatible scalar unit-patch response vector for tip output.

        For non-tip outputs, use response_columns() because the metric is based
        on the full displacement field.
        """
        output = canonical_output_name(output or self.mode_settings.output)
        U_cols = self.response_columns(fe, omega)
        if output != "tip":
            raise ValueError("unit_patch_response is only meaningful for scalar tip output; use response_columns instead")
        return U_cols[tip_reduced_index(fe), :]

    def evaluate_at_natural_frequency(self, fe) -> dict:
        """Evaluate selected mode at its geometry-dependent natural frequency."""
        ms = self.mode_settings
        m = int(ms.target_mode_number)
        if m < 1 or m > len(fe.freq):
            raise ValueError(f"target_mode_number={m} outside available mode range 1..{len(fe.freq)}")

        omega = float(fe.omega[m - 1])
        freq_hz = float(fe.freq[m - 1])
        output = canonical_output_name(ms.output)
        U_cols = self.response_columns(fe, omega)
        h_tip = U_cols[tip_reduced_index(fe), :]

        phase_mode = ms.phase_mode.lower()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases_general(fe, U_cols, output, ms.voltage_amplitude)
        elif phase_mode == "continuous" and output == "tip":
            phase_result = optimize_continuous_phases_tip(fe, U_cols, ms.voltage_amplitude)
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases_metric(
                fe,
                U_cols,
                output,
                ms.voltage_amplitude,
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
            )
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")

        return {
            "target_mode_number": m,
            "omega": omega,
            "freq_hz": freq_hz,
            "output": output,
            "metric_label": metric_label(output),
            "h": h_tip,       # backward-compatible tip unit response
            "h_tip": h_tip,
            "U_cols": U_cols,
            **phase_result,
        }

    def objective(self, z: np.ndarray) -> float:
        fe, layout, penalty = self.build_fe_for_design(z)
        if penalty > 0 or fe is None:
            return float(penalty)

        try:
            inner = self.evaluate_at_natural_frequency(fe)
            score = float(inner["score"])
        except Exception as exc:
            self.evaluation_history.append({"z": np.asarray(z, dtype=float).copy(), "layout": layout, "score": -np.inf, "error": repr(exc)})
            return self.geometry_settings.invalid_penalty

        self.evaluation_history.append(
            {
                "z": np.asarray(z, dtype=float).copy(),
                "layout": layout,
                "score": score,
                "output": inner["output"],
                "freq_hz": float(inner["freq_hz"]),
                "phase_mode": inner["phase_mode"],
                "phase_deg": np.asarray(inner["phase_deg"]).copy(),
                "relative_phase_deg": np.asarray(inner["relative_phase_deg"]).copy(),
                "natural_freqs": fe.freq[: min(8, len(fe.freq))].copy(),
                "response_metrics": inner.get("response_metrics", {}),
            }
        )
        return -score

    def best_eval_from_history(self) -> Optional[dict]:
        valid = [h for h in self.evaluation_history if np.isfinite(h.get("score", -np.inf))]
        if not valid:
            return None
        return max(valid, key=lambda h: h["score"])

    def run_random_search(self) -> OptimizeResult:
        opt = self.optimizer_settings
        bounds = np.asarray(self.make_bounds(), dtype=float)
        rng = np.random.default_rng(opt.seed)
        best_x = None
        best_fun = np.inf
        iterator = range(opt.n_random_samples)
        if opt.show_progress:
            iterator = tqdm(iterator, desc="Random search")

        for _ in iterator:
            z = rng.uniform(bounds[:, 0], bounds[:, 1])
            fixed = bounds[:, 0] == bounds[:, 1]
            z[fixed] = bounds[fixed, 0]
            f = self.objective(z)
            if f < best_fun:
                best_fun = float(f)
                best_x = z.copy()

        return OptimizeResult(x=best_x, fun=best_fun, success=True, message="Random search complete", nfev=opt.n_random_samples)

    def run_powell_refinement(self, x0=None) -> OptimizeResult:
        opt = self.optimizer_settings
        bounds = self.make_bounds()
        if x0 is None:
            x0 = np.array([(a + b) / 2 for a, b in bounds], dtype=float)
        return minimize(
            self.objective,
            np.asarray(x0, dtype=float),
            method="Powell",
            bounds=bounds,
            options={"maxiter": opt.powell_maxiter, "xtol": opt.powell_xtol, "ftol": opt.powell_ftol, "disp": opt.show_progress},
        )

    def run_differential_evolution(self) -> OptimizeResult:
        opt = self.optimizer_settings
        return differential_evolution(
            self.objective,
            bounds=self.make_bounds(),
            maxiter=opt.maxiter,
            popsize=opt.popsize,
            seed=opt.seed,
            polish=opt.polish,
            workers=opt.workers,
            updating="deferred" if opt.workers != 1 else "immediate",
            disp=opt.show_progress,
        )

    def run(self) -> OptimizeResult:
        method = self.optimizer_settings.method.lower()
        if method in ("de", "differential_evolution"):
            return self.run_differential_evolution()
        if method == "random":
            return self.run_random_search()
        if method == "powell":
            return self.run_powell_refinement()
        if method in ("random_powell", "random+powell", "random-powell"):
            r0 = self.run_random_search()
            r1 = self.run_powell_refinement(r0.x)
            r1.random_result = r0
            return r1
        raise ValueError(f"Unknown optimizer method: {self.optimizer_settings.method}")

    def inspect_result(self, result: OptimizeResult) -> dict:
        """Rebuild and evaluate the best design from an OptimizeResult."""
        fe, layout, penalty = self.build_fe_for_design(result.x)
        if penalty > 0 or fe is None:
            raise RuntimeError("Could not rebuild FE model for optimization result")
        inner = self.evaluate_at_natural_frequency(fe)
        return {"result": result, "fe": fe, "layout": layout, "penalty": penalty, "inner": inner}

    def dense_metric_frf_for_plot(self, fe, voltage_vector, *, output: Optional[str] = None, sweep_range_hz=None, n_freq=None) -> dict:
        """Dense mechanical FRF for one voltage pattern and selected output metric."""
        ms = self.mode_settings
        output = canonical_output_name(output or ms.output)
        sweep_range_hz = sweep_range_hz or ms.final_sweep_range_hz
        n_freq = int(n_freq or ms.final_sweep_n_freq)
        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        D = fe.effective_damping_matrix() if hasattr(fe, "effective_damping_matrix") else fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red

        metric = np.zeros(n_freq, dtype=float)
        tip_disp = np.zeros(n_freq, dtype=float)
        mean_abs = np.zeros(n_freq, dtype=float)
        rms = np.zeros(n_freq, dtype=float)
        response_complex_tip = np.zeros(n_freq, dtype=complex)

        for k, omega in enumerate(omega_vec):
            Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
            u_red = np.linalg.solve(Z, fe.Gamma_red @ voltage_vector)
            metric[k] = evaluate_output_metric(fe, u_red, output)
            tip_disp[k] = evaluate_output_metric(fe, u_red, "tip")
            mean_abs[k] = evaluate_output_metric(fe, u_red, "mean_abs")
            rms[k] = evaluate_output_metric(fe, u_red, "rms")
            response_complex_tip[k] = u_red[tip_reduced_index(fe)]

        return {
            "freq": freq,
            "omega": omega_vec,
            "output": output,
            "metric_label": metric_label(output),
            "metric": metric,
            "tip_disp": tip_disp,
            "mean_abs": mean_abs,
            "rms": rms,
            "response_complex": response_complex_tip,
            "voltage_vector": voltage_vector,
        }

    def dense_tip_frf_for_plot(self, fe, voltage_vector, *, sweep_range_hz=None, n_freq=None) -> dict:
        """Backward-compatible dense tip FRF for one voltage pattern."""
        return self.dense_metric_frf_for_plot(
            fe,
            voltage_vector,
            output="tip",
            sweep_range_hz=sweep_range_hz,
            n_freq=n_freq,
        )

    def dense_all_binary_metric_frf_for_plot(self, fe, *, output: Optional[str] = None, sweep_range_hz=None, n_freq=None) -> List[dict]:
        """Dense FRFs for all binary sign patterns using the selected output metric."""
        n = fe.Gamma_red.shape[1]
        out = []
        for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
            signs = np.asarray(signs_tuple, dtype=float)
            frf = self.dense_metric_frf_for_plot(
                fe,
                self.mode_settings.voltage_amplitude * signs,
                output=output or self.mode_settings.output,
                sweep_range_hz=sweep_range_hz,
                n_freq=n_freq,
            )
            frf["signs"] = signs
            frf["label"] = sign_label(signs)
            out.append(frf)
        return out

    def dense_all_binary_frf_for_plot(self, fe, *, sweep_range_hz=None, n_freq=None) -> List[dict]:
        """Backward-compatible dense tip FRFs for all binary sign patterns."""
        return self.dense_all_binary_metric_frf_for_plot(fe, output="tip", sweep_range_hz=sweep_range_hz, n_freq=n_freq)


def build_region_types_from_params(params: PiezoBeamParams, *, h_patch: float = 1e-3, h_gap: float = 1e-3) -> dict:
    """Convenience region_types dictionary for simple piezo/substrate layouts."""
    rhoA_patch = params.b * (params.rho_s * params.hs + 2.0 * params.rho_p * params.hp)
    rhoA_gap = params.b * params.rho_s * params.hs
    return {
        "piezo": {"EI": params.YI, "rhoA": rhoA_patch, "h": h_patch},
        "substrate": {"EI": params.YI_s, "rhoA": rhoA_gap, "h": h_gap},
    }
