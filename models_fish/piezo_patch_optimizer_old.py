"""Patch-placement optimizer for 1D piezoelectric beam FE models.

The optimizer intentionally keeps the notebook thin: configure settings, call
``PiezoPatchOptimizer.run()``, then plot/post-process results.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from tqdm.auto import tqdm

try:  # package imports
    from Modeling.models.beam_properties import PiezoBeamParams
    import Modeling.models.FE3 as FE_module
except Exception:  # local / notebook fallback
    from beam_properties_fish import PiezoBeamParams
    import FE_fish as FE_module


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
    output: str = "tip"         # currently only "tip" is implemented
    final_sweep_range_hz: Tuple[float, float] = (0.1, 10.0)
    final_sweep_n_freq: int = 1000


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


def tip_reduced_index(fe) -> int:
    """Reduced DOF index for tip transverse displacement."""
    tip_full_dof = 2 * (len(fe.geom.x_nodes) - 1)
    idx = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(idx) != 1:
        raise RuntimeError("Could not find tip displacement DOF in reduced system")
    return int(idx[0])


def output_reduced_index(fe, output: str = "tip") -> int:
    if output != "tip":
        raise NotImplementedError("Only output='tip' is implemented in the refactored single-mode optimizer")
    return tip_reduced_index(fe)


def optimize_binary_phases(h: np.ndarray, voltage_amplitude: float = 1.0) -> dict:
    """Brute-force binary signs for a complex unit-patch response vector."""
    h = np.asarray(h, dtype=complex)
    n = h.size
    best_score = -np.inf
    best_signs = None
    best_response = None
    all_results = []

    for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
        signs = np.asarray(signs_tuple, dtype=float)
        response = voltage_amplitude * np.dot(h, signs)
        score = abs(response)
        record = {"signs": signs, "response": response, "score": float(score), "label": sign_label(signs)}
        all_results.append(record)
        if score > best_score:
            best_score = score
            best_signs = signs
            best_response = response

    return {
        "phase_mode": "binary",
        "score": float(best_score),
        "response": best_response,
        "signs": best_signs,
        "phase_rad": np.where(best_signs > 0, 0.0, np.pi),
        "phase_deg": np.where(best_signs > 0, 0.0, 180.0),
        "voltage_vector": voltage_amplitude * best_signs.astype(complex),
        "all_phase_results": all_results,
    }


def optimize_continuous_phases(h: np.ndarray, voltage_amplitude: float = 1.0) -> dict:
    """Analytic continuous phase alignment for a scalar complex output.

    For y = sum_j h_j A exp(i alpha_j), the maximum magnitude is obtained by
    alpha_j = -angle(h_j) plus an arbitrary shared global phase.
    """
    h = np.asarray(h, dtype=complex)
    phase_rad = -np.angle(h)
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    response = np.dot(h, voltage_vector)
    return {
        "phase_mode": "continuous",
        "score": float(abs(response)),
        "response": response,
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(np.mod(phase_rad, 2 * np.pi)),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
    }


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

    def unit_patch_response(self, fe, omega: float, output: Optional[str] = None) -> np.ndarray:
        """Return h_j = output response per unit voltage on patch j at omega."""
        output = output or self.mode_settings.output
        idx_out = output_reduced_index(fe, output)
        D = fe.effective_damping_matrix() if hasattr(fe, "effective_damping_matrix") else fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red
        Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
        U_cols = np.linalg.solve(Z, fe.Gamma_red)
        return U_cols[idx_out, :]

    def evaluate_at_natural_frequency(self, fe) -> dict:
        """Evaluate selected mode at its geometry-dependent natural frequency."""
        ms = self.mode_settings
        m = int(ms.target_mode_number)
        if m < 1 or m > len(fe.freq):
            raise ValueError(f"target_mode_number={m} outside available mode range 1..{len(fe.freq)}")

        omega = float(fe.omega[m - 1])
        freq_hz = float(fe.freq[m - 1])
        h = self.unit_patch_response(fe, omega, ms.output)

        phase_mode = ms.phase_mode.lower()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases(h, ms.voltage_amplitude)
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases(h, ms.voltage_amplitude)
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")

        return {
            "target_mode_number": m,
            "omega": omega,
            "freq_hz": freq_hz,
            "h": h,
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
                "freq_hz": float(inner["freq_hz"]),
                "phase_mode": inner["phase_mode"],
                "signs": None if inner.get("signs") is None else np.asarray(inner["signs"]).copy(),
                "phase_deg": np.asarray(inner["phase_deg"]).copy(),
                "natural_freqs": fe.freq[: min(8, len(fe.freq))].copy(),
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
            # Preserve exact fixed variables.
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

    def dense_tip_frf_for_plot(self, fe, voltage_vector, *, sweep_range_hz=None, n_freq=None) -> dict:
        """Dense mechanical FRF for one voltage pattern using direct FE solves."""
        ms = self.mode_settings
        sweep_range_hz = sweep_range_hz or ms.final_sweep_range_hz
        n_freq = int(n_freq or ms.final_sweep_n_freq)
        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        idx = output_reduced_index(fe, ms.output)
        D = fe.effective_damping_matrix() if hasattr(fe, "effective_damping_matrix") else fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red
        tip_disp = np.zeros(n_freq, dtype=float)
        response_complex = np.zeros(n_freq, dtype=complex)
        for k, omega in enumerate(omega_vec):
            Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
            u = np.linalg.solve(Z, fe.Gamma_red @ voltage_vector)
            response_complex[k] = u[idx]
            tip_disp[k] = abs(u[idx])
        return {"freq": freq, "omega": omega_vec, "tip_disp": tip_disp, "response_complex": response_complex, "voltage_vector": voltage_vector}

    def dense_all_binary_frf_for_plot(self, fe, *, sweep_range_hz=None, n_freq=None) -> List[dict]:
        """Dense FRFs for all binary sign patterns."""
        n = fe.Gamma_red.shape[1]
        out = []
        for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
            signs = np.asarray(signs_tuple, dtype=float)
            frf = self.dense_tip_frf_for_plot(fe, self.mode_settings.voltage_amplitude * signs, sweep_range_hz=sweep_range_hz, n_freq=n_freq)
            frf["signs"] = signs
            frf["label"] = sign_label(signs)
            out.append(frf)
        return out


def build_region_types_from_params(params: PiezoBeamParams, *, h_patch: float = 1e-3, h_gap: float = 1e-3) -> dict:
    """Convenience region_types dictionary for simple piezo/substrate layouts."""
    rhoA_patch = params.b * (params.rho_s * params.hs + 2.0 * params.rho_p * params.hp)
    rhoA_gap = params.b * params.rho_s * params.hs
    return {
        "piezo": {"EI": params.YI, "rhoA": rhoA_patch, "h": h_patch},
        "substrate": {"EI": params.YI_s, "rhoA": rhoA_gap, "h": h_gap},
    }
