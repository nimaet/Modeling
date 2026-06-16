"""Patch-placement optimizer for 1D piezoelectric beam FE models.

Version 3 / multimode refactor
-----------------------------
This module keeps notebooks thin: configure settings, call
``PiezoPatchOptimizer.run()``, then plot/post-process results.

Supported objective kinds
- ``single_mode``: optimize one selected natural frequency/mode.
- ``multi_mode``: optimize a weighted aggregate of several modes. By default,
  each mode gets its own best phase vector for the same geometry. This answers:
  "Can one geometry actuate several target modes well if I can retune phase per
  mode?"
- ``traveling_wave``: optimize a harmonic response for traveling-wave quality
  using a Feeny-style traveling index with nested frequency/phase optimization.
- ``thrust``: optimize Lighthill thrust for prescribed-voltage actuation.
- ``thrust_per_power``: optimize thrust normalized by electrical power.
- ``admittance_match``: match a supplied complex/magnitude admittance target.
- ``multifunctional``: optimize a weighted score of response, thrust, and
  traveling-wave quality.

Supported output metrics
- ``tip``: tip displacement magnitude.
- ``mean_abs``: line-average of |w(x)|.
- ``rms``: RMS of |w(x)|.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from tqdm.auto import tqdm
from ..model.beam_properties import PiezoBeamParams
from ..model.model import PiezoBeamFE, build_geometry_from_types
from ..model.model_helpers import _copy_array_or_value
from .optimizer_settings import GeometrySettings, ObjectiveSettings, OptimizerSettings
from .optimizer_helpers import (
    OBJECTIVE_ADMITTANCE_MATCH,
    OBJECTIVE_MULTI_MODE,
    OBJECTIVE_MULTIFUNCTIONAL,
    OBJECTIVE_SINGLE_MODE,
    OBJECTIVE_THRUST,
    OBJECTIVE_THRUST_PER_POWER,
    OBJECTIVE_TRAVELING_WAVE,
    OBJECTIVE_TYPES,
    OUTPUT_TYPES,
)
from .admittance_optimizer import AdmittanceMatchOptimizer
from .multi_mode_optimizer import MultiModeOptimizer
from .multifunctional_optimizer import MultifunctionalOptimizer
from .single_mode_optimizer import SingleModeOptimizer
from .thrust_optimizer import ThrustOptimizer, ThrustPerPowerOptimizer
from .traveling_wave_optimizer import TravelingWaveOptimizer


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
        objective_settings: Optional[ObjectiveSettings] = None,
        optimizer_settings: Optional[OptimizerSettings] = None,
        *,
        default_h: float = 1e-3,
    ):
        if objective_settings is None:
            objective_settings = ObjectiveSettings()

        self.L = float(L)
        self.region_types = region_types
        self.base_params = base_params
        self.geometry_settings = geometry_settings
        self.objective_settings = objective_settings
        self.optimizer_settings = optimizer_settings or OptimizerSettings()
        self.default_h = default_h
        self.evaluation_history: list[dict] = []
        
        self.n_patches = geometry_settings.Np
        self.n_design_variables = 2 * self.n_patches - 1

    # Main Run Function
    def run(self) -> OptimizeResult:
        method = self.optimizer_settings.method.lower()
        if method == "differential_evolution":
            return self.run_differential_evolution()
        if method == "random":
            return self.run_random_search()
        if method == "powell":
            return self.run_powell_refinement()
        if method == "random_powell":
            r0 = self.run_random_search()
            r1 = self.run_powell_refinement(r0.x)
            r1.random_result = r0
            return r1
        raise ValueError(f"Unknown optimizer method: {self.optimizer_settings.method}")
    
    # Outer Optimizer Objective and Evaluation History
    def objective(self, z: np.ndarray) -> float:
        fe, layout, penalty = self.build_fe_for_design(z)
        if penalty > 0 or fe is None:
            return float(penalty)

        try:
            inner = self.inner_optimizer(fe)
            score = float(inner["score"])
        except Exception as exc:
            if bool(getattr(self.optimizer_settings, "raise_exceptions", False)):
                raise
            self.evaluation_history.append({
                "z": np.asarray(z, dtype=float).copy(),
                "layout": layout,
                "score": -np.inf,
                "error": repr(exc),
            })
            return self.geometry_settings.invalid_penalty

        self.evaluation_history.append(
            {
                "z": np.asarray(z, dtype=float).copy(),
                "layout": _copy_array_or_value(layout),
                "score": score,
                "objective": inner.get("objective"),
                "output": inner.get("output"),
                "freq_hz": _copy_array_or_value(inner.get("freq_hz")),
                "phase_mode": inner.get("phase_mode"),
                "phase_deg": _copy_array_or_value(inner.get("phase_deg")),
                "relative_phase_deg": _copy_array_or_value(inner.get("relative_phase_deg")),
                "natural_freqs": fe.freq[: min(8, len(fe.freq))].copy(),
                "response_metrics": _copy_array_or_value(inner.get("response_metrics", {})),
                "raw_mode_scores": _copy_array_or_value(inner.get("raw_mode_scores", None)),
            }
        )
        return -score

    # SciPy Outer Optimizer Methods
    def run_random_search(self) -> OptimizeResult:
        opt = self.optimizer_settings
        settings = opt.random_search_settings
        bounds = np.asarray(self.make_bounds(), dtype=float)
        rng = np.random.default_rng(opt.seed)
        best_x = None
        best_fun = np.inf
        n_random_samples = int(settings["n_random_samples"])
        iterator = range(n_random_samples)
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

        return self._finalize_scipy_result(
            OptimizeResult(x=best_x, fun=best_fun, success=True, message="Random search complete", nfev=n_random_samples),
            "random_search",
        )

    def run_powell_refinement(self, x0=None) -> OptimizeResult:
        opt = self.optimizer_settings
        settings = opt.powell_settings
        bounds = self.make_bounds()
        if x0 is None:
            x0 = np.array([(a + b) / 2 for a, b in bounds], dtype=float)
        result = minimize(
            self.objective,
            np.asarray(x0, dtype=float),
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": settings["maxiter"],
                "xtol": settings["xtol"],
                "ftol": settings["ftol"],
                "disp": opt.show_progress,
            },
        )
        return self._finalize_scipy_result(result, "powell")

    def run_differential_evolution(self) -> OptimizeResult:
        opt = self.optimizer_settings
        settings = opt.differential_evolution_settings
        workers = settings["workers"]
        result = differential_evolution(
            self.objective,
            bounds=self.make_bounds(),
            maxiter=settings["maxiter"],
            popsize=settings["popsize"],
            seed=opt.seed,
            polish=settings["polish"],
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
            disp=opt.show_progress,
        )
        return self._finalize_scipy_result(result, "differential_evolution")


    # Results Post Processing and Helpers
    def _finalize_scipy_result(self, result: OptimizeResult, optimizer_name: str) -> OptimizeResult:
        success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))
        result.optimizer_name = optimizer_name
        result.optimization_warning = None if success else f"{optimizer_name} did not report success: {message}"
        if result.optimization_warning and bool(getattr(self.optimizer_settings, "raise_optimizer_failures", False)):
            raise RuntimeError(result.optimization_warning)
        return result

    def best_eval_from_history(self) -> Optional[dict]:
        valid = [h for h in self.evaluation_history if np.isfinite(h.get("score", -np.inf))]
        if not valid:
            return None
        return max(valid, key=lambda h: h["score"])

    def inspect_result(self, result: OptimizeResult) -> dict:
        """Rebuild and evaluate the best design from an OptimizeResult."""
        fe, layout, penalty = self.build_fe_for_design(result.x)
        if penalty > 0 or fe is None:
            raise RuntimeError("Could not rebuild FE model for optimization result")
        inner = self.inner_optimizer(fe)
        return {"result": result, "fe": fe, "layout": layout, "penalty": penalty, "inner": inner}

    # Geometry Decoding and Constraints
    def make_region_sequence(self) -> List[str]:
        """Return ['piezo','substrate', ...] for N patch/substrate pairs."""
        if self.n_patches < 1:
            raise ValueError("n_patches must be at least 1")
        seq: List[str] = []
        for _ in range(int(self.n_patches)):
            seq.extend(["piezo", "substrate"])
        return seq

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
        tip_max = self.L if tip_max is None else tip_max
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

    # FE Model Construction

    def build_fe_for_design(self, z: np.ndarray):
        """Return (fe, layout, penalty) for a candidate design."""
        layout = self.decode_design(z)
        penalty = self.geometry_penalty(layout)
        if penalty > 0:
            return None, layout, penalty

        try:
            geom = build_geometry_from_types(
                L=self.L,
                region_types=self.region_types,
                region_sequence=layout["region_sequence"],
                x_starts=layout["x_starts"],
                default_h=self.default_h,
            )
            params = copy.deepcopy(self.base_params)
            params.geometry = geom
            params.sync_patch_count(len(geom.piezos))
            fe = PiezoBeamFE(params)
            return fe, layout, 0.0
        except Exception as exc:
            if getattr(self.optimizer_settings, "raise_exceptions", False):
                raise
            layout["error"] = repr(exc)
            return None, layout, self.geometry_settings.invalid_penalty

    # Inner Objective Dispatch
    def inner_objective(self):
        objective = self.objective_settings.objective
        if objective == OBJECTIVE_SINGLE_MODE:
            return SingleModeOptimizer(self)
        if objective == OBJECTIVE_MULTI_MODE:
            return MultiModeOptimizer(self)
        if objective == OBJECTIVE_TRAVELING_WAVE:
            return TravelingWaveOptimizer(self)
        if objective == OBJECTIVE_THRUST:
            return ThrustOptimizer(self)
        if objective == OBJECTIVE_THRUST_PER_POWER:
            return ThrustPerPowerOptimizer(self)
        if objective == OBJECTIVE_ADMITTANCE_MATCH:
            return AdmittanceMatchOptimizer(self)
        if objective == OBJECTIVE_MULTIFUNCTIONAL:
            return MultifunctionalOptimizer(self)
        raise RuntimeError(f"Unhandled objective {objective}")

    def inner_optimizer(self, fe) -> dict:
        """Evaluate the configured objective for a built FE model."""
        return self.inner_objective().evaluate(fe)
