"""Generic outer optimization loop for patch-placement optimizers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from tqdm.auto import tqdm


def _copy_array_or_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_array_or_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_array_or_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _copy_array_or_value(v) for k, v in value.items()}
    return value


class GenericOuterOptimizer:
    """Reusable outer optimizer that delegates model-specific work to hooks.

    Subclasses provide geometry/FE and objective hooks:
    ``make_bounds()``, ``build_fe_for_design(z)``, and ``inner_optimizer(fe)``.
    """

    optimizer_settings: object
    geometry_settings: object
    evaluation_history: list

    def _raise_exceptions(self) -> bool:
        return bool(getattr(self.optimizer_settings, "raise_exceptions", False))

    def _raise_optimizer_failures(self) -> bool:
        return bool(getattr(self.optimizer_settings, "raise_optimizer_failures", False))

    def objective(self, z: np.ndarray) -> float:
        fe, layout, penalty = self.build_fe_for_design(z)
        if penalty > 0 or fe is None:
            return float(penalty)

        try:
            inner = self.inner_optimizer(fe)
            score = float(inner["score"])
        except Exception as exc:
            if self._raise_exceptions():
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

    def _finalize_scipy_result(self, result: OptimizeResult, optimizer_name: str) -> OptimizeResult:
        success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))
        result.optimizer_name = optimizer_name
        result.optimization_warning = None if success else f"{optimizer_name} did not report success: {message}"
        if result.optimization_warning and self._raise_optimizer_failures():
            raise RuntimeError(result.optimization_warning)
        return result

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

        return self._finalize_scipy_result(
            OptimizeResult(x=best_x, fun=best_fun, success=True, message="Random search complete", nfev=opt.n_random_samples),
            "random_search",
        )

    def run_powell_refinement(self, x0=None) -> OptimizeResult:
        opt = self.optimizer_settings
        bounds = self.make_bounds()
        if x0 is None:
            x0 = np.array([(a + b) / 2 for a, b in bounds], dtype=float)
        result = minimize(
            self.objective,
            np.asarray(x0, dtype=float),
            method="Powell",
            bounds=bounds,
            options={"maxiter": opt.powell_maxiter, "xtol": opt.powell_xtol, "ftol": opt.powell_ftol, "disp": opt.show_progress},
        )
        return self._finalize_scipy_result(result, "powell")

    def run_differential_evolution(self) -> OptimizeResult:
        opt = self.optimizer_settings
        result = differential_evolution(
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
        return self._finalize_scipy_result(result, "differential_evolution")

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
        inner = self.inner_optimizer(fe)
        return {"result": result, "fe": fe, "layout": layout, "penalty": penalty, "inner": inner}
