"""Objective evaluators for piezo patch optimization."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from .actuation import (
    optimize_binary_phases_general,
    optimize_binary_phases_traveling_wave,
    optimize_continuous_phases_metric,
    optimize_continuous_phases_tip,
    optimize_continuous_phases_traveling_wave,
)
from .metrics import (
    canonical_output_name,
    compact_traveling_wave_metrics,
    default_traveling_wave_settings,
    metric_label,
    tip_reduced_index,
    traveling_wave_frequency_from_settings,
)


def _maybe_raise_inner_warning(optimizer: Any, result: dict) -> None:
    warning = result.get("inner_opt_warning")
    if warning and bool(getattr(optimizer.optimizer_settings, "raise_optimizer_failures", False)):
        raise RuntimeError(str(warning))


def reduce_multimode_scores(
    raw_scores: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    normalizers: Optional[Sequence[float]] = None,
    reduction: str = "weighted_sum",
) -> dict:
    """Combine per-mode scores into one scalar objective."""
    raw = np.asarray(raw_scores, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw_scores must be a nonempty 1D sequence")

    if weights is None:
        w = np.ones_like(raw)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != raw.shape:
            raise ValueError("weights must have the same length as raw_scores")

    if normalizers is None:
        norm = np.ones_like(raw)
    else:
        norm = np.asarray(normalizers, dtype=float)
        if norm.shape != raw.shape:
            raise ValueError("normalizers must have the same length as raw_scores")
        if np.any(norm <= 0):
            raise ValueError("normalizers must be positive")

    normalized = raw / norm
    weighted = w * normalized
    name = reduction.lower().strip().replace("-", "_")

    if name in ("weighted_sum", "sum"):
        score = float(np.sum(weighted))
    elif name in ("weighted_mean", "mean", "average"):
        denom = float(np.sum(np.abs(w)))
        score = float(np.sum(weighted) / denom) if denom > 0 else float(np.mean(normalized))
    elif name in ("min", "maximin"):
        score = float(np.min(weighted))
    elif name in ("geometric_mean", "geom", "geom_mean"):
        eps = 1e-300
        vals = np.maximum(weighted, eps)
        score = float(np.exp(np.mean(np.log(vals))))
    else:
        raise ValueError("multi_mode_reduction must be weighted_sum, weighted_mean, min, or geometric_mean")

    return {
        "score": score,
        "raw_scores": raw,
        "weights": w,
        "normalizers": norm,
        "normalized_scores": normalized,
        "weighted_scores": weighted,
        "reduction": name,
    }


class ModeResponseObjective:
    """Single-mode response objective."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def evaluate_mode(self, fe, mode_number: int) -> dict:
        """Evaluate one mode and run the selected phase optimizer."""
        ms = self.settings
        m = int(mode_number)
        if m < 1 or m > len(fe.freq):
            raise ValueError(f"mode_number={m} outside available mode range 1..{len(fe.freq)}")

        omega = float(fe.omega[m - 1])
        freq_hz = float(fe.freq[m - 1])
        output = canonical_output_name(ms.output)
        U_cols = self.optimizer.response_columns(fe, omega)
        h_tip = U_cols[tip_reduced_index(fe), :]

        phase_mode = ms.phase_mode.lower().strip()
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
        _maybe_raise_inner_warning(self.optimizer, phase_result)

        return {
            "mode_number": m,
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

    def evaluate(self, fe) -> dict:
        mode_result = self.evaluate_mode(fe, int(self.settings.single_mode_number))
        return {
            "objective": "single_mode",
            "target_mode_number": mode_result["mode_number"],
            "single_mode_number": mode_result["mode_number"],
            "mode_results": [mode_result],
            **mode_result,
        }


class MultiModeObjective:
    """Multi-mode weighted response objective."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings
        self.mode_objective = ModeResponseObjective(optimizer)

    def evaluate(self, fe) -> dict:
        ms = self.settings
        modes = tuple(int(m) for m in ms.multi_mode_numbers)
        if len(modes) == 0:
            raise ValueError("multi_mode_numbers must contain at least one mode")

        mode_results = [self.mode_objective.evaluate_mode(fe, m) for m in modes]
        raw_scores = np.asarray([r["score"] for r in mode_results], dtype=float)
        reduction = reduce_multimode_scores(
            raw_scores,
            weights=ms.multi_mode_weights,
            normalizers=ms.multi_mode_score_normalizers,
            reduction=ms.multi_mode_reduction,
        )
        output = canonical_output_name(ms.output)

        # Store lists rather than ragged arrays for easy downstream access.
        phase_deg = [r["phase_deg"] for r in mode_results]
        relative_phase_deg = [r["relative_phase_deg"] for r in mode_results]
        voltage_vectors = [r["voltage_vector"] for r in mode_results]

        return {
            "objective": "multi_mode",
            "multi_mode_numbers": modes,
            "target_mode_number": modes,  # compatibility: this is now a tuple
            "single_mode_number": None,
            "mode_results": mode_results,
            "score": float(reduction["score"]),
            "raw_mode_scores": reduction["raw_scores"],
            "normalized_mode_scores": reduction["normalized_scores"],
            "weighted_mode_scores": reduction["weighted_scores"],
            "multi_mode_weights": reduction["weights"],
            "multi_mode_score_normalizers": reduction["normalizers"],
            "multi_mode_reduction": reduction["reduction"],
            "phase_mode": ms.phase_mode,
            "phase_optimizer": "per_mode",
            "output": output,
            "metric_label": f"Multi-mode {reduction['reduction']} of {metric_label(output)}",
            "freq_hz": np.asarray([r["freq_hz"] for r in mode_results], dtype=float),
            "omega": np.asarray([r["omega"] for r in mode_results], dtype=float),
            "phase_deg": phase_deg,
            "relative_phase_deg": relative_phase_deg,
            "phase_rad": [r["phase_rad"] for r in mode_results],
            "relative_phase_rad": [r["relative_phase_rad"] for r in mode_results],
            "voltage_vector": voltage_vectors,
            "signs": [r.get("signs", None) for r in mode_results],
            "response": [r["response"] for r in mode_results],
            "response_red": [r["response_red"] for r in mode_results],
            "response_metrics": {
                "per_mode": [r["response_metrics"] for r in mode_results],
                "selected": float(reduction["score"]),
                "output": output,
            },
            "all_phase_results": [r.get("all_phase_results", None) for r in mode_results],
        }


class TravelingWaveObjective:
    """Traveling-wave response objective."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def evaluate(self, fe) -> dict:
        ms = self.settings
        settings = default_traveling_wave_settings(ms.traveling_wave_settings)
        if settings.get("frequency_bounds_hz", None) is not None:
            return self._evaluate_frequency_optimized(fe, settings)

        freq_hz, _omega = traveling_wave_frequency_from_settings(fe, settings)
        return self._evaluate_at_frequency(
            fe,
            freq_hz,
            settings,
            frequency_metadata={
                "frequency_optimized": False,
                "frequency_bounds_hz": None,
                "frequency_optimizer": "fixed",
                "frequency_optimization_result": None,
            },
        )

    @staticmethod
    def _frequency_bounds_from_settings(settings: dict) -> tuple[float, float]:
        bounds = settings.get("frequency_bounds_hz", None)
        if bounds is None:
            raise ValueError("traveling_wave_settings['frequency_bounds_hz'] is required")
        if len(bounds) != 2:
            raise ValueError("traveling_wave_settings['frequency_bounds_hz'] must contain two values")
        lo, hi = (float(bounds[0]), float(bounds[1]))
        if not (0.0 < lo < hi):
            raise ValueError("traveling_wave_settings['frequency_bounds_hz'] must satisfy 0 < lo < hi")
        return lo, hi

    def _phase_result_at_frequency(self, fe, freq_hz: float, settings: dict) -> tuple[float, float, np.ndarray, dict]:
        ms = self.settings
        freq_hz = float(freq_hz)
        if freq_hz <= 0:
            raise ValueError("Traveling-wave optimization frequency must be positive")
        omega = 2.0 * np.pi * freq_hz
        eval_settings = dict(settings)
        eval_settings["frequency_hz"] = freq_hz
        U_cols = self.optimizer.response_columns(fe, omega)

        phase_mode = ms.phase_mode.lower().strip()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases_traveling_wave(
                fe,
                U_cols,
                eval_settings,
                ms.voltage_amplitude,
            )
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases_traveling_wave(
                fe,
                U_cols,
                eval_settings,
                ms.voltage_amplitude,
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
            )
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")
        _maybe_raise_inner_warning(self.optimizer, phase_result)
        return freq_hz, omega, U_cols, phase_result

    def _evaluate_at_frequency(
        self,
        fe,
        freq_hz: float,
        settings: dict,
        *,
        frequency_metadata: Optional[dict] = None,
    ) -> dict:
        ms = self.settings
        freq_hz, omega, U_cols, phase_result = self._phase_result_at_frequency(fe, freq_hz, settings)
        h_tip = U_cols[tip_reduced_index(fe), :]
        metrics = phase_result["traveling_wave_metrics"]
        result = {
            "objective": "traveling_wave",
            "target_mode_number": None,
            "single_mode_number": None,
            "mode_results": [],
            "score": float(phase_result["score"]),
            "omega": omega,
            "freq_hz": freq_hz,
            "phase_mode": ms.phase_mode,
            "phase_optimizer": phase_result["phase_optimizer"],
            "output": "traveling_wave",
            "metric_label": "Traveling-wave objective score [-]",
            "h": h_tip,
            "h_tip": h_tip,
            "U_cols": U_cols,
            **phase_result,
            "score": float(phase_result["score"]),
            "response_metrics": {
                **phase_result["response_metrics"],
                "selected": float(phase_result["score"]),
                "output": "traveling_wave",
                "traveling_wave": compact_traveling_wave_metrics(metrics),
            },
            "traveling_wave_settings": settings,
            "traveling_wave_metrics": metrics,
        }
        if frequency_metadata is not None:
            result.update(frequency_metadata)
        return result

    def _evaluate_frequency_optimized(self, fe, settings: dict) -> dict:
        bounds = self._frequency_bounds_from_settings(settings)
        reference_freq_hz, _reference_omega = traveling_wave_frequency_from_settings(fe, settings)
        xatol = float(settings.get("frequency_optimization_xatol", 1e-3))
        maxiter = int(settings.get("frequency_optimization_maxiter", 40))

        def neg_score(freq_hz: float) -> float:
            _freq_hz, _omega, _U_cols, phase_result = self._phase_result_at_frequency(
                fe,
                float(freq_hz),
                settings,
            )
            return -float(phase_result["score"])

        opt = minimize_scalar(
            neg_score,
            bounds=bounds,
            method="bounded",
            options={"xatol": xatol, "maxiter": maxiter},
        )
        freq_warning = None
        if not bool(opt.success):
            freq_warning = f"bounded_minimize_scalar did not report success: {opt.message}"
            if bool(getattr(self.optimizer.optimizer_settings, "raise_optimizer_failures", False)):
                raise RuntimeError(freq_warning)
        best_freq_hz = float(opt.x)
        result = self._evaluate_at_frequency(
            fe,
            best_freq_hz,
            settings,
            frequency_metadata={
                "frequency_optimized": True,
                "frequency_bounds_hz": bounds,
                "frequency_optimizer": "bounded_minimize_scalar",
                "frequency_reference_hz": float(reference_freq_hz),
                "frequency_optimization_result": {
                    "x": best_freq_hz,
                    "fun": float(opt.fun),
                    "score": float(-opt.fun),
                    "nfev": int(opt.nfev),
                    "nit": int(opt.nit),
                    "success": bool(opt.success),
                    "message": str(opt.message),
                    "xatol": xatol,
                    "maxiter": maxiter,
                },
                "frequency_optimization_warning": freq_warning,
            },
        )
        return result
