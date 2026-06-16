"""Traveling-wave objective evaluator and traveling-index metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ..model.model_helpers import (
    reduced_to_full_displacement_nodes,
    response_columns,
    trapezoid_node_weights,
)
from .optimizer_helpers import (
    OUTPUT_TRAVELING_WAVE,
    optimize_binary_phases,
    optimize_continuous_phases,
    response_summary,
)


# -----------------------------------------------------------------------------
# Traveling-Wave Frequency and Window Helpers
# -----------------------------------------------------------------------------

def traveling_wave_frequency_bounds_from_settings(fe, settings: Dict[str, Any]) -> Tuple[float, float]:
    """Return frequency bounds for nested traveling-wave frequency optimization."""
    bounds = settings["frequency_bounds_hz"]
    if bounds is not None:
        lo, hi = (float(v) for v in bounds)
        if not (0.0 < lo < hi):
            raise ValueError("traveling_wave_settings['frequency_bounds_hz'] must satisfy 0 < lo < hi")
        return lo, hi

    mode_pair = tuple(int(m) for m in settings["mode_pair"])
    if len(mode_pair) != 2:
        raise ValueError("traveling_wave_settings['mode_pair'] must contain two mode numbers")
    if any(m < 1 or m > len(fe.freq) for m in mode_pair):
        raise ValueError(f"mode_pair={mode_pair} outside available mode range 1..{len(fe.freq)}")

    f0 = float(fe.freq[mode_pair[0] - 1])
    f1 = float(fe.freq[mode_pair[1] - 1])
    lo, hi = min(f0, f1), max(f0, f1)
    if not (0.0 < lo < hi):
        raise ValueError("Computed traveling-wave frequency bounds must satisfy 0 < lo < hi")
    return lo, hi


def traveling_wave_node_window(fe, settings: Dict[str, Any]) -> np.ndarray:
    """Boolean node mask for spatial traveling-wave metrics."""
    x = np.asarray(fe.geom.x_nodes, dtype=float)
    L = float(x[-1] - x[0])
    if L <= 0:
        raise ValueError("Beam length must be positive")

    lo, hi = settings["x_fraction_bounds"]
    lo = float(lo)
    hi = float(hi)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("traveling_wave_settings['x_fraction_bounds'] must satisfy 0 <= lo < hi <= 1")

    eta = (x - x[0]) / L
    mask = (eta >= lo) & (eta <= hi)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Traveling-wave spatial window must include at least three nodes")
    return mask


# -----------------------------------------------------------------------------
# Traveling-Wave Shape Metrics
# -----------------------------------------------------------------------------

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / denom)


def traveling_index_from_complex_shape(W: np.ndarray, weights=None, eps: float = 1e-300) -> float:
    """Feeny-style traveling index from one complex harmonic spatial shape."""
    W = np.asarray(W, dtype=complex)
    A = np.column_stack([np.real(W), np.imag(W)])
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != W.shape:
            raise ValueError("weights must have the same shape as W")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        A = np.sqrt(weights)[:, None] * A
    if A.shape[0] < 2:
        return 0.0
    s = np.linalg.svd(A, compute_uv=False)
    if s.size < 2 or s[0] <= eps:
        return 0.0
    return float(np.clip(s[-1] / s[0], 0.0, 1.0))


def traveling_wave_metrics(fe, u_red: np.ndarray, settings: Dict[str, Any]) -> dict:
    """Evaluate traveling index for one complex harmonic response."""
    eps = float(settings["eps"])
    x_full = np.asarray(fe.geom.x_nodes, dtype=float)
    W_full = reduced_to_full_displacement_nodes(fe, u_red)
    mask = traveling_wave_node_window(fe, settings)

    x = x_full[mask]
    W = W_full[mask]
    weights = trapezoid_node_weights(x_full)[mask]
    if np.sum(weights) <= 0:
        weights = np.ones_like(x, dtype=float)

    amp = np.abs(W)
    traveling_index = traveling_index_from_complex_shape(W, weights=weights, eps=eps)
    amplitude_rms = float(np.sqrt(_weighted_mean(amp ** 2, weights)))

    return {
        "score": float(traveling_index),
        "traveling_index": float(traveling_index),
        "amplitude_rms": amplitude_rms,
        "x": x,
        "W": W,
        "x_full": x_full,
        "W_full": W_full,
        "x_mask": mask,
    }


def compact_traveling_wave_metrics(metrics: dict) -> dict:
    """Return scalar traveling-wave metrics suitable for logs/dataframes."""
    scalar_keys = [
        "score",
        "traveling_index",
        "amplitude_rms",
    ]
    return {k: metrics.get(k) for k in scalar_keys if k in metrics}


# -----------------------------------------------------------------------------
# Traveling-Wave Objective
# -----------------------------------------------------------------------------

class TravelingWaveOptimizer:
    """Traveling-wave response objective."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def evaluate(self, fe) -> dict:
        ms = self.settings
        settings = dict(ms.traveling_wave_settings)
        bounds = traveling_wave_frequency_bounds_from_settings(fe, settings)
        xatol = float(settings["frequency_xtol"])

        def negative_score(freq_hz: float) -> float:
            return -self._phase_result_at_frequency(fe, freq_hz, settings)[2]["score"]

        opt = minimize_scalar(
            negative_score,
            bounds=bounds,
            method="bounded",
            options={"xatol": xatol},
        )
        candidates = [float(bounds[0]), float(bounds[1])]
        if np.isfinite(getattr(opt, "x", np.nan)):
            candidates.append(float(opt.x))

        result = max(
            (self._evaluate_at_frequency(fe, freq_hz, settings, bounds) for freq_hz in candidates),
            key=lambda r: r["score"],
        )
        result["frequency_bounds_hz"] = bounds
        result["frequency_optimizer"] = "bounded_minimize_scalar"
        result["frequency_optimized"] = True
        return result

    def _score_traveling_wave(self, fe, u_red: np.ndarray, settings: dict) -> tuple[float, dict, dict]:
        metrics = traveling_wave_metrics(fe, u_red, settings)
        return float(metrics["score"]), response_summary(fe, u_red, "rms"), metrics

    def _phase_result_at_frequency(self, fe, freq_hz: float, settings: dict) -> tuple[float, float, dict]:
        ms = self.settings
        freq_hz = float(freq_hz)
        if freq_hz <= 0:
            raise ValueError("Traveling-wave optimization frequency must be positive")
        omega = 2.0 * np.pi * freq_hz
        eval_settings = dict(settings)
        U_cols = response_columns(fe, omega)

        def score_function(fe_inner, u_red):
            return self._score_traveling_wave(fe_inner, u_red, eval_settings)

        phase_mode = ms.phase_mode.lower().strip()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases(
                fe,
                U_cols,
                OUTPUT_TRAVELING_WAVE,
                ms.voltage_amplitude,
                score_function=score_function,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases(
                fe,
                U_cols,
                OUTPUT_TRAVELING_WAVE,
                ms.voltage_amplitude,
                score_function=score_function,
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")
        return freq_hz, omega, phase_result

    def _evaluate_at_frequency(
        self,
        fe,
        freq_hz: float,
        settings: dict,
        frequency_bounds_hz: tuple[float, float],
    ) -> dict:
        ms = self.settings
        freq_hz, omega, phase_result = self._phase_result_at_frequency(fe, freq_hz, settings)
        metrics = phase_result["traveling_wave_metrics"]
        return {
            "objective": "traveling_wave",
            "score": float(phase_result["score"]),
            "omega": omega,
            "freq_hz": freq_hz,
            "phase_mode": ms.phase_mode,
            "phase_optimizer": phase_result["phase_optimizer"],
            "output": "traveling_wave",
            "metric_label": "Traveling-wave objective score [-]",
            **phase_result,
            "score": float(phase_result["score"]),
            "response_metrics": {
                **phase_result["response_metrics"],
                "selected": float(phase_result["score"]),
                "output": "traveling_wave",
                "traveling_wave": compact_traveling_wave_metrics(metrics),
            },
            "traveling_wave_settings": settings,
            "frequency_bounds_hz": frequency_bounds_hz,
            "traveling_wave_metrics": metrics,
        }
