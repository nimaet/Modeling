"""Multi-mode standing-wave objective evaluator."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .optimizer_helpers import OUTPUT_TYPES, metric_label
from .single_mode_optimizer import SingleModeOptimizer


# -----------------------------------------------------------------------------
# Multi-Mode Reduction
# -----------------------------------------------------------------------------

def reduce_multimode_scores(
    raw_scores: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    normalizers: Optional[Sequence[float]] = None,
    reduction: str = "weighted_sum",
    eps: float = 1e-300,
) -> dict:
    """Normalize, weight, and reduce per-mode scores into one objective.

    If ``normalizers`` is None, ``normalized_scores`` equals ``raw_scores`` and
    existing reductions preserve the previous numerical behavior. New balance-
    style reductions are just ordinary reduction names and operate on weighted
    normalized scores, same as the other reduction methods.
    """
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
        n = None
        normalized = raw.copy()
    else:
        n = np.asarray(normalizers, dtype=float)
        if n.shape != raw.shape:
            raise ValueError("normalizers must have the same length as raw_scores")
        if np.any(n <= 0.0):
            raise ValueError("normalizers must all be positive")
        normalized = raw / n

    weighted = w * normalized
    name = reduction.lower().strip().replace("-", "_")

    if name == "weighted_sum":
        score = float(np.sum(weighted))
    elif name == "weighted_mean":
        denom = float(np.sum(np.abs(w)))
        score = float(np.sum(weighted) / denom) if denom > 0 else float(np.mean(normalized))
    elif name == "min":
        score = float(np.min(weighted))
    elif name == "geometric_mean":
        vals = np.maximum(weighted, eps)
        score = float(np.exp(np.mean(np.log(vals))))
    elif name == "min_max_ratio":
        score = float(np.min(weighted) / (np.max(weighted) + eps))
    elif name == "cv":
        cv = float(np.std(weighted) / (np.mean(weighted) + eps))
        score = float(1.0 / (1.0 + cv))
    elif name == "gm_am_ratio":
        gm = float(np.exp(np.mean(np.log(np.maximum(weighted, eps)))))
        am = float(np.mean(weighted))
        score = float(gm / (am + eps))
    else:
        raise ValueError(
            "multi_mode_reduction must be weighted_sum, weighted_mean, min, "
            "geometric_mean, min_max_ratio, cv, or gm_am_ratio"
        )

    return {
        "score": score,
        "raw_scores": raw,
        "normalized_scores": normalized,
        "normalizers": n,
        "weights": w,
        "weighted_scores": weighted,
        "reduction": name,
    }


# -----------------------------------------------------------------------------
# External Normalizer Calibration
# -----------------------------------------------------------------------------

def calibrate_multimode_normalizers(
    optimizer: Any,
    fe,
    mode_numbers: Optional[Sequence[int]] = None,
) -> dict:
    """Compute external per-mode normalizers from a reference FE model.

    This helper does not change optimizer settings. Use the returned
    ``multi_mode_normalizers`` in a later ``ObjectiveSettings`` object if you want
    normalization. Leaving that settings field as None preserves old behavior.
    """
    settings = optimizer.objective_settings
    standing = settings.standing_wave_settings
    modes = tuple(int(m) for m in (mode_numbers or standing["multi_mode_numbers"]))
    if len(modes) == 0:
        raise ValueError("mode_numbers must contain at least one mode")

    mode_objective = SingleModeOptimizer(optimizer)
    mode_results = [mode_objective.evaluate_mode(fe, m) for m in modes]
    normalizers = np.asarray([r["score"] for r in mode_results], dtype=float)
    if np.any(normalizers <= 0.0):
        raise ValueError("calibrated multi-mode normalizers must all be positive")

    return {
        "multi_mode_numbers": modes,
        "multi_mode_normalizers": normalizers,
        "mode_results": mode_results,
        "output": settings.output,
        "phase_mode": settings.phase_mode,
    }


# -----------------------------------------------------------------------------
# Multi-Mode Objective
# -----------------------------------------------------------------------------

class MultiModeOptimizer:
    """Multi-mode weighted response objective."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings
        self.mode_objective = SingleModeOptimizer(optimizer)

    def calibrate_normalizers(self, fe, mode_numbers: Optional[Sequence[int]] = None) -> dict:
        """Convenience wrapper for external normalizer calibration."""
        return calibrate_multimode_normalizers(self.optimizer, fe, mode_numbers=mode_numbers)

    def evaluate(self, fe) -> dict:
        ms = self.settings
        standing = ms.standing_wave_settings
        modes = tuple(int(m) for m in standing["multi_mode_numbers"])
        if len(modes) == 0:
            raise ValueError("multi_mode_numbers must contain at least one mode")

        mode_results = [self.mode_objective.evaluate_mode(fe, m) for m in modes]
        raw_scores = np.asarray([r["score"] for r in mode_results], dtype=float)
        reduction = reduce_multimode_scores(
            raw_scores,
            weights=standing["multi_mode_weights"],
            normalizers=standing["multi_mode_normalizers"],
            reduction=standing["multi_mode_reduction"],
        )
        output = ms.output
        if output not in OUTPUT_TYPES:
            raise ValueError("output must be exactly 'tip', 'mean_abs', or 'rms'")

        phase_deg = [r["phase_deg"] for r in mode_results]
        relative_phase_deg = [r["relative_phase_deg"] for r in mode_results]
        voltage_vectors = [r["voltage_vector"] for r in mode_results]

        return {
            "objective": "multi_mode",
            "multi_mode_numbers": modes,
            "mode_results": mode_results,
            "score": float(reduction["score"]),
            "raw_mode_scores": reduction["raw_scores"],
            "normalized_mode_scores": reduction["normalized_scores"],
            "weighted_mode_scores": reduction["weighted_scores"],
            "multi_mode_weights": reduction["weights"],
            "multi_mode_normalizers": reduction["normalizers"],
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
