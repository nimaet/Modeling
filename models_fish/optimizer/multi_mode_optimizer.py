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
    reduction: str = "weighted_sum",
) -> dict:
    """Combine raw per-mode scores into one scalar objective."""
    raw = np.asarray(raw_scores, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw_scores must be a nonempty 1D sequence")

    if weights is None:
        w = np.ones_like(raw)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != raw.shape:
            raise ValueError("weights must have the same length as raw_scores")

    weighted = w * raw
    name = reduction.lower().strip().replace("-", "_")

    if name == "weighted_sum":
        score = float(np.sum(weighted))
    elif name == "weighted_mean":
        denom = float(np.sum(np.abs(w)))
        score = float(np.sum(weighted) / denom) if denom > 0 else float(np.mean(raw))
    elif name == "min":
        score = float(np.min(weighted))
    elif name == "geometric_mean":
        eps = 1e-300
        vals = np.maximum(weighted, eps)
        score = float(np.exp(np.mean(np.log(vals))))
    else:
        raise ValueError("multi_mode_reduction must be weighted_sum, weighted_mean, min, or geometric_mean")

    return {
        "score": score,
        "raw_scores": raw,
        "weights": w,
        "weighted_scores": weighted,
        "reduction": name,
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
            "weighted_mode_scores": reduction["weighted_scores"],
            "multi_mode_weights": reduction["weights"],
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
