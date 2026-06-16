"""Weighted multifunctional prescribed-voltage objective."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model.model_helpers import response_columns
from .optimizer_helpers import (
    OUTPUT_MULTIFUNCTIONAL,
    evaluate_output_metric,
    metric_label,
    optimize_binary_phases,
    optimize_continuous_phases,
    response_summary,
)
from .thrust_optimizer import frequency_from_mode_or_setting, thrust_from_response
from .traveling_wave_optimizer import compact_traveling_wave_metrics, traveling_wave_metrics


# -----------------------------------------------------------------------------
# Multifunctional Scoring
# -----------------------------------------------------------------------------

def _normalized_component(value: float, name: str, settings: dict) -> float:
    normalizer = float(settings["normalizers"].get(name, 1.0))
    if normalizer == 0:
        normalizer = 1.0
    return float(value / normalizer)


class MultifunctionalOptimizer:
    """Optimize a weighted actuation score without harvesting physics."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def _score(self, omega: float, settings: dict, thrust_settings: dict, tw_settings: dict):
        def score_function(fe, u_red, voltage_vector=None):
            components = {}
            if "response" in settings["components"]:
                components["response"] = evaluate_output_metric(fe, u_red, settings["response_output"])
            if "thrust" in settings["components"]:
                components["thrust"] = thrust_from_response(fe, u_red, omega, thrust_settings)
            if "traveling_wave" in settings["components"]:
                components["traveling_wave"] = traveling_wave_metrics(fe, u_red, tw_settings)["score"]

            score = 0.0
            normalized = {}
            for name, value in components.items():
                normalized[name] = _normalized_component(value, name, settings)
                score += float(settings["weights"].get(name, 0.0)) * normalized[name]

            response_metrics = response_summary(fe, u_red, settings["response_output"])
            response_metrics.update(
                {
                    "selected": float(score),
                    "output": OUTPUT_MULTIFUNCTIONAL,
                    "components": components,
                    "normalized_components": normalized,
                }
            )
            extra = {"components": components, "normalized_components": normalized}
            if "traveling_wave" in components:
                extra["traveling_wave"] = compact_traveling_wave_metrics(
                    traveling_wave_metrics(fe, u_red, tw_settings)
                )
            return float(score), response_metrics, extra

        return score_function

    def evaluate(self, fe) -> dict:
        ms = self.settings
        settings = dict(ms.multifunctional_settings)
        thrust_settings = dict(ms.thrust_settings)
        thrust_settings["mode_number"] = settings["mode_number"]
        thrust_settings["frequency_hz"] = settings["frequency_hz"]
        omega, freq_hz, mode_number = frequency_from_mode_or_setting(fe, thrust_settings)

        tw_settings = dict(ms.traveling_wave_settings)
        U_cols = response_columns(fe, omega)
        phase_mode = ms.phase_mode.lower().strip()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases(
                fe,
                U_cols,
                OUTPUT_MULTIFUNCTIONAL,
                ms.voltage_amplitude,
                score_function=self._score(omega, settings, thrust_settings, tw_settings),
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases(
                fe,
                U_cols,
                OUTPUT_MULTIFUNCTIONAL,
                ms.voltage_amplitude,
                score_function=self._score(omega, settings, thrust_settings, tw_settings),
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")

        return {
            "objective": "multifunctional",
            "mode_number": mode_number,
            "omega": omega,
            "freq_hz": freq_hz,
            "output": OUTPUT_MULTIFUNCTIONAL,
            "metric_label": metric_label(OUTPUT_MULTIFUNCTIONAL),
            "multifunctional_settings": settings,
            "thrust_settings": thrust_settings,
            "traveling_wave_settings": tw_settings,
            **phase_result,
        }
