"""Thrust-based prescribed-voltage objective evaluators."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model.model_helpers import (
    complex_power_from_peak_phasors,
    lighthill_thrust,
    response_columns,
)
from .optimizer_helpers import (
    OUTPUT_THRUST,
    OUTPUT_THRUST_PER_POWER,
    evaluate_output_metric,
    metric_label,
    optimize_binary_phases,
    optimize_continuous_phases,
    response_summary,
)


# -----------------------------------------------------------------------------
# Thrust Score Helpers
# -----------------------------------------------------------------------------

def thrust_width_from_settings(fe, settings: dict) -> float:
    """Return thrust width, defaulting to the piezo/beam width in FE params."""
    width = settings.get("width", None)
    if width is None:
        width = getattr(fe.params, "b", None)
    if width is None:
        raise ValueError("thrust_settings['width'] must be set when fe.params.b is unavailable")
    return float(width)


def frequency_from_mode_or_setting(fe, settings: dict) -> tuple[float, float, int | None]:
    """Return (omega, freq_hz, mode_number) from explicit frequency or mode number."""
    if settings.get("frequency_hz", None) is not None:
        freq_hz = float(settings["frequency_hz"])
        if freq_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        return 2.0 * np.pi * freq_hz, freq_hz, None

    mode_number = int(settings["mode_number"])
    if mode_number < 1 or mode_number > len(fe.freq):
        raise ValueError(f"mode_number={mode_number} outside available mode range 1..{len(fe.freq)}")
    return float(fe.omega[mode_number - 1]), float(fe.freq[mode_number - 1]), mode_number


def thrust_from_response(fe, u_red, omega: float, settings: dict) -> float:
    """Evaluate net Lighthill thrust, optionally clipping negative thrust."""
    raw = lighthill_thrust(
        fe,
        u_red,
        omega,
        rho=settings["rho"],
        width=thrust_width_from_settings(fe, settings),
        beta=settings["beta"],
        swimming_speed=settings["swimming_speed"],
    )
    return float(max(raw, 0.0) if settings.get("clip_negative_thrust", True) else raw)


def current_from_prescribed_voltage(fe, omega: float, voltage_vector, response_red) -> np.ndarray:
    """Patch current phasor for prescribed voltage and known mechanical response."""
    voltage_vector = np.asarray(voltage_vector, dtype=complex)
    return 1j * omega * (fe.Cp @ voltage_vector + fe.Gamma_red.T @ response_red)


def electrical_power_denominator(power, settings: dict) -> float:
    """Positive denominator used by thrust-per-power objective variants."""
    mode = settings["power_denominator"]
    eps = float(settings["power_eps"])
    if mode == "apparent":
        return max(float(abs(power)), eps)
    real_power = float(np.real(power))
    if mode == "real_positive":
        return max(real_power, eps)
    if mode == "abs_real":
        return max(abs(real_power), eps)
    raise ValueError("power_denominator must be abs_real, real_positive, or apparent")


def thrust_response_metrics(fe, u_red, output: str, thrust: float, extra: dict) -> dict:
    """Common response metrics plus thrust-specific scalars."""
    metrics = response_summary(fe, u_red, "rms")
    metrics.update(
        {
            "selected": float(thrust if output == OUTPUT_THRUST else extra["score"]),
            "output": output,
            "thrust": float(thrust),
            "tip": evaluate_output_metric(fe, u_red, "tip"),
            "mean_abs": evaluate_output_metric(fe, u_red, "mean_abs"),
            "rms": evaluate_output_metric(fe, u_red, "rms"),
        }
    )
    metrics.update(extra)
    return metrics


# -----------------------------------------------------------------------------
# Thrust Objectives
# -----------------------------------------------------------------------------

class ThrustOptimizer:
    """Maximize Lighthill thrust for a prescribed-voltage actuator layout."""

    output = OUTPUT_THRUST

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def _score(self, omega: float, settings: dict):
        def score_function(fe, u_red, voltage_vector=None):
            thrust = thrust_from_response(fe, u_red, omega, settings)
            extra = {"score": thrust, "raw_thrust": thrust}
            return thrust, thrust_response_metrics(fe, u_red, self.output, thrust, extra), extra

        return score_function

    def _phase_result(self, fe, omega: float, settings: dict) -> dict:
        ms = self.settings
        U_cols = response_columns(fe, omega)
        phase_mode = ms.phase_mode.lower().strip()
        if phase_mode == "binary":
            return optimize_binary_phases(
                fe,
                U_cols,
                self.output,
                ms.voltage_amplitude,
                score_function=self._score(omega, settings),
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        if phase_mode == "continuous":
            return optimize_continuous_phases(
                fe,
                U_cols,
                self.output,
                ms.voltage_amplitude,
                score_function=self._score(omega, settings),
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        raise ValueError("phase_mode must be 'binary' or 'continuous'")

    def evaluate(self, fe) -> dict:
        settings = dict(self.settings.thrust_settings)
        omega, freq_hz, mode_number = frequency_from_mode_or_setting(fe, settings)
        phase_result = self._phase_result(fe, omega, settings)
        return {
            "objective": "thrust",
            "mode_number": mode_number,
            "omega": omega,
            "freq_hz": freq_hz,
            "output": self.output,
            "metric_label": metric_label(self.output),
            "thrust_settings": settings,
            **phase_result,
        }


class ThrustPerPowerOptimizer(ThrustOptimizer):
    """Maximize thrust divided by prescribed-voltage electrical power."""

    output = OUTPUT_THRUST_PER_POWER

    def _score(self, omega: float, settings: dict):
        def score_function(fe, u_red, voltage_vector=None):
            if voltage_vector is None:
                raise ValueError("thrust_per_power objective requires the voltage vector")
            thrust = thrust_from_response(fe, u_red, omega, settings)
            current = current_from_prescribed_voltage(fe, omega, voltage_vector, u_red)
            power = complex_power_from_peak_phasors(voltage_vector, current)
            denominator = electrical_power_denominator(power, settings)
            score = float(thrust / denominator)
            extra = {
                "score": score,
                "thrust": float(thrust),
                "complex_power": power,
                "real_power": float(np.real(power)),
                "apparent_power": float(abs(power)),
                "power_denominator": denominator,
            }
            return score, thrust_response_metrics(fe, u_red, self.output, thrust, extra), extra

        return score_function

    def evaluate(self, fe) -> dict:
        result = super().evaluate(fe)
        result["objective"] = "thrust_per_power"
        result["output"] = self.output
        result["metric_label"] = metric_label(self.output)
        return result
