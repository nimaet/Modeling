"""Admittance-matching objective for prescribed-voltage model validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model.model_helpers import admittance_matrix, solve_harmonic_response, tip_reduced_index


# -----------------------------------------------------------------------------
# Target Parsing and Error Metrics
# -----------------------------------------------------------------------------

def _array_or_none(value, *, dtype=None):
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def admittance_error_score(model_values: np.ndarray, target_values: np.ndarray, settings: dict) -> dict:
    """Convert admittance mismatch into a positive score where 1 is perfect."""
    weights = _array_or_none(settings.get("weights", None), dtype=float)
    diff = np.asarray(model_values) - np.asarray(target_values)
    if settings["relative_error"]:
        diff = diff / np.maximum(np.abs(target_values), settings["eps"])
    error_values = np.abs(diff) ** 2
    if weights is not None:
        error_values = error_values * weights
        denom = max(float(np.sum(weights)), settings["eps"])
        rmse = float(np.sqrt(np.sum(error_values) / denom))
    else:
        rmse = float(np.sqrt(np.mean(error_values)))
    return {"score": float(1.0 / (1.0 + rmse)), "rmse": rmse}


# -----------------------------------------------------------------------------
# Admittance-Match Objective
# -----------------------------------------------------------------------------

class AdmittanceMatchOptimizer:
    """Match model admittance against supplied experimental or reference data."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.settings = optimizer.objective_settings

    def _voltage_vector(self, fe, settings: dict) -> np.ndarray:
        voltage = settings.get("voltage_vector", None)
        if voltage is None:
            return np.ones(fe.Gamma_red.shape[1], dtype=complex)
        voltage = np.asarray(voltage, dtype=complex)
        if voltage.shape != (fe.Gamma_red.shape[1],):
            raise ValueError("admittance_match_settings['voltage_vector'] length must match patch count")
        return voltage

    def evaluate(self, fe) -> dict:
        settings = dict(self.settings.admittance_match_settings)
        frequencies_hz = _array_or_none(settings["frequencies_hz"], dtype=float)
        if frequencies_hz is None or frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
            raise ValueError("admittance_match_settings['frequencies_hz'] must be a nonempty 1D array")
        if np.any(frequencies_hz <= 0):
            raise ValueError("admittance match frequencies must be positive")

        omega = 2.0 * np.pi * frequencies_hz
        admittance = np.asarray([admittance_matrix(fe, w) for w in omega])
        target_admittance = _array_or_none(settings.get("target_admittance", None), dtype=complex)
        target_magnitude = _array_or_none(settings.get("target_magnitude", None), dtype=float)

        if target_admittance is not None:
            if target_admittance.shape != admittance.shape:
                raise ValueError("target_admittance must have shape (n_freq, n_patches, n_patches)")
            score_info = admittance_error_score(admittance, target_admittance, settings)
            matched_quantity = "complex_admittance"
        elif target_magnitude is not None:
            magnitude = np.abs(admittance)
            if target_magnitude.shape != magnitude.shape:
                raise ValueError("target_magnitude must have shape (n_freq, n_patches, n_patches)")
            score_info = admittance_error_score(magnitude, target_magnitude, settings)
            matched_quantity = "admittance_magnitude"
        else:
            raise ValueError("Provide target_admittance or target_magnitude for admittance_match objective")

        voltage_vector = self._voltage_vector(fe, settings)
        response_red = solve_harmonic_response(fe, omega[0], voltage_vector)
        phase_rad = np.mod(np.angle(voltage_vector), 2.0 * np.pi)
        relative_phase_rad = np.mod(phase_rad - phase_rad[0], 2.0 * np.pi)

        return {
            "objective": "admittance_match",
            "score": float(score_info["score"]),
            "rmse": float(score_info["rmse"]),
            "matched_quantity": matched_quantity,
            "freq_hz": frequencies_hz,
            "omega": omega,
            "output": "admittance_match",
            "metric_label": "Admittance match score [-]",
            "phase_mode": "fixed_voltage",
            "phase_optimizer": "none",
            "all_phase_results": None,
            "signs": None,
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": relative_phase_rad,
            "relative_phase_deg": np.rad2deg(relative_phase_rad),
            "voltage_vector": voltage_vector,
            "response": response_red[tip_reduced_index(fe)],
            "response_red": response_red,
            "response_metrics": {
                "selected": float(score_info["score"]),
                "rmse": float(score_info["rmse"]),
                "output": "admittance_match",
            },
            "admittance": admittance,
            "admittance_match_settings": settings,
        }
