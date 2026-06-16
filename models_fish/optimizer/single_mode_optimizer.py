"""Single-mode standing-wave objective evaluator."""

from __future__ import annotations

from typing import Any

from ..model.model_helpers import response_columns
from .optimizer_helpers import (
    OUTPUT_TYPES,
    metric_label,
    optimize_binary_phases,
    optimize_continuous_phases,
)


# -----------------------------------------------------------------------------
# Single-Mode Objective
# -----------------------------------------------------------------------------

class SingleModeOptimizer:
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
        output = ms.output
        if output not in OUTPUT_TYPES:
            raise ValueError("output must be exactly 'tip', 'mean_abs', or 'rms'")
        U_cols = response_columns(fe, omega)

        phase_mode = ms.phase_mode.lower().strip()
        if phase_mode == "binary":
            phase_result = optimize_binary_phases(
                fe,
                U_cols,
                output,
                ms.voltage_amplitude,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        elif phase_mode == "continuous":
            phase_result = optimize_continuous_phases(
                fe,
                U_cols,
                output,
                ms.voltage_amplitude,
                n_starts=ms.continuous_phase_n_starts,
                seed=ms.continuous_phase_seed,
                method=ms.continuous_phase_method,
                inner_workers=ms.inner_workers,
                inner_parallel_min_tasks=ms.inner_parallel_min_tasks,
            )
        else:
            raise ValueError("phase_mode must be 'binary' or 'continuous'")

        return {
            "mode_number": m,
            "omega": omega,
            "freq_hz": freq_hz,
            "output": output,
            "metric_label": metric_label(output),
            **phase_result,
        }

    def evaluate(self, fe) -> dict:
        mode_result = self.evaluate_mode(fe, int(self.settings.standing_wave_settings["single_mode_number"]))
        return {
            "objective": "single_mode",
            **mode_result,
        }
