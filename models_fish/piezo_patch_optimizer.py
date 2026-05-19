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
  using a Feeny-style traveling index, bounded RMS amplitude reward, and
  envelope-uniformity reward.

Supported output metrics
- ``tip``: tip displacement magnitude.
- ``mean_abs`` / ``line_average``: line-average of |w(x)|.
- ``rms``: RMS of |w(x)|.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # package imports
    from Modeling.models_fish.beam_properties_fish import PiezoBeamParams
    import Modeling.models_fish.FE_fish as FE_module
except Exception:  # local / notebook fallback
    try:
        from beam_properties import PiezoBeamParams
    except Exception:  # support previous exported name
        from beam_properties_refactored import PiezoBeamParams
    try:
        import FE3 as FE_module
    except Exception:
        import FE3_refactored as FE_module


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

@dataclass
class GeometrySettings:
    """Outer geometry design settings.

    The design vector is always
    ``z = [L1, g12, L2, g23, ..., g(n-1,n), Ln]`` and has length ``2*Np - 1``.
    The final substrate length is the remaining beam length.
    """

    Np: int = 3
    patch_length_bounds: Tuple[float, float] = (10e-3, 40e-3)
    gap_bounds: Tuple[float, float] = (4e-3, 80e-3)
    tip_substrate_bounds: Tuple[float, Optional[float]] = (0.0, 150e-3)
    total_patch_length_bounds: Optional[Tuple[float, float]] = None
    fixed_patch_lengths: Dict[int, float] = field(default_factory=dict)  # patch index j = 0..Np-1
    fixed_gaps: Dict[int, float] = field(default_factory=dict)           # gap index j = 0..Np-2
    invalid_penalty: float = 1e12


@dataclass
class ObjectiveSettings:
    """Objective and inner phase-optimization settings.

    Parameters
    ----------
    objective:
        ``"single_mode"``, ``"multi_mode"``, or ``"traveling_wave"``.
    target_mode_number:
        Backward-compatible alias for ``single_mode_number``. If supplied, it
        overrides ``single_mode_number``.
    single_mode_number:
        Mode number used for ``objective="single_mode"``. Uses 1-based indexing.
    multi_mode_numbers:
        Mode numbers used for ``objective="multi_mode"``. Uses 1-based indexing.
    multi_mode_weights:
        Optional per-mode weights. If omitted, all modes get equal weight.
    multi_mode_score_normalizers:
        Optional per-mode positive normalizers. Useful when one mode naturally
        has a much larger response. If omitted, no normalization is applied.
    multi_mode_reduction:
        ``"weighted_sum"``/``"sum"``, ``"weighted_mean"``/``"mean"``,
        ``"min"``, or ``"geometric_mean"``.
    multi_mode_phase_policy:
        Currently only ``"per_mode"`` is implemented. This optimizes phase
        independently for each mode while sharing the same geometry.
    """

    objective: str = "single_mode"

    # Backward-compatible alias used by earlier notebooks/runners.
    target_mode_number: Optional[int] = None
    single_mode_number: int = 1
    multi_mode_numbers: Sequence[int] = field(default_factory=lambda: (1, 2, 3))
    multi_mode_weights: Optional[Sequence[float]] = None
    multi_mode_score_normalizers: Optional[Sequence[float]] = None
    multi_mode_reduction: str = "weighted_sum"
    multi_mode_phase_policy: str = "per_mode"

    # Traveling-wave objective settings. Common keys:
    #   frequency_hz: explicit excitation frequency.
    #   frequency_bounds_hz: optional (lo, hi) bounded inner frequency search.
    #   mode_pair: two 1-based modes; used with frequency_fraction if no
    #       frequency_hz is supplied.
    #   frequency_fraction: interpolation between mode_pair frequencies.
    #   amplitude_reference: A_ref in A_rms / (A_ref + A_rms). If None or <= 0,
    #       the amplitude term is disabled.
    #   x_fraction_bounds: spatial window used for wave-quality metrics.
    #   direction: "either", "positive_x"/"tailward", or "negative_x"/"headward".
    traveling_wave_settings: Dict[str, Any] = field(default_factory=dict)

    voltage_amplitude: float = 1.0
    phase_mode: str = "binary"  # "binary" or "continuous"

    # Output metric used in the objective.
    #   "tip" / "tip_disp": tip displacement magnitude.
    #   "mean_abs" / "line_average": integral average of |w(x)| over beam length.
    #   "rms" / "rms_displacement": RMS of |w(x)| over beam length.
    output: str = "tip"

    # Dense final sweep settings for post-processing.
    final_sweep_range_hz: Tuple[float, float] = (0.1, 10.0)
    final_sweep_n_freq: int = 1000

    # Only used when phase_mode="continuous" and output is not scalar tip.
    # For tip output, continuous phase has an analytic solution.
    continuous_phase_n_starts: int = 8
    continuous_phase_seed: Optional[int] = 1
    continuous_phase_method: str = "L-BFGS-B"

    def __post_init__(self):
        if self.target_mode_number is not None:
            self.single_mode_number = int(self.target_mode_number)

        self.objective = canonical_objective_name(self.objective)
        self.phase_mode = self.phase_mode.lower().strip()
        self.output = canonical_output_name(self.output)

        self.single_mode_number = int(self.single_mode_number)
        self.multi_mode_numbers = tuple(int(m) for m in self.multi_mode_numbers)

        if self.multi_mode_weights is not None:
            self.multi_mode_weights = tuple(float(w) for w in self.multi_mode_weights)
            if len(self.multi_mode_weights) != len(self.multi_mode_numbers):
                raise ValueError("multi_mode_weights must have the same length as multi_mode_numbers")

        if self.multi_mode_score_normalizers is not None:
            self.multi_mode_score_normalizers = tuple(float(v) for v in self.multi_mode_score_normalizers)
            if len(self.multi_mode_score_normalizers) != len(self.multi_mode_numbers):
                raise ValueError("multi_mode_score_normalizers must have the same length as multi_mode_numbers")
            if np.any(np.asarray(self.multi_mode_score_normalizers) <= 0):
                raise ValueError("multi_mode_score_normalizers must be positive")

        if self.multi_mode_phase_policy.lower().strip() != "per_mode":
            raise NotImplementedError("Only multi_mode_phase_policy='per_mode' is implemented for now")


# Backward-compatible name used by older notebooks.
SingleModeSettings = ObjectiveSettings


@dataclass
class CircuitSettings:
    R_c: float = 1e3
    K_p: float = 0.02
    K_i: float = 0.0
    K_c: float = 0.0


@dataclass
class OptimizerSettings:
    method: str = "differential_evolution"  # differential_evolution, random, powell, random_powell
    maxiter: int = 15
    popsize: int = 8
    seed: Optional[int] = 1
    polish: bool = False
    # Integer per scipy, or a map-like callable such as ThreadPool.map.
    workers: Any = 1
    n_random_samples: int = 300
    powell_maxiter: int = 80
    powell_xtol: float = 1e-4
    powell_ftol: float = 1e-4
    show_progress: bool = True
    raise_exceptions: bool = False
    raise_optimizer_failures: bool = False


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def canonical_objective_name(objective: str) -> str:
    name = objective.lower().strip().replace("-", "_")
    aliases = {
        "single": "single_mode",
        "single_mode": "single_mode",
        "mode": "single_mode",
        "multi": "multi_mode",
        "multimode": "multi_mode",
        "multi_mode": "multi_mode",
        "traveling": "traveling_wave",
        "travelling": "traveling_wave",
        "traveling_wave": "traveling_wave",
        "travelling_wave": "traveling_wave",
    }
    if name not in aliases:
        raise ValueError("objective must be 'single_mode', 'multi_mode', or 'traveling_wave'")
    return aliases[name]


def sign_label(signs) -> str:
    return "".join("+" if s > 0 else "-" for s in np.asarray(signs))


def make_region_sequence(n_patches: int) -> List[str]:
    """Return ['piezo','substrate', ...] for N patch/substrate pairs."""
    if n_patches < 1:
        raise ValueError("n_patches must be at least 1")
    seq: List[str] = []
    for _ in range(int(n_patches)):
        seq.extend(["piezo", "substrate"])
    return seq


def wrap_phase_rad(phase_rad: np.ndarray) -> np.ndarray:
    """Wrap phases to [0, 2*pi)."""
    return np.mod(np.asarray(phase_rad, dtype=float), 2 * np.pi)


def relative_phase_rad(phase_rad: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """Return phases relative to one patch, wrapped to [0, 2*pi)."""
    phase_rad = np.asarray(phase_rad, dtype=float)
    return wrap_phase_rad(phase_rad - phase_rad[reference_index])


# -----------------------------------------------------------------------------
# Output and traveling-wave metrics
# -----------------------------------------------------------------------------

try:  # package imports
    from Modeling.models_fish.piezo_opt.metrics import (
        canonical_output_name,
        compact_traveling_wave_metrics,
        default_traveling_wave_settings,
        evaluate_output_metric,
        metric_label,
        phase_slope_from_complex_shape,
        reduced_to_full_displacement_nodes,
        response_summary,
        tip_reduced_index,
        trapezoid_node_weights,
        transverse_reduced_indices,
        traveling_index_from_complex_shape,
        traveling_wave_frequency_from_settings,
        traveling_wave_metrics,
        traveling_wave_node_window,
    )
except Exception:  # local / notebook fallback
    from piezo_opt.metrics import (
        canonical_output_name,
        compact_traveling_wave_metrics,
        default_traveling_wave_settings,
        evaluate_output_metric,
        metric_label,
        phase_slope_from_complex_shape,
        reduced_to_full_displacement_nodes,
        response_summary,
        tip_reduced_index,
        trapezoid_node_weights,
        transverse_reduced_indices,
        traveling_index_from_complex_shape,
        traveling_wave_frequency_from_settings,
        traveling_wave_metrics,
        traveling_wave_node_window,
    )


# -----------------------------------------------------------------------------
# Inner phase optimizers
# -----------------------------------------------------------------------------

try:  # package imports
    from Modeling.models_fish.piezo_opt.actuation import (
        optimize_binary_phases_general,
        optimize_binary_phases_traveling_wave,
        optimize_continuous_phases_metric,
        optimize_continuous_phases_tip,
        optimize_continuous_phases_traveling_wave,
    )
except Exception:  # local / notebook fallback
    from piezo_opt.actuation import (
        optimize_binary_phases_general,
        optimize_binary_phases_traveling_wave,
        optimize_continuous_phases_metric,
        optimize_continuous_phases_tip,
        optimize_continuous_phases_traveling_wave,
    )


try:  # package imports
    from Modeling.models_fish.piezo_opt.objectives import (
        ModeResponseObjective,
        MultiModeObjective,
        TravelingWaveObjective,
        reduce_multimode_scores,
    )
except Exception:  # local / notebook fallback
    from piezo_opt.objectives import (
        ModeResponseObjective,
        MultiModeObjective,
        TravelingWaveObjective,
        reduce_multimode_scores,
    )

try:  # package imports
    from Modeling.models_fish.piezo_opt.outer import GenericOuterOptimizer
except Exception:  # local / notebook fallback
    from piezo_opt.outer import GenericOuterOptimizer


# -----------------------------------------------------------------------------
# Main optimizer
# -----------------------------------------------------------------------------

class PiezoPatchOptimizer(GenericOuterOptimizer):
    """Outer geometry optimizer with an inner phase optimization layer."""

    def __init__(
        self,
        L: float,
        region_types: dict,
        base_params: PiezoBeamParams,
        geometry_settings: GeometrySettings,
        objective_settings: Optional[ObjectiveSettings] = None,
        circuit_settings: Optional[CircuitSettings] = None,
        optimizer_settings: Optional[OptimizerSettings] = None,
        *,
        mode_settings: Optional[ObjectiveSettings] = None,
        fe_module=FE_module,
        default_h: float = 1e-3,
    ):
        # ``mode_settings`` is retained for backward compatibility with older notebooks.
        if objective_settings is None:
            objective_settings = mode_settings
        if objective_settings is None:
            objective_settings = ObjectiveSettings()

        self.L = float(L)
        self.region_types = region_types
        self.base_params = base_params
        self.geometry_settings = geometry_settings
        self.objective_settings = objective_settings
        self.circuit_settings = circuit_settings or CircuitSettings()
        self.optimizer_settings = optimizer_settings or OptimizerSettings()
        self.fe_module = fe_module
        self.default_h = default_h
        self.evaluation_history: list[dict] = []

    @property
    def mode_settings(self) -> ObjectiveSettings:
        """Backward-compatible alias for earlier code."""
        return self.objective_settings

    @property
    def n_patches(self) -> int:
        return int(self.geometry_settings.Np)

    @property
    def n_design_variables(self) -> int:
        return 2 * self.n_patches - 1

    def make_region_sequence(self) -> List[str]:
        return make_region_sequence(self.n_patches)

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

    def build_fe_for_design(self, z: np.ndarray):
        """Return (fe, layout, penalty) for a candidate design."""
        layout = self.decode_design(z)
        penalty = self.geometry_penalty(layout)
        if penalty > 0:
            return None, layout, penalty

        try:
            geom = self.fe_module.build_geometry_from_types(
                L=self.L,
                region_types=self.region_types,
                region_sequence=layout["region_sequence"],
                x_starts=layout["x_starts"],
                default_h=self.default_h,
            )
            params = copy.copy(self.base_params)
            params.geometry = geom
            if hasattr(params, "sync_patch_count"):
                params.sync_patch_count(len(geom.piezos))
            else:  # backward compatibility with older parameter class
                params.n_patches = len(geom.piezos)
                params.S = len(geom.piezos)
                params.Q = len(geom.piezos)
                params.Cp = params.Cp_scalar * np.ones(params.S)
            fe = self.fe_module.PiezoBeamFE(params)
            return fe, layout, 0.0
        except Exception as exc:
            if getattr(self.optimizer_settings, "raise_exceptions", False):
                raise
            layout["error"] = repr(exc)
            return None, layout, self.geometry_settings.invalid_penalty

    def effective_damping_matrix(self, fe) -> np.ndarray:
        if hasattr(fe, "effective_damping_matrix"):
            return fe.effective_damping_matrix()
        return fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red

    def response_columns(self, fe, omega: float) -> np.ndarray:
        """Return reduced displacement columns per unit patch voltage at omega."""
        D = self.effective_damping_matrix(fe)
        Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
        return np.linalg.solve(Z, fe.Gamma_red)

    def unit_patch_response(self, fe, omega: float, output: Optional[str] = None) -> np.ndarray:
        """Backward-compatible scalar unit-patch response vector for tip output."""
        output = canonical_output_name(output or self.objective_settings.output)
        U_cols = self.response_columns(fe, omega)
        if output != "tip":
            raise ValueError("unit_patch_response is only meaningful for scalar tip output; use response_columns instead")
        return U_cols[tip_reduced_index(fe), :]

    def _evaluate_one_mode(self, fe, mode_number: int) -> dict:
        """Evaluate one mode and run the selected phase optimizer."""
        return ModeResponseObjective(self).evaluate_mode(fe, mode_number)

    def _evaluate_single_mode_objective(self, fe) -> dict:
        return ModeResponseObjective(self).evaluate(fe)

    def _evaluate_multi_mode_objective(self, fe) -> dict:
        return MultiModeObjective(self).evaluate(fe)

    def _evaluate_traveling_wave_objective(self, fe) -> dict:
        return TravelingWaveObjective(self).evaluate(fe)

    def _objective_for_settings(self):
        objective = canonical_objective_name(self.objective_settings.objective)
        if objective == "single_mode":
            return ModeResponseObjective(self)
        if objective == "multi_mode":
            return MultiModeObjective(self)
        if objective == "traveling_wave":
            return TravelingWaveObjective(self)
        raise RuntimeError(f"Unhandled objective {objective}")

    def inner_optimizer(self, fe) -> dict:
        """Evaluate the configured objective for a built FE model."""
        return self._objective_for_settings().evaluate(fe)

    def evaluate_at_natural_frequency(self, fe) -> dict:
        """Backward-compatible alias for the inner objective evaluation."""
        return self.inner_optimizer(fe)

    def single_mode_calibration_optimizer(
        self,
        mode_number: int,
        *,
        optimizer_settings: Optional[OptimizerSettings] = None,
    ) -> "PiezoPatchOptimizer":
        """Return a single-mode optimizer using this optimizer as a template.

        The calibration optimizer keeps the same geometry, circuit, phase mode,
        output metric, and voltage settings, but switches the objective to one
        selected mode. Its best score is a natural normalizer for that mode in a
        subsequent multi-mode run.
        """
        settings_dict = copy.deepcopy(self.objective_settings.__dict__)
        settings_dict.update(
            {
                "objective": "single_mode",
                "target_mode_number": None,
                "single_mode_number": int(mode_number),
                "multi_mode_weights": None,
                "multi_mode_score_normalizers": None,
            }
        )
        objective_settings = ObjectiveSettings(**settings_dict)

        return PiezoPatchOptimizer(
            L=self.L,
            region_types=copy.deepcopy(self.region_types),
            base_params=copy.copy(self.base_params),
            geometry_settings=copy.deepcopy(self.geometry_settings),
            objective_settings=objective_settings,
            circuit_settings=copy.deepcopy(self.circuit_settings),
            optimizer_settings=copy.copy(optimizer_settings or self.optimizer_settings),
            fe_module=self.fe_module,
            default_h=self.default_h,
        )

    def calibrate_multimode_score_normalizers(
        self,
        modes: Optional[Sequence[int]] = None,
        *,
        optimizer_settings: Optional[OptimizerSettings] = None,
        normalizer_floor: Optional[float] = None,
        apply: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Estimate automatic multi-mode normalizers from single-mode optima.

        Each requested mode is optimized independently using the same design
        bounds and phase/output settings. The resulting best scores can be used
        as ``multi_mode_score_normalizers`` so the multi-mode objective compares
        each mode by its fraction of its own best achievable response.

        Parameters
        ----------
        modes:
            1-based target mode numbers. Defaults to ``multi_mode_numbers`` for
            a multi-mode objective, otherwise the configured single mode.
        optimizer_settings:
            Optional settings for the calibration runs. If omitted, the current
            optimizer settings are reused.
        normalizer_floor:
            Optional lower bound applied to every normalizer. Useful only if a
            mode has a nearly zero best score.
        apply:
            If True, write the computed tuple directly to
            ``self.objective_settings.multi_mode_score_normalizers``.
        verbose:
            Print a compact progress summary.
        """
        ms = self.objective_settings
        if modes is None:
            if canonical_objective_name(ms.objective) == "multi_mode":
                modes = ms.multi_mode_numbers
            else:
                modes = (ms.single_mode_number,)

        modes = tuple(int(m) for m in modes)
        if len(modes) == 0:
            raise ValueError("modes must contain at least one mode number")

        floor = None if normalizer_floor is None else float(normalizer_floor)
        if floor is not None and floor <= 0:
            raise ValueError("normalizer_floor must be positive when provided")

        records = []
        scores = []
        normalizers = []

        for mode_number in modes:
            if verbose:
                print(f"Calibrating mode {mode_number} normalizer...")

            optimizer = self.single_mode_calibration_optimizer(
                mode_number,
                optimizer_settings=optimizer_settings,
            )
            result = optimizer.run()
            best = optimizer.inspect_result(result)
            score = float(best["inner"]["score"])

            if not np.isfinite(score) or score <= 0:
                raise RuntimeError(
                    f"Single-mode calibration for mode {mode_number} returned invalid score {score!r}"
                )

            normalizer = max(score, floor) if floor is not None else score
            scores.append(score)
            normalizers.append(normalizer)
            records.append(
                {
                    "mode_number": mode_number,
                    "score": score,
                    "normalizer": normalizer,
                    "optimizer": optimizer,
                    "result": result,
                    "best": best,
                }
            )

            if verbose:
                print(f"  best score = {score:.6e}, normalizer = {normalizer:.6e}")

        normalizers_tuple = tuple(float(v) for v in normalizers)
        if apply:
            self.objective_settings.multi_mode_score_normalizers = normalizers_tuple

        return {
            "modes": modes,
            "scores": np.asarray(scores, dtype=float),
            "normalizers": np.asarray(normalizers, dtype=float),
            "records": records,
            "applied": bool(apply),
        }

    def get_mode_result(self, inner: dict, mode_number: Optional[int] = None, mode_index: int = 0) -> dict:
        """Extract one mode result from a single- or multi-mode inner result."""
        mode_results = inner.get("mode_results", [inner])
        if mode_number is not None:
            for r in mode_results:
                if int(r["mode_number"]) == int(mode_number):
                    return r
            raise KeyError(f"mode_number={mode_number} not found in mode_results")
        return mode_results[int(mode_index)]

    def dense_metric_frf_for_plot(
        self,
        fe,
        voltage_vector,
        *,
        output: Optional[str] = None,
        sweep_range_hz=None,
        n_freq=None,
    ) -> dict:
        """Dense mechanical FRF for one voltage pattern and selected output metric."""
        ms = self.objective_settings
        output = canonical_output_name(output or ms.output)
        sweep_range_hz = sweep_range_hz or ms.final_sweep_range_hz
        n_freq = int(n_freq or ms.final_sweep_n_freq)
        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        D = self.effective_damping_matrix(fe)

        metric = np.zeros(n_freq, dtype=float)
        tip_disp = np.zeros(n_freq, dtype=float)
        mean_abs = np.zeros(n_freq, dtype=float)
        rms = np.zeros(n_freq, dtype=float)
        response_complex_tip = np.zeros(n_freq, dtype=complex)

        for k, omega in enumerate(omega_vec):
            Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
            u_red = np.linalg.solve(Z, fe.Gamma_red @ voltage_vector)
            metric[k] = evaluate_output_metric(fe, u_red, output)
            tip_disp[k] = evaluate_output_metric(fe, u_red, "tip")
            mean_abs[k] = evaluate_output_metric(fe, u_red, "mean_abs")
            rms[k] = evaluate_output_metric(fe, u_red, "rms")
            response_complex_tip[k] = u_red[tip_reduced_index(fe)]

        return {
            "freq": freq,
            "omega": omega_vec,
            "output": output,
            "metric_label": metric_label(output),
            "metric": metric,
            "tip_disp": tip_disp,
            "mean_abs": mean_abs,
            "rms": rms,
            "response_complex": response_complex_tip,
            "voltage_vector": voltage_vector,
        }

    def dense_metric_frf_for_mode_result(self, fe, mode_result: dict, **kwargs) -> dict:
        """Dense FRF using the voltage vector stored in a single mode_result."""
        return self.dense_metric_frf_for_plot(fe, mode_result["voltage_vector"], output=mode_result.get("output"), **kwargs)

    def dense_traveling_wave_metrics_for_plot(
        self,
        fe,
        voltage_vector,
        *,
        sweep_range_hz=None,
        n_freq=None,
        traveling_wave_settings: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Dense frequency sweep of traveling-wave metrics for one voltage pattern."""
        ms = self.objective_settings
        settings = default_traveling_wave_settings(
            traveling_wave_settings or ms.traveling_wave_settings
        )
        sweep_range_hz = sweep_range_hz or ms.final_sweep_range_hz
        n_freq = int(n_freq or ms.final_sweep_n_freq)
        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        D = self.effective_damping_matrix(fe)

        score = np.zeros(n_freq, dtype=float)
        traveling_index = np.zeros(n_freq, dtype=float)
        amplitude_rms = np.zeros(n_freq, dtype=float)
        amplitude_score = np.zeros(n_freq, dtype=float)
        envelope_cv = np.zeros(n_freq, dtype=float)
        envelope_score = np.zeros(n_freq, dtype=float)
        phase_slope = np.zeros(n_freq, dtype=float)
        direction_score = np.zeros(n_freq, dtype=float)

        for k, omega in enumerate(omega_vec):
            Z = fe.K_red + 1j * omega * D - omega**2 * fe.M_red
            u_red = np.linalg.solve(Z, fe.Gamma_red @ voltage_vector)
            metrics = traveling_wave_metrics(fe, u_red, settings)
            score[k] = metrics["score"]
            traveling_index[k] = metrics["traveling_index"]
            amplitude_rms[k] = metrics["amplitude_rms"]
            amplitude_score[k] = metrics["amplitude_score"]
            envelope_cv[k] = metrics["envelope_cv"]
            envelope_score[k] = metrics["envelope_score"]
            phase_slope[k] = metrics["phase_slope_rad_per_m"]
            direction_score[k] = metrics["direction_score"]

        return {
            "freq": freq,
            "omega": omega_vec,
            "score": score,
            "traveling_index": traveling_index,
            "amplitude_rms": amplitude_rms,
            "amplitude_score": amplitude_score,
            "envelope_cv": envelope_cv,
            "envelope_score": envelope_score,
            "phase_slope_rad_per_m": phase_slope,
            "direction_score": direction_score,
            "voltage_vector": voltage_vector,
            "traveling_wave_settings": settings,
            "metric_label": "Traveling-wave objective score [-]",
        }

    def dense_tip_frf_for_plot(self, fe, voltage_vector, *, sweep_range_hz=None, n_freq=None) -> dict:
        """Backward-compatible dense tip FRF for one voltage pattern."""
        return self.dense_metric_frf_for_plot(
            fe,
            voltage_vector,
            output="tip",
            sweep_range_hz=sweep_range_hz,
            n_freq=n_freq,
        )

    def dense_all_binary_metric_frf_for_plot(self, fe, *, output: Optional[str] = None, sweep_range_hz=None, n_freq=None) -> List[dict]:
        """Dense FRFs for all binary sign patterns using the selected output metric."""
        n = fe.Gamma_red.shape[1]
        out = []
        for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
            signs = np.asarray(signs_tuple, dtype=float)
            frf = self.dense_metric_frf_for_plot(
                fe,
                self.objective_settings.voltage_amplitude * signs,
                output=output or self.objective_settings.output,
                sweep_range_hz=sweep_range_hz,
                n_freq=n_freq,
            )
            frf["signs"] = signs
            frf["label"] = sign_label(signs)
            out.append(frf)
        return out

    def dense_all_binary_frf_for_plot(self, fe, *, sweep_range_hz=None, n_freq=None) -> List[dict]:
        """Backward-compatible dense tip FRFs for all binary sign patterns."""
        return self.dense_all_binary_metric_frf_for_plot(fe, output="tip", sweep_range_hz=sweep_range_hz, n_freq=n_freq)


def build_region_types_from_params(params: PiezoBeamParams, *, h_patch: float = 1e-3, h_gap: float = 1e-3) -> dict:
    """Convenience region_types dictionary for simple piezo/substrate layouts."""
    rhoA_patch = params.b * (params.rho_s * params.hs + 2.0 * params.rho_p * params.hp)
    rhoA_gap = params.b * params.rho_s * params.hs
    return {
        "piezo": {"EI": params.YI, "rhoA": rhoA_patch, "h": h_patch},
        "substrate": {"EI": params.YI_s, "rhoA": rhoA_gap, "h": h_gap},
    }
