# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .optimizer_helpers import (
    OBJECTIVE_ADMITTANCE_MATCH,
    OBJECTIVE_MULTI_MODE,
    OBJECTIVE_MULTIFUNCTIONAL,
    OBJECTIVE_SINGLE_MODE,
    OBJECTIVE_THRUST,
    OBJECTIVE_THRUST_PER_POWER,
    OBJECTIVE_TRAVELING_WAVE,
    OBJECTIVE_TYPES,
    OUTPUT_TYPES,
)
from ..model.model import ThrustSettings, WaterSettings

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Settings Defaults
def default_standing_wave_settings() -> Dict[str, Any]:
    return {
        "single_mode_number": 1,
        "multi_mode_numbers": (1, 2, 3),
        "multi_mode_weights": None,
        "multi_mode_reduction": "weighted_sum",
        "multi_mode_phase_policy": "per_mode",
    }


def default_traveling_wave_objective_settings() -> Dict[str, Any]:
    return {
        "frequency_bounds_hz": None,
        "mode_pair": (1, 2),
        "frequency_xtol": 1e-3,
        "x_fraction_bounds": (0.05, 1.0),
        "eps": 1e-300,
    }


def default_thrust_objective_settings() -> Dict[str, Any]:
    return {
        "mode_number": 1,
        "frequency_hz": None,
        "rho": 1000.0,
        "width": None,
        "beta": 1.0,
        "swimming_speed": 0.0,
        "clip_negative_thrust": True,
        "power_denominator": "abs_real",  # abs_real, real_positive, apparent
        "power_eps": 1e-18,
    }


def default_admittance_match_settings() -> Dict[str, Any]:
    return {
        "frequencies_hz": None,
        "target_admittance": None,
        "target_magnitude": None,
        "target_real_power": None,
        "voltage_vector": None,
        "weights": None,
        "relative_error": True,
        "eps": 1e-30,
    }


def default_multifunctional_settings() -> Dict[str, Any]:
    return {
        "mode_number": 1,
        "frequency_hz": None,
        "components": ("thrust", "traveling_wave", "response"),
        "weights": {"thrust": 1.0, "traveling_wave": 1.0, "response": 1.0},
        "normalizers": {"thrust": 1.0, "traveling_wave": 1.0, "response": 1.0},
        "response_output": "rms",
    }


def default_random_search_settings() -> Dict[str, Any]:
    return {"n_random_samples": 300}


def default_powell_settings() -> Dict[str, Any]:
    return {"maxiter": 80, "xtol": 1e-4, "ftol": 1e-4}


def default_differential_evolution_settings() -> Dict[str, Any]:
    return {"maxiter": 15, "popsize": 8, "polish": False, "workers": 1}


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
        Must be one of the objective constants from ``optimizer_helpers.py``.
    standing_wave_settings:
        Dictionary for ``single_mode`` and ``multi_mode`` fields:
        ``single_mode_number``, ``multi_mode_numbers``, ``multi_mode_weights``,
        ``multi_mode_reduction``, and ``multi_mode_phase_policy``.
    traveling_wave_settings:
        Dictionary for traveling-wave frequency/window/score fields. Defaults
        are populated explicitly during ``__post_init__``.
    """

    objective: str = "single_mode"
    standing_wave_settings: Dict[str, Any] = field(default_factory=default_standing_wave_settings)
    traveling_wave_settings: Dict[str, Any] = field(default_factory=default_traveling_wave_objective_settings)
    thrust_settings: Dict[str, Any] = field(default_factory=default_thrust_objective_settings)
    admittance_match_settings: Dict[str, Any] = field(default_factory=default_admittance_match_settings)
    multifunctional_settings: Dict[str, Any] = field(default_factory=default_multifunctional_settings)

    voltage_amplitude: float = 1.0
    phase_mode: str = "binary"  # "binary" or "continuous"
    output: str = "tip"

    # Dense final sweep settings for post-processing.
    final_sweep_range_hz: Tuple[float, float] = (0.1, 10.0)
    final_sweep_n_freq: int = 1000

    # Only used when phase_mode="continuous" and output is not scalar tip.
    # For tip output, continuous phase has an analytic solution.
    continuous_phase_n_starts: int = 8
    continuous_phase_seed: Optional[int] = 1
    continuous_phase_method: str = "L-BFGS-B"

    inner_workers: int = 1
    inner_parallel_min_tasks: int = 4

    def __post_init__(self):
        if self.objective not in OBJECTIVE_TYPES:
            raise ValueError(f"objective must be one of {OBJECTIVE_TYPES}")
        if self.phase_mode not in ("binary", "continuous"):
            raise ValueError("phase_mode must be exactly 'binary' or 'continuous'")
        if self.objective in (OBJECTIVE_SINGLE_MODE, OBJECTIVE_MULTI_MODE) and self.output not in OUTPUT_TYPES:
            raise ValueError("output must be exactly 'tip', 'mean_abs', or 'rms'")

        standing_defaults = default_standing_wave_settings()
        unknown = set(self.standing_wave_settings) - set(standing_defaults)
        if unknown:
            raise ValueError(f"Unknown standing_wave_settings keys: {sorted(unknown)}")
        standing = {**standing_defaults, **dict(self.standing_wave_settings)}
        standing["single_mode_number"] = int(standing["single_mode_number"])
        standing["multi_mode_numbers"] = tuple(int(m) for m in standing["multi_mode_numbers"])

        if standing["multi_mode_weights"] is not None:
            standing["multi_mode_weights"] = tuple(float(w) for w in standing["multi_mode_weights"])
            if len(standing["multi_mode_weights"]) != len(standing["multi_mode_numbers"]):
                raise ValueError("multi_mode_weights must have the same length as multi_mode_numbers")

        if standing["multi_mode_phase_policy"] != "per_mode":
            raise NotImplementedError("Only multi_mode_phase_policy='per_mode' is implemented for now")
        self.standing_wave_settings = standing

        traveling_defaults = default_traveling_wave_objective_settings()
        unknown = set(self.traveling_wave_settings) - set(traveling_defaults)
        if unknown:
            raise ValueError(f"Unknown traveling_wave_settings keys: {sorted(unknown)}")
        self.traveling_wave_settings = {**traveling_defaults, **dict(self.traveling_wave_settings)}

        thrust_defaults = default_thrust_objective_settings()
        unknown = set(self.thrust_settings) - set(thrust_defaults)
        if unknown:
            raise ValueError(f"Unknown thrust_settings keys: {sorted(unknown)}")
        thrust = {**thrust_defaults, **dict(self.thrust_settings)}
        thrust["mode_number"] = int(thrust["mode_number"])
        if thrust["frequency_hz"] is not None:
            thrust["frequency_hz"] = float(thrust["frequency_hz"])
        for key in ("rho", "beta", "swimming_speed", "power_eps"):
            thrust[key] = float(thrust[key])
        if thrust["width"] is not None:
            thrust["width"] = float(thrust["width"])
        thrust["clip_negative_thrust"] = bool(thrust["clip_negative_thrust"])
        if thrust["power_denominator"] not in ("abs_real", "real_positive", "apparent"):
            raise ValueError("thrust_settings['power_denominator'] must be abs_real, real_positive, or apparent")
        self.thrust_settings = thrust

        admittance_defaults = default_admittance_match_settings()
        unknown = set(self.admittance_match_settings) - set(admittance_defaults)
        if unknown:
            raise ValueError(f"Unknown admittance_match_settings keys: {sorted(unknown)}")
        admittance = {**admittance_defaults, **dict(self.admittance_match_settings)}
        admittance["relative_error"] = bool(admittance["relative_error"])
        admittance["eps"] = float(admittance["eps"])
        self.admittance_match_settings = admittance

        multifunctional_defaults = default_multifunctional_settings()
        unknown = set(self.multifunctional_settings) - set(multifunctional_defaults)
        if unknown:
            raise ValueError(f"Unknown multifunctional_settings keys: {sorted(unknown)}")
        multifunctional = {**multifunctional_defaults, **dict(self.multifunctional_settings)}
        multifunctional["mode_number"] = int(multifunctional["mode_number"])
        if multifunctional["frequency_hz"] is not None:
            multifunctional["frequency_hz"] = float(multifunctional["frequency_hz"])
        multifunctional["components"] = tuple(str(c) for c in multifunctional["components"])
        multifunctional["weights"] = {str(k): float(v) for k, v in dict(multifunctional["weights"]).items()}
        multifunctional["normalizers"] = {str(k): float(v) for k, v in dict(multifunctional["normalizers"]).items()}
        if multifunctional["response_output"] not in OUTPUT_TYPES:
            raise ValueError("multifunctional_settings['response_output'] must be tip, mean_abs, or rms")
        self.multifunctional_settings = multifunctional


@dataclass
class OptimizerSettings:
    method: str = "differential_evolution"  # differential_evolution, random, powell, random_powell
    seed: Optional[int] = 1
    show_progress: bool = True
    raise_exceptions: bool = False
    raise_optimizer_failures: bool = False
    random_search_settings: Dict[str, Any] = field(default_factory=default_random_search_settings)
    powell_settings: Dict[str, Any] = field(default_factory=default_powell_settings)
    differential_evolution_settings: Dict[str, Any] = field(default_factory=default_differential_evolution_settings)

    def __post_init__(self):
        random_defaults = default_random_search_settings()
        powell_defaults = default_powell_settings()
        de_defaults = default_differential_evolution_settings()

        unknown = set(self.random_search_settings) - set(random_defaults)
        if unknown:
            raise ValueError(f"Unknown random_search_settings keys: {sorted(unknown)}")
        unknown = set(self.powell_settings) - set(powell_defaults)
        if unknown:
            raise ValueError(f"Unknown powell_settings keys: {sorted(unknown)}")
        unknown = set(self.differential_evolution_settings) - set(de_defaults)
        if unknown:
            raise ValueError(f"Unknown differential_evolution_settings keys: {sorted(unknown)}")

        self.random_search_settings = {**random_defaults, **dict(self.random_search_settings)}
        self.powell_settings = {**powell_defaults, **dict(self.powell_settings)}
        self.differential_evolution_settings = {**de_defaults, **dict(self.differential_evolution_settings)}


@dataclass
class PostProcessingSettings:
    postprocess_workers: int = 1
    postprocess_parallel_min_tasks: int = 4
    water_settings: WaterSettings = field(default_factory=WaterSettings)
    thrust_settings: ThrustSettings = field(default_factory=ThrustSettings)

    def __post_init__(self):
        self.postprocess_workers = int(self.postprocess_workers)
        self.postprocess_parallel_min_tasks = int(self.postprocess_parallel_min_tasks)

        if isinstance(self.water_settings, dict):
            self.water_settings = WaterSettings(**self.water_settings)
        if isinstance(self.thrust_settings, dict):
            self.thrust_settings = ThrustSettings(**self.thrust_settings)

        if self.postprocess_workers < 1:
            raise ValueError("postprocess_workers must be >= 1")
        if self.postprocess_parallel_min_tasks < 1:
            raise ValueError("postprocess_parallel_min_tasks must be >= 1")

# -----------------------------------------------------------------------------
