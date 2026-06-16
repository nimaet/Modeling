"""Optional post-processing utilities for piezo patch optimization results."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..model.model_helpers import (
    admittance_matrix,
    complex_power_from_peak_phasors,
    linear_added_mass_matrix,
    linear_hydrodynamic_damping_matrix,
    solve_harmonic_response,
    solve_morison_harmonic_response,
    tip_reduced_index,
    lighthill_thrust,
)
from ..optimizer.optimizer_helpers import (
    OUTPUT_TYPES,
    evaluate_output_metric,
    metric_label,
    parallel_map,
)
from ..optimizer.optimizer_settings import PostProcessingSettings
from ..optimizer.traveling_wave_optimizer import traveling_wave_metrics


# -----------------------------------------------------------------------------
# Result Selection and JSON Helpers
# -----------------------------------------------------------------------------

def get_mode_result(inner: dict, mode_number: Optional[int] = None, mode_index: int = 0) -> dict:
    """Extract one mode result from a single- or multi-mode inner result."""
    mode_results = inner.get("mode_results", [inner])
    if mode_number is not None:
        for result in mode_results:
            if int(result["mode_number"]) == int(mode_number):
                return result
        raise KeyError(f"mode_number={mode_number} not found in mode_results")
    return mode_results[int(mode_index)]


def _read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_jsonable(value):
    if isinstance(value, dict):
        if set(value) == {"real", "imag"}:
            real = np.asarray(value["real"])
            imag = np.asarray(value["imag"])
            out = real + 1j * imag
            return out.item() if out.shape == () else out
        return {k: _decode_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        decoded = [_decode_jsonable(v) for v in value]
        if decoded and all(np.isscalar(v) for v in decoded):
            try:
                return np.asarray(decoded)
            except Exception:
                return decoded
        return decoded
    return value


def _infer_L_from_layout(layout: dict) -> float:
    x_starts = np.asarray(layout["x_starts"], dtype=float)
    return float(x_starts[-1] + float(layout.get("tip_substrate", 0.0)))


# -----------------------------------------------------------------------------
# Post-Processing Class
# -----------------------------------------------------------------------------

class PostProcessor:
    """Reusable post-processor for one optimizer/FE result."""

    def __init__(
        self,
        optimizer: Any = None,
        fe: Any = None,
        settings: Optional[PostProcessingSettings] = None,
    ):
        self.optimizer = optimizer
        self.fe = fe
        self.settings = settings or PostProcessingSettings()

    def _workers(self, workers: Optional[int]) -> int:
        return self.settings.postprocess_workers if workers is None else int(workers)

    def _min_tasks(self, min_tasks: Optional[int]) -> int:
        return self.settings.postprocess_parallel_min_tasks if min_tasks is None else int(min_tasks)

    def _water_settings(self):
        return self.settings.water_settings

    def _water_enabled(self) -> bool:
        water = self._water_settings()
        return bool(water.enabled and water.model != "none")

    def _frequency_response_label(self) -> str:
        if not self._water_enabled():
            return "Dry linear frequency response"
        if self._water_settings().model == "linear":
            return "Linear-water frequency response"
        return "Morison nonlinear frequency response estimate"

    def _frequency_response_kind(self) -> str:
        if not self._water_enabled():
            return "linear_frequency_response"
        if self._water_settings().model == "linear":
            return "linear_water_frequency_response"
        return "nonlinear_frequency_response_estimate"

    def _require_water_width(self) -> float:
        water = self._water_settings()
        if water.width is None:
            raise ValueError("water_settings.width must be set when water is enabled")
        return float(water.width)

    def _linear_water_matrices(self):
        water = self._water_settings()
        if not self._water_enabled():
            return None, None

        width = self._require_water_width()
        extra_mass = linear_added_mass_matrix(
            self.fe,
            rho=water.rho,
            width=width,
            added_mass_coefficient=water.added_mass_coefficient,
        )
        extra_damping = linear_hydrodynamic_damping_matrix(
            self.fe,
            damping_per_length=water.linear_damping_coefficient,
        )
        return extra_mass, extra_damping

    def _solve_frequency_response(self, omega, voltage_vector):
        water = self._water_settings()

        if not self._water_enabled():
            return solve_harmonic_response(self.fe, omega, voltage_vector), {}

        if water.model == "linear":
            extra_mass, extra_damping = self._linear_water_matrices()
            return solve_harmonic_response(
                self.fe,
                omega,
                voltage_vector,
                extra_mass=extra_mass,
                extra_damping=extra_damping,
            ), {"water_model": "linear"}

        if water.model == "morison":
            out = solve_morison_harmonic_response(
                self.fe,
                omega,
                voltage_vector,
                rho=water.rho,
                width=self._require_water_width(),
                added_mass_coefficient=water.added_mass_coefficient,
                drag_coefficient=water.drag_coefficient,
                linear_damping_per_length=water.linear_damping_coefficient,
                max_iter=water.morison_max_iter,
                tol=water.morison_tol,
                relaxation=water.morison_relaxation,
            )
            metadata = {
                "water_model": "morison",
                "morison_converged": out["converged"],
                "morison_iterations": out["iterations"],
                "morison_relative_error": out["relative_error"],
            }
            return out["response_red"], metadata

        raise ValueError(f"Unsupported water model: {water.model}")

    def _current_from_response(self, omega, voltage_vector, response_red):
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        return 1j * omega * (
            self.fe.Cp @ voltage_vector + self.fe.Gamma_red.T @ response_red
        )

    def _thrust_settings(self):
        return self.settings.thrust_settings

    def _thrust_enabled(self) -> bool:
        return bool(self._thrust_settings().enabled)

    def _require_thrust_width(self) -> float:
        thrust = self._thrust_settings()
        if thrust.width is None:
            raise ValueError("thrust_settings.width must be set when thrust is enabled")
        return float(thrust.width)

    def _evaluate_thrust(self, omega, response_red):
        thrust = self._thrust_settings()
        if not thrust.enabled:
            return None
        return lighthill_thrust(
            self.fe,
            response_red,
            omega,
            rho=thrust.rho,
            width=self._require_thrust_width(),
            beta=thrust.beta,
            swimming_speed=thrust.swimming_speed,
        )

    def dense_metric_frequency_response_for_plot(
        self,
        voltage_vector,
        *,
        output: Optional[str] = None,
        sweep_range_hz=None,
        n_freq=None,
        postprocess_workers: Optional[int] = None,
        postprocess_parallel_min_tasks: Optional[int] = None,
    ) -> dict:
        """Dense mechanical frequency response for one voltage pattern."""
        if self.optimizer is None or self.fe is None:
            raise ValueError("PostProcessor requires optimizer and fe for dense frequency responses")

        opt_settings = self.optimizer.objective_settings
        output = output or opt_settings.output
        if output not in OUTPUT_TYPES:
            raise ValueError("output must be exactly 'tip', 'mean_abs', or 'rms'")
        sweep_range_hz = sweep_range_hz or opt_settings.final_sweep_range_hz
        n_freq = int(n_freq or opt_settings.final_sweep_n_freq)

        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)
        include_tip_response = output == "tip"

        def evaluate_response(omega):
            # u_red = solve_harmonic_response(self.fe, omega, voltage_vector)
            # metric = evaluate_output_metric(self.fe, u_red, output)
            u_red, metadata = self._solve_frequency_response(omega, voltage_vector)
            metric = evaluate_output_metric(self.fe, u_red, output)
            thrust = self._evaluate_thrust(omega, u_red)

            tip_response = u_red[tip_reduced_index(self.fe)] if include_tip_response else None
            return metric, tip_response, metadata, thrust

        results = parallel_map(
            evaluate_response,
            omega_vec,
            workers=self._workers(postprocess_workers),
            min_tasks=self._min_tasks(postprocess_parallel_min_tasks),
        )
        metric = np.asarray([result[0] for result in results], dtype=float)
        metadata_records = [result[2] for result in results]
        thrust_values = [result[3] for result in results]
        thrust_array = (
            np.asarray(thrust_values, dtype=float)
            if self._thrust_enabled()
            else None
        )
        response_complex_tip = (
            np.asarray([result[1] for result in results], dtype=complex)
            if include_tip_response
            else None
        )

        return {
            "freq": freq,
            "omega": omega_vec,
            "output": output,
            "metric_label": metric_label(output),
            "response_label": self._frequency_response_label(),
            "response_kind": self._frequency_response_kind(),
            "metric": metric,
            "tip_disp": metric if output == "tip" else None,
            "mean_abs": metric if output == "mean_abs" else None,
            "rms": metric if output == "rms" else None,
            "response_complex": response_complex_tip,
            "voltage_vector": voltage_vector,
            "water_model": self._water_settings().model if self._water_enabled() else "none",
            "water_enabled": self._water_enabled(),
            "metadata": metadata_records,
            "thrust_enabled": self._thrust_enabled(),
            "thrust": thrust_array,
            "thrust_label": "Mean thrust estimate [N]",
        }

    def dense_metric_frequency_response_for_mode_result(self, mode_result: dict, **kwargs) -> dict:
        """Dense frequency response using a mode result's voltage vector."""
        return self.dense_metric_frequency_response_for_plot(
            mode_result["voltage_vector"],
            output=mode_result.get("output"),
            **kwargs,
        )

    def dense_all_binary_metric_frequency_responses_for_plot(
        self,
        *,
        output: Optional[str] = None,
        sweep_range_hz=None,
        n_freq=None,
        postprocess_workers: Optional[int] = None,
        postprocess_parallel_min_tasks: Optional[int] = None,
    ) -> list[dict]:
        """Dense frequency responses for all binary sign patterns."""
        if self.optimizer is None or self.fe is None:
            raise ValueError("PostProcessor requires optimizer and fe for binary frequency responses")

        out = []
        for signs_tuple in itertools.product([-1.0, 1.0], repeat=self.fe.Gamma_red.shape[1]):
            signs = np.asarray(signs_tuple, dtype=float)
            response = self.dense_metric_frequency_response_for_plot(
                self.optimizer.objective_settings.voltage_amplitude * signs,
                output=output or self.optimizer.objective_settings.output,
                sweep_range_hz=sweep_range_hz,
                n_freq=n_freq,
                postprocess_workers=postprocess_workers,
                postprocess_parallel_min_tasks=postprocess_parallel_min_tasks,
            )
            response["signs"] = signs
            response["label"] = "".join("+" if s > 0 else "-" for s in signs)
            out.append(response)
        return out

    def dense_traveling_wave_metrics_for_plot(
        self,
        voltage_vector,
        *,
        sweep_range_hz=None,
        n_freq=None,
        traveling_wave_settings: Optional[Dict[str, Any]] = None,
        postprocess_workers: Optional[int] = None,
        postprocess_parallel_min_tasks: Optional[int] = None,
    ) -> dict:
        """Dense frequency sweep of traveling-wave metrics for one voltage pattern."""
        if self.optimizer is None or self.fe is None:
            raise ValueError("PostProcessor requires optimizer and fe for traveling-wave sweeps")

        settings = dict(self.optimizer.objective_settings.traveling_wave_settings)
        if traveling_wave_settings is not None:
            settings.update(dict(traveling_wave_settings))
        sweep_range_hz = sweep_range_hz or self.optimizer.objective_settings.final_sweep_range_hz
        n_freq = int(n_freq or self.optimizer.objective_settings.final_sweep_n_freq)

        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)

        def evaluate_response(omega):
            # u_red = solve_harmonic_response(self.fe, omega, voltage_vector)
            u_red, metadata = self._solve_frequency_response(omega, voltage_vector)
            metrics = traveling_wave_metrics(self.fe, u_red, settings)
            return metrics["score"], metrics["traveling_index"], metrics["amplitude_rms"], metadata

        results = parallel_map(
            evaluate_response,
            omega_vec,
            workers=self._workers(postprocess_workers),
            min_tasks=self._min_tasks(postprocess_parallel_min_tasks),
        )

        metadata_records = [result[3] for result in results]

        return {
            "freq": freq,
            "omega": omega_vec,
            "score": np.asarray([result[0] for result in results], dtype=float),
            "traveling_index": np.asarray([result[1] for result in results], dtype=float),
            "amplitude_rms": np.asarray([result[2] for result in results], dtype=float),
            "voltage_vector": voltage_vector,
            "traveling_wave_settings": settings,
            "metric_label": "Traveling-wave objective score [-]",
            "response_label": self._frequency_response_label(),
            "response_kind": self._frequency_response_kind(),
            "water_model": self._water_settings().model if self._water_enabled() else "none",
            "water_enabled": self._water_enabled(),
            "metadata": metadata_records,
        }

    def dense_current_frequency_response_for_plot(
        self,
        voltage_vector,
        *,
        sweep_range_hz=None,
        n_freq=None,
        postprocess_workers: Optional[int] = None,
        postprocess_parallel_min_tasks: Optional[int] = None,
    ) -> dict:
        """Dense electrical current and power response for I(omega) = Y(omega)V."""
        if self.optimizer is None or self.fe is None:
            raise ValueError("PostProcessor requires optimizer and fe for current frequency responses")

        opt_settings = self.optimizer.objective_settings
        sweep_range_hz = sweep_range_hz or opt_settings.final_sweep_range_hz
        n_freq = int(n_freq or opt_settings.final_sweep_n_freq)

        freq = np.linspace(float(sweep_range_hz[0]), float(sweep_range_hz[1]), n_freq)
        omega_vec = 2.0 * np.pi * freq
        voltage_vector = np.asarray(voltage_vector, dtype=complex)

        def evaluate_current(omega):
            # Y = admittance_matrix(self.fe, omega)
            # current = Y @ voltage_vector

            if not self._water_enabled() or self._water_settings().model == "linear":
                extra_mass, extra_damping = self._linear_water_matrices()
                Y = admittance_matrix(
                    self.fe,
                    omega,
                    extra_mass=extra_mass,
                    extra_damping=extra_damping,
                )
                current = Y @ voltage_vector
                metadata = {"water_model": self._water_settings().model if self._water_enabled() else "none"}
            else:
                u_red, metadata = self._solve_frequency_response(omega, voltage_vector)
                current = self._current_from_response(omega, voltage_vector, u_red)

            power = complex_power_from_peak_phasors(voltage_vector, current)
            return current, power, metadata

        results = parallel_map(
            evaluate_current,
            omega_vec,
            workers=self._workers(postprocess_workers),
            min_tasks=self._min_tasks(postprocess_parallel_min_tasks),
        )

        metadata_records = [result[2] for result in results]
        current = np.vstack([r[0] for r in results])
        power = np.asarray([r[1] for r in results], dtype=complex)

        return {
            "freq": freq,
            "omega": omega_vec,
            "voltage_vector": voltage_vector,
            "current": current,
            "current_magnitude": np.abs(current),
            "current_phase_deg": np.rad2deg(np.angle(current)),
            "complex_power": power,
            "real_power": np.real(power),
            "apparent_power": np.abs(power),
            "current_label": "Patch current magnitude [A]",
            "power_label": "Electrical power [W]",
            "response_label": self._frequency_response_label(),
            "response_kind": self._frequency_response_kind(),
            "water_model": self._water_settings().model if self._water_enabled() else "none",
            "water_enabled": self._water_enabled(),
            "metadata": metadata_records,
        }


# -----------------------------------------------------------------------------
# Saved Artifact Replotting
# -----------------------------------------------------------------------------

def load_saved_case_record(case_dir, *, prefer_pickle: bool = True) -> dict:
    """Load one case folder saved by ``run_sweep``.

    If ``record.pkl`` exists, the returned record can generate full plots. If
    only JSON is available, FE-dependent plots are skipped by the plotter.
    """
    case_dir = Path(case_dir)
    data_dir = case_dir / "data"
    if prefer_pickle:
        record_path = data_dir / "record.pkl"
        if record_path.exists():
            with record_path.open("rb") as f:
                return pickle.load(f)

    case = _decode_jsonable(_read_json(data_dir / "case.json"))
    layout = _decode_jsonable(_read_json(data_dir / "layout.json"))
    inner = _decode_jsonable(_read_json(data_dir / "inner.json"))
    summary_path = data_dir / "summary.json"
    summary = _decode_jsonable(_read_json(summary_path)) if summary_path.exists() else {}
    result_path = data_dir / "optimizer_result.json"
    result = _decode_jsonable(_read_json(result_path)) if result_path.exists() else {}
    return {
        **case,
        "result": result,
        "best": {"layout": layout, "inner": inner},
        "summary": summary,
        "case_dir": case_dir,
    }


def plot_saved_case_artifacts(
    case_dir,
    *,
    L: Optional[float] = None,
    postprocess_settings: Optional[PostProcessingSettings] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    close_plots: bool = True,
    prefer_pickle: bool = True,
    overwrite: bool = True,
) -> dict:
    """Generate plots for one saved case folder."""
    from .plotting import OptimizerPlotter

    case_dir = Path(case_dir)
    record = load_saved_case_record(case_dir, prefer_pickle=prefer_pickle)
    settings_path = case_dir / "data" / "postprocess_settings.json"
    if postprocess_settings is None and settings_path.exists():
        postprocess_settings = PostProcessingSettings(**_read_json(settings_path))
    best = record["best"]
    if L is None:
        L = float(record.get("L", _infer_L_from_layout(best["layout"])))

    plotter = OptimizerPlotter(
        optimizer=record.get("optimizer"),
        postprocess_settings=postprocess_settings,
    )
    options = dict(plot_kwargs or {})
    options.setdefault("show", False)
    figures = plotter.plot_optimizer_result(best=best, L=L, **options)

    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = plots_dir / f"{name}.png"
        if overwrite or not path.exists():
            fig.savefig(path, dpi=200, bbox_inches="tight")

    if close_plots:
        import matplotlib.pyplot as plt

        for fig in figures.values():
            plt.close(fig)

    return {"record": record, "figures": figures, "plots_dir": plots_dir}


def plot_saved_sweep_artifacts(
    sweep_dir,
    *,
    L: Optional[float] = None,
    postprocess_settings: Optional[PostProcessingSettings] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    close_plots: bool = True,
    prefer_pickle: bool = True,
    overwrite: bool = True,
) -> list[dict]:
    """Generate plots for every saved case folder under a sweep directory."""
    sweep_dir = Path(sweep_dir)
    case_dirs = sorted(path.parent.parent for path in sweep_dir.glob("*/data/case.json"))
    results = []
    for case_dir in case_dirs:
        results.append(
            plot_saved_case_artifacts(
                case_dir,
                L=L,
                postprocess_settings=postprocess_settings,
                plot_kwargs=plot_kwargs,
                close_plots=close_plots,
                prefer_pickle=prefer_pickle,
                overwrite=overwrite,
            )
        )
    return results
