"""Small experiment-grid utilities for piezo patch optimization notebooks."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ..optimizer.geometry_optimizer import PiezoPatchOptimizer
from ..optimizer.optimizer_helpers import (
    OBJECTIVE_ADMITTANCE_MATCH,
    OBJECTIVE_MULTI_MODE,
    OBJECTIVE_MULTIFUNCTIONAL,
    OBJECTIVE_SINGLE_MODE,
    OBJECTIVE_THRUST,
    OBJECTIVE_THRUST_PER_POWER,
    OBJECTIVE_TRAVELING_WAVE,
    OBJECTIVE_TYPES,
)
from ..optimizer.optimizer_settings import (
    GeometrySettings,
    ObjectiveSettings,
    OptimizerSettings,
    PostProcessingSettings,
)


# -----------------------------------------------------------------------------
# Sweep Configuration
# -----------------------------------------------------------------------------

@dataclass
class SweepConfig:
    patch_counts: Iterable[int] = (2, 3, 4, 5)

    # Single-mode cases use target_modes.
    target_modes: Iterable[int] = (1, 2, 3)

    # Multi-mode cases use multi_mode_sets. If empty/None and objective includes
    # "multi_mode", the runner uses one set containing all target_modes.
    multi_mode_sets: Optional[Iterable[Sequence[int]]] = None

    # Which objective families to run.
    objectives: Iterable[str] = ("single_mode",)

    phase_modes: Iterable[str] = ("binary", "continuous")
    outputs: Iterable[str] = ("tip",)

    geometry_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "patch_length_bounds": (10e-3, 80e-3),
        "gap_bounds": (3e-3, 40e-3),
        "tip_substrate_bounds": (0.0, None),  # None means use full beam length L
    })
    objective_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "voltage_amplitude": 1.0,
        "final_sweep_range_hz": (0.1, 40.0),
        "final_sweep_n_freq": 1000,
    })
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "method": "differential_evolution",
        "seed": 2,
        "differential_evolution_settings": {
            "maxiter": 10,
            "popsize": 6,
            "polish": False,
            "workers": 1,
        },
        "random_search_settings": {
            "n_random_samples": 250,
        },
        "powell_settings": {
            "maxiter": 80,
        },
    })
    output_dir: Optional[str | Path] = None
    save_data: bool = True
    save_plots: bool = True
    close_plots: bool = True
    postprocess_settings: PostProcessingSettings = field(default_factory=PostProcessingSettings)
    plot_kwargs: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Case Generation
# -----------------------------------------------------------------------------

def _resolve_tip_bounds(bounds, L):
    if bounds is None:
        return (0.0, L)
    lo, hi = bounds
    return (lo, L if hi is None else hi)


def _mode_label(objective: str, target_mode=None, multi_mode_numbers=None) -> str:
    if objective not in OBJECTIVE_TYPES:
        raise ValueError(f"objective must be one of {OBJECTIVE_TYPES}.")
    if objective == OBJECTIVE_SINGLE_MODE:
        return f"m{int(target_mode)}"
    if objective == OBJECTIVE_MULTI_MODE:
        return "m" + "-".join(str(int(m)) for m in multi_mode_numbers)
    if objective in (OBJECTIVE_THRUST, OBJECTIVE_THRUST_PER_POWER, OBJECTIVE_MULTIFUNCTIONAL):
        return f"m{int(target_mode)}"
    if objective == OBJECTIVE_ADMITTANCE_MATCH:
        return "admittance_match"
    return OBJECTIVE_TRAVELING_WAVE


def iter_cases(config: SweepConfig):
    """Yield case dictionaries for the Cartesian sweep."""
    objectives = []
    for name in config.objectives:
        if name not in OBJECTIVE_TYPES:
            raise ValueError(f"objective must be one of {OBJECTIVE_TYPES}.")
        objectives.append(name)
    target_modes = tuple(int(m) for m in config.target_modes)

    if config.multi_mode_sets is None:
        multi_mode_sets = (target_modes,)
    else:
        multi_mode_sets = tuple(tuple(int(m) for m in modes) for modes in config.multi_mode_sets)

    for Np, phase_mode, output in product(config.patch_counts, config.phase_modes, config.outputs):
        for objective in objectives:
            if objective in (OBJECTIVE_SINGLE_MODE, OBJECTIVE_THRUST, OBJECTIVE_THRUST_PER_POWER, OBJECTIVE_MULTIFUNCTIONAL):
                for mode in target_modes:
                    yield {
                        "Np": int(Np),
                        "objective": objective,
                        "target_mode": int(mode),
                        "multi_mode_numbers": None,
                        "phase_mode": phase_mode,
                        "output": output,
                        "mode_label": _mode_label(objective, target_mode=mode),
                    }
            elif objective == OBJECTIVE_MULTI_MODE:
                for modes in multi_mode_sets:
                    yield {
                        "Np": int(Np),
                        "objective": objective,
                        "target_mode": None,
                        "multi_mode_numbers": tuple(int(m) for m in modes),
                        "phase_mode": phase_mode,
                        "output": output,
                        "mode_label": _mode_label(objective, multi_mode_numbers=modes),
                    }
            elif objective == OBJECTIVE_TRAVELING_WAVE:
                yield {
                    "Np": int(Np),
                    "objective": objective,
                    "target_mode": None,
                    "multi_mode_numbers": None,
                    "phase_mode": phase_mode,
                    "output": output,
                    "mode_label": "traveling_wave",
                }
            elif objective == OBJECTIVE_ADMITTANCE_MATCH:
                yield {
                    "Np": int(Np),
                    "objective": objective,
                    "target_mode": None,
                    "multi_mode_numbers": None,
                    "phase_mode": phase_mode,
                    "output": output,
                    "mode_label": "admittance_match",
                }


# -----------------------------------------------------------------------------
# Optimizer Construction
# -----------------------------------------------------------------------------

def make_optimizer_for_case(
    *,
    L,
    region_types,
    base_params,
    Np: int,
    phase_mode: str,
    output: str = "tip",
    config: SweepConfig,
    objective: str = "single_mode",
    target_mode: Optional[int] = None,
    multi_mode_numbers: Optional[Sequence[int]] = None,
):
    geometry_kwargs = dict(config.geometry_kwargs)
    geometry_kwargs["tip_substrate_bounds"] = _resolve_tip_bounds(
        geometry_kwargs.get("tip_substrate_bounds", (0.0, L)), L
    )

    if objective not in OBJECTIVE_TYPES:
        raise ValueError(f"objective must be one of {OBJECTIVE_TYPES}.")
    objective_name = objective
    objective_kwargs = dict(config.objective_kwargs)
    standing_wave_settings = dict(objective_kwargs.pop("standing_wave_settings", {}))

    # Remove possible conflicts; case-specific values below should win.
    for key in ["objective", "phase_mode", "output"]:
        objective_kwargs.pop(key, None)

    if objective_name == OBJECTIVE_SINGLE_MODE:
        if target_mode is None:
            target_mode = 1
        standing = dict(standing_wave_settings)
        standing["single_mode_number"] = int(target_mode)
        objective_settings = ObjectiveSettings(
            objective=OBJECTIVE_SINGLE_MODE,
            standing_wave_settings=standing,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name in (OBJECTIVE_THRUST, OBJECTIVE_THRUST_PER_POWER):
        if target_mode is None:
            target_mode = 1
        thrust_settings = dict(objective_kwargs.pop("thrust_settings", {}))
        thrust_settings["mode_number"] = int(target_mode)
        objective_settings = ObjectiveSettings(
            objective=objective_name,
            thrust_settings=thrust_settings,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == OBJECTIVE_MULTIFUNCTIONAL:
        if target_mode is None:
            target_mode = 1
        multifunctional_settings = dict(objective_kwargs.pop("multifunctional_settings", {}))
        multifunctional_settings["mode_number"] = int(target_mode)
        objective_settings = ObjectiveSettings(
            objective=OBJECTIVE_MULTIFUNCTIONAL,
            multifunctional_settings=multifunctional_settings,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == OBJECTIVE_MULTI_MODE:
        if multi_mode_numbers is None:
            multi_mode_numbers = tuple(config.target_modes)
        standing = dict(standing_wave_settings)
        standing["multi_mode_numbers"] = tuple(int(m) for m in multi_mode_numbers)
        objective_settings = ObjectiveSettings(
            objective=OBJECTIVE_MULTI_MODE,
            standing_wave_settings=standing,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == OBJECTIVE_TRAVELING_WAVE:
        objective_settings = ObjectiveSettings(
            objective=OBJECTIVE_TRAVELING_WAVE,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == OBJECTIVE_ADMITTANCE_MATCH:
        objective_settings = ObjectiveSettings(
            objective=OBJECTIVE_ADMITTANCE_MATCH,
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    else:
        raise RuntimeError(f"Unhandled objective {objective_name}")

    geom_settings = GeometrySettings(Np=Np, **geometry_kwargs)
    opt_settings = OptimizerSettings(**config.optimizer_kwargs)

    return PiezoPatchOptimizer(
        L=L,
        region_types=region_types,
        base_params=base_params,
        geometry_settings=geom_settings,
        objective_settings=objective_settings,
        optimizer_settings=opt_settings,
    )


# -----------------------------------------------------------------------------
# Result Summaries
# -----------------------------------------------------------------------------

def summarize_best(best: dict) -> dict:
    result = best["result"]
    inner = best["inner"]
    layout = best["layout"]
    fe = best["fe"]
    objective = inner.get("objective", "single_mode")
    metrics = inner.get("response_metrics", {})

    if objective == "multi_mode":
        mode_numbers = tuple(int(m) for m in inner["multi_mode_numbers"])
        freq_hz = np.asarray(inner["freq_hz"], dtype=float)
        tip_score = np.asarray([r["response_metrics"].get("tip", np.nan) for r in inner["mode_results"]], dtype=float)
        mean_abs_score = np.asarray([r["response_metrics"].get("mean_abs", np.nan) for r in inner["mode_results"]], dtype=float)
        rms_score = np.asarray([r["response_metrics"].get("rms", np.nan) for r in inner["mode_results"]], dtype=float)
        phase_deg = inner.get("phase_deg", None)
        relative_phase_deg = inner.get("relative_phase_deg", None)
        traveling_metrics = None
    elif objective == OBJECTIVE_TRAVELING_WAVE:
        mode_numbers = ()
        freq_hz = float(inner["freq_hz"])
        tip_score = metrics.get("tip", np.nan)
        mean_abs_score = metrics.get("mean_abs", np.nan)
        rms_score = metrics.get("rms", np.nan)
        phase_deg = inner.get("phase_deg", None)
        relative_phase_deg = inner.get("relative_phase_deg", None)
        traveling_metrics = inner.get("traveling_wave_metrics", {})
    elif objective == OBJECTIVE_ADMITTANCE_MATCH:
        mode_numbers = ()
        freq_hz = np.asarray(inner["freq_hz"], dtype=float)
        tip_score = np.nan
        mean_abs_score = np.nan
        rms_score = np.nan
        phase_deg = inner.get("phase_deg", None)
        relative_phase_deg = inner.get("relative_phase_deg", None)
        traveling_metrics = None
    else:
        mode_number = inner.get("mode_number", None)
        mode_numbers = () if mode_number is None else (int(mode_number),)
        freq_hz = float(inner["freq_hz"])
        tip_score = metrics.get("tip", np.nan)
        mean_abs_score = metrics.get("mean_abs", np.nan)
        rms_score = metrics.get("rms", np.nan)
        phase_deg = inner.get("phase_deg", None)
        relative_phase_deg = inner.get("relative_phase_deg", None)
        traveling_metrics = None

    return {
        "Np": len(layout["xL"]),
        "objective": objective,
        "mode_numbers": mode_numbers,
        "mode_label": "m" + "-".join(str(m) for m in mode_numbers) if mode_numbers else objective,
        "phase_mode": inner["phase_mode"],
        "output": inner.get("output", "tip"),
        "score": inner["score"],
        "tip_score": tip_score,
        "mean_abs_score": mean_abs_score,
        "rms_score": rms_score,
        "raw_mode_scores": inner.get("raw_mode_scores", None),
        "weighted_mode_scores": inner.get("weighted_mode_scores", None),
        "traveling_index": None if traveling_metrics is None else traveling_metrics.get("traveling_index", np.nan),
        "traveling_amplitude_rms": None if traveling_metrics is None else traveling_metrics.get("amplitude_rms", np.nan),
        "freq_hz": freq_hz,
        "best_z_mm": 1e3 * result.x,
        "xL_mm": 1e3 * layout["xL"],
        "xR_mm": 1e3 * layout["xR"],
        "phase_deg": phase_deg,
        "relative_phase_deg": relative_phase_deg,
        "first_five_freqs_hz": fe.freq[:5],
        "success": getattr(result, "success", None),
        "message": getattr(result, "message", ""),
    }


def _fmt_array_short(x, precision=4):
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], np.ndarray):
        return "[" + "; ".join(np.array2string(np.asarray(v), precision=precision) for v in x) + "]"
    if isinstance(x, np.ndarray):
        return np.array2string(x, precision=precision)
    return str(x)


# -----------------------------------------------------------------------------
# Artifact Saving
# -----------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _case_slug(record: dict) -> str:
    parts = [
        f"{int(record['case_id']):03d}",
        f"Np{int(record['Np'])}",
        str(record["objective"]),
        str(record["mode_label"]),
        str(record["phase_mode"]),
        str(record["output"]),
    ]
    return "_".join(part.replace("/", "-").replace(" ", "") for part in parts)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(value), f, indent=2)


def save_case_artifacts(
    record: dict,
    *,
    output_dir,
    L,
    save_data: bool = True,
    save_plots: bool = True,
    close_plots: bool = True,
    postprocess_settings: Optional[PostProcessingSettings] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
) -> Path:
    """Save one sweep record into a case folder with data and plots."""
    case_dir = Path(output_dir) / _case_slug(record)
    data_dir = case_dir / "data"
    plots_dir = case_dir / "plots"
    case_dir.mkdir(parents=True, exist_ok=True)

    if save_data:
        data_dir.mkdir(parents=True, exist_ok=True)
        case_info = {
            key: record[key]
            for key in [
                "case_id",
                "Np",
                "objective",
                "target_mode",
                "multi_mode_numbers",
                "phase_mode",
                "output",
                "mode_label",
            ]
        }
        case_info["L"] = L
        best = record["best"]
        result = record["result"]

        _write_json(data_dir / "case.json", case_info)
        _write_json(data_dir / "postprocess_settings.json", asdict(postprocess_settings or PostProcessingSettings()))
        _write_json(data_dir / "summary.json", record["summary"])
        _write_json(data_dir / "layout.json", best["layout"])
        _write_json(data_dir / "inner.json", best["inner"])
        _write_json(
            data_dir / "optimizer_result.json",
            {
                "x": getattr(result, "x", None),
                "fun": getattr(result, "fun", None),
                "success": getattr(result, "success", None),
                "message": getattr(result, "message", ""),
                "optimizer_name": getattr(result, "optimizer_name", None),
            },
        )
        np.savez(
            data_dir / "arrays.npz",
            z=np.asarray(getattr(result, "x", []), dtype=float),
            xL=np.asarray(best["layout"]["xL"], dtype=float),
            xR=np.asarray(best["layout"]["xR"], dtype=float),
            x_starts=np.asarray(best["layout"]["x_starts"], dtype=float),
        )
        with (data_dir / "record.pkl").open("wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        from .plotting import OptimizerPlotter

        plot_options = dict(plot_kwargs or {})
        plot_options.setdefault("show", False)
        figures = OptimizerPlotter(
            optimizer=record["optimizer"],
            postprocess_settings=postprocess_settings,
        ).plot_record(
            record=record,
            L=L,
            **plot_options,
        )
        for name, fig in figures.items():
            fig.savefig(plots_dir / f"{name}.png", dpi=200, bbox_inches="tight")

        if close_plots:
            import matplotlib.pyplot as plt

            for fig in figures.values():
                plt.close(fig)

    record["case_dir"] = case_dir
    return case_dir


# -----------------------------------------------------------------------------
# Sweep Execution
# -----------------------------------------------------------------------------

def run_sweep(
    *,
    L,
    region_types,
    base_params,
    config: SweepConfig,
    plot_callback: Optional[Callable[[dict], None]] = None,
    save_callback: Optional[Callable[[dict], None]] = None,
    verbose: bool = True,
):
    """Run a Cartesian sweep over patch counts, objectives, phase modes, and outputs.

    Returns a list of records. Each record contains optimizer, result, best, and summary.
    """
    records = []
    cases = list(iter_cases(config))

    for case_id, case in enumerate(cases, start=1):
        if verbose:
            print("\n" + "=" * 80)
            print(
                f"Case {case_id}/{len(cases)}: "
                f"Np={case['Np']}, objective={case['objective']}, "
                f"modes={case['mode_label']}, phase={case['phase_mode']}, output={case['output']}"
            )

        optimizer = make_optimizer_for_case(
            L=L,
            region_types=region_types,
            base_params=base_params,
            Np=case["Np"],
            objective=case["objective"],
            target_mode=case["target_mode"],
            multi_mode_numbers=case["multi_mode_numbers"],
            phase_mode=case["phase_mode"],
            output=case["output"],
            config=config,
        )

        if verbose:
            print("Design bounds:")
            for k, bnd in enumerate(optimizer.make_bounds()):
                print(f"  z[{k}]: {1e3*bnd[0]:.3f} to {1e3*bnd[1]:.3f} mm")

        result = optimizer.run()
        best = optimizer.inspect_result(result)
        summary = summarize_best(best)

        if verbose:
            print(f"Best score [{best['inner'].get('metric_label', 'm/V')}]: {summary['score']:.6e}")
            print("Selected natural frequency/frequencies [Hz]:", _fmt_array_short(summary["freq_hz"]))
            print("Best design z [mm]:", _fmt_array_short(summary["best_z_mm"]))
            print("Patch xL [mm]:", _fmt_array_short(summary["xL_mm"]))
            print("Patch xR [mm]:", _fmt_array_short(summary["xR_mm"]))
            print("Phase [deg]:", _fmt_array_short(summary["phase_deg"]))
            if summary.get("raw_mode_scores") is not None:
                print("Raw mode scores:", _fmt_array_short(summary["raw_mode_scores"]))
                print("Weighted mode scores:", _fmt_array_short(summary["weighted_mode_scores"]))
            if summary["objective"] == "traveling_wave":
                print("Traveling index:", _fmt_array_short(summary["traveling_index"]))
                print("Traveling RMS amplitude:", _fmt_array_short(summary["traveling_amplitude_rms"]))

        record = {
            "case_id": case_id,
            **case,
            "optimizer": optimizer,
            "result": result,
            "best": best,
            "summary": summary,
        }
        records.append(record)

        if plot_callback is not None:
            plot_callback(record)
        if config.output_dir is not None:
            save_case_artifacts(
                record,
                output_dir=config.output_dir,
                L=L,
                save_data=config.save_data,
                save_plots=config.save_plots,
                close_plots=config.close_plots,
                postprocess_settings=config.postprocess_settings,
                plot_kwargs=config.plot_kwargs,
            )
        if save_callback is not None:
            save_callback(record)

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "summaries.json", [record["summary"] for record in records])
        with (output_dir / "records.pkl").open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    return records
