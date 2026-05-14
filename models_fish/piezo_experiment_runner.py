"""Small experiment-grid utilities for piezo patch optimization notebooks.

This runner supports single-mode and multi-mode sweeps. It deliberately keeps
plotting and saving outside the optimizer: pass a ``plot_callback`` or
``save_callback`` if you want per-case artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pandas is optional
    pd = None

try:
    from Modeling.models_fish.piezo_patch_optimizer import (
        CircuitSettings,
        GeometrySettings,
        ObjectiveSettings,
        OptimizerSettings,
        PiezoPatchOptimizer,
    )
except Exception:
    try:
        from Modeling.models_fish.old.piezo_patch_optimizer_singlemode import (
            CircuitSettings,
            GeometrySettings,
            ObjectiveSettings,
            OptimizerSettings,
            PiezoPatchOptimizer,
        )
    except Exception:
        from Modeling.models_fish.piezo_patch_optimizer_multimode import (
            CircuitSettings,
            GeometrySettings,
            ObjectiveSettings,
            OptimizerSettings,
            PiezoPatchOptimizer,
        )


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
    circuit_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "R_c": 1e3,
        "K_p": 0.02,
        "K_i": 0.0,
        "K_c": 0.0,
    })
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "method": "differential_evolution",
        "maxiter": 10,
        "popsize": 6,
        "seed": 2,
        "polish": False,
        "workers": 1,
        "n_random_samples": 250,
        "powell_maxiter": 80,
    })


def _resolve_tip_bounds(bounds, L):
    if bounds is None:
        return (0.0, L)
    lo, hi = bounds
    return (lo, L if hi is None else hi)


def _objective_name(name: str) -> str:
    name = name.lower().strip().replace("-", "_")
    aliases = {
        "single": "single_mode",
        "single_mode": "single_mode",
        "multi": "multi_mode",
        "multimode": "multi_mode",
        "multi_mode": "multi_mode",
        "traveling": "traveling_wave",
        "travelling": "traveling_wave",
        "traveling_wave": "traveling_wave",
        "travelling_wave": "traveling_wave",
    }
    if name not in aliases:
        raise ValueError("Unknown objective. Use single_mode, multi_mode, or traveling_wave.")
    return aliases[name]


def _mode_label(objective: str, target_mode=None, multi_mode_numbers=None) -> str:
    objective = _objective_name(objective)
    if objective == "single_mode":
        return f"m{int(target_mode)}"
    if objective == "multi_mode":
        return "m" + "-".join(str(int(m)) for m in multi_mode_numbers)
    return "traveling_wave"


def iter_cases(config: SweepConfig):
    """Yield case dictionaries for the Cartesian sweep."""
    objectives = [_objective_name(o) for o in config.objectives]
    target_modes = tuple(int(m) for m in config.target_modes)

    if config.multi_mode_sets is None:
        multi_mode_sets = (target_modes,)
    else:
        multi_mode_sets = tuple(tuple(int(m) for m in modes) for modes in config.multi_mode_sets)

    for Np, phase_mode, output in product(config.patch_counts, config.phase_modes, config.outputs):
        for objective in objectives:
            if objective == "single_mode":
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
            elif objective == "multi_mode":
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
            elif objective == "traveling_wave":
                yield {
                    "Np": int(Np),
                    "objective": objective,
                    "target_mode": None,
                    "multi_mode_numbers": None,
                    "phase_mode": phase_mode,
                    "output": output,
                    "mode_label": "traveling_wave",
                }


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

    objective_name = _objective_name(objective)
    objective_kwargs = dict(config.objective_kwargs)

    # Remove possible conflicts; case-specific values below should win.
    for key in ["objective", "target_mode_number", "single_mode_number", "multi_mode_numbers", "phase_mode", "output"]:
        objective_kwargs.pop(key, None)

    if objective_name == "single_mode":
        if target_mode is None:
            target_mode = 1
        objective_settings = ObjectiveSettings(
            objective="single_mode",
            single_mode_number=int(target_mode),
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == "multi_mode":
        if multi_mode_numbers is None:
            multi_mode_numbers = tuple(config.target_modes)
        objective_settings = ObjectiveSettings(
            objective="multi_mode",
            multi_mode_numbers=tuple(int(m) for m in multi_mode_numbers),
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    elif objective_name == "traveling_wave":
        objective_settings = ObjectiveSettings(
            objective="traveling_wave",
            phase_mode=phase_mode,
            output=output,
            **objective_kwargs,
        )
    else:
        raise RuntimeError(f"Unhandled objective {objective_name}")

    geom_settings = GeometrySettings(Np=Np, **geometry_kwargs)
    circuit_settings = CircuitSettings(**config.circuit_kwargs)
    opt_settings = OptimizerSettings(**config.optimizer_kwargs)

    return PiezoPatchOptimizer(
        L=L,
        region_types=region_types,
        base_params=base_params,
        geometry_settings=geom_settings,
        objective_settings=objective_settings,
        circuit_settings=circuit_settings,
        optimizer_settings=opt_settings,
    )


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
    else:
        mode_numbers = (int(inner.get("target_mode_number", inner.get("single_mode_number", 1))),)
        freq_hz = float(inner["freq_hz"])
        tip_score = metrics.get("tip", np.nan)
        mean_abs_score = metrics.get("mean_abs", np.nan)
        rms_score = metrics.get("rms", np.nan)
        phase_deg = inner.get("phase_deg", None)
        relative_phase_deg = inner.get("relative_phase_deg", None)

    return {
        "Np": len(layout["xL"]),
        "objective": objective,
        "mode_numbers": mode_numbers,
        "mode_label": "m" + "-".join(str(m) for m in mode_numbers) if objective != "traveling_wave" else "traveling_wave",
        "phase_mode": inner["phase_mode"],
        "output": inner.get("output", "tip"),
        "score": inner["score"],
        "tip_score": tip_score,
        "mean_abs_score": mean_abs_score,
        "rms_score": rms_score,
        "raw_mode_scores": inner.get("raw_mode_scores", None),
        "weighted_mode_scores": inner.get("weighted_mode_scores", None),
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
        if save_callback is not None:
            save_callback(record)

    return records


def records_to_dataframe(records):
    """Convert sweep records to a compact pandas DataFrame if pandas is available."""
    rows = []
    for r in records:
        s = r["summary"]
        rows.append({
            "case_id": r["case_id"],
            "Np": s["Np"],
            "objective": s["objective"],
            "mode_label": s["mode_label"],
            "phase_mode": s["phase_mode"],
            "output": s["output"],
            "score": s["score"],
            "raw_mode_scores": s.get("raw_mode_scores", None),
            "freq_hz": s["freq_hz"],
            "success": s["success"],
            "message": s["message"],
        })
    if pd is None:
        return rows
    return pd.DataFrame(rows)
