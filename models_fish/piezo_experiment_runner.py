"""Small experiment-grid utilities for piezo patch optimization notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable, Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pandas is optional
    pd = None

try:
    from Modeling.models.piezo_patch_optimizer import (
        CircuitSettings,
        GeometrySettings,
        OptimizerSettings,
        PiezoPatchOptimizer,
        SingleModeSettings,
    )
except Exception:
    from Modeling.models_fish.piezo_patch_optimizer import (
        CircuitSettings,
        GeometrySettings,
        OptimizerSettings,
        PiezoPatchOptimizer,
        SingleModeSettings,
    )


@dataclass
class SweepConfig:
    patch_counts: Iterable[int] = (2, 3, 4, 5)
    target_modes: Iterable[int] = (1, 2, 3)
    phase_modes: Iterable[str] = ("binary", "continuous")
    outputs: Iterable[str] = ("tip",)

    geometry_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "patch_length_bounds": (10e-3, 80e-3),
        "gap_bounds": (3e-3, 40e-3),
        "tip_substrate_bounds": (0.0, None),  # None means use full beam length L
    })
    mode_kwargs: dict[str, Any] = field(default_factory=lambda: {
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


def make_optimizer_for_case(
    *,
    L,
    region_types,
    base_params,
    Np: int,
    target_mode: int,
    phase_mode: str,
    output: str = "tip",
    config: SweepConfig,
):
    geometry_kwargs = dict(config.geometry_kwargs)
    geometry_kwargs["tip_substrate_bounds"] = _resolve_tip_bounds(
        geometry_kwargs.get("tip_substrate_bounds", (0.0, L)), L
    )

    geom_settings = GeometrySettings(Np=Np, **geometry_kwargs)
    mode_settings = SingleModeSettings(
        target_mode_number=target_mode,
        phase_mode=phase_mode,
        output=output,
        **config.mode_kwargs,
    )
    circuit_settings = CircuitSettings(**config.circuit_kwargs)
    opt_settings = OptimizerSettings(**config.optimizer_kwargs)

    return PiezoPatchOptimizer(
        L=L,
        region_types=region_types,
        base_params=base_params,
        geometry_settings=geom_settings,
        mode_settings=mode_settings,
        circuit_settings=circuit_settings,
        optimizer_settings=opt_settings,
    )


def summarize_best(best: dict) -> dict:
    result = best["result"]
    inner = best["inner"]
    layout = best["layout"]
    fe = best["fe"]
    metrics = inner.get("response_metrics", {})
    return {
        "Np": len(layout["xL"]),
        "mode": inner["target_mode_number"],
        "phase_mode": inner["phase_mode"],
        "output": inner.get("output", "tip"),
        "score": inner["score"],
        "tip_score": metrics.get("tip", np.nan),
        "mean_abs_score": metrics.get("mean_abs", np.nan),
        "rms_score": metrics.get("rms", np.nan),
        "freq_hz": inner["freq_hz"],
        "best_z_mm": 1e3 * result.x,
        "xL_mm": 1e3 * layout["xL"],
        "xR_mm": 1e3 * layout["xR"],
        "phase_deg": inner.get("phase_deg", None),
        "relative_phase_deg": inner.get("relative_phase_deg", None),
        "first_five_freqs_hz": fe.freq[:5],
        "success": getattr(result, "success", None),
        "message": getattr(result, "message", ""),
    }


def run_sweep(
    *,
    L,
    region_types,
    base_params,
    config: SweepConfig,
    plot_callback: Optional[Callable[[dict], None]] = None,
    verbose: bool = True,
):
    """Run a Cartesian sweep over patch counts, modes, phase modes, and outputs.

    Returns a list of records. Each record contains optimizer, result, best, and summary.
    """
    records = []
    cases = list(product(config.patch_counts, config.target_modes, config.phase_modes, config.outputs))

    for case_id, (Np, mode, phase_mode, output) in enumerate(cases, start=1):
        if verbose:
            print("\n" + "=" * 80)
            print(f"Case {case_id}/{len(cases)}: Np={Np}, mode={mode}, phase={phase_mode}, output={output}")

        optimizer = make_optimizer_for_case(
            L=L,
            region_types=region_types,
            base_params=base_params,
            Np=Np,
            target_mode=mode,
            phase_mode=phase_mode,
            output=output,
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
            print(f"Selected natural frequency [Hz]: {summary['freq_hz']:.6g}")
            print("Best design z [mm]:", summary["best_z_mm"])
            print("Patch xL [mm]:", summary["xL_mm"])
            print("Patch xR [mm]:", summary["xR_mm"])
            print("Phase [deg]:", summary["phase_deg"])

        record = {
            "case_id": case_id,
            "Np": Np,
            "mode": mode,
            "phase_mode": phase_mode,
            "output": output,
            "optimizer": optimizer,
            "result": result,
            "best": best,
            "summary": summary,
        }
        records.append(record)

        if plot_callback is not None:
            plot_callback(record)

    return records


def records_to_dataframe(records):
    """Convert sweep records to a compact pandas DataFrame if pandas is available."""
    rows = []
    for r in records:
        s = r["summary"]
        rows.append({
            "case_id": r["case_id"],
            "Np": s["Np"],
            "mode": s["mode"],
            "phase_mode": s["phase_mode"],
            "output": s["output"],
            "score": s["score"],
            "tip_score": s["tip_score"],
            "mean_abs_score": s["mean_abs_score"],
            "rms_score": s["rms_score"],
            "freq_hz": s["freq_hz"],
            "success": s["success"],
            "message": s["message"],
        })
    if pd is None:
        return rows
    return pd.DataFrame(rows)
