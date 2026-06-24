"""Geometry performance study helpers.

This module is intentionally outside the optimizer core. It reuses the existing
FE model and single-mode phase optimizer to answer:

``For these fixed patch geometries, what is the best actuation profile and
response amplitude for each target mode?``
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from Modeling.models_fish.model.beam_properties import PiezoBeamParams
from Modeling.models_fish.model.model import (
    PiezoBeamFE,
    build_geometry_arbitrary_piezos,
    build_region_types_from_params,
)
from Modeling.models_fish.optimizer.optimizer_helpers import (
    OBJECTIVE_SINGLE_MODE,
    OUTPUT_TIP,
)
from Modeling.models_fish.optimizer.optimizer_settings import ObjectiveSettings
from Modeling.models_fish.optimizer.single_mode_optimizer import SingleModeOptimizer


# -----------------------------------------------------------------------------
# Geometry Normalization
# -----------------------------------------------------------------------------

@dataclass
class FixedGeometryCase:
    """Named fixed patch geometry used by the study evaluator."""

    name: str
    xL: np.ndarray
    xR: np.ndarray
    metadata: dict[str, Any]


@dataclass
class _SingleModeContext:
    """Small adapter so SingleModeOptimizer can run without an outer optimizer."""

    objective_settings: ObjectiveSettings


def design_vector_to_patch_edges(z: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert optimizer-style ``[L1, g12, L2, ...]`` into patch edge arrays."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size % 2 != 1:
        raise ValueError("A design vector z must be 1D with length 2*Np - 1")

    patch_lengths = z[0::2]
    gaps = z[1::2]
    xL = []
    xR = []
    x = 0.0
    for j, length in enumerate(patch_lengths):
        if length <= 0:
            raise ValueError("Patch lengths must be positive")
        xL.append(x)
        x += float(length)
        xR.append(x)
        if j < len(gaps):
            if gaps[j] < 0:
                raise ValueError("Gaps must be nonnegative")
            x += float(gaps[j])
    return np.asarray(xL, dtype=float), np.asarray(xR, dtype=float)


def normalize_geometry_case(case: dict[str, Any] | FixedGeometryCase, *, name_prefix: str = "geometry") -> FixedGeometryCase:
    """Normalize accepted geometry inputs into explicit patch edge arrays.

    Accepted input dictionaries:
    - ``{"name": "...", "z": [...]}``
    - ``{"name": "...", "xL": [...], "xR": [...]}``
    - ``{"name": "...", "patches": [[xL0, xR0], ...]}``
    """
    if isinstance(case, FixedGeometryCase):
        return case

    if not isinstance(case, dict):
        raise TypeError("Geometry cases must be dictionaries or FixedGeometryCase objects")

    name = str(case.get("name", f"{name_prefix}_{id(case)}"))
    metadata = {k: v for k, v in case.items() if k not in {"name", "z", "xL", "xR", "patches"}}

    if "z" in case:
        xL, xR = design_vector_to_patch_edges(case["z"])
        metadata["z"] = np.asarray(case["z"], dtype=float)
    elif "patches" in case:
        patches = np.asarray(case["patches"], dtype=float)
        if patches.ndim != 2 or patches.shape[1] != 2:
            raise ValueError("patches must have shape (n_patches, 2)")
        xL = patches[:, 0]
        xR = patches[:, 1]
    elif "xL" in case and "xR" in case:
        xL = np.asarray(case["xL"], dtype=float)
        xR = np.asarray(case["xR"], dtype=float)
    else:
        raise ValueError("Each geometry case needs z, xL/xR, or patches")

    if xL.ndim != 1 or xR.ndim != 1 or xL.size != xR.size:
        raise ValueError("xL and xR must be 1D arrays with the same length")
    if xL.size == 0:
        raise ValueError("A performance study geometry must contain at least one patch")

    order = np.argsort(xL)
    xL = xL[order]
    xR = xR[order]
    if np.any(xL >= xR):
        raise ValueError("Each patch must satisfy xL < xR")
    if np.any(xR[:-1] > xL[1:] + 1e-12):
        raise ValueError("Patch intervals must not overlap")

    return FixedGeometryCase(name=name, xL=xL, xR=xR, metadata=metadata)


# -----------------------------------------------------------------------------
# FE Construction
# -----------------------------------------------------------------------------

def _region_types_or_default(base_params: PiezoBeamParams, region_types: dict[str, Any] | None, default_h: float):
    if region_types is not None:
        return region_types
    return build_region_types_from_params(base_params, h_patch=default_h, h_gap=default_h)


def build_fe_for_patch_edges(
    *,
    L: float,
    xL: Sequence[float],
    xR: Sequence[float],
    base_params: PiezoBeamParams,
    region_types: dict[str, Any] | None = None,
    default_h: float = 1e-3,
) -> PiezoBeamFE:
    """Build an FE model for explicit patch edges without running geometry optimization."""
    region_types = _region_types_or_default(base_params, region_types, default_h)
    piezo = region_types["piezo"]
    substrate = region_types["substrate"]

    geom = build_geometry_arbitrary_piezos(
        L=float(L),
        xL=np.asarray(xL, dtype=float),
        xR=np.asarray(xR, dtype=float),
        EI_patch=float(piezo["EI"]),
        rhoA_patch=float(piezo["rhoA"]),
        EI_gap=float(substrate["EI"]),
        rhoA_gap=float(substrate["rhoA"]),
        h_patch=float(piezo.get("h", default_h)),
        h_gap=float(substrate.get("h", default_h)),
    )
    params = copy.deepcopy(base_params)
    params.geometry = geom
    params.sync_patch_count(len(geom.piezos))
    return PiezoBeamFE(params)


def build_fe_for_geometry_case(
    case: dict[str, Any] | FixedGeometryCase,
    *,
    L: float,
    base_params: PiezoBeamParams,
    region_types: dict[str, Any] | None = None,
    default_h: float = 1e-3,
) -> tuple[PiezoBeamFE, FixedGeometryCase]:
    """Normalize a geometry case and build its FE model."""
    normalized = normalize_geometry_case(case)
    fe = build_fe_for_patch_edges(
        L=L,
        xL=normalized.xL,
        xR=normalized.xR,
        base_params=base_params,
        region_types=region_types,
        default_h=default_h,
    )
    return fe, normalized


# -----------------------------------------------------------------------------
# Mode-by-Mode Actuation Study
# -----------------------------------------------------------------------------

def _make_mode_settings(
    mode_number: int,
    *,
    phase_mode: str,
    output: str,
    voltage_amplitude: float,
    objective_settings_kwargs: dict[str, Any] | None,
) -> ObjectiveSettings:
    kwargs = dict(objective_settings_kwargs or {})
    standing_wave_settings = dict(kwargs.pop("standing_wave_settings", {}))
    standing_wave_settings["single_mode_number"] = int(mode_number)
    return ObjectiveSettings(
        objective=OBJECTIVE_SINGLE_MODE,
        standing_wave_settings=standing_wave_settings,
        phase_mode=phase_mode,
        output=output,
        voltage_amplitude=voltage_amplitude,
        **kwargs,
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def mode_result_to_row(
    *,
    geometry: FixedGeometryCase,
    fe: PiezoBeamFE,
    mode_result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one optimized mode result into a table-friendly row."""
    metrics = mode_result.get("response_metrics", {})
    return {
        "geometry": geometry.name,
        "n_patches": int(len(geometry.xL)),
        "mode_number": int(mode_result["mode_number"]),
        "freq_hz": float(mode_result["freq_hz"]),
        "phase_mode": mode_result["phase_mode"],
        "phase_optimizer": mode_result["phase_optimizer"],
        "output": mode_result["output"],
        "score": float(mode_result["score"]),
        "amplitude": float(metrics.get("selected", mode_result["score"])),
        "tip": float(metrics.get("tip", np.nan)),
        "mean_abs": float(metrics.get("mean_abs", np.nan)),
        "rms": float(metrics.get("rms", np.nan)),
        "xL": geometry.xL.copy(),
        "xR": geometry.xR.copy(),
        "freqs_first_five_hz": fe.freq[:5].copy(),
        "phase_deg": np.asarray(mode_result["phase_deg"], dtype=float),
        "relative_phase_deg": np.asarray(mode_result["relative_phase_deg"], dtype=float),
        "voltage_vector": np.asarray(mode_result["voltage_vector"], dtype=complex),
        "signs": None if mode_result.get("signs") is None else np.asarray(mode_result["signs"], dtype=float),
    }


def evaluate_geometry_modes(
    case: dict[str, Any] | FixedGeometryCase,
    *,
    L: float,
    base_params: PiezoBeamParams,
    mode_numbers: Iterable[int],
    region_types: dict[str, Any] | None = None,
    phase_mode: str = "continuous",
    output: str = OUTPUT_TIP,
    voltage_amplitude: float = 1.0,
    default_h: float = 1e-3,
    objective_settings_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize patch actuation independently for each requested mode."""
    fe, geometry = build_fe_for_geometry_case(
        case,
        L=L,
        base_params=base_params,
        region_types=region_types,
        default_h=default_h,
    )

    mode_results = []
    rows = []
    for mode_number in mode_numbers:
        settings = _make_mode_settings(
            int(mode_number),
            phase_mode=phase_mode,
            output=output,
            voltage_amplitude=voltage_amplitude,
            objective_settings_kwargs=objective_settings_kwargs,
        )
        mode_result = SingleModeOptimizer(_SingleModeContext(settings)).evaluate_mode(fe, int(mode_number))
        mode_results.append(mode_result)
        rows.append(mode_result_to_row(geometry=geometry, fe=fe, mode_result=mode_result))

    return {
        "geometry": geometry,
        "fe": fe,
        "mode_results": mode_results,
        "rows": rows,
    }


def make_wide_mode_table(rows: Sequence[dict[str, Any]], *, metric_key: str = "amplitude") -> list[dict[str, Any]]:
    """Create one summary row per geometry with one metric column per mode."""
    by_geometry: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["geometry"]
        mode = int(row["mode_number"])
        out = by_geometry.setdefault(
            name,
            {
                "geometry": name,
                "n_patches": int(row["n_patches"]),
                "output": row["output"],
                "phase_mode": row["phase_mode"],
            },
        )
        out[f"mode_{mode}_freq_hz"] = float(row["freq_hz"])
        out[f"mode_{mode}_{metric_key}"] = float(row[metric_key])
        out[f"mode_{mode}_phase_deg"] = row["phase_deg"]
        out[f"mode_{mode}_voltage_vector"] = row["voltage_vector"]
    return list(by_geometry.values())


def run_geometry_performance_study(
    geometries: Iterable[dict[str, Any] | FixedGeometryCase],
    *,
    L: float,
    base_params: PiezoBeamParams,
    mode_numbers: Iterable[int],
    region_types: dict[str, Any] | None = None,
    phase_mode: str = "continuous",
    output: str = OUTPUT_TIP,
    voltage_amplitude: float = 1.0,
    default_h: float = 1e-3,
    objective_settings_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a set of geometries and return long and wide result tables."""
    rows = []
    geometry_results = []
    for case in geometries:
        result = evaluate_geometry_modes(
            case,
            L=L,
            base_params=base_params,
            mode_numbers=mode_numbers,
            region_types=region_types,
            phase_mode=phase_mode,
            output=output,
            voltage_amplitude=voltage_amplitude,
            default_h=default_h,
            objective_settings_kwargs=objective_settings_kwargs,
        )
        geometry_results.append(result)
        rows.extend(result["rows"])

    return {
        "rows": rows,
        "wide_table": make_wide_mode_table(rows),
        "geometry_results": geometry_results,
        "mode_numbers": tuple(int(m) for m in mode_numbers),
        "phase_mode": phase_mode,
        "output": output,
    }


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_geometry_performance_study(study: dict[str, Any], output_dir: str | Path) -> Path:
    """Save study rows as JSON and CSV for quick spreadsheet comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "geometry_performance_rows.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(study["rows"]), f, indent=2)
    with (output_dir / "geometry_performance_wide.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(study["wide_table"]), f, indent=2)

    csv_columns = [
        "geometry",
        "n_patches",
        "mode_number",
        "freq_hz",
        "phase_mode",
        "output",
        "score",
        "amplitude",
        "tip",
        "mean_abs",
        "rms",
    ]
    with (output_dir / "geometry_performance_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in study["rows"]:
            writer.writerow({key: row.get(key) for key in csv_columns})

    return output_dir
