"""Curvature-node informed geometry generation.

The generated geometry is meant as a comparison point against blind outer
optimization. It first solves the bare substrate beam, finds curvature/strain
zero-crossings for selected modes, unions all node locations, then places one
patch in every interval between adjacent nodes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from Modeling.models_fish.model.beam_properties import PiezoBeamParams
from Modeling.models_fish.model.model import (
    PiezoBeamFE,
    build_geometry_arbitrary_piezos,
    build_region_types_from_params,
)


# -----------------------------------------------------------------------------
# Bare-Beam FE Construction
# -----------------------------------------------------------------------------

def build_bare_beam_fe(
    *,
    L: float,
    base_params: PiezoBeamParams,
    region_types: dict[str, Any] | None = None,
    default_h: float = 1e-3,
) -> PiezoBeamFE:
    """Build a substrate-only beam used for curvature-node extraction."""
    region_types = region_types or build_region_types_from_params(
        base_params,
        h_patch=default_h,
        h_gap=default_h,
    )
    substrate = region_types["substrate"]
    geom = build_geometry_arbitrary_piezos(
        L=float(L),
        xL=np.asarray([], dtype=float),
        xR=np.asarray([], dtype=float),
        EI_patch=float(region_types["piezo"]["EI"]),
        rhoA_patch=float(region_types["piezo"]["rhoA"]),
        EI_gap=float(substrate["EI"]),
        rhoA_gap=float(substrate["rhoA"]),
        h_patch=float(region_types["piezo"].get("h", default_h)),
        h_gap=float(substrate.get("h", default_h)),
    )
    params = copy.deepcopy(base_params)
    params.geometry = geom
    params.sync_patch_count(0)
    return PiezoBeamFE(params)


# -----------------------------------------------------------------------------
# Curvature Sampling and Node Detection
# -----------------------------------------------------------------------------

def _hermite_curvature_row(xi: float, Le: float) -> np.ndarray:
    """Return d2N/dx2 for the Hermite beam interpolation row."""
    return np.asarray(
        [
            (-6.0 + 12.0 * xi) / Le**2,
            (-4.0 + 6.0 * xi) / Le,
            (6.0 - 12.0 * xi) / Le**2,
            (-2.0 + 6.0 * xi) / Le,
        ],
        dtype=float,
    )


def sample_mode_curvature(fe: PiezoBeamFE, mode_number: int, *, samples_per_element: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Sample modal curvature d2w/dx2 along the beam."""
    mode_number = int(mode_number)
    if mode_number < 1 or mode_number > fe.Phi.shape[1]:
        raise ValueError(f"mode_number={mode_number} outside available range 1..{fe.Phi.shape[1]}")
    samples_per_element = int(samples_per_element)
    if samples_per_element < 2:
        raise ValueError("samples_per_element must be >= 2")

    phi = np.asarray(fe.Phi[:, mode_number - 1], dtype=float)
    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)
    xs = []
    kappas = []

    for e in range(len(x_nodes) - 1):
        xa = float(x_nodes[e])
        xb = float(x_nodes[e + 1])
        Le = xb - xa
        if Le <= 0:
            raise ValueError(f"Non-positive element length at element {e}: {Le}")

        # Avoid duplicating element-boundary samples except at the first element.
        xi_values = np.linspace(0.0, 1.0, samples_per_element)
        if e > 0:
            xi_values = xi_values[1:]

        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        qe = phi[dofs]
        for xi in xi_values:
            xs.append(xa + xi * Le)
            kappas.append(float(_hermite_curvature_row(float(xi), Le) @ qe))

    return np.asarray(xs, dtype=float), np.asarray(kappas, dtype=float)


def zero_crossings_from_samples(x: Sequence[float], y: Sequence[float], *, relative_zero_tol: float = 1e-10) -> np.ndarray:
    """Find approximate zero-crossing locations with linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays with the same length")
    if x.size < 2:
        return np.asarray([], dtype=float)

    scale = max(float(np.max(np.abs(y))), 1.0)
    tol = float(relative_zero_tol) * scale
    roots = []

    for i in range(x.size - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if abs(y0) <= tol:
            roots.append(float(x[i]))
        if y0 * y1 < 0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            roots.append(float(x[i] + frac * (x[i + 1] - x[i])))
    if abs(y[-1]) <= tol:
        roots.append(float(x[-1]))

    return np.asarray(sorted(set(np.round(roots, 14))), dtype=float)


def mode_curvature_nodes(
    fe: PiezoBeamFE,
    mode_number: int,
    *,
    samples_per_element: int = 9,
    relative_zero_tol: float = 1e-10,
) -> dict[str, Any]:
    """Return interior curvature nodes for one bare-beam mode."""
    x, curvature = sample_mode_curvature(fe, mode_number, samples_per_element=samples_per_element)
    roots = zero_crossings_from_samples(x, curvature, relative_zero_tol=relative_zero_tol)
    L = float(fe.geom.x_nodes[-1] - fe.geom.x_nodes[0])
    interior = roots[(roots > 1e-12) & (roots < L - 1e-12)]
    return {
        "mode_number": int(mode_number),
        "freq_hz": float(fe.freq[int(mode_number) - 1]),
        "curvature_nodes": interior,
        "x_samples": x,
        "curvature_samples": curvature,
    }


# -----------------------------------------------------------------------------
# Patch Interval Generation
# -----------------------------------------------------------------------------

def _merge_close_nodes(nodes: Sequence[float], *, tolerance: float, L: float) -> np.ndarray:
    nodes = np.asarray(sorted(nodes), dtype=float)
    if nodes.size == 0:
        return np.asarray([0.0, float(L)], dtype=float)

    merged = [float(nodes[0])]
    for node in nodes[1:]:
        if abs(float(node) - merged[-1]) <= tolerance:
            merged[-1] = 0.5 * (merged[-1] + float(node))
        else:
            merged.append(float(node))

    merged[0] = 0.0 if abs(merged[0]) <= tolerance else merged[0]
    merged[-1] = float(L) if abs(merged[-1] - float(L)) <= tolerance else merged[-1]
    out = np.asarray(merged, dtype=float)
    out = out[(out >= -tolerance) & (out <= float(L) + tolerance)]
    out[0] = 0.0
    out[-1] = float(L)
    return out


def patch_edges_between_nodes(
    nodes: Sequence[float],
    *,
    min_patch_length: float = 0.0,
    gap_size: float = 0.0,
    tip_gap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Place one patch in every interval between adjacent node locations.

    ``gap_size`` creates a uniform inactive gap centered on every internal node.
    ``tip_gap`` leaves an inactive substrate gap between the final patch and the
    free tip. No extra margin is added at the root.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("nodes must contain at least two locations")
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("nodes must be strictly increasing")
    gap_size = float(gap_size)
    tip_gap = float(tip_gap)
    if gap_size < 0:
        raise ValueError("gap_size must be nonnegative")
    if tip_gap < 0:
        raise ValueError("tip_gap must be nonnegative")

    left_clearance = np.zeros(nodes.size - 1, dtype=float)
    right_clearance = np.zeros(nodes.size - 1, dtype=float)
    if nodes.size > 2 and gap_size > 0:
        left_clearance[1:] = 0.5 * gap_size
        right_clearance[:-1] = 0.5 * gap_size

    xL = nodes[:-1] + left_clearance
    xR = nodes[1:] - right_clearance
    xR[-1] -= tip_gap
    if np.any(xR <= xL):
        raise ValueError("gap_size/tip_gap is too large for at least one node interval")

    keep = (xR - xL) >= float(min_patch_length)
    return xL[keep].copy(), xR[keep].copy()



def _is_sweep_value(value) -> bool:
    """Return True for list/tuple/array gap inputs used as sweep values."""
    return isinstance(value, (list, tuple, np.ndarray)) and not np.isscalar(value)


def _as_float_tuple(value) -> tuple[float, ...]:
    if _is_sweep_value(value):
        return tuple(float(v) for v in value)
    return (float(value),)

def generate_curvature_node_geometry(
    *,
    L: float,
    base_params: PiezoBeamParams,
    mode_numbers: Iterable[int] | None = None,
    n_modes: int | None = None,
    region_types: dict[str, Any] | None = None,
    default_h: float = 1e-3,
    samples_per_element: int = 9,
    relative_zero_tol: float = 1e-10,
    merge_tolerance: float = 1e-6,
    min_patch_length: float = 0.0,
    gap_size: float = 0.0,
    tip_gap: float = 0.0,
    name: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Generate informed geometry from bare-beam curvature nodes.

    Scalar ``gap_size`` and ``tip_gap`` return one geometry dictionary. Passing a
    list/tuple/array for either value returns every gap/tip-gap combination.
    """
    if _is_sweep_value(gap_size) or _is_sweep_value(tip_gap):
        return generate_curvature_node_geometry_sweep(
            L=L,
            base_params=base_params,
            mode_numbers=mode_numbers,
            n_modes=n_modes,
            region_types=region_types,
            default_h=default_h,
            samples_per_element=samples_per_element,
            relative_zero_tol=relative_zero_tol,
            merge_tolerance=merge_tolerance,
            min_patch_length=min_patch_length,
            gap_sizes=_as_float_tuple(gap_size),
            tip_gaps=_as_float_tuple(tip_gap),
            name=name,
        )

    if mode_numbers is None:
        if n_modes is None:
            raise ValueError("Provide either mode_numbers or n_modes")
        mode_numbers = range(1, int(n_modes) + 1)
    mode_numbers = tuple(int(m) for m in mode_numbers)
    if not mode_numbers:
        raise ValueError("At least one mode number is required")

    fe = build_bare_beam_fe(
        L=L,
        base_params=base_params,
        region_types=region_types,
        default_h=default_h,
    )
    mode_data = [
        mode_curvature_nodes(
            fe,
            mode_number,
            samples_per_element=samples_per_element,
            relative_zero_tol=relative_zero_tol,
        )
        for mode_number in mode_numbers
    ]

    raw_all_nodes = [0.0, float(L)]
    for entry in mode_data:
        raw_all_nodes.extend(float(x) for x in entry["curvature_nodes"])

    # A requested gap cannot fit inside intervals shorter than that gap. Merge
    # such near-duplicate nodes before patch placement while preserving the raw
    # curvature nodes separately for inspection.
    effective_merge_tolerance = max(
        float(merge_tolerance),
        float(gap_size) + float(min_patch_length),
        float(tip_gap) + float(min_patch_length),
    )
    all_nodes = _merge_close_nodes(raw_all_nodes, tolerance=effective_merge_tolerance, L=float(L))
    xL, xR = patch_edges_between_nodes(
        all_nodes,
        min_patch_length=min_patch_length,
        gap_size=gap_size,
        tip_gap=tip_gap,
    )

    if name is None:
        mode_label = "-".join(str(m) for m in mode_numbers)
        name = f"curvature_nodes_modes_{mode_label}"

    return {
        "name": name,
        "source": "bare_beam_curvature_nodes",
        "mode_numbers": mode_numbers,
        "n_patches": int(len(xL)),
        "raw_all_nodes": np.asarray(sorted(raw_all_nodes), dtype=float),
        "all_nodes": all_nodes,
        "xL": xL,
        "xR": xR,
        "patches": np.column_stack([xL, xR]) if len(xL) else np.empty((0, 2)),
        "mode_nodes": mode_data,
        "bare_freq_hz": fe.freq[: max(mode_numbers)].copy(),
        "settings": {
            "samples_per_element": int(samples_per_element),
            "relative_zero_tol": float(relative_zero_tol),
            "merge_tolerance": float(merge_tolerance),
            "effective_merge_tolerance": float(effective_merge_tolerance),
            "min_patch_length": float(min_patch_length),
            "gap_size": float(gap_size),
            "tip_gap": float(tip_gap),
            "default_h": float(default_h),
        },
    }


def generate_curvature_node_geometry_sweep(
    *,
    gap_sizes: Iterable[float] = (0.0,),
    tip_gaps: Iterable[float] = (0.0,),
    **geometry_kwargs,
) -> list[dict[str, Any]]:
    """Generate curvature-node geometries for all gap/tip-gap combinations."""
    geometries = []
    base_name = geometry_kwargs.pop("name", None)
    gap_values = _as_float_tuple(gap_sizes)
    tip_values = _as_float_tuple(tip_gaps)

    for gap_size in gap_values:
        for tip_gap in tip_values:
            name = base_name
            mode_numbers = geometry_kwargs.get("mode_numbers", None)
            n_modes = geometry_kwargs.get("n_modes", None)
            if mode_numbers is None:
                mode_label = f"1-{int(n_modes)}"
            else:
                mode_label = "-".join(str(int(m)) for m in mode_numbers)
            suffix = f"gap_{1e3 * gap_size:.3g}mm_tip_{1e3 * tip_gap:.3g}mm"
            name = f"curvature_nodes_modes_{mode_label}_{suffix}" if name is None else f"{name}_{suffix}"

            geometries.append(
                generate_curvature_node_geometry(
                    **geometry_kwargs,
                    gap_size=gap_size,
                    tip_gap=tip_gap,
                    name=name,
                )
            )
    return geometries


def generate_curvature_node_geometry_gap_sweep(
    *,
    gap_sizes: Iterable[float],
    tip_gaps: Iterable[float] = (0.0,),
    **geometry_kwargs,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for gap/tip-gap geometry sweeps."""
    return generate_curvature_node_geometry_sweep(
        gap_sizes=gap_sizes,
        tip_gaps=tip_gaps,
        **geometry_kwargs,
    )

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_curvature_node_geometry(geometry: dict[str, Any], path: str | Path) -> Path:
    """Save generated nodes and patch intervals as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(geometry), f, indent=2)
    return path
