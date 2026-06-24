"""Preview plots for fixed and curvature-node geometry studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt

from Modeling.models_fish.studies.geometry_performance_study import normalize_geometry_case


# -----------------------------------------------------------------------------
# Basic Layout Preview
# -----------------------------------------------------------------------------

def _geometry_layout(case: dict[str, Any]):
    normalized = normalize_geometry_case(case)
    return normalized.name, np.asarray(normalized.xL, dtype=float), np.asarray(normalized.xR, dtype=float)


def plot_geometry_set(
    geometries: Iterable[dict[str, Any]],
    *,
    L: float,
    ax=None,
    title: str = "Geometry candidates",
):
    """Plot all candidate patch layouts before running a performance study."""
    geometries = list(geometries)
    if ax is None:
        fig_height = max(2.0, 0.65 * len(geometries) + 1.0)
        fig, ax = plt.subplots(figsize=(10, fig_height))
    else:
        fig = ax.get_figure()

    for row, case in enumerate(geometries):
        name, xL, xR = _geometry_layout(case)
        y = float(row)
        ax.plot([0.0, L], [y, y], "k-", lw=2)
        for j, (a, b) in enumerate(zip(xL, xR)):
            ax.add_patch(
                plt.Rectangle(
                    (float(a), y - 0.22),
                    float(b - a),
                    0.44,
                    alpha=0.35,
                    edgecolor="k",
                    facecolor=f"C{j % 10}",
                    linewidth=0.8,
                )
            )
        ax.text(-0.015 * L, y, name, ha="right", va="center", fontsize=9)
        ax.text(L + 0.005 * L, y, f"{len(xL)} patches", ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.18 * L, 1.15 * L)
    ax.set_ylim(-0.7, max(len(geometries) - 0.3, 0.7))
    ax.set_yticks([])
    ax.set_xlabel("x [m]")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Curvature-Node Overlay Preview
# -----------------------------------------------------------------------------

def plot_curvature_node_geometry(
    geometry: dict[str, Any],
    *,
    L: float | None = None,
    ax=None,
    normalize_curvature: bool = True,
    show_all_nodes: bool = True,
    show_curvature_nodes: bool = True,
):
    """Overlay bare-beam modal curvature/strain curves with generated patches."""
    if "mode_nodes" not in geometry:
        raise ValueError("plot_curvature_node_geometry expects a geometry from generate_curvature_node_geometry")

    xL = np.asarray(geometry["xL"], dtype=float)
    xR = np.asarray(geometry["xR"], dtype=float)
    if L is None:
        L = float(np.max([np.max(xR), np.max(geometry["all_nodes"])]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
    else:
        fig = ax.get_figure()

    for j, (a, b) in enumerate(zip(xL, xR)):
        ax.axvspan(float(a), float(b), ymin=0.06, ymax=0.94, alpha=0.18, color="C0")
        ax.text(0.5 * (a + b), 0.92, f"P{j + 1}", ha="center", va="center", fontsize=8, transform=ax.get_xaxis_transform())

    ax.axhline(0.0, color="k", lw=1.0, label="beam")
    for entry in geometry["mode_nodes"]:
        x = np.asarray(entry["x_samples"], dtype=float)
        kappa = np.asarray(entry["curvature_samples"], dtype=float)
        if normalize_curvature:
            scale = max(float(np.max(np.abs(kappa))), 1e-300)
            y = kappa / scale
            ylabel = "Normalized curvature / strain"
        else:
            y = kappa
            ylabel = r"Curvature $d^2\phi/dx^2$"
        ax.plot(x, y, lw=1.8, label=f"mode {entry['mode_number']} ({entry['freq_hz']:.3g} Hz)")

        if show_curvature_nodes:
            for node in np.asarray(entry["curvature_nodes"], dtype=float):
                ax.axvline(node, color="0.35", lw=0.7, ls=":", alpha=0.5)

    if show_all_nodes:
        for node in np.asarray(geometry["all_nodes"], dtype=float):
            ax.axvline(node, color="k", lw=0.9, ls="--", alpha=0.35)

    ax.set_xlim(0.0, float(L))
    ax.set_xlabel("x [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{geometry.get('name', 'curvature-node geometry')} with bare-beam curvature nodes")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax


def preview_geometries(
    geometries: Iterable[dict[str, Any]],
    *,
    L: float,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Generate preview figures for geometry candidates and curvature-node cases."""
    geometries = list(geometries)
    figures = {}
    fig, _ax = plot_geometry_set(geometries, L=L)
    figures["geometry_set"] = fig

    for case in geometries:
        if "mode_nodes" in case:
            fig, _ax = plot_curvature_node_geometry(case, L=L)
            figures[f"curvature_overlay_{case.get('name', len(figures))}"] = fig

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            safe = str(name).replace("/", "-").replace(" ", "_")
            fig.savefig(output_dir / f"{safe}.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    return figures
