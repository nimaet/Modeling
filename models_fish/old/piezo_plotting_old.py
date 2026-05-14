"""Plotting and modal post-processing helpers for piezo patch optimization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_layout(layout: dict, L: float, ax=None):
    """Plot beam layout with shaded active piezo spans."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 1.8))
    else:
        fig = ax.get_figure()

    ax.plot([0, L], [0, 0], "k-", lw=3, label="beam")
    for j, (a, b) in enumerate(zip(layout["xL"], layout["xR"])):
        ax.axvspan(a, b, alpha=0.35, label="active piezo" if j == 0 else None)
        ax.text(0.5 * (a + b), 0.02, f"P{j + 1}", ha="center", va="bottom")
    for x in layout["x_starts"]:
        ax.axvline(x, color="0.7", lw=0.8, ls="--")

    ax.set_xlim(0, L)
    ax.set_yticks([])
    ax.set_xlabel("x [m]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_tip_frf(frf: dict, natural_freq_hz: float | None = None, ax=None, label: str | None = None):
    """Semilog plot of tip displacement FRF."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()
    ax.semilogy(frf["freq"], frf["tip_disp"], lw=2, label=label or "tip FRF")
    if natural_freq_hz is not None:
        ax.axvline(natural_freq_hz, color="k", ls="--", lw=1, label=f"natural freq: {natural_freq_hz:.3g} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Tip displacement magnitude [m/V]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_all_binary_frf(all_results: list[dict], natural_freq_hz: float | None = None, ax=None):
    """Semilog plot of all binary sign-pattern FRFs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()
    for r in all_results:
        ax.semilogy(r["freq"], r["tip_disp"], lw=1.2, label=r.get("label", None))
    if natural_freq_hz is not None:
        ax.axvline(natural_freq_hz, color="k", ls="--", lw=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Tip displacement magnitude [m/V]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax

def plot_all_binary_bar(all_results, ax=None):
    """Compare all binary sign patterns at the selected natural frequency.

    Expected input:
        best["inner"]["all_phase_results"]

    Each result should contain:
        "label", "score"
    """
    if all_results is None:
        raise ValueError(
            "all_results is None. For binary phase mode, pass "
            "best['inner']['all_phase_results'].test"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    labels = []
    scores = []

    for r in all_results:
        labels.append(r.get("label", ""))
        scores.append(float(r["score"]))

    ax.bar(labels, scores)
    ax.set_xlabel("Binary signs")
    ax.set_ylabel("Tip displacement at natural frequency [m/V]")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax

def hermite_mode_shape_and_curvature(fe, mode_number: int, points_per_element: int = 25):
    """Evaluate mode shape and Euler-Bernoulli curvature using Hermite shape functions.

    Returns x, w, curvature arrays. Curvature is d²w/dx².
    """
    if mode_number < 1 or mode_number > fe.Phi.shape[1]:
        raise ValueError("mode_number outside available range")
    phi = fe.Phi[:, mode_number - 1]
    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)

    xs_all = []
    w_all = []
    kappa_all = []
    for e in range(len(x_nodes) - 1):
        x0, x1 = x_nodes[e], x_nodes[e + 1]
        Le = x1 - x0
        if Le <= 0:
            continue
        xi = np.linspace(0.0, 1.0, points_per_element)
        if e > 0:
            xi = xi[1:]  # avoid duplicate node between elements
        x = x0 + xi * Le

        w0 = phi[2 * e]
        th0 = phi[2 * e + 1]
        w1 = phi[2 * (e + 1)]
        th1 = phi[2 * (e + 1) + 1]

        N1 = 1 - 3 * xi**2 + 2 * xi**3
        N2 = Le * (xi - 2 * xi**2 + xi**3)
        N3 = 3 * xi**2 - 2 * xi**3
        N4 = Le * (-xi**2 + xi**3)
        w = N1 * w0 + N2 * th0 + N3 * w1 + N4 * th1

        d2N1 = (-6 + 12 * xi) / Le**2
        d2N2 = (-4 + 6 * xi) / Le
        d2N3 = (6 - 12 * xi) / Le**2
        d2N4 = (-2 + 6 * xi) / Le
        kappa = d2N1 * w0 + d2N2 * th0 + d2N3 * w1 + d2N4 * th1

        xs_all.append(x)
        w_all.append(w)
        kappa_all.append(kappa)

    return np.concatenate(xs_all), np.concatenate(w_all), np.concatenate(kappa_all)


def plot_mode_shape_and_curvature(fe, layout: dict | None = None, mode_number: int = 1, points_per_element: int = 25):
    """Plot mode shape and curvature with optional patch overlays."""
    x, w, kappa = hermite_mode_shape_and_curvature(fe, mode_number, points_per_element)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, w, lw=2)
    axes[0].set_ylabel(f"Mode {mode_number} shape")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, kappa, lw=2)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel(r"Curvature $d^2\phi/dx^2$")
    axes[1].grid(True, alpha=0.3)

    if layout is not None:
        for ax in axes:
            for a, b in zip(layout["xL"], layout["xR"]):
                ax.axvspan(a, b, alpha=0.15)

    fig.tight_layout()
    return fig, axes
