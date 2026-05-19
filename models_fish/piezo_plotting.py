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


def plot_metric_frf(frf: dict, natural_freq_hz: float | None = None, ax=None, label: str | None = None):
    """Semilog plot of any scalar output metric from dense_metric_frf_for_plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    y = frf.get("metric", frf.get("tip_disp"))
    ylabel = frf.get("metric_label", "Output magnitude [m/V]")
    ax.semilogy(frf["freq"], y, lw=2, label=label or frf.get("output", "FRF"))
    if natural_freq_hz is not None:
        ax.axvline(natural_freq_hz, color="k", ls="--", lw=1, label=f"natural freq: {natural_freq_hz:.3g} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_tip_frf(frf: dict, natural_freq_hz: float | None = None, ax=None, label: str | None = None):
    """Backward-compatible semilog plot of tip displacement FRF."""
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


def plot_all_binary_frf(all_results: list[dict], natural_freq_hz: float | None = None, ax=None, use_metric: bool = False):
    """Semilog plot of all binary sign-pattern FRFs.

    use_metric=False keeps old behavior and plots tip displacement.
    use_metric=True plots the selected output metric from dense_metric_frf_for_plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()
    for r in all_results:
        y = r["metric"] if use_metric and "metric" in r else r["tip_disp"]
        ax.semilogy(r["freq"], y, lw=1.2, label=r.get("label", None))
    if natural_freq_hz is not None:
        ax.axvline(natural_freq_hz, color="k", ls="--", lw=1)
    ax.set_xlabel("Frequency [Hz]")
    if use_metric and all_results:
        ax.set_ylabel(all_results[0].get("metric_label", "Output magnitude [m/V]"))
    else:
        ax.set_ylabel("Tip displacement magnitude [m/V]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_all_binary_bar(all_results: list[dict], ax=None, metric_key: str = "score"):
    """Compare all binary sign patterns at one frequency.

    For natural-frequency inner results, use metric_key="score".
    For dense FRF results, use a scalar key or precompute one first.
    """
    if all_results is None:
        raise ValueError("all_results is None. For binary mode, pass a mode_result['all_phase_results'] or best['inner']['all_phase_results'] for single-mode cases.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    labels = [r.get("label", "") for r in all_results]
    scores = [float(r[metric_key]) for r in all_results]
    ax.bar(labels, scores)
    ax.set_xlabel("Binary signs")
    ax.set_ylabel("Output metric at natural frequency [m/V]")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _phase_deg_from_inputs(phase_deg=None, phase_rad=None, voltage_vector=None):
    if phase_deg is not None:
        return np.asarray(phase_deg, dtype=float) % 360.0
    if phase_rad is not None:
        return np.rad2deg(np.asarray(phase_rad, dtype=float)) % 360.0
    if voltage_vector is not None:
        return np.rad2deg(np.angle(np.asarray(voltage_vector, dtype=complex))) % 360.0
    raise ValueError("Provide phase_deg, phase_rad, or voltage_vector")


def plot_patch_phases(layout: dict, phase_deg=None, phase_rad=None, voltage_vector=None, ax=None, annotate: bool = True):
    """Show each patch's phase as a bar positioned at the physical patch location.

    This is useful for continuous phase optimization. Phases are shown in degrees
    wrapped to [0, 360). For binary phases, bars will be at 0 or 180 degrees.
    """
    phase = _phase_deg_from_inputs(phase_deg=phase_deg, phase_rad=phase_rad, voltage_vector=voltage_vector)
    if phase.ndim != 1:
        raise ValueError("plot_patch_phases expects a 1D phase vector. For multi-mode results, use plot_multimode_phase_matrix or pass one mode_result's phase vector.")

    xL = np.asarray(layout["xL"], dtype=float)
    xR = np.asarray(layout["xR"], dtype=float)
    centers = 0.5 * (xL + xR)
    widths = xR - xL

    if len(phase) != len(centers):
        raise ValueError(f"Number of phases ({len(phase)}) does not match number of patches ({len(centers)})")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.get_figure()

    ax.bar(centers, phase, width=0.9 * widths, align="center", edgecolor="k", alpha=0.75)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Patch phase [deg]")
    ax.grid(True, axis="y", alpha=0.3)

    if annotate:
        for j, (x, p) in enumerate(zip(centers, phase)):
            ax.text(x, min(p + 8, 350), f"P{j + 1}\n{p:.1f}°", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_phase_phasors(phase_deg=None, phase_rad=None, voltage_vector=None, ax=None, annotate: bool = True):
    """Polar phasor plot for patch voltage phases."""
    if voltage_vector is not None:
        v = np.asarray(voltage_vector, dtype=complex)
        theta = np.angle(v)
        radius = np.abs(v)
    else:
        phase = _phase_deg_from_inputs(phase_deg=phase_deg, phase_rad=phase_rad)
        if phase.ndim != 1:
            raise ValueError("plot_phase_phasors expects a 1D phase vector. For multi-mode results, pass one mode_result's voltage_vector.")
        theta = np.deg2rad(phase)
        radius = np.ones_like(theta)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    for j, (th, r) in enumerate(zip(theta, radius)):
        ax.plot([0, th], [0, r], marker="o", lw=2, label=f"P{j + 1}")
        if annotate:
            ax.text(th, r * 1.08, f"P{j + 1}", ha="center", va="center")
    ax.set_title("Patch voltage phasors")
    ax.legend(loc="best", bbox_to_anchor=(1.1, 1.1), fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_traveling_wave_shape(inner: dict, ax=None, n_time: int = 8):
    """Plot envelope and time snapshots for a traveling-wave optimizer result."""
    if inner.get("objective") != "traveling_wave":
        raise ValueError("plot_traveling_wave_shape requires a traveling-wave inner result")

    metrics = inner["traveling_wave_metrics"]
    x = np.asarray(metrics.get("x_full", metrics["x"]), dtype=float)
    W = np.asarray(metrics.get("W_full", metrics["W"]), dtype=complex)
    amp = np.abs(W)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    ax.plot(x, amp, "k--", lw=1.5, label="envelope")
    ax.plot(x, -amp, "k--", lw=1.5)

    phases = np.linspace(0.0, 2.0 * np.pi, int(n_time), endpoint=False)
    for phi in phases:
        ax.plot(x, np.real(W * np.exp(1j * phi)), lw=1.0, alpha=0.75)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Transverse displacement [m/V]")
    ax.set_title(
        "Traveling wave: "
        f"T_i={metrics['traveling_index']:.3f}, "
        f"A_rms={metrics['amplitude_rms']:.3e}, "
        f"CV={metrics['envelope_cv']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_traveling_wave_metric_sweep(sweep: dict, ax=None):
    """Plot traveling-wave objective score and traveling index over frequency."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    freq = np.asarray(sweep["freq"], dtype=float)
    ax.plot(freq, sweep["score"], lw=2, label="objective score")
    ax.plot(freq, sweep["traveling_index"], lw=1.5, label="traveling index")
    ax.plot(freq, sweep["envelope_score"], lw=1.2, label="envelope score")
    ax.plot(freq, sweep["amplitude_score"], lw=1.2, label="amplitude score")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Dimensionless score")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_multimode_phase_matrix(inner: dict, ax=None, relative: bool = True, annotate: bool = True):
    """Visualize optimized phase of each patch for each target mode.

    Works with a multi-mode ``best['inner']`` dictionary from the optimizer. Rows
    are modes and columns are patches.
    """
    if inner.get("objective") != "multi_mode":
        # Single-mode fallback: show one row.
        phase = np.asarray(inner["relative_phase_deg" if relative else "phase_deg"], dtype=float)[None, :]
        mode_numbers = [inner.get("target_mode_number", inner.get("single_mode_number", 1))]
    else:
        key = "relative_phase_deg" if relative else "phase_deg"
        phase = np.vstack([np.asarray(v, dtype=float) for v in inner[key]])
        mode_numbers = list(inner["multi_mode_numbers"])

    phase = phase % 360.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * phase.shape[1]), max(2.5, 0.6 * phase.shape[0] + 1.5)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(phase, aspect="auto", vmin=0, vmax=360)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Phase [deg]")
    ax.set_xticks(np.arange(phase.shape[1]))
    ax.set_xticklabels([f"P{j + 1}" for j in range(phase.shape[1])])
    ax.set_yticks(np.arange(phase.shape[0]))
    ax.set_yticklabels([f"Mode {m}" for m in mode_numbers])
    ax.set_xlabel("Patch")
    ax.set_ylabel("Target mode")
    ax.set_title("Optimized patch phases" + (" (relative)" if relative else ""))

    if annotate:
        for i in range(phase.shape[0]):
            for j in range(phase.shape[1]):
                ax.text(j, i, f"{phase[i, j]:.0f}°", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_multimode_score_bar(inner: dict, ax=None):
    """Bar plot of raw and weighted per-mode scores for a multi-mode objective."""
    if inner.get("objective") != "multi_mode":
        raise ValueError("plot_multimode_score_bar requires a multi-mode inner result")

    modes = list(inner["multi_mode_numbers"])
    raw = np.asarray(inner["raw_mode_scores"], dtype=float)
    weighted = np.asarray(inner["weighted_mode_scores"], dtype=float)
    x = np.arange(len(modes))
    width = 0.38

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.bar(x - width / 2, raw, width, label="raw")
    ax.bar(x + width / 2, weighted, width, label="weighted/normalized")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Mode {m}" for m in modes])
    ax.set_ylabel("Score contribution")
    ax.set_title(f"Multi-mode score = {inner['score']:.3e}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
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
            xi = xi[1:]
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
