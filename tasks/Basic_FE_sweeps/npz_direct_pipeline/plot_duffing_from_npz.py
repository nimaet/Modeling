"""
Classic 2D Duffing FE sweep line plots: three Kc subplots in rows.

Direct-npz variant of results/Duffing_paper_visuals/sim_figs/plot_duffing.py.
Instead of loading a combined `results.pkl` (produced by
collect_nD_sweep_results.py), this script reads the per-simulation
`npz/sim_*.npz` files written directly by generic_nD_sweep_SLURMarray.py.
Only `params` (small) is read for every simulation; the large `data` arrays
are decompressed only for the Kc groups selected via KC_SELECTION. See
README.md in this folder.

Run this file in the VS Code Interactive Window. Edit the user settings block below
for the input run directory, output paths, frequency range, and Kc selection.

Plot layout:
    one subplot row per selected Kc value
    X: frequency [Hz]
    Y: FRF magnitude, semilog scale
    Color: excitation amplitude
    Separate legend-only SVG figure
"""
#%%
from __future__ import annotations

import sys
from pathlib import Path
from importlib import reload

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Original sim_figs folder: reused (read-only import) for build_curve_from_data
# / list_entry and for duffing_linear_refs -- neither is pickle-specific.
SIM_FIGS_DIR = Path(
    r"E:\GaTech Dropbox\Seyednima Etemadi\Projects\Metamaterial beam"
    r"\results\Duffing_paper_visuals\sim_figs"
)
if str(SIM_FIGS_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_FIGS_DIR))

PLOT_SETTINGS_DIR = Path(r"E:\GaTech Dropbox\Seyednima Etemadi\Projects\Metamaterial beam\plot_settings")
if str(PLOT_SETTINGS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SETTINGS_DIR))
import plot_settings  # noqa: F401 - applies shared matplotlib style on import
  # reload to apply any edits made in the VS Code Interactive Window
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import duffing_linear_refs
from duffing_plot_helpers import (
    build_curve_from_data,
    list_entry,
)
from npz_direct_helpers import (
    choose_kc_labels,
    collect_unique_amps_from_index,
    kc_scalar,
    load_npz_results_by_kc,
    npz_kc_index,
)
reload(duffing_linear_refs)
build_fe_linear_cases =  duffing_linear_refs.build_fe_linear_cases
reload(plot_settings)


# =========================
# User settings
# =========================
# Folder containing config.json and npz/ (produced by
# generic_nD_sweep_SLURMarray.py, downloaded by download_npz_results.py) --
# NOT a results.pkl path.
RUN_DIR = Path(
    r"E:\GaTech Dropbox\Seyednima Etemadi\Projects"
        r"\Metamaterial beam\Modeling\tasks\FE_studies"
            r"\sim_dat\Duffing_softening_10928098"
)
# RUN_DIR = Path(
#     r"E:\GaTech Dropbox\Seyednima Etemadi\Projects\Metamaterial beam\Modeling\tasks"
#          r"\FE_studies\sim_dat\Duffing_hardening2_10926975"
# )

if "softening" in str(RUN_DIR):
    OUTPUT_PATH = SCRIPT_DIR /'fig_files'/ "FRF_sim_softening.svg"
    COLORBAR_OUTPUT_PATH = SCRIPT_DIR /'fig_files'/ "FRF_sim_softening_colorbar.svg"
elif "hardening" in str(RUN_DIR):
    OUTPUT_PATH = SCRIPT_DIR /'fig_files'/ "FRF_sim_hardening.svg"
    COLORBAR_OUTPUT_PATH = SCRIPT_DIR /'fig_files'/ "FRF_sim_hardening_colorbar.svg"

FREQ_MIN = 1000.0
FREQ_MAX = 3500.0
Y_MIN = 2e-5
Y_MAX = 2e-3
LINE_STRIDE = 1
SAVE_FIGURE = True

# Choose any number of Kc groups. Order maps directly to subplot rows:
#   KC_SELECTION[0] -> first row, KC_SELECTION[1] -> second row, etc.
# Use zero-based indices into the data order, e.g. [2, 4]. Set to None to
# plot all Kc groups in their native order. Indices refer to the native
# sim-index order in npz/ (same order collect_nD_sweep_results.py would have
# produced), so existing KC_SELECTION values from the results.pkl workflow
# carry over unchanged.
KC_SELECTION = [ 2, 6] #for hardening
KC_SELECTION = [ 2, 6] #for softening

# Figure layout knobs.
ROW_HEIGHT = 1.350
FIG_WIDTH = 3.2
FIG_DPI = 150
GRID_HSPACE = 0.02
FIG_ADJUST = dict(left=0.00, right=1, bottom=0.0, top=1)

# Separate legend/colorbar figure knobs.
USE_TRAPEZOID_COLORBAR = True



COLORBAR_FIG_SIZE = (1.0, 2.05)
COLORBAR_TITLE = r"Excitation voltage"
COLORBAR_UNITS = "V"
COLORBAR_TICK_VALUES = [12.5, 20, 30, 40, 50]  # e.g. [0.5, 1.0, 1.5, 2.0]; None uses all amplitudes
COLORBAR_TICK_DECIMALS = 0
COLORBAR_LABEL_SIDE = "right"  # "right" or "left"
COLORBAR_UPRIGHT_SIDE = "right"  # "left" or "right"

# Linear reference overlay knobs.
OVERLAY_LINEAR_CURVES = True
LINEAR_PROJECT_ROOT = Path(r"E:\GaTech Dropbox\Seyednima Etemadi\Projects\Metamaterial beam")
LINEAR_CASES_ENABLED = {
    "FE_Linear_FD": True,
    # "FE_Linear_LowDamp_FD": True,
}
# =========================
# Trapezoid colorbar helpers
# =========================
def color_position(index, count):
    # Match the line color normalization used below.
    return index / max(count, 1)


def colorbar_tick_position(value, amplitudes):
    amp_positions = np.linspace(0.0, 1.0, len(amplitudes))
    return float(np.interp(value, amplitudes, amp_positions))


def draw_voltage_colorbar_legend(ax, amplitudes, cmap, tick_values=None, fe_cases=None):
    if not amplitudes:
        raise ValueError("No amplitude values found in the loaded data.")

    tick_values = amplitudes if tick_values is None else [float(value) for value in tick_values]
    fe_cases = {} if fe_cases is None else fe_cases

    x_center = 0.50
    y_bottom = 0.18
    y_top = 0.82
    width_bottom = 0.34
    width_top = 0.34

    if COLORBAR_UPRIGHT_SIDE == "right":
        x_right = x_center + width_top / 2
        verts = np.array([
            [x_right - width_bottom, y_bottom],
            [x_right, y_bottom],
            [x_right, y_top],
            [x_right - width_top, y_top],
        ])
    else:
        x_left = x_center - width_top / 2
        verts = np.array([
            [x_left, y_bottom],
            [x_left + width_bottom, y_bottom],
            [x_left + width_top, y_top],
            [x_left, y_top],
        ])

    n = len(amplitudes)
    grad = np.linspace(color_position(0, n), color_position(n - 1, n), 1000)[:, None]
    grad = np.repeat(grad, 250, axis=1)

    im = ax.imshow(
        grad,
        extent=[x_center - width_top / 2, x_center + width_top / 2, y_bottom, y_top],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )

    trapezoid = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(trapezoid)
    im.set_clip_path(trapezoid)

    outline = Polygon(verts, closed=True, facecolor="none", edgecolor="0.25", lw=0.7)
    ax.add_patch(outline)

    ax.text(
        0.5,
        0.96,
        COLORBAR_TITLE,
        ha="center",
        va="top",
        # fontsize=8,
    )

    if COLORBAR_LABEL_SIDE == "left":
        tick_x = x_center - width_top / 2 - 0.04
        tick_end_x = tick_x + 0.035
        label_x = tick_x - 0.035
        label_ha = "right"
    else:
        tick_x = x_center + width_top / 2 + 0.04
        tick_end_x = tick_x - 0.035
        label_x = tick_x + 0.035
        label_ha = "left"

    for value in tick_values:
        y_tick = y_bottom + colorbar_tick_position(value, amplitudes) * (y_top - y_bottom)
        ax.plot([tick_x, tick_end_x], [y_tick, y_tick], color="0.25", lw=0.5)
        ax.text(
            label_x,
            y_tick,
            rf"${value:.{COLORBAR_TICK_DECIMALS}f}\,\mathrm{{{COLORBAR_UNITS}}}$",
            ha=label_ha,
            va="center",
            # fontsize=7,
        )

    fe_y = 0.065
    fe_x0 = 0.22
    fe_x1 = 0.40
    fe_label_x = 0.47
    fe_style = dict(color="k", linestyle="--", linewidth=1.0, label="Linear")
    for fe_case_name, enabled in LINEAR_CASES_ENABLED.items():
        if not enabled or fe_case_name not in fe_cases:
            continue
        case = fe_cases[fe_case_name]
        fe_style = dict(
            color=case["color"],
            linestyle=case["style"],
            linewidth=case["linewidth"],
            label=case["label"],
        )
        break

    ax.plot(
        [fe_x0, fe_x1],
        [fe_y, fe_y],
        color=fe_style["color"],
        linestyle=fe_style["linestyle"],
        linewidth=fe_style["linewidth"],
    )
    ax.text(
        fe_label_x,
        fe_y,
        fe_style["label"],
        ha="left",
        va="center",
        # fontsize=8,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

# =========================
# Load data (direct npz, no collect_nD_sweep_results.py step)
# =========================
print(f"Indexing run: {RUN_DIR}")
kc_index = npz_kc_index(RUN_DIR)
all_kc_labels = [group["label"] for group in kc_index]
selected_kc_labels = choose_kc_labels(all_kc_labels, KC_SELECTION)
unique_amps = collect_unique_amps_from_index(kc_index)

print(f"Kc groups: {len(kc_index)}")
print(f"All Kc labels: {all_kc_labels}")
print(f"Selected Kc labels: {selected_kc_labels}")
print(f"Amplitudes: {unique_amps}")

print(f"Loading npz data for selected groups only: {selected_kc_labels}")
results_by_kc = load_npz_results_by_kc(kc_index, selected_kc_labels)

if not unique_amps:
    raise ValueError("No amplitude values found in the loaded data.")
if LINE_STRIDE < 1:
    raise ValueError("LINE_STRIDE must be >= 1")

fe_cases = {}
if OVERLAY_LINEAR_CURVES:
    print("Building FE linear reference curves...")
    fe_cases = build_fe_linear_cases(LINEAR_PROJECT_ROOT)
    enabled_linear = [name for name, enabled in LINEAR_CASES_ENABLED.items() if enabled and name in fe_cases]
    print(f"Linear overlays: {enabled_linear}")


# =========================
# Flat classic plotting block
# =========================
cmap = plt.cm.inferno
cmap = plt.cm.cividis
cmap = plt.cm.plasma
amp_colors = {
    amp: cmap(i / max(len(unique_amps)  , 1))
    for i, amp in enumerate(unique_amps)
}

n_kc = len(selected_kc_labels)
fig = plt.figure(figsize=(FIG_WIDTH, ROW_HEIGHT * n_kc), dpi=FIG_DPI)
gs = fig.add_gridspec(n_kc, 1, hspace=GRID_HSPACE)
axes = []
for row_idx in range(n_kc):
    if row_idx == 0:
        axes.append(fig.add_subplot(gs[row_idx, :]))
    else:
        axes.append(fig.add_subplot(gs[row_idx, :], sharex=axes[0], sharey=axes[0]))
plotted = 0

for ax, label in zip(axes, selected_kc_labels):
    block = results_by_kc[label]
    data_block = block.get("data", {})
    amps = [float(a) for a in block.get("amps", [])]
    kc_val = kc_scalar(block.get("kc_vec"))

    for idx, amp in enumerate(amps):
        freq = list_entry(data_block, "freq", idx)
        curve = build_curve_from_data(data_block, idx)

        if freq is None or curve is None:
            continue

        freq = np.asarray(freq).squeeze()
        curve = np.asarray(curve).squeeze()
        n = min(freq.size, curve.size)
        if n == 0:
            continue

        freq = freq[:n]
        curve = np.abs(curve[:n])

        valid = np.isfinite(freq) & np.isfinite(curve) & (curve > 0)
        if FREQ_MIN is not None:
            valid &= freq >= FREQ_MIN
        if FREQ_MAX is not None:
            valid &= freq <= FREQ_MAX

        if not np.any(valid):
            continue

        ax.semilogy(
            freq[valid][::LINE_STRIDE],
            curve[valid][::LINE_STRIDE],
            color=amp_colors[amp],
            linewidth=2.0,
            alpha=0.95,
        )
        plotted += 1

    for fe_case_name, enabled in LINEAR_CASES_ENABLED.items():
        if not enabled or fe_case_name not in fe_cases:
            continue
        case = fe_cases[fe_case_name]
        ax.semilogy(
            case["freq"],
            case["frf"],
            color=case["color"],
            linestyle=case["style"],
            linewidth=case["linewidth"],
            zorder=5,
        )
    mantissa, exp = f"{kc_val:.2e}".split('e')
    ax.text(
        0.1,
        0.88,
        rf"$K_c = {float(mantissa):.2f} \times 10^{{{int(exp)}}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )
    # ax.grid(True, which="both")
    ax.minorticks_off()
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
    # pass
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5
                                        #    , prune="upper"
                                           ))
    nticks = 5
    # ticks = np.arange(1000, 3500 + 1, (3500 - 1000) / (nticks + 1))
    # ticks = np.round(ticks/100, decimals=0)*100
    # ax.set_xticks(ticks[1:nticks+1])

if FREQ_MIN is not None and FREQ_MAX is not None:
    for ax in axes:
        ax.set_xlim(FREQ_MIN, FREQ_MAX)

if Y_MIN is not None and Y_MAX is not None:
    for ax in axes:
        ax.set_ylim(Y_MIN, Y_MAX)

if plotted == 0:
    raise ValueError("No valid curves were plotted. Check frequency limits and data fields.")
ax = axes[-1]
ax.tick_params(axis="x", labelbottom=True)
fig.canvas.draw()

xticklabels = ax.get_xticklabels()
if "softening" in str(RUN_DIR):
    xticklabels[0].set_ha("left") # for softening
    for ax in axes: # for softening
        ax.tick_params(axis="y", labelleft=False)
elif "hardening" in str(RUN_DIR):
    xticklabels[-1].set_ha("right")# for hardening
# axes[-1].set_xlabel(r"Frequency [Hz]")
# axes[1].set_ylabel(r"FRF magnitude")
# fig.canvas.draw()
import matplotlib.transforms as mtransforms


fig.subplots_adjust(**FIG_ADJUST)

# =========================
# Separate legend/colorbar figures
# =========================

colorbar_fig = None


if USE_TRAPEZOID_COLORBAR:
    colorbar_fig = plt.figure(figsize=COLORBAR_FIG_SIZE, dpi=FIG_DPI)
    colorbar_ax = colorbar_fig.add_subplot(111)
    draw_voltage_colorbar_legend(
        colorbar_ax,
        unique_amps,
        cmap,
        tick_values=COLORBAR_TICK_VALUES,
    )
    colorbar_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.96)

if SAVE_FIGURE:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="svg", transparent=True, bbox_inches="tight")
    print(f"Saved classic Kc-row figure: {OUTPUT_PATH}")


    if colorbar_fig is not None:
        colorbar_fig.savefig(COLORBAR_OUTPUT_PATH, format="svg", transparent=True, bbox_inches="tight")
        print(f"Saved trapezoid colorbar figure: {COLORBAR_OUTPUT_PATH}")

# =========================
# Export plot-ready data for MATLAB recreation
# =========================
EXPORT_MAT_DATA = True
MAT_DATA_OUTPUT_PATH = OUTPUT_PATH.parents[1] / f"{OUTPUT_PATH.stem}_plot_data.mat"



def _matlab_color(color):
    return np.asarray(plt.matplotlib.colors.to_rgb(color), dtype=float)


def _matlab_row_struct(label):
    block = results_by_kc[label]
    data_block = block.get("data", {})
    amps = [float(a) for a in block.get("amps", [])]
    kc_val = kc_scalar(block.get("kc_vec"))
    curves = []

    for idx, amp in enumerate(amps):
        freq = list_entry(data_block, "freq", idx)
        curve = build_curve_from_data(data_block, idx)
        if freq is None or curve is None:
            continue

        freq = np.asarray(freq).squeeze()
        curve = np.asarray(curve).squeeze()
        n = min(freq.size, curve.size)
        if n == 0:
            continue

        freq = freq[:n]
        curve = np.abs(curve[:n])
        valid = np.isfinite(freq) & np.isfinite(curve) & (curve > 0)
        if FREQ_MIN is not None:
            valid &= freq >= FREQ_MIN
        if FREQ_MAX is not None:
            valid &= freq <= FREQ_MAX
        if not np.any(valid):
            continue

        curves.append(
            {
                "amp": float(amp),
                "freq": freq[valid][::LINE_STRIDE],
                "frf": curve[valid][::LINE_STRIDE],
                "color": np.asarray(amp_colors[amp][:3], dtype=float),
                "alpha": float(0.95),
                "linewidth": float(2.0),
            }
        )

    return {
        "label": str(label),
        "kc": float(kc_val),
        "kc_vec": np.asarray(block.get("kc_vec"), dtype=float).ravel(),
        "curves": np.asarray(curves, dtype=object),
    }


if EXPORT_MAT_DATA:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise ImportError("Saving MATLAB data requires scipy. Install scipy or disable EXPORT_MAT_DATA.") from exc

    enabled_fe_cases = []
    for fe_case_name, enabled in LINEAR_CASES_ENABLED.items():
        if not enabled or fe_case_name not in fe_cases:
            continue
        case = fe_cases[fe_case_name]
        enabled_fe_cases.append(
            {
                "name": fe_case_name,
                "freq": np.asarray(case["freq"], dtype=float).squeeze(),
                "frf": np.asarray(case["frf"], dtype=float).squeeze(),
                "color": _matlab_color(case["color"]),
                "style": str(case["style"]),
                "linewidth": float(case["linewidth"]),
                "label": str(case["label"]),
            }
        )

    n_cmap = 1000
    cmap_samples = np.asarray(cmap(np.linspace(0.0, 1.0, n_cmap))[:, :3], dtype=float)
    colorbar_ticks = (
        np.asarray(COLORBAR_TICK_VALUES, dtype=float)
        if COLORBAR_TICK_VALUES is not None
        else np.asarray(unique_amps, dtype=float)
    )

    mat_payload = {
        "plot_data": {
            "source_run_dir": str(RUN_DIR),
            "output_svg": str(OUTPUT_PATH),
            "colorbar_output_svg": str(COLORBAR_OUTPUT_PATH),
            "figure": {
                "width": float(FIG_WIDTH),
                "row_height": float(ROW_HEIGHT),
                "dpi": float(FIG_DPI),
                "grid_hspace": float(GRID_HSPACE),
                "adjust_left": float(FIG_ADJUST["left"]),
                "adjust_right": float(FIG_ADJUST["right"]),
                "adjust_bottom": float(FIG_ADJUST["bottom"]),
                "adjust_top": float(FIG_ADJUST["top"]),
            },
            "limits": {
                "freq_min": np.nan if FREQ_MIN is None else float(FREQ_MIN),
                "freq_max": np.nan if FREQ_MAX is None else float(FREQ_MAX),
                "y_min": np.nan if Y_MIN is None else float(Y_MIN),
                "y_max": np.nan if Y_MAX is None else float(Y_MAX),
            },
            "selected_kc_labels": np.asarray(selected_kc_labels, dtype=object),
            "unique_amps": np.asarray(unique_amps, dtype=float),
            "kc_rows": np.asarray([_matlab_row_struct(label) for label in selected_kc_labels], dtype=object),
            "linear_cases": np.asarray(enabled_fe_cases, dtype=object),
            "is_softening": bool("softening" in str(RUN_DIR)),
            "is_hardening": bool("hardening" in str(RUN_DIR)),
            "colorbar": {
                "fig_width": float(COLORBAR_FIG_SIZE[0]),
                "fig_height": float(COLORBAR_FIG_SIZE[1]),
                "title": str(COLORBAR_TITLE),
                "units": str(COLORBAR_UNITS),
                "tick_values": colorbar_ticks,
                "tick_decimals": float(COLORBAR_TICK_DECIMALS),
                "label_side": str(COLORBAR_LABEL_SIDE),
                "upright_side": str(COLORBAR_UPRIGHT_SIDE),
                "x_center": float(0.50),
                "y_bottom": float(0.18),
                "y_top": float(0.82),
                "width_bottom": float(0.18),
                "width_top": float(0.34),
                "cmap_samples": cmap_samples,
            },
            "style": {
                "font_name": "Times New Roman",
                "font_size": float(10),
                "axes_linewidth": float(0.5),
                "tick_width": float(0.5),
                "tick_direction": "in",
            },
        }
    }
    savemat(MAT_DATA_OUTPUT_PATH, mat_payload, do_compression=True, long_field_names=True)
    print(f"Saved MATLAB plot data: {MAT_DATA_OUTPUT_PATH}")

plt.show()
#%%
