"""Grid of 2D phase portraits, one panel per sweep point, for a single
physical (x_eval) spatial location -- reproduces plot_duffing_ROM.ipynb cell
"02f76d53" (displacement vs velocity at a beam location), minus its Poincare
overlay.

See plot_phase_portrait_grid_patch.py's docstring for why there's no
Poincare overlay: the chirp excitation has no fixed period to
stroboscopically sample from.

Displacement isn't saved directly (only velocity is a ROM state-derived
quantity); it's reconstructed here via cumulative_trapezoid of the velocity
tail window, referenced to the start of that window (so absolute
displacement offset is arbitrary, but the attractor's shape in phase space
is preserved).

X_INDEX can be any x_eval node (0-99 for the sweep script's default 100-point
x_eval) -- the sweep script saves the full veloc array, with no
representative-index restriction.

Assumes the sweep's npz/ + config.json have already been downloaded.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

from npz_direct_helpers_ROM import load_config, make_lazy_results, tail_window

# =========================================================
# CONFIG
# =========================================================
RUN_DIR = Path("../../Basic_ROM_sweeps/sim_dat/ROM_Duffing_freq_sweep_REPLACE_WITH_JOB_ID")
SWEEP_KEY = "K_c"
X_INDEX = 40
TAIL_SECONDS = 0.2
SWEEP_SLICE = slice(0, 40)
NCOLS = 8
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
picked = results[SWEEP_SLICE]
print(f"Plotting {len(picked)}/{len(results)} sim(s) for x_eval index {X_INDEX}")

# =========================================================
# PLOT
# =========================================================
nrows = int(np.ceil(len(picked) / NCOLS))
fig, axes = plt.subplots(nrows, NCOLS, figsize=(2 * NCOLS, 2 * nrows))
axes = np.atleast_2d(axes)

for i, r in enumerate(picked):
    row, col = i // NCOLS, i % NCOLS
    ax = axes[row, col]

    mask = tail_window(r["t"], TAIL_SECONDS)
    t_win = r["t"][mask]
    veloc_win = r["veloc"][X_INDEX, mask]
    disp_win = cumulative_trapezoid(veloc_win, t_win, initial=0)

    ax.scatter(disp_win, veloc_win, s=0.1, color='green', alpha=0.9)
    ax.set_title(f'{r[SWEEP_KEY]:.2e}', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)

for j in range(len(picked), nrows * NCOLS):
    axes[j // NCOLS, j % NCOLS].axis('off')

fig.suptitle(f'Phase portraits for x_eval index {X_INDEX} across {SWEEP_KEY} (last {TAIL_SECONDS}s)', fontsize=14)
fig.tight_layout()
fig.savefig(FIGS_DIR / "phase_portrait_grid_spatial.png", dpi=150)
plt.show()
