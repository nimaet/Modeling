"""Grid of 2D phase portraits, one panel per sweep point, for a single
mechanical mode -- reproduces plot_duffing_ROM.ipynb cell "2881d650" (eta vs
eta_dot), minus its Poincare overlay.

See plot_phase_portrait_grid_patch.py's docstring for why there's no
Poincare overlay: the chirp excitation has no fixed period to stroboscopically
sample from.

MODE_INDEX can be any mode -- the sweep script saves full eta/eta_dot arrays
for every mode, with no representative-index restriction.

Assumes the sweep's npz/ + config.json have already been downloaded.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from npz_direct_helpers_ROM import load_config, make_lazy_results, tail_window

# =========================================================
# CONFIG
# =========================================================
RUN_DIR = Path("../../Basic_ROM_sweeps/sim_dat/ROM_Duffing_freq_sweep_REPLACE_WITH_JOB_ID")
SWEEP_KEY = "K_c"
MODE_INDEX = 0
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
print(f"Plotting {len(picked)}/{len(results)} sim(s) for mode {MODE_INDEX}")

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
    eta = r["eta"][MODE_INDEX, mask]
    eta_dot = r["eta_dot"][MODE_INDEX, mask]

    ax.scatter(eta, eta_dot, s=0.1, color='green', alpha=0.9)
    ax.set_title(f'{r[SWEEP_KEY]:.2e}', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)

for j in range(len(picked), nrows * NCOLS):
    axes[j // NCOLS, j % NCOLS].axis('off')

fig.suptitle(f'Phase portraits for mode {MODE_INDEX} across {SWEEP_KEY} (last {TAIL_SECONDS}s)', fontsize=14)
fig.tight_layout()
fig.savefig(FIGS_DIR / "phase_portrait_grid_mode.png", dpi=150)
plt.show()
