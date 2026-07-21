"""Grid of 2D phase portraits, one panel per sweep point, for a single piezo
patch -- reproduces plot_duffing_ROM.ipynb cell "77cd9e38" (z vs v), minus
its Poincare overlay.

The chirp excitation (f0=500 -> f1=3500 Hz over t_end=1s) never repeats at a
fixed period, so there's no well-defined stroboscopic sample rate to build a
Poincare section from -- only the continuous (z, v) trajectory is plotted,
over a trailing TAIL_SECONDS window (the response is dense enough that the
whole 1s trace is usually too busy to read).

PATCH_INDEX can be any patch -- the sweep script saves full z/v arrays for
every patch, with no representative-index restriction.

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
PATCH_INDEX = 10
TAIL_SECONDS = 0.2
SWEEP_SLICE = slice(0, 40)  # which sweep points to show, in ascending sweep-value order
NCOLS = 8
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
picked = results[SWEEP_SLICE]
print(f"Plotting {len(picked)}/{len(results)} sim(s) for patch {PATCH_INDEX}")

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
    z = r["z"][PATCH_INDEX, mask]
    v = r["v"][PATCH_INDEX, mask]

    ax.scatter(z, v, s=0.1, color='green', alpha=0.9)
    ax.set_title(f'{r[SWEEP_KEY]:.2e}', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)

for j in range(len(picked), nrows * NCOLS):
    axes[j // NCOLS, j % NCOLS].axis('off')

fig.suptitle(f'Phase portraits for patch {PATCH_INDEX} across {SWEEP_KEY} (last {TAIL_SECONDS}s)', fontsize=14)
fig.tight_layout()
fig.savefig(FIGS_DIR / "phase_portrait_grid_patch.png", dpi=150)
plt.show()
