"""Animated grid of 2D phase portraits (one panel per piezo patch), stepping
through K_c sweep points frame by frame -- reproduces plot_duffing_ROM.ipynb
cell "36f9d799", minus its Poincare overlay (the chirp excitation has no
fixed period to stroboscopically sample from -- see
plot_phase_portrait_grid_patch.py's docstring).

Uses the full z/v arrays (saved for every patch, no representative-index
restriction), sliced to a trailing TAIL_SECONDS window per frame.

Assumes the sweep's npz/ + config.json have already been downloaded.
"""
from math import ceil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from npz_direct_helpers_ROM import load_config, make_lazy_results, tail_window

# =========================================================
# CONFIG
# =========================================================
RUN_DIR = Path("../../Basic_ROM_sweeps/sim_dat/ROM_Duffing_freq_sweep_REPLACE_WITH_JOB_ID")
SWEEP_KEY = "K_c"
PATCH_INDICES = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28]
TAIL_SECONDS = 0.2
NCOLS = 4
INTERVAL_MS = 400
SAVE_PATH = Path("figs") / "animated_phase_portrait_grid.gif"

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
print(f"Animating {len(results)} sweep frame(s) for patches {PATCH_INDICES}")

# Fix axis limits across all frames using the full sweep's range, so the
# animation doesn't rescale from frame to frame.
def _windowed(r, key):
    mask = tail_window(r["t"], TAIL_SECONDS)
    return r[key][PATCH_INDICES, :][:, mask]


all_z = np.concatenate([_windowed(r, "z").ravel() for r in results])
all_v = np.concatenate([_windowed(r, "v").ravel() for r in results])
z_pad = 0.05 * (np.nanmax(all_z) - np.nanmin(all_z) + 1e-12)
v_pad = 0.05 * (np.nanmax(all_v) - np.nanmin(all_v) + 1e-12)

# =========================================================
# PLOT / ANIMATE
# =========================================================
n = len(PATCH_INDICES)
nrows = ceil(n / NCOLS)
fig, axes = plt.subplots(nrows, NCOLS, figsize=(3 * NCOLS, 3 * nrows))
axes = np.atleast_1d(axes).flatten()

scatters = []
for i, patch_idx in enumerate(PATCH_INDICES):
    ax = axes[i]
    ax.set_xlabel('flux linkage z [V.s]')
    ax.set_ylabel('voltage v [V]')
    ax.set_title(f'Patch {patch_idx}')
    ax.set_xlim(np.nanmin(all_z) - z_pad, np.nanmax(all_z) + z_pad)
    ax.set_ylim(np.nanmin(all_v) - v_pad, np.nanmax(all_v) + v_pad)
    ax.grid(True, alpha=0.25)
    scatters.append(ax.scatter([], [], s=4))

for j in range(n, len(axes)):
    axes[j].axis('off')


def update(frame_i):
    r = results[frame_i]
    mask = tail_window(r["t"], TAIL_SECONDS)
    for sc, patch_idx in zip(scatters, PATCH_INDICES):
        z = r["z"][patch_idx, mask]
        v = r["v"][patch_idx, mask]
        sc.set_offsets(np.column_stack([z, v]))
    fig.suptitle(f'{SWEEP_KEY} = {r[SWEEP_KEY]:.2e}')
    return scatters


anim = FuncAnimation(fig, update, frames=len(results), interval=INTERVAL_MS, blit=False, repeat=True)
plt.tight_layout()

SAVE_PATH.parent.mkdir(exist_ok=True)
fps = max(1, int(round(1000 / INTERVAL_MS)))
anim.save(str(SAVE_PATH), writer='pillow', fps=fps)
print(f"Saved animation to {SAVE_PATH}")
