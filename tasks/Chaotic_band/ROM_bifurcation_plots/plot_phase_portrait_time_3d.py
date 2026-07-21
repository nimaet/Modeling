"""3D time-series phase portrait at a single sweep point -- reproduces
plot_duffing_ROM.ipynb cells "3a6acc9d"/"774b3352" (piezo voltage v(t) laid
out across several patches, stacked along a "patch index" axis).

The sweep script saves the FULL v/t arrays for every patch (no
representative-index restriction, since nothing here is stroboscopically
trimmed), so PATCH_INDICES can be any patches. TAIL_SECONDS selects a
trailing time window to plot (the full t_end=1s chirp response is dense
enough that plotting all of it is usually too busy to read).

Assumes the sweep's npz/ + config.json have already been downloaded.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from npz_direct_helpers_ROM import load_config, make_lazy_results, nearest_result, tail_window

# =========================================================
# CONFIG
# =========================================================
RUN_DIR = Path("../../Basic_ROM_sweeps/sim_dat/ROM_Duffing_freq_sweep_REPLACE_WITH_JOB_ID")
SWEEP_KEY = "K_c"
TARGET_VALUE = 3e10        # sweep value to plot; nearest available sim is used
PATCH_INDICES = [0, 5, 10, 15, 20, 25]
TAIL_SECONDS = 0.2         # trailing window of the chirp response to plot
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
r = nearest_result(results, TARGET_VALUE, sweep_key=SWEEP_KEY)
print(f"Plotting sim with {SWEEP_KEY}={r[SWEEP_KEY]:.4e} (target was {TARGET_VALUE:.4e})")

t = r["t"]
mask = tail_window(t, TAIL_SECONDS)
t_win = t[mask]
v_win = r["v"][:, mask]  # shape (n_patches, n_samples_in_window)

# =========================================================
# PLOT
# =========================================================
fig = plt.figure(figsize=(8.5, 5.8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(np.linspace(0, 1, len(PATCH_INDICES)))

for i, patch_idx in enumerate(PATCH_INDICES):
    ax.plot(np.full_like(t_win, patch_idx), t_win, v_win[patch_idx, :], color=colors[i], lw=0.8, alpha=0.95)

ax.set_xlabel('Piezo patch index')
ax.set_xticks(PATCH_INDICES)
ax.set_ylabel('time [s]')
ax.set_zlabel('v [V]')
ax.set_title(f'{SWEEP_KEY}={r[SWEEP_KEY]:.2e} -- v(t) across patches (last {TAIL_SECONDS}s)')
ax.grid(False)
fig.tight_layout()
fig.savefig(FIGS_DIR / "phase_portrait_time_3d.png", dpi=150)
plt.show()
