"""FRF overlay across a handful of K_c sweep points -- reproduces
plot_duffing_ROM.ipynb's frequency-response cell (and the amp-sweep FRF
overlay), for the ROM's Kc/chirp sweep (Basic_ROM_sweeps/
generic_nD_sweep_SLURMarray_ROM.py).

Each sim's freq/FRF come from the FFT of its whole-band chirp response
(f0=500 -> f1=3500 Hz over t_end=1s), so this is a genuine frequency-response
curve per sim (unlike a constant-frequency-locked sweep) -- directly
comparable to the FE sweep's equivalent plot for the same K_c grid.

Assumes the sweep's npz/ + config.json have already been downloaded.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from npz_direct_helpers_ROM import load_config, make_lazy_results

# =========================================================
# CONFIG
# =========================================================
RUN_DIR = Path("../../Basic_ROM_sweeps/sim_dat/ROM_Duffing_freq_sweep_REPLACE_WITH_JOB_ID")
SWEEP_KEY = "K_c"          # must match the SweepParam key actually used for this run
N_REPRESENTATIVE = 8       # how many evenly-spaced sweep points to overlay
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
print(f"Loaded {len(results)} sim(s) from {RUN_DIR}")

pick_idx = np.linspace(0, len(results) - 1, min(N_REPRESENTATIVE, len(results))).round().astype(int)
picked = [results[i] for i in np.unique(pick_idx)]

# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(picked)))

for r, color in zip(picked, colors):
    freq = r["freq"]
    frf = np.abs(r["FRF"])
    ax.semilogy(freq, frf, color=color, alpha=0.9, label=f"{SWEEP_KEY}={r[SWEEP_KEY]:.2e}")

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("FRF magnitude (mean |Y| / |X|)")
ax.set_title(f"Frequency response across {SWEEP_KEY} sweep")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIGS_DIR / "spectral_content.png", dpi=150)
plt.show()
