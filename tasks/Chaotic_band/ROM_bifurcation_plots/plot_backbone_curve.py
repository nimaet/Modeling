"""Backbone curve: peak FRF amplitude (and its frequency) vs K_c.

Replaces a literal "bifurcation diagram" for this sweep: true Poincare
sections need a periodic response sampled at a fixed excitation period, but
this sweep's excitation is a whole-band chirp (f0=500 -> f1=3500 Hz), which
never repeats at a fixed period. What IS well-defined from chirp+FFT data is
the classic Duffing "backbone curve" -- how the resonance peak's amplitude
and frequency shift as K_c moves through the softening/hardening range --
which is also the natural quantity to compare against the FE sweep run over
the same K_c grid.

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
SWEEP_KEY = "K_c"
FREQ_BAND = (2500, 3500)  # Hz; restrict the peak search to the resonance neighborhood
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD
# =========================================================
config = load_config(RUN_DIR)
results = make_lazy_results(RUN_DIR, sweep_key=SWEEP_KEY)
print(f"Loaded {len(results)} sim(s) from {RUN_DIR}")

sweep_vals, peak_amps, peak_freqs = [], [], []
for r in results:
    freq = r["freq"]
    frf = np.abs(r["FRF"])
    band_mask = (freq >= FREQ_BAND[0]) & (freq <= FREQ_BAND[1])
    if not np.any(band_mask):
        continue
    band_freq = freq[band_mask]
    band_frf = frf[band_mask]
    peak_i = np.argmax(band_frf)

    sweep_vals.append(r[SWEEP_KEY])
    peak_amps.append(band_frf[peak_i])
    peak_freqs.append(band_freq[peak_i])

sweep_vals = np.asarray(sweep_vals)
peak_amps = np.asarray(peak_amps)
peak_freqs = np.asarray(peak_freqs)

order = np.argsort(sweep_vals)
sweep_vals, peak_amps, peak_freqs = sweep_vals[order], peak_amps[order], peak_freqs[order]

# =========================================================
# PLOT
# =========================================================
fig, (ax_amp, ax_freq) = plt.subplots(1, 2, figsize=(11, 5))

ax_amp.plot(sweep_vals, peak_amps, 'o-', color='tab:blue')
ax_amp.set_xlabel(SWEEP_KEY)
ax_amp.set_ylabel('Peak FRF magnitude')
ax_amp.set_title('Resonance peak amplitude')
ax_amp.grid(True, alpha=0.3)

ax_freq.plot(sweep_vals, peak_freqs, 'o-', color='tab:orange')
ax_freq.set_xlabel(SWEEP_KEY)
ax_freq.set_ylabel('Peak frequency [Hz]')
ax_freq.set_title('Resonance peak frequency (softening/hardening shift)')
ax_freq.grid(True, alpha=0.3)

fig.suptitle(f'Backbone curve over {SWEEP_KEY} sweep (peak search band {FREQ_BAND} Hz)')
fig.tight_layout()
fig.savefig(FIGS_DIR / "backbone_curve.png", dpi=150)
plt.show()
