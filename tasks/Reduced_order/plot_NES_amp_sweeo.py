"""
Plot amplitude sweeps for different K_c values in separate figures
Similar to experimental parametric sweep plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import os
from ROM import modal_simulation, run_time_sim, N, S, L_b
#%%
# ======= Load Main 2D Sweep Data =======
npz_path_main = 'sim_dat/2D_sweep_softening_Kp=0.030_Ki=2100.npz'
npz_path_main = 'sim_dat/NES_kp_sweep_Kc=5.00e+11.npz'
data = np.load(npz_path_main, allow_pickle=True)

amp_list = data['params']
# Kc_list = data['Kc_list']
# results_by_Kc = data['results_by_Kc'].item()  # Convert to dictionary

# print(f"Loaded data: {len(Kc_list)} K_c values, {len(amp_list)} amplitudes")
# print(f"K_c values: {Kc_list}")
print(f"Amplitude values: {amp_list}")

# Sort K_c values
kc_order = np.sort(Kc_list)

# Get a sample frequency array from first result
# sample_kc = list(results_by_Kc.keys())[0]
freq = results_by_Kc[sample_kc]['freq']

print(f"Frequency range: {freq[0]:.1f} - {freq[-1]:.1f} Hz ({len(freq)} points")

# ======= Generate Reference Data =======
K_p = 0.03
K_i = 2100
K_c = 0
t_end = 0.5
f0 = 1000
f1 = 5000
# Short circuit (K_p = 100, K_c = 0)
print("Computing short circuit reference...")
freq_SC, FRF_SC, _ = modal_simulation(K_c=0, K_p=100, K_i=K_i)

# Linear (K_c = 0, same K_p)
print("Computing linear reference...")
freq_linear, FRF_linear, _ = modal_simulation(K_c=0, K_p=K_p, K_i=K_i)


def v_exc(t, A_exc=50, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
t_eval = np.arange(0, t_end, 1/f1/20)
x_eval = np.linspace(0, L_b, 100)
res_timeDomain = run_time_sim(v_exc=v_exc, K_c=K_c, K_p=K_p, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
freq = res_timeDomain['freq']
FRF_linear_timeDomain = res_timeDomain['FRF']

res_timeDomain = run_time_sim(v_exc=v_exc, K_c=K_c, K_p=0.01, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
freq = res_timeDomain['freq']
FRF_linear_timeDomain_lowDamp = res_timeDomain['FRF']
# Linear low damping (K_c = 0, K_p = 0.01)
print("Computing linear low damping reference...")
freq_linear_lowdamp, FRF_linear_lowdamp, _ = modal_simulation(K_c=0, K_p=0.01, K_i=K_i)
#%%
# ======= Plot for Each K_c Value =======
fig_path = os.path.join(os.path.dirname(npz_path_main), 'amp_sweep_by_Kc')
os.makedirs(fig_path, exist_ok=True)

# Create colormap and amplitude-to-color mapping
# Collect all unique amplitudes across all K_c values
all_amps = []
for kc in kc_order:
    all_amps.extend(results_by_Kc[kc]["amps"])
unique_amps = np.sort(np.unique(all_amps))

# Create color mapping
cmap = cm.get_cmap("viridis", len(unique_amps))
amp_to_color = {amp: cmap(i / (len(unique_amps) - 1)) for i, amp in enumerate(unique_amps)}

for kc in kc_order:
    data_kc = results_by_Kc[kc]
    
    plt.figure(figsize=(8, 5))
    
    # Plot amplitude sweeps
    for j, amp in enumerate(data_kc["amps"]):
        color = amp_to_color[amp]
        print(j)
        FRF = data_kc["FRFs"][j]
        plt.semilogy(
            data_kc["freq"], FRF,
            '.',
            color=color,
            label=f"Amp={amp:.2f}"
        )

    # Add reference lines
    # plt.semilogy(freq_linear, FRF_linear, 'r--', label='linear', linewidth=1.5)
    # plt.semilogy(freq_linear_lowdamp, FRF_linear_lowdamp, 'c--', label='linear low damping', linewidth=1.5)
    plt.semilogy(freq_SC, FRF_SC, '--', label='Short circuit', color='purple', linewidth=1.5)
    plt.semilogy(freq, FRF_linear_timeDomain, 'r--', label='Time Domain ROM linear', linewidth=1.5)
    plt.semilogy(freq, FRF_linear_timeDomain_lowDamp, 'c--', label='Time Domain ROM linear low damping', linewidth=1.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel('Average FRF Velocity / Voltage (m/s/V)')
    plt.title(f"kc = {kc:.2e}, Kp = {K_p:.3f}")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.xlim([1200, 3200])
    plt.ylim([2e-5, 6e-4])
    # Save figure
    save_path = os.path.join(fig_path, f"kc_{kc:.2e}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

print(f"\n{'='*60}")
print(f"All plots saved to: {fig_path}")
print(f"{'='*60}")

# ======= Single Legend Figure =======
# Defensive import for notebook cell execution
import matplotlib.patches as mpatches  # noqa: E401
from matplotlib.lines import Line2D    # noqa: E401

amp_handles = [mpatches.Patch(color=amp_to_color[a], label=f"Amp={a:.2f}") for a in unique_amps]
ref_handles = [
    Line2D([], [], color='purple', linestyle='--', label='Short circuit'),
    Line2D([], [], color='r', linestyle='--', label='linear'),
    Line2D([], [], color='c', linestyle='--', label='linear low damping'),
]

fig_leg = plt.figure(figsize=(5, 3))
fig_leg.legend(handles=amp_handles + ref_handles, loc='center', ncol=2, fontsize=9, frameon=True)
plt.axis('off')
plt.tight_layout()
legend_path = os.path.join(fig_path, 'legend_amp_sweep_by_Kc.png')
plt.savefig(legend_path, dpi=300, bbox_inches='tight')
print(f"Saved legend: {legend_path}")
plt.show()
#%