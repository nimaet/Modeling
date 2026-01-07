"""
3D Surface Plot: K_c vs Frequency vs FRF Magnitude
Plots FRF response as a 3D surface where:
- X-axis: K_c (Duffing coefficient)
- Y-axis: Frequency [Hz]
- Z-axis: FRF Magnitude
- Color: Amplitude (excitation amplitude)
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
#%%
# Load results from 2D sweep
print("Loading 2D sweep results...")
data = np.load('sim_dat/2D_sweep_Kp=0.030_Ki=2100.npz', allow_pickle=True)

results_by_Kc = data['results_by_Kc'].item()
amp_list = data['amp_list']
Kc_list = data['Kc_list']

print(f"Loaded {len(results_by_Kc)} K_c values")
print(f"Amplitudes: {amp_list}")
print(f"K_c values: {Kc_list}")

# ======= Prepare data for 3D plotting =======
Kc_values = sorted(results_by_Kc.keys())
freq = results_by_Kc[Kc_values[0]]["freq"]

# Filter to positive frequencies and frequency range of interest
freq_range = (1200, 3200)
freq_idx = np.where((freq >= freq_range[0]) & (freq <= freq_range[1]))[0]
freq_plot = freq[freq_idx]

# Create equidistant K_c positions (indices) regardless of actual K_c values
Kc_positions = np.arange(len(Kc_values))  # 0, 1, 2, 3, ...
Kc_labels = [f'{kc:.0e}' for kc in Kc_values]  # Labels with actual values

# Create meshgrid with equidistant K_c positions
Kc_mesh, Freq_mesh = np.meshgrid(Kc_positions, freq_plot)

# ======= Single unified 3D plot with color for amplitude =======
#%%
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors_amp = cm.get_cmap('viridis')(np.linspace(0, 1, len(amp_list)))

for amp_idx, amp in enumerate(amp_list):
	for kc_idx, kc in enumerate(Kc_values):
		data_kc = results_by_Kc[kc]
		if amp in data_kc["amps"]:
			local_amp_idx = data_kc["amps"].index(amp)
			FRF_full = data_kc["FRFs"][local_amp_idx]
			FRF_plot = FRF_full[freq_idx]
			
			# Plot line for this K_c, amplitude combination (using equidistant position)
			ax.plot(freq_plot, np.full_like(freq_plot, Kc_positions[kc_idx]-1), FRF_plot,
			        color=colors_amp[amp_idx], linewidth=1.5, alpha=0.7,
			        label=f'A={amp:.1f}V' if kc_idx == 0 else "")

ax.set_xlabel('Frequency [Hz]', fontsize=12, fontweight='bold')
ax.set_ylabel('Kc ', fontsize=12, fontweight='bold', labelpad=20)
ax.set_yticks(Kc_positions)
ax.set_yticklabels(Kc_labels, fontsize=10, rotation=45)
# ax.set_zlabel('FRF Magnitude', fontsize=12, fontweight='bold')
ax.set_zlim([1e-5, 8e-4])
ax.set_zscale('log')
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(False)
# Remove pane edges
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')
# Hide y and z axes by setting line width to 0
ax.yaxis.line.set_linewidth(0)
ax.zaxis.line.set_linewidth(0)
# ax.xaxis.line.set_linewidth(0)
# ax.yaxis.label.set_visible(False)
# ax.zaxis.label.set_visible(False)
# ax.set_yticks([])
ax.set_zticks([])
# ax.zaxis.set_tick_params(labelbottom=False, labelleft=False, labelright=False)
# Also hide the tick lines themselves
# ax.zaxis._axinfo['juggled'] = (1, 2, 0)
# for line in ax.zaxis.get_ticklines():
# 	line.set_visible(False)
# ax.set_title('3D Response Surface: Frequency × K_c × FRF (colored by Amplitude)', 
#              fontsize=13, fontweight='bold', pad=20)
ax.view_init(elev=15, azim=80)

# Hide gray background panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# Hide y and z axes

plt.savefig('sim_dat/3D_unified_surface_Kc_freq_amp.png', dpi=300, bbox_inches='tight')
print("Saved: sim_dat/3D_unified_surface_Kc_freq_amp.png")
plt.show()

# Separate legend figure
fig_leg = plt.figure(figsize=(4, 3))
handles = [mpatches.Patch(color=colors_amp[i], label=f'A={amp_list[i]:.1f}V') 
          for i in range(len(amp_list))]
fig_leg.legend(handles=handles, loc='center', fontsize=11, frameon=True)
plt.axis('off')
plt.tight_layout()
plt.savefig('sim_dat/legend_Kc_freq_amp.png', dpi=300, bbox_inches='tight')
print("Saved: sim_dat/legend_Kc_freq_amp.png")
plt.show()

print("\n" + "="*60)
print("3D PLOTTING COMPLETE")
print("="*60)
