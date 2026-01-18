"""
Plot amplitude sweeps for different K_c values in separate figures (FE version)
Similar to the ROM plotting script but for Finite Element results
Includes comparison with ROM and reference data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models.FE1 import PiezoBeamFE, frf_sweep
from Modeling.models.ROM import ROM

# ======= Load FE 2D Sweep Data =======
script_dir = Path(__file__).resolve().parent
npz_path_main = script_dir / 'sim_dat' / 'FE_2D_sweep_softening_Kp=0.030_Ki=2000.npz'

if not npz_path_main.exists():
    # Try alternative naming
    sim_dat_dir = script_dir / 'sim_dat'
    if sim_dat_dir.exists():
        npz_files = list(sim_dat_dir.glob('FE_2D_sweep_*.npz'))
        if npz_files:
            npz_path_main = npz_files[0]
            print(f"Found NPZ file: {npz_path_main.name}")
        else:
            print(f"No FE 2D sweep data found in {sim_dat_dir}")
            print("Please run FE_2D_sweep_amp_Kc.py first")
            exit(1)
    else:
        print(f"Directory {sim_dat_dir} not found")
        exit(1)

data = np.load(npz_path_main, allow_pickle=True)

amp_list = data['amp_list']
Kc_list = data['Kc_list']
results_by_Kc = data['results_by_Kc'].item()  # Convert to dictionary

print(f"Loaded FE data from: {npz_path_main.name}")
print(f"  {len(Kc_list)} K_c values, {len(amp_list)} amplitudes")
print(f"  K_c values: {Kc_list}")
print(f"  Amplitude values: {amp_list}")

# Sort K_c values
kc_order = np.sort(Kc_list)

# Get sample frequency array from first result
sample_kc = list(results_by_Kc.keys())[0]
freq = results_by_Kc[sample_kc]['freq']
print(f"  Frequency range: {freq[0]:.1f} - {freq[-1]:.1f} Hz ({len(freq)} points)")

# ======= FE Reference Data (Frequency Domain) =======
print("\nComputing FE reference responses (frequency domain)...")
K_p = float(data['K_p'])
K_i = float(data['K_i'])

params_fe = PiezoBeamParams(hp=0.252e-3, hs=0.51e-3)
params_fe.zeta_p = 0.0151 * 8
params_fe.zeta_q = 0.0392 * 10

fe = PiezoBeamFE(params_fe, n_el_gap=1, n_el_patch=3)

# Linear case (K_c = 0)
def v_exc_ref(t, A_exc=1.0):
    return A_exc

ode_linear = fe.build_ode_system(
    j_exc=30,
    K_c=0,
    K_i=K_i,
    K_p=K_p,
    R_c=1e3,
    v_exc=v_exc_ref
)

f_fe_ref = np.linspace(1000, 3000, 500)
frf_linear_fd = frf_sweep(ode_linear, f_fe_ref * 2 * np.pi)
FRF_FE_linear = np.mean(np.abs(frf_linear_fd['u_dot']), axis=1)

print(f"  FE linear (K_c=0, K_p={K_p})")

# Short Circuit (very high K_p)
ode_sc = fe.build_ode_system(
    j_exc=30,
    K_c=0,
    K_i=K_i,
    K_p=100,
    R_c=1e3,
    v_exc=v_exc_ref
)

frf_sc_fd = frf_sweep(ode_sc, f_fe_ref * 2 * np.pi)
FRF_FE_SC = np.mean(np.abs(frf_sc_fd['u_dot']), axis=1)

print(f"  FE short circuit (K_c=0, K_p=100)")

# ======= ROM Reference Data (if available) =======
try:
    # Try to load ROM reference data
    rom = ROM(PiezoBeamParams(hp=0.252e-3, hs=0.51e-3), N=40)
    freq_rom, FRF_rom_linear, _ = rom.modal_simulation(K_c=0, K_p=K_p, K_i=K_i)
    freq_rom_sc, FRF_rom_sc, _ = rom.modal_simulation(K_c=0, K_p=100, K_i=K_i)
    have_rom = True
    print(f"  ROM linear (K_c=0, K_p={K_p})")
    print(f"  ROM short circuit (K_c=0, K_p=100)")
except Exception as e:
    have_rom = False
    print(f"  ROM data not available: {e}")

# ======= Create Output Directory =======
fig_path = script_dir / 'sim_dat' / 'FE_amp_sweep_by_Kc'
fig_path.mkdir(parents=True, exist_ok=True)

# ======= Plot for Each K_c Value =======
print(f"\nGenerating plots for each K_c value...")

# Collect all unique amplitudes across all K_c values for consistent coloring
all_amps = []
for kc in kc_order:
    all_amps.extend(results_by_Kc[kc]["amps"])
unique_amps = np.sort(np.unique(all_amps))

# Create color mapping
cmap = cm.get_cmap("viridis", len(unique_amps))
amp_to_color = {amp: cmap(i / (len(unique_amps) - 1)) for i, amp in enumerate(unique_amps)}

for idx, kc in enumerate(kc_order):
    data_kc = results_by_Kc[kc]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot amplitude sweeps
    for j, amp in enumerate(data_kc["amps"]):
        color = amp_to_color[amp]
        FRF = data_kc["FRFs"][j]
        ax.semilogy(
            data_kc["freq"], FRF,
            '.-',
            color=color,
            linewidth=2,
            markersize=4,
            label=f"Amp={amp:.1f}V"
        )
    
    # Add FE reference lines
    ax.semilogy(f_fe_ref, FRF_FE_linear, '--', color='red', linewidth=2.5, 
                label=f'FE linear (K_p={K_p})', alpha=0.8)
    ax.semilogy(f_fe_ref, FRF_FE_SC, '--', color='purple', linewidth=2.5, 
                label='FE short circuit (K_p=100)', alpha=0.8)
    
    # Add ROM reference if available
    if have_rom:
        ax.semilogy(freq_rom, FRF_rom_linear, ':', color='darkred', linewidth=2, 
                    label=f'ROM linear (K_p={K_p})', alpha=0.7)
        ax.semilogy(freq_rom_sc, FRF_rom_sc, ':', color='darkviolet', linewidth=2, 
                    label='ROM short circuit (K_p=100)', alpha=0.7)
    
    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel('FRF Magnitude (m/s/V)', fontsize=12)
    ax.set_title(f"FE Amplitude Sweep: K_c = {kc:.2e}", fontsize=13, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='best', framealpha=0.95)
    ax.set_xlim([1000, 3200])
    ax.set_ylim([1e-5, 1e-3])
    
    plt.tight_layout()
    
    # Save figure
    save_path = fig_path / f"FE_kc_{kc:.2e}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path.name}")
    plt.close(fig)

print(f"\nAll amplitude sweep plots saved to: {fig_path}/")

# ======= Create Legend Figure =======
print("Creating legend figure...")

amp_handles = [mpatches.Patch(color=amp_to_color[a], label=f"Amp={a:.1f}V") 
               for a in unique_amps]

ref_handles = [
    Line2D([], [], color='red', linestyle='--', linewidth=2.5, label=f'FE linear (K_p={K_p})'),
    Line2D([], [], color='purple', linestyle='--', linewidth=2.5, label='FE short circuit (K_p=100)'),
]

if have_rom:
    ref_handles.extend([
        Line2D([], [], color='darkred', linestyle=':', linewidth=2, label=f'ROM linear (K_p={K_p})'),
        Line2D([], [], color='darkviolet', linestyle=':', linewidth=2, label='ROM short circuit (K_p=100)'),
    ])

fig_leg = plt.figure(figsize=(10, 8))
fig_leg.legend(handles=amp_handles + ref_handles, loc='center', ncol=2, fontsize=11, 
               frameon=True, title='FE 2D Amplitude & K_c Sweep', title_fontsize=12)
plt.axis('off')
plt.tight_layout()

legend_path = fig_path / 'FE_legend_amp_sweep_by_Kc.png'
plt.savefig(legend_path, dpi=300, bbox_inches='tight')
print(f"Saved legend: {legend_path.name}")
plt.close(fig_leg)

# ======= Summary Statistics =======
print("\n" + "="*70)
print("FE AMPLITUDE SWEEP PLOT SUMMARY")
print("="*70)
print(f"Number of K_c values: {len(kc_order)}")
print(f"Number of amplitude values: {len(unique_amps)}")
print(f"Total plots generated: {len(kc_order)}")
print(f"Output directory: {fig_path}/")
print("="*70)

# ======= Optional: Comparison Table =======
print("\nFrequency shift analysis (peak frequency vs amplitude):")
print("-" * 70)
for kc in kc_order[:3]:  # Show first 3 K_c values as example
    data_kc = results_by_Kc[kc]
    print(f"\nK_c = {kc:.2e}:")
    for j, amp in enumerate(data_kc["amps"][:3]):  # First 3 amplitudes
        FRF = data_kc["FRFs"][j]
        freq_local = data_kc["freq"]
        # Find peak frequency (in frequency range of interest)
        mask = (freq_local >= 1200) & (freq_local <= 3000)
        if np.any(mask):
            peak_idx = np.argmax(FRF[mask])
            peak_freq = freq_local[np.where(mask)[0][peak_idx]]
            peak_val = FRF[np.where(mask)[0][peak_idx]]
            print(f"  Amp={amp:.1f}V:  peak at {peak_freq:.0f} Hz, magnitude={peak_val:.2e}")

print("\n" + "="*70)
