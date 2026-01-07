# %%
import numpy as np
import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models.FE1 import PiezoBeamFE, frf_sweep
import matplotlib.pyplot as plt
from Modeling.models.ROM import ROM
from numpy import pi	
from IPython.display import clear_output
from matplotlib import cm, colors
import pandas as pd


# %%


# %%
# # ======= K_p sweep in frequency domain =======
K_i = 2000; K_p = 0.02; K_c = 0
t_end = 0.01
f0 = 1000
f1 = 4500
N = 40

params = PiezoBeamParams()
hp, hs = 0.252e-3, 0.51e-3 		
# hp, hs = 0.31e-3, 0.607e-3 		
params.hp = hp; params.hs = hs
params.zeta_p = 0.0151*2*2*1.2
params.zeta_q = 0.0392*2*2*1.6


# %%
# ============================================================
# Convergence Study: n_el_gap and n_el_patch
# ============================================================

# Define parameter ranges for convergence study
n_el_gap_values = [2, 3, 4, 5, 6, 8, 10, 12]
n_el_patch_values = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25]

# Number of modes to compare


# Reference frequencies (from COMSOL)
# f_ref = np.array([
#     5.3055, 33.242, 93.066, 182.36, 301.45, 450.33, 629.03, 837.59,
#     1076.0, 1344.4
# ])
f_ref = np.array([
	5.3055, 33.242, 93.066, 182.36, 301.45, 450.33, 629.03, 837.59,
	1076.0, 1344.4, 1642.8, 1971.1, 2329.5, 2717.8, 3136.2,
	3584.5, 4062.8, 4570.8, 5108.4, 5675.4, 6271.4, 6895.9, 7141.3
])
Nmodes_conv = f_ref.shape[0]
# Storage for convergence results
conv_results_gap = {}
conv_results_patch = {}

# ----------------------------------------
# Study 1: Vary n_el_gap (fix n_el_patch=6)
# ----------------------------------------
print("Computing convergence for n_el_gap...")
n_el_patch_fixed = 6

for n_gap in n_el_gap_values:
    fe_temp = PiezoBeamFE(params, n_el_gap=n_gap, n_el_patch=n_el_patch_fixed)
    freq_temp, _, _ = fe_temp.eigen_analysis()
    f_fe_temp = freq_temp[:Nmodes_conv]
    
    # Compute relative errors
    rel_error = np.abs((f_fe_temp - f_ref) / f_ref) * 100  # percentage error
    conv_results_gap[n_gap] = {
        'frequencies': f_fe_temp,
        'rel_error': rel_error,
        'mean_error': np.mean(rel_error),
        'max_error': np.max(rel_error)
    }
    print(f"  n_el_gap={n_gap:2d}: mean error = {np.mean(rel_error):.4f}%, max error = {np.max(rel_error):.4f}%")

# ----------------------------------------
# Study 2: Vary n_el_patch (fix n_el_gap=6)
# ----------------------------------------
print("\nComputing convergence for n_el_patch...")
n_el_gap_fixed = 6

for n_patch in n_el_patch_values:
    fe_temp = PiezoBeamFE(params, n_el_gap=n_el_gap_fixed, n_el_patch=n_patch)
    freq_temp, _, _ = fe_temp.eigen_analysis()
    f_fe_temp = freq_temp[:Nmodes_conv]
    
    # Compute relative errors
    rel_error = np.abs((f_fe_temp - f_ref) / f_ref) * 100  # percentage error
    conv_results_patch[n_patch] = {
        'frequencies': f_fe_temp,
        'rel_error': rel_error,
        'mean_error': np.mean(rel_error),
        'max_error': np.max(rel_error)
    }
    print(f"  n_el_patch={n_patch:2d}: mean error = {np.mean(rel_error):.4f}%, max error = {np.max(rel_error):.4f}%")

print("\nConvergence study complete!")


# %%
# ============================================================
# Visualization of Convergence Results
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ----------------------------------------
# Plot 1: Mean error vs n_el_gap
# ----------------------------------------
ax = axes[0, 0]
mean_errors_gap = [conv_results_gap[n]['mean_error'] for n in n_el_gap_values]
ax.semilogy(n_el_gap_values, mean_errors_gap, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('n_el_gap', fontsize=12)
ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
ax.set_title(f'Convergence: n_el_gap (n_el_patch={n_el_patch_fixed})', fontsize=13)
ax.grid(True, which='both', alpha=0.3)

# ----------------------------------------
# Plot 2: Mean error vs n_el_patch
# ----------------------------------------
ax = axes[0, 1]
mean_errors_patch = [conv_results_patch[n]['mean_error'] for n in n_el_patch_values]
ax.semilogy(n_el_patch_values, mean_errors_patch, 's-', linewidth=2, markersize=8, color='C1')
ax.set_xlabel('n_el_patch', fontsize=12)
ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
ax.set_title(f'Convergence: n_el_patch (n_el_gap={n_el_gap_fixed})', fontsize=13)
ax.grid(True, which='both', alpha=0.3)

# ----------------------------------------
# Plot 3: Error per mode (n_el_gap)
# ----------------------------------------
ax = axes[1, 0]
for n_gap in [n_el_gap_values[0], n_el_gap_values[len(n_el_gap_values)//2], n_el_gap_values[-1]]:
    rel_err = conv_results_gap[n_gap]['rel_error']
    ax.semilogy(range(1, Nmodes_conv+1), rel_err, 'o-', label=f'n_el_gap={n_gap}', markersize=6)
ax.set_xlabel('Mode Number', fontsize=12)
ax.set_ylabel('Relative Error [%]', fontsize=12)
ax.set_title('Error per Mode (varying n_el_gap)', fontsize=13)
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# ----------------------------------------
# Plot 4: Error per mode (n_el_patch)
# ----------------------------------------
ax = axes[1, 1]
for n_patch in [n_el_patch_values[0], n_el_patch_values[len(n_el_patch_values)//2], n_el_patch_values[-1]]:
    rel_err = conv_results_patch[n_patch]['rel_error']
    ax.semilogy(range(1, Nmodes_conv+1), rel_err, 's-', label=f'n_el_patch={n_patch}', markersize=6)
ax.set_xlabel('Mode Number', fontsize=12)
ax.set_ylabel('Relative Error [%]', fontsize=12)
ax.set_title('Error per Mode (varying n_el_patch)', fontsize=13)
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------
# Summary table
# ----------------------------------------
print("\n" + "="*70)
print("CONVERGENCE STUDY SUMMARY")
print("="*70)
print("\nn_el_gap Convergence (n_el_patch={}):".format(n_el_patch_fixed))
print(f"{'n_el_gap':<10} {'Mean Error [%]':<18} {'Max Error [%]':<15}")
print("-"*45)
for n_gap in n_el_gap_values:
    mean_err = conv_results_gap[n_gap]['mean_error']
    max_err = conv_results_gap[n_gap]['max_error']
    print(f"{n_gap:<10} {mean_err:<18.4f} {max_err:<15.4f}")

print("\nn_el_patch Convergence (n_el_gap={}):".format(n_el_gap_fixed))
print(f"{'n_el_patch':<10} {'Mean Error [%]':<18} {'Max Error [%]':<15}")
print("-"*45)
for n_patch in n_el_patch_values:
    mean_err = conv_results_patch[n_patch]['mean_error']
    max_err = conv_results_patch[n_patch]['max_error']
    print(f"{n_patch:<10} {mean_err:<18.4f} {max_err:<15.4f}")


# %%
# ============================================================
# 2D Convergence Study: Both n_el_gap and n_el_patch
# ============================================================

# Coarser grid for 2D study to manage computation time
n_el_gap_2d = [1, 2, 3, 4, 6, 8, 10]
n_el_patch_2d = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40]


# %%
# ============================================================
# 2D Convergence Study: Both n_el_gap and n_el_patch (PARALLEL)
# ============================================================

from joblib import Parallel, delayed

# Coarser grid for 2D study to manage computation time

# Define worker function for parallel processing
def compute_convergence(i, j, n_gap, n_patch, params_dict, Nmodes_conv, f_ref):
    """Worker function to compute FE eigenfrequencies for given mesh parameters"""
    
    # Recreate params object from dictionary
    params = PiezoBeamParams()
    for key, value in params_dict.items():
        setattr(params, key, value)
    
    # Create FE model and compute eigenfrequencies
    fe_temp = PiezoBeamFE(params, n_el_gap=n_gap, n_el_patch=n_patch)
    freq_temp, _, _ = fe_temp.eigen_analysis()
    f_fe_temp = freq_temp[:Nmodes_conv]
    
    # Compute relative errors
    rel_error = np.abs((f_fe_temp - f_ref) / f_ref) * 100
    mean_error = np.mean(rel_error)
    max_error = np.max(rel_error)
    
    return (i, j, n_gap, n_patch, mean_error, max_error)

# Convert params to dict for serialization
params_dict = vars(params)

# Storage for 2D convergence results
conv_2d_mean = np.zeros((len(n_el_gap_2d), len(n_el_patch_2d)))
conv_2d_max = np.zeros((len(n_el_gap_2d), len(n_el_patch_2d)))

# Create list of all parameter combinations
param_combinations = []
for i, n_gap in enumerate(n_el_gap_2d):
    for j, n_patch in enumerate(n_el_patch_2d):
        param_combinations.append((i, j, n_gap, n_patch))

total_tasks = len(param_combinations)
print(f"Computing 2D convergence study...")
print(f"Total combinations: {total_tasks}")

# Parallel computation with live progress
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_convergence)(i, j, n_gap, n_patch, params_dict, Nmodes_conv, f_ref)
    for i, j, n_gap, n_patch in param_combinations
)

# Store results
for i, j, n_gap, n_patch, mean_error, max_error in results:
    conv_2d_mean[i, j] = mean_error
    conv_2d_max[i, j] = max_error

# ----------------------------------------
# Visualization: 2D heatmaps
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Mean error heatmap
ax = axes[0]
im1 = ax.imshow(conv_2d_mean, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(n_el_patch_2d)))
ax.set_yticks(range(len(n_el_gap_2d)))
ax.set_xticklabels(n_el_patch_2d)
ax.set_yticklabels(n_el_gap_2d)
ax.set_xlabel('n_el_patch', fontsize=12)
ax.set_ylabel('n_el_gap', fontsize=12)
ax.set_title('Mean Relative Error [%]', fontsize=13)
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('Error [%]', fontsize=11)

# Add text annotations
for i in range(len(n_el_gap_2d)):
    for j in range(len(n_el_patch_2d)):
        text = ax.text(j, i, f'{conv_2d_mean[i, j]:.2f}',
                      ha="center", va="center", color="w", fontsize=9)

# Plot 2: Max error heatmap
ax = axes[1]
im2 = ax.imshow(conv_2d_max, cmap='plasma', aspect='auto', origin='lower')
ax.set_xticks(range(len(n_el_patch_2d)))
ax.set_yticks(range(len(n_el_gap_2d)))
ax.set_xticklabels(n_el_patch_2d)
ax.set_yticklabels(n_el_gap_2d)
ax.set_xlabel('n_el_patch', fontsize=12)
ax.set_ylabel('n_el_gap', fontsize=12)
ax.set_title('Maximum Relative Error [%]', fontsize=13)
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('Error [%]', fontsize=11)

# Add text annotations
for i in range(len(n_el_gap_2d)):
    for j in range(len(n_el_patch_2d)):
        text = ax.text(j, i, f'{conv_2d_max[i, j]:.2f}',
                      ha="center", va="center", color="w", fontsize=9)

plt.tight_layout()
plt.show()

print("\n2D convergence study complete!")

# %%
