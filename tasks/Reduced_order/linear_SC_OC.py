"""
Compare Short Circuit (SC) and Open Circuit (OC) conditions
Plots both experimental data and ROM simulations with color-coded circuits
and different linestyles for different data sources
"""

from ROM import modal_simulation
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd

# ======= Simulation Parameters =======
K_i = 0
K_c = 0

# Open Circuit (K_p = 0)
K_p_OC = 0.0
freq_modal_OC, vel_mag_OC, disp_mag_OC = modal_simulation(K_c=K_c, K_p=K_p_OC, K_i=K_i)

# Short Circuit (K_p = large value)
K_p_SC = 1000
freq_modal_SC, vel_mag_SC, disp_mag_SC = modal_simulation(K_c=K_c, K_p=K_p_SC, K_i=K_i)

# ======= Load Experimental Data =======
npz_path_OC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\OC.npz".replace("\\", "/")
npz_path_SC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\SC.npz".replace("\\", "/")

data_OC = np.load(npz_path_OC)
data_SC = np.load(npz_path_SC)

frq_OC_exp = data_OC['freq']
frf_data_OC_exp = data_OC['frf_data']
frq_SC_exp = data_SC['freq']
frf_data_SC_exp = data_SC['frf_data']

# ======= Load COMSOL Data =======
comsol_OC = pd.read_csv('../../comsol/OC_wide.csv')
comsol_SC = pd.read_csv('../../comsol/SC.csv')
#%%
# ======= Plotting =======
plt.figure(figsize=(12, 4))

# Color scheme: Blue for OC, Red for SC
# Linestyle: solid for ROM, dashed for Experiment, dotted for COMSOL

# --- Open Circuit (Blue) ---
# plt.semilogy(freq_modal_OC, vel_mag_OC, '.-', color='blue', linewidth=2.5, 
#              label='ROM - Open Circuit', alpha=0.9)

plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), '--', 
             color='blue', linewidth=2, label='Exp - Open Circuit', alpha=0.8)

# plt.semilogy(comsol_OC['freq'], comsol_OC['w']*2*pi*comsol_OC['freq'], ':', 
#              color='blue', linewidth=2, label='COMSOL - Open Circuit', alpha=0.7)

# --- Short Circuit (Red) ---
# plt.semilogy(freq_modal_SC, vel_mag_SC, '.-', color='red', linewidth=2.5, 
#              label='ROM - Short Circuit', alpha=0.9)

plt.semilogy(frq_SC_exp, np.mean(frf_data_SC_exp[:,:], axis=1), '--', 
             color='red', linewidth=2, label='Exp - Short Circuit', alpha=0.8)

# plt.semilogy(comsol_SC['freq'], comsol_SC['w']*2*pi*comsol_SC['freq'], ':', 
#              color='red', linewidth=2, label='COMSOL - Short Circuit', alpha=0.7)


# --- Formatting ---
plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("Average Velocity/Voltage FRF [m/s/V]", fontsize=12)
# plt.title("Short Circuit vs Open Circuit Comparison", fontsize=14, fontweight='bold')
plt.xlim([1000, 4500])
plt.ylim([4e-5, 6e-4])

plt.grid(True, which='both', alpha=0.3)
plt.legend(loc='best', fontsize=10, framealpha=0.9)




plt.show()
