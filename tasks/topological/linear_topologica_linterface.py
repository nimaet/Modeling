from pathlib import Path
import sys
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from Modeling.models.beam_properties import PiezoBeamParams
import importlib
if 'ROM' in dir():
	importlib.reload(sys.modules['ROM'])
from Modeling.models.ROM import ROM
import numpy as np
from numpy import pi	
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm, colors
import pandas as pd
#%%

K_i = 0; K_p = 0.02; K_c = 0
t_end = 0.1
f0 = 1000; f1 = 8000
N = 40
j_exc = 30

def v_exc(t, A_exc=50, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
params = PiezoBeamParams()
params.zeta_p = 0.0151/5
params.zeta_q = 0.0392/5
rom = ROM(params, N=N)
interfce_idx = 15
beta = 0.25
# # ======= K_p sweep in frequency domain =======

ki0 = 4000
ki2 = ki0 / (1 - beta)**2 
ki1 = ki0 / (1 + beta)**2 
S = rom.S
K_i = np.zeros(S)
for i in range(S):
	if i < interfce_idx:
		K_i[i] = ki1 if i % 2 == 0 else ki2
	else:
		K_i[i] = ki2 if i % 2 == 0 else ki1
print("K_i:", K_i)
#%%
# x_eval = np.linspace(0, rom.p.L_b, 1000)
# x0 = np.zeros(2*N + 2*S)


t_eval = np.arange(0, t_end, 1/f1/20)
x_eval = np.linspace(0, rom.p.L_b, 500)

results = rom.run_time_sim(v_exc=v_exc, j_exc=j_exc, K_c=K_c, K_p=K_p, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
t = results['t']
veloc = results['veloc']
freq = results['freq']
Y = results['Y']
X = results['X']
FRF = results['FRF']

freq_modal, vel_mag, disp_mag, veloc_frq = rom.frequency_response(j_exc=j_exc, K_c=K_c, K_p=K_p, K_i=K_i,
										w=np.linspace(f0, f1, 200)*2*np.pi, x_eval=x_eval)

# Load experimental data
npz_path_OC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\OC.npz".replace("\\", "/")
# npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\10\ki0_kc0_kpSweep\parametric_sweep.npz".replace("\\", "/")  
npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\linear.npz".replace("\\", "/")  
# npz_path_OC =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\OC.npz".replace("\\", "/")
npz_path_SC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\SC.npz".replace("\\", "/")
data_SC = np.load(npz_path_SC)
data_OC = np.load(npz_path_OC)
data_linear = np.load(npz_path_linear)

frq_OC_exp = data_OC['freq']			# (Nfreq,)
frf_data_OC_exp = data_OC['frf_data']	# (Nfiles, Nfreq, Npoints)
frq_SC_exp = data_SC['freq']            # (Nfreq,)
frf_data_SC_exp = data_SC['frf_data']   # (Nfiles, Nfreq, Npoints)
frq_linear_exp = data_linear['freq']            # (Nfreq,)
frf_data_linear_exp = data_linear['frf_data']

comsol_OC = pd.read_csv('../../comsol/OC.csv')
#%%
plt.figure(figsize=(12, 4))
# plt.semilogy(frq_linear_exp, np.mean(frf_data_linear_exp[:,:], axis=1), 'r--', label=' Experiment')
# plt.semilogy(comsol_OC['freq'], comsol_OC['w']*2*pi*comsol_OC['freq'], 'g-', label='COMSOL ')
# plt.semilogy(comsol_OC['freq'], comsol_OC['w'], 'g-', label='COMSOL displacement FRF')
# plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Open circuit Exp.')
# plt.semilogy(frq_SC_exp, np.mean(frf_data_SC_exp[:,:], axis=1), 'b--', label=f'Short circuit Exp.')
# plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order'   )
plt.semilogy(freq, FRF, '.-', linewidth=1.5, label='Time Domain ROM')
plt.semilogy(freq_modal, vel_mag, '.-', label='Frequency Domain ROM'   )
# plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Experiment')
# plt.semilogy(frq_linear, np.mean(frf_data_linear[:,:], axis=1), 'r--', label=' Exp.')

# plt.semilogy(freq_modal, disp_mag*freq_modal*2*np.pi, '-', label='Modal Reduced Order Displacement $j \omega$'   )
plt.legend()
# plt.xlim([1300, 3000])
plt.xlim([f0, f1])
# plt.ylim([1e-5, 1e-3])
# plt.ylim([3e-5, 6e-4])
plt.xlabel("Frequency [Hz]")
plt.ylabel("AverageVelocity/Voltage FRF")
plt.grid(True)
plt.show()





# plt.figure()

# # plt.plot(t, disp[10,:], '.-', label='Disp.')
# plt.plot(t, veloc[10,:], '-', label='Velocity')
# # plt.xlim([0, 0.05 ])
# # plt.ylim(np.array([-1,1])*1e-3)

# plt.xlabel("t [s]")
# # plt.ylabel("displacement")
# plt.legend()
# plt.grid(True)
#%%
# =================== Spectral analysis: v(x, freq) heatmap ===================
# Perform FFT along time axis to get frequency content at each spatial location
dt = t[1] - t[0]  # time step
fs = 1.0 / dt  # sampling frequency

# FFT along time axis (axis=1)
V_freq = np.fft.fft(veloc, axis=1)

# Frequency vector
freq_fft = np.fft.fftfreq(len(t), d=dt)

# Keep only positive frequencies
idx_pos = freq_fft >= 0
freq_fft_pos = freq_fft[idx_pos]
V_freq_pos = V_freq[:, idx_pos]

# Magnitude spectrum
V_mag = np.abs(V_freq_pos)

# Plot heatmap with frequency on y-axis
plt.figure(figsize=(10, 6))
plt.pcolormesh(x_eval, freq_fft_pos, V_mag.T, shading='auto', cmap='viridis')
plt.colorbar(label='|V(x, f)| [m/s]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Position along beam [m]')
plt.title('Spatial-Frequency Velocity Spectrum')
plt.ylim([0, f1])  # Adjust frequency range as needed
plt.tight_layout()
plt.show()
# %%
# =================== Mode shape at single frequency ===================
single_freq = 3250  # Hz - choose your frequency of interest
single_w = single_freq * 2 * np.pi
x_eval_mode = np.linspace(0, rom.p.L_b, 500)

# Get frequency response at single frequency
freq_single, vel_mag_single, disp_mag_single, veloc_frq = rom.frequency_response(
    K_c=K_c, K_p=K_p, K_i=K_i, w=np.array([single_w]), x_eval=x_eval_mode
)

# veloc_frq has shape (n_spatial_points, 1) - extract the mode shape
mode_shape_vel = np.abs(veloc_frq[:, 0])

# Plot the mode shape
plt.figure(figsize=(10, 5))
plt.plot(x_eval_mode, mode_shape_vel, 'b-', linewidth=2)
plt.xlabel('Position along beam [m]')
plt.ylabel('Velocity magnitude [m/s]')
plt.title(f'Mode shape at {single_freq} Hz')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
