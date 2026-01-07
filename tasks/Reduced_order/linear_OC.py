from pathlib import Path
import sys
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from beam_properties import PiezoBeamParams
from ROM import ROM
import numpy as np
from numpy import pi	
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm, colors
import pandas as pd

# # ======= K_p sweep in frequency domain =======
K_i = 0; K_p = 0.0001; K_c = 0
t_end = 0.1
f0 = 1000
f1 = 5000
N = 40


def v_exc(t, A_exc=50, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
params = PiezoBeamParams()
rom = ROM(params, N=N)

# x0 = np.zeros(2*N + 2*S)

f0 = 1000; f1 = 3000
t_eval = np.arange(0, t_end, 1/f1/20)
x_eval = np.linspace(0, rom.p.L_b, 100)

results = rom.run_time_sim(v_exc=v_exc, K_c=K_c, K_p=K_p, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
t = results['t']
veloc = results['veloc']
freq = results['freq']
Y = results['Y']
X = results['X']
FRF = results['FRF']

freq_modal, vel_mag, disp_mag = rom.frequency_response(K_c=K_c, K_p=K_p, K_i=K_i)

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

comsol_OC = pd.read_csv('../../comsol/OC_wide.csv')
#%%
plt.figure(figsize=(12, 4))
# plt.semilogy(frq_linear_exp, np.mean(frf_data_linear_exp[:,:], axis=1), 'r--', label=' Experiment')
plt.semilogy(comsol_OC['freq'], comsol_OC['w']*2*pi*comsol_OC['freq'], 'g-', label='COMSOL ')
# plt.semilogy(comsol_OC['freq'], comsol_OC['w'], 'g-', label='COMSOL displacement FRF')
# plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Open circuit Exp.')
# plt.semilogy(frq_SC_exp, np.mean(frf_data_SC_exp[:,:], axis=1), 'b--', label=f'Short circuit Exp.')
# plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order'   )
plt.semilogy(freq, FRF, '.-', linewidth=1.5, label='Time Domain ROM')
plt.semilogy(freq_modal, vel_mag, '.-', label='Frequency Domain ROM'   )
plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Experiment')
# plt.semilogy(frq_linear, np.mean(frf_data_linear[:,:], axis=1), 'r--', label=' Exp.')

# plt.semilogy(freq_modal, disp_mag*freq_modal*2*np.pi, '-', label='Modal Reduced Order Displacement $j \omega$'   )
plt.legend()
# plt.xlim([1300, 3000])
plt.xlim([1000, 4500])
# plt.ylim([1e-5, 1e-3])
plt.ylim([3e-5, 6e-4])
plt.xlabel("Frequency [Hz]")
plt.ylabel("AverageVelocity/Voltage FRF")
plt.grid(True)
plt.show()





plt.figure()

# plt.plot(t, disp[10,:], '.-', label='Disp.')
plt.plot(t, veloc[10,:], '-', label='Velocity')
# plt.xlim([0, 0.05 ])
# plt.ylim(np.array([-1,1])*1e-3)

plt.xlabel("t [s]")
# plt.ylabel("displacement")
plt.legend()
plt.grid(True)