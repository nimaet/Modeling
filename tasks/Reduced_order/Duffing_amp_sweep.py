from ROM import modal_simulation, run_time_sim,odefun, mode_shape, N, S, L_b
import ROM
import numpy as np
from numpy import pi	
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm, colors
import pandas as pd
from joblib import Parallel, delayed

# # ======= K_p sweep in frequency domain =======
K_i = 2100; K_p = 0.03; K_c = 5e9
param_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125 # Amplitude sweep
t_end = 1
f0 = 1300
f1 = 3000




x0 = np.zeros(2*N + 2*S)

f0 = 1000; f1 = 3000
t_eval = np.arange(0, t_end, 1/f1/20)
x_eval = np.linspace(0, L_b, 100)


# param_list = np.arange(0.01, 0.8, 0.05) # K_p sweep
# A_list = np.array([0.05, 0.15, 0.25, 0.4]) * 125
# param_list = np.array([0.05, 0.4]) * 125

# ======= Function to run single simulation =======
def run_single_simulation(par, K_c, K_p, K_i, t_end, f0, f1, x_eval, t_eval):
	"""Run a single simulation for given amplitude parameter"""
	print(f"Amp = {par} ...")	

	def v_exc(t, A_exc=par, f0=f0, f1=f1, t_end=t_end):
		return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
	
	res = run_time_sim(v_exc=v_exc, K_c=K_c, K_p=K_p, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
	
	return {
		"param": par,
		"t": res['t'],
		"veloc": res['veloc'],
		"freq": res['freq'],
		"Y": res['Y'],
		"X": res['X'],
		"FRF": res['FRF']
	}

# ======= Run simulations in parallel =======
print("Running simulations in parallel...")
sim_results = Parallel(n_jobs=-1, verbose=10)(
	delayed(run_single_simulation)(par, K_c, K_p, K_i, t_end, f0, f1, x_eval, t_eval) 
	for par in param_list
)

# ======= Collect results =======
results = {
	"params": [],
	"param_name": "Amp",
	"t": None,
	"veloc": [],
	"freq": None,
	"Y": [],
	"X": [],
	"FRF": []
}

for i, res in enumerate(sim_results):
	results["params"].append(res["param"])
	results["veloc"].append(res["veloc"])
	results["Y"].append(res["Y"])
	results["FRF"].append(res["FRF"])
	results["X"].append(res["X"])
	
	# Save time and frequency (from first simulation)
	if results["freq"] is None:
		results["freq"] = res["freq"]
		results["t"] = res["t"]

freq_modal, vel_mag, disp_mag = modal_simulation(K_c=K_c, K_p=K_p, K_i=K_i)

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

# plt.figure(figsize=(12, 5))
# # plt.semilogy(frq_linear_exp, np.mean(frf_data_linear_exp[:,:], axis=1), 'r--', label=' Experiment')
# # plt.semilogy(comsol_OC['freq'], comsol_OC['w']*2*pi*comsol_OC['freq'], 'g-', label='COMSOL ')
# # plt.semilogy(comsol_OC['freq'], comsol_OC['w'], 'g-', label='COMSOL displacement FRF')
# # plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Open circuit Exp.')
# # plt.semilogy(frq_SC_exp, np.mean(frf_data_SC_exp[:,:], axis=1), 'b--', label=f'Short circuit Exp.')
# # plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order'   )
# # plt.semilogy(freq, FRF, '.-', linewidth=1.5, label='Time Domain ROM')
# # plt.semilogy(freq_modal, vel_mag, '.-', label='Frequency Domain ROM'   )
# # plt.semilogy(frq_OC_exp, np.mean(frf_data_OC_exp[:,:], axis=1), 'k--', label=f'Experiment')
# # plt.semilogy(frq_linear, np.mean(frf_data_linear[:,:], axis=1), 'r--', label=' Exp.')

# # plt.semilogy(freq_modal, disp_mag*freq_modal*2*np.pi, '-', label='Modal Reduced Order Displacement $j \omega$'   )
# plt.legend()
# # plt.xlim([1300, 3000])
# plt.xlim([1000, 4500])
# # plt.ylim([1e-5, 1e-3])
# plt.ylim([3e-5, 6e-4])
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("AverageVelocity/Voltage FRF")
# plt.grid(True)
# plt.show()


#%%
np.savez('./'+f"sim_dat/amp_sweep_Kc={K_c:.2e}_Kp={K_p:.3f}_Ki={K_i:.0f}"+'.npz', **results)
plt.figure(figsize=(6,4))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results['params'])))   # gradient from low A → high A
i = -1
for A, FRF in zip(results["params"], results["FRF"]):
	i +=1
	# plt.semilogy(results["freq"], FRF, linewidth=1.2, label=f"A={A:.2f} V")
	plt.semilogy(results["freq"], FRF, '.',color=colors[i], linewidth=3, label=f"{results['param_name']}={A:.2f} V")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF Magnitude")
plt.title(f"kc={K_c:.2e}, kp={K_p:0.3f}")
plt.xlim([1300, 3000])
plt.ylim([3e-5, 6e-4])
plt.grid(True)
plt.legend()
plt.show()
