"""
2D Parameter Sweep: Amplitude and K_c (Duffing nonlinearity)
Sweeps over both excitation amplitude and cubic stiffness coefficient
Runs simulations in parallel using joblib
"""

from ROM import modal_simulation, run_time_sim, N, S, L_b
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import itertools

# ======= Sweep Parameters =======
K_i = 2100
K_p = 0.03
t_end = 1
f0 = 1000
f1 = 3000

# Parameter lists
amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125  # Amplitude sweep
Kc_list = - np.array([2e9, 4e9, 8e9, 1e10, 1.2e10, 1.6e10, 2e10, 2.25e10, 2.5e10])  # K_c (Duffing) sweep

print(f"Amplitude sweep: {amp_list}")
print(f"K_c sweep: {Kc_list}")
print(f"Total simulations: {len(amp_list) * len(Kc_list)}")

# Setup
x0 = np.zeros(2*N + 2*S)
t_eval = np.arange(0, t_end, 1/f1/20)
x_eval = np.linspace(0, L_b, 100)

# ======= Function to run single simulation =======
def run_single_simulation_2d(amp, kc, K_p, K_i, t_end, f0, f1, x_eval, t_eval):
	"""Run a single simulation for given amplitude and K_c parameters"""
	print(f"  Amp = {amp:.1f} V, K_c = {kc:.2e}")
	
	def v_exc(t, A_exc=amp, f0=f0, f1=f1, t_end=t_end):
		return A_exc*np.sin(2*np.pi*(f0 + t*(f1-f0)/t_end)*t)
	
	res = run_time_sim(v_exc=v_exc, K_c=kc, K_p=K_p, K_i=K_i, t_end=t_end, x_eval=x_eval, t_eval=t_eval)
	
	return {
		"amp": amp,
		"kc": kc,
		"t": res['t'],
		"veloc": res['veloc'],
		"freq": res['freq'],
		"Y": res['Y'],
		"X": np.fft.fft(v_exc(res['t'], A_exc=amp)),
		"FRF": res['FRF']
	}

# ======= Run 2D sweep in parallel =======
print("Running 2D parameter sweep in parallel...")
param_pairs = list(itertools.product(amp_list, Kc_list))

sim_results = Parallel(n_jobs=16, verbose=10)(
	delayed(run_single_simulation_2d)(amp, kc, K_p, K_i, t_end, f0, f1, x_eval, t_eval) 
	for amp, kc in param_pairs
)

# ======= Organize results by K_c =======
results_by_Kc = {}
common_freq = None
common_t = None

for res in sim_results:
	kc = res["kc"]
	if kc not in results_by_Kc:
		results_by_Kc[kc] = {
			"amps": [],
			"FRFs": [],
			"velocs": [],
			"Ys": [],
			"Xs": [],
			"freq": None,
			"t": None
		}
	
	results_by_Kc[kc]["amps"].append(res["amp"])
	results_by_Kc[kc]["FRFs"].append(res["FRF"])
	results_by_Kc[kc]["velocs"].append(res["veloc"])
	results_by_Kc[kc]["Ys"].append(res["Y"])
	results_by_Kc[kc]["Xs"].append(res["X"])
	
	if results_by_Kc[kc]["freq"] is None:
		results_by_Kc[kc]["freq"] = res["freq"]
		results_by_Kc[kc]["t"] = res["t"]
	
	if common_freq is None:
		common_freq = res["freq"]
		common_t = res["t"]

# ======= Create subplots for each K_c value =======
fig, axes = plt.subplots(len(Kc_list), 1, figsize=(10, 4*len(Kc_list)))
if len(Kc_list) == 1:
	axes = [axes]

for idx, kc in enumerate(sorted(results_by_Kc.keys())):
	ax = axes[idx]
	data = results_by_Kc[kc]
	
	cmap = plt.cm.viridis
	colors = cmap(np.linspace(0, 1, len(data["amps"])))
	
	for i, (amp, FRF) in enumerate(zip(data["amps"], data["FRFs"])):
		ax.semilogy(data["freq"], FRF, '.', color=colors[i], linewidth=2, 
		           label=f"A = {amp:.1f} V")
	
	ax.set_xlim([1300, 3000])
	ax.set_ylim([3e-5, 6e-4])
	ax.grid(True)
	ax.legend(loc='best', fontsize=8)
	ax.set_xlabel("Frequency [Hz]")
	ax.set_ylabel("FRF Magnitude")
	ax.set_title(f"Amplitude Sweep at K_c = {kc:.2e}")

plt.tight_layout()
plt.savefig(f'sim_dat/2D_sweep_amp_Kc_Kp={K_p:.3f}_Ki={K_i:.0f}.png', dpi=300, bbox_inches='tight')
print(f"Figure saved to: sim_dat/2D_sweep_amp_Kc_Kp={K_p:.3f}_Ki={K_i:.0f}.png")
plt.show()

# # ======= Create comparison figure: amplitude effect at different K_c =======
# fig2, axes2 = plt.subplots(1, len(amp_list), figsize=(4*len(amp_list), 4))
# if len(amp_list) == 1:
# 	axes2 = [axes2]

# for amp_idx, amp in enumerate(amp_list):
# 	ax = axes2[amp_idx]
# 	cmap = plt.cm.plasma
# 	colors = cmap(np.linspace(0, 1, len(Kc_list)))
	
# 	for kc_idx, kc in enumerate(sorted(results_by_Kc.keys())):
# 		data = results_by_Kc[kc]
# 		if amp in data["amps"]:
# 			local_idx = data["amps"].index(amp)
# 			ax.semilogy(data["freq"], data["FRFs"][local_idx], '.', 
# 			           color=colors[kc_idx], linewidth=2, label=f"K_c = {kc:.2e}")
	
# 	ax.set_xlim([1300, 3000])
# 	ax.set_ylim([3e-5, 6e-4])
# 	ax.grid(True)
# 	ax.legend(loc='best', fontsize=8)
# 	ax.set_xlabel("Frequency [Hz]")
# 	ax.set_ylabel("FRF Magnitude")
# 	ax.set_title(f"K_c Effect at A = {amp:.1f} V")

# plt.tight_layout()
# plt.savefig(f'sim_dat/2D_sweep_Kc_effect_Kp={K_p:.3f}_Ki={K_i:.0f}.png', dpi=300, bbox_inches='tight')
# print(f"Figure saved to: sim_dat/2D_sweep_Kc_effect_Kp={K_p:.3f}_Ki={K_i:.0f}.png")
# plt.show()
#%%
# ======= Save results =======
np.savez(f'sim_dat/2D_sweep_softening_Kp={K_p:.3f}_Ki={K_i:.0f}.npz', 
         amp_list=amp_list, 
         Kc_list=Kc_list,
		 K_p=K_p,
		 K_i=K_i,
         results_by_Kc=results_by_Kc)
print(f"Results saved to: sim_dat/2D_sweep_Kp={K_p:.3f}_Ki={K_i:.0f}.npz")

print("\n" + "="*60)
print("2D SWEEP COMPLETE")
print("="*60)
print(f"Amplitude range: {amp_list[0]:.1f} - {amp_list[-1]:.1f} V")
print(f"K_c range: {Kc_list[0]:.2e} - {Kc_list[-1]:.2e}")
print(f"K_p: {K_p}, K_i: {K_i}")
print(f"Total simulations completed: {len(sim_results)}")
print("="*60)
