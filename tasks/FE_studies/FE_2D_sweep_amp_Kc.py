"""
2D Parameter Sweep: Amplitude and K_c (Duffing nonlinearity) using Finite Element
Sweeps over both excitation amplitude and cubic stiffness coefficient
Runs simulations in parallel using joblib
Similar to the ROM version but using FE discretization
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from joblib import Parallel, delayed
import itertools
import traceback
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models.FE1 import PiezoBeamFE, solve_newmark

# ======= Sweep Parameters =======
K_i = 2000
K_p = 0.03
R_c = 1e3
t_end = 1.0
f0 = 1000
f1 = 3000
dt = 1/f1/50

# Parameter lists
amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125  # Amplitude sweep
Kc_list = - np.array([2e9, 4e9, 8e9, 1e10, 1.2e10, 1.6e10, 2e10, 2.25e10, 2.5e10])  # K_c (Duffing) sweep

print(f"Amplitude sweep: {amp_list}")
print(f"K_c sweep: {Kc_list}")
print(f"Total simulations: {len(amp_list) * len(Kc_list)}")

# Setup FE model
params_fe = PiezoBeamParams(hp=0.252e-3, hs=0.51e-3)
params_fe.zeta_p = 0.0151 * 8
params_fe.zeta_q = 0.0392 * 10
fe = PiezoBeamFE(params_fe, n_el_gap=1, n_el_patch=3)

# ======= Function to run single simulation =======
def run_single_simulation_2d_fe(amp, kc, K_p, K_i, R_c, fe_params, dt, t_end, f0, f1):
	"""Run a single FE simulation for given amplitude and K_c parameters.
	
	Returns:
		{"status": "ok", "result": {...}} on success
		{"status": "failed", "error": error_message, "amp": amp, "kc": kc} on failure
	"""
	try:
		print(f"  Amp = {amp:.1f} V, K_c = {kc:.2e}")
		
		# Recreate FE object in each worker (for parallel safety)
		fe_local = PiezoBeamFE(fe_params, n_el_gap=1, n_el_patch=3)
		
		# Create excitation function for this amplitude
		def v_exc(t_var, A_exc=amp, f0=f0, f1=f1, t_end=t_end):
			return A_exc * np.sin(2*np.pi*(f0 + t_var*(f1-f0)/t_end) * t_var)
		
		# Build ODE system
		ode = fe_local.build_ode_system(
			j_exc=30,
			K_c=kc,
			K_i=K_i,
			K_p=K_p,
			R_c=R_c,
			v_exc=v_exc
		)
		
		# Run time-domain simulation
		result = solve_newmark(
			ode=ode,
			dt=dt,
			t_end=t_end,
			beta=0.25,
			gamma=0.5,
			newton_tol=1e-8,
			newton_maxiter=8,
			x0=np.zeros(ode.M.shape[0]),
			x_dot0=np.zeros(ode.M.shape[0]),
			do_spectral=True
		)
		
		return {
			"status": "ok",
			"result": {
				"amp": amp,
				"kc": kc,
				"t": result['t'],
				"u": result['u'],
				"u_dot": result['u_dot'],
				"u_ddot": result['u_ddot'],
				"q": result['q'],
				"v": result['v'],
				"freq": result['spectral']['freq'],
				"FRF": result['spectral']['FRF'],
				"spectrum": result['spectral']
			}
		}
	
	except Exception as e:
		print(f"\n{'='*70}")
		print(f"ERROR: Simulation failed for Amp={amp:.1f}V, K_c={kc:.2e}")
		print(f"Exception: {type(e).__name__}: {str(e)}")
		print(f"{'='*70}\n")
		
		return {
			"status": "failed",
			"amp": amp,
			"kc": kc,
			"error": str(e),
			"exception_type": type(e).__name__
		}

# ======= Run 2D sweep in parallel =======
print("Running 2D parameter sweep in parallel...")
print(f"Note: Errors in individual simulations will NOT stop the sweep.")
print(f"Partial results will be saved for successfully completed simulations.\n")

param_pairs = list(itertools.product(amp_list, Kc_list))

all_results = Parallel(n_jobs=32, verbose=10)(
	delayed(run_single_simulation_2d_fe)(amp, kc, K_p, K_i, R_c, params_fe, dt, t_end, f0, f1) 
	for amp, kc in param_pairs
)

# ======= Separate successful and failed simulations =======
sim_results = []
failed_sims = []

for res in all_results:
	if res["status"] == "ok":
		sim_results.append(res["result"])
	else:
		failed_sims.append({
			"amp": res["amp"],
			"kc": res["kc"],
			"exception": res.get("exception_type", "Unknown"),
			"message": res["error"]
		})

print(f"\n{'='*70}")
print(f"SWEEP STATUS:")
print(f"  Total simulations: {len(all_results)}")
print(f"  Successful: {len(sim_results)}")
print(f"  Failed: {len(failed_sims)}")
print(f"{'='*70}\n")

if failed_sims:
	print("FAILED SIMULATIONS:")
	print("-" * 70)
	for fail in failed_sims:
		print(f"  Amp={fail['amp']:.1f}V, K_c={fail['kc']:.2e}: {fail['exception']}")
		print(f"    → {fail['message']}")
	print("-" * 70 + "\n")

# ======= Organize results by K_c =======
results_by_Kc = {}
common_freq = None

for res in sim_results:
	kc = res["kc"]
	if kc not in results_by_Kc:
		results_by_Kc[kc] = {
			"amps": [],
			"FRFs": [],
			"time_histories": [],
			"freq": None,
			"t": None
		}
	
	results_by_Kc[kc]["amps"].append(res["amp"])
	results_by_Kc[kc]["FRFs"].append(res["FRF"])
	
	# Store complete time domain output
	results_by_Kc[kc]["time_histories"].append({
		"amp": res["amp"],
		"t": res["t"],
		"u": res["u"],
		"u_dot": res["u_dot"],
		"u_ddot": res["u_ddot"],
		"q": res["q"],
		"v": res["v"]
	})
	
	if results_by_Kc[kc]["freq"] is None:
		results_by_Kc[kc]["freq"] = res["freq"]
		results_by_Kc[kc]["t"] = res["t"]
	
	if common_freq is None:
		common_freq = res["freq"]

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
	ax.set_title(f"FE: Amplitude Sweep at K_c = {kc:.2e}")

plt.tight_layout()

# Create sim_dat directory if it doesn't exist
sim_dat_dir = Path(__file__).parent / 'sim_dat'
sim_dat_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(sim_dat_dir / f'FE_2D_sweep_amp_Kc_Kp={K_p:.3f}_Ki={K_i:.0f}.png', dpi=300, bbox_inches='tight')
print(f"Figure saved to: {sim_dat_dir / f'FE_2D_sweep_amp_Kc_Kp={K_p:.3f}_Ki={K_i:.0f}.png'}")
plt.close(fig)

# ======= Save results =======
npz_filename = f'FE_2D_sweep_softening_Kp={K_p:.3f}_Ki={K_i:.0f}.npz'
np.savez(sim_dat_dir / npz_filename, 
         amp_list=amp_list, 
         Kc_list=Kc_list,
         K_p=K_p,
         K_i=K_i,
         results_by_Kc=results_by_Kc)
print(f"Results saved to: {sim_dat_dir / npz_filename}")

# ======= Save error log if there were failures =======
if failed_sims:
	error_log = {
		"timestamp": datetime.now().isoformat(),
		"total_simulations": len(all_results),
		"successful_simulations": len(sim_results),
		"failed_simulations": len(failed_sims),
		"failed_cases": failed_sims,
		"sweep_parameters": {
			"K_p": K_p,
			"K_i": K_i,
			"R_c": R_c,
			"t_end": t_end,
			"f0": f0,
			"f1": f1,
			"dt": dt
		}
	}
	
	error_log_path = sim_dat_dir / f'FE_2D_sweep_errors_Kp={K_p:.3f}_Ki={K_i:.0f}.json'
	with open(error_log_path, 'w') as f:
		json.dump(error_log, f, indent=2)
	print(f"Error log saved to: {error_log_path}")

print("\n" + "="*70)
print("2D FE SWEEP COMPLETE")
print("="*70)
print(f"Amplitude range: {amp_list[0]:.1f} - {amp_list[-1]:.1f} V")
print(f"K_c range: {Kc_list[0]:.2e} - {Kc_list[-1]:.2e}")
print(f"K_p: {K_p}, K_i: {K_i}")
print(f"Total simulations attempted: {len(all_results)}")
print(f"Successful simulations: {len(sim_results)}")
if failed_sims:
	print(f"Failed simulations: {len(failed_sims)}")
	print(f"Success rate: {100*len(sim_results)/len(all_results):.1f}%")
print("\nOUTPUTS SAVED:")
print(f"  - NPZ file with all results: {npz_filename}")
print(f"  - Visualization: FE_2D_sweep_amp_Kc_*.png")
if failed_sims:
	print(f"  - Error log: FE_2D_sweep_errors_*.json")
print("="*70)
