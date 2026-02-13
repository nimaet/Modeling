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
kc0 = 1.6e10*2
gamma = 0.2
ki= 1800
ki0 = (1-gamma)*ki; ki1 = (1+gamma)*ki
K_i = 1800; K_p = 0.03; K_c = -1.6e10 ; R_c = 1e3
K_i = np.array([ki0, ki1]*15+[ki0])
t_end = 1.0
f0 = 1000
f1 = 3000
dt = 1/f1/50
npz_filename = f'FE_2D_sweep_alternating_Kp={K_p:.3f}_Ki={ki:.0f}.npz'
amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125  # Amplitude sweep
# Kc_list = -np.array([ 4e9, 8e9, 1e10, 1.2e10, 1.3e10, 1.4e10, 1.5e10, 1.6e10])  # K_c (Duffing) sweep
Kc0_list = [  2e10, 3e10, 410, 5e10, 6e10]  # K_c (Duffing) sweep
Kc_list = np.array([[kc0, -kc0]*15  for kc0 in Kc0_list])
print(f"Amplitude sweep: {amp_list}")
# print(f"K_c sweep: {Kc_list}")
print(f"Total simulations: {len(amp_list) * len(Kc_list)}")

# Setup FE model
params_fe = PiezoBeamParams(hp=0.252e-3, hs=0.51e-3
                            , d31= -1.48e-10,eps_r=1700
                            )
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
	print("shape of kc:", kc.shape)
	try:
		print(f"  Amp = {amp:.1f} V, K_c = {kc[0]:.2e}")
		
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
		print(f"ERROR: Simulation failed for Amp={amp:.1f}V, K_c={kc[0]:.2e}")
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

# ...existing code...

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
        # kc is a vector; print a concise repr
        kc_vec = np.asarray(fail['kc'])
        kc_repr = f"[{kc_vec[0]:.2e}, {kc_vec[1]:.2e}, ...]"
        print(f"  Amp={fail['amp']:.1f}V, K_c={kc_repr}: {fail['exception']}")
        print(f"    → {fail['message']}")
    print("-" * 70 + "\n")

# ======= Organize results by K_c VECTOR =======
results_by_Kc = {}
kc_catalog = []   # unique K_c vectors (arrays)
kc_labels = []    # representative scalar for sorting (abs of first element)

for res in sim_results:
    kc_vec = np.asarray(res["kc"])
    # hashable key: rounded tuple to avoid float noise
    kc_key = tuple(np.round(kc_vec.astype(float), 8))
    if kc_key not in results_by_Kc:
        results_by_Kc[kc_key] = {
            "Kc_vector": kc_vec,
            "amps": [],
            "FRFs": [],
            "time_histories": [],
            "freq": None,
            "t": None
        }
        kc_catalog.append(kc_vec)
        kc_labels.append(float(np.abs(kc_vec[0])))

    results_by_Kc[kc_key]["amps"].append(res["amp"])
    results_by_Kc[kc_key]["FRFs"].append(res["FRF"])

    # Store complete time domain output
    results_by_Kc[kc_key]["time_histories"].append({
        "amp": res["amp"],
        "t": res["t"],
        "u": res["u"],
        "u_dot": res["u_dot"],
        "u_ddot": res["u_ddot"],
        "q": res["q"],
        "v": res["v"]
    })

    if results_by_Kc[kc_key]["freq"] is None:
        results_by_Kc[kc_key]["freq"] = res["freq"]
        results_by_Kc[kc_key]["t"] = res["t"]

# ======= Sort keys by representative K_c magnitude =======
sort_idx = np.argsort(kc_labels)
sorted_kc_keys = [tuple(np.round(kc_catalog[i].astype(float), 8)) for i in sort_idx]

# ======= Create subplots for each K_c VECTOR =======
fig, axes = plt.subplots(len(sorted_kc_keys), 1, figsize=(10, 4*len(sorted_kc_keys)))
if len(sorted_kc_keys) == 1:
    axes = [axes]

for idx, kc_key in enumerate(sorted_kc_keys):
    ax = axes[idx]
    data = results_by_Kc[kc_key]

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(data["amps"])))

    for i, (amp, FRF) in enumerate(zip(data["amps"], data["FRFs"])):
        ax.semilogy(data["freq"], FRF, '.', color=colors[i], linewidth=2,
                    label=f"A = {amp:.1f} V")

    kc_vec = data["Kc_vector"]
    kc_label = f"[{kc_vec[0]:.2e}, {kc_vec[1]:.2e}, ...]"

    ax.set_xlim([1300, 3000])
    ax.set_ylim([3e-5, 6e-4])
    ax.grid(True)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("FRF Magnitude")
    ax.set_title(f"FE: Amplitude Sweep at K_c = {kc_label}")

plt.tight_layout()

# Create sim_dat directory if it doesn't exist
sim_dat_dir = Path(__file__).parent / 'sim_dat'
sim_dat_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(sim_dat_dir / f'FE_2D_sweep_alternating_Kp={K_p:.3f}_Ki={K_i:.0f}.png', dpi=300, bbox_inches='tight')
print(f"Figure saved to: {sim_dat_dir / f'FE_2D_sweep_alternating_Kp={K_p:.3f}_Ki={K_i:.0f}.png'}")
plt.close(fig)

# ======= Save results =======
# Keep compatibility with plotting by saving both a scalar representative (abs(first)) and full vectors.
np.savez(sim_dat_dir / npz_filename,
         amp_list=amp_list,
         # scalar representative for sorting/labels in downstream plots
         Kc_list=np.array([np.abs(vec[0]) for vec in kc_catalog], dtype=float),
         # full K_c vectors for exact reproduction
         Kc_vectors=np.array(kc_catalog, dtype=object),
         K_p=K_p,
         K_i=K_i,
         R_c=R_c,
         t_end=t_end,
         f0=f0,
         f1=f1,
         dt=dt,
         # FE model parameters
         params_fe_hp=params_fe.hp,
         params_fe_hs=params_fe.hs,
         params_fe_d31=params_fe.d31,
         params_fe_eps_r=params_fe.eps_r,
         params_fe_zeta_p=params_fe.zeta_p,
         params_fe_zeta_q=params_fe.zeta_q,
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

    error_log_path = sim_dat_dir / f'FE_2D_sweep_alternating_errors_Kp={K_p:.3f}_Ki={K_i:.0f}.json'
    with open(error_log_path, 'w') as f:
        json.dump(error_log, f, indent=2)
    print(f"Error log saved to: {error_log_path}")

# ...existing code...