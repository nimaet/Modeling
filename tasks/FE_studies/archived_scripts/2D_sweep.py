import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from numpy import pi
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import pickle
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

# =========================================================
# Project imports
# =========================================================
import sys
project_root = Path.cwd().parents[2]
sys.path.append(str(project_root))

import Modeling
from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models import FE_helpers
import Modeling.models.FE3 as FE_module

# =========================================================
# Sweep configuration
# =========================================================
SAVE_PREFIX = "sweep"

params_fe = PiezoBeamParams(
	hp=0.252e-3,
	hs=0.51e-3,
	d31=-1.48e-10,
	eps_r=1700,
)

params_fe.zeta_p = 0.0151 * 8
params_fe.zeta_q = 0.0392 * 10

K_p = 0.015
interface_idx = 10
beta = 0.0

ki0 = 1800
ki1 = ki0 / (1 - beta)**2
ki2 = ki0 / (1 + beta)**2

K_i = np.array(
	[ki1, ki2] * (interface_idx // 2)
	+ [ki2, ki1] * (15 - interface_idx // 2)
	+ [ki2]
)

R_c = 1e3

t_end = 1
f0 = 1000
f1 = 3000
dt = 1 / f1 / 50

amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125
# amp_list = np.array([0.05, 0.1,  0.2, 0.25,  0.4]) * 125
kc_magnitudes = np.array([ 8e9, 2e10, 3e10])
# kc_magnitudes = np.array([4e10, 6e10])

Kc_cases = [
	{
		"label": f"Kc_{i}",
		"kc_vec": np.array(
			[kc, kc] * (interface_idx // 2)
			+ [kc, kc] * (15 - interface_idx // 2)
		)
	}
	for i, kc in enumerate(kc_magnitudes)
]

# =========================================================
# Output specification
# =========================================================
OUTPUT_SPEC = {
	"freq": lambda out: out["spectral"]["freq"],
	"X":    lambda out: out["spectral"]["X"],
	"Y":    lambda out: out["spectral"]["Y"],
	"FRF":  lambda out: out["spectral"]["FRF"],
}

# =========================================================
# Filesystem helpers
# =========================================================
sim_dat_dir = Path.cwd() / "sim_dat"
sim_dat_dir.mkdir(parents=True, exist_ok=True)

def unique_dir(path: Path) -> Path:
	if not path.exists():
		path.mkdir(parents=True, exist_ok=False)
		return path
	counter = 1
	while True:
		candidate = path.with_name(f"{path.name}_{counter}")
		if not candidate.exists():
			candidate.mkdir(parents=True, exist_ok=False)
			return candidate
		counter += 1

# =========================================================
# Simulation kernel (workers only execute this)
# =========================================================
def run_single_simulation(
	amp, kc_vec, kc_label,
	K_p, K_i, R_c,
	fe_params,
	dt, t_end, f0, f1,
	output_spec,
	intermediate_dir
):
	try:
		fe = FE_module.PiezoBeamFE(fe_params)

		def v_exc(t):
			return amp * np.sin(
				2 * np.pi * (f0 + t * (f1 - f0) / t_end) * t
			)

		ode = fe.build_ode_system(
			j_exc=30,
			K_c=kc_vec,
			K_i=K_i,
			K_p=K_p,
			R_c=R_c,
			v_exc=v_exc
		)

		out = FE_helpers.solve_newmark(
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

		saved = {
			"amp": amp,
			"kc_label": kc_label,
			"kc_vec": kc_vec
		}

		for name, extractor in output_spec.items():
			saved[name] = extractor(out)

		fname = f"amp_{amp:.3f}_{kc_label}.npz"
		np.savez_compressed(intermediate_dir / fname, **saved)

		return dict(ok=True, file=fname, amp=amp, kc_label=kc_label)

	except Exception as e:
		return dict(ok=False, amp=amp, kc_label=kc_label,
		            error=str(e), exception=type(e).__name__)

# =========================================================
# Build sweep grid
# =========================================================
sweep = [
	{
		"amp": amp,
		"kc_label": case["label"],
		"kc_vec": case["kc_vec"]
	}
	for amp in amp_list
	for case in Kc_cases
]

print(f"Total simulations: {len(sweep)}")

# =========================================================
# CREATE RUN DIRECTORY (ONCE, MAIN PROCESS)
# =========================================================
run_dir = unique_dir(sim_dat_dir / SAVE_PREFIX)

intermediate_dir = run_dir / "intermediate_npz"
intermediate_dir.mkdir(parents=True, exist_ok=True)

png_path    = run_dir / f"{SAVE_PREFIX}.png"
pickle_path = run_dir / f"{SAVE_PREFIX}.pkl"
err_path    = run_dir / f"{SAVE_PREFIX}_errors.json"

print(f"Run output folder: {run_dir}")
print("Running sweep in parallel...\n")

# =========================================================
# Run sweep
# =========================================================
results = Parallel(n_jobs=22, verbose=10)(
	delayed(run_single_simulation)(
		s["amp"],
		s["kc_vec"],
		s["kc_label"],
		K_p, K_i, R_c,
		params_fe,
		dt, t_end, f0, f1,
		OUTPUT_SPEC,
		intermediate_dir
	)
	for s in sweep
)

# =========================================================
# Separate success / failure
# =========================================================
successful = [r for r in results if r["ok"]]
failed     = [r for r in results if not r["ok"]]

print(f"Successful: {len(successful)}")
print(f"Failed:     {len(failed)}")

# =========================================================
# Aggregate results
# =========================================================
results_by_kc = defaultdict(lambda: {
	"kc_label": None,
	"kc_vec": None,
	"amps": [],
	"data": defaultdict(list)
})

for r in successful:
	data = np.load(intermediate_dir / r["file"], allow_pickle=True)
	label = data["kc_label"].item()

	block = results_by_kc[label]
	block["kc_label"] = label
	block["kc_vec"] = data["kc_vec"]
	block["amps"].append(float(data["amp"]))

	for key in OUTPUT_SPEC:
		block["data"][key].append(data[key])

# =========================================================
# Plotting
# =========================================================
fig, axes = plt.subplots(len(Kc_cases), 1, figsize=(10, 4*len(Kc_cases)), sharex=True)
if len(Kc_cases) == 1:
	axes = [axes]

for ax, case in zip(axes, Kc_cases):
	block = results_by_kc[case["label"]]
	colors = plt.cm.viridis(np.linspace(0, 1, len(block["amps"])))

	for amp, freq, FRF, col in zip(
		block["amps"],
		block["data"]["freq"],
		block["data"]["FRF"],
		colors
	):
		ax.semilogy(freq, FRF, color=col, label=f"A={amp:.1f}")

	ax.set_title(case["label"])
	ax.set_ylabel("FRF")
	ax.grid(True)
	ax.legend(fontsize=8)

axes[-1].set_xlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.close(fig)

# =========================================================
# Save outputs
# =========================================================
with open(pickle_path, "wb") as f:
	pickle.dump(dict(
		Kc_labels=[case["label"] for case in Kc_cases],
		results_by_kc=dict(results_by_kc),
		failed=failed,
		run_dir=str(run_dir)
	), f)

with open(err_path, "w") as f:
	json.dump({
		"failed": failed,
		"n_failed": len(failed),
		"n_success": len(successful)
	}, f, indent=2)

config_dict = {
	"amp_list": amp_list.tolist(),
	"kc_magnitudes": kc_magnitudes.tolist(),
	"K_p": float(K_p),
	"K_i": K_i.tolist(),
	"R_c": float(R_c),
	"dt": float(dt),
	"f0": float(f0),
	"f1": float(f1),
	"t_end": float(t_end),
	"interface_idx": int(interface_idx),
	"beta": float(beta),
	"run_dir": str(run_dir)
}

with open(run_dir / f"{SAVE_PREFIX}.json", "w") as f:
	json.dump(config_dict, f, indent=2)

print("All outputs saved successfully.")