import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from numpy import pi
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Callable
import json
import os
import shutil
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
# SWEEP SPEC INFRASTRUCTURE
# =========================================================
from dataclasses import dataclass

@dataclass
class SweepParam:
	key: str
	values: list
	target: str | list[str] | None = None
	builder: Callable | None = None
	description: str = ""

	def resolve(self, value):
		if self.builder is not None:
			return self.builder(value)
		return value


class SweepSpec:
	def __init__(self, params: list[SweepParam]):
		self.params = params
		self.keys = [p.key for p in params]

	def build_grid(self):
		value_lists = [p.values for p in self.params]
		grid = []

		for combo in product(*value_lists):
			entry = {}
			for p, v in zip(self.params, combo):
				entry[p.key] = v
			grid.append(entry)

		return grid

	def apply(self, base_params: dict, sweep_entry: dict):
		params = dict(base_params)

		for p in self.params:
			if p.key not in sweep_entry:
				continue

			val = p.resolve(sweep_entry[p.key])
			target = p.target if p.target is not None else p.key

			if isinstance(target, (list, tuple)):
				# Multiple targets: broadcast a scalar to all of them,
				# or zip a matching-length sequence value across them
				# (e.g. one "freq" key driving both f0 and f1 for a
				# single-tone excitation).
				if isinstance(val, (list, tuple)):
					for t, v in zip(target, val):
						params[t] = v
				else:
					for t in target:
						params[t] = val
			else:
				params[target] = val

		return params

	def validate(self, base_params: dict):
		targets = set()
		for p in self.params:
			t = p.target if p.target is not None else p.key
			ts = t if isinstance(t, (list, tuple)) else [t]
			for tt in ts:
				if tt in targets:
					raise ValueError(f"Duplicate sweep target detected: {tt}")
				targets.add(tt)

		for t in targets:
			if t not in base_params and t not in ("amp", "f0", "f1"):
				print(f"[SweepSpec] Warning: '{t}' not in base_params")


# =========================================================
# BASE MODEL PARAMETERS
# =========================================================
SAVE_PREFIX = "Duffing_freq_sweep"
sim_dat_dir = Path.cwd() / "sim_dat"
sim_dat_dir.mkdir(parents=True, exist_ok=True)

params_fe = PiezoBeamParams(
	hp=0.252e-3,
	hs=0.51e-3,
	d31=-1.45e-10,
	eps_r=1700,
)

interface_idx = 10
beta = 0.3
ki0 = 4000


def Ki_builder(beta):
	ki1 = ki0 / (1 - beta)**2
	ki2 = ki0 / (1 + beta)**2
	return np.array(
		[ki1, ki2] * (interface_idx // 2)
		+ [ki1, ki2] * (15 - interface_idx // 2)
		+ [ki1]
	)


def Kc_builder(kc):
	return np.array(
		[-kc, kc] * (interface_idx // 2)
		+ [kc, -kc] * (15 - interface_idx // 2)
	)


BASE_PARAMS = dict(
	K_p=0.025,
	K_i=1800,
	K_c=3e10,
	R_c=1e3,
	amp = 50,
)

# =========================================================
# TIME / EXCITATION
# =========================================================
# t_end = 1
f0 = 500
f1 = 3500
# dt = 1 / max(f0, f1) / 50

# =========================================================
# DEFINE SWEEP
# =========================================================
# amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 ]) * 125

# Kc_magnitudes =  np.array([8.00e9, 1.20e10 1.50e10, 1.80e10, 2.50e10, 2.75e10, 3.00e10]) #for hardening
# Kc_magnitudes =  -np.array([ 8.00e9, 1.0e10, 1.20e10, 1.30e10, 1.4e10, 1.6e10, 1.8e10, 2.0e10 ]) #for softening

Kc_soft =  -np.array([5.0e8, 1.0e9, 2.0e9, 4.00e9 , 8.00e9, 1.0e10, 1.20e10, 1.30e10, 1.4e10, 1.6e10, 1.8e10, 2.0e10 ]) 
Kc_hard =  np.array([5e8, 1e9, 2.0e9, 4.0e9, 8.00e9, 1.20e10, 1.50e10, 1.80e10, 2.50e10, 2.75e10, 3.00e10]) 

sweep_spec = SweepSpec([
	# SweepParam(
	# 	key="amp",
	# 	values=amp_list.tolist(),
	# 	description="Excitation amplitude",
	# ),

	# SweepParam(
	# 	key="K_c",
	# 	values=Kc_soft.tolist()[::-1]+Kc_hard.tolist(),
	# 	target="K_c",
	# 	description="nonlinear inductance",
	# ),

	# Excitation frequency sweep (single-tone): one "freq" value is
	# broadcast to both f0 and f1, so v_exc's chirp formula collapses
	# to a constant-frequency sine at that value. When this param is
	# absent, f0/f1 fall back to the module-level chirp (f0 -> f1
	# over t_end), so the default sweep is unaffected.
	SweepParam(
		key="freq",
		values=np.linspace(1000, 3000, 500).tolist(),
		target=["f0", "f1"],
		description="Excitation frequency (single-tone)",
	),
])

# =========================================================
# OUTPUT EXTRACTION
# =========================================================
OUTPUT_SPEC = {
	"t": lambda out: out["t"],
	"u": lambda out: out["u"],
	"u_dot": lambda out: out["u_dot"],
	"v": lambda out: out["v"],
	"z": lambda out: out["z"],
	# "FRF": lambda out: out['spectral']['FRF'],
	# "freq": lambda out: out['spectral']['freq'],
}

# =========================================================
# UTILITIES
# =========================================================
def to_jsonable(obj):
	if isinstance(obj, dict):
		return {k: to_jsonable(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [to_jsonable(v) for v in obj]
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, Path):
		return str(obj)
	return f'{obj:0.3e}' if isinstance(obj, float) else obj


# =========================================================
# BUILD SWEEP GRID
# =========================================================
sweep_spec.validate(BASE_PARAMS)
SWEEP_GRID = sweep_spec.build_grid()

print(f"Total simulations: {len(SWEEP_GRID)}")

# Recompute dt in case a frequency sweep pushes above the default f0/f1
# range (dt must resolve the highest excitation frequency actually used).
def _max_swept_freq():
	max_freq = max(f0, f1)
	for p in sweep_spec.params:
		target = p.target if p.target is not None else p.key
		targets = target if isinstance(target, (list, tuple)) else [target]
		if "f0" in targets or "f1" in targets:
			max_freq = max(max_freq, max(p.resolve(v) for v in p.values))
	return max_freq


# dt = 1 / _max_swept_freq() / 50

# =========================================================
# SINGLE SIMULATION
# =========================================================
def run_single_simulation(
	index,
	sweep_entry,
	base_params,
	fe_params,
	# dt, t_end, f0, f1,
	output_spec,
	out_dir
):

	try:
		params = sweep_spec.apply(base_params, sweep_entry)
		amp = params["amp"]
		f0_val = params.get("f0", f0)
		f1_val = params.get("f1", f1)

		fe = FE_module.PiezoBeamFE(fe_params)

		def v_exc(t):
			return amp * np.sin(
				2 * np.pi * (f0_val + t * (f1_val - f0_val) / t_end) * t
			)

		ode = fe.build_ode_system(
			j_exc=30,
			K_c=params.get("K_c"),
			K_i=params["K_i"],
			K_p=params["K_p"],
			R_c=params["R_c"],
			v_exc=v_exc
		)
		dt = 1 / max(f0_val, f1_val) / 50
		t_end = 1/f0_val * 10000
		out = FE_helpers.solve_newmark(
			ode=ode,
			dt=dt,
			t_end=t_end,
			beta=0.25,
			gamma=0.5,
			newton_tol=1e-6,
			newton_maxiter=32,
			x0=np.zeros(ode.M.shape[0]),
			x_dot0=np.zeros(ode.M.shape[0]),
			do_spectral=False
		)

		data = {k: fn(out) for k, fn in output_spec.items()}
		data['v_exc'] = v_exc(out['t'])

		np.savez_compressed(
			out_dir / f"sim_{index:05d}.npz",
			params=params,
			data=data
		)

		return dict(ok=True, index=index)

	except Exception as e:
		return dict(
			ok=False,
			index=index,
			sweep_entry=sweep_entry,
			error=str(e),
			exception=type(e).__name__
		)


# =========================================================
# SLURM ARRAY INDEX
# =========================================================
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"

array_task_count = int(
	os.environ.get("SLURM_ARRAY_TASK_COUNT")
	or (
		int(os.environ.get("SLURM_ARRAY_TASK_MAX", array_id))
		- int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
		+ 1
	)
)

print(f"Running array task {array_id} of {array_task_count}")

assigned_indices = list(range(array_id, len(SWEEP_GRID), array_task_count))

if not assigned_indices:
	print(f"Array task {array_id:05d} has no assigned simulations. Exiting.")
	raise SystemExit(0)

# =========================================================
# DIRECTORY SETUP (RUN ONCE)
# =========================================================
run_dir = sim_dat_dir / f"{SAVE_PREFIX}_{array_job_id}"
npz_dir = run_dir / "npz"
status_dir = run_dir / "status"

# Use node-local temp storage for heavy intermediate writes, then copy once.
tmp_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
task_tmp_dir = tmp_root / f"{SAVE_PREFIX}_{array_job_id}" / f"task_{array_id:05d}"
task_tmp_dir.mkdir(parents=True, exist_ok=True)

run_dir.mkdir(parents=True, exist_ok=True)
npz_dir.mkdir(parents=True, exist_ok=True)
status_dir.mkdir(parents=True, exist_ok=True)

if array_id == 0:


	CONFIG = {
		"created_at": datetime.now().isoformat(),
		"time": dict( f0=f0, f1=f1),
		"sweep_spec": [
			{
				"key": p.key,
				"values": p.values,
				"target": p.target,
				"description": p.description,
			}
			for p in sweep_spec.params
		],
		"fe_params": to_jsonable(vars(params_fe)),
		"base_params": {
			k: to_jsonable(v)
			for k, v in BASE_PARAMS.items()
		},
	}

	with open(run_dir / "config.json", "w") as f:
		json.dump(to_jsonable(CONFIG), f, indent=2)


# =========================================================
# RUN ASSIGNED CASES (HYBRID: ARRAY + JOBLIB)
# =========================================================
def run_and_store(index):
	sweep_entry = SWEEP_GRID[index]
	local_dir = task_tmp_dir / f"sim_{index:05d}"
	local_dir.mkdir(parents=True, exist_ok=True)

	result = run_single_simulation(
		index,
		sweep_entry,
		BASE_PARAMS,
		params_fe,
		# dt, t_end, f0, f1,
		OUTPUT_SPEC,
		local_dir
	)

	if result["ok"]:
		local_npz = local_dir / f"sim_{index:05d}.npz"
		shared_npz = npz_dir / f"sim_{index:05d}.npz"
		shutil.copy2(local_npz, shared_npz)

	with open(status_dir / f"result_{index:05d}.json", "w") as f:
		json.dump(to_jsonable(result), f, indent=2)

	return result


n_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
n_jobs = min(n_jobs, len(assigned_indices))

print(
	f"Array task {array_id:05d} processing indices {assigned_indices} "
	f"with n_jobs={n_jobs}"
)

if n_jobs == 1:
	results = [run_and_store(i) for i in assigned_indices]
else:
	results = Parallel(n_jobs=n_jobs, prefer="processes")(
		delayed(run_and_store)(i) for i in assigned_indices
	)

ok_count = sum(1 for r in results if r.get("ok", False))
fail_count = len(results) - ok_count
print(f"Array task {array_id:05d} complete. success={ok_count}, failed={fail_count}")
print(f"Saved task outputs to: {run_dir}")