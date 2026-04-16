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
	target: str | None = None
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

			val = sweep_entry[p.key]
			target = p.target if p.target is not None else p.key
			params[target] = p.resolve(val)

		return params

	def validate(self, base_params: dict):
		targets = set()
		for p in self.params:
			t = p.target if p.target is not None else p.key
			if t in targets:
				raise ValueError(f"Duplicate sweep target detected: {t}")
			targets.add(t)

		for t in targets:
			if t not in base_params and t != "amp":
				print(f"[SweepSpec] Warning: '{t}' not in base_params")


# =========================================================
# BASE MODEL PARAMETERS
# =========================================================
SAVE_PREFIX = "NES"
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
	K_p=0.018,
	K_i=0.001,
	K_c=3e10,
	R_c=1e3,
)

# =========================================================
# TIME / EXCITATION
# =========================================================
t_end = 1
f0 = 500
f1 = 3000
dt = 1 / max(f0, f1) / 50

# =========================================================
# DEFINE SWEEP
# =========================================================
amp_list = np.array([ 0.2, 0.4]) * 125
kc_magnitudes = np.linspace(6e11, 3e12, 4)

sweep_spec = SweepSpec([
	SweepParam(
		key="amp",
		values=amp_list.tolist(),
		description="Excitation amplitude",
	),

	SweepParam(
		key="K_c",
		values=kc_magnitudes.tolist(),
		target="K_c",
		description="nonlinear inductance",
	),

	SweepParam(
		key="K_p",
		values=(np.linspace(0.01, 0.1, 12)).tolist(),
		target="K_p",
		description="linear inductance",
	),
])

# =========================================================
# OUTPUT EXTRACTION
# =========================================================
OUTPUT_SPEC = {
	"t": lambda out: out["t"],
	"u_dot": lambda out: out["u_dot"],
	"v": lambda out: out["v"],
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

# =========================================================
# SINGLE SIMULATION
# =========================================================
def run_single_simulation(
	index,
	sweep_entry,
	base_params,
	fe_params,
	dt, t_end, f0, f1,
	output_spec,
	out_dir
):

	try:
		params = sweep_spec.apply(base_params, sweep_entry)
		amp = params["amp"]

		fe = FE_module.PiezoBeamFE(fe_params)

		def v_exc(t):
			return amp * np.sin(
				2 * np.pi * (f0 + t * (f1 - f0) / t_end) * t
			)

		ode = fe.build_ode_system(
			j_exc=30,
			K_c=params.get("K_c"),
			K_i=params["K_i"],
			K_p=params["K_p"],
			R_c=params["R_c"],
			v_exc=v_exc
		)

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
			do_spectral=True
		)

		data = {k: fn(out) for k, fn in output_spec.items()}

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
		"time": dict(dt=dt, t_end=t_end, f0=f0, f1=f1),
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
		dt, t_end, f0, f1,
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