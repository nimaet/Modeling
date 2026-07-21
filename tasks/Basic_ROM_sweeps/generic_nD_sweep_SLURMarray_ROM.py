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
from Modeling.models.ROM1 import ROM

# =========================================================
# SWEEP SPEC INFRASTRUCTURE
# =========================================================
# Identical to generic_nD_sweep_SLURMarray.py (Basic_FE_sweeps) -- kept as a
# self-contained copy here on purpose, matching this repo's existing
# convention of duplicating this infra per sweep script rather than
# factoring it into a shared module.
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
SAVE_PREFIX = "ROM_Duffing_freq_sweep"
sim_dat_dir = Path.cwd() / "sim_dat"
sim_dat_dir.mkdir(parents=True, exist_ok=True)

params_rom = PiezoBeamParams(
	hp=0.252e-3,
	hs=0.51e-3,
)

N_modes = 20

# Build the ROM once, at module scope. Unlike Basic_FE_sweeps' PiezoBeamFE
# (rebuilt per simulation), ROM(...) does an eigenvalue solve plus mode
# shape/coupling integrals that only depend on params_rom/N_modes -- never
# on the swept electrical gains (K_c, K_i, K_p, R_c). Rebuilding it per grid
# point would redo that expensive setup for nothing, so it's built once here
# and shared (read-only) across every simulation and joblib worker.
rom = ROM(params=params_rom, N=N_modes, modal_damping_ratios=np.array([0.0065] * N_modes))
x_eval = np.linspace(0, rom.p.L_b, 100)

print('First natural frequency [Hz]:', rom.omega[0] / (2 * np.pi))

# Note: unlike the FE model, ROM1's odefun() applies K_c/K_i/K_p/R_c as
# plain scalars broadcast across every piezo patch (no per-interface
# alternating array like Basic_FE_sweeps' Ki_builder/Kc_builder). So the
# swept Kc values below are used directly, with no builder function needed.
BASE_PARAMS = dict(
	K_p=0.015,
	K_i=1820,
	K_c=3e10,
	R_c=1e3,
	amp=50,
)

# =========================================================
# TIME / EXCITATION
# =========================================================
t_end = 1
f0 = 500
f1 = 3500
dt = 1 / max(f0, f1) / 50

rtol = 1e-7
atol = 1e-9

# =========================================================
# DEFINE SWEEP
# =========================================================
# Same Kc grid as Basic_FE_sweeps/generic_nD_sweep_SLURMarray.py, for a
# direct ROM-vs-FE comparison at matched sweep points.
Kc_soft = -np.array([5.0e8, 1.0e9, 2.0e9, 4.00e9, 8.00e9, 1.0e10, 1.20e10, 1.30e10, 1.4e10, 1.6e10, 1.8e10, 2.0e10])
Kc_hard = np.array([5e8, 1e9, 2.0e9, 4.0e9, 8.00e9, 1.20e10, 1.50e10, 1.80e10, 2.50e10, 2.75e10, 3.00e10])

sweep_spec = SweepSpec([
	SweepParam(
		key="K_c",
		values=Kc_soft.tolist()[::-1] + Kc_hard.tolist(),
		target="K_c",
		description="nonlinear inductance",
	),

	# Excitation frequency sweep (single-tone): one "freq" value is
	# broadcast to both f0 and f1, so v_exc's chirp formula collapses
	# to a constant-frequency sine at that value. When this param is
	# absent, f0/f1 fall back to the module-level chirp (f0 -> f1
	# over t_end), so the default sweep is unaffected.
	# SweepParam(
	# 	key="freq",
	# 	values=np.linspace(1000, 3000, 11).tolist(),
	# 	target=["f0", "f1"],
	# 	description="Excitation frequency (single-tone)",
	# ),
])

# =========================================================
# OUTPUT EXTRACTION
# =========================================================
OUTPUT_SPEC = {
	"t": lambda out: out["t"],
	"veloc": lambda out: out["veloc"],
	"eta": lambda out: out["eta"],
	"eta_dot": lambda out: out["eta_dot"],
	"z": lambda out: out["z"],
	"v": lambda out: out["v"],
	"FRF": lambda out: out["FRF"],
	"freq": lambda out: out["freq"],
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


dt = 1 / _max_swept_freq() / 50

# =========================================================
# SINGLE SIMULATION
# =========================================================
def run_single_simulation(
	index,
	sweep_entry,
	base_params,
	dt, t_end, f0, f1,
	output_spec,
	out_dir
):

	try:
		params = sweep_spec.apply(base_params, sweep_entry)
		amp = params["amp"]
		f0_val = params.get("f0", f0)
		f1_val = params.get("f1", f1)

		def v_exc(t):
			return amp * np.sin(
				2 * np.pi * (f0_val + t * (f1_val - f0_val) / t_end) * t
			)

		t_eval = np.arange(0, t_end, dt)

		out = rom.run_time_sim(
			v_exc=v_exc,
			j_exc=30,
			R_c=params["R_c"],
			K_p=params["K_p"],
			K_i=params["K_i"],
			K_c=params.get("K_c"),
			t_end=t_end,
			t_eval=t_eval,
			x_eval=x_eval,
			rtol=rtol,
			atol=atol,
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
		"rom_params": to_jsonable(vars(params_rom)),
		"N_modes": N_modes,
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
