import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from numpy import pi
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Callable
import json
import pickle

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
# BASE MODEL PARAMETERS (NOT SWEPT)
# =========================================================
params_fe = PiezoBeamParams(
	hp=0.252e-3,
	hs=0.51e-3,
	d31=-1.48e-10,
	eps_r=1700,
)

params_fe.zeta_p = 0.0151 * 8
params_fe.zeta_q = 0.0392 * 10

interface_idx = 10
beta = 0.0

ki0 = 1800
ki1 = ki0 / (1 - beta)**2
ki2 = ki0 / (1 + beta)**2

K_i_base = np.array(
	[ki1, ki2] * (interface_idx // 2)
	+ [ki2, ki1] * (15 - interface_idx // 2)
	+ [ki2]
)

BASE_PARAMS = dict(
	K_p=0.015,
	K_i=K_i_base,
	R_c=1e3,
)

# =========================================================
# TIME / EXCITATION
# =========================================================
t_end = 0.01
f0 = 1000
f1 = 3000
dt = 1 / f1 / 50

# =========================================================
# DEFINE SWEEP (EDIT ONLY THIS BLOCK)
# =========================================================
amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125
kc_magnitudes = np.array([8e9, 2e10, 3e10])

sweep_spec = SweepSpec([
	SweepParam(
		key="amp",
		values=amp_list.tolist(),
		description="Excitation amplitude",
	),

	SweepParam(
		key="K_c",
		values=kc_magnitudes.tolist(),
		# target="K_c",
		builder=lambda kc: np.array(
			[kc, kc] * (interface_idx // 2)
			+ [kc, kc] * (15 - interface_idx // 2)
		),
		description="Nonlinear stiffness distribution",
	),
])

# =========================================================
# OUTPUT EXTRACTION
# =========================================================
OUTPUT_SPEC = {
	"freq": lambda out: out["spectral"]["freq"],
	"FRF":  lambda out: out["spectral"]["FRF"],
	"X":    lambda out: out["spectral"]["X"],
	"Y":    lambda out: out["spectral"]["Y"],
}

# =========================================================
# FILESYSTEM
# =========================================================
SAVE_PREFIX = "sweep"
sim_dat_dir = Path.cwd() / "sim_dat"
sim_dat_dir.mkdir(parents=True, exist_ok=True)

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
	return obj

def unique_dir(path: Path) -> Path:
	if not path.exists():
		path.mkdir(parents=True)
		return path

	i = 1
	while True:
		cand = path.with_name(f"{path.name}_{i}")
		if not cand.exists():
			cand.mkdir(parents=True)
			return cand
		i += 1

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
			newton_tol=1e-8,
			newton_maxiter=8,
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
# RUN DIRECTORY + CONFIG SAVE
# =========================================================
run_dir = unique_dir(sim_dat_dir / SAVE_PREFIX)
npz_dir = run_dir / "npz"
npz_dir.mkdir(parents=True)

CONFIG = {
	"created_at": datetime.now().isoformat(),
	"fe_params": to_jsonable(vars(params_fe)),
	"base_params": {
		k: to_jsonable(v)
		for k, v in BASE_PARAMS.items()
	},
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
}

with open(run_dir / "config.json", "w") as f:
	json.dump(to_jsonable(CONFIG), f, indent=2)

# =========================================================
# RUN SWEEP
# =========================================================
results = Parallel(n_jobs=22, verbose=10)(
	delayed(run_single_simulation)(
		i, s, BASE_PARAMS,
		params_fe,
		dt, t_end, f0, f1,
		OUTPUT_SPEC,
		npz_dir
	)
	for i, s in enumerate(SWEEP_GRID)
)

# =========================================================
# COLLECT RESULTS INTO PKL
# =========================================================
successful = []
failed = []

for r in results:
	if r["ok"]:
		npz = np.load(npz_dir / f"sim_{r['index']:05d}.npz", allow_pickle=True)
		successful.append({
			"params": npz["params"].item(),
			"data": npz["data"].item(),
		})
	else:
		failed.append(r)

PKL_DATA = {
	"meta": {
		"run_dir": str(run_dir),
		"created_at": CONFIG["created_at"],
	},
	"sweep": {
		"keys": sweep_spec.keys,
		"grid": SWEEP_GRID,
	},
	"results": successful,
	"failed": failed,
}

with open(run_dir / "results.pkl", "wb") as f:
	pickle.dump(PKL_DATA, f)

with open(run_dir / "errors.json", "w") as f:
	json.dump(to_jsonable(failed), f, indent=2)

print(f"Completed: {len(successful)} success, {len(failed)} failed")
print(f"Saved to: {run_dir}")