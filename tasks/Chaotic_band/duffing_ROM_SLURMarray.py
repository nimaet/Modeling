import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import chirp

try:
	from joblib import Parallel, delayed
	HAS_JOBLIB = True
except ImportError:
	HAS_JOBLIB = False


# =========================================================
# Project imports
# =========================================================
script_dir = Path(__file__).resolve().parent
cwd = Path.cwd().resolve()
search_paths = [script_dir, *script_dir.parents, cwd, *cwd.parents]
project_root = next((p for p in search_paths if (p / "Modeling").is_dir()), None)
if project_root is None:
	raise RuntimeError("Could not find project root containing Modeling/.")
sys.path.insert(0, str(project_root))

from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models.ROM1 import ROM


# =========================================================
# Sweep definition
# =========================================================
SAVE_PREFIX = "duffing_ROM_freq"

# This is intentionally a simple 1D sweep. Change only these values for most runs.
SWEEP_NAME = "drive_freq_hz"
SWEEP_VALUES = np.linspace(2955.0, 2980.0, 20)


# =========================================================
# Model / circuit / excitation parameters
# =========================================================
N_MODES = 20
MODAL_DAMPING_RATIO = 0.0065

J_EXC = 30
AMP = 50.0
F0_HZ = 500.0
R_C = 1000.0
K_P = 0.015
K_I = 1820.0
K_C = 3e10

N_PERIODS_TOTAL = 10000
N_PERIODS_KEEP = 2000
POINTS_PER_PERIOD = 100
X_EVAL_POINTS = 100

RTOL = 1e-7
ATOL = 1e-9


# =========================================================
# Output directories
# =========================================================
sim_dat_dir = script_dir / "sim_dat"
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


def make_rom():
	params_rom = PiezoBeamParams(hp=0.252e-3, hs=0.51e-3)
	return ROM(
		params=params_rom,
		N=N_MODES,
		modal_damping_ratios=np.array([MODAL_DAMPING_RATIO] * N_MODES),
	)


def stroboscopic_sample(t, response, points_per_period, n_periods_keep):
	response = np.asarray(response)
	strobe_indices = np.arange(0, len(t), points_per_period)
	t_strobe = t[strobe_indices]

	if response.ndim == 1:
		response_strobe = response[strobe_indices]
	else:
		response_strobe = response[..., strobe_indices]

	return t_strobe[-n_periods_keep:], response_strobe[..., -n_periods_keep:]


def run_single_simulation(index, drive_freq_hz, out_dir):
	T = 1.0 / drive_freq_hz
	dt = T / POINTS_PER_PERIOD
	t_end = N_PERIODS_TOTAL * T
	t_eval = np.arange(0.0, t_end, dt)
	rom = make_rom()
	x_eval = np.linspace(0.0, rom.p.L_b, X_EVAL_POINTS)

	def v_exc(t):
		t1 = t_end / 10.0
		t = np.asarray(t)
		steady = AMP * np.cos(2.0 * np.pi * drive_freq_hz * t)
		ramp = AMP * chirp(t, f0=F0_HZ, f1=drive_freq_hz, t1=t1, method="linear")
		return np.where(t < t1, ramp, steady)

	out = rom.run_time_sim(
		v_exc=v_exc,
		j_exc=J_EXC,
		R_c=R_C,
		K_p=K_P,
		K_i=K_I,
		K_c=K_C,
		t_end=t_end,
		t_eval=t_eval,
		x_eval=x_eval,
		rtol=RTOL,
		atol=ATOL,
	)

	t_strobe, eta_poincare = stroboscopic_sample(
		out["t"], out["eta"], POINTS_PER_PERIOD, N_PERIODS_KEEP
	)
	_, eta_dot_poincare = stroboscopic_sample(
		out["t"], out["eta_dot"], POINTS_PER_PERIOD, N_PERIODS_KEEP
	)
	_, z_poincare = stroboscopic_sample(
		out["t"], out["z"], POINTS_PER_PERIOD, N_PERIODS_KEEP
	)
	_, v_poincare = stroboscopic_sample(
		out["t"], out["v"], POINTS_PER_PERIOD, N_PERIODS_KEEP
	)

	out_file = out_dir / f"sim_{index:05d}.npz"
	np.savez_compressed(
		out_file,
		index=np.array(index, dtype=np.int64),
		sweep_name=np.array(SWEEP_NAME),
		sweep_value=np.array(drive_freq_hz, dtype=float),
		drive_freq_hz=np.array(drive_freq_hz, dtype=float),
		amp=np.array(AMP, dtype=float),
		f0_hz=np.array(F0_HZ, dtype=float),
		K_p=np.array(K_P, dtype=float),
		K_i=np.array(K_I, dtype=float),
		K_c=np.array(K_C, dtype=float),
		R_c=np.array(R_C, dtype=float),
		N_modes=np.array(N_MODES, dtype=np.int64),
		n_periods_total=np.array(N_PERIODS_TOTAL, dtype=np.int64),
		n_periods_keep=np.array(N_PERIODS_KEEP, dtype=np.int64),
		points_per_period=np.array(POINTS_PER_PERIOD, dtype=np.int64),
		t=out["t"],
		x_eval=out["x_eval"],
		eta=out["eta"],
		eta_dot=out["eta_dot"],
		z=out["z"],
		v=out["v"],
		veloc=out["veloc"],
		freq=out["freq"],
		FRF=out["FRF"],
		Y=out["Y"],
		t_strobe=t_strobe,
		eta_poincare=eta_poincare,
		eta_dot_poincare=eta_dot_poincare,
		z_poincare=z_poincare,
		v_poincare=v_poincare,
	)

	return {
		"ok": True,
		"index": index,
		SWEEP_NAME: float(drive_freq_hz),
		"file": str(out_file.name),
	}


# =========================================================
# SLURM array setup
# =========================================================
SWEEP_VALUES = np.asarray(SWEEP_VALUES, dtype=float)

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

assigned_indices = list(range(array_id, len(SWEEP_VALUES), array_task_count))
print(f"Total simulations: {len(SWEEP_VALUES)}")
print(f"Array task {array_id:05d} of {array_task_count} has indices {assigned_indices}")

if not assigned_indices:
	print(f"Array task {array_id:05d} has no assigned simulations. Exiting.")
	raise SystemExit(0)

run_dir = sim_dat_dir / f"{SAVE_PREFIX}_{array_job_id}"
npz_dir = run_dir / "npz"
status_dir = run_dir / "status"

tmp_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
task_tmp_dir = tmp_root / f"{SAVE_PREFIX}_{array_job_id}" / f"task_{array_id:05d}"

run_dir.mkdir(parents=True, exist_ok=True)
npz_dir.mkdir(parents=True, exist_ok=True)
status_dir.mkdir(parents=True, exist_ok=True)
task_tmp_dir.mkdir(parents=True, exist_ok=True)

if array_id == 0:
	config = {
		"created_at": datetime.now().isoformat(),
		"save_prefix": SAVE_PREFIX,
		"sweep_name": SWEEP_NAME,
		"sweep_values": SWEEP_VALUES,
		"N_modes": N_MODES,
		"modal_damping_ratio": MODAL_DAMPING_RATIO,
		"j_exc": J_EXC,
		"amp": AMP,
		"f0_hz": F0_HZ,
		"R_c": R_C,
		"K_p": K_P,
		"K_i": K_I,
		"K_c": K_C,
		"n_periods_total": N_PERIODS_TOTAL,
		"n_periods_keep": N_PERIODS_KEEP,
		"points_per_period": POINTS_PER_PERIOD,
		"x_eval_points": X_EVAL_POINTS,
		"rtol": RTOL,
		"atol": ATOL,
	}

	with open(run_dir / "config.json", "w") as f:
		json.dump(to_jsonable(config), f, indent=2)

	with open(run_dir / "manifest.csv", "w") as f:
		f.write(f"index,{SWEEP_NAME},npz_file,status_file\n")
		for i, value in enumerate(SWEEP_VALUES):
			f.write(f"{i},{value:.16g},npz/sim_{i:05d}.npz,status/result_{i:05d}.json\n")


def run_and_store(index):
	drive_freq_hz = float(SWEEP_VALUES[index])
	local_dir = task_tmp_dir / f"sim_{index:05d}"
	local_dir.mkdir(parents=True, exist_ok=True)

	try:
		result = run_single_simulation(index, drive_freq_hz, local_dir)
		shutil.copy2(local_dir / result["file"], npz_dir / result["file"])
		result["file"] = str(Path("npz") / result["file"])
	except Exception as e:
		result = {
			"ok": False,
			"index": index,
			SWEEP_NAME: drive_freq_hz,
			"error": str(e),
			"exception": type(e).__name__,
		}

	with open(status_dir / f"result_{index:05d}.json", "w") as f:
		json.dump(to_jsonable(result), f, indent=2)

	return result


n_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
n_jobs = min(n_jobs, len(assigned_indices))

print(f"Task {array_id:05d} running {len(assigned_indices)} simulation(s) with n_jobs={n_jobs}")

if n_jobs == 1 or not HAS_JOBLIB:
	results = [run_and_store(i) for i in assigned_indices]
else:
	results = Parallel(n_jobs=n_jobs, prefer="processes")(
		delayed(run_and_store)(i) for i in assigned_indices
	)

ok_count = sum(1 for r in results if r.get("ok", False))
fail_count = len(results) - ok_count
print(f"Array task {array_id:05d} complete. success={ok_count}, failed={fail_count}")
print(f"Saved outputs to: {run_dir}")
