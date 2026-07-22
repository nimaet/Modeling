"""Reduce raw per-simulation npz outputs into one small consolidated file.

Runs on the cluster, alongside the raw `npz/sim_*.npz` files written by
generic_nD_sweep_SLURMarray.py -- NOT after downloading them. Each npz's
(large) `data` dict is read once here, boiled down to a handful of scalars
per sim via REDUCTION_SPEC (see reduction_spec.py), and only that small
result is written out. That output is what you download locally, instead of
the raw npz's.

This is a parallel, opt-in step -- it does not replace or modify
collect_nD_sweep_results.py, and does not touch the raw npz/ folder. Rerun it
any time you want to add/change a metric in reduction_spec.py; each run gets
its own sequential output file (reduced_results__v001.pkl, v002.pkl, ...),
so old reductions of the same run are never silently overwritten.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import pickle
import re
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from reduction_spec import (
    REDUCTION_SPEC,
    MECH_DOFS,
    ELEC_DOFS,
    STEADY_STATE_FRACTION,
    N_POINCARE_PERIODS,
)

# =========================================================
# LOCAL CONFIGURATION
# =========================================================
# Set these here instead of passing command-line arguments.
# Update ARRAY_JOB_ID when reducing a different sweep run.
SAVE_PREFIX = "Duffing_gain_sweep"
SIM_DAT_DIR = Path.cwd() / "sim_dat"
ARRAY_JOB_ID = "11043124"
RUN_DIR = None
N_JOBS = 4
PROGRESS_EVERY = 50

VERSION_RE = re.compile(r"reduced_results__v(\d+)$")


def resolve_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR).resolve()

    if not ARRAY_JOB_ID:
        raise RuntimeError(
            "Could not determine run directory. Set RUN_DIR or ARRAY_JOB_ID in the script."
        )

    return (SIM_DAT_DIR / f"{SAVE_PREFIX}_{ARRAY_JOB_ID}").resolve()


def build_grid_from_config(config):
    sweep_spec = config.get("sweep_spec", [])
    keys = [p["key"] for p in sweep_spec]
    value_lists = [p.get("values", []) for p in sweep_spec]

    grid = [dict(zip(keys, combo)) for combo in product(*value_lists)]
    return keys, grid


def next_version(run_dir: Path) -> int:
    """Next sequential version number for this run's reduced_results__vNNN.pkl.

    Scans existing files rather than trusting reduction_manifest.json alone,
    so a version is never reused even if the manifest was hand-edited or
    deleted.
    """
    versions = []
    for path in run_dir.glob("reduced_results__v*.pkl"):
        m = VERSION_RE.match(path.stem)
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0) + 1


def reduce_one(index, npz_path, sweep_entry):
    """Load one raw npz and apply every reducer in REDUCTION_SPEC to it.

    Results go straight into a pickle (not JSON), so reduced values, params,
    and sweep_entry are kept as native numpy/python objects -- no jsonable
    conversion needed or wanted here.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            params = npz["params"].item()
            data = npz["data"].item()

        reduced = {
            name: fn(data, params, sweep_entry)
            for name, fn in REDUCTION_SPEC.items()
        }

        return dict(
            ok=True,
            index=index,
            sweep_entry=sweep_entry,
            params=params,
            reduced=reduced,
        )

    except Exception as e:
        return dict(
            ok=False,
            index=index,
            sweep_entry=sweep_entry,
            error=str(e),
            exception=type(e).__name__,
        )


def main():
    run_dir = resolve_run_dir()
    config_path = run_dir / "config.json"
    npz_dir = run_dir / "npz"
    status_dir = run_dir / "status"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    sweep_keys, sweep_grid = build_grid_from_config(config)
    expected_total = len(sweep_grid)

    print(f"Reducing run: {run_dir}")
    print(f"Reducers: {list(REDUCTION_SPEC.keys())}")

    status_files = sorted(status_dir.glob("result_*.json"))
    seen_indices = set()
    failed = []
    tasks = []

    for status_file in status_files:
        with open(status_file, "r") as f:
            status = json.load(f)

        index = int(status.get("index", -1))
        if index >= 0:
            seen_indices.add(index)

        if not status.get("ok", False):
            failed.append(status)
            continue

        npz_path = npz_dir / f"sim_{index:05d}.npz"
        if not npz_path.exists():
            failed.append(
                {
                    "ok": False,
                    "index": index,
                    "error": f"Missing npz output: {npz_path.name}",
                    "exception": "MissingNPZ",
                }
            )
            continue

        sweep_entry = sweep_grid[index] if 0 <= index < expected_total else {}
        tasks.append((index, npz_path, sweep_entry))

    for index in range(expected_total):
        if index not in seen_indices:
            failed.append(
                {"ok": False, "index": index, "error": "Missing status file for task", "exception": "MissingStatus"}
            )

    print(f"Found {len(tasks)} reducible sims, {len(failed)} already-failed/missing")

    n_jobs = max(1, min(N_JOBS, len(tasks))) if tasks else 1
    if n_jobs == 1:
        raw_results = [reduce_one(*task) for task in tasks]
    else:
        raw_results = Parallel(n_jobs=n_jobs)(delayed(reduce_one)(*task) for task in tasks)

    results = []
    for i, r in enumerate(raw_results, start=1):
        if r["ok"]:
            results.append(r)
        else:
            failed.append(r)

        if i % max(1, PROGRESS_EVERY) == 0 or i == len(raw_results):
            print(f"Progress: {i}/{len(raw_results)} | success={len(results)} failed={len(failed)}")

    version = next_version(run_dir)
    out_path = run_dir / f"reduced_results__v{version:03d}.pkl"

    pkl_data = {
        "meta": {
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            "version": version,
            "reduction_spec_keys": list(REDUCTION_SPEC.keys()),
            "mech_dofs": MECH_DOFS,
            "elec_dofs": ELEC_DOFS,
            "steady_state_fraction": STEADY_STATE_FRACTION,
            "n_poincare_periods": N_POINCARE_PERIODS,
            "expected_total": expected_total,
        },
        "sweep": {"keys": sweep_keys, "grid": sweep_grid},
        "results": results,
        "failed": failed,
    }

    with open(out_path, "wb") as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest_path = run_dir / "reduction_manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    manifest.append(
        {
            "version": version,
            "file": out_path.name,
            "created_at": pkl_data["meta"]["created_at"],
            "reducers": list(REDUCTION_SPEC.keys()),
            "mech_dofs": MECH_DOFS,
            "elec_dofs": ELEC_DOFS,
            "steady_state_fraction": STEADY_STATE_FRACTION,
            "n_poincare_periods": N_POINCARE_PERIODS,
            "n_success": len(results),
            "n_failed": len(failed),
        }
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Updated manifest: {manifest_path}")
    print(f"Success: {len(results)} | Failed: {len(failed)}")


if __name__ == "__main__":
    print("Starting nD sweep reduction...")
    main()
