import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


# =========================================================
# LOCAL CONFIGURATION
# =========================================================
# Set these here instead of passing command-line arguments.
# Update ARRAY_JOB_ID when collecting a different sweep run.
SAVE_PREFIX = "Duffing_gain_sweep"
SIM_DAT_DIR = Path.cwd() / "sim_dat"
ARRAY_JOB_ID = "11043124"
RUN_DIR = None
PROGRESS_EVERY = 1


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
    return f"{obj:0.3e}" if isinstance(obj, float) else obj


def build_grid_from_config(config):
    from itertools import product

    from itertools import product

    sweep_spec = config.get("sweep_spec", [])
    keys = [p["key"] for p in sweep_spec]
    value_lists = [p.get("values", []) for p in sweep_spec]

    grid = []
    for combo in product(*value_lists):
        grid.append({k: v for k, v in zip(keys, combo)})

    return keys, grid

def resolve_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR).resolve()
def resolve_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR).resolve()

    if not ARRAY_JOB_ID:
    if not ARRAY_JOB_ID:
        raise RuntimeError(
            "Could not determine run directory. Set RUN_DIR or ARRAY_JOB_ID in the script."
            "Could not determine run directory. Set RUN_DIR or ARRAY_JOB_ID in the script."
        )

    return (SIM_DAT_DIR / f"{SAVE_PREFIX}_{ARRAY_JOB_ID}").resolve()
    return (SIM_DAT_DIR / f"{SAVE_PREFIX}_{ARRAY_JOB_ID}").resolve()


def main():
    run_dir = resolve_run_dir()
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

    successful = []
    failed = []

    status_files = sorted(status_dir.glob("result_*.json"))
    seen_indices = set()

    print(f"Collecting run: {run_dir}")
    print(f"Found {len(status_files)} status files, expected {expected_total} cases")

    for idx, status_file in enumerate(status_files, start=1):
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

        npz = np.load(npz_path, allow_pickle=True)
        sweep_entry = sweep_grid[index] if 0 <= index < expected_total else None

        successful.append(
            {
                "index": index,
                "sweep_entry": to_jsonable(sweep_entry),
                "params": npz["params"].item(),
                "data": npz["data"].item(),
            }
        )

        if idx % max(1, PROGRESS_EVERY) == 0 or idx == len(status_files):
        if idx % max(1, PROGRESS_EVERY) == 0 or idx == len(status_files):
            print(
                f"Progress: {idx}/{len(status_files)} status files | "
                f"success={len(successful)} failed={len(failed)}"
            )

    missing_count = 0
    for index in range(expected_total):
        if index not in seen_indices:
            missing_count += 1
            failed.append(
                {
                    "ok": False,
                    "index": index,
                    "error": "Missing status file for task",
                    "exception": "MissingStatus",
                }
            )

    if missing_count:
        print(f"Detected {missing_count} missing status files")

    pkl_data = {
        "meta": {
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            "status_files_found": len(status_files),
            "expected_total": expected_total,
        },
        "sweep": {
            "keys": sweep_keys,
            "grid": sweep_grid,
        },
        "results": successful,
        "failed": failed,
    }

    with open(run_dir / "results.pkl", "wb") as f:
        pickle.dump(pkl_data, f)

    with open(run_dir / "errors.json", "w") as f:
        json.dump(to_jsonable(failed), f, indent=2)

    print(f"Collected run: {run_dir}")
    print(f"Expected tasks: {expected_total}")
    print(f"Success: {len(successful)} | Failed: {len(failed)}")


if __name__ == "__main__":
    print("Starting nD sweep result collection...")
    main()
