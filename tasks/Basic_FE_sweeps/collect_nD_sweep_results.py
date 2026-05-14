import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import pickle
import os
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np


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
    sweep_spec = config.get("sweep_spec", [])
    keys = [p["key"] for p in sweep_spec]
    value_lists = [p.get("values", []) for p in sweep_spec]

    grid = []
    for combo in product(*value_lists):
        grid.append({k: v for k, v in zip(keys, combo)})

    return keys, grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate SLURM array sweep outputs into results.pkl and errors.json"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Absolute or relative run directory (e.g., sim_dat/NES_1234567)",
    )
    parser.add_argument(
        "--array-job-id",
        default=None,
        help="Array job id used in run folder naming (NES_<array_job_id>)",
    )
    parser.add_argument(
        "--save-prefix",
        default="NES",
        help="Run folder prefix used by workers",
    )
    parser.add_argument(
        "--sim-dat-dir",
        type=Path,
        default=Path.cwd() / "sim_dat",
        help="Parent directory containing run folders",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N status files processed",
    )
    return parser.parse_args()


def resolve_run_dir(args):
    if args.run_dir is not None:
        return args.run_dir.resolve()

    array_job_id = (
        args.array_job_id
        or os.environ.get("SWEEP_ARRAY_JOB_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
    )
    if not array_job_id:
        raise RuntimeError(
            "Could not determine run directory. Provide --run-dir or --array-job-id."
        )

    return (args.sim_dat_dir / f"{args.save_prefix}_{array_job_id}").resolve()


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args)
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

        if idx % max(1, args.progress_every) == 0 or idx == len(status_files):
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
