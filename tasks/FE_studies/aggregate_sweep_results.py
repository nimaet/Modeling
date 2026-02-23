"""
Post-processing script: aggregate intermediate NPZ files into final results.
Runs AFTER all SLURM array jobs complete.

Usage:
    python aggregate_sweep_results.py <run_folder>

Example:
    python aggregate_sweep_results.py ~/sim_dat/Alternating_Kc_sweep_20260223_120000
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

# =========================================================
# Configuration (same as notebook/single_case runner)
# =========================================================
SAVE_PREFIX = "Alternating_Kc_"

K_p = 0.032
interface_idx = 10
beta = 0
ki0 = 2000
ki1 = ki0 / (1 - beta)**2
ki2 = ki0 / (1 + beta)**2

K_i = np.array([ki1, ki2] * (interface_idx // 2)
               + [ki2, ki1] * (15 - interface_idx // 2)
               + [ki2])

R_c = 1e3
t_end = 0.01
f0 = 1000
f1 = 3000
dt = 1 / f1 / 50

amp_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125

kc_magnitudes = [1e10, 2e10]
Kc_cases = [
    {"label": f"Kc_{i}"} for i, kc in enumerate(kc_magnitudes)
]

OUTPUT_SPEC = {
    "freq": None,
    "X": None,
    "Y": None,
    "u_dot": None,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <run_folder>")
        print(f"Example: python {sys.argv[0]} ~/sim_dat/Alternating_Kc_sweep_20260223_120000")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser()
    intermediate_dir = run_dir / "intermediate_npz"

    if not run_dir.exists():
        print(f"ERROR: run folder not found: {run_dir}")
        sys.exit(1)

    if not intermediate_dir.exists():
        print(f"ERROR: intermediate folder not found: {intermediate_dir}")
        sys.exit(1)

    # =========================================================
    # Load intermediate NPZ files
    # =========================================================
    npz_files = sorted(intermediate_dir.glob("amp_*.npz"))
    print(f"Found {len(npz_files)} intermediate NPZ files in {intermediate_dir}")

    results_by_kc = defaultdict(lambda: {
        "kc_label": None,
        "kc_vec": None,
        "amps": [],
        "data": defaultdict(list)
    })

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)

        label = data["kc_label"].item()
        block = results_by_kc[label]

        block["kc_label"] = label
        block["kc_vec"] = data["kc_vec"]
        block["amps"].append(float(data["amp"]))

        for key in OUTPUT_SPEC.keys():
            if key in data:
                block["data"][key].append(data[key])

    print(f"Aggregated {len(results_by_kc)} Kc cases")
    for label, block in results_by_kc.items():
        print(f"  {label}: {len(block['amps'])} amplitudes")

    # =========================================================
    # Save aggregated results
    # =========================================================
    save_dict = {
        "amp_list": amp_list,
        "K_p": K_p,
        "K_i": K_i,
        "R_c": R_c,
        "dt": dt,
        "f0": f0,
        "f1": f1,
        "t_end": t_end,
        "output_spec_keys": list(OUTPUT_SPEC.keys()),
        "results_by_kc": dict(results_by_kc),
        "run_dir": str(run_dir)
    }

    pickle_path = run_dir / f"{SAVE_PREFIX}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved pickle: {pickle_path}")

    # =========================================================
    # Save config JSON
    # =========================================================
    config_dict = {
        "amp_list": amp_list.tolist(),
        "kc_magnitudes": kc_magnitudes,
        "Kc_cases": [{"label": case["label"]} for case in Kc_cases],
        "K_p": float(K_p),
        "K_i": K_i.tolist(),
        "R_c": float(R_c),
        "dt": float(dt),
        "f0": float(f0),
        "f1": float(f1),
        "t_end": float(t_end),
        "interface_idx": int(interface_idx),
        "beta": float(beta),
        "output_spec_keys": list(OUTPUT_SPEC.keys()),
        "run_dir": str(run_dir),
        "num_intermediate_files": len(npz_files)
    }

    config_path = run_dir / f"{SAVE_PREFIX}.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Saved config: {config_path}")

    print("\nAggregation complete!")
    print(f"Load in notebook with: RUN_FOLDER = '{run_dir.name}'")


if __name__ == "__main__":
    main()
