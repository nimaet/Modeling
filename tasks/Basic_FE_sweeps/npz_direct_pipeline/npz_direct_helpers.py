"""Load per-simulation npz outputs directly, without collect_nD_sweep_results.py.

generic_nD_sweep_SLURMarray.py writes one `npz/sim_{index:05d}.npz` per
simulation, each holding a small `params` dict and a large `data` dict
(t, u_dot, v, freq, FRF, ...). collect_nD_sweep_results.py normally loads
every npz's `params` and `data` into memory to build one combined
`results.pkl`, which plot_duffing.py then loads back into memory in full
before picking a couple of Kc groups to actually plot.

This module skips that combined pickle. It groups sims by Kc in two passes:

  1. `npz_kc_index` reads only `params` (small) from every sim to build the
     Kc/amp grouping and index -- `data` is never touched here.
  2. `load_npz_results_by_kc` reads `data` (large) only for the Kc labels
     you actually select for plotting.

The returned `results_by_kc` shape matches
`duffing_plot_helpers.to_results_by_kc` exactly, so downstream plotting code
does not need to change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_hashable(value):
    """Convert arrays/lists to hashable keys for grouping Kc values."""
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).ravel().tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def kc_scalar(kc_value) -> float:
    """Use the first Kc entry as the scalar coordinate/label for a Kc vector."""
    arr = np.asarray(kc_value).ravel()
    if arr.size == 0:
        return np.nan
    return float(arr[0])


def choose_kc_labels(labels: list[str], selection):
    """Choose Kc labels by zero-based index into the native (first-seen) order."""
    if not labels:
        raise ValueError("No Kc groups found.")

    if selection is None:
        return labels

    chosen = []
    for item in selection:
        if not isinstance(item, (int, np.integer)):
            raise TypeError("KC_SELECTION must contain integer indices only.")
        chosen.append(labels[int(item)])

    if not chosen:
        raise ValueError("KC_SELECTION must resolve to at least one Kc label.")

    return chosen


def load_config(run_dir) -> dict:
    config_path = Path(run_dir) / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _npz_paths(run_dir):
    npz_dir = Path(run_dir) / "npz"
    if not npz_dir.exists():
        raise FileNotFoundError(f"No npz/ folder found under {run_dir}")
    return sorted(npz_dir.glob("sim_*.npz"))


def npz_kc_index(run_dir):
    """Pass 1 (cheap): read only `params` from every sim npz and group by Kc.

    Walks npz files in ascending sim-index order, which is the same order
    collect_nD_sweep_results.py's `results` list ended up in (it iterates
    sorted, zero-padded `status/result_*.json` files) -- so KC_SELECTION
    indices mean the same thing here as they did against a results.pkl.

    Returns a list of group dicts in that native order:
        {"label": str, "kc_vec": np.ndarray, "rows": [(amp, npz_path), ...]}
    """
    npz_paths = _npz_paths(run_dir)
    if not npz_paths:
        raise FileNotFoundError(
            f"No sim_*.npz files found under {Path(run_dir) / 'npz'}. "
            "Check RUN_DIR and that download_npz_results.py actually "
            "transferred the npz/ folder for this run (e.g. the include "
            "pattern/shard didn't match, or the run only has a results.pkl "
            "from the old collect_nD_sweep_results.py workflow)."
        )

    grouped = {}
    order = []
    skipped = []

    for npz_path in npz_paths:
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                params = npz["params"].item()
            kc_value = params["K_c"]
            amp_value = float(params["amp"])
        except Exception as exc:
            skipped.append((npz_path.name, repr(exc)))
            continue  # unreadable / incomplete sim output -- skip like a failed sim

        group_key = make_hashable(kc_value)
        if group_key not in grouped:
            kc_vec = np.atleast_1d(np.asarray(kc_value))
            grouped[group_key] = {
                "label": f"Kc={kc_scalar(kc_vec):.2e}",
                "kc_vec": kc_vec,
                "rows": [],
            }
            order.append(group_key)

        grouped[group_key]["rows"].append((amp_value, npz_path))

    if not grouped:
        preview = "\n".join(f"  {name}: {err}" for name, err in skipped[:5])
        raise ValueError(
            f"Found {len(npz_paths)} npz files under {run_dir}, but none had "
            f"a readable params['K_c']/params['amp']. First failures:\n{preview}"
        )

    if skipped:
        print(
            f"[npz_kc_index] Skipped {len(skipped)}/{len(npz_paths)} npz files "
            "that failed to load or were missing K_c/amp:"
        )
        for name, err in skipped[:10]:
            print(f"  {name}: {err}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return [grouped[key] for key in order]


def collect_unique_amps_from_index(kc_index) -> list[float]:
    """Amplitudes across all Kc groups -- cheap, from the params-only index."""
    amps = set()
    for group in kc_index:
        for amp, _ in group["rows"]:
            amps.add(float(amp))
    return sorted(amps)


def load_npz_results_by_kc(kc_index, selected_labels):
    """Pass 2: load `data` (the large arrays) only for the selected Kc labels.

    Returns the same {label: {kc_label, kc_vec, amps, data}} shape as
    duffing_plot_helpers.to_results_by_kc.
    """
    selected = set(selected_labels)
    results_by_kc = {}

    for group in kc_index:
        if group["label"] not in selected:
            continue

        rows = sorted(group["rows"], key=lambda row: row[0])
        amps, freq, frf, u_dot, y, v, t = [], [], [], [], [], [], []

        for amp_value, npz_path in rows:
            with np.load(npz_path, allow_pickle=True) as npz:
                data = npz["data"].item()
            amps.append(amp_value)
            freq.append(data.get("freq"))
            frf.append(data.get("FRF"))
            u_dot.append(data.get("u_dot"))
            y.append(data.get("Y"))
            v.append(data.get("v"))
            t.append(data.get("t"))

        results_by_kc[group["label"]] = {
            "kc_label": group["label"],
            "kc_vec": group["kc_vec"],
            "amps": amps,
            "data": {
                "freq": freq,
                "FRF": frf,
                "u_dot": u_dot,
                "Y": y,
                "v": v,
                "t": t,
            },
        }

    return results_by_kc
