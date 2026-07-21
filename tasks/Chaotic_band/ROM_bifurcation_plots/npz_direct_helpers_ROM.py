"""Shared npz-loading helpers for the ROM sweep plotting scripts in this
folder.

Assumes Basic_ROM_sweeps/generic_nD_sweep_SLURMarray_ROM.py has already been
run on the cluster and its `npz/sim_*.npz` + `config.json` have already been
downloaded locally (e.g. via Basic_ROM_sweeps/npz_direct_pipeline/
download_npz_results.py) -- this module does no downloading itself.

Each sim npz holds a small `params` dict (the resolved sweep point: K_c,
K_i, K_p, R_c, amp) and a `data` dict with FULL raw arrays for every mode,
patch, and x_eval node -- no stroboscopic/Poincare pre-sampling and no
representative-index restriction, since the sweep script's default
excitation is a single whole-band chirp (f0=500 -> f1=3500 Hz over
t_end=1s), not a constant-frequency lock:
    t, veloc, eta, eta_dot, z, v, freq, FRF, v_exc

Because the excitation chirps continuously and never repeats at a fixed
period, there is no well-defined "period" to stroboscopically sample --
Poincare sections / bifurcation diagrams in the classic sense (which need a
periodic response) aren't meaningful for this data. Plotting scripts here
stick to what full continuous time series support: spectral content (FRF),
continuous phase portraits, and backbone (peak-response-vs-Kc) curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LazyNPZ:
    """Load a single array from a compressed .npz archive only when requested."""

    def __init__(self, path, **metadata):
        self.path = str(path)
        self._metadata = dict(metadata)
        self._cache = {}
        with np.load(self.path, allow_pickle=True) as data:
            self._keys = tuple(data.files)

    @property
    def keys(self):
        return self._keys

    def __contains__(self, key):
        return key in self._metadata or key in self._keys

    def __getitem__(self, key):
        if key in self._metadata:
            return self._metadata[key]
        if key not in self._keys:
            raise KeyError(key)
        if key not in self._cache:
            with np.load(self.path, allow_pickle=True) as data:
                self._cache[key] = data[key]
        return self._cache[key]

    def get(self, key, default=None):
        return self[key] if key in self else default


def load_config(run_dir) -> dict:
    config_path = Path(run_dir) / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _npz_paths(run_dir):
    npz_dir = Path(run_dir) / "npz"
    if not npz_dir.exists():
        raise FileNotFoundError(f"No npz/ folder found under {run_dir}")
    paths = sorted(npz_dir.glob("sim_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No sim_*.npz files found under {npz_dir}. Check RUN_DIR and "
            "that the npz/ folder for this run was actually downloaded "
            "(see Basic_ROM_sweeps/npz_direct_pipeline/download_npz_results.py)."
        )
    return paths


def sweep_index(run_dir, sweep_key="K_c"):
    """Read only `params` from every sim npz; return a value-sorted index.

    Cheap: `params` is a small dict, `data` (the large arrays) is never
    touched here. Returns a list of {"value": float, "path": Path, "index":
    int} sorted ascending by params[sweep_key].
    """
    npz_paths = _npz_paths(run_dir)

    entries = []
    skipped = []
    for npz_path in npz_paths:
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                params = npz["params"].item()
            value = float(np.asarray(params[sweep_key]).ravel()[0])
            sim_index = int(npz_path.stem.split("_")[1])
        except Exception as exc:
            skipped.append((npz_path.name, repr(exc)))
            continue
        entries.append({"value": value, "path": npz_path, "index": sim_index})

    if not entries:
        preview = "\n".join(f"  {name}: {err}" for name, err in skipped[:5])
        raise ValueError(
            f"Found {len(npz_paths)} npz files under {run_dir}, but none had "
            f"a readable params['{sweep_key}']. First failures:\n{preview}"
        )

    if skipped:
        print(f"[sweep_index] Skipped {len(skipped)}/{len(npz_paths)} unreadable npz files:")
        for name, err in skipped[:10]:
            print(f"  {name}: {err}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    entries.sort(key=lambda e: e["value"])
    return entries


def make_lazy_results(run_dir, sweep_key="K_c"):
    """One LazyNPZ handle per sim, tagged with the swept value, value-sorted."""
    index = sweep_index(run_dir, sweep_key=sweep_key)
    return [LazyNPZ(e["path"], **{sweep_key: e["value"]}) for e in index]


def nearest_result(results, target_value, sweep_key="K_c"):
    """Pick the loaded result whose tagged sweep value is closest to target_value."""
    return min(results, key=lambda r: abs(r[sweep_key] - target_value))


def tail_window(t, window_seconds):
    """Boolean mask selecting the last `window_seconds` of a time series.

    Used in place of the old period-based stroboscopic tail slicing -- the
    chirp excitation here has no fixed period, so windows are defined in
    wall-clock seconds instead of "periods".
    """
    t = np.asarray(t)
    return t >= (t[-1] - window_seconds)
