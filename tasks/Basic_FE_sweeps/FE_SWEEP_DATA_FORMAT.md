# FE Duffing Sweep Data Format

## Purpose

This document describes the pickle data structures used by `Plot_Duffing.ipynb` to load, inspect, and plot FE Duffing frequency-response sweep results. It is meant as a practical contract for future scripts that need to save results in the same format, or load existing results for plotting.

The notebook supports two related formats:

- **Preferred/new format:** a record-list pickle with top-level `meta`, `sweep`, `results`, and `failed`, following the style in `archived_scripts/2D_sweep_3.py`.
- **Legacy plotting format:** a grouped pickle with top-level `results_by_kc`, used by existing files such as `sim_dat/softening/sweep_1/softening.pkl` and older plotting notebooks.

Future simulation code should write the preferred/new format. Plotting code may convert it into the legacy `results_by_kc` view because many existing plotting cells expect that grouped shape.

## Folder Layout

Typical run folders live under:

```text
sim_dat/<run_name>/
```

Common files are:

```text
sim_dat/<run_name>/results.pkl          # preferred/new aggregate pickle
sim_dat/<run_name>/<run_name>.pkl       # legacy aggregate pickle, or old naming style
sim_dat/<run_name>/config.json          # optional run configuration
sim_dat/<run_name>/errors.json          # optional failures for preferred format
sim_dat/<run_name>/<run_name>_errors.json
sim_dat/<run_name>/npz/                 # optional intermediate per-simulation files
sim_dat/<run_name>/intermediate_npz/    # optional legacy intermediate files
sim_dat/<run_name>/plots/               # generated figures
```

`Plot_Duffing.ipynb` chooses a run with `RUN_FOLDER`, resolves it relative to `Path.cwd() / "sim_dat"`, finds `.pkl` files in that folder, and loads the last sorted pickle candidate.

## Preferred Save Format

Use this format for new scripts. It stores one result record per successful sweep point.

Top-level pickle structure:

```python
PKL_DATA = {
    "meta": {
        "run_dir": str(run_dir),
        "created_at": created_at_iso_string,
        # optional: code version, notes, hostname, script name, etc.
    },
    "sweep": {
        "keys": ["amp", "K_c"],
        "grid": [
            {"amp": 6.25, "K_c": 8e9},
            {"amp": 12.5, "K_c": 8e9},
            # one dict per intended simulation
        ],
    },
    "results": [
        {
            "index": 0,
            "sweep_entry": {"amp": 6.25, "K_c": 8e9},
            "params": {
                "amp": 6.25,
                "K_c": np.array([...]),
                "K_i": np.array([...]),
                "K_p": 0.015,
                "R_c": 1000.0,
            },
            "data": {
                "freq": np.ndarray,  # shape: (n_freq,)
                "FRF": np.ndarray,   # shape: (n_freq,), preferred plot curve
                "X": np.ndarray,     # shape: (n_freq,), optional
                "Y": np.ndarray,     # shape: (n_freq, n_nodes), optional
            },
        },
    ],
    "failed": [
        {
            "ok": False,
            "index": 3,
            "sweep_entry": {"amp": 25.0, "K_c": 8e9},
            "error": "...",
            "exception": "...",
        },
    ],
}
```

Required top-level keys:

- `meta`: dictionary with run metadata. `run_dir` and `created_at` are recommended.
- `sweep`: dictionary describing the sweep space.
- `sweep["keys"]`: ordered list of swept parameter names, for example `['amp', 'K_c']`.
- `sweep["grid"]`: list of intended sweep entries. `sweep["grid"][i]` should describe result index `i`.
- `results`: list of successful result records.
- `failed`: list of failed simulation records. Use an empty list if all simulations succeed.

Required fields for each `results[i]`:

- `index`: integer index into `sweep["grid"]`.
- `sweep_entry`: the sweep values for this run, such as `amp` and `K_c`.
- `params`: resolved model parameters used by the FE solver. If `K_c` is expanded from a scalar into a vector, store the vector here.
- `data`: output arrays used for plotting and inspection.

Preferred `data` fields:

- `freq`: frequency vector in Hz, shape `(n_freq,)`.
- `FRF`: frequency response magnitude/curve, shape `(n_freq,)`.
- `X`: optional spectral state, observed shape `(n_freq,)`.
- `Y`: optional spatial spectral response, observed shape `(n_freq, n_nodes)`.
- `u_dot`: optional velocity response. If `FRF` is absent, plotting can derive a curve from `u_dot`.
- `v`, `t`: optional time-domain data when needed.

## Legacy Compatibility Format

Existing plot code often expects `results_by_kc`, where data are grouped first by Kc case, then by amplitude index.

Top-level structure:

```python
LEGACY_DATA = {
    "Kc_labels": ["Kc_0", "Kc_1", "Kc_2"],
    "results_by_kc": {
        "Kc_0": {
            "kc_label": "Kc_0",
            "kc_vec": np.ndarray,  # shape: (n_kc_entries,)
            "amps": [6.25, 12.5, 18.75],
            "data": {
                "freq": [np.ndarray, np.ndarray, np.ndarray],
                "FRF": [np.ndarray, np.ndarray, np.ndarray],
                "X": [np.ndarray, np.ndarray, np.ndarray],
                "Y": [np.ndarray, np.ndarray, np.ndarray],
            },
        },
    },
    "failed": [],
    "run_dir": str(run_dir),
    # optional: K_p, K_i, amp_list, R_c, dt, f0, f1, t_end
}
```

Observed example from `sim_dat/softening/sweep_1/softening.pkl`:

- Top-level keys: `Kc_labels`, `results_by_kc`, `failed`, `run_dir`, `K_p`, `K_i`.
- `Kc_labels`: `['Kc_0', 'Kc_1', 'Kc_2', 'Kc_3', 'Kc_4']`.
- `results_by_kc['Kc_0']['amps']`: 7 amplitudes, from `6.25` to `50.0` V.
- `kc_vec`: shape `(30,)`.
- First amplitude entry shapes:
  - `freq`: `(75001,)`, `float64`.
  - `FRF`: `(75001,)`, `float64`.
  - `X`: `(75001,)`, `complex128`.
  - `Y`: `(75001, 157)`, `complex128`.

For each Kc block, every list in `block["data"]` should have the same length as `block["amps"]`. Entry `j` in each list corresponds to amplitude `block["amps"][j]`.

## Required Fields For Plotting

To make a run plottable by `Plot_Duffing.ipynb`, provide:

- An amplitude value, named `amp`, either in `result["params"]`, `result["sweep_entry"]`, or the corresponding `sweep["grid"]` entry.
- A Kc value, named `K_c`, either in `result["params"]`, `result["sweep_entry"]`, or the corresponding `sweep["grid"]` entry.
- `data["freq"]`, with shape `(n_freq,)`.
- A response curve with the same frequency length. Preferred order:
  - `data["FRF"]`, used directly.
  - `data["u_dot"]`, converted to `mean(abs(u_dot), axis=1)` if 2D, otherwise `abs(u_dot).squeeze()`.
  - `data["Y"]`, converted similarly if no `FRF` or `u_dot` exists.

For spatial plots, also provide:

- `data["Y"]`, shape `(n_freq, n_nodes)`, where rows align with `freq`.

## Minimal Save Example

```python
import pickle
from datetime import datetime
import numpy as np

run_dir = Path("sim_dat/my_run")
run_dir.mkdir(parents=True, exist_ok=True)

sweep_grid = []
results = []
failed = []

for index, (amp, kc_scalar) in enumerate(parameter_pairs):
    sweep_entry = {"amp": float(amp), "K_c": float(kc_scalar)}
    sweep_grid.append(sweep_entry)

    try:
        kc_vec = build_kc_vector(kc_scalar)
        out = run_fe_solver(amp=amp, K_c=kc_vec)

        results.append({
            "index": index,
            "sweep_entry": sweep_entry,
            "params": {
                "amp": float(amp),
                "K_c": kc_vec,
                "K_i": K_i,
                "K_p": float(K_p),
                "R_c": float(R_c),
            },
            "data": {
                "freq": out["spectral"]["freq"],
                "FRF": out["spectral"]["FRF"],
                "X": out["spectral"].get("X"),
                "Y": out["spectral"].get("Y"),
            },
        })
    except Exception as exc:
        failed.append({
            "ok": False,
            "index": index,
            "sweep_entry": sweep_entry,
            "error": str(exc),
            "exception": type(exc).__name__,
        })

pkl_data = {
    "meta": {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
    },
    "sweep": {
        "keys": ["amp", "K_c"],
        "grid": sweep_grid,
    },
    "results": results,
    "failed": failed,
}

with open(run_dir / "results.pkl", "wb") as f:
    pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## Minimal Load And Plot Example

```python
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

with open("sim_dat/my_run/results.pkl", "rb") as f:
    pkl = pickle.load(f)


def make_hashable(value):
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).ravel().tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def get_record_value(result, idx, key):
    if key in result.get("params", {}):
        return result["params"][key]
    if key in result.get("sweep_entry", {}):
        return result["sweep_entry"][key]
    entry = pkl.get("sweep", {}).get("grid", [])[idx]
    return entry[key]


def build_curve(data):
    if data.get("FRF") is not None:
        return np.asarray(data["FRF"])
    if data.get("u_dot") is not None:
        u_dot = np.asarray(data["u_dot"])
        return np.mean(np.abs(u_dot), axis=1) if u_dot.ndim == 2 else np.abs(u_dot).squeeze()
    if data.get("Y") is not None:
        y = np.asarray(data["Y"])
        return np.mean(np.abs(y), axis=1) if y.ndim == 2 else np.abs(y).squeeze()
    return None


# Reuse legacy data directly when present.
if "results_by_kc" in pkl:
    results_by_kc = pkl["results_by_kc"]
else:
    grouped = {}
    for idx, result in enumerate(pkl.get("results", [])):
        amp = float(get_record_value(result, idx, "amp"))
        kc = get_record_value(result, idx, "K_c")
        group_key = make_hashable(kc)

        if group_key not in grouped:
            kc_arr = np.atleast_1d(np.asarray(kc))
            grouped[group_key] = {
                "label": f"Kc={float(kc_arr.ravel()[0]):.2e}",
                "kc_vec": kc_arr,
                "rows": [],
            }

        grouped[group_key]["rows"].append({
            "amp": amp,
            "data": result.get("data", {}),
        })

    results_by_kc = {}
    for group in grouped.values():
        rows = sorted(group["rows"], key=lambda row: row["amp"])
        results_by_kc[group["label"]] = {
            "kc_label": group["label"],
            "kc_vec": group["kc_vec"],
            "amps": [row["amp"] for row in rows],
            "data": {
                "freq": [row["data"].get("freq") for row in rows],
                "FRF": [build_curve(row["data"]) for row in rows],
                "Y": [row["data"].get("Y") for row in rows],
            },
        }

# Plot one Kc group.
label = sorted(results_by_kc.keys())[0]
block = results_by_kc[label]

for amp, freq, frf in zip(block["amps"], block["data"]["freq"], block["data"]["FRF"]):
    freq = np.asarray(freq)
    frf = np.asarray(frf).squeeze()
    if len(freq) == len(frf):
        plt.semilogy(freq, np.abs(frf), label=f"A={amp:.1f} V")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF magnitude")
plt.title(label)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
```

## Validation Checklist

Before saving a run, check the following:

- `len(sweep["grid"]) == len(results) + len(failed)` if every intended simulation is accounted for.
- Every successful result has `index`, `sweep_entry`, `params`, and `data`.
- Each result has an amplitude value `amp` and a Kc value `K_c` or `kc_vec`.
- `freq` is one-dimensional and in Hz.
- The plotted curve has the same length as `freq`.
- If `Y` is stored, `Y.shape[0] == len(freq)` and `Y.shape[1] == n_nodes`.
- In legacy `results_by_kc`, every `block["data"][key]` list has length `len(block["amps"])`.
- Amplitudes within each Kc group are sorted before plotting if line/color order matters.
- Failed simulations are recorded in `failed`; do not silently drop them.
- Large arrays stay as NumPy arrays in pickle. JSON sidecar files should store only metadata or JSON-converted summaries.

## Task Brief For Another AI

You are documenting and maintaining the FE Duffing sweep data format used by `Plot_Duffing.ipynb`. The notebook loads `.pkl` files from `sim_dat/<run_folder>`. It supports old pickle files with `results_by_kc` and newer pickle files with `meta`, `sweep`, `results`, and `failed`.

When writing new code, prefer the newer record-list format:

```python
{
    "meta": {...},
    "sweep": {"keys": [...], "grid": [...]},
    "results": [{"index": i, "sweep_entry": {...}, "params": {...}, "data": {...}}],
    "failed": [...],
}
```

When plotting, either read legacy `results_by_kc` directly or convert the new `results` list into grouped blocks keyed by Kc label. Plotting expects amplitude-indexed arrays grouped by Kc. The primary plot fields are `freq` and `FRF`; if `FRF` is absent, derive the curve from `u_dot` or `Y` using an absolute-value magnitude. Spatial plots use `Y` with shape `(n_freq, n_nodes)`.

Keep this document practical: include required keys, optional keys, array shapes, a save example, a load/plot example, and validation checks. Do not remove legacy compatibility unless all notebooks that read `results_by_kc` have been updated.
