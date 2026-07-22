"""Plot a bifurcation diagram + backbone amplitude curve from a reduced pickle.

Reads reduced_results__*.pkl produced by reduce_nD_sweep_results.py (via
download_reduced_results.py) -- never the raw npz's or a results.pkl. Run in
the VS Code Interactive Window, like plot_duffing_from_npz.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
RUN_DIR = Path("sim_dat/Duffing_gain_sweep_11043124")
VERSION = None  # None -> use the most recent entry in reduction_manifest.json
DOF_INDEX = 0  # position within reduction_spec.py's MECH_DOFS list to plot
SAVE_FIG = True
FIG_DIR = Path(__file__).parent / "fig_files"


def resolve_reduced_path(run_dir: Path, version: int | None) -> Path:
    if version is not None:
        path = run_dir / f"reduced_results__v{version:03d}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No reduced results for version={version}: {path}")
        return path

    manifest_path = run_dir / "reduction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No reduction_manifest.json under {run_dir}. Set VERSION explicitly, "
            "or run reduce_nD_sweep_results.py + download_reduced_results.py first."
        )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"{manifest_path} is empty.")

    latest = max(manifest, key=lambda m: m["version"])
    return run_dir / latest["file"]


def load_reduced(run_dir: Path, version: int | None = None) -> dict:
    path = resolve_reduced_path(run_dir, version)
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_bifurcation_arrays(pkl_data: dict, dof_index: int = 0):
    """Return (freqs, amps, bif_freqs, bif_amps) for one MECH_DOFS entry.

    response_amplitude/poincare_samples are arrays shaped over
    reduction_spec.py's MECH_DOFS (one value/row per selected mechanical
    DOF) -- `dof_index` picks which row to plot here (0 = first DOF in that
    list, not an absolute node index).

    freqs/amps: one entry per sim -- the backbone/FRF curve.
    bif_freqs/bif_amps: one entry per Poincare sample (many per sim) -- the
    scatter bifurcation diagram. A sim with a period-1 response contributes a
    tight cluster of near-identical points; period-doubling or chaos spreads
    across several distinct bands at that frequency.
    """
    freqs, amps = [], []
    bif_freqs, bif_amps = [], []

    for r in pkl_data["results"]:
        reduced = r["reduced"]
        freq = reduced["excitation_freq"]
        freqs.append(freq)
        amps.append(np.atleast_1d(reduced["response_amplitude"])[dof_index])

        samples = np.atleast_2d(reduced["poincare_samples"])[dof_index]
        bif_freqs.append(np.full(samples.shape, freq))
        bif_amps.append(samples)

    order = np.argsort(freqs)
    freqs = np.asarray(freqs)[order]
    amps = np.asarray(amps)[order]

    bif_freqs = np.concatenate(bif_freqs) if bif_freqs else np.array([])
    bif_amps = np.concatenate(bif_amps) if bif_amps else np.array([])

    return freqs, amps, bif_freqs, bif_amps


def main():
    pkl_data = load_reduced(RUN_DIR, VERSION)
    print(f"Sims: {len(pkl_data['results'])} succeeded, {len(pkl_data['failed'])} failed")
    print(f"mech_dofs={pkl_data['meta'].get('mech_dofs')} -- plotting DOF_INDEX={DOF_INDEX}")

    freqs, amps, bif_freqs, bif_amps = build_bifurcation_arrays(pkl_data, DOF_INDEX)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(freqs, amps, ".-", ms=4)
    axes[0].set_xlabel("Excitation frequency [Hz]")
    axes[0].set_ylabel("Response amplitude (u_dot, peak-to-peak/2)")
    axes[0].set_title("Backbone / FRF amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bif_freqs, bif_amps, ".", ms=1.5, alpha=0.5)
    axes[1].set_xlabel("Excitation frequency [Hz]")
    axes[1].set_ylabel("Poincare-sampled u_dot")
    axes[1].set_title("Bifurcation diagram")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if SAVE_FIG:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIG_DIR / f"{RUN_DIR.name}_bifurcation.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
