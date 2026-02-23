"""
Single sweep case runner for SLURM job arrays.
Runs one (amplitude, Kc) simulation and saves intermediate result.

Usage:
    python run_single_sweep_case.py <case_index> [output_dir]

Example:
    python run_single_sweep_case.py 0 ~/sim_dat/Alternating_Kc__1
    # Runs case 0 and saves to ~/sim_dat/Alternating_Kc__1/intermediate_npz/
"""

import sys
import os
from pathlib import Path
import numpy as np
import importlib

# =========================================================
# Setup paths
# =========================================================
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import Modeling
importlib.reload(Modeling)
from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models import FE_helpers
import Modeling.models.FE3 as FE_module

import numpy as np

# =========================================================
# Sweep Configuration (COPY from notebook Cell 1)
# =========================================================
SAVE_PREFIX = "Alternating_Kc_"

params_fe = PiezoBeamParams(
    hp=0.252e-3, hs=0.51e-3,
    d31=-1.48e-10, eps_r=1700,
)
params_fe.zeta_p = 0.0151 * 8
params_fe.zeta_q = 0.0392 * 10

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
    {
        "label": f"Kc_{i}",
        "kc_vec": np.array(
            [kc, kc] * (interface_idx // 2)
            + [kc, kc] * (15 - interface_idx // 2)
        )
    }
    for i, kc in enumerate(kc_magnitudes)
]

OUTPUT_SPEC = {
    "freq": lambda out: out["spectral"]["freq"],
    "X": lambda out: out["spectral"]["X"],
    "Y": lambda out: out["spectral"]["Y"],
    "u_dot": lambda out: out["spectral"]["FRF"],
}

# =========================================================
# Build sweep grid (same as notebook)
# =========================================================
sweep = [
    {
        "amp": amp,
        "kc_label": case["label"],
        "kc_vec": case["kc_vec"]
    }
    for amp in amp_list
    for case in Kc_cases
]


# =========================================================
# Simulation kernel (same as notebook)
# =========================================================
def run_single_simulation(
    amp, kc, kc_label,
    K_p, K_i, R_c,
    fe_params,
    dt, t_end, f0, f1,
    output_spec,
    output_dir
):
    """Run one case and save to output_dir/intermediate_npz/"""
    try:
        fe = FE_module.PiezoBeamFE(fe_params)

        def v_exc(t):
            return amp * np.sin(
                2 * np.pi * (f0 + t * (f1 - f0) / t_end) * t
            )

        ode = fe.build_ode_system(
            j_exc=30,
            K_c=kc,
            K_i=K_i,
            K_p=K_p,
            R_c=R_c,
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

        saved_data = {
            "amp": amp,
            "kc_label": kc_label,
            "kc_vec": kc
        }

        for name, extractor in output_spec.items():
            saved_data[name] = extractor(out)

        fname = f"amp_{amp:.3f}_{kc_label}.npz"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / fname, **saved_data)

        return {
            "ok": True,
            "amp": amp,
            "kc_label": kc_label,
            "file": fname,
            "msg": f"Case {amp:.3f} / {kc_label} completed successfully"
        }

    except Exception as e:
        return {
            "ok": False,
            "amp": amp,
            "kc_label": kc_label,
            "error": str(e),
            "exception": type(e).__name__,
            "msg": f"Case {amp:.3f} / {kc_label} failed: {type(e).__name__}"
        }


# =========================================================
# Main: run one case by index
# =========================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_index> [output_dir]")
        print(f"Total cases: {len(sweep)}")
        sys.exit(1)

    case_idx = int(sys.argv[1])
    
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2]) / "intermediate_npz"
    else:
        output_dir = Path.cwd() / "sim_dat" / SAVE_PREFIX / "intermediate_npz"

    if case_idx < 0 or case_idx >= len(sweep):
        print(f"ERROR: case_index {case_idx} out of range [0, {len(sweep)-1}]")
        sys.exit(1)

    case = sweep[case_idx]
    print(f"Running case {case_idx} / {len(sweep) - 1}")
    print(f"  Amplitude: {case['amp']:.3f}")
    print(f"  Kc label: {case['kc_label']}")
    print(f"  Output: {output_dir}")

    result = run_single_simulation(
        amp=case["amp"],
        kc=case["kc_vec"],
        kc_label=case["kc_label"],
        K_p=K_p,
        K_i=K_i,
        R_c=R_c,
        fe_params=params_fe,
        dt=dt,
        t_end=t_end,
        f0=f0,
        f1=f1,
        output_spec=OUTPUT_SPEC,
        output_dir=output_dir
    )

    print(f"Result: {result['msg']}")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
