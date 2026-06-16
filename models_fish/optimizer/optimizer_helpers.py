"""Shared constants, output metrics, and phase optimizers."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from ..model.model_helpers import (
    reduced_to_full_displacement_nodes,
    tip_reduced_index,
    trapezoid_node_weights,
)

from concurrent.futures import ThreadPoolExecutor

def parallel_map(func, items, *, workers=1, min_tasks=4):
    items = list(items)
    if workers is None or workers <= 1 or len(items) < min_tasks:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(func, items))

# -----------------------------------------------------------------------------
# Objective and Output Constants
# -----------------------------------------------------------------------------

OBJECTIVE_SINGLE_MODE = "single_mode"
OBJECTIVE_MULTI_MODE = "multi_mode"
OBJECTIVE_TRAVELING_WAVE = "traveling_wave"
OBJECTIVE_THRUST = "thrust"
OBJECTIVE_THRUST_PER_POWER = "thrust_per_power"
OBJECTIVE_ADMITTANCE_MATCH = "admittance_match"
OBJECTIVE_MULTIFUNCTIONAL = "multifunctional"
OBJECTIVE_TYPES = (
    OBJECTIVE_SINGLE_MODE,
    OBJECTIVE_MULTI_MODE,
    OBJECTIVE_TRAVELING_WAVE,
    OBJECTIVE_THRUST,
    OBJECTIVE_THRUST_PER_POWER,
    OBJECTIVE_ADMITTANCE_MATCH,
    OBJECTIVE_MULTIFUNCTIONAL,
)

OUTPUT_TIP = "tip"
OUTPUT_MEAN_ABS = "mean_abs"
OUTPUT_RMS = "rms"
OUTPUT_TRAVELING_WAVE = "traveling_wave"
OUTPUT_THRUST = "thrust"
OUTPUT_THRUST_PER_POWER = "thrust_per_power"
OUTPUT_MULTIFUNCTIONAL = "multifunctional"
OUTPUT_TYPES = (OUTPUT_TIP, OUTPUT_MEAN_ABS, OUTPUT_RMS)

PhaseScoreFunction = Callable[[Any, np.ndarray], tuple[float, dict, Optional[dict]]]


# -----------------------------------------------------------------------------
# Standing-Wave Output Metrics
# -----------------------------------------------------------------------------

def metric_label(output: str) -> str:
    if output not in (
        *OUTPUT_TYPES,
        OUTPUT_TRAVELING_WAVE,
        OUTPUT_THRUST,
        OUTPUT_THRUST_PER_POWER,
        OUTPUT_MULTIFUNCTIONAL,
    ):
        raise ValueError("Unknown output metric.")
    if output == OUTPUT_TIP:
        return "Tip displacement magnitude [m/V]"
    if output == OUTPUT_MEAN_ABS:
        return "Line-average displacement magnitude [m/V]"
    if output == OUTPUT_RMS:
        return "RMS beam displacement magnitude [m/V]"
    if output == OUTPUT_TRAVELING_WAVE:
        return "Traveling-wave objective score [-]"
    if output == OUTPUT_THRUST:
        return "Mean thrust estimate [N]"
    if output == OUTPUT_THRUST_PER_POWER:
        return "Thrust per electrical power [N/W]"
    if output == OUTPUT_MULTIFUNCTIONAL:
        return "Weighted multifunctional objective score [-]"
    return "Output metric"


def evaluate_output_metric(fe, u_red: np.ndarray, output: str = OUTPUT_TIP) -> float:
    """Evaluate a scalar standing-wave output metric."""
    if output not in OUTPUT_TYPES:
        raise ValueError("Unknown output metric. Use 'tip', 'mean_abs', or 'rms'.")

    if output == OUTPUT_TIP:
        return float(abs(u_red[tip_reduced_index(fe)]))

    w_nodes = reduced_to_full_displacement_nodes(fe, u_red)
    weights = trapezoid_node_weights(fe.geom.x_nodes)
    L = float(fe.geom.x_nodes[-1] - fe.geom.x_nodes[0])
    if L <= 0:
        raise ValueError("Beam length must be positive")

    if output == OUTPUT_MEAN_ABS:
        return float(np.sum(weights * np.abs(w_nodes)) / L)

    if output == OUTPUT_RMS:
        return float(np.sqrt(np.sum(weights * np.abs(w_nodes) ** 2) / L))

    raise ValueError(f"Unhandled output metric {output}")


def response_summary(fe, u_red: np.ndarray, output: str) -> dict:
    """Return common standing-wave response metrics for a reduced vector."""
    if output not in OUTPUT_TYPES:
        raise ValueError("Unknown output metric. Use 'tip', 'mean_abs', or 'rms'.")
    return {
        OUTPUT_TIP: evaluate_output_metric(fe, u_red, OUTPUT_TIP),
        OUTPUT_MEAN_ABS: evaluate_output_metric(fe, u_red, OUTPUT_MEAN_ABS),
        OUTPUT_RMS: evaluate_output_metric(fe, u_red, OUTPUT_RMS),
        "selected": evaluate_output_metric(fe, u_red, output),
        "output": output,
    }


# -----------------------------------------------------------------------------
# Phase Optimization
# -----------------------------------------------------------------------------

def _score_response(
    fe,
    u_red: np.ndarray,
    output: str,
    score_function: Optional[PhaseScoreFunction],
    voltage_vector=None,
) -> tuple[float, dict, Optional[dict]]:
    if score_function is not None:
        try:
            return score_function(fe, u_red, voltage_vector)
        except TypeError:
            return score_function(fe, u_red)
    return float(evaluate_output_metric(fe, u_red, output)), response_summary(fe, u_red, output), None


def _phase_record(
    fe,
    U_cols,
    voltage_vector,
    output,
    *,
    score_function: Optional[PhaseScoreFunction] = None,
    signs=None,
    label=None,
) -> dict:
    u_red = U_cols @ voltage_vector
    phase_rad = np.mod(np.angle(voltage_vector), 2 * np.pi)
    rel_phase = np.mod(phase_rad - phase_rad[0], 2 * np.pi)
    score, response_metrics, extra_metrics = _score_response(
        fe,
        u_red,
        output,
        score_function,
        voltage_vector=voltage_vector,
    )
    record = {
        "signs": signs,
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": rel_phase,
        "relative_phase_deg": np.rad2deg(rel_phase),
        "voltage_vector": voltage_vector,
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "score": score,
        "response_metrics": response_metrics,
    }
    if label is not None:
        record["label"] = label
    if extra_metrics is not None:
        if output == OUTPUT_TRAVELING_WAVE:
            record["traveling_wave_metrics"] = extra_metrics
        else:
            record["objective_metrics"] = extra_metrics
    return record


def optimize_binary_phases(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
    *,
    score_function: Optional[PhaseScoreFunction] = None,
    inner_workers: int = 1,
    inner_parallel_min_tasks: int = 4,
) -> dict:
    """Brute-force all binary patch signs for standing- or traveling-wave scores."""
    U_cols = np.asarray(U_cols, dtype=complex)
    if output not in (*OUTPUT_TYPES, OUTPUT_TRAVELING_WAVE, OUTPUT_THRUST, OUTPUT_THRUST_PER_POWER, OUTPUT_MULTIFUNCTIONAL):
        raise ValueError("Unsupported phase-optimization output.")
    sign_patterns = list(itertools.product([-1.0, 1.0], repeat=U_cols.shape[1]))

    def evaluate_signs(signs_tuple):
        signs = np.asarray(signs_tuple, dtype=float)
        return _phase_record(
            fe,
            U_cols,
            voltage_amplitude * signs.astype(complex),
            output,
            score_function=score_function,
            signs=signs,
            label="".join("+" if s > 0 else "-" for s in signs),
        )

    # Serial reference version:
    # all_results = []
    # best_record = None
    # for signs_tuple in sign_patterns:
    #     record = evaluate_signs(signs_tuple)
    #     all_results.append(record)
    #     if best_record is None or record["score"] > best_record["score"]:
    #         best_record = record

    all_results = parallel_map(
        evaluate_signs,
        sign_patterns,
        workers=inner_workers,
        min_tasks=inner_parallel_min_tasks,
    )
    best_record = max(all_results, key=lambda r: r["score"])

    if best_record is None:
        raise RuntimeError("No binary phase candidates were evaluated")

    result = {
        "phase_mode": "binary",
        "phase_optimizer": "brute_force_binary_traveling_wave" if output == OUTPUT_TRAVELING_WAVE else "brute_force_binary",
        "all_phase_results": all_results,
        **{k: best_record[k] for k in (
            "score",
            "response",
            "response_red",
            "response_metrics",
            "signs",
            "phase_rad",
            "phase_deg",
            "relative_phase_rad",
            "relative_phase_deg",
            "voltage_vector",
        )},
    }
    if output == OUTPUT_TRAVELING_WAVE:
        result["traveling_wave_metrics"] = best_record["traveling_wave_metrics"]
    elif "objective_metrics" in best_record:
        result["objective_metrics"] = best_record["objective_metrics"]
    return result


def optimize_continuous_phases(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
    *,
    score_function: Optional[PhaseScoreFunction] = None,
    n_starts: int = 8,
    seed: Optional[int] = 1,
    method: str = "L-BFGS-B",
    inner_workers: int = 1,
    inner_parallel_min_tasks: int = 4,
) -> dict:
    """Optimize continuous patch phases for standing- or traveling-wave scores."""
    U_cols = np.asarray(U_cols, dtype=complex)
    if output not in (*OUTPUT_TYPES, OUTPUT_TRAVELING_WAVE, OUTPUT_THRUST, OUTPUT_THRUST_PER_POWER, OUTPUT_MULTIFUNCTIONAL):
        raise ValueError("Unsupported phase-optimization output.")
    n = U_cols.shape[1]

    if output == OUTPUT_TIP:
        phase_rad = -np.angle(U_cols[tip_reduced_index(fe), :])
        phase_rad = np.mod(phase_rad - phase_rad[0], 2 * np.pi)
        optimizer_name = "analytic_tip_alignment"
        opt_metadata = {}
    elif n == 1:
        phase_rad = np.array([0.0])
        optimizer_name = "single_patch_traveling_wave" if output == OUTPUT_TRAVELING_WAVE else "single_patch"
        opt_metadata = {}
    else:
        rng = np.random.default_rng(seed)

        def make_voltage(alpha_free: np.ndarray) -> np.ndarray:
            return voltage_amplitude * np.exp(1j * np.concatenate([[0.0], np.asarray(alpha_free, dtype=float)]))

        def neg_score(alpha_free: np.ndarray) -> float:
            voltage = make_voltage(alpha_free)
            return -_score_response(
                fe,
                U_cols @ voltage,
                output,
                score_function,
                voltage_vector=voltage,
            )[0]

        starts = [np.zeros(n - 1)]
        for signs_tuple in itertools.product([0.0, np.pi], repeat=n - 1):
            starts.append(np.asarray(signs_tuple, dtype=float))
            if len(starts) >= max(2, min(n_starts, 2 ** (n - 1) + 1)):
                break
        if output == OUTPUT_TRAVELING_WAVE:
            if len(starts) < n_starts:
                starts.append(np.linspace(0.0, np.pi, n, endpoint=False)[1:])
            if len(starts) < n_starts:
                starts.append(np.linspace(0.0, -np.pi, n, endpoint=False)[1:] % (2 * np.pi))
        while len(starts) < n_starts:
            starts.append(rng.uniform(0.0, 2 * np.pi, size=n - 1))

        def run_start(x0):
            return minimize(
                neg_score,
                x0,
                method=method,
                bounds=[(0.0, 2 * np.pi)] * (n - 1),
            )

        # Serial reference version:
        # best_res = None
        # for x0 in starts:
        #     res = run_start(x0)
        #     if best_res is None or res.fun < best_res.fun:
        #         best_res = res

        results = parallel_map(
            run_start,
            starts,
            workers=inner_workers,
            min_tasks=inner_parallel_min_tasks,
        )
        best_res = min(results, key=lambda r: r.fun)

        phase_rad = np.mod(np.concatenate([[0.0], best_res.x]), 2 * np.pi)
        suffix = "_traveling_wave" if output == OUTPUT_TRAVELING_WAVE else ""
        optimizer_name = f"numeric_{method}{suffix}"
        opt_metadata = {}

    record = _phase_record(
        fe,
        U_cols,
        voltage_amplitude * np.exp(1j * phase_rad),
        output,
        score_function=score_function,
    )
    return {
        "phase_mode": "continuous",
        "phase_optimizer": optimizer_name,
        "all_phase_results": None,
        **record,
        **opt_metadata,
    }
