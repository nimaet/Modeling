"""Phase/actuation optimizers for piezo patch optimization."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from .metrics import (
    canonical_output_name,
    default_traveling_wave_settings,
    evaluate_output_metric,
    response_summary,
    tip_reduced_index,
    traveling_wave_metrics,
)


def sign_label(signs) -> str:
    return "".join("+" if s > 0 else "-" for s in np.asarray(signs))


def wrap_phase_rad(phase_rad: np.ndarray) -> np.ndarray:
    """Wrap phases to [0, 2*pi)."""
    return np.mod(np.asarray(phase_rad, dtype=float), 2 * np.pi)


def relative_phase_rad(phase_rad: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """Return phases relative to one patch, wrapped to [0, 2*pi)."""
    phase_rad = np.asarray(phase_rad, dtype=float)
    return wrap_phase_rad(phase_rad - phase_rad[reference_index])


def scipy_result_metadata(result, optimizer_name: str) -> dict:
    """Return compact warning metadata for an inner SciPy optimization result."""
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", ""))
    return {
        "inner_opt_success": success,
        "inner_opt_message": message,
        "inner_opt_warning": None if success else f"{optimizer_name} did not report success: {message}",
    }


def optimize_binary_phases_general(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
) -> dict:
    """Brute-force binary signs for a full reduced-response matrix.

    U_cols[:, j] is the reduced displacement response per unit voltage on patch j.
    """
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    output = canonical_output_name(output)

    best_score = -np.inf
    best_signs = None
    best_response_red = None
    all_results = []

    for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
        signs = np.asarray(signs_tuple, dtype=float)
        voltage_vector = voltage_amplitude * signs.astype(complex)
        u_red = U_cols @ voltage_vector
        score = evaluate_output_metric(fe, u_red, output)
        phase_rad = np.where(signs > 0, 0.0, np.pi)
        record = {
            "signs": signs,
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": relative_phase_rad(phase_rad),
            "relative_phase_deg": np.rad2deg(relative_phase_rad(phase_rad)),
            "voltage_vector": voltage_vector,
            "response": u_red[tip_reduced_index(fe)],
            "response_red": u_red,
            "score": float(score),
            "label": sign_label(signs),
            "response_metrics": response_summary(fe, u_red, output),
        }
        all_results.append(record)
        if score > best_score:
            best_score = score
            best_signs = signs
            best_response_red = u_red

    best_phase_rad = np.where(best_signs > 0, 0.0, np.pi)
    best_voltage_vector = voltage_amplitude * best_signs.astype(complex)
    return {
        "phase_mode": "binary",
        "phase_optimizer": "brute_force_binary",
        "score": float(best_score),
        "response": best_response_red[tip_reduced_index(fe)],  # backward-compatible scalar tip response
        "response_red": best_response_red,
        "response_metrics": response_summary(fe, best_response_red, output),
        "signs": best_signs,
        "phase_rad": best_phase_rad,
        "phase_deg": np.rad2deg(best_phase_rad),
        "relative_phase_rad": relative_phase_rad(best_phase_rad),
        "relative_phase_deg": np.rad2deg(relative_phase_rad(best_phase_rad)),
        "voltage_vector": best_voltage_vector,
        "all_phase_results": all_results,
    }


def optimize_continuous_phases_tip(
    fe,
    U_cols: np.ndarray,
    voltage_amplitude: float = 1.0,
) -> dict:
    """Analytic continuous phase alignment for scalar tip displacement."""
    h = np.asarray(U_cols[tip_reduced_index(fe), :], dtype=complex)
    # A global phase does not change response magnitude. Use patch 1 as reference.
    phase_rad = relative_phase_rad(-np.angle(h), reference_index=0)
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    u_red = U_cols @ voltage_vector
    return {
        "phase_mode": "continuous",
        "phase_optimizer": "analytic_tip_alignment",
        "score": evaluate_output_metric(fe, u_red, "tip"),
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "response_metrics": response_summary(fe, u_red, "tip"),
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": phase_rad,
        "relative_phase_deg": np.rad2deg(phase_rad),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
    }


def optimize_continuous_phases_metric(
    fe,
    U_cols: np.ndarray,
    output: str,
    voltage_amplitude: float = 1.0,
    *,
    n_starts: int = 8,
    seed: Optional[int] = 1,
    method: str = "L-BFGS-B",
) -> dict:
    """Numerically optimize continuous patch phases for a non-scalar output metric.

    The first patch phase is fixed to zero because global phase does not affect
    metrics based on displacement magnitude. This reduces the search dimension
    from Np to Np-1.
    """
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    output = canonical_output_name(output)

    if n == 1:
        phase_rad = np.array([0.0])
        voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
        u_red = U_cols @ voltage_vector
        return {
            "phase_mode": "continuous",
            "phase_optimizer": "single_patch",
            "score": evaluate_output_metric(fe, u_red, output),
            "response": u_red[tip_reduced_index(fe)],
            "response_red": u_red,
            "response_metrics": response_summary(fe, u_red, output),
            "signs": np.sign(np.real(voltage_vector)),
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": phase_rad,
            "relative_phase_deg": np.rad2deg(phase_rad),
            "voltage_vector": voltage_vector,
            "all_phase_results": None,
        }

    rng = np.random.default_rng(seed)

    def make_voltage(alpha_free: np.ndarray) -> np.ndarray:
        phase_rad = np.concatenate([[0.0], np.asarray(alpha_free, dtype=float)])
        return voltage_amplitude * np.exp(1j * phase_rad)

    def neg_score(alpha_free: np.ndarray) -> float:
        u_red = U_cols @ make_voltage(alpha_free)
        return -evaluate_output_metric(fe, u_red, output)

    starts: List[np.ndarray] = [np.zeros(n - 1)]

    # Include binary-like initial guesses; often good for beam modes.
    for signs_tuple in itertools.product([0.0, np.pi], repeat=n - 1):
        starts.append(np.asarray(signs_tuple, dtype=float))
        if len(starts) >= max(2, min(n_starts, 2 ** (n - 1) + 1)):
            break

    while len(starts) < n_starts:
        starts.append(rng.uniform(0.0, 2 * np.pi, size=n - 1))

    bounds = [(0.0, 2 * np.pi)] * (n - 1)
    best_res = None
    for x0 in starts:
        res = minimize(neg_score, x0, method=method, bounds=bounds)
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    phase_rad = wrap_phase_rad(np.concatenate([[0.0], best_res.x]))
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    u_red = U_cols @ voltage_vector
    opt_metadata = scipy_result_metadata(best_res, f"numeric_{method}")
    return {
        "phase_mode": "continuous",
        "phase_optimizer": f"numeric_{method}",
        "score": evaluate_output_metric(fe, u_red, output),
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "response_metrics": response_summary(fe, u_red, output),
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": phase_rad,
        "relative_phase_deg": np.rad2deg(phase_rad),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
        "inner_opt_result": best_res,
        **opt_metadata,
    }


def optimize_binary_phases_traveling_wave(
    fe,
    U_cols: np.ndarray,
    settings: Optional[Dict[str, Any]] = None,
    voltage_amplitude: float = 1.0,
) -> dict:
    """Brute-force binary signs for the traveling-wave objective."""
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    best_score = -np.inf
    best_record = None
    all_results = []

    for signs_tuple in itertools.product([-1.0, 1.0], repeat=n):
        signs = np.asarray(signs_tuple, dtype=float)
        voltage_vector = voltage_amplitude * signs.astype(complex)
        u_red = U_cols @ voltage_vector
        metrics = traveling_wave_metrics(fe, u_red, settings)
        phase_rad = np.where(signs > 0, 0.0, np.pi)
        record = {
            "signs": signs,
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": relative_phase_rad(phase_rad),
            "relative_phase_deg": np.rad2deg(relative_phase_rad(phase_rad)),
            "voltage_vector": voltage_vector,
            "response": u_red[tip_reduced_index(fe)],
            "response_red": u_red,
            "score": float(metrics["score"]),
            "label": sign_label(signs),
            "traveling_wave_metrics": metrics,
            "response_metrics": response_summary(fe, u_red, "rms"),
        }
        all_results.append(record)
        if record["score"] > best_score:
            best_score = record["score"]
            best_record = record

    if best_record is None:
        raise RuntimeError("No binary traveling-wave phase candidates were evaluated")

    return {
        "phase_mode": "binary",
        "phase_optimizer": "brute_force_binary_traveling_wave",
        "score": float(best_record["score"]),
        "response": best_record["response"],
        "response_red": best_record["response_red"],
        "response_metrics": best_record["response_metrics"],
        "traveling_wave_metrics": best_record["traveling_wave_metrics"],
        "signs": best_record["signs"],
        "phase_rad": best_record["phase_rad"],
        "phase_deg": best_record["phase_deg"],
        "relative_phase_rad": best_record["relative_phase_rad"],
        "relative_phase_deg": best_record["relative_phase_deg"],
        "voltage_vector": best_record["voltage_vector"],
        "all_phase_results": all_results,
    }


def optimize_continuous_phases_traveling_wave(
    fe,
    U_cols: np.ndarray,
    settings: Optional[Dict[str, Any]] = None,
    voltage_amplitude: float = 1.0,
    *,
    n_starts: int = 8,
    seed: Optional[int] = 1,
    method: str = "L-BFGS-B",
) -> dict:
    """Numerically optimize continuous patch phases for traveling-wave quality."""
    U_cols = np.asarray(U_cols, dtype=complex)
    n = U_cols.shape[1]
    cfg = default_traveling_wave_settings(settings)

    if n == 1:
        phase_rad = np.array([0.0])
        voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
        u_red = U_cols @ voltage_vector
        metrics = traveling_wave_metrics(fe, u_red, cfg)
        return {
            "phase_mode": "continuous",
            "phase_optimizer": "single_patch_traveling_wave",
            "score": float(metrics["score"]),
            "response": u_red[tip_reduced_index(fe)],
            "response_red": u_red,
            "response_metrics": response_summary(fe, u_red, "rms"),
            "traveling_wave_metrics": metrics,
            "signs": np.sign(np.real(voltage_vector)),
            "phase_rad": phase_rad,
            "phase_deg": np.rad2deg(phase_rad),
            "relative_phase_rad": phase_rad,
            "relative_phase_deg": np.rad2deg(phase_rad),
            "voltage_vector": voltage_vector,
            "all_phase_results": None,
        }

    rng = np.random.default_rng(seed)

    def make_voltage(alpha_free: np.ndarray) -> np.ndarray:
        phase_rad = np.concatenate([[0.0], np.asarray(alpha_free, dtype=float)])
        return voltage_amplitude * np.exp(1j * phase_rad)

    def neg_score(alpha_free: np.ndarray) -> float:
        u_red = U_cols @ make_voltage(alpha_free)
        return -float(traveling_wave_metrics(fe, u_red, cfg)["score"])

    starts: List[np.ndarray] = [np.zeros(n - 1)]
    for signs_tuple in itertools.product([0.0, np.pi], repeat=n - 1):
        starts.append(np.asarray(signs_tuple, dtype=float))
        if len(starts) >= max(2, min(n_starts, 2 ** (n - 1) + 1)):
            break

    # Add progressive phase guesses in both directions; these often resemble
    # simple traveling-wave phasing better than random starts.
    if len(starts) < n_starts:
        starts.append(np.linspace(0.0, np.pi, n, endpoint=False)[1:])
    if len(starts) < n_starts:
        starts.append(np.linspace(0.0, -np.pi, n, endpoint=False)[1:] % (2 * np.pi))

    while len(starts) < n_starts:
        starts.append(rng.uniform(0.0, 2 * np.pi, size=n - 1))

    bounds = [(0.0, 2 * np.pi)] * (n - 1)
    best_res = None
    for x0 in starts:
        res = minimize(neg_score, x0, method=method, bounds=bounds)
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    phase_rad = wrap_phase_rad(np.concatenate([[0.0], best_res.x]))
    voltage_vector = voltage_amplitude * np.exp(1j * phase_rad)
    u_red = U_cols @ voltage_vector
    metrics = traveling_wave_metrics(fe, u_red, cfg)
    opt_metadata = scipy_result_metadata(best_res, f"numeric_{method}_traveling_wave")
    return {
        "phase_mode": "continuous",
        "phase_optimizer": f"numeric_{method}_traveling_wave",
        "score": float(metrics["score"]),
        "response": u_red[tip_reduced_index(fe)],
        "response_red": u_red,
        "response_metrics": response_summary(fe, u_red, "rms"),
        "traveling_wave_metrics": metrics,
        "signs": np.sign(np.real(voltage_vector)),
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "relative_phase_rad": phase_rad,
        "relative_phase_deg": np.rad2deg(phase_rad),
        "voltage_vector": voltage_vector,
        "all_phase_results": None,
        "inner_opt_result": best_res,
        **opt_metadata,
    }
