"""Output and traveling-wave metrics for piezo patch optimization."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def tip_reduced_index(fe) -> int:
    """Reduced DOF index for tip transverse displacement."""
    tip_full_dof = 2 * (len(fe.geom.x_nodes) - 1)
    idx = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(idx) != 1:
        raise RuntimeError("Could not find tip displacement DOF in reduced system")
    return int(idx[0])


def transverse_reduced_indices(fe) -> np.ndarray:
    """Reduced indices corresponding to transverse displacement DOFs w, excluding fixed DOFs."""
    return np.where((np.asarray(fe.free_dofs) % 2) == 0)[0]


def reduced_to_full_displacement_nodes(fe, u_red: np.ndarray) -> np.ndarray:
    """Convert reduced mechanical response vector to nodal transverse displacement array."""
    u_red = np.asarray(u_red, dtype=complex)
    full = np.zeros(fe.Ndof, dtype=complex)
    full[fe.free_dofs] = u_red
    return full[0::2]


def trapezoid_node_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Integration weights for nodal values on a nonuniform 1D mesh."""
    x = np.asarray(x_nodes, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x_nodes must be a 1D array with at least two nodes")
    weights = np.zeros_like(x)
    dx = np.diff(x)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return weights


def canonical_output_name(output: str) -> str:
    output = output.lower().strip()
    aliases = {
        "tip": "tip",
        "tip_disp": "tip",
        "tip_displacement": "tip",
        "line_average": "mean_abs",
        "avg": "mean_abs",
        "average": "mean_abs",
        "average_displacement": "mean_abs",
        "mean": "mean_abs",
        "mean_abs": "mean_abs",
        "mean_abs_displacement": "mean_abs",
        "rms": "rms",
        "rms_displacement": "rms",
    }
    if output not in aliases:
        raise ValueError("Unknown output metric. Use 'tip', 'mean_abs'/'line_average', or 'rms'.")
    return aliases[output]


def metric_label(output: str) -> str:
    output = canonical_output_name(output)
    if output == "tip":
        return "Tip displacement magnitude [m/V]"
    if output == "mean_abs":
        return "Line-average displacement magnitude [m/V]"
    if output == "rms":
        return "RMS beam displacement magnitude [m/V]"
    return "Output metric"


def evaluate_output_metric(fe, u_red: np.ndarray, output: str = "tip") -> float:
    """Evaluate scalar output metric from a reduced displacement response vector.

    For harmonic complex response u_hat, this evaluates the amplitude metric.
    """
    output = canonical_output_name(output)

    if output == "tip":
        return float(abs(u_red[tip_reduced_index(fe)]))

    w_nodes = reduced_to_full_displacement_nodes(fe, u_red)
    weights = trapezoid_node_weights(fe.geom.x_nodes)
    L = float(fe.geom.x_nodes[-1] - fe.geom.x_nodes[0])
    if L <= 0:
        raise ValueError("Beam length must be positive")

    if output == "mean_abs":
        return float(np.sum(weights * np.abs(w_nodes)) / L)

    if output == "rms":
        return float(np.sqrt(np.sum(weights * np.abs(w_nodes) ** 2) / L))

    raise ValueError(f"Unhandled output metric {output}")


def response_summary(fe, u_red: np.ndarray, output: str) -> dict:
    """Return common response metrics for a reduced displacement vector."""
    output = canonical_output_name(output)
    return {
        "tip": evaluate_output_metric(fe, u_red, "tip"),
        "mean_abs": evaluate_output_metric(fe, u_red, "mean_abs"),
        "rms": evaluate_output_metric(fe, u_red, "rms"),
        "selected": evaluate_output_metric(fe, u_red, output),
        "output": output,
    }


def default_traveling_wave_settings(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return traveling-wave settings with conservative defaults."""
    cfg = {
        "frequency_hz": None,
        "mode_pair": (1, 2),
        "frequency_fraction": 0.5,
        "amplitude_reference": 1e-4,
        "ti_power": 2.0,
        "amplitude_power": 1.0,
        "envelope_power": 1.0,
        "x_fraction_bounds": (0.05, 1.0),
        "direction": "either",
        "phase_slope_amplitude_floor": 0.05,
        "eps": 1e-300,
    }
    if settings is not None:
        cfg.update(dict(settings))
    return cfg


def traveling_wave_frequency_from_settings(fe, settings: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """Return excitation frequency and angular frequency for a traveling-wave objective."""
    cfg = default_traveling_wave_settings(settings)
    if cfg.get("frequency_hz", None) is not None:
        freq_hz = float(cfg["frequency_hz"])
        if freq_hz <= 0:
            raise ValueError("traveling_wave_settings['frequency_hz'] must be positive")
        return freq_hz, 2.0 * np.pi * freq_hz

    mode_pair = tuple(int(m) for m in cfg.get("mode_pair", (1, 2)))
    if len(mode_pair) != 2:
        raise ValueError("traveling_wave_settings['mode_pair'] must contain two mode numbers")
    if any(m < 1 or m > len(fe.freq) for m in mode_pair):
        raise ValueError(f"mode_pair={mode_pair} outside available mode range 1..{len(fe.freq)}")

    f0 = float(fe.freq[mode_pair[0] - 1])
    f1 = float(fe.freq[mode_pair[1] - 1])
    frac = float(cfg.get("frequency_fraction", 0.5))
    freq_hz = f0 + frac * (f1 - f0)
    if freq_hz <= 0:
        raise ValueError("Computed traveling-wave frequency must be positive")
    return freq_hz, 2.0 * np.pi * freq_hz


def traveling_wave_node_window(fe, settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Boolean node mask for spatial traveling-wave metrics."""
    cfg = default_traveling_wave_settings(settings)
    x = np.asarray(fe.geom.x_nodes, dtype=float)
    L = float(x[-1] - x[0])
    if L <= 0:
        raise ValueError("Beam length must be positive")

    lo, hi = cfg.get("x_fraction_bounds", (0.05, 1.0))
    lo = float(lo)
    hi = float(hi)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("traveling_wave_settings['x_fraction_bounds'] must satisfy 0 <= lo < hi <= 1")

    eta = (x - x[0]) / L
    mask = (eta >= lo) & (eta <= hi)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Traveling-wave spatial window must include at least three nodes")
    return mask


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / denom)


def traveling_index_from_complex_shape(W: np.ndarray, eps: float = 1e-300) -> float:
    """Feeny-style traveling index from one complex harmonic spatial shape.

    The index is the ratio of minor to major singular values of
    ``[real(W), imag(W)]``. It is 1 for a circular/pure traveling component and
    0 for a purely standing component.
    """
    W = np.asarray(W, dtype=complex)
    A = np.column_stack([np.real(W), np.imag(W)])
    if A.shape[0] < 2:
        return 0.0
    s = np.linalg.svd(A, compute_uv=False)
    if s.size < 2 or s[0] <= eps:
        return 0.0
    return float(np.clip(s[-1] / s[0], 0.0, 1.0))


def phase_slope_from_complex_shape(x: np.ndarray, W: np.ndarray, *, amplitude_floor: float = 0.05) -> float:
    """Fit spatial phase slope d(angle(W))/dx over sufficiently excited nodes."""
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=complex)
    amp = np.abs(W)
    if amp.size < 2 or np.max(amp) <= 0:
        return 0.0

    keep = amp >= float(amplitude_floor) * float(np.max(amp))
    if np.count_nonzero(keep) < 2:
        keep = np.ones_like(amp, dtype=bool)

    phase = np.unwrap(np.angle(W[keep]))
    xx = x[keep]
    if len(xx) < 2 or np.ptp(xx) <= 0:
        return 0.0
    return float(np.polyfit(xx, phase, 1)[0])


def traveling_wave_metrics(fe, u_red: np.ndarray, settings: Optional[Dict[str, Any]] = None) -> dict:
    """Evaluate traveling-wave quality metrics for one complex harmonic response."""
    cfg = default_traveling_wave_settings(settings)
    eps = float(cfg.get("eps", 1e-300))
    x_full = np.asarray(fe.geom.x_nodes, dtype=float)
    W_full = reduced_to_full_displacement_nodes(fe, u_red)
    mask = traveling_wave_node_window(fe, cfg)

    x = x_full[mask]
    W = W_full[mask]
    weights = trapezoid_node_weights(x_full)[mask]
    if np.sum(weights) <= 0:
        weights = np.ones_like(x, dtype=float)

    amp = np.abs(W)
    traveling_index = traveling_index_from_complex_shape(W, eps=eps)
    amplitude_rms = float(np.sqrt(_weighted_mean(amp ** 2, weights)))

    envelope_mean = _weighted_mean(amp, weights)
    envelope_var = _weighted_mean((amp - envelope_mean) ** 2, weights)
    envelope_std = float(np.sqrt(max(envelope_var, 0.0)))
    envelope_cv = float(envelope_std / (envelope_mean + eps))
    envelope_score = float(1.0 / (1.0 + envelope_cv))

    amplitude_reference = cfg.get("amplitude_reference", 1e-4)
    if amplitude_reference is None or float(amplitude_reference) <= 0:
        amplitude_score = 1.0
        amplitude_reference_value = None
    else:
        amplitude_reference_value = float(amplitude_reference)
        amplitude_score = float(amplitude_rms / (amplitude_reference_value + amplitude_rms + eps))

    phase_slope = phase_slope_from_complex_shape(
        x,
        W,
        amplitude_floor=float(cfg.get("phase_slope_amplitude_floor", 0.05)),
    )
    direction = str(cfg.get("direction", "either")).lower().strip()
    if direction in ("either", "any", "none"):
        direction_score = 1.0
    elif direction in ("positive_x", "+x", "tailward", "tail"):
        # With Re(W exp(i omega t)), decreasing spatial phase travels in +x.
        direction_score = 1.0 if phase_slope <= 0.0 else 0.0
    elif direction in ("negative_x", "-x", "headward", "head"):
        direction_score = 1.0 if phase_slope >= 0.0 else 0.0
    else:
        raise ValueError("traveling_wave_settings['direction'] must be 'either', 'positive_x', or 'negative_x'")

    ti_term = traveling_index ** float(cfg.get("ti_power", 2.0))
    amp_term = amplitude_score ** float(cfg.get("amplitude_power", 1.0))
    env_term = envelope_score ** float(cfg.get("envelope_power", 1.0))
    score = float(ti_term * amp_term * env_term * direction_score)

    return {
        "score": score,
        "traveling_index": float(traveling_index),
        "traveling_index_term": float(ti_term),
        "amplitude_rms": amplitude_rms,
        "amplitude_reference": amplitude_reference_value,
        "amplitude_score": float(amplitude_score),
        "envelope_mean": float(envelope_mean),
        "envelope_std": envelope_std,
        "envelope_cv": envelope_cv,
        "envelope_score": envelope_score,
        "phase_slope_rad_per_m": phase_slope,
        "direction": direction,
        "direction_score": float(direction_score),
        "x": x,
        "W": W,
        "x_full": x_full,
        "W_full": W_full,
        "x_mask": mask,
    }


def compact_traveling_wave_metrics(metrics: dict) -> dict:
    """Return scalar traveling-wave metrics suitable for logs/dataframes."""
    scalar_keys = [
        "score",
        "traveling_index",
        "traveling_index_term",
        "amplitude_rms",
        "amplitude_reference",
        "amplitude_score",
        "envelope_mean",
        "envelope_std",
        "envelope_cv",
        "envelope_score",
        "phase_slope_rad_per_m",
        "direction",
        "direction_score",
    ]
    return {k: metrics.get(k) for k in scalar_keys if k in metrics}
