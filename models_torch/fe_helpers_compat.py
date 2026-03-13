"""
Compatibility helpers to use Modeling.models.FE_helpers with models_torch.

This module intentionally mirrors the ODE container contract expected by
FE_helpers.frf_sweep and FE_helpers.solve_newmark, without modifying FE_helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class PiezoBeamODESystemCompat:
    M: np.ndarray
    M_mech: np.ndarray
    K_mech: np.ndarray
    C: np.ndarray
    D: np.ndarray
    f_ext_freq_domain: np.ndarray
    f_int: Callable[[np.ndarray], np.ndarray]
    K_tan: Callable[[np.ndarray], np.ndarray]
    f_ext: Callable[[float], np.ndarray]
    v_exc: Callable[[float], np.ndarray]
    N_mech: int
    N_elec: int


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _modal_damping_matrix(M: np.ndarray, K: np.ndarray, zeta: float) -> np.ndarray:
    # Generalized eigenproblem K phi = w^2 M phi
    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(M, K))
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    eigvals = np.maximum(eigvals, 0.0)
    omega = np.sqrt(eigvals)

    idx = np.argsort(omega)
    omega = omega[idx]
    eigvecs = eigvecs[:, idx]

    # Mass-normalize modal vectors
    for i in range(eigvecs.shape[1]):
        nrm = np.sqrt(eigvecs[:, i].T @ M @ eigvecs[:, i])
        if nrm > 0:
            eigvecs[:, i] /= nrm

    c_modal = 2.0 * zeta * omega
    C_modal = np.diag(c_modal)

    # Back-transform
    V = eigvecs
    V_inv = np.linalg.inv(V)
    return V_inv.T @ C_modal @ V_inv


def _modal_damping_matrix_from_zeta_dict(
    M: np.ndarray,
    K: np.ndarray,
    zeta_dict: dict,
) -> np.ndarray:
    """Build damping matrix using FE3-style per-mode zeta dictionary."""
    eigvals, eigvecs = eigh(K, M)
    eigvals = np.maximum(np.real(eigvals), 0.0)
    omega = np.sqrt(eigvals)

    zeta = np.array([
        zeta_dict.get(i + 1, zeta_dict.get("rest", 0.0))
        for i in range(len(omega))
    ])
    c_modal = 2.0 * zeta * omega
    C_modal = np.diag(c_modal)

    V_inv = np.linalg.inv(eigvecs)
    return V_inv.T @ C_modal @ V_inv


def build_ode_system_compat(
    fe,
    j_exc=None,
    R_c: float = 1e3,
    K_p: float = 0.02,
    K_i=0.0,
    K_c: float = 0.0,
    v_exc: Callable[[float], np.ndarray] = lambda t: 1.0,
    freq_domain_amps=1.0,
    c_alpha: float = 0.0,
    c_beta: float = 0.0,
    modal_zeta: float | None = None,
    zeta_dict: dict | None = None,
) -> PiezoBeamODESystemCompat:
    """
    Build FE3-compatible ODE container from PiezoBeamFE_Torch.

    The returned object can be passed directly to FE_helpers.frf_sweep
    and FE_helpers.solve_newmark.
    """
    K_red_t, M_red_t, Gamma_red_t = fe.build_KM_Gamma()

    K_red = _to_numpy(K_red_t)
    M_red = _to_numpy(M_red_t)
    Gamma_red = _to_numpy(Gamma_red_t)

    N = M_red.shape[0]
    S = Gamma_red.shape[1]

    if j_exc is None:
        j_exc = np.arange(S)
    j_exc = np.atleast_1d(j_exc).astype(int)
    if j_exc.size == 0:
        raise ValueError("j_exc must contain at least one excited piezo index")
    if np.any((j_exc < 0) | (j_exc >= S)):
        raise ValueError(f"j_exc indices out of range [0, {S-1}]")
    j_exc = np.unique(j_exc)

    idx_all = np.arange(S)
    idx_f = np.setdiff1d(idx_all, j_exc)

    Gamma_f = Gamma_red[:, idx_f]
    Gamma_e = Gamma_red[:, j_exc]

    if np.isscalar(K_i):
        K_i = float(K_i) * np.ones(len(idx_f))
    else:
        K_i = np.delete(np.asarray(K_i, dtype=float), j_exc)
        if len(K_i) != len(idx_f):
            raise ValueError(f"K_i length mismatch: expected {len(idx_f)}, got {len(K_i)}")

    if zeta_dict is not None:
        D = _modal_damping_matrix_from_zeta_dict(M_red, K_red, zeta_dict) + c_alpha * M_red + c_beta * K_red
    elif modal_zeta is not None:
        D = _modal_damping_matrix(M_red, K_red, float(modal_zeta)) + c_alpha * M_red + c_beta * K_red
    else:
        D = c_alpha * M_red + c_beta * K_red

    Cp = float(_to_numpy(fe.tp.Cp_scalar()))
    M_elec = Cp * np.eye(len(idx_f))

    M_ODE = np.block([
        [M_red, np.zeros((N, len(idx_f)))],
        [np.zeros((len(idx_f), N)), M_elec],
    ])

    C_ODE = np.block([
        [D, -Gamma_f],
        [Gamma_f.T, (K_p / R_c) * np.eye(len(idx_f))],
    ])

    def f_int(x: np.ndarray) -> np.ndarray:
        u = x[:N]
        qf = x[N:]
        return np.concatenate([
            K_red @ u,
            (K_i / R_c) * qf + (K_c / R_c) * qf ** 3,
        ])

    def K_tan(x: np.ndarray) -> np.ndarray:
        qf = x[N:]
        Kqq = (np.diag(K_i) / R_c) + (3.0 * K_c / R_c) * np.diag(qf ** 2)
        return np.block([
            [K_red, np.zeros((N, len(qf)))],
            [np.zeros((len(qf), N)), Kqq],
        ])

    def f_ext(t: float) -> np.ndarray:
        v_t = v_exc(t)
        if np.isscalar(v_t):
            v_t = np.full(len(j_exc), float(v_t))
        else:
            v_t = np.asarray(v_t)
            if v_t.shape[0] != len(j_exc):
                raise ValueError(
                    f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}"
                )
        return np.concatenate([Gamma_e @ v_t, np.zeros(len(idx_f))])

    amps = np.asarray(freq_domain_amps)
    if amps.ndim == 0:
        amps = np.full(len(j_exc), complex(amps))
    if amps.shape[0] != len(j_exc):
        raise ValueError(
            f"freq_domain_amps length mismatch: expected {len(j_exc)}, got {amps.shape[0]}"
        )

    f_ext_freq_domain = np.concatenate([
        Gamma_e @ amps,
        np.zeros(len(idx_f), dtype=complex),
    ])

    return PiezoBeamODESystemCompat(
        M=M_ODE,
        M_mech=M_red,
        K_mech=K_red,
        C=C_ODE,
        D=D,
        f_ext_freq_domain=f_ext_freq_domain,
        f_int=f_int,
        K_tan=K_tan,
        f_ext=f_ext,
        v_exc=v_exc,
        N_mech=N,
        N_elec=len(idx_f),
    )
