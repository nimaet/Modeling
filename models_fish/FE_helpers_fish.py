"""Frequency-response and time-domain helper functions for piezo beam ODE systems."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm


def compute_frf_from_time_domain(t, veloc, v_exc_values, *, verbose: bool = False):
    """Compute an FFT-based FRF from a time-domain velocity response.

    Parameters
    ----------
    t : array_like
        Time vector [s].
    veloc : ndarray
        Velocity field with shape ``(n_time, n_spatial)``.
    v_exc_values : array_like
        Excitation values with shape ``(n_time,)``.
    verbose : bool, optional
        If True, print FFT array shapes for debugging.
    """
    t = np.asarray(t)
    veloc = np.asarray(veloc)
    v_exc_values = np.asarray(v_exc_values)

    Nt = len(t)
    if Nt < 2:
        return {"freq": None, "FRF": None, "Y": None, "X": None}

    dt = t[1] - t[0]
    Y = np.fft.fft(veloc, axis=0)
    X = np.fft.fft(v_exc_values)
    freq = np.fft.fftfreq(Nt, d=dt)

    if verbose:
        print("X shape:", X.shape, "Y shape:", Y.shape, "freq shape:", freq.shape, "veloc shape:", veloc.shape)

    idx = freq >= 0
    freq = freq[idx]
    Y = Y[idx, :]
    X = X[idx]

    X_mag = np.abs(X)
    X_mag = np.where(X_mag < 1e-10, 1.0, X_mag)
    FRF = np.mean(np.abs(Y), axis=1) / X_mag
    return {"freq": freq, "FRF": FRF, "Y": Y, "X": X}


def solve_newmark(
    ode,
    dt,
    t_end,
    beta=0.25,
    gamma=0.5,
    newton_tol=1e-9,
    newton_maxiter=5,
    x0=None,
    x_dot0=None,
    do_spectral=True,
    spectral_verbose: bool = False,
):
    """Solve a coupled ODE system with Newmark-beta time integration."""
    try:
        from Modeling.models.newmark import newmark_beta_nonlinear
    except Exception:
        from newmark import newmark_beta_nonlinear

    ndof = ode.M.shape[0]
    if x0 is None:
        x0 = np.zeros(ndof)
    if x_dot0 is None:
        x_dot0 = np.zeros(ndof)

    a0 = np.linalg.solve(ode.M, ode.f_ext(0.0) - ode.C @ x_dot0 - ode.f_int(x0))
    n_steps = int(t_end / dt)

    x, x_dot, x_ddot = newmark_beta_nonlinear(
        M=ode.M,
        C=ode.C,
        f_int=ode.f_int,
        K_tan=ode.K_tan,
        f_ext=ode.f_ext,
        u0=x0,
        v0=x_dot0,
        a0_init=a0,
        dt=dt,
        n_steps=n_steps,
        beta=beta,
        gamma=gamma,
        newton_tol=newton_tol,
        newton_maxiter=newton_maxiter,
    )

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    N_mech = ode.N_mech

    u = x[:, :N_mech:2]
    u_dot = x_dot[:, :N_mech:2]
    u_ddot = x_ddot[:, :N_mech:2]
    q = x[:, N_mech:]
    v = x_dot[:, N_mech:]

    result = {"t": t, "u": u, "u_dot": u_dot, "u_ddot": u_ddot, "q": q, "v": v, "x": x, "x_dot": x_dot, "x_ddot": x_ddot}

    if do_spectral:
        v_exc_values = ode.v_exc(t)
        if np.ndim(v_exc_values) == 2:
            v_exc_values = np.sqrt(np.mean(v_exc_values**2, axis=0))
        if len(v_exc_values) != u_ddot.shape[0]:
            raise ValueError("Excitation length mismatch with time vector")
        result["spectral"] = compute_frf_from_time_domain(t, u_dot, v_exc_values, verbose=spectral_verbose)
    else:
        result["spectral"] = None

    return result


def frequency_response_linear(ode, omega: float):
    """Linear frequency response of the full coupled ODE system."""
    M = ode.M
    C = ode.C
    K = ode.K_tan(np.zeros(M.shape[0]))
    Z = -omega**2 * M + 1j * omega * C + K
    return np.linalg.solve(Z, ode.f_ext_freq_domain)


def frequency_response_mechanical(ode, omega: float, f_hat=None):
    """Linear frequency response for the mechanical subsystem only."""
    M = ode.M_mech
    C = ode.D
    K = ode.K_mech
    if f_hat is None:
        f_hat = ode.f_ext_freq_domain[: ode.N_mech]
    Z = -omega**2 * M + 1j * omega * C + K
    return np.linalg.solve(Z, f_hat)


def frequency_response_mechanical_modal(ode, omega: float, freq_max: float = 5000.0, n_modes_max: int | None = None, f_hat=None):
    """Modal-reduced mechanical frequency response."""
    M = ode.M_mech
    C = ode.D
    K = ode.K_mech
    if f_hat is None:
        f_hat = ode.f_ext_freq_domain[: ode.N_mech]

    evals, evecs = eigh(K, M)
    omega_n = np.sqrt(np.maximum(evals, 0.0))
    f_n = omega_n / (2 * np.pi)

    mode_indices = np.where(f_n <= freq_max)[0]
    if n_modes_max is not None:
        mode_indices = mode_indices[:n_modes_max]
    if len(mode_indices) == 0:
        raise ValueError("No modes selected for modal response")

    Phi = evecs[:, mode_indices]
    Mm = Phi.T @ M @ Phi
    Cm = Phi.T @ C @ Phi
    Km = Phi.T @ K @ Phi
    fm = Phi.T @ f_hat
    q_hat = np.linalg.solve(-omega**2 * Mm + 1j * omega * Cm + Km, fm)
    return Phi @ q_hat


def frf_sweep(ode, omega_vec, *, show_progress: bool = True):
    """Compute full coupled frequency response over an angular-frequency vector."""
    omega_vec = np.asarray(omega_vec, dtype=float)
    ndof = ode.M.shape[0]
    N_mech = ode.N_mech
    X = np.zeros((len(omega_vec), ndof), dtype=complex)

    iterator = tqdm(omega_vec, desc="FRF sweep") if show_progress else omega_vec
    for k, w in enumerate(iterator):
        X[k] = frequency_response_linear(ode, w)

    u = X[:, :N_mech:2]
    q = X[:, N_mech:]
    u_dot = 1j * omega_vec[:, None] * u
    v = 1j * omega_vec[:, None] * q

    return {"omega": omega_vec, "freq": omega_vec / (2 * np.pi), "u": u, "u_dot": u_dot, "q": q, "v": v, "X": X}


def frf_sweep_mechanical(ode, omega_vec, *, show_progress: bool = True, modal: bool = False, **modal_kwargs):
    """Compute mechanical-only FRF sweep.

    This replaces the older ``frf_sweep_SC`` helper while keeping a compatibility
    alias below.
    """
    omega_vec = np.asarray(omega_vec, dtype=float)
    N_mech = ode.N_mech
    X = np.zeros((len(omega_vec), N_mech), dtype=complex)
    iterator = tqdm(omega_vec, desc="Mechanical FRF sweep") if show_progress else omega_vec
    for k, w in enumerate(iterator):
        if modal:
            X[k] = frequency_response_mechanical_modal(ode, w, **modal_kwargs)
        else:
            X[k] = frequency_response_mechanical(ode, w)

    u = X[:, :N_mech:2]
    u_dot = 1j * omega_vec[:, None] * u
    return {"omega": omega_vec, "freq": omega_vec / (2 * np.pi), "u": u, "u_dot": u_dot, "X": X}


# Backward-compatible aliases. The old functions referenced ode.f_ext_unit;
# these aliases use ode.f_ext_freq_domain instead.
frequency_response_SC = frequency_response_mechanical
frequency_response_SC_modal = frequency_response_mechanical_modal
frf_sweep_SC = frf_sweep_mechanical
