"""FE-model and response helper functions for piezo beam systems."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Generic Result Helpers
# -----------------------------------------------------------------------------

def _copy_array_or_value(value):
    """Recursively copy arrays and containers for lightweight result snapshots."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_array_or_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_array_or_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _copy_array_or_value(v) for k, v in value.items()}
    return value


# -----------------------------------------------------------------------------
# Frequency Response and Time-Domain Helpers
# -----------------------------------------------------------------------------
    
def compute_frequency_response_from_time_domain(t, veloc, v_exc_values, *, verbose: bool = False):
    """Compute an FFT-based frequency response from a time-domain velocity response.

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
        return {"freq": None, "frequency_response": None, "Y": None, "X": None}

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
    frequency_response = np.mean(np.abs(Y), axis=1) / X_mag
    return {"freq": freq, "frequency_response": frequency_response, "Y": Y, "X": X}


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
        result["spectral"] = compute_frequency_response_from_time_domain(t, u_dot, v_exc_values, verbose=spectral_verbose)
    else:
        result["spectral"] = None

    return result


# -----------------------------------------------------------------------------
# Reduced-DOF and Mesh Helpers
# -----------------------------------------------------------------------------

def tip_reduced_index(fe) -> int:
    """Reduced DOF index for tip transverse displacement."""
    tip_full_dof = 2 * (len(fe.geom.x_nodes) - 1)
    idx = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(idx) != 1:
        raise RuntimeError("Could not find tip displacement DOF in reduced system")
    return int(idx[0])


def tip_slope_reduced_index(fe) -> int:
    """Reduced DOF index for tip rotation/slope."""
    tip_full_dof = 2 * (len(fe.geom.x_nodes) - 1) + 1
    idx = np.where(fe.free_dofs == tip_full_dof)[0]
    if len(idx) != 1:
        raise RuntimeError("Could not find tip slope DOF in reduced system")
    return int(idx[0])


def reduced_to_full_displacement_nodes(fe, u_red: np.ndarray) -> np.ndarray:
    """Convert a reduced mechanical response vector to nodal displacement."""
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

# -----------------------------------------------------------------------------
# Linear Hydrodynamic Helpers
# -----------------------------------------------------------------------------

def hermite_consistent_mass_matrix_per_length(Le: float) -> np.ndarray:
    """Consistent Euler-Bernoulli element mass matrix for unit mass per length."""
    return (Le / 420.0) * np.array(
        [
            [156, 22 * Le, 54, -13 * Le],
            [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
            [54, 13 * Le, 156, -22 * Le],
            [-13 * Le, -3 * Le**2, -22 * Le, 4 * Le**2],
        ],
        dtype=float,
    )


def linear_added_mass_matrix(
    fe,
    rho: float,
    width: float,
    added_mass_coefficient: float = 1.0,
) -> np.ndarray:
    """Assemble reduced linear hydrodynamic added mass matrix."""
    rho = float(rho)
    width = float(width)
    cm = float(added_mass_coefficient)

    if rho <= 0:
        raise ValueError("rho must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if cm < 0:
        raise ValueError("added_mass_coefficient must be nonnegative")

    m_added_per_length = 0.25 * np.pi * rho * width**2 * cm

    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)
    M_full = np.zeros_like(fe.M, dtype=float)

    for e in range(len(x_nodes) - 1):
        Le = x_nodes[e + 1] - x_nodes[e]
        if Le <= 0:
            raise ValueError(f"Non-positive element length at element {e}: {Le}")

        Me = m_added_per_length * hermite_consistent_mass_matrix_per_length(Le)
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        M_full[np.ix_(dofs, dofs)] += Me

    return M_full[np.ix_(fe.free_dofs, fe.free_dofs)]

def linear_hydrodynamic_damping_matrix(fe, damping_per_length: float) -> np.ndarray:
    """Assemble reduced linear hydrodynamic damping matrix."""
    damping_per_length = float(damping_per_length)
    if damping_per_length < 0:
        raise ValueError("damping_per_length must be nonnegative")

    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)
    C_full = np.zeros_like(fe.M, dtype=float)

    for e in range(len(x_nodes) - 1):
        Le = x_nodes[e + 1] - x_nodes[e]
        if Le <= 0:
            raise ValueError(f"Non-positive element length at element {e}: {Le}")

        Ce = damping_per_length * hermite_consistent_mass_matrix_per_length(Le)
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        C_full[np.ix_(dofs, dofs)] += Ce

    return C_full[np.ix_(fe.free_dofs, fe.free_dofs)]


# -----------------------------------------------------------------------------
# Nonlinear Hydrodynamic Helpers
# -----------------------------------------------------------------------------
def reduced_to_full_dofs(fe, q_red: np.ndarray) -> np.ndarray:
    """Expand a reduced DOF vector into full FE DOFs."""
    q_red = np.asarray(q_red)
    full = np.zeros(fe.Ndof, dtype=q_red.dtype)
    full[fe.free_dofs] = q_red
    return full


def hermite_shape_functions(xi: float, Le: float) -> np.ndarray:
    """Hermite beam shape functions for transverse displacement."""
    return np.array(
        [
            1 - 3 * xi**2 + 2 * xi**3,
            Le * (xi - 2 * xi**2 + xi**3),
            3 * xi**2 - 2 * xi**3,
            Le * (-xi**2 + xi**3),
        ],
        dtype=float,
    )

def morison_quadratic_drag_force(
    fe,
    qdot_red,
    rho: float,
    width: float,
    drag_coefficient: float,
    *,
    n_gauss: int = 3,
) -> np.ndarray:
    """Reduced nonlinear Morison drag force f_d(qdot)."""
    rho = float(rho)
    width = float(width)
    cd = float(drag_coefficient)
    if rho <= 0:
        raise ValueError("rho must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if cd < 0:
        raise ValueError("drag_coefficient must be nonnegative")

    qdot_full = reduced_to_full_dofs(fe, qdot_red)
    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)
    f_full = np.zeros(fe.Ndof, dtype=float)

    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    for e in range(len(x_nodes) - 1):
        Le = x_nodes[e + 1] - x_nodes[e]
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        qde = qdot_full[dofs]

        for s, w in zip(gp, gw):
            xi = 0.5 * (s + 1.0)
            weight = 0.5 * Le * w
            N = hermite_shape_functions(xi, Le)
            wdot = float(N @ qde)
            load = 0.5 * rho * width * cd * abs(wdot) * wdot
            f_full[dofs] += N * load * weight

    return f_full[fe.free_dofs]


def equivalent_morison_damping_matrix(
    fe,
    omega: float,
    u_red,
    rho: float,
    width: float,
    drag_coefficient: float,
    *,
    n_gauss: int = 3,
) -> np.ndarray:
    """Amplitude-dependent equivalent damping for harmonic Morison drag."""
    rho = float(rho)
    width = float(width)
    cd = float(drag_coefficient)
    if rho <= 0:
        raise ValueError("rho must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if cd < 0:
        raise ValueError("drag_coefficient must be nonnegative")

    u_full = reduced_to_full_dofs(fe, np.asarray(u_red, dtype=complex))
    x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)
    C_full = np.zeros_like(fe.M, dtype=float)

    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    for e in range(len(x_nodes) - 1):
        Le = x_nodes[e + 1] - x_nodes[e]
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        ue = u_full[dofs]

        for s, w in zip(gp, gw):
            xi = 0.5 * (s + 1.0)
            weight = 0.5 * Le * w
            N = hermite_shape_functions(xi, Le)
            v_hat = 1j * omega * (N @ ue)
            c_eq = 0.5 * rho * width * cd * (8.0 / (3.0 * np.pi)) * abs(v_hat)
            C_full[np.ix_(dofs, dofs)] += c_eq * np.outer(N, N) * weight

    return C_full[np.ix_(fe.free_dofs, fe.free_dofs)]


def solve_morison_harmonic_response(
    fe,
    omega: float,
    voltage_vector,
    *,
    rho: float,
    width: float,
    added_mass_coefficient: float = 1.0,
    drag_coefficient: float = 1.0,
    linear_damping_per_length: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    relaxation: float = 0.7,
) -> dict:
    """Fixed-point nonlinear frequency response estimate with Morison drag."""
    max_iter = int(max_iter)
    tol = float(tol)
    relaxation = float(relaxation)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")

    M_added = linear_added_mass_matrix(fe, rho, width, added_mass_coefficient)
    C_linear = linear_hydrodynamic_damping_matrix(fe, linear_damping_per_length)

    voltage_vector = np.asarray(voltage_vector, dtype=complex)
    u = solve_harmonic_response(
        fe,
        omega,
        voltage_vector,
        extra_mass=M_added,
        extra_damping=C_linear,
    )

    converged = False
    for iteration in range(1, max_iter + 1):
        C_nl = equivalent_morison_damping_matrix(
            fe, omega, u, rho, width, drag_coefficient
        )
        u_new = solve_harmonic_response(
            fe,
            omega,
            voltage_vector,
            extra_mass=M_added,
            extra_damping=C_linear + C_nl,
        )

        denom = max(np.linalg.norm(u_new), 1e-30)
        rel_error = np.linalg.norm(u_new - u) / denom
        u = relaxation * u_new + (1.0 - relaxation) * u

        if rel_error < tol:
            converged = True
            break

    return {
        "response_red": u,
        "added_mass": M_added,
        "linear_damping": C_linear,
        "equivalent_nonlinear_damping": C_nl,
        "iterations": iteration,
        "relative_error": rel_error,
        "converged": converged,
    }

# -----------------------------------------------------------------------------
# Thrust Model Helpers
# -----------------------------------------------------------------------------

def tip_displacement_phasor(fe, response_red):
    """Return complex tip displacement phasor from reduced response."""
    return response_red[tip_reduced_index(fe)]


def tip_velocity_phasor(fe, response_red, omega: float):
    """Return complex tip velocity phasor for harmonic response."""
    return 1j * omega * tip_displacement_phasor(fe, response_red)


def tip_slope_phasor(fe, response_red):
    """Return complex tip slope phasor from reduced response."""
    return response_red[tip_slope_reduced_index(fe)]


def _validate_lighthill_inputs(rho: float, width: float, beta: float):
    rho = float(rho)
    width = float(width)
    beta = float(beta)
    if rho <= 0:
        raise ValueError("rho must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    return rho, width, beta


def lighthill_quiescent_thrust_from_tip_velocity(
    tip_velocity_hat,
    rho: float,
    width: float,
    beta: float = 1.0,
) -> float:
    """Mean thrust estimate in quiescent water from peak tip velocity phasor."""
    rho, width, beta = _validate_lighthill_inputs(rho, width, beta)
    return float(np.pi * rho * width**2 * beta / 16.0 * abs(tip_velocity_hat) ** 2)


def lighthill_thrust_with_swimming_speed(
    tip_velocity_hat,
    tip_slope_hat,
    swimming_speed: float,
    rho: float,
    width: float,
    beta: float = 1.0,
) -> float:
    """Mean Lighthill thrust estimate with finite swimming-speed correction.

    Uses peak phasors. The time averages are
    mean(v_tip**2) = 0.5*|v_tip_hat|**2 and
    mean(slope_tip**2) = 0.5*|slope_tip_hat|**2.
    """
    rho, width, beta = _validate_lighthill_inputs(rho, width, beta)
    swimming_speed = float(swimming_speed)
    if swimming_speed < 0:
        raise ValueError("swimming_speed must be nonnegative")

    m_virtual = 0.25 * np.pi * rho * width**2 * beta
    mean_v2 = 0.5 * abs(tip_velocity_hat) ** 2
    mean_slope2 = 0.5 * abs(tip_slope_hat) ** 2
    return float(0.5 * m_virtual * (mean_v2 - swimming_speed**2 * mean_slope2))


def lighthill_quiescent_thrust(
    fe,
    response_red,
    omega: float,
    rho: float,
    width: float,
    beta: float = 1.0,
) -> float:
    """Mean thrust estimate from a reduced harmonic response."""
    v_tip = tip_velocity_phasor(fe, response_red, omega)
    return lighthill_quiescent_thrust_from_tip_velocity(v_tip, rho, width, beta)


def lighthill_thrust(
    fe,
    response_red,
    omega: float,
    rho: float,
    width: float,
    beta: float = 1.0,
    swimming_speed: float = 0.0,
) -> float:
    """Mean Lighthill thrust estimate from a reduced harmonic response."""
    v_tip = tip_velocity_phasor(fe, response_red, omega)
    if float(swimming_speed) == 0.0:
        return lighthill_quiescent_thrust_from_tip_velocity(v_tip, rho, width, beta)
    return lighthill_thrust_with_swimming_speed(
        v_tip,
        tip_slope_phasor(fe, response_red),
        swimming_speed,
        rho,
        width,
        beta,
    )


# -----------------------------------------------------------------------------
# Frequency-Response Helpers
# -----------------------------------------------------------------------------

def dynamic_stiffness_from_fe(fe, omega, extra_mass=None, extra_damping=None):
    """Return reduced dynamic stiffness Z = K + i*omega*C - omega**2*M."""
    M = fe.M_red if extra_mass is None else fe.M_red + extra_mass
    C0 = fe.effective_damping_matrix()
    C = C0 if extra_damping is None else C0 + extra_damping
    return fe.K_red + 1j * omega * C - omega**2 * M


def solve_harmonic_response(fe, omega, voltage_vector, extra_mass=None, extra_damping=None):
    """Solve Z(omega) U = Gamma V for reduced mechanical response U."""
    Z = dynamic_stiffness_from_fe(fe, omega, extra_mass=extra_mass, extra_damping=extra_damping)
    return np.linalg.solve(Z, fe.Gamma_red @ np.asarray(voltage_vector, dtype=complex))


def response_columns(fe, omega, extra_mass=None, extra_damping=None):
    """Return reduced displacement columns per unit patch voltage at omega."""
    Z = dynamic_stiffness_from_fe(fe, omega, extra_mass=extra_mass, extra_damping=extra_damping)
    return np.linalg.solve(Z, fe.Gamma_red)


def admittance_matrix(fe, omega, extra_mass=None, extra_damping=None):
    """Return dry multi-patch electrical admittance for peak voltage phasors.

    The convention is I = Y V with
    Y = i*omega*Cp + i*omega*Gamma.T*Z(omega)^-1*Gamma.
    """
    H = response_columns(
        fe,
        omega,
        extra_mass=extra_mass,
        extra_damping=extra_damping,
    )
    return 1j * omega * fe.Cp + 1j * omega * (fe.Gamma_red.T @ H)


def complex_power_from_peak_phasors(voltage_vector, current_vector):
    """Return complex average power S = 0.5*V^H*I for peak phasors."""
    return 0.5 * np.vdot(
        np.asarray(voltage_vector, dtype=complex),
        np.asarray(current_vector, dtype=complex),
    )


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


def frequency_response_sweep(ode, omega_vec, *, show_progress: bool = True):
    """Compute full coupled frequency response over an angular-frequency vector."""
    omega_vec = np.asarray(omega_vec, dtype=float)
    ndof = ode.M.shape[0]
    N_mech = ode.N_mech
    X = np.zeros((len(omega_vec), ndof), dtype=complex)

    iterator = tqdm(omega_vec, desc="Frequency response sweep") if show_progress else omega_vec
    for k, w in enumerate(iterator):
        X[k] = frequency_response_linear(ode, w)

    u = X[:, :N_mech:2]
    q = X[:, N_mech:]
    u_dot = 1j * omega_vec[:, None] * u
    v = 1j * omega_vec[:, None] * q

    return {"omega": omega_vec, "freq": omega_vec / (2 * np.pi), "u": u, "u_dot": u_dot, "q": q, "v": v, "X": X}


def mechanical_frequency_response_sweep(ode, omega_vec, *, show_progress: bool = True, modal: bool = False, **modal_kwargs):
    """Compute mechanical-only frequency response sweep."""
    omega_vec = np.asarray(omega_vec, dtype=float)
    N_mech = ode.N_mech
    X = np.zeros((len(omega_vec), N_mech), dtype=complex)
    iterator = tqdm(omega_vec, desc="Mechanical frequency response sweep") if show_progress else omega_vec
    for k, w in enumerate(iterator):
        if modal:
            X[k] = frequency_response_mechanical_modal(ode, w, **modal_kwargs)
        else:
            X[k] = frequency_response_mechanical(ode, w)

    u = X[:, :N_mech:2]
    u_dot = 1j * omega_vec[:, None] * u
    return {"omega": omega_vec, "freq": omega_vec / (2 * np.pi), "u": u, "u_dot": u_dot, "X": X}
