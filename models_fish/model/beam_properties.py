"""Physical parameters and section-property utilities for piezoelectric beam FE models."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PiezoBeamParams:
    """Material, geometric, electrical, and damping parameters for a piezo beam.

    Notes
    -----
    The default ``xL``/``xR`` patch array is still generated from ``w_p``, ``w_s``,
    and ``n_patches`` for backward compatibility. Optimizer code can override the
    actual FE layout by attaching a ``geometry`` attribute to an instance before
    constructing ``PiezoBeamFE``.
    """

    # ===================== Geometry =====================
    w_p: float = 10e-3          # patch width [m]
    w_s: float = 0.265625e-3    # spacing between patches [m]
    n_patches: int = 31         # number of default unit-cell patches
    b: float = 10e-3            # beam width [m]
    hp: float = 0.252e-3        # piezo thickness [m]
    hs: float = 0.51e-3         # substrate thickness [m]

    # ===================== Materials =====================
    rho_p: float = 7600.0       # piezo density [kg/m^3]
    rho_s: float = 2700.0       # substrate density [kg/m^3]
    E_s: float = 70e9           # substrate Young's modulus [Pa]
    s11: float = 1.5873e-11     # piezo compliance [m^2/N]
    s44: float = 4.75e-11       # piezo shear compliance [m^2/N]
    d31: float = -1.45e-10      # piezo strain constant [C/N]
    eps0: float = 8.854e-12     # vacuum permittivity [F/m]
    eps_r: float = 1700.0       # relative permittivity [-]
    nu_s: float = 0.33          # substrate Poisson ratio [-]

    # Rayleigh damping calibration frequencies [rad/s]
    omega_p: float = 2 * np.pi * 100
    omega_q: float = 2 * np.pi * 6000

    # Optional modal damping lookup used by FE3.PiezoBeamFE.damping_matrix_from_modal_damping.
    zeta_dict: dict = field(
        default_factory=lambda: {
            1: 0.02, 2: 0.025, 3: 0.03, 4: 0.007, 5: 0.0075,
            6: 0.0085, 7: 0.008, 8: 0.007, 9: 0.007, 10: 0.0075,
            11: 0.008, 12: 0.009, 13: 0.01, 14: 0.012,
            15: 0.014, 16: 0.014, 17: 0.013, 18: 0.013,
            19: 0.015, 20: 0.023, 21: 0.024, 22: 0.025,
            "rest": 0.4,
        }
    )

    # ===================== Derived quantities =====================
    L_b: float = field(init=False)
    S: int = field(init=False)
    Q: int = field(init=False)
    E_p: float = field(init=False)
    G_p: float = field(init=False)
    G_s: float = field(init=False)
    e31: float = field(init=False)
    eps33_bar: float = field(init=False)
    eps33: float = field(init=False)
    m: float = field(init=False)
    YI: float = field(init=False)
    YI_s: float = field(init=False)
    Cp_scalar: float = field(init=False)
    Cp: np.ndarray = field(init=False)
    theta_mech: float = field(init=False)
    xL: np.ndarray = field(init=False)
    xR: np.ndarray = field(init=False)
    c_alpha: float = field(init=False)
    c_beta: float = field(init=False)

    _zeta_p: float = field(default=0.0001, repr=False)
    _zeta_q: float = field(default=0.0001, repr=False)

    def __post_init__(self) -> None:
        self._recompute_derived_quantities()

    def _recompute_derived_quantities(self) -> None:
        """Recompute all derived quantities after base parameters change."""
        self.Q = int(self.n_patches)
        self.S = self.Q

        # Elastic moduli
        self.E_p = 1.0 / self.s11
        self.G_p = 1.0 / self.s44
        self.G_s = self.E_s / (2.0 * (1.0 + self.nu_s))

        # Piezoelectric and dielectric constants
        self.e31 = self.d31 / self.s11
        self.eps33_bar = self.eps0 * self.eps_r
        self.eps33 = self.eps33_bar - self.d31**2 / self.s11

        # Default repeated patch locations
        j = np.arange(1, self.S + 1)
        self.xL = (j - 1) * self.w_p + j * self.w_s
        self.xR = self.xL + self.w_p
        self.L_b = float(self.xR[-1] + self.w_s) if self.S > 0 else 0.0

        # Mass and bending stiffness
        self.m = self.b * (self.rho_s * self.hs + 2.0 * self.rho_p * self.hp)

        term1 = self.E_s * self.hs**3 / 8.0
        term2 = self.E_p * ((self.hp + self.hs / 2.0) ** 3 - self.hs**3 / 8.0)
        self.YI = 2.0 * self.b / 3.0 * (term1 + term2)
        self.YI_s = self.b * self.E_s * self.hs**3 / 12.0

        # Capacitance and electromechanical coupling
        self.Cp_scalar = 2.0 * self.eps33 * self.w_p * self.b / self.hp
        self.Cp = self.Cp_scalar * np.ones(self.S)

        hpc = 0.5 * (self.hp + self.hs)
        self.theta_mech = 2.0 * self.e31 * self.b * hpc

        self._update_c_alpha_beta()

    def sync_patch_count(self, n_patches: int) -> None:
        """Update patch-count bookkeeping for externally supplied geometries.

        This does not overwrite an attached ``geometry``. It only keeps ``S``,
        ``Q``, ``n_patches``, and the capacitance vector consistent with the
        number of active piezo regions in that geometry.
        """
        self.n_patches = int(n_patches)
        self.Q = int(n_patches)
        self.S = int(n_patches)
        self.Cp = self.Cp_scalar * np.ones(self.S)

    def _update_c_alpha_beta(self) -> None:
        """Compute Rayleigh damping coefficients from two calibration points."""
        A = np.array(
            [
                [self.omega_p / (2 * self.YI), 1 / (2 * self.m * self.omega_p)],
                [self.omega_q / (2 * self.YI), 1 / (2 * self.m * self.omega_q)],
            ]
        )
        b = np.array([self._zeta_p, self._zeta_q])
        self.c_beta, self.c_alpha = np.linalg.solve(A, b)

    @property
    def zeta_p(self) -> float:
        return self._zeta_p

    @zeta_p.setter
    def zeta_p(self, value: float) -> None:
        self._zeta_p = float(value)
        self._update_c_alpha_beta()

    @property
    def zeta_q(self) -> float:
        return self._zeta_q

    @zeta_q.setter
    def zeta_q(self, value: float) -> None:
        self._zeta_q = float(value)
        self._update_c_alpha_beta()

    def rayleigh_zeta(self, omega: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the calibrated Rayleigh damping ratio at angular frequency."""
        omega = np.asarray(omega, dtype=float)
        return self.c_alpha / (2 * omega * self.m) + self.c_beta * omega / (2 * self.YI)

    def plot_zeta_vs_omega(self, omega_range=None, ax=None):
        """Plot damping ratio versus frequency. Kept for backward compatibility."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if omega_range is None:
            omega_range = np.linspace(0.5 * self.omega_p, 2 * self.omega_q, 500)

        zeta = self.rayleigh_zeta(omega_range)
        ax.plot(omega_range / (2 * np.pi), 100 * zeta, linewidth=2, label="Rayleigh damping")
        ax.plot(
            self.omega_p / (2 * np.pi),
            100 * self._zeta_p,
            "o",
            markersize=8,
            label=f"Point 1: ({self.omega_p/(2*np.pi):.1f} Hz, {100*self._zeta_p:.2f}%)",
        )
        ax.plot(
            self.omega_q / (2 * np.pi),
            100 * self._zeta_q,
            "s",
            markersize=8,
            label=f"Point 2: ({self.omega_q/(2*np.pi):.1f} Hz, {100*self._zeta_q:.2f}%)",
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Damping Ratio ζ [%]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def homogenized_parameters(self, K_i, K_c, R_c) -> dict:
        """Compute homogenized parameters used by the ROM/perturbation model."""
        K_i_eff = float(np.mean(K_i)) if np.ndim(K_i) > 0 else float(K_i)

        m_bar = self.m
        EI_bar = self.YI
        Cp_bar = self.Cp_scalar / self.w_p
        L_bar = R_c / K_i_eff * self.w_p
        Lc_bar = R_c / K_c * self.w_p
        theta_bar = self.theta_mech

        return {
            "m_bar": m_bar,
            "EI_bar": EI_bar,
            "Cp_bar": Cp_bar,
            "L_bar": L_bar,
            "Lc_bar": Lc_bar,
            "theta_bar": theta_bar,
        }

    def nondimensional_scales(self, K_i, K_c, R_c) -> dict:
        """Compute nondimensional scales required by the perturbation solution."""
        h = self.homogenized_parameters(K_i, K_c, R_c)
        m_bar = h["m_bar"]
        EI_bar = h["EI_bar"]
        Cp_bar = h["Cp_bar"]
        L_bar = h["L_bar"]
        Lc_bar = h["Lc_bar"]
        theta_bar = h["theta_bar"]

        t0 = np.sqrt(Cp_bar * L_bar)
        x0 = (EI_bar * Cp_bar * L_bar / m_bar) ** 0.25
        lambda0 = np.sqrt(abs(Lc_bar / L_bar))
        w0 = np.sqrt(Cp_bar * abs(Lc_bar) / (m_bar * L_bar))
        theta_tilde = theta_bar / np.sqrt(Cp_bar * EI_bar)

        return {
            "t0": float(t0),
            "x0": float(x0),
            "lambda0": float(lambda0),
            "w0": float(w0),
            "theta_tilde": float(theta_tilde),
        }


def compute_EI_and_rhoA(E_layers, rho_layers, h_layers, b_layers):
    """Compute bending stiffness EI and mass per unit length rhoA.

    The section is assumed symmetric about the mid-plane. Layer 0 is the
    center/mid-layer and layers 1, 2, ... are mirrored outward.
    """
    E_layers = np.asarray(E_layers, dtype=float)
    rho_layers = np.asarray(rho_layers, dtype=float)
    h_layers = np.asarray(h_layers, dtype=float)
    b_layers = np.asarray(b_layers, dtype=float)

    if not (len(E_layers) == len(rho_layers) == len(h_layers) == len(b_layers)):
        raise ValueError("E_layers, rho_layers, h_layers, and b_layers must have the same length")

    y = [h_layers[0] / 2.0]
    for i in range(1, len(h_layers)):
        y.append(y[i - 1] + h_layers[i])

    EI_half = 0.0
    rhoA_half = 0.0
    for i, (E, rho, h, b) in enumerate(zip(E_layers, rho_layers, h_layers, b_layers)):
        if i == 0:
            EI_half += E * b * y[i] ** 3 / 3.0
            rhoA_half += rho * b * h / 2.0
        else:
            EI_half += E * b * (y[i] ** 3 - y[i - 1] ** 3) / 3.0
            rhoA_half += rho * b * h

    return 2.0 * EI_half, 2.0 * rhoA_half
