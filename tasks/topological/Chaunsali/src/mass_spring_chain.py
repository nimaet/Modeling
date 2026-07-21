"""Core reusable model for 1D nonlinear mass-spring chains.

This module contains the physical chain, state packing, time integration, and
simple force helpers.  Periodic sweep workflows live in ``src.sweep_studies``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


ArrayLike = np.ndarray | Iterable[float]
ForceFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ChainResponse:
    """Time-domain response returned by :meth:`MassSpringChain.simulate`.

    The paper uses ``u`` and ``v`` for site-1 and site-2 displacement
    envelopes, so this response uses explicit displacement/velocity names.
    """

    t: np.ndarray
    y: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    energy: np.ndarray | None = None


@dataclass(frozen=True)
class SolveOptions:
    """Options passed to ``scipy.integrate.solve_ivp``."""

    method: str = "RK45"
    rtol: float = 1e-9
    atol: float = 1e-10

class MassSpringChain:
    """Finite 1D chain with fixed end supports and cubic spring forces.

    There are ``N`` moving masses and ``N + 1`` springs. Spring ``0`` connects
    the left wall to mass ``0``; spring ``N`` connects mass ``N - 1`` to the
    right wall. Spring extension is defined as ``Delta_n = u_n - u_{n-1}``
    after adding zero-displacement boundary nodes.
    """

    def __init__(
        self,
        mass: float,
        k_springs: ArrayLike,
        alpha_springs: ArrayLike | None = None,
        c_dashpot: float | ArrayLike = 0.0,
    ):
        self.mass = float(mass)
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")

        self.k_springs = np.asarray(k_springs, dtype=float)
        if self.k_springs.ndim != 1 or self.k_springs.size < 2:
            raise ValueError("k_springs must be a 1D array with at least two springs")

        if alpha_springs is None:
            self.alpha_springs = np.zeros_like(self.k_springs)
        else:
            self.alpha_springs = np.asarray(alpha_springs, dtype=float)

        if self.alpha_springs.shape != self.k_springs.shape:
            raise ValueError("alpha_springs must have the same shape as k_springs")

        self.c_dashpot = self._coerce_spring_array(c_dashpot, "c_dashpot")
        self.N = self.k_springs.size - 1

    @classmethod
    def alternating(
        cls,
        N: int,
        mass: float = 1.0,
        k: float = 1.0,
        gamma: float = 0.0,
        alpha_even: float = 0.0,
        alpha_odd: float = 0.0,
        c_dashpot: float | ArrayLike = 0.0,
        start: str = "low",
    ) -> "MassSpringChain":
        """Build the alternating chain used in ``nonlinear_chain.ipynb``.

        With ``start="low"``, even springs use ``k * (1 - gamma)`` and
        ``alpha_even``; odd springs use ``k * (1 + gamma)`` and ``alpha_odd``.
        Use ``alpha_even=alpha_hard`` and ``alpha_odd=alpha_soft`` to reproduce
        the notebook convention.
        """
        spring_index = np.arange(int(N) + 1)
        even = spring_index % 2 == 0
        if start == "high":
            even = ~even
        elif start != "low":
            raise ValueError("start must be either 'low' or 'high'")

        k_springs = float(k) * np.where(even, 1.0 - gamma, 1.0 + gamma)
        alpha_springs = np.where(even, alpha_even, alpha_odd)
        return cls(
            mass=mass,
            k_springs=k_springs,
            alpha_springs=alpha_springs,
            c_dashpot=c_dashpot,
        )

    def copy_with(self, **changes) -> "MassSpringChain":
        """Return a modified chain without mutating the original object."""
        data = {
            "mass": self.mass,
            "k_springs": self.k_springs.copy(),
            "alpha_springs": self.alpha_springs.copy(),
            "c_dashpot": self.c_dashpot.copy(),
        }
        data.update(changes)
        return MassSpringChain(**data)

    def zeros_state(self) -> np.ndarray:
        """Return ``[displacement0, velocity0]`` initialized to zero."""
        return np.zeros(2 * self.N)

    def state(
        self,
        displacement0: ArrayLike | None = None,
        velocity0: ArrayLike | None = None,
    ) -> np.ndarray:
        """Pack displacement and velocity arrays into the solver state."""
        displacement = (
            np.zeros(self.N)
            if displacement0 is None
            else self._coerce_mass_array(displacement0, "displacement0")
        )
        velocity = (
            np.zeros(self.N)
            if velocity0 is None
            else self._coerce_mass_array(velocity0, "velocity0")
        )
        return np.concatenate([displacement, velocity])

    def split_state(self, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Split a packed state into displacement and velocity vectors."""
        y = np.asarray(y, dtype=float)
        if y.shape != (2 * self.N,):
            raise ValueError(f"state must have shape ({2 * self.N},)")
        return y[: self.N], y[self.N :]

    def spring_extensions(self, displacement: ArrayLike) -> np.ndarray:
        """Return spring extensions for one displacement vector."""
        displacement = self._coerce_mass_array(displacement, "displacement")
        return np.diff(np.concatenate(([0.0], displacement, [0.0])))

    def spring_extensions_history(self, displacement: ArrayLike) -> np.ndarray:
        """Return spring extensions for a displacement history ``(nt, N)``."""
        displacement = np.asarray(displacement, dtype=float)
        if displacement.ndim == 1:
            return self.spring_extensions(displacement)
        if displacement.ndim != 2 or displacement.shape[1] != self.N:
            raise ValueError(f"displacement history must have shape (nt, {self.N})")
        boundaries = np.zeros((displacement.shape[0], self.N + 2), dtype=float)
        boundaries[:, 1:-1] = displacement
        return np.diff(boundaries, axis=1)

    def spring_forces(self, displacement: ArrayLike) -> np.ndarray:
        """Return nonlinear spring forces ``k*Delta + alpha*Delta**3``."""
        delta = self.spring_extensions(displacement)
        return self.k_springs * delta + self.alpha_springs * delta**3

    def damping_forces(self, velocity: ArrayLike) -> np.ndarray:
        """Return dashpot forces on each spring from relative velocities."""
        delta_velocity = self.spring_extensions(velocity)
        return self.c_dashpot * delta_velocity

    def acceleration(
        self,
        t: float,
        displacement: ArrayLike,
        velocity: ArrayLike,
        force: ForceFunction | None = None,
    ) -> np.ndarray:
        """Return accelerations for displacement, velocity, and external force."""
        external = np.zeros(self.N) if force is None else np.asarray(force(t), dtype=float)
        if external.shape != (self.N,):
            raise ValueError(f"external force must have shape ({self.N},)")
        elastic = self.spring_forces(displacement)
        damping = self.damping_forces(velocity)
        return (elastic[1:] - elastic[:-1] + damping[1:] - damping[:-1] + external) / self.mass

    def rhs(self, t: float, y: ArrayLike, force: ForceFunction | None = None) -> np.ndarray:
        """First-order state derivative for ``solve_ivp``."""
        displacement, velocity = self.split_state(y)
        return np.concatenate([velocity, self.acceleration(t, displacement, velocity, force)])

    def make_rhs(self, force: ForceFunction | None = None) -> Callable[[float, np.ndarray], np.ndarray]:
        """Create a ``solve_ivp`` RHS closure."""
        return lambda t, y: self.rhs(t, y, force)

    def simulate(
        self,
        t_eval: ArrayLike,
        y0: ArrayLike | None = None,
        force: ForceFunction | None = None,
        options: SolveOptions | None = None,
        include_energy: bool = False,
    ) -> ChainResponse:
        """Integrate the chain on a prescribed time grid."""
        options = SolveOptions() if options is None else options
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("t_eval must be a 1D array with at least two entries")
        y0 = self.zeros_state() if y0 is None else np.asarray(y0, dtype=float)
        if y0.shape != (2 * self.N,):
            raise ValueError(f"y0 must have shape ({2 * self.N},)")

        sol = solve_ivp(
            self.make_rhs(force),
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method=options.method,
            t_eval=t_eval,
            rtol=options.rtol,
            atol=options.atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        y = sol.y.T
        displacement = y[:, : self.N]
        velocity = y[:, self.N :]
        energy = self.total_energy(displacement, velocity) if include_energy else None
        return ChainResponse(
            t=sol.t,
            y=y,
            displacement=displacement,
            velocity=velocity,
            energy=energy,
        )

    def mass_matrix(self) -> np.ndarray:
        """Return the lumped mass matrix."""
        return self.mass * np.eye(self.N)

    def stiffness_matrix(self) -> np.ndarray:
        """Return the linearized fixed-boundary stiffness matrix."""
        K = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            k_left = self.k_springs[i]
            k_right = self.k_springs[i + 1]
            K[i, i] = k_left + k_right
            if i > 0:
                K[i, i - 1] = -k_left
            if i < self.N - 1:
                K[i, i + 1] = -k_right
        return K

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return natural frequencies and mode shapes of the linearized chain."""
        omega2, modes = eigh(self.stiffness_matrix(), self.mass_matrix())
        omega = np.sqrt(np.clip(omega2, 0.0, None))
        return omega, modes

    def total_energy(self, displacement: ArrayLike, velocity: ArrayLike) -> np.ndarray:
        """Return mechanical energy for a state or response history."""
        displacement = np.asarray(displacement, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if displacement.ndim == 1:
            displacement = displacement[None, :]
        if velocity.ndim == 1:
            velocity = velocity[None, :]
        if displacement.shape != velocity.shape or displacement.shape[1] != self.N:
            raise ValueError(f"displacement and velocity must both have shape (nt, {self.N})")
        delta = self.spring_extensions_history(displacement)
        kinetic = 0.5 * self.mass * np.sum(velocity**2, axis=1)
        potential = np.sum(
            0.5 * self.k_springs * delta**2 + 0.25 * self.alpha_springs * delta**4,
            axis=1,
        )
        return kinetic + potential

    def _coerce_mass_array(self, values: ArrayLike, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.N,):
            raise ValueError(f"{name} must have shape ({self.N},)")
        return values

    def _coerce_spring_array(self, values: float | ArrayLike, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            return np.full_like(self.k_springs, float(values))
        if values.shape != self.k_springs.shape:
            raise ValueError(f"{name} must be scalar or have shape {self.k_springs.shape}")
        return values


def sinusoidal_force(
    N: int,
    mass_index: int,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> ForceFunction:
    """Return ``F_i(t) = amplitude*sin(frequency*t + phase)`` at one mass."""
    N = int(N)
    mass_index = int(mass_index)
    if mass_index < 0 or mass_index >= N:
        raise ValueError("mass_index is outside the chain")

    def force(t: float) -> np.ndarray:
        values = np.zeros(N)
        values[mass_index] = amplitude * np.sin(frequency * t + phase)
        return values

    return force


def scalar_force_on_mass(
    N: int,
    mass_index: int,
    scalar_force: Callable[[float], float],
) -> ForceFunction:
    """Lift a scalar force function into a vector force on one mass."""
    N = int(N)
    mass_index = int(mass_index)
    if mass_index < 0 or mass_index >= N:
        raise ValueError("mass_index is outside the chain")

    def force(t: float) -> np.ndarray:
        values = np.zeros(N)
        values[mass_index] = scalar_force(t)
        return values

    return force
