"""Periodic forcing and sweep utilities for mass-spring-chain studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed

from .mass_spring_chain import ArrayLike, MassSpringChain, SolveOptions, sinusoidal_force


@dataclass(frozen=True)
class PeriodicCaseResult:
	"""Compact result returned by a periodic forced response case."""

	amplitude: float
	frequency: float
	period: float
	observed_mass: int
	observed_indices: np.ndarray
	t: np.ndarray
	displacement_obs: np.ndarray
	velocity_obs: np.ndarray
	t_strobe: np.ndarray
	displacement_poincare: np.ndarray
	velocity_poincare: np.ndarray
	peak_displacement: float
	rms_displacement: float

	def as_dict(self) -> dict[str, object]:
		"""Return a notebook-friendly dictionary."""
		return {
			"amplitude": self.amplitude,
			"frequency": self.frequency,
			"period": self.period,
			"observed_mass": self.observed_mass,
			"observed_indices": self.observed_indices,
			"t": self.t,
			"displacement_obs": self.displacement_obs,
			"velocity_obs": self.velocity_obs,
			"t_strobe": self.t_strobe,
			"displacement_poincare": self.displacement_poincare,
			"velocity_poincare": self.velocity_poincare,
			"peak_displacement": self.peak_displacement,
			"rms_displacement": self.rms_displacement,
		}


@dataclass(frozen=True)
class PeriodicSampling:
	"""Sampling controls for periodic forcing studies."""

	points_per_period: int = 100
	n_periods_total: int = 500
	n_periods_keep: int = 100

	def time_grid(self, frequency: float) -> tuple[float, np.ndarray]:
		if frequency <= 0.0:
			raise ValueError("frequency must be positive for periodic sampling")
		period = 2.0 * np.pi / float(frequency)
		n_steps = int(self.n_periods_total) * int(self.points_per_period)
		return period, np.linspace(0.0, self.n_periods_total * period, n_steps + 1)


def stroboscopic_sample(
	t: ArrayLike,
	response: ArrayLike,
	points_per_period: int,
	n_periods_keep: int,
) -> dict[str, np.ndarray]:
	"""Sample a response once per forcing period."""
	t = np.asarray(t, dtype=float)
	response = np.asarray(response)
	strobe_indices = np.arange(0, len(t), int(points_per_period))
	if response.ndim == 1:
		response_strobe = response[strobe_indices]
	else:
		response_strobe = response[..., strobe_indices]
	return {
		"t_strobe": t[strobe_indices][-int(n_periods_keep) :],
		"response_poincare": response_strobe[..., -int(n_periods_keep) :],
	}


class PeriodicSweepStudy:
	"""Run amplitude/frequency sweeps for a :class:`MassSpringChain`."""

	def __init__(
		self,
		chain: MassSpringChain,
		forced_mass: int,
		phase_mass_indices: Iterable[int] | None = None,
		y0: ArrayLike | None = None,
		sampling: PeriodicSampling | None = None,
		solve_options: SolveOptions | None = None,
		observed_mass: int | None = None,
	):
		self.chain = chain
		self.forced_mass = int(forced_mass)
		self.phase_mass_indices = _unique_indices(
			[] if phase_mass_indices is None else phase_mass_indices,
			chain.N,
		)
		self.y0 = chain.zeros_state() if y0 is None else np.asarray(y0, dtype=float)
		if self.y0.shape != (2 * chain.N,):
			raise ValueError(f"y0 must have shape ({2 * chain.N},)")
		self.sampling = PeriodicSampling() if sampling is None else sampling
		self.solve_options = SolveOptions() if solve_options is None else solve_options
		self.observed_mass = self.forced_mass if observed_mass is None else int(observed_mass)
		_validate_mass_index(self.forced_mass, chain.N, "forced_mass")
		_validate_mass_index(self.observed_mass, chain.N, "observed_mass")

	def with_options(
		self,
		*,
		sampling: PeriodicSampling | None = None,
		solve_options: SolveOptions | None = None,
		y0: ArrayLike | None = None,
	) -> "PeriodicSweepStudy":
		"""Return a copy with selected settings replaced."""
		return PeriodicSweepStudy(
			chain=self.chain,
			forced_mass=self.forced_mass,
			phase_mass_indices=self.phase_mass_indices,
			y0=self.y0 if y0 is None else y0,
			sampling=self.sampling if sampling is None else sampling,
			solve_options=self.solve_options if solve_options is None else solve_options,
			observed_mass=self.observed_mass,
		)

	@staticmethod
	def default_n_jobs(values: Iterable[object], n_jobs: int | None = None) -> int:
		values = list(values)
		if n_jobs is not None:
			return int(n_jobs)
		return min(len(values), max(1, os.cpu_count() or 1), 30)

	def run_case(
		self,
		amplitude: float,
		frequency: float,
		*,
		excited_mass: int | None = None,
		phase_indices: Iterable[int] | None = None,
		y0: ArrayLike | None = None,
		sampling: PeriodicSampling | None = None,
		solve_options: SolveOptions | None = None,
		observed_mass: int | None = None,
		as_dict: bool = False,
	) -> PeriodicCaseResult | dict[str, object]:
		excited_mass = self.forced_mass if excited_mass is None else int(excited_mass)
		_validate_mass_index(excited_mass, self.chain.N, "excited_mass")
		observed_mass = self.observed_mass if observed_mass is None else int(observed_mass)
		_validate_mass_index(observed_mass, self.chain.N, "observed_mass")

		phase_indices = self.phase_mass_indices if phase_indices is None else _unique_indices(phase_indices, self.chain.N)
		sampling = self.sampling if sampling is None else sampling
		solve_options = self.solve_options if solve_options is None else solve_options
		y0 = self.y0 if y0 is None else np.asarray(y0, dtype=float)

		period, t_eval = sampling.time_grid(frequency)
		force = sinusoidal_force(self.chain.N, excited_mass, amplitude, frequency)
		out = self.chain.simulate(t_eval=t_eval, y0=y0, force=force, options=solve_options)

		observed_indices = np.asarray(sorted(set([excited_mass, observed_mass, *phase_indices])), dtype=int)
		displacement_obs = out.displacement[:, observed_indices]
		velocity_obs = out.velocity[:, observed_indices]
		keep_start = max(0, len(out.t) - sampling.n_periods_keep * sampling.points_per_period)
		obs_col = list(observed_indices).index(observed_mass)

		displacement_strobe = stroboscopic_sample(
			out.t,
			displacement_obs.T,
			sampling.points_per_period,
			sampling.n_periods_keep,
		)
		velocity_strobe = stroboscopic_sample(
			out.t,
			velocity_obs.T,
			sampling.points_per_period,
			sampling.n_periods_keep,
		)

		result = PeriodicCaseResult(
			amplitude=float(amplitude),
			frequency=float(frequency),
			period=period,
			observed_mass=observed_mass,
			observed_indices=observed_indices,
			t=out.t,
			displacement_obs=displacement_obs,
			velocity_obs=velocity_obs,
			t_strobe=displacement_strobe["t_strobe"],
			displacement_poincare=displacement_strobe["response_poincare"],
			velocity_poincare=velocity_strobe["response_poincare"],
			peak_displacement=float(np.max(np.abs(displacement_obs[keep_start:, obs_col]))),
			rms_displacement=float(np.sqrt(np.mean(displacement_obs[keep_start:, obs_col] ** 2))),
		)
		return result.as_dict() if as_dict else result

	def amplitude_sweep(
		self,
		amplitudes: Iterable[float],
		frequency: float,
		*,
		n_jobs: int | None = None,
		as_dict: bool = False,
		**case_options,
	) -> list[PeriodicCaseResult] | list[dict[str, object]]:
		amplitudes = list(amplitudes)
		n_jobs = self.default_n_jobs(amplitudes, n_jobs)
		if n_jobs == 1:
			return [
				self.run_case(a, frequency, as_dict=as_dict, **case_options)
				for a in amplitudes
			]
		return Parallel(n_jobs=n_jobs, verbose=12)(
			delayed(self.run_case)(a, frequency, as_dict=as_dict, **case_options)
			for a in amplitudes
		)

	def frequency_sweep(
		self,
		frequencies: Iterable[float],
		amplitude: float,
		*,
		n_jobs: int | None = None,
		as_dict: bool = False,
		**case_options,
	) -> list[PeriodicCaseResult] | list[dict[str, object]]:
		frequencies = list(frequencies)
		n_jobs = self.default_n_jobs(frequencies, n_jobs)
		if n_jobs == 1:
			return [
				self.run_case(amplitude, w, as_dict=as_dict, **case_options)
				for w in frequencies
			]
		return Parallel(n_jobs=n_jobs, verbose=12)(
			delayed(self.run_case)(amplitude, w, as_dict=as_dict, **case_options)
			for w in frequencies
		)

	def run_amplitude_sweep(self, amplitudes, frequency, **kwargs):
		kwargs = _legacy_case_kwargs(kwargs)
		return self.amplitude_sweep(amplitudes, frequency, as_dict=True, **kwargs)

	def run_frequency_sweep(self, frequencies, amplitude, **kwargs):
		kwargs = _legacy_case_kwargs(kwargs)
		return self.frequency_sweep(frequencies, amplitude, as_dict=True, **kwargs)


class PeriodicChainSweepStudy(PeriodicSweepStudy):
	"""Constructor-compatible periodic sweep helper."""

	def __init__(
		self,
		N,
		m,
		k_springs,
		alpha_springs,
		c_dashpot,
		forced_mass,
		phase_mass_indices,
		y0,
		points_per_period=100,
		n_periods_total=10000,
		n_periods_keep=5000,
		observed_mass=None,
		default_rtol=1e-7,
		default_atol=1e-9,
	):
		chain = MassSpringChain(
			mass=m,
			k_springs=k_springs,
			alpha_springs=alpha_springs,
			c_dashpot=c_dashpot,
		)
		if int(N) != chain.N:
			raise ValueError(f"N={N} is inconsistent with {len(k_springs)} springs")
		super().__init__(
			chain=chain,
			forced_mass=forced_mass,
			phase_mass_indices=phase_mass_indices,
			y0=y0,
			sampling=PeriodicSampling(
				points_per_period=points_per_period,
				n_periods_total=n_periods_total,
				n_periods_keep=n_periods_keep,
			),
			solve_options=SolveOptions(rtol=default_rtol, atol=default_atol),
			observed_mass=observed_mass,
		)

	@property
	def N(self) -> int:
		return self.chain.N

	@property
	def m(self) -> float:
		return self.chain.mass

	@property
	def k_springs(self) -> np.ndarray:
		return self.chain.k_springs

	@property
	def alpha_springs(self) -> np.ndarray:
		return self.chain.alpha_springs

	@property
	def c_dashpot(self) -> np.ndarray:
		return self.chain.c_dashpot

	@property
	def points_per_period(self) -> int:
		return self.sampling.points_per_period

	@property
	def n_periods_total(self) -> int:
		return self.sampling.n_periods_total

	@property
	def n_periods_keep(self) -> int:
		return self.sampling.n_periods_keep

	@property
	def default_rtol(self) -> float:
		return self.solve_options.rtol

	@property
	def default_atol(self) -> float:
		return self.solve_options.atol

	def spring_extensions(self, displacement):
		return self.chain.spring_extensions(displacement)

	def spring_forces(self, displacement):
		return self.chain.spring_forces(displacement)

	def damping_force(self, velocity):
		return self.chain.damping_forces(velocity)

	def make_periodic_force(self, amplitude, frequency, excited_mass=None):
		excited_mass = self.forced_mass if excited_mass is None else excited_mass
		return sinusoidal_force(self.N, excited_mass, amplitude, frequency)

	def acceleration(self, t, displacement, velocity, external_force):
		return self.chain.acceleration(t, displacement, velocity, external_force)

	def rhs(self, t, y, external_force):
		return self.chain.rhs(t, y, external_force)

	def make_rhs(self, external_force):
		return self.chain.make_rhs(external_force)

	def stroboscopic_sample(self, t, response, points_per_period=None, n_periods_keep=None):
		return stroboscopic_sample(
			t,
			response,
			self.points_per_period if points_per_period is None else points_per_period,
			self.n_periods_keep if n_periods_keep is None else n_periods_keep,
		)

	def simulate_chain(
		self,
		t_eval,
		external_force=None,
		y0=None,
		method="RK45",
		rtol=None,
		atol=None,
	):
		options = replace(
			self.solve_options,
			method=method,
			rtol=self.default_rtol if rtol is None else rtol,
			atol=self.default_atol if atol is None else atol,
		)
		response = self.chain.simulate(
			t_eval=t_eval,
			y0=self.y0 if y0 is None else y0,
			force=external_force,
			options=options,
		)
		return {
			"t": response.t,
			"y": response.y,
			"displacement": response.displacement,
			"velocity": response.velocity,
		}

	def run_case(self, amplitude, frequency, **kwargs):
		kwargs = _legacy_case_kwargs(kwargs)
		kwargs.pop("as_dict", None)
		return super().run_case(amplitude, frequency, as_dict=True, **kwargs)


def _legacy_case_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
	kwargs = dict(kwargs)
	if "y0_case" in kwargs:
		kwargs["y0"] = kwargs.pop("y0_case")

	sampling_keys = {"points_per_period", "n_periods_total", "n_periods_keep"}
	solve_keys = {"rtol", "atol"}
	sampling_values = {key: kwargs.pop(key) for key in list(kwargs) if key in sampling_keys and kwargs[key] is not None}
	solve_values = {key: kwargs.pop(key) for key in list(kwargs) if key in solve_keys and kwargs[key] is not None}

	if sampling_values:
		sampling_defaults = PeriodicSampling()
		kwargs["sampling"] = PeriodicSampling(
			points_per_period=int(sampling_values.get("points_per_period", sampling_defaults.points_per_period)),
			n_periods_total=int(sampling_values.get("n_periods_total", sampling_defaults.n_periods_total)),
			n_periods_keep=int(sampling_values.get("n_periods_keep", sampling_defaults.n_periods_keep)),
		)
	if solve_values:
		solve_defaults = SolveOptions()
		kwargs["solve_options"] = SolveOptions(
			rtol=float(solve_values.get("rtol", solve_defaults.rtol)),
			atol=float(solve_values.get("atol", solve_defaults.atol)),
		)
	return kwargs


def _unique_indices(indices: Iterable[int], N: int) -> list[int]:
	unique = sorted(set(int(idx) for idx in indices))
	for idx in unique:
		_validate_mass_index(idx, N, "mass index")
	return unique


def _validate_mass_index(index: int, N: int, name: str) -> None:
	if index < 0 or index >= N:
		raise ValueError(f"{name}={index} is outside the chain with N={N}")

