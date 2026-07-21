"""Batch version of nonlinear_chain.ipynb using reusable src modules.

The original notebook mixes study setup, system definitions, simulations, and
plots in cells. This script keeps the same studies but imports the chain model
from ``src.mass_spring_chain`` instead of redefining spring forces and the RHS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import chirp

from src.mass_spring_chain import MassSpringChain, SolveOptions, sinusoidal_force


@dataclass(frozen=True)
class StudyConfig:
	N: int = 512
	m: float = 1.0
	k: float = 1.0
	c_dashpot: float = 1e-5
	alpha_hard: float = 0.08 / 3.0
	alpha_soft: float = -0.08 / 3.0
	gamma: float = 0.08 / 4.0
	forced_mass: int = 1
	force_amplitude: float = 0.0
	force_frequency: float = 1.0
	force_phase: float = 0.0
	n_periods_total: int = 500
	n_periods_keep: int = 100
	points_per_period: int = 100
	random_seed: int = 42


class LazyNPZ:
	"""Load arrays from a compressed npz archive on demand."""

	def __init__(self, path, **metadata):
		self.path = str(path)
		self._metadata = dict(metadata)
		self._cache = {}
		with np.load(self.path, allow_pickle=True) as data:
			self._keys = tuple(data.files)

	@property
	def keys(self):
		return self._keys

	def __contains__(self, key):
		return key in self._metadata or key in self._keys

	def __getitem__(self, key):
		if key in self._metadata:
			return self._metadata[key]
		if key not in self._keys:
			raise KeyError(key)
		if key not in self._cache:
			with np.load(self.path, allow_pickle=True) as data:
				self._cache[key] = data[key]
		return self._cache[key]

	def get(self, key, default=None):
		return self[key] if key in self else default


def build_chain(config: StudyConfig) -> MassSpringChain:
	return MassSpringChain.alternating(
		N=config.N,
		mass=config.m,
		k=config.k,
		gamma=config.gamma,
		alpha_even=config.alpha_hard,
		alpha_odd=config.alpha_soft,
		c_dashpot=config.c_dashpot,
		start="low",
	)


def localized_initial_state(chain: MassSpringChain, velocity_mass: int = 1, velocity: float = 1e-4) -> np.ndarray:
	u0 = np.zeros(chain.N)
	v0 = np.zeros(chain.N)
	v0[velocity_mass] = velocity
	return chain.state(u0, v0)


def periodic_time_grid(config: StudyConfig) -> tuple[float, float, np.ndarray]:
	dt = 1.0 / (config.force_frequency / (2.0 * np.pi)) / config.points_per_period
	t_final = config.n_periods_total * config.points_per_period * dt
	return dt, t_final, np.arange(0.0, t_final + dt, dt)


def eigen_dispersion(chain: MassSpringChain, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	omega_n, modes = chain.modes()
	print(f"Assembled M and K for N={chain.N} moving masses with fixed end supports.")
	print(f"First five natural frequencies (rad/s): {omega_n[:5]}")
	print("Modes are stored column-wise; eigenvalues satisfy K phi = omega^2 M phi.")
	print(f"Maximum natural frequency (rad/s): {omega_n.max():.6g}")

	mode_fft = np.fft.fftshift(np.fft.fft(modes, axis=0), axes=0)
	mode_power = np.abs(mode_fft) ** 2
	mode_power /= np.sum(mode_power, axis=0, keepdims=True) + 1e-15
	k_vals_eig = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(chain.N, d=1.0))

	fig, ax = plt.subplots(figsize=(9, 5))
	positive_power = mode_power[mode_power > 0]
	norm = LogNorm(vmin=max(positive_power.min(), 1e-12), vmax=mode_power.max())
	pcm = ax.pcolormesh(k_vals_eig, omega_n, mode_power.T, shading="auto", cmap="magma", norm=norm)
	fig.colorbar(pcm, ax=ax, label="normalized FFT power of eigenvector")

	dominant_k_idx = np.argmax(mode_power, axis=0)
	ax.scatter(
		k_vals_eig[dominant_k_idx],
		omega_n,
		s=18,
		facecolors="none",
		edgecolors="cyan",
		linewidths=0.8,
		label="dominant bin per mode",
	)
	ax.set_xlabel("discrete angular wavenumber k")
	ax.set_ylabel("natural frequency omega")
	ax.set_title("Dispersion estimate from FFT of eigenvectors")
	ax.set_xlim(k_vals_eig[0], k_vals_eig[-1])
	ax.legend(loc="upper left")
	fig.tight_layout()
	fig.savefig(output_dir / "eigenvector_dispersion.png", dpi=200)
	plt.close(fig)
	return omega_n, modes, mode_power, k_vals_eig


def filter_significant_omega_from_power(power, omega_vals, relative_threshold=1e-4, min_keep=5):
	power = np.asarray(power)
	omega_vals = np.asarray(omega_vals)

	if power.ndim == 1:
		freq_power = power
		power_matrix = None
	elif power.ndim == 2:
		freq_power = np.sum(power, axis=1)
		power_matrix = power
	else:
		raise ValueError("power must be either 1D or 2D with shape (n_omega, n_k)")

	if omega_vals.shape[0] != freq_power.shape[0]:
		raise ValueError("omega_vals must match the omega axis of power")

	if freq_power.size == 0:
		freq_mask = np.array([], dtype=bool)
	else:
		freq_mask = freq_power > np.max(freq_power) * relative_threshold
		if freq_mask.sum() < min_keep:
			strongest = np.argsort(freq_power)[-min(min_keep, freq_power.size):]
			freq_mask = np.zeros_like(freq_power, dtype=bool)
			freq_mask[strongest] = True

	power_masked = freq_power[freq_mask] if power_matrix is None else power_matrix[freq_mask, :]
	return power_masked, omega_vals[freq_mask], freq_mask, freq_power


def time_domain_dispersion(
	chain: MassSpringChain,
	config: StudyConfig,
	output_dir: Path,
	mode_power: np.ndarray | None = None,
	omega_n: np.ndarray | None = None,
	k_vals_eig: np.ndarray | None = None,
	fft_periods: int = 1200,
	fft_points_per_period: int = 20,
	velocity_level: float = 0.1,
):
	omega_ref = np.sqrt(2.0 * config.k / config.m)
	fft_dt = 2.0 * np.pi / omega_ref / fft_points_per_period
	fft_t_final = fft_periods * 2.0 * np.pi / omega_ref
	fft_t_eval = np.arange(0.0, fft_t_final + fft_dt, fft_dt)

	rng = np.random.default_rng(config.random_seed)
	u0_td = np.zeros(chain.N)
	v0_td = velocity_level * rng.standard_normal(chain.N)
	y0_td = chain.state(u0_td, v0_td)

	td_run = chain.simulate(
		t_eval=fft_t_eval,
		y0=y0_td,
		force=lambda _t: np.zeros(chain.N),
		options=SolveOptions(rtol=1e-8, atol=1e-9),
	)

	u_td = td_run.displacement - np.mean(td_run.displacement, axis=0, keepdims=True)
	u_omega = np.fft.rfft(u_td, axis=0)
	u_k_omega = np.fft.fftshift(np.fft.fft(u_omega, axis=1), axes=1)
	power = np.abs(u_k_omega) ** 2
	omega_vals = 2.0 * np.pi * np.fft.rfftfreq(len(td_run.t), d=fft_dt)
	k_vals = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(chain.N, d=1.0))

	power_masked, omega_vals_masked, _freq_mask, _freq_power = filter_significant_omega_from_power(
		power,
		omega_vals,
		relative_threshold=1e-4,
		min_keep=5,
	)
	top_k_idx = np.argsort(power_masked, axis=1)[:, -3:]

	fig, ax = plt.subplots(figsize=(9, 5))
	for rank in range(top_k_idx.shape[1]):
		ax.scatter(
			k_vals[top_k_idx[:, rank]],
			omega_vals_masked,
			s=10,
			alpha=0.25,
			c="red",
			label="top wavenumber bins" if rank == 0 else None,
		)
	ax.set_xlabel("discrete angular wavenumber k")
	ax.set_ylabel("angular frequency omega")
	ax.set_title("Dispersion map from time-domain simulation, random initial velocity")
	ax.set_xlim(k_vals[0], k_vals[-1])
	ax.set_ylim(0, 2)
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(output_dir / "time_domain_dispersion.png", dpi=200)
	plt.close(fig)

	if mode_power is not None and omega_n is not None and k_vals_eig is not None:
		fig, ax = plt.subplots(figsize=(9, 5))
		positive_power = mode_power[mode_power > 0]
		norm = LogNorm(vmin=max(positive_power.min(), 1e-12), vmax=mode_power.max())
		pcm = ax.pcolormesh(k_vals_eig, omega_n, mode_power.T, shading="auto", cmap="magma", norm=norm)
		fig.colorbar(pcm, ax=ax, label="normalized FFT power of eigenvector")
		for rank in range(top_k_idx.shape[1]):
			ax.scatter(
				k_vals[top_k_idx[:, rank]],
				omega_vals_masked,
				s=14,
				alpha=0.35,
				c="cyan",
				edgecolors="none",
				label="time-domain ridge" if rank == 0 else None,
			)
		ax.set_xlabel("discrete angular wavenumber k")
		ax.set_ylabel("frequency omega")
		ax.set_title("Time-domain dispersion overlaid on eigenanalysis dispersion")
		ax.set_xlim(k_vals[0], k_vals[-1])
		ax.set_ylim(0, np.max(omega_n))
		ax.legend(loc="upper left")
		fig.tight_layout()
		fig.savefig(output_dir / "time_domain_on_eigen_dispersion.png", dpi=200)
		plt.close(fig)

	return {
		"omega_vals_masked": omega_vals_masked,
		"k_vals": k_vals,
		"top_k_idx": top_k_idx,
		"power_masked": power_masked,
	}


def simulate_velocity_case(chain: MassSpringChain, config: StudyConfig, velocity_level, fft_t_eval, fft_dt):
	rng = np.random.default_rng(config.random_seed)
	y0_td = chain.state(np.zeros(chain.N), velocity_level * rng.standard_normal(chain.N))
	td_run = chain.simulate(
		t_eval=fft_t_eval,
		y0=y0_td,
		force=lambda _t: np.zeros(chain.N),
		options=SolveOptions(rtol=1e-8, atol=1e-9),
	)

	u_td = td_run.displacement - np.mean(td_run.displacement, axis=0, keepdims=True)
	u_omega = np.fft.rfft(u_td, axis=0)
	u_k_omega = np.fft.fftshift(np.fft.fft(u_omega, axis=1), axes=1)
	power = np.abs(u_k_omega) ** 2
	omega_vals = 2.0 * np.pi * np.fft.rfftfreq(len(td_run.t), d=fft_dt)
	k_vals = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(chain.N, d=1.0))
	power_masked, omega_vals_masked, _freq_mask, _freq_power = filter_significant_omega_from_power(
		power,
		omega_vals,
		relative_threshold=1e-4,
		min_keep=5,
	)
	return {
		"velocity_level": float(velocity_level),
		"omega_vals_masked": omega_vals_masked,
		"k_vals": k_vals,
		"top_k_idx": np.argsort(power_masked, axis=1)[:, -3:],
	}


def velocity_level_sweep(
	chain: MassSpringChain,
	config: StudyConfig,
	output_dir: Path,
	velocity_levels=np.array([0.05, 0.1, 0.25, 0.5, 1.0]),
	fft_periods: int = 1200,
	fft_points_per_period: int = 20,
	n_jobs: int | None = None,
):
	omega_ref = np.sqrt(2.0 * config.k / config.m)
	fft_dt = 2.0 * np.pi / omega_ref / fft_points_per_period
	fft_t_final = fft_periods * 2.0 * np.pi / omega_ref
	fft_t_eval = np.arange(0.0, fft_t_final + fft_dt, fft_dt)
	n_workers = min(len(velocity_levels), 30) if n_jobs is None else int(n_jobs)
	ridge_cases = Parallel(n_jobs=n_workers, verbose=12)(
		delayed(simulate_velocity_case)(chain, config, velocity_level, fft_t_eval, fft_dt)
		for velocity_level in velocity_levels
	)

	cmap = plt.cm.inferno
	norm = Normalize(vmin=float(np.min(velocity_levels)), vmax=float(np.max(velocity_levels)) + 1.0)
	fig, ax = plt.subplots(figsize=(10, 5))
	for case in ridge_cases:
		color = cmap(norm(case["velocity_level"]))
		for rank in range(case["top_k_idx"].shape[1]):
			ax.scatter(
				case["k_vals"][case["top_k_idx"][:, rank]],
				case["omega_vals_masked"],
				s=10,
				alpha=0.5,
				color=color,
				edgecolors=color,
			)

	sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	fig.colorbar(sm, ax=ax, label="initial velocity level")
	ax.set_xlabel("discrete angular wavenumber k")
	ax.set_ylabel("angular frequency omega")
	ax.set_title("Dispersion ridges from increasing initial velocity levels")
	ax.set_xlim(ridge_cases[0]["k_vals"][0], ridge_cases[0]["k_vals"][-1])
	ax.set_ylim(1.25, 1.6)
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_dir / "velocity_level_dispersion_ridges.png", dpi=200)
	plt.close(fig)
	return ridge_cases


def chirp_frf_sweep(
	chain: MassSpringChain,
	config: StudyConfig,
	output_dir: Path,
	amp_sweep_values=np.linspace(0.01, 3.0, 16),
	f0: float = 0.01,
	f1: float = 3.0,
	points_per_period: int = 200,
	n_periods_total: int = 500,
	n_jobs: int | None = None,
):
	results_dir = output_dir / "results_chirp_amp"
	results_dir.mkdir(exist_ok=True)
	dt = 1.0 / (points_per_period * f1)
	t_end = n_periods_total / f1 * 2.0
	t_eval = np.arange(0.0, t_end + dt, dt)
	y0 = localized_initial_state(chain)

	def run_case(amplitude):
		def scalar_chirp(t):
			return amplitude * chirp(t, f0=f0, f1=f1, t1=t_end, method="linear")

		def external_force_chirp(t):
			force = np.zeros(chain.N)
			force[config.forced_mass] = scalar_chirp(t)
			return force

		response = chain.simulate(
			t_eval=t_eval,
			y0=y0,
			force=external_force_chirp,
			options=SolveOptions(rtol=1e-9, atol=1e-10),
		)
		force_ts = np.asarray([scalar_chirp(tt) for tt in response.t])
		freqs = np.fft.rfftfreq(response.displacement.shape[0], d=dt)
		u_fft = np.fft.rfft(response.displacement.T, axis=1)
		v_fft = np.fft.rfft(response.velocity.T, axis=1)
		force_fft = np.fft.rfft(force_ts)
		fname = results_dir / f"chirp_frf_A{int(round(1000 * amplitude))}_{uuid.uuid4().hex}.npz"
		np.savez_compressed(
			fname,
			t=response.t,
			force=force_ts,
			u_fft=u_fft,
			v_fft=v_fft,
			force_fft=force_fft,
			freqs=freqs,
			amplitude=amplitude,
			f0=f0,
			f1=f1,
		)
		return {"amplitude": float(amplitude), "file": str(fname)}

	n_workers = min(len(amp_sweep_values), 30) if n_jobs is None else int(n_jobs)
	results_meta = Parallel(n_jobs=n_workers, verbose=12)(delayed(run_case)(a) for a in amp_sweep_values)
	results_chirp = [LazyNPZ(m["file"], amplitude=float(m["amplitude"])) for m in results_meta]

	fig, ax = plt.subplots(figsize=(8, 4))
	amplitudes = np.array([r["amplitude"] for r in results_chirp], dtype=float)
	norm = plt.Normalize(amplitudes.min(), amplitudes.max())
	cmap = plt.cm.viridis
	for result in sorted(results_chirp, key=lambda item: item["amplitude"]):
		force_mag = np.maximum(np.abs(result["force_fft"]), 1e-15)
		frf = np.mean(np.abs(result["u_fft"][20:, :]), axis=0) / force_mag
		ax.semilogy(result["freqs"] * 2.0 * np.pi, frf, color=cmap(norm(result["amplitude"])), label=f"A={result['amplitude']:.3f}")
	ax.set_xlabel("angular frequency omega")
	ax.set_ylabel("FRF magnitude |X/F|")
	ax.set_title(f"Chirp-based FRF sweep at mass {config.forced_mass}")
	ax.grid(True)
	ax.legend(ncol=2, fontsize=8)
	ax.set_xlim(0, 3)
	ax.set_ylim(1e-2, 10)
	fig.tight_layout()
	fig.savefig(output_dir / "chirp_frf_sweep.png", dpi=200)
	plt.close(fig)

	print(f"Prepared {len(results_chirp)} chirp FRF result handle(s) using {n_workers} worker(s).")
	return results_chirp


def quick_free_response(chain: MassSpringChain, config: StudyConfig, output_dir: Path):
	_dt, _t_final, t_eval = periodic_time_grid(config)
	y0 = localized_initial_state(chain)
	force = sinusoidal_force(
		chain.N,
		config.forced_mass,
		config.force_amplitude,
		config.force_frequency,
		config.force_phase,
	)
	response = chain.simulate(t_eval=t_eval, y0=y0, force=force, options=SolveOptions(), include_energy=True)
	history_mass_indices = sorted(set([0, chain.N // 4, config.forced_mass, 3 * chain.N // 4, chain.N - 1]))

	fig, ax = plt.subplots(figsize=(9, 4.5))
	im = ax.imshow(
		response.displacement.T,
		aspect="auto",
		origin="lower",
		extent=[response.t[0], response.t[-1], 0, chain.N - 1],
		cmap="RdBu_r",
	)
	ax.set_xlabel("time")
	ax.set_ylabel("mass index")
	ax.set_title("Displacement field u_n(t)")
	fig.colorbar(im, ax=ax, label="displacement")
	fig.tight_layout()
	fig.savefig(output_dir / "free_response_field.png", dpi=200)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(8, 3.5))
	for mass_idx in history_mass_indices:
		ax.plot(response.t, response.displacement[:, mass_idx], label=f"mass {mass_idx}")
	ax.set_xlabel("time")
	ax.set_ylabel("displacement")
	ax.set_title("Mass displacement time histories")
	ax.legend(ncol=3)
	fig.tight_layout()
	fig.savefig(output_dir / "free_response_histories.png", dpi=200)
	plt.close(fig)

	if response.energy is not None and response.energy[0] != 0.0:
		rel_change = (response.energy[-1] - response.energy[0]) / response.energy[0]
		print(f"Relative mechanical energy change: {rel_change:.3e}")


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--sections",
		nargs="+",
		default=["eigen", "time", "velocity", "chirp"],
		choices=["eigen", "time", "velocity", "chirp", "free", "all"],
		help="Study sections to run. Use a smaller list while iterating; all default sections are compute-heavy.",
	)
	parser.add_argument("--output-dir", default="results_nonlinear_chain_from_src")
	parser.add_argument("--n-jobs", type=int, default=None)
	args, unknown = parser.parse_known_args()
	if unknown:
		print(f"Ignoring external arguments from the interactive runner: {unknown}")
	return args


def main():
	plt.rcParams.update({"figure.figsize": (8, 4), "axes.grid": True})
	args = parse_args()
	sections = set(args.sections)
	if "all" in sections:
		sections = {"eigen", "time", "velocity", "chirp", "free"}

	output_dir = Path(args.output_dir)
	output_dir.mkdir(exist_ok=True)

	config = StudyConfig()
	chain = build_chain(config)
	omega_n = modes = mode_power = k_vals_eig = None

	if "eigen" in sections or "time" in sections:
		omega_n, modes, mode_power, k_vals_eig = eigen_dispersion(chain, output_dir)

	if "time" in sections:
		time_domain_dispersion(chain, config, output_dir, mode_power, omega_n, k_vals_eig)

	if "velocity" in sections:
		velocity_level_sweep(chain, config, output_dir, n_jobs=args.n_jobs)

	if "chirp" in sections:
		chirp_frf_sweep(chain, config, output_dir, n_jobs=args.n_jobs)

	if "free" in sections:
		quick_free_response(chain, config, output_dir)

	print(f"Saved outputs in {output_dir.resolve()}")


if __name__ == "__main__":
	main()
