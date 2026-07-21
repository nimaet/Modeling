"""Reproduce Chaunsali & Theocharis Fig. 4(a-f) interactively.

Run this file cell-by-cell.  The cases use bulk profiles from Fig. 3 as initial
conditions and then evolve the full discrete cubic chain freely.
"""

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.mass_spring_chain import MassSpringChain, SolveOptions


# %%
@dataclass(frozen=True)
class Fig4Config:
	N: int = 299
	gamma: float = 0.08 / 4.0
	Gamma: float = 0.08 / 3.0
	site2_small_amplitude: float = 0.98
	site1_large_amplitude: float = 3.0
	soliton_center_cell: float = 74.5
	omega_m: float = np.sqrt(2.0)
	n_periods: int = 160
	points_per_period: int = 40
	plot_max_spring: int = 180
	plot_min_period: float = 0.0
	plot_max_period: float = 160.0
	plot_stride_time: int = 3
	plot_zmax: float = 3.0
	plot_cloud_size: float = 5
	plot_color_limits: dict[str, tuple[float, float] | None] | None = None
	rtol: float = 1e-8
	atol: float = 1e-10
	run_full_simulation: bool = True
	output_dir: Path = Path("results_fig4_all")


config = Fig4Config()

# Optional per-panel color limits.  Leave as None for independent auto-scaling.
# Example:
config = Fig4Config(
	plot_color_limits={
		"4a_symmetry_preserving": (0.0, 1),
		"4b_symmetry_preserving_soliton": (0.0, 1.4),
		"4c_symmetry_preserving_large": (0.0, 3.0),
		"4d_symmetry_breaking": (0.0, 1),
		"4e_symmetry_breaking_soliton": (0.0, 1.4),
		"4f_symmetry_breaking_large": (0.0, 3.0),
	}
)

# The paper normalizes time by the midgap period tau_m = 2*pi/Omega_m.
tau_m = 2.0 * np.pi / config.omega_m
t_eval = np.linspace(
	0.0,
	config.n_periods * tau_m,
	config.n_periods * config.points_per_period + 1,
)

print(f"tau_m = {tau_m:.6g}")
print(f"time samples = {t_eval.size}")
print(f"gamma = {config.gamma:.6g}, Gamma = {config.Gamma:.6g}")


# %%
# Model builders and analytical Fig. 3 profiles.
def build_fig4_chain(config: Fig4Config, termination: str) -> MassSpringChain:
	"""Build the finite chain for one of the two left-boundary terminations."""
	if termination == "symmetry_preserving":
		start = "low"
	elif termination == "symmetry_breaking":
		start = "high"
	else:
		raise ValueError("termination must be 'symmetry_preserving' or 'symmetry_breaking'")

	return MassSpringChain.alternating(
		N=config.N,
		mass=1.0,
		k=1.0,
		gamma=config.gamma,
		alpha_even=config.Gamma,
		alpha_odd=-config.Gamma,
		c_dashpot=0.0,
		start=start,
	)


def evanescent_site2_profile(config: Fig4Config, n_cells: int) -> np.ndarray:
	"""Fig. 3(d) envelope on the paper's site-1=0 axis."""
	z = np.arange(n_cells, dtype=float)
	plateau_inverse = 3.0 * config.Gamma / (4.0 * config.gamma)
	return (
		plateau_inverse
		+ (1.0 / config.site2_small_amplitude**2 - plateau_inverse) * np.exp(4.0 * config.gamma * z)
	) ** -0.5


def evanescent_site1_profile(config: Fig4Config, n_cells: int) -> np.ndarray:
	"""Fig. 3(f) envelope on the paper's site-2=0 axis."""
	z = np.arange(n_cells, dtype=float)
	plateau_inverse = 3.0 * config.Gamma / (4.0 * config.gamma)
	return (
		plateau_inverse
		+ (1.0 / config.site1_large_amplitude**2 - plateau_inverse) * np.exp(-4.0 * config.gamma * z)
	) ** -0.5


def soliton_profile(config: Fig4Config, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
	"""Fig. 3(b) heteroclinic soliton envelope centered away from boundaries."""
	z = np.arange(n_cells, dtype=float) - config.soliton_center_cell
	amplitude = np.sqrt(2.0 * config.gamma / (3.0 * config.Gamma))
	transition = np.tanh(4.0 * config.gamma * z)
	site1_envelope = amplitude * (1.0 + transition)
	site2_envelope = amplitude * (1.0 - transition)
	return site1_envelope, site2_envelope


def profile_envelopes(config: Fig4Config, profile: str, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
	"""Return the paper's site-1/site-2 envelopes before applying (-1)^n."""
	if profile == "evanescent_trivial":
		return np.zeros(n_cells), evanescent_site2_profile(config, n_cells)
	if profile == "soliton":
		return soliton_profile(config, n_cells)
	if profile == "evanescent_nontrivial":
		return evanescent_site1_profile(config, n_cells), np.zeros(n_cells)
	raise ValueError("unknown profile name")


def sublattice_indices(chain: MassSpringChain, termination: str, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
	"""Return mass indices carrying the paper's site-1 and site-2 envelopes."""
	if termination == "symmetry_preserving":
		# Mass ordering: site1_1, site2_1, site1_2, site2_2, ...
		site1_mass_indices = 2 * np.arange(n_cells)
		site2_mass_indices = site1_mass_indices + 1
	elif termination == "symmetry_breaking":
		# Mass ordering after cutting the unit cell: site2_1, site1_2, site2_2, site1_3, ...
		site2_mass_indices = 2 * np.arange(n_cells)
		site1_mass_indices = site2_mass_indices + 1
	else:
		raise ValueError("termination must be 'symmetry_preserving' or 'symmetry_breaking'")

	return site1_mass_indices, site2_mass_indices


def initial_state_from_profile(
	chain: MassSpringChain,
	config: Fig4Config,
	termination: str,
	profile: str,
) -> np.ndarray:
	"""Map a Fig. 3 bulk profile onto the selected finite boundary."""
	disp0 = np.zeros(chain.N)
	vel0 = np.zeros(chain.N)
	n_cells = (chain.N + 1) // 2
	phase = (-1.0) ** np.arange(n_cells)
	site1_mass_indices, site2_mass_indices = sublattice_indices(chain, termination, n_cells)
	site1_envelope, site2_envelope = profile_envelopes(config, profile, n_cells)

	site1_inside = site1_mass_indices < chain.N
	site2_inside = site2_mass_indices < chain.N
	disp0[site1_mass_indices[site1_inside]] = phase[site1_inside] * site1_envelope[site1_inside]
	disp0[site2_mass_indices[site2_inside]] = phase[site2_inside] * site2_envelope[site2_inside]
	return chain.state(disp0, vel0)


def run_free_response(chain: MassSpringChain, y0: np.ndarray, config: Fig4Config):
	return chain.simulate(
		t_eval=t_eval,
		y0=y0,
		force=None,
		options=SolveOptions(rtol=config.rtol, atol=config.atol),
		include_energy=True,
	)


def strain_history(chain: MassSpringChain, response) -> np.ndarray:
	return chain.spring_extensions_history(response.displacement)


# %%
# Define all six paper panels.  Top row preserves the unit-cell boundary;
# bottom row cuts the unit cell at the left boundary.
fig4_cases = {
	"4a_symmetry_preserving": {
		"termination": "symmetry_preserving",
		"profile": "evanescent_trivial",
		"title": "Fig. 4a candidate: symmetry-preserving boundary",
	},
	"4b_symmetry_preserving_soliton": {
		"termination": "symmetry_preserving",
		"profile": "soliton",
		"title": "Fig. 4b candidate: symmetry-preserving boundary, soliton IC",
	},
	"4c_symmetry_preserving_large": {
		"termination": "symmetry_preserving",
		"profile": "evanescent_nontrivial",
		"title": "Fig. 4c candidate: symmetry-preserving boundary, large-amplitude IC",
	},
	"4d_symmetry_breaking": {
		"termination": "symmetry_breaking",
		"profile": "evanescent_trivial",
		"title": "Fig. 4d candidate: symmetry-breaking boundary",
	},
	"4e_symmetry_breaking_soliton": {
		"termination": "symmetry_breaking",
		"profile": "soliton",
		"title": "Fig. 4e candidate: symmetry-breaking boundary, soliton IC",
	},
	"4f_symmetry_breaking_large": {
		"termination": "symmetry_breaking",
		"profile": "evanescent_nontrivial",
		"title": "Fig. 4f candidate: symmetry-breaking boundary, large-amplitude IC",
	},
}

# Assemble each finite chain and map the chosen analytical profile to masses.
for case in fig4_cases.values():
	case["chain"] = build_fig4_chain(config, case["termination"])
	case["y0"] = initial_state_from_profile(
		case["chain"],
		config,
		case["termination"],
		case["profile"],
	)
	case["initial_strain"] = case["chain"].spring_extensions(case["y0"][: config.N])

print("First six spring linear/cubic coefficients:")
for name, case in fig4_cases.items():
	chain = case["chain"]
	print(name)
	print("  k     =", np.array2string(chain.k_springs[:6], precision=4))
	print("  alpha =", np.array2string(chain.alpha_springs[:6], precision=4))
	print("  |strain(0)| =", np.array2string(np.abs(case["initial_strain"][:6]), precision=4))


# %%
# Quick IC check before running the ODE solve.
fig, axes = plt.subplots(len(fig4_cases), 1, figsize=(9, 2.3 * len(fig4_cases)), sharex=True)
for ax, (name, case) in zip(axes, fig4_cases.items()):
	ax.plot(np.arange(1, config.N + 2), np.abs(case["initial_strain"]), lw=1.5)
	ax.set_ylabel(r"$|\Delta \xi_j(0)|$")
	ax.set_title(case["title"])
	ax.grid(True, alpha=0.25)
axes[-1].set_xlabel("spring index")
fig.tight_layout()
plt.show()


# %%
# Fig. 3-style envelope profiles: site 1 and site 2 are shown separately.
def plot_fig3_style_profiles(config: Fig4Config, *, max_cell: int = 150):
	n_cells = (config.N + 1) // 2
	cell_index = np.arange(n_cells)
	cell_mask = cell_index < max_cell
	profile_specs = [
		("evanescent_trivial", "Fig. 3d IC: small-amplitude evanescent"),
		("soliton", "Fig. 3b IC: kink soliton"),
		("evanescent_nontrivial", "Fig. 3f IC: large-amplitude evanescent"),
	]

	fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
	for ax, (profile, title) in zip(axes, profile_specs):
		site1, site2 = profile_envelopes(config, profile, n_cells)
		ax.plot(cell_index[cell_mask], site1[cell_mask], color="tab:red", lw=2.0, label="site 1")
		ax.plot(cell_index[cell_mask], site2[cell_mask], color="tab:blue", lw=2.0, label="site 2")
		ax.axhline(0.0, color="0.25", lw=0.8)
		ax.set_title(title)
		ax.set_xlabel("unit-cell index")
		ax.grid(True, alpha=0.25)
	axes[0].set_ylabel("envelope displacement")
	axes[0].legend(loc="best")
	fig.tight_layout()
	plt.show()
	return fig, axes


fig_profiles, axes_profiles = plot_fig3_style_profiles(config)


# %%
# Main compute cell: free response from the selected initial conditions.
if config.run_full_simulation:
	for name, case in fig4_cases.items():
		print(f"Running {name}...")
		case["response"] = run_free_response(case["chain"], case["y0"], config)
		case["strain"] = strain_history(case["chain"], case["response"])
		if case["response"].energy is not None:
			energy = case["response"].energy
			rel_drift = (energy[-1] - energy[0]) / energy[0]
			print(f"  relative energy drift = {rel_drift:.3e}")
else:
	print("Set config.run_full_simulation=True, then rerun this cell to compute Fig. 4(a-f).")


# %%
# Plot paper-style strain maps: x is spring index, y is tau/tau_m.
def color_limits_for_case(case_name: str, values: np.ndarray, config: Fig4Config) -> tuple[float, float]:
	"""Return per-panel color limits, or auto-scale this panel if unset."""
	limits = None if config.plot_color_limits is None else config.plot_color_limits.get(case_name)
	if limits is not None:
		vmin, vmax = limits
		return float(vmin), max(float(vmax), float(vmin) + 1e-12)

	vmax = float(np.max(values)) if values.size else config.plot_zmax
	return 0.0, max(vmax, 1e-12)


def plot_strain_map(
	case_name,
	case,
	*,
	max_spring: int | None = None,
	min_period: float | None = None,
	max_period: float | None = None,
	stride_time: int | None = None,
):
	max_spring = config.plot_max_spring if max_spring is None else int(max_spring)
	min_period = config.plot_min_period if min_period is None else float(min_period)
	max_period = config.plot_max_period if max_period is None else float(max_period)
	stride_time = config.plot_stride_time if stride_time is None else int(stride_time)
	response = case["response"]
	strain = np.abs(case["strain"])
	s_plot = np.arange(1, strain.shape[1] + 1)
	t_all = response.t / tau_m
	t_mask = (t_all >= min_period) & (t_all <= max_period)
	s_mask = s_plot <= max_spring
	t_plot = t_all[t_mask][::stride_time]
	strain_plot = strain[t_mask, :][::stride_time, :][:, s_mask]
	color_min, color_max = color_limits_for_case(case_name, strain_plot, config)

	fig, ax = plt.subplots(figsize=(8.5, 4.5))
	mesh = ax.pcolormesh(
		s_plot[s_mask],
		t_plot,
		strain_plot,
		shading="auto",
		cmap="YlGnBu",
		vmin=color_min,
		vmax=color_max,
	)
	ax.set_xlim(1, max_spring)
	ax.set_ylim(min_period, max_period)
	ax.set_xlabel("spring index")
	ax.set_ylabel(r"$\tau / \tau_m$")
	ax.set_title(case["title"])
	fig.colorbar(mesh, ax=ax, label=r"$|\Delta \xi|$")
	fig.tight_layout()
	plt.show()
	return fig, ax


if all("strain" in case for case in fig4_cases.values()):
	for case_name, case in fig4_cases.items():
		plot_strain_map(case_name, case)
else:
	print("Run the simulation cell before plotting strain maps.")


# %%
# Paper-like 3D version of Fig. 4: first column is (a,b,c), second is (d,e,f).
def plot_fig4_surface_grid(
	fig4_cases: dict,
	config: Fig4Config,
	*,
	max_spring: int | None = None,
	min_period: float | None = None,
	max_period: float | None = None,
	stride_time: int | None = None,
	zmax: float | None = None,
	cmap: str = "YlGnBu",
):
	max_spring = config.plot_max_spring if max_spring is None else int(max_spring)
	min_period = config.plot_min_period if min_period is None else float(min_period)
	max_period = config.plot_max_period if max_period is None else float(max_period)
	stride_time = config.plot_stride_time if stride_time is None else int(stride_time)
	zmax = config.plot_zmax if zmax is None else float(zmax)

	case_order = [
		("4a_symmetry_preserving", "(a)", 1),
		("4b_symmetry_preserving_soliton", "(b)", 3),
		("4c_symmetry_preserving_large", "(c)", 5),
		("4d_symmetry_breaking", "(d)", 2),
		("4e_symmetry_breaking_soliton", "(e)", 4),
		("4f_symmetry_breaking_large", "(f)", 6),
	]

	fig = plt.figure(figsize=(9.5, 12.0))
	axes = []
	colormap = plt.get_cmap(cmap)

	for case_name, panel_label, subplot_index in case_order:
		case = fig4_cases[case_name]
		response = case["response"]
		strain = np.abs(case["strain"])

		t_all = response.t / tau_m
		s_all = np.arange(1, strain.shape[1] + 1)
		t_mask = (t_all >= min_period) & (t_all <= max_period)
		s_mask = s_all <= max_spring

		t_plot = t_all[t_mask][::stride_time]
		s_plot = s_all[s_mask]
		z_plot = strain[t_mask, :][::stride_time, :][:, s_mask]
		s_mesh, t_mesh = np.meshgrid(s_plot, t_plot)
		s_cloud = s_mesh.ravel()
		t_cloud = t_mesh.ravel()
		z_cloud = z_plot.ravel()
		color_min, color_max = color_limits_for_case(case_name, z_plot, config)
		norm = plt.Normalize(vmin=color_min, vmax=color_max)

		ax = fig.add_subplot(3, 2, subplot_index, projection="3d")
		axes.append(ax)
		ax.scatter(
			s_cloud,
			t_cloud,
			z_cloud,
			c=z_cloud,
			cmap=colormap,
			norm=norm,
			s=config.plot_cloud_size,
			marker=".",
			linewidths=0.0,
			alpha=1.0,
		)

		ax.set_title(panel_label, pad=2)
		ax.set_xlim(1, max_spring)
		ax.set_ylim(min_period, max_period)
		ax.set_zlim(0, zmax)
		ax.set_xlabel("spring index", labelpad=4)
		ax.set_ylabel(r"$\tau / \tau_m$", labelpad=4)
		ax.set_zlabel(r"$|\Delta \xi|$", labelpad=4)
		ax.view_init(elev=25, azim=-125)
		ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
		ax.tick_params(axis="z", which="major", labelsize=8, pad=0)

		sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
		sm.set_array([])
		fig.colorbar(sm, ax=ax, shrink=0.58, pad=0.02, label=r"$|\Delta \xi|$")

	fig.suptitle("Chaunsali & Theocharis Fig. 4 reproduction candidates", y=0.98)
	fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.95, wspace=0.02, hspace=0.02)
	plt.show()
	return fig, axes


if all("strain" in case for case in fig4_cases.values()):
	fig_surface, axes_surface = plot_fig4_surface_grid(fig4_cases, config, max_spring=300, max_period=60)
	config.output_dir.mkdir(exist_ok=True)
	fig_surface.savefig(config.output_dir / "fig4_surface_grid.png", dpi=250)
	print(f"Saved paper-style surface grid to {config.output_dir / 'fig4_surface_grid.png'}")
else:
	print("Run the simulation cell before plotting the 3D surface grid.")


# %%
# Save raw arrays so we can revisit plotting without rerunning the ODE solve.
if all("strain" in case for case in fig4_cases.values()):
	config.output_dir.mkdir(exist_ok=True)

	for name, case in fig4_cases.items():
		response = case["response"]
		np.savez_compressed(
			config.output_dir / f"{name}.npz",
			t=response.t,
			displacement=response.displacement,
			velocity=response.velocity,
			strain=case["strain"],
			energy=response.energy,
			k_springs=case["chain"].k_springs,
			alpha_springs=case["chain"].alpha_springs,
			initial_state=case["y0"],
		)

	print(f"Saved arrays in {config.output_dir.resolve()}")
else:
	print("Run the simulation cell before saving arrays.")

# %%
