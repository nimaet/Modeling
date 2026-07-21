"""Interactive Duffing/Lyapunov experiments.

Run this file with VS Code's "Run Current File in Interactive Window", or run
individual cells marked by # %%.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

from duffing_lyapunov import (
	analyze_duffing_once,
	sweep_phase_poincare,
	sweep_lyapunov,
)


# %%
# Baseline parameters
zeta = 0.01
omega_n = 2*np.pi
alpha = 2e5
F = 10.0
Omega = 0.9*omega_n

base_parameters = dict(
	zeta=zeta,
	omega_n=omega_n,
	alpha=alpha,
	F=F,
	Omega=Omega,
	x0=0.0,
	v0=0.0,
	delta0=1e-8,
)


# %%
# Single run: Lyapunov exponent, phase portrait, Poincare map, convergence plot.
result = analyze_duffing_once(
	**base_parameters,
	n_transient_periods=200,
	n_measure_periods=600,
	steps_per_period=200,
	store_trajectory=True,
)

print(f"lambda_max = {result['lambda_max']:.8g}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

ax = axes[0]
ax.plot(result["x"], result["v"], lw=0.55, alpha=0.55, color="tab:blue", label="Phase portrait")
ax.scatter(
	result["x_poincare"],
	result["v_poincare"],
	s=15,
	color="tab:red",
	edgecolor="white",
	linewidth=0.3,
	label="Poincare map",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$\\dot{x}$")
ax.set_title(f"$\\Omega/\\omega_n={result['Omega']/omega_n:.4g}$, $F={result['F']:.4g}$")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

ax = axes[1]
ax.plot(result["period_history"], result["lambda_history"], color="tab:green")
ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("Measured forcing periods")
ax.set_ylabel("$\\lambda_{max}$")
ax.set_title(f"Converged $\\lambda_{{max}}={result['lambda_max']:.4g}$")
ax.grid(True, alpha=0.25)


# %%
# Shared options for faster scalar parameter sweeps.
sweep_options = dict(
	n_transient_periods=150,
	n_measure_periods=350,
	steps_per_period=120,
	rtol=1e-8,
	atol=1e-10,
)


# %%
# Frequency sweep in parallel.
omega_ratios = np.linspace(0.7, 1.3, 31)
Omega_values = omega_ratios*omega_n

frequency_sweep = sweep_lyapunov(
	"Omega",
	Omega_values,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	**sweep_options,
)

lambda_vs_Omega = frequency_sweep["lambda_max"]

fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
ax.plot(omega_ratios, lambda_vs_Omega, "o-", ms=4)
ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("$\\Omega/\\omega_n$")
ax.set_ylabel("$\\lambda_{max}$")
ax.grid(True, alpha=0.25)


# %%
# Amplitude sweep in parallel.
F_values = np.linspace(1.0, 20.0, 31)

amplitude_sweep = sweep_lyapunov(
	"F",
	F_values,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	**sweep_options,
)

lambda_vs_F = amplitude_sweep["lambda_max"]

fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
ax.plot(F_values, lambda_vs_F, "o-", ms=4)
ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("$F$")
ax.set_ylabel("$\\lambda_{max}$")
ax.grid(True, alpha=0.25)


# %%
# Inspect sweep arrays directly in the Interactive Window.
lambda_vs_Omega, lambda_vs_F


# %%
# Shared options for phase portrait / Poincare map grids.
# This reruns the sweep while storing trajectory data, so keep these settings
# lighter than the scalar Lyapunov sweep until you know how much detail you need.
map_options = dict(
	n_transient_periods=500,
	n_measure_periods=1000,
	steps_per_period=200,
	rtol=1e-8,
	atol=1e-10,
)

ncols = 5
phase_color = "green"
poincare_color = "red"
phase_size = 1
poincare_size = 2.0
phase_alpha = 0.35
poincare_alpha = 0.8


# %%
# Frequency-sweep phase portraits and Poincare sections in a grid.
omega_ratios_map = np.linspace(0.7, 1.3, 25)
Omega_values_map = omega_ratios_map*omega_n

frequency_maps = sweep_phase_poincare(
	"Omega",
	Omega_values_map,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	**map_options,
)

results = frequency_maps["results"]
nplots = len(results)
nrows = int(np.ceil(nplots/ncols))
# %% plot Frequency-sweep phase portraits and Poincare sections in a grid.
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.8*nrows))
axes = np.atleast_1d(axes).ravel()

for ax, result_i, omega_ratio in zip(axes, results, omega_ratios_map):
	ax.scatter(result_i["x"], result_i["v"], s=phase_size, color=phase_color, alpha=phase_alpha, linewidths=0)
	ax.scatter(
		result_i["x_poincare"],
		result_i["v_poincare"],
		s=poincare_size,
		color=poincare_color,
		alpha=poincare_alpha,
		linewidths=0,
	)
	ax.set_title(f"{omega_ratio:.4g}", fontsize=10)
	ax.set_xlabel("$x$", fontsize=8)
	ax.set_ylabel("$\\dot{x}$", fontsize=8)
	# ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.35)
	ax.tick_params(labelsize=8)

for ax in axes[nplots:]:
	pass
fig.tight_layout()
	# ax.set_visible(False)

fig.suptitle(
	"Phase portraits (green) and Poincare sections (blue) across excitation frequency",
	fontsize=13,
)


# %%
# Amplitude-sweep phase portraits and Poincare sections in a grid.
F_values_map = np.linspace(1.0, 20.0, 25)

amplitude_maps = sweep_phase_poincare(
	"F",
	F_values_map,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	**map_options,
)

results = amplitude_maps["results"]
nplots = len(results)
nrows = int(np.ceil(nplots/ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.8*nrows)
						#  , constrained_layout=True
						 )
axes = np.atleast_1d(axes).ravel()

for ax, result_i, F_i in zip(axes, results, F_values_map):
	ax.scatter(result_i["x"], result_i["v"], s=phase_size, color=phase_color, alpha=phase_alpha, linewidths=0)
	ax.scatter(
		result_i["x_poincare"],
		result_i["v_poincare"],
		s=poincare_size,
		color=poincare_color,
		alpha=poincare_alpha,
		linewidths=0,
	)
	ax.set_title(f"{F_i:.4g}", fontsize=10)
	ax.set_xlabel("$x$", fontsize=8)
	ax.set_ylabel("$\\dot{x}$", fontsize=8)
	# ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.35)
	ax.tick_params(labelsize=8)

for ax in axes[nplots:]:
	ax.set_visible(False)

fig.suptitle(
	"Phase portraits (green) and Poincare sections (blue) across forcing amplitude",
	fontsize=13,
)
