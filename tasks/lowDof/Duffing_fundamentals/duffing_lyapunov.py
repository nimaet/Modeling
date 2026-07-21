import numpy as np
from scipy.integrate import solve_ivp

try:
	from joblib import Parallel, delayed
except ImportError:
	Parallel = None
	delayed = None


def duffing_augmented(t, Y, zeta, omega_n, alpha, F, Omega):
	x, v = Y[0], Y[1]
	dx, dv = Y[2], Y[3]

	x_dot = v
	v_dot = -2*zeta*omega_n*v - omega_n**2*x - alpha*x**3 + F*np.cos(Omega*t)

	J11 = 0.0
	J12 = 1.0
	J21 = -omega_n**2 - 3*alpha*x**2
	J22 = -2*zeta*omega_n

	dx_dot = J11*dx + J12*dv
	dv_dot = J21*dx + J22*dv

	return np.array([x_dot, v_dot, dx_dot, dv_dot])


def _renormalize_perturbation(y, delta0):
	perturbation_norm = np.linalg.norm(y[2:4])
	if perturbation_norm == 0.0:
		raise RuntimeError("Perturbation vector collapsed to zero; increase delta0.")
	y[2:4] = delta0*y[2:4]/perturbation_norm
	return perturbation_norm


def analyze_duffing_once(
	zeta,
	omega_n,
	alpha,
	F,
	Omega,
	x0=0.0,
	v0=0.0,
	delta0=1e-8,
	n_transient_periods=200,
	n_measure_periods=600,
	steps_per_period=200,
	store_trajectory=True,
	rtol=1e-9,
	atol=1e-11,
):
	"""Compute Lyapunov exponent, phase trajectory, and Poincare points in one pass."""
	T = 2*np.pi/Omega
	y = np.array([x0, v0, delta0, 0.0], dtype=float)
	t0 = 0.0

	for _ in range(n_transient_periods):
		sol = solve_ivp(
			duffing_augmented,
			(t0, t0 + T),
			y,
			args=(zeta, omega_n, alpha, F, Omega),
			rtol=rtol,
			atol=atol,
			max_step=T/steps_per_period,
		)
		y = sol.y[:, -1]
		_renormalize_perturbation(y, delta0)
		t0 += T

	t_hist = []
	x_hist = []
	v_hist = []
	x_poincare = []
	v_poincare = []
	lambda_history = []
	period_history = []
	log_growth_sum = 0.0
	total_time = 0.0

	for period_idx in range(1, n_measure_periods + 1):
		t_eval = None
		if store_trajectory:
			t_eval = np.linspace(t0, t0 + T, steps_per_period + 1)

		sol = solve_ivp(
			duffing_augmented,
			(t0, t0 + T),
			y,
			args=(zeta, omega_n, alpha, F, Omega),
			t_eval=t_eval,
			rtol=rtol,
			atol=atol,
			max_step=T/steps_per_period,
		)

		if not sol.success:
			raise RuntimeError(sol.message)

		if store_trajectory:
			start = 0 if period_idx == 1 else 1
			t_hist.append(sol.t[start:])
			x_hist.append(sol.y[0, start:])
			v_hist.append(sol.y[1, start:])
			x_poincare.append(sol.y[0, -1])
			v_poincare.append(sol.y[1, -1])

		y = sol.y[:, -1]
		perturbation_norm = _renormalize_perturbation(y, delta0)
		log_growth_sum += np.log(perturbation_norm/delta0)
		total_time += T
		lambda_history.append(log_growth_sum/total_time)
		period_history.append(period_idx)
		t0 += T

	result = {
		"lambda_max": log_growth_sum/total_time,
		"lambda_history": np.array(lambda_history),
		"period_history": np.array(period_history),
		"T": T,
		"Omega": Omega,
		"F": F,
	}

	if store_trajectory:
		result.update({
			"t": np.concatenate(t_hist),
			"x": np.concatenate(x_hist),
			"v": np.concatenate(v_hist),
			"x_poincare": np.array(x_poincare),
			"v_poincare": np.array(v_poincare),
		})

	return result


def _sweep_one_value(parameter_name, value, base_parameters, analysis_options):
	parameters = base_parameters.copy()
	options = analysis_options.copy()
	options.pop("store_trajectory", None)
	parameters[parameter_name] = value
	result = analyze_duffing_once(
		**parameters,
		store_trajectory=False,
		**options,
	)
	return result["lambda_max"]


def sweep_lyapunov(
	parameter_name,
	parameter_values,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	backend="loky",
	**analysis_options,
):
	"""Sweep one parameter and compute lambda_max values, optionally in parallel."""
	parameter_values = np.asarray(parameter_values, dtype=float)

	if n_jobs == 1:
		lambdas = [
			_sweep_one_value(parameter_name, value, base_parameters, analysis_options)
			for value in parameter_values
		]
	elif Parallel is None:
		raise ImportError("joblib is required for parallel sweeps. Install it or use n_jobs=1.")
	else:
		lambdas = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
			delayed(_sweep_one_value)(parameter_name, value, base_parameters, analysis_options)
			for value in parameter_values
		)

	return {
		"parameter_name": parameter_name,
		"parameter_values": parameter_values,
		"lambda_max": np.array(lambdas),
		"base_parameters": base_parameters.copy(),
		"analysis_options": analysis_options.copy(),
	}


def _sweep_one_map(parameter_name, value, base_parameters, analysis_options):
	parameters = base_parameters.copy()
	options = analysis_options.copy()
	parameters[parameter_name] = value
	result = analyze_duffing_once(
		**parameters,
		store_trajectory=True,
		**options,
	)
	return result


def sweep_phase_poincare(
	parameter_name,
	parameter_values,
	base_parameters,
	n_jobs=-1,
	verbose=10,
	backend="loky",
	**analysis_options,
):
	"""Sweep one parameter and keep phase/Poincare data for grid plots."""
	parameter_values = np.asarray(parameter_values, dtype=float)
	options = analysis_options.copy()
	options.pop("store_trajectory", None)

	if n_jobs == 1:
		results = [
			_sweep_one_map(parameter_name, value, base_parameters, options)
			for value in parameter_values
		]
	elif Parallel is None:
		raise ImportError("joblib is required for parallel sweeps. Install it or use n_jobs=1.")
	else:
		results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
			delayed(_sweep_one_map)(parameter_name, value, base_parameters, options)
			for value in parameter_values
		)

	return {
		"parameter_name": parameter_name,
		"parameter_values": parameter_values,
		"results": results,
		"lambda_max": np.array([result["lambda_max"] for result in results]),
		"base_parameters": base_parameters.copy(),
		"analysis_options": options.copy(),
	}

