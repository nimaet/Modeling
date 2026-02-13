
import numpy as np
import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from scipy.linalg import eigh
# from Modeling.models.beam_properties import PiezoBeamParams
# from beam_properties import PiezoBeamParams
from Modeling.models.newmark import newmark_beta_nonlinear
from tqdm import tqdm

def compute_frf_from_time_domain(t, veloc, v_exc_values):
	"""
	Compute Frequency Response Function (FRF) from time domain response.
	
	Performs FFT on velocity field and excitation signal, then computes
	the spatially-averaged transfer function.
	
	Parameters
	----------
	t : array_like
		Time vector [s]
	veloc : ndarray
		Velocity field array of shape (n_time, n_spatial) where:
		- n_spatial: number of spatial points
		- n_time: number of time samples
	v_exc_values : array_like
		Excitation voltage/velocity values at each time point,
		shape (n_time,)
	
	Returns
	-------
	result : dict
		Dictionary containing:
		- 'freq': frequency array [Hz]
		- 'FRF': frequency response function (spatially averaged magnitude)
		- 'Y': FFT of velocity field, shape (n_freq, n_spatial)
		- 'X': FFT of excitation signal, shape (n_freq,)
	
	Notes
	-----
	- Only positive frequencies are returned
	- FRF is computed as: mean(|Y|) / |X| over spatial dimension
	- If time vector has < 2 samples, returns None values
	"""
	t = np.asarray(t)
	veloc = np.asarray(veloc)
	v_exc_values = np.asarray(v_exc_values)
	
	Nt = len(t)
	
	if Nt < 2:
		return {
			'freq': None,
			'FRF': None,
			'Y': None,
			'X': None
		}
	
	dt = t[1] - t[0]
	
	# FFT of velocity field (spatial x temporal)
	Y = np.fft.fft(veloc, axis=0)
	
	# FFT of excitation signal
	X = np.fft.fft(v_exc_values)
	
	# Frequency vector [Hz]
	freq = np.fft.fftfreq(Nt, d=dt)
	
	# Keep only positive frequencies
	print("X shape:", X.shape, "Y shape:", Y.shape, "freq shape:", freq.shape, "veloc shape:", veloc.shape)
	idx = freq >= 0
	freq = freq[idx]
	Y = Y[idx, :]
	X = X[idx]
	
	# FRF: spatially averaged magnitude of transfer function
	# Avoid division by zero
	X_mag = np.abs(X)
	X_mag = np.where(X_mag < 1e-10, 1.0, X_mag)  # Replace near-zero with 1 to avoid division by zero
	
	FRF = np.mean(np.abs(Y), axis=1) / X_mag
	
	return {
		'freq': freq,
		'FRF': FRF,
		'Y': Y,
		'X': X
	}


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
):
	ndof = ode.M.shape[0]

	if x0 is None:
		x0 = np.zeros(ndof)
	if x_dot0 is None:
		x_dot0 = np.zeros(ndof)

	a0 = np.linalg.solve(
		ode.M,
		ode.f_ext(0.0) - ode.C @ x_dot0 - ode.f_int(x0)
	)

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
		newton_maxiter=newton_maxiter
	)

	t = np.linspace(0.0, n_steps*dt, n_steps+1)

	# Extract mechanical and electrical DOFs
	N_mech = ode.N_mech
	N_elec = ode.N_elec
	
	u = x[:, :N_mech:2]              # mechanical displacement (n_steps+1, N_mech)
	u_dot = x_dot[:, :N_mech:2]      # mechanical velocity (n_steps+1, N_mech)
	u_ddot = x_ddot[:, :N_mech:2]    # mechanical acceleration (n_steps+1, N_mech)
	q = x[:, N_mech:]              # electrical charge (n_steps+1, N_elec)
	v = x_dot[:, N_mech:]          # voltage/dq dt (n_steps+1, N_elec)

	result = {
		't': t,
		'u': u,                    # mechanical displacement
		'u_dot': u_dot,            # mechanical velocity
		'u_ddot': u_ddot,          # mechanical acceleration
		'q': q,                    # electrical charge
		'v': v,                    # voltage (dq/dt)
		# Keep full state for backwards compatibility if needed
		'x': x,
		'x_dot': x_dot,
		'x_ddot': x_ddot
	}

	if do_spectral:
		# Harmonize excitation length with solver time vecto
		v_exc_values = ode.v_exc(t)
		assert len(v_exc_values) == u_ddot.shape[0], "Excitation length mismatch with time vector"
		# if v_exc_values.shape[0] == t.shape[0] - 1:
		# 	# Common case when using np.arange for excitation sampling
		# 	v_exc_values = np.append(v_exc_values, v_exc_values[-1])
		# elif v_exc_values.shape[0] != t.shape[0]:
		# 	t_exc = np.linspace(t[0], t[-1], v_exc_values.shape[0])
		# 	v_exc_values = np.interp(t, t_exc, v_exc_values)

		spec = compute_frf_from_time_domain(
			t=t,
			veloc=u_dot,
			v_exc_values=v_exc_values
		)
		result['spectral'] = spec
	else:
		result['spectral'] = None

	return result


def frequency_response_linear(ode,	omega ):
	"""
	Linear frequency-domain response of the coupled ODE system.

	Solves:
	(-ω² M + iω C + K) x̂ = f̂
	"""

	M = ode.M
	C = ode.C
	K = ode.K_tan(np.zeros(M.shape[0]))
	# harmonic forcing amplitude
	f_hat = ode.f_ext_unit
	Z = -omega**2 * M + 1j*omega*C + K
	x_hat = np.linalg.solve(Z, f_hat)

	return x_hat

def frf_sweep(ode, omega_vec):
	"""
	Compute frequency response sweep over a range of frequencies.
	
	Parameters
	----------
	ode : PiezoBeamODESystem
		ODE system object with M, C, K_tan, etc.
	omega_vec : array_like
		Array of angular frequencies [rad/s]
	
	Returns
	-------
	result : dict
		Dictionary containing:
		- 'omega': angular frequency array [rad/s]
		- 'freq': frequency array [Hz]
		- 'u': mechanical displacement response (n_freq, N_mech), complex
		- 'u_dot': mechanical velocity response (n_freq, N_mech), complex
		- 'q': electrical charge response (n_freq, N_elec), complex
		- 'v': voltage response (n_freq, N_elec), complex
		- 'X': full state vector (n_freq, ndof), complex (for backwards compatibility)
	"""
	ndof = ode.M.shape[0]
	N_mech = ode.N_mech
	N_elec = ode.N_elec
	
	X = np.zeros((len(omega_vec), ndof), dtype=complex)

	for k, w in enumerate(tqdm(omega_vec, desc="FRF sweep")):
		X[k] = frequency_response_linear(ode, w)

	# Separate mechanical and electrical DOFs
	u = X[:, :N_mech:2]                    # mechanical displacement
	q = X[:, N_mech:]                    # electrical charge
	
	# Velocity and voltage (derivatives in frequency domain)
	u_dot = np.zeros_like(u)
	v = np.zeros_like(q)
	
	for k, w in enumerate(omega_vec):
		u_dot[k] = 1j * w * u[k]         # velocity = iω * displacement
		v[k] = 1j * w * q[k]             # voltage = iω * charge

	return {
		'omega': omega_vec,
		'freq': omega_vec / (2*np.pi),
		'u': u,
		'u_dot': u_dot,
		'q': q,
		'v': v,
		'X': X  # full state for backwards compatibility
	}

