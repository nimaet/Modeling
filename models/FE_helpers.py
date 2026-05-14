
import numpy as np
import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
# from Modeling.models.beam_properties import PiezoBeamParams
# from beam_properties import PiezoBeamParams
from Modeling.models.newmark import newmark_beta_nonlinear
from tqdm import tqdm


from dataclasses import dataclass


@dataclass(frozen=True)
class PiezoBeamFEROMSystem:
	Phi: np.ndarray
	M_modal: np.ndarray
	C_modal: np.ndarray
	K_modal: np.ndarray
	Gamma_modal: np.ndarray
	Gamma_exc: np.ndarray
	Gamma_free: np.ndarray
	M_elec: np.ndarray
	K_elec: np.ndarray
	f_ext: callable
	v_exc: callable
	N_modal: int
	N_elec: int
	N_mech: int
	free_dofs: np.ndarray
	modal_indices: np.ndarray
	R_c: float
	K_p: float
	K_i: np.ndarray
	K_c: float
	freq_domain_amps: np.ndarray


def _pack_rom_result(t, eta, eta_dot, eta_ddot, q, q_dot, rom):
	u = eta @ rom.Phi.T
	u_dot = eta_dot @ rom.Phi.T
	u_ddot = eta_ddot @ rom.Phi.T

	return {
		"t": t,
		"u": u,
		"u_dot": u_dot,
		"u_ddot": u_ddot,
		"q": q,
		"v": q_dot,
		"x": eta,
		"x_dot": eta_dot,
		"x_ddot": eta_ddot,
	}


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
	X = X[ idx]
	
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
		# Harmonize excitation length with solver time vector
		v_exc_values = ode.v_exc(t)
		
		# Handle multi-piezo excitation: convert to 1D signal
		if v_exc_values.ndim == 2:
			# Multi-piezo case: shape is (n_piezos, n_time)
			# Use RMS across piezos to get effective excitation magnitude
			v_exc_values = np.sqrt(np.mean(v_exc_values**2, axis=0))
		
		assert len(v_exc_values) == u_ddot.shape[0], "Excitation length mismatch with time vector"

		spec = compute_frf_from_time_domain(
			t=t,
			veloc=u_dot,
			v_exc_values=v_exc_values
		)
		result['spectral'] = spec
	else:
		result['spectral'] = None

	return result


def build_fe_rom_system(
	fe,
	j_exc=[30],
	R_c=1e3,
	K_p=0.02,
	K_i=0.0,
	K_c=0.0,
	v_exc=lambda t: 1,
	freq_domain_amps=np.array([1.0]),
	n_modes_max: int | None = None,
	freq_max: float = 5000.0,
):
	"""
	Project a full FE beam model to a modal-reduced helper-level ROM.

	The mechanical FE coordinates are reduced to modal coordinates, while the
	electrical piezo states remain in the same free-piezo physical form used by
	the full coupled FE system.
	"""
	if not hasattr(fe, "Phi") or not hasattr(fe, "omega"):
		fe.eigen_analysis()

	N = fe.M_red.shape[0]
	S = fe.Gamma_red.shape[1]

	j_exc = np.atleast_1d(j_exc).astype(int)
	if j_exc.size == 0:
		raise ValueError("j_exc must contain at least one excited piezo index")
	if np.any((j_exc < 0) | (j_exc >= S)):
		raise ValueError(f"j_exc indices out of range [0, {S-1}]")
	j_exc = np.unique(j_exc)

	idx_all = np.arange(S)
	idx_f = np.setdiff1d(idx_all, j_exc)

	if np.isscalar(K_i):
		K_i = K_i * np.ones(len(idx_f))
	else:
		K_i = np.asarray(K_i)
		if K_i.shape[0] == S:
			K_i = np.delete(K_i, j_exc)
		if K_i.shape[0] != len(idx_f):
			raise ValueError(f"K_i length mismatch: expected {len(idx_f)}, got {K_i.shape[0]}")

	if hasattr(fe, "C_red"):
		D = fe.C_red + fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red
	else:
		D = fe.params.c_alpha * fe.M_red + fe.params.c_beta * fe.K_red

	mode_mask = fe.freq <= freq_max
	modal_indices = np.where(mode_mask)[0]
	if n_modes_max is not None:
		modal_indices = modal_indices[:n_modes_max]
	if modal_indices.size == 0:
		raise ValueError("No modes selected for the reduced-order model")

	Phi_full = fe.Phi[fe.free_dofs, :]
	Phi_r = Phi_full[:, modal_indices]

	M_modal = Phi_r.T @ fe.M_red @ Phi_r
	C_modal = Phi_r.T @ D @ Phi_r
	K_modal = Phi_r.T @ fe.K_red @ Phi_r
	Gamma_modal = Phi_r.T @ fe.Gamma_red
	Gamma_free = Gamma_modal[:, idx_f]
	Gamma_exc = Gamma_modal[:, j_exc]
	M_elec = fe.params.Cp_scalar * np.eye(len(idx_f))

	def f_ext(t):
		v_t = v_exc(t)
		if np.isscalar(v_t):
			v_t = np.full(len(j_exc), v_t)
		else:
			v_t = np.asarray(v_t)
			if v_t.shape[0] != len(j_exc):
				raise ValueError(f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}")
		return np.concatenate([Gamma_exc @ v_t, np.zeros(len(idx_f))])

	return PiezoBeamFEROMSystem(
		Phi=Phi_r,
		M_modal=M_modal,
		C_modal=C_modal,
		K_modal=K_modal,
		Gamma_modal=Gamma_modal,
		Gamma_exc=Gamma_exc,
		Gamma_free=Gamma_free,
		M_elec=M_elec,
		K_elec=(K_p / R_c) * np.eye(len(idx_f)),
		f_ext=f_ext,
		v_exc=v_exc,
		N_modal=len(modal_indices),
		N_elec=len(idx_f),
		N_mech=N,
		free_dofs=fe.free_dofs,
		modal_indices=modal_indices,
		R_c=R_c,
		K_p=K_p,
		K_i=np.asarray(K_i),
		K_c=K_c,
		freq_domain_amps=np.asarray(freq_domain_amps),
	)


def build_fe_rom_system_from_ode(
	ode,
	freq_domain_amps=np.array([1.0]),
	n_modes_max: int | None = None,
	freq_max: float = 5000.0,
):
	"""
	Project an already assembled FE ODE system to a modal ROM.

	This version starts from the full coupled ODE object so the free/fixed DOF
	partition and the coupled FE ordering are inherited directly from ode.
	"""
	N = ode.M_mech.shape[0]
	N_elec = ode.N_elec
	D = ode.D if hasattr(ode, "D") else ode.C[:N, :N]
	Gamma_free = np.asarray(ode.Gamma_free)
	Gamma_exc = np.asarray(ode.Gamma_exc)

	if not hasattr(ode, "M_mech") or not hasattr(ode, "K_mech"):
		raise ValueError("ode must expose M_mech and K_mech for ROM projection")

	evals, evecs = eigh(ode.K_mech, ode.M_mech)
	omega_n = np.sqrt(np.maximum(evals, 0.0))
	f_n = omega_n / (2 * np.pi)
	mode_mask = f_n <= freq_max
	modal_indices = np.where(mode_mask)[0]
	if n_modes_max is not None:
		modal_indices = modal_indices[:n_modes_max]
	if modal_indices.size == 0:
		raise ValueError("No modes selected for the reduced-order model")

	Phi = evecs[:, modal_indices]
	for i in range(Phi.shape[1]):
		norm_i = np.sqrt(Phi[:, i].T @ ode.M_mech @ Phi[:, i])
		Phi[:, i] /= norm_i

	omega_r = omega_n[modal_indices]
	Omega2 = np.diag(omega_r**2)
	Zeta = np.diag(np.diag(Phi.T @ D @ Phi) / (2.0 * np.maximum(omega_r, 1e-14)))
	Gamma_free_r = Phi.T @ Gamma_free
	Gamma_exc_r = Phi.T @ Gamma_exc

	M_elec = ode.M[N:, N:]
	K_elec = ode.C[N:, N:]
	M_modal = np.eye(Phi.shape[1])
	C_modal = 2.0 * Zeta @ np.diag(omega_r)
	K_modal = Omega2

	def f_ext(t):
		f_full = ode.f_ext(t)
		if np.ndim(f_full) == 1:
			f_mech = f_full[:N]
			f_elec = f_full[N:]
		else:
			raise ValueError("ode.f_ext(t) must return a 1D vector for ROM projection")
		return np.concatenate([Phi.T @ f_mech, f_elec])

	return PiezoBeamFEROMSystem(
		Phi=Phi,
		M_modal=M_modal,
		C_modal=C_modal,
		K_modal=K_modal,
		Gamma_modal=Gamma_free_r,
		Gamma_exc=Gamma_exc_r,
		Gamma_free=Gamma_free_r,
		M_elec=M_elec,
		K_elec=K_elec,
		f_ext=f_ext,
		v_exc=ode.v_exc,
		N_modal=Phi.shape[1],
		N_elec=N_elec,
		N_mech=N,
		free_dofs=np.arange(N),
		modal_indices=modal_indices,
		R_c=getattr(ode, "R_c", 1e3),
		K_p=getattr(ode, "K_p", 0.0),
		K_i=np.zeros(N_elec),
		K_c=0.0,
		freq_domain_amps=np.asarray(freq_domain_amps),
	)


def solve_ivp_rom(
	rom,
	dt,
	t_end,
	x0=None,
	x_dot0=None,
	method="RK45",
	rtol=1e-8,
	atol=1e-10,
	do_spectral=True,
):
	"""
	Integrate the FE ROM with solve_ivp using a first-order state.

	State ordering is [eta, eta_dot, lam, lam_dot], where eta are modal amplitudes
	and lam are the electrical states.
	"""
	n_modal = rom.N_modal
	n_elec = rom.N_elec
	if x0 is None:
		x0 = np.zeros(n_modal)
	if x_dot0 is None:
		x_dot0 = np.zeros(n_modal)

	eta0 = np.asarray(x0, dtype=float)
	eta_dot0 = np.asarray(x_dot0, dtype=float)
	lam0 = np.zeros(n_elec, dtype=float)
	lam_dot0 = np.zeros(n_elec, dtype=float)
	y0 = np.concatenate([eta0, eta_dot0, lam0, lam_dot0])

	M_ode = np.block([
		[rom.M_modal, np.zeros((n_modal, n_elec))],
		[np.zeros((n_elec, n_modal)), rom.M_elec],
	])
	C_ode = np.block([
		[rom.C_modal, np.zeros((n_modal, n_elec))],
		[rom.Gamma_free.T, rom.K_elec],
	])
	K_ode = np.block([
		[rom.K_modal, -rom.Gamma_exc],
		[np.zeros((n_elec, n_modal)), np.zeros((n_elec, n_elec))],
	])

	def f_int(x):
		eta = x[:n_modal]
		lam = x[n_modal:]
		return np.concatenate([
			rom.K_modal @ eta,
			(rom.K_i / rom.R_c) * lam + (rom.K_c / rom.R_c) * lam**3,
		])

	def K_tan(x):
		lam = x[n_modal:]
		Klam = np.diag(rom.K_i / rom.R_c) + (3.0 * rom.K_c / rom.R_c) * np.diag(lam**2)
		return np.block([
			[rom.K_modal, -rom.Gamma_exc],
			[np.zeros((n_elec, n_modal)), Klam],
		])

	def rhs(t, y):
		eta = y[:n_modal]
		eta_dot = y[n_modal:2 * n_modal]
		lam = y[2 * n_modal:2 * n_modal + n_elec]
		lam_dot = y[2 * n_modal + n_elec:]
		x = np.concatenate([eta, lam])
		x_dot = np.concatenate([eta_dot, lam_dot])
		x_ddot = np.linalg.solve(M_ode, rom.f_ext(t) - C_ode @ x_dot - f_int(x))
		eta_ddot = x_ddot[:n_modal]
		lam_ddot = x_ddot[n_modal:]

		return np.concatenate([eta_dot, eta_ddot, lam_dot, lam_ddot])

	t_eval = np.linspace(0.0, t_end, int(np.round(t_end / dt)) + 1)
	sol = solve_ivp(rhs, [0.0, t_end], y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
	if not sol.success:
		raise RuntimeError(sol.message)

	eta = sol.y[:n_modal, :].T
	eta_dot = sol.y[n_modal:2 * n_modal, :].T
	lam = sol.y[2 * n_modal:2 * n_modal + n_elec, :].T
	lam_dot = sol.y[2 * n_modal + n_elec:, :].T

	eta_ddot = np.zeros_like(eta)
	x_ddot_hist = np.zeros((len(sol.t), n_modal + n_elec))
	for k, t in enumerate(sol.t):
		x = np.concatenate([eta[k], lam[k]])
		x_dot = np.concatenate([eta_dot[k], lam_dot[k]])
		x_ddot = np.linalg.solve(M_ode, rom.f_ext(t) - C_ode @ x_dot - f_int(x))
		x_ddot_hist[k] = x_ddot
		eta_ddot[k] = x_ddot[:n_modal]

	result = _pack_rom_result(sol.t, eta, eta_dot, eta_ddot, lam, lam_dot, rom)
	result["x_ddot"] = x_ddot_hist

	if do_spectral:
		v_exc_values = rom.v_exc(sol.t)
		if np.ndim(v_exc_values) == 2:
			v_exc_values = np.sqrt(np.mean(v_exc_values**2, axis=0))
		result["spectral"] = compute_frf_from_time_domain(sol.t, result["u_dot"], v_exc_values)
	else:
		result["spectral"] = None

	return result


def frequency_response_fe_rom(rom, omega):
	"""
	Frequency-domain response of the FE ROM.

	Solves the reduced coupled harmonic system in modal coordinates and returns
	the reconstructed mechanical response in physical coordinates.
	"""
	M_ode = np.block([
		[rom.M_modal, np.zeros((rom.N_modal, rom.N_elec))],
		[np.zeros((rom.N_elec, rom.N_modal)), rom.M_elec],
	])
	C_ode = np.block([
		[rom.C_modal, np.zeros((rom.N_modal, rom.N_elec))],
		[rom.Gamma_free.T, rom.K_elec],
	])
	K_ode = np.block([
		[rom.K_modal, -rom.Gamma_exc],
		[np.zeros((rom.N_elec, rom.N_modal)), np.zeros((rom.N_elec, rom.N_elec))],
	])
	f_hat = rom.f_ext(0.0)
	if f_hat.shape[0] != (rom.N_modal + rom.N_elec):
		raise ValueError("ROM forcing dimension mismatch")

	Z = -omega**2 * M_ode + 1j * omega * C_ode + K_ode
	x_hat = np.linalg.solve(Z, f_hat)
	eta_hat = x_hat[:rom.N_modal]
	u_hat = rom.Phi @ eta_hat
	return u_hat, x_hat


def frf_sweep_fe_rom(rom, omega_vec):
	"""
	Compute a frequency response sweep for the FE ROM.
	"""
	U = np.zeros((len(omega_vec), rom.N_mech), dtype=complex)
	X = np.zeros((len(omega_vec), rom.N_modal + rom.N_elec), dtype=complex)
	for k, w in enumerate(tqdm(omega_vec, desc="ROM FRF sweep")):
		U[k], X[k] = frequency_response_fe_rom(rom, w)

	u_dot = np.zeros_like(U)
	for k, w in enumerate(omega_vec):
		u_dot[k] = 1j * w * U[k]

	return {
		"omega": omega_vec,
		"freq": omega_vec / (2 * np.pi),
		"u": U,
		"u_dot": u_dot,
		"X": X,
	}


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
	f_hat = ode.f_ext_freq_domain
	Z = -omega**2 * M + 1j*omega*C + K
	x_hat = np.linalg.solve(Z, f_hat)

	return x_hat

def frequency_response_SC(ode,	omega ):
	"""
	Linear frequency-domain response of the coupled ODE system.

	Solves:
	(-ω² M + iω C + K) x̂ = f̂
	"""

	M = ode.M_mech
	C = ode.D
	K = ode.K_mech
	# harmonic forcing amplitude
	f_hat = ode.f_ext_unit.copy()
	f_hat = f_hat[:ode.N_mech]  # Only mechanical forcing
	Z = -omega**2 * M + 1j*omega*C + K
	x_hat = np.linalg.solve(Z, f_hat)

	return x_hat

import numpy as np
from scipy.linalg import eigh


def frequency_response_SC_modal(
	ode,
	omega: float,
	freq_max: float = 5000,
	n_modes_max: int | None = None
):
	"""
	Modal-reduced linear frequency-domain response
	with residual flexibility correction.

	Parameters
	----------
	ode : PiezoBeamODESystem
	omega : float
		Excitation frequency [rad/s]
	freq_max : float
		Maximum frequency of interest in Hz
		(used to determine modal cutoff)
	n_modes_max : int | None
		Optional hard cap on number of retained modes

	Returns
	-------
	x_hat : ndarray
		Complex displacement response (mechanical DOFs only)
	"""

	M = ode.M_mech
	C = ode.D
	K = ode.K_mech

	f_hat = ode.f_ext_unit[:ode.N_mech]

	# -----------------------------------------
	# 1) Solve eigenproblem
	# -----------------------------------------
	# K phi = w^2 M phi
	evals, evecs = eigh(K, M)

	omega_n = np.sqrt(np.maximum(evals, 0.0))  # rad/s
	f_n = omega_n / (2*np.pi)

	# -----------------------------------------
	# 2) Select modes
	# -----------------------------------------
	# Keep modes up to ~2.5 × freq_max
	cutoff = 1 * freq_max
	mode_mask = f_n <= cutoff

	if n_modes_max is not None:
		mode_indices = np.where(mode_mask)[0][:n_modes_max]
	else:
		mode_indices = np.where(mode_mask)[0]

	Phi = evecs[:, mode_indices]
	omega_r = omega_n[mode_indices]

	# -----------------------------------------
	# 3) Modal matrices
	# -----------------------------------------
	Mm = Phi.T @ M @ Phi
	Cm = Phi.T @ C @ Phi
	Km = Phi.T @ K @ Phi
	fm = Phi.T @ f_hat

	# -----------------------------------------
	# 4) Solve reduced system
	# -----------------------------------------
	Zm = -omega**2 * Mm + 1j*omega*Cm + Km
	q_hat = np.linalg.solve(Zm, fm)

	# Back to physical coordinates
	x_modal = Phi @ q_hat

	# -----------------------------------------
	# 5) Residual flexibility correction
	# -----------------------------------------
	# K^{-1} f  minus modal static contribution
	K_inv_f = np.linalg.solve(K, f_hat)

	Km_static = Phi @ np.linalg.solve(Km, fm)

	x_residual = K_inv_f - Km_static

	x_hat = x_modal #+ x_residual

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


def frf_sweep_SC(ode, omega_vec):
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
	N_mech = ode.N_mech
	X = np.zeros((len(omega_vec), N_mech), dtype=complex)
	for k, w in enumerate(tqdm(omega_vec, desc="FRF sweep")):
		# X[k] = frequency_response_SC(ode, w) 
		X[k] = frequency_response_SC_modal(ode, w) 
	# Separate mechanical and electrical DOFs
	u = X[:, :N_mech:2]                    # mechanical displacement
	# Velocity and voltage (derivatives in frequency domain)
	u_dot = np.zeros_like(u)
	
	for k, w in enumerate(omega_vec):
		u_dot[k] = 1j * w * u[k]         # velocity = iω * displacement
	return {
		'omega': omega_vec,
		'freq': omega_vec / (2*np.pi),
		'u': u,
		'u_dot': u_dot,
		'X': X  # full state for backwards compatibility
	}

