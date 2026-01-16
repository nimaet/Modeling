import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

from Modeling.models.jit_stuff import odefun_jit, rk4_integrate
# from Modeling.tasks.Reduced_order.linear_Kp_sweep import v_exc
# add Modeling/ to Python path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from Modeling.models.beam_properties import PiezoBeamParams

class ROM:
	def __init__(self, params=PiezoBeamParams(), N=30):
		self.p = params
		self.N = N
		self.S = params.S

		self._compute_eigen()
		self._compute_modes()
		self._compute_coupling()
		self.rayleigh_damping()

	def _compute_eigen(self):
		def eigen_eq(lam):
			return 1.0 + np.cos(lam)*np.cosh(lam)

		def lambda_(i):
			lam0 = np.pi*(i + 0.5)
			return fsolve(eigen_eq, lam0)[0]

		self.lambda_vals = np.array([lambda_(i) for i in range(self.N)])
		self.beta = self.lambda_vals / self.p.L_b
		self.omega = self.beta**2 * np.sqrt(self.p.YI / self.p.m)
		self.omega2 = self.omega**2

	def _compute_modes(self):
		def stable_exp_term(lam, x):
			num = np.sin(lam) + np.cos(lam) + np.exp(-lam)
			den = np.cos(lam)*np.exp(-lam) + 0.5 + 0.5*np.exp(-2*lam)
			return 0.5*(num / den) * np.exp(lam*(x/self.p.L_b - 1.0))

		def sigma_r(lam):
			num = 2*np.sin(lam)*np.exp(-lam) - 1 + np.exp(-2*lam)
			den = 2*np.cos(lam)*np.exp(-lam) + 1 + np.exp(-2*lam)
			return num / den

		self._stable_exp_term = stable_exp_term
		self.sigma_vals = np.array([sigma_r(l) for l in self.lambda_vals])

	def mode_shape(self, r, x):
		lam = self.lambda_vals[r]
		sigma = self.sigma_vals[r]
		x = np.asarray(x)

		term = self._stable_exp_term(lam, x)

		phi = (
			np.cos(lam*x/self.p.L_b)
			+ sigma*np.sin(lam*x/self.p.L_b)
			- term
			- 0.5*(1 - sigma)*np.exp(-lam*x/self.p.L_b)
		)

		return np.sqrt(1.0/(self.p.m*self.p.L_b)) * phi

	def mode_shape_dx(self, r, x):
		lam = self.lambda_vals[r]
		sigma = self.sigma_vals[r]
		x = np.asarray(x)

		theta = lam*x/self.p.L_b
		term = self._stable_exp_term(lam, x)
		term_dx = term * (lam/self.p.L_b)

		last_exp = 0.5*(1 - sigma)*np.exp(-lam*x/self.p.L_b)
		last_exp_dx = -last_exp * (lam/self.p.L_b)

		phi_x = (
			-np.sin(theta)*(lam/self.p.L_b)
			+ sigma*np.cos(theta)*(lam/self.p.L_b)
			- term_dx
			- last_exp_dx
		)

		return np.sqrt(1.0/(self.p.m*self.p.L_b)) * phi_x

	def _compute_coupling(self):
		self.Gamma = np.zeros((self.N, self.S))
		for r in range(self.N):
			for j in range(self.S):
				self.Gamma[r, j] = (
					self.mode_shape_dx(r, self.p.xR[j])
					- self.mode_shape_dx(r, self.p.xL[j])
				)

	# ===================== damping (your fitted Rayleigh model) =====================
	def rayleigh_damping(self):
		self.zeta = (
			self.p.c_alpha/(2*self.omega*self.p.m)
			+ self.p.c_beta*self.omega/(2*self.p.YI)
		)

		self.damp = 2*self.zeta*self.omega

	# ===================== ODE =====================
	def odefun(self, t, x, v_exc, j_exc, R_c, K_c=0, K_p=0, K_i=0):
		N = self.N
		S = self.S

		eta = x[0:N]
		eta_dot = x[N:2*N]
		z = x[2*N:2*N+S]
		v = x[2*N+S:2*N+2*S]

		v[j_exc] = v_exc(t)

		eta_ddot = (
			- self.damp*eta_dot
			- self.omega2*eta
			+ self.p.theta_mech*(self.Gamma @ v)
		)

		strain_coupling = self.Gamma.T @ eta_dot
		duffing = z**3

		v_dot = (
			-(K_p/R_c)*v
			-(K_i/R_c)*z
			- self.p.theta_mech*strain_coupling
			-(K_c/R_c)*duffing
		) / self.p.Cp

		z_dot = v

		return np.concatenate([eta_dot, eta_ddot, z_dot, v_dot])

	# ===================== frequency response =====================
	def frequency_response(self, j_exc=30, R_c=1e3, K_c=0, K_p=0, K_i=0, w=None, x_eval=None):
		if w is None:
			w = np.arange(0.1, 4500, 2.5)*2*np.pi

		N = self.N
		S = self.S
		dim = 2*N + 2*S

		A = np.zeros((dim, dim))
		A[0:N, N:2*N] = np.eye(N)
		A[N:2*N, 0:N] = -np.diag(self.omega2)
		A[N:2*N, N:2*N] = -np.diag(self.damp)
		A[N:2*N, 2*N+S:2*N+2*S] = self.p.theta_mech*self.Gamma
		A[2*N:2*N+S, 2*N+S:2*N+2*S] = np.eye(S)
		A[2*N+S:2*N+2*S, N:2*N] = -(self.p.theta_mech/self.p.Cp_scalar)*self.Gamma.T
		A[2*N+S:2*N+2*S, 2*N:2*N+S] = -np.eye(S)*K_i/R_c/self.p.Cp_scalar
		A[2*N+S:2*N+2*S, 2*N+S:2*N+2*S] = -np.eye(S)*K_p/R_c/self.p.Cp_scalar
		A[N:2*N, 2*N + j_exc] = 0.0

		f = np.zeros(dim)
		f[N:2*N] = self.p.theta_mech*self.Gamma[:, j_exc]

		I = np.eye(dim)
		Y = []

		for wi in w:
			y = np.linalg.solve(1j*wi*I - A, f)
			Y.append(y)
		Y = np.array(Y)    # shape (n_freq, dim)
		# Extract modal coordinates from frequency-response solution
		eta     = Y[:, 0:N].T        # shape (N, n_freq)
		eta_dot = Y[:, N:2*N].T      # shape (N, n_freq)
		if x_eval is None:
			x_eval = np.linspace(0, self.p.L_b, 100)

		npts = len(x_eval)
		nfreq = eta.shape[1]

		disp  = np.zeros((npts, nfreq), dtype=complex)
		veloc = np.zeros((npts, nfreq), dtype=complex)

		for r in range(N):
			phi_r  = self.mode_shape(r, x_eval)       # shape (npts,)
			disp  += np.outer(phi_r, eta[r, :])
			veloc += np.outer(phi_r, eta_dot[r, :])
		vel_mag = np.mean(np.abs(veloc), axis=0 )
		disp_mag = np.mean(np.abs(disp), axis=0 )

		freq_modal = w / (2*np.pi)
		return freq_modal, vel_mag, disp_mag, veloc

	# ===================== dispersion analysis =====================
	def dispersion_analysis(self, j_exc=30, R_c=1e3, K_c=0, K_p=0, K_i=0, w=None, x_eval=None):
		"""
		Compute dispersion relation using spatial spectral analysis on velocity field.
		
		Parameters:
		-----------
		j_exc : int
			Index of excited shunt
		R_c, K_c, K_p, K_i : float
			Shunt circuit parameters
		w : array_like, optional
			Angular frequency array. Default: 0.1 to 4500 Hz
		x_eval : array_like, optional
			Spatial positions for evaluation. Default: 100 points along beam
			
		Returns:
		--------
		freq : array
			Frequency array [Hz]
		wavenumber : array
			Wavenumber array [rad/m]
		spectrum : 2D array
			Wavenumber-frequency spectrum magnitude
		"""
		if not np.isscalar(K_i):
			K_i = np.diag(K_i)

		if w is None:
			w = np.arange(0.1, 4500, 2)*2*np.pi
		
		if x_eval is None:
			x_eval = np.linspace(0, self.p.L_b, 1000)
		
		N = self.N
		S = self.S
		dim = 2*N + 2*S

		# Build system matrix (same as frequency_response)
		A = np.zeros((dim, dim))
		A[0:N, N:2*N] = np.eye(N)
		A[N:2*N, 0:N] = -np.diag(self.omega2)
		A[N:2*N, N:2*N] = -np.diag(self.damp)
		A[N:2*N, 2*N+S:2*N+2*S] = self.p.theta_mech*self.Gamma
		A[2*N:2*N+S, 2*N+S:2*N+2*S] = np.eye(S)
		A[2*N+S:2*N+2*S, N:2*N] = -(self.p.theta_mech/self.p.Cp_scalar)*self.Gamma.T
		A[2*N+S:2*N+2*S, 2*N:2*N+S] = -np.eye(S)*K_i/R_c/self.p.Cp_scalar
		A[2*N+S:2*N+2*S, 2*N+S:2*N+2*S] = -np.eye(S)*K_p/R_c/self.p.Cp_scalar
		A[N:2*N, 2*N + j_exc] = 0.0

		f = np.zeros(dim)
		f[N:2*N] = self.p.theta_mech*self.Gamma[:, j_exc]

		I = np.eye(dim)
		
		npts = len(x_eval)
		nfreq = len(w)
		
		# Allocate velocity field array
		veloc = np.zeros((npts, nfreq), dtype=complex)
		
		# Solve for each frequency
		for i, wi in enumerate(w):
			y = np.linalg.solve(1j*wi*I - A, f)
			eta_dot = y[N:2*N]
			
			# Reconstruct velocity at spatial points
			for r in range(N):
				phi_r = self.mode_shape(r, x_eval)
				veloc[:, i] += phi_r * eta_dot[r]
		
		# Spatial FFT along beam for each frequency
		dx = x_eval[1] - x_eval[0]
		spectrum_spatial = np.fft.fft(veloc, axis=0)
		wavenumber = 2*np.pi*np.fft.fftfreq(npts, d=dx)
		
		# Get magnitude spectrum
		spectrum = np.abs(spectrum_spatial)
		
		# Shift to center zero frequency/wavenumber for better visualization
		spectrum = np.fft.fftshift(spectrum, axes=0)
		wavenumber = np.fft.fftshift(wavenumber)
		
		# Transpose spectrum so frequency is on y-axis: spectrum[freq_idx, wavenumber_idx]
		spectrum = spectrum.T
		
		freq = w / (2*np.pi)
		result = {
			'freq': freq,
			'wavenumber': wavenumber,
			'spectrum': spectrum,
			'veloc': veloc}

		return result

	def run_time_sim(
		self,
		v_exc,
		j_exc=30,
		R_c=1e3,
		K_c=0,
		K_p=0,
		K_i=0,
		t_end=0.1,
		t_eval=None,
		x_eval=None,
		rtol=1e-9,
		atol=1e-10
	):
		if x_eval is None:
			x_eval = np.linspace(0, self.p.L_b, 100)

		N = self.N
		S = self.S

		x0 = np.zeros(2*N + 2*S)

		sol = solve_ivp(
			lambda t, x: self.odefun(
				t, x,
				v_exc=v_exc,
				j_exc=j_exc,
				R_c=R_c,
				K_c=K_c,
				K_p=K_p,
				K_i=K_i
			),
			(0.0, t_end),
			x0,
			t_eval=t_eval,
			method='RK45',
			rtol=rtol,
			atol=atol
		)

		# reconstruct velocity field
		eta_dot = sol.y[N:2*N, :]
		veloc = np.zeros((len(x_eval), eta_dot.shape[1]))

		for r in range(N):
			veloc += np.outer(self.mode_shape(r, x_eval), eta_dot[r, :])
				# =========================
		# Spectrum / FRF analysis
		# =========================
		t = sol.t
		Nt = len(t)

		if Nt > 1:
			dt = t[1] - t[0]
			fs = 1.0 / dt

			# FFT of velocity field (space x time)
			Y = np.fft.fft(veloc, axis=1)

			# FFT of excitation signal
			X = np.fft.fft(v_exc(t))

			# Frequency vector
			freq = np.fft.fftfreq(Nt, d=dt)

			# Positive frequencies only
			idx = freq >= 0
			freq = freq[idx]
			Y = Y[:, idx]
			X = X[idx]

			# FRF (spatially averaged magnitude)
			FRF = np.mean(np.abs(Y), axis=0) / np.abs(X)
		else:
			freq = None
			Y = None
			X = None
			FRF = None
	
		return {
			't': sol.t,
			'state': sol.y,
			'veloc': veloc,
			'x_eval': x_eval,
			'freq': freq,
			'Y': Y,
			'X': X,
			'FRF': FRF
		}
	

	def run_time_sim_jit(
		self,
		v_exc,
		j_exc=30,
		R_c=1e3,
		K_c=0,
		K_p=0,
		K_i=0,
		t_end=0.1,
		t_eval=None,
		x_eval=None
	):
		if x_eval is None:
			x_eval = np.linspace(0, self.p.L_b, 100)

		if t_eval is None:
			raise ValueError("t_eval must be provided for fixed-step RK4")

		N = self.N
		S = self.S
		dim = 2*N + 2*S

		x0 = np.zeros(dim)

		t_arr = np.asarray(t_eval, dtype=np.float64)
		v_exc_arr = np.asarray(v_exc(t_arr), dtype=np.float64)

		# ---- homogenize Ki for JIT ----
		if np.ndim(K_i) > 0:
			K_i_eff = float(np.mean(K_i))
		else:
			K_i_eff = float(K_i)

		# ---- allocate state OUTSIDE JIT ----
		dim = 2*N + 2*S
		Nt = t_arr.size
		state = np.zeros((dim, Nt), dtype=np.float64)
		state[:, 0] = x0

		rk4_integrate(
			state,
			t_arr,
			v_exc_arr,
			N,
			S,
			self.damp,
			self.omega2,
			self.Gamma,
			self.p.theta_mech,
			self.p.Cp_scalar,
			float(R_c),
			float(K_c),
			float(K_p),
			float(K_i_eff),
			int(j_exc)
		)



		# ---- reconstruct velocity field ----
		eta_dot = state[N:2*N, :]
		veloc = np.zeros((len(x_eval), eta_dot.shape[1]))

		for r in range(N):
			veloc += np.outer(self.mode_shape(r, x_eval), eta_dot[r, :])

		return {
			't': t_eval,
			'state': state,
			'veloc': veloc,
			'x_eval': x_eval
		}


	def homogenized_parameters(self, K_i, K_c, R_c):
		"""
		Compute homogenized parameters entering Eq. (13) of the paper,
		consistent with the ROM realization.

		Parameters
		----------
		K_i : float or array
			Integral gain(s) of shunt (1/L equivalent)
		K_c : float
			Cubic gain of shunt (1/L_c equivalent)
		R_c : float
			Shunt resistance

		Returns
		-------
		params : dict
			Homogenized parameters:
			m_bar, EI_bar, Cp_bar, L_bar, Lc_bar, theta_bar
		"""

		# effective Ki (homogenized in space)
		if np.ndim(K_i) > 0:
			K_i_eff = np.mean(K_i)
		else:
			K_i_eff = K_i

		# ---- mechanical ----
		m_bar  = self.p.m
		EI_bar = self.p.YI
		theta1 = self.p.theta_mech/self.p.YI
		theta2 = 0
		# ---- electrical ----
		Cp_bar = self.p.Cp_scalar/self.p.w_p

		# inductances (from ROM realization)
		L_bar  = R_c / K_i_eff * self.p.w_p
		Lc_bar = R_c / K_c * self.p.w_p

		# ---- electromechanical coupling ----
		# NOTE:
		# theta_mech corresponds to the *local* coupling θ in Eq. (4)
		# ROM + Γ implicitly perform the spatial averaging, so we take:
		theta_bar = self.p.theta_mech

		return {
			'm_bar': m_bar,
			'EI_bar': EI_bar,
			'Cp_bar': Cp_bar,
			'L_bar': L_bar,
			'Lc_bar': Lc_bar,
			'theta_bar': theta_bar
		}

	def nondimensional_scales(self, K_i, K_c, R_c):
		"""
		Compute nondimensional scales (Eq. 15–16) required for
		the perturbation solution (Eq. 51–52).

		Returns
		-------
		scales : dict
			t0, x0, lambda0, w0, theta_tilde
		"""

		h = self.homogenized_parameters(K_i, K_c, R_c)

		m_bar     = h['m_bar']
		EI_bar    = h['EI_bar']
		Cp_bar    = h['Cp_bar']
		L_bar     = h['L_bar']
		Lc_bar    = h['Lc_bar']
		theta_bar = h['theta_bar']

		# ---- Eq. (16) ----
		t0 = np.sqrt(Cp_bar * L_bar)

		x0 = (EI_bar * Cp_bar * L_bar / m_bar)**0.25

		lambda0 = np.sqrt(abs(Lc_bar / L_bar))

		w0 = np.sqrt(Cp_bar * abs(Lc_bar) / (m_bar * L_bar))

		# ---- Eq. (15) ----
		theta_tilde = theta_bar / np.sqrt(Cp_bar * EI_bar)

		return {
			't0': t0,
			'x0': x0,
			'lambda0': lambda0,
			'w0': w0,
			'theta_tilde': theta_tilde
		}

