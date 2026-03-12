from dataclasses import dataclass, field
import numpy as np

@dataclass
class PiezoBeamParams:
	# ===================== Geometry =====================
	# L_b: float = 0.3185			# beam length [m]
	w_p: float = 10e-3			# patch width [m]
	w_s: float = 0.265625e-3		# spacing between patches [m]
	n_patches: int = 31				# number of unit cells (patches)
	b: float = 10e-3			# beam width [m]
	hp: float = 0.252e-3			# piezo thickness [m]
	hs: float = 0.51e-3			# substrate thickness [m]

	# ===================== Materials =====================
	rho_p: float = 7600.0			# piezo density [kg/m^3]
	rho_s: float = 2700.0			# substrate density [kg/m^3]
	E_s: float = 70e9			# substrate Young's modulus [Pa]
	s11: float = 1.5873e-11			# piezo compliance [m^2/N]
	s44: float = 4.75e-11			# piezo shear compliance [m^2/N]
	d31: float = -1.45e-10			# piezo strain constant [C/N]
	eps0: float = 8.854e-12			# vacuum permittivity
	eps_r: float = 1700			# relative permittivity
	nu_s: float = 0.33			# Poisson ratio (substrate)
	# zeta_p: float = 0.0001
	# zeta_q: float = 0.0001
	omega_p: float = 2*np.pi*100
	omega_q: float = 2*np.pi*6000
	zeta_dict: dict = field(default_factory=lambda: { 1: 0.02, 2: 0.025, 3: 0.03, 4: 0.007, 5: 0.0075, 6:0.0085, 7:0.008, 8:0.007,
			 9: 0.007, 10: 0.0075, 11: 0.008, 12: 0.009, 13: 0.01, 14: 0.012, 15: 0.014, 16: 0.014,
			 17: 0.013, 18: 0.013, 19: 0.015, 20: 0.023, 21: 0.024, 22: 0.025, 'rest': 0.4
			 })

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


	def __post_init__(self):
		# number of patches
		self.Q = self.n_patches
		self.S = self.Q

		# elastic moduli
		self.E_p = 1.0 / self.s11
		self.G_p = 1.0 / self.s44
		self.G_s = self.E_s / (2.0*(1.0 + self.nu_s))

		# piezoelectric constant
		self.e31 = self.d31 / self.s11

		# dielectric constants
		self.eps33_bar = self.eps0 * self.eps_r
		self.eps33 = self.eps33_bar - self.d31**2 / self.s11

		# patch locations
		j = np.arange(1, self.S + 1)
		self.xL = (j - 1)*self.w_p + j*self.w_s
		self.xR = self.xL + self.w_p
		self.L_b = self.xR[-1]+self.w_s
		# mass per unit length
		self.m = self.b * (self.rho_s*self.hs + 2.0*self.rho_p*self.hp)

		# bending stiffness
		term1 = self.E_s * self.hs**3 / 8.0
		term2 = self.E_p * ((self.hp + self.hs/2.0)**3 - self.hs**3/8.0)
		self.YI = 2.0*self.b/3.0 * (term1 + term2)
		self.YI_s = self.b* self.E_s * self.hs**3 / 12
		# capacitance
		self.Cp_scalar = 2.0*self.eps33 * self.w_p * self.b / self.hp
		self.Cp = self.Cp_scalar * np.ones(self.S)

		# electromechanical coupling
		hpc = 0.5*(self.hp + self.hs)
		self.theta_mech = 2.0*self.e31 * self.b * hpc
		self._update_c_alpha_beta()


	def _update_c_alpha_beta(self):
		"""
		Compute Rayleigh damping coefficients from two calibration points.
		
		Rayleigh damping: C = c_alpha * M + c_beta * K
		- c_alpha: mass-proportional coefficient (dominates at low frequencies)
		- c_beta: stiffness-proportional coefficient (dominates at high frequencies)
		
		Modal damping ratio: ζ(ω) = c_alpha/(2ω) + c_beta*ω/2
		"""
		A = np.array([
			[self.omega_p/(2*self.YI), 1/(2*self.m*self.omega_p)],
			[self.omega_q/(2*self.YI), 1/(2*self.m*self.omega_q)]
		])
		b = np.array([self._zeta_p, self._zeta_q])

		self.c_beta, self.c_alpha = np.linalg.solve(A, b)

	@property
	def zeta_p(self):
		return self._zeta_p

	@zeta_p.setter
	def zeta_p(self, value):
		self._zeta_p = value
		self._update_c_alpha_beta()

	@property
	def zeta_q(self):
		return self._zeta_q

	@zeta_q.setter
	def zeta_q(self, value):
		self._zeta_q = value
		self._update_c_alpha_beta()

	def plot_zeta_vs_omega(self, omega_range=None, ax=None):
		"""
		Plot damping ratio (zeta) vs angular frequency (omega) using the two calibration points.
		
		The damping ratio is calculated using the Rayleigh damping model:
		zeta(omega) = c_alpha / (2*omega) + c_beta * omega / 2
		
		Parameters:
		-----------
		omega_range : array-like, optional
			Range of angular frequencies to plot [rad/s]. 
			If None, uses a range from 0.5*omega_p to 2*omega_q
		ax : matplotlib axes, optional
			Axes object to plot on. If None, creates a new figure
			
		Returns:
		--------
		fig, ax : matplotlib figure and axes objects
		"""
		import matplotlib.pyplot as plt
		
		# Create figure if not provided
		if ax is None:
			fig, ax = plt.subplots(figsize=(8, 6))
		else:
			fig = ax.get_figure()
		
		# Define omega range
		if omega_range is None:
			omega_range = np.linspace(0.5*self.omega_p, 2*self.omega_q, 500)
		
		# Calculate zeta using Rayleigh damping model
		zeta = self.c_alpha / (2*omega_range * self.m) + self.c_beta * omega_range / (2*self.YI)
		
		# Plot the curve
		ax.plot(omega_range/(2*np.pi), zeta*100, 'b-', linewidth=2, label='Rayleigh damping')
		
		# Mark the calibration points
		ax.plot(self.omega_p/(2*np.pi), self._zeta_p*100, 'ro', markersize=10, 
				label=f'Point 1: ({self.omega_p/(2*np.pi):.1f} Hz, {self._zeta_p*100:.2f}%)')
		ax.plot(self.omega_q/(2*np.pi), self._zeta_q*100, 'rs', markersize=10,
				label=f'Point 2: ({self.omega_q/(2*np.pi):.1f} Hz, {self._zeta_q*100:.2f}%)')
		
		# Formatting
		ax.set_xlabel('Frequency [Hz]', fontsize=12)
		ax.set_ylabel('Damping Ratio ζ [%]', fontsize=12)
		ax.set_title('Damping Ratio vs Frequency', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best', fontsize=10)
		
		plt.tight_layout()
		
		return fig, ax

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
		m_bar  = self.m
		EI_bar = self.YI
		theta1 = self.theta_mech/self.YI
		theta2 = 0
		# ---- electrical ----
		Cp_bar = self.Cp_scalar/self.w_p

		# inductances (from ROM realization)
		L_bar  = R_c / K_i_eff * self.w_p
		Lc_bar = R_c / K_c * self.w_p

		# ---- electromechanical coupling ----
		# NOTE:
		# theta_mech corresponds to the *local* coupling θ in Eq. (4)
		# ROM + Γ implicitly perform the spatial averaging, so we take:
		theta_bar = self.theta_mech

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
			't0': float(t0),
			'x0': float(x0),
			'lambda0': float(lambda0),
			'w0': float(w0),
			'theta_tilde': float(theta_tilde)
		}