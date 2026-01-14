from dataclasses import dataclass, field
import numpy as np

@dataclass
class PiezoBeamParams:
	# ===================== Geometry =====================
	L_b: float = 0.3185			# beam length [m]
	w_p: float = 10e-3			# patch width [m]
	w_s: float = 0.265625e-3		# spacing between patches [m]
	Q: int = 31				# number of unit cells (patches)
	b: float = 10e-3			# beam width [m]
	hp: float = 0.252e-3			# piezo thickness [m]
	hs: float = 0.51e-3			# substrate thickness [m]

	# ===================== Materials =====================
	rho_p: float = 7600.0			# piezo density [kg/m^3]
	rho_s: float = 2700.0			# substrate density [kg/m^3]
	E_s: float = 70e9			# substrate Young's modulus [Pa]
	s11: float = 1.5873e-11			# piezo compliance [m^2/N]
	s44: float = 4.75e-11			# piezo shear compliance [m^2/N]
	d31: float = -1.75e-10			# piezo strain constant [C/N]
	eps0: float = 8.854e-12			# vacuum permittivity
	eps_r: float = 1900.0			# relative permittivity
	nu_s: float = 0.33			# Poisson ratio (substrate)
	# zeta_p: float = 0.0151
	# zeta_q: float = 0.0392
	omega_p: float = 2*np.pi*1070
	omega_q: float = 2*np.pi*5892.5

	# ===================== Derived quantities =====================
	S: int = field(init=False)
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
	_zeta_p: float = field(default=0.0151, repr=False)
	_zeta_q: float = field(default=0.0392, repr=False)


	def __post_init__(self):
		# number of patches
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
