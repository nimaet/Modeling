# %
import numpy as np
from scipy.optimize import root_scalar, fsolve
from scipy.integrate import simpson as simps
# conf = 'series'
# # === Step 3: Define material+geo constants (Table 2.1) ===
# rho_p, rho_s = 7800, 8500          # [kg/m³]
# E_p, E_s   = 66e9, 100e9           # [Pa]
# e31, eps33 = -12.3, 14.8e-9        # [C/m²], [F/m]
# L, b       = 56e-3, 31.75e-3     # [m]
# hp, hs     = 0.267e-3, 0.127e-3    # [m]
# m = b*(rho_s * hs + 2 * rho_p * hp)
# M_t     = 0         # [kg]
# I_t = 0

# # Etrurk=== Step 3: Define material+geo constants (Table 2.1) ===
# # rho_p, rho_s = 7750, 2700          # [kg/m³]
# # E_p, E_s   = 61e9, 70e9           # [Pa]
# # e31, eps33 = -10.4, 13.3e-9        # [C/m²], [F/m]
# # L, b       = 30e-3, 5e-3     # [m]
# # hp, hs     = 0.15e-3, 0.05e-3    # [m]
# # m = b*(rho_s * hs + 2 * rho_p * hp)
# # I_t = 0  # kg·m², rotational inertia of the tip mass
# # M_t     = 0         # [kg]
# # Flexural rigidity YI via (3.10)
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import simpson as simps

class PiezoBeamFRF:
	def __init__(
		self, conf='series',
		rho_p=7800, rho_s=8500,
		E_p=66e9, E_s=100e9,
		e31=-12.3, eps33=14.8e-9,
		L=56e-3, b=31.75e-3,
		hp=0.267e-3, hs=0.127e-3,
		N_modes=4, zeta=0.01,
		M_t=0, I_t=0
	):
		self.conf = conf
		self.rho_p = rho_p
		self.rho_s = rho_s
		self.E_p = E_p
		self.E_s = E_s
		self.e31 = e31
		self.eps33 = eps33
		self.L = L
		self.b = b
		self.hp = hp
		self.hs = hs
		self.N_modes = N_modes
		self.zeta = zeta
		self.M_t = M_t
		self.I_t = I_t

		self.m = self.b * (self.rho_s * self.hs + 2 * self.rho_p * self.hp)
		self.hpc = (self.hp + self.hs)/2

		# Flexural rigidity
		term1 = self.E_s * self.hs**3 / 8
		term2 = self.E_p * ((self.hp + self.hs/2)**3 - self.hs**3/8)
		self.YI = 2 * self.b / 3 * (term1 + term2)

		self._compute_modes_and_shapes()
		self._compute_modal_properties()

	def eigen_eq(self, lam):
		t1 = 1 + np.cos(lam)*np.cosh(lam)
		t2 = lam*self.M_t/(self.m*self.L)*(np.cos(lam)*np.sinh(lam) - np.sin(lam)*np.cosh(lam))
		t3 = (lam**3 * self.I_t/ self.L**3 / self.m)*(np.cosh(lam)*np.sin(lam) + np.sinh(lam)*np.cos(lam))
		t4 = (lam**4 * self.M_t * self.I_t)/(self.m**2 * self.L**4)*(1 - np.cos(lam)*np.cosh(lam))
		return t1 + t2 - t3 + t4

	def _lambda_i(self, i):
		result = fsolve(self.eigen_eq, np.pi * (i + 0.5))
		return result[0]

	def _compute_modes_and_shapes(self):
		self.lambda_r = np.array([self._lambda_i(i) for i in range(self.N_modes)])
		self.omega_r = np.array([lam**2 * np.sqrt(self.YI/(self.m*self.L**4)) for lam in self.lambda_r])

		self.zeta_r = []
		for lam in self.lambda_r:
			num = (np.sin(lam) - np.sinh(lam)
				   + (self.M_t/(self.m*self.L))*(np.cos(lam)-np.cosh(lam)))
			den = (np.cos(lam) + np.cosh(lam)
				   - (self.M_t/(self.m*self.L))*(np.sin(lam)-np.sinh(lam)))
			self.zeta_r.append(num/den)
		self.zeta_r = np.array(self.zeta_r)

		self.xg = np.linspace(0, self.L, 500)
		self.A_r = []
		for i, lam in enumerate(self.lambda_r):
			phi_vals = self.phi_raw(self.xg, lam, self.zeta_r[i])
			dphiL = (lam/self.L) * (
				-np.sin(lam) - np.sinh(lam) + self.zeta_r[i] * (np.cos(lam) - np.cosh(lam))
			)
			I1 = self.m * simps(phi_vals**2, x =self.xg)
			I2 = self.M_t * (phi_vals[-1]**2)
			I3 = self.I_t * (dphiL**2)
			norm = I1 + I2 + I3
			self.A_r.append(1.0 / np.sqrt(norm))
		self.A_r = np.array(self.A_r)

	def phi_raw(self, x, lam, zr):
		arg = lam * x / self.L
		return (np.cos(arg) - np.cosh(arg)
				+ zr * (np.sin(arg) - np.sinh(arg)))

	def phi(self, r, x):
		return self.A_r[r] * self.phi_raw(x, self.lambda_r[r], self.zeta_r[r])

	def dphi_dx_at_L(self, r):
		lam, zr = self.lambda_r[r], self.zeta_r[r]
		arg = lam
		return self.A_r[r]*(lam/self.L)*(
			-np.sin(arg)-np.sinh(arg)
			+ zr*(np.cos(arg)-np.cosh(arg))
		)

	def _compute_modal_properties(self):
		self.sigma_r, self.tau_r = [], []
		for r in range(self.N_modes):
			ph = self.phi(r, self.xg)
			self.sigma_r.append(-self.m*simps(ph, x = self.xg) - self.M_t*self.phi(r, self.L))
			self.tau_r.append(-self.m*simps(self.xg*ph, x = self.xg) - self.M_t*self.L*self.phi(r, self.L))
		self.sigma_r = np.array(self.sigma_r)
		self.tau_r = np.array(self.tau_r)

		if self.conf == 'parallel':
			self.theta_r = 2*self.e31*self.b*self.hpc * np.array([self.dphi_dx_at_L(r) for r in range(self.N_modes)])
			self.C_p_eq  = 2*self.eps33*self.b*self.L/self.hp
		elif self.conf == 'series':
			self.theta_r = self.e31*self.b*self.hpc * np.array([self.dphi_dx_at_L(r) for r in range(self.N_modes)])
			self.C_p_eq  = self.eps33*self.b*self.L/self.hp/2
		else:
			raise ValueError('Invalid configuration: choose "series" or "parallel".')

	# FRFs and external interface methods
	def R_l(self, omega):
		return 1e3  # Default: 1kΩ; override as needed

	def FRF_alpha(self, omega, R_l=None):
		zeta = self.zeta
		if R_l is None:
			R_l = self.R_l
		den_mech = self.omega_r**2 - omega**2 + 1j*2*zeta*self.omega_r*omega
		num = np.sum(-1j*omega*self.theta_r*self.sigma_r / den_mech)
		den = (1/R_l(omega) + 1j*omega*self.C_p_eq
			   + np.sum(1j*omega*self.theta_r**2 / den_mech))
		return num/den

	def FRF_mu(self, omega, R_l=None):
		zeta = self.zeta
		if R_l is None:
			R_l = self.R_l
		den_mech = self.omega_r**2 - omega**2 + 1j*2*zeta*self.omega_r*omega
		num = np.sum(-1j*omega*self.theta_r*self.tau_r / den_mech)
		den = (1/R_l(omega) + 1j*omega*self.C_p_eq
			   + np.sum(1j*omega*self.theta_r**2 / den_mech))
		return num/den

	def FRF_beta(self, omega, x, R_l=None):
		zeta = self.zeta
		if R_l is None:
			R_l = self.R_l
		den_mech = self.omega_r**2 - omega**2 + 1j*2*zeta*self.omega_r*omega
		num_alpha = np.sum(1j*omega * self.theta_r * self.sigma_r / den_mech)
		den_elec  = (
			1.0/R_l(omega)
			+ 1j*omega*self.C_p_eq
			+ np.sum(1j*omega * self.theta_r**2 / den_mech)
		)
		alpha = num_alpha/den_elec
		s = 0+0j
		for r in range(self.N_modes):
			coeff = self.sigma_r[r]  - self.theta_r[r]*alpha
			s += coeff*self.phi(r, x)/den_mech[r]
		return s

	def FRF_trans(self, omega, x, R_l=None):
		return self.FRF_beta(omega, x, R_l)*omega**2

	def FRF_psi(self, omega, x, R_l=None):
		zeta = self.zeta
		if R_l is None:
			R_l = self.R_l
		den_mech = self.omega_r**2 - omega**2 + 1j*2*zeta*self.omega_r*omega
		den_elec = 1/R_l(omega) + 1j*omega*self.C_p_eq + np.sum(1j*omega*self.theta_r**2/den_mech)
		s = 0+0j
		for r in range(self.N_modes):
			denom = den_mech[r]
			coeff = (self.tau_r[r]
					 - self.theta_r[r]/den_elec*(1j*omega*self.theta_r[r]/denom))
			s += coeff*(1j*omega*self.theta_r[r]*self.tau_r[r]/denom)*self.phi(r, x)/denom
		return s
	
if __name__=='__main__':	
	import matplotlib.pyplot as plt
	# %matplotlib widget
	# Frequency range: 1 Hz to 5000 Hz
	frequencies = np.linspace(1, 1000, 10000)
	omegas = 2 * np.pi * frequencies

	# Load resistances and legend labels
	R_l_list = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
	labels   = ['100 Ω', '1 kΩ', '10 kΩ', '100 kΩ', '1 MΩ', '10 MΩ']
	beam = PiezoBeamFRF(conf='series')  # or 'parallel
	fig, ax = plt.subplots(1,2,figsize=(12,6))

	for Rl, lbl in zip(R_l_list, labels):
		# Temporarily override R_l(ω) to be constant = Rl
		def R_l(omega, Rl=Rl):
			return Rl
		# Compute α(ω) across the band
		alpha_vals = np.array([beam.FRF_alpha(w, R_l=R_l) for w in omegas])
		dyn_compl = np.array([beam.FRF_beta(w, L, R_l=R_l) for w in omegas])

		# Plot magnitude |α| (V/g) on log–log axes
		
		ax[0].set_xlabel('Frequency (Hz)', fontsize=15)
		# plt.ylim([1e-5, 1e2])
		ax[0].set_xlim([0, 1000])
		ax[0].set_title('Voltage FRF α(ω) for Various Load Resistances')
		ax[0].set_ylabel('|α(ω)| [V/g]')
		ax[0].semilogy(frequencies, np.abs(alpha_vals), linewidth=1.5, label=lbl)
		ax[1].set_xlim([85, 145])
		# plt.ylim([1e1, 1e3])
		ax[1].semilogy(frequencies, np.abs(dyn_compl*(frequencies*2*np.pi)**2), linewidth=1.5, label=lbl)
		ax[1].set_title('Tip displacement/ base acceleration FRF')
		ax[1].set_ylabel(r'$[\mu m/g]$')

	ax[1].set_xlabel('Frequency (Hz)')


	ax[1].legend(title='Rₗ', loc='upper right')
	ax[1].grid(which='both', linestyle='--', linewidth=0.5)
	# plt.tight_layout()
	plt.show()

