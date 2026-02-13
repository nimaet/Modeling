import numpy as np
import scipy
import scipy
from scipy.interpolate import interp1d


class HomogenizedModel:
	def __init__(self, ref_scales, hom_params, branch):
		self.ref_scales = ref_scales
		self.hom_params = hom_params
		self.theta_tilde = ref_scales['theta_tilde']
		self.branch = branch
		if branch == 'acoustic':
			self.Omega_ndim = np.linspace(
				0.001,
				1/np.sqrt(1 + self.theta_tilde**2)*0.997,
				5000
			)
		elif branch == 'optical':
			self.Omega_ndim = np.linspace(1.01, 3.0, 5000)
		self.Omega_dim = self.Omega_ndim / self.ref_scales['t0']
	# ======================
	# Original functions
	# ======================

	def q_fun_ndim(self, Omega):
		q4 = (Omega**4 - Omega**2) / ((1 + self.theta_tilde**2)*Omega**2 - 1)
		return q4**0.25

	def q_fun_dim(self, omega0):
		omega0_ndim = omega0 * self.ref_scales['t0']
		q0 = self.q_fun_ndim(omega0_ndim)
		return q0 / self.ref_scales['x0']

	def P_vg_fun_dim(self):
		Omega = self.Omega_ndim
		q = self.q_fun_ndim(Omega)
		dOmega_dq = np.gradient(Omega, q)
		d2Omega_dq2 = np.gradient(dOmega_dq, q)

		dOmega_dq_func = interp1d(
			Omega, dOmega_dq, kind='cubic', fill_value='extrapolate'
		)
		d2Omega_dq2_func = interp1d(
			Omega, d2Omega_dq2, kind='cubic', fill_value='extrapolate'
		)

		P = lambda omega0: (
			0.5
			* d2Omega_dq2_func(omega0 * self.ref_scales['t0'])
			* self.ref_scales['x0']**2
			/ self.ref_scales['t0']
		)

		v_g = lambda omega0: (
			dOmega_dq_func(omega0 * self.ref_scales['t0'])
			* self.ref_scales['x0']
			/ self.ref_scales['t0']
		)

		return v_g, P

	def P_vg_fun_ndim(self):
		Omega = self.Omega_ndim
		q = self.q_fun_ndim(Omega)
		dOmega_dq = np.gradient(Omega, q)
		d2Omega_dq2 = np.gradient(dOmega_dq, q)

		dOmega_dq_func = interp1d(
			Omega, dOmega_dq, kind='cubic', fill_value='extrapolate'
		)
		d2Omega_dq2_func = interp1d(
			Omega, d2Omega_dq2, kind='cubic', fill_value='extrapolate'
		)

		P = lambda omega0: 0.5 * d2Omega_dq2_func(omega0)
		v_g = lambda omega0: dOmega_dq_func(omega0)

		return v_g, P

	def eigen_vector_ndim(self, omega0):
		q0 = self.q_fun_ndim(omega0)
		C_w1 = (
			1j * self.theta_tilde * omega0 * q0**2
			/ (-omega0**2 + q0**4)
		)
		C_Lambda1 = 1
		return np.array((C_w1, C_Lambda1))

	def F(self, DT, DX):
		return np.array([
			[DT**2 + DX**4, -self.theta_tilde*DT*DX**2],
			[self.theta_tilde*DT*DX**2, DT**2 + 1]
		])

	def F_DT(self, DT, DX):
		return np.array([
			[2*DT, -self.theta_tilde*DX**2],
			[self.theta_tilde*DX**2, 2*DT]
		])

	def Q_fun_dim(self, omega0):
		q0 = self.q_fun_dim(omega0)
		omega0_ndim = omega0 * self.ref_scales['t0']
		eig_vec = self.eigen_vector_ndim(omega0_ndim)

		m_bar = self.hom_params['m_bar']
		Cp_bar = self.hom_params['Cp_bar']

		F_matrix = np.array([
			[2*m_bar*omega0, 1j*self.theta_tilde*q0**2],
			[-1j*self.theta_tilde*q0**2, 2*Cp_bar*omega0]
		])

		coeff = -3 / (
			self.hom_params['Lc_bar']
			* eig_vec.conj().T @ F_matrix @ eig_vec
		)
		return coeff
	
	def sec_envelope(self, omega0, eps_phi):
		v_g, P = self.P_vg_fun_dim()

		Q0 = self.Q_fun_dim(omega0)
		vg0 = v_g(omega0)
		P0 = P(omega0)
		q0 = self.q_fun_dim(omega0)

		envelope_params = {
			'omega0': omega0,
			'vg0': vg0,
			'P0': P0,
			'Q0': Q0,
			'q0': q0,
			'eps_phi': eps_phi
		}

		def A_func(X, T):
			sech_arg = (
				eps_phi * np.sqrt(Q0/(2*P0)) * (X - vg0*T)
			)
			phase_arg = q0*X - omega0*T
			decay = 1#np.exp(1j * 0.5 * eps_phi**2 * Q0 * T)
			
			return (
				eps_phi / np.cosh(sech_arg)
			)

		return A_func, envelope_params
	
	def envelope(self, omega0, eps_phi):
		v_g, P = self.P_vg_fun_dim()

		Q0 = self.Q_fun_dim(omega0)
		vg0 = v_g(omega0)
		P0 = P(omega0)
		q0 = self.q_fun_dim(omega0)

		envelope_params = {
			'omega0': omega0,
			'vg0': vg0,
			'P0': P0,
			'Q0': Q0,
			'q0': q0,
			'eps_phi': eps_phi
		}
		print('sqrt Q0/(2P0)= ', np.sqrt(Q0/(2*P0)))
		def A_func(X, T):
			sech_arg = (
				eps_phi * np.sqrt(Q0/(2*P0)) * (X - vg0*T)
			)
			phase_arg = q0*X - omega0*T
			decay = np.exp(1j * 0.5 * eps_phi**2 * Q0 * T)

			return (
				eps_phi / np.cosh(sech_arg)
				* np.exp(1j*phase_arg)
				* decay
			)

		return A_func, envelope_params

	def make_v_exc(self, omega0, eps_phi, t_eval,  t_shift=0):
		envelope_func, _ = self.envelope(omega0, eps_phi)
		flux_linkage = envelope_func(
			0,
			t_eval - t_shift
		) * 1000
		voltage = scipy.integrate.cumulative_trapezoid(
			flux_linkage, t_eval, initial=0
		) 
		voltage = np.real(voltage)
		return interp1d(
			t_eval, voltage, kind='cubic', fill_value='extrapolate'
		)
	
	def peak_focus(self):

		omega0 = self.Omega_dim
		v_g, P_fun = self.P_vg_fun_dim()

		focusing = np.array([
			self.Q_fun_dim(om0) / P_fun(om0)
			for om0 in omega0
		])

		idx = np.argmax(np.real(focusing))

		return {
			'omega0': omega0[idx],
			'Q': self.Q_fun_dim(omega0[idx]),
			'P': P_fun(omega0[idx]),
			'Q_over_P': focusing[idx]
		}
