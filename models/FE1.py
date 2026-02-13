import numpy as np
import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
from scipy.linalg import eigh
from Modeling.models.beam_properties import PiezoBeamParams
from dataclasses import dataclass
# from Modeling.models.utils import compute_frf_from_time_domain
from Modeling.models.newmark import newmark_beta_nonlinear
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class PiezoBeamODESystem:
	M: np.ndarray
	C: np.ndarray
	f_ext_unit: np.ndarray
	f_int: callable
	K_tan: callable
	f_ext: callable
	v_exc: callable
	N_mech: int
	N_elec: int


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


class PiezoBeamFE:
	def __init__(self, params, n_el_patch=3, n_el_gap=2):
		self.params = params
		self.n_el_patch = n_el_patch
		self.n_el_gap   = n_el_gap

		self._build_mesh()
		self._build_element_properties()
		self._assemble_KM()
		self._build_Gamma()
		self._apply_bc()

	# --------------------------------------------------
	# Mesh + segmentation
	# --------------------------------------------------
	def _build_mesh(self):

		p = self.params
		xL, xR, L_b, S = p.xL, p.xR, p.L_b, p.S

		segments = []

		if xL[0] > 0:
			segments.append(("gap", 0.0, xL[0]))

		for j in range(S):
			segments.append(("patch", xL[j], xR[j]))
			if j < S-1 and xR[j] < xL[j+1]:
				segments.append(("gap", xR[j], xL[j+1]))

		if xR[-1] < L_b:
			segments.append(("gap", xR[-1], L_b))

		x_nodes = [0.0]
		for seg_type, xa, xb in segments:
			n = self.n_el_patch if seg_type == "patch" else self.n_el_gap
			xs = np.linspace(xa, xb, n+1)
			x_nodes.extend(xs[1:])

		self.x_nodes = np.array(x_nodes)
		self.Nnodes  = len(self.x_nodes)

		self.elements = [(i, i+1) for i in range(self.Nnodes-1)]

		elem_type = []
		for (i, j) in self.elements:
			xm = 0.5*(self.x_nodes[i] + self.x_nodes[j])
			in_patch = any((xm >= xL[k]) and (xm <= xR[k]) for k in range(S))
			elem_type.append("patch" if in_patch else "gap")

		self.elem_type = elem_type

	# --------------------------------------------------
	# Element properties
	# --------------------------------------------------
	def _build_element_properties(self):

		p = self.params

		rhoA_patch = p.b * (p.rho_s*p.hs + 2*p.rho_p*p.hp)
		rhoA_gap   = p.b * p.rho_s * p.hs

		EI_patch = p.YI
		EI_gap   = p.E_s * p.b * p.hs**3 / 12

		self.rhoA_elem = []
		self.EI_elem   = []

		for t in self.elem_type:
			if t == "patch":
				self.rhoA_elem.append(rhoA_patch)
				self.EI_elem.append(EI_patch)
			else:
				self.rhoA_elem.append(rhoA_gap)
				self.EI_elem.append(EI_gap)

	# --------------------------------------------------
	# Assembly
	# --------------------------------------------------
	def _assemble_KM(self):

		Ndof = 2*self.Nnodes
		self.Ndof = Ndof

		K = np.zeros((Ndof, Ndof))
		M = np.zeros((Ndof, Ndof))

		for e, (i, j) in enumerate(self.elements):
			Le = self.x_nodes[j] - self.x_nodes[i]

			EI = self.EI_elem[e]
			Ke = (EI / Le**3) * np.array([
				[ 12,     6*Le,  -12,     6*Le ],
				[ 6*Le,  4*Le**2, -6*Le,  2*Le**2 ],
				[-12,    -6*Le,   12,    -6*Le ],
				[ 6*Le,  2*Le**2, -6*Le,  4*Le**2 ]
			])

			rhoA = self.rhoA_elem[e]
			Me = (rhoA * Le / 420) * np.array([
				[156,     22*Le,   54,     -13*Le],
				[22*Le,  4*Le**2, 13*Le,   -3*Le**2],
				[54,      13*Le,  156,     -22*Le],
				[-13*Le, -3*Le**2, -22*Le,  4*Le**2]
			])

			dofs = [2*i, 2*i+1, 2*j, 2*j+1]

			for a in range(4):
				for b in range(4):
					K[dofs[a], dofs[b]] += Ke[a,b]
					M[dofs[a], dofs[b]] += Me[a,b]

		self.K = K
		self.M = M

	# --------------------------------------------------
	# Electromechanical coupling
	# --------------------------------------------------
	def _build_Gamma(self):

		p = self.params
		S = p.S

		Gamma = np.zeros((self.Ndof, S))
		node_index = {x: i for i, x in enumerate(self.x_nodes)}

		for j in range(S):
			kL = node_index[p.xL[j]]
			kR = node_index[p.xR[j]]

			Gamma[2*kR + 1, j] +=  p.theta_mech
			Gamma[2*kL + 1, j] += -p.theta_mech

		self.Gamma = Gamma

	# --------------------------------------------------
	# Boundary conditions
	# --------------------------------------------------
	def _apply_bc(self):

		fixed_dofs = [0, 1]

		all_dofs = np.arange(self.Ndof)
		self.free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

		self.K_red = self.K[np.ix_(self.free_dofs, self.free_dofs)]
		self.M_red = self.M[np.ix_(self.free_dofs, self.free_dofs)]
		self.Gamma_red = self.Gamma[self.free_dofs, :]

	def eigen_analysis(self):
		"""
		Generalized eigenvalue analysis:
			K φ = ω² M φ
		Returns mass-normalized full-order mode shapes.
		"""

		# ----------------------------
		# Sanity checks
		# ----------------------------
		assert np.linalg.norm(self.K - self.K.T) < 1e-10

		eigvals_test = np.linalg.eigvalsh(self.K_red)
		try:
			assert np.all(eigvals_test > 0)
		except AssertionError:
			print("Warning: Not all eigenvalues of K_red are positive.")

		# ----------------------------
		# Generalized EVP (reduced)
		# ----------------------------
		eigvals, eigvecs = eigh(self.K_red, self.M_red)

		omega = np.sqrt(eigvals)					# rad/s
		freq  = omega / (2*np.pi)					# Hz

		idx = np.argsort(freq)
		self.omega = omega[idx]
		self.freq  = freq[idx]
		eigvecs = eigvecs[:, idx]

		Nmodes = len(self.freq)
		Ndof   = self.Ndof

		# ----------------------------
		# Lift to full DOF space
		# ----------------------------
		Phi = np.zeros((Ndof, Nmodes))

		for i in range(Nmodes):
			Phi[self.free_dofs, i] = eigvecs[:, i]

		# ----------------------------
		# Mass normalization (FULL M)
		# ----------------------------
		for i in range(Nmodes):
			norm = np.sqrt(Phi[:, i].T @ self.M @ Phi[:, i])
			Phi[:, i] /= norm

		self.Phi = Phi

		# consistency checks
		assert np.allclose(Phi.T @ self.M @ Phi, np.eye(Nmodes), atol=1e-8)
		# assert np.allclose(
		# 	Phi.T @ self.K @ Phi,
		# 	np.diag(self.omega**2),
		# 	atol=1e-6
		# )


		return self.freq, self.omega, self.Phi
	
	def build_ode_system(
		self,
		j_exc=30,
		R_c=1e3,
		K_p=0.02,
		K_i=0.0,
		K_c=0.0,
		v_exc = lambda t: 0.0
	):

		
		"""
		Build and return a thread-safe ODE system object.
		Does NOT modify FE state.
		"""

		# ----------------------------
		# DOF bookkeeping
		# ----------------------------
		N = self.M_red.shape[0]
		S = self.Gamma_red.shape[1]

		idx_all = np.arange(S)
		idx_f = np.delete(idx_all, j_exc)

		Gamma_f = self.Gamma_red[:, idx_f]
		Gamma_e = self.Gamma_red[:, j_exc]

		# ----------------------------
		# Excitation (closure-safe)
		# ----------------------------
		if np.isscalar(K_i):
				K_i = K_i * np.ones(len(idx_f))
		else:
			K_i = np.delete(K_i, j_exc)


		# ----------------------------
		# Damping
		# ----------------------------
		D = self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red

		# ----------------------------
		# Block mass matrix
		# ----------------------------
		Cp_scalar = self.params.Cp_scalar
		M_elec = Cp_scalar * np.eye(len(idx_f))

		M_ODE = np.block([
			[self.M_red,					np.zeros((N, len(idx_f)))],
			[np.zeros((len(idx_f), N)),	M_elec]
		])

		# ----------------------------
		# Block damping / coupling
		# ----------------------------
		C_ODE = np.block([
			[D,					-Gamma_f],
			[Gamma_f.T,	(K_p/R_c)*np.eye(len(idx_f))]
		])

		# ----------------------------
		# Internal force
		# ----------------------------
		def f_int(x):
			u = x[:N]
			qf = x[N:]

			fu = self.K_red @ u
			fq = (K_i/R_c)*qf + (K_c/R_c)*qf**3

			return np.concatenate((fu, fq))

		# ----------------------------
		# Tangent stiffness
		# ----------------------------
		def K_tan(x):
			qf = x[N:]

			Kqq = (np.diag(K_i)/R_c) *np.eye(len(qf)) \
				+ (3*K_c/R_c)*np.diag(qf**2)

			return np.block([
				[self.K_red,					np.zeros((N, len(qf)))],
				[np.zeros((len(qf), N)),	Kqq]
			])

		# ----------------------------
		# External force
		# ----------------------------
		def f_ext(t):
			fu = Gamma_e * v_exc(t)
			fq = np.zeros(len(idx_f))
			return np.concatenate((fu, fq))
		fu = Gamma_e 
		fq = np.zeros(len(idx_f))
		f_ext_unit = np.concatenate((fu, fq))
		return PiezoBeamODESystem(
			M=M_ODE,
			C=C_ODE,
			f_int=f_int,
			K_tan=K_tan,
			f_ext=f_ext,
			v_exc=v_exc,
			f_ext_unit=f_ext_unit,
			N_mech=N,
			N_elec=len(idx_f)
		)



if __name__ == "__main__":
	print("Piezoelectric beam FE model test")
	params = PiezoBeamParams()
	hp, hs = 0.252e-3, 0.51e-3 		
	params.hp = hp; params.hs = hs
	fe = PiezoBeamFE(params)
	freq, omega, Phi = fe.eigen_analysis()
	Ndof = Phi.shape[0]
	x_nodes = fe.x_nodes
	w_dofs = np.arange(0, Ndof, 2)
	for i in range(3):
		w_mode = Phi[w_dofs, i]
		plt.plot(x_nodes, w_mode, label=f"Mode {i+1}")
	fe = PiezoBeamFE(params)

	ode = fe.build_ode_system(
		j_exc=30,
		A_exc=50,
		f0=1e3,
		f1=2e3,
		t_end=0.05
	)

	# Initial conditions
	ndof = ode.M.shape[0]
	x0 = np.zeros(ndof)
	x_dot0 = np.zeros(ndof)

	a0 = np.linalg.solve(
		ode.M,
		ode.f_ext(0.0) - ode.C @ x_dot0 - ode.f_int(x0)
	)

	# # Time integration (unchanged)
	# x, x_dot, x_ddot = newmark_beta_nonlinear(
	# 	M=fe.M_ODE,
	# 	C=fe.C_ODE,
	# 	f_int=fe.f_int,
	# 	K_tan=fe.K_tan,
	# 	f_ext=fe.f_ext,
	# 	u0=x0,
	# 	v0=x_dot0,
	# 	a0_init=a0,
	# 	dt=dt,
	# 	n_steps=n_steps
	# )
	# build FE + ODE
	fe = PiezoBeamFE(params)

	ode = fe.build_ode_system(
		j_exc=30,
		K_c=0.0,    # linear circuit
		K_i=0.0
	)

	# frequency range
	f = np.linspace(50, 3000, 2000)
	omega = 2*np.pi*f

	# compute FRF
	X = frf_sweep(ode, omega)

	# split DOFs
	N = ode.N_mech
	U = X[:, :N]          # mechanical
	w = X[:, :N:2]   # transverse displacement
	Q = X[:, N:]          # electrical
	plt.figure(figsize=(7, 4))
	plt.semilogy(f, np.linalg.norm(Q, axis=1), 'r', lw=2)
	plt.semilogy(f, np.linalg.norm(w, axis=1), 'b', lw=2)

	plt.xlabel("Frequency [Hz]")
	plt.ylabel("|Electrical charge|")
	plt.title("Linear FRF – Electrical response")
	plt.grid(True)
	plt.tight_layout()
	plt.show()


