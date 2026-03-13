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

from scipy.linalg import eigh

# ============================================================
# Geometry container (NEW)
# ============================================================

@dataclass
class GeometrySpec:
	x_nodes: np.ndarray
	elem_EI: np.ndarray
	elem_rhoA: np.ndarray
	# NEW (pure geometry, no FE math)
	piezos: list   # each piezo has xL, xR, theta (optional)



# ============================================================
# ODE system container (UNCHANGED)
# ============================================================

@dataclass(frozen=True)
class PiezoBeamODESystem:
	M: np.ndarray
	M_mech: np.ndarray
	K_mech: np.ndarray
	C: np.ndarray
	D: np.ndarray
	f_ext_freq_domain: np.ndarray
	f_int: callable
	K_tan: callable
	f_ext: callable
	v_exc: callable
	N_mech: int
	N_elec: int


class PiezoBeamFE:
	def __init__(self, params,  n_el_patch=3, n_el_gap=2):
		self.params = params
		self.n_el_patch = n_el_patch
		self.n_el_gap   = n_el_gap
		# ---- geometry selection (NEW) ----
		if hasattr(self.params, "geometry"):
			self.geom = self.params.geometry
		else:
			self.geom = self._build_default_geometry()

		# ---- assembly ----
		self._build_Gamma()  # build Gamma based on provided geometry
		self._assemble_KM()
		self._apply_bc()
		self.eigen_analysis()
		self.damping_matrix_from_modal_damping()

	# ========================================================
	# Default geometry (EXTRACTED, UNCHANGED LOGIC)
	# ========================================================

	def _build_default_geometry(self):
		p = self.params
		xL, xR, L_b, n_patches = p.xL, p.xR, p.L_b, p.n_patches

		segments = []
		if xL[0] > 0:
			segments.append(("gap", 0.0, xL[0]))

		for j in range(n_patches):
			segments.append(("patch", xL[j], xR[j]))
			if j < n_patches-1 and xR[j] < xL[j+1]:
				segments.append(("gap", xR[j], xL[j+1]))

		if xR[-1] < L_b:
			segments.append(("gap", xR[-1], L_b))

		x_nodes = [0.0]
		elem_type = []

		for seg_type, xa, xb in segments:
			n = self.n_el_patch if seg_type == "patch" else self.n_el_gap
			xs = np.linspace(xa, xb, n+1)
			for k in range(n):
				elem_type.append(seg_type)
			x_nodes.extend(xs[1:])

		x_nodes = np.array(x_nodes)
		Nnodes = len(x_nodes)

		# ---- element properties ----
		rhoA_patch = p.b * (p.rho_s*p.hs + 2*p.rho_p*p.hp)
		rhoA_gap   = p.b * p.rho_s * p.hs

		EI_patch = p.YI
		EI_gap   = p.E_s * p.b * p.hs**3 / 12

		elem_rhoA = []
		elem_EI   = []

		for t in elem_type:
			if t == "patch":
				elem_rhoA.append(rhoA_patch)
				elem_EI.append(EI_patch)
			else:
				elem_rhoA.append(rhoA_gap)
				elem_EI.append(EI_gap)

		elem_rhoA = np.array(elem_rhoA)
		elem_EI   = np.array(elem_EI)
		# build default piezo descriptors from params
		piezos = []
		for j in range(p.S):
			piezos.append({
				"xL": p.xL[j],
				"xR": p.xR[j],
			})

		return GeometrySpec(
			x_nodes=x_nodes,
			elem_EI=elem_EI,
			elem_rhoA=elem_rhoA,
			piezos=piezos
		)


	# ========================================================
	# Assembly
	# ========================================================
	def _build_Gamma(self):
		x_nodes = self.geom.x_nodes
		piezos  = self.geom.piezos
		Nnodes = len(x_nodes)
		S = len(piezos)
		Ndof = 2 * Nnodes
		Gamma = np.zeros((Ndof, S))
		node_index = {x: i for i, x in enumerate(x_nodes)}
		for j in range(S):
			xL_j = piezos[j]['xL']
			xR_j = piezos[j]['xR']
			if xL_j not in node_index:
				raise ValueError(f"Piezo {j}: xL={xL_j} not found in mesh nodes")
			if xR_j not in node_index:
				raise ValueError(f"Piezo {j}: xR={xR_j} not found in mesh nodes")
			kL = node_index[xL_j]
			kR = node_index[xR_j]
			Gamma[2*kR + 1, j] +=  self.params.theta_mech
			Gamma[2*kL + 1, j] += -self.params.theta_mech
		self.Gamma = Gamma

	def _assemble_KM(self):
		x_nodes = self.geom.x_nodes
		elem_EI = self.geom.elem_EI
		elem_rhoA = self.geom.elem_rhoA

		Nnodes = len(x_nodes)
		Ndof = 2 * Nnodes
		self.Ndof = Ndof

		K = np.zeros((Ndof, Ndof))
		M = np.zeros((Ndof, Ndof))

		for e in range(len(elem_EI)):
			i = e
			j = e + 1
			Le = x_nodes[j] - x_nodes[i]

			EI = elem_EI[e]
			rhoA = elem_rhoA[e]

			Ke = (EI / Le**3) * np.array([
				[ 12,     6*Le,  -12,     6*Le ],
				[ 6*Le,  4*Le**2, -6*Le,  2*Le**2 ],
				[-12,    -6*Le,   12,    -6*Le ],
				[ 6*Le,  2*Le**2, -6*Le,  4*Le**2 ]
			])

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


	def _apply_bc(self):
		all_dofs = np.arange(self.Ndof)
		self.fixed_dofs = [0, 1]
		self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)

		self.K_red = self.K[np.ix_(self.free_dofs, self.free_dofs)]
		self.M_red = self.M[np.ix_(self.free_dofs, self.free_dofs)]
		self.Gamma_red = self.Gamma[self.free_dofs, :]

	# ========================================================
	# Eigen analysis (UNCHANGED)
	# ========================================================

	def eigen_analysis(self):
		eigvals, eigvecs = eigh(self.K_red, self.M_red)
		omega = np.sqrt(eigvals)
		freq = omega / (2*np.pi)

		idx = np.argsort(freq)
		omega = omega[idx]
		freq = freq[idx]
		eigvecs = eigvecs[:, idx]

		Phi = np.zeros((self.Ndof, len(freq)))
		for i in range(len(freq)):
			Phi[self.free_dofs, i] = eigvecs[:, i]
			Phi[:, i] /= np.sqrt(Phi[:, i].T @ self.M @ Phi[:, i])

		self.freq = freq
		self.omega = omega
		self.Phi = Phi
		self.eigvecs = eigvecs
		return freq, omega, Phi

	def damping_matrix_from_modal_damping(self):
		self.zeta = np.array([self.params.zeta_dict.get(i+1, self.params.zeta_dict['rest']) for i in range(self.M_red.shape[0])])	
		assert len(self.zeta) == len(self.omega), "Length of zeta must match number of modes" 
		C_modal = 2 *self.zeta * self.omega
		# C_red = self.Phi[self.free_dofs, :].T @ (self.params.c_alpha*self.M + self.params.c_beta*self.K) @ self.Phi[self.free_dofs, :]
		C_modal_matrix = np.diag(C_modal)
		C_reconstructed = np.linalg.inv(self.eigvecs).T @ C_modal_matrix @ np.linalg.inv(self.eigvecs)
		self.C_red = C_reconstructed
		return C_reconstructed 
	# ========================================================
	# ODE construction (UNCHANGED)
	# ========================================================

	def build_ode_system(
		self,
		j_exc=[30],
		R_c=1e3,
		K_p=0.02,
		K_i=0.0,
		K_c=0.0,
		v_exc=lambda t: 1, # can be scalar or vector of length len(j_exc)
		freq_domain_amps=np.array([1.0]),
	):
		"""Build ODE system for coupled piezo-beam dynamics.
		
		Args:
		  j_exc: indices of excited (actuator) piezos
		  R_c: circuit resistance
		  K_p, K_i, K_c: proportional, integral, cubic feedback gains
		  v_exc: excitation voltage function v_exc(t)
		"""
		# Mechanical DOF count and number of piezos
		N = self.M_red.shape[0]
		S = self.Gamma_red.shape[1]

		# Validate and normalize excited piezo indices
		j_exc = np.atleast_1d(j_exc).astype(int)
		if j_exc.size == 0:
			raise ValueError("j_exc must contain at least one excited piezo index")
		if np.any((j_exc < 0) | (j_exc >= S)):
			raise ValueError(f"j_exc indices out of range [0, {S-1}]")
		j_exc = np.unique(j_exc)

		# idx_f: indices of free (sensing/feedback) piezos (all piezos minus excited ones)
		idx_all = np.arange(S)
		idx_f = np.setdiff1d(idx_all, j_exc)

		# Split Gamma matrix into feedback and excitation columns
		Gamma_f = self.Gamma_red[:, idx_f]  # coupling for free piezos
		Gamma_e = self.Gamma_red[:, j_exc]  # coupling for excited piezos

		# Prepare integral gain K_i for free piezos
		if np.isscalar(K_i):
			K_i = K_i * np.ones(len(idx_f))
		else:
			K_i = np.delete(K_i, j_exc)
			if len(K_i) != len(idx_f):
				raise ValueError(f"K_i length mismatch: expected {len(idx_f)}, got {len(K_i)}")

		# Damping and mass matrices
		if hasattr(self, 'C_red'):
			D = self.C_red + self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red
		else: 
			D = self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red
		# print('alpha, beta', self.params.c_alpha, self.params.c_beta)
		M_elec = self.params.Cp_scalar * np.eye(len(idx_f))

		# Combined ODE state matrix (mech DOFs + electrical DOFs)
		M_ODE = np.block([
			[self.M_red, np.zeros((N, len(idx_f)))],
			[np.zeros((len(idx_f), N)), M_elec]
		])

		# Damping/stiffness matrix
		C_ODE = np.block([
			[D, -Gamma_f],
			[Gamma_f.T, (K_p/R_c)*np.eye(len(idx_f))]
		])

		# Internal (nonlinear) force function
		def f_int(x):
			u = x[:N]
			qf = x[N:]
			return np.concatenate([
				self.K_red @ u,
				(K_i/R_c)*qf + (K_c/R_c)*qf**3
			])

		# Tangent stiffness (for Newton methods)
		def K_tan(x):
			qf = x[N:]
			Kqq = (np.diag(K_i)/R_c) + (3*K_c/R_c)*np.diag(qf**2)
			return np.block([
				[self.K_red, np.zeros((N, len(qf)))],
				[np.zeros((len(qf), N)), Kqq]
			])

		# External force from excitation voltages
		def f_ext(t):
			v_t = v_exc(t)
			if np.isscalar(v_t):
				v_t = np.full(len(j_exc), v_t)
			else:
				v_t = np.asarray(v_t)
				if v_t.shape[0] != len(j_exc):
					raise ValueError(
						f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}"
					)
			return np.concatenate([Gamma_e @ v_t, np.zeros(len(idx_f))])

		f_ext_freq_domain = np.concatenate([Gamma_e @ (np.ones(len(j_exc)) * freq_domain_amps), np.zeros(len(idx_f))])
		# f_ext_freq_domain = f_ext(0.0)
		return PiezoBeamODESystem(
			M=M_ODE,
			M_mech=self.M_red,
			K_mech=self.K_red,
			C=C_ODE,
			D=D,
			f_int=f_int,
			K_tan=K_tan,
			f_ext=f_ext,
			v_exc=v_exc,
			f_ext_freq_domain=f_ext_freq_domain,
			N_mech=N,
			N_elec=len(idx_f)
		)


	def build_ode_system_base_excitation(
			self,
			u_base,                 # base displacement function u_base(t)
			du_base=None,           # optional velocity
			ddu_base=None,          # optional acceleration
			j_exc=[30],
			R_c=1e3,
			K_p=0.02,
			K_i=0.0,
			K_c=0.0,
			v_exc=lambda t: 0.0,
	):
		"""
		ODE system with prescribed base displacement.

		Base DOF = DOF 0 (before BC removal).
		The base motion enters as equivalent forcing.
		"""

		N = self.M_red.shape[0]
		S = self.Gamma_red.shape[1]

		# Allow no electrical actuation by passing j_exc=None or j_exc=[]
		if j_exc is None:
			j_exc = np.array([], dtype=int)
		else:
			j_exc = np.atleast_1d(j_exc).astype(int)
			j_exc = np.unique(j_exc)

		if np.any((j_exc < 0) | (j_exc >= S)):
			raise ValueError(f"j_exc indices out of range [0, {S-1}]")

		idx_all = np.arange(S)
		idx_f = np.setdiff1d(idx_all, j_exc)

		# Normalize K_i to the free piezo channels
		if np.isscalar(K_i):
			K_i = K_i * np.ones(len(idx_f))
		else:
			K_i = np.asarray(K_i)
			if K_i.shape[0] == S:
				K_i = np.delete(K_i, j_exc)
			if K_i.shape[0] != len(idx_f):
				raise ValueError(f"K_i length mismatch: expected {len(idx_f)}, got {K_i.shape[0]}")

		Gamma_f = self.Gamma_red[:, idx_f]
		Gamma_e = self.Gamma_red[:, j_exc]

		# -----------------------------------
		# Partition full matrices
		# -----------------------------------

		b = 0   # base dof index in full system

		free = self.free_dofs

		Mfb = self.M[np.ix_(free, [b])]
		# Cfb = self.params.c_alpha*self.M[np.ix_(free, [b])] + self.params.c_beta*self.K[np.ix_(free, [b])]
		# Cfb = np.zeros_like(self.K[:  0])
		Kfb = self.K[np.ix_(free, [b])]

		# -----------------------------------
		# Damping
		# -----------------------------------

		if hasattr(self, 'C_red'):
			D = self.C_red + self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red
		else:
			D = self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red

		M_elec = self.params.Cp_scalar * np.eye(len(idx_f))

		M_ODE = np.block([
			[self.M_red, np.zeros((N, len(idx_f)))],
			[np.zeros((len(idx_f), N)), M_elec]
		])

		C_ODE = np.block([
			[D, -Gamma_f],
			[Gamma_f.T, (K_p/R_c)*np.eye(len(idx_f))]
		])

		# -----------------------------------
		# internal forces
		# -----------------------------------

		def f_int(x):
			u = x[:N]
			qf = x[N:]

			return np.concatenate([
				self.K_red @ u,
				(K_i/R_c)*qf + (K_c/R_c)*qf**3
			])

		def K_tan(x):
			qf = x[N:]

			Kqq = (np.diag(K_i)/R_c) + (3*K_c/R_c)*np.diag(qf**2)

			return np.block([
				[self.K_red, np.zeros((N, len(qf)))],
				[np.zeros((len(qf), N)), Kqq]
			])

		# -----------------------------------
		# base forcing term
		# -----------------------------------

		def base_force(t):

			u = u_base(t)

			if du_base is None:
				ud = 0.0
			else:
				ud = du_base(t)

			if ddu_base is None:
				udd = 0.0
			else:
				udd = ddu_base(t)

			return (
				- Mfb.flatten()*udd
				# - Cfb.flatten()*ud
				# - Kfb.flatten()*u
			)

		# -----------------------------------
		# external force
		# -----------------------------------

		def f_ext(t):

			if len(j_exc) == 0:
				piezo_force = np.zeros(N)
			else:
				v_t = v_exc(t)

				if np.isscalar(v_t):
					v_t = np.full(len(j_exc), v_t)
				else:
					v_t = np.asarray(v_t)
					if v_t.shape[0] != len(j_exc):
						raise ValueError(
							f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}"
						)

				piezo_force = Gamma_e @ v_t

			mech_force = piezo_force + base_force(t)

			return np.concatenate([
				mech_force,
				np.zeros(len(idx_f))
			])

		f_ext_freq_domain = f_ext(0.0)

		return PiezoBeamODESystem(
			M=M_ODE,
			M_mech=self.M_red,
			K_mech=self.K_red,
			C=C_ODE,
			D=D,
			f_int=f_int,
			K_tan=K_tan,
			f_ext=f_ext,
			v_exc=v_exc,
			f_ext_freq_domain=f_ext_freq_domain,
			N_mech=N,
			N_elec=len(idx_f)
		)

	def build_ode_system_nonlocal(
			self,
			j_exc=[30],
			R_c=1e3,
			K_p=0.02,
			v_exc=lambda t: 1,
			freq_domain_amps=np.array([1.0]),
			electrical_network=None,
	):
		"""
		Build ODE system for coupled piezo–beam dynamics using
		an energy-based electrical network description.

		Electrical DOFs are flux linkages φ.
		Nonlocal and nonlinear elements are defined via energy functions.
		"""

		# -----------------------------
		# Mechanical / electrical sizes
		# -----------------------------
		N = self.M_red.shape[0]
		S = self.Gamma_red.shape[1]

		j_exc = np.atleast_1d(j_exc).astype(int)
		j_exc = np.unique(j_exc)

		if np.any((j_exc < 0) | (j_exc >= S)):
			raise ValueError("j_exc index out of range")

		idx_all = np.arange(S)
		idx_f = np.setdiff1d(idx_all, j_exc)
		Nf = len(idx_f)

		Gamma_f = self.Gamma_red[:, idx_f]
		Gamma_e = self.Gamma_red[:, j_exc]

		# -----------------------------
		# Mass and damping matrices
		# -----------------------------
		if hasattr(self, 'C_red'):
			D = self.C_red + self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red
		else: 
			D = self.params.c_alpha*self.M_red + self.params.c_beta*self.K_red
			
		M_elec = self.params.Cp_scalar * np.eye(Nf)

		M_ODE = np.block([
			[self.M_red, np.zeros((N, Nf))],
			[np.zeros((Nf, N)), M_elec]
		])

		C_ODE = np.block([
			[D, -Gamma_f],
			[Gamma_f.T, (K_p / R_c) * np.eye(Nf)]
		])

		# -----------------------------
		# Electrical internal force
		# -----------------------------
		def electrical_force(qf):
			f = np.zeros(Nf)

			if electrical_network is None:
				return f

			for elem in electrical_network["elements"]:
				i, j = elem["nodes"]

				phi_i = qf[i]
				phi_j = 0.0 if j in (None, "gnd") else qf[j]

				dphi = phi_i - phi_j
				g = elem["grad"](dphi)

				f[i] += g
				if j not in (None, "gnd"):
					f[j] -= g

			return f

		# -----------------------------
		# Electrical tangent matrix
		# -----------------------------
		def electrical_tangent(qf):
			K = np.zeros((Nf, Nf))

			if electrical_network is None:
				return K

			for elem in electrical_network["elements"]:
				i, j = elem["nodes"]

				phi_i = qf[i]
				phi_j = 0.0 if j in (None, "gnd") else qf[j]

				dphi = phi_i - phi_j
				k = elem["hess"](dphi)

				K[i, i] += k
				if j not in (None, "gnd"):
					K[j, j] += k
					K[i, j] -= k
					K[j, i] -= k

			return K

		# -----------------------------
		# Internal force (full system)
		# -----------------------------
		def f_int(x):
			u = x[:N]
			qf = x[N:]

			return np.concatenate([
				self.K_red @ u,
				electrical_force(qf) / R_c
			])

		# -----------------------------
		# Tangent stiffness (Newton)
		# -----------------------------
		def K_tan(x):
			qf = x[N:]

			Kqq = electrical_tangent(qf) / R_c

			return np.block([
				[self.K_red, np.zeros((N, Nf))],
				[np.zeros((Nf, N)), Kqq]
			])

		# -----------------------------
		# External excitation force
		# -----------------------------
		def f_ext(t):
			v_t = v_exc(t)

			if np.isscalar(v_t):
				v_t = np.full(len(j_exc), v_t)
			else:
				v_t = np.asarray(v_t)
				if v_t.shape[0] != len(j_exc):
					raise ValueError("v_exc length mismatch")

			return np.concatenate([
				Gamma_e @ v_t,
				np.zeros(Nf)
			])

		f_ext_freq_domain = np.concatenate([
			Gamma_e @ (np.ones(len(j_exc)) * freq_domain_amps),
			np.zeros(Nf)
		])

		return PiezoBeamODESystem(
			M=M_ODE,
			M_mech=self.M_red,
			K_mech=self.K_red,
			C=C_ODE,
			D=D,
			f_int=f_int,
			K_tan=K_tan,
			f_ext=f_ext,
			v_exc=v_exc,
			f_ext_freq_domain=f_ext_freq_domain,
			N_mech=N,
			N_elec=Nf
		)
# ============================================================
# Example geometry builder (NEW)
# ============================================================
def build_geometry_arbitrary_piezos(
	L: float,
	xL: np.ndarray,
	xR: np.ndarray,
	EI_patch: float,
	rhoA_patch: float,
	EI_gap: float,
	rhoA_gap: float,
	h_patch: float, # element size in patch regions
	h_gap: float # element size in gap regions
)-> GeometrySpec:

	"""
	Build geometry with explicit meshing per segment.

	- Different EI / rhoA in piezo vs gap regions
	- User-controlled element size per region
	- Exact nodes at all piezo edges
	"""

	xL = np.asarray(xL, dtype=float)
	xR = np.asarray(xR, dtype=float)

	assert len(xL) == len(xR)
	assert np.all(xL < xR)
	assert np.all(xL >= 0.0) and np.all(xR <= L)

	# --------------------------------------------------
	# Build segments: (type, x_start, x_end)
	# --------------------------------------------------
	segments = []

	if xL[0] > 0.0:
		segments.append(("gap", 0.0, xL[0]))

	for j in range(len(xL)):
		segments.append(("patch", xL[j], xR[j]))
		if j < len(xL)-1 and xR[j] < xL[j+1]:
			segments.append(("gap", xR[j], xL[j+1]))

	if xR[-1] < L:
		segments.append(("gap", xR[-1], L))

	# --------------------------------------------------
	# Mesh each segment independently
	# --------------------------------------------------
	x_nodes = [0.0]
	elem_EI = []
	elem_rhoA = []

	for seg_type, xa, xb in segments:
		Ls = xb - xa
		h = h_patch if seg_type == "patch" else h_gap
		n_el = max(1, int(np.ceil(Ls / h)))

		xs = np.linspace(xa, xb, n_el + 1)

		for k in range(n_el):
			x_nodes.append(xs[k+1])

			if seg_type == "patch":
				elem_EI.append(EI_patch)
				elem_rhoA.append(rhoA_patch)
			else:
				elem_EI.append(EI_gap)
				elem_rhoA.append(rhoA_gap)

	x_nodes = np.array(x_nodes)
	elem_EI = np.array(elem_EI)
	elem_rhoA = np.array(elem_rhoA)

	# --------------------------------------------------
	# Piezo descriptors (geometry only)
	# --------------------------------------------------
	piezos = []
	for j in range(len(xL)):
		piezos.append({
			"xL": xL[j],
			"xR": xR[j],
		})

	# --------------------------------------------------
	# Sanity: piezo edges must be exact nodes
	# --------------------------------------------------
	for x in np.concatenate([xL, xR]):
		if x not in x_nodes:
			raise RuntimeError(f"Piezo edge x={x} missing from mesh")

	return GeometrySpec(
		x_nodes=x_nodes,
		elem_EI=elem_EI,
		elem_rhoA=elem_rhoA,
		piezos=piezos
	)


def build_geometry_with_regions(
	L: float,
	regions: list,
	piezos: list,
	default_h: float = 1e-3
) -> GeometrySpec:
	"""
	Build geometry with arbitrary region types and properties.
	
	This function allows you to define a beam with multiple region types,
	each with custom material properties (EI, rhoA) and mesh density.
	Regions are specified explicitly, and the function automatically meshes
	and assembles them.
	
	Parameters
	----------
	L : float
		Total beam length [m]
	
	regions : list of dict
		List of region definitions. Each region is a dict with:
		- 'x_start': float - start position [m]
		- 'x_end': float - end position [m]
		- 'EI': float - bending stiffness [N·m²]
		- 'rhoA': float - mass per unit length [kg/m]
		- 'h': float (optional) - element size [m], defaults to default_h
		- 'name': str (optional) - region name for debugging
		
		Example:
		[
			{'x_start': 0.0, 'x_end': 0.05, 'EI': 1e-3, 'rhoA': 0.01, 'h': 2e-3, 'name': 'substrate'},
			{'x_start': 0.05, 'x_end': 0.10, 'EI': 5e-3, 'rhoA': 0.05, 'h': 1e-3, 'name': 'piezo'},
			{'x_start': 0.10, 'x_end': 0.20, 'EI': 2e-2, 'rhoA': 0.10, 'h': 3e-3, 'name': 'mass'}
		]
	
	piezos : list of dict
		List of piezo patch definitions. Each piezo is a dict with:
		- 'xL': float - left edge position [m]
		- 'xR': float - right edge position [m]
		
		Example:
		[{'xL': 0.05, 'xR': 0.10}, {'xL': 0.15, 'xR': 0.20}]
	
	default_h : float, optional
		Default element size if not specified in region [m]
	
	Returns
	-------
	GeometrySpec
		Geometry specification with nodes, element properties, and piezo locations
	
	Notes
	-----
	- Regions must not overlap
	- Regions should cover [0, L] (gaps will raise a warning)
	- Piezo edges must align with region boundaries or be interior to regions
	- Element sizes control mesh density independently per region
	
	Example
	-------
	>>> # Beam with substrate, heavy mass, and piezo regions
	>>> regions = [
	...     {'x_start': 0.0, 'x_end': 0.1, 'EI': 1e-3, 'rhoA': 0.01, 'name': 'substrate'},
	...     {'x_start': 0.1, 'x_end': 0.15, 'EI': 5e-2, 'rhoA': 0.5, 'h': 0.5e-3, 'name': 'mass'},
	...     {'x_start': 0.15, 'x_end': 0.25, 'EI': 3e-3, 'rhoA': 0.03, 'name': 'piezo'},
	...     {'x_start': 0.25, 'x_end': 0.4, 'EI': 1e-3, 'rhoA': 0.01, 'name': 'substrate'}
	... ]
	>>> piezos = [{'xL': 0.15, 'xR': 0.25}]
	>>> geom = build_geometry_with_regions(L=0.4, regions=regions, piezos=piezos)
	"""
	
	# --------------------------------------------------
	# Validate and sort regions
	# --------------------------------------------------
	regions = sorted(regions, key=lambda r: r['x_start'])
	
	# Check for overlaps and gaps
	for i in range(len(regions)-1):
		if regions[i]['x_end'] > regions[i+1]['x_start']:
			raise ValueError(
				f"Region overlap: region {i} ends at {regions[i]['x_end']}, "
				f"but region {i+1} starts at {regions[i+1]['x_start']}"
			)
		if regions[i]['x_end'] < regions[i+1]['x_start']:
			import warnings
			warnings.warn(
				f"Gap between regions: [{regions[i]['x_end']}, {regions[i+1]['x_start']}]"
			)
	
	# Check coverage
	if regions[0]['x_start'] > 0:
		import warnings
		warnings.warn(f"Gap at beam start: [0, {regions[0]['x_start']}]")
	if regions[-1]['x_end'] < L:
		import warnings
		warnings.warn(f"Gap at beam end: [{regions[-1]['x_end']}, {L}]")
	
	# --------------------------------------------------
	# Mesh each region independently
	# --------------------------------------------------
	x_nodes = [0.0]
	elem_EI = []
	elem_rhoA = []
	
	for reg in regions:
		xa = reg['x_start']
		xb = reg['x_end']
		EI = reg['EI']
		rhoA = reg['rhoA']
		h = reg.get('h', default_h)  # element size
		
		Ls = xb - xa
		n_el = max(1, int(np.ceil(Ls / h)))
		
		xs = np.linspace(xa, xb, n_el + 1)
		
		# Add nodes and elements
		for k in range(n_el):
			# Only add node if not duplicate (handle region boundaries)
			if xs[k+1] not in x_nodes or np.abs(xs[k+1] - x_nodes[-1]) > 1e-12:
				x_nodes.append(xs[k+1])
			
			elem_EI.append(EI)
			elem_rhoA.append(rhoA)
	
	x_nodes = np.array(x_nodes)
	elem_EI = np.array(elem_EI)
	elem_rhoA = np.array(elem_rhoA)
	
	# --------------------------------------------------
	# Validate piezo positions
	# --------------------------------------------------
	piezos_validated = []
	for p in piezos:
		xL = p['xL']
		xR = p['xR']
		
		# Check if piezo edges are at nodes (within tolerance)
		tol = 1e-12
		xL_found = any(np.abs(x_nodes - xL) < tol)
		xR_found = any(np.abs(x_nodes - xR) < tol)
		
		if not xL_found:
			raise RuntimeError(
				f"Piezo left edge xL={xL} not found in mesh nodes. "
				f"Ensure piezo edges align with region boundaries."
			)
		if not xR_found:
			raise RuntimeError(
				f"Piezo right edge xR={xR} not found in mesh nodes. "
				f"Ensure piezo edges align with region boundaries."
			)
		
		piezos_validated.append({'xL': xL, 'xR': xR})
	
	return GeometrySpec(
		x_nodes=x_nodes,
		elem_EI=elem_EI,
		elem_rhoA=elem_rhoA,
		piezos=piezos_validated
	)


def build_geometry_from_types(
	L: float,
	region_types: dict,
	region_sequence: list,
	x_starts: list,
	default_h: float = 1e-3
) -> GeometrySpec:
	"""
	Build beam geometry using a simplified region definition interface.
	
	Instead of specifying full region dictionaries for each segment, define
	region types once and reference them by name. Regions with type names
	containing "piezo" (case-insensitive) are automatically treated as piezos.
	
	Parameters
	----------
	L : float
		Total beam length.
	region_types : dict
		Dictionary mapping region type names to property dicts.
		Each property dict must contain:
		  - 'EI': bending stiffness
		  - 'rhoA': mass per unit length
		  - 'h': element size (optional, uses default_h if not specified)
		Example:
		  {
		    'substrate': {'EI': 1e-4, 'rhoA': 0.01, 'h': 5e-3},
		    'piezo': {'EI': 5e-3, 'rhoA': 0.05, 'h': 2e-3},
		    'mass': {'EI': 2e-2, 'rhoA': 0.5, 'h': 1e-3}
		  }
	region_sequence : list
		List of region type names (strings) defining the beam layout from left to right.
		Any region type name containing "piezo" (case-insensitive) will be treated as a piezo.
		Example: ['substrate', 'piezo', 'substrate', 'mass', 'piezo', 'substrate']
	x_starts : list or array
		Start positions for each region in region_sequence.
		Must have same length as region_sequence.
		Each region extends from x_starts[i] to x_starts[i+1] (or L for last region).
		Example: [0, 0.05, 0.08, 0.15, 0.20, 0.23]
	default_h : float, optional
		Default element size for regions that don't specify 'h'.
		Default is 1e-3.
	
	Returns
	-------
	GeometrySpec
		Named tuple with x_nodes, elem_EI, elem_rhoA, piezos.
	
	Raises
	------
	ValueError
		If region_sequence and x_starts have different lengths.
		If a region type name is not found in region_types.
	
	Example
	-------
	>>> region_types = {
	...     'substrate': {'EI': 1e-4, 'rhoA': 0.01, 'h': 5e-3},
	...     'piezo': {'EI': 5e-3, 'rhoA': 0.05, 'h': 2e-3},
	...     'mass': {'EI': 2e-2, 'rhoA': 0.5, 'h': 1e-3},
	...     'stiff': {'EI': 1e-1, 'rhoA': 0.1, 'h': 2e-3}
	... }
	>>> region_sequence = ['substrate', 'piezo', 'substrate', 'mass', 'stiff', 'piezo', 'substrate']
	>>> x_starts = [0, 5e-3, 30e-3, 100e-3, 125e-3, 200e-3, 225e-3]
	>>> geom = build_geometry_from_types(0.4, region_types, region_sequence, x_starts)
	>>> # Regions at indices 1 and 5 are automatically detected as piezos
	"""
	# Validate inputs
	if len(region_sequence) != len(x_starts):
		raise ValueError(
			f"region_sequence and x_starts must have same length. "
			f"Got {len(region_sequence)} and {len(x_starts)}."
		)
	
	# Check all region types exist
	for name in region_sequence:
		if name not in region_types:
			raise ValueError(
				f"Region type '{name}' not found in region_types dict. "
				f"Available types: {list(region_types.keys())}"
			)
	
	# Build regions list and automatically detect piezos
	regions = []
	piezos = []
	x_starts = np.asarray(x_starts)
	
	for i, type_name in enumerate(region_sequence):
		props = region_types[type_name]
		
		x_start = x_starts[i]
		# x_end is either next x_start or beam length
		x_end = x_starts[i+1] if i+1 < len(x_starts) else L
		
		region = {
			'x_start': x_start,
			'x_end': x_end,
			'EI': props['EI'],
			'rhoA': props['rhoA'],
			'h': props.get('h', default_h)
		}
		regions.append(region)
		
		# Automatically detect piezos by name
		if 'piezo' in type_name.lower():
			piezos.append({
				'xL': x_start,
				'xR': x_end
			})
	
	# Use build_geometry_with_regions to do the actual meshing
	return build_geometry_with_regions(L, regions, piezos, default_h)


def geometry_from_params(
	params: PiezoBeamParams,
	h_patch: float,
	h_gap: float
) -> GeometrySpec:
	"""
	Build a GeometrySpec equivalent to PiezoBeamParams geometry
	using build_geometry_arbitrary_piezos.
	"""

	# ---- beam length ----
	L: float = params.L_b

	# ---- piezo locations ----
	xL: np.ndarray = params.xL
	xR: np.ndarray = params.xR

	# ---- mass per unit length ----
	rhoA_patch: float = params.b * (
		params.rho_s * params.hs + 2.0 * params.rho_p * params.hp
	)
	rhoA_gap: float = params.b * params.rho_s * params.hs

	# ---- bending stiffness ----
	EI_patch: float = params.YI
	EI_gap: float = params.YI_s

	return build_geometry_arbitrary_piezos(
		L=L,
		xL=xL,
		xR=xR,
		EI_patch=EI_patch,
		rhoA_patch=rhoA_patch,
		EI_gap=EI_gap,
		rhoA_gap=rhoA_gap,
		h_patch=h_patch,
		h_gap=h_gap
	)

def build_linear_electrical_network(
		Nf: int,
		K_i: float | np.ndarray,
		K_i_nl: float,
		periodic: bool = False,
):
	"""
	Generate an energy-based electrical network with:
	- local linear inductors at each piezo (K_i)
	- nonlocal linear inductors between nearest neighbors (K_i_nl)

	Flux-based formulation: U = 1/2 * K * (Δφ)^2
	"""

	elements = []

	# -------------------------
	# Local inductors
	# -------------------------
	if np.isscalar(K_i):
		K_i = K_i * np.ones(Nf)

	for i in range(Nf):
		Ki = K_i[i]

		elements.append({
			"nodes": (i, None),
			"energy": lambda dphi, Ki=Ki: 0.5 * Ki * dphi**2,
			"grad":   lambda dphi, Ki=Ki: Ki * dphi,
			"hess":   lambda dphi, Ki=Ki: Ki,
		})

	# -------------------------
	# Nonlocal inductors
	# -------------------------
	for i in range(Nf - 1):
		elements.append({
			"nodes": (i, i + 1),
			"energy": lambda dphi, Knl=K_i_nl: 0.5 * Knl * dphi**2,
			"grad":   lambda dphi, Knl=K_i_nl: Knl * dphi,
			"hess":   lambda dphi, Knl=K_i_nl: Knl,
		})

	if periodic and Nf > 2:
		elements.append({
			"nodes": (Nf - 1, 0),
			"energy": lambda dphi, Knl=K_i_nl: 0.5 * Knl * dphi**2,
			"grad":   lambda dphi, Knl=K_i_nl: Knl * dphi,
			"hess":   lambda dphi, Knl=K_i_nl: Knl,
		})

	return {"elements": elements}