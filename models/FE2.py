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


# ============================================================
# FE MODEL
# ============================================================
import numpy as np
from dataclasses import dataclass
from scipy.linalg import eigh

# ============================================================
# Geometry container (NEW)
# ============================================================

@dataclass
class GeometrySpec:
	x_nodes: np.ndarray          # nodal coordinates
	elem_EI: np.ndarray          # bending stiffness per element
	elem_rhoA: np.ndarray        # mass per length per element
	Gamma: np.ndarray            # electromechanical coupling (full DOF)


# ============================================================
# ODE system container (UNCHANGED)
# ============================================================

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


# ============================================================
# FE MODEL
# ============================================================

class PiezoBeamFE:
	def __init__(self, params, n_el_patch=3, n_el_gap=2):
		self.params = params
		self.n_el_patch = n_el_patch
		self.n_el_gap   = n_el_gap

		# ---- geometry selection (NEW) ----
		if hasattr(self.params, "geometry"):
			self.geom = self.params.geometry
		else:
			self.geom = self._build_default_geometry()

		# ---- assembly ----
		self._assemble_KM()
		self._apply_bc()

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

		# ---- electromechanical coupling ----
		Ndof = 2 * Nnodes
		Gamma = np.zeros((Ndof, n_patches))
		node_index = {x: i for i, x in enumerate(x_nodes)}

		for j in range(n_patches):
			kL = node_index[p.xL[j]]
			kR = node_index[p.xR[j]]
			Gamma[2*kR + 1, j] +=  p.theta_mech
			Gamma[2*kL + 1, j] += -p.theta_mech

		return GeometrySpec(
			x_nodes=x_nodes,
			elem_EI=elem_EI,
			elem_rhoA=elem_rhoA,
			Gamma=Gamma
		)

	# ========================================================
	# Assembly
	# ========================================================

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

	# ========================================================
	# Boundary conditions
	# ========================================================

	def _apply_bc(self):
		fixed_dofs = [0, 1]
		all_dofs = np.arange(self.Ndof)
		self.free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

		self.K_red = self.K[np.ix_(self.free_dofs, self.free_dofs)]
		self.M_red = self.M[np.ix_(self.free_dofs, self.free_dofs)]
		self.Gamma_red = self.geom.Gamma[self.free_dofs, :]

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
		return freq, omega, Phi

	# ========================================================
	# ODE construction (UNCHANGED)
	# ========================================================

	def build_ode_system(
		self,
		j_exc=0,
		R_c=1e3,
		K_p=0.02,
		K_i=0.0,
		K_c=0.0,
		v_exc=lambda t: 0.0
	):
		N = self.M_red.shape[0]
		S = self.Gamma_red.shape[1]

		idx_all = np.arange(S)
		idx_f = np.delete(idx_all, j_exc)

		Gamma_f = self.Gamma_red[:, idx_f]
		Gamma_e = self.Gamma_red[:, j_exc]

		if np.isscalar(K_i):
			K_i = K_i * np.ones(len(idx_f))
		else:
			K_i = np.delete(K_i, j_exc)

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

		def f_ext(t):
			return np.concatenate([Gamma_e * v_exc(t), np.zeros(len(idx_f))])

		f_ext_unit = np.concatenate([Gamma_e, np.zeros(len(idx_f))])

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
