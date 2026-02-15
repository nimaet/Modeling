import numpy as np
from dataclasses import dataclass


@dataclass
class GeometrySpec:
	"""
	Pure geometry container.
	NO FE matrices here.
	"""
	x_nodes: np.ndarray
	elem_EI: np.ndarray
	elem_rhoA: np.ndarray
	piezos: list   # ← REQUIRED by FE2

def build_geometry_arbitrary_piezos(
	L: float,
	xL: np.ndarray,
	xR: np.ndarray,
	EI_patch: float,
	rhoA_patch: float,
	EI_gap: float,
	rhoA_gap: float,
	h_patch: float,
	h_gap: float
):

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
