"""
PiezoBeamFE_Torch — differentiable Euler-Bernoulli piezo-beam FE model.

Assembly strategy
-----------------
The standard FE stiffness matrix is linear in the element bending stiffness:

    K = Σ_e  EI_e * K_hat[e]

where K_hat[e] is the *fixed* (non-differentiable) EI-normalised element
stiffness pattern matrix (depends only on element length Le, which is
determined by the mesh geometry — not by material parameters).

Similarly:

    M = Σ_e  rhoA_e * M_hat[e]
    Γ = θ_mech * Γ_hat           (Γ_hat encodes piezo node locations)

Because K_hat, M_hat, Γ_hat are fixed tensors (buffers) while EI_e, rhoA_e
and θ_mech are differentiable functions of hp, hs, E_s, …, the entire
K, M, Γ pipeline is part of the autograd computation graph.

Generalised eigenvalue problem
-------------------------------
torch.linalg.eigh solves only the *standard* symmetric eigenvalue problem.
We reduce  K v = λ M v  via a Cholesky transform:

    M = L Lᵀ             (Cholesky, positive-definite)
    A = L⁻¹ K L⁻ᵀ       (symmetric, well-conditioned)
    A u = λ u            → torch.linalg.eigh(A)
    v = L⁻ᵀ u            (back-transform)

All three steps (solve_triangular, eigh, back-transform) are differentiable
in PyTorch ≥ 2.0, so ∂ωᵢ/∂hp etc. can be computed via .backward().

Parameters that ARE differentiable
------------------------------------
  hp, hs, b, E_s, s11, d31, eps0, eps_r, rho_s, rho_p
  (anything that changes element EI / rhoA or θ_mech)

Parameters that are NOT differentiable (fixed mesh topology)
-------------------------------------------------------------
  w_p, w_s, n_patches
  Changing these would alter node positions and therefore mesh connectivity;
  a new PiezoBeamFE_Torch instance must be constructed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Modeling.models_torch.beam_params_torch import PiezoBeamParamsTorch


# ─────────────────────────────────────────────────────────────────────────────
# Custom differentiable generalised eigensolver
# ─────────────────────────────────────────────────────────────────────────────

class _GenEighFn(torch.autograd.Function):
    """
    Differentiable generalised symmetric eigenvalue problem  K v = λ M v.

    Forward:  scipy.linalg.eigh (accurate, mass-normalised eigenvectors).
    Backward: Rayleigh-quotient adjoint:

        ∂L/∂K =  V  diag(∂L/∂λ)        V^T
        ∂L/∂M = −V  diag(λ ⊙ ∂L/∂λ)  V^T

    where V are mass-normalised columns (V^T M V = I).
    Only the first n_out eigenpairs are computed via subset_by_index.
    """

    @staticmethod
    def forward(ctx, K, M, n_out):
        from scipy.linalg import eigh
        K_np = K.detach().cpu().double().numpy()
        M_np = M.detach().cpu().double().numpy()
        N    = K_np.shape[0]
        n_c  = min(n_out, N)
        eigvals_np, eigvecs_np = eigh(
            K_np, M_np, subset_by_index=[0, n_c - 1],
        )
        eigvals = torch.from_numpy(eigvals_np.copy()).to(K)
        eigvecs = torch.from_numpy(eigvecs_np.copy()).to(K)
        ctx.save_for_backward(eigvals, eigvecs)
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, grad_eigvals, grad_eigvecs):
        eigvals, V = ctx.saved_tensors   # V: (Nfree, n_out), mass-normalised

        # ∂L/∂K = V diag(g_λ) V^T
        g_lam = grad_eigvals                               # (n_out,)
        grad_K = V @ torch.diag(g_lam) @ V.T              # (Nfree, Nfree)

        # ∂L/∂M = -V diag(λ ⊙ g_λ) V^T
        grad_M = -(V @ torch.diag(eigvals * g_lam) @ V.T)

        return grad_K, grad_M, None   # None for n_out (non-tensor arg)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh description (pure numpy, built once, never differentiated)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _MeshGeometry:
    x_nodes:           np.ndarray   # node x-coordinates  [Nnodes]
    elem_is_patch:     np.ndarray   # bool mask, shape (n_elem,)
    piezo_node_pairs:  List[Tuple[int, int]]  # (kL, kR) per piezo, len=S
    free_dofs:         np.ndarray   # free DOF indices (BC applied)
    fixed_dofs:        List[int]    # fixed DOF indices


@dataclass
class GeometrySpecTorch:
    """
    Arbitrary region geometry container for PiezoBeamFE_Torch.

    Attributes
    ----------
    x_nodes : np.ndarray
        Global node coordinates (size Nnodes).
    elem_EI : np.ndarray
        Per-element bending stiffness EI (size Nnodes-1).
    elem_rhoA : np.ndarray
        Per-element mass per unit length rhoA (size Nnodes-1).
    piezos : list of dict
        Piezo descriptors with keys {'xL', 'xR'}.
    """

    x_nodes: np.ndarray
    elem_EI: np.ndarray
    elem_rhoA: np.ndarray
    piezos: List[Dict[str, float]]


def build_geometry_arbitrary_piezos(
    L: float,
    xL: np.ndarray,
    xR: np.ndarray,
    EI_patch: float,
    rhoA_patch: float,
    EI_gap: float,
    rhoA_gap: float,
    h_patch: float,
    h_gap: float,
) -> GeometrySpecTorch:
    """Build piecewise patch/gap geometry with independent mesh densities."""
    xL = np.asarray(xL, dtype=float)
    xR = np.asarray(xR, dtype=float)

    if len(xL) != len(xR):
        raise ValueError("xL and xR must have the same length.")
    if not np.all(xL < xR):
        raise ValueError("Every piezo must satisfy xL < xR.")
    if not np.all(xL >= 0.0) or not np.all(xR <= L):
        raise ValueError("Piezo edges must lie inside [0, L].")

    segments: List[Tuple[str, float, float]] = []
    if xL[0] > 0.0:
        segments.append(("gap", 0.0, xL[0]))

    for j in range(len(xL)):
        segments.append(("patch", float(xL[j]), float(xR[j])))
        if j < len(xL) - 1 and xR[j] < xL[j + 1]:
            segments.append(("gap", float(xR[j]), float(xL[j + 1])))

    if xR[-1] < L:
        segments.append(("gap", float(xR[-1]), float(L)))

    x_nodes = [0.0]
    elem_EI = []
    elem_rhoA = []

    for seg_type, xa, xb in segments:
        Ls = xb - xa
        h = h_patch if seg_type == "patch" else h_gap
        n_el = max(1, int(np.ceil(Ls / h)))
        xs = np.linspace(xa, xb, n_el + 1)

        for k in range(n_el):
            x_nodes.append(float(xs[k + 1]))
            if seg_type == "patch":
                elem_EI.append(float(EI_patch))
                elem_rhoA.append(float(rhoA_patch))
            else:
                elem_EI.append(float(EI_gap))
                elem_rhoA.append(float(rhoA_gap))

    x_nodes = np.asarray(x_nodes, dtype=float)
    elem_EI = np.asarray(elem_EI, dtype=float)
    elem_rhoA = np.asarray(elem_rhoA, dtype=float)

    piezos = [{"xL": float(a), "xR": float(b)} for a, b in zip(xL, xR)]

    tol = 1e-12
    for x in np.concatenate([xL, xR]):
        if not np.any(np.abs(x_nodes - x) < tol):
            raise RuntimeError(
                f"Piezo edge x={x} missing from mesh. "
                "Choose h_patch/h_gap so piezo edges are included."
            )

    return GeometrySpecTorch(
        x_nodes=x_nodes,
        elem_EI=elem_EI,
        elem_rhoA=elem_rhoA,
        piezos=piezos,
    )


def build_geometry_with_regions(
    L: float,
    regions: List[Dict[str, float]],
    piezos: List[Dict[str, float]],
    default_h: float = 1e-3,
) -> GeometrySpecTorch:
    """Build geometry from explicit region dictionaries."""
    if len(regions) == 0:
        raise ValueError("regions cannot be empty.")

    regions = sorted(regions, key=lambda r: r["x_start"])

    for i in range(len(regions) - 1):
        if regions[i]["x_end"] > regions[i + 1]["x_start"]:
            raise ValueError(
                f"Region overlap between indices {i} and {i+1}: "
                f"{regions[i]['x_end']} > {regions[i + 1]['x_start']}"
            )

    x_nodes = [0.0]
    elem_EI = []
    elem_rhoA = []

    for reg in regions:
        xa = float(reg["x_start"])
        xb = float(reg["x_end"])
        EI = float(reg["EI"])
        rhoA = float(reg["rhoA"])
        h = float(reg.get("h", default_h))

        if xb <= xa:
            raise ValueError(f"Region has non-positive length: [{xa}, {xb}]")

        n_el = max(1, int(np.ceil((xb - xa) / h)))
        xs = np.linspace(xa, xb, n_el + 1)

        for k in range(n_el):
            x_next = float(xs[k + 1])
            if abs(x_next - x_nodes[-1]) > 1e-12:
                x_nodes.append(x_next)
            elem_EI.append(EI)
            elem_rhoA.append(rhoA)

    x_nodes = np.asarray(x_nodes, dtype=float)
    elem_EI = np.asarray(elem_EI, dtype=float)
    elem_rhoA = np.asarray(elem_rhoA, dtype=float)

    piezos_validated: List[Dict[str, float]] = []
    tol = 1e-12
    for pz in piezos:
        xL = float(pz["xL"])
        xR = float(pz["xR"])
        if not np.any(np.abs(x_nodes - xL) < tol):
            raise RuntimeError(
                f"Piezo left edge xL={xL} not found in mesh nodes. "
                "Ensure piezo edges align with region boundaries."
            )
        if not np.any(np.abs(x_nodes - xR) < tol):
            raise RuntimeError(
                f"Piezo right edge xR={xR} not found in mesh nodes. "
                "Ensure piezo edges align with region boundaries."
            )
        piezos_validated.append({"xL": xL, "xR": xR})

    return GeometrySpecTorch(
        x_nodes=x_nodes,
        elem_EI=elem_EI,
        elem_rhoA=elem_rhoA,
        piezos=piezos_validated,
    )


def build_geometry_from_types(
    L: float,
    region_types: Dict[str, Dict[str, float]],
    region_sequence: List[str],
    x_starts: List[float],
    default_h: float = 1e-3,
) -> GeometrySpecTorch:
    """
    Build arbitrary geometry from named region types.

    Region names containing "piezo" are auto-detected as piezo spans.
    """
    if len(region_sequence) != len(x_starts):
        raise ValueError(
            "region_sequence and x_starts must have same length. "
            f"Got {len(region_sequence)} and {len(x_starts)}."
        )

    for name in region_sequence:
        if name not in region_types:
            raise ValueError(
                f"Region type '{name}' not found in region_types. "
                f"Available types: {list(region_types.keys())}"
            )

    x_starts_np = np.asarray(x_starts, dtype=float)
    regions: List[Dict[str, float]] = []
    piezos: List[Dict[str, float]] = []

    for i, type_name in enumerate(region_sequence):
        props = region_types[type_name]
        x_start = float(x_starts_np[i])
        x_end = float(x_starts_np[i + 1] if i + 1 < len(x_starts_np) else L)

        region = {
            "x_start": x_start,
            "x_end": x_end,
            "EI": float(props["EI"]),
            "rhoA": float(props["rhoA"]),
            "h": float(props.get("h", default_h)),
        }
        regions.append(region)

        if "piezo" in type_name.lower():
            piezos.append({"xL": x_start, "xR": x_end})

    return build_geometry_with_regions(L=L, regions=regions, piezos=piezos, default_h=default_h)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PiezoBeamFE_Torch(nn.Module):
    """
    Differentiable Euler-Bernoulli piezo-beam finite element model.

    Parameters
    ----------
    tp : PiezoBeamParamsTorch
        Differentiable material / cross-section parameters.
    n_el_patch : int
        Number of beam elements per piezo patch (default 3).
    n_el_gap : int
        Number of beam elements per inter-patch gap  (default 2).

    Attributes (buffers, non-differentiable)
    ----------------------------------------
    K_hat     (n_elem, Ndof, Ndof)   EI-normalised element stiffness patterns
    M_hat     (n_elem, Ndof, Ndof)   rhoA-normalised element mass patterns
    Gamma_hat (Ndof, S)              ±1 coupling pattern
    is_patch  (n_elem,) bool         True → patch element

    Typical workflow
    ----------------
    >>> tp = PiezoBeamParamsTorch()
    >>> fe = PiezoBeamFE_Torch(tp)
    >>> out = fe()                         # full forward pass
    >>> out['freq'][:5]                    # first 5 natural frequencies [Hz]
    >>> loss = out['freq'][0]              # e.g. optimise first mode
    >>> loss.backward()
    >>> tp.hp.grad                         # dω₁/d(hp)
    """

    def __init__(
        self,
        tp: PiezoBeamParamsTorch,
        geometry: Optional[GeometrySpecTorch] = None,
        elem_property_fn: Optional[Callable[[PiezoBeamParamsTorch], Tuple[torch.Tensor, torch.Tensor]]] = None,
        n_el_patch: int = 3,
        n_el_gap:   int = 2,
    ) -> None:
        super().__init__()
        self.tp         = tp
        self.geometry   = geometry
        self.elem_property_fn = elem_property_fn
        self.n_el_patch = n_el_patch
        self.n_el_gap   = n_el_gap
        self._use_explicit_elem_props = geometry is not None
        self._use_callable_elem_props = (
            self._use_explicit_elem_props and self.elem_property_fn is not None
        )

        # Build fixed mesh geometry (no grad)
        if self._use_explicit_elem_props:
            self._mesh = self._build_mesh_from_geometry(geometry)
            if not self._use_callable_elem_props:
                self.register_buffer(
                    "_elem_EI_explicit",
                    torch.tensor(np.asarray(geometry.elem_EI, dtype=float), dtype=self.tp.dtype),
                )
                self.register_buffer(
                    "_elem_rhoA_explicit",
                    torch.tensor(np.asarray(geometry.elem_rhoA, dtype=float), dtype=self.tp.dtype),
                )
        else:
            self._mesh = self._build_mesh()

        # Precompute fixed pattern matrices → register as buffers
        self._precompute_patterns()

        # Store free-DOF index tensor as a buffer for fast repeated indexing
        self.register_buffer(
            "_free_dofs_t",
            torch.tensor(self._mesh.free_dofs, dtype=torch.long),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Mesh construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_mesh(self) -> _MeshGeometry:
        tp   = self.tp
        xL   = tp.xL_np()
        xR   = tp.xR_np()
        L_b  = tp.L_b()
        n_p  = tp.n_patches

        # ── segment list ─────────────────────────────────────────────────
        segments: List[Tuple[str, float, float]] = []
        if xL[0] > 1e-12:
            segments.append(("gap", 0.0, xL[0]))
        for j in range(n_p):
            segments.append(("patch", xL[j], xR[j]))
            if j < n_p - 1 and xR[j] < xL[j + 1] - 1e-12:
                segments.append(("gap", xR[j], xL[j + 1]))
        if xR[-1] < L_b - 1e-12:
            segments.append(("gap", xR[-1], L_b))

        # ── mesh nodes and element type flags ─────────────────────────────
        x_nodes: List[float]  = [0.0]
        elem_is_patch: List[bool] = []
        for seg_type, xa, xb in segments:
            n  = self.n_el_patch if seg_type == "patch" else self.n_el_gap
            xs = np.linspace(xa, xb, n + 1)
            for _ in range(n):
                elem_is_patch.append(seg_type == "patch")
            x_nodes.extend(xs[1:].tolist())

        x_nodes_arr = np.array(x_nodes)

        # ── piezo boundary node look-up ────────────────────────────────────
        # Round to femtometre precision to avoid floating-point mismatches
        _prec = 1e-15
        node_index = {round(x / _prec): i for i, x in enumerate(x_nodes_arr)}

        piezo_node_pairs: List[Tuple[int, int]] = []
        for j in range(tp.S):
            kL_key = round(float(xL[j]) / _prec)
            kR_key = round(float(xR[j]) / _prec)
            if kL_key not in node_index:
                raise ValueError(
                    f"Piezo {j}: xL={xL[j]:.6e} not found among mesh nodes. "
                    "Check that n_el_patch / n_el_gap divide the segments evenly."
                )
            if kR_key not in node_index:
                raise ValueError(
                    f"Piezo {j}: xR={xR[j]:.6e} not found among mesh nodes."
                )
            piezo_node_pairs.append((node_index[kL_key], node_index[kR_key]))

        # ── boundary conditions (cantilever: DOFs 0 and 1 fixed) ──────────
        Ndof      = 2 * len(x_nodes_arr)
        fixed_dofs = [0, 1]
        free_dofs  = np.setdiff1d(np.arange(Ndof), fixed_dofs)

        return _MeshGeometry(
            x_nodes          = x_nodes_arr,
            elem_is_patch    = np.array(elem_is_patch, dtype=bool),
            piezo_node_pairs = piezo_node_pairs,
            free_dofs        = free_dofs,
            fixed_dofs       = fixed_dofs,
        )

    def _build_mesh_from_geometry(self, geometry: GeometrySpecTorch) -> _MeshGeometry:
        """Build FE mesh metadata from user-provided arbitrary region geometry."""
        x_nodes_arr = np.asarray(geometry.x_nodes, dtype=float)
        n_elem = len(x_nodes_arr) - 1

        if n_elem <= 0:
            raise ValueError("geometry.x_nodes must contain at least two nodes.")
        if len(geometry.elem_EI) != n_elem or len(geometry.elem_rhoA) != n_elem:
            raise ValueError(
                "geometry.elem_EI and geometry.elem_rhoA must each have length len(x_nodes)-1."
            )

        # Round to femtometre precision to avoid floating-point mismatches.
        _prec = 1e-15
        node_index = {round(float(x) / _prec): i for i, x in enumerate(x_nodes_arr)}

        piezo_node_pairs: List[Tuple[int, int]] = []
        for j, pz in enumerate(geometry.piezos):
            xL = float(pz["xL"])
            xR = float(pz["xR"])
            kL_key = round(xL / _prec)
            kR_key = round(xR / _prec)
            if kL_key not in node_index:
                raise ValueError(f"Piezo {j}: xL={xL:.6e} not found among mesh nodes.")
            if kR_key not in node_index:
                raise ValueError(f"Piezo {j}: xR={xR:.6e} not found among mesh nodes.")
            piezo_node_pairs.append((node_index[kL_key], node_index[kR_key]))

        Ndof = 2 * len(x_nodes_arr)
        fixed_dofs = [0, 1]
        free_dofs = np.setdiff1d(np.arange(Ndof), fixed_dofs)

        return _MeshGeometry(
            x_nodes=x_nodes_arr,
            elem_is_patch=np.zeros(n_elem, dtype=bool),
            piezo_node_pairs=piezo_node_pairs,
            free_dofs=free_dofs,
            fixed_dofs=fixed_dofs,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Pre-compute fixed pattern matrices (registered as buffers)
    # ─────────────────────────────────────────────────────────────────────

    def _precompute_patterns(self) -> None:
        """
        Build and register three fixed (non-differentiable) buffers:

          K_hat[e]     — element stiffness pattern,  normalised by 1/EI
          M_hat[e]     — element mass pattern,        normalised by 1/rhoA
          Gamma_hat    — coupling sign pattern (±1 at piezo boundary nodes)

        Then:
            K     = (EI_vec   [:, None, None] * K_hat).sum(0)
            M     = (rhoA_vec [:, None, None] * M_hat).sum(0)
            Gamma = theta_mech * Gamma_hat
        """
        mesh   = self._mesh
        dtype  = self.tp.dtype
        x_nodes      = mesh.x_nodes
        elem_is_patch = mesh.elem_is_patch
        n_elem        = len(elem_is_patch)
        Nnodes        = len(x_nodes)
        Ndof          = 2 * Nnodes
        S             = len(mesh.piezo_node_pairs)

        K_hat_np = np.zeros((n_elem, Ndof, Ndof))
        M_hat_np = np.zeros((n_elem, Ndof, Ndof))

        for e in range(n_elem):
            Le    = float(x_nodes[e + 1] - x_nodes[e])
            dofs  = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            ix    = np.ix_(dofs, dofs)

            # EI-normalised stiffness pattern (dimensionless shape × 1/Le³)
            K_hat_np[e][ix] = np.array([
                [ 12,      6*Le,    -12,      6*Le   ],
                [  6*Le,   4*Le**2,  -6*Le,   2*Le**2],
                [-12,     -6*Le,     12,     -6*Le   ],
                [  6*Le,   2*Le**2,  -6*Le,   4*Le**2],
            ]) / Le**3

            # rhoA-normalised mass pattern (dimensionless shape × Le)
            M_hat_np[e][ix] = (Le / 420.0) * np.array([
                [ 156,     22*Le,    54,    -13*Le   ],
                [  22*Le,   4*Le**2, 13*Le,  -3*Le**2],
                [  54,     13*Le,  156,    -22*Le   ],
                [ -13*Le,  -3*Le**2, -22*Le,  4*Le**2],
            ])

        # Gamma coupling sign pattern
        Gamma_hat_np = np.zeros((Ndof, S))
        for j, (kL, kR) in enumerate(mesh.piezo_node_pairs):
            Gamma_hat_np[2*kR + 1, j] += +1.0   # slope DOF at right end
            Gamma_hat_np[2*kL + 1, j] += -1.0   # slope DOF at left end

        # is_patch mask (bool → used in torch.where during assembly)
        self.register_buffer(
            "K_hat",
            torch.tensor(K_hat_np, dtype=dtype),
        )
        self.register_buffer(
            "M_hat",
            torch.tensor(M_hat_np, dtype=dtype),
        )
        self.register_buffer(
            "Gamma_hat",
            torch.tensor(Gamma_hat_np, dtype=dtype),
        )
        self.register_buffer(
            "is_patch",
            torch.tensor(elem_is_patch, dtype=torch.bool),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Differentiable assembly
    # ─────────────────────────────────────────────────────────────────────

    def assemble_KM(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble global stiffness K and mass M (full DOF, before BC).

        K = Σ_e  EI_e   * K_hat[e]
        M = Σ_e  rhoA_e * M_hat[e]

        Gradients flow through EI_e and rhoA_e back to hp, hs, E_s, etc.
        """
        if self._use_explicit_elem_props:
            if self._use_callable_elem_props:
                EI_vec, rhoA_vec = self.elem_property_fn(self.tp)
                if EI_vec.ndim != 1 or rhoA_vec.ndim != 1:
                    raise ValueError("elem_property_fn must return two 1D tensors.")
                if EI_vec.shape[0] != self.K_hat.shape[0] or rhoA_vec.shape[0] != self.K_hat.shape[0]:
                    raise ValueError(
                        "elem_property_fn returned wrong vector length. "
                        f"Expected {self.K_hat.shape[0]} elements."
                    )
            else:
                EI_vec = self._elem_EI_explicit
                rhoA_vec = self._elem_rhoA_explicit
        else:
            tp = self.tp
            EI_patch   = tp.YI()         # scalar tensor
            EI_gap     = tp.YI_s()       # scalar tensor
            rhoA_patch = tp.rhoA_patch() # scalar tensor
            rhoA_gap   = tp.rhoA_gap()   # scalar tensor

            # Per-element vectors: choose patch or gap value, shape (n_elem,)
            # torch.where broadcasts scalar tensors onto the bool mask correctly.
            EI_vec   = torch.where(self.is_patch, EI_patch,   EI_gap)
            rhoA_vec = torch.where(self.is_patch, rhoA_patch, rhoA_gap)

        # Vectorised assembly: (n_elem,1,1) * (n_elem,Ndof,Ndof) → sum
        K = (EI_vec  [:, None, None] * self.K_hat).sum(0)
        M = (rhoA_vec[:, None, None] * self.M_hat).sum(0)
        return K, M

    def assemble_Gamma(self) -> torch.Tensor:
        """
        Assemble coupling matrix Γ = θ_mech * Γ_hat.

        Gradient flows through theta_mech → d31, hs, hp, b, s11.
        """
        return self.tp.theta_mech() * self.Gamma_hat

    # ─────────────────────────────────────────────────────────────────────
    # Boundary conditions
    # ─────────────────────────────────────────────────────────────────────

    def apply_bc(
        self,
        K:     torch.Tensor,
        M:     torch.Tensor,
        Gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract free-DOF sub-matrices (cantilever BC: DOFs 0,1 fixed)."""
        f = self._free_dofs_t
        K_red     = K    [f[:, None], f[None, :]]
        M_red     = M    [f[:, None], f[None, :]]
        Gamma_red = Gamma[f, :]
        return K_red, M_red, Gamma_red

    # ─────────────────────────────────────────────────────────────────────
    # Differentiable generalised eigenvalue problem  K v = λ M v
    # ─────────────────────────────────────────────────────────────────────

    def eigen_analysis(
        self,
        K_red:   torch.Tensor,
        M_red:   torch.Tensor,
        n_modes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve  K_red v = λ M_red v  via _GenEighFn (custom autograd).

        Forward : scipy.linalg.eigh (accurate, mass-normalised eigenvectors).
        Backward: Rayleigh-quotient adjoint (analytically exact):

            ∂L/∂K =  V diag(∂L/∂λ) V^T
            ∂L/∂M = -V diag(λ * ∂L/∂λ) V^T

        Returns
        -------
        omega   : (n_modes,)         natural frequencies [rad/s]
        freq    : (n_modes,)         natural frequencies [Hz]
        eigvecs : (Nfree, n_modes)   mass-normalised mode shapes
        """
        Nfree = K_red.shape[0]
        n_out = n_modes if n_modes is not None else Nfree
        eigvals, eigvecs = _GenEighFn.apply(K_red, M_red, n_out)
        omega = torch.sqrt(torch.clamp(eigvals, min=0.0))
        freq  = omega / (2.0 * torch.pi)
        return omega, freq, eigvecs

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: full pipeline
    # ─────────────────────────────────────────────────────────────────────

    def build_KM_Gamma(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble K, M, Γ and apply cantilever BC. All outputs are differentiable."""
        K, M  = self.assemble_KM()
        Gamma = self.assemble_Gamma()
        return self.apply_bc(K, M, Gamma)

    def forward(
        self,
        n_modes: Optional[int] = None,
    ) -> dict:
        """
        Full differentiable forward pass:
        assemble → apply BC → solve generalised eigenvalue problem.

        Parameters
        ----------
        n_modes : int or None
            If given, return only the first n_modes eigenfrequencies/vectors.

        Returns
        -------
        dict with keys:
            omega      (n_modes,)         [rad/s]
            freq       (n_modes,)         [Hz]
            eigvecs    (Nfree, n_modes)   mass-normalised mode shapes
            K_red      (Nfree, Nfree)     reduced stiffness matrix
            M_red      (Nfree, Nfree)     reduced mass matrix
            Gamma_red  (Nfree, S)         reduced coupling matrix
        """
        K_red, M_red, Gamma_red = self.build_KM_Gamma()
        omega, freq, eigvecs    = self.eigen_analysis(K_red, M_red, n_modes=n_modes)
        return dict(
            omega     = omega,
            freq      = freq,
            eigvecs   = eigvecs,
            K_red     = K_red,
            M_red     = M_red,
            Gamma_red = Gamma_red,
        )
