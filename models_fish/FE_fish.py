"""Euler-Bernoulli finite-element model for piezoelectric patch beams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
from scipy.linalg import eigh

try:  # package import
    from Modeling.models_fish.beam_properties_fish import PiezoBeamParams
except Exception:  # local / notebook fallback
    from beam_properties_fish import PiezoBeamParams


@dataclass
class GeometrySpec:
    """Pure geometry/material description used by ``PiezoBeamFE``."""

    x_nodes: np.ndarray
    elem_EI: np.ndarray
    elem_rhoA: np.ndarray
    piezos: list  # each item has at least {'xL': float, 'xR': float}


@dataclass(frozen=True)
class PiezoBeamODESystem:
    """Container for coupled mechanical/electrical ODE systems."""
    
    M: np.ndarray
    M_mech: np.ndarray
    K_mech: np.ndarray
    C: np.ndarray
    D: np.ndarray
    f_ext_freq_domain: np.ndarray
    f_int: Callable
    K_tan: Callable
    f_ext: Callable
    v_exc: Callable
    N_mech: int
    N_elec: int


class PiezoBeamFE:
    """2-node Hermite Euler-Bernoulli FE model with piezo moment-pair forcing."""

    def __init__(self, params: PiezoBeamParams, n_el_patch: int = 3, n_el_gap: int = 2):
        self.params = params
        self.n_el_patch = int(n_el_patch)
        self.n_el_gap = int(n_el_gap)

        self.geom = getattr(self.params, "geometry", None)
        if self.geom is None:
            self.geom = geometry_from_params(self.params, self.n_el_patch, self.n_el_gap)
        elif hasattr(self.params, "sync_patch_count"):
            self.params.sync_patch_count(len(self.geom.piezos))

        self._build_Gamma()
        self._assemble_KM()
        self._apply_bc()
        self.eigen_analysis()
        self.damping_matrix_from_modal_damping()

    # ------------------------------------------------------------------
    # Assembly helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_node_index(x_nodes: np.ndarray, x: float, tol: float = 1e-10) -> int:
        idx = np.where(np.abs(x_nodes - x) <= tol)[0]
        if len(idx) != 1:
            raise ValueError(f"Could not identify a unique mesh node at x={x:.16g}; matches={idx}")
        return int(idx[0])

    def _build_Gamma(self) -> None:
        x_nodes = np.asarray(self.geom.x_nodes, dtype=float)
        piezos = self.geom.piezos
        ndof = 2 * len(x_nodes)
        gamma = np.zeros((ndof, len(piezos)))

        for j, pz in enumerate(piezos):
            kL = self._find_node_index(x_nodes, float(pz["xL"]))
            kR = self._find_node_index(x_nodes, float(pz["xR"]))
            theta = float(pz.get("theta", self.params.theta_mech))
            gamma[2 * kL + 1, j] += -theta
            gamma[2 * kR + 1, j] += theta

        self.Gamma = gamma

    def _assemble_KM(self) -> None:
        x_nodes = np.asarray(self.geom.x_nodes, dtype=float)
        elem_EI = np.asarray(self.geom.elem_EI, dtype=float)
        elem_rhoA = np.asarray(self.geom.elem_rhoA, dtype=float)

        if len(x_nodes) != len(elem_EI) + 1:
            raise ValueError("Expected len(x_nodes) == len(elem_EI) + 1")
        if len(elem_EI) != len(elem_rhoA):
            raise ValueError("elem_EI and elem_rhoA must have the same length")

        n_nodes = len(x_nodes)
        self.Ndof = 2 * n_nodes
        K = np.zeros((self.Ndof, self.Ndof))
        M = np.zeros((self.Ndof, self.Ndof))

        for e, (EI, rhoA) in enumerate(zip(elem_EI, elem_rhoA)):
            i, j = e, e + 1
            Le = x_nodes[j] - x_nodes[i]
            if Le <= 0:
                raise ValueError(f"Non-positive element length at element {e}: {Le}")

            Ke = (EI / Le**3) * np.array(
                [
                    [12, 6 * Le, -12, 6 * Le],
                    [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                    [-12, -6 * Le, 12, -6 * Le],
                    [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
                ],
                dtype=float,
            )

            Me = (rhoA * Le / 420.0) * np.array(
                [
                    [156, 22 * Le, 54, -13 * Le],
                    [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
                    [54, 13 * Le, 156, -22 * Le],
                    [-13 * Le, -3 * Le**2, -22 * Le, 4 * Le**2],
                ],
                dtype=float,
            )

            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(dofs, dofs)] += Ke
            M[np.ix_(dofs, dofs)] += Me

        self.K = K
        self.M = M

    def _apply_bc(self) -> None:
        all_dofs = np.arange(self.Ndof)
        self.fixed_dofs = np.array([0, 1], dtype=int)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)

        self.K_red = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        self.M_red = self.M[np.ix_(self.free_dofs, self.free_dofs)]
        self.Gamma_red = self.Gamma[self.free_dofs, :]

    # ------------------------------------------------------------------
    # Modal analysis and damping
    # ------------------------------------------------------------------
    def eigen_analysis(self):
        eigvals, eigvecs = eigh(self.K_red, self.M_red)
        eigvals = np.maximum(eigvals, 0.0)
        omega = np.sqrt(eigvals)
        freq = omega / (2 * np.pi)

        idx = np.argsort(freq)
        omega = omega[idx]
        freq = freq[idx]
        eigvecs = eigvecs[:, idx]

        Phi = np.zeros((self.Ndof, len(freq)))
        for i in range(len(freq)):
            Phi[self.free_dofs, i] = eigvecs[:, i]
            norm = np.sqrt(Phi[:, i].T @ self.M @ Phi[:, i])
            if norm > 0:
                Phi[:, i] /= norm

        self.freq = freq
        self.omega = omega
        self.Phi = Phi
        self.eigvecs = eigvecs
        return freq, omega, Phi

    def damping_matrix_from_modal_damping(self):
        zeta_rest = self.params.zeta_dict.get("rest", 0.0)
        self.zeta = np.array(
            [self.params.zeta_dict.get(i + 1, zeta_rest) for i in range(self.M_red.shape[0])],
            dtype=float,
        )
        if len(self.zeta) != len(self.omega):
            raise ValueError("Length of zeta must match number of modes")

        C_modal = np.diag(2.0 * self.zeta * self.omega)
        inv_phi = np.linalg.inv(self.eigvecs)
        self.C_red = inv_phi.T @ C_modal @ inv_phi
        return self.C_red

    def effective_damping_matrix(self) -> np.ndarray:
        """Return reduced damping matrix used in linear frequency response."""
        modal_damping = getattr(self, "C_red", 0.0)
        return modal_damping + self.params.c_alpha * self.M_red + self.params.c_beta * self.K_red

    # ------------------------------------------------------------------
    # ODE construction helpers
    # ------------------------------------------------------------------
    def _normalize_excited_indices(self, j_exc, allow_empty: bool = False) -> np.ndarray:
        S = self.Gamma_red.shape[1]
        if j_exc is None:
            j_exc = [] if allow_empty else [0]
        j_exc = np.atleast_1d(j_exc).astype(int)
        if not allow_empty and j_exc.size == 0:
            raise ValueError("j_exc must contain at least one excited piezo index")
        if np.any((j_exc < 0) | (j_exc >= S)):
            raise ValueError(f"j_exc indices out of range [0, {S - 1}]")
        return np.unique(j_exc)

    def _split_piezo_channels(self, j_exc: np.ndarray):
        idx_all = np.arange(self.Gamma_red.shape[1])
        idx_f = np.setdiff1d(idx_all, j_exc)
        return idx_f, self.Gamma_red[:, idx_f], self.Gamma_red[:, j_exc]

    @staticmethod
    def _normalize_free_gain(K_i, idx_f, S: int) -> np.ndarray:
        if np.isscalar(K_i):
            return float(K_i) * np.ones(len(idx_f))
        K_i = np.asarray(K_i, dtype=float)
        if K_i.shape[0] == S:
            return K_i[idx_f]
        if K_i.shape[0] == len(idx_f):
            return K_i
        raise ValueError(f"K_i length mismatch: expected {len(idx_f)} or {S}, got {K_i.shape[0]}")

    def _base_block_matrices(self, idx_f, Gamma_f, K_p: float, R_c: float):
        N = self.M_red.shape[0]
        Nf = len(idx_f)
        D = self.effective_damping_matrix()
        M_elec = self.params.Cp_scalar * np.eye(Nf)
        M_ODE = np.block([[self.M_red, np.zeros((N, Nf))], [np.zeros((Nf, N)), M_elec]])
        C_ODE = np.block([[D, -Gamma_f], [Gamma_f.T, (K_p / R_c) * np.eye(Nf)]])
        return M_ODE, C_ODE, D

    def build_ode_system(
        self,
        j_exc=None,
        R_c: float = 1e3,
        K_p: float = 0.02,
        K_i=0.0,
        K_c: float = 0.0,
        v_exc=lambda t: 1.0,
        freq_domain_amps=None,
    ) -> PiezoBeamODESystem:
        """Build a coupled ODE system with piezo voltage excitation."""
        N = self.M_red.shape[0]
        S = self.Gamma_red.shape[1]
        j_exc = self._normalize_excited_indices(j_exc, allow_empty=False)
        idx_f, Gamma_f, Gamma_e = self._split_piezo_channels(j_exc)
        K_i_free = self._normalize_free_gain(K_i, idx_f, S)
        M_ODE, C_ODE, D = self._base_block_matrices(idx_f, Gamma_f, K_p, R_c)

        if freq_domain_amps is None:
            freq_domain_amps = np.ones(len(j_exc))
        freq_domain_amps = np.asarray(freq_domain_amps, dtype=float)

        def f_int(x):
            u = x[:N]
            qf = x[N:]
            return np.concatenate([self.K_red @ u, (K_i_free / R_c) * qf + (K_c / R_c) * qf**3])

        def K_tan(x):
            qf = x[N:]
            Kqq = np.diag(K_i_free / R_c) + (3 * K_c / R_c) * np.diag(qf**2)
            return np.block([[self.K_red, np.zeros((N, len(qf)))], [np.zeros((len(qf), N)), Kqq]])

        def f_ext(t):
            v_t = v_exc(t)
            if np.isscalar(v_t):
                v_t = np.full(len(j_exc), float(v_t))
            else:
                v_t = np.asarray(v_t)
                if v_t.shape[0] != len(j_exc):
                    raise ValueError(f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}")
            return np.concatenate([Gamma_e @ v_t, np.zeros(len(idx_f))])

        f_ext_freq_domain = np.concatenate([Gamma_e @ freq_domain_amps, np.zeros(len(idx_f))])
        return PiezoBeamODESystem(M_ODE, self.M_red, self.K_red, C_ODE, D, f_ext_freq_domain, f_int, K_tan, f_ext, v_exc, N, len(idx_f))

    def build_ode_system_base_excitation(
        self,
        u_base,
        du_base=None,
        ddu_base=None,
        j_exc=None,
        R_c: float = 1e3,
        K_p: float = 0.02,
        K_i=0.0,
        K_c: float = 0.0,
        v_exc=lambda t: 0.0,
    ) -> PiezoBeamODESystem:
        """Build an ODE system with prescribed root/base displacement."""
        N = self.M_red.shape[0]
        S = self.Gamma_red.shape[1]
        j_exc = self._normalize_excited_indices(j_exc, allow_empty=True)
        idx_f, Gamma_f, Gamma_e = self._split_piezo_channels(j_exc)
        K_i_free = self._normalize_free_gain(K_i, idx_f, S)
        M_ODE, C_ODE, D = self._base_block_matrices(idx_f, Gamma_f, K_p, R_c)

        base_dof = 0
        Mfb = self.M[np.ix_(self.free_dofs, [base_dof])]

        def f_int(x):
            u = x[:N]
            qf = x[N:]
            return np.concatenate([self.K_red @ u, (K_i_free / R_c) * qf + (K_c / R_c) * qf**3])

        def K_tan(x):
            qf = x[N:]
            Kqq = np.diag(K_i_free / R_c) + (3 * K_c / R_c) * np.diag(qf**2)
            return np.block([[self.K_red, np.zeros((N, len(qf)))], [np.zeros((len(qf), N)), Kqq]])

        def base_force(t):
            udd = 0.0 if ddu_base is None else ddu_base(t)
            return -Mfb.flatten() * udd

        def f_ext(t):
            if len(j_exc) == 0:
                piezo_force = np.zeros(N)
            else:
                v_t = v_exc(t)
                if np.isscalar(v_t):
                    v_t = np.full(len(j_exc), float(v_t))
                else:
                    v_t = np.asarray(v_t)
                    if v_t.shape[0] != len(j_exc):
                        raise ValueError(f"v_exc length mismatch: expected {len(j_exc)}, got {v_t.shape[0]}")
                piezo_force = Gamma_e @ v_t
            return np.concatenate([piezo_force + base_force(t), np.zeros(len(idx_f))])

        return PiezoBeamODESystem(M_ODE, self.M_red, self.K_red, C_ODE, D, f_ext(0.0), f_int, K_tan, f_ext, v_exc, N, len(idx_f))

    def build_ode_system_nonlocal(
        self,
        j_exc=None,
        R_c: float = 1e3,
        K_p: float = 0.02,
        v_exc=lambda t: 1.0,
        freq_domain_amps=None,
        electrical_network=None,
    ) -> PiezoBeamODESystem:
        """Build ODE system with an energy-based nonlocal electrical network."""
        N = self.M_red.shape[0]
        j_exc = self._normalize_excited_indices(j_exc, allow_empty=False)
        idx_f, Gamma_f, Gamma_e = self._split_piezo_channels(j_exc)
        Nf = len(idx_f)
        M_ODE, C_ODE, D = self._base_block_matrices(idx_f, Gamma_f, K_p, R_c)

        if freq_domain_amps is None:
            freq_domain_amps = np.ones(len(j_exc))
        freq_domain_amps = np.asarray(freq_domain_amps, dtype=float)

        def electrical_force(qf):
            f = np.zeros(Nf)
            if electrical_network is None:
                return f
            for elem in electrical_network["elements"]:
                i, j = elem["nodes"]
                phi_i = qf[i]
                phi_j = 0.0 if j in (None, "gnd") else qf[j]
                g = elem["grad"](phi_i - phi_j)
                f[i] += g
                if j not in (None, "gnd"):
                    f[j] -= g
            return f

        def electrical_tangent(qf):
            K = np.zeros((Nf, Nf))
            if electrical_network is None:
                return K
            for elem in electrical_network["elements"]:
                i, j = elem["nodes"]
                phi_i = qf[i]
                phi_j = 0.0 if j in (None, "gnd") else qf[j]
                k = elem["hess"](phi_i - phi_j)
                K[i, i] += k
                if j not in (None, "gnd"):
                    K[j, j] += k
                    K[i, j] -= k
                    K[j, i] -= k
            return K

        def f_int(x):
            u = x[:N]
            qf = x[N:]
            return np.concatenate([self.K_red @ u, electrical_force(qf) / R_c])

        def K_tan(x):
            qf = x[N:]
            Kqq = electrical_tangent(qf) / R_c
            return np.block([[self.K_red, np.zeros((N, Nf))], [np.zeros((Nf, N)), Kqq]])

        def f_ext(t):
            v_t = v_exc(t)
            if np.isscalar(v_t):
                v_t = np.full(len(j_exc), float(v_t))
            else:
                v_t = np.asarray(v_t)
                if v_t.shape[0] != len(j_exc):
                    raise ValueError("v_exc length mismatch")
            return np.concatenate([Gamma_e @ v_t, np.zeros(Nf)])

        f_ext_freq_domain = np.concatenate([Gamma_e @ freq_domain_amps, np.zeros(Nf)])
        return PiezoBeamODESystem(M_ODE, self.M_red, self.K_red, C_ODE, D, f_ext_freq_domain, f_int, K_tan, f_ext, v_exc, N, Nf)


# ----------------------------------------------------------------------
# Geometry builders
# ----------------------------------------------------------------------
def _validate_monotone_regions(regions: list, L: float, tol: float = 1e-12) -> list:
    if not regions:
        raise ValueError("At least one region is required")
    regions = sorted(regions, key=lambda r: r["x_start"])
    if regions[0]["x_start"] < -tol or regions[-1]["x_end"] > L + tol:
        raise ValueError("Regions must lie within [0, L]")
    for i in range(len(regions) - 1):
        if regions[i]["x_end"] > regions[i + 1]["x_start"] + tol:
            raise ValueError(f"Region overlap between indices {i} and {i + 1}")
        if regions[i]["x_end"] < regions[i + 1]["x_start"] - tol:
            warnings.warn(f"Gap between regions: [{regions[i]['x_end']}, {regions[i+1]['x_start']}]")
    if regions[0]["x_start"] > tol:
        warnings.warn(f"Gap at beam start: [0, {regions[0]['x_start']}]")
    if regions[-1]["x_end"] < L - tol:
        warnings.warn(f"Gap at beam end: [{regions[-1]['x_end']}, {L}]")
    return regions


def build_geometry_with_regions(L: float, regions: list, piezos: list, default_h: float = 1e-3) -> GeometrySpec:
    """Build geometry from explicit region dictionaries and piezo edge locations."""
    tol = 1e-12
    regions = _validate_monotone_regions(regions, L, tol=tol)

    x_nodes = [float(regions[0]["x_start"])]
    if abs(x_nodes[0]) > tol:
        x_nodes.insert(0, 0.0)
    elem_EI = []
    elem_rhoA = []

    for reg in regions:
        xa = float(reg["x_start"])
        xb = float(reg["x_end"])
        if xb <= xa:
            raise ValueError(f"Region has non-positive length: {reg}")
        h = float(reg.get("h", default_h))
        if h <= 0:
            raise ValueError("Element size h must be positive")
        n_el = max(1, int(np.ceil((xb - xa) / h)))
        xs = np.linspace(xa, xb, n_el + 1)

        if abs(x_nodes[-1] - xa) > tol:
            x_nodes.append(xa)

        for k in range(n_el):
            x_next = float(xs[k + 1])
            if abs(x_nodes[-1] - x_next) > tol:
                x_nodes.append(x_next)
            elem_EI.append(float(reg["EI"]))
            elem_rhoA.append(float(reg["rhoA"]))

    x_nodes = np.asarray(x_nodes, dtype=float)
    elem_EI = np.asarray(elem_EI, dtype=float)
    elem_rhoA = np.asarray(elem_rhoA, dtype=float)

    piezos_validated = []
    for pz in piezos:
        xL = float(pz["xL"])
        xR = float(pz["xR"])
        if not np.any(np.abs(x_nodes - xL) <= tol):
            raise RuntimeError(f"Piezo left edge xL={xL} not found in mesh nodes")
        if not np.any(np.abs(x_nodes - xR) <= tol):
            raise RuntimeError(f"Piezo right edge xR={xR} not found in mesh nodes")
        out = {"xL": xL, "xR": xR}
        if "theta" in pz:
            out["theta"] = float(pz["theta"])
        piezos_validated.append(out)

    return GeometrySpec(x_nodes=x_nodes, elem_EI=elem_EI, elem_rhoA=elem_rhoA, piezos=piezos_validated)


def build_geometry_from_types(
    L: float,
    region_types: dict,
    region_sequence: list,
    x_starts: list,
    default_h: float = 1e-3,
) -> GeometrySpec:
    """Build geometry using named region types and a left-to-right sequence.

    Any type name containing ``"piezo"`` is automatically registered as an active
    piezo region.
    """
    if len(region_sequence) != len(x_starts):
        raise ValueError("region_sequence and x_starts must have the same length")

    x_starts = np.asarray(x_starts, dtype=float)
    if np.any(np.diff(x_starts) < -1e-12):
        raise ValueError("x_starts must be nondecreasing")

    regions = []
    piezos = []
    for i, type_name in enumerate(region_sequence):
        if type_name not in region_types:
            raise ValueError(f"Region type '{type_name}' not found. Available: {list(region_types)}")
        props = region_types[type_name]
        x_start = float(x_starts[i])
        x_end = float(x_starts[i + 1]) if i + 1 < len(x_starts) else float(L)
        if x_end < x_start:
            raise ValueError(f"Region '{type_name}' has negative length")
        region = {
            "x_start": x_start,
            "x_end": x_end,
            "EI": props["EI"],
            "rhoA": props["rhoA"],
            "h": props.get("h", default_h),
            "name": type_name,
        }
        regions.append(region)
        if "piezo" in type_name.lower():
            piezos.append({"xL": x_start, "xR": x_end})

    return build_geometry_with_regions(L, regions, piezos, default_h=default_h)


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
) -> GeometrySpec:
    """Build a substrate/patch geometry from explicit piezo edge arrays."""
    xL = np.asarray(xL, dtype=float)
    xR = np.asarray(xR, dtype=float)
    if len(xL) != len(xR):
        raise ValueError("xL and xR must have the same length")
    if len(xL) == 0:
        regions = [{"x_start": 0.0, "x_end": L, "EI": EI_gap, "rhoA": rhoA_gap, "h": h_gap, "name": "substrate"}]
        return build_geometry_with_regions(L, regions, [], default_h=h_gap)
    if np.any(xL >= xR) or np.any(xL < 0.0) or np.any(xR > L):
        raise ValueError("Invalid piezo edge arrays")

    segments = []
    if xL[0] > 0.0:
        segments.append(("substrate", 0.0, xL[0]))
    for j in range(len(xL)):
        segments.append(("piezo", xL[j], xR[j]))
        if j < len(xL) - 1 and xR[j] < xL[j + 1]:
            segments.append(("substrate", xR[j], xL[j + 1]))
    if xR[-1] < L:
        segments.append(("substrate", xR[-1], L))

    regions = []
    for name, xa, xb in segments:
        is_piezo = name == "piezo"
        regions.append(
            {
                "x_start": float(xa),
                "x_end": float(xb),
                "EI": EI_patch if is_piezo else EI_gap,
                "rhoA": rhoA_patch if is_piezo else rhoA_gap,
                "h": h_patch if is_piezo else h_gap,
                "name": name,
            }
        )
    piezos = [{"xL": float(a), "xR": float(b)} for a, b in zip(xL, xR)]
    return build_geometry_with_regions(L, regions, piezos, default_h=min(h_patch, h_gap))


def geometry_from_params(params: PiezoBeamParams, h_patch: float, h_gap: float) -> GeometrySpec:
    """Build a default GeometrySpec equivalent to ``PiezoBeamParams``."""
    rhoA_patch = params.b * (params.rho_s * params.hs + 2.0 * params.rho_p * params.hp)
    rhoA_gap = params.b * params.rho_s * params.hs
    return build_geometry_arbitrary_piezos(
        L=params.L_b,
        xL=params.xL,
        xR=params.xR,
        EI_patch=params.YI,
        rhoA_patch=rhoA_patch,
        EI_gap=params.YI_s,
        rhoA_gap=rhoA_gap,
        h_patch=h_patch,
        h_gap=h_gap,
    )


def build_linear_electrical_network(Nf: int, K_i, K_i_nl: float, periodic: bool = False) -> dict:
    """Generate a local/nonlocal linear inductor network in flux variables."""
    elements = []
    if np.isscalar(K_i):
        K_i = float(K_i) * np.ones(Nf)
    else:
        K_i = np.asarray(K_i, dtype=float)
        if len(K_i) != Nf:
            raise ValueError("K_i must be scalar or length Nf")

    for i, Ki in enumerate(K_i):
        elements.append(
            {
                "nodes": (i, None),
                "energy": lambda dphi, Ki=Ki: 0.5 * Ki * dphi**2,
                "grad": lambda dphi, Ki=Ki: Ki * dphi,
                "hess": lambda dphi, Ki=Ki: Ki,
            }
        )

    for i in range(Nf - 1):
        elements.append(
            {
                "nodes": (i, i + 1),
                "energy": lambda dphi, Knl=K_i_nl: 0.5 * Knl * dphi**2,
                "grad": lambda dphi, Knl=K_i_nl: Knl * dphi,
                "hess": lambda dphi, Knl=K_i_nl: Knl,
            }
        )

    if periodic and Nf > 2:
        elements.append(
            {
                "nodes": (Nf - 1, 0),
                "energy": lambda dphi, Knl=K_i_nl: 0.5 * Knl * dphi**2,
                "grad": lambda dphi, Knl=K_i_nl: Knl * dphi,
                "hess": lambda dphi, Knl=K_i_nl: Knl,
            }
        )

    return {"elements": elements}
