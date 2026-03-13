"""
PiezoBeamParamsTorch — differentiable material/cross-section parameters.

All scalar physical parameters (hp, hs, b, E_s, s11, d31, eps_r, rho_s,
rho_p) are stored as nn.Parameter tensors so that any quantity derived from
them (YI, rhoA, theta_mech, Cp_scalar, …) is part of the autograd graph.

Mesh-topology scalars (w_p, w_s, n_patches) are intentionally *not*
differentiable: they define node positions and therefore mesh connectivity,
which cannot be varied continuously without remeshing.

Usage
-----
>>> from Modeling.models_torch import PiezoBeamParamsTorch
>>> tp = PiezoBeamParamsTorch()           # initialise from defaults
>>> tp.hp.grad                            # None until backward() is called

You can also initialise from an existing PiezoBeamParams instance:

>>> from Modeling.models.beam_properties import PiezoBeamParams
>>> p = PiezoBeamParams(hp=0.3e-3)
>>> tp = PiezoBeamParamsTorch(params=p)

To freeze a parameter (e.g. keep rho_s fixed during optimisation):

>>> tp.rho_s.requires_grad_(False)
"""

from __future__ import annotations

from typing import Optional
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PiezoBeamParamsTorch(nn.Module):
    """
    Differentiable piezoelectric beam material/cross-section parameters.

    Learnable parameters (nn.Parameter, requires_grad=True)
    --------------------------------------------------------
    hp      piezo layer thickness               [m]
    hs      substrate thickness                  [m]
    b       beam width                           [m]
    E_s     substrate Young's modulus            [Pa]
    s11     piezo compliance s_11^E              [m²/N]
    d31     piezo strain constant d_31           [C/N]
    eps0    vacuum permittivity                  [F/m]
    eps_r   relative dielectric constant         [-]
    rho_s   substrate density                    [kg/m³]
    rho_p   piezo density                        [kg/m³]

    Fixed scalars (plain Python floats, NOT in autograd graph)
    ----------------------------------------------------------
    w_p     patch width                          [m]
    w_s     inter-patch spacing                  [m]
    n_patches  number of patches                 [-]
    S       = n_patches (alias)

    Derived differentiable quantities (call as methods)
    ---------------------------------------------------
    E_p()         1 / s11
    e31()         d31 / s11
    eps33()       eps0*eps_r - d31**2/s11
    YI()          bending stiffness EI of patch elements
    YI_s()        bending stiffness of substrate (gap elements)
    rhoA_patch()  mass/length on a patch section
    rhoA_gap()    mass/length on a gap section
    theta_mech()  electromechanical coupling coefficient
    Cp_scalar()   capacitance of a single patch
    """

    def __init__(
        self,
        params=None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        device = device or torch.device("cpu")
        self.dtype  = dtype
        self.device = device

        # ── Resolve defaults from existing dataclass if not supplied ──────
        if params is None:
            params = self._load_default_params()

        def _p(val: float) -> nn.Parameter:
            """Wrap a scalar as a learnable parameter."""
            return nn.Parameter(torch.tensor(float(val), dtype=dtype, device=device))

        # ── Learnable parameters ──────────────────────────────────────────
        self.hp    = _p(params.hp)
        self.hs    = _p(params.hs)
        self.b     = _p(params.b)
        self.E_s   = _p(params.E_s)
        self.s11   = _p(params.s11)
        self.d31   = _p(params.d31)
        self.eps0  = _p(params.eps0)
        self.eps_r = _p(params.eps_r)
        self.rho_s = _p(params.rho_s)
        self.rho_p = _p(params.rho_p)

        # ── Fixed mesh/geometry scalars (NOT in autograd graph) ───────────
        # Changing these would require rebuilding the FE mesh.
        self.w_p       = float(params.w_p)
        self.w_s       = float(params.w_s)
        self.n_patches = int(params.n_patches)
        self.S         = int(params.S)

        # Store zeta dict for optional damping use (not in gradient graph)
        self.zeta_dict: dict = dict(params.zeta_dict)

    # ── Private helper ────────────────────────────────────────────────────

    @staticmethod
    def _load_default_params():
        """Import PiezoBeamParams from the sibling models package."""
        try:
            from Modeling.models.beam_properties import PiezoBeamParams
            return PiezoBeamParams()
        except ImportError:
            # Fallback: load the file directly using its absolute path
            _here = Path(__file__).resolve().parents[1] / "models" / "beam_properties.py"
            spec  = importlib.util.spec_from_file_location("beam_properties", _here)
            mod   = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.PiezoBeamParams()

    # ── Derived differentiable quantities ─────────────────────────────────

    def E_p(self) -> torch.Tensor:
        """Piezo Young's modulus  E_p = 1 / s11  [Pa]."""
        return 1.0 / self.s11

    def e31(self) -> torch.Tensor:
        """Piezoelectric coupling constant  e_31 = d31 / s11  [C/m²]."""
        return self.d31 / self.s11

    def eps33(self) -> torch.Tensor:
        """Clamped dielectric constant  ε₃₃ = ε₀εᵣ − d31²/s11  [F/m]."""
        return self.eps0 * self.eps_r - self.d31 ** 2 / self.s11

    def YI(self) -> torch.Tensor:
        """Bending stiffness EI of a patch (beam + two piezo layers)  [N·m²]."""
        term1 = self.E_s * self.hs ** 3 / 8.0
        term2 = self.E_p() * ((self.hp + self.hs / 2.0) ** 3 - self.hs ** 3 / 8.0)
        return 2.0 * self.b / 3.0 * (term1 + term2)

    def YI_s(self) -> torch.Tensor:
        """Bending stiffness of bare substrate (gap elements)  [N·m²]."""
        return self.b * self.E_s * self.hs ** 3 / 12.0

    def rhoA_patch(self) -> torch.Tensor:
        """Mass per unit length — patch section (substrate + 2 piezo layers)  [kg/m]."""
        return self.b * (self.rho_s * self.hs + 2.0 * self.rho_p * self.hp)

    def rhoA_gap(self) -> torch.Tensor:
        """Mass per unit length — gap section (bare substrate)  [kg/m]."""
        return self.b * self.rho_s * self.hs

    def theta_mech(self) -> torch.Tensor:
        """Electromechanical coupling coefficient  θ = 2 e31 b (hp+hs)/2  [N/V]."""
        hpc = 0.5 * (self.hp + self.hs)
        return 2.0 * self.e31() * self.b * hpc

    def Cp_scalar(self) -> torch.Tensor:
        """Capacitance of a single piezo patch  [F]."""
        return 2.0 * self.eps33() * self.w_p * self.b / self.hp

    # ── Geometry helpers (fixed, return numpy arrays) ─────────────────────

    def xL_np(self) -> np.ndarray:
        """Left edge of each patch [m], numpy."""
        j = np.arange(1, self.S + 1)
        return (j - 1) * self.w_p + j * self.w_s

    def xR_np(self) -> np.ndarray:
        """Right edge of each patch [m], numpy."""
        return self.xL_np() + self.w_p

    def L_b(self) -> float:
        """Total beam length [m]."""
        return float(self.xR_np()[-1] + self.w_s)

    # ── forward (convenience summary) ────────────────────────────────────

    def forward(self) -> dict:
        """Return all differentiable derived quantities as a dict of tensors."""
        return dict(
            E_p        = self.E_p(),
            e31        = self.e31(),
            eps33      = self.eps33(),
            YI         = self.YI(),
            YI_s       = self.YI_s(),
            rhoA_patch = self.rhoA_patch(),
            rhoA_gap   = self.rhoA_gap(),
            theta_mech = self.theta_mech(),
            Cp_scalar  = self.Cp_scalar(),
        )
