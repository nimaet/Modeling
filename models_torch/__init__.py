"""
models_torch — differentiable PyTorch wrappers for the piezoelectric beam FE model.

The key idea
------------
The global stiffness K and mass M matrices are assembled as:

    K = Σ_e  EI_e   * K_hat[e]
    M = Σ_e  rhoA_e * M_hat[e]

where K_hat / M_hat are *fixed* (non-differentiable) pattern matrices that
depend only on the mesh geometry, while EI_e and rhoA_e are *differentiable*
scalars computed from the material/cross-section parameters (hp, hs, E_s, …).

This means gradients of any scalar function of K or M (e.g. eigenfrequencies)
can be back-propagated all the way to hp, hs, E_s, d31, rho_s, rho_p, etc.

Modules
-------
PiezoBeamParamsTorch   – differentiable parameter container (nn.Module)
PiezoBeamFE_Torch      – differentiable FE assembly + generalized eigensolver
"""

from Modeling.models_torch.beam_params_torch import PiezoBeamParamsTorch
from Modeling.models_torch.fe_torch import (
    GeometrySpecTorch,
    PiezoBeamFE_Torch,
    build_geometry_arbitrary_piezos,
    build_geometry_from_types,
    build_geometry_with_regions,
)
from Modeling.models_torch.fe_helpers_compat import (
    PiezoBeamODESystemCompat,
    build_ode_system_compat,
)

__all__ = [
    "PiezoBeamParamsTorch",
    "PiezoBeamFE_Torch",
    "GeometrySpecTorch",
    "build_geometry_arbitrary_piezos",
    "build_geometry_with_regions",
    "build_geometry_from_types",
    "PiezoBeamODESystemCompat",
    "build_ode_system_compat",
]
