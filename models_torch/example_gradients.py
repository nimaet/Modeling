"""
example_gradients.py — Demonstration of differentiable K/M matrices.

Run from the Metamaterial beam/ workspace root, e.g.:

    python -m Modeling.models_torch.example_gradients

What this script shows
-----------------------
1. Build differentiable K_red, M_red, Γ_red from default parameters.
2. Compute the first 5 natural frequencies [Hz].
3. Back-propagate the first natural frequency loss → obtain dω₁/d(param)
   for every learnable parameter in PiezoBeamParamsTorch.
4. Finite-difference verification of the hp gradient.
5. Simple gradient-descent optimisation: push ω₁ toward a target frequency.
"""

import sys
from pathlib import Path

# Allow running directly from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from Modeling.models_torch import PiezoBeamParamsTorch, PiezoBeamFE_Torch


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build the differentiable model
# ─────────────────────────────────────────────────────────────────────────────

tp = PiezoBeamParamsTorch(dtype=torch.float64)
fe = PiezoBeamFE_Torch(tp, n_el_patch=3, n_el_gap=2)

print("Model built successfully.")
print(f"  n_elem   = {len(fe.is_patch)}")
print(f"  Ndof     = {fe.K_hat.shape[1]}")
print(f"  Nfree    = {len(fe._mesh.free_dofs)}")
print(f"  S        = {tp.S}  (number of piezos)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Forward pass (no gradient tracking needed here)
# ─────────────────────────────────────────────────────────────────────────────

with torch.no_grad():
    out_ref = fe(n_modes=10)

print("\nFirst 10 natural frequencies (Hz):")
for i, f_ in enumerate(out_ref['freq']):
    print(f"  Mode {i+1:2d}: {f_.item():10.3f} Hz")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gradient of ω₁ w.r.t. all differentiable parameters
# ─────────────────────────────────────────────────────────────────────────────

# Zero out any previous gradients
tp.zero_grad()

out = fe(n_modes=5)
loss = out['freq'][0]   # first natural frequency
loss.backward()

print(f"\nFirst natural frequency: {loss.item():.4f} Hz")
print("\n∂f₁/∂param  (analytical gradients via autograd):")

for name, param in tp.named_parameters():
    if param.grad is not None:
        print(f"  d(f1)/d({name:<10s}) = {param.grad.item(): .6e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Finite-difference check for hp gradient
# ─────────────────────────────────────────────────────────────────────────────

print("\nFinite-difference verification for hp:")

eps  = 1e-7
hp0  = tp.hp.item()

with torch.no_grad():
    tp.hp.data.fill_(hp0 + eps)
    f1_plus  = fe(n_modes=1)['freq'][0].item()

    tp.hp.data.fill_(hp0 - eps)
    f1_minus = fe(n_modes=1)['freq'][0].item()

    tp.hp.data.fill_(hp0)   # restore

fd_grad      = (f1_plus - f1_minus) / (2 * eps)
analytic_grad = tp.hp.grad.item()  # from step 3

print(f"  Analytic  : {analytic_grad: .6e}")
print(f"  FD (2-pt) : {fd_grad: .6e}")
print(f"  Rel. error: {abs(fd_grad - analytic_grad) / (abs(analytic_grad) + 1e-30):.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Simple gradient-descent to shift ω₁ to a target frequency
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Optimisation demo: shift f₁ toward 110 Hz by varying hp only ---")

# Freeze all parameters except hp
for name, param in tp.named_parameters():
    param.requires_grad_(name == "hp")

target_f1 = 110.0   # Hz
optimizer  = torch.optim.Adam([tp.hp], lr=1e-6)

for step in range(40):
    optimizer.zero_grad()
    freq1 = fe(n_modes=1)['freq'][0]
    loss  = (freq1 - target_f1) ** 2
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == 39:
        print(f"  step {step:3d}  f₁ = {freq1.item():.4f} Hz  hp = {tp.hp.item()*1e3:.4f} mm")

# Restore requires_grad for all parameters
for param in tp.parameters():
    param.requires_grad_(True)

print("\nDone.")
