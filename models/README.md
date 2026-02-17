# FE3 Finite-Element Model (Euler–Bernoulli Beam + Piezo Coupling)

This note documents the mathematics implemented in `FE3.py` for the beam finite-element model and the coupled electromechanical ODE.

## 1) Mechanical beam FE model

The beam uses 2-node Euler–Bernoulli elements with 4 DOFs per element:

\[
\mathbf{u}_e = [w_i,\ \theta_i,\ w_j,\ \theta_j]^T
\]

where $w$ is transverse displacement and $\theta$ is rotation.

For each element $e$ of length $L_e$, bending stiffness $EI_e$, and mass per length $(\rho A)_e$, the code assembles:

\[
\mathbf{K}_e = \frac{EI_e}{L_e^3}
\begin{bmatrix}
12 & 6L_e & -12 & 6L_e \\
6L_e & 4L_e^2 & -6L_e & 2L_e^2 \\
-12 & -6L_e & 12 & -6L_e \\
6L_e & 2L_e^2 & -6L_e & 4L_e^2
\end{bmatrix}
\]

\[
\mathbf{M}_e = \frac{(\rho A)_e L_e}{420}
\begin{bmatrix}
156 & 22L_e & 54 & -13L_e \\
22L_e & 4L_e^2 & 13L_e & -3L_e^2 \\
54 & 13L_e & 156 & -22L_e \\
-13L_e & -3L_e^2 & -22L_e & 4L_e^2
\end{bmatrix}
\]

Global assembly gives:

\[
\mathbf{M}\,\ddot{\mathbf{u}} + \mathbf{K}\,\mathbf{u} = \mathbf{f}
\]

with DOF ordering

\[
[w_0,\theta_0,w_1,\theta_1,\dots,w_{N_n-1},\theta_{N_n-1}]^T.
\]

## 2) Piezo-mechanical coupling matrix

For each piezo patch $j$ spanning nodes at $x_{L,j}$ and $x_{R,j}$, the coupling column $\Gamma_{:,j}$ is built only on rotational DOFs:

\[
\Gamma_{2k_R+1,j} += \theta_{\text{mech}},\qquad
\Gamma_{2k_L+1,j} += -\theta_{\text{mech}}.
\]

So each patch contributes an equal-and-opposite bending moment pair at its two edges.

## 3) Boundary conditions and reduced matrices

Clamped boundary at node 0 is applied by removing DOFs $[0,1]$:

\[
\mathbf{u}_f = \mathbf{u}[\text{free\_dofs}],
\]

\[
\mathbf{K}_r = \mathbf{K}_{ff},\quad
\mathbf{M}_r = \mathbf{M}_{ff},\quad
\Gamma_r = \Gamma_f.
\]

## 4) Eigenanalysis

Natural modes are computed from the generalized eigenproblem:

\[
\mathbf{K}_r\,\boldsymbol{\phi}_r = \omega^2\,\mathbf{M}_r\,\boldsymbol{\phi}_r.
\]

Frequencies are:

\[
f = \frac{\omega}{2\pi}.
\]

Modes are expanded to full DOF size and mass-normalized:

\[
\boldsymbol{\phi}_i^T\mathbf{M}\boldsymbol{\phi}_i = 1.
\]

## 5) Coupled electromechanical ODE used in `build_ode_system`

Let:
- $N$ = number of reduced mechanical DOFs,
- $S$ = number of piezos,
- excited piezo indices = $j_{exc}$,
- feedback/free piezo indices = $j_f$,
- $\Gamma_e = \Gamma_r[:,j_{exc}]$,
- $\Gamma_f = \Gamma_r[:,j_f]$.

Define state:

\[
\mathbf{x} = \begin{bmatrix}\mathbf{u}\\\mathbf{q}_f\end{bmatrix},
\]

where $\mathbf{u}\in\mathbb{R}^N$ and $\mathbf{q}_f\in\mathbb{R}^{|j_f|}$.

Rayleigh damping:

\[
\mathbf{D} = c_\alpha\mathbf{M}_r + c_\beta\mathbf{K}_r.
\]

Electrical capacitance block:

\[
\mathbf{M}_{elec} = C_p\,\mathbf{I}.
\]

The code forms

\[
\mathbf{M}_{ODE}=
\begin{bmatrix}
\mathbf{M}_r & \mathbf{0}\\
\mathbf{0} & \mathbf{M}_{elec}
\end{bmatrix},
\qquad
\mathbf{C}_{ODE}=
\begin{bmatrix}
\mathbf{D} & -\Gamma_f\\
\Gamma_f^T & (K_p/R_c)\mathbf{I}
\end{bmatrix}.
\]

Internal force term:

\[
\mathbf{f}_{int}(\mathbf{x})=
\begin{bmatrix}
\mathbf{K}_r\mathbf{u}\\
(K_i/R_c)\,\mathbf{q}_f + (K_c/R_c)\,\mathbf{q}_f^{\circ 3}
\end{bmatrix},
\]

where $\mathbf{q}_f^{\circ 3}$ is elementwise cubic.

External forcing from excited piezos:

\[
\mathbf{f}_{ext}(t)=
\begin{bmatrix}
\Gamma_e\,\mathbf{v}_{exc}(t)\\
\mathbf{0}
\end{bmatrix}.
\]

So the implemented first-order-in-state second-order dynamics is:

\[
\mathbf{M}_{ODE}\,\ddot{\mathbf{x}} + \mathbf{C}_{ODE}\,\dot{\mathbf{x}} + \mathbf{f}_{int}(\mathbf{x}) = \mathbf{f}_{ext}(t).
\]

## 6) Tangent stiffness/Jacobian block used for Newton iterations

\[
\mathbf{K}_{tan}(\mathbf{x})=
\begin{bmatrix}
\mathbf{K}_r & \mathbf{0}\\
\mathbf{0} & \mathbf{K}_{qq}(\mathbf{q}_f)
\end{bmatrix},
\]

with

\[
\mathbf{K}_{qq}(\mathbf{q}_f)
=
\frac{1}{R_c}\operatorname{diag}(K_i)
+\frac{3K_c}{R_c}\operatorname{diag}(\mathbf{q}_f^{\circ 2}).
\]

## 7) Geometry/material assignment

`FE3.py` supports:
- default patch-gap geometry from `params`,
- arbitrary piezo regions (`build_geometry_arbitrary_piezos`),
- fully custom region definitions (`build_geometry_with_regions`),
- type-based region layouts (`build_geometry_from_types`).

All produce:
- node coordinates `x_nodes`,
- per-element $EI_e$ and $(\rho A)_e$,
- piezo edge locations used to build $\Gamma$.
