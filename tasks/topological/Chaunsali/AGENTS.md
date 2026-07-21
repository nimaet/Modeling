# Nima_topological_chain project instructions

This project studies nonlinear/topological phononic chains inspired by Chaunsali & Theocharis (2019), especially self-induced topological transitions, nonlinear edge modes, kink solitons, and locally resonant chain variants.

Key references:
- docs/Chaunsali_Theocharis_2019.pdf
- docs/Equations.txt

Main modeling conventions:
- Use nondimensional time tau.
- Use gamma for stiffness mismatch.
- Use Gamma for cubic nonlinearity in the locally resonant attachment model.
- Main-chain displacement is u.
- Local resonator displacement is eta.
- Relative displacement is r_n = eta_n - u_n.
- For two-site locally resonant unit cells, use:
  r_1,n = eta_1,n - u_1,n
  r_2,n = eta_2,n - u_2,n

Coding conventions:
- Python scripts must use TAB indentation characters.
- Prefer symbolic derivations with SymPy when deriving envelope equations.
- Do not rewrite whole files when a small patch is requested.
- Keep numerical experiments reproducible: include parameters, initial conditions, and plotting labels.

Research goals:
- Reproduce Chaunsali & Theocharis results rigorously.
- Compare discrete simulations, Bloch dispersion, and homogenized/envelope models.
- Extend the analysis to locally resonant chains with nonlinear local attachments.
- Track how amplitude changes effective stiffness, band gaps, edge modes, and soliton/domain-wall behavior.