# %%
from IPython.display import clear_output
from matplotlib import cm, colors
import numpy as np
from numpy import pi
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import tqdm
# ===================== Geometry =====================

L_b = 0.3185						# beam length [m]
w_p = 10e-3						# patch width [m]
w_s = 0.265625e-3					# spacing between patches [m]
Q = 31								# number of unit cells
S = Q								# one patch per cell
# positions

xL = np.array([(j-1)*w_p + j*w_s for j in range(1, S+1)], dtype=float)
xR = xL + w_p
eps0 = 8.854e-12
eps_r = 1900.0

d31 = -1.75e-10
s11 = 1.5873e-11
e31 = d31 / s11   
# print(f"e31ff = {e31:e} C/m^2")
eps33_bar = eps0 * eps_r
eps33 = eps33_bar - d31**2 / s11
# print(f"eps33_bar = {eps33_bar:e} F/m")

#  ===================== Beam material (6061 Al) =====================
rho_p, rho_s = 8500, 2700          # [kg/m³]
E_p, E_s   = 1/s11, 70e9           # [Pa]						# Pa
b = 10e-3						# m
hp, hs = 0.23e-3, 0.5e-3 					# m
hpc = (hp + hs)/2
#===================================Stiffness==========================
m = b*(rho_s * hs + 2 * rho_p * hp)
term1 = E_s * hs**3 / 8
term2 = E_p * ((hp + hs/2)**3 - hs**3/8)
YI = 2*b/3 * (term1 + term2)
# e31 = -14.76 #-5.4e-10						# C/m^2 (typical for PZT-5A)
# e31, eps33 = -14.76, 14.8e-9   

# A_term = (hp + 0.5*hs)**2 - (hs**2)/4.0
Cp_scalar = 2*eps33*  w_p * b / hp
Cp = Cp_scalar * np.ones(S)
theta_mech = 2*e31 * b *hpc		# parallel case (3.9)

# ===================== Piezo capacitance =====================



# ===================== Cantilever eigenvalues =====================

N = 40							# number of modes

def eigen_eq(lam):
	return 1.0 + np.cos(lam)*np.cosh(lam)

def lambda_(i):
	lam0 = np.pi*(i+0.5)
	return fsolve(eigen_eq, lam0)[0]

lambda_vals = np.array([lambda_(i) for i in range(N)], dtype=float)

beta = lambda_vals / L_b
omega = beta**2 * np.sqrt(YI / m)

# Rayleigh damping = 1% per mode approx (adjust if needed)


# ===================== Mass-normalized mode shapes (your eq. 2.10) =====================
def stable_exp_term(lam, x):
	num = np.sin(lam) + np.cos(lam) + np.exp(-lam)
	den = np.cos(lam)*np.exp(-lam) + 0.5 + 0.5*np.exp(-2*lam)
	return 1/2*(num / den) * np.exp(lam*(x/L_b - 1.0))

def sigma_r(lam):
	num = 2*np.sin(lam)*np.exp(-lam) - 1 + np.exp(-2*lam)
	den = 2*np.cos(lam)*np.exp(-lam) + 1 + np.exp(-2*lam)
	return num / den
# sigma_vals = np.array([sigma_r(r) for r in range(N)], dtype=float)
sigma_vals = np.array([sigma_r(lambda_vals[r]) for r in range(N)], dtype=float)

def mode_shape(r, x):
	print("hp =", hp)
	lam   = lambda_vals[r]
	sigma = sigma_vals[r]
	x = np.asarray(x)

	term = stable_exp_term(lam, x)

	phi = (
		np.cos(lam*x/L_b)
		+ sigma*np.sin(lam*x/L_b)
		- term
		- 1/2*(1-sigma)*np.exp( -lam*x/L_b )
	)
	return np.sqrt(1/(m*L_b)) * phi

def mode_shape_dx(r, x):
	lam   = lambda_vals[r]
	sigma = sigma_vals[r]
	x = np.asarray(x)

	theta = lam*x/L_b

	# the stabilized exponential growth term
	term = stable_exp_term(lam, x)
	term_dx = term * (lam/L_b)

	# derivative of the final (1-sigma)/2 * exp(-lam x / L)
	last_exp = 0.5*(1 - sigma) * np.exp(-lam*x/L_b)
	last_exp_dx = -last_exp * (lam/L_b)

	# assemble phi_x
	phi_x = (
		-np.sin(theta) * (lam/L_b)
		+ sigma*np.cos(theta) * (lam/L_b)
		- term_dx
		- last_exp_dx
	)

	return np.sqrt(1/(m*L_b)) * phi_x





# ===================== Forcing integral (your eq. 2.24) =====================

Phi_int = np.zeros(N)
for r in range(N):
	lam = lambda_vals[r]
	sigma = sigma_vals[r]
	Phi_int[r] = (2 * sigma / lam) * np.sqrt(L_b / m)

# ===================== Coupling matrix =====================

Gamma = np.zeros((N, S))
for r in range(N):
	for j in range(S):
		Gamma[r, j] = mode_shape_dx(r, xR[j]) - mode_shape_dx(r, xL[j])

# ===================== Base excitation: chirp 1–5 kHz =====================

j_exc =30
A_exc = 50
f0 = 1e3
f1 = 1.5e3
t_end = 0.1

def v_exc(t, A_exc=A_exc, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)

# ===================== Duffing shunt parameters (all set to 1) =====================

R_b = 1e6
R_c = 1e3

zeta_p = 0.0151
zeta_q = 0.0392
omega_p = 2*np.pi*1070
omega_q = 2*np.pi*5892.5 
A = np.array([
	[omega_p/(2*YI), 1/(2*m*omega_p)],
	[omega_q/(2*YI), 1/(2*m*omega_q)]
])
b = np.array([zeta_p, zeta_q])
c_beta, c_alpha = np.linalg.solve(A, b)
print("C_beta  =", c_beta)
print("C_alpha =", c_alpha)

# zeta = 0.003 * np.ones(N)
zeta = c_alpha / (2*omega*m) + c_beta * omega / (2*YI)
# zete = 
omega2 = omega**2
damp = 2*zeta*omega
# damp = 2*0.029 * np.ones(N)

def odefun(t, x, v_exc=v_exc, K_c=0, K_p=0, K_i=0):
	eta = x[0:N]
	eta_dot = x[N:2*N]
	z = x[2*N:2*N+S]
	v = x[2*N+S:2*N+2*S]
	v[j_exc] = v_exc(t)
	# mechanical
	eta_ddot = - damp*eta_dot - omega2*eta + theta_mech*(Gamma @ v)
	# voltage equation (replaces Y*v term with Kp*v + Ki*z)
	v_dot = np.zeros(S)
	z_dot = np.zeros(S)
	strain_coupling = Gamma.T @ eta_dot
	# nonlinear Duffing term: Kc * z^3
	duffing = z**3
	# voltage ODE
	num = -(K_p/R_c * v) - (K_i/R_c * z) - theta_mech * strain_coupling - (K_c/R_c * duffing)
	v_dot =  num / Cp	# integrator state
	z_dot = v
	# override voltage directly
	# v[j_exc] = v_exc(t)
	# v_dot[j_exc] = 0.0
	# z_dot[j_exc] = v_exc(t)
		
	return np.concatenate([eta_dot, eta_ddot, v, v_dot])

# ===================== Solve =====================


def modal_simulation(K_c = 0, K_p = 0.00, K_i = 0):
	def f_eval(x):
		return odefun(0, x, v_exc=lambda t: 0, K_c=K_c, K_p=K_p, K_i=K_i)   # freeze excitation

	# M = 2                   # number of samples
	# eps = 1e-1                 # perturbation radius around equilibrium
	dim = 2*N + 2*S
	# X = eps * (2*np.random.rand(dim, M) - 1)
	# Y = np.zeros((dim, M))
	# for k in range(M):
	#     Y[:, k] = f_eval(X[:, k])

	A12 = np.eye(N)
	A21 = -np.diag(omega2)
	A22 = -np.diag(damp)
	A24 = theta_mech*Gamma
	A34 = np.eye(S)
	A42 = -(theta_mech/Cp_scalar)*Gamma.T
	A43 = -np.eye(S)*K_i/R_c/Cp_scalar
	A44 = -np.eye(S)*K_p/R_c/Cp_scalar
	A = np.zeros((dim, dim))
	A[0:N, N:2*N] = A12
	A[N:2*N, 0:N] = A21
	A[N:2*N, N:2*N] = A22
	A[N:2*N, 2*N+S:2*N+2*S] = A24
	A[2*N:2*N+S, 2*N+S:2*N+2*S] = A34
	A[2*N+S:2*N+2*S, N:2*N] = A42
	A[2*N+S:2*N+2*S, 2*N:2*N+S] = A43
	A[2*N+S:2*N+2*S, 2*N+S:2*N+2*S] = A44
	A[N:2*N, 2*N + j_exc] = 0.0   # THIS IS YOUR HANDWRITTEN RULE
	# Y_anl = A @ X
	# # Error between analytical linearization and sampled data
	# residual = Y - Y_anl
	# abs_err_norm = np.linalg.norm(residual)
	# rel_err_norm = abs_err_norm / (np.linalg.norm(Y))
	# per_sample_rel_err = np.linalg.norm(residual, axis=0) / (np.linalg.norm(Y, axis=0) )
	# print(f"Residual Frobenius norm: {abs_err_norm:.4e}")
	# print(f"Relative Frobenius norm: {rel_err_norm:.4e}")
	# print(f"Mean per-sample relative error: {per_sample_rel_err.mean():.4e}")
	# print(f"Max per-sample relative error: {per_sample_rel_err.max():.4e}")
	# Build forcing vector f
	# --------------------------------------------------------------
	f = np.zeros(dim)
	b = theta_mech * Gamma[:, j_exc]   # mechanical forcing term
	f[N:2*N] = b                       # insert only in eta_ddot block

	# --------------------------------------------------------------
	# Frequency response
	# --------------------------------------------------------------
	def freq_response(omega_list):
		Y = []
		I = np.eye(dim)
		for w in omega_list:
			M = 1j*w * I - A
			y = np.linalg.solve(M, f)
			Y.append(y)
		return np.array(Y)   # shape (len(w), dim)

	# Example frequency sweep
	w = np.arange(0.1, 4500, 2.5 ) * 2*np.pi   # rad/s
	Y = freq_response(w)

	# Extract modal coordinates from frequency-response solution
	eta     = Y[:, 0:N].T        # shape (N, n_freq)
	eta_dot = Y[:, N:2*N].T      # shape (N, n_freq)
	x_eval = np.linspace(0, L_b, 100)

	npts = len(x_eval)
	nfreq = eta.shape[1]

	disp  = np.zeros((npts, nfreq), dtype=complex)
	veloc = np.zeros((npts, nfreq), dtype=complex)

	for r in range(N):
		phi_r  = mode_shape(r, x_eval)       # shape (npts,)
		disp  += np.outer(phi_r, eta[r, :])
		veloc += np.outer(phi_r, eta_dot[r, :])
	vel_mag = np.mean(np.abs(veloc), axis=0 )
	disp_mag = np.mean(np.abs(disp), axis=0 )

	freq_modal = w / (2*np.pi)
	return freq_modal, vel_mag, disp_mag

x_eval = np.linspace(0, L_b, 100)

def run_time_sim(v_exc=v_exc, K_c=0, K_p=0, K_i=0, t_end=t_end, x_eval=x_eval, t_eval=None):
	# --- local excitation function with this amplitude ---
	x0 = np.zeros(2*N + 2*S)

	sol = solve_ivp(
		lambda t, x: odefun(t, x, v_exc=v_exc, K_c= K_c, K_p=K_p, K_i=K_i),
		(0, t_end),
		x0,
		t_eval=t_eval,
		method='RK45',
		rtol=1e-9,
		atol=1e-10
		)
	# reconstruct velocities on beam
	eta     = sol.y[0:N, :]
	eta_dot = sol.y[N:2*N, :]
	veloc   = np.zeros((len(x_eval), eta.shape[1]))
	for r in range(N):
		veloc += np.outer(mode_shape(r, x_eval), eta_dot[r,:])
	# ----- Compute and plot spectrum -----
	t = sol.t
	y = veloc                # velocity signal at x_eval index 10
	Nt = len(t)
	dt = t[1] - t[0]                  # sampling period
	fs = 1/dt                         # sampling frequency

	Y = np.fft.fft(y, axis=1)
	X = np.fft.fft(v_exc(t))
	freq = np.fft.fftfreq(Nt, d=dt)

	# take only positive frequencies
	idx = freq >= 0
	freq = freq[idx]
	Y = Y[:,idx]
	X = X[idx]
	FRF = np.mean(np.abs(Y), axis=0)/ np.abs(X)
	return {
		't': sol.t,
		'veloc': veloc,
		'freq': freq,
		'Y': Y,
		'X': X,
		'FRF': FRF
	}


# ===================== Plot =====================