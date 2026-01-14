
from numba import njit
import numpy as np	

from numba import njit, float64, int64

from numba import njit

@njit(cache=True, fastmath=True)
def rk4_integrate(
	X,          # <-- preallocated state array
	t,
	v_exc_arr,
	N,
	S,
	damp,
	omega2,
	Gamma,
	theta_mech,
	Cp,
	R_c,
	K_c,
	K_p,
	K_i,
	j_exc
):
	Nt = t.size
	dim = X.shape[0]

	for n in range(Nt - 1):
		dt = t[n+1] - t[n]

		k1 = odefun_jit(
			t[n], X[:, n],
			N, S, damp, omega2, Gamma,
			theta_mech, Cp, R_c, K_c, K_p, K_i, j_exc,
			v_exc_arr[n]
		)

		k2 = odefun_jit(
			t[n] + 0.5*dt,
			X[:, n] + 0.5*dt*k1,
			N, S, damp, omega2, Gamma,
			theta_mech, Cp, R_c, K_c, K_p, K_i, j_exc,
			v_exc_arr[n]
		)

		k3 = odefun_jit(
			t[n] + 0.5*dt,
			X[:, n] + 0.5*dt*k2,
			N, S, damp, omega2, Gamma,
			theta_mech, Cp, R_c, K_c, K_p, K_i, j_exc,
			v_exc_arr[n]
		)

		k4 = odefun_jit(
			t[n] + dt,
			X[:, n] + dt*k3,
			N, S, damp, omega2, Gamma,
			theta_mech, Cp, R_c, K_c, K_p, K_i, j_exc,
			v_exc_arr[n]
		)

		X[:, n+1] = X[:, n] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


from numba import njit

@njit(cache=True, fastmath=True)
def odefun_jit(
	t,
	x,
	N,
	S,
	damp,
	omega2,
	Gamma,
	theta_mech,
	Cp,
	R_c,
	K_c,
	K_p,
	K_i,
	j_exc,
	v_exc_t
):
	eta = x[0:N]
	eta_dot = x[N:2*N]
	z = x[2*N:2*N+S]
	v = x[2*N+S:2*N+2*S]

	# excitation
	v[j_exc] = v_exc_t

	# mechanical
	eta_ddot = (
		- damp * eta_dot
		- omega2 * eta
		+ theta_mech * (Gamma @ v)
	)

	# electrical
	strain = Gamma.T @ eta_dot
	v_dot = (
		-(K_p / R_c) * v
		-(K_i / R_c) * z
		- theta_mech * strain
		-(K_c / R_c) * (z ** 3)
	) / Cp

	z_dot = v

	out = np.empty_like(x)
	out[0:N] = eta_dot
	out[N:2*N] = eta_ddot
	out[2*N:2*N+S] = z_dot
	out[2*N+S:2*N+2*S] = v_dot
	return out
