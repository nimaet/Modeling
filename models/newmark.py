# %%
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
USE_TQDM = sys.stdout.isatty()
def newmark_beta_nonlinear(
	M, C,
	f_int, K_tan,
	f_ext,
	u0, v0, a0_init,
	dt, n_steps,
	beta=0.25, gamma=0.5,
	newton_tol=1e-8,
	newton_maxiter=20
):
	"""
	Nonlinear Newmark-beta time integration with Newton-Raphson.

	M, C : (n,n) arrays
	f_int(u) : internal force vector (n,)
	K_tan(u) : tangent stiffness matrix (n,n)
	f_ext(t) : external force vector (n,)
	u0, v0, a0_init : initial displacement, velocity, acceleration
	"""

	n = len(u0)

	u = np.zeros((n_steps+1, n))
	v = np.zeros((n_steps+1, n))
	a = np.zeros((n_steps+1, n))

	u[0] = u0
	v[0] = v0
	a[0] = a0_init

	# Newmark constants
	a0 = 1.0 / (beta * dt**2)
	a1 = gamma / (beta * dt)
	a2 = 1.0 / (beta * dt)
	a3 = 1.0 / (2*beta) - 1.0
	a4 = gamma / beta - 1.0
	a5 = dt * (gamma / (2*beta) - 1.0)

	t = 0.0

	for nstep in tqdm(range(n_steps), desc="Newmark Integration", unit="step", disable=not USE_TQDM):
		t += dt

		# Predictor (constant acceleration predictor)
		u_trial = (
			u[nstep]
			+ dt*v[nstep]
			+ dt**2*(0.5 - beta)*a[nstep]
		)

		# Newton-Raphson iteration
		for it in range(newton_maxiter):

			# acceleration and velocity from trial displacement
			a_trial = (
				a0*(u_trial - u[nstep])
				- a2*v[nstep]
				- a3*a[nstep]
			)

			v_trial = (
				v[nstep]
				+ dt*((1-gamma)*a[nstep] + gamma*a_trial)
			)

			# residual
			res = (
				M @ a_trial
				+ C @ v_trial
				+ f_int(u_trial)
				- f_ext(t)
			)

			res_norm = np.linalg.norm(res)
			if res_norm < newton_tol:
				break

			# tangent matrix
			K_eff = (
				a0*M
				+ a1*C
				+ K_tan(u_trial)
			)

			# Newton update
			du = np.linalg.solve(K_eff, -res)
			u_trial += du

		else:
			raise RuntimeError(
				f"Newton did not converge at step {nstep}, "
				f"residual = {res_norm}"
			)

		# accept converged solution
		u[nstep+1] = u_trial
		a[nstep+1] = (
			a0*(u[nstep+1] - u[nstep])
			- a2*v[nstep]
			- a3*a[nstep]
		)
		v[nstep+1] = (
			v[nstep]
			+ dt*((1-gamma)*a[nstep] + gamma*a[nstep+1])
		)

	return u, v, a


if __name__ == "__main__":
    import numpy as np

    # Duffing parameters
    m = 1.0
    c = 0.02
    k = 1.0
    alpha = 10.0      # cubic stiffness (hardening)

    # matrices
    M = np.array([[m]])
    C = np.array([[c]])
    def f_int(u):
        return np.array([
            k*u[0] + alpha*u[0]**3
        ])

    def K_tan(u):
        return np.array([
            [k + 3*alpha*u[0]**2]
        ])
    F0 = 0.2
    omega = 1.0

    def f_ext(t):
        return np.array([
            F0*np.sin(omega*t)
        ])
    u0 = np.array([0.0])
    v0 = np.array([0.0])

    # consistent initial acceleration
    a0_init = np.linalg.solve(
        M,
        f_ext(0.0) - C@v0 - f_int(u0)
    )
    dt = 0.01
    n_steps = 5000

    beta = 0.25
    gamma = 0.5
    u, v, a = newmark_beta_nonlinear(
        M=M,
        C=C,
        f_int=f_int,
        K_tan=K_tan,
        f_ext=f_ext,
        u0=u0,
        v0=v0,
        a0_init=a0_init,
        dt=dt,
        n_steps=n_steps,
        beta=beta,
        gamma=gamma,
        newton_tol=1e-10,
        newton_maxiter=25
    )


    # Duffing RHS for RK45
    def duffing_rhs(t, y):
        u = y[0]
        v = y[1]

        du = v
        dv = (f_ext(t)[0] - c*v - k*u - alpha*u**3) / m

        return [du, dv]
    t0 = 0.0
    t1 = n_steps * dt
    t_eval = np.linspace(t0, t1, n_steps+1)
    sol = solve_ivp(
        duffing_rhs,
        [t0, t1],
        y0=[u0[0], v0[0]],
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    u_rk = sol.y[0]
    v_rk = sol.y[1]
    u_nm = u[:,0]
    v_nm = v[:,0]
    err_inf = np.max(np.abs(u_nm - u_rk))
    print("Max displacement difference:", err_inf)


    plt.figure()
    plt.plot(u_nm, v_nm, label="Newmark")
    plt.plot(u_rk, v_rk, "--", label="RK45")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.legend()
    plt.grid(True)
    plt.show()




