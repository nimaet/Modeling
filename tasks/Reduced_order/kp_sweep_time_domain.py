"""
Time domain K_p sweep with FFT analysis
Performs parametric sweep of K_p in time domain and extracts FRF from FFT
"""

import numpy as np
import matplotlib.pyplot as plt
from ROM import N, S, L_b, odefun, mode_shape
from scipy.integrate import solve_ivp

# Simulation parameters
K_c = 5e11
t_end = 1.0
K_i = 0.0
j_exc = 30
A_exc = 50
f0 = 1000
f1 = 5000

def v_exc(t, A_exc=A_exc, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0 + t*(f1-f0)/t_end) *t)

def run_sim(K_p):
	"""Run time domain simulation for given K_p"""
	x0 = np.zeros(2*N + 2*S)
	t_eval = np.arange(0, t_end, 1/f1/20)
	
	sol = solve_ivp(
		lambda t, x: odefun(t, x, v_exc=v_exc, K_c=K_c, K_p=K_p, K_i=K_i, j_exc=j_exc),
		(0, t_end),
		x0,
		t_eval=t_eval,
		method='RK45',
		rtol=1e-9,
		atol=1e-10
	)
	
	# Reconstruct velocities
	eta = sol.y[0:N, :]
	eta_dot = sol.y[N:2*N, :]
	x_eval = np.linspace(0, L_b, 100)
	veloc = np.zeros((len(x_eval), eta.shape[1]))
	for r in range(N):
		veloc += np.outer(mode_shape(r, x_eval), eta_dot[r,:])
	
	return sol.t, veloc

# Parameter sweep
param_list = np.arange(0.01, 0.8, 0.05)

results = {
	"params": [],
	"param_name": "K_p",
	"t": None,
	"veloc": [],
	"freq": None,
	"Y": [],
	"X": [],
	"FRF": []
}

print(f"Starting K_p sweep (K_c = {K_c:.2e})...")
for par in param_list:
	print(f"  K_p = {par:.3f} ...")
	
	# Run simulation
	t, veloc = run_sim(par)
	results["params"].append(par)
	results["veloc"].append(veloc)
	
	# FFT parameters
	Nt = len(t)
	dt = t[1] - t[0]
	freq = np.fft.fftfreq(Nt, d=dt)
	
	# Velocity FFT
	Y = np.fft.fft(veloc, axis=1)
	results["Y"].append(Y)
	
	# Save time and frequency (only once)
	if results["freq"] is None:
		results["freq"] = freq
		results["t"] = t
	
	# Excitation FFT
	results["X"].append(np.fft.fft(v_exc(t, A_exc=A_exc)))

# Extract frequency and compute FRF
freq = results["freq"]
idx = freq >= 0
freq_pos = freq[idx]

for i, Y in enumerate(results["Y"]):
	X = results["X"][i]
	FRF = np.mean(np.abs(Y), axis=0) / (np.abs(X))
	results["FRF"].append(FRF)

# Save results
np.savez("./sim_dat/kp_sweep_time_domain.npz", **results)
print("Results saved to sim_dat/kp_sweep_time_domain.npz")

# Plot results
plt.figure(figsize=(9, 5))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results['params'])))

for i, (par, FRF) in enumerate(zip(results["params"], results["FRF"])):
	plt.semilogy(results["freq"], FRF, color=colors[i], 
	             linewidth=2, label=f"K_p={par:.2f}")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF Magnitude")
plt.title(f"K_p Sweep (K_c={K_c:.2e}, K_i={K_i})")
plt.xlim([500, 5000])
plt.grid(True)
plt.legend(loc='best', ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig('sim_dat/kp_sweep_time_domain.png', dpi=300, bbox_inches='tight')
plt.show()

print("K_p sweep complete!")
