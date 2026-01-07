"""
Amplitude sweep analysis
Investigates nonlinear response by varying excitation amplitude
"""

import numpy as np
import matplotlib.pyplot as plt
from ROM import N, S, L_b, odefun, mode_shape
from scipy.integrate import solve_ivp

# Simulation parameters
K_c = 6e9
K_p = 0.03
t_end = 1.0
K_i = 2100
j_exc = 30
f0 = 1000
f1 = 5000

def run_sim(A_exc=50, K_c=K_c, K_p=K_p, K_i=K_i):
	"""Run time domain simulation for given amplitude"""
	def v_exc(t):
		return A_exc*np.sin(2*np.pi*(f0 + t*(f1-f0)/t_end) *t)
	
	x0 = np.zeros(2*N + 2*S)
	t_eval = np.arange(0, t_end, 1/f1/20)
	
	sol = solve_ivp(
		lambda t, x: odefun(t, x, v_exc=v_exc, K_c=K_c, K_p=K_p, K_i=K_i),
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
param_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125
# param_list = np.array([0.05, 0.4]) * 125  # Quick test

results = {
	"params": [],
	"param_name": "Amp",
	"t": None,
	"veloc": [],
	"freq": None,
	"Y": [],
	"X": [],
	"FRF": []
}

print(f"Starting amplitude sweep...")
print(f"K_c = {K_c:.2e}, K_p = {K_p}, K_i = {K_i}")

for par in param_list:
	print(f"  Amplitude = {par:.2f} V ...")
	
	# Run simulation
	t, veloc = run_sim(A_exc=par)
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
	def v_exc_func(t):
		return par*np.sin(2*np.pi*(f0 + t*(f1-f0)/t_end) *t)
	results["X"].append(np.fft.fft(v_exc_func(t)))

# Extract frequency and compute FRF
freq = results["freq"]
idx = freq >= 0
freq_pos = freq[idx]

for i, Y in enumerate(results["Y"]):
	X = results["X"][i]
	FRF = np.mean(np.abs(Y), axis=0) / (np.abs(X))
	results["FRF"].append(FRF)

# Save results
np.savez("./sim_dat/amp_sweep.npz", **results)
print("Results saved to sim_dat/amp_sweep.npz")

# Plot results
plt.figure(figsize=(9, 5))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results['params'])))

for i, (par, FRF) in enumerate(zip(results["params"], results["FRF"])):
	plt.semilogy(results["freq"], FRF, color=colors[i], 
	             linewidth=3, label=f"A={par:.1f} V")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF Magnitude")
plt.title(f"Amplitude Sweep (K_c={K_c:.2e}, K_p={K_p})")
plt.xlim([1300, 3000])
plt.ylim([1e-6, 1e-2])
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('sim_dat/amp_sweep.png', dpi=300, bbox_inches='tight')
plt.show()

print("Amplitude sweep complete!")
