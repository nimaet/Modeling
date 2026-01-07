from ROM import modal_simulation, odefun, mode_shape, N, S, L_b
import ROM
import numpy as np	
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# %% [markdown]
K_i = 2100
param_list = [0.0, 0.01, 0.05] + list(np.arange(0.1, 0.9, 0.1)) + [1000] # K_p sweep

results_freqDomain = {
	"params": [],
	"param_name": "K_p",
	"freq": None,         # same for all sweeps
	"FRF": []             # FRF curves for each amplitude
}
for par in param_list:
    print(f"Running K_p = {par:.2f} ...")
    freq_modal, vel_mag, disp_mag = modal_simulation(K_p=par, K_i=K_i)
    results_freqDomain["params"].append(par)
    results_freqDomain["FRF"].append(vel_mag)
    if results_freqDomain["freq"] is None:
        results_freqDomain["freq"] = freq_modal
        

plt.figure(figsize=(8,6))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results_freqDomain['params'])))   # gradient from low A → high A
for i, (vel_mag, par) in enumerate(zip(results_freqDomain["FRF"], results_freqDomain["params"])):
    line, = plt.plot(results_freqDomain["freq"], vel_mag, label=f'K_p={results_freqDomain["params"][i]:.2f}', color=colors[i])
    if par==0.0:
        line.set_label('Open Circuit');line.set_linestyle('--'); line.set_color('black')
    if par > 900:
        line.set_label('Short Circuit');line.set_linestyle('--'); line.set_color('blue')
    
plt.legend()
plt.xlim([0, 1000])
plt.xlim([1300, 3000])
# plt.xlim([2500, 3000])
plt.ylim([1e-5, 6e-4])
plt.xlabel("Frequency [Hz]")
plt.ylabel("AverageVelocity/Voltage FRF")
plt.grid(True)
plt.show()

# %%


# %% [markdown]
# # NES kp Sweep

# %%


# %%
# ===========================
# Sanity check: linearization
# ===========================



freq_modal, vel_mag, disp_mag = modal_simulation(K_c=0, K_p=0.01, K_i=2100)

npz_path_OC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\OC.npz".replace("\\", "/")
# npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\10\ki0_kc0_kpSweep\parametric_sweep.npz".replace("\\", "/")  
npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\linear.npz".replace("\\", "/")  
# npz_path_OC =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\OC.npz".replace("\\", "/")
npz_path_SC = r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\SC.npz".replace("\\", "/")
data_SC = np.load(npz_path_SC)
data_OC = np.load(npz_path_OC)
data_linear = np.load(npz_path_linear)
frq_OC = data_OC['freq']			# (Nfreq,)
frf_data_OC = data_OC['frf_data']	# (Nfiles, Nfreq, Npoints)
frq_SC = data_SC['freq']            # (Nfreq,)
frf_data_SC = data_SC['frf_data']   # (Nfiles, Nfreq, Npoints)
frq_linear = data_linear['freq']            # (Nfreq,)
frf_data_linear = data_linear['frf_data']

# %matplotlib widget
from IPython.display import clear_output
from matplotlib import cm, colors

import pandas as pd
comsol_OC = pd.read_csv('../../comsol/OC_wide.csv')
plt.figure(figsize=(10,6))


plt.semilogy(frq_linear, np.mean(frf_data_linear[:,:], axis=1), 'r--', label=' Exp.')
# plt.semilogy(comsol_OC['freq'], comsol_OC['w']*2*pi*comsol_OC['freq'], 'g-', label='COMSOL O.C. ')
# plt.semilogy(comsol_OC['freq'], comsol_OC['w'], 'g-', label='COMSOL displacement FRF')
# plt.semilogy(frq_OC, np.mean(frf_data_OC[:,:], axis=1), 'k--', label=f'Open circuit Exp.')
# plt.semilogy(frq_SC, np.mean(frf_data_SC[:,:], axis=1), 'b--', label=f'Short circuit Exp.')
# plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order'   )

# plt.title("Velocity Spectrum at x = x_eval[10]")

plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order O.C.'   )
# plt.semilogy(freq_modal, disp_mag*freq_modal*2*np.pi, '-', label='Modal Reduced Order Displacement $j \omega$'   )

plt.legend()
plt.xlim([1300, 3000])
# plt.xlim([1000, 4500])
plt.ylim([1e-5, 1e-3])
plt.xlabel("Frequency [Hz]")
plt.ylabel("AverageVelocity/Voltage FRF")
plt.grid(True)
plt.show()


# %%

import pandas as pd
df_fr2000 = pd.read_csv('../../comsol/OC.csv')
df_fr2000.head()
df_fr2000['w']


x0 = np.zeros(2*N + 2*S)
t_end = 1
f0 = 1000; f1 = 3000
t_eval = np.arange(0, t_end, 1/f1/20)

K_c = 5e10
K_p = 0.01
K_i = 2100
def v_exc(t, A_exc=50, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)

sol = solve_ivp(
	lambda t, x: odefun(t, x, v_exc=v_exc, K_c= K_c, K_p=K_p, K_i=K_i),
    # odefun,
	(0, t_end),
	x0,
	t_eval=t_eval,
	method='RK45',
	rtol=1e-9,
	atol=1e-10
	)

t = sol.t
eta = sol.y[0:N, :]
eta_dot = sol.y[N:2*N, :]
v = sol.y[2*N:2*N+S, :]

x_eval = np.linspace(0, L_b, 100)
disp = np.zeros([len(x_eval), eta.shape[1]] )
veloc = np.zeros([len(x_eval), eta.shape[1]] )
for i in range(eta.shape[0]):
    disp += np.outer(mode_shape(i, x_eval), eta[i, :])
    veloc += np.outer(mode_shape(i, x_eval), eta_dot[i, :])


# %%

# %matplotlib widget
plt.figure()

# plt.plot(t, disp[10,:], '.-', label='Disp.')
plt.plot(t, veloc[10,:], '-', label='Velocity')
# plt.xlim([0, 0.05 ])
# plt.ylim(np.array([-1,1])*1e-3)

plt.xlabel("t [s]")
# plt.ylabel("displacement")
plt.legend()
plt.grid(True)


# %%
npz_path_OC =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\7\kc0_kp_sweep\OCSC\OC.npz".replace("\\", "/")
# npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\10\ki0_kc0_kpSweep\parametric_sweep.npz".replace("\\", "/")  
npz_path_linear =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\linear.npz".replace("\\", "/")  
# npz_path_OC =r"Z:\Nima\Synthetic_impedance\long_beam\ssdsl_dat\ssdsl_dat_Nov\11\OC.npz".replace("\\", "/")
data_OC = np.load(npz_path_OC)
data_linear = np.load(npz_path_linear)
frq_OC = data_OC['freq']			# (Nfreq,)
frf_data_OC = data_OC['frf_data']	# (Nfiles, Nfreq, Npoints)

frq_linear = data_linear['freq']            # (Nfreq,)
frf_data_linear = data_linear['frf_data']

# %matplotlib widget


# ----- Compute and plot spectrum -----
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

plt.figure()
plt.semilogy(freq, np.mean(np.abs(Y), axis=0)/ np.abs(X), '.-', linewidth=1.5, label='Numerical')

# plt.semilogy(frq_linear, np.mean(frf_data_linear[:,:], axis=1), 'r--', label=' Exp.')
plt.semilogy(frq_OC, np.mean(frf_data_OC[:,:], axis=1), 'k--', label=f'Open circuit Exp.')
plt.semilogy(freq_modal, vel_mag, '.-', label='Modal Reduced Order'   )
plt.xlabel("Frequency [Hz]")
plt.xlim([800, 4500])

plt.ylabel("Amplitude")
# plt.title("Velocity Spectrum at x = x_eval[10]")
plt.grid(True)
plt.legend()


# %% [markdown]
# # kp Sweep

# %%
K_c = 5e11
t_end = 1
K_i = 0.0
def run_sim(K_p):
	# --- local excitation function with this amplitude ---
	
	def v_exc(t, A_exc=50, f0=1000, f1=5000, t_end=t_end):
		return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
	x0 = np.zeros(2*N + 2*S)
	t_eval = np.arange(0, t_end, 1/f1/20)

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

	return sol.t, veloc

param_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125 # Amplitude sweep
param_list = np.arange(0.01, 0.8, 0.05) # K_p sweep
# A_list = np.array([0.05, 0.15, 0.25, 0.4]) * 125
# A_list = np.array([0.05, 0.4]) * 125

results = {
	"params": [],
	"param_name": "K_p",
	"t": None,
	"veloc": [],
	"freq": None,         # same for all sweeps
	"Y": [],              # velocity FFT
	"X": [],            # excitation FFT (same for all sweeps)
	"FRF": []             # FRF curves for each amplitude
}

for par in param_list:
	print(f"{results['param_name']} = {par} ...")

	# --- Run simulation ---
	t, veloc = run_sim(par)    # ensure run_sim returns v_in too
	results["params"].append(par)
	results["veloc"].append(veloc)
	# --- FFT parameters ---
	Nt = len(t)
	dt = t[1] - t[0]
	freq = np.fft.fftfreq(Nt, d=dt)
	# Velocity FFT
	Y = np.fft.fft(veloc, axis=1)
	results["Y"].append(Y)

	# Save time frequency (only once)
	if results["freq"] is None:
		results["freq"] = freq
		results["t"] = t
		
	# Excitation FFT X (only once)
	results["X"].append(np.fft.fft(v_exc(t, A_exc=A_exc)))

# extract freq, X for convenience
freq = results["freq"]
# keep only positive frequencies
idx = freq >= 0
freq_pos = freq[idx]


for i, Y in enumerate(results["Y"]):
	X = results["X"][i]
	X_pos = X[idx]
	Y_pos = Y[:, idx]
	FRF = np.mean(np.abs(Y), axis=0) / (np.abs(X))
	results["FRF"].append(FRF)

# overwrite positive frequencies for clarity
# results["freq"] = freq_pos



np.savez("./sim_dat/kc_sweep.npz", **results)
plt.figure(figsize=(9,5))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results['Params'])))   # gradient from low A → high A
i = -1
for A, FRF in zip(results["Params"], results["FRF"]):
	i +=1
	# plt.semilogy(results["freq"], FRF, linewidth=1.2, label=f"A={A:.2f} V")
	plt.semilogy(results["freq"], FRF, color=colors[i], linewidth=3, label=f"{results['param_name']}={A/125:.2f} V")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF Magnitude")
plt.title(f"kc={K_c:.2e}, kp={K_p:0.3f}")
plt.xlim([500, 5000])
plt.grid(True)
plt.legend()
plt.show()


# %% [markdown]
# # Amplitude Sweep

# %%
K_c = 6e9
K_p = 0.01
t_end = 1
K_i = 2100
def v_exc(t, A_exc=A_exc, f0=1000, f1=5000, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
def run_sim(A_exc=50, K_c=K_c, K_p=K_p, K_i=K_i):
	# --- local excitation function with this amplitude ---
	

	x0 = np.zeros(2*N + 2*S)
	t_eval = np.arange(0, t_end, 1/f1/20)

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

	return sol.t, veloc

param_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125 # Amplitude sweep
# param_list = np.arange(0.01, 0.8, 0.05) # K_p sweep
# A_list = np.array([0.05, 0.15, 0.25, 0.4]) * 125
# A_list = np.array([0.05, 0.4]) * 125

results = {
	"params": [],
	"param_name": "Amp",
	"t": None,
	"veloc": [],
	"freq": None,         # same for all sweeps
	"Y": [],              # velocity FFT
	"X": [],            # excitation FFT (same for all sweeps)
	"FRF": []             # FRF curves for each amplitude
}

for par in param_list:
	print(f"{results['param_name']} = {par} ...")

	# --- Run simulation ---
	t, veloc = run_sim(A_exc=par)    # ensure run_sim returns v_in too
	results["params"].append(par)
	results["veloc"].append(veloc)
	# --- FFT parameters ---
	Nt = len(t)
	dt = t[1] - t[0]
	freq = np.fft.fftfreq(Nt, d=dt)
	# Velocity FFT
	Y = np.fft.fft(veloc, axis=1)
	results["Y"].append(Y)

	# Save time frequency (only once)
	if results["freq"] is None:
		results["freq"] = freq
		results["t"] = t
		
	# Excitation FFT X (only once)
	results["X"].append(np.fft.fft(v_exc(t, A_exc=par)))

# extract freq, X for convenience
freq = results["freq"]
# keep only positive frequencies
idx = freq >= 0
freq_pos = freq[idx]


for i, Y in enumerate(results["Y"]):
	X = results["X"][i]
	X_pos = X[idx]
	Y_pos = Y[:, idx]
	FRF = np.mean(np.abs(Y), axis=0) / (np.abs(X))
	results["FRF"].append(FRF)

# overwrite positive frequencies for clarity
# results["freq"] = freq_pos



# %%


np.savez("./sim_dat/amp_sweep.npz", **results)
plt.figure(figsize=(9,5))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(results['params'])))   # gradient from low A → high A
i = -1
for A, FRF in zip(results["params"], results["FRF"]):
	i +=1
	# plt.semilogy(results["freq"], FRF, linewidth=1.2, label=f"A={A:.2f} V")
	plt.semilogy(results["freq"], FRF, color=colors[i], linewidth=3, label=f"{results['param_name']}={A/125:.2f} V")

plt.xlabel("Frequency [Hz]")
plt.ylabel("FRF Magnitude")
plt.title(f"kc={K_c:.2e}, kp={K_p:0.3f}")
plt.xlim([1300, 3000])
plt.grid(True)
plt.legend()
plt.show()


# %%
plt.figure(figsize=(9,5))
plt.semilogy(freq, np.abs(results["X"][1]), '.', linewidth=2, label='Excitation')
# plt.xlim([500, 10000])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Excitation Spectrum")
plt.grid(True)
plt.legend()
plt.show()

# %%
len(t)

# %%
# Generate a chirp signal
t_end = 0.1
t = np.arange(0, t_end, 1/f1/20)
chirp_signal = v_exc(t, A_exc=A_exc, f0=f0, f1=2000, t_end=t_end)


# Compute and plot spectrum
Nt = len(t)
dt = t[1] - t[0]
freq = np.fft.fftfreq(Nt, d=dt)
X_chirp = np.fft.fft(chirp_signal)

# Take only positive frequencies
idx = freq >= 0
freq_pos = freq[idx]
X_pos = X_chirp[idx]

plt.figure(figsize=(12, 4))
plt.semilogy(freq, np.abs(X_chirp), '.-', linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title(f"Chirp Spectrum: {f0/1e3:.1f}–{f1/1e3:.1f} kHz")
plt.xlim([0, 12000])
plt.grid(True)
plt.show()

# %%
%matplotlib widget
plt.figure()

plt.plot(t, v[30, :])
# plt.xlim([0, 1e-6 ])
# plt.ylim([-1,1])
plt.xlabel("t [s]")
plt.ylabel("eta_1(t)")
plt.grid(True)



# %%

plt.figure()
plt.plot(t, v[20, :])
plt.xlabel("t [s]")
plt.ylabel("v_1(t) [V]")
plt.grid(True)
plt.show()




