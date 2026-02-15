import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from pyparsing import line
import numpy as np
# import pyvista as pv
import imageio.v2 as imageio
from tqdm import trange

def animate_field_1d(
	t,
	u,
	x=None,
	filename=None,
	fps=30,
	stride=1,
	scale=1.0,
	ylabel="Field",
	xlabel="x",
	title="1D Field Animation"
):
	"""
	Animate a 1D space–time field u(t, x).

	Parameters
	----------
	t : (Nt,) array
		Time vector
	u : (Nt, Nx) array
		Field values
	x : (Nx,) array or None
		Spatial coordinates; if None, uses index array
	filename : str or None
		If provided, saves animation as GIF
	fps : int
		Frames per second
	stride : int
		Frame stride
	scale : float
		Visualization scaling
	"""

	assert u.shape[0] == len(t), "u and t size mismatch"

	if x is None:
		x = np.arange(u.shape[1])
	else:
		assert u.shape[1] == len(x), "u and x size mismatch"

	u_plot = scale * u
	frames = range(0, len(t), stride)
	total_frames = len(list(frames))

	def progress_callback(frame, total):
		print(f"\rSaving frame {frame+1}/{total}", end="")

	fig, ax = plt.subplots(figsize=(7, 4))
	line, = ax.plot([], [], lw=2)

	ax.set_xlim(x[0], x[-1])
	ax.set_ylim(1.1*np.min(u_plot), 1.1*np.max(u_plot))
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True)

	def init():
		line.set_data([], [])
		return (line,)
	time_text = ax.text(
		0.02, 0.95, "", transform=ax.transAxes
	)

	def update(frame):
		line.set_data(x, u_plot[frame])
		time_text.set_text(f"t = {t[frame]:.4f} s")
		return (line, time_text)

	# def update(frame):
	# 	line.set_data(x, u_plot[frame])
	# 	ax.set_title(f"{title} — t = {t[frame]:.4f} s")
	# 	return (line,)

	ani = animation.FuncAnimation(
		fig,
		update,
		frames=frames,
		init_func=init,
		blit=True
	)

	if filename is not None:
		# writer = PillowWriter(fps=fps)
		writer = FFMpegWriter(fps=fps, bitrate=2000)
		ani.save(filename, writer=writer,
		   progress_callback=progress_callback
		   )
		print(f"Saved animation to {filename}")

	# plt.show()
	return ani




import numpy as np
# import pyvista as pv
import imageio.v2 as imageio
from tqdm import trange


def animate_field_1d_pyvista(
	t,
	u,
	x=None,
	filename="beam.mp4",
	fps=30,
	stride=1,
	scale=1.0,
	ylabel="Displacement"
):
	
	assert u.shape[0] == len(t), "u and t size mismatch"

	if x is None:
		x = np.arange(u.shape[1])
	else:
		assert u.shape[1] == len(x), "u and x size mismatch"

	u_plot = scale * u

	# ----------------------------
	# Create 2D line geometry
	# ----------------------------
	points = np.column_stack((x, u_plot[0], np.zeros_like(x)))

	line = pv.PolyData(points)
	line = pv.PolyData(points)

	plotter = pv.Plotter()
	plotter.add_mesh(line, color="black", line_width=3)

	plotter.view_xy()
	plotter.enable_parallel_projection()
	plotter.show_axes()
	# plotter.disable_interactivity()
	plotter.off_screen = True

	plotter.open_movie(filename, framerate=fps)
	time_text = plotter.add_text("", position="upper_left")
	
	for k in trange(0, len(t), stride, desc="Saving animation"):
		points[:, 1] = u_plot[k]
		line.points = points
		# time_text.SetText(0, f"t = {t[k]:.4f} s")
		plotter.write_frame()    # save frame
		# plotter.render()

	plotter.close()

	print(f"Saved animation to {filename}")

# ...existing code...

def animate_field_1d_with_envelope(
	t,
	u,
	envelope_func,
	x=None,
	filename=None,
	fps=30,
	stride=1,
	scale=1.0,
	ylabel="Field",
	xlabel="x",
	title="1D Field Animation with Envelope",
	y_lim_scale=1.2
):
	"""
	Animate a 1D space–time field u(t, x) with envelope overlay.

	Parameters
	----------
	t : (Nt,) array
		Time vector
	u : (Nt, Nx) array
		Field values
	envelope_func : callable
		Envelope function with signature envelope_func(x, t) returning complex array
	x : (Nx,) array or None
		Spatial coordinates; if None, uses index array
	filename : str or None
		If provided, saves animation as MP4/GIF
	fps : int
		Frames per second
	stride : int
		Frame stride
	scale : float
		Visualization scaling for field u
	ylabel : str
		Label for y-axis
	xlabel : str
		Label for x-axis
	title : str
		Animation title
	"""

	assert u.shape[0] == len(t), "u and t size mismatch"

	if x is None:
		x = np.arange(u.shape[1])
	else:
		assert u.shape[1] == len(x), "u and x size mismatch"

	u_plot = scale * u
	frames = range(0, len(t), stride)
	total_frames = len(list(frames))

	fig, ax = plt.subplots(figsize=(10, 5))
	line_field, = ax.plot([], [], 'k-', lw=2, label='FE ')
	line_envelope, = ax.plot([], [], 'r--', lw=2, label='Analytical')

	ax.set_xlim(x[0], x[-1])
	y_min = np.min(u_plot[:, 1:-3])
	y_max = np.max(u_plot[:, 1:-3]) 
	ax.set_ylim(y_lim_scale * y_min , y_lim_scale * y_max )
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right')

	def init():
		line_field.set_data([], [])
		line_envelope.set_data([], [])
		return (line_field, line_envelope)

	time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

	def update(frame_idx):
		# Evaluate envelope at all spatial points and current time
		envelope_vals = envelope_func(x, t[frame_idx])
		envelope_plot = np.real(envelope_vals)

		line_field.set_data(x, u_plot[frame_idx])
		line_envelope.set_data(x, envelope_plot)
		time_text.set_text(f"t = {t[frame_idx]:.4f} s")
		return (line_field, line_envelope, time_text)

	ani = animation.FuncAnimation(
		fig,
		update,
		frames=frames,
		init_func=init,
		blit=True
	)
	def progress_callback(frame, total):
		print(f"\rSaving frame {frame+1}/{total}", end="")
	if filename is not None:
		writer = FFMpegWriter(fps=fps, bitrate=2000)
		ani.save(filename, writer=writer, 
		   progress_callback=progress_callback
		   )
		print(f"Saved animation to {filename}")

	return ani


def quick_frf_check(
	K_i: float = 0.0,
	K_p: float = 1e-5,
	K_c: float = 0.0,
	R_c: float = 1e3,
	f0: float = 800.0,
	f1: float = 4500.0,
	t_end: float = 0.1,
	A_exc: float = 50.0,
	j_exc: int = 30,
	hp: float = 0.252e-3,
	hs: float = 0.51e-3,
	d31: float = -1.48e-10,
	eps_r: float = 1700.0,
	zeta_p: float = 0.0151 * 8,
	zeta_q: float = 0.0392 * 10,
	h_patch: float = 2e-3,
	h_gap: float = 5e-3,
	n_freq: int = 500,
	plot: bool = True
):
	import numpy as np
	import sys
	from pathlib import Path
	import importlib
	import matplotlib.pyplot as plt

	# project path
	project_root = Path.cwd().parents[2]
	sys.path.append(str(project_root))

	import Modeling
	import Modeling.models.FE2 as FE_module
	from Modeling.models.beam_properties import PiezoBeamParams
	from Modeling.models import FE_helpers

	importlib.reload(Modeling)
	importlib.reload(FE_module)

	# excitation
	dt = 1 / f1 / 20

	def v_exc(t):
		return A_exc * np.sin(2 * np.pi * (f0 + t * (f1 - f0) / t_end) * t)

	# FE parameters
	params_fe = PiezoBeamParams(
		hp=hp,
		hs=hs,
		d31=d31,
		eps_r=eps_r
	)

	params_fe.zeta_p = zeta_p
	params_fe.zeta_q = zeta_q

	# geometry
	geom = FE_module.geometry_from_params(
		params=params_fe,
		h_patch=h_patch,
		h_gap=h_gap
	)

	params_fe.geometry = geom

	# FE model
	fe = FE_module.PiezoBeamFE(params_fe)

	ode = fe.build_ode_system(
		j_exc=j_exc,
		K_c=K_c,
		K_i=K_i,
		K_p=K_p,
		v_exc=v_exc
	)

	# frequency-domain FRF
	f_fe = np.linspace(f0, f1, n_freq)
	frf_fd = FE_helpers.frf_sweep(ode, f_fe * 2 * np.pi)

	w_dot_fd = frf_fd['u_dot']
	freq_fd = frf_fd['freq']
	vel_fd = np.mean(np.abs(w_dot_fd), axis=1)

	# time-domain solve
	ndof = ode.M.shape[0]
	x0 = np.zeros(ndof)
	x_dot0 = np.zeros(ndof)

	result = FE_helpers.solve_newmark(
		ode=ode,
		dt=dt,
		t_end=t_end,
		beta=0.25,
		gamma=0.5,
		newton_tol=1e-8,
		newton_maxiter=8,
		x0=x0,
		x_dot0=x_dot0
	)

	spec_td = result['spectral']

	if spec_td is not None and spec_td['freq'] is not None:
		freq_td = spec_td['freq']
		frf_td = spec_td['FRF']
	else:
		freq_td = frf_td = None

	# plot
	if plot:
		plt.figure(figsize=(10, 4))
		plt.semilogy(freq_fd, vel_fd, 'k-', lw=2, label='FE freq-domain')
		if freq_td is not None:
			plt.semilogy(freq_td, frf_td, '.-', label='FE time-domain')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('FRF magnitude')
		plt.xlim([f0, f1])
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		plt.show()

	return {
		'freq_fd': freq_fd,
		'vel_fd': vel_fd,
		'freq_td': freq_td,
		'frf_td': frf_td,
		'ode': ode,
		'result_td': result
	}
