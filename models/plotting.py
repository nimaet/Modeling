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