import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


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

	fig, ax = plt.subplots(figsize=(7, 4))
	line, = ax.plot([], [], lw=2)

	ax.set_xlim(x[0], x[-1])
	ax.set_ylim(1.2*np.min(u_plot), 1.2*np.max(u_plot))
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True)

	def init():
		line.set_data([], [])
		return (line,)

	def update(frame):
		line.set_data(x, u_plot[frame])
		ax.set_title(f"{title} — t = {t[frame]:.4f} s")
		return (line,)

	ani = animation.FuncAnimation(
		fig,
		update,
		frames=frames,
		init_func=init,
		blit=True
	)

	if filename is not None:
		writer = PillowWriter(fps=fps)
		ani.save(filename, writer=writer)
		print(f"Saved animation to {filename}")

	plt.show()
	return ani
