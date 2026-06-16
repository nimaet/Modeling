"""Class-based plotting helpers for piezo patch optimization results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..optimizer.optimizer_helpers import (
    OBJECTIVE_MULTI_MODE,
    OBJECTIVE_SINGLE_MODE,
    OBJECTIVE_TRAVELING_WAVE,
    OUTPUT_TIP,
)
from ..optimizer.optimizer_settings import PostProcessingSettings
from .postprocess import PostProcessor


class OptimizerPlotter:
    """Stateful plotter for optimizer and saved sweep results."""

    def __init__(
        self,
        optimizer=None,
        postprocess_settings: PostProcessingSettings | None = None,
    ):
        self.optimizer = optimizer
        self.postprocess_settings = postprocess_settings or PostProcessingSettings()

    # ------------------------------------------------------------------
    # Basic Result Plots
    # ------------------------------------------------------------------

    def plot_layout(self, layout: dict, L: float, ax=None):
        """Plot beam layout with shaded active piezo spans."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 1.8))
        else:
            fig = ax.get_figure()

        ax.plot([0, L], [0, 0], "k-", lw=3, label="beam")
        for j, (a, b) in enumerate(zip(layout["xL"], layout["xR"])):
            ax.axvspan(a, b, alpha=0.35, label="active piezo" if j == 0 else None)
            ax.text(0.5 * (a + b), 0.02, f"P{j + 1}", ha="center", va="bottom")
        for x in layout["x_starts"]:
            ax.axvline(x, color="0.7", lw=0.8, ls="--")

        ax.set_xlim(0, L)
        ax.set_yticks([])
        ax.set_xlabel("x [m]")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, ax

    def plot_metric_frequency_response(
        self,
        response: dict,
        natural_freq_hz: float | None = None,
        ax=None,
        label: str | None = None,
    ):
        """Semilog plot of a scalar output metric from a dense frequency response."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        y = response.get("metric", response.get("tip_disp"))
        ylabel = response.get("metric_label", "Output magnitude [m/V]")
        response_label = response.get("response_label", "Frequency response")
        ax.semilogy(response["freq"], y, lw=2, label=label or response.get("output", "frequency response"))
        if natural_freq_hz is not None:
            ax.axvline(natural_freq_hz, color="k", ls="--", lw=1, label=f"natural freq: {natural_freq_hz:.3g} Hz")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(response_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def plot_thrust_frequency_response(
        self,
        response: dict,
        natural_freq_hz: float | None = None,
        ax=None,
    ):
        """Plot postprocessed mean thrust estimate over frequency."""
        if response.get("thrust") is None:
            raise ValueError("response does not contain thrust data")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        ax.plot(response["freq"], response["thrust"], lw=2)
        if natural_freq_hz is not None:
            ax.axvline(natural_freq_hz, color="k", ls="--", lw=1)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(response.get("thrust_label", "Mean thrust estimate [N]"))
        ax.set_title("Lighthill quiescent-water thrust estimate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax
    
    def plot_all_binary_frequency_responses(
        self,
        all_results: list[dict],
        natural_freq_hz: float | None = None,
        ax=None,
        use_metric: bool = False,
    ):
        """Semilog plot of all binary sign-pattern frequency responses."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        for result in all_results:
            y = result["metric"] if use_metric and "metric" in result else result["tip_disp"]
            ax.semilogy(result["freq"], y, lw=1.2, label=result.get("label", None))
        if natural_freq_hz is not None:
            ax.axvline(natural_freq_hz, color="k", ls="--", lw=1)
        ax.set_xlabel("Frequency [Hz]")
        if use_metric and all_results:
            ax.set_ylabel(all_results[0].get("metric_label", "Output magnitude [m/V]"))
        else:
            ax.set_ylabel("Tip displacement magnitude [m/V]")
        if all_results:
            ax.set_title(all_results[0].get("response_label", "Frequency responses"))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig, ax

    def plot_all_binary_bar(self, all_results: list[dict], ax=None, metric_key: str = "score"):
        """Compare all binary sign patterns at one frequency."""
        if all_results is None:
            raise ValueError(
                "all_results is None. Pass mode_result['all_phase_results'] "
                "or best['inner']['all_phase_results'] for single-mode cases."
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        labels = [result.get("label", "") for result in all_results]
        scores = [float(result[metric_key]) for result in all_results]
        ax.bar(labels, scores)
        ax.set_xlabel("Binary signs")
        ax.set_ylabel("Output metric at natural frequency [m/V]")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Phase Plots
    # ------------------------------------------------------------------

    @staticmethod
    def phase_deg_from_inputs(phase_deg=None, phase_rad=None, voltage_vector=None):
        if phase_deg is not None:
            return np.asarray(phase_deg, dtype=float) % 360.0
        if phase_rad is not None:
            return np.rad2deg(np.asarray(phase_rad, dtype=float)) % 360.0
        if voltage_vector is not None:
            return np.rad2deg(np.angle(np.asarray(voltage_vector, dtype=complex))) % 360.0
        raise ValueError("Provide phase_deg, phase_rad, or voltage_vector")

    def plot_patch_phases(
        self,
        layout: dict,
        phase_deg=None,
        phase_rad=None,
        voltage_vector=None,
        ax=None,
        annotate: bool = True,
    ):
        """Show each patch's phase as a bar positioned at the physical patch location."""
        phase = self.phase_deg_from_inputs(phase_deg=phase_deg, phase_rad=phase_rad, voltage_vector=voltage_vector)
        if phase.ndim != 1:
            raise ValueError("plot_patch_phases expects a 1D phase vector.")

        xL = np.asarray(layout["xL"], dtype=float)
        xR = np.asarray(layout["xR"], dtype=float)
        centers = 0.5 * (xL + xR)
        widths = xR - xL

        if len(phase) != len(centers):
            raise ValueError(f"Number of phases ({len(phase)}) does not match number of patches ({len(centers)})")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 3.5))
        else:
            fig = ax.get_figure()

        ax.bar(centers, phase, width=0.9 * widths, align="center", edgecolor="k", alpha=0.75)
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Patch phase [deg]")
        ax.grid(True, axis="y", alpha=0.3)

        if annotate:
            for j, (x, p) in enumerate(zip(centers, phase)):
                ax.text(x, min(p + 8, 350), f"P{j + 1}\n{p:.1f} deg", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        return fig, ax

    def plot_phase_phasors(
        self,
        phase_deg=None,
        phase_rad=None,
        voltage_vector=None,
        ax=None,
        annotate: bool = True,
    ):
        """Polar phasor plot for patch voltage phases."""
        if voltage_vector is not None:
            v = np.asarray(voltage_vector, dtype=complex)
            theta = np.angle(v)
            radius = np.abs(v)
        else:
            phase = self.phase_deg_from_inputs(phase_deg=phase_deg, phase_rad=phase_rad)
            if phase.ndim != 1:
                raise ValueError("plot_phase_phasors expects a 1D phase vector.")
            theta = np.deg2rad(phase)
            radius = np.ones_like(theta)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        else:
            fig = ax.get_figure()

        for j, (th, r) in enumerate(zip(theta, radius)):
            ax.plot([0, th], [0, r], marker="o", lw=2, label=f"P{j + 1}")
            if annotate:
                ax.text(th, r * 1.08, f"P{j + 1}", ha="center", va="center")
        ax.set_title("Patch voltage phasors")
        ax.legend(loc="best", bbox_to_anchor=(1.1, 1.1), fontsize=8)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Objective-Specific Plots
    # ------------------------------------------------------------------

    def plot_traveling_wave_shape(self, inner: dict, ax=None, n_time: int = 8):
        """Plot envelope and time snapshots for a traveling-wave optimizer result."""
        if inner.get("objective") != OBJECTIVE_TRAVELING_WAVE:
            raise ValueError("plot_traveling_wave_shape requires a traveling-wave inner result")

        metrics = inner["traveling_wave_metrics"]
        x = np.asarray(metrics.get("x_full", metrics["x"]), dtype=float)
        W = np.asarray(metrics.get("W_full", metrics["W"]), dtype=complex)
        amp = np.abs(W)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        ax.plot(x, amp, "k--", lw=1.5, label="envelope")
        ax.plot(x, -amp, "k--", lw=1.5)

        phases = np.linspace(0.0, 2.0 * np.pi, int(n_time), endpoint=False)
        for phi in phases:
            ax.plot(x, np.real(W * np.exp(1j * phi)), lw=1.0, alpha=0.75)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("Transverse displacement [m/V]")
        ax.set_title(
            "Traveling wave: "
            f"T_i={metrics['traveling_index']:.3f}, "
            f"A_rms={metrics['amplitude_rms']:.3e}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def plot_traveling_wave_metric_sweep(self, sweep: dict, ax=None):
        """Plot traveling-wave objective score and traveling index over frequency."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        freq = np.asarray(sweep["freq"], dtype=float)
        ax.plot(freq, sweep["score"], lw=2, label="objective score")
        if not np.array_equal(np.asarray(sweep["score"]), np.asarray(sweep["traveling_index"])):
            ax.plot(freq, sweep["traveling_index"], lw=1.5, label="traveling index")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Dimensionless score")
        ax.set_title(sweep.get("response_label", "Traveling-wave frequency sweep"))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def plot_multimode_phase_matrix(
        self,
        inner: dict,
        ax=None,
        relative: bool = True,
        annotate: bool = True,
    ):
        """Visualize optimized phase of each patch for each target mode."""
        if inner.get("objective") != OBJECTIVE_MULTI_MODE:
            raise ValueError("plot_multimode_phase_matrix requires a multi-mode inner result")

        key = "relative_phase_deg" if relative else "phase_deg"
        phase = np.vstack([np.asarray(v, dtype=float) for v in inner[key]]) % 360.0
        mode_numbers = list(inner["multi_mode_numbers"])

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * phase.shape[1]), max(2.5, 0.6 * phase.shape[0] + 1.5)))
        else:
            fig = ax.get_figure()

        im = ax.imshow(phase, aspect="auto", vmin=0, vmax=360)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Phase [deg]")
        ax.set_xticks(np.arange(phase.shape[1]))
        ax.set_xticklabels([f"P{j + 1}" for j in range(phase.shape[1])])
        ax.set_yticks(np.arange(phase.shape[0]))
        ax.set_yticklabels([f"Mode {m}" for m in mode_numbers])
        ax.set_xlabel("Patch")
        ax.set_ylabel("Target mode")
        ax.set_title("Optimized patch phases" + (" (relative)" if relative else ""))

        if annotate:
            for i in range(phase.shape[0]):
                for j in range(phase.shape[1]):
                    ax.text(j, i, f"{phase[i, j]:.0f} deg", ha="center", va="center", fontsize=9)

        fig.tight_layout()
        return fig, ax

    def plot_multimode_score_bar(self, inner: dict, ax=None):
        """Bar plot of raw and weighted per-mode scores for a multi-mode objective."""
        if inner.get("objective") != OBJECTIVE_MULTI_MODE:
            raise ValueError("plot_multimode_score_bar requires a multi-mode inner result")

        modes = list(inner["multi_mode_numbers"])
        raw = np.asarray(inner["raw_mode_scores"], dtype=float)
        weighted = np.asarray(inner["weighted_mode_scores"], dtype=float)
        x = np.arange(len(modes))
        width = 0.38

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        ax.bar(x - width / 2, raw, width, label="raw")
        ax.bar(x + width / 2, weighted, width, label="weighted")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Mode {m}" for m in modes])
        ax.set_ylabel("Score contribution")
        ax.set_title(f"Multi-mode score = {inner['score']:.3e}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Mode Shape and Curvature
    # ------------------------------------------------------------------

    @staticmethod
    def hermite_mode_shape_and_curvature(fe, mode_number: int, points_per_element: int = 25):
        """Evaluate mode shape and Euler-Bernoulli curvature using Hermite shape functions."""
        if mode_number < 1 or mode_number > fe.Phi.shape[1]:
            raise ValueError("mode_number outside available range")
        phi = fe.Phi[:, mode_number - 1]
        x_nodes = np.asarray(fe.geom.x_nodes, dtype=float)

        xs_all = []
        w_all = []
        kappa_all = []
        for e in range(len(x_nodes) - 1):
            x0, x1 = x_nodes[e], x_nodes[e + 1]
            Le = x1 - x0
            if Le <= 0:
                continue
            xi = np.linspace(0.0, 1.0, points_per_element)
            if e > 0:
                xi = xi[1:]
            x = x0 + xi * Le

            w0 = phi[2 * e]
            th0 = phi[2 * e + 1]
            w1 = phi[2 * (e + 1)]
            th1 = phi[2 * (e + 1) + 1]

            N1 = 1 - 3 * xi**2 + 2 * xi**3
            N2 = Le * (xi - 2 * xi**2 + xi**3)
            N3 = 3 * xi**2 - 2 * xi**3
            N4 = Le * (-xi**2 + xi**3)
            w = N1 * w0 + N2 * th0 + N3 * w1 + N4 * th1

            d2N1 = (-6 + 12 * xi) / Le**2
            d2N2 = (-4 + 6 * xi) / Le
            d2N3 = (6 - 12 * xi) / Le**2
            d2N4 = (-2 + 6 * xi) / Le
            kappa = d2N1 * w0 + d2N2 * th0 + d2N3 * w1 + d2N4 * th1

            xs_all.append(x)
            w_all.append(w)
            kappa_all.append(kappa)

        return np.concatenate(xs_all), np.concatenate(w_all), np.concatenate(kappa_all)

    def plot_mode_shape_and_curvature(
        self,
        fe,
        layout: dict | None = None,
        mode_number: int = 1,
        points_per_element: int = 25,
    ):
        """Plot mode shape and curvature with optional patch overlays."""
        x, w, kappa = self.hermite_mode_shape_and_curvature(fe, mode_number, points_per_element)

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(x, w, lw=2)
        axes[0].set_ylabel(f"Mode {mode_number} shape")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, kappa, lw=2)
        axes[1].axhline(0.0, color="0.5", lw=0.8)
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel(r"Curvature $d^2\phi/dx^2$")
        axes[1].grid(True, alpha=0.3)

        if layout is not None:
            for ax in axes:
                for a, b in zip(layout["xL"], layout["xR"]):
                    ax.axvspan(a, b, alpha=0.15)

        fig.tight_layout()
        return fig, axes
    
    def plot_current_frequency_response(
        self,
        current_response: dict,
        natural_freq_hz: float | None = None,
        ax=None,
    ):
        """Plot patch current magnitudes over frequency."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        freq = np.asarray(current_response["freq"], dtype=float)
        current_mag = np.asarray(current_response["current_magnitude"], dtype=float)

        for j in range(current_mag.shape[1]):
            ax.semilogy(freq, current_mag[:, j], lw=1.8, label=f"P{j + 1}")

        if natural_freq_hz is not None:
            ax.axvline(natural_freq_hz, color="k", ls="--", lw=1)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(current_response.get("current_label", "Patch current magnitude [A]"))
        ax.set_title(current_response.get("response_label", "Current frequency response"))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax
    
    def plot_electrical_power_frequency_response(
        self,
        current_response: dict,
        ax=None,
    ):
        """Plot real and apparent electrical power from peak phasors."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()

        freq = np.asarray(current_response["freq"], dtype=float)
        ax.plot(freq, current_response["real_power"], lw=2, label="real power")
        ax.plot(freq, current_response["apparent_power"], lw=1.5, label="apparent power")

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(current_response.get("power_label", "Electrical power [W]"))
        ax.set_title(current_response.get("response_label", "Electrical power frequency response"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Full Optimizer Result Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def is_multimode_inner(inner: dict) -> bool:
        return inner.get("objective", OBJECTIVE_SINGLE_MODE) == OBJECTIVE_MULTI_MODE

    @staticmethod
    def is_traveling_wave_inner(inner: dict) -> bool:
        return inner.get("objective") == OBJECTIVE_TRAVELING_WAVE

    @staticmethod
    def mode_results_from_inner(inner: dict) -> list[dict]:
        if "mode_results" in inner and inner["mode_results"] is not None:
            return list(inner["mode_results"])
        return [inner]

    def select_mode_results(self, inner: dict, mode="all") -> list[dict]:
        """Select mode result dictionaries by list index, mode number, or all."""
        mode_results = self.mode_results_from_inner(inner)
        if mode == "all":
            return mode_results

        if isinstance(mode, int):
            if 0 <= mode < len(mode_results):
                return [mode_results[mode]]
            matches = [result for result in mode_results if int(result.get("mode_number", -999)) == int(mode)]
            if matches:
                return matches

        raise ValueError("mode must be 'all', a mode_result index, or an actual mode_number")

    def plot_optimizer_result(
        self,
        *,
        best: dict,
        L: float,
        optimizer=None,
        mode="all",
        plot_layout_flag=True,
        plot_phase_flag=True,
        plot_frequency_response_flag=True,
        plot_binary_flag=True,
        plot_curvature_flag=True,
        plot_multimode_summary=True,
        plot_traveling_wave_flag=True,
        plot_current_response_flag=False,
        show=True,
    ):
        """Plot the standard summary figures for an inspected optimizer result."""
        optimizer = optimizer or self.optimizer
        layout = best["layout"]
        inner = best["inner"]
        fe = best.get("fe")
        postprocessor = PostProcessor(optimizer, fe, self.postprocess_settings) if optimizer is not None and fe is not None else None

        is_multi = self.is_multimode_inner(inner)
        is_traveling = self.is_traveling_wave_inner(inner)
        mode_results = self.select_mode_results(inner, mode=mode)
        figures = {}

        def remember(name: str, fig):
            figures[name] = fig
            return fig

        print("=" * 80)
        print("Plotting optimizer result")
        print("Objective:", inner.get("objective", "single_mode"))
        print("Total score:", inner.get("score"))
        print("Output:", inner.get("output", None))
        print("Phase mode:", inner.get("phase_mode", None))
        print("Patch xL [mm]:", 1e3 * np.asarray(layout["xL"]))
        print("Patch xR [mm]:", 1e3 * np.asarray(layout["xR"]))

        if plot_layout_flag:
            fig, _ax = self.plot_layout(layout, L)
            remember("layout", fig)

        if is_multi and plot_multimode_summary:
            fig, _ax = self.plot_multimode_score_bar(inner)
            remember("multimode_scores", fig)
            fig, _ax = self.plot_multimode_phase_matrix(inner, relative=True)
            remember("multimode_phase_matrix", fig)

        if is_traveling and plot_traveling_wave_flag:
            fig, _ax = self.plot_traveling_wave_shape(inner)
            remember("traveling_wave_shape", fig)
            if postprocessor is not None:
                sweep = postprocessor.dense_traveling_wave_metrics_for_plot(inner["voltage_vector"])
                fig, _ax = self.plot_traveling_wave_metric_sweep(sweep)
                remember("traveling_wave_metric_sweep", fig)

        for mode_result in mode_results:
            mode_number = int(mode_result.get("mode_number", 1))
            freq_hz = float(mode_result["freq_hz"])
            best_response = None

            if is_traveling:
                print("-" * 80)
                print("Traveling-wave optimized frequency [Hz]:", f"{freq_hz:.6g}")
            else:
                print("-" * 80)
                print(f"Mode {mode_number}")
                print(f"Natural frequency [Hz]: {freq_hz:.6g}")

            print(f"Mode score: {mode_result['score']:.6e}")
            print("Phase [deg]:", np.asarray(mode_result.get("phase_deg", [])))
            print("Relative phase [deg]:", np.asarray(mode_result.get("relative_phase_deg", [])))

            if plot_phase_flag:
                fig, _ax = self.plot_patch_phases(layout, phase_deg=mode_result["relative_phase_deg"])
                remember(f"mode_{mode_number}_patch_phases", fig)
                fig, _ax = self.plot_phase_phasors(voltage_vector=mode_result["voltage_vector"])
                remember(f"mode_{mode_number}_phase_phasors", fig)

            if plot_frequency_response_flag and postprocessor is not None and not is_traveling:
                best_response = postprocessor.dense_metric_frequency_response_for_mode_result(mode_result)
                fig, _ax = self.plot_metric_frequency_response(
                    best_response,
                    natural_freq_hz=freq_hz,
                    label=f"mode {mode_number} optimized phase",
                )
                remember(f"mode_{mode_number}_frequency_response", fig)
            elif plot_frequency_response_flag and postprocessor is not None and is_traveling:
                best_response = postprocessor.dense_metric_frequency_response_for_mode_result(
                    {**mode_result, "output": OUTPUT_TIP},
                )
                fig, _ax = self.plot_metric_frequency_response(
                    best_response,
                    natural_freq_hz=freq_hz,
                    label="traveling-wave optimized phase",
                )
                remember("traveling_wave_tip_frequency_response", fig)

            if best_response is not None and best_response.get("thrust") is not None:
                fig, _ax = self.plot_thrust_frequency_response(best_response, natural_freq_hz=freq_hz)
                remember(f"mode_{mode_number}_thrust_response", fig)
                
            if plot_binary_flag and mode_result.get("phase_mode", inner.get("phase_mode")) == "binary":
                if postprocessor is not None:
                    all_binary = postprocessor.dense_all_binary_metric_frequency_responses_for_plot(
                        output=mode_result.get("output", inner.get("output", OUTPUT_TIP)),
                    )
                    fig, _ax = self.plot_all_binary_frequency_responses(all_binary, natural_freq_hz=freq_hz, use_metric=True)
                    remember(f"mode_{mode_number}_all_binary_frequency_responses", fig)
                if mode_result.get("all_phase_results", None) is not None:
                    fig, _ax = self.plot_all_binary_bar(mode_result["all_phase_results"])
                    remember(f"mode_{mode_number}_all_binary_scores", fig)

            if plot_curvature_flag and fe is not None and not is_traveling:
                fig, _axes = self.plot_mode_shape_and_curvature(fe, layout, mode_number=mode_number)
                remember(f"mode_{mode_number}_shape_curvature", fig)

            if plot_current_response_flag and postprocessor is not None:
                current_response = postprocessor.dense_current_frequency_response_for_plot(
                    mode_result["voltage_vector"]
                )
                fig, _ax = self.plot_current_frequency_response(
                    current_response,
                    natural_freq_hz=freq_hz,
                )
                remember(f"mode_{mode_number}_current_response", fig)

                fig, _ax = self.plot_electrical_power_frequency_response(current_response)
                remember(f"mode_{mode_number}_electrical_power", fig)

        if show:
            plt.show()
        return figures

    def plot_record(self, record: dict, L: float, mode="all", **kwargs):
        """Plot directly from one record returned by ``run_sweep``."""
        optimizer = record.get("optimizer", self.optimizer)
        return self.plot_optimizer_result(
            optimizer=optimizer,
            best=record["best"],
            L=L,
            mode=mode,
            **kwargs,
        )

    def plot_after_optimization(self, optimizer, result, L: float, mode="all", **kwargs):
        """Inspect and plot a completed optimization result."""
        return self.plot_optimizer_result(
            optimizer=optimizer,
            best=optimizer.inspect_result(result),
            L=L,
            mode=mode,
            **kwargs,
        )
