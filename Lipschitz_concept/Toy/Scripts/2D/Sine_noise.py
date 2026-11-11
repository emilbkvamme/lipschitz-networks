# GOAL: How much plane-wave (sine) noise do the Lipschitz networks remove from a 2D Gaussian, as a function of L?
# (Script version of the plane-wave section of Toy_gaussian_2d.ipynb.)

# HOW: The target is a wide Gaussian with plane waves on top, y(x) = f(x) + sum_j A_j sin(k_j . x).
# A plane wave needs the slope A ||k||_dual, where ||.||_dual is the dual of the networks' input norm
# (l2 -> ||k||_2, l1 -> ||k||_inf). If the bound is what removes the waves, the fraction of a wave that survives
# depends only on A ||k||_dual / L.
#   1. The data: the Gaussian, the noise and the noisy Gaussian, with their spectra
#   2. Networks for a few L: the fits, cuts and spectra, and the fraction of every wave that survives
#   3. L scan: MSE to the clean Gaussian and the noise removed, against L, with the unconstrained network

# PLOTS: in Plots/2D/Sine_noise


import os
import sys
import pickle
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import REPORT_STYLE, noise_removed, plot_dir, results_dir
from Toy_nd_utils import *

QUICK = False                      # True: small grid, short training, only to check that everything runs
PLOT_DIR = plot_dir("2D", "Sine_noise")
RESULTS_FILE = os.path.join(results_dir("2D", "Sine_noise"), "networks_quick.pkl" if QUICK else "networks.pkl")
LOAD_SAVED = True                  # True: use the saved networks if they exist, instead of training again
INPUT_NORM = "l2"                  # the norm of the networks (the notebook used the default, l1)

# The data, as in the notebook (but with the isotropic Gaussian its text describes)
DIM = 2
N_POINTS = 40 if QUICK else 160    # points per axis
X_MIN, X_MAX = -5, 5
SIGMA = 1.0
AMPLITUDES = [0.5, 0.3, 0.7, 0.4, 0.6]
MODES = [(12, 0), (25, 25), (0, 30), (13, 26), (20, 15)]   # whole numbers of periods across the grid along x_1 and x_2

# Networks, as in the notebook
L_VALUES = [0.1, 0.25, 0.5, 1.0]   # the networks shown in detail. The Gaussian needs 0.1, the waves 4 to 13
L_RATIOS_SCAN = np.logspace(-1.5, 0, 3 if QUICK else 12)   # the L scan, as L / L_max (L_max: slope of the noisy data)
SEEDS = [0, 1] if QUICK else [0, 1, 2]
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5

# Figure format
PANEL_SIZE = (3.1, 2.8)
C_NET, C_REF, C_MUTED = "tab:orange", "#9a9893", "#6b6a66"


#### CALCULATIONS ####
def make_data():
    """
    The clean Gaussian, the plane-wave noise and the noisy Gaussian on a periodic grid (no double end point),
    so that every wave sits exactly on one Fourier mode. Returns axes, X, clean, noise, noisy and the wave vectors.
    """
    axes, X = make_grid((X_MAX - X_MIN) / 2, N_POINTS, DIM, endpoint=False)
    clean = gaussian_nd(X, np.zeros(DIM), isotropic_cov(SIGMA, DIM))
    wavevectors = [wavevector(modes, X_MAX - X_MIN) for modes in MODES]
    noise = plane_wave_noise(X, AMPLITUDES, wavevectors)
    return axes, X, clean, noise, clean + noise, wavevectors


def dual_norm(v, norm):
    """The dual of the input norm, along the last axis: the norm the gradient is bounded in."""
    if norm == "l1":
        return np.max(np.abs(v), axis=-1)
    if norm == "l2":
        return np.linalg.norm(v, axis=-1)
    return np.sum(np.abs(v), axis=-1)       # linf


def wave_slope(A, k, norm):
    """The slope the wave A sin(k . x) needs in the networks' norm: A ||k||_dual."""
    return A * dual_norm(np.asarray(k), norm)


def largest_slope(f, axes, norm):
    """The largest slope of the grid function f in the networks' norm (central differences)."""
    return np.max(dual_norm(grid_gradient(f, axes), norm))


def train_networks(x, y, L_values, shape):
    """
    Lipschitz networks for every L and seed, and the unconstrained network for every seed.
    Returns predictions[L] with shape (seeds, grid) and predictions_unconstrained with shape (seeds, grid).
    """
    jobs = []
    for L in L_values:
        for seed in SEEDS:
            jobs.append(dict(kind="lipschitz", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x, y=y,
                             model_kwargs=dict(input_dim=DIM, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                               depth=DEPTH, lipschitz_bound=L, input_norm=INPUT_NORM)))
    for seed in SEEDS:
        jobs.append(dict(kind="unconstrained", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x, y=y,
                         model_kwargs=dict(input_dim=DIM, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)))
    print(f"Training {len(jobs)} networks on {len(x)} points")
    results = iter(train_many(jobs))

    predictions = {}
    for L in L_values:
        predictions[L] = np.array([next(results)["prediction"].reshape(shape) for _ in SEEDS])
    predictions_unconstrained = np.array([next(results)["prediction"].reshape(shape) for _ in SEEDS])
    return predictions, predictions_unconstrained


def surviving_fractions(axes, predictions, A, k):
    """The amplitude of the wave in every prediction divided by its amplitude in the data, one value per seed."""
    return np.array([mode_amplitude_nd(axes, prediction, k) / A for prediction in predictions])


def print_summary(axes, clean, noise, predictions, predictions_unconstrained, wavevectors):
    """The surviving fraction of every wave, and the MSE to the clean Gaussian and the noise removed, per network."""
    print("Fraction of each wave that survives (mean ± std over seeds)")
    for L, runs in [*predictions.items(), ("Unconstrained", predictions_unconstrained)]:
        print(f"{'L = ' + format(L, 'g') if L != 'Unconstrained' else L}:")
        for A, k in zip(AMPLITUDES, wavevectors):
            fractions = surviving_fractions(axes, runs, A, k)
            needed = f", needs {wave_slope(A, k, INPUT_NORM) / L:6.2f} L" if L != "Unconstrained" else ""
            print(f"   k = ({k[0]:5.2f}, {k[1]:5.2f}){needed}: {fractions.mean():.2f} ± {fractions.std():.2f}")

    print(f"\nMSE to the clean Gaussian (noisy data itself: {np.mean(noise ** 2):.2e}) and noise removed")
    for L, runs in [*predictions.items(), ("Unconstrained", predictions_unconstrained)]:
        name = f"L = {L:g}" if L != "Unconstrained" else L
        mse = [np.mean((p - clean) ** 2) for p in runs]
        removed = [100 * noise_removed(p, clean, noise) for p in runs]
        print(f"   {name:14s}: MSE {np.mean(mse):.2e} ± {np.std(mse):.1e}, "
              f"noise removed {np.mean(removed):5.1f} ± {np.std(removed):.1f} %")


#### PLOTS ####
def plot_noise_inputs(axes, clean, noise, noisy, wavevectors, file_name, k_max=40):
    """
    Rows: the Gaussian, the noise and the noisy Gaussian. Columns: the function, its 2D Fourier spectrum (one dot per
    mode, darker and larger = stronger, log scale) and the radial spectrum (amplitude against |k|).
    The wave vectors are marked in red.
    """
    rows = [(clean, "Gaussian", "black"), (noise, "Noise", "orange"), (noisy, "Gaussian + noise", "grey")]
    extent = [axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]]

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(3, 3, figsize=(6.3, 1.9 * 3), gridspec_kw=dict(hspace=0.35, wspace=0.7))
        for i, (f, name, color) in enumerate(rows):
            axs[i, 0].imshow(f.T, origin="lower", extent=extent, cmap="viridis")
            axs[i, 0].set_ylabel(f"{name}\n" + r"$x_2$")

            K, coefficients = fourier_modes(axes, f)
            shown = np.all(np.abs(K) <= k_max, axis=-1)
            log_amplitude = np.log10(np.maximum(2 * coefficients[shown], 1e-8))
            order = np.argsort(log_amplitude)                    # the strongest modes last, on top
            dot_size = 2 + 40 * np.clip((log_amplitude[order] + 6) / 6, 0, 1) ** 3
            axs[i, 1].scatter(K[shown][order, 0], K[shown][order, 1], c=log_amplitude[order], s=dot_size,
                              vmin=-6, vmax=0, cmap="Greys", linewidths=0)
            axs[i, 1].set_xlim(-k_max, k_max)
            axs[i, 1].set_ylim(-k_max, k_max)
            axs[i, 1].set_aspect("equal")
            axs[i, 1].set_ylabel(r"$k_2$")

            axs[i, 2].plot(*radial_spectrum(axes, f), color=color, lw=0.8)
            axs[i, 2].set_xlim(0, k_max)
            axs[i, 2].set_yscale("log")
            axs[i, 2].set_ylim(1e-6, 1)
            axs[i, 2].set_ylabel("Amplitude")

            for k in wavevectors:                                # at +k and -k in the 2D spectrum
                axs[i, 1].plot([k[0], -k[0]], [k[1], -k[1]], "o", mfc="none", color="red", ms=9, mew=0.8)
                axs[i, 2].axvline(np.linalg.norm(k), color="red", ls=":", lw=0.8)

        axs[-1, 0].set_xlabel(r"$x_1$")
        axs[-1, 1].set_xlabel(r"$k_1$")
        axs[-1, 2].set_xlabel(r"$|k|$")
        fig.align_ylabels(axs[:, 0])
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_noise_fits(axes, clean, noisy, predictions, predictions_unconstrained, wavevectors, file_name, k_max=20):
    """One row per network. Columns: the fit (seed 0), a cut along x_1 through the centre, and the radial spectrum."""
    rows = [(f"$L = {L:g}$", runs) for L, runs in predictions.items()] + [("Unconstr.", predictions_unconstrained)]
    extent = [axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]]
    k_noisy, amplitude_noisy = radial_spectrum(axes, noisy)
    k_clean, amplitude_clean = radial_spectrum(axes, clean)

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(rows), 3, figsize=(6.3, 1.6 * len(rows)), sharex="col",
                                gridspec_kw=dict(hspace=0.15, wspace=0.45))
        for i, (name, runs) in enumerate(rows):
            axs[i, 0].imshow(runs[0].T, origin="lower", extent=extent, cmap="viridis",
                             vmin=clean.min(), vmax=clean.max())    # the same colour scale as the clean Gaussian
            axs[i, 0].set_ylabel(name)

            axs[i, 1].plot(axes[0], central_line(noisy), color="lightgrey", lw=0.8)
            axs[i, 1].plot(axes[0], central_line(clean), color="black", lw=1.0, ls="--")
            for prediction in runs:
                axs[i, 1].plot(axes[0], central_line(prediction), color=C_NET, alpha=0.7, lw=1.0)

            axs[i, 2].plot(k_noisy, amplitude_noisy, color="lightgrey", lw=0.8)
            axs[i, 2].plot(k_clean, amplitude_clean, color="black", lw=1.0, ls="--")
            for prediction in runs:
                axs[i, 2].plot(*radial_spectrum(axes, prediction), color=C_NET, alpha=0.7, lw=1.0)
            for k in wavevectors:
                axs[i, 2].axvline(np.linalg.norm(k), color="red", ls=":", lw=0.8)
            axs[i, 2].set_yscale("log")
            axs[i, 2].set_ylim(1e-6, 1)
            axs[i, 2].set_xlim(0, k_max)

        axs[-1, 0].set_xlabel(r"$x_1$")
        axs[-1, 1].set_xlabel(r"$x_1$ ($x_2 = 0$)")
        axs[-1, 2].set_xlabel(r"$|k|$")
        handles = [plt.Line2D([], [], color="lightgrey", lw=0.8, label="Training data (noisy)"),
                   plt.Line2D([], [], color="black", lw=1.0, ls="--", label="Clean Gaussian"),
                   plt.Line2D([], [], color=C_NET, lw=1.0, label="Network (one line per seed)"),
                   plt.Line2D([], [], color="red", ls=":", lw=0.8, label="Noise wave vectors")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_surviving_fractions(axes, predictions, wavevectors, file_name):
    """
    The fraction of every wave that survives, against the slope it needs relative to L. Left in the l2 norm
    (A ||k||_2 / L), right in the l1-network norm (A ||k||_inf / L). The waves fall on one curve in the panel of the
    networks' own norm, and the diagonal waves are shifted in the other.
    """
    markers = ["o", "s", "D", "^", "v"]
    L_values = np.array(list(predictions.keys()))
    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True, gridspec_kw=dict(wspace=0.1))
        for A, k, modes, marker in zip(AMPLITUDES, wavevectors, MODES, markers):
            fractions = [surviving_fractions(axes, predictions[L], A, k) for L in L_values]
            mean = [f.mean() for f in fractions]
            std = [f.std() for f in fractions]
            for ax, norm in [(axs[0], "l2"), (axs[1], "l1")]:
                ax.errorbar(wave_slope(A, k, norm) / L_values, mean, yerr=std, fmt=marker, capsize=3, ms=4,
                            label=f"modes {modes}")
        for ax, label in [(axs[0], r"$A\|k\|_2 / L$"), (axs[1], r"$A\|k\|_\infty / L$")]:
            ax.axvline(1, color=C_MUTED, ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(label)
        axs[0].set_ylabel("Fraction that survives")
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)   # below the x labels
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_measure_vs_L(L_ratios, values, values_unconstrained, y_label, file_name, wave_ratios, noisy_level=None,
                      log_y=False):
    """
    One measure (seeds as rows, values: (L, seeds)) against L / L_max, with the unconstrained network as a band.
    Dotted red lines: where each wave starts to be clipped, A ||k||_dual / L_max.
    """
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        lipschitz_line = ax.errorbar(L_ratios, values.mean(axis=1), yerr=values.std(axis=1), fmt="o-", color=C_NET,
                                     lw=1.2, markersize=4, capsize=3, label="Lipschitz")
        mean, std = values_unconstrained.mean(), values_unconstrained.std()
        unconstrained_line = ax.axhline(mean, color=C_REF, lw=1.2, ls="--", label="Unconstrained")
        ax.axhspan(mean - std, mean + std, color=C_REF, alpha=0.3, lw=0)
        handles = [lipschitz_line, unconstrained_line]
        if noisy_level is not None:
            handles.append(ax.axhline(noisy_level, color="black", lw=0.8, ls=":", label="Noisy data"))
        for ratio in wave_ratios:
            ax.axvline(ratio, color="red", ls=":", lw=0.8)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$L / L_\mathrm{max}$")
        ax.set_ylabel(y_label)
        ax.legend(handles=handles)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def main():
    # 1. The data
    axes, X, clean, noise, noisy, wavevectors = make_data()
    x = X.reshape(-1, DIM)
    L_true = largest_slope(clean, axes, INPUT_NORM)
    L_max = max(L_true, largest_slope(noisy, axes, INPUT_NORM))
    print(f"Grid {N_POINTS}^{DIM}, norm {INPUT_NORM}: slope of the Gaussian {L_true:.4f}, of the noisy data {L_max:.2f}")
    for A, k in zip(AMPLITUDES, wavevectors):
        print(f"Wave k = ({k[0]:5.2f}, {k[1]:5.2f}), A = {A}: needs slope {wave_slope(A, k, INPUT_NORM):.3f}")
    plot_noise_inputs(axes, clean, noise, noisy, wavevectors, "sine_noise_input_and_spectra.pdf")

    # Training: the networks for a few L and the L scan (saved, so the plots can be remade without training)
    L_scan = L_RATIOS_SCAN * L_max
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "rb") as file:
            saved = pickle.load(file)
        print(f"Loaded the networks from {RESULTS_FILE}")
    else:
        predictions, predictions_unconstrained = train_networks(x, noisy.ravel(), L_VALUES, clean.shape)
        predictions_scan, predictions_scan_unconstrained = train_networks(x, noisy.ravel(), L_scan, clean.shape)
        saved = dict(predictions=predictions, predictions_unconstrained=predictions_unconstrained,
                     predictions_scan=predictions_scan, predictions_scan_unconstrained=predictions_scan_unconstrained,
                     L_scan=L_scan, L_max=L_max)
        with open(RESULTS_FILE, "wb") as file:
            pickle.dump(saved, file)

    # 2. Networks for a few L
    print_summary(axes, clean, noise, saved["predictions"], saved["predictions_unconstrained"], wavevectors)
    plot_noise_fits(axes, clean, noisy, saved["predictions"], saved["predictions_unconstrained"], wavevectors,
                    "sine_noise_fits_and_spectra.pdf")
    plot_surviving_fractions(axes, saved["predictions"], wavevectors, "sine_noise_surviving_fraction.pdf")

    # 3. L scan: MSE to the clean Gaussian and the noise removed
    wave_ratios = [wave_slope(A, k, INPUT_NORM) / saved["L_max"] for A, k in zip(AMPLITUDES, wavevectors)]
    measures = [(lambda p: np.mean((p - clean) ** 2), "MSE to clean Gaussian", "sine_noise_mse_vs_L.pdf", True),
                (lambda p: 100 * noise_removed(p, clean, noise), "Noise removed (%)", "sine_noise_removed_vs_L.pdf", False)]
    for measure, y_label, file_name, log_y in measures:
        values = np.array([[measure(p) for p in saved["predictions_scan"][L]] for L in saved["L_scan"]])
        values_unconstrained = np.array([measure(p) for p in saved["predictions_scan_unconstrained"]])
        plot_measure_vs_L(saved["L_scan"] / saved["L_max"], values, values_unconstrained, y_label, file_name,
                          wave_ratios, noisy_level=np.mean(noise ** 2) if log_y else None, log_y=log_y)
    return 0


if __name__ == "__main__":
    main()
