# GOAL: Investigate the effect of overlaid sine noise on the training of the Lipschitz Network. We want the network to function as a regularizer, smoothing out the noise and recovering the underlying Gaussian.
# 
# HOW: 
#   Generate the sine noise
#       Plane waves in multiple directions with random phases. Must be smaller in magnitude than the Guussian
#   Scan over a range of Lipschitz constants

# PLOTS: MSE/MAE evolution, FWHM evolution, Fits at different L,  

#======= IMPORTS =======
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import REPORT_STYLE, noise_removed, plot_dir, results_dir, save_results, load_results
from Toy_nd_utils import (gaussian_nd, isotropic_cov, isotropic_lipschitz_constant, make_grid, wavevector,
                          mode_amplitude_nd, radial_spectrum, central_line, fwhm_equivalent, train_many)

#======= SETTINGS =======
QUICK = "--quick" in sys.argv      # python Sine_noise.py --quick: fewer points, short training, only to check that everything runs
LOAD_SAVED = True                  # True: use the saved networks if they were trained on the same data and settings
# Not networks.pkl: that file in the same folder is from the old Sine_noise.py and has another structure
NETWORKS_FILE = os.path.join(results_dir("2D", "Sine_noise"), "L_scan_quick.pkl" if QUICK else "L_scan.pkl")

# The Gaussian: round, sigma = 1, in the box [-4, 4)^2 (at the edge it is 3e-4 of its peak)
DIM = 2
SIGMA = 1.0
MEAN = np.zeros(DIM)
COV = isotropic_cov(SIGMA, DIM)
HALF_WIDTH = 4.0
LENGTH = 2 * HALF_WIDTH            # side of the box
PEAK = float(gaussian_nd(MEAN, MEAN, COV))
L_TRUE = isotropic_lipschitz_constant(SIGMA, DIM)   # the same in the l1 and l2 norm for a round Gaussian

# The data: random training points, and a periodic grid (endpoint=False) for evaluation and Fourier spectra
N_TRAIN = 2000 if QUICK else 10000   # 10000: about 4 points per wavelength of the shortest wave
N_GRID = 64 if QUICK else 200        # grid points per axis, must be above 2 x the largest mode (24)
SAMPLE_SEED = 0                      # the training points
NOISE_SEED = 42                      # the phases of the waves

# The sine noise: plane waves A sin(k . x + phase), with a whole number of periods across the box along each axis
# (MODES), so that every wave is periodic on the grid and gives one clean peak in its Fourier spectrum.
# Each wave is set by the slope it needs: A sin(k . x) has slope A |k|, so a wave that needs L = r L_true has
# A = r L_true / |k|. (10, 0) and (6, 8) have the same |k| in different directions: an l2 network should treat them equally.
MODES = np.array([[5, 0], [0, 8], [10, 0], [6, 8], [12, 9], [-15, 15], [0, 24]])
SLOPE_RATIOS = np.array([0.25, 0.5, 0.75, 0.75, 1, 1.5, 2])   # L / L_true needed to fit each wave (noise rms 10% of the peak)
WAVEVECTORS = wavevector(MODES, LENGTH)               # k = 2 pi m / LENGTH
WAVENUMBERS = np.linalg.norm(WAVEVECTORS, axis=1)     # |k|
AMPLITUDES = SLOPE_RATIOS * L_TRUE / WAVENUMBERS
PAIR = [2, 3]                      # the places of (10, 0) and (6, 8) in MODES: the same |k| in different directions

# The networks, as in 1D and Optimal_solution
INPUT_NORM = "l2"                  # in the l2 norm a round Gaussian, and waves in every direction, are treated equally
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5
LEARNING_RATE = 3e-3
SEEDS = [0, 1] if QUICK else [0, 1, 2]
# The L scan, from below the smallest wave (0.25 L_true) to above the slope of all waves and the Gaussian together
# (at most 1 + 6.75 = 7.75 L_true), where nothing limits the fit anymore. Steps of a quarter decade, so 1 is included.
L_RATIOS = np.logspace(-1, 1.5, 3 if QUICK else 11)
# Training until converged (see train_until_converged in Toy_utils): the learning rate is halved when the loss stops
# improving, and training stops after n_halvings halvings. The same as in 1D
CONVERGENCE = (dict(check_every=50, rel_tol=1e-3, patience=1, n_halvings=1, max_steps=300) if QUICK else
               dict(check_every=500, rel_tol=1e-3, patience=2, n_halvings=6, max_steps=200000))

# Where to train. The networks are tiny, so a GPU is not faster for one network because of the computing, but
# because its time per step hardly grows with the number of points. Several networks share the GPU, one process each,
# and every process needs its own CPU core: N_WORKERS at most the number of cores you have on the GPU machine.
# (Not the Mac GPU, "mps": its sort, used by GroupSort, is slower than the CPU.)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS = 8 if DEVICE == "cuda" else None     # None: all CPU cores but one

# The plots, in the same format as in 1D (Scripts/1D/Noise_response.py)
PLOT_DIR = plot_dir("2D", "Sine_noise")
L_RATIOS_SHOW = [0.3, 1, 3, 10]    # the networks shown in the fits (the nearest L in the scan)
K_MAX_SHOWN = 25                   # highest |k| in the spectra (the shortest wave has |k| = 18.8)
# The measures of the L scan: name -> (y label, file name, logarithmic y axis)
MEASURES = {"MSE": ("MSE", "sine_noise_mse_vs_L.pdf", True),
            "MAE": ("MAE", "sine_noise_mae_vs_L.pdf", True),
            "FWHM": (r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$", "sine_noise_fwhm_vs_L.pdf", True),
            "removed": ("Noise removed (%)", "sine_noise_removed_vs_L.pdf", False)}
# The MAE spans only about one decade. Matplotlib then labels the minor ticks as well, and the long labels push the
# axis label out of the figure, so its labelled ticks are set by hand (only those inside the axis are shown)
MAE_Y_TICKS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
# The scan plots sit side by side in the report, so they get a fixed size and margins and are saved without tight
# cropping. The legend sits under the axes, in two columns
SCAN_SIZE = (3.1, 3.5)
SCAN_MARGINS = dict(left=0.26, right=0.96, top=0.970, bottom=0.330)
SCAN_LEGEND = dict(loc="upper center", bbox_to_anchor=((0.5 - 0.26) / (0.96 - 0.26), -0.2), ncol=2,
                   handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
PANEL_SIZE = (3.1, 2.8)            # inches, about the size of a 0.48\textwidth subfigure (two side by side)
C_TRUE, C_MUTED, C_REF = "#1f1f1f", "#6b6a66", "#9a9893"
C_NET, C_NOISE, C_DATA, C_MODE = "tab:orange", "tab:orange", "grey", "red"

#======= FUNCTIONS =======
def sine_noise(X, phases):
    """The sine noise, the sum of A sin(k . x + phase) over all waves, at the points X of shape (..., 2)."""
    noise = np.zeros(X.shape[:-1])
    for A, k, phase in zip(AMPLITUDES, WAVEVECTORS, phases):
        noise = noise + A * np.sin(X @ k + phase)     # X @ k is k . x at every point
    return noise


def make_data():
    """
    The data as a dictionary:
        X_train, clean_train, noise_train, noisy_train: N_TRAIN random points in the box and the values there,
        axes, X_grid, clean_grid, noise_grid: the periodic evaluation grid and the values there.
    The noise is a function of position, so the same waves (same phases) are evaluated at both sets of points.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    X_train = rng.uniform(-HALF_WIDTH, HALF_WIDTH, size=(N_TRAIN, DIM))
    axes, X_grid = make_grid(HALF_WIDTH, N_GRID, DIM, endpoint=False)

    noise_rng = np.random.default_rng(NOISE_SEED)
    phases = noise_rng.uniform(0, 2 * np.pi, len(AMPLITUDES))   # random, but the same in every run

    data = {}
    data["X_train"] = X_train
    data["clean_train"] = gaussian_nd(X_train, MEAN, COV)
    data["noise_train"] = sine_noise(X_train, phases)
    data["noisy_train"] = data["clean_train"] + data["noise_train"]
    data["axes"] = axes
    data["X_grid"] = X_grid
    data["clean_grid"] = gaussian_nd(X_grid, MEAN, COV)
    data["noise_grid"] = sine_noise(X_grid, phases)
    return data


def check_noise(data):
    """Prints every wave (its size, the L it needs, and its amplitude measured from the grid) and the total noise level."""
    print("L_true = %.4f, peak = %.4f" % (L_TRUE, PEAK))
    print("mode        k sigma   A/peak   A|k|/L_true   measured A/A")
    for m, k, k_length, A in zip(MODES, WAVEVECTORS, WAVENUMBERS, AMPLITUDES):
        measured = mode_amplitude_nd(data["axes"], data["noise_grid"], k)   # should give back A
        print("%-10s  %6.1f   %6.3f   %10.2f   %12.3f" % (m, k_length * SIGMA, A / PEAK, A * k_length / L_TRUE, measured / A))

    rms_expected = np.sqrt(np.sum(AMPLITUDES ** 2) / 2)   # each wave has rms A / sqrt(2)
    print("noise rms / peak: expected %.3f, training points %.3f" % (rms_expected / PEAK, np.std(data["noise_train"]) / PEAK))

#======= L-SCAN =======
def L_scan(L_ratio, L_true, data, seeds):
    """
    Trains a Lipschitz network for every L = L_ratio * L_true and seed, and the unconstrained network for every seed,
    on the noisy training points until converged, and evaluates them on the grid. Returns a dictionary with
        grid: the Lipschitz networks on the grid, shape (seeds, L values, N_GRID, N_GRID),
        histories[seed][L]: their training histories (step, loss, MSE to the clean Gaussian, learning rate),
        unconstrained_grid, unconstrained_histories: the same for the unconstrained network (without the L axis).
    """
    L_arr = L_ratio * L_true
    jobs = []
    for L in L_arr:
        for seed in seeds:
            job = {"kind": "lipschitz", "seed": seed,
                   "model_kwargs": {"input_dim": DIM, "hidden_dim": HIDDEN_DIM, "output_dim": 1, "n_groups": N_GROUPS,
                                    "depth": DEPTH, "lipschitz_bound": L, "input_norm": INPUT_NORM}}
            jobs.append(job)
    for seed in seeds:   # the unconstrained network: how much of the noise a network copies without a slope limit
        job = {"kind": "unconstrained", "seed": seed,
               "model_kwargs": {"input_dim": DIM, "hidden_dim": HIDDEN_DIM, "output_dim": 1, "depth": DEPTH}}
        jobs.append(job)

    # The same data and training settings for every network
    for job in jobs:
        job["x"] = data["X_train"]
        job["y"] = data["noisy_train"]
        job["y_clean"] = data["clean_train"]               # to record the MSE to the clean Gaussian during training
        job["x_eval"] = data["X_grid"].reshape(-1, DIM)    # the grid as a list of points, shape (N_GRID^2, 2)
        job["lr"] = LEARNING_RATE
        job["until_converged"] = True
        job["convergence"] = CONVERGENCE
        job["device"] = DEVICE

    print("Training %d networks until converged on the %s" % (len(jobs), DEVICE))
    start = time.time()
    results = train_many(jobs, n_workers=N_WORKERS)     # in the same order as the jobs
    print("Trained in %.1f min" % ((time.time() - start) / 60))

    n_L = len(L_ratio)
    n_seeds = len(seeds)
    networks = {}
    networks["grid"] = np.zeros((n_seeds, n_L, N_GRID, N_GRID))
    networks["histories"] = [[None] * n_L for seed in seeds]
    for j in range(n_L):
        for i in range(n_seeds):
            result = results[j * n_seeds + i]     # the jobs were made with the L loop outside and the seed loop inside
            networks["grid"][i, j] = result["prediction"].reshape(N_GRID, N_GRID)
            networks["histories"][i][j] = result["history"]

    networks["unconstrained_grid"] = np.zeros((n_seeds, N_GRID, N_GRID))
    networks["unconstrained_histories"] = [None] * n_seeds
    for i in range(n_seeds):
        result = results[n_L * n_seeds + i]       # after all the Lipschitz networks
        networks["unconstrained_grid"][i] = result["prediction"].reshape(N_GRID, N_GRID)
        networks["unconstrained_histories"][i] = result["history"]
    return networks


def same_array(a, b):
    """
    True if a and b have the same shape and the same values up to rounding. Not exact equality: the networks may be
    trained on another computer (the cluster), which can round the last digit of sin and exp differently.
    """
    return a.shape == b.shape and np.allclose(a, b)


def saved_or_trained_networks(data):
    """
    The networks of the L scan. Loaded from NETWORKS_FILE if they were trained on the same data, L values and
    convergence settings as now, otherwise trained again and saved together with those.
    """
    if LOAD_SAVED and os.path.exists(NETWORKS_FILE):
        saved = load_results(NETWORKS_FILE)
        same_data = same_array(saved["X_train"], data["X_train"]) and same_array(saved["noisy_train"], data["noisy_train"])
        same_L = same_array(saved["L_ratios"], L_RATIOS) and saved["seeds"] == SEEDS
        same_training = saved["convergence"] == CONVERGENCE
        if same_data and same_L and same_training:
            return saved["networks"]
        print("The saved networks used other data, L values or settings: training again")

    networks = L_scan(L_RATIOS, L_TRUE, data, SEEDS)
    save_results({"X_train": data["X_train"], "noisy_train": data["noisy_train"], "L_ratios": L_RATIOS,
                  "seeds": SEEDS, "convergence": CONVERGENCE, "networks": networks}, NETWORKS_FILE)
    return networks

#======= MEASURES =======
def measures(prediction, data):
    """
    The measures of one fit on the grid: MSE and MAE to the clean Gaussian, the width at half maximum relative to
    the true width (fwhm_equivalent: the diameter of the circle with the same area as the region above half maximum),
    and the share of the noise removed in % (see noise_removed in Toy_utils: kept = copied^2, removed = 1 - kept).
    The sine noise exists everywhere, so the noise removed is measured on the grid, not only at the training points.
    """
    error = prediction - data["clean_grid"]
    fwhm_true = fwhm_equivalent(data["clean_grid"], data["axes"])
    return {"MSE": np.mean(error ** 2),
            "MAE": np.mean(np.abs(error)),
            "FWHM": fwhm_equivalent(prediction, data["axes"]) / fwhm_true,
            "removed": 100 * noise_removed(prediction, data["clean_grid"], data["noise_grid"])}


def network_measures(data, networks):
    """The measures of every network: lipschitz[name] of shape (seeds, L values) and unconstrained[name] of shape (seeds,)."""
    n_seeds, n_L = networks["grid"].shape[:2]
    lipschitz = {}
    unconstrained = {}
    for name in MEASURES:
        lipschitz[name] = np.zeros((n_seeds, n_L))
        unconstrained[name] = np.zeros(n_seeds)
    for i in range(n_seeds):
        for j in range(n_L):
            for name, value in measures(networks["grid"][i, j], data).items():
                lipschitz[name][i, j] = value
        for name, value in measures(networks["unconstrained_grid"][i], data).items():
            unconstrained[name][i] = value
    return lipschitz, unconstrained


def surviving_fractions(axes, prediction):
    """
    The fraction of every sine wave that survives in a fit on the grid: the amplitude of its Fourier mode in the fit
    divided by its amplitude in the data. The grid spans the box exactly, so every wave sits on one Fourier mode.
    """
    fractions = np.zeros(len(AMPLITUDES))
    for i in range(len(AMPLITUDES)):
        fractions[i] = mode_amplitude_nd(axes, prediction, WAVEVECTORS[i]) / AMPLITUDES[i]
    return fractions


def network_fractions(data, networks):
    """The surviving fraction of every wave in the networks, mean over seeds, shape (L values, waves)."""
    n_seeds, n_L = networks["grid"].shape[:2]
    fractions = np.zeros((n_L, len(AMPLITUDES)))
    for j in range(n_L):
        for i in range(n_seeds):
            fractions[j] = fractions[j] + surviving_fractions(data["axes"], networks["grid"][i, j]) / n_seeds
    return fractions

#======= PLOTS =======
def format_spectrum_axes(ax):
    """The shared format of the spectra, as in 1D: logarithmic amplitude, and the |k| of every wave marked."""
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-1)
    ax.set_yticks([1e-6, 1e-4, 1e-2])          # no tick labels at the edges, so they do not run into the next row
    ax.set_xlim(0, K_MAX_SHOWN)
    ax.set_ylabel("Amplitude")
    for k_wave in WAVENUMBERS:
        ax.axvline(k_wave, color=C_MODE, ls=":", lw=0.8)


def plot_data(data, file_name):
    """
    The data, as in 1D: the clean Gaussian, the noise and the noisy Gaussian as maps on the grid (left), and their
    radial Fourier spectra (right) with the |k| of the waves marked. The radial spectrum adds up the modes with the
    same |k|, so the peak of (10, 0) and (6, 8) together is sqrt(2) A.
    """
    noisy = data["clean_grid"] + data["noise_grid"]
    extent = [-HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH]
    noise_limit = np.abs(data["noise_grid"]).max()                     # a colour scale symmetric around zero
    k, clean_spectrum = radial_spectrum(data["axes"], data["clean_grid"])
    noise_spectrum = radial_spectrum(data["axes"], data["noise_grid"])[1]
    noisy_spectrum = radial_spectrum(data["axes"], noisy)[1]
    with plt.rc_context(REPORT_STYLE):
        # wspace: room between the colour bars of the maps and the axis labels of the spectra
        fig, axs = plt.subplots(3, 2, figsize=(6.3, 7.5), sharex="col", gridspec_kw=dict(hspace=0.15, wspace=0.95))
        images = [axs[0, 0].imshow(data["clean_grid"].T, origin="lower", extent=extent, cmap="viridis"),   # .T: imshow wants (x_2, x_1)
                  axs[1, 0].imshow(data["noise_grid"].T, origin="lower", extent=extent, cmap="RdBu_r",
                                   vmin=-noise_limit, vmax=noise_limit),
                  axs[2, 0].imshow(noisy.T, origin="lower", extent=extent, cmap="viridis")]
        for image, ax, label in zip(images, axs[:, 0], [r"$f$", r"$n$", r"$f + n$"]):
            fig.colorbar(image, ax=ax, label=label)
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.set_ylabel(r"$x_2$")
        axs[2, 0].set_xlabel(r"$x_1$")

        axs[0, 1].plot(k, clean_spectrum, color=C_TRUE, lw=1.2)
        axs[1, 1].plot(k, noise_spectrum, color=C_NOISE, lw=1.0)
        axs[2, 1].plot(k, noisy_spectrum, color=C_DATA, lw=1.0)
        axs[2, 1].plot(k, clean_spectrum, color=C_TRUE, lw=1.2, ls="--")
        for ax in axs[:, 1]:
            format_spectrum_axes(ax)
        axs[2, 1].set_xlabel(r"$|k|$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_fits_and_spectra(data, L_ratios, networks, file_name):
    """
    One row per network (the L in L_RATIOS_SHOW, then the unconstrained network), as in 1D. Left: cuts through the
    peak along x_1 (at x_2 = 0) of the noisy Gaussian, the clean Gaussian (dashed) and the fits, one line per seed.
    Right: the radial Fourier spectra of the same, on the whole grid. A wave the network removes shows up as a peak of
    the noisy data that is missing in the spectrum of the fit.
    """
    x = data["axes"][0]
    noisy = data["clean_grid"] + data["noise_grid"]
    noisy_cut = central_line(noisy)                  # central_line: the values along x_1 at x_2 = 0
    clean_cut = central_line(data["clean_grid"])
    k, clean_spectrum = radial_spectrum(data["axes"], data["clean_grid"])
    noisy_spectrum = radial_spectrum(data["axes"], noisy)[1]
    rows = []
    shown = []                                           # the places in the scan already shown, so no L appears twice
    for ratio in L_RATIOS_SHOW:
        j = np.argmin(np.abs(np.log(L_ratios / ratio)))  # the nearest L in the scan, on the log scale of the scan
        if j in shown:
            continue
        shown.append(j)
        rows.append((rf"${L_ratios[j]:.2g}\,L_\mathrm{{true}}$", networks["grid"][:, j]))
    rows.append(("Unconstr.", networks["unconstrained_grid"]))

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(rows), 2, figsize=(6.3, 1.7 * len(rows)), sharex="col",
                                gridspec_kw=dict(hspace=0.12, wspace=0.45))
        for (name, predictions), ax_fit, ax_spectrum in zip(rows, axs[:, 0], axs[:, 1]):
            ax_fit.plot(x, noisy_cut, color="lightgrey", lw=0.8)
            ax_fit.plot(x, clean_cut, color=C_TRUE, lw=1.0, ls="--")
            for prediction in predictions:
                ax_fit.plot(x, central_line(prediction), color=C_NET, alpha=0.7, lw=1.0)
            ax_fit.set_ylabel(name)

            ax_spectrum.plot(k, noisy_spectrum, color="lightgrey", lw=0.8)
            ax_spectrum.plot(k, clean_spectrum, color=C_TRUE, lw=1.0, ls="--")
            for prediction in predictions:
                ax_spectrum.plot(k, radial_spectrum(data["axes"], prediction)[1], color=C_NET, alpha=0.7, lw=0.8)
            format_spectrum_axes(ax_spectrum)

        axs[-1, 0].set_xlabel(r"$x_1$ ($x_2 = 0$)")
        axs[-1, 1].set_xlabel(r"$|k|$")
        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        handles = [plt.Line2D([], [], color="lightgrey", lw=0.8, label="Noisy Gaussian"),
                   plt.Line2D([], [], color=C_TRUE, lw=1.0, ls="--", label="Clean Gaussian"),
                   plt.Line2D([], [], color=C_NET, lw=1.0, label="Network (one line per seed)"),
                   plt.Line2D([], [], color=C_MODE, ls=":", lw=0.8, label="Sine modes")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_scan(name, L_ratios, lipschitz, unconstrained, noisy):
    """
    One measure against L / L_true, as in 1D: the networks (mean over seeds, the spread min to max as error bars), the
    unconstrained network (dashed, its spread as a band) and the noisy data itself (dotted, if noisy[name] is not None).
    L_true is marked, and the slope every wave needs (SLOPE_RATIOS).
    """
    y_label, file_name, log_y = MEASURES[name]
    values = lipschitz[name]
    reference = unconstrained[name]
    mean = values.mean(axis=0)
    spread = [mean - values.min(axis=0), values.max(axis=0) - mean]
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=SCAN_SIZE)
        fig.subplots_adjust(**SCAN_MARGINS)
        handles = [ax.errorbar(L_ratios, mean, yerr=spread, fmt="o", color=C_NET, markersize=3.5, lw=1.0, capsize=2,
                               label="Networks"),
                   ax.axhline(reference.mean(), color=C_REF, lw=1.2, ls="--", label="Unconstrained")]
        ax.axhspan(reference.min(), reference.max(), color=C_REF, alpha=0.3, lw=0)
        if noisy[name] is not None:
            handles.append(ax.axhline(noisy[name], color=C_TRUE, lw=0.8, ls=":", label="Noisy data"))
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")             # L_true
        for ratio in SLOPE_RATIOS:                                  # the slope every wave needs, in units of L_true
            mode_line = ax.axvline(ratio, color=C_MODE, ls=":", lw=0.8, label="Sine modes")
        handles.append(mode_line)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if name == "MAE":
            ax.yaxis.set_major_locator(plt.FixedLocator(MAE_Y_TICKS))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))   # 0.005 instead of 5x10^-3
            ax.yaxis.set_minor_formatter(plt.NullFormatter())            # minor ticks stay, without labels
        if name == "FWHM":
            ax.axhline(1.0, color=C_MUTED, lw=0.8, ls="--")             # the true width
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))   # 1, 2, 3 instead of 10^0, 2x10^0
            ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%g"))
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        ax.legend(handles=handles, **SCAN_LEGEND)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_surviving_fractions(L_ratios, fractions, file_name):
    """
    The fraction of every sine wave that survives in the networks (mean over seeds), against the slope it needs
    relative to L, A|k| / L = SLOPE_RATIOS / (L / L_true). If the bound is what removes the waves, all waves follow
    one curve, and a wave passes when A|k| < L. As in 1D.
    """
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for j, ratio in enumerate(L_ratios):
            ax.plot(SLOPE_RATIOS / ratio, fractions[j], "o", color=C_NET, markersize=3.5)
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"$A|k| / L$")
        ax.set_ylabel("Fraction that survives")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)

#======= MAIN =======
def main():
    data = make_data()
    check_noise(data)
    plot_data(data, "sine_noise_data.pdf")
    networks = saved_or_trained_networks(data)

    lipschitz, unconstrained = network_measures(data, networks)
    noisy = measures(data["clean_grid"] + data["noise_grid"], data)    # the noisy data itself, as a reference
    noisy["removed"] = None            # the noisy data keeps all of its noise (0 % removed): no line, as in 1D
    plot_fits_and_spectra(data, L_RATIOS, networks, "sine_noise_fits_and_spectra.pdf")
    for name in MEASURES:
        plot_scan(name, L_RATIOS, lipschitz, unconstrained, noisy)

    fractions = network_fractions(data, networks)
    plot_surviving_fractions(L_RATIOS, fractions, "sine_noise_surviving_fraction.pdf")
    # The isotropy check: in the l2 norm, two waves with the same |k| should survive equally, whatever their direction
    print("Surviving fraction of (10, 0) and (6, 8), the same |k| in different directions:")
    for j, ratio in enumerate(L_RATIOS):
        print("  L = %5.2f L_true:   %.3f   %.3f" % (ratio, fractions[j, PAIR[0]], fractions[j, PAIR[1]]))
    return

if __name__ == "__main__":
    main()