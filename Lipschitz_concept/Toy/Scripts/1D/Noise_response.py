# GOAL: How Lipschitz networks respond to noise on a 1D Gaussian: sine-wave noise (a few frequencies) and white noise
# (all frequencies). Does the bound on the slope filter out the noise, and how does that compare with an unconstrained
# network?
# (Script version of the noise sections of Toy_gaussian.ipynb.)

# HOW: The training data are the Gaussian plus noise at random points. A sine wave A sin(omega x) needs the slope
# A omega, so an L-Lipschitz network cannot follow the waves with A omega > L, while the Gaussian needs only L_true.
# The networks are trained until their loss stops improving (not for a fixed number of steps), so the results show
# what the networks converge to, and not how far they got in a given time.
#   1. The data: the Gaussian, the noise and the noisy Gaussian, with their Fourier spectra
#   2. Networks for a few L (and the unconstrained network): the fits and their spectra, and how the training converged
#   3. L scan: MSE and MAE to the clean Gaussian, the noise removed and the noise kept, for the networks, the
#      unconstrained network and the noisy data itself. Plus a zoom of the MSE and MAE around L_true, where they drop fast
#   4. Sine noise only: the fraction of every wave that survives, against the slope it needs relative to L
# The networks are trained on random points and evaluated on an evenly spaced grid (unseen points), which the Fourier
# spectra need. The noise removed and kept are measured at the training points, where the noise is known for both
# kinds (the white noise only exists there).

# PLOTS: in Plots/1D/Sine_noise and Plots/1D/White_noise


import os
import sys
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import (REPORT_STYLE, gaussian_function, gaussian_lipschitz_constant, fourier_spectrum, noise_removed,
                       plot_dir, results_dir, save_results, load_results)
from Toy_nd_utils import train_many

QUICK = False                      # True: few points, short training, only to check that everything runs
LOAD_SAVED = True                  # True: use the saved networks if they exist, instead of training again
NOISE_KINDS = ["sine", "white"]
PLOT_DIRS = {"sine": plot_dir("1D", "Sine_noise"), "white": plot_dir("1D", "White_noise")}

# The Gaussian and the points, as in the notebook (but random training points)
X_MIN, X_MAX = -5, 5
SIGMA = 2.0                        # a wide Gaussian, so that the noise is much faster than the Gaussian
L_TRUE = gaussian_lipschitz_constant(SIGMA)
N_TRAIN = 300 if QUICK else 2000   # random training points
N_GRID = 400 if QUICK else 2000    # evenly spaced evaluation points, for the Fourier spectra
SAMPLE_SEED = 0                    # the same training points in every run
NOISE_SEED = 42                    # the same noise in every run

# Sine noise, as in the notebook: whole numbers of periods (MODES) across the box, so every wave sits on one Fourier mode
AMPLITUDES = 0.1 * np.array([0.01, 0.01, 0.05, 0.07, 0.15, 0.1, 0.09, 0.08])
MODES = [10, 20, 45, 60, 75, 95, 110, 120]
FREQUENCIES = 2 * np.pi * np.array(MODES) / (X_MAX - X_MIN)
# White noise with the same power (mean squared noise) as the sine noise, sum(A^2) / 2
NOISE_STD = np.sqrt(np.sum(AMPLITUDES ** 2) / 2)

# Networks. The scan covers L from below L_true to above the slopes of the noise (L / L_true)
L_RATIOS_SHOW = [0.5, 1, 3, 12]    # the networks shown in detail
L_RANGE = {"sine": (0.1, 100), "white": (0.1, 1000)}
# Zoom around L_true, where the MSE and MAE drop fast. A separate scan with its own saved networks, so the main scan
# does not have to be trained again. Log spaced from 0.5 to 2, so symmetric around 1 on the log axis, and with an odd
# number of points L_true itself is one of them
ZOOM_RANGE = (0.5, 2.0)
N_ZOOM_NETWORKS = 3 if QUICK else 19
ZOOM_X_TICKS = [0.5, 1, 2]
# The errors to the clean Gaussian are also plotted with a linear y axis (the log axis makes their fast drop below
# L_true look like a jump), and in the zoom
ERROR_MEASURES = ["MSE", "MAE"]
N_SCAN_NETWORKS = 3 if QUICK else 12
SEEDS = [0, 1] if QUICK else [0, 1, 2]
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5
# Training until converged (see train_until_converged in Toy_utils): every check_every steps the loss is compared with
# the best so far, the learning rate is halved when it stops improving, and training stops after n_halvings halvings
CONVERGENCE = (dict(check_every=50, rel_tol=1e-3, patience=1, n_halvings=1, max_steps=300) if QUICK else
               dict(check_every=500, rel_tol=1e-3, patience=2, n_halvings=6, max_steps=200000))

# The measures of the L scan: name -> (y label, file name ending, logarithmic y axis)
MEASURES = {"MSE": ("MSE", "mse_vs_L", True),
            "MAE": ("MAE", "mae_vs_L", True),
            "removed": ("Noise removed (%)", "removed_vs_L", False),
            "kept": ("Noise kept (%)", "kept_vs_L", True)}
# The MAE spans only about one decade. Matplotlib then labels the minor ticks as well (2x10^-3, 3x10^-3, ...), and the
# long labels push the axis label out of the figure, so its labelled ticks are set by hand (only those inside the axis
# are shown)
MAE_Y_TICKS = [0.002, 0.005, 0.01, 0.02, 0.05]

# Figure format
PANEL_SIZE = (3.1, 2.8)            # inches, about the size of a 0.48\textwidth subfigure (two side by side)
# The scan plots sit side by side in the report, so they get a fixed size and margins and are saved without tight
# cropping. The legend sits under the axes, in two columns
SCAN_SIZE = (3.1, 3.5)             # the axes are as large as in the Optimal_solution scans, the legend has two rows
SCAN_MARGINS = dict(left=0.26, right=0.96, top=0.970, bottom=0.330)
SCAN_LEGEND = dict(loc="upper center", bbox_to_anchor=((0.5 - 0.26) / (0.96 - 0.26), -0.2), ncol=2,
                   handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
OMEGA_MAX_SINE = 120               # highest frequency shown in the sine-noise spectra
C_TRUE, C_MUTED, C_REF = "#1f1f1f", "#6b6a66", "#9a9893"
C_NET, C_NOISE, C_DATA, C_MODE = "tab:orange", "tab:orange", "grey", "red"


#### CALCULATIONS ####
def sine_noise(x, phases):
    """The sine-wave noise, the sum of A sin(omega x + phase) over all waves, at the points x."""
    noise = np.zeros_like(x)
    for A, omega, phase in zip(AMPLITUDES, FREQUENCIES, phases):
        noise = noise + A * np.sin(omega * x + phase)
    return noise


def make_data(kind):
    """
    The data for one kind of noise ("sine" or "white"), as a dictionary:
        x_train, clean_train, noise_train, noisy_train: N_TRAIN random points (sorted) and the values there,
        x_grid, clean_grid: the evenly spaced evaluation grid (without the end point, so it is periodic),
        noise_grid: the sine noise on the grid. The sine noise is a function and exists everywhere, the white noise
            is an independent random number at every training point and only exists there (None).
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    x_train = np.sort(rng.uniform(X_MIN, X_MAX, N_TRAIN))
    x_grid = np.linspace(X_MIN, X_MAX, N_GRID, endpoint=False)
    data = dict(x_train=x_train, clean_train=gaussian_function(x_train, 0.0, SIGMA),
                x_grid=x_grid, clean_grid=gaussian_function(x_grid, 0.0, SIGMA))

    noise_rng = np.random.default_rng(NOISE_SEED)
    if kind == "sine":
        phases = noise_rng.uniform(0, 2 * np.pi, len(AMPLITUDES))      # random, but the same in every run
        data["noise_train"] = sine_noise(x_train, phases)
        data["noise_grid"] = sine_noise(x_grid, phases)
    else:
        data["noise_train"] = NOISE_STD * noise_rng.standard_normal(N_TRAIN)
        data["noise_grid"] = None
    data["noisy_train"] = data["clean_train"] + data["noise_train"]
    return data


def point_spectrum(x, f, omega):
    """
    Amplitude spectrum of the values f at irregular points x, at the angular frequencies omega, scaled like
    fourier_spectrum (a term A sin(omega x) gives a peak of height A). The FFT needs evenly spaced points, so the
    Fourier sum is written out instead. On random points it has a background of about 2 rms(f) / sqrt(N) at all frequencies.
    """
    amplitude = 2 * np.abs(np.exp(-1j * np.outer(omega, x)) @ f) / len(x)
    amplitude[omega == 0] = amplitude[omega == 0] / 2
    return amplitude


def data_spectra(kind, data):
    """
    The spectra of the clean Gaussian, the noise and the noisy training data, at the frequencies of the grid.
    The sine noise exists everywhere, so its spectra come from the grid. The white noise only exists at the random
    training points, so its spectra (and that of the noisy data) come from those points (point_spectrum).
    Returns omega and the three spectra.
    """
    omega, clean = fourier_spectrum(data["x_grid"], data["clean_grid"])
    if kind == "sine":
        noise = fourier_spectrum(data["x_grid"], data["noise_grid"])[1]
        noisy = fourier_spectrum(data["x_grid"], data["clean_grid"] + data["noise_grid"])[1]
    else:
        noise = point_spectrum(data["x_train"], data["noise_train"], omega)
        noisy = point_spectrum(data["x_train"], data["noisy_train"], omega)
    return omega, clean, noise, noisy


def network_L_ratios(kind):
    """The L / L_true of the trained networks: the logarithmic scan of the noise kind, plus the values shown in detail."""
    low, high = L_RANGE[kind]
    return np.unique(np.concatenate([np.logspace(np.log10(low), np.log10(high), N_SCAN_NETWORKS), L_RATIOS_SHOW]))


def train_networks(data, L_ratios):
    """
    Lipschitz networks for every L / L_true and seed, and the unconstrained network for every seed, trained on the noisy
    training points until converged. Every network is evaluated on the grid and at the training points.
    Returns a dictionary with grid and train (the predictions, shape (seeds, L values, points)), histories[seed][L]
    (the training histories), and the same for the unconstrained network (without the L axis).
    """
    x_eval = np.concatenate([data["x_grid"], data["x_train"]]).reshape(-1, 1)
    common = dict(lr=LEARNING_RATE, until_converged=True, convergence=CONVERGENCE, x=data["x_train"].reshape(-1, 1),
                  y=data["noisy_train"], y_clean=data["clean_train"], x_eval=x_eval)
    jobs = []
    for ratio in L_ratios:
        for seed in SEEDS:
            jobs.append(dict(kind="lipschitz", seed=seed, **common,
                             model_kwargs=dict(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                               depth=DEPTH, lipschitz_bound=ratio * L_TRUE)))
    for seed in SEEDS:
        jobs.append(dict(kind="unconstrained", seed=seed, **common,
                         model_kwargs=dict(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)))
    print(f"Training {len(jobs)} networks until converged")
    results = iter(train_many(jobs))

    n_seeds, n_L = len(SEEDS), len(L_ratios)
    networks = dict(grid=np.zeros((n_seeds, n_L, N_GRID)), train=np.zeros((n_seeds, n_L, N_TRAIN)),
                    histories=[[None] * n_L for _ in SEEDS],
                    unconstrained_grid=np.zeros((n_seeds, N_GRID)), unconstrained_train=np.zeros((n_seeds, N_TRAIN)),
                    unconstrained_histories=[None] * n_seeds)
    for j in range(n_L):
        for i in range(n_seeds):
            result = next(results)
            networks["grid"][i, j] = result["prediction"][:N_GRID]
            networks["train"][i, j] = result["prediction"][N_GRID:]
            networks["histories"][i][j] = result["history"]
    for i in range(n_seeds):
        result = next(results)
        networks["unconstrained_grid"][i] = result["prediction"][:N_GRID]
        networks["unconstrained_train"][i] = result["prediction"][N_GRID:]
        networks["unconstrained_histories"][i] = result["history"]
    return networks


def results_file(kind, scan):
    """The file with the saved networks of one noise kind and scan ("" for the main scan, "_zoom" for the zoom)."""
    quick = "_quick" if QUICK else ""
    return os.path.join(results_dir("1D", f"{kind.capitalize()}_noise"), f"networks{scan}{quick}.pkl")


def saved_or_trained_networks(kind, data, L_ratios, scan=""):
    """
    The networks of one noise kind and scan ("" for the main scan, "_zoom" for the zoom around L_true). Loaded from
    their file (results_file) if they were trained on the same points, L values and convergence settings as now,
    otherwise trained again and saved together with those.
    """
    file_name = results_file(kind, scan)
    if LOAD_SAVED and os.path.exists(file_name):
        saved = load_results(file_name)
        if (np.array_equal(saved["x_train"], data["x_train"]) and np.array_equal(saved["L_ratios"], L_ratios)
                and saved["convergence"] == CONVERGENCE):
            return saved["networks"]
        print(f"The saved networks in {file_name} used other points, L values or settings: training again")
    networks = train_networks(data, L_ratios)
    save_results(dict(x_train=data["x_train"], L_ratios=L_ratios, convergence=CONVERGENCE, networks=networks),
                 file_name)
    return networks


def measures(prediction_grid, prediction_train, data):
    """
    The measures of one fit: MSE and MAE to the clean Gaussian on the grid, and the share of the noise removed and kept
    (in %) at the training points (see noise_removed in Toy_utils: kept = copied^2, removed = 1 - kept).
    """
    removed = 100 * noise_removed(prediction_train, data["clean_train"], data["noise_train"])
    return {"MSE": np.mean((prediction_grid - data["clean_grid"]) ** 2),
            "MAE": np.mean(np.abs(prediction_grid - data["clean_grid"])),
            "removed": removed,
            "kept": 100 - removed}


def network_measures(data, networks):
    """The measures of every network: lipschitz[name] of shape (seeds, L values) and unconstrained[name] of shape (seeds,)."""
    n_seeds, n_L = networks["grid"].shape[:2]
    lipschitz = {name: np.zeros((n_seeds, n_L)) for name in MEASURES}
    unconstrained = {name: np.zeros(n_seeds) for name in MEASURES}
    for i in range(n_seeds):
        for j in range(n_L):
            for name, value in measures(networks["grid"][i, j], networks["train"][i, j], data).items():
                lipschitz[name][i, j] = value
        for name, value in measures(networks["unconstrained_grid"][i], networks["unconstrained_train"][i], data).items():
            unconstrained[name][i] = value
    return lipschitz, unconstrained


def surviving_fractions(x_grid, prediction_grid):
    """
    The fraction of every sine wave that survives in a fit on the grid: the amplitude of its Fourier mode in the fit
    divided by its amplitude in the data. The grid spans the box exactly, so wave number n sits on Fourier mode n.
    """
    amplitude = fourier_spectrum(x_grid, prediction_grid)[1]
    return np.array([amplitude[mode] / A for mode, A in zip(MODES, AMPLITUDES)])


def print_summary(kind, data, L_ratios, networks, lipschitz, unconstrained):
    """How the training converged, and the MSE and the noise kept of the networks at every L (mean over seeds)."""
    print(f"\n{kind} noise: noise rms {np.sqrt(np.mean(data['noise_train'] ** 2)):.4f}, "
          f"Gaussian peak {data['clean_grid'].max():.4f}, L_true = {L_TRUE:.4f}")
    if kind == "sine":
        print("Slopes the waves need, A omega / L_true:", np.round(AMPLITUDES * FREQUENCIES / L_TRUE, 2))
    print(f"{'L/L_true':>9} {'steps':>8} {'converged':>10} {'MSE':>10} {'kept (%)':>10}")
    for j, ratio in enumerate(L_ratios):
        histories = [networks["histories"][i][j] for i in range(len(SEEDS))]
        steps = np.mean([history["step"][-1] for history in histories])
        converged = sum(history["converged"] for history in histories)
        print(f"{ratio:9.3g} {steps:8.0f} {converged:>8}/{len(SEEDS)}"
              f" {lipschitz['MSE'][:, j].mean():10.2e} {lipschitz['kept'][:, j].mean():10.3g}")
    histories = networks["unconstrained_histories"]
    print(f"{'Unconstr.':>9} {np.mean([h['step'][-1] for h in histories]):8.0f} "
          f"{sum(h['converged'] for h in histories):>8}/{len(SEEDS)}"
          f" {unconstrained['MSE'].mean():10.2e} {unconstrained['kept'].mean():10.3g}")
    print(f"Noisy data itself: MSE {np.mean(data['noise_train'] ** 2):.2e}, kept 100 %")


#### PLOTS ####
def format_spectrum_axes(ax, kind):
    """The shared format of the spectra: logarithmic amplitude; the sine noise with its frequencies marked."""
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_yticks([1e-5, 1e-3, 1e-1])          # no tick labels at the edges, so they do not run into the next row
    ax.set_ylabel("Amplitude")
    if kind == "sine":
        ax.set_xlim(0, OMEGA_MAX_SINE)
        for omega in FREQUENCIES:
            ax.axvline(omega, color=C_MODE, ls=":", lw=0.8)
    else:
        ax.set_xscale("log")                    # white noise covers all frequencies


def plot_data(kind, data, spectra, file_name):
    """The signals (left) and their Fourier spectra (right): the Gaussian, the noise, and the noisy training data."""
    omega, clean_spectrum, noise_spectrum, noisy_spectrum = spectra
    line_width = 1.0 if kind == "sine" else 0.5   # thin lines for the dense white noise
    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(3, 2, figsize=(6.3, 7.5), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.4))
        axs[0, 0].plot(data["x_grid"], data["clean_grid"], color=C_TRUE, lw=1.2)
        axs[0, 0].set_ylabel(r"$f(x)$")
        axs[1, 0].plot(data["x_train"], data["noise_train"], color=C_NOISE, lw=line_width)
        axs[1, 0].set_ylabel(r"$n(x)$")
        axs[2, 0].plot(data["x_train"], data["noisy_train"], color=C_DATA, lw=line_width)
        axs[2, 0].plot(data["x_grid"], data["clean_grid"], color=C_TRUE, lw=1.2, ls="--")
        axs[2, 0].set_ylabel(r"$f(x) + n(x)$")
        axs[2, 0].set_xlabel(r"$x$")

        axs[0, 1].plot(omega, clean_spectrum, color=C_TRUE, lw=1.2)
        axs[1, 1].plot(omega, noise_spectrum, color=C_NOISE, lw=line_width)
        axs[2, 1].plot(omega, noisy_spectrum, color=C_DATA, lw=line_width)
        axs[2, 1].plot(omega, clean_spectrum, color=C_TRUE, lw=1.2, ls="--")
        axs[2, 1].set_xlabel(r"$\omega$")
        for ax in axs[:, 1]:
            format_spectrum_axes(ax, kind)
        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        fig.savefig(f"{PLOT_DIRS[kind]}/{file_name}")
        plt.close(fig)


def plot_fits_and_spectra(kind, data, spectra, L_ratios, networks, file_name):
    """
    One row per network (the L values shown in detail, then the unconstrained network): the fits on the left, with the
    noisy training data and the clean Gaussian, and their Fourier spectra on the right. One line per seed.
    """
    omega, clean_spectrum, _, noisy_spectrum = spectra
    rows = []
    for ratio in L_RATIOS_SHOW:
        j = np.argmin(np.abs(L_ratios - ratio))
        rows.append((rf"${ratio:g}\,L_\mathrm{{true}}$", networks["grid"][:, j]))
    rows.append(("Unconstr.", networks["unconstrained_grid"]))
    data_width = 0.8 if kind == "sine" else 0.5

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(rows), 2, figsize=(6.3, 1.7 * len(rows)), sharex="col",
                                gridspec_kw=dict(hspace=0.12, wspace=0.45))
        for (name, predictions), ax_fit, ax_spectrum in zip(rows, axs[:, 0], axs[:, 1]):
            ax_fit.plot(data["x_train"], data["noisy_train"], color="lightgrey", lw=data_width)
            ax_fit.plot(data["x_grid"], data["clean_grid"], color=C_TRUE, lw=1.0, ls="--")
            for prediction in predictions:
                ax_fit.plot(data["x_grid"], prediction, color=C_NET, alpha=0.7, lw=1.0)
            ax_fit.set_ylabel(name)

            ax_spectrum.plot(omega, noisy_spectrum, color="lightgrey", lw=data_width)
            ax_spectrum.plot(omega, clean_spectrum, color=C_TRUE, lw=1.0, ls="--")
            for prediction in predictions:
                ax_spectrum.plot(omega, fourier_spectrum(data["x_grid"], prediction)[1], color=C_NET, alpha=0.7, lw=0.8)
            format_spectrum_axes(ax_spectrum, kind)

        axs[-1, 0].set_xlabel(r"$x$")
        axs[-1, 1].set_xlabel(r"$\omega$")
        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        handles = [plt.Line2D([], [], color="lightgrey", lw=0.8, label="Training data (noisy)"),
                   plt.Line2D([], [], color=C_TRUE, lw=1.0, ls="--", label="Clean Gaussian"),
                   plt.Line2D([], [], color=C_NET, lw=1.0, label="Network (one line per seed)")]
        if kind == "sine":
            handles.append(plt.Line2D([], [], color=C_MODE, ls=":", lw=0.8, label="Noise frequencies"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
        fig.savefig(f"{PLOT_DIRS[kind]}/{file_name}")
        plt.close(fig)


def plot_scan(kind, name, L_ratios, lipschitz, unconstrained, data_value, file_name, x_ticks=None, linear_y=False):
    """
    One measure against L / L_true: the networks (mean over seeds, the spread min to max as error bars), the
    unconstrained network (dashed, its spread as a band) and the noisy data itself (dotted).
    L_true is marked, and for the sine noise the slope every wave needs (the waves inside the scanned range).
    x_ticks: the labelled ticks of the L axis. Needed for the zoom, which spans less than a decade: matplotlib then
    labels the minor ticks as well, and the labels run into each other (as in Optimal_solution).
    linear_y: a linear y axis starting at zero, instead of the axis in MEASURES. The values are then shown in units of a
    power of ten, given in the axis label, so the tick labels stay short.
    """
    y_label, _, log_y = MEASURES[name]
    unit = 1.0
    if linear_y:
        log_y = False
        exponent = int(np.floor(np.log10(lipschitz[name].max())))
        unit = 10.0 ** exponent
        y_label = rf"{y_label} / $10^{{{exponent}}}$"
    values = lipschitz[name] / unit
    reference = unconstrained[name] / unit
    mean = values.mean(axis=0)
    spread = [mean - values.min(axis=0), values.max(axis=0) - mean]
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=SCAN_SIZE)
        fig.subplots_adjust(**SCAN_MARGINS)
        handles = [ax.errorbar(L_ratios, mean, yerr=spread, fmt="o", color=C_NET, markersize=3.5, lw=1.0, capsize=2,
                               label="Networks"),
                   ax.axhline(reference.mean(), color=C_REF, lw=1.2, ls="--", label="Unconstrained")]
        ax.axhspan(reference.min(), reference.max(), color=C_REF, alpha=0.3, lw=0)
        if data_value is not None:
            handles.append(ax.axhline(data_value / unit, color=C_TRUE, lw=0.8, ls=":", label="Noisy data"))
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")             # L_true
        if kind == "sine":
            needed = AMPLITUDES * FREQUENCIES / L_TRUE             # the slope every wave needs, in units of L_true
            inside = needed[(needed >= L_ratios.min()) & (needed <= L_ratios.max())]
            for ratio in inside:
                mode_line = ax.axvline(ratio, color=C_MODE, ls=":", lw=0.8, label="Sine modes")
            if len(inside) > 0:
                handles.append(mode_line)
        ax.set_xscale("log")
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))   # 0.5 instead of 5x10^-1
            ax.xaxis.set_minor_formatter(plt.NullFormatter())            # minor ticks stay, without labels
        if log_y:
            ax.set_yscale("log")
        if linear_y:
            ax.set_ylim(bottom=0)
        if name == "MAE" and log_y:
            ax.yaxis.set_major_locator(plt.FixedLocator(MAE_Y_TICKS))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))   # 0.005 instead of 5x10^-3
            ax.yaxis.set_minor_formatter(plt.NullFormatter())            # minor ticks stay, without labels
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        ax.legend(handles=handles, **SCAN_LEGEND)
        fig.savefig(f"{PLOT_DIRS[kind]}/{file_name}")
        plt.close(fig)


def plot_surviving_fractions(data, L_ratios, networks, file_name):
    """
    The fraction of every sine wave that survives in the networks (mean over seeds), against the slope it needs
    relative to L, A omega / L. If the bound is what removes the waves, all waves follow one curve, and a wave passes
    when A omega < L.
    """
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for j, ratio in enumerate(L_ratios):
            fractions = np.mean([surviving_fractions(data["x_grid"], prediction) for prediction in networks["grid"][:, j]],
                                axis=0)
            ax.plot(AMPLITUDES * FREQUENCIES / (ratio * L_TRUE), fractions, "o", color=C_NET, markersize=3.5)
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"$A\omega / L$")
        ax.set_ylabel("Fraction that survives")
        fig.savefig(f"{PLOT_DIRS['sine']}/{file_name}")
        plt.close(fig)


def plot_convergence(kind, L_ratios, networks, file_name):
    """
    How the training converged, for the networks shown in detail (seed 0) and the unconstrained network: the training
    loss (MSE to the noisy data, left) and the MSE to the clean Gaussian at the training points (right) against the
    training step.
    """
    colors = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_RATIOS_SHOW)))
    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw=dict(wspace=0.45))
        handles = []
        for ratio, color in zip(L_RATIOS_SHOW, colors):
            history = networks["histories"][0][np.argmin(np.abs(L_ratios - ratio))]
            axs[0].plot(history["step"], history["loss"], color=color, lw=1.2)
            axs[1].plot(history["step"], history["mse_clean"], color=color, lw=1.2)
            handles.append(plt.Line2D([], [], color=color, lw=1.2, label=rf"${ratio:g}\,L_\mathrm{{true}}$"))
        history = networks["unconstrained_histories"][0]
        axs[0].plot(history["step"], history["loss"], color=C_REF, lw=1.2)
        axs[1].plot(history["step"], history["mse_clean"], color=C_REF, lw=1.2)
        handles.append(plt.Line2D([], [], color=C_REF, lw=1.2, label="Unconstr."))

        for ax, y_label in zip(axs, ["MSE to noisy data", "MSE to clean Gaussian"]):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Training step")
            ax.set_ylabel(y_label)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                   handlelength=1.4, columnspacing=1.0)
        fig.savefig(f"{PLOT_DIRS[kind]}/{file_name}")
        plt.close(fig)


def main():
    for kind in NOISE_KINDS:
        print(f"\n========== {kind} noise ==========")
        data = make_data(kind)
        spectra = data_spectra(kind, data)
        plot_data(kind, data, spectra, f"{kind.capitalize()}_noise_gaussian.pdf")

        # Networks (trained until converged, or loaded)
        L_ratios = network_L_ratios(kind)
        networks = saved_or_trained_networks(kind, data, L_ratios)
        lipschitz, unconstrained = network_measures(data, networks)
        print_summary(kind, data, L_ratios, networks, lipschitz, unconstrained)

        plot_fits_and_spectra(kind, data, spectra, L_ratios, networks, f"{kind}_noise_fits_and_spectra.pdf")
        plot_convergence(kind, L_ratios, networks, f"{kind}_noise_convergence.pdf")
        noisy_data = {"MSE": np.mean(data["noise_train"] ** 2), "MAE": np.mean(np.abs(data["noise_train"])),
                      "removed": None, "kept": None}
        for name, (_, ending, _) in MEASURES.items():
            plot_scan(kind, name, L_ratios, lipschitz, unconstrained, noisy_data[name], f"{kind}_noise_{ending}.pdf")
        for name in ERROR_MEASURES:
            plot_scan(kind, name, L_ratios, lipschitz, unconstrained, noisy_data[name],
                      f"{kind}_noise_{MEASURES[name][1]}_linear.pdf", linear_y=True)

        # Zoom around L_true (its own networks; the unconstrained networks are trained again with it, same seeds)
        zoom_ratios = np.logspace(np.log10(ZOOM_RANGE[0]), np.log10(ZOOM_RANGE[1]), N_ZOOM_NETWORKS)
        zoom_networks = saved_or_trained_networks(kind, data, zoom_ratios, "_zoom")
        zoom_lipschitz, zoom_unconstrained = network_measures(data, zoom_networks)
        print("\nZoom around L_true:")
        print_summary(kind, data, zoom_ratios, zoom_networks, zoom_lipschitz, zoom_unconstrained)
        for name in ERROR_MEASURES:
            ending = MEASURES[name][1]
            plot_scan(kind, name, zoom_ratios, zoom_lipschitz, zoom_unconstrained, noisy_data[name],
                      f"{kind}_noise_{ending}_zoom.pdf", x_ticks=ZOOM_X_TICKS)
            plot_scan(kind, name, zoom_ratios, zoom_lipschitz, zoom_unconstrained, noisy_data[name],
                      f"{kind}_noise_{ending}_zoom_linear.pdf", x_ticks=ZOOM_X_TICKS, linear_y=True)
        if kind == "sine":
            plot_surviving_fractions(data, L_ratios, networks, "sine_noise_surviving_fraction.pdf")
    return 0


# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
