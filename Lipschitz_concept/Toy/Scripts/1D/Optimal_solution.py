# GOAL: The best possible fit of a 1D Gaussian with an L-Lipschitz function (the floor), and whether trained
# Lipschitz networks reach it. (Script version of the first part of Toy_gaussian.ipynb.)

# HOW:
#   1. The Lipschitz constant of the Gaussian as a function of sigma, L_true = 1 / (sigma^2 sqrt(2 pi e))
#   2. The floor for a few L below L_true: the fits and their slopes
#   3. MSE and FWHM of the floor as a function of L, a wide scan and a zoom around L_true (the kink)
#   4. Integrated squared and absolute error of the floor for several sigma (scaling check)
#   5. Lipschitz networks trained for the same L as in 2, against the floor, with the unconstrained network

# PLOTS: in Plots/1D/Optimal_solution


import os
import sys
import pickle
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import REPORT_STYLE, gaussian_function, gaussian_lipschitz_constant, FWHM_calculation, plot_dir, results_dir
from Toy_nd_utils import train_many
from lipschitz_floor import lipschitz_floor

QUICK = False                      # True: few points, short training, only to check that everything runs
PLOT_DIR = plot_dir("1D", "Optimal_solution")
RESULTS_FILE = os.path.join(results_dir("1D", "Optimal_solution"), "networks_quick.pkl" if QUICK else "networks.pkl")
LOAD_SAVED = True                  # True: use the saved networks if they exist, instead of training again

# The Gaussian, as in the notebook
SIGMA = 0.1
N_POINTS = 400 if QUICK else 4000  # the notebook evaluated everything on 2 * 2000 points
X_RANGE = (-5, 5)
L_SHOW = [1, 3, 10, 20]            # the L of the floor fits and the networks (all below L_true = 24.2)

# Networks, as in the notebook
SEEDS = [0, 1] if QUICK else [0, 1, 2]
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5

# Figure format
PANEL_SIZE = (3.1, 2.8)            # inches, about the size of a 0.48\textwidth subfigure (two side by side)
# Figures that sit side by side in the report (floor next to networks) get the same fixed size and margins,
# and are saved without tight cropping, so their axes line up
PAIR_SIZE = (3.1, 3.5)
PAIR_MARGINS = dict(left=0.24, right=0.96, top=0.97, bottom=0.33)
PAIR_LEGEND = dict(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3,
                   handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
C_TRUE, C_MUTED, C_REF = "#1f1f1f", "#6b6a66", "#9a9893"
FLOOR_SHADES = plt.cm.Blues(np.linspace(0.4, 0.95, len(L_SHOW)))       # theory: blue, light -> dark = small -> large L
NETWORK_SHADES = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_SHOW)))   # networks: orange


#### CALCULATIONS ####
def make_gaussian(sigma, n_points, x_range):
    """The Gaussian on evenly spaced points. Returns x and f(x) as flat arrays."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    return x, gaussian_function(x, 0.0, sigma)


def floor_fits(x, y, L_values):
    """The floor for every L. Returns floors[L]."""
    floors = {}
    for L in L_values:
        floors[L], mse = lipschitz_floor(x, y, L)
        print(f"Floor L = {L:g}: MSE {mse:.3e}, peak {floors[L].max():.3f} (true {y.max():.3f})")
    return floors


def floor_scan(x, y, L_values):
    """MSE and FWHM of the floor for every L. Returns two arrays."""
    MSE, FWHM = [], []
    for L in L_values:
        floor, mse = lipschitz_floor(x, y, L)
        MSE.append(mse)
        FWHM.append(FWHM_calculation(x, floor))
    return np.array(MSE), np.array(FWHM)


def integrated_floor_errors(sigma_values, L_ratios, points_per_sigma):
    """
    Integrated squared and absolute error of the floor for every sigma (rows) and L / L_true (columns).
    The grid is scaled with sigma, so every combination is resolved equally well and contains the whole floor.
    """
    ISE = np.zeros((len(sigma_values), len(L_ratios)))
    IAE = np.zeros_like(ISE)
    for i, sigma in enumerate(sigma_values):
        L_true = gaussian_lipschitz_constant(sigma)
        for j, ratio in enumerate(L_ratios):
            L = ratio * L_true
            tent_half_width = np.sqrt(1 / L)          # the floor of a unit-area peak is a tent of half-width sqrt(1 / L)
            half_width = max(6 * sigma, 1.5 * tent_half_width)
            x = np.linspace(-half_width, half_width, int(2 * half_width / sigma * points_per_sigma) + 1)
            y = gaussian_function(x, 0.0, sigma)
            floor, _ = lipschitz_floor(x, y, L)
            ISE[i, j] = np.trapezoid((floor - y) ** 2, x)
            IAE[i, j] = np.trapezoid(np.abs(floor - y), x)
        print(f"Integrated errors: sigma = {sigma:.2f} done")
    return ISE, IAE


def train_networks(x, y, L_values):
    """
    Lipschitz networks for every L and seed, and the unconstrained network for every seed, trained on the clean
    Gaussian. Returns fits[L] and slopes[L] with shape (seeds, points), and the same for the unconstrained network.
    """
    x_in = x.reshape(-1, 1)
    jobs = []
    for L in L_values:
        for seed in SEEDS:
            jobs.append(dict(kind="lipschitz", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x_in, y=y,
                             model_kwargs=dict(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                               depth=DEPTH, lipschitz_bound=L)))
    for seed in SEEDS:
        jobs.append(dict(kind="unconstrained", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x_in, y=y,
                         model_kwargs=dict(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)))
    print(f"Training {len(jobs)} networks")
    results = iter(train_many(jobs))

    fits, slopes = {}, {}
    for L in L_values:
        runs = [next(results) for _ in SEEDS]
        fits[L] = np.array([run["prediction"] for run in runs])
        slopes[L] = np.array([run["gradient"].ravel() for run in runs])
    runs = [next(results) for _ in SEEDS]
    fits_unconstrained = np.array([run["prediction"] for run in runs])
    slopes_unconstrained = np.array([run["gradient"].ravel() for run in runs])
    return fits, slopes, fits_unconstrained, slopes_unconstrained


def print_network_table(x, y, floors, fits, slopes, fits_unconstrained, slopes_unconstrained, L_true):
    """The networks (mean +- std over seeds) against the floor: used slope, peak, FWHM and integrated squared error."""
    print(f"{'L':>5} {'L_hat / L':>14} {'peak: net | floor':>22} {'FWHM: net | floor':>22} {'ISE: net | floor':>26}")
    for L in fits:
        L_hat = np.abs(slopes[L]).max(axis=1) / L
        peaks = fits[L].max(axis=1)
        widths = np.array([FWHM_calculation(x, f) for f in fits[L]])
        ise = np.array([np.trapezoid((f - y) ** 2, x) for f in fits[L]])
        print(f"{L:5g} {L_hat.mean():8.3f} ± {L_hat.std():.3f}"
              f"   {peaks.mean():7.3f} | {floors[L].max():6.3f}"
              f"   {widths.mean():7.3f} | {FWHM_calculation(x, floors[L]):6.3f}"
              f"   {ise.mean():9.4f} | {np.trapezoid((floors[L] - y) ** 2, x):9.4f}")
    largest_slope = np.abs(slopes_unconstrained).max(axis=1)
    widths = np.array([FWHM_calculation(x, f) for f in fits_unconstrained])
    ise = np.array([np.trapezoid((f - y) ** 2, x) for f in fits_unconstrained])
    print(f"Unconstrained: largest slope {largest_slope.mean():.1f} ± {largest_slope.std():.1f} (L_true = {L_true:.1f}), "
          f"peak {fits_unconstrained.max(axis=1).mean():.3f} (true {y.max():.3f}), "
          f"FWHM {widths.mean():.3f} (true {FWHM_calculation(x, y):.3f}), ISE {ise.mean():.2e}")


def zoom_limits(x, y, floors):
    """The x range where the floors differ visibly from the truth (1% of the peak), with a small margin."""
    differs = np.any([np.abs(floor - y) > 0.01 * y.max() for floor in floors.values()], axis=0)
    half_width = 1.15 * np.abs(x[differs]).max() if differs.any() else 4 * SIGMA
    return max(x.min(), -half_width), min(x.max(), half_width)


#### PLOTS ####
def plot_lipschitz_constant(sigma_values, file_name):
    """L_true against sigma (log y)."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.plot(sigma_values, gaussian_lipschitz_constant(sigma_values), color=C_TRUE, lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$L_\mathrm{true} = (\sigma^2 \sqrt{2\pi e})^{-1}$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_curves(x, truth, curves, shades, x_limits, y_label, file_name, unconstrained=None, bands=None):
    """
    The shared format of the fit and slope figures: one line per L (shades), the truth dashed on top,
    optionally the unconstrained network dotted and a band (min to max over seeds) around every line.
    """
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=PAIR_SIZE)
        fig.subplots_adjust(**PAIR_MARGINS)
        lines = []
        for (L, curve), color in zip(curves.items(), shades):
            if bands is not None:
                ax.fill_between(x, bands[L][0], bands[L][1], color=color, alpha=0.3, lw=0)
            line, = ax.plot(x, curve, color=color, lw=1.4, label=rf"$L = {L:g}$")
            lines.append(line)
        if unconstrained is not None:
            unconstrained_line, = ax.plot(x, unconstrained, color=C_REF, lw=1.2, ls=":", label="Unconstr.")
        truth_line, = ax.plot(x, truth, color=C_TRUE, lw=1.2, ls="--", label="Truth")   # last, so it stays visible
        lines.append(truth_line)
        if unconstrained is not None:
            lines.append(unconstrained_line)
        ax.set_xlim(*x_limits)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(y_label)
        ax.legend(handles=lines, **PAIR_LEGEND)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_floor_scan(L_ratios, values, y_label, file_name, log_y=False):
    """One measure of the floor (MSE or FWHM) against L / L_true, with L_true marked."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.plot(L_ratios, values, "o-", color=FLOOR_SHADES[-1], lw=1.2, markersize=3)
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_integrated_errors(sigma_values, L_ratios, ISE, IAE, file_name):
    """The squared and the absolute error side by side, one shared legend underneath."""
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_values)))
    # Zeros (L >= L_true, where the floor is the truth) cannot be drawn on a log axis
    ISE = np.where(ISE > 0, ISE, np.nan)
    IAE = np.where(IAE > 0, IAE, np.nan)

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw=dict(wspace=0.45))
        for values, ax, y_label in [(ISE, axs[0], r"$\int (g - f)^2\,\mathrm{d}x$"),
                                    (IAE, axs[1], r"$\int |g - f|\,\mathrm{d}x$")]:
            for i, sigma in enumerate(sigma_values):
                ax.plot(L_ratios, values[i], color=colors[i], lw=1.2, label=rf"$\sigma = {sigma:.2f}$")
            ax.axvline(1, color=C_MUTED, lw=0.6, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$L / L_\mathrm{true}$")
            ax.set_ylabel(y_label)
        axs[0].yaxis.set_major_locator(plt.LogLocator(numticks=4))   # the squared error spans many decades
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5,
                   handlelength=1.0, columnspacing=1.0)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def main():
    x, y = make_gaussian(SIGMA, N_POINTS, X_RANGE)
    L_true = gaussian_lipschitz_constant(SIGMA)
    print(f"sigma = {SIGMA}, L_true = {L_true:.2f}, {N_POINTS} points on {X_RANGE}")

    # 1. The Lipschitz constant
    plot_lipschitz_constant(np.logspace(-1, 0.7, 40), "gaussian_lipschitz_constant.pdf")

    # 2. Floor fits and their slopes
    floors = floor_fits(x, y, L_SHOW)
    x_limits = zoom_limits(x, y, floors)
    plot_curves(x, y, floors, FLOOR_SHADES, x_limits, r"$f(x)$", "floor_fits.pdf")
    floor_slopes = {L: np.gradient(floor, x) for L, floor in floors.items()}
    plot_curves(x, np.gradient(y, x), floor_slopes, FLOOR_SHADES, x_limits, r"$\mathrm{d}f/\mathrm{d}x$",
                "floor_slopes.pdf")

    # 3. MSE and FWHM against L, a wide scan and a zoom around L_true
    FWHM_true = FWHM_calculation(x, y)
    n_scan = 8 if QUICK else 40
    for L_ratios, suffix in [(np.logspace(np.log10(0.01), np.log10(2), n_scan), ""),
                             (np.logspace(np.log10(0.5), np.log10(1.01), n_scan), "_zoom")]:
        MSE, FWHM = floor_scan(x, y, L_ratios * L_true)
        plot_floor_scan(L_ratios, MSE, "MSE", f"floor_MSE_vs_L{suffix}.pdf")
        plot_floor_scan(L_ratios, FWHM / FWHM_true, r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$",
                        f"floor_FWHM_vs_L{suffix}.pdf", log_y=(suffix == ""))
        print(f"Floor scan{suffix} done")

    # 4. Integrated errors for several sigma (IAE should not depend on sigma, ISE scales as 1 / sigma)
    sigma_values = np.logspace(-1, 0.5, 3 if QUICK else 10)
    L_ratios = np.logspace(-2, np.log10(2), 6 if QUICK else 30)
    ISE, IAE = integrated_floor_errors(sigma_values, L_ratios, points_per_sigma=10 if QUICK else 40)
    plot_integrated_errors(sigma_values, L_ratios, ISE, IAE, "floor_integrated_errors.pdf")

    # 5. Networks against the floor (saved, so the plots can be remade without training)
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "rb") as file:
            fits, slopes, fits_unconstrained, slopes_unconstrained = pickle.load(file)
        print(f"Loaded the networks from {RESULTS_FILE}")
    else:
        fits, slopes, fits_unconstrained, slopes_unconstrained = train_networks(x, y, L_SHOW)
        with open(RESULTS_FILE, "wb") as file:
            pickle.dump((fits, slopes, fits_unconstrained, slopes_unconstrained), file)

    print_network_table(x, y, floors, fits, slopes, fits_unconstrained, slopes_unconstrained, L_true)
    mean_fits = {L: fits[L].mean(axis=0) for L in L_SHOW}
    bands = {L: (fits[L].min(axis=0), fits[L].max(axis=0)) for L in L_SHOW}
    plot_curves(x, y, mean_fits, NETWORK_SHADES, x_limits, r"$f(x)$", "network_fits.pdf",
                unconstrained=fits_unconstrained.mean(axis=0), bands=bands)
    mean_slopes = {L: slopes[L].mean(axis=0) for L in L_SHOW}
    plot_curves(x, np.gradient(y, x), mean_slopes, NETWORK_SHADES, x_limits, r"$\mathrm{d}f/\mathrm{d}x$",
                "network_slopes.pdf", unconstrained=slopes_unconstrained.mean(axis=0))
    return 0


if __name__ == "__main__":
    main()
