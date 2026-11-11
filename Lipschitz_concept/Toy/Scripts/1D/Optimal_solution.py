# GOAL: The best possible fit of a 1D Gaussian with an L-Lipschitz function (the floor), and whether trained
# Lipschitz networks reach it. (Script version of the first part of Toy_gaussian.ipynb.)

# HOW:
#   1. The Lipschitz constant of the Gaussian as a function of sigma, L_true = 1 / (sigma^2 sqrt(2 pi e))
#   2. The floor for a few L below L_true: the fits and their slopes
#   3. MSE and FWHM as a function of L, a wide scan and a zoom around L_true (the kink): the floor alone, and
#      Lipschitz networks trained for every L (mean and spread over seeds) against the floor
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
# One file per scan and number of training steps, e.g. network_scan_zoom_6000steps.pkl ({} = scan suffix, steps)
NETWORK_SCAN_FILE = os.path.join(results_dir("1D", "Optimal_solution"),
                                 "network_scan{}_{}steps_quick.pkl" if QUICK else "network_scan{}_{}steps.pkl")
LOAD_SAVED = True                  # True: use the saved networks if they exist, instead of training again

# The Gaussian
SIGMA = 0.1
N_POINTS = 400 if QUICK else 4000  # the notebook evaluated everything on 2 * 2000 points
SAMPLE_SEED = 0                    # the same random points in every run (the saved networks are trained on them)
X_RANGE = (-5, 5)
L_TRUE = gaussian_lipschitz_constant(SIGMA)
L_SHOW = np.array([0.05, 0.20, 0.65, 0.9])*L_TRUE          # the L of the floor fits and the networks, as fractions of L_true = 24.2

# Networks, as in the notebook
SEEDS = [0, 1] if QUICK else [0, 1, 2]
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5

# The L scans, MSE and FWHM against L / L_true, by file suffix: the range of L / L_true, the labelled x ticks and the
# training steps of the networks. The zoom gets more steps, to test whether the networks close to L_true are undertrained
SCANS = {"": dict(L_range=(0.01, 2.0), x_ticks=None, steps=STEPS),
         "_zoom": dict(L_range=(0.5, 1.01), x_ticks=[0.5, 0.7, 1.0], steps=200 if QUICK else 20000)}
N_SCAN_FLOOR = 8 if QUICK else 40          # L values per scan for the floor (cheap)
N_SCAN_NETWORKS = 3 if QUICK else 12       # L values per scan for the networks (len(SEEDS) networks each, ~1 min per network)

# Figure format
PANEL_SIZE = (3.1, 2.8)            # inches, about the size of a 0.48\textwidth subfigure (two side by side)
# Figures that sit side by side in the report (floor next to networks) get the same fixed size and margins,
# and are saved without tight cropping, so their axes line up
# The legend has two columns (the labels 0.65 L_true are too wide for three), centred on the figure, and room for
# three rows: four L values, the truth and the unconstrained network. The axes are 2.24 inches high.
PAIR_SIZE = (3.1, 3.75)
PAIR_MARGINS = dict(left=0.24, right=0.96, top=0.972, bottom=0.375)
PAIR_LEGEND = dict(loc="upper center", bbox_to_anchor=((0.5 - 0.24) / (0.96 - 0.24), -0.2), ncol=2,
                   handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
C_TRUE, C_MUTED, C_REF = "#1f1f1f", "#6b6a66", "#9a9893"
FLOOR_SHADES = plt.cm.Blues(np.linspace(0.4, 0.95, len(L_SHOW)))       # theory: blue, light -> dark = small -> large L
NETWORK_SHADES = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_SHOW)))   # networks: orange


#### CALCULATIONS ####
def make_gaussian(sigma, n_points, x_range):
    """
    The Gaussian on random points. Must be random to ensure that we do not inherit grid effects.
    The points are sorted, which the slopes (np.gradient), integrals (np.trapezoid) and line plots need. This gives the
    networks no extra information: the loss is a mean over all points, so their order has no effect on training.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    x = np.sort(rng.uniform(x_range[0], x_range[1], n_points))
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


def lipschitz_job(x, y, L, seed, steps=STEPS):
    """The training job of one Lipschitz network with bound L on the points (x, y), for train_many."""
    return dict(kind="lipschitz", seed=seed, steps=steps, lr=LEARNING_RATE, x=x.reshape(-1, 1), y=y,
                model_kwargs=dict(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                  depth=DEPTH, lipschitz_bound=L))


def network_scan(x, y, L_values, steps):
    """
    Lipschitz networks for every L and seed, trained on the clean Gaussian as in 5, for the given number of steps.
    Returns the MSE on the training points and the FWHM of every network, two arrays of shape (seeds, L values).
    """
    jobs = [lipschitz_job(x, y, L, seed, steps) for L in L_values for seed in SEEDS]
    print(f"Training {len(jobs)} networks for the L scan, {steps} steps each")
    results = iter(train_many(jobs))

    MSE = np.zeros((len(SEEDS), len(L_values)))
    FWHM = np.zeros_like(MSE)
    for j in range(len(L_values)):
        for i in range(len(SEEDS)):
            prediction = next(results)["prediction"]
            MSE[i, j] = np.mean((prediction - y) ** 2)
            FWHM[i, j] = FWHM_calculation(x, prediction)
    return MSE, FWHM


def network_scans(x, y, L_true):
    """
    The network scan for every entry of SCANS. Every scan is saved in its own file (NETWORK_SCAN_FILE, with the number
    of steps in the name) together with the training points and the L values, so the plots can be remade without
    training. A saved scan is only used if both are the same as now. Changing the steps of one scan therefore trains
    only that scan again, and the results for the old number of steps are kept.
    Returns scans[suffix] = (L_ratios, MSE, FWHM), with MSE and FWHM of shape (seeds, L values).
    """
    scans = {}
    for suffix, scan in SCANS.items():
        low, high = scan["L_range"]
        L_ratios = np.logspace(np.log10(low), np.log10(high), N_SCAN_NETWORKS)
        file_name = NETWORK_SCAN_FILE.format(suffix, scan["steps"])

        if LOAD_SAVED and os.path.exists(file_name):
            with open(file_name, "rb") as file:
                saved = pickle.load(file)
            if np.array_equal(saved["x"], x) and np.array_equal(saved["L_ratios"], L_ratios):
                print(f"Loaded the network scan from {file_name}")
                scans[suffix] = (L_ratios, saved["MSE"], saved["FWHM"])
                continue
            print(f"The saved network scan in {file_name} used other points or L values: training again")

        MSE, FWHM = network_scan(x, y, L_ratios * L_true, scan["steps"])
        with open(file_name, "wb") as file:
            pickle.dump(dict(x=x, L_ratios=L_ratios, MSE=MSE, FWHM=FWHM), file)
        scans[suffix] = (L_ratios, MSE, FWHM)
    return scans


def train_networks(x, y, L_values):
    """
    Lipschitz networks for every L and seed, and the unconstrained network for every seed, trained on the clean
    Gaussian. Returns fits[L] and slopes[L] with shape (seeds, points), and the same for the unconstrained network.
    """
    x_in = x.reshape(-1, 1)
    jobs = [lipschitz_job(x, y, L, seed) for L in L_values for seed in SEEDS]
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


def saved_or_trained_networks(x, y):
    """
    The networks of 5, for every L in L_SHOW. Loaded from RESULTS_FILE if they were trained on the same points and
    L values as now, otherwise trained again and saved together with the points and L values.
    Returns fits, slopes, fits_unconstrained, slopes_unconstrained (see train_networks).
    """
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "rb") as file:
            saved = pickle.load(file)
        if np.array_equal(saved["x"], x) and np.array_equal(saved["L_values"], L_SHOW):
            print(f"Loaded the networks from {RESULTS_FILE}")
            return saved["fits"], saved["slopes"], saved["fits_unconstrained"], saved["slopes_unconstrained"]
        print(f"The saved networks in {RESULTS_FILE} used other points or L values: training again")

    fits, slopes, fits_unconstrained, slopes_unconstrained = train_networks(x, y, L_SHOW)
    with open(RESULTS_FILE, "wb") as file:
        pickle.dump(dict(x=x, L_values=L_SHOW, fits=fits, slopes=slopes, fits_unconstrained=fits_unconstrained,
                         slopes_unconstrained=slopes_unconstrained), file)
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


def plot_curves(x, truth, curves, shades, x_limits, y_label, file_name, L_true, unconstrained=None, bands=None):
    """
    The shared format of the fit and slope figures: one line per L (shades), labelled as a fraction of L_true like the
    L scans, the truth dashed on top, optionally the unconstrained network dotted and a band (min to max over seeds)
    around every line.
    """
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=PAIR_SIZE)
        fig.subplots_adjust(**PAIR_MARGINS)
        lines = []
        for (L, curve), color in zip(curves.items(), shades):
            if bands is not None:
                ax.fill_between(x, bands[L][0], bands[L][1], color=color, alpha=0.3, lw=0)
            line, = ax.plot(x, curve, color=color, lw=1.4, label=rf"${L / L_true:.2f}\,L_\mathrm{{true}}$")
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


def format_scan_axes(ax, y_label, log_y, x_ticks):
    """
    The shared format of the L scans: log L / L_true axis with L_true marked.
    x_ticks: the labelled ticks of the L axis. Needed for the zoom, which spans less than a decade: matplotlib then
    labels the minor ticks as well (6x10^-1, 7x10^-1, ...) and the labels run into each other.
    """
    ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))   # 0.5 instead of 5x10^-1
        ax.xaxis.set_minor_formatter(plt.NullFormatter())            # minor ticks stay, without labels
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$L / L_\mathrm{true}$")
    ax.set_ylabel(y_label)


def plot_floor_scan(L_ratios, values, y_label, file_name, log_y=False, x_ticks=None):
    """One measure of the floor (MSE or FWHM) against L / L_true, with L_true marked."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.plot(L_ratios, values, "o-", color=FLOOR_SHADES[-1], lw=1.2, markersize=3)
        format_scan_axes(ax, y_label, log_y, x_ticks)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_network_scan(L_floor, floor_values, L_networks, network_values, y_label, file_name, log_y=False,
                      x_ticks=None):
    """
    One measure (MSE or FWHM) of the trained networks against L / L_true: the mean over seeds, with the spread
    (min to max over seeds) as error bars, on top of the floor (line) as the reference.
    """
    mean = network_values.mean(axis=0)
    spread = [mean - network_values.min(axis=0), network_values.max(axis=0) - mean]
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.plot(L_floor, floor_values, color=FLOOR_SHADES[-1], lw=1.2, label="Floor")
        ax.errorbar(L_networks, mean, yerr=spread, fmt="o", color=NETWORK_SHADES[-1], markersize=3.5, lw=1.0,
                    capsize=2, label="Networks")
        format_scan_axes(ax, y_label, log_y, x_ticks)
        ax.legend(loc="upper right")
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
    plot_curves(x, y, floors, FLOOR_SHADES, x_limits, r"$f(x)$", "floor_fits.pdf", L_true)
    floor_slopes = {L: np.gradient(floor, x) for L, floor in floors.items()}
    plot_curves(x, np.gradient(y, x), floor_slopes, FLOOR_SHADES, x_limits, r"$\mathrm{d}f/\mathrm{d}x$",
                "floor_slopes.pdf", L_true)

    # 3. MSE and FWHM against L, a wide scan and a zoom around L_true: the floor alone (fine scan), and the networks
    #    (coarser scan, trained or loaded) with the floor as the reference
    FWHM_true = FWHM_calculation(x, y)
    FWHM_label = r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$"
    networks = network_scans(x, y, L_true)
    for suffix, scan in SCANS.items():
        low, high = scan["L_range"]
        x_ticks = scan["x_ticks"]
        L_ratios = np.logspace(np.log10(low), np.log10(high), N_SCAN_FLOOR)
        MSE, FWHM = floor_scan(x, y, L_ratios * L_true)
        plot_floor_scan(L_ratios, MSE, "MSE", f"floor_MSE_vs_L{suffix}.pdf", x_ticks=x_ticks)
        plot_floor_scan(L_ratios, FWHM / FWHM_true, FWHM_label, f"floor_FWHM_vs_L{suffix}.pdf",
                        log_y=(suffix == ""), x_ticks=x_ticks)

        L_networks, MSE_networks, FWHM_networks = networks[suffix]
        plot_network_scan(L_ratios, MSE, L_networks, MSE_networks, "MSE", f"network_MSE_vs_L{suffix}.pdf",
                          x_ticks=x_ticks)
        plot_network_scan(L_ratios, FWHM / FWHM_true, L_networks, FWHM_networks / FWHM_true, FWHM_label,
                          f"network_FWHM_vs_L{suffix}.pdf", log_y=(suffix == ""), x_ticks=x_ticks)
        print(f"Scan{suffix} done")

    # 4. Integrated errors for several sigma (IAE should not depend on sigma, ISE scales as 1 / sigma)
    sigma_values = np.logspace(-1, 0.5, 3 if QUICK else 10)
    L_ratios = np.logspace(-2, np.log10(2), 6 if QUICK else 30)
    ISE, IAE = integrated_floor_errors(sigma_values, L_ratios, points_per_sigma=10 if QUICK else 40)
    plot_integrated_errors(sigma_values, L_ratios, ISE, IAE, "floor_integrated_errors.pdf")

    # 5. Networks against the floor (saved, so the plots can be remade without training)
    fits, slopes, fits_unconstrained, slopes_unconstrained = saved_or_trained_networks(x, y)

    print_network_table(x, y, floors, fits, slopes, fits_unconstrained, slopes_unconstrained, L_true)
    mean_fits = {L: fits[L].mean(axis=0) for L in L_SHOW}
    bands = {L: (fits[L].min(axis=0), fits[L].max(axis=0)) for L in L_SHOW}
    plot_curves(x, y, mean_fits, NETWORK_SHADES, x_limits, r"$f(x)$", "network_fits.pdf", L_true,
                unconstrained=fits_unconstrained.mean(axis=0), bands=bands)
    mean_slopes = {L: slopes[L].mean(axis=0) for L in L_SHOW}
    plot_curves(x, np.gradient(y, x), mean_slopes, NETWORK_SHADES, x_limits, r"$\mathrm{d}f/\mathrm{d}x$",
                "network_slopes.pdf", L_true, unconstrained=slopes_unconstrained.mean(axis=0))
    return 0


if __name__ == "__main__":
    main()
