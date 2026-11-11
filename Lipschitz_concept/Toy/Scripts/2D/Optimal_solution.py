# GOAL: The best possible fit of a 2D Gaussian with an L-Lipschitz function (the floor), and whether trained
# Lipschitz networks reach it. (Script version of the first part of Toy_gaussian_2d.ipynb, the 2D version of
# Scripts/1D/Optimal_solution.py.)

# HOW:
#   1. The Lipschitz constant of the Gaussian as a function of sigma, for 1, 2 and 3 dimensions
#   2. The floor for a few L below L_true: cuts along an axis and along the diagonal, the slopes, the half-maximum
#      contours and where the slope budget is used up
#   3. MSE and widths (FWHM) of the floor as a function of L, a wide scan and a zoom around L_true
#   4. Integrated squared and absolute error of the floor for several sigma (scaling check)
#   5. Lipschitz networks trained for the same L as in 2, against the floor, with the unconstrained network. The
#      networks are trained on random points drawn uniformly in the box, and evaluated on the grid (all unseen points)
# One norm is used everywhere (INPUT_NORM): the input norm of the networks, the floor and L_true. In the l2 norm the
# floor of a round Gaussian stays round, in the l1 norm (used in the notebook) it turns into a diamond.
# The floors and the networks are saved, so the plots can be remade without computing them again.
# Runtime of a full run: about 40 minutes (the floors and the networks are computed several at a time).

# PLOTS: in Plots/2D/Optimal_solution


import os
import sys
import multiprocessing as mp
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import REPORT_STYLE, plot_dir, results_dir, save_results, load_results
from Toy_nd_utils import (gaussian_nd, isotropic_cov, gaussian_lipschitz_constant_nd, isotropic_lipschitz_constant,
                          pyramid_radius, make_grid, grid_integral, grid_gradient, central_line, profile,
                          fwhm_equivalent, fwhm_along, train_many)
from lipschitz_floor import lipschitz_floor_nd

QUICK = False                      # True: small grids, short training, only to check that everything runs
PLOT_DIR = plot_dir("2D", "Optimal_solution")
FLOORS_FILE = os.path.join(results_dir("2D", "Optimal_solution"), "floors_quick.pkl" if QUICK else "floors.pkl")
NETWORKS_FILE = os.path.join(results_dir("2D", "Optimal_solution"), "networks_quick.pkl" if QUICK else "networks.pkl")
LOAD_SAVED = True                  # True: use the saved floors and networks if they exist, instead of computing again
INPUT_NORM = "l2"                  # the norm of the networks, the floor and L_true: "l2", or "l1" as in the notebook

# The Gaussian, as in the notebook: isotropic with sigma = 0.1
DIM = 2
SIGMA = 0.1
L_RATIOS_SHOW = [0.04, 0.12, 0.4, 0.8]   # L / L_true of the floor fits and the networks (about the L = 1, 3, 10, 20 of 1D)
POINTS_PER_SIGMA = 4 if QUICK else 15    # grid of the floor fits and the networks

# The scans and the integrated errors, on coarser grids
POINTS_PER_SIGMA_SCAN = 3 if QUICK else 8
L_RATIOS_SCAN = np.logspace(np.log10(0.01), np.log10(2), 6 if QUICK else 40)
L_RATIOS_ZOOM = np.logspace(np.log10(0.4), np.log10(1.01), 6 if QUICK else 40)
SIGMA_VALUES = np.logspace(-1, 0.5, 2 if QUICK else 5)
L_RATIOS_INTEGRATED = np.logspace(-2, np.log10(2), 4 if QUICK else 20)
FLOOR_WORKERS = 5                  # floors computed side by side (an l2 floor on the large grid: 1.5 min, 1.7 GB memory)

# Networks, as in the notebook
SEEDS = [0, 1] if QUICK else [0, 1, 2]
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5
# Training data: random points drawn uniformly in the box of the grid, the same sample for every network.
# Uniform sampling weights the box evenly, like the grid, so the floor on the grid stays the right reference.
N_TRAIN = 1300 if QUICK else 17424  # as many points as the earlier training grid (every second point of the 263^2 grid)
SAMPLE_SEED = 0

# Directions of the cuts through the peak
AXIS_DIRECTION = np.array([1.0, 0.0])
DIAGONAL_DIRECTION = np.array([1.0, 1.0])

# Figure format, as in 1D
PANEL_SIZE = (3.1, 2.8)            # inches, about the size of a 0.48\textwidth subfigure (two side by side)
# Figures that sit side by side in the report (floor next to networks) get the same fixed size and margins,
# and are saved without tight cropping, so their axes line up
PAIR_SIZE = (3.1, 3.7)
PAIR_MARGINS = dict(left=0.24, right=0.96, top=0.97, bottom=0.37)
PAIR_LEGEND = dict(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2,
                   handlelength=1.0, columnspacing=0.8, handletextpad=0.4, frameon=False)
C_TRUE, C_MUTED, C_REF = "#1f1f1f", "#6b6a66", "#9a9893"
FLOOR_SHADES = plt.cm.Blues(np.linspace(0.4, 0.95, len(L_RATIOS_SHOW)))       # theory: blue, light -> dark = small -> large L
NETWORK_SHADES = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_RATIOS_SHOW)))   # networks: orange


#### CALCULATIONS ####
def lipschitz_constant(sigma):
    """L_true of the isotropic Gaussian in INPUT_NORM (for a round Gaussian the same in l1 and l2)."""
    return gaussian_lipschitz_constant_nd(isotropic_cov(sigma, DIM), norm=INPUT_NORM)


def dual_norm(gradient):
    """
    The dual of INPUT_NORM along the last axis: the norm the gradient is bounded in.
    l1 -> the largest partial derivative, l2 -> the length of the gradient.
    """
    if INPUT_NORM == "l1":
        return np.max(np.abs(gradient), axis=-1)
    return np.linalg.norm(gradient, axis=-1)


def grid_half_width(sigma, L_smallest):
    """
    Half-width of a grid that is wide enough for the floor with the smallest L to come down to zero inside it.
    pyramid_radius is the size of the l1 pyramid, the widest of the possible floor shapes, so it is safe for l2 as well.
    """
    return max(6 * sigma, 1.2 * pyramid_radius(L_smallest, DIM))


def make_gaussian(sigma, half_width, points_per_sigma):
    """The Gaussian on [-half_width, half_width]^DIM, with an odd number of points so the peak sits on a grid point."""
    n_points = 2 * int(half_width / sigma * points_per_sigma) + 1
    axes, X = make_grid(half_width, n_points, DIM)
    return axes, X, gaussian_nd(X, np.zeros(DIM), isotropic_cov(sigma, DIM))


def _floor_worker(job):
    """One floor, for floors_many. job = (y, axes, L)."""
    y, axes, L = job
    floor, _ = lipschitz_floor_nd(y, axes, L, norm=INPUT_NORM)
    return floor


def floors_many(jobs):
    """The floor for every job (y, axes, L), FLOOR_WORKERS at a time in separate processes. Returns them in order."""
    with mp.get_context("spawn").Pool(min(len(jobs), FLOOR_WORKERS)) as pool:
        return pool.map(_floor_worker, jobs, chunksize=1)


def widths(f, axes):
    """The three widths: equivalent FWHM (the diameter of the disc with the same area), along an axis and the diagonal."""
    return {"equivalent": fwhm_equivalent(f, axes),
            "axis": fwhm_along(axes, f, AXIS_DIRECTION),
            "diagonal": fwhm_along(axes, f, DIAGONAL_DIRECTION)}


def floor_fits(y, axes, L_true):
    """The floor for every L in L_RATIOS_SHOW. Returns floors[L / L_true]."""
    floors = dict(zip(L_RATIOS_SHOW, floors_many([(y, axes, r * L_true) for r in L_RATIOS_SHOW])))
    for r, floor in floors.items():
        print(f"Floor L = {r:g} L_true: MSE {np.mean((floor - y) ** 2):.3e}, peak {floor.max():.3f} (true {y.max():.3f})")
    return floors


def floor_scan(y, axes, L_values):
    """MSE and the three widths of the floor for every L. Returns a dictionary of arrays."""
    floors = floors_many([(y, axes, L) for L in L_values])
    floor_widths = [widths(floor, axes) for floor in floors]
    scan = {"MSE": np.array([np.mean((floor - y) ** 2) for floor in floors])}
    for name in ["equivalent", "axis", "diagonal"]:
        scan[name] = np.array([w[name] for w in floor_widths])
    return scan


def integrated_floor_errors(sigma_values, L_ratios):
    """
    Integrated squared and absolute error of the floor for every sigma (rows) and L / L_true (columns).
    The grid is scaled with sigma, so every combination is resolved equally well and contains the whole floor.
    """
    jobs = []
    for sigma in sigma_values:
        L_true = lipschitz_constant(sigma)
        for ratio in L_ratios:
            axes, _, y = make_gaussian(sigma, grid_half_width(sigma, ratio * L_true), POINTS_PER_SIGMA_SCAN)
            jobs.append((y, axes, ratio * L_true))
    floors = floors_many(jobs)

    ISE = np.zeros((len(sigma_values), len(L_ratios)))
    IAE = np.zeros_like(ISE)
    for k, ((y, axes, _), floor) in enumerate(zip(jobs, floors)):
        i, j = divmod(k, len(L_ratios))
        ISE[i, j] = grid_integral((floor - y) ** 2, axes)
        IAE[i, j] = grid_integral(np.abs(floor - y), axes)

    # Scaling check: below L_true, the IAE and sigma^DIM * ISE should not depend on sigma
    below = L_ratios < 1
    scaled_ISE = sigma_values[:, None] ** DIM * ISE
    print(f"Integrated errors, largest relative spread over sigma: "
          f"IAE {np.max(np.ptp(IAE[:, below], axis=0) / IAE[:, below].mean(axis=0)):.1e}, "
          f"sigma^{DIM} ISE {np.max(np.ptp(scaled_ISE[:, below], axis=0) / scaled_ISE[:, below].mean(axis=0)):.1e}")
    return ISE, IAE


def sample_training_points(half_width):
    """N_TRAIN points drawn uniformly in [-half_width, half_width]^DIM, and the clean Gaussian at those points."""
    rng = np.random.default_rng(SAMPLE_SEED)
    x_train = rng.uniform(-half_width, half_width, size=(N_TRAIN, DIM))
    return x_train, gaussian_nd(x_train, np.zeros(DIM), isotropic_cov(SIGMA, DIM))


def train_networks(X, y, x_train, y_train, L_true):
    """
    Lipschitz networks for every L in L_RATIOS_SHOW and seed, and the unconstrained network for every seed, trained on
    the clean Gaussian at the random points x_train. They are evaluated on the grid X (for the plots and measures)
    and at the training points (for the training MSE).
    Returns a dictionary with fits[L / L_true] (seeds, grid), gradients[L / L_true] (seeds, grid, DIM) and
    train_mse[L / L_true] (seeds), and the same for the unconstrained network.
    """
    x_grid = X.reshape(-1, DIM)
    n_grid = len(x_grid)
    x_eval = np.concatenate([x_grid, x_train])     # first the grid, then the training points
    jobs = []
    for r in L_RATIOS_SHOW:
        for seed in SEEDS:
            jobs.append(dict(kind="lipschitz", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x_train, y=y_train,
                             x_eval=x_eval,
                             model_kwargs=dict(input_dim=DIM, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                               depth=DEPTH, lipschitz_bound=r * L_true, input_norm=INPUT_NORM)))
    for seed in SEEDS:
        jobs.append(dict(kind="unconstrained", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x_train, y=y_train,
                         x_eval=x_eval, model_kwargs=dict(input_dim=DIM, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)))
    print(f"Training {len(jobs)} networks on {len(x_train)} random points, evaluated on the {n_grid} grid points")
    results = iter(train_many(jobs))

    def collect(runs):
        """The grid fits, grid gradients and training MSE of the runs of one network (one per seed)."""
        fits = np.array([run["prediction"][:n_grid].reshape(y.shape) for run in runs])
        gradients = np.array([run["gradient"][:n_grid].reshape(y.shape + (DIM,)) for run in runs])
        train_mse = np.array([np.mean((run["prediction"][n_grid:] - y_train) ** 2) for run in runs])
        return fits, gradients, train_mse

    networks = {"fits": {}, "gradients": {}, "train_mse": {}}
    for r in L_RATIOS_SHOW:
        networks["fits"][r], networks["gradients"][r], networks["train_mse"][r] = collect([next(results) for _ in SEEDS])
    (networks["fits_unconstrained"], networks["gradients_unconstrained"],
     networks["train_mse_unconstrained"]) = collect([next(results) for _ in SEEDS])
    return networks


def print_network_table(axes, y, floors, networks, L_true):
    """
    The networks (mean over seeds) against the floor: used slope, peak, equivalent FWHM, and the MSE at the training
    points, on the grid (unseen points) and of the floor on the grid.
    """
    print(f"{'L/L_true':>8} {'L_hat / L':>14} {'peak: net | floor':>20} {'FWHM: net | floor':>20} "
          f"{'MSE: train | grid | floor':>32}")
    for r in L_RATIOS_SHOW:
        fits = networks["fits"][r]
        L_hat = np.array([dual_norm(g).max() for g in networks["gradients"][r]]) / (r * L_true)
        peaks = np.array([f.max() for f in fits])
        fwhm = np.array([fwhm_equivalent(f, axes) for f in fits])
        grid_mse = np.array([np.mean((f - y) ** 2) for f in fits])
        print(f"{r:8g} {L_hat.mean():8.3f} ± {L_hat.std():.3f}"
              f"   {peaks.mean():7.3f} | {floors[r].max():7.3f}"
              f"   {fwhm.mean():7.4f} | {fwhm_equivalent(floors[r], axes):7.4f}"
              f"   {networks['train_mse'][r].mean():9.2e} | {grid_mse.mean():9.2e} | {np.mean((floors[r] - y) ** 2):9.2e}")
    fits = networks["fits_unconstrained"]
    largest_slope = np.array([dual_norm(g).max() for g in networks["gradients_unconstrained"]])
    print(f"Unconstrained: largest slope {largest_slope.mean():.1f} ± {largest_slope.std():.1f} (L_true = {L_true:.1f}), "
          f"peak {np.mean([f.max() for f in fits]):.3f} (true {y.max():.3f}), "
          f"FWHM {np.mean([fwhm_equivalent(f, axes) for f in fits]):.4f} (true {fwhm_equivalent(y, axes):.4f}), "
          f"MSE train {networks['train_mse_unconstrained'].mean():.2e} | grid {np.mean([np.mean((f - y) ** 2) for f in fits]):.2e}")


def zoom_limit(t, truth, cuts):
    """The distance from the peak where the cuts differ visibly from the truth (1% of the peak), with a small margin."""
    differs = np.any([np.abs(cut - truth) > 0.01 * np.nanmax(truth) for cut in cuts.values()], axis=0)
    return min(t.max(), 1.15 * np.abs(t[differs]).max()) if differs.any() else 4 * SIGMA


#### PLOTS ####
def plot_lipschitz_constant(sigma_values, file_name):
    """L_true against sigma for 1, 2 and 3 dimensions (log-log)."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for n, ls in zip([1, 2, 3], [":", "-", "--"]):
            ax.plot(sigma_values, isotropic_lipschitz_constant(sigma_values, n), color=C_TRUE, ls=ls, lw=1.2,
                    label=f"$n = {n}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$L_\mathrm{true}$")
        ax.legend(frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_training_points(x_train, axes, y, limit, file_name):
    """The random training points inside the zoomed window, with the half-maximum contour of the Gaussian (dashed)."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.scatter(x_train[:, 0], x_train[:, 1], s=1, color=C_MUTED, linewidths=0)
        ax.contour(axes[0], axes[1], y.T, levels=[y.max() / 2], colors=[C_TRUE], linewidths=1.2, linestyles="--")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_curves(x, truth, curves, shades, x_limit, x_label, y_label, file_name, unconstrained=None, bands=None):
    """
    The shared format of the cut and slope figures, as in 1D: one line per L / L_true (shades), the truth dashed on top,
    optionally the unconstrained network dotted and a band (min to max over seeds) around every line.
    """
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=PAIR_SIZE)
        fig.subplots_adjust(**PAIR_MARGINS)
        lines = []
        for (r, curve), color in zip(curves.items(), shades):
            if bands is not None:
                ax.fill_between(x, bands[r][0], bands[r][1], color=color, alpha=0.3, lw=0)
            line, = ax.plot(x, curve, color=color, lw=1.4, label=rf"${r:g}\,L_\mathrm{{true}}$")
            lines.append(line)
        if unconstrained is not None:
            unconstrained_line, = ax.plot(x, unconstrained, color=C_REF, lw=1.2, ls=":", label="Unconstr.")
        truth_line, = ax.plot(x, truth, color=C_TRUE, lw=1.2, ls="--", label="Truth")   # last, so it stays visible
        lines.append(truth_line)
        if unconstrained is not None:
            lines.append(unconstrained_line)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(handles=lines, **PAIR_LEGEND)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_contours(axes, truth, fields, shades, limit, file_name, floors=None):
    """
    Half-maximum contours of every field (one per L / L_true, in shades) and of the truth (dashed).
    With floors given (the network figure), the half-maximum contours of the floors are drawn dashed underneath.
    Same fixed size as the cut figures, so the floor and network contours can sit side by side.
    """
    with plt.rc_context(REPORT_STYLE), plt.rc_context({"savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=PAIR_SIZE)
        fig.subplots_adjust(**PAIR_MARGINS)
        handles = []
        for (r, field), color, floor_color in zip(fields.items(), shades, FLOOR_SHADES):
            if floors is not None:
                ax.contour(axes[0], axes[1], floors[r].T, levels=[floors[r].max() / 2], colors=[floor_color],
                           linewidths=1.0, linestyles="--")
            ax.contour(axes[0], axes[1], field.T, levels=[field.max() / 2], colors=[color], linewidths=1.4)
            handles.append(plt.Line2D([], [], color=color, lw=1.4, label=rf"${r:g}\,L_\mathrm{{true}}$"))
        ax.contour(axes[0], axes[1], truth.T, levels=[truth.max() / 2], colors=[C_TRUE], linewidths=1.2,
                   linestyles="--")
        handles.append(plt.Line2D([], [], color=C_TRUE, lw=1.2, ls="--", label="Truth"))
        if floors is not None:
            handles.append(plt.Line2D([], [], color=FLOOR_SHADES[-1], lw=1.0, ls="--", label="Floor"))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))       # few ticks, so the labels fit the fixed margins
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend(handles=handles, **PAIR_LEGEND)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_slope_budget(axes, floors, L_true, limit, file_name):
    """
    Where the floors use their full slope budget: the slope (in the dual norm of INPUT_NORM) divided by L, one map
    per L / L_true, in a 2 x 2 grid (in one row the tick labels of the small maps overlap).
    Central differences round off the kinks, so the maps can dip below 1 there.
    """
    extent = [axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]]
    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex=True, sharey=True,
                                gridspec_kw=dict(wspace=0.1, hspace=0.25))
        for ax, (r, floor) in zip(axs.flat, floors.items()):
            budget = dual_norm(grid_gradient(floor, axes)) / (r * L_true)
            image = ax.imshow(budget.T, origin="lower", extent=extent, vmin=0, vmax=1, cmap="Blues")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_title(rf"${r:g}\,L_\mathrm{{true}}$")
        for ax in axs[-1]:
            ax.set_xlabel(r"$x_1$")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$x_2$")
        fig.colorbar(image, ax=axs, shrink=0.8, label=r"Slope / $L$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_floor_scan(L_ratios, curves, y_label, file_name, log_y=False):
    """
    One measure of the floor against L / L_true, with L_true marked. curves: {label: values}; with more than one
    curve (the three widths) they are drawn solid, dashed and dotted, with a legend.
    """
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for (label, values), ls in zip(curves.items(), ["-", "--", ":"]):
            ax.plot(L_ratios, values, ls, color=FLOOR_SHADES[-1], lw=1.2, label=label)
        ax.axvline(1.0, color=C_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())          # 1, 2, 3 instead of 10^0, 2x10^0
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        if len(curves) > 1:
            ax.legend(frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_integrated_errors(sigma_values, L_ratios, ISE, IAE, file_name):
    """The squared and the absolute error side by side, one shared legend underneath (as in 1D)."""
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_values)))
    # Zeros (L >= L_true, where the floor is the truth) cannot be drawn on a log axis
    ISE = np.where(ISE > 0, ISE, np.nan)
    IAE = np.where(IAE > 0, IAE, np.nan)

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw=dict(wspace=0.45))
        for values, ax, y_label in [(ISE, axs[0], r"$\int (g - f)^2\,\mathrm{d}^2x$"),
                                    (IAE, axs[1], r"$\int |g - f|\,\mathrm{d}^2x$")]:
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
                   handlelength=1.0, columnspacing=1.0, frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def main():
    L_true = lipschitz_constant(SIGMA)
    half_width = grid_half_width(SIGMA, min(L_RATIOS_SHOW) * L_true)
    axes, X, y = make_gaussian(SIGMA, half_width, POINTS_PER_SIGMA)
    print(f"sigma = {SIGMA}, norm {INPUT_NORM}, L_true = {L_true:.2f}, "
          f"grid {len(axes[0])}^{DIM} points on [-{half_width:.3f}, {half_width:.3f}]^{DIM}")

    # 1. The Lipschitz constant, and a numerical check of the formula on a fine grid
    axes_check, _, y_check = make_gaussian(SIGMA, 5 * SIGMA, 80)
    print(f"L_true from the formula {L_true:.3f}, largest slope on a fine grid "
          f"{dual_norm(grid_gradient(y_check, axes_check)).max():.3f}")
    plot_lipschitz_constant(np.logspace(-1, 0.7, 40), "gaussian_lipschitz_constant.pdf")

    # The floors: fits, scans and integrated errors (saved, so the plots can be remade without computing them again)
    if LOAD_SAVED and os.path.exists(FLOORS_FILE):
        saved = load_results(FLOORS_FILE)
    else:
        saved = {"floors": floor_fits(y, axes, L_true)}
        half_width_scan = grid_half_width(SIGMA, L_RATIOS_SCAN.min() * L_true)   # one grid for both scans
        axes_scan, _, y_scan = make_gaussian(SIGMA, half_width_scan, POINTS_PER_SIGMA_SCAN)
        print(f"Scan grid: {len(axes_scan[0])}^{DIM} points")
        saved["widths_true"] = widths(y_scan, axes_scan)
        saved["scan"] = floor_scan(y_scan, axes_scan, L_RATIOS_SCAN * L_true)
        print("Wide scan done")
        saved["scan_zoom"] = floor_scan(y_scan, axes_scan, L_RATIOS_ZOOM * L_true)
        print("Zoomed scan done")
        saved["ISE"], saved["IAE"] = integrated_floor_errors(SIGMA_VALUES, L_RATIOS_INTEGRATED)
        save_results(saved, FLOORS_FILE)
    floors = saved["floors"]

    # 2. Floor fits: cuts along the axis and the diagonal (t = distance from the peak), slopes, contours, slope budget
    t = np.linspace(-half_width, half_width, 1201)
    truth_cuts = {name: profile(axes, y, direction, t)
                  for name, direction in [("axis", AXIS_DIRECTION), ("diagonal", DIAGONAL_DIRECTION)]}
    floor_cuts = {name: {r: profile(axes, floors[r], direction, t) for r in L_RATIOS_SHOW}
                  for name, direction in [("axis", AXIS_DIRECTION), ("diagonal", DIAGONAL_DIRECTION)]}
    limit = zoom_limit(t, truth_cuts["axis"], floor_cuts["axis"])
    for name in ["axis", "diagonal"]:
        plot_curves(t, truth_cuts[name], floor_cuts[name], FLOOR_SHADES, limit, rf"$t$ (along the {name})", r"$f$",
                    f"floor_fits_{name}.pdf")
    slope_true = central_line(grid_gradient(y, axes)[..., 0])
    floor_slopes = {r: central_line(grid_gradient(floors[r], axes)[..., 0]) for r in L_RATIOS_SHOW}
    plot_curves(axes[0], slope_true, floor_slopes, FLOOR_SHADES, limit, r"$x_1$ ($x_2 = 0$)",
                r"$\partial f / \partial x_1$", "floor_slopes.pdf")
    plot_contours(axes, y, floors, FLOOR_SHADES, limit, "floor_half_max_contours.pdf")
    plot_slope_budget(axes, floors, L_true, limit, "floor_slope_budget.pdf")

    # 3. MSE and widths against L, a wide scan and a zoom around L_true
    for L_ratios, scan, suffix in [(L_RATIOS_SCAN, saved["scan"], ""), (L_RATIOS_ZOOM, saved["scan_zoom"], "_zoom")]:
        plot_floor_scan(L_ratios, {"MSE": scan["MSE"]}, "MSE", f"floor_MSE_vs_L{suffix}.pdf")
        relative_widths = {name.capitalize(): scan[name] / saved["widths_true"][name]
                           for name in ["equivalent", "axis", "diagonal"]}
        plot_floor_scan(L_ratios, relative_widths, r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$",
                        f"floor_FWHM_vs_L{suffix}.pdf", log_y=(suffix == ""))

    # 4. Integrated errors for several sigma (IAE should not depend on sigma, ISE scales as 1 / sigma^2)
    plot_integrated_errors(SIGMA_VALUES, L_RATIOS_INTEGRATED, saved["ISE"], saved["IAE"], "floor_integrated_errors.pdf")

    # 5. Networks against the floor, trained on random points (saved, so the plots can be remade without training)
    x_train, y_train = sample_training_points(half_width)
    plot_training_points(x_train, axes, y, limit, "training_points.pdf")
    if LOAD_SAVED and os.path.exists(NETWORKS_FILE):
        networks = load_results(NETWORKS_FILE)
    else:
        networks = train_networks(X, y, x_train, y_train, L_true)
        save_results(networks, NETWORKS_FILE)
    print_network_table(axes, y, floors, networks, L_true)

    for name, direction in [("axis", AXIS_DIRECTION), ("diagonal", DIAGONAL_DIRECTION)]:
        cuts = {r: np.array([profile(axes, fit, direction, t) for fit in networks["fits"][r]]) for r in L_RATIOS_SHOW}
        mean_cuts = {r: cuts[r].mean(axis=0) for r in L_RATIOS_SHOW}
        bands = {r: (cuts[r].min(axis=0), cuts[r].max(axis=0)) for r in L_RATIOS_SHOW}
        unconstrained = np.mean([profile(axes, fit, direction, t) for fit in networks["fits_unconstrained"]], axis=0)
        plot_curves(t, truth_cuts[name], mean_cuts, NETWORK_SHADES, limit, rf"$t$ (along the {name})", r"$f$",
                    f"network_fits_{name}.pdf", unconstrained=unconstrained, bands=bands)
    network_slopes = {r: central_line(networks["gradients"][r][..., 0].mean(axis=0)) for r in L_RATIOS_SHOW}
    unconstrained_slope = central_line(networks["gradients_unconstrained"][..., 0].mean(axis=0))
    plot_curves(axes[0], slope_true, network_slopes, NETWORK_SHADES, limit, r"$x_1$ ($x_2 = 0$)",
                r"$\partial f / \partial x_1$", "network_slopes.pdf", unconstrained=unconstrained_slope)
    mean_fits = {r: networks["fits"][r].mean(axis=0) for r in L_RATIOS_SHOW}
    plot_contours(axes, y, mean_fits, NETWORK_SHADES, limit, "network_half_max_contours.pdf", floors=floors)
    return 0


# The floors and the networks are computed in parallel processes, which on macOS start a fresh Python that imports
# this file. The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
