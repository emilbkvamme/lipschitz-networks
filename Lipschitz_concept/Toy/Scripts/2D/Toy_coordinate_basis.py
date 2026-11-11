# GOAL: Investigating the effect of different coordinate bases on the performance of the Lipschitz networks

# HOW: Generating a 2D Gaussian, and then changing the coordinate basis of the input space: Cartesian to polar/spherical/logscaled (are there more?)

#STEPS:
    # 1. Generate a 2D Gaussian in Cartesian coordinates and then transform it to the new coordinates
    # 2. Train the Lipschitz network on the transformed data and measure the performance metrics (relative error, MSE, FWHM)
    # Iterate through all the coordinate bases and compare performances

import os
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_nd_utils import *
from Toy_utils import REPORT_STYLE, plot_dir, results_dir, save_results, load_results

QUICK = False        # True: short training, only to check that everything runs
INPUT_NORM = "l2"   # norm of the Lipschitz bound in the input space (the same for every basis)
PLOT_DIR = plot_dir("2D", "Coordinate_basis")
RESULTS_FILE = os.path.join(results_dir("2D", "Coordinate_basis"), "results_quick.pkl" if QUICK else "results.pkl")
LOAD_SAVED = True                  # True: make the plots from RESULTS_FILE if it exists, instead of training again


#### COORDINATE TRANSFORMATIONS ####
# All functions take points X of shape (..., n) and return the new coordinates with the same shape.

def cartesian_to_hyperspherical(X):
    """
    Cartesian (x_1, ..., x_n) to hyperspherical (r, phi_1, ..., phi_{n-1}).
    n = 2 gives polar coordinates (r, phi), n = 3 spherical coordinates (r, phi_1, phi_2).
        r       = |x|
        phi_k   = angle between x_k and the remaining coordinates (x_{k+1}, ..., x_n), in [0, pi]   for k = 1, ..., n-2
        phi_n-1 = arctan2(x_n, x_{n-1}), in (-pi, pi]   (the last angle goes all the way round)
    """
    n = X.shape[-1]
    r = np.sqrt(np.sum(X ** 2, axis=-1))
    angles = []
    for k in range(n - 2):
        rest = np.sqrt(np.sum(X[..., k + 1:] ** 2, axis=-1))   # length of the remaining coordinates
        angles.append(np.arctan2(rest, X[..., k]))              # same as arccos(x_k / |x_k..x_n|), but no 0/0 at the axes
    angles.append(np.arctan2(X[..., n - 1], X[..., n - 2]))
    return np.stack([r] + angles, axis=-1)


def hyperspherical_to_cartesian(X):
    """
    Hyperspherical (r, phi_1, ..., phi_{n-1}) back to Cartesian (x_1, ..., x_n):
        x_1     = r cos(phi_1)
        x_2     = r sin(phi_1) cos(phi_2)
        ...
        x_{n-1} = r sin(phi_1) ... sin(phi_{n-2}) cos(phi_{n-1})
        x_n     = r sin(phi_1) ... sin(phi_{n-2}) sin(phi_{n-1})
    """
    n = X.shape[-1]
    r, angles = X[..., 0], X[..., 1:]
    coordinates = []
    sin_product = r                                    # r sin(phi_1) ... sin(phi_{k-1}), built up one angle at a time
    for k in range(n - 1):
        coordinates.append(sin_product * np.cos(angles[..., k]))
        sin_product = sin_product * np.sin(angles[..., k])
    coordinates.append(sin_product)
    return np.stack(coordinates, axis=-1)


def cartesian_to_log(X, scale=1.0):
    """
    Signed log scaling of every coordinate: u = sign(x) log(1 + |x| / scale).
    Works for negative x (a plain log does not), is close to x / scale near 0 and grows like log|x| far out.
    """
    return np.sign(X) * np.log1p(np.abs(X) / scale)


def log_to_cartesian(U, scale=1.0):
    """Inverse of cartesian_to_log: x = sign(u) scale (exp(|u|) - 1)."""
    return np.sign(U) * scale * np.expm1(np.abs(U))


def L_scan(u, truth, axes, L_values, seeds, steps):
    """
    Trains one Lipschitz network per L value and seed on the inputs u, shape (number of points, n), with the
    targets truth (the Gaussian on the Cartesian grid, flattened in the same order as u).
    The predictions are compared with the truth on the Cartesian grid, so every basis is measured on the same points.
    Returns MSE, AE (mean absolute error), FWHM and SLOPE (the largest Cartesian slope max |grad_x prediction|),
    each of shape (number of L values, number of seeds), and the predictions, shape (L values, seeds, *grid shape).
    """
    jobs, labels = [], []
    for i, L in enumerate(L_values):
        for j, seed in enumerate(seeds):
            jobs.append(dict(kind="lipschitz", seed=seed, steps=steps, lr=3e-3, x=u, y=truth.ravel(),
                             model_kwargs=dict(input_dim=u.shape[1], hidden_dim=64, output_dim=1, n_groups=8,
                                               depth=5, lipschitz_bound=L, input_norm=INPUT_NORM)))
            labels.append((i, j))

    results = train_many(jobs)             # all networks in parallel, same order as jobs

    MSE = np.zeros((len(L_values), len(seeds)))
    AE = np.zeros((len(L_values), len(seeds)))
    FWHM = np.zeros((len(L_values), len(seeds)))
    SLOPE = np.zeros((len(L_values), len(seeds)))
    PREDICTIONS = np.zeros((len(L_values), len(seeds)) + truth.shape)
    for (i, j), result in zip(labels, results):
        prediction = result["prediction"].reshape(truth.shape)   # back on the Cartesian grid
        MSE[i, j] = np.mean((prediction - truth) ** 2)
        AE[i, j] = np.mean(np.abs(prediction - truth))
        FWHM[i, j] = fwhm_equivalent(prediction, axes)
        # Slope in physical (Cartesian) space, from central differences on the grid. The l2 length of the
        # gradient matches INPUT_NORM = "l2". The network's own gradient would be with respect to u instead.
        SLOPE[i, j] = np.max(np.linalg.norm(grid_gradient(prediction, axes), axis=-1))
        PREDICTIONS[i, j] = prediction
    return {"MSE": MSE, "AE": AE, "FWHM": FWHM, "SLOPE": SLOPE, "predictions": PREDICTIONS}


def coordinate_basis_scan():
    """
    One common 2D Gaussian on a Cartesian grid. For every basis the grid points are transformed to the new
    coordinates, and an L scan is trained on (new coordinates, Gaussian values).
    """
    BASES = ["cartesian", "polar", "log"]
    L_RATIOS = [0.1, 0.5, 1.0] if QUICK else [0.1, 0.3, 0.6, 1.0, 2.0]
    SEEDS = [0, 1, 2]
    STEPS = 100 if QUICK else 6000
    POINTS_PER_SIGMA = 4 if QUICK else 8

    # The common Gaussian: anisotropic, so that the angle matters in polar coordinates
    N = 2
    SIGMA_LONG, SIGMA_SHORT = 1.0, 0.5
    MU = np.zeros(N)
    COVARIANCE = np.diag([SIGMA_LONG ** 2, SIGMA_SHORT ** 2])
    L_TRUE = gaussian_lipschitz_constant_nd(cov=COVARIANCE, norm=INPUT_NORM)   # in Cartesian coordinates
    L_values = [ratio * L_TRUE for ratio in L_RATIOS]   # the same L values for every basis

    # The grid must contain the Gaussian (5 sigma) and the floor for the smallest L
    half_width = max(5 * SIGMA_LONG, 1.2 * pyramid_radius(min(L_values), N))
    axes, X = make_grid(half_width, 2 * int(half_width / SIGMA_SHORT * POINTS_PER_SIGMA) + 1, N)   # odd: peak on the grid
    truth = gaussian_nd(X, MU, COVARIANCE)
    FWHM_TRUE = fwhm_equivalent(truth, axes)
    x = X.reshape(-1, N)                  # the grid points as (number of points, N)
    print(f"L_true (cartesian, {INPUT_NORM}) = {L_TRUE:.4f}, FWHM_true = {FWHM_TRUE:.4f}, grid {len(axes[0])}^{N} points")

    scan_results = {"L_ratios": L_RATIOS, "L_true": L_TRUE, "FWHM_true": FWHM_TRUE, "axes": axes, "truth": truth}
    for basis in BASES:
        # The same points in the new coordinates
        if basis == "cartesian":
            u = x
        elif basis == "polar":
            u = cartesian_to_hyperspherical(x)
        elif basis == "log":
            u = cartesian_to_log(x, scale=SIGMA_SHORT)

        scan_results[basis] = L_scan(u, truth, axes, L_values, SEEDS, STEPS)
        for i, ratio in enumerate(L_RATIOS):
            print(f"{basis:9s}, L / L_true = {ratio:.2f}: MSE {scan_results[basis]['MSE'][i].mean():.2e}, "
                  f"AE {scan_results[basis]['AE'][i].mean():.2e}, "
                  f"FWHM / FWHM_true = {scan_results[basis]['FWHM'][i].mean() / FWHM_TRUE:.3f}, "
                  f"slope / L_true = {scan_results[basis]['SLOPE'][i].mean() / L_TRUE:.3f}")

    return scan_results

#### PLOTS ####
def plot_basis_comparison(scan_results, measure, y_label, file_name, divide_by=1.0, log_y=True):
    """One line per basis: the mean over the seeds against L / L_true, with the spread over the seeds as error bars."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        for basis, label in [("cartesian", "Cartesian"), ("polar", "Polar"), ("log", "Log")]:
            values = scan_results[basis][measure] / divide_by      # shape (number of L values, number of seeds)
            ax.errorbar(scan_results["L_ratios"], values.mean(axis=1), yerr=values.std(axis=1), fmt="o-",
                        lw=1.2, markersize=4, capsize=3, label=label)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        ax.legend(frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_error_maps(scan_results, L_index, file_prefix):
    """
    One map per basis of the error (prediction - truth) on the Cartesian grid, at the L value L_ratios[L_index],
    averaged over the seeds. All maps share the same colour limits, so they can be compared directly.
    Saved as {file_prefix}_{basis}.pdf.
    """
    bases = [("cartesian", "Cartesian"), ("polar", "Polar"), ("log", "Log")]
    axes, truth = scan_results["axes"], scan_results["truth"]

    # The error of every basis, and the common colour limit: the largest error in any of them
    errors = {}
    for basis, label in bases:
        errors[basis] = scan_results[basis]["predictions"][L_index].mean(axis=0) - truth
    limit = max(np.max(np.abs(error)) for error in errors.values())

    with plt.rc_context(REPORT_STYLE):
        for basis, label in bases:
            fig, ax = plt.subplots(figsize=(3.1, 2.8))
            mesh = ax.pcolormesh(axes[0], axes[1], errors[basis].T,      # .T: axis 0 of the grid is x, pcolormesh wants (y, x)
                                 cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))       # whole-number ticks
            ax.yaxis.set_major_locator(plt.MultipleLocator(2))
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            colorbar = fig.colorbar(mesh, ax=ax)
            colorbar.set_label(r"$f_\mathrm{pred} - f_\mathrm{true}$")
            colorbar.formatter.set_powerlimits((-2, 2))              # small errors as e.g. 2 x 10^-3
            colorbar.update_ticks()
            fig.savefig(f"{PLOT_DIR}/{file_prefix}_{basis}.pdf")
            plt.close(fig)
#######

# NEED TO ADD: fixing the error plots, they are difficult to read, need the dotted contour of the truth
# saving the predictions so that I can access them for plotting the error maps or for testing effect of other basis choices. 
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        scan_results = load_results(RESULTS_FILE)
    else:
        scan_results = coordinate_basis_scan()
        save_results(scan_results, RESULTS_FILE)

    plot_basis_comparison(scan_results, "MSE", "MSE", "basis_mse_vs_L.pdf")
    plot_basis_comparison(scan_results, "AE", "Mean absolute error", "basis_ae_vs_L.pdf")
    plot_basis_comparison(scan_results, "FWHM", r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$", "basis_fwhm_vs_L.pdf",
                          divide_by=scan_results["FWHM_true"], log_y=False)
    plot_basis_comparison(scan_results, "SLOPE", r"$\max |\nabla_x f| / L_\mathrm{true}$", "basis_slope_vs_L.pdf",
                          divide_by=scan_results["L_true"], log_y=False)

    L_index = scan_results["L_ratios"].index(1.0)     # the error maps at L = L_true
    plot_error_maps(scan_results, L_index, "basis_error_map_L1p0")
    return 0

if __name__ == "__main__":
    main()
