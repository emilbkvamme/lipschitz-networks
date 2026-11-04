#GOAL: Investigate the effect of missing data for the Lipschitz models. Can it retrieve cirrcular symmetry or other symmetries in the data without a complete picture.

# HOW: Generate a toy dataset with a circular symmetry. Ommit a piece of the data and see if the Lipschitz model can still retrieve the symmetry
# Scan over angular size for missing data?

# PLOTS: plot of the initial data FWHM, quarter half, and peak. This will be in dotted lines. The Lipschitz models will be plotted in solid coloured lines around. One model at least needs to have a 90 degree missing piece, one 45 degree, one missing two  oppositely palced 90 degrees. Isotropic gaussian





import os
import pickle
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

from Toy_nd_utils import *
from Toy_utils import REPORT_STYLE
from Toy_coordinate_basis import cartesian_to_hyperspherical

QUICK = False                       # True: short training, only to check that everything runs
PLOT_DIR = "Plots/Toy_data_missing"
RESULTS_FILE = "Results/Toy_data_missing/results.pkl"   # the trained predictions and measures, saved after the scan
LOAD_SAVED = True                  # True: make the plots from RESULTS_FILE if it exists, instead of training again
BASES = ["cartesian", "polar"]     # the input coordinates of the networks (polar = 2D spherical: (r, phi))
LR_LIPSCHITZ = 1e-2                # the Lipschitz networks learn slowly with 3e-3 (their step size scales with L)
LR_UNCONSTRAINED = 3e-3
INPUT_NORM = "l2"                  # rotation invariant: the bound itself has no preferred direction
CONTOUR_LEVELS = [0.25, 0.5, 0.9]  # contours at these fractions of the peak: quarter, half maximum and near the peak


def wedge_mask(X, gaps):
    """
    True for the points X, shape (..., 2), that lie inside one of the missing wedges.
    gaps: list of (centre, width) in degrees, e.g. [(0, 90)] removes the wedge from -45 to 45 degrees,
    and [(0, 90), (180, 90)] two opposite wedges. The tip of every wedge is at the origin.
    """
    angle = np.degrees(np.arctan2(X[..., 1], X[..., 0]))          # in (-180, 180]
    missing = np.zeros(X.shape[:-1], dtype=bool)
    for centre, width in gaps:
        difference = (angle - centre + 180) % 360 - 180             # angle to the gap centre, in [-180, 180)
        missing = missing | (np.abs(difference) < width / 2)
    return missing


def ring_mask(X, r_inner, r_outer):
    """
    True for the points X, shape (..., 2), with r_inner <= |x| < r_outer.
    r_inner = 0 removes a disk around the peak (peak drop), r_inner > 0 a ring on the flank.
    """
    r = np.sqrt(np.sum(X ** 2, axis=-1))
    return (r >= r_inner) & (r < r_outer)


def L_scan(X, truth, missing, axes, L_ratios, L_true, seeds, steps, basis="cartesian"):
    """
    Trains Lipschitz networks (one per L = ratio * L_true and seed) and unconstrained networks (one per seed) on the grid points
    that are not missing, and evaluates them on the full grid.
    X: grid points, shape (*grid shape, 2), truth: the Gaussian on the grid, missing: the mask from wedge_mask or ring_mask.
    basis: "cartesian" trains on (x_1, x_2), "polar" on (r, phi). The results are always on the Cartesian grid.
    Returns, for every ratio in L_ratios and for "unconstrained", a dictionary with
        MSE_gap, AE_gap: the errors inside the gap (where there was no data), one value per seed,
        MSE_rest:        the error on the points that were trained on, one value per seed,
        FWHM:            fwhm_equivalent of the prediction, one value per seed,
        predictions:     the predictions on the grid, shape (number of seeds, *grid shape).
    """
    x = X.reshape(-1, 2)
    if basis == "polar":
        x = cartesian_to_hyperspherical(x)  # the same points in (r, phi); the grid order is unchanged
    keep = ~missing.ravel()                 # the training points
    keys = list(L_ratios) + ["unconstrained"]

    jobs, labels = [], []
    for key in keys:
        if key == "unconstrained":
            kind = "unconstrained"
            lr = LR_UNCONSTRAINED
            model_kwargs = dict(input_dim=2, hidden_dim=64, output_dim=1, depth=5)
        else:
            kind = "lipschitz"
            lr = LR_LIPSCHITZ
            model_kwargs = dict(input_dim=2, hidden_dim=64, output_dim=1, n_groups=8, depth=5,
                                lipschitz_bound=key * L_true, input_norm=INPUT_NORM)
        for seed in seeds:
            jobs.append(dict(kind=kind, seed=seed, steps=steps, lr=lr,
                             x=x[keep], y=truth.ravel()[keep],   # train only where there is data
                             x_eval=x,                           # evaluate on the full grid, gap included
                             model_kwargs=model_kwargs))
            labels.append(key)

    results = train_many(jobs)              # all networks in parallel, same order as jobs

    scan = {key: {"MSE_gap": [], "AE_gap": [], "MSE_rest": [], "FWHM": [], "predictions": []} for key in keys}
    for key, result in zip(labels, results):
        prediction = result["prediction"].reshape(truth.shape)
        error = prediction - truth
        if missing.any():
            scan[key]["MSE_gap"].append(np.mean(error[missing] ** 2))
            scan[key]["AE_gap"].append(np.mean(np.abs(error[missing])))
        else:                                # full-data reference: there is no gap
            scan[key]["MSE_gap"].append(np.nan)
            scan[key]["AE_gap"].append(np.nan)
        scan[key]["MSE_rest"].append(np.mean(error[~missing] ** 2))
        scan[key]["FWHM"].append(fwhm_equivalent(prediction, axes))
        scan[key]["predictions"].append(prediction)

    # Lists -> arrays
    for key in keys:
        for name in scan[key]:
            scan[key][name] = np.array(scan[key][name])
    return scan


def make_missing_cases(X, sigma):
    """The missing points of every case: angular gaps (wedges) and radial gaps (peak drop and ring)."""
    return {"wedge90":    wedge_mask(X, [(0, 90)]),
            "wedge45":    wedge_mask(X, [(0, 45)]),
            "opposite90": wedge_mask(X, [(0, 90), (180, 90)]),
            "peak_drop":  ring_mask(X, 0, 1.0 * sigma),             # the top of the peak, inside the half maximum (1.18 sigma)
            "ring":       ring_mask(X, 1.0 * sigma, 1.5 * sigma)}   # the flank: steepest slope (r = sigma) and the half maximum


def missing_data_scan():
    """
    For every basis and missing-data case: an L scan of Lipschitz networks (l2 bound) and the unconstrained networks,
    trained on the remaining points of an isotropic 2D Gaussian and evaluated on the full grid.
    Also a full-data reference ("none") per basis. Returns everything in one dictionary:
        results[basis][case] = the output of L_scan, results["missing"][case] = the mask of the case.
    """
    L_RATIOS = [0.5, 1.0] if QUICK else [0.3, 0.6, 1.0, 2.0]
    SEEDS = [0, 1, 2]
    STEPS = 300 if QUICK else 20000
    POINTS_PER_SIGMA = 4 if QUICK else 8

    # The isotropic Gaussian
    N = 2
    SIGMA = 1.0
    COVARIANCE = isotropic_cov(sigma=SIGMA, n=N)
    L_TRUE = gaussian_lipschitz_constant_nd(cov=COVARIANCE, norm=INPUT_NORM)

    # The training grid must contain the Gaussian (5 sigma) and the floor for the smallest L
    half_width = max(5 * SIGMA, 1.2 * pyramid_radius(min(L_RATIOS) * L_TRUE, N))
    axes, X = make_grid(half_width, 2 * int(half_width / SIGMA * POINTS_PER_SIGMA) + 1, N)   # odd: peak on the grid
    truth = gaussian_nd(X, np.zeros(N), COVARIANCE)
    print(f"L_true ({INPUT_NORM}) = {L_TRUE:.4f}, grid {len(axes[0])}^{N} points, {STEPS} steps")

    # The cases, with the full-data reference first
    missing_cases = {"none": np.zeros(truth.shape, dtype=bool)}
    missing_cases.update(make_missing_cases(X, SIGMA))

    results = {"L_ratios": L_RATIOS, "L_true": L_TRUE, "axes": axes, "truth": truth, "steps": STEPS,
               "missing": missing_cases}
    for basis in BASES:
        results[basis] = {}
        for name, missing in missing_cases.items():
            scan = L_scan(X, truth, missing, axes, L_RATIOS, L_TRUE, SEEDS, STEPS, basis=basis)
            results[basis][name] = scan
            for ratio in L_RATIOS:
                print(f"{basis:9s} {name:10s}, L / L_true = {ratio:.2f}: MSE in gap {scan[ratio]['MSE_gap'].mean():.2e} "
                      f"(unconstrained {scan['unconstrained']['MSE_gap'].mean():.2e}), "
                      f"MSE on data {scan[ratio]['MSE_rest'].mean():.2e}")
    return results


def save_results(results, file_name):
    """Saves the results dictionary (numpy arrays and numbers) with pickle."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as file:
        pickle.dump(results, file)
    print(f"Saved the results to {file_name}")


def load_results(file_name):
    """Loads the results dictionary saved by save_results."""
    with open(file_name, "rb") as file:
        return pickle.load(file)


def make_plots(results):
    """
    All plots from the saved results, so they can be changed without training again. Per basis, case and L:
    a contour plot and a slice plot, both with the unconstrained network overlaid. Per case: the gap error vs L
    for both bases.
    """
    axes, truth, L_ratios, L_true = results["axes"], results["truth"], results["L_ratios"], results["L_true"]
    for name, missing in results["missing"].items():
        if name == "none":
            continue                       # the reference has no gap; it only appears in the slice plots
        for basis in BASES:
            scan = results[basis][name]
            unconstrained_mean = scan["unconstrained"]["predictions"].mean(axis=0)     # mean over the seeds
            for ratio in L_ratios:
                ratio_text = f"{ratio:g}".replace(".", "p")                             # 0.5 -> 0p5 for the file name
                contour_plot(axes, truth, missing, scan[ratio]["predictions"].mean(axis=0),
                             f"contour_{basis}_{name}_L{ratio_text}.pdf", unconstrained=unconstrained_mean)
                slice_plot(axes, truth, missing, ratio * L_true, scan[ratio]["predictions"].mean(axis=0),
                           results[basis]["none"][ratio]["predictions"].mean(axis=0), unconstrained_mean,
                           f"slice_{basis}_{name}_L{ratio_text}.pdf", show_band=(ratio >= 1))

        plot_gap_error_vs_L({basis: results[basis][name] for basis in BASES}, L_ratios, "MSE_gap", "MSE in gap",
                            f"gap_mse_vs_L_{name}.pdf")

    # Overview of all cases and bases, one figure per L
    for ratio in L_ratios:
        ratio_text = f"{ratio:g}".replace(".", "p")
        plot_contour_overview(results, ratio, f"contour_overview_L{ratio_text}.pdf")
        plot_slice_overview(results, ratio, ["peak_drop", "ring"], f"slice_overview_L{ratio_text}.pdf")


#### PLOTS ####
def plot_truth_and_missing_data(axes, truth, missing, file_name):
    """
    The true Gaussian as a colour map, with the missing points greyed out, and the true contours at
    CONTOUR_LEVELS times the peak as dotted lines (they continue through the gap).
    """
    shown = np.ma.masked_where(missing, truth)        # masked points are not coloured, the grey background shows
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        ax.set_facecolor("0.85")                       # colour of the missing region
        mesh = ax.pcolormesh(axes[0], axes[1], shown.T, cmap="viridis", shading="auto", rasterized=True,  # .T: pcolormesh wants (y, x)
                             vmin=0, vmax=truth.max())        # the same colour scale in every case, also when the peak is missing
        ax.contour(axes[0], axes[1], truth.T, levels=[level * truth.max() for level in CONTOUR_LEVELS],
                   colors="white", linestyles="dotted", linewidths=1.2)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        colorbar = fig.colorbar(mesh, ax=ax)
        colorbar.set_label(r"$f(x)$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def contour_plot(axes, truth, missing, prediction, file_name, unconstrained=None, half_width=3.0):
    """
    Contours at CONTOUR_LEVELS times the true peak: the truth dotted (black), the Lipschitz prediction solid (blue)
    and, if given, the unconstrained prediction solid (orange). The gap (no training data) is shaded grey.
    prediction and unconstrained are grids like truth, e.g. the mean over the seeds.
    Only the region |x_i| < half_width is shown (the training grid can be wider).
    """
    levels = [level * truth.max() for level in CONTOUR_LEVELS]   # the same absolute levels for every curve

    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        ax.contourf(axes[0], axes[1], missing.T.astype(float), levels=[0.5, 1.5], colors=["0.88"])   # the gap
        ax.contour(axes[0], axes[1], truth.T, levels=levels, colors="black", linestyles="dotted", linewidths=1.2)
        ax.contour(axes[0], axes[1], prediction.T, levels=levels, colors="tab:blue", linewidths=1.2)
        if unconstrained is not None:
            ax.contour(axes[0], axes[1], unconstrained.T, levels=levels, colors="tab:orange", linewidths=1.2)

        # Contour lines have no legend labels, so the legend is built from plain lines in the same style
        handles = [plt.Line2D([], [], color="black", ls="dotted", lw=1.2, label="Truth"),
                   plt.Line2D([], [], color="tab:blue", lw=1.2, label="Lipschitz")]
        if unconstrained is not None:
            handles.append(plt.Line2D([], [], color="tab:orange", lw=1.2, label="Unconstrained"))
        ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=len(handles), handlelength=1.2, columnspacing=0.8, handletextpad=0.4)

        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_contour_overview(results, ratio, file_name, half_width=3.0):
    """
    All missing-data cases side by side (columns) for every basis (rows), at L = ratio * L_true.
    In every panel, as in contour_plot: the truth dotted (black), the Lipschitz network solid (blue), the
    unconstrained network solid (orange), all as the mean over the seeds, and the gap shaded grey.
    """
    axes, truth = results["axes"], results["truth"]
    levels = [level * truth.max() for level in CONTOUR_LEVELS]
    case_titles = {"wedge90": "Wedge 90°", "wedge45": "Wedge 45°", "opposite90": "Opposite 90°",
                   "peak_drop": "Peak drop", "ring": "Ring"}
    basis_titles = {"cartesian": "Cartesian", "polar": "Polar"}
    cases = [name for name in results["missing"] if name != "none"]

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(BASES), len(cases), figsize=(2.1 * len(cases), 2.2 * len(BASES) + 0.7),
                                sharex=True, sharey=True, squeeze=False)
        for row, basis in enumerate(BASES):
            for column, name in enumerate(cases):
                ax = axs[row, column]
                scan = results[basis][name]
                missing = results["missing"][name]
                ax.contourf(axes[0], axes[1], missing.T.astype(float), levels=[0.5, 1.5], colors=["0.88"])   # the gap
                ax.contour(axes[0], axes[1], truth.T, levels=levels, colors="black", linestyles="dotted", linewidths=1.0)
                ax.contour(axes[0], axes[1], scan["unconstrained"]["predictions"].mean(axis=0).T, levels=levels,
                           colors="tab:orange", linewidths=1.0)
                ax.contour(axes[0], axes[1], scan[ratio]["predictions"].mean(axis=0).T, levels=levels,
                           colors="tab:blue", linewidths=1.0)
                ax.set_xlim(-half_width, half_width)
                ax.set_ylim(-half_width, half_width)
                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                ax.yaxis.set_major_locator(plt.MultipleLocator(2))
                if row == 0:
                    ax.set_title(case_titles.get(name, name), fontsize=12)
                if column == 0:
                    ax.set_ylabel(f"{basis_titles[basis]}\n$x_2$")
                if row == len(BASES) - 1:
                    ax.set_xlabel(r"$x_1$")

        # One legend for all panels, below the figure
        handles = [plt.Line2D([], [], color="black", ls="dotted", lw=1.2, label="Truth"),
                   plt.Line2D([], [], color="tab:blue", lw=1.2, label=f"Lipschitz, $L = {ratio:g}\\,L_\\mathrm{{true}}$"),
                   plt.Line2D([], [], color="tab:orange", lw=1.2, label="Unconstrained")]
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.08, 1, 1))     # leave room for the legend at the bottom
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def draw_slice(ax, axes, truth, missing, L, prediction, prediction_full, unconstrained, show_band=True, half_width=3.0):
    """
    Draws one slice along x_1 through the peak (x_2 = 0) on ax: the truth (dotted), the Lipschitz network trained
    with the gap (solid) and on the full data (dashed), and the unconstrained network trained with the gap.
    The gap is shaded. The grey band is what the bound allows given the remaining data: every L-Lipschitz function
    (l2) through the training points lies between max_i(y_i - L |x - x_i|) and min_i(y_i + L |x - x_i|).
    The band is only meaningful for L >= L_true (below, the data itself is steeper than L).
    """
    middle = len(axes[1]) // 2                       # the index of x_2 = 0 (the grid has an odd number of points)
    x1 = axes[0]

    # The gap along the slice, one shaded span per missing grid cell
    spacing = x1[1] - x1[0]
    for value in x1[missing[:, middle]]:
        ax.axvspan(value - spacing / 2, value + spacing / 2, color="0.92", lw=0)

    if show_band:
        # Distances from every slice point to every training point, shape (slice points, training points)
        X_train = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)[~missing]
        y_train = truth[~missing]
        slice_points = np.stack([x1, np.zeros_like(x1)], axis=-1)
        distances = np.linalg.norm(slice_points[:, None, :] - X_train[None, :, :], axis=-1)
        lower = np.max(y_train - L * distances, axis=1)
        upper = np.min(y_train + L * distances, axis=1)
        ax.fill_between(x1, lower, upper, color="0.6", alpha=0.35, lw=0, label="Allowed by $L$")

    ax.plot(x1, truth[:, middle], color="black", ls="dotted", lw=1.2, label="Truth")
    ax.plot(x1, prediction_full[:, middle], color="tab:blue", ls="--", lw=1.2, label="Lipschitz, full data")
    ax.plot(x1, prediction[:, middle], color="tab:blue", lw=1.2, label="Lipschitz, gap")
    ax.plot(x1, unconstrained[:, middle], color="tab:orange", lw=1.2, label="Unconstrained, gap")

    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(0, 1.15 * truth.max())
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))


def slice_plot(axes, truth, missing, L, prediction, prediction_full, unconstrained, file_name,
               show_band=True, half_width=3.0):
    """One slice (see draw_slice) as its own figure."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        draw_slice(ax, axes, truth, missing, L, prediction, prediction_full, unconstrained, show_band, half_width)
        ax.set_xlabel(r"$x_1$  ($x_2 = 0$)")
        ax.set_ylabel(r"$f(x)$")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2,
                  handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_slice_overview(results, ratio, cases, file_name, half_width=3.0):
    """
    Slices along x_1 (see draw_slice) for the given cases (columns) and every basis (rows), at L = ratio * L_true.
    All curves are the mean over the seeds. Meant for the radial gaps (peak drop and ring), where the side view
    shows the height of the fit better than the contours.
    """
    axes, truth, L_true = results["axes"], results["truth"], results["L_true"]
    case_titles = {"wedge90": "Wedge 90°", "wedge45": "Wedge 45°", "opposite90": "Opposite 90°",
                   "peak_drop": "Peak drop", "ring": "Ring"}
    basis_titles = {"cartesian": "Cartesian", "polar": "Polar"}

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(BASES), len(cases), figsize=(3.1 * len(cases), 2.6 * len(BASES) + 0.9),
                                sharex=True, sharey=True, squeeze=False)
        for row, basis in enumerate(BASES):
            for column, name in enumerate(cases):
                ax = axs[row, column]
                scan = results[basis][name]
                draw_slice(ax, axes, truth, results["missing"][name], ratio * L_true,
                           scan[ratio]["predictions"].mean(axis=0),
                           results[basis]["none"][ratio]["predictions"].mean(axis=0),
                           scan["unconstrained"]["predictions"].mean(axis=0),
                           show_band=(ratio >= 1), half_width=half_width)
                if row == 0:
                    ax.set_title(case_titles.get(name, name), fontsize=12)
                if column == 0:
                    ax.set_ylabel(f"{basis_titles[basis]}\n$f(x)$")
                if row == len(BASES) - 1:
                    ax.set_xlabel(r"$x_1$  ($x_2 = 0$)")

        # One legend for all panels, below the figure
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, handlelength=1.4, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0.13, 1, 1))     # leave room for the legend at the bottom
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def plot_gap_error_vs_L(scans, L_ratios, measure, y_label, file_name):
    """
    One measure (e.g. MSE_gap) against L / L_true. scans = {basis: output of L_scan}.
    Colour: the network (Lipschitz blue, unconstrained orange). Line style: the basis (Cartesian solid, polar dashed).
    Lipschitz: mean over the seeds with the spread as error bars. Unconstrained: does not depend on L, a horizontal line.
    """
    styles = {"cartesian": ("-", "o", "Cartesian"), "polar": ("--", "s", "Polar")}
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        for basis, scan in scans.items():
            line_style, marker, basis_label = styles[basis]
            lipschitz = np.array([scan[ratio][measure] for ratio in L_ratios])   # shape (number of L values, number of seeds)
            unconstrained = scan["unconstrained"][measure]                      # shape (number of seeds,)
            ax.errorbar(L_ratios, lipschitz.mean(axis=1), yerr=lipschitz.std(axis=1), fmt=marker, ls=line_style,
                        color="tab:blue", lw=1.2, markersize=4, capsize=3, label=f"Lipschitz, {basis_label.lower()}")
            ax.axhline(unconstrained.mean(), color="tab:orange", ls=line_style, lw=1.2,
                       label=f"Unconstrained, {basis_label.lower()}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$L / L_\mathrm{true}$")
        ax.set_ylabel(y_label)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, handlelength=1.8)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # The isotropic Gaussian, shown only where it is not tiny
    N = 2
    SIGMA = 1.0
    axes, X = make_grid(3 * SIGMA, 121, N)
    truth = gaussian_nd(X, np.zeros(N), isotropic_cov(sigma=SIGMA, n=N))

    for name, missing in make_missing_cases(X, SIGMA).items():
        plot_truth_and_missing_data(axes, truth, missing, f"truth_missing_{name}.pdf")

    # Train only if there are no saved results (or LOAD_SAVED is False), then make every plot from the results
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        print(f"Loading the results from {RESULTS_FILE}")
        results = load_results(RESULTS_FILE)
    else:
        results = missing_data_scan()
        save_results(results, RESULTS_FILE)
    make_plots(results)
    return 0

if __name__ == "__main__":
    main()
