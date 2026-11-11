# Stage 1 of the noise analysis: does the metric of the Lipschitz bound match the signal?
#
# A Lipschitz network assumes |f(x) - f(y)| <= L ||x - y||, measured in a NORM of the input. The first layer of
# LipschitzModel decides which norm (input_norm in models.py):
#   l1   (the default so far): every partial derivative is at most L
#   l2   (Euclidean):          the slope is at most L in every direction, the same for every rotation
#   linf:                      |df/dx_1| + |df/dx_2| <= L
# In 1D the three are the same. From 2D on they are not, and only l2 treats all directions equally.
#
# The test: an elongated 2D Gaussian (long width SIGMA_SHORT * ASPECT, short width SIGMA_SHORT), rotated by an angle.
# For every norm and angle we compute the floor (the best possible fit) and train networks, at the SAME bound L for
# every norm and angle. If the bound is rotation invariant (l2), nothing should change with the angle. For l1 and linf
# the error and the width in the narrow direction should change with the angle: the bound depends on the coordinate
# axes, so the resolution of the network depends on how the feature is oriented relative to them.
#
# Design choices:
#   - The overall width does not matter: rescaling x maps one width onto another (shown in the 1D and 2D notebooks),
#     so only the SHAPE (ASPECT) and the ANGLE are scanned. ASPECT = 1 (round) is the control: rotating a round
#     Gaussian changes nothing, so every norm must give a flat line.
#   - Angles from 0 to 45 degrees are enough: the l1 and linf norms are unchanged by 90 degree rotations and by
#     mirroring in the diagonal, so 45 + a gives the same as 45 - a.
#   - The bound is set relative to the slope of the Gaussian in the l2 norm, L = L_RATIO * L_true_l2, the same number
#     for every norm and angle (the l2 slope does not change with the angle).
#   - No noise yet: this stage is only about the metric. Noise comes in stage 2, with the metric chosen here.
#
# Run from the Lipschitz_concept folder:  python Toy_nd_norms_rotations_complete.py
# The figures and the numbers (results.npz) are saved in Plots/Toy_nd_norms.

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt

# "Homemade" modules
from Toy_utils import REPORT_STYLE
from Toy_nd_utils import gaussian_nd, pyramid_radius, make_grid, grid_integral, profile, train_many
from lipschitz_floor import lipschitz_floor_nd


##### SETTINGS #####
QUICK = False        # True: coarse grid, few angles and short training, only to check that everything runs

NORMS = ["l1", "l2", "linf"]
NORM_LABELS = {"l1": r"$\ell_1$", "l2": r"$\ell_2$", "linf": r"$\ell_\infty$"}
NORM_COLORS = {"l1": "#2a78d6", "l2": "#7b4fb3", "linf": "#eb6834"}   # the true Gaussian is black

# The Gaussians
SIGMA_SHORT = 0.1                                       # width in the narrow direction
ASPECTS = [1.0, 3.0]                                    # long width / short width. 1 = round (the control)
ANGLES_DEGREES = np.linspace(0, 45, 4 if QUICK else 10) # rotation of the long direction away from the x_1 axis
POINTS_PER_SIGMA = 2 if QUICK else 4                    # grid points per SIGMA_SHORT (the narrow width sets the grid)

# The floors: bound L = L_RATIO * (slope of the Gaussian in the l2 norm), the same for every norm and angle
L_RATIOS_FLOOR = [0.25, 0.5]

# The networks: one elongated Gaussian, a few angles, one bound
ASPECT_NETWORKS = 3.0
ANGLES_NETWORKS = [0.0, 22.5, 45.0]
L_RATIO_NETWORKS = 0.5
SEEDS = [0, 1] if QUICK else [0, 1, 2]
HIDDEN_DIM = 64
N_GROUPS = 8
DEPTH = 5
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3

# Figures: fixed size and margins (saved without the tight cropping), so all the single plots are the same size in LaTeX
FIG_SIZE = (3.1, 2.8)
FIG_MARGINS = dict(left=0.24, right=0.96, top=0.96, bottom=0.18)

# Output folder, next to this script, so the script can be run from anywhere
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plots", "Toy_nd_norms")


##### THE GAUSSIAN #####
def rotated_covariance(sigma_long, sigma_short, angle_degrees):
    """Covariance of a 2D Gaussian with widths sigma_long and sigma_short, the long direction at the given angle."""
    angle = np.radians(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag([sigma_long ** 2, sigma_short ** 2]) @ rotation.T


def gaussian_slope(cov, norm):
    """
    Lipschitz constant of the Gaussian N(0, cov) in the given norm of the input: the largest value of the DUAL norm
    of its gradient. grad f = -P x f(x) with P = cov^-1; writing x = t u with u^T P u = 1, the slope is largest at
    t = 1, which leaves L = exp(-1/2) f_max max_u ||P u||_dual (the same derivation as gaussian_lipschitz_constant_nd).
        l1:   dual norm inf,  max_u ||P u||_inf = max_i sqrt(P_ii)
        l2:   dual norm l2,   max_u ||P u||_2   = sqrt(largest eigenvalue of P)
        linf: dual norm l1,   max_u ||P u||_1   = max over sign vectors s of sqrt(s^T P s)
    """
    n = cov.shape[0]
    precision = np.linalg.inv(cov)
    f_max = 1 / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
    if norm == "l1":
        factor = np.sqrt(np.max(np.diag(precision)))
    elif norm == "l2":
        factor = np.sqrt(np.max(np.linalg.eigvalsh(precision)))
    elif norm == "linf":
        factor = 0.0
        for signs in np.stack(np.meshgrid(*[[-1.0, 1.0]] * n, indexing="ij"), axis=-1).reshape(-1, n):
            factor = max(factor, np.sqrt(signs @ precision @ signs))
    else:
        raise ValueError(f"norm ({norm}) must be 'l1', 'l2' or 'linf'")
    return np.exp(-0.5) * f_max * factor


def grid_for(aspect, L_smallest):
    """
    One grid for all angles of a Gaussian with the given aspect. Wide enough for the long direction at any angle
    (5 long widths) and for the floor with the smallest L to come down to zero (the pyramid of the l1 floor is the
    widest of the three norms; 1.3 times its size covers the other two). Odd number of points, so the peak is on the grid.
    """
    half_width = max(5 * SIGMA_SHORT * aspect, 1.3 * pyramid_radius(L_smallest, 2))
    spacing = SIGMA_SHORT / POINTS_PER_SIGMA
    n_points = 2 * int(half_width / spacing) + 1
    return make_grid(half_width, n_points, 2)


##### MEASURES #####
def relative_squared_error(g, f, axes):
    """Integrated squared error of g relative to f, divided by the integral of f^2 (the same for every width)."""
    return grid_integral((g - f) ** 2, axes) / grid_integral(f ** 2, axes)


def width_along(axes, f, direction):
    """
    Full width at half maximum of the grid function f along the line through the origin in the given direction.
    The half-maximum crossings are interpolated between the profile points, so the result is not limited to steps.
    """
    t_max = axes[0][-1] - axes[0][0]
    t = np.linspace(-t_max, t_max, 8001)
    values = profile(axes, f, direction, t)
    half_max = np.nanmax(values) / 2
    above = np.where(values >= half_max)[0]
    first, last = above[0], above[-1]
    t_left = t[first - 1] + (half_max - values[first - 1]) * (t[first] - t[first - 1]) / (values[first] - values[first - 1])
    t_right = t[last] + (half_max - values[last]) * (t[last + 1] - t[last]) / (values[last + 1] - values[last])
    return t_right - t_left


def directions(angle_degrees):
    """Unit vectors along the long and the short direction of the Gaussian rotated by the given angle."""
    angle = np.radians(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])


def largest_slope(gradient, norm):
    """Largest slope of a network in the given norm of the input: the largest dual norm of its gradient over the grid."""
    dual_order = {"l1": np.inf, "l2": 2, "linf": 1}[norm]
    return np.max(np.linalg.norm(gradient, ord=dual_order, axis=-1))


##### THE FLOORS #####
def floor_scan():
    """
    Floors for every aspect, bound, norm and angle. Returns
        floor_results[(aspect, L_ratio, norm)] = dictionary of arrays over the angles: relative error, peak, widths
    and the floors themselves for the contour figure (aspect ASPECT_NETWORKS, bound L_RATIO_NETWORKS).
    """
    floor_results = {}
    floors_for_contours = {}
    for aspect in ASPECTS:
        L_true_l2 = gaussian_slope(rotated_covariance(SIGMA_SHORT * aspect, SIGMA_SHORT, 0.0), "l2")
        axes, X = grid_for(aspect, min(L_RATIOS_FLOOR) * L_true_l2)
        print(f"Aspect {aspect}: L_true (l2) = {L_true_l2:.2f}, grid {len(axes[0])}^2 points on "
              f"[-{axes[0][-1]:.2f}, {axes[0][-1]:.2f}]^2")

        for L_ratio in L_RATIOS_FLOOR:
            L = L_ratio * L_true_l2
            for norm in NORMS:
                results = {"relative_error": [], "peak": [], "width_long": [], "width_short": []}
                start = time.time()
                for angle in ANGLES_DEGREES:
                    cov = rotated_covariance(SIGMA_SHORT * aspect, SIGMA_SHORT, angle)
                    y = gaussian_nd(X, np.zeros(2), cov)
                    floor, mse = lipschitz_floor_nd(y, axes, L, norm=norm)
                    long_direction, short_direction = directions(angle)

                    # Everything relative to the true Gaussian, so the numbers are the same for every width
                    results["relative_error"].append(relative_squared_error(floor, y, axes))
                    results["peak"].append(floor.max() / y.max())
                    results["width_long"].append(width_along(axes, floor, long_direction) / width_along(axes, y, long_direction))
                    results["width_short"].append(width_along(axes, floor, short_direction) / width_along(axes, y, short_direction))

                    if aspect == ASPECT_NETWORKS and L_ratio == L_RATIO_NETWORKS and angle in (ANGLES_DEGREES[0], ANGLES_DEGREES[-1]):
                        floors_for_contours[(norm, angle)] = (axes, floor, y)

                floor_results[(aspect, L_ratio, norm)] = {name: np.array(values) for name, values in results.items()}
                print(f"   L / L_true = {L_ratio}, {norm:4s}: {len(ANGLES_DEGREES)} floors in {time.time() - start:.1f} s, "
                      f"relative error from {min(results['relative_error']):.3e} to {max(results['relative_error']):.3e}")
    return floor_results, floors_for_contours


##### THE NETWORKS #####
def network_scan():
    """
    Lipschitz networks with every input norm, trained on the elongated Gaussian at every angle in ANGLES_NETWORKS,
    with the same bound L = L_RATIO_NETWORKS * L_true_l2. All networks are trained in parallel (train_many).
    Returns network_results[(norm, angle)] = dictionary with the predictions and measures for every seed, and the grid.
    """
    L_true_l2 = gaussian_slope(rotated_covariance(SIGMA_SHORT * ASPECT_NETWORKS, SIGMA_SHORT, 0.0), "l2")
    L = L_RATIO_NETWORKS * L_true_l2
    axes, X = grid_for(ASPECT_NETWORKS, L)
    x = X.reshape(-1, 2)

    # One training job per norm, angle and seed, in this order
    jobs = []
    targets = {}
    for norm in NORMS:
        for angle in ANGLES_NETWORKS:
            y = gaussian_nd(X, np.zeros(2), rotated_covariance(SIGMA_SHORT * ASPECT_NETWORKS, SIGMA_SHORT, angle))
            targets[angle] = y
            for seed in SEEDS:
                jobs.append(dict(kind="lipschitz", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x, y=y.ravel(),
                                 model_kwargs=dict(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                                   depth=DEPTH, lipschitz_bound=L, input_norm=norm)))
    print(f"Training {len(jobs)} networks on a {len(axes[0])}^2 grid, L = {L:.2f}")
    start = time.time()
    results = train_many(jobs)
    print(f"   done in {time.time() - start:.0f} s")

    # Sort the results back per norm and angle, in the same order as the jobs
    network_results = {}
    k = 0
    for norm in NORMS:
        for angle in ANGLES_NETWORKS:
            y = targets[angle]
            long_direction, short_direction = directions(angle)
            entry = {"prediction": [], "relative_error": [], "width_short": [], "slope_used": []}
            for seed in SEEDS:
                prediction = results[k]["prediction"].reshape(y.shape)
                gradient = results[k]["gradient"]
                entry["prediction"].append(prediction)
                entry["relative_error"].append(relative_squared_error(prediction, y, axes))
                entry["width_short"].append(width_along(axes, prediction, short_direction) / width_along(axes, y, short_direction))
                entry["slope_used"].append(largest_slope(gradient, norm) / L)   # 1 = the whole budget is used
                k += 1
            network_results[(norm, angle)] = {name: np.array(values) for name, values in entry.items()}
            print(f"   {norm:4s}, angle {angle:4.1f}: relative error {np.mean(entry['relative_error']):.3e}, "
                  f"slope used {np.mean(entry['slope_used']):.3f} L")
    return network_results, axes, targets


##### PLOTS #####
def new_figure():
    """One single plot in the report format, always the same size."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(**FIG_MARGINS)
    return fig, ax


def plot_slope_needed():
    """How large a bound each norm needs to fit the Gaussian exactly (L_true), relative to the l2 norm, against the angle."""
    angles = np.linspace(0, 45, 91)
    fig, ax = new_figure()
    for norm in NORMS:
        ratio = []
        for angle in angles:
            cov = rotated_covariance(SIGMA_SHORT * ASPECT_NETWORKS, SIGMA_SHORT, angle)
            ratio.append(gaussian_slope(cov, norm) / gaussian_slope(cov, "l2"))
        ax.plot(angles, ratio, color=NORM_COLORS[norm], lw=1.4, label=NORM_LABELS[norm])
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel(r"$L_\mathrm{true} / L_\mathrm{true}^{\ell_2}$")
    ax.legend(frameon=False)
    fig.savefig(f"{PLOT_DIR}/slope_needed_vs_angle.pdf")
    plt.close(fig)


def plot_floor_measure(floor_results, measure, y_label, file_prefix):
    """One figure per aspect and bound: the measure of the floor against the angle, one line per norm."""
    for aspect in ASPECTS:
        for L_ratio in L_RATIOS_FLOOR:
            fig, ax = new_figure()
            for norm in NORMS:
                ax.plot(ANGLES_DEGREES, floor_results[(aspect, L_ratio, norm)][measure], "o-", color=NORM_COLORS[norm],
                        lw=1.2, markersize=3, label=NORM_LABELS[norm])
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel(y_label)
            ax.legend(frameon=False)
            name = f"{file_prefix}_aspect{aspect:g}_L{L_ratio:g}".replace(".", "p")     # no dots in file names (LaTeX)
            fig.savefig(f"{PLOT_DIR}/{name}.pdf")
            plt.close(fig)


def plot_network_error(network_results, floor_results):
    """Relative error of the networks against the angle (mean and spread over seeds), with the floor as a line."""
    fig, ax = new_figure()
    for norm in NORMS:
        mean = [network_results[(norm, angle)]["relative_error"].mean() for angle in ANGLES_NETWORKS]
        std = [network_results[(norm, angle)]["relative_error"].std() for angle in ANGLES_NETWORKS]
        ax.errorbar(ANGLES_NETWORKS, mean, yerr=std, fmt="o", color=NORM_COLORS[norm], markersize=4, capsize=3,
                    label=NORM_LABELS[norm])
        if (ASPECT_NETWORKS, L_RATIO_NETWORKS, norm) in floor_results:
            ax.plot(ANGLES_DEGREES, floor_results[(ASPECT_NETWORKS, L_RATIO_NETWORKS, norm)]["relative_error"],
                    color=NORM_COLORS[norm], lw=0.8, ls="--")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Relative squared error")
    ax.legend(frameon=False)
    fig.savefig(f"{PLOT_DIR}/network_error_vs_angle.pdf")
    plt.close(fig)


def plot_contours(floors_for_contours, network_results, network_axes):
    """
    Half-maximum contours, one column per norm, one row per angle (the first and the last): the true Gaussian
    (black, dashed), the floor (colour, dashed) and the networks (colour, solid, mean over seeds, if trained at that angle).
    Full text width.
    """
    angles = [ANGLES_DEGREES[0], ANGLES_DEGREES[-1]]
    fig, axs = plt.subplots(len(angles), len(NORMS), figsize=(6.3, 4.4), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for row, angle in enumerate(angles):
        for column, norm in enumerate(NORMS):
            ax = axs[row, column]
            axes, floor, y = floors_for_contours[(norm, angle)]
            ax.contour(axes[0], axes[1], y.T, levels=[y.max() / 2], colors="black", linewidths=1.0, linestyles="--")
            ax.contour(axes[0], axes[1], floor.T, levels=[floor.max() / 2], colors=NORM_COLORS[norm],
                       linewidths=1.0, linestyles="--")
            if (norm, angle) in network_results:
                mean_fit = network_results[(norm, angle)]["prediction"].mean(axis=0)
                ax.contour(network_axes[0], network_axes[1], mean_fit.T, levels=[mean_fit.max() / 2],
                           colors=NORM_COLORS[norm], linewidths=1.4)
            limit = 2.2 * SIGMA_SHORT * ASPECT_NETWORKS
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(NORM_LABELS[norm])
            if column == 0:
                ax.set_ylabel(f"{angle:g} degrees\n" + r"$x_2$")
            if row == len(angles) - 1:
                ax.set_xlabel(r"$x_1$")
    legend_lines = [plt.Line2D([], [], color="black", lw=1.0, ls="--"),
                    plt.Line2D([], [], color="grey", lw=1.0, ls="--"),
                    plt.Line2D([], [], color="grey", lw=1.4)]
    fig.legend(legend_lines, ["Gaussian", "Floor", "Network"], loc="lower center", bbox_to_anchor=(0.5, 0.95),
               ncol=3, frameon=False)
    fig.savefig(f"{PLOT_DIR}/half_max_contours.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.rcParams.update(REPORT_STYLE)
    plt.rcParams["savefig.bbox"] = "standard"      # fixed figure sizes, see FIG_SIZE
    print(f"QUICK = {QUICK}. Saving the figures in {PLOT_DIR}")

    # 1) How large a bound each norm needs, against the angle (analytic, no fitting)
    plot_slope_needed()

    # 2) The floors: what the bound itself does, for every norm and angle
    floor_results, floors_for_contours = floor_scan()
    plot_floor_measure(floor_results, "relative_error", "Relative squared error", "floor_error_vs_angle")
    plot_floor_measure(floor_results, "width_short", r"$\mathrm{FWHM}_\mathrm{short}/\mathrm{FWHM}_\mathrm{true}$",
                       "floor_width_short_vs_angle")
    plot_floor_measure(floor_results, "peak", "Peak / true peak", "floor_peak_vs_angle")

    # 3) The networks: do they behave like their floors, and is the l2 network rotation invariant?
    network_results, network_axes, _ = network_scan()
    plot_network_error(network_results, floor_results)
    plot_contours(floors_for_contours, network_results, network_axes)

    # 4) Save the numbers, so the figures can be remade without computing everything again
    saved = {}
    for (aspect, L_ratio, norm), results in floor_results.items():
        for name, values in results.items():
            saved[f"floor_{name}_aspect{aspect:g}_L{L_ratio:g}_{norm}"] = values
    for (norm, angle), results in network_results.items():
        for name in ["relative_error", "width_short", "slope_used"]:
            saved[f"network_{name}_{norm}_angle{angle:g}"] = results[name]
    np.savez(f"{PLOT_DIR}/results.npz", angles_floor=ANGLES_DEGREES, angles_networks=np.array(ANGLES_NETWORKS), **saved)
    print("Saved the figures and results.npz")


# The training runs in parallel processes (train_many), which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main(), and the new processes only import the functions.
if __name__ == "__main__":
    main()
