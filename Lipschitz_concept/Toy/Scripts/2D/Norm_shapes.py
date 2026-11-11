# GOAL: A slide figure of how the input norm decides the shape of the best L-Lipschitz fit. With the l1 norm the floor
# of a round Gaussian is a pyramid (diamond contours), with the l2 norm a cone (round contours), and trained Lipschitz
# networks with the same norm do the same.

# HOW: The 2D Gaussian of Optimal_solution.py (sigma = 0.1) at L = 0.12 L_true, where the floor runs at its full slope.
#   One panel per norm, seen from above: the contours of a Lipschitz network trained with that input_norm (random
#   training points, as in Optimal_solution.py), on top of the contours of the floor in the same norm
# l1 bounds every partial derivative on its own, max_i |df/dx_i| <= L, so along the diagonals the slope can reach
# sqrt(2) L. l2 bounds the length of the gradient, |grad f| <= L, the same in every direction.

# PLOTS: in Plots/2D/Norms_and_rotations (PDF for the report, PNG at 300 dpi for the slide)


import os
import sys
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_utils import REPORT_STYLE, plot_dir, results_dir, save_results, load_results
from Toy_nd_utils import (gaussian_nd, isotropic_cov, isotropic_lipschitz_constant, pyramid_radius, make_grid,
                          grid_gradient, fwhm_along, train_many)
from lipschitz_floor import lipschitz_floor_nd

QUICK = False                      # True: small grid, short training, only to check that everything runs
PLOT_DIR = plot_dir("2D", "Norms_and_rotations")
RESULTS_FILE = os.path.join(results_dir("2D", "Norms_and_rotations"),
                            "norm_shapes_quick.pkl" if QUICK else "norm_shapes.pkl")
LOAD_SAVED = True                  # True: use the saved floors and networks if they exist, instead of computing again

# The Gaussian and the bound
DIM = 2
SIGMA = 0.1
L_RATIO = 0.12                     # L / L_true: low enough for the floor to run at its full slope around the peak
NORMS = ["l1", "l2"]
POINTS_PER_SIGMA = 4 if QUICK else 8

# Networks, as in Optimal_solution.py: random training points, uniform in the box
N_TRAIN = 1000 if QUICK else 8000  # about the same density of points as in Optimal_solution.py
SAMPLE_SEED = 0
SEED = 0
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3
HIDDEN_DIM, N_GROUPS, DEPTH = 64, 8, 5

# Figure format
C_FLOOR, C_NET = "#2a78d6", "tab:orange"
LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])   # contour levels, as fractions of the peak of the floor
# No titles: the panels are labelled l1 (left) and l2 (right) in the report caption or on the slide


#### CALCULATIONS ####
def make_data():
    """
    The Gaussian on a grid that is wide enough for the floor, and N_TRAIN random training points in the same box.
    Returns L, axes, X, the Gaussian on the grid, and the training points with their Gaussian values.
    """
    L = L_RATIO * isotropic_lipschitz_constant(SIGMA, DIM)      # L_true is the same in l1 and l2 for a round Gaussian
    half_width = max(6 * SIGMA, 1.2 * pyramid_radius(L, DIM))
    axes, X = make_grid(half_width, 2 * int(half_width / SIGMA * POINTS_PER_SIGMA) + 1, DIM)
    y = gaussian_nd(X, np.zeros(DIM), isotropic_cov(SIGMA, DIM))
    rng = np.random.default_rng(SAMPLE_SEED)
    x_train = rng.uniform(-half_width, half_width, size=(N_TRAIN, DIM))
    y_train = gaussian_nd(x_train, np.zeros(DIM), isotropic_cov(SIGMA, DIM))
    return L, axes, X, y, x_train, y_train


def train_networks(X, x_train, y_train, L, shape):
    """One Lipschitz network per norm, trained on the random points and evaluated on the grid. Returns fits[norm]."""
    jobs = [dict(kind="lipschitz", seed=SEED, steps=STEPS, lr=LEARNING_RATE, x=x_train, y=y_train,
                 x_eval=X.reshape(-1, DIM),
                 model_kwargs=dict(input_dim=DIM, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS,
                                   depth=DEPTH, lipschitz_bound=L, input_norm=norm))
            for norm in NORMS]
    print(f"Training {len(jobs)} networks on {len(x_train)} random points")
    results = train_many(jobs)
    return {norm: result["prediction"].reshape(shape) for norm, result in zip(NORMS, results)}


def print_shape_check(axes, functions, L, name):
    """The partial derivatives halfway down the flank on the diagonal, and the FWHM along the diagonal / along the axis."""
    middle = len(axes[0]) // 2
    k = middle + middle // 4                      # a grid point on the diagonal, on the flank of the peak
    for norm, f in functions.items():
        g1, g2 = grid_gradient(f, axes)[k, k] / L
        ratio = fwhm_along(axes, f, [1, 1]) / fwhm_along(axes, f, [1, 0])
        print(f"{name:8s} {norm}: on the diagonal df/dx1 = {g1:+.2f} L, df/dx2 = {g2:+.2f} L "
              f"(slope {np.hypot(g1, g2):.2f} L), FWHM diagonal / axis = {ratio:.3f}")


#### PLOT ####
def plot_norm_shapes(axes, floors, fits, file_name, figsize, font_size):
    """
    One panel per norm (l1, l2), seen from above: the contours of the trained network (solid) on top of those of the
    floor (dashed), at the same levels. The figure is made at the size it is shown at, so the text keeps its size:
    full text width with 12 pt text for the report, about half a Beamer slide with 8 pt text for the slide.
    """
    limit = axes[0][-1]
    line_scale = font_size / 12                    # thinner lines in the small slide figure
    sizes = {name: font_size for name in ["font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                                          "legend.fontsize"]}
    with plt.rc_context(REPORT_STYLE), plt.rc_context(sizes):
        fig, axs = plt.subplots(1, len(NORMS), figsize=figsize, sharey=True, gridspec_kw=dict(wspace=0.1))
        for ax, norm in zip(axs, NORMS):
            levels = LEVELS * floors[norm].max()
            ax.contour(axes[0], axes[1], floors[norm].T, levels=levels, colors=[C_FLOOR], linestyles="--",
                       linewidths=1.0 * line_scale)
            ax.contour(axes[0], axes[1], fits[norm].T, levels=levels, colors=[C_NET], linewidths=1.4 * line_scale)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_xlabel(r"$x_1$")
        axs[0].set_ylabel(r"$x_2$")                    # the y axis is shared, so only the left panel is labelled

        handles = [plt.Line2D([], [], color=C_FLOOR, ls="--", lw=1.0 * line_scale, label="Best possible fit (floor)"),
                   plt.Line2D([], [], color=C_NET, lw=1.4 * line_scale, label="Trained Lipschitz network")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}.pdf", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/{file_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    L, axes, X, y, x_train, y_train = make_data()
    print(f"sigma = {SIGMA}, L = {L_RATIO} L_true = {L:.2f}, grid {len(axes[0])}^{DIM} points")

    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        saved = load_results(RESULTS_FILE)
    else:
        saved = {"floors": {norm: lipschitz_floor_nd(y, axes, L, norm=norm)[0] for norm in NORMS},
                 "fits": train_networks(X, x_train, y_train, L, y.shape)}
        save_results(saved, RESULTS_FILE)

    print_shape_check(axes, saved["floors"], L, "Floor")
    print_shape_check(axes, saved["fits"], L, "Network")
    plot_norm_shapes(axes, saved["floors"], saved["fits"], "norm_shapes_pyramid_and_cone", figsize=(6.3, 3.4),
                     font_size=12)                                    # report: full text width
    plot_norm_shapes(axes, saved["floors"], saved["fits"], "norm_shapes_pyramid_and_cone_slide", figsize=(3.0, 1.75),
                     font_size=8)                                     # Beamer: about half the slide width
    return 0


# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
