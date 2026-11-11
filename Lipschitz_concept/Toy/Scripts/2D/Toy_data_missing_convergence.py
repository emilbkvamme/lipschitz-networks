# GOAL: Check whether the Lipschitz networks in Toy_data_missing.py have converged.
# If the error keeps falling with more training steps (or a larger learning rate), the gap results are undertrained.
# If it levels off above the unconstrained network, the difference is real.

# HOW: The 90 degree wedge, two opposite 90 degree wedges and the full-data reference, L = 1 and 2 L_true, one seed,
# trained for an increasing number of steps with two learning rates. The unconstrained network as comparison.

import os
import csv
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the Toy folder
from Toy_nd_utils import *
from Toy_utils import REPORT_STYLE, plot_dir, results_dir, save_results, load_results
from Toy_data_missing import wedge_mask, INPUT_NORM, CONTOUR_LEVELS

QUICK = False                       # True: very short training, only to check that everything runs
PLOT_DIR = plot_dir("2D", os.path.join("Missing_data", "Convergence"))
RESULTS_FILE = os.path.join(results_dir("2D", os.path.join("Missing_data", "Convergence")),
                            "results_quick.pkl" if QUICK else "results.pkl")   # QUICK runs get their own file, so they never replace a full run
LOAD_SAVED = True                  # True: make the plots from RESULTS_FILE if it exists, instead of training again


def convergence_test():
    STEPS = [100, 300] if QUICK else [6000, 20000, 50000]
    LEARNING_RATES = [3e-3, 1e-2]
    L_RATIOS = [1.0, 2.0]
    SEED = 0
    POINTS_PER_SIGMA = 4 if QUICK else 8

    # The same Gaussian and grid as in missing_data_scan (its smallest L ratio there is 0.3)
    N = 2
    SIGMA = 1.0
    COVARIANCE = isotropic_cov(sigma=SIGMA, n=N)
    L_TRUE = gaussian_lipschitz_constant_nd(cov=COVARIANCE, norm=INPUT_NORM)
    half_width = max(5 * SIGMA, 1.2 * pyramid_radius(0.3 * L_TRUE, N))
    axes, X = make_grid(half_width, 2 * int(half_width / SIGMA * POINTS_PER_SIGMA) + 1, N)
    truth = gaussian_nd(X, np.zeros(N), COVARIANCE)
    x = X.reshape(-1, N)
    middle = len(axes[0]) // 2           # the index of the peak, x = 0

    cases = {"none": np.zeros(truth.shape, dtype=bool),            # full data: all the error comes from fitting
             "wedge90": wedge_mask(X, [(0, 90)]),
             "opposite90": wedge_mask(X, [(0, 90), (180, 90)])}

    # All jobs; labels[i] = (case, network, learning rate, steps). The unconstrained network only with lr 3e-3.
    jobs, labels = [], []
    for case, missing in cases.items():
        keep = ~missing.ravel()
        for steps in STEPS:
            for ratio in L_RATIOS:
                for lr in LEARNING_RATES:
                    jobs.append(dict(kind="lipschitz", seed=SEED, steps=steps, lr=lr,
                                     x=x[keep], y=truth.ravel()[keep], x_eval=x,
                                     model_kwargs=dict(input_dim=N, hidden_dim=64, output_dim=1, n_groups=8, depth=5,
                                                       lipschitz_bound=ratio * L_TRUE, input_norm=INPUT_NORM)))
                    labels.append((case, f"Lipschitz L={ratio:g}", lr, steps))
            jobs.append(dict(kind="unconstrained", seed=SEED, steps=steps, lr=3e-3,
                             x=x[keep], y=truth.ravel()[keep], x_eval=x,
                             model_kwargs=dict(input_dim=N, hidden_dim=64, output_dim=1, depth=5)))
            labels.append((case, "Unconstrained", 3e-3, steps))

    print(f"Training {len(jobs)} networks, grid {len(axes[0])}^{N} points")
    results = train_many(jobs)

    # The measures of every network
    table = {}
    print(f"{'case':10s} {'network':16s} {'lr':>7s} {'steps':>6s} {'MSE data':>10s} {'MSE gap':>10s} {'peak/true':>9s}")
    for (case, network, lr, steps), result in zip(labels, results):
        missing = cases[case]
        prediction = result["prediction"].reshape(truth.shape)
        error = prediction - truth
        MSE_rest = np.mean(error[~missing] ** 2)
        MSE_gap = np.mean(error[missing] ** 2) if missing.any() else np.nan
        peak_ratio = prediction[middle, middle] / truth[middle, middle]
        table[(case, network, lr, steps)] = {"MSE_rest": MSE_rest, "MSE_gap": MSE_gap, "peak_ratio": peak_ratio,
                                             "prediction": prediction}
        print(f"{case:10s} {network:16s} {lr:7.0e} {steps:6d} {MSE_rest:10.2e} {MSE_gap:10.2e} {peak_ratio:9.3f}")

    return table, STEPS, LEARNING_RATES, L_RATIOS, axes, truth, cases


#### PLOTS ####
def save_table(table, file_name):
    """Saves the table as a CSV file (one row per network), so the runs can be compared later."""
    with open(f"{PLOT_DIR}/{file_name}", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "network", "lr", "steps", "MSE_data", "MSE_gap", "peak_ratio"])
        for (case, network, lr, steps), values in table.items():
            writer.writerow([case, network, lr, steps,
                             f"{values['MSE_rest']:.4e}", f"{values['MSE_gap']:.4e}", f"{values['peak_ratio']:.4f}"])


def plot_convergence_overview(table, steps_list, learning_rates, L_ratios, file_name):
    """
    Overview against the number of training steps. Rows: full data and the wedge cases. Columns: MSE on the training
    points, MSE in the gap, and the peak height / true peak. One line per L and learning rate, the unconstrained dashed.
    The full data has no gap, so that panel holds the legend.
    """
    cases = [("none", "Full data"), ("wedge90", "Wedge 90"), ("opposite90", "Opposite 90")]
    measures = [("MSE_rest", "MSE on data", True), ("MSE_gap", "MSE in gap", True),
                ("peak_ratio", "Peak / true peak", False)]

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(cases), 3, figsize=(8.0, 2.6 * len(cases)), sharex=True)
        for row, (case, case_label) in enumerate(cases):
            for column, (measure, y_label, log_y) in enumerate(measures):
                ax = axs[row, column]
                if case == "none" and measure == "MSE_gap":
                    ax.axis("off")                  # no gap in the full data: this panel holds the legend
                    continue
                for ratio, color in zip(L_ratios, ["tab:blue", "tab:green"]):
                    for lr, marker in zip(learning_rates, ["o", "s"]):
                        values = [table[(case, f"Lipschitz L={ratio:g}", lr, steps)][measure] for steps in steps_list]
                        ax.plot(steps_list, values, marker=marker, color=color, lw=1.2, markersize=4,
                                label=f"$L = {ratio:g}\\,L_\\mathrm{{true}}$, lr {lr:g}")
                values = [table[(case, "Unconstrained", 3e-3, steps)][measure] for steps in steps_list]
                ax.plot(steps_list, values, "o--", color="tab:orange", lw=1.2, markersize=4, label="Unconstrained")
                if measure == "peak_ratio":
                    ax.axhline(1.0, color="black", ls="dotted", lw=1.0)   # the true peak
                ax.set_xscale("log")
                ax.set_xticks(steps_list, [f"{steps:g}" for steps in steps_list])   # only the trained step values
                ax.xaxis.set_minor_locator(plt.NullLocator())
                if log_y:
                    ax.set_yscale("log")
                ax.set_title(f"{case_label}: {y_label}", fontsize=12)
                if row == len(cases) - 1:
                    ax.set_xlabel("Training steps")

        # One legend for all panels, in the empty panel
        handles, labels = axs[1, 0].get_legend_handles_labels()
        axs[0, 1].legend(handles, labels, frameon=False, loc="center", handlelength=1.6)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)
def plot_contour_convergence(table, case, missing, axes, truth, steps_list, learning_rates, L_ratios, file_name,
                             half_width=2.5):
    """
    Contours at CONTOUR_LEVELS times the true peak, one panel per training length (rows) and Lipschitz setting
    (columns: L and learning rate). In every panel: the truth dotted (black), the Lipschitz network solid (blue),
    the unconstrained network with the same number of steps solid (orange), and the gap shaded grey.
    """
    levels = [level * truth.max() for level in CONTOUR_LEVELS]
    settings = [(ratio, lr) for ratio in L_ratios for lr in learning_rates]   # the columns

    with plt.rc_context(REPORT_STYLE):
        fig, axs = plt.subplots(len(steps_list), len(settings), figsize=(2.2 * len(settings), 2.2 * len(steps_list) + 0.6),
                                sharex=True, sharey=True, squeeze=False)
        for row, steps in enumerate(steps_list):
            unconstrained = table[(case, "Unconstrained", 3e-3, steps)]["prediction"]
            for column, (ratio, lr) in enumerate(settings):
                ax = axs[row, column]
                prediction = table[(case, f"Lipschitz L={ratio:g}", lr, steps)]["prediction"]
                if missing.any():
                    ax.contourf(axes[0], axes[1], missing.T.astype(float), levels=[0.5, 1.5], colors=["0.88"])
                ax.contour(axes[0], axes[1], truth.T, levels=levels, colors="black", linestyles="dotted", linewidths=1.0)
                ax.contour(axes[0], axes[1], unconstrained.T, levels=levels, colors="tab:orange", linewidths=1.0)
                ax.contour(axes[0], axes[1], prediction.T, levels=levels, colors="tab:blue", linewidths=1.0)
                ax.set_xlim(-half_width, half_width)
                ax.set_ylim(-half_width, half_width)
                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                ax.yaxis.set_major_locator(plt.MultipleLocator(2))
                if row == 0:
                    ax.set_title(f"$L = {ratio:g}\\,L_\\mathrm{{true}}$, lr {lr:g}", fontsize=12)
                if column == 0:
                    ax.set_ylabel(f"{steps:g} steps")
                if row == len(steps_list) - 1:
                    ax.set_xlabel(r"$x_1$")

        # One legend for all panels, below the figure
        handles = [plt.Line2D([], [], color="black", ls="dotted", lw=1.2, label="Truth"),
                   plt.Line2D([], [], color="tab:blue", lw=1.2, label="Lipschitz"),
                   plt.Line2D([], [], color="tab:orange", lw=1.2, label="Unconstrained")]
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.06, 1, 1))     # leave room for the legend at the bottom
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)
#######


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    if LOAD_SAVED and os.path.exists(RESULTS_FILE):
        results = load_results(RESULTS_FILE)
    else:
        results = convergence_test()
        save_results(results, RESULTS_FILE)
    table, steps_list, learning_rates, L_ratios, axes, truth, cases = results
    save_table(table, "convergence_table.csv")
    plot_convergence_overview(table, steps_list, learning_rates, L_ratios, "convergence_overview.pdf")
    for case, missing in cases.items():
        plot_contour_convergence(table, case, missing, axes, truth, steps_list, learning_rates, L_ratios,
                                 f"convergence_contours_{case}.pdf")
    return 0


if __name__ == "__main__":
    main()
