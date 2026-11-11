# Stage 2 of the noise analysis: what does the Lipschitz bound do against noise?
#
# The signal is a round 2D Gaussian. Three kinds of noise, three parts:
#
#   A) Noise on the OUTPUT (the y values), redrawn many times:
#        - sine noise: plane waves with random phases (deterministic shape, random position)
#        - white noise: an independent random number at every point
#      For every L, the networks are trained on every noise draw. Over the draws, the error to the true Gaussian
#      splits into
#        bias^2   = (average fit - truth)^2           what every fit gets wrong in the same way (the cost of the bound)
#        variance = spread of the fits between draws  how much the fit follows the particular noise
#      Expected: bias rises below L_true (it is the floor), variance rises above it, the total is lowest near L_true.
#      The floor of every noisy draw is computed as well: what an ideal L-network would give.
#
#   B) Noise on the INPUT (the x values) during training: the network sees x + delta instead of x, with the y of x.
#      The best possible fit is then the Gaussian smeared by the noise, E[y | observed x] = (f * noise)(x), which for
#      Gaussian noise of width s is again a Gaussian, of width sqrt(sigma^2 + s^2). So the fit gets wider, and we
#      compare that with the widening caused by the bound (the floor of the smeared Gaussian).
#
#   C) Perturbations of the input AFTER training: how much can the output change when x is moved by at most epsilon?
#        - random perturbations: x + epsilon * (random direction)
#        - adversarial perturbations: the direction that changes the output the most, found by gradient ascent (PGD)
#      A Lipschitz network can never change by more than L * epsilon (the guarantee); an unconstrained network has no
#      such limit. The networks here are trained on white-noise data, where the unconstrained network learns steep
#      spikes that an attack can use.
#
# The bound is measured in INPUT_NORM (l2 by default: the norm that treats every direction equally, from stage 1).
#
# Run from the Lipschitz_concept folder:  python Toy_nd_noise.py
# The figures and the numbers (results.npz) are saved in Plots/Toy_nd_noise.

import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt

# "Homemade" modules
from Toy_utils import REPORT_STYLE, train_full_batch
from Toy_nd_utils import (gaussian_nd, isotropic_cov, make_grid, grid_integral, profile, wavevector,
                          mode_amplitude_nd, train_many)
from lipschitz_floor import lipschitz_floor_nd
from models import LipschitzModel, UnconstrainedModel


##### SETTINGS #####
QUICK = True        # True: few noise draws and short training, only to check that everything runs

INPUT_NORM = "l2"   # the norm of the input the bound is measured in ("l1", "l2" or "linf"), see models.py

# The signal: a round 2D Gaussian on a periodic grid (periodic, so the plane waves sit exactly on Fourier modes)
SIGMA = 0.5
X_HALF_WIDTH = 2.5                  # grid on [-2.5, 2.5)^2, five widths on each side
N_POINTS = 41                       # points per axis: a sparse grid, so an unconstrained network CAN follow the noise

# A) Output noise. Both have about 10 % of the peak height, and the same power as each other
NOISE_DRAWS = 2 if QUICK else 10                # independent noise draws (each network is trained on every draw)
L_RATIOS = [0.25, 0.5, 1.0, 2.0, 4.0]           # L / L_true (L_true: slope of the Gaussian in INPUT_NORM)
WAVE_MODES = [(4, 0), (0, 8), (10, 10)]         # whole periods across the grid: along x_1, along x_2, diagonal
NOISE_LEVEL = 0.1                               # noise amplitude relative to the peak of the Gaussian

# B) Input noise during training: width of the noise on x, relative to SIGMA
INPUT_NOISE_RATIOS = [0.0, 0.25, 0.5, 1.0]
INPUT_NOISE_COPIES = 2 if QUICK else 4          # every grid point is used this many times, each with its own noise
L_RATIOS_INPUT = [0.5, 1.0, 2.0]

# C) Perturbations after training
EPSILON_RATIOS = [0.01, 0.03, 0.1, 0.3]         # size of the perturbation (Euclidean length), relative to SIGMA
L_RATIOS_ATTACK = [0.5, 1.0, 2.0]
RANDOM_DIRECTIONS = 20                          # random directions tried per point
ATTACK_STEPS = 20                               # gradient ascent steps of the attack

# Networks and training (the same settings as in the notebooks)
HIDDEN_DIM = 64
N_GROUPS = 8
DEPTH = 5
STEPS = 100 if QUICK else 6000
LEARNING_RATE = 3e-3

NORM_LABELS = {"l1": r"$\ell_1$", "l2": r"$\ell_2$", "linf": r"$\ell_\infty$"}
C_TRUE, C_FLOOR, C_REF = "#1f1f1f", "#2a78d6", "#9a9893"

# Figures: fixed size and margins (saved without the tight cropping), so all the single plots are the same size in LaTeX
FIG_SIZE = (3.1, 2.8)
FIG_MARGINS = dict(left=0.24, right=0.96, top=0.96, bottom=0.18)

# Output folder, next to this script, so the script can be run from anywhere
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plots", "Toy_nd_noise")


##### THE SIGNAL #####
def setup_signal():
    """The grid, the points as an (N, 2) array, the true Gaussian on the grid, and its slope L_true in INPUT_NORM."""
    axes, X = make_grid(X_HALF_WIDTH, N_POINTS, 2, endpoint=False)
    truth = gaussian_nd(X, np.zeros(2), isotropic_cov(SIGMA, 2))
    # For a round Gaussian the largest gradient has length exp(-1/2) f_max / sigma (Euclidean). Its direction can be
    # along an axis or along the diagonal, so in the other norms the slope is the largest dual norm of that gradient:
    #   l1 (dual inf): along an axis, the same;  linf (dual l1): along the diagonal, sqrt(2) times larger
    slope_l2 = np.exp(-0.5) * truth.max() / SIGMA
    L_true = {"l1": slope_l2, "l2": slope_l2, "linf": np.sqrt(2) * slope_l2}[INPUT_NORM]
    return axes, X, X.reshape(-1, 2), truth, L_true


def wave_slope(A, k):
    """Largest slope of the plane wave A sin(k . x) in INPUT_NORM: the dual norm of A k."""
    dual_order = {"l1": np.inf, "l2": 2, "linf": 1}[INPUT_NORM]
    return A * np.linalg.norm(k, ord=dual_order)


def sine_noise_draw(X, amplitude, wavevectors, rng):
    """Plane waves with the given wave vectors and a random phase each: the shape is fixed, the position is random."""
    noise = np.zeros(X.shape[:-1])
    for k in wavevectors:
        noise = noise + amplitude * np.sin(X @ k + rng.uniform(0, 2 * np.pi))
    return noise


##### MEASURES #####
def width_along_axis(axes, f):
    """FWHM of the grid function f along the x_1 axis through the origin, with the half-maximum crossings interpolated."""
    t = np.linspace(axes[0][0], axes[0][-1], 8001)
    values = profile(axes, f, np.array([1.0, 0.0]), t)
    half_max = np.nanmax(values) / 2
    above = np.where(values >= half_max)[0]
    first, last = above[0], above[-1]
    t_left = t[first - 1] + (half_max - values[first - 1]) * (t[first] - t[first - 1]) / (values[first] - values[first - 1])
    t_right = t[last] + (half_max - values[last]) * (t[last + 1] - t[last]) / (values[last + 1] - values[last])
    return t_right - t_left


def bias_variance(fits, truth, noise_power):
    """
    Bias^2, variance and total error of a set of fits (one per noise draw), each averaged over the grid and divided by
    the noise power (the mean squared noise): 1 means as far from the truth as the noisy data itself.
    total = bias^2 + variance is the expected squared error of one fit.

    With a finite number R of draws, the average fit still contains a part of the noise (of size variance / R), which
    would show up as bias. So the variance is taken with R - 1 in the denominator (unbiased), and variance / R is
    subtracted from the measured bias. With few draws the corrected bias can come out slightly below zero; it is then
    set to zero (it is zero within the statistical uncertainty).
    """
    fits = np.asarray(fits)
    R = len(fits)
    mean_fit = fits.mean(axis=0)
    variance = np.mean(fits.var(axis=0, ddof=1)) / noise_power
    bias2 = max(np.mean((mean_fit - truth) ** 2) / noise_power - variance / R, 0.0)
    return bias2, variance, bias2 + variance


def lipschitz_jobs(x, y, L_values, seed, x_eval=None):
    """Training jobs for train_many: one Lipschitz network per L, and one unconstrained network, all on (x, y)."""
    jobs = []
    for L in L_values:
        job = dict(kind="lipschitz", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x, y=y,
                   model_kwargs=dict(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS, depth=DEPTH,
                                     lipschitz_bound=L, input_norm=INPUT_NORM))
        if x_eval is not None:
            job["x_eval"] = x_eval
        jobs.append(job)
    job = dict(kind="unconstrained", seed=seed, steps=STEPS, lr=LEARNING_RATE, x=x, y=y,
               model_kwargs=dict(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH))
    if x_eval is not None:
        job["x_eval"] = x_eval
    jobs.append(job)
    return jobs


##### A) OUTPUT NOISE #####
def output_noise_part(axes, X, x, truth, L_true, noise_kind):
    """
    Trains every network on every noise draw of one kind ("sine" or "white"), and computes the floors.
    Returns a dictionary with the bias^2, variance and total error for every L (networks and floors) and for the
    unconstrained network, the fits of the first draw (for the example figure) and, for sine noise, the fraction of
    every wave that survives.
    """
    rng = np.random.default_rng(0)
    amplitude = NOISE_LEVEL * truth.max()
    wavevectors = [wavevector(m, 2 * X_HALF_WIDTH) for m in WAVE_MODES]
    L_values = [r * L_true for r in L_RATIOS]

    # The noise draws. White noise gets the same power as the sine noise (three waves of amplitude A: 3 A^2 / 2)
    noise_draws = []
    for draw in range(NOISE_DRAWS):
        if noise_kind == "sine":
            noise_draws.append(sine_noise_draw(X, amplitude, wavevectors, rng))
        else:
            noise_draws.append(np.sqrt(3 * amplitude ** 2 / 2) * rng.standard_normal(truth.shape))
    noise_power = np.mean(np.array(noise_draws) ** 2)

    # Networks: every L and the unconstrained network, on every draw (one seed per draw), all in parallel
    jobs = []
    for draw in range(NOISE_DRAWS):
        jobs += lipschitz_jobs(x, (truth + noise_draws[draw]).ravel(), L_values, seed=draw)
    start = time.time()
    results = train_many(jobs)
    print(f"   {noise_kind} noise: trained {len(jobs)} networks in {time.time() - start:.0f} s")

    network_fits = {L: [] for L in L_values}
    unconstrained_fits = []
    k = 0
    for draw in range(NOISE_DRAWS):
        for L in L_values:
            network_fits[L].append(results[k]["prediction"].reshape(truth.shape))
            k += 1
        unconstrained_fits.append(results[k]["prediction"].reshape(truth.shape))
        k += 1

    # Floors: the best L-Lipschitz fit to every noisy draw
    floor_fits = {L: [] for L in L_values}
    for L in L_values:
        for draw in range(NOISE_DRAWS):
            floor, _ = lipschitz_floor_nd(truth + noise_draws[draw], axes, L, norm=INPUT_NORM)
            floor_fits[L].append(floor)

    # Bias^2, variance and total error
    part = {"network": [], "floor": []}
    for L in L_values:
        part["network"].append(bias_variance(network_fits[L], truth, noise_power))
        part["floor"].append(bias_variance(floor_fits[L], truth, noise_power))
        print(f"      L / L_true = {L / L_true:4.2f}: network bias^2 {part['network'][-1][0]:.3f}, "
              f"variance {part['network'][-1][1]:.3f} | floor bias^2 {part['floor'][-1][0]:.3f}, "
              f"variance {part['floor'][-1][1]:.3f}  (relative to the noise power)")
    part["network"] = np.array(part["network"])
    part["floor"] = np.array(part["floor"])
    part["unconstrained"] = np.array(bias_variance(unconstrained_fits, truth, noise_power))
    print(f"      unconstrained: bias^2 {part['unconstrained'][0]:.3f}, variance {part['unconstrained'][1]:.3f}")

    # Example fits from the first draw, for the cut figure
    part["example_noisy"] = truth + noise_draws[0]
    part["example_fits"] = {L: network_fits[L][0] for L in L_values}
    part["example_unconstrained"] = unconstrained_fits[0]

    # Sine noise: fraction of every wave that survives in the fit (its amplitude does not depend on the phase)
    if noise_kind == "sine":
        part["wave_slopes"] = np.array([wave_slope(amplitude, k) for k in wavevectors])
        part["surviving"] = np.zeros((len(L_values), len(wavevectors)))
        for i, L in enumerate(L_values):
            for j, k in enumerate(wavevectors):
                fractions = [mode_amplitude_nd(axes, fit - truth, k) / amplitude for fit in network_fits[L]]
                part["surviving"][i, j] = np.mean(fractions)
    return part


##### B) INPUT NOISE DURING TRAINING #####
def input_noise_part(axes, X, x, truth, L_true):
    """
    Trains networks on inputs with noise (x + delta, with the y of x), for every noise width, and measures the width
    of the fit along x_1. References: the width of the smeared Gaussian, and of the floor of the smeared Gaussian.
    """
    rng = np.random.default_rng(1)
    L_values = [r * L_true for r in L_RATIOS_INPUT]
    jobs = []
    for s_ratio in INPUT_NOISE_RATIOS:
        x_train = np.concatenate([x + s_ratio * SIGMA * rng.standard_normal(x.shape) for _ in range(INPUT_NOISE_COPIES)])
        y_train = np.concatenate([truth.ravel()] * INPUT_NOISE_COPIES)
        jobs += lipschitz_jobs(x_train, y_train, L_values, seed=0, x_eval=x)   # evaluated on the clean grid
    start = time.time()
    results = train_many(jobs)
    print(f"   input noise: trained {len(jobs)} networks in {time.time() - start:.0f} s")

    true_width = width_along_axis(axes, truth)
    part = {"network_width": np.zeros((len(INPUT_NOISE_RATIOS), len(L_values))),
            "unconstrained_width": np.zeros(len(INPUT_NOISE_RATIOS)),
            "smeared_width": np.zeros(len(INPUT_NOISE_RATIOS)),
            "smeared_floor_width": np.zeros((len(INPUT_NOISE_RATIOS), len(L_values)))}
    k = 0
    for i, s_ratio in enumerate(INPUT_NOISE_RATIOS):
        # The best possible fit: the Gaussian smeared by the input noise, a Gaussian of width sqrt(sigma^2 + s^2)
        smeared = gaussian_nd(X, np.zeros(2), isotropic_cov(np.sqrt(SIGMA ** 2 + (s_ratio * SIGMA) ** 2), 2))
        part["smeared_width"][i] = width_along_axis(axes, smeared) / true_width
        for j, L in enumerate(L_values):
            part["network_width"][i, j] = width_along_axis(axes, results[k]["prediction"].reshape(truth.shape)) / true_width
            floor, _ = lipschitz_floor_nd(smeared, axes, L, norm=INPUT_NORM)
            part["smeared_floor_width"][i, j] = width_along_axis(axes, floor) / true_width
            k += 1
        part["unconstrained_width"][i] = width_along_axis(axes, results[k]["prediction"].reshape(truth.shape)) / true_width
        k += 1
        print(f"      s / sigma = {s_ratio:4.2f}: smeared Gaussian {part['smeared_width'][i]:.3f}, networks "
              + ", ".join(f"{w:.3f}" for w in part["network_width"][i])
              + f", unconstrained {part['unconstrained_width'][i]:.3f}  (FWHM / true FWHM)")
    return part


##### C) PERTURBATIONS AFTER TRAINING #####
def largest_change_random(model, x, epsilon, rng):
    """Largest |f(x + delta) - f(x)| over RANDOM_DIRECTIONS random directions of length epsilon, for every point."""
    with torch.no_grad():
        f0 = model(x)
        largest = torch.zeros_like(f0)
        for _ in range(RANDOM_DIRECTIONS):
            direction = torch.tensor(rng.standard_normal(x.shape), dtype=torch.float32)
            direction = direction / direction.norm(dim=1, keepdim=True)
            largest = torch.maximum(largest, (model(x + epsilon * direction) - f0).abs())
    return largest.numpy().ravel()


def largest_change_adversarial(model, x, epsilon, rng):
    """
    Adversarial attack (projected gradient ascent, PGD): for every point, the perturbation delta with Euclidean length
    at most epsilon that changes the output the most. Starts from a random direction, takes ATTACK_STEPS steps along
    the gradient of (f(x + delta) - f(x))^2, and pulls delta back onto the ball after every step.
    """
    with torch.no_grad():
        f0 = model(x)
    delta = torch.tensor(rng.standard_normal(x.shape), dtype=torch.float32)
    delta = 0.5 * epsilon * delta / delta.norm(dim=1, keepdim=True)
    for _ in range(ATTACK_STEPS):
        delta.requires_grad_(True)
        change = ((model(x + delta) - f0) ** 2).sum()
        gradient, = torch.autograd.grad(change, delta)
        with torch.no_grad():
            step = gradient / (gradient.norm(dim=1, keepdim=True) + 1e-12)      # unit step along the gradient
            delta = delta + 0.25 * epsilon * step
            length = delta.norm(dim=1, keepdim=True)
            delta = delta * torch.clamp(epsilon / (length + 1e-12), max=1.0)    # back onto the ball of radius epsilon
    with torch.no_grad():
        return (model(x + delta) - f0).abs().numpy().ravel()


def guarantee_factor():
    """
    The Lipschitz guarantee for a perturbation of Euclidean length epsilon: |f(x + delta) - f(x)| <= factor * L * epsilon.
    The factor converts the Euclidean length into INPUT_NORM: ||delta||_1 <= sqrt(2) ||delta||_2, ||delta||_inf <= ||delta||_2.
    """
    return {"l1": np.sqrt(2), "l2": 1.0, "linf": 1.0}[INPUT_NORM]


def perturbation_part(axes, X, x, truth, L_true):
    """
    Trains Lipschitz networks and an unconstrained network on ONE white-noise draw (in this process, since the attack
    needs the trained model itself), and measures the largest change of the output for random and adversarial
    perturbations of every size.
    """
    rng = np.random.default_rng(2)
    noisy = truth + np.sqrt(3 / 2) * NOISE_LEVEL * truth.max() * rng.standard_normal(truth.shape)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(noisy.ravel(), dtype=torch.float32).reshape(-1, 1)

    models = {}
    for L_ratio in L_RATIOS_ATTACK:
        torch.manual_seed(0)
        model = LipschitzModel(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, n_groups=N_GROUPS, depth=DEPTH,
                               lipschitz_bound=L_ratio * L_true, input_norm=INPUT_NORM)
        models[f"L = {L_ratio:g} L_true"] = (train_full_batch(model, x_tensor, y_tensor, steps=STEPS, lr=LEARNING_RATE), L_ratio * L_true)
    torch.manual_seed(0)
    model = UnconstrainedModel(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)
    models["Unconstrained"] = (train_full_batch(model, x_tensor, y_tensor, steps=STEPS, lr=LEARNING_RATE), None)
    print(f"   perturbations: trained {len(models)} networks")

    part = {"epsilon": np.array(EPSILON_RATIOS) * SIGMA, "random": {}, "adversarial": {}, "L": {}}
    for name, (model, L) in models.items():
        part["L"][name] = L
        part["random"][name] = []
        part["adversarial"][name] = []
        for epsilon in part["epsilon"]:
            part["random"][name].append(largest_change_random(model, x_tensor, epsilon, rng).max())
            part["adversarial"][name].append(largest_change_adversarial(model, x_tensor, epsilon, rng).max())
        part["random"][name] = np.array(part["random"][name])
        part["adversarial"][name] = np.array(part["adversarial"][name])
        print(f"      {name:16s}: largest change random " + ", ".join(f"{v:.3f}" for v in part["random"][name])
              + " | adversarial " + ", ".join(f"{v:.3f}" for v in part["adversarial"][name]))
    return part


##### PLOTS #####
def new_figure():
    """One single plot in the report format, always the same size."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(**FIG_MARGINS)
    return fig, ax


def new_figure_legend_below():
    """A single plot with room for a legend below it, always the same size (taller than new_figure)."""
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], FIG_SIZE[1] + 1.4))
    fig.subplots_adjust(left=FIG_MARGINS["left"], right=FIG_MARGINS["right"], top=0.97, bottom=0.42)
    return fig, ax


def legend_below(ax, handles=None):
    """Legend under the plot, in two columns."""
    if handles is None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handlelength=1.2, columnspacing=0.8,
                  frameon=False)
    else:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handlelength=1.2,
                  columnspacing=0.8, frameon=False)


def L_ratio_ticks(ax, L_ratios):
    """x ticks at the L ratios, written as plain numbers (the log axis would otherwise write overlapping 3x10^-1 etc.)."""
    ax.set_xticks(L_ratios)
    ax.set_xticklabels([f"{r:g}" for r in L_ratios])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_bias_variance(part, noise_kind):
    """
    Bias^2, variance and total error against L / L_true: networks (points, solid) and floors (dashed), the total error
    of the unconstrained network as a horizontal line. Zeros (a floor with no bias) cannot be drawn on a log axis.
    """
    fig, ax = new_figure_legend_below()
    colors = {"total": "black", "bias": C_FLOOR, "variance": "#eb6834"}
    handles = []
    for index, name in [(2, "total"), (0, "bias"), (1, "variance")]:
        label = {"total": "Total", "bias": r"Bias$^2$", "variance": "Variance"}[name]
        line, = ax.plot(L_RATIOS, part["network"][:, index], "o-", color=colors[name], lw=1.2, markersize=4, label=label)
        handles.append(line)
        floor_values = np.where(part["floor"][:, index] > 0, part["floor"][:, index], np.nan)
        ax.plot(L_RATIOS, floor_values, color=colors[name], lw=0.8, ls="--")
    handles.append(plt.Line2D([], [], color="grey", lw=0.8, ls="--", label="Floor"))
    handles.append(ax.axhline(part["unconstrained"][2], color=C_REF, lw=1.2, ls=":", label="Unconstr. total"))
    ax.axvline(1, color="black", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    L_ratio_ticks(ax, L_RATIOS)
    ax.set_xlabel(r"$L / L_\mathrm{true}$")
    ax.set_ylabel("Error / noise power")
    legend_below(ax, handles)
    fig.savefig(f"{PLOT_DIR}/{noise_kind}_noise_bias_variance.pdf")
    plt.close(fig)


def plot_example_cuts(axes, truth, part, L_true, noise_kind):
    """Cut along x_1 through the centre: noisy data, truth, and the fits of the first draw. Legend below the plot."""
    fig, ax = new_figure_legend_below()
    middle = len(axes[1]) // 2
    ax.plot(axes[0], part["example_noisy"][:, middle], color="lightgrey", lw=0.8, label="Data")
    shades = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_RATIOS)))
    for (L, fit), color, r in zip(part["example_fits"].items(), shades, L_RATIOS):
        ax.plot(axes[0], fit[:, middle], color=color, lw=1.2, label=rf"$L = {r:g}\,L_\mathrm{{true}}$")
    ax.plot(axes[0], part["example_unconstrained"][:, middle], color=C_REF, lw=1.0, ls=":", label="Unconstr.")
    ax.plot(axes[0], truth[:, middle], color=C_TRUE, lw=1.2, ls="--", label="Truth")
    ax.set_xlabel(r"$x_1$ (with $x_2 = 0$)")
    ax.set_ylabel(r"$f$")
    legend_below(ax)
    fig.savefig(f"{PLOT_DIR}/{noise_kind}_noise_example_cuts.pdf")
    plt.close(fig)


def plot_surviving(part):
    """Sine noise: fraction of every wave that survives, against the slope it needs relative to the bound, A |k| / L."""
    fig, ax = new_figure()
    L_values = np.array(L_RATIOS) * part["L_true"]
    markers = ["o", "s", "D"]
    for j, (mode, marker) in enumerate(zip(WAVE_MODES, markers)):
        ax.plot(part["wave_slopes"][j] / L_values, part["surviving"][:, j], marker, ms=4, label=f"modes {mode}")
    ax.axvline(1, color="black", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$A\|k\| / L$")
    ax.set_ylabel("Fraction that survives")
    ax.legend(frameon=False)
    fig.savefig(f"{PLOT_DIR}/sine_noise_surviving_fraction.pdf")
    plt.close(fig)


def plot_input_noise(part):
    """Width of the fit against the input noise width: networks (points), smeared Gaussian and its floors (lines)."""
    fig, ax = new_figure_legend_below()
    shades = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_RATIOS_INPUT)))
    for j, (r, color) in enumerate(zip(L_RATIOS_INPUT, shades)):
        ax.plot(INPUT_NOISE_RATIOS, part["network_width"][:, j], "o", color=color, ms=4, label=rf"$L = {r:g}\,L_\mathrm{{true}}$")
        ax.plot(INPUT_NOISE_RATIOS, part["smeared_floor_width"][:, j], color=color, lw=0.8, ls="--")
    ax.plot(INPUT_NOISE_RATIOS, part["unconstrained_width"], "s", color=C_REF, ms=4, label="Unconstr.")
    ax.plot(INPUT_NOISE_RATIOS, part["smeared_width"], color=C_TRUE, lw=1.2, label="Smeared truth")
    ax.set_xlabel(r"Input noise $s / \sigma$")
    ax.set_ylabel(r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{true}$")
    legend_below(ax)
    fig.savefig(f"{PLOT_DIR}/input_noise_width.pdf")
    plt.close(fig)


def plot_perturbations(part, L_true):
    """
    Largest change of the output against the perturbation size: adversarial (solid, points) and random (dotted), and
    the Lipschitz guarantee L * epsilon of every Lipschitz network (thin dashed). Legend below the plot.
    """
    fig, ax = new_figure_legend_below()
    shades = list(plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_RATIOS_ATTACK)))) + [C_REF]
    for (name, L), color in zip(part["L"].items(), shades):
        if L is None:
            label = "Unconstr."
        else:
            label = rf"$L = {L / L_true:g}\,L_\mathrm{{true}}$"
        ax.plot(part["epsilon"], part["adversarial"][name], "o-", color=color, lw=1.2, ms=3, label=label)
        ax.plot(part["epsilon"], part["random"][name], color=color, lw=0.8, ls=":")
        if L is not None:
            ax.plot(part["epsilon"], guarantee_factor() * L * part["epsilon"], color=color, lw=0.6, ls="--")
    ax.plot(part["epsilon"], L_true * part["epsilon"], color=C_TRUE, lw=0.8, ls="-.", label=r"$L_\mathrm{true}\,\epsilon$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Perturbation size $\epsilon$")
    ax.set_ylabel(r"Largest $|f(x+\delta) - f(x)|$")
    legend_below(ax)
    fig.savefig(f"{PLOT_DIR}/perturbation_largest_change.pdf")
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.rcParams.update(REPORT_STYLE)
    plt.rcParams["savefig.bbox"] = "standard"      # fixed figure sizes, see FIG_SIZE
    print(f"QUICK = {QUICK}, input norm {INPUT_NORM}. Saving the figures in {PLOT_DIR}")

    axes, X, x, truth, L_true = setup_signal()
    print(f"Gaussian sigma = {SIGMA}, L_true = {L_true:.3f} ({INPUT_NORM}), grid {N_POINTS}^2 points")
    saved = {"L_ratios": np.array(L_RATIOS), "L_true": L_true}

    # A) Noise on the output
    print("A) Output noise")
    for noise_kind in ["sine", "white"]:
        part = output_noise_part(axes, X, x, truth, L_true, noise_kind)
        part["L_true"] = L_true
        plot_bias_variance(part, noise_kind)
        plot_example_cuts(axes, truth, part, L_true, noise_kind)
        if noise_kind == "sine":
            plot_surviving(part)
            saved["sine_surviving"] = part["surviving"]
            saved["sine_wave_slopes"] = part["wave_slopes"]
        for name in ["network", "floor", "unconstrained"]:
            saved[f"{noise_kind}_{name}_bias2_variance_total"] = part[name]

    # B) Noise on the input during training
    print("B) Input noise during training")
    part = input_noise_part(axes, X, x, truth, L_true)
    plot_input_noise(part)
    for name, values in part.items():
        saved[f"input_noise_{name}"] = values

    # C) Perturbations after training
    print("C) Perturbations after training")
    part = perturbation_part(axes, X, x, truth, L_true)
    plot_perturbations(part, L_true)
    saved["perturbation_epsilon"] = part["epsilon"]
    for name in part["L"]:
        key = name.replace(" ", "").replace("=", "").replace(".", "p")
        saved[f"perturbation_random_{key}"] = part["random"][name]
        saved[f"perturbation_adversarial_{key}"] = part["adversarial"][name]

    np.savez(f"{PLOT_DIR}/results.npz", **saved)
    print("Saved the figures and results.npz")


# The training runs in parallel processes (train_many), which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main(), and the new processes only import the functions.
if __name__ == "__main__":
    main()
