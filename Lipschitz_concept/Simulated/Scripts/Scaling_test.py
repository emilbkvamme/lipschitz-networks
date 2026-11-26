# WHAT: Performance of the Lipschitz network on the simulated qq->Zg data as a function of the amount of data.
#       pT + angles inputs, l2 norm, fixed bounds L = L_RATIO * L_data, with L_data estimated on ALL training events.
# WHY: With few events there is little to go on between them. The bound limits how much the network can change from
#      one event to the next, so it may need fewer events than the unconstrained network to reach the same error.
#      L_data itself grows with the number of events (the neighbours lie closer together, so steeper slopes are seen).
#      The L_data of a small sample (the local L_data) is therefore lower than the L the target needs, and is shown
#      next to the bounds for comparison.

# HOW:
    # 1. Load the data, transform it to pT + angles, take log10 of the target, split 60-20-20 and standardize, as in
    #    L_scan_pt_angles_l2.py. The L_data of ALL training events sets the bounds of the networks.
    # 2. For every seed and fraction: a random subset of the training and validation events (the same fraction of both,
    #    as if the simulation had produced fewer events). The test set is always complete, so every network is measured
    #    equally precisely. The L_data of the training subset (the local L_data) is calculated too.
    # 3. On every subset: one Lipschitz network per L_RATIO and one unconstrained network. Every network gets the same
    #    number of gradient steps, so a network on a small subset is not also trained for a shorter time.
    # 4. Plots: MSE, MAE and used slope against the number of training events, and for every subset size the
    #    distribution of the test errors (truth - prediction) of its networks, and the mean error against the true |M|^2
    #    (where does the bound cut off the steepest peaks?), with the distribution of the true |M|^2 below it.


# ======== IMPORTS ============#
import os
import sys
import math
import numpy as np
import torch

# The functions for the data, the training and the plots are in Simulation_utils.py (in the Simulated folder, one up)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Simulation_utils import (max_cores_from_argv, limit_cores, int_from_argv, scaled_lr, get_simulated_data, BASES, basis_from_argv,
                              drop_constant_columns, split_sim_data, standardize, lipschitz_bound_estimate_of_data,
                              LipschitzModel, UnconstrainedModel, fit_many, used_slope, save_results, load_results,
                              save_figure, plt, PANEL_SIZE, C_UNCONSTRAINED, training_summary,
                              TRAIN_VERSION)

# The basis of the inputs: --basis NAME on the command line, a key of BASES in Simulation_utils.py
#   logpt_cossin (default): log pT, (cos phi, sin phi) in [-1, 1], rapidity y
#   logpt_cossin_costheta:  the same with cos(theta) in [-1, 1] instead of y
#   pt_angles:              pT, phi in [-pi, pi], y (the basis of the earlier runs)
BASIS = basis_from_argv(default="logpt_cossin")
# The name of the plot and result folders of this test. The pt_angles basis keeps the folder of the earlier runs.
TEST_NAME = "Scaling_test" if BASIS == "pt_angles" else f"Scaling_test_{BASIS}"
INPUT_NORM = "l2"               # the norm of the network inputs, as in the l2 L scans
QUICK = "--quick" in sys.argv   # a fast test of the whole script: are there any errors before the long run?


# ======== FUNCTIONS ============#
def epochs_for_same_steps(epochs_of_all_events, n_train_all, n_train, batch_size):
    """
    The number of epochs on n_train events that gives as many gradient steps as epochs_of_all_events epochs on all
    n_train_all training events. One epoch is one pass over the training events, in steps of batch_size events.
    Example: all 454849 training events give 445 steps per epoch, 454 events give 1. So 200 epochs on all events
    (89000 steps) are 89000 epochs on 454 events.
    """
    steps_per_epoch_all = math.ceil(n_train_all / batch_size)
    steps_per_epoch = math.ceil(n_train / batch_size)
    return math.ceil(epochs_of_all_events * steps_per_epoch_all / steps_per_epoch)


def model_path(model_dir, seed, n_train, name):
    """The file of one trained network in model_dir (None: the networks are not saved)."""
    return None if model_dir is None else os.path.join(model_dir, f"seed{seed}_{n_train}_events_{name}.pt")


def scaling_one_seed(fractions, L_arr, train_data, val_data, test_data, model_params, training_params, input_norm, seed,
                     max_cores=None, model_dir=None):
    """
    For one seed: a random subset of the training and validation events for every fraction, and on each subset one
    Lipschitz network per L and one unconstrained network, all trained in parallel.
    training_params: num_epochs, max_iter and lr_patience are given in epochs of ALL training events, and are converted
    to the same number of gradient steps on each subset.
    max_cores: the most cores used at the same time (None: the default of fit_many).
    model_dir: folder for the trained networks (one .pt file each). A network already saved there with the same
    settings and data is loaded instead of trained again. None: nothing is saved.
    Returns one dictionary per network, in the order (fraction, then L, the unconstrained network last): the fraction,
    the number of training events, L (None for the unconstrained network), the L_data of the training subset, the
    number of epochs it was allowed, the losses per epoch, the test predictions, the trained weights, the test MSE and
    MAE, and the largest slope the network uses on the test events.
    """
    # One random order of the training and of the validation events per seed. The subset of a fraction is the first
    # events in this order, so the subsets of one seed lie inside each other: a larger subset only adds events.
    generator = torch.Generator().manual_seed(seed)
    train_order = torch.randperm(len(train_data[0]), generator=generator)
    val_order = torch.randperm(len(val_data[0]), generator=generator)

    jobs = []
    infos = [] # What each job is: the fraction, the subset size, L, the L_data of the subset, the epochs it is allowed
    for fraction in fractions:
        n_train = int(fraction * len(train_order))
        n_val = int(fraction * len(val_order))
        train_subset = (train_data[0][train_order[:n_train]], train_data[1][train_order[:n_train]])
        val_subset = (val_data[0][val_order[:n_val]], val_data[1][val_order[:n_val]])

        # The local L_data: what the Lipschitz estimate gives with only these training events
        L_data_subset = lipschitz_bound_estimate_of_data(train_subset[0], train_subset[1], norm=2, max_cores=max_cores)
        print(f"Seed {seed}, fraction {fraction}: {n_train} training and {n_val} validation events, "
              f"local L_data = {L_data_subset:.3f}")

        # The same number of gradient steps on every subset. A small subset has few steps per epoch, so it is only
        # evaluated once per "epoch of all events" (eval_every subset epochs): every subset then gets about the same
        # number of evaluations, and the early stopping patience and the learning rate patience (which count
        # evaluations) keep the meaning they have on all events. Before, the smallest subset was evaluated after every
        # single step, which took most of its time.
        num_epochs = epochs_for_same_steps(training_params["num_epochs"], len(train_order), n_train,
                                           training_params["batch_size"])
        eval_every = epochs_for_same_steps(1, len(train_order), n_train, training_params["batch_size"])
        fit_settings = dict(trn_data=train_subset, val_data=val_subset, test_data=test_data, seed=seed,
                            lr=training_params["lr"], batch_size=training_params["batch_size"], num_epochs=num_epochs,
                            max_iter=training_params["max_iter"], lr_patience=training_params["lr_patience"],
                            eval_every=eval_every)

        # The output starts at the mean target of the training subset: the only mean a sample of this size would give
        subset_model_params = dict(**model_params, output_bias_init=float(train_subset[1].mean()))

        for L in L_arr:
            lipschitz_kwargs = dict(**subset_model_params, lipschitz_bound=float(L), input_norm=input_norm)
            jobs.append(dict(model_cls=LipschitzModel, model_kwargs=lipschitz_kwargs, **fit_settings,
                             model_path=model_path(model_dir, seed, n_train, f"L{float(L):.6g}")))
            infos.append(dict(fraction=fraction, n_train=n_train, L=float(L), L_data_subset=L_data_subset,
                              num_epochs=num_epochs))
        jobs.append(dict(model_cls=UnconstrainedModel, model_kwargs=subset_model_params, **fit_settings,
                         model_path=model_path(model_dir, seed, n_train, "unconstrained")))
        infos.append(dict(fraction=fraction, n_train=n_train, L=None, L_data_subset=L_data_subset,
                          num_epochs=num_epochs))

    print(f"Seed {seed}: training {len(jobs)} networks")
    results = fit_many(jobs, max_cores=max_cores)

    y_test = test_data[1].numpy()
    for job, info, result in zip(jobs, infos, results):
        result.update(info)
        model = job["model_cls"](**job["model_kwargs"]) # The trained network, rebuilt from its weights
        model.load_state_dict(result["state_dict"])

        errors = result["predictions"] - y_test # In units of the target, log10 |M|^2
        result["MSE"] = float(np.mean(errors**2))
        result["MAE"] = float(np.mean(np.abs(errors)))
        result["max_slope"], result["slope_999"] = used_slope(model, test_data[0], input_norm)
    return results


def print_scaling_summary(results):
    """One line per network: the subset, the test errors, the largest slope it uses, and how the training went."""
    print(f"{'events':>8} {'L':>7} {'L_local':>8} {'MSE':>10} {'MAE':>8} {'slope':>7} {'epochs':>8} {'best':>8}")
    for result in results:
        name = "none" if result["L"] is None else f"{result['L']:.3g}"
        trained, best, _, capped = training_summary(result, result["num_epochs"])
        note = ""
        if capped:
            note = "  hit the cap (best weights kept)" if "eval_epochs" in result else "  hit the cap: last weights, not the best"
        print(f"{result['n_train']:8d} {name:>7} {result['L_data_subset']:8.3f} {result['MSE']:10.3e} "
              f"{result['MAE']:8.4f} {result['max_slope']:7.3f} {trained:8d} {best:8d}{note}")


def collect_scaling(all_results, measure, L):
    """
    One measure (e.g. "MSE") of the networks with bound L (None: the unconstrained networks), for every seed and
    fraction: one row per seed, one column per fraction.
    """
    values = []
    for results in all_results:
        row = []
        for result in results:
            if result["L"] == L:
                row.append(result[measure])
        values.append(row)
    return np.array(values)


def plot_errorbars(ax, x, values, color, label):
    # The mean over the seeds, with the lowest and highest seed as error bar (as plot_scan in Simulation_utils)
    mean = values.mean(axis=0)
    error = [mean - values.min(axis=0), values.max(axis=0) - mean]
    ax.errorbar(x, mean, yerr=error, fmt="o-", color=color, label=label)


def plot_vs_events(n_events, all_results, L_arr, L_ratios, measure, y_label, plot_dir, file_name):
    """
    One measure against the number of training events: the Lipschitz networks in shades of orange, from light
    (smallest L) to dark (largest L), the unconstrained network in grey.
    """
    shades = plt.cm.Oranges(np.linspace(0.5, 0.9, len(L_arr)))
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for i in range(len(L_arr)):
        plot_errorbars(ax, n_events, collect_scaling(all_results, measure, float(L_arr[i])), shades[i],
                       f"{L_ratios[i]:g}" + r" $L_\mathrm{data}$")
    plot_errorbars(ax, n_events, collect_scaling(all_results, measure, None), C_UNCONSTRAINED, "Unconstrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training events")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    save_figure(fig, plot_dir, file_name)


def plot_used_slope_vs_events(n_events, all_results, L_arr, L_ratios, L_data_all, plot_dir, file_name):
    """
    The largest slope each network uses on the test events, against the number of training events, in units of the
    L_data of all training events. The bounds are fixed (scaled to the L_data of all events), so they are horizontal
    dotted lines. The local L_data, estimated on each training subset, changes with the number of events: black line.
    """
    shades = plt.cm.Oranges(np.linspace(0.5, 0.9, len(L_arr)))
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for i in range(len(L_arr)):
        plot_errorbars(ax, n_events, collect_scaling(all_results, "max_slope", float(L_arr[i])) / L_data_all,
                       shades[i], f"{L_ratios[i]:g}" + r" $L_\mathrm{data}$")
        ax.axhline(L_ratios[i], color=shades[i], ls=":") # The bound of this network
    plot_errorbars(ax, n_events, collect_scaling(all_results, "max_slope", None) / L_data_all, C_UNCONSTRAINED,
                   "Unconstrained")
    # The local L_data is the same for every network on a subset, so it is taken from the unconstrained networks
    plot_errorbars(ax, n_events, collect_scaling(all_results, "L_data_subset", None) / L_data_all, "black",
                   r"Local $L_\mathrm{data}$")
    ax.set_xscale("log")
    # Linear y: the slopes only run from about 0.8 to 2 L_data
    ax.set_xlabel("Training events")
    ax.set_ylabel(r"Used slope / $L_\mathrm{data}$")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2) # Below: the lines fill the axes
    save_figure(fig, plot_dir, file_name)


def find_network(results, fraction, L):
    """The network of one seed trained on the subset of this fraction, with bound L (None: the unconstrained one)."""
    for result in results:
        if result["fraction"] == fraction and result["L"] == L:
            return result


def plot_errors_one_fraction(results, fraction, L_arr, L_ratios, y_test, plot_dir, file_name):
    """
    Truth minus prediction (log10 |M|^2) on the test events, for the networks of one seed trained on the subset of one
    fraction, as step histograms: the Lipschitz networks in shades of orange, from light (smallest L) to dark (largest
    L), the unconstrained network in grey.
    The errors shrink by about a factor 100 from the smallest to the largest subset, so every subset gets its own x range:
    symmetric, out to the 99.9th percentile of |error| of the widest of its networks. The 0.1% largest errors of that
    network (and the few beyond that range of the other networks) are not shown.
    """
    shades = plt.cm.Oranges(np.linspace(0.5, 0.9, len(L_arr)))
    errors = []
    colors = []
    labels = []
    for i in range(len(L_arr)):
        network = find_network(results, fraction, float(L_arr[i]))
        errors.append(y_test[:, 0] - network["predictions"][:, 0])
        colors.append(shades[i])
        labels.append(f"{L_ratios[i]:g}" + r" $L_\mathrm{data}$")
    network = find_network(results, fraction, None)
    errors.append(y_test[:, 0] - network["predictions"][:, 0])
    colors.append(C_UNCONSTRAINED)
    labels.append("Unconstrained")

    # One x range for all the networks of this subset, so their histograms can be compared bin by bin
    limit = 0.0
    for error in errors:
        limit = max(limit, np.percentile(np.abs(error), 99.9))
    bins = np.linspace(-limit, limit, 101)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for error, color, label in zip(errors, colors, labels):
        ax.hist(error, bins=bins, histtype="step", color=color, label=label)
    ax.set_yscale("log")
    ax.set_title(f"{network['n_train']} training events") # The plots of the subsets look alike otherwise
    ax.set_xlabel(r"Truth $-$ prediction ($\log_{10}|M|^2$)")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2) # Below: the histograms fill the axes
    save_figure(fig, plot_dir, file_name)


def mean_error_in_bins(all_results, fraction, L, y_true, in_bins):
    """
    The mean of truth minus prediction of the test events in every bin, for the networks with bound L (None: the
    unconstrained ones) trained on the subset of this fraction: one row per seed, one column per bin.
    in_bins: one boolean array per bin, True for the test events in that bin.
    """
    values = []
    for results in all_results:
        network = find_network(results, fraction, L)
        error = y_true - network["predictions"][:, 0]
        row = []
        for in_bin in in_bins:
            row.append(error[in_bin].mean())
        values.append(row)
    return np.array(values)


def plot_bias_vs_truth(all_results, fraction, L_arr, L_ratios, y_test, plot_dir, file_name, min_events=5):
    """
    The mean of truth minus prediction (log10 |M|^2) of the test events in bins of the true log10 |M|^2, for the
    networks trained on the subset of one fraction: the Lipschitz networks in shades of orange, the unconstrained
    network in grey. The mean over the seeds, with the lowest and highest seed as error bar.
    Positive: the network predicts too low. A network whose slope is limited cannot climb all the way up the steepest
    peaks of |M|^2, so it predicts too low there, and more so the higher the peak.
    Bins with fewer than min_events test events are left out: their mean would rest on one or two events.
    Lower panel: the distribution of the true log10 |M|^2 of the test events, in the same bins, to compare the shape of
    the bias with how many events there are at each height. The training subset has the same shape, with
    (training events / test events) times as many events in every bin.
    """
    y_true = y_test[:, 0]
    edges = np.arange(-1, 3.01, 0.25) # The true log10 |M|^2 runs from -0.78 to 2.84
    centres = []
    in_bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (y_true >= lo) & (y_true < hi)
        if in_bin.sum() >= min_events:
            centres.append((lo + hi) / 2)
            in_bins.append(in_bin)

    shades = plt.cm.Oranges(np.linspace(0.5, 0.9, len(L_arr)))
    # Two panels on the same x axis: the bias on top, the distribution of the target below (half as high)
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(PANEL_SIZE[0], 1.5 * PANEL_SIZE[1]), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax.axhline(0, color="black", lw=0.8) # No bias: the network is right on average
    for i in range(len(L_arr)):
        plot_errorbars(ax, centres, mean_error_in_bins(all_results, fraction, float(L_arr[i]), y_true, in_bins),
                       shades[i], f"{L_ratios[i]:g}" + r" $L_\mathrm{data}$")
    plot_errorbars(ax, centres, mean_error_in_bins(all_results, fraction, None, y_true, in_bins), C_UNCONSTRAINED,
                   "Unconstrained")
    n_train = find_network(all_results[0], fraction, None)["n_train"]
    ax.set_title(f"{n_train} training events") # The plots of the subsets look alike otherwise
    ax.set_ylabel(r"Mean truth $-$ prediction")
    ax.legend(frameon=False, loc="upper left")

    ax_hist.hist(y_true, bins=edges, histtype="step", color="black")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel(r"True $\log_{10}|M|^2$")
    ax_hist.set_ylabel("Test events")
    save_figure(fig, plot_dir, file_name)


def main():
    # PATHS, for this script only. By default relative to this file, so the script runs from any folder and on any
    # machine. Point them elsewhere to read the data from, or write the plots and results to, another place.
    SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the Simulated folder
    DATA_PATH = os.path.join(SIM_DIR, "zg.npy")                    # the simulated qq->Zg events
    PLOT_DIR = os.path.join(SIM_DIR, "Plots", TEST_NAME)
    RESULTS_DIR = os.path.join(SIM_DIR, "Results", TEST_NAME)      # the result files and the trained networks

    # PARAMETERS
    FRACTIONS = [0.001, 0.01, 0.1, 0.5, 1]
    SEEDS = [0, 1, 2, 3, 4] # A different subset and a different network start for every seed
    # The bounds, as multiples of the L_data of ALL training events. From L_scan_pt_angles_l2: at 1 L_data the bound
    # still limits the network (the MSE falls steeply up to 1.23 L_data), from about 1.5 L_data it no longer does.
    L_RATIOS = [1.0, 2.0]
    # The most cores the run may use (networks trained at the same time). The Mac has 6 fast cores (the 2 slow ones
    # would hold up a round). Change it on the command line with --max-cores N.
    MAX_CORES = max_cores_from_argv(default=6)

    # The same training settings as the l2 L scans. num_epochs, max_iter and lr_patience are in epochs of ALL training
    # events (445 steps each), and every subset gets the same number of steps (see epochs_for_same_steps)
    LR_PATIENCE = 5 # Halve the learning rate when the validation loss has not improved for more than this long
    # Batch size: --batch-size N on the command line (default 1024, 445 steps per epoch). The learning rate at the start
    # is 3e-3 at batch 1024 and scales with sqrt(batch size / 1024) (see scaled_lr in train_eval.py).
    BATCH_SIZE = int_from_argv("--batch-size", 1024)
    LR = scaled_lr(3e-3, BATCH_SIZE)
    NUM_EPOCHS = 200 # The cap: 200 * 445 = 89000 gradient steps for every network
    MAX_ITER = 20 # Early stopping patience: stop when the validation loss has not improved for this long
    DEPTH = 3
    HIDDEN_DIM = 64

    if QUICK:
        FRACTIONS = [FRACTIONS[0], FRACTIONS[-1]] # The smallest and the largest subset
        SEEDS = [0]
        NUM_EPOCHS = 2
        MAX_ITER = 2
        LR_PATIENCE = 1
    ending = "_quick" if QUICK else "" # The quick run gets its own result and plot files
    if BATCH_SIZE != 1024:
        ending += f"_bs{BATCH_SIZE}" # other batch sizes get their own result and plot files

    limit_cores(MAX_CORES) # The threads of this process too (the slope calculations, the neighbour search)

    # LOAD SIMULATION DATA: the 16 Cartesian momentum columns, and |M|^2
    data, target = get_simulated_data(DATA_PATH)

    # TRANSFORMATION OF DATA, as in L_scan_pt_angles_l2.py
    log_target = np.log10(target)
    data = BASES[BASIS](data) # 16 -> 10 (pt_angles) or 12 input variables
    data = drop_constant_columns(data)

    # TRAIN-VALIDATION-TEST SPLIT of all events, then standardized inputs. The subsets are taken from these training and
    # validation events, and the test set is the same for every subset.
    # Standardized with the mean and std of ALL training events, not of each subset: L is a slope in standardized units,
    # so one standardization keeps L the same quantity on every subset.
    train_data, val_data, test_data = split_sim_data(data, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)

    # L OF ALL TRAINING EVENTS: sets the bounds of the networks on every subset
    L_data_all = lipschitz_bound_estimate_of_data(train_data[0], train_data[1], norm=2, max_cores=MAX_CORES)
    L_arr = np.array(L_RATIOS) * L_data_all
    print(f"L_data of all training events = {L_data_all:.3f} (l2), bounds L = {np.round(L_arr, 3)}")

    # TRAINING AND EVALUATION OF NETWORKS
    # output_bias_init is not here: it is the mean target of each training subset, set in scaling_one_seed
    model_params = dict(input_dim=data.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH)
    training_params = dict(lr=LR, lr_patience=LR_PATIENCE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           max_iter=MAX_ITER)

    # One file per seed, saved as soon as the seed is done (Results/Scaling_test/scaling_seed<seed>.pkl). A seed that is
    # already saved with the same settings is loaded instead of trained again, so seeds can be added later.
    settings = dict(fractions=FRACTIONS, L_arr=[float(L) for L in L_arr], model_params=model_params,
                    training_params=training_params, input_norm=INPUT_NORM, train_version=TRAIN_VERSION,
                    eval_once_per_all_events_epoch=True)
    all_results = [] # One list of networks per seed
    for seed in SEEDS:
        file_name = os.path.join(RESULTS_DIR, f"scaling{ending}_seed{seed}.pkl")
        model_dir = os.path.join(RESULTS_DIR, f"scaling{ending}_models") # every trained network, as a .pt file
        saved = None
        if os.path.exists(file_name):
            saved = load_results(file_name)
        if saved is not None and saved["settings"] == settings:
            print(f"Seed {seed} is already done: {file_name}")
            results = saved["results"]
        else:
            results = scaling_one_seed(FRACTIONS, L_arr, train_data, val_data, test_data, model_params, training_params,
                                       INPUT_NORM, seed, MAX_CORES, model_dir)
            save_results(dict(settings=settings, results=results), file_name)
        print_scaling_summary(results)
        all_results.append(results)

    # PLOTTING
    n_events = collect_scaling(all_results, "n_train", None)[0] # The number of training events of every fraction
    plot_vs_events(n_events, all_results, L_arr, L_RATIOS, "MSE", "MSE", PLOT_DIR, f"mse_vs_events{ending}.pdf")
    plot_vs_events(n_events, all_results, L_arr, L_RATIOS, "MAE", "MAE", PLOT_DIR, f"mae_vs_events{ending}.pdf")
    plot_used_slope_vs_events(n_events, all_results, L_arr, L_RATIOS, L_data_all, PLOT_DIR,
                              f"used_slope_vs_events{ending}.pdf")
    # The test errors of the networks of every subset, one plot per subset. Seed 0, as the other plots of single networks
    for i in range(len(FRACTIONS)):
        plot_errors_one_fraction(all_results[0], FRACTIONS[i], L_arr, L_RATIOS, test_data[1].numpy(), PLOT_DIR,
                                 f"errors_{n_events[i]}_events{ending}.pdf")
    # The mean error against the true |M|^2, one plot per subset: where does the bound cut the peaks?
    for i in range(len(FRACTIONS)):
        plot_bias_vs_truth(all_results, FRACTIONS[i], L_arr, L_RATIOS, test_data[1].numpy(), PLOT_DIR,
                           f"bias_vs_truth_{n_events[i]}_events{ending}.pdf")
    return

# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
