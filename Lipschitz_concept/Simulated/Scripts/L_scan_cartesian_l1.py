# WHAT: Using the Lipschitz network on simulation data. Here using qq->Zg process.
# WHY: To understand the impact of the Lipschitz constraint on the network's performance on real data.
# IDEA: Evaluate its performance on the simulated data and compare to the unconstrained network. Doing a scan in the lipshcitz constant L of the network, evaluating whether we can determine the L from the MSE or MAE of the network output as a function of L.

# HOW:
    # 1. Load simulation data. Drop the inputs that are the same in every event, take log10 of the target |M|^2,
    #    split the events 60-20-20 and standardize the inputs with the mean and std of the training events.
    # 2. L_data: the smallest L that fits all training events exactly. The scan runs over L / L_data, as the toy
    #    studies run over L / L_true.
    # 3. Train one Lipschitz network per L and one unconstrained network, for several seeds. Each seed is saved as soon
    #    as it is done. Adam, with the learning rate halved whenever the validation loss stops improving.
    #    For every network: the test MSE and MAE, and the largest slope it actually uses.
    # 4. Plots: MSE, MAE and used slope against L / L_data, the validation losses, and the test errors of a few networks.

# ======== IMPORTS ============#
import os
import pickle
import sys
import matplotlib
matplotlib.use("Agg") # No windows: the figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

SIM_DIR = os.path.dirname(os.path.abspath(__file__))       # the Simulated folder, with the data
sys.path.insert(0, os.path.join(SIM_DIR, "..", "Utils"))    # the Utils folder, with models.py and train_eval.py
sys.path.insert(0, os.path.join(SIM_DIR, "..", "Toy"))      # the Toy folder, with the plot style of the report

from models import LipschitzModel, UnconstrainedModel
from train_eval import fit_many
from Toy_utils import REPORT_STYLE

RESULTS_DIR = os.path.join(SIM_DIR, "Results")
PLOT_DIR = os.path.join(SIM_DIR, "Plots")

# True: a trial run with only the smallest and the largest L and the unconstrained network, one seed and no early
# stopping, to see after how many epochs the validation loss stops improving. False: the full scan.
TRIAL = False

# Plot format, as in the toy studies
plt.rcParams.update(REPORT_STYLE)
PANEL_SIZE = (3.1, 2.8)
C_NET = "tab:orange"
C_UNCONSTRAINED = "#9a9893"

#========= DATA HANDLING ============#
# Data cleaning is less relevant for simulation data. Also, we want the Lipschitz networks to be robust against bad data.
def data_cleaning_sim_data(data):
    return
def data_transform_cartesian_to_spherical(data):
    return
def data_transform_logarithmic(data):
    return
# SHOULD ADD THE OPPOSITE TRANSFORMATIONS AS WELL

def drop_constant_columns(inputs):
    # The px and py of the two incoming partons are 0 in every event
    varies = inputs.std(axis=0) > 0
    return inputs[:, varies]

def split_sim_data(inputs, target, seed=0):
    # inputs: (events, variables), target: (events,). The network works in float32, the data is float64.
    # The target gets shape (events, 1), the same as the network output, so that MSELoss compares event by event.
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)

    # Random 60-20-20 split of the events, the same every run (fixed seed)
    n_events = len(X)
    n_train, n_val = int(0.6 * n_events), int(0.2 * n_events)
    order = torch.randperm(n_events, generator=torch.Generator().manual_seed(seed))
    train, val, test = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

    # (inputs, targets) tuples, as fit() in train_eval.py takes them; fit() makes the DataLoaders itself
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])

def standardize(train_data, val_data, test_data):
    # Every input variable gets mean 0 and standard deviation 1, with the mean and std of the TRAINING events only:
    # the validation and test events must not influence the training.
    # With all inputs on the same scale the bound max_i |df/dx_i| <= L treats them alike, and L is in units of
    # the target (log10 |M|^2) per standard deviation of an input.
    mean, std = train_data[0].mean(dim=0), train_data[0].std(dim=0)
    return [((X - mean) / std, y) for X, y in (train_data, val_data, test_data)]

def lipschitz_constant_of_data(X, y, n_neighbours=10):
    """
    The smallest L that fits all the events exactly: the steepest slope |y_a - y_b| / ||x_a - x_b||_1 between two
    events. The target has no noise, so every pair of events gives a lower limit on L. The steepest pairs lie close
    together, so only the n_neighbours nearest neighbours of every event are checked (about a minute).
    """
    X = X.numpy()
    y = y.numpy().reshape(-1, 1)
    distances, neighbours = cKDTree(X).query(X, k=n_neighbours + 1, p=1)  # p=1: the l1 distance, the norm of the networks
    distances, neighbours = distances[:, 1:], neighbours[:, 1:]           # the nearest "neighbour" is the event itself
    slopes = np.abs(y[neighbours, 0] - y) / distances                     # one row per event, one column per neighbour
    return float(slopes.max())

#========= NETWORKS AND TRAINING ============#

def used_slope(model, X):
    """
    The slope a trained network uses at the events X: for every event the largest |df/dx_i|, which is what the l1
    bound of the Lipschitz networks limits to L. Returns the largest value over the events, and the 99.9th percentile
    (less sensitive to a single spike of the unconstrained network).
    """
    X = X.clone().requires_grad_(True)
    output = model(X).sum() # Summed over the events: one backward pass then gives the gradient at every event
    gradient = torch.autograd.grad(output, X)[0]
    slopes = gradient.abs().max(dim=1).values.numpy()
    return float(slopes.max()), float(np.percentile(slopes, 99.9))

def L_scan_sim_data(L_arr, train_data, val_data, test_data, model_params, training_params, seed):
    """
    Running a scan of Lipschitz constants on the simulation data, for one seed: one Lipschitz network per L and one
    unconstrained network as the baseline, all trained in parallel. They all get the same data, settings and seed,
    so the only difference between them is the constraint.
    Returns one dictionary per network (the Lipschitz networks in the order of L_arr, the unconstrained one last) with
    L (None for the unconstrained network), the losses per epoch, the test predictions, the trained weights, the test
    MSE and MAE, and the largest slope the network uses on the test events.
    """
    data = dict(trn_data=train_data, val_data=val_data, test_data=test_data)
    jobs = []
    for L in L_arr:
        jobs.append(dict(model_cls=LipschitzModel, model_kwargs=dict(**model_params, lipschitz_bound=float(L)),
                         seed=seed, **data, **training_params))
    jobs.append(dict(model_cls=UnconstrainedModel, model_kwargs=model_params, seed=seed, **data, **training_params))

    print(f"Seed {seed}: training {len(jobs)} networks")
    results = fit_many(jobs)

    y_test = test_data[1].numpy()
    for job, result in zip(jobs, results):
        model = job["model_cls"](**job["model_kwargs"]) # The trained network, rebuilt from its weights
        model.load_state_dict(result["state_dict"])

        errors = result["predictions"] - y_test # In units of the target, log10 |M|^2
        result["L"] = job["model_kwargs"].get("lipschitz_bound") # None for the unconstrained network
        result["MSE"] = float(np.mean(errors**2))
        result["MAE"] = float(np.mean(np.abs(errors)))
        result["max_slope"], result["slope_999"] = used_slope(model, test_data[0])
    return results

def print_summary(results, num_epochs):
    """
    One line per network: the test errors, the largest slope it uses, and how the training went.
    "Within 1%" is the first epoch where the validation loss is within 1% of its best value: about where it stops improving.
    """
    print(f"{'L':>8} {'MSE':>10} {'MAE':>8} {'slope':>7} {'epochs':>7} {'best':>5} {'within 1%':>10}")
    for result in results:
        name = "none" if result["L"] is None else f"{result['L']:.3g}"
        val_losses = np.array(result["val_losses"])
        best = int(np.argmin(val_losses)) + 1
        within = int(np.argmax(val_losses <= 1.01 * val_losses.min())) + 1
        # train() only goes back to the best weights when early stopping ends the training
        cap = "  hit the cap: last weights, not the best" if len(val_losses) == num_epochs else ""
        print(f"{name:>8} {result['MSE']:10.3e} {result['MAE']:8.4f} {result['max_slope']:7.3f} "
              f"{len(val_losses):7d} {best:5d} {within:10d}{cap}")

def save_results(results, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as file:
        pickle.dump(results, file)
    print(f"Saved the results to {file_name}")

def load_results(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)

def collect(all_results, measure):
    """One measure (e.g. "MSE") of every network and seed: one row per seed, the unconstrained network in the last column."""
    values = []
    for results in all_results:
        values.append([result[measure] for result in results])
    return np.array(values)

#========= PLOTS ============#

def plot_scan(L_ratios, values, y_label, file_name, show_bound=False):
    """
    One measure against L / L_data. values has one row per seed and one column per network (from collect).
    Lipschitz networks: the mean over the seeds, with the lowest and highest seed as error bar.
    Unconstrained network: a dashed line at its mean. show_bound adds the bound "slope = L" (for the used slope).
    """
    lipschitz = values[:, :-1]
    unconstrained = values[:, -1]
    mean = lipschitz.mean(axis=0)
    error = [mean - lipschitz.min(axis=0), lipschitz.max(axis=0) - mean]

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.errorbar(L_ratios, mean, yerr=error, fmt="o-", color=C_NET, label="Lipschitz")
    ax.axhline(unconstrained.mean(), color=C_UNCONSTRAINED, ls="--", label="Unconstrained")
    if show_bound:
        ax.plot(L_ratios, L_ratios, color="black", ls=":", label="Bound")
    ax.axvline(1, color="black", lw=0.8) # L = L_data
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g")) # 0.1 and 1 instead of 10^-1 and 10^0
    ax.set_xlabel(r"$L / L_\mathrm{data}$")
    ax.set_ylabel(y_label)
    if show_bound:
        # The networks follow the bound at small L and level off at large L, so the lower right corner stays free
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.legend(frameon=False)
    fig.savefig(os.path.join(PLOT_DIR, file_name))
    plt.close(fig)

def plot_training_losses(L_ratios, results, file_name):
    """
    The validation loss per epoch of every network of one seed: the Lipschitz networks in shades of orange,
    from light (smallest L) to dark (largest L), the unconstrained network in grey.
    """
    shades = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_ratios)))
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for i in range(len(L_ratios)):
        label = None
        if i == 0 or i == len(L_ratios) - 1: # Only the smallest and the largest L in the legend
            label = r"$L / L_\mathrm{data}$" + f" = {L_ratios[i]:.2g}"
        ax.plot(range(1, len(results[i]["val_losses"]) + 1), results[i]["val_losses"], color=shades[i], label=label)
    ax.plot(range(1, len(results[-1]["val_losses"]) + 1), results[-1]["val_losses"], color=C_UNCONSTRAINED,
            label="Unconstrained")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(PLOT_DIR, file_name))
    plt.close(fig)

def plot_selected_fits(L_ratios, results, y_test, file_name):
    """
    Prediction minus truth on the test events, for the smallest L, L = L_data (the ratio closest to 1), the largest L
    and the unconstrained network of one seed, as step histograms.
    """
    selected = [0, int(np.argmin(np.abs(np.log(L_ratios)))), len(L_ratios) - 1]
    shades = plt.cm.Oranges(np.linspace(0.4, 0.95, len(selected)))
    bins = np.linspace(-1, 1, 101)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for shade, i in zip(shades, selected):
        errors = results[i]["predictions"][:, 0] - y_test[:, 0]
        ax.hist(errors, bins=bins, histtype="step", color=shade, label=f"{L_ratios[i]:.2g}" + r" $L_\mathrm{data}$")
    errors = results[-1]["predictions"][:, 0] - y_test[:, 0]
    ax.hist(errors, bins=bins, histtype="step", color=C_UNCONSTRAINED, label="Unconstrained")
    ax.set_yscale("log")
    ax.set_xlabel(r"Prediction $-$ truth ($\log_{10}|M|^2$)")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2) # Below: the histograms fill the axes
    fig.savefig(os.path.join(PLOT_DIR, file_name))
    plt.close(fig)

def plot_predictions(L_ratios, results, y_test, file_name):
    """
    The distribution of the predicted log10 |M|^2 on the test events, with the true distribution on top: every L of
    one seed in shades of orange, from light (smallest L) to dark (largest L), the unconstrained network in grey and
    the truth dashed black. Shows which part of the true distribution the networks reach, and where they miss.
    """
    shades = plt.cm.Oranges(np.linspace(0.4, 0.95, len(L_ratios)))
    # The true log10 |M|^2 runs from -0.78 to 2.84, the predictions a bit further (-1.6 to 3.3): wide enough for all
    bins = np.linspace(-2, 3.5, 111)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for i in range(len(L_ratios)):
        label = None
        if i == 0 or i == len(L_ratios) - 1: # Only the smallest and the largest L in the legend
            label = f"{L_ratios[i]:.2g}" + r" $L_\mathrm{data}$"
        ax.hist(results[i]["predictions"][:, 0], bins=bins, histtype="step", color=shades[i], label=label)
    ax.hist(results[-1]["predictions"][:, 0], bins=bins, histtype="step", color=C_UNCONSTRAINED, label="Unconstrained")
    ax.hist(y_test[:, 0], bins=bins, histtype="step", color="black", ls="--", label="Truth") # Last: on top
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}|M|^2$")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2) # Below, as in plot_selected_fits
    fig.savefig(os.path.join(PLOT_DIR, file_name))
    plt.close(fig)


def main():
    # DEFINING PATHS AND PARAMETERS
    L_RATIOS = np.logspace(-1.5, 0.5, 9) # L / L_data from 0.03 to 3, with a point at exactly 1
    SEEDS = [0, 1, 2, 3, 4] # One full scan per seed, for the spread between trainings
    path_to_sim_data = os.path.join(SIM_DIR, "zg.npy") # Path to the simulation data

    LR = 3e-3 # The learning rate at the start
    LR_PATIENCE = 5 # Halve the learning rate when the validation loss has not improved for more than this many epochs
    BATCH_SIZE = 1024 # 444 steps per epoch, about 2.5 s per epoch on one core (batch 64: about 6 s)
    NUM_EPOCHS = 200 # The cap: early stopping normally ends the training before
    MAX_ITER = 20 # Early stopping patience: stop when the validation loss has not improved for this many epochs
    DEPTH = 3
    HIDDEN_DIM = 64
    # Optimizer: fit() always uses Adam

    if TRIAL:
        L_RATIOS = L_RATIOS[[0, -1]] # The smallest and the largest L
        SEEDS = [0]
        NUM_EPOCHS = 300
        MAX_ITER = NUM_EPOCHS # No early stopping: the whole validation curve
    ending = "_trial" if TRIAL else "" # The trial run gets its own result and plot files

    # LOAD SIMULATION DATA
    data = np.load(path_to_sim_data, allow_pickle=True)
    target = data[:, -1] # Last variable is the |M|^2
    data = data[:, :-1] # The rest are input features

    # TRANSFORMATION OF DATA
    # |M|^2 runs from 0.17 to 700, so the networks learn log10 |M|^2. The inputs are not logged: the momentum
    # components are often 0 or negative.
    log_target = np.log10(target)
    data = drop_constant_columns(data) # 16 -> 12 input variables

    # TRAIN-VALIDATION-TEST SPLIT, then standardized inputs
    train_data, val_data, test_data = split_sim_data(data, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)

    # L OF THE DATA, and the L values of the scan
    L_data = lipschitz_constant_of_data(train_data[0], train_data[1])
    L_arr = L_RATIOS * L_data
    print(f"L_data = {L_data:.3f}, scanning L = {np.round(L_arr, 3)}")

    # TRAINING AND EVALUATION OF NETWORKS
    # The settings both models take. The output starts at the mean training target, which makes the Lipschitz
    # networks train much faster (see Lipschitz_testing.ipynb).
    model_params = dict(input_dim=data.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH,
                        output_bias_init=float(train_data[1].mean()))
    training_params = dict(lr=LR, lr_patience=LR_PATIENCE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           max_iter=MAX_ITER)
    settings = dict(L_arr=[float(L) for L in L_arr], model_params=model_params, training_params=training_params)

    # One scan per seed, saved as soon as it is done: a crash only loses the seed that was running.
    # A seed that is already saved with the same settings is loaded instead of trained again.
    all_results = [] # One list of networks per seed
    for seed in SEEDS:
        file_name = os.path.join(RESULTS_DIR, f"L_scan{ending}_seed{seed}.pkl")
        saved = None
        if os.path.exists(file_name):
            saved = load_results(file_name)
        if saved is not None and saved["settings"] == settings:
            print(f"Seed {seed} is already done: {file_name}")
            results = saved["results"]
        else:
            results = L_scan_sim_data(L_arr, train_data, val_data, test_data, model_params, training_params, seed)
            save_results(dict(settings=settings, results=results), file_name)
        print_summary(results, NUM_EPOCHS)
        all_results.append(results)

    # PLOTTING
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_scan(L_RATIOS, collect(all_results, "MSE"), "MSE", f"mse_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "MAE"), "MAE", f"mae_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "max_slope") / L_data, r"Used slope / $L_\mathrm{data}$",
              f"used_slope_vs_L{ending}.pdf", show_bound=True)
    plot_training_losses(L_RATIOS, all_results[0], f"validation_losses{ending}.pdf")
    if not TRIAL: # The trial has only two L values
        plot_selected_fits(L_RATIOS, all_results[0], test_data[1].numpy(), "test_errors_selected.pdf")
    plot_predictions(L_RATIOS, all_results[0], test_data[1].numpy(), f"predictions{ending}.pdf")
    return

# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
