# IDEA: Many of the iterations on the simulated data are going to be verys similar, so it is useful to have a script with utilities that can be imported and put together

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg") # No windows: the figures are only saved to files (also works on the cluster, without a screen)
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
import torch

SIM_DIR = os.path.dirname(os.path.abspath(__file__))       # the Simulated folder, with the data
sys.path.insert(0, os.path.join(SIM_DIR, "..", "Utils"))    # the Utils folder, with models.py and train_eval.py

from models import LipschitzModel, UnconstrainedModel
from train_eval import fit_many
from plot_style import REPORT_STYLE, PANEL_SIZE

# Plot format of the report, for every script that imports these utilities
plt.rcParams.update(REPORT_STYLE)
C_NET = "tab:orange"            # the Lipschitz networks
C_UNCONSTRAINED = "#9a9893"     # the unconstrained network

#======= FOLDERS ===========
# The test scripts are in Simulated/Scripts. Every test writes to Plots/<test> and Results/<test> inside the Simulated
# folder, no matter which folder the script is run from.
DATA_PATH = os.path.join(SIM_DIR, "zg.npy") # the simulated qq->Zg events

def plot_dir(test):
    """The plot folder of one test, e.g. plot_dir("L_scan_cartesian_l2"). save_figure makes it if it does not exist."""
    return os.path.join(SIM_DIR, "Plots", test)

def results_dir(test):
    """The result folder of one test, e.g. results_dir("L_scan_cartesian_l2"). save_results makes it if it does not exist."""
    return os.path.join(SIM_DIR, "Results", test)

#======= DATA CLEANING ===========
def get_simulated_data(path_to_sim_data):
    assert os.path.exists(path_to_sim_data), f"Path {path_to_sim_data} does not exist. Please check the path."

    data_raw = np.load(path_to_sim_data)

    data, target = data_raw[:, :-1], data_raw[:, -1] # Extracting the |M|^2 values as the target, and the rest of the data as the input data
    return data, target

def drop_constant_columns(inputs):
    # In cartesian coord: the two incoming particles have px=py=0 in every event. Constant columns can be dropped
    varies = inputs.std(axis=0) > 0
    return inputs[:, varies]

def split_sim_data(inputs, target, seed=0):
    # Turning the data into split torch tensors for training
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)

    # Random 60-20-20 split of the events, the same every run (fixed seed)
    n_events = len(X)
    n_train, n_val = int(0.6 * n_events), int(0.2 * n_events)
    order = torch.randperm(n_events, generator=torch.Generator().manual_seed(seed))
    train, val, test = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])

def standardize(train_data, val_data, test_data):
    mean, std = train_data[0].mean(dim=0), train_data[0].std(dim=0)
    return [((X - mean) / std, y) for X, y in (train_data, val_data, test_data)]

def cartesian_to_pt_angles(data):
    """
    Transforming the four particles in cartesian coordinates to a transverse momentum cylindrical coordinate system.
    Input:
        data: numpy array of shape (n_events, 16) from get_simulated_data, the four particles (q1, q2, Z, g) in cartesian coordinates (E, px, py, pz)
    Output:
        new_data: numpy array of shape (n_events, 10), the columns E_1, E_2, E_Z, pT_Z, phi_Z, y_Z, E_g, pT_g, phi_g, y_g
    Numpy, not torch: the data is still float64 here, so the logs and angles are calculated at full precision.
    """
    q1, q2, Z, g = data[:, 0:4], data[:, 4:8], data[:, 8:12], data[:, 12:] # The four particles in cartesian coordinates

    # Incoming partons move only along the z-axis, all information they carry is their energy
    E1, E2 = q1[:, 0], q2[:, 0]

    # The Z-boson
    E_Z = Z[:, 0]
    pT_Z = np.sqrt(Z[:, 1]**2 + Z[:, 2]**2) # sqrt(px^2 + py^2)
    phi_Z = np.arctan2(Z[:, 2], Z[:, 1]) # arctan2(py, px)
    y_Z = 0.5 * np.log((Z[:, 0] + Z[:, 3]) / (Z[:, 0] - Z[:, 3])) # 0.5 * log((E + pz) / (E - pz))

    # The gluon
    E_g = g[:, 0]
    pT_g = np.sqrt(g[:, 1]**2 + g[:, 2]**2) # sqrt(px^2 + py^2)
    phi_g = np.arctan2(g[:, 2], g[:, 1]) # arctan2(py, px)
    y_g = 0.5 * np.log((g[:, 0] + g[:, 3]) / (g[:, 0] - g[:, 3])) # 0.5 * log((E + pz) / (E - pz))

    # Stacking the new features together: one row per event, one column per variable
    new_data = np.stack([E1, E2, E_Z, pT_Z, phi_Z, y_Z, E_g, pT_g, phi_g, y_g], axis=1)
    return new_data


def lipschitz_bound_estimate_of_data(X, y, n_neighbours=10, norm=1):
    """ Calculating a lower limit of the Lipschitz constant of the data by calculating local slopes between local neighbourhoods of points in the data. The slopes are defined as |y_a - y_b| / ||x_a - x_b||_l between two events. The largest slope of each cluster is taken as a local Lipschitz constant. The complete Lipschitz constant of the underlying data is the L>=max(L_local)
    Inputs:
        X: torch tensor of shape (n_events, n_features)
        y: torch tensor of shape (n_events, 1)
        n_neighbours: int, number of nearest neighbours to consider for each event
        norm: the p of the p-norm for the distances: 1 (Manhattan, the default: the norm of the l1 networks), 2 (Euclidean) or np.inf (the max norm). cKDTree needs np.inf, not a string.
    Outputs:
        L: float, the estimated Lipschitz constant of the data. The maximum local slope.
    """
    X = X.numpy()
    y = y.numpy().reshape(-1, 1)

    distances, neighbours = cKDTree(X).query(X, k=n_neighbours + 1, p=norm) # Extracts the 10 nearest neighbours of each event in an array [event, neighbours]

    distances, neighbours = distances[:, 1:], neighbours[:, 1:]   # The algorithm returns the event itself as its own nearest neighbour, so it mustbe omitted to avoid division by zero in the slopes.
    slopes = np.abs(y[neighbours, 0] - y) / distances                 # one row per event, one column per neighbour,  

    max_slope = float(slopes.max())
    return max_slope

#======= DATA SCANNING ===========
def used_slope(model, X, input_norm):
    """
    The slope a trained network uses at the events X, measured the way the bound of the Lipschitz networks limits it:
        input_norm "l1": |f(a) - f(b)| <= L ||a - b||_1, the same as every |df/dx_i| <= L: the largest |df/dx_i|
        input_norm "l2": |f(a) - f(b)| <= L ||a - b||_2, the same as ||grad f||_2 <= L: the length of the gradient
    Returns the largest value over the events, and the 99.9th percentile (less sensitive to a single spike of the
    unconstrained network).
    """
    X = X.clone().requires_grad_(True)
    output = model(X).sum() # Summed over the events: one backward pass then gives the gradient at every event
    gradient = torch.autograd.grad(output, X)[0] # (events, inputs)
    if input_norm == "l1":
        slopes = gradient.abs().max(dim=1).values
    elif input_norm == "l2":
        slopes = gradient.norm(dim=1)
    else:
        raise ValueError(f"input_norm ({input_norm}) must be 'l1' or 'l2'")
    slopes = slopes.numpy()
    return float(slopes.max()), float(np.percentile(slopes, 99.9))

def L_scan_sim_data(L_arr, train_data, val_data, test_data, model_params, training_params, input_norm, seed,
                    n_workers=None):
    """
    Running a scan of Lipschitz constants on the simulation data, for one seed: one Lipschitz network per L and one
    unconstrained network as the baseline, all trained in parallel. They all get the same data, settings and seed,
    so the only difference between them is the constraint.
    input_norm ("l1" or "l2") is the norm of the inputs the Lipschitz bound is measured in. Only the Lipschitz networks
    take it; the used slope of every network is measured in it. n_workers: networks trained at the same time
    (None: the default of fit_many, half the cores).
    Returns one dictionary per network (the Lipschitz networks in the order of L_arr, the unconstrained one last) with
    L (None for the unconstrained network), the losses per epoch, the test predictions, the trained weights, the test
    MSE and MAE, and the largest slope the network uses on the test events.
    """
    data = dict(trn_data=train_data, val_data=val_data, test_data=test_data)
    jobs = []
    for L in L_arr:
        lipschitz_kwargs = dict(**model_params, lipschitz_bound=float(L), input_norm=input_norm)
        jobs.append(dict(model_cls=LipschitzModel, model_kwargs=lipschitz_kwargs, seed=seed, **data, **training_params))
    jobs.append(dict(model_cls=UnconstrainedModel, model_kwargs=model_params, seed=seed, **data, **training_params))

    print(f"Seed {seed}: training {len(jobs)} networks")
    results = fit_many(jobs, n_workers=n_workers)

    y_test = test_data[1].numpy()
    for job, result in zip(jobs, results):
        model = job["model_cls"](**job["model_kwargs"]) # The trained network, rebuilt from its weights
        model.load_state_dict(result["state_dict"])

        errors = result["predictions"] - y_test # In units of the target, log10 |M|^2
        result["L"] = job["model_kwargs"].get("lipschitz_bound") # None for the unconstrained network
        result["MSE"] = float(np.mean(errors**2))
        result["MAE"] = float(np.mean(np.abs(errors)))
        result["max_slope"], result["slope_999"] = used_slope(model, test_data[0], input_norm)
    return results

def L_scan_all_seeds(L_arr, train_data, val_data, test_data, model_params, training_params, input_norm, seeds,
                     file_start, n_workers=None):
    """
    The L scan of L_scan_sim_data for every seed. Each seed is saved as soon as it is done, to
    file_start + "_seed<seed>.pkl": a crash only loses the seed that was running. A seed that is already saved with the
    same settings is loaded instead of trained again, so more seeds can be added later without training the old ones again.
    The settings do NOT include the data itself, so every basis (or other change of the data) needs its own file_start.
    Returns one list of networks per seed, as L_scan_sim_data returns them: the input of collect.
    """
    settings = dict(L_arr=[float(L) for L in L_arr], model_params=model_params, training_params=training_params,
                    input_norm=input_norm)

    all_results = [] # One list of networks per seed
    for seed in seeds:
        file_name = f"{file_start}_seed{seed}.pkl"
        saved = None
        if os.path.exists(file_name):
            saved = load_results(file_name)
        if saved is not None and saved["settings"] == settings:
            print(f"Seed {seed} is already done: {file_name}")
            results = saved["results"]
        else:
            results = L_scan_sim_data(L_arr, train_data, val_data, test_data, model_params, training_params, input_norm,
                                      seed, n_workers)
            save_results(dict(settings=settings, results=results), file_name)
        print_summary(results, training_params["num_epochs"])
        all_results.append(results)
    return all_results

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

def collect(all_results, measure):
    """One measure (e.g. "MSE") of every network and seed: one row per seed, the unconstrained network in the last column."""
    values = []
    for results in all_results:
        values.append([result[measure] for result in results])
    return np.array(values)

#======= SAVE AND LOAD ===========
def save_results(results, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as file:
        pickle.dump(results, file)
    print(f"Saved the results to {file_name}")

def load_results(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)

#======= DATA PLOTTING ===========
# Every plot function takes plot_dir, the folder of one test (e.g. Plots/Cartesian), so the tests do not overwrite
# each other's figures.

def save_figure(fig, plot_dir, file_name):
    # Saves the figure in plot_dir (made if it does not exist yet) and closes it, so the figures do not pile up in memory
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, file_name))
    plt.close(fig)

def plot_scan(L_ratios, values, y_label, plot_dir, file_name, show_bound=False):
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
    save_figure(fig, plot_dir, file_name)

def plot_training_losses(L_ratios, results, plot_dir, file_name):
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
    save_figure(fig, plot_dir, file_name)

def plot_selected_fits(L_ratios, results, y_test, plot_dir, file_name):
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
    save_figure(fig, plot_dir, file_name)

def plot_predictions(L_ratios, results, y_test, plot_dir, file_name):
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
    save_figure(fig, plot_dir, file_name)





