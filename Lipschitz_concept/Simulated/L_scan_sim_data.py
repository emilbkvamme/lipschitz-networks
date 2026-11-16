# WHAT: Using the Lipschitz network on simulation data. Here using qq->Zg process.
# WHY: To understand the impact of the Lipschitz constraint on the network's performance on real data.
# IDEA: Evaluate its performance on the simulated data and compare to the unconstrained network. Doing a scan in the lipshcitz constant L of the network, evaluating whether we can determine the L from the MSE or MAE of the network output as a function of L.

# HOW:
    # 1. Load simulation data. Clean it, transform to a spherical coordinate system (or logarithmic scale), make dataloader objects ready for training and evaluation.
    # 2. Train and evaluate the Lipschitz network and the unconstrained network on the data, for a range of L values. Save the results.

# ======== IMPORTS ============#
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Utils"))   # the Utils folder, with train_eval.py
from train_eval import fit, fit_many

#========= DATA HANDLING ============#
# Data cleaning is less relevant for simulation data. Also, we want the Lipschitz networks to be robust against bad data.
def data_cleaning_sim_data(data):
    return
def data_transform_cartesian_to_spherical(data):
    return
def data_transform_logarithmic(data):
    return
# SHOULD ADD THE OPPOSITE TRANSFORMATIONS AS WELL

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

#========= NETWORKS AND TRAINING ============#

def L_scan_sim_data(L_arr, data, training_params, device):
    return

#========= PLOTS ============#

def plot_scan_MAE():
    return
def plot_scan_MSE():
    return
def plot_scan_selected_fits():
    return
def plot_training_losses():
    return


def main():
    # DEFINING PATHS AND PARAMETERS
    L_arr = np.logspace(-1, 2, 10) # The Lipschitz constants to scan over
    path_to_sim_data = "zg.npy" # Path to the simulation data

    LR = 3e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 20000
    MAX_ITER = 500
    OPTMIZER = "Adam"
    DEPTH = 3
    HIDDEN_DIM = 64

    # LOAD SIMULATION DATA
    data = np.load(path_to_sim_data, allow_pickle=True)
    target = data[:, -1] # Last variable is the |M|^2
    data = data[:, :-1] # The rest are input features


    
    # TRANSFORMATION OF DATA
    # Log10-scaling test
    log_target = np.log10(target)
    log_data = np.log10(data)
    
    # TRAIN-VALIDATION-TEST SPLIT
    train_data, val_data, test_data = split_sim_data(data, target)

    # TRAINING AND EVALUATION OF NETWORKS

    # PLOTTING
    plot_training_losses()
    plot_scan_MAE()
    plot_scan_MSE()
    plot_scan_selected_fits()
    return

if __name__ == "__main__":
    main()