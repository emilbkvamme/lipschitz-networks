# WHAT: Using the Lipschitz network on simulation data. Here using qq->Zg process. Cartesian inputs, l1 norm, L around L_data.
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
import sys
import numpy as np

# The functions for the data, the scan and the plots are in Simulation_utils.py (in the Simulated folder, one up),
# shared by all the tests on the simulated data. Importing it also sets the plot style of the report.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Simulation_utils import (DATA_PATH, plot_dir, results_dir, get_simulated_data, drop_constant_columns,
                              split_sim_data, standardize, lipschitz_bound_estimate_of_data, L_scan_all_seeds, collect,
                              plot_scan, plot_training_losses, plot_selected_fits, plot_predictions)

TEST_NAME = "L_scan_cartesian_l1"   # the name of the plot and result folders of this test
INPUT_NORM = "l1"                   # the norm of the network inputs (the default of LipschitzModel)

# True: a trial run with only the smallest and the largest L and the unconstrained network, one seed and no early
# stopping, to see after how many epochs the validation loss stops improving. False: the full scan.
TRIAL = False


def main():
    # DEFINING PATHS AND PARAMETERS
    L_RATIOS = np.logspace(-1.5, 0.5, 9) # L / L_data from 0.03 to 3, with a point at exactly 1
    SEEDS = [0, 1, 2, 3, 4] # One full scan per seed, for the spread between trainings
    PLOT_DIR = plot_dir(TEST_NAME)

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

    # LOAD SIMULATION DATA: the 16 Cartesian momentum columns, and |M|^2
    data, target = get_simulated_data(DATA_PATH)

    # TRANSFORMATION OF DATA
    # |M|^2 runs from 0.17 to 700, so the networks learn log10 |M|^2. The inputs are not logged: the momentum
    # components are often 0 or negative.
    log_target = np.log10(target)
    data = drop_constant_columns(data) # 16 -> 12 input variables

    # TRAIN-VALIDATION-TEST SPLIT, then standardized inputs
    train_data, val_data, test_data = split_sim_data(data, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)

    # L OF THE DATA, and the L values of the scan
    L_data = lipschitz_bound_estimate_of_data(train_data[0], train_data[1], norm=1) # p = 1: the l1 distance, as the networks
    L_arr = L_RATIOS * L_data
    print(f"L_data = {L_data:.3f}, scanning L = {np.round(L_arr, 3)}")

    # TRAINING AND EVALUATION OF NETWORKS
    # The settings both models take. The output starts at the mean training target, which makes the Lipschitz
    # networks train much faster (see Lipschitz_testing.ipynb).
    model_params = dict(input_dim=data.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH,
                        output_bias_init=float(train_data[1].mean()))
    training_params = dict(lr=LR, lr_patience=LR_PATIENCE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           max_iter=MAX_ITER)

    # One scan per seed, saved as soon as it is done (Results/L_scan_cartesian_l1/L_scan_seed<seed>.pkl). A seed that
    # is already saved with the same settings is loaded instead of trained again.
    all_results = L_scan_all_seeds(L_arr, train_data, val_data, test_data, model_params, training_params, INPUT_NORM,
                                   SEEDS, os.path.join(results_dir(TEST_NAME), f"L_scan{ending}"))

    # PLOTTING
    plot_scan(L_RATIOS, collect(all_results, "MSE"), "MSE", PLOT_DIR, f"mse_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "MAE"), "MAE", PLOT_DIR, f"mae_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "max_slope") / L_data, r"Used slope / $L_\mathrm{data}$",
              PLOT_DIR, f"used_slope_vs_L{ending}.pdf", show_bound=True)
    plot_training_losses(L_RATIOS, all_results[0], PLOT_DIR, f"validation_losses{ending}.pdf")
    if not TRIAL: # The trial has only two L values
        plot_selected_fits(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, "test_errors_selected.pdf")
    plot_predictions(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, f"predictions{ending}.pdf")
    return

# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
