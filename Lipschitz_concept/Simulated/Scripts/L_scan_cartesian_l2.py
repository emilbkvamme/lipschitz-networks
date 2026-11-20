# WHAT: L scan on the simulated qq->Zg data. Cartesian inputs, l2 norm, L from just below L_data to far above it.
# WHY: L_data, the steepest slope between neighbouring events, is only a LOWER bound on the L the target needs. The
#      first scan (L_scan_cartesian_l1.py, 0.03 - 3 L_data) looked around it. Here: what happens when L passes the bound
#      by far? Does the error keep falling, level off at the unconstrained network, or is there a minimum?
#      The l2 norm, |f(a) - f(b)| <= L ||a - b||_2, is rotationally invariant: the bound is the same in every direction
#      of the inputs. The l1 norm is not: it allows more slope along the diagonals than along the axes.

# HOW:
    # 1. Load simulation data. Drop the inputs that are the same in every event, take log10 of the target |M|^2,
    #    split the events 60-20-20 and standardize the inputs with the mean and std of the training events.
    # 2. L_data with the l2 distance, the same norm as the networks.
    # 3. Train one Lipschitz network per L (17 values from 0.8 to 25 L_data, evenly spaced on a log scale) and one
    #    unconstrained network, for several seeds. Each seed is saved as soon as it is done.
    # 4. Plots, as in the l1 scan: MSE, MAE and used slope against L / L_data, the validation losses, the test errors of
    #    a few networks, and the distribution of the predictions.

# RUN (from the Scripts folder):  python L_scan_cartesian_l2.py           the full scan
#                                 python L_scan_cartesian_l2.py --quick   a test of a few minutes: 2 L values, 1 seed, 2 epochs

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

TEST_NAME = "L_scan_cartesian_l2"   # the name of the plot and result folders of this test
INPUT_NORM = "l2"                   # the norm of the network inputs: rotationally invariant
QUICK = "--quick" in sys.argv       # a fast test of the whole script: are there any errors before the long run?


def main():
    # DEFINING PATHS AND PARAMETERS
    # L / L_data: 0.8, 0.99, 1.23, ..., 20.2, 25, evenly spaced on a log scale (a factor 1.24 apart). 17 values + the
    # unconstrained network = 18 networks = 3 full rounds of N_WORKERS: with 12 values it would also take 3 rounds.
    L_RATIOS = np.geomspace(0.8, 25, 17)
    # One full scan per seed, for the spread between trainings. About 60-75 min per seed on the Mac (3 rounds of ~20-25 min).
    # More seeds can be added later: the saved seeds are loaded, only the new ones are trained.
    SEEDS = [0, 1, 2]
    N_WORKERS = 6 # Networks trained at the same time: the Mac has 6 fast cores (the 2 slow ones would hold up a round)

    # The same training settings as the l1 scan, so that only the norm and the L range differ
    LR = 3e-3 # The learning rate at the start
    LR_PATIENCE = 5 # Halve the learning rate when the validation loss has not improved for more than this many epochs
    BATCH_SIZE = 1024 # 444 steps per epoch
    NUM_EPOCHS = 200 # The cap: early stopping normally ends the training before
    MAX_ITER = 20 # Early stopping patience: stop when the validation loss has not improved for this many epochs
    DEPTH = 3
    HIDDEN_DIM = 64
    # Optimizer: fit() always uses Adam

    if QUICK:
        L_RATIOS = L_RATIOS[[0, -1]] # The smallest and the largest L
        SEEDS = [0]
        NUM_EPOCHS = 2
        MAX_ITER = 2
    ending = "_quick" if QUICK else "" # The quick run gets its own result and plot files
    PLOT_DIR = plot_dir(TEST_NAME)

    # LOAD SIMULATION DATA: the 16 Cartesian momentum columns, and |M|^2
    data, target = get_simulated_data(DATA_PATH)

    # TRANSFORMATION OF DATA
    # |M|^2 runs from 0.17 to 700, so the networks learn log10 |M|^2
    log_target = np.log10(target)
    data = drop_constant_columns(data) # 16 -> 12 input variables: px and py of the incoming partons are always 0

    # TRAIN-VALIDATION-TEST SPLIT, then standardized inputs
    train_data, val_data, test_data = split_sim_data(data, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)

    # L OF THE DATA, and the L values of the scan
    L_data = lipschitz_bound_estimate_of_data(train_data[0], train_data[1], norm=2) # p = 2: the l2 distance, as the networks
    L_arr = L_RATIOS * L_data
    print(f"L_data = {L_data:.3f} (l2), scanning L = {np.round(L_arr, 3)}")

    # TRAINING AND EVALUATION OF NETWORKS
    # The settings both models take. The output starts at the mean training target, which makes the Lipschitz
    # networks train much faster (see Lipschitz_testing.ipynb).
    model_params = dict(input_dim=data.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH,
                        output_bias_init=float(train_data[1].mean()))
    training_params = dict(lr=LR, lr_patience=LR_PATIENCE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           max_iter=MAX_ITER)

    # One scan per seed, saved as soon as it is done (Results/L_scan_cartesian_l2/L_scan_seed<seed>.pkl). A seed that
    # is already saved with the same settings is loaded instead of trained again.
    all_results = L_scan_all_seeds(L_arr, train_data, val_data, test_data, model_params, training_params, INPUT_NORM,
                                   SEEDS, os.path.join(results_dir(TEST_NAME), f"L_scan{ending}"), n_workers=N_WORKERS)

    # PLOTTING
    plot_scan(L_RATIOS, collect(all_results, "MSE"), "MSE", PLOT_DIR, f"mse_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "MAE"), "MAE", PLOT_DIR, f"mae_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "max_slope") / L_data, r"Used slope / $L_\mathrm{data}$",
              PLOT_DIR, f"used_slope_vs_L{ending}.pdf", show_bound=True)
    plot_training_losses(L_RATIOS, all_results[0], PLOT_DIR, f"validation_losses{ending}.pdf")
    if not QUICK: # The quick run has only two L values
        plot_selected_fits(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, "test_errors_selected.pdf")
    plot_predictions(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, f"predictions{ending}.pdf")
    return

# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
