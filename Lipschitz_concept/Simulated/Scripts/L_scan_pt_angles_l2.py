# WHAT: L scan on the simulated qq->Zg data. pT + angles inputs, l2 norm, L from just below L_data to far above it.
# WHY: The same scan as L_scan_cartesian_l2.py, with the particles described by the coordinates along and around the
#      beam instead of (E, px, py, pz). Does the basis change how close the networks get to the target, and how
#      much of their slope budget they use?
#      In this basis a rotation around the beam only shifts phi_Z and phi_g, and |M|^2 does not depend on phi at all.

# HOW:
    # 1. Load simulation data, transform it to pT + angles (cartesian_to_pt_angles: E_1, E_2, and E, pT, phi, y of the
    #    Z and the gluon, 10 columns), take log10 of the target |M|^2, split the events 60-20-20 and standardize the
    #    inputs with the mean and std of the training events.
    # 2. L_data with the l2 distance, the same norm as the networks. In this basis, so the scan runs over L / L_data
    #    of this basis, and the x-axes mean the same as in the Cartesian scan.
    # 3. Train one Lipschitz network per L (17 values from 0.8 to 25 L_data, evenly spaced on a log scale) and one
    #    unconstrained network, for several seeds. Each seed is saved as soon as it is done.
    # 4. Plots, as in the Cartesian scans.

# RUN (from the Scripts folder):  python L_scan_pt_angles_l2.py           the full scan
#                                 python L_scan_pt_angles_l2.py --quick   a test of a few minutes: 2 L values, 1 seed, 2 epochs
#                                 --basis pt_angles | logpt_cossin (default) | logpt_cossin_costheta   the input basis

# ======== IMPORTS ============#
import os
import sys
import numpy as np

# The functions for the data, the scan and the plots are in Simulation_utils.py (in the Simulated folder, one up),
# shared by all the tests on the simulated data. Importing it also sets the plot style of the report.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Simulation_utils import (max_cores_from_argv, limit_cores, int_from_argv, scaled_lr, get_simulated_data, BASES, basis_from_argv,
                              drop_constant_columns, split_sim_data, standardize, lipschitz_bound_estimate_of_data,
                              L_scan_all_seeds, collect, plot_scan, plot_training_losses, plot_selected_fits,
                              plot_predictions)

# The basis of the inputs: --basis NAME on the command line, a key of BASES in Simulation_utils.py
#   logpt_cossin (default): log pT, (cos phi, sin phi) in [-1, 1], rapidity y
#   logpt_cossin_costheta:  the same with cos(theta) in [-1, 1] instead of y
#   pt_angles:              pT, phi in [-pi, pi], y (the basis of the earlier runs)
BASIS = basis_from_argv(default="logpt_cossin")
# The name of the plot and result folders of this test. The pt_angles basis keeps the folder of the earlier runs.
TEST_NAME = "L_scan_pt_angles_l2" if BASIS == "pt_angles" else f"L_scan_{BASIS}_l2"
INPUT_NORM = "l2"                   # the norm of the network inputs, as in L_scan_cartesian_l2.py
QUICK = "--quick" in sys.argv       # a fast test of the whole script: are there any errors before the long run?


def main():
    # PATHS, for this script only. By default relative to this file, so the script runs from any folder and on any
    # machine. Point them elsewhere to read the data from, or write the plots and results to, another place.
    SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the Simulated folder
    DATA_PATH = os.path.join(SIM_DIR, "zg.npy")                    # the simulated qq->Zg events
    PLOT_DIR = os.path.join(SIM_DIR, "Plots", TEST_NAME)
    RESULTS_DIR = os.path.join(SIM_DIR, "Results", TEST_NAME)      # the result files and the trained networks

    # PARAMETERS: the same as L_scan_cartesian_l2.py, so that only the basis differs
    # L / L_data: 0.8, 0.99, 1.23, ..., 20.2, 25, evenly spaced on a log scale (a factor 1.24 apart). 17 values + the
    # unconstrained network = 18 networks = 3 full rounds of MAX_CORES = 6: with 12 values it would also take 3 rounds.
    L_RATIOS = np.geomspace(0.8, 25, 15)
    # One full scan per seed, for the spread between trainings. About 60-75 min per seed on the Mac (3 rounds of ~20-25 min).
    # More seeds can be added later: the saved seeds are loaded, only the new ones are trained.
    SEEDS = [0, 1, 2]
    # The most cores the run may use (networks trained at the same time). The Mac has 6 fast cores (the 2 slow ones
    # would hold up a round). Change it on the command line with --max-cores N.
    MAX_CORES = max_cores_from_argv(default=6)

    LR_PATIENCE = 5 # Halve the learning rate when the validation loss has not improved for more than this many epochs
    # Batch size: --batch-size N on the command line (default 1024, 445 steps per epoch). The learning rate at the start
    # is 3e-3 at batch 1024 and scales with sqrt(batch size / 1024) (see scaled_lr in train_eval.py).
    BATCH_SIZE = int_from_argv("--batch-size", 1024)
    LR = scaled_lr(3e-3, BATCH_SIZE)
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
    if BATCH_SIZE != 1024:
        ending += f"_bs{BATCH_SIZE}" # other batch sizes get their own result and plot files

    limit_cores(MAX_CORES) # The threads of this process too (the slope calculations, the neighbour search)

    # LOAD SIMULATION DATA: the 16 Cartesian momentum columns, and |M|^2
    data, target = get_simulated_data(DATA_PATH)

    # TRANSFORMATION OF DATA
    # |M|^2 runs from 0.17 to 700, so the networks learn log10 |M|^2
    log_target = np.log10(target)
    data = BASES[BASIS](data) # 16 -> 10 (pt_angles) or 12 input variables (see Sanity_check.py for the checks)
    data = drop_constant_columns(data)  # No constant columns in this basis, but a constant column would give std = 0
                                        # and nan in standardize

    # TRAIN-VALIDATION-TEST SPLIT, then standardized inputs
    train_data, val_data, test_data = split_sim_data(data, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)

    # L OF THE DATA, and the L values of the scan
    L_data = lipschitz_bound_estimate_of_data(train_data[0], train_data[1], norm=2, max_cores=MAX_CORES) # p = 2: the l2 distance, as the networks
    L_arr = L_RATIOS * L_data
    print(f"L_data = {L_data:.3f} (l2), scanning L = {np.round(L_arr, 3)}")

    # TRAINING AND EVALUATION OF NETWORKS
    # The settings both models take. The output starts at the mean training target, which makes the Lipschitz
    # networks train much faster (see Lipschitz_testing.ipynb).
    model_params = dict(input_dim=data.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, depth=DEPTH,
                        output_bias_init=float(train_data[1].mean()))
    training_params = dict(lr=LR, lr_patience=LR_PATIENCE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           max_iter=MAX_ITER)

    # One scan per seed, saved as soon as it is done (Results/L_scan_pt_angles_l2/L_scan_seed<seed>.pkl). A seed that
    # is already saved with the same settings is loaded instead of trained again.
    all_results = L_scan_all_seeds(L_arr, train_data, val_data, test_data, model_params, training_params, INPUT_NORM,
                                   SEEDS, os.path.join(RESULTS_DIR, f"L_scan{ending}"), max_cores=MAX_CORES)

    # PLOTTING
    plot_scan(L_RATIOS, collect(all_results, "MSE"), "MSE", PLOT_DIR, f"mse_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "MAE"), "MAE", PLOT_DIR, f"mae_vs_L{ending}.pdf")
    plot_scan(L_RATIOS, collect(all_results, "max_slope") / L_data, r"Used slope / $L_\mathrm{data}$",
              PLOT_DIR, f"used_slope_vs_L{ending}.pdf", show_bound=True)
    plot_training_losses(L_RATIOS, all_results[0], PLOT_DIR, f"validation_losses{ending}.pdf")
    if not QUICK: # The quick run has only two L values
        plot_selected_fits(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, f"test_errors_selected{ending}.pdf")
    plot_predictions(L_RATIOS, all_results[0], test_data[1].numpy(), PLOT_DIR, f"predictions{ending}.pdf")
    return

# The networks are trained in parallel processes, which on macOS start a fresh Python that imports this file.
# The guard makes sure only the original process runs main().
if __name__ == "__main__":
    main()
