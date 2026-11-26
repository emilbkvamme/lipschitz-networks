# Lipschitz Networks

Research code and experiments for studying Lipschitz-constrained neural networks
applications in HEP.

## Contents
(Subject to change)

- `Lipschitz_concept/`: models, training utilities, experiments, notebooks, and plots
- `Lipschitz_concept/Toy/`: toy experiments and visualizations scripts
- `requirements.txt`: project dependencies
- `requirements-lock.txt`: pinned environment dependencies


## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebooks in `Lipschitz_concept/` contain the experiments and visualizations.

## Running the scripts

- **Paths.** No machine-specific paths are hard-coded. Every script and notebook sets its own paths in a `PATHS` block
  at the top (e.g. `DATA_PATH`, `PLOT_DIR`, `RESULTS_DIR`), relative to its own location by default, so it runs from
  any folder. Change them there to read from or write to somewhere else.
- **Cores.** The parallel sweeps (`fit_many` in `Utils/train_eval.py`, `train_many` in `Toy/Toy_nd_utils.py`) take
  `max_cores`, the most cores a run may use. The scripts read it from the command line:
  `python L_scan_pt_angles_l2.py --max-cores 4`.
- **Saved models.** `fit` and `train_one` take a `model_path`: the trained network is saved there (`.pt`), and a later
  run with the same model, settings, seed and data loads it instead of training it again. The Simulated scans save
  every network to `Results/<test>/<file>_models/`. Load one with `load_model(path)` from `Utils/train_eval.py`.

## Contributing

Issues and pull requests are welcome. Please describe the motivation and
include relevant experiment details when proposing changes.
