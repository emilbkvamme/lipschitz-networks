#!/bin/bash
# The L scan of Toy/Scripts/2D/Sine_noise.py on the CPU partition of the LPNHE cluster: all 36 networks
# (11 L values x 3 seeds + 3 unconstrained) train at the same time, one core each, plus one core for the main process.
# Submit from this folder:   sbatch sine_noise_cpu.sh             the full scan
#                            sbatch sine_noise_cpu.sh --quick     a test of a few minutes
#SBATCH --job-name=sine_noise_cpu
#SBATCH --output=%x_%j.out         # the log, in the folder you submit from (%x = job name, %j = job ID)
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=37         # 36 networks + the main process
#SBATCH --mem=48G                  # about 0.7 GB per network process (measured on the Mac)
#SBATCH --time=08:00:00            # the slowest network: at most 200000 steps x ~30 ms = ~2 h, with a margin

REPO="$HOME/lipschitz-networks"    # where you cloned the repository
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=1           # one thread per process: the parallel part is one process per network
export CUDA_VISIBLE_DEVICES=""     # GPUs are not allowed on the cpu partition: make sure PyTorch does not see one

cd "$REPO/Lipschitz_concept/Toy/Scripts/2D"
echo "Running on $(hostname), job $SLURM_JOB_ID, $SLURM_CPUS_PER_TASK cores"
pixi run --manifest-path "$REPO/pixi.toml" python -u Sine_noise.py "$@"    # -u: print to the log right away
