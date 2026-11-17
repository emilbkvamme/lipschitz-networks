#!/bin/bash
# The L scan of Toy/Scripts/2D/Sine_noise.py on a MIG slice of the H200 (partition gpu-mig). The networks are tiny,
# so a slice (1/7 of the GPU) is as fast for them as the whole card, and leaves the rest to other users.
# N_WORKERS = 8 networks share the slice at a time, each process needs one core to send its work to the GPU.
# Submit from this folder:   sbatch sine_noise_gpu.sh             the full scan
#                            sbatch sine_noise_gpu.sh --quick     a test of a few minutes
#SBATCH --job-name=sine_noise_gpu
#SBATCH --output=%x_%j.out         # the log, in the folder you submit from (%x = job name, %j = job ID)
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:mig_1g35gb:1    # required on gpu-mig: one slice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # N_WORKERS = 8 network processes + the main process
#SBATCH --mem=32G                  # a process with CUDA takes more memory than on the CPU, about 2-3 GB
#SBATCH --time=08:00:00

REPO="$HOME/lipschitz-networks"    # where you cloned the repository
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=1

cd "$REPO/Lipschitz_concept/Toy/Scripts/2D"
echo "Running on $(hostname), job $SLURM_JOB_ID, GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi
pixi run --manifest-path "$REPO/pixi.toml" python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
pixi run --manifest-path "$REPO/pixi.toml" python -u Sine_noise.py "$@"    # -u: print to the log right away
