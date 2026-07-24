#!/bin/bash
# B3 matched-energy pair: the fixed-T, same-C_dec attainability witness.
# Run A = n_dec declared fp16 iterations on dear operands + idle pad to a
# fixed 10 s window; run B = the SAME n_dec iterations on cheap operands +
# tuned covert int8 fill + pad to the same window. Pre-flight planning skips
# any utilisation above the busy-time bound h_max; artifacts are validated
# (pads, capacity, convergence, sensor acceptance) before being written.
#
# Submit:   sbatch scripts/run_matched_pair.slurm
#
# Walltime: capture is ~20 min (calibration + <=3 h-points x ~21 windows of
# 10 s); 00:45 keeps slack while staying short enough to backfill on a
# saturated milan-gpu queue (cf. the 02:15 trick in run_bench.slurm).
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=asav-pair-b3
#SBATCH --output=results/bench/slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader || true

# fail fast: the gpu-marked smoke runs a miniature capture end to end
python -m pytest tests/test_matched_pair.py -m gpu -q

python -u scripts/run_matched_pair.py --out results/bench --device 0

echo "=== done  $(date -u) ==="
