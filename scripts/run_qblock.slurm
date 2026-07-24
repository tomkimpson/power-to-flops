#!/bin/bash
# Useful-operand workload (referee B5): int8 TRAINED-weight transformer-block
# cells on one A100. Output feeds analyze_bands.py --qblock (the one-sided-LCB
# r_useful band pricing capability-ladder rungs 4-5).
#
# Prerequisite: the trained-weight artifacts must already be present and hash-
# matching, i.e. scripts/fetch_qblock_weights.py has been run (see
# data/qblock/README.md). The capture hard-fails otherwise.
#
# Submit:   sbatch scripts/run_qblock.slurm
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=asav-qblock-r2
#SBATCH --output=results/bench/slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader || true

# GPU pre-flight: the qblock forward must run before spending measurement time
# (same pattern as run_b2_capture.slurm).
python -m pytest tests/test_qblock.py -q -m gpu

# Load + quantize + validate the trained-weight forward against the fp32
# reference on the largest cell before metering anything.
python -u scripts/run_qblock.py --preflight-only --device 0

python -u scripts/run_qblock.py --out results/bench --device 0

echo "=== done  $(date -u) ==="
