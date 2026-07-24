#!/bin/bash
# Marginal useful-cost dose-response (referee B5, third round): the Exp-7
# construction with the int8 TRAINED-weight transformer forward as the swept
# covert dose, on one A100. Output feeds analyze_bands.py --qblock-marg (the
# incremental r_useful_marg edge replacing --qblock as the ladder denominator
# for rungs 4-5, so baseline power cancels in the dose slope).
#
# Prerequisite: the trained-weight artifacts must already be present and hash-
# matching, i.e. scripts/fetch_qblock_weights.py has been run (see
# data/qblock/README.md). The capture hard-fails otherwise.
#
# Submit:   sbatch scripts/run_qblock_marginal.slurm
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=asav-qblock-marg
#SBATCH --output=results/bench/slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader || true

# GPU pre-flight: the qblock forward + the marginal-dose smoke must pass before
# spending measurement time.
python -m pytest tests/test_qblock.py tests/test_marginal.py -q -m gpu

# Load + quantize + validate the trained-weight forward against the fp32
# reference on the largest cell before metering anything.
python -u scripts/run_qblock_marginal.py --preflight-only --device 0

python -u scripts/run_qblock_marginal.py --out results/bench --device 0

echo "=== done  $(date -u) ==="
