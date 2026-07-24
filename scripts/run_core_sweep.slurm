#!/bin/bash
# R2.5 cross-device core sweep: the exchange-band corners + covert edge on
# several distinct A100s (review C5/C6 — device-to-device spread directly
# measures the hardware-gap / re-execution-transfer rung).
#
# Each array task is ONE allocation -> hopefully one new device; distinct GPU
# UUIDs are counted post hoc from the manifests:
#     grep -h '"gpu_uuid"' results/bench/exp4_core_*_manifest.json | sort -u
# Target >= 3 distinct UUIDs within <= 6 allocations: submit 1-4 first, and if
# short, resubmit the reserve with the already-seen hosts excluded, e.g.:
#     sbatch --array=5-6 --exclude=gina15,gina305 scripts/run_core_sweep.slurm
#
# Submit:   sbatch scripts/run_core_sweep.slurm
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=asav-core-r2
#SBATCH --array=1-4
#SBATCH --output=results/bench/slurm-%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== host $(hostname)  task ${SLURM_ARRAY_TASK_ID}  $(date -u) ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader || true

python -u scripts/run_bench.py \
    --experiment 4 \
    --out results/bench \
    --device 0

echo "=== done task ${SLURM_ARRAY_TASK_ID}  $(date -u) ==="
echo "distinct devices so far:"
grep -h '"gpu_uuid"' results/bench/exp4_core_*_manifest.json | sort -u || true
