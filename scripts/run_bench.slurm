#!/bin/bash
# A2 NVML bench harness — experiments 1-3 + 7 on ONE A100, ONE allocation.
#
# All sweeps run sequentially in a single job so every record shares one GPU
# UUID (review C5: the 2026-06/07 records spanned three GPUs because each
# experiment was its own array task — the analyze_bands guard now refuses
# that). Exp 7 (the referee-B1/B2 marginal covert-cost capture) feeds the SAME
# headline as exp 1, so it must share the device too. Budget: exp3 ~12 min +
# exp1 ~25 min + exp2 ~10 min + exp7 ~30 min ≈ 1h30 with overheads; 02:15:00
# keeps ~45 min of slack while staying short enough to backfill on a saturated
# partition (a 03:30:00 request sat 2 days deep in the 2026-07-22 queue).
#
# Submit:   sbatch scripts/run_bench.slurm
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:15:00
#SBATCH --job-name=asav-bench-r2
#SBATCH --output=results/bench/slurm-%x-%j.out

set -euo pipefail

# Land in the repo root (the dir this script lives in is scripts/).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
# `|| true`: head closes the pipe early -> nvidia-smi gets SIGPIPE; do not let
# `set -o pipefail` abort the job over provenance logging.
nvidia-smi -q | head -40 || true   # driver / clocks / persistence / power-limit

# R2.1 clock-control probe: re-confirms per allocated node that clock locking /
# power capping are (still) privilege-denied, logging the denial as evidence.
# Never fatal — the stratify-by-achieved-clock fallback is the expected path.
python -u scripts/probe_gpu.py || true

# Fail-fast smoke for the NEW exp-7 harness (tiny grid into a tmp dir, ~1 min
# GPU): a crash here aborts the job BEFORE any records exist, so a rerun never
# strands exps 1-3 on a card exp 7 can't rejoin (the analyze_bands UUID guard
# refuses mixed devices). The sleep gives the card time to decay toward idle
# before exp 3's cold no_ctx/ctx/cached cells (same rationale as the probe).
echo "=== exp7 smoke  $(date -u) ==="
python -m pytest tests/test_marginal.py -m gpu -q -p no:debugging
sleep 90

# Exp 3 FIRST: its no_ctx idle cell measures the context-free idle state, and
# the probe ran in a separate (already exited) process, so no CUDA context
# exists when exp 3 starts; the probe's single ~1.5 s matmul has ~a minute of
# python/torch startup to decay before the first idle window. Then the long
# factorial, then the operand sweep.
for EXP in 3 1 2; do
    echo "=== experiment ${EXP}  $(date -u) ==="
    python -u scripts/run_bench.py \
        --experiment "${EXP}" \
        --out results/bench \
        --device 0
done

# Exp 7 last: the marginal covert-cost dose sweep (referee B1/B2). Same
# process order rationale as above does not apply (no no_ctx cell), it just
# needs the same physical card as exps 1-3.
echo "=== experiment 7  $(date -u) ==="
python -u scripts/run_marginal.py --out results/bench --device 0

echo "=== done  $(date -u) ==="
