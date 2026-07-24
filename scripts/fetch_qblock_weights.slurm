#!/bin/bash
# Extract the trained Pythia-6.9B layer for the qblock benchmark (referee B5),
# OFFLINE, on one A100. The model must already be cached under $HF_HOME (run
# `hf download EleutherAI/pythia-6.9b --revision main` on a login node first)
# and the corpus materialised to data/qblock/wikitext103_corpus.txt.
#
# Writes data/qblock/pythia-6.9b_layer15_{weights.npz,acts.npz,manifest.json}
# and hard-fails if the crosscheck (fp32 reference reproduces the model's own
# hidden_states[L+1]) does not pass — so a bad extraction never yields an
# artifact you would commit.
#
# Submit:   sbatch scripts/fetch_qblock_weights.slurm
#
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=asav-qblock-fetch
#SBATCH --output=results/bench/slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/bench data/qblock

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate error-correction          # transformers 5.4 + torch + scipy

# Cache is on /fred (visible to the compute node); force offline so the job
# never reaches for the network (compute nodes have none).
export HF_HOME=/fred/oz022/tkimpson/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader || true

python -u scripts/fetch_qblock_weights.py \
    --out data/qblock \
    --model EleutherAI/pythia-6.9b --revision main --layer 15 \
    --n-seg 8 --seg-len 2048 --dtype float16 --device cuda \
    --corpus-file data/qblock/wikitext103_corpus.txt

echo "=== done  $(date -u) ==="
