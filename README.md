# power-to-flops

**Can power draw constrain covert AI compute?**

A verifier wants to constrain the total compute that an adversarial prover is running, given only an off-chip power meter.

How tight can the verifier make those constraints? 

## Setup

```bash
git clone git@github.com:tomkimpson/power-to-flops.git
cd power-to-flops
pip install -r requirements.txt        # CPU-only: tests, analysis, all six figures
pip install -r requirements-gpu.txt    # only to re-capture on hardware (pinned torch)
```

## Reproducing the results

Reproducing every number and figure needs **no GPU and no LaTeX** — it consumes the
committed records in `results/bench/`. `pip install -e .` is optional; the scripts put the
repo root on `sys.path` themselves.

**Tests** — numerical anchors guard the closed form, the band extraction, and every
reported headline value:

```bash
pytest -m "not gpu"    # the gpu-marked tests need a CUDA card with NVML
```

**Capture identifiers.** Record filenames, CLI flags and Slurm scripts use *capture ids*
(`exp1`, `exp7`, `qblock_marginal`, …), which are **not** the manuscript's experiment
numbers — captures are numbered in the order they were written, manuscript experiments in
the order the argument needs them. The map is in the `powertoflops.bench` package
docstring, which is the authority. Don't renumber a capture to match a manuscript number.

**The measured-band artifact.** `results/bench/measured_bands.json` is the single source
every figure and every quoted number reads from, and its `inputs` block is authoritative
for which record priced which band. `results/bench/` carries the full capture history
rather than a curated subset, so anything not named in that block is context, not a
reported number. Rebuild the artifact from explicitly named records — nothing is selected
newest-by-mtime:

```bash
python scripts/analyze_bands.py \
    --exp1 results/bench/exp1_exchange_e2958cc757b9.jsonl \
    --exp2 results/bench/exp2_operand_cbfbb110b29e.jsonl \
    --exp3 results/bench/exp3_overhead_b9af294e862e.jsonl \
    --exp7 results/bench/exp7_marginal_b73b5094c430.jsonl \
    --qblock results/bench/qblock_5ba826fed96c.jsonl \
    --qblock-marg results/bench/qblock_marginal_5dd152cc216f.jsonl \
    --allow-multi-device
```

`--qblock-marg` is required: without it the artifact carries no `r_useful_marg` and the
bottom two ladder rungs cannot be priced. `--allow-multi-device` declares that the qblock
captures ran on other cards of the same SKU.

The cross-device artifact (per-device bands plus the 3-card spread) is separate. `--core`
takes one path per flag and appends, so repeat the flag rather than globbing:

```bash
python scripts/analyze_bands.py \
    --core results/bench/exp4_core_29be9ab775bd.jsonl \
    --core results/bench/exp4_core_65295e34a0a2.jsonl \
    --core results/bench/exp4_core_99c98d91aec3.jsonl \
    --core results/bench/exp4_core_dcf016ca4ecf.jsonl \
    --out results/bench/measured_bands_multidevice.json
```

**Figures.** All six read their input record from `measured_bands.json`'s `inputs` block,
so a figure can only plot the records that priced the bands it is captioned against:

```bash
python scripts/plot_beta_band_figures.py   # figures/{exchange_band,operand_envelope,overhead_affine,marginal_dose}.*
python scripts/plot_beta_phys.py           # figures/beta_phys_curve.*     (the headline β(h) curve)
python scripts/plot_capability_ladder.py   # figures/capability_ladder.*   (β down the rungs)
```

Regenerated figures are not always byte-identical to the committed ones, and that is not a
regression. `plotstyle.save` suppresses the PDF `CreationDate`, so a rerun on unchanged data
is deterministic on one machine, but two things still differ across machines. **Fonts** are a
real content change: `plotstyle.py` asks for `[Helvetica, Arial, DejaVu Sans]`, and the
committed figures were rendered on a node carrying neither of the first two, so matplotlib
fell through to its bundled DejaVu Sans. Regenerating anywhere Helvetica *is* installed — any
Mac — silently re-renders every figure in a different typeface. Pin `font.sans-serif` to
`["DejaVu Sans"]` first, or regenerate where Helvetica is absent. **Compression** is not a
content change: a different zlib build re-encodes identical bytes differently. Measured on one
Mac, all six PDFs came back two bytes apart with every decompressed stream equal, and three
PNGs differed in IDAT encoding while being pixel-identical. So a dirty `git status figures/`
after a rerun is a prompt to compare decompressed PDF streams or decoded PNG pixels, not
proof that anything moved.

## Measurement campaign

Re-capturing needs A100s with NVML under Slurm. The recipes in `scripts/*.slurm` are
written for a specific cluster; one may need to adapt the header for
another site. Every record carries a provenance manifest with GPU UUID, config snapshot,
and git commit.

```bash
sbatch scripts/run_bench.slurm             # exp1-exp3 then exp7, ONE GPU, ONE allocation
sbatch scripts/run_core_sweep.slurm        # exp4: compact core sweep across distinct A100s
sbatch scripts/fetch_qblock_weights.slurm  # extract the trained Pythia-6.9B layer
sbatch scripts/run_qblock.slurm            # trained-weight int8 block, total quotient (disclosure only)
sbatch scripts/run_qblock_marginal.slurm   # qblock_marginal: useful-dose sweep (prices rungs 4-5)
sbatch scripts/run_matched_pair.slurm      # the fixed-T, same-C_dec cheating witnesses
```

Four protocol facts are load-bearing, and the analysis enforces the first:

1. **`exp1`, `exp2`, `exp3` and `exp7` must share one GPU in one allocation.** `exp7`
   measures the *marginal* covert cost that becomes the denominator of every reported
   bound, so it must sit on the same silicon as the declared band it divides.
   `analyze_bands.py` refuses mixed UUIDs for these four unless explicitly overridden; the
   qblock captures sit outside that guard.
2. **`exp4` is deliberately cross-device** — its whole purpose is the device-to-device
   spread (2–3 % across cards of the same SKU), so it gets its own artifact.
3. **Clock locking and power capping are privilege-denied** on shared nodes, so the DVFS
   axis is passive: each window's achieved SM clock is *recorded* and analysis stratifies
   on it. `scripts/probe_gpu.py` runs as a job prolog to log the denial per allocated node
   rather than assuming it.
4. **NVML power is a 10 Hz zero-order hold** on these A100s, which sets the window length
   (`target_window_s = 1.5 s`, far above the 100 ms hold; idle windows use 5 s for a
   stable energy delta).

Kernel dispatch is load-bearing for the int8 result: the covert edge comes from
`torch._int_mm` dispatching cuBLASLt IMMA kernels in the exact torch/CUDA build pinned in
`requirements-gpu.txt`, with the B operand **column-major**. A different build may dispatch
different kernels and move the band edges. Re-measure rather than assume.

## Reading the code

Comments and docstrings carry about 130 short review tags — `referee B1`–`B6`,
`review C1`–`C7`, `R2.x`, `R3.x`. Each records *why* a particular guard, band definition or
protocol constraint exists and which objection it answers; the tag is always followed by the
substance, so nothing external is needed to read them. They are not requirements and not
open items.

## Paper

```bash
cd paper && latexmk -pdf main.tex
```
