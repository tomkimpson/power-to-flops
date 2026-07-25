# power-to-flops

**Can power draw constrain covert compute?**

A verifier wants to constrain the total compute that an adversarial prover is running, given only an off-chip power meter.


## Setup

```bash
git clone git@github.com:tomkimpson/power-to-flops.git
cd power-to-flops
pip install -r requirements.txt        # CPU-only: tests, analysis, all five figures
pip install -r requirements-gpu.txt    # only to re-capture on hardware (pinned torch)
```

## Reproducing the results

Reproducing every number and figure below needs **no GPU and no LaTeX** — it consumes the
committed records in `results/bench/`. `pip install -e .` is optional; the scripts put the
repo root on `sys.path` themselves.

**Tests** — numerical anchors guard the closed form, the band extraction, and every
reported headline value:

```bash
pytest -m "not gpu"    # 263 passed, 7 deselected (GPU tests skip cleanly without torch)
```

### Experiment numbering

The manuscript numbers its experiments **1–5**. Records, CLI flags and Slurm scripts use
**capture identifiers**, which match the filenames in `results/bench/` and are *not* the
same numbers. Everywhere below, `expN` means the capture identifier.

| manuscript | capture id | record that prices it | flag | GPU |
|---|---|---|---|---|
| Experiment 1 — exchange band | `exp1` | `exp1_exchange_e2958cc757b9.jsonl` (340) | `--exp1` | `GPU-ac1cbae9` |
| Experiment 2 — operand sweep | `exp2` | `exp2_operand_cbfbb110b29e.jsonl` (135) | `--exp2` | `GPU-ac1cbae9` |
| Experiment 3 — overhead band | `exp3` | `exp3_overhead_b9af294e862e.jsonl` (85) | `--exp3` | `GPU-ac1cbae9` |
| Experiment 4 — marginal covert cost | `exp7` | `exp7_marginal_b73b5094c430.jsonl` (70) | `--exp7` | `GPU-ac1cbae9` |
| Experiment 5 — marginal useful cost | `qblock_marginal` | `qblock_marginal_5dd152cc216f.jsonl` (60) | `--qblock-marg` | `GPU-747045fa` |
| *(unnumbered)* cross-device spread | `exp4` | `exp4_core_*.jsonl` | `--core` | 3 further cards |
| *(unnumbered)* matched-energy witnesses | `matched_pair` | `matched_pair_*.jsonl` | — | `GPU-108c1549` |
| *(disclosure only)* useful total quotient | `qblock` | `qblock_5ba826fed96c.jsonl` (40) | `--qblock` | `GPU-757ff02a` |

**Mind the collision:** capture `exp4` is the cross-device sweep, whereas *manuscript*
Experiment 4 is the marginal-dose intervention (capture `exp7`). The two schemes disagree
on the number 4; the table is the authority.

Capture ids 5 and 6 were the matched-pair and qblock harnesses (`bench/runner.py`), which
the manuscript reports without numbering. Nothing is missing from either list.

**The measured-band artifact.** `results/bench/measured_bands.json` is the single source
every figure and every quoted number reads from. Rebuild it from explicitly named records
— nothing is selected newest-by-mtime:

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

Two flags carry real meaning. `--qblock-marg` (manuscript Experiment 5) is **required**:
without it the artifact carries no `r_useful_marg` and the bottom two ladder rungs cannot
be priced. `--allow-multi-device` is needed because `exp1`–`exp3` and `exp7` share one card
by design while the qblock captures ran on other cards of the same SKU — the tool otherwise
**refuses** mixed-device input (see "Measurement campaign" below for why).

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

**Figures.** All five read their input record from `measured_bands.json`'s `inputs` block,
so a figure can only plot the records that priced the bands it is captioned against:

```bash
python scripts/plot_beta_band_figures.py   # figures/{exchange_band,operand_envelope,overhead_affine,marginal_dose}.*
python scripts/plot_beta_phys.py           # figures/beta_phys_curve.*     (the headline β(h) curve)
python scripts/plot_capability_ladder.py   # figures/capability_ladder.*   (β down the rungs)
```

`plotstyle.save` suppresses the PDF `CreationDate`, so re-running on unchanged data is
byte-identical and a clean `git status figures/` is a real regression check. One caveat:
`powertoflops/plotstyle.py` requests `font.sans-serif = [Helvetica, Arial, DejaVu Sans]`,
and the committed figures were rendered on a node carrying **none** of the first two, so
matplotlib fell through to its bundled DejaVu Sans. Regenerating on a machine that *does*
have Helvetica (any Mac) silently re-renders every figure in a different typeface. Either
regenerate where Helvetica is absent, or pin `font.sans-serif` to `["DejaVu Sans"]` first.




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
   `analyze_bands.py` refuses mixed UUIDs for these unless explicitly overridden.
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

## Repository layout

| path | contents |
|---|---|
| `powertoflops/` | library: `floor.py` (closed form), `measured.py` (band artifact loader), `joint_model.py` (the headline), `ladder.py` (the rungs), `plotstyle.py` |
| `powertoflops/bench/` | NVML capture harness plus the pure-Python band extraction that consumes its records |
| `scripts/` | entry points: `analyze_bands.py`, `analyze_pair.py`, three `plot_*.py`, the capture drivers, and the Slurm recipes |
| `tests/` | pytest suite; `gpu`-marked tests need a CUDA GPU + NVML |
| `results/bench/` | the committed measurement records, their manifests, and `measured_bands.json` |
| `data/qblock/` | provenance for the trained-weight artifacts (the large `.npz` are not tracked — see the README there) |
| `figures/` | the five manuscript figures, regenerated by `scripts/plot_*.py` |
| `paper/` | the manuscript (ICML 2026 two-column, preprint mode) |

## Record provenance

`results/bench/` carries the full capture history, not a curated subset. The authoritative
list of what actually prices the reported bands is the `inputs` block of
`measured_bands.json`, tabulated under "Experiment numbering" above; the rest is context.

**Superseded captures, retained for the record.** Earlier runs of the same experiments,
replaced by the records in that table and used by no reported number:
`exp1_exchange_796438566c67`, `exp1_exchange_a754ce2c43a9`, `exp2_operand_b835c2c71f65`,
`exp2_operand_efb7b61a6922`, `exp3_overhead_016bc12dea38`, `exp3_overhead_2e655de0e9ca`,
`qblock_d7ba70b76931`.

**Matched-energy witnesses.** Three fixed-`T`, same-`C_dec` pairs support the attained
`0.41` bound, at declared utilisation `h` = 0.10, 0.15 and 0.216:
`matched_pair_c1762bafbb5d`, `matched_pair_03ae3aaefa9f`, `matched_pair_133496492841`.
`matched_pair_c922a1f23b27` is **not** one of them — it is an earlier-schema exploratory
pair with an fp32 declared format at `h = 0.5`, above the measured busy-time ceiling
`h_max ≈ 0.24`. It carries no `acceptance` block, so it cannot support an acceptance claim
and none is made from it.

**The `analysis_git_commit` fields.** Every `*_manifest.json` and `measured_bands.json`
stamps the commit of the code that produced it. Records captured before this repository
existed carry hashes from the upstream history, which **do not resolve here** — `git show`
will fail on them. That is not a broken chain: the check that matters is local and
stronger, namely that re-running the `analyze_bands.py` command above reproduces the
committed `measured_bands.json`, with `analysis_git_commit` the only field that differs.
Regenerating on a different CPU also perturbs regression outputs in the last one or two
digits (BLAS reduction order), well below any reported precision.

## Paper

```bash
cd paper && latexmk -pdf main.tex
```
