# power-to-flops

**Can power draw constrain covert compute?**

A verifier wants to constrain the total compute that an adversarial prover is running, given only an off-chip power meter.


## Setup

```bash
git clone git@github.com:tomkimpson/power-to-flops.git
cd power-to-flops
pip install -r requirements.txt        # CPU-only: tests, analysis, all six figures
pip install -r requirements-gpu.txt    # only to re-capture on hardware (pinned torch)
```

## Reproducing the results

Reproducing every number and figure below needs **no GPU and no LaTeX** — it consumes the
committed records in `results/bench/`. `pip install -e .` is optional; the scripts put the
repo root on `sys.path` themselves.

**Tests** — numerical anchors guard the closed form, the band extraction, and every
reported headline value:

```bash
pytest -m "not gpu"    # 205 passed, 5 skipped, 2 deselected
```

That is the count for the CPU-only install above, and it is the one that guards every
reported number. Five modules — `test_gpu_smoke`, `test_qblock`, `test_qweights`,
`test_sweeps`, `test_workload` — are gated by a module-level
`pytest.importorskip("torch")` and vanish at collection, so roughly a sixth of the suite is
invisible without `requirements-gpu.txt`. Three of those carry no `gpu` mark at all: they
exercise the capture harness, which needs torch but not a device. Installing torch
therefore *raises* the collected count and the deselected count (7 tests are `gpu`-marked
repo-wide) rather than turning skips into passes.

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
| *(unnumbered)* matched-energy witnesses | `matched_pair` | `matched_pair_*.json` (three of four; see below) | — | `GPU-108c1549` |
| *(disclosure only)* useful total quotient | `qblock` | `qblock_5ba826fed96c.jsonl` (40) | `--qblock` | `GPU-757ff02a` |

**Mind the collision:** capture `exp4` is the cross-device sweep, whereas *manuscript*
Experiment 4 is the marginal-dose intervention (capture `exp7`). The two schemes disagree
on the number 4; the table is the authority.

Capture ids 5 and 6 were the matched-pair and qblock harnesses (`bench/runner.py`), which
the manuscript reports without numbering. Nothing is missing from either list.

One more scheme leaks through, in the records rather than the filenames: each `.jsonl` row
carries an integer `experiment` field from the harness's own enumeration, and for the
qblock-marginal capture that integer is **8** (`QBLOCK_MARG_EXPERIMENT`; its zero-dose
reference cells carry `0`). So `qblock_marginal_5dd152cc216f.jsonl` — manuscript
Experiment 5 — contains rows tagged `"experiment": 8`. There is no `exp8_*` file and no
manuscript Experiment 8; the integer is an internal id only.

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

`--qblock-marg` (manuscript Experiment 5) is **required**: without it the artifact carries
no `r_useful_marg` and the bottom two ladder rungs cannot be priced.

`--allow-multi-device` is declarative here, not load-bearing. The mixed-device guard in
`powertoflops.bench.analysis` covers `exp1`, `exp2`, `exp3` and `exp7` only — the records
that jointly price the floor, and which do share one card — so **this command succeeds
without the flag and produces an identical artifact**. The qblock captures ran on other
cards of the same SKU; their UUIDs are recorded in the artifact's `inputs` block but are
never checked against the others. Pass the flag to state that mixing plainly; do not read
it as evidence that the qblock devices were validated. See "Measurement campaign" below for
why the guard is scoped the way it is.

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

`plotstyle.save` suppresses the PDF `CreationDate`, so re-running on unchanged data is
deterministic *on one machine*: rerun twice and `git status figures/` stays clean. Across
machines it is weaker than that, in two independent ways.

**Fonts.** `powertoflops/plotstyle.py` requests
`font.sans-serif = [Helvetica, Arial, DejaVu Sans]`, and the committed figures were
rendered on a node carrying **none** of the first two, so matplotlib fell through to its
bundled DejaVu Sans. Regenerating where Helvetica *is* available (any Mac) silently
re-renders every figure in a different typeface — a real content change, ~10 % of pixels.
Either regenerate where Helvetica is absent, or pin `font.sans-serif` to `["DejaVu Sans"]`
first.

**Compression.** Even with the font pinned, a different zlib build re-encodes the same
bytes differently: on one Mac all six PDFs came back content-identical (every decompressed
stream equal) but two bytes apart, and three PNGs differed in IDAT encoding while being
pixel-identical. So a dirty `git status figures/` after a rerun is a prompt to check, not
proof of a regression. The reliable check is content, not bytes: compare the decompressed
PDF streams, or the decoded PNG pixels.




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
   `analyze_bands.py` refuses mixed UUIDs **for these four** unless explicitly overridden.
   The qblock captures are outside that guard: they price the useful rungs against their
   own dose intercept rather than against `exp1`'s declared band, so they do not need to
   share the card — but nothing checks their UUIDs either.
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
| `figures/` | the six manuscript figures, regenerated by `scripts/plot_*.py` |
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
`0.41` bound, at declared utilisation `h` = 0.10, 0.15 and 0.216, all on `GPU-108c1549`:
`matched_pair_c1762bafbb5d`, `matched_pair_03ae3aaefa9f`, `matched_pair_133496492841`
(single-object `.json`, not `.jsonl` — these predate the record-stream format).
`matched_pair_c922a1f23b27` is **not** one of them — it is an earlier-schema exploratory
pair on `GPU-aa412071`, with an fp32 declared format at `h = 0.5`, above the measured
busy-time ceiling `h_max ≈ 0.24`. It carries no `acceptance` block, so it cannot support an
acceptance claim and none is made from it. A `matched_pair_*.json` glob picks up all four;
the three named above are the witnesses.

**The commit stamps.** `measured_bands.json` carries `analysis_git_commit`; the per-record
`*_manifest.json` files carry `git_commit`. Both name the commit of the code that produced
the file. Two caveats, neither of which breaks the chain:

- Records captured before this repository existed carry hashes from the upstream history,
  which **do not resolve here** — `git show` will fail on them.
- The stamp on a committed artifact is necessarily the commit the analysis *ran at*, which
  is the parent of the commit that records the result. It can never name its own commit.

The check that matters is local and stronger: re-running the `analyze_bands.py` command
above reproduces the committed `measured_bands.json`, with the commit stamp the only field
that differs. Regenerating on a different CPU also perturbs regression outputs in the last
one or two digits (BLAS reduction order), well below any reported precision.

## Where this repository came from

It was split out of a larger private research repository at upstream commit
`09e9408d36b00e5973f550338ee6a64d0dffba16` (2026-07-25), so that the manuscript and the
closure of what reproduces its numbers stand on their own. History was not carried: this
repository starts from a single import commit. Nothing was re-measured for the split —
every record, figure and reported number is bit-for-bit what the source repository carried.

Three things changed in the copy, none of which touch a number:

1. The library package was renamed `code` → `powertoflops`. The old name shadowed the
   Python standard library's `code` module, which newer `pdb` imports eagerly. **Do not
   rename it back.**
2. Figures were flattened from `figures/paper-beta/` into `figures/`;
   `paper/main.tex`'s `\graphicspath` follows.
3. Code, tests and configuration outside the manuscript's dependency closure were left
   behind, along with the source repository's design notes and review correspondence.

**Review tags in the comments.** Comments and docstrings carry short tags — `referee B1`–
`B6`, `review C1`–`C9`, `R2.x`, `R3.x`, about 130 of them. These are provenance markers
from the source repository's review rounds: each records *why* a particular guard, band
definition or protocol constraint exists. They are not requirements or open items, and you
do not need the review documents to read the code — the tag is always followed by the
substance. Rough key:

- **`referee B*`** — the round that forced every term to be evaluated inside one jointly
  feasible scenario, and the covert cost to be a genuine *incremental* lower bound.
- **`review C*`** — the earlier round on measurement design: single-device discipline,
  operand-residency-not-problem-size, format-dependent capacity.
- **`R2.*` / `R3.*`** — steps of the re-measurement campaign that produced the records in
  `results/bench/`.

## Paper

```bash
cd paper && latexmk -pdf main.tex
```
