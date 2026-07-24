# power-to-flops

**Can power draw constrain covert compute?**

Compute governance rests on counting operations: past a threshold, a training run
triggers registration, reporting, and audit obligations. One channel a verifier can read
without trusting the operator's hardware is an off-chip power meter, and the hope is that
an energy budget bounds an operation count. This repository establishes exactly when it
does — and by how much.

The energy cost of an operation is unknown and data-dependent, so inverting a power trace
returns a **set**, not a number. The set's width is a floor `β` on the covert compute the
trace admits, measured in *machines* of declared-format capacity. We derive `β` in closed
form from named physical bands, measure those bands on NVIDIA A100s, and price what each
additional verifier capability buys.

We do **not** claim energy accounting is vacuous. It is loose against a bare meter and
tightens substantially as capabilities are added.

## Results

Every term is evaluated inside **one jointly feasible scenario** — one declared format,
one covert format, one observation window sharing the machine — with the covert cost
measured as a genuine incremental lower bound.

| quantity | value | status |
|---|---|---|
| Exchange-rate span on one A100 | **24.8×** (0.79–19.5 pJ/op) | measured (Exp 1) |
| Overhead power band `P₀` | **[59.8, 101.0] W** | measured (Exp 3) |
| Attained covert work, accepted by the sensor's own rule | **0.41 machines** at declared utilisation `h ≈ 0.22` | measured (3 witnesses) |
| Busy-time ceiling `h_max` | **≈ 0.24** | measured |
| Model worst case, precision checked | **`β*_phys` = 1.16** at `h* = 0.42` | model **upper bound** |
| Model worst case restricted to feasible `h` | **≈ 0.77** | model |
| Against a bare meter (format not even pinned) | **1.91** | model |

The `1.16` peak sits *above* the utilisation any single window can hold, so it is an upper
bound rather than an attained width. The attained result is the `0.41`: three fixed-`T`,
same-`C_dec` witnesses that hide four-tenths of a machine of covert int8 work while the
meter raises no alarm.

The identified set then narrows down the **capability ladder**:

| rung | `β*_phys` | condition it rests on |
|---|---|---|
| bare meter | 1.91 | — |
| precision declared & checked | 1.16 | declared-format execution |
| clock & locality observed | *unpriced* (prospective) | + prospective |
| declared work re-executed | 0.34 | + operating point observed |
| covert work on useful data | 0.071 | + operating point observed |
| covert work at top of useful band | 0.057 | + operating point observed |

The rungs **interlock** — re-execution pays only at an observed operating point — and the
chain is cumulative, not a menu.

## Setup

```bash
git clone git@github.com:tomkimpson/power-to-flops.git
cd power-to-flops
pip install -r requirements.txt        # CPU-only: tests, analysis, all five figures
pip install -r requirements-gpu.txt    # only to re-capture on hardware (pinned torch)
```

Reproducing every number and figure below needs **no GPU and no LaTeX** — it consumes the
committed records in `results/bench/`. `pip install -e .` is optional; the scripts put the
repo root on `sys.path` themselves.

## Reproducing the results

**Tests** — numerical anchors guard the closed form, the band extraction, and every
reported headline value:

```bash
pytest -m "not gpu"    # 263 passed, 7 deselected (GPU tests skip cleanly without torch)
```

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

Two flags carry real meaning. `--qblock-marg` (Exp 8) is **required**: without it the
artifact carries no `r_useful_marg` and the bottom two ladder rungs cannot be priced.
`--allow-multi-device` is needed because Exp 1–3 and Exp 7 share one card by design while
the qblock captures ran on other cards of the same SKU — the tool otherwise **refuses**
mixed-device input (see "Measurement campaign" below for why).

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
python scripts/plot_beta_band_figures.py   # figures/{exchange_band,operand_envelope,overhead_affine}.*
python scripts/plot_beta_phys.py           # figures/beta_phys_curve.*     (the headline β(h) curve)
python scripts/plot_capability_ladder.py   # figures/capability_ladder.*   (β down the rungs)
```

Each script prints its own anchors, so a regression is visible without opening the PDF.
`plotstyle.save` suppresses the PDF `CreationDate`, so re-running on unchanged data
produces **byte-identical** files — `git status figures/` staying clean is a real check,
not timestamp luck.

**The attained bound.** Re-validate one of the three matched-pair witnesses:

```bash
python scripts/analyze_pair.py results/bench/matched_pair_133496492841.json
# -> β_witness = 0.411 machines, NO ALARM on any cheating repeat
```

## Measurement campaign

Re-capturing needs A100s with NVML under Slurm. The recipes in `scripts/*.slurm` are
written for OzSTAR (`--account=oz022`, `--partition=milan-gpu`); adapt the header for
another site. Every record carries a provenance manifest with GPU UUID, config snapshot,
and git commit.

```bash
sbatch scripts/run_bench.slurm             # Exp 1-3 then Exp 7, ONE GPU, ONE allocation
sbatch scripts/run_core_sweep.slurm        # Exp 4: compact core sweep across distinct A100s
sbatch scripts/fetch_qblock_weights.slurm  # extract the trained Pythia-6.9B layer
sbatch scripts/run_qblock.slurm            # trained-weight int8 block, total quotient (disclosure only)
sbatch scripts/run_qblock_marginal.slurm   # Exp 8: incremental useful-dose sweep (prices rungs 4-5)
sbatch scripts/run_matched_pair.slurm      # the fixed-T, same-C_dec cheating witnesses
```

Four protocol facts are load-bearing, and the analysis enforces the first:

1. **Exp 1, 2, 3 and 7 must share one GPU in one allocation.** Exp 7 measures the
   *marginal* covert cost that becomes the denominator of every reported bound, so it must
   sit on the same silicon as the declared band it divides. `analyze_bands.py` refuses
   mixed UUIDs for these unless explicitly overridden.
2. **Exp 4 is deliberately cross-device** — its whole purpose is the device-to-device
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

`PROVENANCE.md` records where this repository was split from and what that means for the
`analysis_git_commit` fields inside the records.

## Paper

```bash
cd paper && latexmk -pdf main.tex
```

`paper/main.pdf` is a build artifact and is **not** tracked — build it from source rather
than trusting a committed copy.
