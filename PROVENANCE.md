# Provenance

This repository was split out of a larger private research repository so that the
manuscript in `paper/` and the closure of what reproduces its numbers stand on their own.
Nothing was re-measured for the split: every record, every figure, and every reported
number here is bit-for-bit what the source repository carried.

## Where it came from

| | |
|---|---|
| Source repository | `tomkimpson/analogue-sensors-for-ai-verification` (private) |
| Commit at time of split | `09e9408d36b00e5973f550338ee6a64d0dffba16` |
| Date of split | 2026-07-25 |
| History | **not carried.** This repository starts from a single import commit. |

Three things changed in the copy, none of which touch a number:

1. The library package was renamed `code` → `powertoflops`. The old name shadowed the
   Python standard library's `code` module, which newer `pdb` imports eagerly.
2. Figures were flattened from `figures/paper-beta/` into `figures/`, since only one
   manuscript's figures travel. `paper/main.tex`'s `\graphicspath` follows.
3. Code, tests, and configuration outside the manuscript's dependency closure were left
   behind — the tracker, detection, and simulation work streams of the source repository,
   along with its design notes and review correspondence.

Verified after the split: the full test suite passes (263 passed, 7 GPU tests deselected),
`measured_bands.json` reproduces exactly from the committed records, and all ten figure
files regenerate **byte-identically** to the source repository's committed versions.

## Review tags in the comments

Comments and docstrings carry short tags — `referee B1`–`B6`, `review C1`–`C9`,
`R2.1`–`R2.9`, `R3.x` — about 128 of them. These are provenance markers from the source
repository's review rounds: each records *why* a particular guard, band definition, or
protocol constraint exists, and which review objection it answers. They are not
requirements or open items, and you do not need the review documents to read the code —
the tag is always followed by the substance. Rough key:

- **`referee B*`** — the round that forced every term to be evaluated inside one jointly
  feasible scenario, and the covert cost to be a genuine *incremental* lower bound.
- **`review C*`** — the earlier round on measurement design: single-device discipline,
  operand-residency-not-problem-size, format-dependent capacity.
- **`R2.*` / `R3.*`** — steps of the re-measurement campaign that produced the records in
  `results/bench/`.

## The `analysis_git_commit` fields

Every `results/bench/*_manifest.json` and `measured_bands.json` records an
`analysis_git_commit` — the commit of the code that produced it. **Those hashes resolve in
the source repository, not this one.** `measured_bands.json`, for instance, carries
`59d3691af86971cdab9bfc3788653afb6de365b0` (2026-07-24), which is a commit in the source
history that was not carried here.

This is not a broken chain, but it is one you cannot follow with `git show` in this
repository. The verifiable claim is stronger and local: re-running the documented
`analyze_bands.py` command in *this* repository reproduces the committed
`measured_bands.json` exactly, `analysis_git_commit` being the only field that differs.
The README's reproduction section is the check that matters.

## Which records price the manuscript

`results/bench/` carries the full capture history, not a curated subset — the superseded
records are small and some are referenced in the manuscript's disclosure passages. The
authoritative list of what actually prices the reported bands is the `inputs` block of
`measured_bands.json`:

| role | record | records | GPU |
|---|---|---|---|
| Exp 1 — exchange band | `exp1_exchange_e2958cc757b9.jsonl` | 340 | `GPU-ac1cbae9` |
| Exp 2 — operand sweep | `exp2_operand_cbfbb110b29e.jsonl` | 135 | `GPU-ac1cbae9` |
| Exp 3 — overhead band | `exp3_overhead_b9af294e862e.jsonl` | 85 | `GPU-ac1cbae9` |
| Exp 7 — marginal covert cost | `exp7_marginal_b73b5094c430.jsonl` | 70 | `GPU-ac1cbae9` |
| Exp 8 total (disclosure only) | `qblock_5ba826fed96c.jsonl` | 40 | `GPU-757ff02a` |
| Exp 8 marginal — prices rungs 4–5 | `qblock_marginal_5dd152cc216f.jsonl` | 60 | `GPU-747045fa` |

Exp 1, 2, 3 and 7 share one card in one allocation, by design and by enforcement (see the
README's "Measurement campaign"). Captured on A100-SXM4-80GB, driver 595.84,
2026-07-22.

**Superseded, retained for the record** — earlier captures of the same experiments,
replaced by the runs above and used by no reported number:
`exp1_exchange_796438566c67`, `exp1_exchange_a754ce2c43a9`, `exp2_operand_b835c2c71f65`,
`exp2_operand_efb7b61a6922`, `exp3_overhead_016bc12dea38`, `exp3_overhead_2e655de0e9ca`,
`qblock_d7ba70b76931`.

**Cross-device (Exp 4)** — `exp4_core_{29be9ab775bd,65295e34a0a2,99c98d91aec3,dcf016ca4ecf}`
feed `measured_bands_multidevice.json`, a separate artifact. These are *deliberately*
four different cards; the 2–3 % spread is the result.

**Matched-pair witnesses.** Three fixed-`T`, same-`C_dec` pairs support the attained
`0.41` bound, at declared utilisation `h` = 0.10, 0.15, and 0.216:
`matched_pair_c1762bafbb5d`, `matched_pair_03ae3aaefa9f`, `matched_pair_133496492841`.

`matched_pair_c922a1f23b27` is **not** one of them: it is an earlier-schema exploratory
pair with an fp32 declared format at `h = 0.5`, above the measured busy-time ceiling
`h_max ≈ 0.24`. It carries no `acceptance` block, so it cannot support an acceptance claim
and none is made from it.

## Untracked data

The trained-weight artifacts behind the Exp 8 useful-work edge (ladder rungs 4–5) are too
large to track. Their **sha256 hashes are committed** in
`data/qblock/pythia-6.9b_layer15_manifest.json`, and `powertoflops.bench.qweights` refuses
to load a `.npz` whose hash does not match — the benchmark hard-fails rather than falling
back to synthetic weights, so a measurement cannot silently run on stale or random data.

| file | size |
|---|---|
| `pythia-6.9b_layer15_weights.npz` | 402 MB |
| `pythia-6.9b_layer15_acts.npz` | 134 MB |
| `wikitext103_corpus.txt` | 400 KB |

At the time of the split these lived at
`/fred/oz022/tkimpson/analogue-sensors-for-ai-verification/data/qblock/` on the OzSTAR
`/fred` filesystem. Regenerate them from scratch with:

```bash
python scripts/fetch_qblock_weights.py --out data/qblock \
    --model EleutherAI/pythia-6.9b --revision main --layer 15 \
    --n-seg 8 --seg-len 2048 --dtype float16 --device cuda
```

See `data/qblock/README.md` for the offline-compute-node procedure and the
`transformers` pin that the extraction cross-check depends on.
