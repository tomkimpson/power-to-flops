# Experiment numbering

Three numbering schemes touch this project, and they do not agree. The README's table is
the quick reference; this note is the full account, including the two traps.

## The three schemes

1. **Manuscript numbers, 1–5.** What `paper/main.tex` calls Experiment *N*. Gap-free by
   construction, and the only scheme a reader of the paper ever sees.
2. **Capture identifiers.** The prefix on the record filenames in `results/bench/`, and the
   CLI flag on `scripts/analyze_bands.py` and `scripts/run_bench.py`. Inherited from the
   capture harness's own sequence, which was never renumbered when the paper was.
3. **The internal `experiment` integer.** A field on each row of the `.jsonl` records, set
   by `powertoflops/bench/runner.py` and its siblings. Usually equals the capture id — but
   not for the qblock-marginal capture.

| manuscript | capture id | internal `experiment` | record |
|---|---|---|---|
| Experiment 1 — exchange band | `exp1` | 1 | `exp1_exchange_e2958cc757b9.jsonl` |
| Experiment 2 — operand sweep | `exp2` | 2 | `exp2_operand_cbfbb110b29e.jsonl` |
| Experiment 3 — overhead band | `exp3` | 3 | `exp3_overhead_b9af294e862e.jsonl` |
| Experiment 4 — marginal covert cost | `exp7` | 7 | `exp7_marginal_b73b5094c430.jsonl` |
| Experiment 5 — marginal useful cost | `qblock_marginal` | **8** | `qblock_marginal_5dd152cc216f.jsonl` |
| *(unnumbered)* cross-device spread | `exp4` | 4 | `exp4_core_*.jsonl` |
| *(unnumbered)* matched-energy witnesses | `matched_pair` | *(none)* | `matched_pair_*.json` |
| *(disclosure only)* useful total quotient | `qblock` | 6 | `qblock_5ba826fed96c.jsonl` |

Two structural notes on that last column. Every `.jsonl` capture also carries some rows
tagged `experiment: 0` — the idle and zero-dose reference cells that anchor each sweep
(40 of 340 in `exp1`, 10 of 70 in `exp7`, and so on). And the matched-pair files are not
record streams at all: they are single-object `.json` witnesses with no `experiment` field,
keyed instead by `declared` / `covert` / `acceptance` blocks.

## Trap 1 — the schemes collide on the number 4

Capture `exp4` is the **cross-device sweep**. Manuscript **Experiment 4** is the
marginal-dose intervention, whose records are `exp7`.

This is the one place where following the wrong scheme gives you a plausible wrong answer
rather than a missing file. A referee who reads "Experiment 4 — the marginal covert cost"
in the paper, opens `results/bench/`, and reaches for `exp4_core_*.jsonl` gets four
cross-device core sweeps on three different cards — real records, wrong experiment. The
manuscript footnote in the *Setup and provenance* paragraph names this explicitly.

## Trap 2 — there is an internal `experiment: 8`, but no Experiment 8

`powertoflops/bench/qblock.py` sets `QBLOCK_MARG_EXPERIMENT = 8`, so 50 of the 60 rows in
`qblock_marginal_5dd152cc216f.jsonl` carry `"experiment": 8` (the other 10 being the
zero-dose reference cells). The id is deliberately distinct from the total-quotient qblock
capture's 6 so the two can never be pooled.

There is no `exp8_*` file and no manuscript Experiment 8. If you find the integer 8 in a
record, it means manuscript Experiment 5.

## Why the capture ids were not renumbered

They are baked into committed record filenames, the `--expN` flags, and the Slurm scripts.
Renaming them would either invalidate every path in `measured_bands.json`'s `inputs` block
or require rewriting the committed records — both worse than documenting the mapping. Ids 5
and 6 belong to the matched-pair and qblock harnesses (`bench/runner.py`), which the
manuscript reports without numbering, which is why the capture sequence looks gappy from
the paper's side.

## Checking the mapping yourself

The `inputs` block of `results/bench/measured_bands.json` is authoritative for which record
priced which band:

```bash
python -c "
import json, re
d = json.load(open('results/bench/measured_bands.json'))
for k, v in d['inputs'].items():
    name = re.findall(r'[\w.-]+\.jsonl', v['path'])[0]
    print(f\"{k:14s} {name:38s} n={v['n_records']:4d}  {v['uuids'][0][:12]}\")
"
```

which prints, for the committed artifact:

```
exp1           exp1_exchange_e2958cc757b9.jsonl       n= 340  GPU-ac1cbae9
exp2           exp2_operand_cbfbb110b29e.jsonl        n= 135  GPU-ac1cbae9
exp3           exp3_overhead_b9af294e862e.jsonl       n=  85  GPU-ac1cbae9
exp7           exp7_marginal_b73b5094c430.jsonl       n=  70  GPU-ac1cbae9
qblock         qblock_5ba826fed96c.jsonl              n=  40  GPU-757ff02a
qblock_marg    qblock_marginal_5dd152cc216f.jsonl     n=  60  GPU-747045fa
```

Two quirks in that block, hence the regex rather than a plain `basename`:

- The `inputs` keys are a **fourth** naming — the analyser's own argument names
  (`qblock_marg`, not `qblock_marginal`), matching the `--qblock-marg` flags.
- `path` is a plain string for `exp1`/`exp2`/`exp3`/`exp7`, but for `qblock` and
  `qblock_marg` it is the *stringified list* `"['results/bench/qblock_….jsonl']"`. Those two
  flags use `action="append"`, and the list is `str()`-ed on the way into the artifact.
