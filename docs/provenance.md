# Provenance

Where this repository came from, and what that means for the commit hashes inside the
records. Recovered from the `PROVENANCE.md` that used to sit at the repository root; kept
here because the pointers are load-bearing for anyone auditing the records, but they are
not front-page material.

## The split

This repository was split out of a larger private research repository at upstream commit
`09e9408d36b00e5973f550338ee6a64d0dffba16` (2026-07-25), so that the manuscript in `paper/`
and the closure of what reproduces its numbers stand on their own. History was not carried:
this repository starts from a single import commit.

Nothing was re-measured for the split. Every record, every figure and every reported number
is bit-for-bit what the source repository carried.

Three things changed in the copy, none of which touch a number:

1. The library package was renamed `code` → `powertoflops`. The old name shadowed the
   Python standard library's `code` module, which newer `pdb` imports eagerly. **Do not
   rename it back** — the failure is an obscure one at debugger start-up, not an import
   error you would catch in tests.
2. Figures were flattened from `figures/paper-beta/` into `figures/`, since only one
   manuscript's figures travel. `paper/main.tex`'s `\graphicspath` follows.
3. Code, tests and configuration outside the manuscript's dependency closure were left
   behind — the tracker, detection and simulation work streams of the source repository,
   along with its design notes and review correspondence.

## The commit stamps

`measured_bands.json` carries `analysis_git_commit`; the per-record `*_manifest.json` files
carry `git_commit`. Both name the commit of the code that produced the file. Two caveats,
neither of which breaks the chain:

- Records captured before this repository existed carry hashes from the upstream history.
  Those **do not resolve here** — `git show` will fail on them.
- A stamp on a committed artifact is necessarily the commit the analysis *ran at*, which is
  the parent of the commit that records the result. It can never name its own commit.

The check that matters is local and stronger than either: re-running the documented
`analyze_bands.py` command reproduces the committed `measured_bands.json`, with the commit
stamp the only field that differs. Regenerating on a different CPU also perturbs regression
outputs in the last one or two digits (BLAS reduction order), far below any reported
precision.

## Review tags in the comments

Comments and docstrings carry short tags — `referee B1`–`B6`, `review C1`–`C9`, `R2.x`,
`R3.x`, about 130 of them. These are provenance markers from the source repository's review
rounds: each records *why* a particular guard, band definition or protocol constraint
exists, and which objection it answers. They are **not** requirements or open items, and
you do not need the review documents to read the code — the tag is always followed by the
substance. Rough key:

- **`referee B*`** — the round that forced every term to be evaluated inside one jointly
  feasible scenario, and the covert cost to be a genuine *incremental* lower bound.
- **`review C*`** — the earlier round on measurement design: single-device discipline,
  operand-residency-not-problem-size, format-dependent capacity.
- **`R2.*` / `R3.*`** — steps of the re-measurement campaign that produced the records in
  `results/bench/`.

## Figure byte-reproducibility

`plotstyle.save` suppresses the PDF `CreationDate`, so re-running a plot script on unchanged
data is deterministic *on one machine*. Across machines it is weaker, in two independent
ways, and the README's regression check should be read with both in mind.

**Fonts (a real content change).** `powertoflops/plotstyle.py` requests
`font.sans-serif = [Helvetica, Arial, DejaVu Sans]`. The committed figures were rendered on
a node carrying neither of the first two, so matplotlib fell through to its bundled DejaVu
Sans. Regenerating where Helvetica *is* available — any Mac — silently re-renders every
figure in a different typeface, around 10 % of pixels. Either regenerate where Helvetica is
absent, or pin `font.sans-serif` to `["DejaVu Sans"]` first.

**Compression (not a content change).** Even with the font pinned, a different zlib build
re-encodes identical bytes differently. Measured on one Mac: all six PDFs came back two
bytes apart with every decompressed stream equal, and three PNGs differed in IDAT encoding
while being pixel-identical. So a dirty `git status figures/` after a rerun is a prompt to
check, not proof of a regression. To check content rather than bytes, compare the
decompressed PDF streams or the decoded PNG pixels.
