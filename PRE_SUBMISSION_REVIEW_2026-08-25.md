# Pre-submission review — 2026-08-25

**Manuscript:** Can Power Draw Constrain Covert AI Compute?
**File:** `paper/main.tex` (2351 lines, 23 pp) · `paper/references.bib` (38 entries)
**Target:** arXiv preprint (ICML 2026 template, `[preprint]` mode)
**Method:** six parallel reviewers (prose · consistency · claims · mathematics · figures · contribution); every load-bearing finding below re-verified by hand against the source and the artefacts.

---

## Overall assessment

The science holds. Two reviewers independently re-derived the closed form from the stated
assumptions and got the paper's result; both then recomputed every published number from
`results/bench/*.json` and reproduced **all of them**, including β\* = 1.1556 → 1.16,
the full ladder 1.953/1.156/0.337/0.071/0.059, the branch 0.662/0.225, and the witnesses
0.191/0.286/0.411. Dimensional analysis is clean, the limits behave, and the artefact
regenerates the committed bands and re-renders every figure pixel-identically. **No
mathematical or numerical error was found in the derivation or the results.**

Everything below is framing, disclosure, and hygiene. Three items are genuine defects a
referee will catch; the rest is a tidy-up list.

**Recommendation: REVISE BEFORE SENDING** — one false sentence, one broken sum, one
missing availability statement, plus a bib placeholder that should not ship to arXiv.

---

## Priority 1 — fix before it goes out

**[CRITICAL] `main.tex:1392` — "On the same card" is false, and the paper says so itself.**
The matched-energy witnesses ran on `GPU-108c1549`; the declared edges they are measured
against came from `GPU-ac1cbae9` (Experiments 1–3). Verified directly:
`matched_pair_*.json` → `GPU-108c1549`, `measured_bands.json` → `GPU-ac1cbae9`.
Appendix C:2041–2043 states this plainly — *"the witnesses are therefore each read against
declared edges measured on a different card"* — and the cross-device sweep bounds the
error at 2–3%. So the science is fine and the appendix is honest; the body sentence is
simply wrong. This is the one place a reader can catch a false statement using nothing but
the paper. One-clause fix.

**[MAJOR] `main.tex:1212` — the displayed arithmetic does not evaluate.**
`$13.5 \times 1.88 \times 1.04 = 26.5$` — the printed factors give **26.40**. The
unrounded medians (13.5258 × 1.8813 × 1.0410 = 26.490) do give 26.5, and 26.5/24.82 = 1.067
is what supports the "holds to 7%" claim (26.4 would give 6%). A referee will multiply
these three numbers. Add a digit or write ≈26.5.

**[MAJOR] No code- or data-availability statement anywhere in the PDF.**
`grep -i "github|zenodo|available at|code avail|data avail" main.tex` returns **nothing**.
The paper refers to "the repository" (1746), "the released records" (2045) and "the release"
(2049) without naming any of them; the only pointer is a LaTeX comment on line 10, invisible
in the PDF. The repo is public. Given the artefact is the paper's strongest asset — it
regenerates the bands and every figure, and 226 tests pin the results — this is the most
damaging omission in the submission and the cheapest to fix. Add the URL, and state the
software stack (torch 2.5.1+cu124, CUDA 12.4, Python 3.10.16): `requirements-gpu.txt` itself
warns that kernel dispatch moves `r_lo^cov`, the denominator of every reported bound.

**[MAJOR] `references.bib:321` — stale placeholder ships in the arXiv tarball.**
`% ---- PLACEHOLDER (TODO: replace with citable reference from Mauricio Baker) ----`
describing a β_MFU bound no longer in the paper. It is a BibTeX comment so nothing renders,
but arXiv screens submitted source, and "placeholder text left in the manuscript" is one of
the two patterns it treats as evidence of undisclosed LLM use. Same class:
`main.tex:77` `% Placeholder: further authors to be added here.` — which also implies the
author list is not final. Delete both. Also swap `todonotes` to `[disable]` (line 60) as
the preamble comment instructs; no `\todo` calls remain.

---

## Priority 2 — claims and framing

**[MAJOR] β is called a "floor" and an "upper limit" in the same paper.**
Flagged independently by two reviewers. §1:178 *"derives a **floor** on the covert compute
a power trace admits"*, §3.3 titled "The floor", §2.1:244 *"every floor we derive is a floor
on what even a noiseless meter can establish"* — against Appendix B:1933 *"The model bound
β\* = 1.16 is an **upper** limit"* and eq:relaxation *"true identified-set width ≤ β(h)"*,
with Discussion (vii):1680 agreeing. Both readings are defensible — a floor with respect to
the noiseless-meter idealisation, a ceiling with respect to the corner relaxation — but the
paper never reconciles them in one place, and the two words point in opposite directions for
one symbol. Most likely single referee comment in the paper.

**[MAJOR] The abstract's 0.059 drops a prerequisite the body calls undeployable.**
The re-execution rung, inherited by every rung below, needs an observed operating point.
§5:1557–1564 says a verifier without on-site clock sensing *"stalls at β\* ≈ 1"* and that
*"no methods are currently deployable"*; it also rests on an assumed, unmeasured hardware
gap ε ≈ 0.05 pJ/op. Both conditions live in a table footnote (`‡`, 1498) and in §5 prose.
The abstract and conclusion quote 0.059 unconditioned, and those are the sentences a policy
reader will lift. One reviewer computed the deployable-today figure — meter + checked
format + useful-work restriction + pinned overhead, no clock — at **β\* ≈ 0.39**. Worth
either a clause in the abstract or an extra ladder row.

**[MAJOR] The 0.19–0.41 witness hides zero-operand work.**
Verified: all witness records carry `"dist": "zeros", "dtype": "int8"`. The paper *is*
honest about this — limitation (ii):1650 says the corner cell *"is int8 on zeros, which
compute nothing"*, and §4:1276 says the covert operands are zeros. But the abstract states
*"We hid 0.19 to 0.41 machines of undeclared work undetected"* with no such qualifier, and
that is the one genuinely-measured result behind the weakness claim. Priced at the paper's
own useful-work edge it is ≈0.04–0.08 machines. Carry the qualifier up to where the claim
is made.

**[MAJOR] Headline β\* = 1.16 is called "worst-case" but is measured with the clock pinned.**
§2.2:258 gives the prover the clocks; the declared band behind 1.16 was sampled with the
governor inside ±3% (1206, 2054). Limitation (iv):1656 concedes a DVFS-free prover raises
both s and b. So §5:1423 calling 1.16 *"a worst-case bound under Assumptions 1 and 2 alone"*
is not right — it is conditional on a clock-stable prover. Anti-conservative direction, so
it does not threaten the conclusion, but the wording will draw fire.

**[MINOR but load-bearing] `r_useful = 2.53` pJ/op must be an infimum over all useful covert
work; it is a minimum over two shapes of one quantised block of one model.** The bottom two
rungs rest entirely on it, and it carries a 4.9× lever. An adversary optimises against
exactly this quantity — larger batch, KV-cached inference, int4, structured sparsity.
Worth a sentence in limitations acknowledging it is not an infimum.

**[MINOR] Threat-model gaps a governance referee will raise in the first five minutes.**
None is mentioned anywhere: (a) **on-site energy storage** — UPS banks, which every data
centre has by construction, let a prover charge outside the priced window and discharge
inside it, adding compute joules the meter never sees; this forges nothing, so Assumption 1
does not exclude it; (b) **compute outside the metered boundary** (second feed, colluding
sites, undeclared racks); (c) **window aggregation** — β is scale-invariant in T, so 0.059
per window accumulates over a year, and covert work in the gaps between windows is never
discussed. One sentence each, as exclusion or acknowledged gap.

---

## Priority 3 — analyses a referee will ask for

- **No H100, B200, Hopper, Blackwell, fp8 or fp4 appears anywhere in the manuscript.** The
  A100 is 2020 silicon and is not what treaties target. γ is a ratio of published peaks and
  needs no new measurement: on Blackwell the covert ladder extends to fp4, γ ≈ 4 rather
  than 2, and the closed form gives β\* ≈ 1.56 at the headline rung — i.e. the A100 numbers
  probably *understate* the problem. (Nicely, the bottom rung barely moves, since energy
  binds there.) A short datasheet table would close this.
- **No translation to absolute FLOP.** The intro motivates with the 10²⁶ threshold; β is
  per-device, per-window, dimensionless, and the paper never connects them. At the headline
  (β = 1.16, h\* = 0.42) hidden work is **2.8× the declared work**; at the bottom rung it is
  6%. That contrast is the paper's policy payload and it is currently left for the reader to
  compute. Relatedly, **"machine" is never defined** — in a governance abstract, "more than
  a full machine" will be read as a cluster, not one A100 over one window.
- **Sensitivity is swept only on ε.** `b` is ~70% of the bottom rung and ΔP₀'s upper edge is
  the peak of a decaying five-window transient (n = 5). The paper's own median-based variant
  (23.9 W) moves the bottom rung **0.059 → 0.042**, a 28% swing from one estimator choice.
  The headline is robust (1.16 → 1.11); the bottom rung is not. A sensitivity table over all
  inputs, not just ε, would pre-empt this.
- **Uncertainty is never propagated to β.** 1.16 and 0.059 are quoted to 3 and 2 s.f. with no
  interval, while the paper itself shows β moving 22% on the published-vs-measured peak
  choice and 0.29–0.54 across the ε sweep. Either quote intervals or drop to ≈1.2 and ≈0.06.
- **NVML is never validated against an independent instrument**, and the harness already
  contains an unused integrating sampler that would have made the comparison nearly free.
  Yang et al. find only ~25% of runtime is sampled on A100; R² > 0.999 excludes curvature but
  not a multiplicative slope bias — and that slope is the denominator of every bound.
- **The die-counter → wall-meter direction is asserted, not argued.** The paper says a
  retrofitted meter "widens ΔP₀, and so widens β", but `r_lo^cov` is *also* read at the die
  and also rises at the wall. A constant conversion efficiency scales numerator and
  denominator alike, leaving `s` invariant and `b` nearly so. Work it through the closed form.
- **Exp. 3 time-multiplexes** (declared burst → covert dose → idle pad) rather than co-running
  the streams; real co-resident covert work contends for SMs, HBM and L2. And the witness
  only works at h ≤ 0.24, below realistic training MFU. Worth noting — though see the free
  win below.

**One free strengthening:** h\* = 0.42 is presented as an adversarial worst case over a free
parameter, but h is hardware-operation utilisation and frontier transformer training runs at
MFU ≈ 30–55%. The worst case coincides with the realistic operating point. Saying so turns
β\* = 1.16 from a contrived corner into the expected case.

---

## Priority 4 — internal consistency

- **[MAJOR] `C_hi`/`C_lo` carries two unrelated meanings** (flagged twice): the *compute*
  interval in ops at eq:projection/eq:chi/fig:noninjective, and the *effective-capacitance*
  band edges at eq:edgefactor:811–814, where line 817 says only "writing C for C_eff to keep
  the subscripts readable" without noticing the collision 500 lines earlier. `w_C` propagates
  into eq:widthfactor and Table 1. Rename one pair.
- **[MAJOR] Appendix D's "fp16 operand floor 1.16 pJ/op" (2223, 2239) is never reconciled
  with §4's `r_lo^dec` = 1.10 pJ/op for the same format.** They are genuinely different
  measurements (`format_bands.fp16.r_lo` = 1.0960 vs `r_lo_op` = 1.1612) but a reader will
  see one quantity disagreeing with itself. Worse, 1.16 pJ/op collides numerically with
  β\* = 1.16.
- **[MAJOR] `main.tex:1394` — "Every cheating run passes the no-alarm test of eq:noalarm"
  is not the test applied.** The witness clears `E_B ≤ mean(E_A) + 3·SD(E_A)` = 1986.3 J,
  whereas eq:noalarm's ceiling is ≈2485 J. True *a fortiori*, and the empirical rule is ~20%
  tighter — which strengthens the result — but the two rules should not be equated.
- **[MAJOR] Appendix B's "why the two ends do not meet" is mis-explained.** It attributes the
  gap to the achieved rates reaching none of the box edges, quoting 1.84/0.82/0.54 pJ/op —
  but those are *marginal* energies compared against *total-quotient* band edges, the very
  estimands Appendix A insists on keeping apart (hence 0.82 sitting below the 1.10 floor).
  Numerically the witness reaches **93% of its declared edge**, and 84% of the shortfall is
  the un-exercised ΔE₀ the witness forgoes by construction. The honest reading is a
  *stronger* result: the box is tight on rates, and the conservatism lives in the overhead
  band.
- `main.tex:1995` — covert edge printed as 0.54 pJ/op; artefact is 0.5347, which rounds to
  **0.53**. Overstates how close the witness sits to the 0.51 edge.
- `main.tex:1231` — ΔP₀ = 23.9 W is quoted one sentence before the ±5% tolerance that
  produces it; a reader checking against 62.9/96.2 gets 16.7 W.
- Appendix D "a monotone 1410 → 1335 MHz slide" — the per-family median SM clocks run
  1410 → **1320** MHz; 1335 is Experiment 1's clock floor.
- Appendix C:2037 lists `GPU-757ff02a` as measuring "the trained-weight total quotient,
  reported only as a contrast" — but no such quotient appears anywhere in the paper. One of
  the seven cards measures nothing the reader can see.
- `main.tex:2133` — "~34 MB against the 40 MB L2: its resident arm marginally spills". 34 < 40,
  so as written it fits.
- Table 2 rounds the β_en(1) column to 2.4 and 1.6 while the text and fig:betafloor quote
  2.39 and 1.62; the same column carries 0.072 and 0.059.
- `main.tex:1587`, `1632` — the ratios 1.47 and 4.9× don't reproduce from the printed
  operands (0.76/0.51 = 1.49, 2.53/0.51 = 4.96); both are right on unrounded 0.516.
- `\label{app:estimands}` sits after an unnumbered `\paragraph`, so its PDF anchor jumps to
  Equation (40) rather than Appendix A. `app:boundkind` and `app:witness` are two labels on
  one section, so `\cref`s at 756, 827 and 1403 all land in the same place.
- **β is a maximisation over accepted traces, not over one trace.** eq:betadef defines the
  width for a *given* trace; the reported β pushes E to the no-alarm ceiling. That is the
  correct security object, but eq:betadef as written is a different, smaller quantity. One
  sentence after eq:noalarm fixes it.
- **Band containment is used everywhere but is not an Assumption.** The derivation needs
  `r̄^cov ≥ r_lo^cov` (the load-bearing one — it divides the slack), stated as eq:incremental
  only in an unnumbered appendix paragraph and never referenced from §3. Elevate it.
- `seferis2025structuring` is in the bib and never cited; BibTeX will silently drop it.

---

## Priority 5 — figures and tables

- **`fig:capability_ladder` — the blue dashed chain line runs straight through the orange
  "0.23" label at print size**, striking out the "2" and "3". Confirmed on the rendered page
  and in the source PNG (`plot_capability_ladder.py:132`, `xytext=(7,-9)`). Visible rendering
  defect; worth fixing.
- **`figures/overhead_affine.{pdf,png}` is generated and committed but included nowhere**
  (verified: only five `\includegraphics` calls, none of them this one) — while Appendix A's
  "Is the affine map true?" paragraph (1865–1904) argues exactly its content in prose, and
  the affine map underpins the entire forward model. The plot already exists and shows the
  pooled-vs-like-for-like split (R² = 0.85 vs 0.99) a referee will ask about.
- `fig:exchange-band` carries a black double-headed "24.8×" arrow the caption never mentions.
- `fig:marginal-dose` caption discloses a data exclusion — *"the flagged high-tail repeat is
  excluded"* — with **no criterion given and no mention anywhere else in the paper**, in the
  experiment that sets the covert denominator. Predictable referee stop.
- `fig:operand-envelope` caption calls the band "the observed spread w₀ = 1.83", but it is
  min-to-max of per-cell *medians*; several plotted windows sit visibly above it.
- Neither `fig:betafloor` nor `fig:ladder` shows any uncertainty on the headline numbers,
  though the text below the latter reports its own ε sweep as 0.29–0.54.
- `fig:betashape` caption uses s, b, γ without defining them.
- `fig:exchange-scales` — panel 4's "DVFS changes every switching cost" overflows its
  rectangle on both sides.
- Table 1's block (b) edges (19.5/0.786/0.51) are not the edges producing β\* = 1.16 — a
  reader computing from "the inputs" table gets ≈17. The caption says "no format pinned", but
  relabelling the block would help.
- `fig:ladder` marks every priced rung `†` while Table 2 uses `†` and `‡` differently, so a
  reader following the dagger never sees the operating-point condition.

---

## Priority 6 — prose and hygiene

`main.tex:1440` sentence ends with no full stop · `1642` "assumptions … push" → *pushes* ·
`1647` "framework and closed form … is" → *are* · `1654` hyphen used as an em-dash
("grounded - band edges"), renders visibly wrong · `298` opens with `` and closes with a
straight `"` · `1163–1173` Experiment 1's opening sentence has no main verb · `1387`
"favorable" is the lone Americanism in a British-English manuscript · `183` "e.g" missing its
period · `1157` "percent" vs "per cent" elsewhere · `1625` sentence opens on lowercase
"int8" · `1623` blank line after `\paragraph` breaks the run-in heading · `1684`
`\paragraph{Richer telemetry}` missing its terminal period · `1453`, `1457` two sentence
fragments · `175`, `1708` "to a future work" → *to future work* · `1079`, `1647` raw `\ref`
in a `\cref` manuscript · `178`, `1464` sentence-initial lowercase `\cref` → `\Cref` ·
DVFS, NVML, HBM and SKU are never expanded · Experiment 4 narrated in past tense while 1–3
are present · MOSFET capitalisation differs between Appendix A and the figure.

---

## arXiv LLM-compliance gate

**Verdict: WARN.**

- Chatbot meta-comments / conversational artefacts: **none** in `main.tex`. The one hit is
  the deliberate AI-usage declaration at 1744, which is what arXiv wants to see.
- Placeholder text: **two**, both in source comments that ship in the tarball —
  `references.bib:321` and `main.tex:77`. Delete before submitting. This is what moves the
  verdict off PASS.
- Citation integrity: all 37 cited keys resolve to bib entries; no citation lacks an entry.
- **Not checked:** whether each reference is a *real* publication. That needs external DOI /
  Semantic Scholar / arXiv lookups. Run `/check-refs` before submitting — hallucinated
  references are the other arXiv-ban pattern and I have not ruled them out.

---

## Venue note

The template is `[preprint]` and 23 pages, which is right for arXiv and is what the source
comments say you intend. For an **ICML main-track** submission it would additionally need:
8-page main text (currently ~14 before appendices), anonymisation (currently full author
list, email and MATS acknowledgement), and an Impact Statement (absent). Worth a decision
rather than a discovery in January.

---

## Literature the reviewers expected and did not find

Ranked by how hard the omission is to defend. All verified to exist.

1. **Wasil, Reed, Miller & Barnett, "Verification methods for international AI agreements",
   arXiv:2408.16074** — nearest prior work; already lists energy/power as a route to
   detecting undeclared datacentres. The novelty claim needs positioning against it.
2. **Kocher, Jaffe & Jun, "Differential Power Analysis", CRYPTO 1999** — the work that
   established "power traces constrain what computation happened". Currently absent.
3. **Yang, Adamek & Armour, SC24** (DOI 10.1109/SC41406.2024.00028) — the peer-reviewed
   version of the cited `yang2023parttime` preprint; its 25%-sampling finding bears directly
   on the NVML question above.
4. **Petrie, Aarne, Ammann & Dalrymple, flexHEG, arXiv:2506.15093** — the leading on-chip
   answer to the same problem.
5. **Newkirk et al., arXiv:2506.14551** — wall-level 8×H100 node power; quantifies the
   die-counter-vs-wall-meter gap limitation (v) waves at.
6. **Brundage et al., "Toward Trustworthy AI Development", arXiv:2004.07213** — canonical
   origin of "verifiable claims".
7. **Brass & Aarne, *Location Verification for AI Chips*, IAPS 2024** — same move as this
   paper (governance guarantee from a physical constraint) in a sibling channel.
8. **Arafa et al., CF'20** (DOI 10.1145/3387902.3392613) — validates NVML energy against
   purpose-built hardware measurement.

---

## Top three actions

1. **Fix the three factual defects:** the "same card" sentence (1392), the 26.5 product
   (1212), and the missing code/data-availability statement.
2. **Delete both placeholders** (`references.bib:321`, `main.tex:77`) and disable
   `todonotes` — these are arXiv-screening patterns, not style.
3. **Reconcile "floor" with "upper limit"**, and carry the two suppressed qualifiers into
   the abstract: that 0.059 presumes a clock capability the body calls undeployable, and
   that the 0.19–0.41 witness hides zero-operand work.
