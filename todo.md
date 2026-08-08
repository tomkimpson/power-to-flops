# TODO

## Checks

- [ ] **Reconcile the Experiment 1 factor-width numbers between the manuscript
  and the code.** The comment block at `powertoflops/bench/bands.py:256`
  ("on the committed A100 artifact this yields...") records different values
  from `paper/main.tex`:

  | quantity | `bands.py` comment | `main.tex` |
  |---|---|---|
  | $w_\alpha$ | 1.90 | 1.88 |
  | product $w_\kappa w_\alpha w_C$ | 26.8 | 26.5 |
  | pooled measured span | 24.9× | 24.8× |
  | residual (factor interaction) | 8% | 7% |
  | $w_\alpha w_C$ vs fp16's measured 2.00× | 1.98 (1%) | 1.96 (2%) |

  $w_\kappa = 13.5$ and $w_C = 1.04$ agree in both.

  The comment is hand-written prose, not generated output, so it is the more
  likely of the two to have drifted — but confirm by re-running `factor_widths`
  on the committed artifact rather than assuming. Whichever is stale, fix it.

  Why it matters: the $w_\alpha w_C \approx 2$ agreement with the fp16 sub-band
  is what the paper cites as licensing the one-factor-per-rung reading of the
  capability ladder (`tab:inputs` caption, and §5's opener), so the two
  percentages should not disagree.

- [ ] **Pin down the reference for the marginal-vs-total-quotient gap in
  Experiment 3.** §4.2 states that pricing the covert kernels as a total
  quotient gives "$0.26$ (int8) to $0.39$\,pJ/op (fp16) more" than the
  marginal cost. The two figures do not appear to be computed against the
  same reference. From the zeros cells of `results/bench/exp1_exchange_*.jsonl`:

  | | resident | streamed | median | median − marginal |
  |---|---|---|---|---|
  | fp16 (marginal 0.765) | 1.114 | 1.195 | 1.155 | **0.39** ✓ |
  | int8 (marginal 0.529) | 0.805 | 0.835 | 0.820 | **0.29** ✗ |

  So fp16's $0.39$ reproduces as the gap against the locality-median zeros
  quotient, but int8's $0.26$ does not — that route gives $0.29$. The int8
  figure does reproduce as $0.786 - 0.53 = 0.256$, i.e. against
  $r_{\mathrm{lo}}^{\mathrm{dec}}$ (the Exp-1 band edge), which is a different
  reference from the one fp16 uses.

  Decide which reference is intended — locality-median standalone quotient, or
  the band edge — and recompute both figures consistently. Note the covert
  kernels run one operand set reused (`marginal.py` builds the covert
  `MatmulSpec` without a locality arm), so *resident* may be the right
  comparison for both, giving $0.28$ / $0.35$.

  Why it matters: this is the only number in the body quantifying what the
  marginal estimand buys over the total quotient, i.e. why Experiment 3 exists
  at all (referee B2). It is recomputable from the released artifacts in a few
  lines, so an inconsistency is likely to be found.
