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
