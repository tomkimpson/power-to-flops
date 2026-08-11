"""Capacity-clipped floor beta_phys(h) — the headline figure.

Evaluates beta_phys(h) = min[S(h), (1-h) kappa] (powertoflops.floor) over declared
utilisation h for the two JOINTLY format-consistent covert scenarios of the
fp16-declared case study (powertoflops.joint_model, referee B1):

  * int8 covert (kappa = 2): the adversary's best case — the fp16 declared
    band priced inside its own format, covert work at the int8 Exp-7 marginal
    lower confidence limit, capacity share (1-h) of the int8 peak. Its worst
    case over h is the paper's headline.
  * fp16 covert (kappa = 1): covert work that must itself look like fp16 —
    the realistic reference.

Every input is validated by powertoflops.joint_model.validate_joint_case: declared
extrema from the declared format's own Exp-1 cells, the covert denominator
from the Exp-7 marginal LCB (referee B2 — never a total-E/C edge), capacity
and kappa from the config peaks. beta is in units of the declared-format
capacity (A100 dense fp16 peak). The unclipped energy-slack index S(h)
(eq:beta) is drawn faint: energy is the binding constraint for h < h*,
capacity for h > h*, and both bind at h*.

Reproduce:
    python scripts/plot_beta_phys.py [--bench-dir results/bench]
Output:
    figures/beta_phys_curve.png / .pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.floor import beta_phys, beta_phys_worst_case, energy_slack  # noqa: E402
from powertoflops.joint_model import headline  # noqa: E402
from powertoflops.measured import load_measured_bands  # noqa: E402
from powertoflops.plotstyle import C, apply_house_style, save  # noqa: E402

apply_house_style()

# Reviewer's anchor from the pre-R2 (2026-07-16) bands: printed as a check so a
# normalisation regression is visible at a glance (also pinned in test_floor).
_OLD = dict(r=9.084265191935023e-13, r_hi=2.059471073953942e-11,
            dP0=91.51573280060181 - 84.96540837711991,
            F_max=312.0e12)


def _curve(res: dict):
    """(h grid, beta_phys curve, slack line) for one joint case."""
    case = res["case"]
    h = np.linspace(0.0, 1.0, 2001)
    args = dict(r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec,
                r_lo_cov=case.r_lo_cov, dE0=case.dP0, C_max=case.C_max)
    return h, beta_phys(h, **args, kappa=case.kappa), energy_slack(h, **args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()
    bench_dir = pathlib.Path(args.bench_dir)

    mb = load_measured_bands(bench_dir / "measured_bands.json")
    res8 = headline(mb, declared="fp16", covert="int8")
    res16 = headline(mb, declared="fp16", covert="fp16")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    top = 1.12 * max(res8["case"].kappa, res8["S1"], res16["S1"])

    for res, col in ((res8, C["blue"]), (res16, C["orange"])):
        h, b, s = _curve(res)
        # Faint unclipped energy-slack line S(h): binding below the crossing.
        ax.plot(h, s, color=col, lw=0.8, ls=(0, (4, 2)), alpha=0.45, zorder=2)
        # Capacity clip (1-h) kappa.
        ax.plot(h, (1.0 - h) * res["case"].kappa, color=C["grey"], lw=0.7,
                ls=":", zorder=1)
        kappa = res["case"].kappa
        ax.plot(h, b, color=col, lw=1.3, zorder=3,
                label=rf"{res['case'].covert_format} covert "
                      rf"($\gamma = {kappa:g}$)")
        hs, bs = res["h_star"], res["beta_star"]
        ax.plot(hs, bs, "o", ms=4, color=col, zorder=4)
        ax.annotate(rf"$\beta = {bs:.2f}$ at $h = {hs:.2f}$", (hs, bs),
                    textcoords="offset points", xytext=(6, 6), fontsize=6.5,
                    color=col)

    ax.axhline(1.0, color=C["grey"], lw=0.8, ls="--", zorder=1)
    ax.text(0.985, 1.0, r"$\beta = 1$: whole machine", ha="right",
            va="bottom", fontsize=6.5, color=C["grey"])

    # Binding-regime annotations (referee B1: energy DOES bind below h*).
    h8s = res8["h_star"]
    ax.annotate("energy binds", (0.45 * h8s, 0.10 * top), fontsize=6.5,
                color=C["grey"], ha="center")
    ax.annotate("capacity binds", (h8s + 0.55 * (1 - h8s), 0.10 * top),
                fontsize=6.5, color=C["grey"], ha="center")
    ax.axvline(h8s, color=C["grey"], lw=0.5, ls=":", alpha=0.6, zorder=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, top)
    ax.set_xlabel(r"declared utilisation $h$")
    ax.set_ylabel(r"clipped floor $\beta(h)$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", borderaxespad=1.2)

    fig.tight_layout(pad=0.4)
    out = save(fig, "beta_phys_curve")

    for res in (res8, res16):
        case = res["case"]
        print(f"{case.declared_format} declared / {case.covert_format} covert "
              f"(kappa={case.kappa:g}): S(0) = {res['b']:.3f}, "
              f"S(1) = {res['S1']:.2f}, worst case beta = "
              f"{res['beta_star']:.3f} at h = {res['h_star']:.4f}")
        print(f"  inputs: declared band [{case.r_lo_dec:.3e}, "
              f"{case.r_hi_dec:.3e}] J/op, covert LCB {case.r_lo_cov:.3e} "
              f"J/op, dP0 = {case.dP0:.1f} W")
    old_args = dict(r_hi_dec=_OLD["r_hi"], r_lo_dec=_OLD["r"],
                    r_lo_cov=_OLD["r"], dE0=_OLD["dP0"], C_max=_OLD["F_max"])
    oh, ob = beta_phys_worst_case(**old_args, kappa=1.0)
    print(f"reviewer anchor (old bands): S(1) = "
          f"{energy_slack(1.0, **old_args):.2f}, beta = {ob:.4f} at h = {oh:.4f} "
          "(expect 21.69, 0.9569 at 0.0431)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
