"""Capability-ladder figure — beta* down the rungs, log scale.

Plots the cumulative ladder of powertoflops.ladder.build_ladder (referee B4): every
priced rung is the capacity-clipped, band-relaxed worst case
beta* = max_h beta_phys(h) under the B1 joint model (fp16-declared case
study, int8 covert at the Exp-7 marginal LCB), the same quantity tab:ladder
quotes. The clock/locality rung is unpriced (B6: w0 is descriptive, attested
telemetry prospective) and is drawn as an open slot. Daggered labels mark the
conditional chain (declared-format execution; + operating point from the
re-execution rung down).

The covert-format branch (build_covert_format_branch) is drawn as hollow
markers hanging off the two rungs where its denominator is measured, NOT as a
continuation of the chain: it reaches the same dial as the measured-workload rung, so
it does not compose with the rungs below (see that function). Also prints the
hardware-gap sensitivity sweep and the "power + paperwork" stall value quoted
in the manuscript.

Reproduce:
    python scripts/plot_capability_ladder.py [--bench-dir results/bench]
Output:
    figures/capability_ladder.png / .pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.ladder import (  # noqa: E402
    DVFS_FACTOR,
    build_covert_format_branch,
    build_ladder,
    check_monotone,
    power_and_paperwork_stall,
)
from powertoflops.measured import load_measured_bands  # noqa: E402
from powertoflops.plotstyle import C, apply_house_style, save  # noqa: E402

apply_house_style()

# Hardware-gap sensitivity sweep [J/op]: measured same-SKU spread (0.02-0.04),
# the stated assumption (0.05), a cross-vendor guess (0.2). Quoted in the
# re-execution paragraph of the manuscript.
GAP_SWEEP = (0.02e-12, 0.03e-12, 0.04e-12, 0.05e-12, 0.2e-12)
# DVFS rate factor "of order 1.5" (paper, clock rung): with the clock
# unobserved, re-execution transfers with a DVFS-wide error, not the gap.

LABELS = {
    "bare meter": "bare meter",
    "precision declared & checked": "precision declared\n& checked",
    "clock & locality observed": "clock & locality\nobserved",
    "declared work re-executed": "declared work\nre-executed",
    "covert work on useful data": "Exp. 4 covert\nconfigurations",
    "overhead band pinned to the window": "overhead band\npinned to window",
}

# Which chain rung each branch point re-prices, so the branch marker lands on
# that rung's x position rather than extending the chain.
BRANCH_ANCHOR = {
    "branch: covert format known, at the precision rung":
        "precision declared & checked",
    "branch: covert format known, at the re-execution rung":
        "declared work re-executed",
}


def _fmt(beta: float) -> str:
    return f"{beta:.2f}" if beta >= 0.095 else f"{beta:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()
    bench_dir = pathlib.Path(args.bench_dir)

    mb = load_measured_bands(bench_dir / "measured_bands.json")
    rungs = build_ladder(mb)
    check_monotone(rungs)
    # The ladder starts at the precision-checked rung; the bare-meter setting
    # is not plotted (it lives in the appendix, not the main-text ladder).
    rungs = [r for r in rungs if r.name != "bare meter"]
    branch = build_covert_format_branch(mb)
    at_x = {r.name: i for i, r in enumerate(rungs)}

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.axhline(1.0, color=C["grey"], lw=0.8, ls="--", zorder=1)
    ax.text(len(rungs) - 0.55, 1.09,
            r"$\beta^{*} = 1$: one whole machine",
            ha="right", va="bottom", fontsize=6.5, color=C["grey"])

    priced = [r for r in rungs if r.priced]
    xs = [at_x[r.name] for r in priced]
    betas = [r.beta_star for r in priced]

    # Descent path: solid while the verifier is gaining capabilities, dashed
    # once the rungs are forced by the threat model instead (type M).
    i_solid = max(i for i, r in enumerate(priced) if r.rung_type != "M")
    ax.plot(xs[:i_solid + 1], betas[:i_solid + 1], color=C["blue"], lw=1.2,
            zorder=2)
    ax.plot(xs[i_solid:], betas[i_solid:], color=C["blue"], lw=1.2,
            ls=(0, (2, 2)), zorder=2)
    for i, (x, r) in enumerate(zip(xs, priced)):
        marker = "s" if r.rung_type == "M" else "o"
        ax.plot(x, r.beta_star, marker, ms=4.5, mfc=C["blue"], mec=C["blue"],
                zorder=3)
        ax.annotate(_fmt(r.beta_star), (x, r.beta_star),
                    textcoords="offset points", xytext=(7, 4), fontsize=7)

    # The covert-format branch: hollow diamonds hanging off the two rungs it
    # re-prices, joined to them by a thin drop line. Drawn in a second colour
    # and never joined into the chain — it does not compose downward.
    bx = [at_x[BRANCH_ANCHOR[r.name]] for r in branch]
    by = [r.beta_star for r in branch]
    for x, r in zip(bx, branch):
        chain = next(c for c in priced if c.name == BRANCH_ANCHOR[r.name])
        ax.plot([x, x], [chain.beta_star, r.beta_star], color=C["orange"],
                lw=0.7, ls=":", zorder=2)
    ax.plot(bx, by, "D", ms=4.0, mfc="white", mec=C["orange"], mew=1.0,
            zorder=3, ls="none", label="branch: covert format known (fp16)")
    for x, y in zip(bx, by):
        ax.annotate(_fmt(y), (x, y), textcoords="offset points",
                    xytext=(7, -9), fontsize=7, color=C["orange"])
    ax.legend(loc="lower left", fontsize=5.5, frameon=False,
              handlelength=1.4, borderaxespad=0.1, labelcolor=C["orange"])

    # unpriced clock rung: an open slot, not a value
    i_clock = next(i for i, r in enumerate(rungs) if not r.priced)
    ax.axvspan(i_clock - 0.18, i_clock + 0.18, color=C["grey"], alpha=0.15,
               zorder=0)
    ax.text(i_clock, 0.048, "unpriced\n(prospective)", ha="center",
            va="bottom", fontsize=5.5, color=C["grey"], rotation=90)

    labels = []
    for r in rungs:
        dagger = r"$^{\dagger}$" if r.conditions and r.priced else ""
        labels.append(LABELS[r.name] + dagger)

    ax.set_yscale("log")
    ax.set_ylim(4e-2, 4)
    ax.set_xlim(-0.4, len(rungs) - 0.15)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=38, ha="right")
    ax.set_ylabel(r"band-relaxed $\beta^{*} = \max_h \beta(h)$")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout(pad=0.4)
    out = save(fig, "capability_ladder")

    for r in rungs + branch:
        val = (f"beta* = {r.beta_star:.3f} @ h* = {r.h_star:.2f}  "
               f"[S(1) = {r.S1:.3f}]") if r.priced else "unpriced (prospective)"
        cond = f"  cond: {', '.join(r.conditions)}" if r.conditions else ""
        print(f"{r.name:56s} {val}{cond}")
    print("\nhardware-gap sensitivity (re-execution rung):")
    for gap in GAP_SWEEP:
        reexec = build_ladder(mb, gap=gap)[3]
        print(f"  gap = {gap * 1e12:.2f} pJ/op -> beta* = {reexec.beta_star:.3f}")
    print(f"power + paperwork stall (clock unobserved, DVFS x{DVFS_FACTOR}): "
          f"beta* = {power_and_paperwork_stall(mb):.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
