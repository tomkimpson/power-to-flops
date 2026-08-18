"""Capability-ladder figure — beta*_phys down the rungs, log scale.

Plots the cumulative ladder of powertoflops.ladder.build_ladder (referee B4): every
priced rung is the capacity-clipped physical worst case
beta*_phys = max_h beta_phys(h) under the B1 joint model (fp16-declared case
study, int8 covert at the Exp-7 marginal LCB), the same quantity tab:ladder
quotes. The clock/locality rung is unpriced (B6: w0 is descriptive, attested
telemetry prospective) and is drawn as an open slot. Daggered labels mark the
conditional chain (declared-format execution; + operating point from the
re-execution rung down). Also prints the hardware-gap sensitivity sweep and
the "power + paperwork" stall value quoted in the manuscript.

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

from powertoflops.config import A100_DENSE_PEAK_OPS  # noqa: E402
from powertoflops.floor import beta_phys_worst_case  # noqa: E402
from powertoflops.ladder import (  # noqa: E402
    COND_OPERATING_POINT,
    DECLARED_FORMAT,
    DVFS_FACTOR,
    build_ladder,
    check_monotone,
    power_and_paperwork_stall,
)
from powertoflops.measured import covert_edge, format_band, load_measured_bands  # noqa: E402
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
    "covert work on useful data": "covert work on\nuseful data",
    "covert work at top of useful band": "covert at top of\nuseful band",
}




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

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.axhline(1.0, color=C["grey"], lw=0.8, ls="--", zorder=1)
    ax.text(len(rungs) - 0.55, 1.09,
            r"$\beta^{*}_{\mathrm{phys}} = 1$: one whole machine",
            ha="right", va="bottom", fontsize=6.5, color=C["grey"])

    xs = [i for i, r in enumerate(rungs) if r.priced]
    betas = [r.beta_star for r in rungs if r.priced]

    # descent path across the unpriced slot; the assumption rung dashed
    ax.plot(xs[:-1], betas[:-1], color=C["blue"], lw=1.2, zorder=2)
    ax.plot(xs[-2:], betas[-2:], color=C["blue"], lw=1.2, ls=(0, (2, 2)),
            zorder=2)
    for x, r in zip(xs, (r for r in rungs if r.priced)):
        marker = {"M": "s", "A": "D"}.get(r.rung_type, "o")
        mfc = "white" if r.rung_type == "A" else C["blue"]
        ax.plot(x, r.beta_star, marker, ms=4.5, mfc=mfc, mec=C["blue"],
                zorder=3)
        txt = f"{r.beta_star:.2f}" if r.beta_star >= 0.095 else f"{r.beta_star:.3f}"
        ax.annotate(txt, (x, r.beta_star), textcoords="offset points",
                    xytext=(7, 4), fontsize=7)

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
    ax.set_xlim(-0.4, len(rungs) - 0.45)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=38, ha="right")
    ax.set_ylabel(r"worst case $\beta^{*}_{\mathrm{phys}} = \max_h \beta_{\mathrm{phys}}(h)$")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout(pad=0.4)
    out = save(fig, "capability_ladder")

    for r in rungs:
        val = (f"beta* = {r.beta_star:.3f} @ h* = {r.h_star:.2f}  "
               f"[S(1) = {r.S1:.3f}]") if r.priced else "unpriced (prospective)"
        cond = f"  cond: {', '.join(r.conditions)}" if r.conditions else ""
        print(f"{r.name:36s} {val}{cond}")
    print("\nhardware-gap sensitivity (re-execution rung):")
    for gap in GAP_SWEEP:
        reexec = build_ladder(mb, gap=gap)[3]
        print(f"  gap = {gap * 1e12:.2f} pJ/op -> beta* = {reexec.beta_star:.3f}")
    print(f"power + paperwork stall (clock unobserved, DVFS x{DVFS_FACTOR}): "
          f"beta* = {power_and_paperwork_stall(mb):.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
