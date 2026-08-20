"""Talk variant of the capability-ladder figure (talk/talk.tex, results slide).

Same ladder as scripts/plot_capability_ladder.py with three changes for the
talk: the final assumption rung ("covert at top of useful band", type A) is
dropped, the axis labels use plain beta (the talk never introduces beta*_phys
or the max over h), and output goes to talk/figures/ so the paper's figure is
untouched.

Reproduce:
    python scripts/plot_capability_ladder_talk.py [--bench-dir results/bench]
Output:
    talk/figures/capability_ladder_talk.png / .pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.ladder import build_ladder, check_monotone  # noqa: E402
from powertoflops.measured import load_measured_bands  # noqa: E402
from powertoflops.plotstyle import C, apply_house_style  # noqa: E402

apply_house_style()

LABELS = {
    "precision declared & checked": "precision declared\n& checked",
    "clock & locality observed": "clock & locality\nobserved",
    "declared work re-executed": "declared work\nre-executed",
    "covert work on useful data": "covert work on\nuseful data",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()
    bench_dir = pathlib.Path(args.bench_dir)

    mb = load_measured_bands(bench_dir / "measured_bands.json")
    rungs = build_ladder(mb)
    check_monotone(rungs)
    # Talk scope: no bare-meter setting (cut from the slide) and no final
    # assumption rung (type A) -- the ladder ends at the useful-data rung.
    rungs = [r for r in rungs
             if r.name != "bare meter" and r.rung_type != "A"]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.axhline(1.0, color=C["grey"], lw=0.8, ls="--", zorder=1)
    ax.text(len(rungs) - 0.55, 1.09, r"$\beta = 1$: one whole machine",
            ha="right", va="bottom", fontsize=6.5, color=C["grey"])

    xs = [i for i, r in enumerate(rungs) if r.priced]
    priced = [r for r in rungs if r.priced]
    ax.plot(xs, [r.beta_star for r in priced], color=C["blue"], lw=1.2,
            zorder=2)
    for x, r in zip(xs, priced):
        marker = "s" if r.rung_type == "M" else "o"
        ax.plot(x, r.beta_star, marker, ms=4.5, mfc=C["blue"], mec=C["blue"],
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
    ax.set_ylabel(r"floor $\beta$")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout(pad=0.4)
    outdir = pathlib.Path(__file__).resolve().parent.parent / "talk" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"capability_ladder_talk.{ext}", dpi=300)

    for r in rungs:
        val = f"beta = {r.beta_star:.3f}" if r.priced else "unpriced (prospective)"
        print(f"{r.name:36s} {val}")
    print(f"-> {outdir}/capability_ladder_talk.png/.pdf")


if __name__ == "__main__":
    main()
