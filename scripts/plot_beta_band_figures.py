"""Standalone single-column band figures (Sec. 3 experiments).

The manuscript is two-column ICML (\\columnwidth ~ 3.25 in), so each of the three
measured results gets one standalone figure authored at that *native* width,
rendering 1:1 with no text rescaling. The band helpers come from
powertoflops.bench.bands, so the plotted numbers are by construction the ones in
the text.

Reproduce:
    python scripts/plot_beta_band_figures.py [--bench-dir results/bench]
Output:
    figures/exchange_band.{pdf,png}
    figures/operand_envelope.{pdf,png}
    figures/overhead_affine.{pdf,png}
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.bands import (  # noqa: E402
    clean_overhead_records,
    clean_records,
    exchange_band,
    operand_core,
    overhead_band,
)
from powertoflops.bench.records import read_jsonl, record_for  # noqa: E402
from powertoflops.plotstyle import (  # noqa: E402
    C,
    WIDTH_ICML_COL,
    apply_house_style,
    save,
)

apply_house_style()

# Locality -> semantic colour and human label (Okabe-Ito). Keys are the operand
# residency tags the records actually carry (config.locality_buffers); the kernel
# is compute-bound in BOTH arms, so the retired "DRAM-bound" label is not used.
LOC_STYLE = {
    "resident": (C["blue"], "L2-resident"),
    "streamed": (C["orange"], "streamed"),
}
DTYPE_ORDER = ["fp32", "tf32", "bf16", "fp16", "int8"]

# Textbook switching-activity ordering + readable axis labels.
OPERAND_ORDER = ["zeros", "low_entropy", "sparse_struct", "declared_typical",
                 "dense_random", "adversarial_toggle"]
DIST_LABEL = {
    "zeros": "zeros",
    "low_entropy": "low-entropy",
    "sparse_struct": "sparse",
    "declared_typical": "declared",
    "dense_random": "dense random",
    "adversarial_toggle": "adversarial",
}



def fig_exchange_band(bench_dir: pathlib.Path) -> None:
    recs = clean_records(read_jsonl(record_for(1, bench_dir)))
    cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in recs:
        cell[(r.dtype, r.locality)].append(r.r_j_per_op)
    med = {k: float(np.median(v)) for k, v in cell.items()}

    dtypes = [d for d in DTYPE_ORDER if any(k[0] == d for k in med)] + \
             sorted({k[0] for k in med} - set(DTYPE_ORDER))
    localities = sorted({k[1] for k in med})
    r_lo, r_hi = exchange_band(recs)
    lo, hi = r_lo * 1e12, r_hi * 1e12

    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.5), layout="constrained")
    width = 0.8 / max(len(localities), 1)
    rng = np.random.default_rng(0)   # deterministic jitter for the run overlay
    for j, loc in enumerate(localities):
        ys = [med.get((d, loc), np.nan) * 1e12 for d in dtypes]
        xs = np.arange(len(dtypes)) + j * width
        color, label = LOC_STYLE.get(loc, (C["green"], loc))
        ax.bar(xs, ys, width=width, label=label, color=color, zorder=3)
        # Overlay the individual runs (n~4 per cell) so the per-bar spread is visible.
        for xi, d in zip(xs, dtypes):
            pts = np.array(cell.get((d, loc), [])) * 1e12
            if pts.size == 0:
                continue
            jit = rng.uniform(-0.18, 0.18, size=pts.shape) * width
            ax.scatter(xi + jit, pts, s=6, facecolors="white",
                       edgecolors=C["black"], linewidths=0.35, zorder=4)

    # Band edges as thin dashed reference lines (a full axhspan would shade
    # nearly the whole axis and read as background), bracketed by a
    # double-arrow in a left gutter with the ratio set beside the arrow.
    ax.axhline(lo, ls="--", color=C["grey"], lw=0.7, zorder=1)
    ax.axhline(hi, ls="--", color=C["grey"], lw=0.7, zorder=1)
    x_arrow = -0.55
    ax.annotate("", xy=(x_arrow, hi), xytext=(x_arrow, lo),
                arrowprops=dict(arrowstyle="<->", color=C["black"], lw=0.9))
    ax.text(x_arrow - 0.28, (lo + hi) / 2, fr"${hi/lo:.1f}\times$",
            rotation=90, ha="center", va="center", fontsize=7)

    x_right = (len(dtypes) - 1) + width * (len(localities) - 1)
    ax.set_xlim(-1.05, x_right + 0.55)
    ax.set_xticks(np.arange(len(dtypes)) + width * (len(localities) - 1) / 2)
    ax.set_xticklabels(dtypes)
    ax.tick_params(axis="x", which="minor", bottom=False)  # categorical axis
    ax.set_ylabel(r"Energy per nominal op, $r = E/C$ (pJ/op)")
    ax.set_xlabel(r"Numeric precision $\kappa$")
    ax.set_ylim(0, hi * 1.08)
    # Anchor the legend below the r_hi reference line so the two don't collide.
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88),
              title=r"Operand locality $C_{\mathrm{eff}}$")
    save(fig, "exchange_band")
    plt.close(fig)
    print(f"[1] r_hi/r_lo = {r_hi/r_lo:.2f}x -> figures/exchange_band.*")


def fig_operand_envelope(bench_dir: pathlib.Path) -> None:
    recs = clean_records(read_jsonl(record_for(2, bench_dir)))
    by_dist: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_dist[r.operand_dist_a].append(r.r_j_per_op)

    dists = [d for d in OPERAND_ORDER if d in by_dist] + \
            [d for d in by_dist if d not in OPERAND_ORDER]
    x = np.arange(len(dists))
    r_lo_cov, r_lo_op, r_hi_op, w0 = operand_core(recs)
    lo, hi = r_lo_op * 1e12, r_hi_op * 1e12

    fig, ax = plt.subplots(figsize=(WIDTH_ICML_COL, 2.6), layout="constrained")

    # Shade the operand envelope [r_lo^0, r_hi^0] = the spread w0 isolates.
    ax.axhspan(lo, hi, color=C["grey"], alpha=0.12, zorder=0)

    # Categorical x: plot runs at the exact category position — they stack
    # vertically in r (the meaningful axis); the tight clustering is the point.
    y_top = hi
    for i, d in enumerate(dists):
        vals = np.array(by_dist[d]) * 1e12  # -> pJ/op
        y_top = max(y_top, float(vals.max()))
        ax.scatter(np.full_like(vals, i), vals, s=12, alpha=0.7, color=C["blue"],
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Covert edge = cheapest rate reached (operand zeros); the band floor.
    cov_line = ax.axhline(
        r_lo_cov * 1e12, ls="--", color=C["vermillion"], lw=1.1, zorder=2,
        label=fr"$r_{{\mathrm{{lo}}}}^{{\mathrm{{cov}}}} = {r_lo_cov*1e12:.2f}$ pJ/op")

    # Bracket the envelope span and label the operand core w0 in a right gutter.
    x_arrow = (len(dists) - 1) + 0.62
    ax.annotate("", xy=(x_arrow, hi), xytext=(x_arrow, lo),
                arrowprops=dict(arrowstyle="<->", color=C["black"], lw=0.9))
    ax.text(x_arrow + 0.24, (lo + hi) / 2, fr"$w_0 = {w0:.2f}$",
            rotation=90, ha="center", va="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABEL.get(d, d) for d in dists],
                       rotation=30, ha="right")
    ax.tick_params(axis="x", which="minor", bottom=False)  # categorical axis
    ax.set_xlim(-0.6, x_arrow + 0.62)
    ax.set_ylim(lo * 0.90, y_top * 1.05)
    ax.set_ylabel(r"Energy per nominal op, $r = E/C$ (pJ/op)")
    ax.set_xlabel(r"Operand value distribution (activity $\alpha$)")
    ax.legend(handles=[cov_line], frameon=False, loc="upper left")
    save(fig, "operand_envelope")
    plt.close(fig)
    print(f"[2] w0 = {w0:.3f}, r_lo^cov = {r_lo_cov:.3e} J/op "
          f"-> figures/operand_envelope.*")


def fig_overhead_affine(bench_dir: pathlib.Path) -> None:
    recs = clean_overhead_records(read_jsonl(record_for(3, bench_dir)))
    F = np.array([r.ops_nominal / r.duration_s for r in recs]) / 1e12   # Top/s
    P = np.array([r.mean_power_w for r in recs])
    ob = overhead_band(recs)
    P0_lo, P0_hi, slope_r, affine_r2 = ob.P0_lo, ob.P0_hi, ob.r_marginal, ob.affine_r2

    # fit in Top/s units for the line, extended to F = 0 to expose the intercept
    slope, intercept = np.polyfit(F, P, 1)
    xs = np.linspace(0, F.max() * 1.02, 100)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(WIDTH_ICML_COL, 3.0), sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [3, 1]})
    ax.axhspan(P0_lo, P0_hi, color=C["grey"], alpha=0.35, zorder=0,
               label=fr"$P_0$ band [{P0_lo:.0f}, {P0_hi:.0f}] W")
    ax.plot(xs, slope * xs + intercept, color=C["vermillion"], lw=1.2, zorder=2,
            label="affine fit")
    ax.scatter(F, P, s=10, color=C["blue"], zorder=3, label="measured windows")
    ax.set_ylabel(r"Mean power, $P$ (W)")
    # Headline fit numbers in an unframed in-panel box rather than a title.
    ax.text(0.04, 0.96,
            fr"$R^2 = {affine_r2:.3f}$" "\n" fr"$r = {slope_r*1e12:.2f}$ pJ/op",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.legend(frameon=False, loc="lower right")

    resid = P - (slope * F + intercept)
    rmax = float(np.abs(resid).max()) * 1.25
    axr.axhline(0, color=C["black"], lw=0.8, zorder=1)
    axr.scatter(F, resid, s=10, color=C["blue"], zorder=3)
    axr.set_ylim(-rmax, rmax)
    axr.set_ylabel("Residual (W)")
    axr.set_xlabel(r"nominal-op rate, $F$ (Top/s)")
    save(fig, "overhead_affine")
    plt.close(fig)
    print(f"[3] P0 band [{P0_lo:.1f}, {P0_hi:.1f}] W, affine R^2 {affine_r2:.4f} "
          f"-> figures/overhead_affine.*")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()
    bench_dir = pathlib.Path(args.bench_dir)

    fig_exchange_band(bench_dir)
    fig_operand_envelope(bench_dir)
    fig_overhead_affine(bench_dir)


if __name__ == "__main__":
    main()
