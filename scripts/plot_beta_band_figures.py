"""Standalone single-column band figures (Sec. 3 experiments).

The manuscript is two-column ICML (\\columnwidth ~ 3.25 in), so each measured
result gets one standalone figure authored at that *native* width, rendering
1:1 with no text rescaling. The band helpers come from
powertoflops.bench.bands, so the plotted numbers are by construction the ones in
the text.

Reproduce:
    python scripts/plot_beta_band_figures.py [--bench-dir results/bench]
Output:
    figures/exchange_band.{pdf,png}
    figures/operand_envelope.{pdf,png}
    figures/overhead_affine.{pdf,png}
    figures/marginal_dose.{pdf,png}
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
    _cell_key,
    _exchange_cell,
    _median_per_cell,
    clean_marginal_records,
    clean_overhead_records,
    clean_records,
    exchange_band,
    marginal_covert_cost,
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
    # Group the factorial's cell medians by (precision, locality): the bar is
    # their median, the whisker their min-max range. Because a cell includes the
    # operand distribution (bands._exchange_cell), that range IS the operand
    # (alpha) axis at fixed precision and locality — the factor the bars
    # otherwise average away — and the extreme whisker ends coincide with
    # exchange_band()'s r_lo and r_hi by construction.
    cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, r_cell in _median_per_cell(recs, key=_exchange_cell).items():
        cell[(key[0], key[1])].append(r_cell)
    med = {k: float(np.median(v)) for k, v in cell.items()}
    op_lo = {k: min(v) for k, v in cell.items()}
    op_hi = {k: max(v) for k, v in cell.items()}
    # Raw windows stay as the scatter overlay (repeat spread within a cell).
    runs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in recs:
        runs[(r.dtype, r.locality)].append(r.r_j_per_op)

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
            pts = np.array(runs.get((d, loc), [])) * 1e12
            if pts.size == 0:
                continue
            jit = rng.uniform(-0.18, 0.18, size=pts.shape) * width
            ax.scatter(xi + jit, pts, s=5, facecolors="white",
                       edgecolors=C["black"], linewidths=0.3, zorder=4,
                       alpha=0.75)
        # Operand (alpha) range within each cell: the third swept factor, which
        # the bar median hides. Drawn over the scatter so the span reads as the
        # deliberate axis rather than run noise.
        errs = np.array([[ys[i] - op_lo.get((d, loc), np.nan) * 1e12,
                          op_hi.get((d, loc), np.nan) * 1e12 - ys[i]]
                         for i, d in enumerate(dtypes)]).T
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=C["black"],
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)

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
    # The whisker gets its own proxy handle: all three swept factors are then
    # named in the figure rather than only the two carried by bar position.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D(
        [0], [0], color=C["black"], lw=0.9, marker="_", markersize=4))
    labels.append(r"operand range $\alpha$")
    ax.legend(handles, labels, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.88),
              title=r"Locality $C_{\mathrm{eff}}$ (bars)")
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
    r_lo_op, r_hi_op, w0 = operand_core(recs)
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

    # Envelope floor = cheapest rate reached (operand zeros), as a total
    # quotient E/C. This is the DESCRIPTIVE operand edge r_lo^0, NOT the
    # covert denominator r_lo^cov (that is the Exp-7 marginal edge, which is
    # cheaper because it strips the baseline power this quotient amortises).
    floor_line = ax.axhline(
        r_lo_op * 1e12, ls="--", color=C["vermillion"], lw=1.1, zorder=2,
        label=fr"$r_{{\mathrm{{lo}}}}^{{0}} = {r_lo_op*1e12:.2f}$ pJ/op")

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
    ax.legend(handles=[floor_line], frameon=False, loc="upper left")
    save(fig, "operand_envelope")
    plt.close(fig)
    print(f"[2] w0 = {w0:.3f}, r_lo^0 = {r_lo_op:.3e} J/op "
          f"-> figures/operand_envelope.*")


def fig_overhead_affine(bench_dir: pathlib.Path) -> None:
    recs = clean_overhead_records(read_jsonl(record_for(3, bench_dir)))
    ob = overhead_band(recs)
    P0_lo, P0_hi, slope_r, affine_r2 = ob.P0_lo, ob.P0_hi, ob.r_marginal, ob.affine_r2

    # Aggregate the raw windows to sweep cells and plot the per-cell median with
    # std error bars over its repeats. The cells are exactly what marginal_rate
    # (hence the quoted r and R^2) regresses over, so the plotted points, the
    # fitted line, and the caption numbers all refer to the same estimand.
    by_cell: dict[tuple, list] = defaultdict(list)
    for r in recs:
        by_cell[_cell_key(r)].append(r)
    F = np.array([np.median([r.ops_nominal / r.duration_s for r in rows])
                  for rows in by_cell.values()]) / 1e12   # Top/s
    P = np.array([float(np.median([r.mean_power_w for r in rows]))
                  for rows in by_cell.values()])
    Perr = np.array([float(np.std([r.mean_power_w for r in rows]))
                     for rows in by_cell.values()])

    # Line from the cell-median fit (marginal_rate): slope in W per Top/s, and the
    # intercept in W. Extended to F = 0 to expose the P_0 intercept.
    slope = slope_r * 1e12
    intercept = ob.fit_intercept_w
    xs = np.linspace(0, F.max() * 1.02, 100)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(WIDTH_ICML_COL, 3.0), sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [3, 1]})
    ax.axhspan(P0_lo, P0_hi, color=C["grey"], alpha=0.35, zorder=0,
               label=fr"$P_0$ band [{P0_lo:.0f}, {P0_hi:.0f}] W")
    ax.plot(xs, slope * xs + intercept, color=C["vermillion"], lw=1.2, zorder=2,
            label="affine fit")
    ax.errorbar(F, P, yerr=Perr, fmt="o", ms=3, color=C["blue"], ecolor=C["blue"],
                elinewidth=0.8, capsize=1.5, zorder=3, label="cell median $\\pm$ SD")
    ax.set_ylabel(r"Mean power, $P$ (W)")
    # Headline fit numbers in an unframed in-panel box rather than a title.
    ax.text(0.04, 0.96,
            fr"$R^2 = {affine_r2:.3f}$" "\n" fr"$r = {slope_r*1e12:.2f}$ pJ/op",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.legend(frameon=False, loc="lower right")

    resid = P - (slope * F + intercept)
    rmax = float(np.abs(resid).max()) * 1.25
    axr.axhline(0, color=C["black"], lw=0.8, zorder=1)
    axr.errorbar(F, resid, yerr=Perr, fmt="o", ms=3, color=C["blue"], ecolor=C["blue"],
                 elinewidth=0.8, capsize=1.5, zorder=3)
    axr.set_ylim(-rmax, rmax)
    axr.set_ylabel("Residual (W)")
    axr.set_xlabel(r"nominal-op rate, $F$ (Top/s)")
    save(fig, "overhead_affine")
    plt.close(fig)
    print(f"[3] P0 band [{P0_lo:.1f}, {P0_hi:.1f}] W, affine R^2 {affine_r2:.4f} "
          f"({len(by_cell)} cells) -> figures/overhead_affine.*")


# Covert format -> (colour, label) for the dose-response figure.
COV_STYLE = {
    "int8": (C["blue"], "int8"),
    "fp16": (C["orange"], "fp16"),
}


def fig_marginal_dose(bench_dir: pathlib.Path) -> None:
    """Experiment 7: window energy vs covert dose, the marginal-cost estimand.

    Unlike the between-shape overhead fit (fig_overhead_affine), this intervenes
    on the covert dose alone inside a fixed window, so its slope is the genuine
    marginal cost dE/dC_cov the floor's covert denominator uses. Near-perfect
    linearity (R^2 > 0.999) is the load-bearing result; points are per-dose-cell
    medians with std error bars over the repeats.
    """
    all_recs = read_jsonl(record_for(7, bench_dir))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(WIDTH_ICML_COL, 3.0), sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [3, 1]})

    dose_max = 0.0
    for fmt, (colour, label) in COV_STYLE.items():
        recs = clean_marginal_records(all_recs, fmt)
        by_dose: dict[int, list] = defaultdict(list)
        for r in recs:
            by_dose[r.cov_iters].append(r)
        # Per dose cell: median covert ops (x) and median window energy (y),
        # with the std of energy over the cell's repeats as the error bar.
        D = np.array([float(np.median([r.cov_ops_nominal for r in rows]))
                      for rows in by_dose.values()])
        E = np.array([float(np.median([r.energy_j for r in rows]))
                      for rows in by_dose.values()])
        Eerr = np.array([float(np.std([r.energy_j for r in rows]))
                         for rows in by_dose.values()])

        mc = marginal_covert_cost(all_recs, fmt)   # slope (J/op), intercept (J), R^2
        dose_max = max(dose_max, float(D.max()))
        xs = np.linspace(0, D.max(), 100)
        ax.plot(xs / 1e15, mc.r_marg * xs + mc.intercept_j,
                color=colour, lw=1.2, zorder=2)
        ax.errorbar(D / 1e15, E, yerr=Eerr, fmt="o", ms=3, color=colour,
                    ecolor=colour, elinewidth=0.8, capsize=1.5, zorder=3,
                    label=fr"{label}: {mc.r_marg*1e12:.2f} pJ/op, "
                          fr"$R^2 = {mc.r2:.4f}$")

        resid = E - (mc.r_marg * D + mc.intercept_j)
        axr.errorbar(D / 1e15, resid, yerr=Eerr, fmt="o", ms=3, color=colour,
                     ecolor=colour, elinewidth=0.8, capsize=1.5, zorder=3)

    ax.set_ylabel(r"Window energy, $E$ (J)")
    ax.legend(frameon=False, loc="upper left")
    axr.axhline(0, color=C["black"], lw=0.8, zorder=1)
    axr.set_ylabel("Residual (J)")
    axr.set_xlabel(r"covert dose, $C_{\mathrm{cov}}$ ($10^{15}$ nominal ops)")
    save(fig, "marginal_dose")
    plt.close(fig)
    print(f"[7] dose-response, doses up to {dose_max/1e15:.2f}e15 ops "
          f"-> figures/marginal_dose.*")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()
    bench_dir = pathlib.Path(args.bench_dir)

    fig_exchange_band(bench_dir)
    fig_operand_envelope(bench_dir)
    fig_overhead_affine(bench_dir)
    fig_marginal_dose(bench_dir)


if __name__ == "__main__":
    main()
