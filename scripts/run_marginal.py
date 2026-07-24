"""Run Exp 7 — the controlled marginal covert-cost capture (referee B1/B2).

Entry point invoked by scripts/run_bench.slurm (after exps 3/1/2, same
allocation so all records share one GPU UUID — review C5). Writes
results/bench/exp7_marginal_<sweep_id>.jsonl (+ .csv, + provenance manifest);
feed the JSONL to scripts/analyze_bands.py --exp7 to land r_marg_cov in
measured_bands.json — the covert denominator of the joint headline model
(powertoflops.joint_model).

Usage:
    python scripts/run_marginal.py --out results/bench
    python scripts/run_marginal.py --dry-run     # print the dose grid, no GPU

All knobs come from powertoflops.config.DEFAULT.bench (marg_* fields).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.marginal import (  # noqa: E402
    build_marginal_sweep,
    capacity_frac_of_dose,
)
from powertoflops.config import DEFAULT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/bench", help="output directory")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the dose grid without touching the GPU")
    args = ap.parse_args()

    cfg = DEFAULT.bench
    pts = build_marginal_sweep(cfg)
    n_windows = len(pts) * (cfg.warmup + cfg.repeats)
    print(f"exp7 marginal: declared {cfg.marg_declared_dtype} "
          f"{cfg.marg_declared_dist} {cfg.marg_declared_shape} at "
          f"h={cfg.marg_declared_frac} of a {cfg.marg_window_s:.0f}s window; "
          f"{len(pts)} cells x {cfg.repeats} repeats (+{cfg.warmup} warmup) "
          f"~= {n_windows * cfg.marg_window_s / 60:.0f} min of windows")
    for p in pts:
        cap = capacity_frac_of_dose(cov_frac=p.cov_frac,
                                    declared_frac=cfg.marg_declared_frac)
        print(f"  cov={p.cov_dtype:>5s} f={p.cov_frac:.2f} "
              f"(capacity share {cap:.2f} of the (1-h) residual)")

    if args.dry_run:
        print("dry-run: no GPU work performed.")
        return

    # Import the GPU path lazily so --dry-run never requires a CUDA device.
    from powertoflops.bench.marginal import run_marginal

    out_path = run_marginal(args.out, cfg, device_index=args.device)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
