"""Run Exp 8 — the marginal useful-cost dose-response (referee B5, third round).

The Exp-7 construction (fixed fp16 declared burst, idle-padded to a fixed
window) with the trained-weight qblock forward as the swept covert dose, so
baseline power cancels in the dose slope and the useful edge becomes a genuine
INCREMENTAL lower bound — not the standalone total quotient E/C of run_qblock,
which folds in idle baseline (anti-conservative as a covert denominator).
Writes results/bench/qblock_marginal_<sweep_id>.jsonl (+ .csv, + provenance
manifest); feed the JSONL to scripts/analyze_bands.py --qblock-marg to land
r_useful_marg_lo/hi in measured_bands.json — the covert denominator of the
capability ladder's bottom two rungs (powertoflops.ladder).

Prerequisite: the trained-weight artifacts named in powertoflops.config.DEFAULT.bench
(qblock_weights_npz / qblock_acts_npz / qblock_manifest) must exist locally.
Regenerate them on a network-connected node with

    python scripts/fetch_qblock_weights.py --out data/qblock

The runner hard-fails (naming that script) if they are missing or their sha256
does not match the committed manifest.

Usage:
    python scripts/run_qblock_marginal.py --out results/bench
    python scripts/run_qblock_marginal.py --preflight-only   # validate, no capture
    python scripts/run_qblock_marginal.py --dry-run          # print grid, no GPU

All knobs come from powertoflops.config.DEFAULT.bench (qblock_marg_* + marg_* fields).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.marginal import (  # noqa: E402
    build_dose_sweep,
    capacity_frac_of_dose,
)
from powertoflops.bench.qblock import qblock_arm_label  # noqa: E402
from powertoflops.config import DEFAULT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/bench", help="output directory")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the dose grid without touching the GPU")
    ap.add_argument("--preflight-only", action="store_true",
                    help="load + quantize + validate the qblock forward against "
                         "the fp32 reference on the largest cell, then exit")
    args = ap.parse_args()

    cfg = DEFAULT.bench
    arms = [qblock_arm_label(s, b) for s, b in cfg.qblock_marg_cells]
    pts = build_dose_sweep(arms, cfg.qblock_marg_cov_fracs,
                           cfg.marg_declared_frac)
    n_windows = len(pts) * (cfg.warmup + cfg.repeats)
    print(f"exp8 qblock-marginal: declared {cfg.marg_declared_dtype} "
          f"{cfg.marg_declared_shape} at h={cfg.marg_declared_frac} of a "
          f"{cfg.marg_window_s:.0f}s window; {len(arms)} arms x "
          f"{len(cfg.qblock_marg_cov_fracs)} doses x {cfg.repeats} repeats "
          f"(+{cfg.warmup} warmup) ~= "
          f"{n_windows * cfg.marg_window_s / 60:.0f} min of windows")
    for p in pts:
        cap = capacity_frac_of_dose(cov_frac=p.cov_frac,
                                    declared_frac=cfg.marg_declared_frac)
        print(f"  arm={p.cov_dtype:>16s} f={p.cov_frac:.2f} "
              f"(capacity share {cap:.2f} of the (1-h) residual)")
    print(f"weights: {cfg.qblock_weights_npz}")

    if args.dry_run:
        print("dry-run: no GPU work performed.")
        return

    if args.preflight_only:
        # Lazy GPU import so --dry-run never requires CUDA. The preflight uses
        # the same largest-cell artifact/device sanity check as run_qblock.
        from powertoflops.bench.qblock import preflight_qblock

        metrics = preflight_qblock(cfg, device_index=args.device)
        print(f"pre-flight OK: cosine={metrics['cosine']:.4f} "
              f"rel_l2={metrics['rel_l2']:.4f} max_abs={metrics['max_abs']:.3g} "
              f"finite={metrics['all_finite']}")
        return

    from powertoflops.bench.qblock import run_qblock_marginal

    out_path = run_qblock_marginal(args.out, cfg, device_index=args.device)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
