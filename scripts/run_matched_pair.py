"""Measure the B3 matched-energy pair (fixed T, same C_dec) on the GPU.

Entry point invoked by scripts/run_matched_pair.slurm. Run A = n_dec declared
iterations on dear operands + idle pad to the fixed window; run B = the SAME
n_dec declared iterations on cheap operands + a tuned covert fill + pad to
the same window. Writes results/bench/matched_pair_<id>.json (+ provenance
manifest) per feasible utilisation h; every artifact passes
powertoflops.bench.pair.validate_pair_payload before it is written. Inspect with
scripts/analyze_pair.py.

Usage:
    python scripts/run_matched_pair.py --out results/bench
    python scripts/run_matched_pair.py --h 0.15 0.25     # override h grid
    python scripts/run_matched_pair.py --dry-run         # print plan, no GPU

All other knobs come from powertoflops.config.DEFAULT.bench (pair_* fields).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.config import DEFAULT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/bench", help="output directory")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--h", type=float, nargs="+", default=None,
                    help="utilisations to capture (default: config grid "
                         "+ auto 0.9*h_max edge point)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pair plan without touching the GPU")
    args = ap.parse_args()

    cfg = DEFAULT.bench
    hs = tuple(args.h) if args.h else cfg.pair_h_values
    shape = "x".join(map(str, cfg.pair_declared_shape))
    print(f"window T = {cfg.pair_window_s:.0f} s, "
          f"h grid {hs}"
          f"{' + auto 0.9*h_max edge' if cfg.pair_h_auto_edge and not args.h else ''}")
    print(f"run A (honest):   n_dec x {cfg.pair_declared_dtype} {shape} "
          f"{cfg.pair_dear_dist} ({cfg.pair_declared_locality}) + idle pad")
    print(f"run B (cheating): SAME n_dec x {cfg.pair_declared_dtype} {shape} "
          f"{cfg.pair_cheap_dist} + tuned {cfg.pair_covert_dtype}/"
          f"{cfg.pair_covert_dist} covert fill + pad "
          f"(|dE|/E < {cfg.pair_tol_frac:.0%}, <= {cfg.pair_max_tune} rounds, "
          f"then frozen)")
    print(f"scored: {cfg.pair_repeats} interleaved A/B repeats per h; "
          f"acceptance rule: alarm iff E_obs > max(honest repeats)")

    if args.dry_run:
        print("dry-run: no GPU work performed.")
        return

    # Import the GPU path lazily so --dry-run never requires a CUDA device.
    from powertoflops.bench.pair import run_matched_pair

    paths = run_matched_pair(args.out, cfg, device_index=args.device,
                             h_values=tuple(args.h) if args.h else None)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
