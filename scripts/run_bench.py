"""Run one A2 bench experiment on the GPU and write its records.

Entry point invoked by scripts/run_bench.slurm (one experiment per array task)
and scripts/run_core_sweep.slurm (experiment 4, one device per array task).
Experiments (R2 campaign; see README.md, "Measurement campaign"):
    1  exchange-rate band r_hi/r_lo  (precision x operand x locality factorial)
    2  operand envelope -> r_lo^0, w0  (operand VALUES only, interleaved rounds;
       descriptive quotients, NOT the covert edge — that is exp7)
    3  marginal rate dP/dF + overhead band  (sustained-F ladder + idle variants)
    4  cross-device core sweep  (band corners; one short allocation per device)

Usage:
    python scripts/run_bench.py --experiment 2 --out results/bench
    python scripts/run_bench.py --experiment 2 --dry-run     # print sweep, no GPU

Output: results/bench/exp{e}_{name}_{sweep_id}.jsonl (+ .csv mirror).
All knobs come from powertoflops.config.DEFAULT.bench (no magic numbers here).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.runner import build_sweep  # noqa: E402
from powertoflops.config import DEFAULT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=int, required=True, choices=(1, 2, 3, 4))
    ap.add_argument("--out", default="results/bench", help="output directory")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the sweep without touching the GPU")
    args = ap.parse_args()

    cfg = DEFAULT.bench
    pts = build_sweep(args.experiment, cfg)
    print(f"experiment {args.experiment}: {len(pts)} sweep cells "
          f"x {cfg.repeats} repeats (+{cfg.warmup} warmup), "
          f"reference every {cfg.ref_every} cells")
    for p in pts:
        extra = f" util={p.util_frac}" if p.util_frac is not None else ""
        clk = f" clk={p.sm_clock}" if p.sm_clock is not None else ""
        idle = f" idle={p.idle_variant}" if p.idle_variant else ""
        print(f"  exp{p.experiment} {p.dtype:>5} {p.locality:>8} "
              f"{p.M}x{p.N}x{p.K} dist={p.dist_a} nbuf={p.n_buffers}"
              f"{extra}{clk}{idle}")

    if args.dry_run:
        print("dry-run: no GPU work performed.")
        return

    # Import the GPU path lazily so --dry-run never requires a CUDA device.
    from powertoflops.bench.runner import run_experiment

    out_path = run_experiment(args.experiment, args.out, cfg, device_index=args.device)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
