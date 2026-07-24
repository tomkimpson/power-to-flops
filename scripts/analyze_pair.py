"""Validate and summarize a B3 matched-pair artifact against the theorem.

Re-runs powertoflops.bench.pair.validate_pair_payload (an invalid artifact exits
nonzero — the referee's hard-failure requirement holds at analysis time, not
only at capture time) and prints the witness table the manuscript quotes:
per-repeat energies, the sensor's one-sided acceptance decision, feasibility
margins, and the linkage of the witness parameters to the measured bands and
the S(h)/beta_phys(h) corner (powertoflops.bench.pair.summarize_pair).

Usage:
    python scripts/analyze_pair.py results/bench/matched_pair_<id>.json \
        [--bands results/bench/measured_bands.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.pair import (  # noqa: E402
    PairValidationError,
    summarize_pair,
    validate_pair_payload,
)
from powertoflops.measured import load_measured_bands  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", help="matched_pair_<id>.json to check")
    ap.add_argument("--bands", default="results/bench/measured_bands.json",
                    help="measured bands artifact for the linkage block")
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.artifact).read_text())

    try:
        validate_pair_payload(payload)
    except PairValidationError as exc:
        print(f"INVALID WITNESS: {exc}")
        sys.exit(1)

    acc = payload["acceptance"]
    print(f"pair {payload['pair_id']}  h = {payload['h']:.3f}  "
          f"(h_max = {payload['h_max']:.3f})  T = {payload['window_s']:.1f} s  "
          f"GPU {payload['gpu_uuid']}")
    print(f"declared: {payload['declared']['iters_a']} x "
          f"{payload['declared']['dtype']} "
          f"{payload['declared']['dist_dear']} (A) / "
          f"{payload['declared']['dist_cheap']} (B) — identical counts")
    print(f"covert:   {payload['covert']['iters']} x "
          f"{payload['covert']['dtype']}/{payload['covert']['dist']}  "
          f"capacity_frac = {payload['capacity_frac_used']:.3f}")
    print(f"E_A = {acc['mean_e_a_j']:.1f} ± {acc['std_e_a_j']:.1f} J   "
          f"E_B = {acc['mean_e_b_j']:.1f} ± {acc['std_e_b_j']:.1f} J   "
          f"|dE|/E = {acc['delta_frac']:.2%}")
    max_thr = acc.get("threshold_max_j")
    max_note = f" (sample max {max_thr:.1f} J)" if max_thr is not None else ""
    print(f"acceptance ({acc['rule']}): threshold {acc['threshold_j']:.1f} J"
          f"{max_note}, verdicts {acc['verdicts']} -> "
          f"{'NO ALARM on any cheating repeat' if acc['all_accepted'] else 'ALARMED'}")
    pads_a = [r["pad_s"] for r in payload["repeats_a"]]
    pads_b = [r["pad_s"] for r in payload["repeats_b"]]
    print(f"pads: A [{min(pads_a):.2f}, {max(pads_a):.2f}] s  "
          f"B [{min(pads_b):.2f}, {max(pads_b):.2f}] s (all > 0)")

    bands_path = pathlib.Path(args.bands)
    if bands_path.exists():
        s = summarize_pair(payload, load_measured_bands(bands_path))
        print("--- linkage to the corner (ass:corners / prop iii-iv) ---")
        print(f"beta_witness = {s['beta_witness']:.3f} machines of declared "
              f"capacity, vs S(h) = {s['S_h']:.3f}, "
              f"beta_phys(h) = {s['beta_phys_h']:.3f} (kappa = {s['kappa']:.1f})")
        print(f"achieved marginal rates [pJ/op]: dear declared "
              f"{1e12 * s['r_marg_dec_dear']:.3f}, cheap declared "
              f"{1e12 * s['r_marg_dec_cheap']:.3f} "
              f"(band [{1e12 * s['r_band_dec'][0]:.3f}, "
              f"{1e12 * s['r_band_dec'][1]:.3f}]), covert "
              f"{1e12 * s['r_marg_cov_achieved']:.3f} "
              f"(exp7 LCB {1e12 * s['r_lo_cov_lcb']:.3f})")
    else:
        print(f"(bands artifact {bands_path} not found — linkage block skipped)")


if __name__ == "__main__":
    main()
