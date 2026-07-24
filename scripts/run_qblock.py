"""Measure the int8 quantized-transformer-block cells (referee B5) on the GPU.

Entry point invoked by scripts/run_qblock.slurm. Prices the useful-operand
band from a TRAINED-weight int8 transformer layer: one extracted Pythia-6.9B
layer, quantization scales applied at dequant, the forward validated against an
fp32 reference before anything is metered. Writes
results/bench/qblock_<sweep_id>.jsonl (+ .csv, + provenance manifest carrying
the weights provenance and the pre-flight validation metrics); feed the JSONL
to scripts/analyze_bands.py --qblock to land r_useful_lo/hi (one-sided LCB) in
measured_bands.json.

Prerequisite: the trained-weight artifacts named in powertoflops.config.DEFAULT.bench
(qblock_weights_npz / qblock_acts_npz / qblock_manifest) must exist locally.
Regenerate them on a network-connected node with

    python scripts/fetch_qblock_weights.py --out data/qblock

The runner hard-fails (naming that script) if they are missing or their sha256
does not match the committed manifest — a measurement never falls back to
random weights (the retired benchmark's defect).

Usage:
    python scripts/run_qblock.py --out results/bench
    python scripts/run_qblock.py --preflight-only   # load+validate, no capture
    python scripts/run_qblock.py --dry-run          # print cells, no GPU

All knobs come from powertoflops.config.DEFAULT.bench (qblock_* fields).
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
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cells without touching the GPU")
    ap.add_argument("--preflight-only", action="store_true",
                    help="load + quantize + validate the forward against the "
                         "fp32 reference on the largest cell, then exit "
                         "(no measurement) — the artifact/device sanity check")
    args = ap.parse_args()

    cfg = DEFAULT.bench
    print(f"qblock: d_model={cfg.qblock_d_model} n_head={cfg.qblock_n_head} "
          f"ffn_mult={cfg.qblock_ffn_mult}; {len(cfg.qblock_cells)} cells "
          f"x {cfg.qblock_repeats} repeats (+{cfg.warmup} warmup), "
          f"soak {cfg.qblock_soak_s:g}s")
    for seq, batch in cfg.qblock_cells:
        print(f"  seq={seq} batch={batch} tokens={seq * batch}")
    print(f"weights: {cfg.qblock_weights_npz}")

    if args.dry_run:
        print("dry-run: no GPU work performed.")
        return

    if args.preflight_only:
        # Lazy GPU import so --dry-run never requires CUDA.
        from powertoflops.bench.qblock import preflight_qblock

        metrics = preflight_qblock(cfg, device_index=args.device)
        print(f"pre-flight OK: cosine={metrics['cosine']:.4f} "
              f"rel_l2={metrics['rel_l2']:.4f} max_abs={metrics['max_abs']:.3g} "
              f"finite={metrics['all_finite']}")
        return

    from powertoflops.bench.qblock import run_qblock

    out_path = run_qblock(args.out, cfg, device_index=args.device)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
