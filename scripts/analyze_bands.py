"""Turn measured bench records into the measured-bands artifact.

Thin CLI over :mod:`powertoflops.bench.analysis`: reads the three explicitly named
record files, extracts the bands, and writes ``measured_bands.json`` — the
small, git-trackable artifact that powertoflops.measured loads.

Inputs are EXPLICIT (review C5): name each experiment's JSONL — nothing is
picked newest-by-mtime. Records from more than one GPU are refused unless
``--allow-multi-device`` is passed, in which case the artifact carries
per-device bands and the device-to-device spread alongside the (flagged)
pooled numbers. The payload records input paths, sweep IDs, GPU UUIDs, and
the analysis git commit.

Usage (single-device bands, R2.9):
    python scripts/analyze_bands.py \\
        --exp1 results/bench/exp1_exchange_<sweep>.jsonl \\
        --exp2 results/bench/exp2_operand_<sweep>.jsonl \\
        --exp3 results/bench/exp3_overhead_<sweep>.jsonl \\
        [--qblock results/bench/qblock_<sweep>.jsonl] \\
        [--qblock-marg results/bench/qblock_marginal_<sweep>.jsonl] \\
        [--exp7 results/bench/exp7_marginal_<sweep>.jsonl] \\
        [--allow-multi-device] [--out results/bench/measured_bands.json]

Usage (cross-device core artifact, R2.5 — per-device bands + spread):
    python scripts/analyze_bands.py \\
        --core results/bench/exp4_core_<a>.jsonl \\
        --core results/bench/exp4_core_<b>.jsonl ... \\
        [--out results/bench/measured_bands_multidevice.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.analysis import (  # noqa: E402
    MixedDeviceError,
    band_payload,
    core_payload,
)
from powertoflops.bench.records import read_jsonl  # noqa: E402


def _qblock_provenance(qblock_paths: list[str]) -> dict | None:
    """Weights/validation provenance for the qblock captures (referee B5).

    Each qblock JSONL written by ``run_qblock`` has a sibling
    ``<stem>_manifest.json`` carrying, under ``extra``, the trained-weight
    ``weights_provenance`` (model/revision/layer + weights sha256), the
    activations sha256, and the GPU pre-flight ``validation`` metrics. This
    reads those sidecars for every ``--qblock`` file and returns one merged
    provenance block for the ``useful_band`` artifact.

    Returns ``None`` if no qblock files are given or none has a manifest (an
    old hand-made record); raises if two captures disagree on the weights
    sha256 — a multi-card useful band must price off ONE trained artifact.
    """
    if not qblock_paths:
        return None

    captures: list[dict] = []
    weights_prov: dict | None = None
    acts_sha: str | None = None
    for path in qblock_paths:
        p = pathlib.Path(path)
        manifest_path = p.with_name(f"{p.stem}_manifest.json")
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extra = manifest.get("extra", {})
        prov = extra.get("weights_provenance")
        if prov is not None:
            sha = prov.get("weights_sha256")
            if weights_prov is None:
                weights_prov, acts_sha = prov, extra.get("acts_sha256")
            elif sha != weights_prov.get("weights_sha256"):
                raise SystemExit(
                    f"error: {manifest_path.name} prices off weights "
                    f"{sha}, but an earlier capture used "
                    f"{weights_prov.get('weights_sha256')} — a multi-card "
                    f"useful band must share one trained-weight artifact")
        captures.append({
            "manifest": manifest_path.name,
            "sweep_id": manifest.get("sweep_id"),
            "gpu_uuid": manifest.get("gpu_uuid"),
            "validation": extra.get("validation"),
        })

    if weights_prov is None:
        return None
    return {**weights_prov, "acts_sha256": acts_sha, "captures": captures}


def _core_main(args) -> None:
    records = [r for path in args.core for r in read_jsonl(path)]
    payload = core_payload(records, inputs={"core": list(args.core)})
    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(args.core[0]).resolve().parent
           / "measured_bands_multidevice.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"  {payload['n_devices']} device(s): "
          f"{', '.join(payload['gpu_uuids'])}")
    for u, bands in payload["per_device"].items():
        r_lo, r_hi = bands.get("r_lo"), bands.get("r_hi")
        if r_lo is not None:
            print(f"  {u}: r_lo/r_hi = {r_lo:.3e} / {r_hi:.3e} J/op"
                  + (f"  r_lo^0 = {bands['r_lo_op']:.3e}"
                     if "r_lo_op" in bands else ""))
    for k, (lo, hi) in payload["device_spread"].items():
        print(f"  spread {k}: [{lo:.3e}, {hi:.3e}]  ({hi / lo:.3f}x)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1", help="Exp-1 (exchange) records JSONL")
    ap.add_argument("--exp2", help="Exp-2 (operand) records JSONL")
    ap.add_argument("--exp3", help="Exp-3 (overhead) records JSONL")
    ap.add_argument("--qblock", action="append", default=[],
                    help="qblock (useful-operand) records JSONL (repeatable "
                         "for the multi-card capture; may come from a "
                         "separate allocation/device — multi-UUID input takes "
                         "the min-over-devices LCB and records the spread)")
    ap.add_argument("--qblock-marg", action="append", default=[],
                    dest="qblock_marg",
                    help="qblock-marginal capture (manuscript Experiment 5): "
                         "marginal useful-cost dose-response records JSONL "
                         "(referee B5; repeatable for the multi-card capture). "
                         "The incremental useful edge replacing --qblock as the "
                         "ladder denominator; like --qblock it may be a separate "
                         "device (multi-UUID takes the min-over-devices LCB)")
    ap.add_argument("--exp7", default=None,
                    help="Exp-7 (marginal covert cost) records JSONL "
                         "(optional; MUST be same-device as exp1-3 — it "
                         "feeds the same headline, so it joins the guard)")
    ap.add_argument("--core", action="append", default=[],
                    help="Exp-4 core-sweep JSONL (repeatable); switches to the "
                         "cross-device artifact and ignores --exp1/2/3")
    ap.add_argument("--allow-multi-device", action="store_true",
                    help="permit exp1-3 records from multiple GPUs; emits "
                         "per-device bands + spread instead of refusing")
    ap.add_argument("--out", default=None,
                    help="output path (default: measured_bands.json beside "
                         "the exp1 input, or measured_bands_multidevice.json "
                         "beside the first --core input)")
    args = ap.parse_args()

    if args.core:
        _core_main(args)
        return
    if not (args.exp1 and args.exp2 and args.exp3):
        ap.error("--exp1/--exp2/--exp3 are required (or use --core)")

    exp1 = read_jsonl(args.exp1)
    exp2 = read_jsonl(args.exp2)
    exp3 = read_jsonl(args.exp3)
    qblock = ([r for path in args.qblock for r in read_jsonl(path)]
              if args.qblock else None)
    qblock_marg = ([r for path in args.qblock_marg for r in read_jsonl(path)]
                   if args.qblock_marg else None)
    exp7 = read_jsonl(args.exp7) if args.exp7 else None

    try:
        payload = band_payload(
            exp1, exp2, exp3, qblock=qblock, qblock_marg=qblock_marg, exp7=exp7,
            inputs={"exp1": args.exp1, "exp2": args.exp2, "exp3": args.exp3,
                    "qblock": (list(args.qblock) or None),
                    "qblock_marg": (list(args.qblock_marg) or None),
                    "exp7": args.exp7},
            qblock_provenance=_qblock_provenance(args.qblock),
            qblock_marg_provenance=_qblock_provenance(args.qblock_marg),
            allow_multi_device=args.allow_multi_device,
        )
    except MixedDeviceError as exc:
        sys.exit(f"error: {exc}")

    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(args.exp1).resolve().parent / "measured_bands.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"wrote {out}")
    if payload["multi_device"]:
        print(f"  WARNING: pooled across {len(payload['gpu_uuids'])} GPUs "
              f"({', '.join(payload['gpu_uuids'])}); per-device bands + spread "
              f"recorded in the artifact")
    print(f"  r_lo/r_hi   = {payload['r_lo']:.3e} / {payload['r_hi']:.3e} J/op  "
          f"(total E/C band, {payload['r_hi'] / payload['r_lo']:.1f}x)")
    print(f"  r_marginal  = {payload['r_marginal']:.3e} J/op  (dP/dF, Exp-3 fit)")
    if "r_marginal_cubes" in payload:
        print(f"  r_marg^cube = {payload['r_marginal_cubes']:.3e} J/op  "
              f"(idle + compute-bound cubes, R^2 = "
              f"{payload['r_marginal_cubes_r2']:.3f}, "
              f"n = {payload['r_marginal_cubes_n_cells']})")
    print(f"  r_lo^0      = {payload['r_lo_op']:.3e} J/op   w0 = {payload['w0']:.3f}"
          "   (Exp-2 descriptive; not the covert edge)")
    print(f"  P0 band     = [{payload['P0_lo']:.1f}, {payload['P0_hi']:.1f}] W  "
          f"(observed idle +/- tolerance); fit intercept "
          f"{payload['P0_fit_intercept']:.1f} +/- {payload['P0_fit_se']:.1f} W "
          f"(diagnostic)")
    if payload.get("P0_lo_resident") is not None:
        print(f"  P0 resident = [{payload['P0_lo_resident']:.1f}, "
              f"{payload['P0_hi_resident']:.1f}] W  "
              f"(resident-operand idle cells only, "
              f"n = {payload['n_idle_resident']}) -- the ladder's "
              f"overhead-pinning rung")
    print(f"  affine R^2  = {payload['affine_r2']:.4f}")
    print(f"  sigma_dec   = {payload['sigma_dec']:.3e} J/op")
    if "r_useful_lo" in payload:
        ub = payload["useful_band"]
        print(f"  r_useful    = [{payload['r_useful_lo']:.3e}, "
              f"{payload['r_useful_hi']:.3e}] J/op  (qblock one-sided "
              f"LCB{100 * (1 - ub['alpha']):.0f}; points "
              f"[{ub['r_lo_point']:.3e}, {ub['r_hi_point']:.3e}], "
              f"v{ub['version']})")
        if "device_spread" in ub:
            lo, hi = ub["device_spread"]["r_lo"]
            print(f"  r_useful device spread (r_lo): [{lo:.3e}, {hi:.3e}] "
                  f"over {len(ub['per_device'])} cards")
    if "r_useful_marg_lo" in payload:
        umb = payload["useful_marginal_band"]
        print(f"  r_useful^marg = [{payload['r_useful_marg_lo']:.3e}, "
              f"{payload['r_useful_marg_hi']:.3e}] J/op  (qblock-marginal dose slope "
              f"one-sided LCB{100 * (1 - umb['alpha']):.0f}; points "
              f"[{umb['r_lo_point']:.3e}, {umb['r_hi_point']:.3e}], "
              f"{umb['n_cells']} arms) -- the ladder covert denominator (B5)")
    for fmt, band in payload["format_bands"].items():
        print(f"  band[{fmt:>5s}] = [{band['r_lo']:.3e}, {band['r_hi']:.3e}] "
              f"J/op  ({band['n_cells']} cells)")
    for fmt, mc in payload.get("r_marg_cov", {}).items():
        stratum = (f"{mc['stratum_mhz']} MHz" if mc["stratum_mhz"] is not None
                   else f"MIXED strata {sorted(mc['strata'])}")
        print(f"  r_marg[{fmt:>4s}] = {mc['r_marg']:.3e} J/op  "
              f"(LCB{100 * (1 - mc['alpha']):.0f} = {mc['r_marg_lcb95']:.3e}, "
              f"R^2 = {mc['r2']:.4f}, n = {mc['n_cells']}, {stratum})")
    n_strata = len(payload["clock_strata"]["exp1"])
    print(f"  clock strata (exp1): {n_strata}")


if __name__ == "__main__":
    main()
