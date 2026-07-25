"""Assemble the measured-bands artifact with provenance and a device guard.

Pure record-list -> payload-dict logic behind ``scripts/analyze_bands.py``
(kept importable and GPU-free so the guard is unit-testable). Two review-C5
failure modes are closed here:

  * inputs are explicit — the caller names the record files; nothing is
    selected newest-by-mtime;
  * records from different GPUs are never pooled silently. One band means one
    device; mixing raises :class:`MixedDeviceError` unless the caller passes
    ``allow_multi_device=True``, in which case the payload carries per-device
    bands and the device-to-device spread instead of pretending to a single
    tighter band than any one card supports.
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from typing import Mapping, Sequence

from .bands import (
    clean_records,
    exchange_band,
    exchange_band_by_format,
    exchange_band_strata,
    extract_bands,
    marginal_covert_cost,
    marginal_rate_compute_bound,
    operand_core,
    operand_core_strata,
    overhead_band,
    sigma_dec_from,
    useful_band,
    useful_marginal_band,
    _median_per_cell,
)
from .records import RunRecord


class MixedDeviceError(ValueError):
    """Records from more than one GPU were passed without allow_multi_device."""


def gather_uuids(*record_lists: Sequence[RunRecord]) -> list[str]:
    """Sorted union of gpu_uuid over all record lists."""
    return sorted({r.gpu_uuid for records in record_lists for r in records})


def analysis_git_commit() -> str:
    """HEAD commit of this checkout (best effort; 'unknown' off-repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _input_provenance(name: str, records: Sequence[RunRecord],
                      path: object | None) -> dict:
    return {
        "path": str(path) if path is not None else None,
        "sweep_ids": sorted({r.sweep_id for r in records}),
        "uuids": gather_uuids(records),
        "n_records": len(records),
    }


def _device_bands(exp1: Sequence[RunRecord], exp2: Sequence[RunRecord],
                  exp3: Sequence[RunRecord]) -> dict:
    """Band quantities computable from ONE device's records; absent exps skipped."""
    out: dict = {}
    try:
        out["r_lo"], out["r_hi"] = exchange_band(exp1)
    except ValueError:
        pass
    try:
        r_lo_op, r_hi_op, w0 = operand_core(exp2)
        out.update(r_lo_op=r_lo_op, r_hi_op=r_hi_op, w0=w0,
                   sigma_dec=sigma_dec_from(exp2))
    except ValueError:
        pass
    try:
        ob = overhead_band(exp3)
        out.update(P0_lo=ob.P0_lo, P0_hi=ob.P0_hi,
                   P0_fit_intercept=ob.fit_intercept_w, P0_fit_se=ob.fit_se_w,
                   r_marginal=ob.r_marginal, affine_r2=ob.affine_r2)
    except ValueError:
        pass
    return out


def band_payload(
    exp1: Sequence[RunRecord],
    exp2: Sequence[RunRecord],
    exp3: Sequence[RunRecord],
    *,
    qblock: Sequence[RunRecord] | None = None,
    qblock_marg: Sequence[RunRecord] | None = None,
    exp7: Sequence[RunRecord] | None = None,
    inputs: Mapping[str, object] | None = None,
    qblock_provenance: Mapping[str, object] | None = None,
    qblock_marg_provenance: Mapping[str, object] | None = None,
    allow_multi_device: bool = False,
) -> dict:
    """The measured_bands.json payload: bands + provenance (+ per-device split).

    ``inputs`` maps "exp1"/"exp2"/"exp3" (and "qblock"/"exp7") to the
    record-file paths for the provenance block. Raises :class:`MixedDeviceError`
    on records from more than one GPU unless ``allow_multi_device`` — and then
    the pooled bands are still computed but flagged ``multi_device: true`` and
    accompanied by ``per_device`` bands and the ``device_spread`` [min, max]
    across devices.

    ``qblock`` (R2.7 / referee B5) contributes the useful-operand band. It is
    a separate quantity, never pooled into the exp1-3 band, so it may come from
    a different allocation/device: its uuids are recorded in ``qblock_uuids``
    and deliberately excluded from the single-device guard. Since the B5
    rebuild the ``r_useful_lo``/``r_useful_hi`` keys carry one-sided 95% LCB
    endpoints and ride with a versioned ``useful_band`` block (version 2);
    pre-B5 artifacts lack that block and fail the ladder's guard. A multi-UUID
    qblock (the 3-card capture) takes the MIN over devices of each device's
    LCB — the safe direction for a lower limit — and records per-device bands
    plus the spread. ``qblock_provenance`` (the extraction manifest / capture
    ``extra`` block) is recorded verbatim under ``weights_provenance``.

    ``qblock_marg`` (referee B5, third round) contributes the MARGINAL useful
    edge ``r_useful_marg_lo``/``r_useful_marg_hi`` from the Exp-8 dose-response
    (:func:`powertoflops.bench.bands.useful_marginal_band`) — the incremental covert
    cost of useful work, which replaces the total-quotient ``qblock`` band as
    the ladder denominator. Like qblock it is a covert denominator on a possibly
    separate device, so it is excluded from the single-device guard and a
    multi-UUID capture takes the min-over-devices lower limit.

    ``exp7`` (referee B1/B2) contributes the per-covert-format marginal costs
    ``r_marg_cov``. Unlike qblock it feeds the SAME headline as the exp1 bands
    (powertoflops.joint_model pairs an exp1 declared band with an exp7 covert edge),
    so its records ARE included in the single-device guard.
    """
    uuids = gather_uuids(exp1, exp2, exp3, exp7 or [])
    if len(uuids) > 1 and not allow_multi_device:
        raise MixedDeviceError(
            f"records span {len(uuids)} GPUs ({', '.join(uuids)}); one band "
            f"means one device — pass matching input files, or set "
            f"allow_multi_device to get per-device bands + spread instead"
        )

    br = extract_bands(exp1, exp2, exp3)
    payload = dataclasses.asdict(br)
    # Per-declared-format bands (referee B1): the pooled r_lo/r_hi above stays
    # as the descriptive span; the joint headline model reads these instead.
    payload["format_bands"] = exchange_band_by_format(exp1)

    # R3.3 sensitivity check beside the pooled marginal fit: idle anchors +
    # compute-bound cubes only. Degenerate ladders (no big cubes) just omit it.
    try:
        sub = marginal_rate_compute_bound(exp3)
        payload.update(
            r_marginal_cubes=sub.r_marginal,
            r_marginal_cubes_se=sub.se_slope,
            r_marginal_cubes_r2=sub.r2,
            r_marginal_cubes_n_cells=sub.n_cells,
        )
    except ValueError:
        pass

    prov = exp2[0] if exp2 else (exp1[0] if exp1 else exp3[0])
    inputs = inputs or {}
    payload.update(
        gpu_name=prov.gpu_name,
        driver_version=prov.driver_version,
        measured_utc=prov.timestamp_utc,
        gpu_uuids=uuids,
        multi_device=len(uuids) > 1,
        inputs={
            "exp1": _input_provenance("exp1", exp1, inputs.get("exp1")),
            "exp2": _input_provenance("exp2", exp2, inputs.get("exp2")),
            "exp3": _input_provenance("exp3", exp3, inputs.get("exp3")),
        },
        n_records=dict(exp1=len(exp1), exp2=len(exp2), exp3=len(exp3)),
        analysis_git_commit=analysis_git_commit(),
        clock_strata={
            "exp1": exchange_band_strata(exp1),
            "exp2": operand_core_strata(exp2),
        },
    )

    if qblock:
        q_uuids = gather_uuids(qblock)
        if len(q_uuids) > 1:
            per_dev = {
                u: useful_band([r for r in qblock if r.gpu_uuid == u])
                for u in q_uuids
            }
            ub_block = {
                "version": 2,
                # min over devices: the safe direction for lower limits
                "r_lo": min(ub.r_lo for ub in per_dev.values()),
                "r_hi": min(ub.r_hi for ub in per_dev.values()),
                "r_lo_point": min(ub.r_lo_point for ub in per_dev.values()),
                "r_hi_point": max(ub.r_hi_point for ub in per_dev.values()),
                "alpha": next(iter(per_dev.values())).alpha,
                "per_device": {u: dataclasses.asdict(ub)
                               for u, ub in per_dev.items()},
                "device_spread": {
                    k: [min(getattr(ub, k) for ub in per_dev.values()),
                        max(getattr(ub, k) for ub in per_dev.values())]
                    for k in ("r_lo", "r_hi")
                },
            }
        else:
            ub_block = {"version": 2,
                        **dataclasses.asdict(useful_band(qblock))}
        if qblock_provenance is not None:
            ub_block["weights_provenance"] = dict(qblock_provenance)
        payload.update(
            r_useful_lo=ub_block["r_lo"],
            r_useful_hi=ub_block["r_hi"],
            useful_band=ub_block,
            qblock_uuids=q_uuids,
        )
        payload["inputs"]["qblock"] = _input_provenance(
            "qblock", qblock, inputs.get("qblock"))

    if qblock_marg:
        # Exp 8 (referee B5, third round): the MARGINAL useful edge — a covert
        # denominator, so (like qblock) a separate quantity excluded from the
        # single-device guard, min-over-devices for the multi-UUID lower limit.
        qm_uuids = gather_uuids(qblock_marg)
        if len(qm_uuids) > 1:
            per_dev = {
                u: useful_marginal_band(
                    [r for r in qblock_marg if r.gpu_uuid == u])
                for u in qm_uuids
            }
            umb_block = {
                "version": 1,
                "r_lo": min(ub.r_lo for ub in per_dev.values()),
                "r_hi": min(ub.r_hi for ub in per_dev.values()),
                "r_lo_point": min(ub.r_lo_point for ub in per_dev.values()),
                "r_hi_point": max(ub.r_hi_point for ub in per_dev.values()),
                "alpha": next(iter(per_dev.values())).alpha,
                "per_device": {u: dataclasses.asdict(ub)
                               for u, ub in per_dev.items()},
                "device_spread": {
                    k: [min(getattr(ub, k) for ub in per_dev.values()),
                        max(getattr(ub, k) for ub in per_dev.values())]
                    for k in ("r_lo", "r_hi")
                },
            }
        else:
            umb_block = {"version": 1,
                         **dataclasses.asdict(useful_marginal_band(qblock_marg))}
        if qblock_marg_provenance is not None:
            umb_block["weights_provenance"] = dict(qblock_marg_provenance)
        payload.update(
            r_useful_marg_lo=umb_block["r_lo"],
            r_useful_marg_hi=umb_block["r_hi"],
            useful_marginal_band=umb_block,
            qblock_marg_uuids=qm_uuids,
        )
        payload["inputs"]["qblock_marg"] = _input_provenance(
            "qblock_marg", qblock_marg, inputs.get("qblock_marg"))

    if exp7:
        cov_dtypes = sorted({r.cov_dtype for r in exp7 if r.cov_dtype})
        payload["r_marg_cov"] = {
            d: dataclasses.asdict(marginal_covert_cost(exp7, d))
            for d in cov_dtypes
        }
        payload["inputs"]["exp7"] = _input_provenance(
            "exp7", exp7, inputs.get("exp7"))
        payload["n_records"]["exp7"] = len(exp7)

    if len(uuids) > 1:
        per_device = {
            u: _device_bands(
                [r for r in exp1 if r.gpu_uuid == u],
                [r for r in exp2 if r.gpu_uuid == u],
                [r for r in exp3 if r.gpu_uuid == u],
            )
            for u in uuids
        }
        payload.update(per_device=per_device,
                       device_spread=_spread_over(per_device))

    return payload


def _spread_over(per_device: Mapping[str, Mapping[str, float]]) -> dict:
    """[min, max] across devices for every quantity >= 2 devices measured."""
    spread: dict[str, list[float]] = {}
    keys = {k for bands in per_device.values() for k in bands}
    for k in sorted(keys):
        vals = [bands[k] for bands in per_device.values()
                if k in bands and not isinstance(bands[k], int)]
        if len(vals) >= 2:
            spread[k] = [min(vals), max(vals)]
    return spread


def _core_bands(core: Sequence[RunRecord]) -> dict:
    """One device's core-sweep quantities: exchange corners + cheapest cell.

    ``r_lo_op`` is the cheapest zero-operand cell as a total quotient E/C. It
    is a descriptive per-device edge for the cross-device spread, NOT the
    covert denominator (that is the Exp-7 marginal edge).
    """
    clean = clean_records(core)
    dense = [r for r in clean if r.operand_dist_a == "dense_random"]
    zeros = [r for r in clean if r.operand_dist_a == "zeros"]
    out: dict = {"n_records": len(clean)}
    if dense:
        out["r_lo"], out["r_hi"] = exchange_band(dense)
    if zeros:
        cells = _median_per_cell(zeros, key=lambda r: (r.dtype, r.locality))
        out["r_lo_op"] = min(cells.values())
    return out


def core_payload(
    core: Sequence[RunRecord],
    *,
    inputs: Mapping[str, object] | None = None,
) -> dict:
    """The cross-device artifact (R2.5): per-device core bands + spread.

    The Exp-4 core sweep runs the exchange-band corners plus the covert edge on
    each of several devices (one allocation each). Unlike :func:`band_payload`,
    multiple UUIDs are the point here — the deliverable is the device-to-device
    spread, which directly measures the hardware-gap / re-execution-transfer
    rung (review C5/C6).
    """
    if not core:
        raise ValueError("core_payload: no records")
    uuids = gather_uuids(core)
    per_device = {
        u: _core_bands([r for r in core if r.gpu_uuid == u]) for u in uuids
    }
    inputs = inputs or {}
    return {
        "per_device": per_device,
        "device_spread": _spread_over(per_device),
        "gpu_uuids": uuids,
        "n_devices": len(uuids),
        "inputs": {"core": _input_provenance("core", core, inputs.get("core"))},
        "n_records": len(core),
        "analysis_git_commit": analysis_git_commit(),
    }
