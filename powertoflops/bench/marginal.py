"""Exp 7: controlled marginal covert-cost capture (referee B1/B2).

The floor's covert denominator must be the MARGINAL energy of an extra covert
op — the derivative dE/dC_cov holding everything else fixed — not the total
quotient E/C, which amortises baseline power over the ops (referee B2).
Experiment 3 cannot identify that derivative (its slope is a between-shape
regression); this experiment does, by intervening on the dose alone:

    every measured window = the SAME calibrated fp16 declared burst
                          + cov_iters covert matmuls of ONE covert format
                          + an idle pad to the SAME fixed wall window,

swept over covert busy fractions (including zero — the intercept anchor) for
each covert format. Declared work, shapes, operand pools, buffers, and window
length are all held fixed; only the dose varies, so the regression slope of
window energy on covert nominal ops (powertoflops.bench.bands.marginal_covert_cost)
is the marginal covert cost. Clock locking is privilege-denied on the platform
(driver-denied), so per-window clock telemetry is recorded and the
analysis verifies a common achieved-clock stratum post hoc; the fixed declared
anchor keeps the governor far better pinned than Exp 2's disjoint workload
families (referee B6).

Cell order is the campaign convention: seeded shuffle plus the standard
drift-reference splice (powertoflops.bench.runner machinery), so slow thermal drift is
regressable. Records land as :class:`powertoflops.bench.records.RunRecord` rows with
the covert-dose fields set; the workload fields describe the DECLARED burst.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
import uuid as uuid_mod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import A100_DENSE_PEAK_OPS, BenchConfig, DEFAULT
from .analysis import analysis_git_commit
from .records import REFERENCE_DIST, RunRecord, build_manifest, to_csv, to_jsonl

MARGINAL_EXPERIMENT = 7

# Ceiling on declared + covert busy time as a fraction of the window: the
# idle pad must never underflow, or the window stretches and the fixed-T_w
# design (equal baseline energy in every window) is broken.
MAX_BUSY_FRAC = 0.9


@dataclass(frozen=True)
class MarginalPoint:
    """One Exp-7 cell: a covert format at one dose fraction."""

    cov_dtype: str
    cov_frac: float            # target covert busy fraction of the window
    is_reference: bool = False


def build_dose_sweep(
    arm_names: Sequence[str],
    cov_fracs: Sequence[float],
    declared_frac: float,
) -> list[MarginalPoint]:
    """Format-agnostic dose grid: covert arms x dose fractions.

    The pure core shared by Exp 7 (arms = covert *formats*,
    :func:`build_marginal_sweep`) and qblock-marginal (arms = qblock *cells*,
    :func:`powertoflops.bench.qblock.run_qblock_marginal`). Every arm carries its own
    zero-dose anchor (tagged with the arm label so each arm's regression has its
    intercept cell). ``cov_fracs`` must include the zero dose; raises when the
    worst-case busy fraction would underflow the idle pad.
    """
    worst = declared_frac + max(cov_fracs)
    if worst > MAX_BUSY_FRAC:
        raise ValueError(
            f"declared_frac {declared_frac} + max dose {max(cov_fracs)} = "
            f"{worst:.2f} exceeds MAX_BUSY_FRAC {MAX_BUSY_FRAC} — the idle pad "
            f"would underflow")
    if min(cov_fracs) != 0.0:
        raise ValueError("cov_fracs must include the zero dose "
                         "(the intercept anchor)")
    return [MarginalPoint(cov_dtype=a, cov_frac=f)
            for a in arm_names for f in cov_fracs]


def build_marginal_sweep(cfg: BenchConfig = DEFAULT.bench) -> list[MarginalPoint]:
    """The (unshuffled) Exp-7 grid: covert formats x dose fractions.

    Every covert format carries its own zero-dose anchor (tagged with the
    format, so each arm's regression has its intercept cell). Raises when the
    grid could underflow the pad or a covert format has no configured peak.
    """
    for dtype in cfg.marg_cov_dtypes:
        if dtype not in A100_DENSE_PEAK_OPS:
            raise ValueError(f"no configured peak for covert dtype {dtype!r}")
    return build_dose_sweep(cfg.marg_cov_dtypes, cfg.marg_cov_fracs,
                            cfg.marg_declared_frac)


def capacity_frac_of_dose(*, cov_frac: float, declared_frac: float) -> float:
    """Dose feasibility: covert busy share over the residual (1 - h) share.

    Covert work at achieved rate <= peak occupies cov_frac of the window's
    wall time, so its op count is at most cov_frac * F_peak * T_w; the
    capacity clip allows (1 - h) * F_peak * T_w. The ratio is therefore
    bounded by cov_frac / (1 - declared_frac) independent of the format —
    < 1 for every configured dose, which the runner asserts.
    """
    return cov_frac / (1.0 - declared_frac)


# ---- GPU harness -------------------------------------------------------------


def run_marginal(out_dir: str | Path, cfg: BenchConfig = DEFAULT.bench,
                 device_index: int = 0) -> Path:
    """Run the Exp-7 dose sweep; write exp7_marginal_<id>.jsonl/.csv + manifest.

    Calibrates the declared burst once and each covert format's per-matmul
    time once, then measures every cell (seeded shuffle + drift-reference
    splices) with the standard warmup/repeats/trim-flag conventions. Feed the
    JSONL to scripts/analyze_bands.py --exp7.
    """
    import random

    import torch

    from . import device as dev
    from .meter import Meter, measure_workload
    from .workload import MatmulSpec, make_matmul, ops_nominal_matmul, \
        seconds_per_matmul

    out_dir = Path(out_dir)
    sweep_id = uuid_mod.uuid4().hex[:12]
    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"cuda:{device_index}")
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    meter = Meter(device_index)

    n_buf = dict(cfg.locality_buffers)[cfg.marg_declared_locality]
    Md, Nd, Kd = cfg.marg_declared_shape
    Mc, Nc, Kc = cfg.marg_cov_shape

    records: list[RunRecord] = []
    try:
        # --- calibrate the fixed declared burst (once) -----------------------
        spec_d = MatmulSpec(Md, Nd, Kd, cfg.marg_declared_dtype,
                            dist_a=cfg.marg_declared_dist,
                            dist_b=cfg.marg_declared_dist,
                            iters=cfg.calib_matmuls, n_buffers=n_buf)
        t_decl = seconds_per_matmul(spec_d, n_time=cfg.calib_matmuls,
                                    device=device, generator=generator)
        n_decl = max(1, int(round(cfg.marg_declared_frac * cfg.marg_window_s
                                  / t_decl)))
        work_decl = make_matmul(dataclasses.replace(spec_d, iters=n_decl),
                                device=device, generator=generator)
        ops_decl = ops_nominal_matmul(Md, Nd, Kd) * n_decl

        # --- calibrate each covert format's per-matmul time (once) ----------
        t_cov: dict[str, float] = {}
        for dtype in cfg.marg_cov_dtypes:
            spec_c = MatmulSpec(Mc, Nc, Kc, dtype,
                                dist_a=cfg.marg_cov_dist,
                                dist_b=cfg.marg_cov_dist,
                                iters=cfg.calib_matmuls)
            t_cov[dtype] = seconds_per_matmul(
                spec_c, n_time=cfg.calib_matmuls, device=device,
                generator=generator)

        # --- reference cell (campaign drift convention) ----------------------
        spec_ref = MatmulSpec(Md, Nd, Kd, "fp16", iters=cfg.calib_matmuls,
                              n_buffers=n_buf)
        t_ref = seconds_per_matmul(spec_ref, n_time=cfg.calib_matmuls,
                                   device=device, generator=generator)
        n_ref = max(1, min(cfg.max_iters,
                           int(round(cfg.target_window_s / t_ref))))
        work_ref = make_matmul(dataclasses.replace(spec_ref, iters=n_ref),
                               device=device, generator=generator)
        ops_ref = ops_nominal_matmul(Md, Nd, Kd) * n_ref

        # --- execution order: seeded shuffle + reference splices -------------
        pts = build_marginal_sweep(cfg)
        rng = random.Random(cfg.seed)
        rng.shuffle(pts)
        ordered: list[MarginalPoint] = []
        for i, p in enumerate(pts):
            if cfg.ref_every > 0 and i % cfg.ref_every == 0:
                ordered.append(MarginalPoint("", 0.0, is_reference=True))
            ordered.append(p)

        def _emit(sp: MarginalPoint, samples, *, iters, ops, cov_iters,
                  cov_ops, pads) -> None:
            info = dev.query_device(device_index)
            throttled = dev.is_throttled(device_index)
            stamp = datetime.now(timezone.utc).isoformat()
            for k, s in enumerate(samples):
                records.append(RunRecord(
                    experiment=0 if sp.is_reference else MARGINAL_EXPERIMENT,
                    sweep_id=sweep_id, repeat_index=k,
                    M=Md, N=Nd, K=Kd, iters=iters, ops_nominal=ops,
                    dtype=cfg.marg_declared_dtype,
                    locality=cfg.marg_declared_locality,
                    operand_dist_a=(REFERENCE_DIST if sp.is_reference
                                    else cfg.marg_declared_dist),
                    operand_dist_b=(REFERENCE_DIST if sp.is_reference
                                    else cfg.marg_declared_dist),
                    util_frac=None,
                    energy_j=s.energy_j, duration_s=s.duration_s,
                    mean_power_w=s.mean_power_w, meter_method=s.method,
                    r_j_per_op=(s.energy_j / ops) if ops > 0 else float("nan"),
                    sm_clock_p50_mhz=s.sm_clock_p50_mhz,
                    sm_clock_min_mhz=s.sm_clock_min_mhz,
                    sm_clock_max_mhz=s.sm_clock_max_mhz,
                    temperature_max_c=s.temperature_max_c,
                    throttle_union=s.throttle_union,
                    gpu_name=info.name, gpu_uuid=info.uuid,
                    driver_version=info.driver_version,
                    cuda_version=info.cuda_version,
                    torch_version=torch.__version__,
                    persistence_mode=info.persistence_mode,
                    sm_clock_mhz=info.sm_clock_mhz,
                    mem_clock_mhz=info.mem_clock_mhz,
                    power_limit_w=info.power_limit_w,
                    temperature_c=info.temperature_c,
                    throttled=throttled,
                    throttle_reasons=info.throttle_reasons,
                    clocks_locked=False, timestamp_utc=stamp,
                    trimmed=s.trimmed,
                    cov_dtype=sp.cov_dtype,
                    cov_dist=(cfg.marg_cov_dist if sp.cov_dtype else ""),
                    cov_iters=cov_iters, cov_ops_nominal=cov_ops,
                    window_pad_s=(pads[cfg.warmup + k]
                                  if pads and cfg.warmup + k < len(pads)
                                  else 0.0),
                ))

        for sp in ordered:
            if sp.is_reference:
                samples = measure_workload(meter, work_ref, warmup=1,
                                           repeats=cfg.repeats,
                                           discard_frac=cfg.discard_frac)
                _emit(sp, samples, iters=n_ref, ops=ops_ref, cov_iters=0,
                      cov_ops=0, pads=[])
                continue

            cap = capacity_frac_of_dose(cov_frac=sp.cov_frac,
                                        declared_frac=cfg.marg_declared_frac)
            assert cap < 1.0, (
                f"dose {sp.cov_frac} infeasible next to declared "
                f"{cfg.marg_declared_frac}")
            n_cov = (0 if sp.cov_frac == 0.0 else
                     max(1, int(round(sp.cov_frac * cfg.marg_window_s
                                      / t_cov[sp.cov_dtype]))))
            cov_ops = ops_nominal_matmul(Mc, Nc, Kc) * n_cov
            work_cov = (make_matmul(
                MatmulSpec(Mc, Nc, Kc, sp.cov_dtype,
                           dist_a=cfg.marg_cov_dist,
                           dist_b=cfg.marg_cov_dist, iters=n_cov),
                device=device, generator=generator) if n_cov else None)

            pads: list[float] = []

            def window() -> None:
                t0 = time.perf_counter()
                work_decl()
                if work_cov is not None:
                    work_cov()
                torch.cuda.synchronize()
                pad = cfg.marg_window_s - (time.perf_counter() - t0)
                pads.append(max(0.0, pad))
                if pad <= 0.0:
                    print(f"  WARNING: window underflowed pad by {-pad:.2f}s "
                          f"(dose {sp.cov_dtype} f={sp.cov_frac})", flush=True)
                else:
                    time.sleep(pad)

            print(f"  cell cov={sp.cov_dtype:>5s} f={sp.cov_frac:.2f} "
                  f"n_cov={n_cov}", flush=True)
            samples = measure_workload(meter, window, warmup=cfg.warmup,
                                       repeats=cfg.repeats,
                                       discard_frac=cfg.discard_frac)
            _emit(sp, samples, iters=n_decl, ops=ops_decl, cov_iters=n_cov,
                  cov_ops=cov_ops, pads=pads)
            del work_cov
            torch.cuda.empty_cache()
    finally:
        meter.close()

    base = out_dir / f"exp7_marginal_{sweep_id}"
    to_jsonl(records, base.with_suffix(".jsonl"))
    to_csv(records, base.with_suffix(".csv"))

    manifest = build_manifest(
        sweep_id=sweep_id, experiment=MARGINAL_EXPERIMENT,
        config=dataclasses.asdict(cfg),
        gpu_uuid=records[-1].gpu_uuid if records else "unknown",
        gpu_name=records[-1].gpu_name if records else "unknown",
        driver_version=records[-1].driver_version if records else "unknown",
        git_commit=analysis_git_commit(), hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(records),
        outputs=[base.with_suffix(".jsonl").name,
                 base.with_suffix(".csv").name],
    )
    (base.parent / f"{base.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return base.with_suffix(".jsonl")
