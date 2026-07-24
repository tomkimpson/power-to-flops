"""Sweep construction and experiment orchestration (ties workload+meter+device).

:func:`build_sweep` turns a :class:`powertoflops.config.BenchConfig` into the list of
:class:`SweepPoint` cells for one experiment; :func:`run_experiment` runs them on
the GPU and writes :class:`powertoflops.bench.records.RunRecord` rows to JSONL+CSV.

Thermal-drift control: the cell order is shuffled (seeded) and a fixed reference
cell is spliced in every ``BenchConfig.ref_every`` runs. Reference rows are
tagged ``REFERENCE_DIST`` so the pure band extraction drops them (they exist only
to make slow drift regressable post hoc).

Exp 3 (marginal rate + overhead) mixes two cell kinds: sustained-F ladder cells
(100 % busy matmul windows whose rate varies through shape alone — no duty-cycle
mixing, review C3) and idle-variation cells (``idle_variant`` set, zero ops) that
bound the controllable part of P0. Exp-3 ordering is pinned (:func:`ordered_points`):
``no_ctx`` runs before anything touches CUDA, ``ctx``/``cached`` while the card is
still cold, ``post_load``/``cooled`` after the ladder.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..config import BenchConfig, DEFAULT
from . import device as dev
from .analysis import analysis_git_commit
from .meter import Meter, measure_workload
from .records import REFERENCE_DIST, RunRecord, build_manifest, to_csv, to_jsonl
from .workload import MatmulSpec, ops_nominal_matmul, make_matmul, seconds_per_matmul

_EXP_NAME = {1: "exchange", 2: "operand", 3: "overhead", 4: "core",
             5: "pair", 6: "qblock"}

# Exp-3 idle variants pinned BEFORE the ladder (cold states, in this order) and
# AFTER it (hot then cooled). no_ctx must precede anything that touches CUDA.
_IDLE_PRE = ("no_ctx", "ctx", "cached")
_IDLE_POST = ("post_load", "cooled")


@dataclass(frozen=True)
class SweepPoint:
    """One cell of an experiment's grid (declarative; closures built at run time)."""

    experiment: int
    M: int
    N: int
    K: int
    dtype: str
    locality: str
    dist_a: str
    dist_b: str
    iters: int
    sm_clock: int | None = None
    util_frac: float | None = None
    n_buffers: int = 1          # operand sets rotated per iteration (residency)
    idle_variant: str = ""      # Exp-3 idle cell label ("" = busy cell)
    is_reference: bool = False


def _n_buffers(cfg: BenchConfig, locality: str) -> int:
    mapping = dict(cfg.locality_buffers)
    if locality not in mapping:
        raise ValueError(f"unknown locality {locality!r}; "
                         f"known: {sorted(mapping)}")
    return mapping[locality]


def _reference_point(cfg: BenchConfig) -> SweepPoint:
    """Canonical drift-reference cell (fixed dense fp16 streamed matmul).

    The *workload* is dense_random (so operand_fill is happy); the record is
    tagged REFERENCE_DIST at write time (see run_experiment) so band extraction
    drops it. Keeping these apart is what the earlier all-skipped run got wrong.
    """
    M, N, K = cfg.exp1_shape
    return SweepPoint(
        experiment=0, M=M, N=N, K=K, dtype="fp16", locality="streamed",
        dist_a="dense_random", dist_b="dense_random", iters=cfg.iters,
        n_buffers=_n_buffers(cfg, "streamed"), is_reference=True,
    )


def build_sweep(experiment: int, cfg: BenchConfig = DEFAULT.bench) -> list[SweepPoint]:
    """Build the (unshuffled) list of cells for ``experiment`` in {1,2,3,4}."""
    pts: list[SweepPoint] = []
    M1, N1, K1 = cfg.exp1_shape
    if experiment == 1:
        # Full precision x locality x operand factorial at ONE fixed shape
        # (locality = residency via n_buffers, review C4).
        clocks = cfg.dvfs_sm_clocks if cfg.dvfs_sm_clocks else (None,)
        for dtype in cfg.dtypes:
            for locality, n_buf in cfg.locality_buffers:
                for dist in cfg.operand_dists:
                    for clk in clocks:
                        pts.append(SweepPoint(
                            experiment=1, M=M1, N=N1, K=K1, dtype=dtype,
                            locality=locality, dist_a=dist, dist_b=dist,
                            iters=cfg.iters, sm_clock=clk, n_buffers=n_buf,
                        ))
    elif experiment == 2:
        # exp2_rounds copies of each dist: after the seeded shuffle each dist
        # samples several well-separated thermal/clock states.
        for _ in range(cfg.exp2_rounds):
            for dist in cfg.operand_dists:
                pts.append(SweepPoint(
                    experiment=2, M=M1, N=N1, K=K1, dtype=cfg.exp2_dtype,
                    locality=cfg.exp2_locality, dist_a=dist, dist_b=dist,
                    iters=cfg.iters,
                    n_buffers=_n_buffers(cfg, cfg.exp2_locality),
                ))
    elif experiment == 3:
        # Sustained-F ladder (every window 100% busy; F varies through shape
        # alone) + idle-variation cells bounding controllable P0.
        for M, N, K in cfg.exp3_ladder:
            pts.append(SweepPoint(
                experiment=3, M=M, N=N, K=K, dtype=cfg.exp3_dtype,
                locality=cfg.exp3_locality, dist_a="dense_random",
                dist_b="dense_random", iters=cfg.iters, util_frac=1.0,
                n_buffers=_n_buffers(cfg, cfg.exp3_locality),
            ))
        for variant in cfg.exp3_idle_variants:
            pts.append(SweepPoint(
                experiment=3, M=0, N=0, K=0, dtype=cfg.exp3_dtype,
                locality=cfg.exp3_locality, dist_a="idle", dist_b="idle",
                iters=0, util_frac=0.0, idle_variant=variant,
            ))
    elif experiment == 4:
        # Compact cross-device core: exchange-band corners + the covert edge.
        for dtype in cfg.core_dtypes:
            for locality, n_buf in cfg.locality_buffers:
                pts.append(SweepPoint(
                    experiment=4, M=M1, N=N1, K=K1, dtype=dtype,
                    locality=locality, dist_a="dense_random",
                    dist_b="dense_random", iters=cfg.iters, n_buffers=n_buf,
                ))
        for locality, n_buf in cfg.locality_buffers:
            pts.append(SweepPoint(
                experiment=4, M=M1, N=N1, K=K1, dtype=cfg.core_zeros_dtype,
                locality=locality, dist_a="zeros", dist_b="zeros",
                iters=cfg.iters, n_buffers=n_buf,
            ))
    else:
        raise ValueError(f"experiment must be 1, 2, 3 or 4; got {experiment}")
    return pts


def _ordered_with_refs(pts: list[SweepPoint], cfg: BenchConfig) -> list[SweepPoint]:
    """Shuffle cells (seeded) and splice a reference cell every ref_every runs."""
    rng = random.Random(cfg.seed)
    shuffled = pts[:]
    rng.shuffle(shuffled)
    ref = _reference_point(cfg)
    out: list[SweepPoint] = []
    for i, p in enumerate(shuffled):
        if cfg.ref_every > 0 and i % cfg.ref_every == 0:
            out.append(ref)
        out.append(p)
    return out


def _ordered_exp3(pts: list[SweepPoint], cfg: BenchConfig) -> list[SweepPoint]:
    """Exp-3 order: pre-idles, shuffled ladder (with refs), post-idles.

    A documented deviation from full randomisation: ``no_ctx`` must run before
    ANY CUDA work (it measures the context-free idle state — a reference cell
    at position 0 would already have heated the card and created a context),
    ``ctx``/``cached`` while still cold, ``post_load`` immediately after the
    ladder and ``cooled`` after the cooldown sleep.
    """
    idles = {p.idle_variant: p for p in pts if p.idle_variant}
    ladder = [p for p in pts if not p.idle_variant]
    middle = _ordered_with_refs(ladder, cfg)
    pre = [idles[v] for v in _IDLE_PRE if v in idles]
    post = [idles[v] for v in _IDLE_POST if v in idles]
    return pre + middle + post


def ordered_points(experiment: int, pts: list[SweepPoint],
                   cfg: BenchConfig) -> list[SweepPoint]:
    """Execution order for a sweep (Exp 3 has pinned idle cells; see above)."""
    if experiment == 3:
        return _ordered_exp3(pts, cfg)
    return _ordered_with_refs(pts, cfg)


def _calibrate_iters(sp: SweepPoint, cfg: BenchConfig, device, generator,
                     target_s: float) -> int:
    """Choose the matmul count that fills ``target_s`` of wall time for this cell.

    Equalises window length across dtypes (which differ ~30x in throughput) so the
    NVML energy counter spans many updates. Falls back to ``cfg.iters`` if
    target_window_s <= 0 or timing fails.
    """
    if cfg.target_window_s <= 0:
        return max(sp.iters, 1)
    spec = MatmulSpec(sp.M, sp.N, sp.K, sp.dtype, "dense_random", "dense_random",
                      iters=cfg.calib_matmuls, n_buffers=sp.n_buffers)
    try:
        t_one = seconds_per_matmul(spec, n_time=cfg.calib_matmuls,
                                   device=device, generator=generator)
    except Exception:
        return max(sp.iters, 1)
    if t_one <= 0:
        return max(sp.iters, 1)
    return max(1, min(cfg.max_iters, int(round(target_s / t_one))))


def _build_work(sp: SweepPoint, cfg: BenchConfig, device, generator):
    """Return (work_closure, ops_per_window, iters) for a cell.

    Idle cells (``sp.idle_variant``) do zero ops: the "work" is holding the
    cell's controllable-P0 state for ``exp3_idle_window_s`` while the meter
    brackets it. The ``cached`` variant additionally holds a full streamed
    operand pool so HBM occupancy is part of the measured state.
    """
    if sp.idle_variant:
        window_s = cfg.exp3_idle_window_s
        pool: list = []
        if sp.idle_variant == "cached":
            M, N, K = cfg.exp1_shape
            n_buf = _n_buffers(cfg, cfg.exp3_locality)
            spec = MatmulSpec(M, N, K, cfg.exp3_dtype, iters=0, n_buffers=n_buf)
            pool.append(make_matmul(spec, device=device, generator=generator))

        def idle_work() -> None:
            _ = pool   # keep the operand pool alive through the window
            time.sleep(window_s)

        return idle_work, 0, 0

    iters = _calibrate_iters(sp, cfg, device, generator, cfg.target_window_s)
    spec = MatmulSpec(sp.M, sp.N, sp.K, sp.dtype, sp.dist_a, sp.dist_b,
                      iters=iters, n_buffers=sp.n_buffers)
    work = make_matmul(spec, device=device, generator=generator)
    ops = ops_nominal_matmul(sp.M, sp.N, sp.K) * iters
    return work, ops, iters


def run_experiment(
    experiment: int,
    out_dir: str | Path,
    cfg: BenchConfig = DEFAULT.bench,
    device_index: int = 0,
) -> Path:
    """Run all cells of ``experiment`` on the GPU; write JSONL+CSV+manifest.

    Returns the JSONL path. The sidecar ``*_manifest.json`` carries the run's
    provenance (device UUID, config snapshot, git commit, host, SLURM job,
    timestamps) so the analysis never has to trust file mtimes (review C5).
    """
    out_dir = Path(out_dir)
    sweep_id = uuid.uuid4().hex[:12]
    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"cuda:{device_index}")
    # Created lazily: the Exp-3 "no_ctx" idle cell must be measured BEFORE any
    # CUDA context exists in this process, and constructing a CUDA generator
    # initialises one. Meter is pure NVML (no CUDA context).
    generator: torch.Generator | None = None

    meter = Meter(device_index)
    pts = ordered_points(experiment, build_sweep(experiment, cfg), cfg)
    records: list[RunRecord] = []
    try:
        for sp in pts:
            if generator is None and sp.idle_variant != "no_ctx":
                generator = torch.Generator(device=device).manual_seed(cfg.seed)
            if sp.idle_variant == "cooled":
                time.sleep(cfg.exp3_cooldown_s)
            clocks_locked = (
                dev.try_lock_clocks(device_index, sp.sm_clock)
                if sp.sm_clock is not None else False
            )
            try:
                work, ops, iters = _build_work(sp, cfg, device, generator)
                # Idle cells: no warmup (nothing to autotune) and no high-tail
                # exclusion — the overhead band wants the FULL observed range.
                samples = measure_workload(
                    meter, work,
                    warmup=0 if sp.idle_variant else cfg.warmup,
                    repeats=cfg.repeats,
                    discard_frac=0.0 if sp.idle_variant else cfg.discard_frac,
                )
            except Exception as exc:
                # A cell may fail (dtype unsupported on this silicon, OOM, ...).
                # Skip it rather than losing the whole experiment; the gap is
                # visible in the records (e.g. int8 absent from the band).
                print(f"  SKIP cell exp{sp.experiment} {sp.dtype} {sp.locality} "
                      f"dist={sp.dist_a}: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                if sp.sm_clock is not None:
                    dev.reset_locked_clocks(device_index)
                continue
            info = dev.query_device(device_index)
            throttled = dev.is_throttled(device_index)
            stamp = datetime.now(timezone.utc).isoformat()
            for k, s in enumerate(samples):
                records.append(RunRecord(
                    experiment=sp.experiment if not sp.is_reference else 0,
                    sweep_id=sweep_id, repeat_index=k,
                    M=sp.M, N=sp.N, K=sp.K, iters=iters, ops_nominal=ops,
                    dtype=sp.dtype, locality=sp.locality,
                    operand_dist_a=(REFERENCE_DIST if sp.is_reference else sp.dist_a),
                    operand_dist_b=(REFERENCE_DIST if sp.is_reference else sp.dist_b),
                    util_frac=sp.util_frac,
                    idle_variant=sp.idle_variant,
                    energy_j=s.energy_j, duration_s=s.duration_s,
                    mean_power_w=s.mean_power_w, meter_method=s.method,
                    r_j_per_op=(s.energy_j / ops) if ops > 0 else float("nan"),
                    sm_clock_p50_mhz=s.sm_clock_p50_mhz,
                    sm_clock_min_mhz=s.sm_clock_min_mhz,
                    sm_clock_max_mhz=s.sm_clock_max_mhz,
                    temperature_max_c=s.temperature_max_c,
                    throttle_union=s.throttle_union,
                    gpu_name=info.name, gpu_uuid=info.uuid,
                    driver_version=info.driver_version, cuda_version=info.cuda_version,
                    torch_version=torch.__version__,
                    persistence_mode=info.persistence_mode,
                    sm_clock_mhz=info.sm_clock_mhz, mem_clock_mhz=info.mem_clock_mhz,
                    power_limit_w=info.power_limit_w, temperature_c=info.temperature_c,
                    throttled=throttled, throttle_reasons=info.throttle_reasons,
                    clocks_locked=clocks_locked, timestamp_utc=stamp,
                    trimmed=s.trimmed,
                ))
            if sp.sm_clock is not None:
                dev.reset_locked_clocks(device_index)
    finally:
        meter.close()

    base = out_dir / f"exp{experiment}_{_EXP_NAME[experiment]}_{sweep_id}"
    to_jsonl(records, base.with_suffix(".jsonl"))
    to_csv(records, base.with_suffix(".csv"))

    manifest = build_manifest(
        sweep_id=sweep_id,
        experiment=experiment,
        config=dataclasses.asdict(cfg),
        gpu_uuid=records[-1].gpu_uuid if records else "unknown",
        gpu_name=records[-1].gpu_name if records else "unknown",
        driver_version=records[-1].driver_version if records else "unknown",
        git_commit=analysis_git_commit(),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(records),
        outputs=[base.with_suffix(".jsonl").name, base.with_suffix(".csv").name],
    )
    manifest_path = base.parent / f"{base.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return base.with_suffix(".jsonl")
