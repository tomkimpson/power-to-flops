"""The measured-run record schema and its JSONL / CSV (de)serialisation.

One :class:`RunRecord` is one *measured window* (a synced burst of back-to-back
matmuls, or one duty-cycle window for Exp 3). The schema carries everything the
pure band-extraction (:mod:`powertoflops.bench.bands`) needs to compute r = E/C and
isolate each CMOS driver, plus the device-state provenance and measurement-
hygiene flags (clocks, temperature, throttling) that let the analysis exclude
contaminated windows rather than silently averaging them in.

This module is pure (no torch, no NVML): it is importable and testable anywhere,
and is what `pytest -m "not gpu"` exercises. JSONL is the primary on-disk format
(append-safe, one row per window, survives a killed job); a flat CSV mirror is
written alongside for human inspection. Both are small and git-trackable.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Operand-distribution tag for the thermal-drift reference cell (powertoflops.bench.runner
# splices one in periodically). Band extraction drops rows carrying this tag — they
# exist only to make slow drift regressable, not to widen any band.
REFERENCE_DIST = "reference_drift"

# The VALIDATED trained-weight qblock distribution tag (referee B5). Records
# from the retired random-weight benchmark carry "qblock_typical" and are
# refused by band extraction — they measured a numerically invalid forward.
QBLOCK_VALID_DIST = "qblock_pythia"

# NVML clocksThrottleReasons bits that INVALIDATE a measurement (clocks cut below
# the steady operating point): HwSlowdown | SwThermalSlowdown | HwThermalSlowdown |
# HwPowerBrakeSlowdown. Deliberately EXCLUDES SwPowerCap (0x4) and GpuIdle (0x1):
# a power-capped run at the card's limit is the *intended* full-power regime — the
# densest operands legitimately sit there — and idle is just between-kernel. Band
# extraction filters on (throttle_reasons & HARMFUL_THROTTLE_MASK), not the
# coarser ``throttled`` bool, so the high-energy cells are kept.
HARMFUL_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80


@dataclass(frozen=True)
class RunRecord:
    """One measured window. Fields group as: identity / workload / measurement /
    device-state provenance / timestamp."""

    # --- experiment identity ---
    experiment: int            # 1 | 2 | 3 (paper Sec. 3.6); 4 core, 5 pair,
                               # 6 qblock, 7 marginal (referee B1/B2)
    sweep_id: str              # uuid shared by all rows of one run_experiment call
    repeat_index: int          # which of BenchConfig.repeats this window is

    # --- workload ---
    M: int
    N: int
    K: int
    iters: int                 # matmuls in the window (Exp 1/2) or run in it (Exp 3)
    ops_nominal: int           # 2*M*N*K*iters — nominal op count C of the window
    dtype: str                 # fp32 | tf32 | fp16 | bf16 | int8
    locality: str              # cache | dram
    operand_dist_a: str        # operand-value distribution of A (the Exp-2 axis)
    operand_dist_b: str        # ... and of B
    util_frac: float | None    # Exp-3 duty cycle d in [0,1]; None for Exp 1/2

    # --- measurement ---
    energy_j: float            # window energy [J] (counter delta or power integral)
    duration_s: float          # wall time of the synced window [s]
    mean_power_w: float        # energy_j / duration_s [W]
    meter_method: str          # nvml_energy_counter | nvml_power_integral
    r_j_per_op: float          # energy_j / ops_nominal [J/op] (E/C; derived, stored)

    # --- device state at run time (provenance + hygiene) ---
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    cuda_version: str
    torch_version: str
    persistence_mode: bool
    sm_clock_mhz: int
    mem_clock_mhz: int
    power_limit_w: float
    temperature_c: float
    throttled: bool
    throttle_reasons: int      # NVML clocksThrottleReasons bitmask
    clocks_locked: bool        # did try_lock_clocks succeed this run?

    # --- timestamp ---
    timestamp_utc: str

    # Capture-time exclusion flag (powertoflops.bench.meter._flag_trimmed): high-energy
    # tail or non-physical counter miss. ALL repeats are saved; band extraction
    # drops flagged rows. Defaults False so pre-flag JSONL still loads.
    trimmed: bool = False

    # --- per-window telemetry (R2.1 clock stratification) ---
    # Sampled DURING the measured window by powertoflops.bench.meter._TelemetrySampler,
    # unlike the per-cell snapshot fields above (queried once after a cell's
    # repeats). Clock locking is privilege-denied on shared nodes, so analysis
    # stratifies by the achieved clock instead; None = telemetry absent
    # (legacy records, fake meters) — None not NaN, since NaN breaks dataclass
    # equality and strict JSON. throttle_union ORs every throttle bit seen
    # mid-window, catching transients the post-cell snapshot misses.
    sm_clock_p50_mhz: float | None = None
    sm_clock_min_mhz: float | None = None
    sm_clock_max_mhz: float | None = None
    temperature_max_c: float | None = None
    throttle_union: int = 0

    # Exp-3 idle-variation label (R2.4): which controllable-P0 state an idle
    # window measured (no_ctx | ctx | cached | post_load | cooled); busy cells
    # carry "".
    idle_variant: str = ""

    # --- Exp-7 covert dose (referee B1/B2 marginal capture) ---
    # An Exp-7 window is a fixed declared burst (described by the workload
    # fields above: dtype/shape/iters/ops_nominal are DECLARED-side) plus
    # ``cov_iters`` covert matmuls of ``cov_dtype``, idle-padded to a fixed
    # wall window; ``window_pad_s`` is the sleep remainder. The marginal
    # covert cost is the regression slope of energy_j on cov_ops_nominal
    # across doses (powertoflops.bench.bands.marginal_covert_cost). r_j_per_op keeps
    # its E/C definition but with C = DECLARED ops only — never band it.
    # Defaults keep every pre-Exp-7 JSONL loading unchanged.
    cov_dtype: str = ""
    cov_dist: str = ""
    cov_iters: int = 0
    cov_ops_nominal: int = 0
    window_pad_s: float = 0.0

    @staticmethod
    def field_names() -> list[str]:
        """Column order for the CSV mirror (dataclass field order)."""
        return [f.name for f in dataclasses.fields(RunRecord)]


def _record_to_dict(rec: RunRecord) -> dict:
    return dataclasses.asdict(rec)


def build_manifest(
    *,
    sweep_id: str,
    experiment: int,
    config: dict,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
    git_commit: str,
    hostname: str,
    slurm_job_id: str | None,
    started_utc: str,
    finished_utc: str,
    n_records: int,
    outputs: list[str],
    extra: dict | None = None,
) -> dict:
    """The per-run provenance manifest written beside each record file.

    Everything needed to re-identify a measurement without trusting file
    mtimes (review C5): which sweep, which device, which code (git commit),
    which config snapshot, where it ran, and what it produced. Pure dict
    assembly so it is testable without a GPU; the runner supplies the values.
    ``extra`` (optional) is merged under an ``"extra"`` key for run-specific
    provenance (e.g. qblock weights provenance + validation metrics, B5).
    """
    manifest = {
        "sweep_id": sweep_id,
        "experiment": experiment,
        "config": config,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "git_commit": git_commit,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "n_records": n_records,
        "outputs": outputs,
    }
    if extra is not None:
        manifest["extra"] = extra
    return manifest


def to_jsonl(records: Iterable[RunRecord], path: str | Path) -> None:
    """Append records as JSON lines (creates parent dirs; opens in append mode).

    Append, not overwrite, so a restarted or array job adds to the same file and
    a killed job keeps the rows already written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_record_to_dict(rec), sort_keys=True))
            fh.write("\n")


# Field renames across schema versions (old JSON key -> current field). The
# nominal-ops rename (review C7: int8 MACs are not FLOPs) changed the schema;
# committed pre-rename JSONL must keep loading.
_LEGACY_KEYS = {
    "flops": "ops_nominal",
    "r_j_per_flop": "r_j_per_op",
}


def read_jsonl(path: str | Path) -> list[RunRecord]:
    """Read a JSONL file back into a list of :class:`RunRecord`.

    Maps legacy field names (:data:`_LEGACY_KEYS`) so records written before a
    schema rename still load; missing new-schema fields with defaults (e.g.
    ``trimmed``) fall back to those defaults.
    """
    path = Path(path)
    out: list[RunRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for old, new in _LEGACY_KEYS.items():
                if old in row and new not in row:
                    row[new] = row.pop(old)
            out.append(RunRecord(**row))
    return out


def record_for(experiment: int, bench_dir: str | Path = "results/bench",
               *, bands: str | Path | None = None) -> Path:
    """Path to the exp<N> record that priced the committed bands.

    Figures must plot the records the manuscript's numbers came from, so the
    record is named by ``measured_bands.json``'s ``inputs`` block rather than
    picked newest-by-mtime: git does not preserve mtimes, so mtime selection
    silently varies by checkout and a fresh clone can plot a retired capture
    under a caption quoting the current one (referee review, 2026-07-25).
    Only the basename is taken, so ``--bench-dir`` still redirects to a scratch
    copy. Fails loudly rather than falling back to a guess.
    """
    bench_dir = Path(bench_dir)
    bands_path = Path(bands) if bands is not None else bench_dir / "measured_bands.json"
    if not bands_path.exists():
        raise FileNotFoundError(
            f"{bands_path} not found — figures are pinned to the records named in "
            f"its `inputs` block; regenerate it with scripts/analyze_bands.py")
    entry = (json.loads(bands_path.read_text(encoding="utf-8"))
             .get("inputs", {}).get(f"exp{experiment}") or {})
    named = entry.get("path")
    if not named:
        raise KeyError(
            f"{bands_path} names no exp{experiment} input — cannot pin the figure "
            f"record; regenerate with scripts/analyze_bands.py --exp{experiment} ...")
    path = bench_dir / Path(named).name
    if not path.exists():
        raise FileNotFoundError(f"{path} (named by {bands_path}) is missing")
    return path


def to_csv(records: Iterable[RunRecord], path: str | Path) -> None:
    """Write a flat CSV mirror (overwrites) for spreadsheet / quick-look.

    LF line endings (the csv module defaults to CRLF, which trips
    ``git diff --check`` on every committed artifact — review checklist).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = RunRecord.field_names()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow(_record_to_dict(rec))
