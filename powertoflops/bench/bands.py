"""Pure extraction of the verification-floor bands from measured records.

These functions turn lists of :class:`powertoflops.bench.records.RunRecord` into the
measured band widths that ground the closed-form floor (paper Sec. 3.6):

    Exp 1 -> (r_lo, r_hi)                       the declared exchange-rate band
    Exp 2 -> (r_lo_op, r_hi_op, w0)             the descriptive operand envelope
    Exp 3 -> OverheadBand                       observed-idle band + marginal-r fit

    Exp 2's edges are total quotients E/C and are DESCRIPTIVE only: they price
    no bound. The covert denominator r_lo^cov is the Exp-7 marginal dose slope
    (:func:`marginal_covert_cost`, reached via
    :func:`powertoflops.measured.covert_edge`), never an Exp-2 edge.

    The per-record ``r_j_per_op`` is the total quotient E/C (a secondary,
    clearly-labelled quantity); the theory's marginal rate dP/dF is estimated
    separately by :func:`marginal_rate` (review C2).

They are pure functions of record lists — no GPU, no NVML — so the whole
analysis is unit-testable on synthetic records (tests/test_bands.py) and feeds
:func:`powertoflops.measured.config_with_measured` to print the first *measured* beta(0).

Aggregation convention: harmful-throttle, drift-reference, and idle-stalled
windows are dropped (the last by a per-cell power guard, see
:func:`_drop_stalled`), then each sweep *cell* (a fixed driver setting) is
collapsed to its median r over repeats before the band edges are taken.
Median-then-min/max plus the power guard is robust to the occasional preempted
or thermally-excursed window the trim in measure_workload missed.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from .records import (
    HARMFUL_THROTTLE_MASK,
    QBLOCK_VALID_DIST,
    REFERENCE_DIST,
    RunRecord,
)


@dataclass(frozen=True)
class BandResult:
    """All measured bands plus the provenance needed for powertoflops.measured."""

    # Exp 1: declared exchange-rate band [J/op].
    r_lo: float
    r_hi: float

    # Exp 2: the descriptive operand envelope (total quotients E/C). These
    # price NO bound -- the covert denominator is the Exp-7 marginal edge.
    r_lo_op: float             # operand-envelope lower edge = cheapest operand cell
    r_hi_op: float             # operand-envelope upper edge
    w0: float                  # operand core ratio r_hi_op / r_lo_op

    # Exp 3: overhead band [W] + affine fit diagnostics. The band is the
    # OBSERVED idle range widened by IDLE_TOL_FRAC; the fit intercept is a
    # separate diagnostic, never the band (review C3).
    P0_lo: float
    P0_hi: float
    P0_fit_intercept: float    # affine-fit intercept [W] (diagnostic, not band)
    P0_fit_se: float           # 1 standard error of that intercept [W]
    r_marginal: float          # Exp-3 fitted dP/dF [J/op] — the theory's marginal r
    affine_r2: float           # R^2 of the affine fit (convexity flag if < 1)

    # Channel companion: re-execution declared-rate std [J/op].
    sigma_dec: float


def _clean_r(records: Iterable[RunRecord]) -> list[RunRecord]:
    """Drop harmfully-throttled, drift-reference, trimmed, and non-physical-r rows.

    "Harmful" = thermal / HW-slowdown throttling (HARMFUL_THROTTLE_MASK), NOT the
    SW power cap, which is the intended full-power regime the densest operands sit
    in — filtering the bool ``throttled`` would wrongly discard exactly the
    high-energy cells that set the upper band edges. A non-positive r_j_per_op is
    a counter-miss, not a measurement, so it is excluded too. ``trimmed`` rows are
    the pre-specified high-tail exclusion, applied HERE rather than at capture:
    every repeat survives on disk (powertoflops.bench.meter saves all samples flagged),
    so the exclusion is auditable and re-runnable with a different discard_frac.
    """
    return [
        r for r in records
        if not (r.throttle_reasons & HARMFUL_THROTTLE_MASK)
        and r.operand_dist_a != REFERENCE_DIST
        and not r.trimmed
        and math.isfinite(r.r_j_per_op) and r.r_j_per_op > 0.0
    ]


# A compute-bound window draws near its cell's peak power; a window that draws far
# less spent most of its wall time idle/stalled (host preemption, a context hiccup,
# contention on a shared node), so the energy-counter bracket folded idle draw into
# E and r = E/C is inflated. Such a window is not the compute-bound measurement the
# protocol assumes, so it is dropped relative to the cell's peak.
MIN_COMPUTE_POWER_FRAC = 0.5


def _cell_key(r: RunRecord) -> tuple:
    """Sweep-cell identity (one fixed driver setting), across all experiments.

    Shape is part of the key: the Exp-3 ladder (R2.4) varies sustained F
    through shape at otherwise fixed settings, and qblock cells differ only in
    (seq, batch) -> (M, N). idle_variant separates the Exp-3 idle cells, and
    the Exp-7 covert-dose fields separate its dose cells (constant defaults
    on every other experiment). Legacy records are unaffected (their shape
    was a function of locality and idle_variant / cov_* are defaults).
    """
    return (r.experiment, r.M, r.N, r.K, r.dtype, r.locality, r.sm_clock_mhz,
            r.operand_dist_a, r.util_frac, r.idle_variant,
            r.cov_dtype, r.cov_iters)


def _drop_stalled(records: Sequence[RunRecord]) -> list[RunRecord]:
    """Drop idle-contaminated windows: mean power < frac x the cell's peak power.

    The reference is the cell's *peak* mean power (its most-saturated window), so a
    legitimate few-percent within-cell spread is kept while a stalled window (often
    ~0.2x peak) is removed. A singleton cell is its own peak and is never dropped.
    """
    by_cell: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_cell[_cell_key(r)].append(r)
    kept: list[RunRecord] = []
    for rows in by_cell.values():
        peak = max(r.mean_power_w for r in rows)
        threshold = MIN_COMPUTE_POWER_FRAC * peak
        kept.extend(r for r in rows if r.mean_power_w >= threshold)
    return kept


def clean_records(records: Iterable[RunRecord]) -> list[RunRecord]:
    """Band-extraction record filter: row-wise hygiene then a per-cell stall guard.

    Applies :func:`_clean_r` (drop harmful-throttle, drift-reference, non-physical
    rows) then :func:`_drop_stalled` (drop idle-contaminated windows relative to
    each cell's peak power). Plotting scripts use this so their displayed points
    match the bands exactly, rather than re-implementing (and drifting from) the
    filter.
    """
    return _drop_stalled(_clean_r(records))


def _median_per_cell(records: Sequence[RunRecord], key) -> dict[object, float]:
    """Collapse records to median r_j_per_op per cell defined by ``key(rec)``."""
    buckets: dict[object, list[float]] = defaultdict(list)
    for rec in records:
        buckets[key(rec)].append(rec.r_j_per_op)
    return {k: float(np.median(v)) for k, v in buckets.items()}


def _exchange_cell(r: RunRecord) -> tuple:
    """Exp-1 cell identity: one driver setting of the factorial."""
    return (r.dtype, r.locality, r.sm_clock_mhz, r.operand_dist_a)


def exchange_band(exp1: Sequence[RunRecord]) -> tuple[float, float]:
    """Exp 1: (r_lo, r_hi) over precision x locality x operand (x DVFS) cells.

    Each cell is a (dtype, locality, sm_clock, operand dist) setting; the band
    edges are the min and max of the per-cell median r [J/op] — the full
    driver-settings envelope of the R2.2 factorial. (Pre-R2 sweeps were
    dense_random-only, so their band is unchanged by the operand term.)
    Exp 2's operand core still isolates the pure activity factor at fixed
    precision/locality. This is the first term of beta(0).
    """
    clean = clean_records(exp1)
    if not clean:
        raise ValueError("exchange_band: no un-throttled Exp-1 records")
    cells = _median_per_cell(clean, key=_exchange_cell)
    rs = list(cells.values())
    return min(rs), max(rs)


def exchange_band_by_format(exp1: Sequence[RunRecord]) -> dict[str, dict]:
    """Exp 1: per-declared-format exchange bands {dtype: {r_lo, r_hi, n_cells}}.

    The pooled :func:`exchange_band` spans the whole factorial, so its min and
    max generally come from *different* formats — a descriptive span, never a
    declared band for the floor. A declared workload runs at ONE format, so
    the joint feasibility model (referee B1; powertoflops.joint_model) prices the
    honest interpretation inside that format's own band, taken here over that
    format's cells only (same cleaning and median-per-cell convention).
    """
    clean = clean_records(exp1)
    if not clean:
        raise ValueError("exchange_band_by_format: no un-throttled Exp-1 records")
    out: dict[str, dict] = {}
    for dtype in sorted({r.dtype for r in clean}):
        cells = _median_per_cell([r for r in clean if r.dtype == dtype],
                                 key=_exchange_cell)
        rs = list(cells.values())
        out[dtype] = {"r_lo": min(rs), "r_hi": max(rs), "n_cells": len(cells)}
    return out


# Clock-stratification bin width [MHz] (R2.1): clock locking is privilege-
# denied on shared nodes, so bands are additionally reported per achieved-clock
# stratum. 45 MHz = 3x the A100's 15 MHz P-state step — fine enough to separate
# boost plateaus, coarse enough that strata keep multiple cells.
CLOCK_BIN_MHZ = 45


def clock_stratum(r: RunRecord, bin_mhz: int = CLOCK_BIN_MHZ) -> int:
    """Achieved-clock stratum label [MHz] for a record.

    Prefers the per-window median clock (meter._TelemetrySampler); legacy
    records without telemetry fall back to the per-cell snapshot clock.
    """
    clock = (r.sm_clock_p50_mhz if r.sm_clock_p50_mhz is not None
             else float(r.sm_clock_mhz))
    return int(round(clock / bin_mhz) * bin_mhz)


def _by_stratum(records: Sequence[RunRecord]) -> dict[int, list[RunRecord]]:
    out: dict[int, list[RunRecord]] = defaultdict(list)
    for r in records:
        out[clock_stratum(r)].append(r)
    return out


def exchange_band_strata(exp1: Sequence[RunRecord]) -> dict[int, dict]:
    """Per-achieved-clock-stratum exchange bands (R2.1 fallback reporting).

    Returns {stratum_mhz: {r_lo, r_hi, n_cells, n_records}}. The cross-stratum
    spread is the DVFS-sensitivity diagnostic the pooled band hides.
    """
    out: dict[int, dict] = {}
    for stratum, rows in sorted(_by_stratum(clean_records(exp1)).items()):
        cells = _median_per_cell(rows, key=_exchange_cell)
        rs = list(cells.values())
        out[stratum] = {"r_lo": min(rs), "r_hi": max(rs),
                        "n_cells": len(cells), "n_records": len(rows)}
    return out


def operand_core_strata(exp2: Sequence[RunRecord]) -> dict[int, dict]:
    """Per-achieved-clock-stratum operand cores (R2.3).

    Returns {stratum_mhz: {r_lo_op, r_hi_op, w0, n_cells, n_records}}.
    """
    out: dict[int, dict] = {}
    for stratum, rows in sorted(_by_stratum(clean_records(exp2)).items()):
        cells = _median_per_cell(rows, key=lambda r: r.operand_dist_a)
        rs = list(cells.values())
        r_lo, r_hi = min(rs), max(rs)
        out[stratum] = {"r_lo_op": r_lo, "r_hi_op": r_hi, "w0": r_hi / r_lo,
                        "n_cells": len(cells), "n_records": len(rows)}
    return out


# Per-cell repeatability floor for the useful-band LCBs: NVML board-power
# readings repeat to ~0.5% on these A100s, so a tighter observed sd is meter
# resolution, not workload stability (same convention as the matched-pair
# sigma floor, powertoflops.bench.pair).
USEFUL_SIGMA_FLOOR_FRAC = 0.005

# Minimum untrimmed repeats per qblock cell for a meaningful Student-t LCB.
USEFUL_MIN_REPEATS = 5


@dataclass(frozen=True)
class UsefulBand:
    """Useful-operand band from the validated qblock capture (referee B5).

    Both endpoints are one-sided lower confidence limits at level 1 - alpha
    (Student-t on the per-cell mean r): beta places the covert cost in the
    DENOMINATOR, so any endpoint used as a covert price must not be
    overstated — that holds for the floor (rung 4) and equally for the
    band-top assumption price (rung 5). The point estimates are carried for
    disclosure; ``strata`` reports per-achieved-clock descriptive medians
    (clock locking is privilege-denied on the platform, B6 convention).
    """

    r_lo: float                # min over cells of the per-cell LCB [J/op]
    r_hi: float                # LCB of the cell with the largest mean [J/op]
    r_lo_point: float          # min cell mean (descriptive)
    r_hi_point: float          # max cell mean (descriptive)
    alpha: float               # one-sided level of the LCBs
    n_cells: int
    per_cell: dict             # {"MxNxK": {mean, sd, n, lcb}}
    strata: dict = field(default_factory=dict)
                               # {stratum_mhz: {r_lo, r_hi, n_records}}


def useful_band(
    qblock: Sequence[RunRecord], *, alpha: float = 0.05,
) -> UsefulBand:
    """Useful-operand band (:class:`UsefulBand`) over qblock (seq, batch) cells.

    Only records from the VALIDATED trained-weight benchmark are accepted
    (operand dist ``qblock_pythia``): the retired random-weight capture ran a
    numerically invalid forward (referee B5) and its records can never price
    a band again. Cells are keyed by shape (M=seq, N=batch, K=d_model); each
    needs >= USEFUL_MIN_REPEATS untrimmed repeats for the LCB.

    ESTIMAND NOTE (referee B2/B5). ``r_j_per_op`` is a standalone total quotient
    E/C, so this band folds in baseline power the covert work does not draw
    incrementally — anti-conservative as a covert-cost DENOMINATOR. It is
    therefore RETIRED as the ladder denominator and kept only for disclosure /
    back-compat. The shipped fix is :func:`useful_marginal_band` (qblock-marginal
    dose-response, :func:`powertoflops.bench.qblock.run_qblock_marginal`), whose slope
    cancels the baseline; :func:`powertoflops.ladder.build_ladder` prices rungs 4-5 off
    that marginal edge via :func:`powertoflops.measured.useful_marginal_edge`.
    """
    clean = clean_records(qblock)
    if not clean:
        raise ValueError("useful_band: no usable qblock records")
    bad = sorted({r.operand_dist_a for r in clean
                  if r.operand_dist_a != QBLOCK_VALID_DIST})
    if bad:
        raise ValueError(
            f"useful_band: records carry operand dist {bad}, expected "
            f"{QBLOCK_VALID_DIST!r} — records from the retired random-weight "
            f"qblock (invalid forward, referee B5) cannot price a band")

    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in clean:
        buckets[(r.M, r.N, r.K)].append(r.r_j_per_op)

    per_cell: dict[str, dict] = {}
    for (M, N, K), rs in sorted(buckets.items()):
        n = len(rs)
        if n < USEFUL_MIN_REPEATS:
            raise ValueError(
                f"useful_band: cell {M}x{N}x{K} has only {n} untrimmed "
                f"repeats; >= {USEFUL_MIN_REPEATS} repeats are required for "
                f"the one-sided LCB")
        mean = float(np.mean(rs))
        sd = max(float(np.std(rs, ddof=1)), USEFUL_SIGMA_FLOOR_FRAC * mean)
        lcb = mean - float(stats.t.ppf(1.0 - alpha, n - 1)) * sd / math.sqrt(n)
        per_cell[f"{M}x{N}x{K}"] = {"mean": mean, "sd": sd, "n": n,
                                    "lcb": lcb}

    cells = list(per_cell.values())
    top = max(cells, key=lambda c: c["mean"])
    strata = {
        stratum: {
            "r_lo": min(_median_per_cell(rows,
                                         key=lambda r: (r.M, r.N, r.K)).values()),
            "r_hi": max(_median_per_cell(rows,
                                         key=lambda r: (r.M, r.N, r.K)).values()),
            "n_records": len(rows),
        }
        for stratum, rows in sorted(_by_stratum(clean).items())
    }
    return UsefulBand(
        r_lo=min(c["lcb"] for c in cells),
        r_hi=top["lcb"],
        r_lo_point=min(c["mean"] for c in cells),
        r_hi_point=top["mean"],
        alpha=alpha,
        n_cells=len(cells),
        per_cell=per_cell,
        strata=strata,
    )


def operand_core(
    exp2: Sequence[RunRecord],
) -> tuple[float, float, float]:
    """Exp 2: (r_lo_op, r_hi_op, w0) over operand-value cells.

    Only the operand *values* vary (precision/shape fixed), so the spread
    tracks the activity factor alpha; the clock was governor-selected rather
    than locked, so read it as a descriptive within-device envelope rather
    than a clean isolation of alpha. Both edges are total quotients E/C and
    the ratio of the extreme cells is the operand core w0 = r_hi^0 / r_lo^0.

    None of these three quantities prices any bound. In particular the
    cheapest cell is NOT the covert denominator r_lo^cov: that is the Exp-7
    marginal dose slope, obtained via
    :func:`powertoflops.measured.covert_edge`.
    """
    clean = clean_records(exp2)
    if not clean:
        raise ValueError("operand_core: no un-throttled Exp-2 records")
    cells = _median_per_cell(clean, key=lambda r: r.operand_dist_a)
    rs = list(cells.values())
    r_lo_op, r_hi_op = min(rs), max(rs)
    w0 = r_hi_op / r_lo_op
    return r_lo_op, r_hi_op, w0


def sigma_dec_from(exp2: Sequence[RunRecord], dist: str = "declared_typical") -> float:
    """Std of r [J/op] over the declared-typical operand runs (primes sigma_dec).

    The re-execution channel reads the *actual declared operands*, so only the
    activity jitter on declared-typical data survives as a band — exactly the
    spread of r across repeats at ``dist`` (A2 Exp 2). Returns
    0.0 if fewer than two such runs are present.
    """
    rs = [r.r_j_per_op for r in clean_records(exp2) if r.operand_dist_a == dist]
    if len(rs) < 2:
        return 0.0
    return float(np.std(rs, ddof=1))


@dataclass(frozen=True)
class MarginalRate:
    """Affine fit P = r_marginal * F + intercept over per-cell (F, P) medians.

    ``r_marginal`` is the marginal exchange rate dP/dF the theory floor uses
    (review C2) — the slope of power against sustained op rate — as opposed to
    the total-quotient E/C stored per record in ``r_j_per_op``, which folds
    the overhead P0/F into the rate. Each sweep cell is collapsed to its median
    (F, P) before the OLS so a single bad window cannot tilt the slope (same
    median-then-fit convention as the band edges).
    """

    r_marginal: float          # dP/dF [J/op]
    intercept_w: float         # fitted P at F = 0 [W]
    se_slope: float            # 1 standard error of the slope [J/op]
    se_intercept: float        # 1 standard error of the intercept [W]
    r2: float                  # R^2 of the affine fit over cell medians
    n_cells: int               # cells entering the fit


def marginal_rate(records: Sequence[RunRecord]) -> MarginalRate:
    """Estimate the marginal rate r = dP/dF from a sustained-rate sweep.

    Collapses each cell (:func:`_cell_key`) to its median sustained op rate
    F = ops_nominal / duration and median power P, then fits P = slope*F + intercept
    across cells. Callers pass already-cleaned records (the cleaning rule is
    per-use: rate bands need finite r, the overhead/marginal fit needs the
    zero-FLOP idle rows kept).
    """
    by_cell: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_cell[_cell_key(r)].append(r)
    if len(by_cell) < 2:
        raise ValueError("marginal_rate: need >= 2 sweep cells to fit a slope")

    F = np.array([np.median([r.ops_nominal / r.duration_s for r in rows])
                  for rows in by_cell.values()], dtype=float)
    P = np.array([np.median([r.mean_power_w for r in rows])
                  for rows in by_cell.values()], dtype=float)

    if np.ptp(F) > 0 and len(by_cell) >= 3:
        (slope, intercept), cov = np.polyfit(F, P, 1, cov=True)
        se_slope = float(np.sqrt(cov[0, 0]))
        se_intercept = float(np.sqrt(cov[1, 1]))
    else:
        (slope, intercept) = np.polyfit(F, P, 1)
        se_slope = se_intercept = 0.0

    resid = P - (slope * F + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((P - P.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return MarginalRate(
        r_marginal=float(slope), intercept_w=float(intercept),
        se_slope=se_slope, se_intercept=se_intercept,
        r2=r2, n_cells=len(by_cell),
    )


def marginal_rate_compute_bound(
    exp3: Iterable[RunRecord],
    cube_min_dim: int = 2048,
) -> MarginalRate:
    """Sensitivity subfit of :func:`marginal_rate`: idle anchors + big cubes only.

    The full-ladder fit honestly includes launch- and bandwidth-bound shapes,
    which bend the affine map and depress its R^2 (review C2). This variant
    keeps the idle cells (F = 0 anchors) and the compute-bound cubes
    (M == N == K >= ``cube_min_dim``) — the cells where the affine model is
    expected to hold — so the paper can quote it as a one-line sensitivity
    check next to the pooled slope. Cubes alone (without the idle anchors) are
    degenerate on the R2 ladder: they all sit at the board power cap with
    little F spread.
    """
    clean = clean_overhead_records(exp3)
    sub = [
        r for r in clean
        if r.util_frac == 0.0
        or (r.M == r.N == r.K and r.M >= cube_min_dim)
    ]
    return marginal_rate(sub)


@dataclass(frozen=True)
class MarginalCovertCost:
    """Exp-7 dose-response fit E = r_marg * C_cov + intercept at one covert format.

    ``r_marg`` is the marginal covert energy cost dE/dC_cov [J/op]: the extra
    joules per covert op given the declared workload, window length, shape and
    thermal regime are all held fixed and only the covert dose varies (referee
    B2's intervention estimand — NOT the total quotient E/C, which amortises
    baseline power over the ops). ``r_marg_lcb95`` is the one-sided lower
    confidence limit at level 1 - alpha (Student-t on the OLS slope over
    per-cell medians): beta places the covert cost in the DENOMINATOR, so a
    verifier-safe upper bound on beta needs a defensible LOWER bound on the
    cost — the floor calculations use the LCB, never the point slope.
    """

    r_marg: float              # dE/dC_cov [J/op] (point slope)
    r_marg_lcb95: float        # one-sided lower confidence limit on the slope
    se_slope: float            # 1 standard error of the slope [J/op]
    r2: float                  # R^2 of the dose fit over cell medians
    n_cells: int               # dose cells entering the fit
    intercept_j: float         # fitted window energy at zero dose [J]
    dose0_energy_j: float      # measured median energy of the zero-dose cell [J]
                               # (nan when the sweep carried no zero dose) —
                               # the intercept's consistency anchor
    stratum_mhz: int | None    # common achieved-clock stratum; None if mixed
    cov_dtype: str             # covert format this fit prices
    alpha: float               # one-sided level of the LCB
    strata: dict = field(default_factory=dict)
                               # {stratum_mhz: {r_marg, n_cells}} subfits when
                               # the clock split across strata (diagnostic)


def clean_marginal_records(
    records: Iterable[RunRecord], cov_dtype: str,
) -> list[RunRecord]:
    """Exp-7 record filter for one covert format's dose regression.

    Row hygiene as :func:`_clean_r` but on the window ENERGY, not r_j_per_op
    (whose declared-only C makes it meaningless across doses), and with NO
    cross-dose stall guard: mean window power grows with the dose by design,
    so :func:`_drop_stalled`'s per-cell peak reference stays valid within a
    dose cell but any cross-dose power comparison would discard the low doses
    wholesale. Zero-dose rows are kept only when tagged with this arm's
    ``cov_dtype`` (the sweep builder tags each arm's own zero-dose anchor).
    """
    rows = [
        r for r in records
        if r.cov_dtype == cov_dtype
        and not (r.throttle_reasons & HARMFUL_THROTTLE_MASK)
        and r.operand_dist_a != REFERENCE_DIST
        and not r.trimmed
        and math.isfinite(r.energy_j) and r.energy_j > 0.0
    ]
    return _drop_stalled(rows)


def _dose_fit(cells: dict[int, tuple[float, float]], alpha: float):
    """OLS of median energy on median covert ops over dose cells.

    Returns (slope, lcb, se_slope, r2, intercept). ``cells`` maps cov_iters ->
    (median C_cov, median E). Needs >= 3 cells for a finite slope SE.
    """
    if len(cells) < 3:
        raise ValueError(
            f"marginal_covert_cost: need >= 3 dose cells for a slope with an "
            f"SE, got {len(cells)}")
    x = np.array([c for c, _ in cells.values()], dtype=float)
    y = np.array([e for _, e in cells.values()], dtype=float)
    if np.ptp(x) == 0.0:
        raise ValueError("marginal_covert_cost: all dose cells share one "
                         "covert op count — no slope is identifiable")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    n = len(cells)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_slope = math.sqrt(float(np.sum(resid ** 2)) / (n - 2) / sxx)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    lcb = float(slope) - float(stats.t.ppf(1.0 - alpha, n - 2)) * se_slope
    return float(slope), lcb, se_slope, r2, float(intercept)


def marginal_covert_cost(
    exp7: Sequence[RunRecord],
    cov_dtype: str,
    *,
    alpha: float = 0.05,
) -> MarginalCovertCost:
    """Exp 7: marginal covert energy cost for one covert format (:class:`MarginalCovertCost`).

    Collapses each dose cell (cov_iters at fixed everything else) to its median
    (C_cov, E) and fits E = r_marg * C_cov + intercept across doses — a
    within-cell intervention on the covert dose, unlike Exp 3's between-shape
    regression (referee B2). Clock locking is privilege-denied on the platform,
    so the common achieved-clock stratum is verified post hoc: ``stratum_mhz``
    is the single stratum when there is one; otherwise it is None and
    ``strata`` carries per-stratum subfit slopes so the spread is inspectable.
    """
    clean = clean_marginal_records(exp7, cov_dtype)
    cells: dict[int, tuple[float, float]] = {}
    by_dose: dict[int, list[RunRecord]] = defaultdict(list)
    for r in clean:
        by_dose[r.cov_iters].append(r)
    for dose, rows in by_dose.items():
        cells[dose] = (float(np.median([r.cov_ops_nominal for r in rows])),
                       float(np.median([r.energy_j for r in rows])))

    slope, lcb, se_slope, r2, intercept = _dose_fit(cells, alpha)

    dose0 = by_dose.get(0)
    dose0_energy = (float(np.median([r.energy_j for r in dose0]))
                    if dose0 else float("nan"))

    strata_rows = _by_stratum(clean)
    strata: dict[int, dict] = {}
    if len(strata_rows) == 1:
        stratum_mhz: int | None = next(iter(strata_rows))
    else:
        stratum_mhz = None
        for stratum, rows in sorted(strata_rows.items()):
            sub: dict[int, list[RunRecord]] = defaultdict(list)
            for r in rows:
                sub[r.cov_iters].append(r)
            sub_cells = {
                d: (float(np.median([r.cov_ops_nominal for r in rr])),
                    float(np.median([r.energy_j for r in rr])))
                for d, rr in sub.items()
            }
            if len(sub_cells) >= 3:
                s_slope, *_ = _dose_fit(sub_cells, alpha)
                strata[stratum] = {"r_marg": s_slope, "n_cells": len(sub_cells)}

    return MarginalCovertCost(
        r_marg=slope, r_marg_lcb95=lcb, se_slope=se_slope, r2=r2,
        n_cells=len(cells), intercept_j=intercept, dose0_energy_j=dose0_energy,
        stratum_mhz=stratum_mhz, cov_dtype=cov_dtype, alpha=alpha,
        strata=strata,
    )


# Prefix tagging a qblock covert arm in the dose sweep's cov_dtype field
# (one arm per (seq, batch) cell, e.g. "qblock_2048x8").
QBLOCK_MARG_ARM_PREFIX = "qblock_"


@dataclass(frozen=True)
class UsefulMarginalBand:
    """Marginal useful-cost band (referee B5, third round).

    The incremental covert cost of USEFUL transformer work, from an Exp-7
    dose-response with the trained qblock forward as the swept dose
    (:func:`powertoflops.bench.qblock.run_qblock_marginal`). Replaces the total-quotient
    :class:`UsefulBand` as the ladder denominator: baseline power cancels in the
    dose slope, so both endpoints are genuine one-sided LOWER limits on the
    marginal cost dE/dC_cov — the estimand beta needs in its denominator (unlike
    a total quotient E/C, which is anti-conservative there). Arms are qblock
    (seq, batch) cells: ``r_lo`` is the cheapest arm's LCB (the covert floor,
    ladder rung 4) and ``r_hi`` the LCB of the arm with the largest slope (the
    band-top assumption price, rung 5). Point slopes are carried for disclosure.
    """

    r_lo: float                # min over arms of the per-arm marginal LCB [J/op]
    r_hi: float                # marginal LCB of the arm with the largest slope
    r_lo_point: float          # min arm point slope (descriptive)
    r_hi_point: float          # max arm point slope (descriptive)
    alpha: float               # one-sided level of the LCBs
    n_cells: int               # qblock arms entering the band
    per_cell: dict             # {arm_label: asdict(MarginalCovertCost)}


def useful_marginal_band(
    qblock_marg: Sequence[RunRecord], *, alpha: float = 0.05,
) -> UsefulMarginalBand:
    """Aggregate the per-arm qblock-marginal dose fits into the marginal useful band.

    Each qblock arm (tagged in ``cov_dtype`` as ``qblock_<seq>x<batch>``) is fit
    by :func:`marginal_covert_cost` to its marginal slope + one-sided LCB; the
    band takes the min LCB over arms (the floor) and the largest-slope arm's LCB
    (the top). Refuses a capture carrying no qblock arms.
    """
    arms = sorted({r.cov_dtype for r in qblock_marg
                   if r.cov_dtype.startswith(QBLOCK_MARG_ARM_PREFIX)})
    if not arms:
        raise ValueError(
            "useful_marginal_band: no qblock covert arms (cov_dtype "
            f"'{QBLOCK_MARG_ARM_PREFIX}<seq>x<batch>') in the records — capture "
            "qblock-marginal with scripts/run_qblock_marginal.py")
    fits = {a: marginal_covert_cost(qblock_marg, a, alpha=alpha) for a in arms}
    top = max(fits.values(), key=lambda f: f.r_marg)
    return UsefulMarginalBand(
        r_lo=min(f.r_marg_lcb95 for f in fits.values()),
        r_hi=top.r_marg_lcb95,
        r_lo_point=min(f.r_marg for f in fits.values()),
        r_hi_point=top.r_marg,
        alpha=alpha,
        n_cells=len(fits),
        per_cell={a: dataclasses.asdict(f) for a, f in fits.items()},
    )


# Widening applied to the observed idle range to form the overhead band
# [P0_lo, P0_hi]: NVML documents +/-5% power-reading accuracy only for the
# Fermi/Kepler generations and publishes no GA100 spec for the total-energy
# counter, so we adopt that figure as a working meter tolerance and stretch
# the observed idle extrema by it (widening only widens the bound).
IDLE_TOL_FRAC = 0.05


@dataclass(frozen=True)
class OverheadBand:
    """Exp-3 result: observed-idle overhead band + affine-fit diagnostics.

    The band [P0_lo, P0_hi] is the OBSERVED idle (util_frac == 0) power range
    widened by IDLE_TOL_FRAC. The affine-fit intercept is reported separately
    (``fit_intercept_w`` +/- ``fit_se_w``) — an extrapolation diagnostic, never
    the band itself; it becomes the band only as an explicit fallback when the
    sweep carried no idle rows at all (``n_idle == 0``).
    """

    P0_lo: float               # overhead band lower edge [W]
    P0_hi: float               # overhead band upper edge [W]
    fit_intercept_w: float     # affine-fit intercept [W] (diagnostic)
    fit_se_w: float            # 1 standard error of the intercept [W]
    r_marginal: float          # fitted dP/dF [J/op] (the theory's marginal r)
    affine_r2: float           # R^2 of the affine fit (convexity flag if < 1)
    n_idle: int                # observed idle rows the band rests on


def clean_overhead_records(records: Iterable[RunRecord]) -> list[RunRecord]:
    """Exp-3 record filter: like :func:`clean_records` but KEEPS zero-FLOP rows.

    The rate-band cleaner requires finite r > 0, which silently deleted every
    idle (zero-FLOP, r = NaN) row — the review-C3 bug that turned the overhead
    band into intercept±1SE. Idle rows carry no r but a perfectly good power
    measurement, so the validity test here is on the power, with the r test
    applied only to rows that did compute. The per-cell stall guard still runs
    (idle rows form their own util_frac == 0 cell, so it never drops them).
    """
    kept = [
        r for r in records
        if not (r.throttle_reasons & HARMFUL_THROTTLE_MASK)
        and r.operand_dist_a != REFERENCE_DIST
        and not r.trimmed
        and math.isfinite(r.mean_power_w) and r.mean_power_w > 0.0
        and (r.ops_nominal == 0
             or (math.isfinite(r.r_j_per_op) and r.r_j_per_op > 0.0))
    ]
    return _drop_stalled(kept)


def overhead_band(exp3: Sequence[RunRecord]) -> OverheadBand:
    """Exp 3: observed-idle overhead band + affine diagnostics (:class:`OverheadBand`).

    Fit mean power P against nominal-op rate F = ops_nominal / duration across the sweep
    (idle rows included at F = 0): P = r*F + P0. The band [P0_lo, P0_hi] is the
    observed idle (util_frac == 0) range widened by IDLE_TOL_FRAC; the intercept
    +/- 1SE is reported alongside as a diagnostic, and is used as the band only
    when no idle rows exist. ``affine_r2`` flags convexity (DVFS bending the
    map) when it drops below ~1.
    """
    clean = clean_overhead_records(exp3)
    if len(clean) < 2:
        raise ValueError("overhead_band: need >= 2 un-throttled Exp-3 records")

    fit = marginal_rate(clean)

    # Overhead band: the directly observed idle range + the stated tolerance.
    idle = [r.mean_power_w for r in clean if r.util_frac == 0.0]
    if idle:
        P0_lo = min(idle) * (1.0 - IDLE_TOL_FRAC)
        P0_hi = max(idle) * (1.0 + IDLE_TOL_FRAC)
    else:
        P0_lo = fit.intercept_w - fit.se_intercept
        P0_hi = fit.intercept_w + fit.se_intercept

    return OverheadBand(
        P0_lo=float(P0_lo), P0_hi=float(P0_hi),
        fit_intercept_w=fit.intercept_w, fit_se_w=fit.se_intercept,
        r_marginal=fit.r_marginal, affine_r2=fit.r2, n_idle=len(idle),
    )


def extract_bands(
    exp1: Sequence[RunRecord],
    exp2: Sequence[RunRecord],
    exp3: Sequence[RunRecord],
) -> BandResult:
    """Compose all three experiments into a single :class:`BandResult`."""
    r_lo, r_hi = exchange_band(exp1)
    r_lo_op, r_hi_op, w0 = operand_core(exp2)
    ob = overhead_band(exp3)
    sigma_dec = sigma_dec_from(exp2)
    return BandResult(
        r_lo=r_lo,
        r_hi=r_hi,
        r_lo_op=r_lo_op,
        r_hi_op=r_hi_op,
        w0=w0,
        P0_lo=ob.P0_lo,
        P0_hi=ob.P0_hi,
        P0_fit_intercept=ob.fit_intercept_w,
        P0_fit_se=ob.fit_se_w,
        r_marginal=ob.r_marginal,
        affine_r2=ob.affine_r2,
        sigma_dec=sigma_dec,
    )
