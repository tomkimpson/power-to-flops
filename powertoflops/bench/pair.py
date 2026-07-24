"""Matched-energy pair: the fixed-T, same-C_dec attainability witness (B3).

The identified-set proposition prices a cheater who does the SAME trusted
declared work in the SAME observation window as an honest run, differing only
in operand cost and covert fill. The witness therefore holds both fixed by
construction (referee B3; the first-round pair halved the declared iterations
and let the wall float, so it demonstrated only a loose energy degeneracy):

  * run A (honest):  n_dec declared iterations on DEAR operands, then an idle
    pad to the fixed window T;
  * run B (cheating): the SAME n_dec declared iterations on CHEAP operands,
    plus a covert fill tuned so E_B matches E_A, then a pad to the same T.

Pre-flight planning (:func:`plan_pair`) solves the covert dose from short
calibration windows and hard-fails when the requested utilisation cannot fit
the window — run B's busy time grows ~2 covert ops per declared op, which
caps the demonstrable utilisation at h_max (~0.27 on the measured A100
bands, below the headline h* = 0.42; the manuscript owns this as a scope
note). Artifacts are validated (:func:`validate_pair_payload`) BEFORE being
written: pad underflow, capacity overrun, C_dec mismatch, non-convergence, or
a sensor alarm on any cheating repeat is a hard failure, never a publishable
witness. The sensor's actual one-sided acceptance decision (alarm iff
E_obs > max honest repeat) is evaluated and recorded per repeat.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
import uuid as uuid_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import A100_DENSE_PEAK_OPS, BenchConfig, DEFAULT
from ..floor import beta_phys, energy_slack
from ..measured import MeasuredBands, covert_edge, format_band
from .analysis import analysis_git_commit
from .marginal import MAX_BUSY_FRAC
from .records import build_manifest


class PairInfeasibleError(RuntimeError):
    """The requested pair cannot fit the fixed window / capacity constraints."""


class PairValidationError(RuntimeError):
    """A pair artifact violates the witness construction; it must not be used."""


# ---- pure tuning logic (no torch / NVML) ------------------------------------


@dataclass(frozen=True)
class TuneResult:
    """Outcome of the covert-iteration tuning loop."""

    n_covert: int                    # final covert iteration count
    e_b: float                       # last measured E_B [J]
    converged: bool                  # |E_A - E_B| / E_A < tol_frac reached
    history: tuple = field(default_factory=tuple)   # ((n_covert, e_b), ...)


def solve_covert_iters(e_target: float, e_declared_b: float,
                       e_per_covert: float) -> int:
    """Initial covert count: (E_A - E_declared_B) / e_covert, floored at 0."""
    return max(0, int(round((e_target - e_declared_b) / e_per_covert)))


def tune_covert_iters(
    measure_b: Callable[[int], float],
    e_a: float,
    n0: int,
    e_per_covert: float,
    *,
    tol_frac: float,
    max_rounds: int,
) -> TuneResult:
    """Walk n_covert until measure_b(n) matches e_a within tol_frac.

    Newton-style correction n += (E_A - E_B) / slope. The first step uses the
    calibrated per-iteration covert energy; from the second step the slope is
    re-estimated by secant from the last two measured points, clamped to
    [0.25, 4] x the calibration value to reject noise-driven or sign-flipped
    estimates. The secant recovers a badly biased calibration slope in one
    step (Slurm 14633255: the fixed-slope loop failed to converge in three
    rounds because the interleaved marginal was ~1.3x the calibration value).
    A rig whose energy does not respond to n exhausts ``max_rounds`` and
    reports ``converged=False`` with the best-effort count.
    """
    n = max(0, n0)
    history: list[tuple[int, float]] = []
    e_b = float("nan")
    slope = e_per_covert
    for _ in range(max_rounds):
        e_b = measure_b(n)
        history.append((n, e_b))
        if abs(e_a - e_b) / e_a < tol_frac:
            return TuneResult(n_covert=n, e_b=e_b, converged=True,
                              history=tuple(history))
        if len(history) >= 2:
            (n_prev, e_prev), (n_cur, e_cur) = history[-2], history[-1]
            dn = n_cur - n_prev
            if dn != 0:
                secant = (e_cur - e_prev) / dn
                if secant > 0.0:            # keep last slope on a noise flip
                    slope = min(max(secant, 0.25 * e_per_covert),
                                4.0 * e_per_covert)
        n = max(0, n + int(round((e_a - e_b) / slope)))
    return TuneResult(n_covert=history[-1][0], e_b=e_b, converged=False,
                      history=tuple(history))


def capacity_frac_used(*, covert_ops: float, h: float, wall_s: float,
                       peak_ops_per_s: float) -> float:
    """Covert ops as a fraction of what (1-h) of the format peak allows.

    < 1 means the covert schedule is capacity-feasible alongside the declared
    share; this is the joint-feasibility witness the formal proposition (R3.2)
    cites for the band corner.
    """
    return covert_ops / ((1.0 - h) * peak_ops_per_s * wall_s)


# ---- B3 fixed-T, same-C_dec construction (pure planning + validation) --------


@dataclass(frozen=True)
class PairCalibration:
    """Per-iteration marginal costs and durations from short calibration windows.

    Marginal energies are intercept-anchored (Exp-7 style): m = (E_window -
    E_idle) / n for a burst of n iterations inside a fixed idle-padded window,
    so the baseline P0 share — which runs for the full window in both runs of
    the pair regardless — never enters the gap arithmetic.
    """

    T_s: float          # fixed observation window [s]
    e_idle_j: float     # measured idle energy over one T_s window [J]
    m_dear_j: float     # marginal energy per declared iter, dear operands [J]
    m_cheap_j: float    # marginal energy per declared iter, cheap operands [J]
    m_cov_j: float      # marginal energy per covert iter [J]
    t_dear_s: float     # wall time per declared iter, dear operands [s]
    t_cheap_s: float    # wall time per declared iter, cheap operands [s]
    t_cov_s: float      # wall time per covert iter [s]
    ops_dec: float      # nominal ops per declared iteration
    ops_cov: float      # nominal ops per covert iteration


@dataclass(frozen=True)
class PairPlan:
    """Pre-flight solution for one utilisation h; every field is predicted."""

    h: float
    n_dec: int          # declared iterations — identical in runs A and B
    n_cov0: int         # covert iterations closing the energy gap
    busy_a_s: float     # predicted busy time of run A (dear declared only)
    busy_b_s: float     # predicted busy time of run B (cheap declared + covert)
    h_max: float        # largest h whose busier run still fits max_busy_frac*T
    capacity_frac: float  # covert ops vs the (1-h) covert-capacity clip
    e_gap_j: float      # predicted energy the covert fill must supply [J]


def pair_h_max(
    calib: PairCalibration,
    *,
    peak_dec_ops: float,
    max_busy_frac: float = MAX_BUSY_FRAC,
) -> float:
    """Largest utilisation h whose busier run still fits max_busy_frac * T.

    Run B's busy time scales linearly in h (the covert dose is proportional
    to the declared count), so feasibility is a single threshold. Raises
    :class:`PairInfeasibleError` when the calibration shows no dear-vs-cheap
    energy gap — then no witness is constructible at any h.
    """
    gap_per_iter = calib.m_dear_j - calib.m_cheap_j
    if gap_per_iter <= 0.0:
        raise PairInfeasibleError(
            f"no energy gap to fill: dear declared iterations cost "
            f"{calib.m_dear_j:.4g} J vs cheap {calib.m_cheap_j:.4g} J — the "
            f"witness needs the dear operands to be strictly dearer")
    iters_per_h = peak_dec_ops * calib.T_s / calib.ops_dec
    ratio_cov = gap_per_iter / calib.m_cov_j     # covert iters per declared iter
    per_h_busy_a = iters_per_h * calib.t_dear_s
    per_h_busy_b = iters_per_h * (calib.t_cheap_s + ratio_cov * calib.t_cov_s)
    return max_busy_frac * calib.T_s / max(per_h_busy_a, per_h_busy_b)


def plan_pair(
    calib: PairCalibration,
    h: float,
    *,
    peak_dec_ops: float,
    peak_cov_ops: float,
    max_busy_frac: float = MAX_BUSY_FRAC,
    safety: float = 0.95,
) -> PairPlan:
    """Solve the same-C_dec, same-T pair at utilisation h; hard-fail if it
    cannot fit.

    h is the theorem's declared utilisation C_dec / (F_max^dec * T).
    Requesting h > safety * h_max (see :func:`pair_h_max`) raises
    :class:`PairInfeasibleError`, as does an empty energy gap (cheap operands
    no cheaper than dear) or a covert dose beyond the (1-h) capacity clip of
    eq:capclip.
    """
    gap_per_iter = calib.m_dear_j - calib.m_cheap_j
    h_max = pair_h_max(calib, peak_dec_ops=peak_dec_ops,
                       max_busy_frac=max_busy_frac)
    iters_per_h = peak_dec_ops * calib.T_s / calib.ops_dec
    ratio_cov = gap_per_iter / calib.m_cov_j     # covert iters per declared iter
    if h > safety * h_max:
        raise PairInfeasibleError(
            f"h = {h:.3f} exceeds {safety:.2f} * h_max = "
            f"{safety * h_max:.3f}: run B's cheap-declared + covert busy time "
            f"cannot fit {max_busy_frac:.2f} of the {calib.T_s:.1f} s window")

    n_dec = int(round(h * iters_per_h))
    n_cov0 = int(round(n_dec * ratio_cov))
    if n_dec < 1 or n_cov0 < 1:
        raise PairInfeasibleError(
            f"degenerate plan at h = {h:.4f}: n_dec = {n_dec}, "
            f"n_cov0 = {n_cov0} — the witness needs nonzero declared and "
            f"covert work")

    cap_frac = capacity_frac_used(
        covert_ops=n_cov0 * calib.ops_cov, h=h, wall_s=calib.T_s,
        peak_ops_per_s=peak_cov_ops)
    if cap_frac >= 1.0:
        raise PairInfeasibleError(
            f"covert dose exceeds the capacity clip: {cap_frac:.3f} of the "
            f"(1-h) covert-format budget at h = {h:.3f}")

    return PairPlan(
        h=h, n_dec=n_dec, n_cov0=n_cov0,
        busy_a_s=n_dec * calib.t_dear_s,
        busy_b_s=n_dec * calib.t_cheap_s + n_cov0 * calib.t_cov_s,
        h_max=h_max, capacity_frac=cap_frac,
        e_gap_j=n_dec * gap_per_iter,
    )


DEFAULT_ACCEPT_Z = 3.0   # one-sided ~0.13% false-alarm rate (Gaussian)


def tuning_target(e_a_ref_j: float, tol_frac: float) -> float:
    """The energy the cheater tunes toward: (1 - tol) * the honest reference.

    A real adversary aims just below the honest energy, not exactly at it:
    undershooting by the tolerance keeps E_B inside the one-sided acceptance
    region while remaining energy-degenerate within tolerance.
    """
    return e_a_ref_j * (1.0 - tol_frac)


def acceptance_threshold(e_a_repeats, *, z: float = DEFAULT_ACCEPT_Z,
                         sigma_floor_frac: float = 0.005) -> float:
    """One-sided no-alarm threshold: mean + z * max(std, floor).

    A stated false-alarm rule, not the sample maximum: with a noisy meter and
    a handful of repeats the empirical max is an overfit, indefensible
    threshold (it alarmed the pair in Slurm jobs 14632514/14632592). ``z``
    sets the false-alarm rate (default 3 sigma, ~0.13% one-sided Gaussian);
    the honest dear run samples the top of the proposition's acceptance band
    \\cref{eq:noalarm}, so the cheating run matched to its energy is accepted
    exactly when the meter cannot separate them.

    The spread is floored at ``sigma_floor_frac`` of the mean (default 0.5%,
    the window-to-window repeatability of NVML energy under its 10 Hz hold):
    no verifier can set an alarm tighter than the meter's own resolution, and
    without the floor a lucky pair of near-identical honest repeats yields an
    unphysically tight threshold that a perfectly matched run trips (Slurm
    14638999: E_B 392.0 J vs a 391.5 J threshold from two 388-390 J repeats).
    Needs >= 2 repeats for a variance.
    """
    n = len(e_a_repeats)
    if n < 2:
        raise ValueError("acceptance threshold needs >= 2 honest repeats "
                         "for a false-alarm variance")
    mean = float(np.mean(e_a_repeats))
    sigma = max(float(np.std(e_a_repeats, ddof=1)), sigma_floor_frac * mean)
    return mean + z * sigma


def acceptance_verdicts(e_b_repeats, threshold_j: float) -> list:
    """Per-repeat sensor decisions: True = accepted (no alarm), E <= threshold."""
    return [float(e) <= threshold_j for e in e_b_repeats]


DEFAULT_DEGENERACY_MAX = 0.05   # NVML stated accuracy: indistinguishable below


def validate_pair_payload(payload: dict, *, wall_tol_frac: float = 0.02,
                          degeneracy_max: float = DEFAULT_DEGENERACY_MAX) -> None:
    """Hard analysis failure unless the artifact is a valid witness (B3).

    Raises :class:`PairValidationError` when the payload violates the
    construction the proposition prices: an explicit nonzero covert workload,
    identical declared work in both runs, both runs on the fixed window (pad
    strictly positive, duration within ``wall_tol_frac`` of window_s), covert
    dose inside the capacity clip, converged tuning, and the one-sided sensor
    rule — recomputed from the repeats, not trusted from the file — accepting
    every cheating repeat. The mean energies must agree within
    ``degeneracy_max`` (NVML's stated ±5% accuracy: below it the two runs are
    indistinguishable to the meter, which is the operational meaning of the
    energy degeneracy; a cheating run drawing *less* than honest is the safe
    direction and is caught only if the gap is gross).
    """
    def fail(msg: str):
        raise PairValidationError(msg)

    if int((payload.get("covert") or {}).get("iters", 0)) < 1:
        fail("no covert work: the witness requires an explicit nonzero covert "
             "workload — a zero-dose pair is just a cheaper declared run")

    decl = payload.get("declared") or {}
    if "iters_a" not in decl or "iters_b" not in decl:
        fail("malformed pair artifact: missing declared iteration counts")
    if decl["iters_a"] != decl["iters_b"]:
        fail(f"C_dec not held fixed: {decl['iters_a']} declared iterations in "
             f"run A vs {decl['iters_b']} in run B — not the witness the "
             f"proposition prices")

    for key in ("window_s", "repeats_a", "repeats_b", "tuning",
                "capacity_frac_used"):
        if key not in payload:
            fail(f"malformed pair artifact: missing {key!r} (repeats and the "
                 f"fixed window are required)")

    window_s = float(payload["window_s"])
    reps_a, reps_b = payload["repeats_a"], payload["repeats_b"]
    if not reps_a or not reps_b:
        fail("no scored repeats: both runs need measured repeat windows")

    for name, reps in (("A", reps_a), ("B", reps_b)):
        for i, rep in enumerate(reps):
            if rep["pad_s"] <= 0.0:
                fail(f"pad underflow in run {name} repeat {i} "
                     f"(pad_s = {rep['pad_s']:.4g}): busy work overflowed the "
                     f"fixed window")
            if abs(rep["duration_s"] - window_s) / window_s > wall_tol_frac:
                fail(f"run {name} repeat {i} missed the fixed window: "
                     f"duration {rep['duration_s']:.3f} s vs window_s "
                     f"{window_s:.3f} s (tol {wall_tol_frac:.0%})")

    if float(payload["capacity_frac_used"]) >= 1.0:
        fail(f"covert schedule violates the capacity clip: "
         f"capacity_frac_used = {payload['capacity_frac_used']:.3f}")

    if not payload["tuning"].get("converged", False):
        fail("energy match did not converge: the tuning loop exhausted its "
             "rounds outside tolerance")

    z = float((payload.get("acceptance") or {}).get("z", DEFAULT_ACCEPT_Z))
    threshold = acceptance_threshold([r["energy_j"] for r in reps_a], z=z)
    verdicts = acceptance_verdicts([r["energy_j"] for r in reps_b], threshold)
    if not all(verdicts):
        alarmed = [i for i, ok in enumerate(verdicts) if not ok]
        fail(f"sensor alarms on cheating repeats {alarmed}: energies exceed "
             f"the {z:g}-sigma honest envelope {threshold:.2f} J — not an "
             f"accepted witness")

    mean_a = float(np.mean([r["energy_j"] for r in reps_a]))
    mean_b = float(np.mean([r["energy_j"] for r in reps_b]))
    delta = abs(mean_a - mean_b) / mean_a
    if delta > degeneracy_max:
        fail(f"pair is not energy-degenerate: mean |dE|/E = {delta:.2%} "
             f"exceeds {degeneracy_max:.0%} (NVML accuracy) — a run this much "
             f"cheaper is distinguishable from honest, not a matched witness")


def summarize_pair(payload: dict, mb: MeasuredBands) -> dict:
    """Link the witness parameters to the theorem's corner assumptions.

    Returns the witnessed covert compute in machines of declared capacity
    (beta_witness = C_cov / (F_max^dec * T)) next to the model's S(h) and
    beta_phys(h) at the same utilisation, plus the achieved marginal rates
    against the measured band edges — the numbers the manuscript's witness
    paragraph quotes.
    """
    decl, cov = payload["declared"], payload["covert"]
    calib = payload["calibration"]
    peak_dec = A100_DENSE_PEAK_OPS[decl["dtype"]]
    peak_cov = A100_DENSE_PEAK_OPS[cov["dtype"]]
    kappa = peak_cov / peak_dec
    h, T_s = float(payload["h"]), float(payload["window_s"])

    r_lo_dec, r_hi_dec = format_band(mb, decl["dtype"])
    r_lo_cov = covert_edge(mb, cov["dtype"])
    d_p0 = mb.P0_hi - mb.P0_lo
    args = dict(r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
                dE0=d_p0, C_max=peak_dec)

    c_cov = float(cov["iters"]) * float(cov["ops_nominal_per_iter"])
    return {
        "beta_witness": c_cov / (peak_dec * T_s),
        "S_h": float(energy_slack(h, **args)),
        "beta_phys_h": float(beta_phys(h, **args, kappa=kappa)),
        "kappa": kappa,
        "h": h,
        "r_marg_dec_dear": calib["m_dear_j"] / calib["ops_dec"],
        "r_marg_dec_cheap": calib["m_cheap_j"] / calib["ops_dec"],
        "r_marg_cov_achieved": calib["m_cov_j"] / calib["ops_cov"],
        "r_band_dec": (r_lo_dec, r_hi_dec),
        "r_lo_cov_lcb": r_lo_cov,
    }


# ---- GPU harness -------------------------------------------------------------


def run_matched_pair(out_dir: str | Path, cfg: BenchConfig = DEFAULT.bench,
                     device_index: int = 0,
                     h_values: tuple | None = None) -> list[Path]:
    """Capture the fixed-T, same-C_dec witness; one artifact per feasible h.

    Calibrates marginal per-iteration energies and durations from short
    idle-padded windows (intercept-anchored, so the always-on baseline never
    enters the gap arithmetic), pre-flight-plans each requested utilisation
    (infeasible ones are skipped with a loud log; the whole capture fails if
    none survive), verifies and FREEZES the covert dose with at most
    cfg.pair_max_tune measured corrections, then scores cfg.pair_repeats
    interleaved A/B window pairs at frozen parameters. Every artifact passes
    :func:`validate_pair_payload` before it is written; a validation failure
    dumps failed_pair_<id>.json for diagnosis and raises.
    """
    import torch

    from . import device as dev
    from .meter import Meter
    from .workload import MatmulSpec, make_matmul, ops_nominal_matmul, \
        seconds_per_matmul

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"cuda:{device_index}")
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    meter = Meter(device_index)

    T = cfg.pair_window_s
    n_buf = dict(cfg.locality_buffers)[cfg.pair_declared_locality]
    Md, Nd, Kd = cfg.pair_declared_shape
    Mc, Nc, Kc = cfg.pair_covert_shape
    ops_dec = ops_nominal_matmul(Md, Nd, Kd)
    ops_cov = ops_nominal_matmul(Mc, Nc, Kc)
    peak_dec = A100_DENSE_PEAK_OPS[cfg.pair_declared_dtype]
    peak_cov = A100_DENSE_PEAK_OPS[cfg.pair_covert_dtype]

    spec_dear = MatmulSpec(Md, Nd, Kd, cfg.pair_declared_dtype,
                           dist_a=cfg.pair_dear_dist,
                           dist_b=cfg.pair_dear_dist, n_buffers=n_buf)
    spec_cheap = dataclasses.replace(spec_dear,
                                     dist_a=cfg.pair_cheap_dist,
                                     dist_b=cfg.pair_cheap_dist)
    spec_cov = MatmulSpec(Mc, Nc, Kc, cfg.pair_covert_dtype,
                          dist_a=cfg.pair_covert_dist,
                          dist_b=cfg.pair_covert_dist)

    def _measure_window(work) -> tuple:
        """One fixed-T window: (optional busy work) + sync + idle pad.

        Returns (EnergySample, pad_s). pad_s <= 0 means the busy work
        overflowed the window — the sample is then invalid by construction
        and validation downstream must (and does) reject it.
        """
        pad_box: list[float] = []

        def window() -> None:
            t0 = time.perf_counter()
            if work is not None:
                work()
                torch.cuda.synchronize()
            pad = T - (time.perf_counter() - t0)
            pad_box.append(pad)
            if pad > 0.0:
                time.sleep(pad)

        with meter.measure() as read:
            window()
        return read(), pad_box[0]

    def _repeat_dict(sample, pad_s: float, order_index: int) -> dict:
        return {
            "energy_j": sample.energy_j,
            "duration_s": sample.duration_s,
            "mean_power_w": sample.mean_power_w,
            "meter_method": sample.method,
            "pad_s": pad_s,
            "sm_clock_p50_mhz": sample.sm_clock_p50_mhz,
            "sm_clock_min_mhz": sample.sm_clock_min_mhz,
            "sm_clock_max_mhz": sample.sm_clock_max_mhz,
            "temperature_max_c": sample.temperature_max_c,
            "throttle_union": sample.throttle_union,
            "order_index": order_index,
        }

    artifact_paths: list[Path] = []
    try:
        # --- warm the card, then calibrate --------------------------------
        # Calibration runs in the same warm state as the scored windows: a
        # dear burst first, then the idle window, then the three marginal
        # bursts at pair_calib_busy_frac of the window.
        print("calibrating (warm-up + idle + 3 bursts)...", flush=True)
        t_dear = seconds_per_matmul(
            dataclasses.replace(spec_dear, iters=cfg.calib_matmuls),
            n_time=cfg.calib_matmuls, device=device, generator=generator)
        t_cheap = seconds_per_matmul(
            dataclasses.replace(spec_cheap, iters=cfg.calib_matmuls),
            n_time=cfg.calib_matmuls, device=device, generator=generator)
        t_cov = seconds_per_matmul(
            dataclasses.replace(spec_cov, iters=cfg.calib_matmuls),
            n_time=cfg.calib_matmuls, device=device, generator=generator)

        n_warm = max(1, int(round(0.5 * T / t_dear)))
        make_matmul(dataclasses.replace(spec_dear, iters=n_warm),
                    device=device, generator=generator)()
        torch.cuda.synchronize()

        e_idles = [_measure_window(None)[0].energy_j
                   for _ in range(cfg.pair_calib_repeats)]
        e_idle = float(np.median(e_idles))

        def _marginal(spec, t_iter: float) -> float:
            n_cal = max(1, int(round(cfg.pair_calib_busy_frac * T / t_iter)))
            w = make_matmul(dataclasses.replace(spec, iters=n_cal),
                            device=device, generator=generator)
            w()                                    # warm (uncounted)
            torch.cuda.synchronize()
            energies = []
            for _ in range(cfg.pair_calib_repeats):
                s, pad = _measure_window(w)
                if pad <= 0.0:
                    raise RuntimeError(
                        f"calibration burst overflowed the {T:.1f} s window "
                        f"(n_cal={n_cal})")
                energies.append(s.energy_j)
            return (float(np.median(energies)) - e_idle) / n_cal

        calib = PairCalibration(
            T_s=T, e_idle_j=e_idle,
            m_dear_j=_marginal(spec_dear, t_dear),
            m_cheap_j=_marginal(spec_cheap, t_cheap),
            m_cov_j=_marginal(spec_cov, t_cov),
            t_dear_s=t_dear, t_cheap_s=t_cheap, t_cov_s=t_cov,
            ops_dec=ops_dec, ops_cov=ops_cov,
        )
        print(f"calibration: idle {e_idle:.1f} J/window, "
              f"m_dear {calib.m_dear_j:.4f} m_cheap {calib.m_cheap_j:.4f} "
              f"m_cov {calib.m_cov_j:.4f} J/iter", flush=True)

        # --- choose utilisations: requested + optional auto edge ----------
        h_max = pair_h_max(calib, peak_dec_ops=peak_dec)
        print(f"h_max = {h_max:.3f} (busy-time bound at achieved rates)",
              flush=True)

        hs = list(h_values if h_values is not None else cfg.pair_h_values)
        if cfg.pair_h_auto_edge:
            edge = 0.9 * h_max
            if all(abs(edge - h) > 0.01 for h in hs):
                hs.append(edge)
        hs.sort()

        info = dev.query_device(device_index)
        for h in hs:
            try:
                plan = plan_pair(calib, h, peak_dec_ops=peak_dec,
                                 peak_cov_ops=peak_cov,
                                 safety=cfg.pair_safety_frac)
            except PairInfeasibleError as exc:
                print(f"SKIP h={h:.3f}: {exc}", flush=True)
                continue
            print(f"h={h:.3f}: n_dec={plan.n_dec} n_cov0={plan.n_cov0} "
                  f"busy A {plan.busy_a_s:.2f}s B {plan.busy_b_s:.2f}s "
                  f"gap {plan.e_gap_j:.0f} J", flush=True)

            work_a = make_matmul(
                dataclasses.replace(spec_dear, iters=plan.n_dec),
                device=device, generator=generator)
            work_b_decl = make_matmul(
                dataclasses.replace(spec_cheap, iters=plan.n_dec),
                device=device, generator=generator)

            # measure_b mirrors the EXACT scoring cadence: a full A window
            # (declared + idle pad) precedes the measured B window, so B is
            # tuned in the same thermal state it is scored in. A bare A burst
            # without the pad left B ~4% hot at tuning vs scoring (Slurm
            # 14635584); the 8.5 s pad's cooling is what had to be replicated.
            def measure_b(n_covert: int) -> float:
                w_cov = make_matmul(
                    dataclasses.replace(spec_cov, iters=max(n_covert, 1)),
                    device=device, generator=generator)

                def busy() -> None:
                    work_b_decl()
                    if n_covert > 0:
                        w_cov()

                _measure_window(work_a)            # full padded A window
                return _measure_window(busy)[0].energy_j

            # -- warm BOTH paths to thermal steady state before freezing ---
            # The covert int8 kernel heats the card as it runs; freezing the
            # dose in a cooler regime than it is scored in put E_B ~50 J
            # above E_A during scoring (Slurm 14632684). Collect the warm A
            # windows and tune against their mean.
            warm_a: list[float] = []
            for _ in range(cfg.pair_warmup_windows):
                s_warm, pad_warm = _measure_window(work_a)
                if pad_warm <= 0.0:
                    raise RuntimeError(f"run A overflowed the window at h={h:.3f}")
                warm_a.append(s_warm.energy_j)
                measure_b(plan.n_cov0)
            e_a_ref = float(np.mean(warm_a[-cfg.pair_warmup_windows:]))

            # -- tune against the warm A mean, then freeze -----------------
            # A small undershoot keeps E_B inside the one-sided acceptance
            # region; interleaved measure_b makes the tuned dose hold at
            # scoring, so the pair stays energy-degenerate within tolerance.
            e_target = tuning_target(e_a_ref, 0.5 * cfg.pair_tol_frac)
            res = tune_covert_iters(measure_b, e_target, plan.n_cov0,
                                    calib.m_cov_j,
                                    tol_frac=cfg.pair_tol_frac,
                                    max_rounds=cfg.pair_max_tune)
            if not res.converged:
                raise PairValidationError(
                    f"energy match did not converge at h={h:.3f}: "
                    f"history={list(res.history)}")
            n_cov = res.n_covert
            if n_cov < 1:
                raise PairValidationError(
                    f"tuning drove the covert dose to zero at h={h:.3f} "
                    f"(history={list(res.history)}): no covert work is not a "
                    f"witness")
            print(f"  tuned n_cov={n_cov} in {len(res.history)} round(s); "
                  f"frozen", flush=True)

            # -- scored interleaved repeats at frozen parameters -----------
            work_cov = make_matmul(
                dataclasses.replace(spec_cov, iters=n_cov),
                device=device, generator=generator)

            def busy_b() -> None:
                work_b_decl()
                work_cov()

            repeats_a, repeats_b = [], []
            for i in range(cfg.pair_repeats):
                s_a, pad_a = _measure_window(work_a)
                repeats_a.append(_repeat_dict(s_a, pad_a, 2 * i))
                s_b, pad_b = _measure_window(busy_b)
                repeats_b.append(_repeat_dict(s_b, pad_b, 2 * i + 1))
                print(f"  repeat {i}: E_A {s_a.energy_j:.1f} J "
                      f"E_B {s_b.energy_j:.1f} J", flush=True)

            e_a_list = [r["energy_j"] for r in repeats_a]
            e_b_list = [r["energy_j"] for r in repeats_b]
            threshold = acceptance_threshold(e_a_list, z=DEFAULT_ACCEPT_Z)
            verdicts = acceptance_verdicts(e_b_list, threshold)
            cap_frac = capacity_frac_used(
                covert_ops=float(ops_cov) * n_cov, h=h, wall_s=T,
                peak_ops_per_s=peak_cov)

            pair_id = uuid_mod.uuid4().hex[:12]
            payload = {
                "pair_id": pair_id,
                "window_s": T,
                "h": h,
                "h_max": h_max,
                "h_definition": "C_dec / (F_max_dec * T), nominal ops",
                "declared": {
                    "dtype": cfg.pair_declared_dtype,
                    "shape": list(cfg.pair_declared_shape),
                    "locality": cfg.pair_declared_locality,
                    "dist_dear": cfg.pair_dear_dist,
                    "dist_cheap": cfg.pair_cheap_dist,
                    "iters_a": plan.n_dec, "iters_b": plan.n_dec,
                    "ops_nominal_per_iter": ops_dec,
                },
                "covert": {
                    "dtype": cfg.pair_covert_dtype,
                    "dist": cfg.pair_covert_dist,
                    "shape": list(cfg.pair_covert_shape),
                    "iters": n_cov,
                    "ops_nominal_per_iter": ops_cov,
                    "m_j_per_iter": calib.m_cov_j,
                },
                "calibration": dataclasses.asdict(calib),
                "plan": dataclasses.asdict(plan),
                "repeats_a": repeats_a,
                "repeats_b": repeats_b,
                "tuning": {"n_covert": n_cov, "converged": res.converged,
                           "e_a_ref_j": e_a_ref,
                           "history": [list(x) for x in res.history]},
                "acceptance": {
                    "rule": f"alarm iff E_obs > mean + {DEFAULT_ACCEPT_Z:g}*std "
                            f"of honest repeats",
                    "z": DEFAULT_ACCEPT_Z,
                    "threshold_j": threshold,
                    "threshold_max_j": float(max(e_a_list)),  # sensitivity
                    "verdicts": verdicts,
                    "all_accepted": all(verdicts),
                    "mean_e_a_j": float(np.mean(e_a_list)),
                    "std_e_a_j": float(np.std(e_a_list, ddof=1)),
                    "mean_e_b_j": float(np.mean(e_b_list)),
                    "std_e_b_j": float(np.std(e_b_list, ddof=1)),
                    "delta_frac": abs(float(np.mean(e_a_list))
                                      - float(np.mean(e_b_list)))
                    / float(np.mean(e_a_list)),
                },
                "capacity_frac_used": cap_frac,
                "capacity_feasible": cap_frac < 1.0,
                "tol_frac": cfg.pair_tol_frac,
                "gpu_uuid": info.uuid,
                "gpu_name": info.name,
                "driver_version": info.driver_version,
                "sm_clock_mhz": info.sm_clock_mhz,
                "started_utc": started_utc,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }

            try:
                validate_pair_payload(payload)
            except PairValidationError:
                debug_path = out_dir / f"failed_pair_{pair_id}.json"
                debug_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True))
                print(f"VALIDATION FAILED; diagnostic dump: {debug_path}",
                      flush=True)
                raise

            out_path = out_dir / f"matched_pair_{pair_id}.json"
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            manifest = build_manifest(
                sweep_id=pair_id, experiment=5,
                config=dataclasses.asdict(cfg),
                gpu_uuid=info.uuid, gpu_name=info.name,
                driver_version=info.driver_version,
                git_commit=analysis_git_commit(),
                hostname=socket.gethostname(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                started_utc=started_utc,
                finished_utc=payload["finished_utc"],
                n_records=2 * cfg.pair_repeats,
                outputs=[out_path.name],
            )
            (out_dir / f"matched_pair_{pair_id}_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True))
            print(f"h={h:.3f}: wrote {out_path.name} "
                  f"(all_accepted={all(verdicts)})", flush=True)
            artifact_paths.append(out_path)

            del work_a, work_b_decl, work_cov
            torch.cuda.empty_cache()
    finally:
        meter.close()

    if not artifact_paths:
        raise PairInfeasibleError(
            "no requested utilisation produced a valid witness artifact")
    return artifact_paths
