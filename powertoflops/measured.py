"""Wire measured A2 bands into the config without disturbing the placeholders.

:data:`powertoflops.config.DEFAULT` holds Horowitz-2014 *placeholder* bands so the A0/A1
tests and figures stay reproducible. A2 measures the real bands (paper Sec. 3.6;
:mod:`powertoflops.bench.bands`) and writes them to ``results/bench/measured_bands.json``.
This module loads that artifact and returns a *new* frozen :class:`Config` with
the six measured fields swapped in — ``DEFAULT`` is never mutated, so every
existing test and placeholder figure is inert to the measurement. A measured
beta(0) is then just ``floor.beta_bare(...)`` fed the overridden config (see
``scripts/plot_beta_phys.py``).

No GPU / NVML dependency: this is plain dataclass plumbing over JSON.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config, DEFAULT


@dataclass(frozen=True)
class MeasuredBands:
    """The measured band fields, plus measurement provenance.

    Maps onto config placeholders as:
        r_lo, r_hi      -> FloorParams.r_lo, .r_hi      (Exp 1 exchange band)
        P0_lo, P0_hi    -> FloorParams.P0_lo, .P0_hi    (Exp 3 overhead band)
        sigma_dec       -> ChannelParams.sigma_dec      (Exp 2 declared-typical)

    ``FloorParams.r_lo_cov`` -- the covert denominator -- is NOT taken from
    Exp 2. It is the Exp-7 marginal dose slope, reached via
    :func:`covert_edge`; see :func:`config_with_measured`.

    ``r_lo_op`` and ``w0`` (= r_hi_op/r_lo_op) are Exp-2 total quotients kept
    for reporting only. They are descriptive -- the operand families ran at
    different governor-selected clocks -- and price no bound.
    """

    r_lo: float
    r_hi: float
    r_lo_op: float
    w0: float
    P0_lo: float
    P0_hi: float
    sigma_dec: float

    # Useful-operand band (R2.7 qblock; analyze_bands --qblock). None when the
    # artifact predates the measurement — consumers must fail loudly rather
    # than fall back to the retired hard-coded floor (review C6). Since the
    # B5 rebuild these are one-sided 95% LCB endpoints from the VALIDATED
    # trained-weight benchmark; ``useful_band`` carries the versioned detail
    # block (version >= 2), whose absence marks a retired pre-B5 artifact.
    r_useful_lo: float | None = None
    r_useful_hi: float | None = None
    useful_band: dict | None = None

    # MARGINAL useful-operand band (qblock-marginal dose-response; referee B5, third
    # round). The incremental covert cost of useful transformer work, from an
    # Exp-7-style dose sweep with the qblock forward as the swept covert dose
    # (analyze_bands.py --qblock-marg). Both endpoints are one-sided 95% LOWER
    # limits on the dose slope dE/dC_cov, so baseline power cancels and the
    # ladder denominator is a genuine marginal lower bound — unlike ``useful_*``
    # above, which is a total quotient E/C (anti-conservative). None until the
    # qblock-marginal capture lands; ``useful_marginal_band`` carries the versioned block.
    r_useful_marg_lo: float | None = None
    r_useful_marg_hi: float | None = None
    useful_marginal_band: dict | None = None

    # Per-declared-format exchange bands {dtype: {r_lo, r_hi, n_cells}}
    # (referee B1: the pooled [r_lo, r_hi] mixes formats, so the joint model
    # prices the declared side inside ONE format's band via
    # :func:`format_band`). None when the artifact predates the split.
    format_bands: dict | None = None

    # Exp-3 fitted dP/dF [J/op] (was silently dropped by the loader; kept as
    # a diagnostic beside the Exp-7 estimand below).
    r_marginal: float | None = None

    # Exp-7 marginal covert cost per covert format:
    # {dtype: {r_marg, r_marg_lcb95, se_slope, r2, ...}} — the asdict of
    # :class:`powertoflops.bench.bands.MarginalCovertCost`. The floor's covert
    # denominator is the one-sided LCB via :func:`covert_edge`. None when the
    # artifact predates Exp 7 — consumers must fail loudly (referee B2).
    r_marg_cov: dict | None = None

    # provenance
    gpu_name: str = "unknown"
    driver_version: str = "unknown"
    source_path: str = ""
    measured_utc: str = ""


def load_measured_bands(path: str | Path) -> MeasuredBands:
    """Load a ``measured_bands.json`` artifact (as written by analyze_bands.py).

    Unknown keys in the JSON are ignored so the artifact can carry extra
    diagnostics (e.g. affine_r2, r_marginal, P0_fit_intercept) without breaking
    the loader.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(MeasuredBands)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs["source_path"] = str(path)   # stamp the real load path (overrides any in-file value)
    return MeasuredBands(**kwargs)


def format_band(mb: MeasuredBands, fmt: str) -> tuple[float, float]:
    """The declared exchange band (r_lo, r_hi) [J/op] of ONE format.

    Fails loudly on artifacts that predate the per-format split (referee B1):
    falling back to the pooled band would silently reintroduce the cross-format
    mix the split exists to prevent.
    """
    if not mb.format_bands:
        raise ValueError(
            f"artifact {mb.source_path or '<unknown>'} carries no per-format "
            f"bands (format_bands) — regenerate it with scripts/analyze_bands.py "
            f"from this checkout")
    if fmt not in mb.format_bands:
        raise ValueError(
            f"no measured band for format {fmt!r}; artifact has "
            f"{sorted(mb.format_bands)}")
    band = mb.format_bands[fmt]
    return float(band["r_lo"]), float(band["r_hi"])


def covert_edge(mb: MeasuredBands, fmt: str) -> float:
    """The verifier-safe covert energy cost [J/op] for covert format ``fmt``.

    This is the one-sided LOWER confidence limit on the Exp-7 marginal slope
    (beta has the covert cost in its denominator, so bounding beta from above
    needs the cost bounded from below). Fails loudly on artifacts that predate
    Exp 7 rather than falling back to a total-E/C edge (referee B2).
    """
    if not mb.r_marg_cov:
        raise ValueError(
            f"artifact {mb.source_path or '<unknown>'} carries no Exp-7 "
            f"marginal covert costs (r_marg_cov) — capture Exp 7 and "
            f"regenerate with scripts/analyze_bands.py --exp7")
    if fmt not in mb.r_marg_cov:
        raise ValueError(
            f"no Exp-7 marginal covert cost for format {fmt!r}; artifact has "
            f"{sorted(mb.r_marg_cov)}")
    return float(mb.r_marg_cov[fmt]["r_marg_lcb95"])


def useful_marginal_edge(mb: MeasuredBands) -> tuple[float, float]:
    """The marginal useful-cost band (r_lo, r_hi) [J/op] for the ladder rungs.

    Both endpoints are one-sided LOWER limits on the qblock-marginal dose slope dE/dC_cov
    (referee B5, third round). Fails loudly on artifacts that predate the qblock-marginal
    capture rather than falling back to the total-quotient ``r_useful`` band,
    which is anti-conservative as a covert denominator (referee B2).
    """
    if mb.r_useful_marg_lo is None or mb.r_useful_marg_hi is None:
        raise ValueError(
            f"artifact {mb.source_path or '<unknown>'} carries no marginal "
            f"useful band (r_useful_marg) — capture qblock-marginal with "
            f"scripts/run_qblock_marginal.py and regenerate with "
            f"scripts/analyze_bands.py --qblock-marg (referee B5)")
    return float(mb.r_useful_marg_lo), float(mb.r_useful_marg_hi)


def config_with_measured(mb: MeasuredBands, base: Config = DEFAULT) -> Config:
    """Return a new Config with the measured fields overriding ``base``.

    Uses :func:`dataclasses.replace` so ``base`` (default ``DEFAULT``) is left
    untouched. Everything not measured here — hfu_dec, F_max, T, z_gamma,
    r_bar_dec, the SimParams and BenchConfig — is carried through unchanged.

    ``floor.r_lo_cov`` is the int8 Exp-7 marginal LCB via :func:`covert_edge`,
    NOT an Exp-2 total quotient: the covert cost divides the energy slack, so
    it must be a one-sided lower bound on the *incremental* cost of an added
    covert operation. An Exp-2 quotient amortises baseline power the covert
    work does not draw and would understate beta.
    """
    floor = dataclasses.replace(
        base.floor,
        r_lo=mb.r_lo,
        r_hi=mb.r_hi,
        r_lo_cov=covert_edge(mb, "int8"),
        P0_lo=mb.P0_lo,
        P0_hi=mb.P0_hi,
    )
    channel = dataclasses.replace(base.channel, sigma_dec=mb.sigma_dec)
    return dataclasses.replace(base, floor=floor, channel=channel)
