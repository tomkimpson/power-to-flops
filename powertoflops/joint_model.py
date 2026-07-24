"""Jointly format-consistent headline floor (referee B1) — the ONLY headline path.

The first-round headline mixed formats: an fp32 declared upper edge and an int8
covert floor against fp16 capacity with kappa = 2 — extrema that cannot coexist
in one observation window (at the old h* = 0.07 the dear-fp32 honest reading
needs 21.8 Top/s of fp32 against a 19.5 Top/s peak). This module replaces that
with a joint feasible set over ONE declared format and ONE covert format:

  * the declared band [r_lo_dec, r_hi_dec] comes from the declared format's own
    Exp-1 cells (powertoflops.measured.format_band), and h is utilisation of the
    declared format's OWN peak, so every honest interpretation is attainable by
    construction;
  * the covert cost is the Exp-7 marginal lower confidence limit for the covert
    format (powertoflops.measured.covert_edge) — the B2 estimand, baseline power
    excluded;
  * capacity: declared and covert work time-multiplex a single window; covert
    work fills the residual (1 - h) share at the covert format's peak, giving
    the clip (1 - h) * kappa with kappa = F_max^cov / F_max^dec. Format
    switching between the two streams is allowed and its overhead is treated
    as zero — the adversary-favorable direction: any real switching cost only
    shrinks the covert share, so the clip is an upper envelope.

Every case is validated against the measured artifact and the config peaks
before a number is produced (:func:`validate_joint_case` raises
:class:`FormatMixError` on any cross-format combination), and the regression
test tests/test_joint_model.py pins the rejection of the old mixed inputs.

The headline scenario is the fp16-declared case study (user decision
2026-07-22): maximising over declared formats is a unit artifact, not physics —
beta is measured in machines of DECLARED capacity, so a slow declared format
(fp32, kappa = 32 against int8) inflates beta by shrinking the yardstick.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import A100_DENSE_PEAK_OPS
from .floor import beta_phys_worst_case, energy_slack
from .measured import MeasuredBands, covert_edge, format_band


class FormatMixError(ValueError):
    """A floor input combines quantities from incompatible formats."""


@dataclass(frozen=True)
class JointCase:
    """One jointly feasible (declared format, covert format) floor scenario.

    ``C_max`` is the per-second form F_max^dec [op/s] and ``dP0`` the overhead
    POWER band width [W]: the window length T cancels between them in the
    overhead term dE0 / (r_lo_cov * F_max^dec * T), so the scalar floor needs
    no T (same convention as scripts/plot_beta_phys.py).
    """

    declared_format: str
    covert_format: str
    r_lo_dec: float            # declared-format band floor [J/op]
    r_hi_dec: float            # declared-format band ceiling [J/op]
    r_lo_cov: float            # covert marginal cost LCB [J/op]
    C_max: float               # declared-format peak rate [op/s]
    kappa: float               # F_max^cov / F_max^dec
    dP0: float                 # overhead power band width [W]


def _close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def validate_joint_case(
    case: JointCase, mb: MeasuredBands, *, rtol: float = 1e-9,
) -> None:
    """Raise :class:`FormatMixError` unless every input belongs to its format.

    Checks, against the measured artifact and the config peaks:
      * (r_lo_dec, r_hi_dec) are exactly the declared format's own Exp-1 band;
      * r_lo_cov is exactly the covert format's Exp-7 marginal LCB;
      * C_max is the declared format's peak and kappa the covert/declared
        peak ratio (both from config.A100_DENSE_PEAK_OPS).

    This is the B1 regression guard: reconstructing the first-round headline
    inputs (pooled fp32 upper edge + int8 floor + fp16 capacity) fails here.
    """
    for role, fmt in (("declared", case.declared_format),
                      ("covert", case.covert_format)):
        if fmt not in A100_DENSE_PEAK_OPS:
            raise FormatMixError(
                f"unknown {role} format {fmt!r}; peaks exist for "
                f"{sorted(A100_DENSE_PEAK_OPS)}")

    r_lo, r_hi = format_band(mb, case.declared_format)
    if not (_close(case.r_lo_dec, r_lo, rtol) and _close(case.r_hi_dec, r_hi, rtol)):
        raise FormatMixError(
            f"declared band [{case.r_lo_dec:.6e}, {case.r_hi_dec:.6e}] J/op is "
            f"not the measured {case.declared_format} band "
            f"[{r_lo:.6e}, {r_hi:.6e}] — declared-side extrema must come from "
            f"the declared format's own cells")

    edge = covert_edge(mb, case.covert_format)
    if not _close(case.r_lo_cov, edge, rtol):
        raise FormatMixError(
            f"covert edge {case.r_lo_cov:.6e} J/op is not the "
            f"{case.covert_format} Exp-7 marginal LCB {edge:.6e}")

    peak_dec = A100_DENSE_PEAK_OPS[case.declared_format]
    peak_cov = A100_DENSE_PEAK_OPS[case.covert_format]
    if not _close(case.C_max, peak_dec, rtol):
        raise FormatMixError(
            f"C_max {case.C_max:.4e} op/s is not the {case.declared_format} "
            f"peak {peak_dec:.4e}")
    if not _close(case.kappa, peak_cov / peak_dec, rtol):
        raise FormatMixError(
            f"kappa {case.kappa:.4f} is not "
            f"F_max[{case.covert_format}]/F_max[{case.declared_format}] = "
            f"{peak_cov / peak_dec:.4f}")


def headline(
    mb: MeasuredBands,
    declared: str = "fp16",
    covert: str = "int8",
) -> dict:
    """Build, validate, and solve one joint floor case from the artifact.

    Returns {h_star, beta_star, S1, s, b, case}: the worst-case utilisation and
    clipped floor (via floor.beta_phys_worst_case), the energy-slack index at
    full utilisation S(1), and the slack line's slope s and intercept
    b = S(0). Energy is the binding constraint for h < h_star, capacity for
    h > h_star, and both bind at h_star (when b < kappa).
    """
    r_lo_dec, r_hi_dec = format_band(mb, declared)
    case = JointCase(
        declared_format=declared,
        covert_format=covert,
        r_lo_dec=r_lo_dec,
        r_hi_dec=r_hi_dec,
        r_lo_cov=covert_edge(mb, covert),
        C_max=A100_DENSE_PEAK_OPS[declared],
        kappa=A100_DENSE_PEAK_OPS[covert] / A100_DENSE_PEAK_OPS[declared],
        dP0=mb.P0_hi - mb.P0_lo,
    )
    validate_joint_case(case, mb)

    args = dict(r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec,
                r_lo_cov=case.r_lo_cov, dE0=case.dP0, C_max=case.C_max)
    h_star, beta_star = beta_phys_worst_case(**args, kappa=case.kappa)
    s = (case.r_hi_dec - case.r_lo_dec) / case.r_lo_cov
    b = case.dP0 / (case.r_lo_cov * case.C_max)
    return {
        "h_star": h_star,
        "beta_star": beta_star,
        "S1": float(energy_slack(1.0, **args)),
        "s": s,
        "b": b,
        "case": case,
    }
