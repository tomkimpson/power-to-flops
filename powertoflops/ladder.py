"""Capability ladder priced under the joint model (referee B4).

Rebuilds tab:ladder / fig:ladder of the manuscript as a *cumulative* chain: each
rung grants the verifier one more capability on top of the rungs above it and
is re-priced as the capacity-clipped physical worst case
max_h beta_phys(h) (:func:`powertoflops.floor.beta_phys_worst_case`) under the B1
joint model — fp16-declared case study (beta in machines of fp16 capacity,
user decision 2026-07-22), int8 covert work at the Exp-7 marginal LCB
(:func:`powertoflops.measured.covert_edge`), kappa = F_max^int8 / F_max^fp16. S(1) is
carried alongside only as the explicitly labelled energy-only diagnostic
(beta_phys(1) = 0 for every single-format rung; the bare rung's cross-format
clip leaves beta_phys(1) = kappa - 1, still far below its S(1), so S(1) is
nowhere the physical floor).

Semantics fixed by the second-round review (user decisions 2026-07-23):

  * **Conditional chain.** The *unconditional* bare rung is the genuine
    power-only floor of subsec:poweronly (referee B4, third round): a bare meter
    cannot separate declared from covert work or one format from another, so
    both band edges are the full measured envelope (five tested formats) — the
    honest ceiling is the
    dearest measured rate (the pooled r_hi, fp32) and the covert/declared floor
    is the cheapest incremental covert op (the int8 marginal edge). No precision
    is assumed on either side (the earlier fp16-ceiling / int8-floor hybrid was
    the inconsistency the referee flagged). The precision rung then keeps the
    fp16-band arithmetic — it IS the headline case — but carries the disclosed
    condition COND_FORMAT, inherited by every rung below. Re-execution transfers
    only at a matched operating point, adding COND_OPERATING_POINT (prospective:
    attested clock telemetry does not yet exist).
  * **Clock & locality rung is unpriced** (priced=False): B6 bans Exp-2's w0
    from pricing a security-guarantee rung, and the trusted on-site observable
    it models is prospective. The row stays as the slot where attested
    telemetry would land; w0 remains descriptive prose.
  * **The bottom rung is "covert at the top of the useful band"** — the top of
    the MARGINAL useful band (qblock-marginal dose-response, :func:`powertoflops.measured.
    useful_marginal_edge`), replacing the w0-derived dense rate; monotone below
    the useful-data rung by construction.

The first-round defect this module retires: the old script fed the useful
band's *width* into the declared-side numerator, so a rung where the verifier
gained a capability WIDENED the identified set (0.16 -> 0.43). Here the
declared width below the re-execution rung is always the hardware gap, and
only the covert edge moves. :func:`check_monotone` enforces the referee's
acceptance criterion that strengthening the verifier never worsens the bound.

Every priced input is validated against the measured artifact (same guard
style as :func:`powertoflops.joint_model.validate_joint_case`). The bare rung reads
the pooled ceiling r_hi — that IS the power-only band by construction
(subsec:poweronly), and it enters only as a numerator-widening honest ceiling,
never as a covert denominator. w0, the pooled floor r_lo, and the pooled/Exp-2
covert edge r_lo_cov are never read (B6).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import A100_DENSE_PEAK_OPS
from .floor import beta_phys_worst_case, energy_slack
from .joint_model import headline
from .measured import (
    MeasuredBands,
    covert_edge,
    format_band,
    useful_marginal_edge,
)

# Residual declared-band width after re-execution at a matched operating
# point: the prover-vs-verifier hardware gap. An assumption informed by the
# measured same-SKU spread (2-3%), not a cross-hardware measurement — the
# manuscript presents the rung as a sensitivity analysis in this number.
HW_GAP = 0.05e-12  # J/op

# Ratio between the dearest and cheapest energy per operation the SAME declared
# computation can exhibit across the DVFS range when the verifier cannot observe
# the operating point. Used only by :func:`power_and_paperwork_stall`, the
# "what if re-execution transfers without a matched clock" sensitivity the
# manuscript quotes; an order-of-magnitude assumption, not a measurement.
DVFS_FACTOR = 1.5

# Disclosed conditions of the conditional chain (referee B4, option
# "label conditional"): rungs carrying COND_FORMAT assume the declared work is
# executed in its declared format (off-site checking cannot enforce this;
# facility-local format enforcement is the open capability). COND_OPERATING_POINT
# is re-execution's clock-matched transfer requirement; PROSPECTIVE marks
# capabilities whose trusted read path does not yet exist.
COND_FORMAT = "declared-format execution"
COND_OPERATING_POINT = "operating point observed"
PROSPECTIVE = "prospective"

DECLARED_FORMAT = "fp16"
COVERT_FORMAT = "int8"


class LadderMonotonicityError(ValueError):
    """A rung that strengthens the verifier reports a wider identified set."""


@dataclass(frozen=True)
class Rung:
    """One ladder rung: capability semantics plus its priced worst case.

    Unpriced rungs (priced=False) carry None in every numeric field. For
    priced rungs, (r_hi_dec, r_lo_dec) is the residual declared band the
    cheater can exploit, r_lo_cov the covert-side denominator edge, and
    (beta_star, h_star) = max_h beta_phys(h) with S1 the energy-only
    diagnostic S(1).
    """

    name: str
    rung_type: str | None            # D/C/T/M/A tags as in tab:ladder
    conditions: tuple[str, ...]
    priced: bool
    r_hi_dec: float | None = None
    r_lo_dec: float | None = None
    r_lo_cov: float | None = None
    beta_star: float | None = None
    h_star: float | None = None
    S1: float | None = None


def _priced(
    mb: MeasuredBands,
    name: str,
    rung_type: str | None,
    conditions: tuple[str, ...],
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    kappa_dec: float = 1.0,
) -> Rung:
    kappa = A100_DENSE_PEAK_OPS[COVERT_FORMAT] / A100_DENSE_PEAK_OPS[DECLARED_FORMAT]
    args = dict(r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
                dE0=mb.P0_hi - mb.P0_lo,
                C_max=A100_DENSE_PEAK_OPS[DECLARED_FORMAT])
    h_star, beta_star = beta_phys_worst_case(**args, kappa=kappa,
                                             kappa_dec=kappa_dec)
    return Rung(
        name=name, rung_type=rung_type, conditions=conditions, priced=True,
        r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
        beta_star=beta_star, h_star=h_star,
        S1=float(energy_slack(1.0, **args)),
    )


def build_ladder(mb: MeasuredBands, *, gap: float = HW_GAP) -> list[Rung]:
    """The six rungs of tab:ladder, priced from the measured artifact.

    ``gap`` is the re-execution hardware-gap assumption [J/op]; the manuscript
    quotes the default and a sensitivity sweep around it.
    """
    edge = covert_edge(mb, COVERT_FORMAT)          # Exp-7 marginal LCB [J/op]
    useful_lo, useful_hi = useful_marginal_edge(mb)  # qblock-marginal dose LCBs (B5)

    # Rung 0 — bare meter, unconditional: the genuine power-only floor of
    # subsec:poweronly. A bare meter cannot separate declared from covert work,
    # nor one format from another, so both edges are the full measured envelope
    # (five tested formats; referee B4, third round): the honest ceiling is the
    # dearest measured rate
    # (the pooled r_hi, fp32) and the covert/declared floor is the cheapest
    # incremental covert op (the int8 marginal edge). No precision is assumed on
    # either side. r_hi enters only as a numerator-widening honest ceiling (a
    # total quotient there is conservative); the covert edge stays the Exp-7
    # marginal LCB (B2). Because the declared side is not pinned to a format
    # either, the cheating schedule executes the declared ops in the cheap
    # covert format, occupying h/kappa of resource-time rather than h: the
    # format-consistent capacity clip is kappa - h, not (1 - h) kappa
    # (kappa_dec = kappa; numerical-audit finding 2).
    kappa = A100_DENSE_PEAK_OPS[COVERT_FORMAT] / A100_DENSE_PEAK_OPS[DECLARED_FORMAT]
    bare = _priced(mb, "bare meter", None, (),
                   r_hi_dec=mb.r_hi, r_lo_dec=edge, r_lo_cov=edge,
                   kappa_dec=kappa)

    # Rung 1 — precision declared & checked. CONDITIONAL: the fp16-band
    # arithmetic assumes declared-format execution, which the off-site check
    # cannot enforce. This case is exactly the B1 headline; reuse it so the
    # cross-link can never drift.
    got = headline(mb, DECLARED_FORMAT, COVERT_FORMAT)
    case = got["case"]
    precision = Rung(
        name="precision declared & checked", rung_type="D,C",
        conditions=(COND_FORMAT,), priced=True,
        r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec, r_lo_cov=case.r_lo_cov,
        beta_star=got["beta_star"], h_star=got["h_star"], S1=got["S1"],
    )

    # Rung 2 — clock & locality observed. Unpriced (B6: w0 is descriptive
    # evidence, not a bound; attested telemetry is prospective). Kept as the
    # slot a trusted on-site observable would fill.
    clock = Rung(
        name="clock & locality observed", rung_type="T",
        conditions=(COND_FORMAT, PROSPECTIVE), priced=False,
    )

    # Rungs 3-5 — the declared band collapses to the hardware gap once the
    # declared work is re-executed at a matched operating point; after that
    # only the covert edge moves (the correct cumulative arithmetic — the
    # useful band's width never re-enters the declared numerator). Rungs 4-5
    # price the covert denominator at the MARGINAL useful edge (qblock-marginal
    # dose-response, useful_lo/useful_hi), not the retired total-quotient
    # useful band, so baseline power does not inflate the covert cost (B5).
    chain = (COND_FORMAT, COND_OPERATING_POINT)
    reexec = _priced(mb, "declared work re-executed", "C", chain,
                     r_hi_dec=edge + gap, r_lo_dec=edge, r_lo_cov=edge)
    useful = _priced(mb, "covert work on useful data", "M", chain,
                     r_hi_dec=edge + gap, r_lo_dec=edge, r_lo_cov=useful_lo)
    top = _priced(mb, "covert work at top of useful band", "A", chain,
                  r_hi_dec=edge + gap, r_lo_dec=edge, r_lo_cov=useful_hi)

    return [bare, precision, clock, reexec, useful, top]


def power_and_paperwork_stall(mb: MeasuredBands) -> float:
    """beta*_phys when re-execution transfers WITHOUT an observed clock.

    The "power plus paperwork" verifier of the manuscript: it holds
    declarations and off-site re-execution but cannot see the operating point,
    so the residual declared width is the DVFS-wide error
    (:data:`DVFS_FACTOR` - 1) times the mid-band declared rate rather than the
    matched-point hardware gap :data:`HW_GAP`. The rung stalls near one whole
    machine, which is the point being made: paperwork alone does not buy the
    re-execution rung's collapse to ~0.34.
    """
    r_lo_dec, r_hi_dec = format_band(mb, DECLARED_FORMAT)
    edge = covert_edge(mb, COVERT_FORMAT)
    width = (DVFS_FACTOR - 1.0) * 0.5 * (r_lo_dec + r_hi_dec)
    _, beta = beta_phys_worst_case(
        r_hi_dec=edge + width, r_lo_dec=edge, r_lo_cov=edge,
        dE0=mb.P0_hi - mb.P0_lo,
        C_max=A100_DENSE_PEAK_OPS[DECLARED_FORMAT],
        kappa=A100_DENSE_PEAK_OPS[COVERT_FORMAT] / A100_DENSE_PEAK_OPS[DECLARED_FORMAT])
    return beta


def check_monotone(rungs: list[Rung]) -> None:
    """Referee acceptance criterion: a stronger verifier never does worse.

    Raises :class:`LadderMonotonicityError` unless beta_star is non-increasing
    along the priced sub-chain (unpriced rungs are skipped, not compared).
    """
    priced = [r for r in rungs if r.priced]
    for above, below in zip(priced, priced[1:]):
        if below.beta_star > above.beta_star:
            raise LadderMonotonicityError(
                f"rung {below.name!r} (beta* = {below.beta_star:.4f}) is wider "
                f"than the weaker rung {above.name!r} "
                f"(beta* = {above.beta_star:.4f}) — a gained capability must "
                f"not widen the identified set")
