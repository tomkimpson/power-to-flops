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
    cannot separate declared from covert work or one format from another, so the
    DECLARED BAND widens to the full measured envelope (five tested formats) —
    ceiling at the dearest measured rate (the pooled r_hi, fp32), floor at the
    cheapest (the pooled r_lo, int8 zeros) — while the covert denominator stays
    the int8 marginal edge, which no verifier capability ever reaches. No
    precision is assumed on either side (the earlier fp16-ceiling / int8-floor
    hybrid was the inconsistency the referee flagged). The declared floor and
    the covert floor are never collapsed onto one symbol: same physical corner,
    different algebraic roles, and each role admits only its own estimand
    (see :func:`build_ladder`). The precision rung then keeps the
    fp16-band arithmetic — it IS the headline case — but carries the disclosed
    condition COND_FORMAT, inherited by every rung below. Re-execution transfers
    only at a matched operating point, adding COND_OPERATING_POINT (prospective:
    attested clock telemetry does not yet exist).
  * **Clock & locality rung is unpriced** (priced=False): B6 bans Exp-2's w0
    from pricing a security-guarantee rung, and the trusted on-site observable
    it models is prospective. The row stays as the slot where attested
    telemetry would land; w0 remains descriptive prose.
  * **The bottom rung pins the overhead band to the priced window** — the
    Exp-3 idle cells compatible with resident operands
    (:func:`powertoflops.measured.resident_overhead_band`), narrowing dE0 alone. It
    replaced "covert at the top of the useful band", which moved the covert
    denominator to the top of the marginal useful band: that rung read as a
    strengthening but priced a *weaker* adversary than the rung above it, so it
    said nothing the useful-data rung had not.
  * **The covert-format branch is not a rung** (:func:`build_covert_format_branch`).
    Assuming the covert format buys both a dearer denominator and kappa = 1,
    but it acts on the same dial as the useful-data rung, so it is priced only
    where its denominator is measured — see that function.

The first-round defect this module retires: the old script fed the useful
band's *width* into the declared-side numerator, so a rung where the verifier
gained a capability WIDENED the identified set (0.16 -> 0.43). Here the
declared width below the re-execution rung is always the hardware gap, and
only the covert edge moves. :func:`check_monotone` enforces the referee's
acceptance criterion that strengthening the verifier never worsens the bound.

Every priced input is validated against the measured artifact (same guard
style as :func:`powertoflops.joint_model.validate_joint_case`). The bare rung reads
the pooled band [r_lo, r_hi] — that IS the power-only declared band by
construction (subsec:poweronly), and both edges enter only the numerator, never
as a covert denominator; a total quotient is the correct estimand there. Every
covert denominator on every rung is a marginal LCB. w0 and the Exp-2 operand
quotient r_lo_op are never read (B6).
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
    resident_overhead_band,
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

# The covert-format branch's own disclosed condition: covert work is assumed to
# run in a known format. Unlike COND_FORMAT this can never be checked — no
# capability reaches work the meter does not resolve — so it rests on a
# threat-model argument about what the stolen capacity is for.
COND_COVERT_FORMAT = "covert-format assumption"

DECLARED_FORMAT = "fp16"
COVERT_FORMAT = "int8"
BRANCH_COVERT_FORMAT = "fp16"


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
    dE0: float | None = None,
    covert_format: str = COVERT_FORMAT,
) -> Rung:
    """Price one rung. ``dE0`` defaults to the unrestricted overhead width;
    ``covert_format`` sets the capacity ratio kappa (the branch's only lever
    besides its denominator)."""
    kappa = A100_DENSE_PEAK_OPS[covert_format] / A100_DENSE_PEAK_OPS[DECLARED_FORMAT]
    args = dict(r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
                dE0=mb.P0_hi - mb.P0_lo if dE0 is None else dE0,
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
    useful_lo, _ = useful_marginal_edge(mb)        # qblock-marginal dose LCB (B5)

    # Rung 0 — bare meter, unconditional: the genuine power-only floor of
    # subsec:poweronly. A bare meter cannot separate declared from covert work,
    # nor one format from another, so the DECLARED BAND widens to the full
    # measured envelope (five tested formats; referee B4, third round): ceiling
    # at the dearest measured rate (the pooled r_hi, fp32), floor at the
    # cheapest (the pooled r_lo, int8 zeros). No precision is assumed on either
    # side.
    #
    # The declared floor is mb.r_lo, NOT the covert edge. The two describe the
    # same physical corner of the hardware, but they sit in different algebraic
    # roles and each admits only its own estimand (app:estimands): the declared
    # band enters through a DIFFERENCE, so total quotients suffice and mb.r_lo
    # is the right number; the covert floor is a DIVISOR, so it admits only the
    # one-sided marginal LCB. Collapsing them onto one symbol — the earlier
    # r_lo_dec=edge — asked a single number to carry both estimands, and mixed
    # them by a factor 1.5 (0.786 vs 0.515 pJ/op). Numerically the collapse
    # barely moved the rung (beta* 1.954 -> 1.953, since r_hi dominates the
    # difference), but it left the paper unable to say which estimand the rung
    # used, so the manuscript carried an apologetic "mixture of estimands"
    # caveat that this de-collapse retires.
    #
    # Because the declared side is not pinned to a format either, the cheating
    # schedule executes the declared ops in the cheap covert format, occupying
    # h/kappa of resource-time rather than h: the format-consistent capacity
    # clip is kappa - h, not (1 - h) kappa (kappa_dec = kappa; numerical-audit
    # finding 2).
    kappa = A100_DENSE_PEAK_OPS[COVERT_FORMAT] / A100_DENSE_PEAK_OPS[DECLARED_FORMAT]
    bare = _priced(mb, "bare meter", None, (),
                   r_hi_dec=mb.r_hi, r_lo_dec=mb.r_lo, r_lo_cov=edge,
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
    # only the covert edge and the overhead width move (the correct cumulative
    # arithmetic — the useful band's width never re-enters the declared
    # numerator). Rungs 4-5 price the covert denominator at the low edge of the
    # MARGINAL useful band (qblock-marginal dose-response, useful_lo), not the
    # retired total-quotient useful band, so baseline power does not inflate
    # the covert cost (B5). Its high edge prices nothing: a dearer covert
    # option is a weaker adversary, and the bound takes the cheapest.
    #
    # The residual band is anchored at the DECLARED floor these rungs inherit
    # from the precision rung (fp16's own r_lo), not at the covert edge. Only
    # the width r_hi_dec - r_lo_dec enters the floor, so the anchor is
    # numerically inert — but anchoring on the covert edge would assert
    # r_lo_dec == r_lo_cov, which is the estimand collapse the bare rung above
    # exists to avoid (see the rung-0 comment). Same width, honest symbols.
    chain = (COND_FORMAT, COND_OPERATING_POINT)
    dec_floor, _ = format_band(mb, DECLARED_FORMAT)
    reexec = _priced(mb, "declared work re-executed", "C", chain,
                     r_hi_dec=dec_floor + gap, r_lo_dec=dec_floor, r_lo_cov=edge)
    useful = _priced(mb, "covert work on useful data", "M", chain,
                     r_hi_dec=dec_floor + gap, r_lo_dec=dec_floor, r_lo_cov=useful_lo)

    # Rung 5 — the overhead band pinned to the window being priced. The fourth
    # dial of the manuscript's eq:betadials, and like the useful-data rung it
    # moves under a threat-model restriction rather than a gained capability:
    # the corner being bounded has the card executing throughout, so its
    # operands are resident and the cold-card idle cells (no CUDA context, or a
    # context with no operand pool) are states that window cannot be in. Only
    # dE0 moves; the covert denominator stays the useful edge inherited above.
    P0_lo_res, P0_hi_res = resident_overhead_band(mb)
    pinned = _priced(mb, "overhead band pinned to the window", "M", chain,
                     r_hi_dec=dec_floor + gap, r_lo_dec=dec_floor,
                     r_lo_cov=useful_lo, dE0=P0_hi_res - P0_lo_res)

    return [bare, precision, clock, reexec, useful, pinned]


def build_covert_format_branch(mb: MeasuredBands, *,
                               gap: float = HW_GAP) -> list[Rung]:
    """The covert-format branch of tab:ladder: covert work known to be fp16.

    Priced apart from :func:`build_ladder`'s chain, and deliberately not
    appended to it. Assuming the covert format moves two things at once — the
    covert denominator rises from the int8 marginal edge to the fp16 one, and
    the capacity ratio falls to kappa = 1, halving the ceiling rather than
    narrowing the width beneath it — but it reaches the SAME dial as the
    useful-data rung, so the two do not compose. Stacking them needs the
    marginal cost of a *useful fp16* block, which is unmeasured: scaling the
    measured int8 block by the ratio of silicon edges has no mechanism behind
    it (the block sits ~5-6x above that edge, so its cost is data movement and
    quantisation overhead, not arithmetic). The branch is therefore quoted at
    the two rungs where its denominator is measured and no further.

    Returned rungs are NOT part of the monotone chain :func:`check_monotone`
    checks; they are re-pricings of the precision and re-execution rungs.
    """
    branch_edge = covert_edge(mb, BRANCH_COVERT_FORMAT)
    dec_lo, dec_hi = format_band(mb, DECLARED_FORMAT)
    at_precision = _priced(
        mb, "branch: covert format known, at the precision rung",
        "A", (COND_FORMAT, COND_COVERT_FORMAT),
        r_hi_dec=dec_hi, r_lo_dec=dec_lo, r_lo_cov=branch_edge,
        covert_format=BRANCH_COVERT_FORMAT)
    at_reexec = _priced(
        mb, "branch: covert format known, at the re-execution rung",
        "A", (COND_FORMAT, COND_OPERATING_POINT, COND_COVERT_FORMAT),
        r_hi_dec=dec_lo + gap, r_lo_dec=dec_lo, r_lo_cov=branch_edge,
        covert_format=BRANCH_COVERT_FORMAT)
    return [at_precision, at_reexec]


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
    # Anchored at the declared floor, not the covert edge: only the width enters,
    # and collapsing the two floors onto one symbol is what subsec:poweronly
    # forbids (see :func:`build_ladder`).
    _, beta = beta_phys_worst_case(
        r_hi_dec=r_lo_dec + width, r_lo_dec=r_lo_dec, r_lo_cov=edge,
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
