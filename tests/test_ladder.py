"""Capability-ladder rebuild (referee B4, no GPU).

The first-round ladder was non-cumulative and mislabeled: the useful-operand
rung fed the covert band *width* into the declared-side numerator (so a rung
where the verifier gains a capability WIDENED the set, 0.16 -> 0.43), the
table quoted S(1) while the figure said beta, and the precision rung's
arithmetic excluded secretly-cheaper execution of declared work that no
off-site check can rule out. These tests pin the repair: powertoflops.ladder prices
every rung as max_h beta_phys(h) under the B1 joint model (fp16-declared case
study, int8 covert at the Exp-7 marginal LCB), the priced chain is monotone
non-increasing (referee-required), the clock rung is unpriced/prospective
(B6: w0 may not price a guarantee rung), and rungs below the precision rung
carry the disclosed declared-format-execution condition.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from powertoflops.config import A100_DENSE_PEAK_OPS
from powertoflops.floor import beta_phys_worst_case, energy_slack
from powertoflops.joint_model import headline
from powertoflops.ladder import (
    COND_COVERT_FORMAT,
    COND_FORMAT,
    COND_OPERATING_POINT,
    DVFS_FACTOR,
    LadderMonotonicityError,
    build_covert_format_branch,
    build_ladder,
    check_monotone,
    power_and_paperwork_stall,
)
from powertoflops.measured import MeasuredBands, covert_edge, load_measured_bands

ARTIFACT = pathlib.Path(__file__).resolve().parent.parent / (
    "results/bench/measured_bands.json")


def _mb(**over) -> MeasuredBands:
    """Artifact stub shaped like a post-Exp-7, post-qblock measured_bands.json."""
    base = dict(
        r_lo=0.8e-12, r_hi=20.0e-12, r_lo_op=1.2e-12, w0=1.8,
        P0_lo=60.0, P0_hi=100.0, sigma_dec=1.0e-13,
        # Resident-operand overhead band: narrower than [P0_lo, P0_hi] by the
        # cold-card cells the priced window cannot be in (the pinned rung).
        P0_lo_resident=70.0, P0_hi_resident=100.0, n_idle_resident=15,
        r_useful_lo=1.8e-12, r_useful_hi=2.4e-12,
        r_useful_marg_lo=2.0e-12, r_useful_marg_hi=2.9e-12,
        format_bands={
            "fp16": {"r_lo": 1.1e-12, "r_hi": 2.2e-12, "n_cells": 12},
            "fp32": {"r_lo": 3.0e-12, "r_hi": 20.0e-12, "n_cells": 12},
            "int8": {"r_lo": 0.8e-12, "r_hi": 1.5e-12, "n_cells": 12},
        },
        r_marg_cov={
            "int8": {"r_marg": 0.60e-12, "r_marg_lcb95": 0.55e-12,
                     "se_slope": 2.0e-14, "r2": 0.99, "n_cells": 6,
                     "intercept_j": 900.0, "dose0_energy_j": 905.0,
                     "stratum_mhz": 1395, "cov_dtype": "int8",
                     "alpha": 0.05, "strata": {}},
            "fp16": {"r_marg": 1.00e-12, "r_marg_lcb95": 0.90e-12,
                     "se_slope": 4.0e-14, "r2": 0.99, "n_cells": 6,
                     "intercept_j": 900.0, "dose0_energy_j": 905.0,
                     "stratum_mhz": 1395, "cov_dtype": "fp16",
                     "alpha": 0.05, "strata": {}},
        },
    )
    base.update(over)
    return MeasuredBands(**base)


def _expected(mb: MeasuredBands, r_hi_dec, r_lo_dec, r_lo_cov, kappa_dec=1.0,
              dE0=None, covert_format="int8"):
    """Independent composition: what a priced rung must evaluate to."""
    kappa = A100_DENSE_PEAK_OPS[covert_format] / A100_DENSE_PEAK_OPS["fp16"]
    args = dict(r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
                dE0=mb.P0_hi - mb.P0_lo if dE0 is None else dE0,
                C_max=A100_DENSE_PEAK_OPS["fp16"])
    h, b = beta_phys_worst_case(**args, kappa=kappa, kappa_dec=kappa_dec)
    return h, b, float(energy_slack(1.0, **args))


# ---- structure -------------------------------------------------------------


def test_ladder_has_six_rungs_in_order():
    rungs = build_ladder(_mb())
    assert [r.name for r in rungs] == [
        "bare meter",
        "precision declared & checked",
        "clock & locality observed",
        "declared work re-executed",
        "covert work on useful data",
        "overhead band pinned to the window",
    ]
    assert [r.rung_type for r in rungs] == [
        None, "D,C", "T", "C", "M", "M"]


def test_clock_rung_is_unpriced_and_prospective():
    # B6: w0 must not price a security-guarantee rung; the clock/locality row
    # stays as an unpriced prospective capability.
    clock = build_ladder(_mb())[2]
    assert clock.priced is False
    assert clock.beta_star is None and clock.h_star is None and clock.S1 is None
    assert "prospective" in clock.conditions


def test_conditional_chain_is_disclosed():
    rungs = build_ladder(_mb())
    assert rungs[0].conditions == ()                      # bare rung unconditional
    for r in rungs[1:2] + rungs[3:]:
        assert COND_FORMAT in r.conditions                # inherited downward
    for r in rungs[3:]:
        assert COND_OPERATING_POINT in r.conditions       # re-exec needs the clock


# ---- pricing ---------------------------------------------------------------


def test_bare_rung_is_full_envelope_power_only():
    # Referee B4 (third round): a bare meter cannot separate declared from
    # covert work nor one format from another, so the DECLARED BAND is the full
    # hardware envelope (subsec:poweronly) — ceiling at the dearest measured
    # rate (the pooled r_hi, fp32), floor at the cheapest (the pooled r_lo, int8
    # zeros). No precision is assumed on either side (the retired fp16-ceiling /
    # int8-floor hybrid was the inconsistency flagged), so the ceiling is
    # deliberately NOT the fp16 band.
    mb = _mb()
    bare = build_ladder(mb)[0]
    assert bare.r_hi_dec == mb.r_hi
    assert bare.r_hi_dec != mb.format_bands["fp16"]["r_hi"]   # not the old hybrid
    assert bare.r_lo_dec == mb.r_lo
    assert bare.r_lo_cov == covert_edge(mb, "int8")
    # Cross-format clip (numerical-audit finding 2): the bare rung pins no
    # declared format, so its declared ops execute in the covert format and
    # occupy h/kappa of resource-time — the clip is kappa - h (kappa_dec =
    # kappa), not the single-format (1 - h) kappa.
    kappa = A100_DENSE_PEAK_OPS["int8"] / A100_DENSE_PEAK_OPS["fp16"]
    h, b, s1 = _expected(mb, bare.r_hi_dec, bare.r_lo_dec, bare.r_lo_cov,
                         kappa_dec=kappa)
    assert bare.beta_star == pytest.approx(b)
    assert bare.h_star == pytest.approx(h)
    assert bare.S1 == pytest.approx(s1)
    h1, b1, _ = _expected(mb, bare.r_hi_dec, bare.r_lo_dec, bare.r_lo_cov)
    assert bare.beta_star > b1                                # relaxed clip widens
    assert bare.h_star > h1


def test_no_rung_collapses_the_declared_floor_onto_the_covert_floor():
    # The de-collapse invariant (subsec:poweronly, app:estimands). r_lo^dec and
    # r_lo^cov can describe the same physical corner of the hardware, but they
    # sit in different algebraic roles and admit different estimands: the
    # declared band enters through a DIFFERENCE (total quotients suffice), the
    # covert floor is a DIVISOR (one-sided marginal LCB only). Every priced
    # rung must therefore keep them on separate numbers, and no rung may divide
    # by a total quotient — a quotient amortises baseline power that a marginal
    # covert op does not pay, so it overstates the divisor and UNDERSTATES beta,
    # the one direction a security floor cannot be wrong in.
    #
    # NOT asserted: an ordering between the two. r_lo^cov is the cheaper number
    # wherever both describe the unconstrained hardware floor, but the
    # useful-data rungs deliberately lift r_lo^cov ABOVE the declared floor —
    # that threat-model restriction (hidden zero-ops are no threat) is exactly
    # what those rungs buy.
    mb = _mb()
    quotients = {mb.r_lo, mb.r_hi, mb.r_lo_op, mb.r_useful_lo, mb.r_useful_hi}
    for rung in (r for r in build_ladder(mb) if r.priced):
        assert rung.r_lo_cov != rung.r_lo_dec, rung.name
        assert rung.r_lo_cov not in quotients, rung.name


def test_precision_rung_equals_headline():
    # The conditional precision rung IS the B1 headline case.
    mb = _mb()
    precision = build_ladder(mb)[1]
    got = headline(mb, "fp16", "int8")
    assert precision.beta_star == got["beta_star"]
    assert precision.h_star == got["h_star"]
    assert precision.S1 == pytest.approx(got["S1"])


def test_reexec_rung_declared_width_is_the_gap():
    mb = _mb()
    gap = 0.05e-12
    reexec = build_ladder(mb, gap=gap)[3]
    assert reexec.r_hi_dec - reexec.r_lo_dec == pytest.approx(gap)
    assert reexec.r_lo_cov == covert_edge(mb, "int8")


def test_useful_band_width_never_enters_declared_numerator():
    # The first-round defect: plot_capability_ladder.py fed r_useful_hi + gap
    # into the declared ceiling, so gaining a capability WIDENED the set.
    # Correct cumulative arithmetic keeps the declared width at the gap and
    # moves only the covert edge.
    mb = _mb()
    gap = 0.05e-12
    useful = build_ladder(mb, gap=gap)[4]
    assert useful.r_hi_dec - useful.r_lo_dec == pytest.approx(gap)
    assert useful.r_lo_cov == mb.r_useful_marg_lo
    h, b, s1 = _expected(mb, useful.r_hi_dec, useful.r_lo_dec, useful.r_lo_cov)
    assert useful.beta_star == pytest.approx(b)
    assert useful.S1 == pytest.approx(s1)


def test_pinned_rung_moves_only_the_overhead_width():
    # The bottom rung is the fourth dial of eq:betadials and nothing else: it
    # narrows dE0 to the resident-operand band and inherits the useful-data
    # rung's denominator and declared width unchanged.
    mb = _mb()
    gap = 0.05e-12
    useful, pinned = build_ladder(mb, gap=gap)[4:6]
    assert pinned.r_lo_cov == useful.r_lo_cov
    assert pinned.r_hi_dec == useful.r_hi_dec
    assert pinned.r_lo_dec == useful.r_lo_dec
    dE0 = mb.P0_hi_resident - mb.P0_lo_resident
    assert dE0 < mb.P0_hi - mb.P0_lo                       # a restriction
    h, b, s1 = _expected(mb, pinned.r_hi_dec, pinned.r_lo_dec, pinned.r_lo_cov,
                         dE0=dE0)
    assert pinned.beta_star == pytest.approx(b)
    assert pinned.h_star == pytest.approx(h)
    assert pinned.S1 == pytest.approx(s1)


def test_pinned_rung_never_falls_back_to_the_published_band():
    # A pre-split artifact must fail loudly. Reusing [P0_lo, P0_hi] would price
    # the rung at the published width and report a restriction never applied.
    with pytest.raises(ValueError, match="P0_.*resident"):
        build_ladder(_mb(P0_lo_resident=None, P0_hi_resident=None))


# ---- the covert-format branch ----------------------------------------------


def test_branch_reprices_two_rungs_at_the_fp16_covert_edge():
    mb = _mb()
    gap = 0.05e-12
    at_precision, at_reexec = build_covert_format_branch(mb, gap=gap)
    edge = covert_edge(mb, "fp16")
    assert edge > covert_edge(mb, "int8")            # dearer denominator
    for r in (at_precision, at_reexec):
        assert r.r_lo_cov == edge
        assert COND_COVERT_FORMAT in r.conditions
    # Its two anchors: the precision rung's full declared band, and the
    # re-execution rung's residual gap.
    dec_lo, dec_hi = mb.format_bands["fp16"]["r_lo"], mb.format_bands["fp16"]["r_hi"]
    assert (at_precision.r_lo_dec, at_precision.r_hi_dec) == (dec_lo, dec_hi)
    assert at_reexec.r_hi_dec - at_reexec.r_lo_dec == pytest.approx(gap)
    for r in (at_precision, at_reexec):
        h, b, s1 = _expected(mb, r.r_hi_dec, r.r_lo_dec, r.r_lo_cov,
                             covert_format="fp16")
        assert r.beta_star == pytest.approx(b)
        assert r.S1 == pytest.approx(s1)


def test_branch_halves_the_capacity_ceiling():
    # The branch's distinctive half: kappa = F_max^fp16 / F_max^fp16 = 1, so
    # beta* < 1 holds by capacity alone whatever the energy slack. Feeding an
    # absurd declared band drives the chain rung to its ceiling of kappa = 2
    # and the branch to 1.
    mb = _mb(format_bands={"fp16": {"r_lo": 1.1e-12, "r_hi": 1.0e-6,
                                    "n_cells": 12}})
    chain = build_ladder(mb)[1]
    branch = build_covert_format_branch(mb)[0]
    assert chain.beta_star == pytest.approx(2.0, rel=1e-3)
    assert branch.beta_star == pytest.approx(1.0, rel=1e-3)


def test_branch_is_not_appended_to_the_chain():
    # It reaches the SAME dial as the useful-data rung, so it must not be
    # composed with the rungs below it: check_monotone governs the chain only,
    # and the branch's names never appear in it.
    mb = _mb()
    names = {r.name for r in build_ladder(mb)}
    for r in build_covert_format_branch(mb):
        assert r.name not in names
        assert r.name.startswith("branch:")


def test_branch_sits_below_the_rungs_it_reprices():
    # A dearer covert denominator and a halved ceiling can only narrow the set.
    mb = _mb()
    chain = {r.name: r for r in build_ladder(mb)}
    at_precision, at_reexec = build_covert_format_branch(mb)
    assert at_precision.beta_star < chain["precision declared & checked"].beta_star
    assert at_reexec.beta_star < chain["declared work re-executed"].beta_star


def test_branch_fails_loudly_without_an_fp16_covert_edge():
    mb = _mb(r_marg_cov={"int8": {"r_marg": 0.60e-12,
                                  "r_marg_lcb95": 0.55e-12}})
    with pytest.raises(ValueError, match="fp16"):
        build_covert_format_branch(mb)


def test_exp2_quotients_do_not_price_any_rung():
    # B6 ban: Exp-2's w0 and operand edge r_lo_op must not enter ANY priced rung
    # — guarantee rungs price off the Exp-7 marginal edge and the per-format
    # bands. Poisoning them leaves every priced rung finite and unchanged.
    # (The pooled envelope edges r_lo/r_hi are NOT in this ban: they are the
    # bare rung's declared band by construction — see the two tests below.)
    clean = build_ladder(_mb())
    poisoned = build_ladder(_mb(w0=math.nan, r_lo_op=math.nan))
    for c, p in zip(clean, poisoned):
        if c.priced:
            assert math.isfinite(p.beta_star)
            assert p.beta_star == c.beta_star


@pytest.mark.parametrize("edge", ["r_hi", "r_lo"])
def test_pooled_envelope_prices_only_the_bare_rung(edge):
    # B4 (third round): the pooled envelope [r_lo, r_hi] IS the power-only
    # DECLARED band, so both its edges price the bare rung by construction
    # (subsec:poweronly) — but no capability rung below it, which prices inside
    # one format. Poisoning either edge must break only the bare rung.
    #
    # r_lo joined r_hi here when the bare rung was de-collapsed: it previously
    # read the covert marginal edge as its declared floor, mixing estimands.
    clean = build_ladder(_mb())
    poisoned = build_ladder(_mb(**{edge: math.nan}))
    assert not math.isfinite(poisoned[0].beta_star)          # bare reads both edges
    for c, p in zip(clean[1:], poisoned[1:]):                 # capability rungs intact
        if c.priced:
            assert math.isfinite(p.beta_star)
            assert p.beta_star == c.beta_star


# ---- monotonicity (referee-required) ---------------------------------------


def test_priced_chain_is_monotone_non_increasing():
    rungs = build_ladder(_mb())
    check_monotone(rungs)                                  # must not raise
    priced = [r.beta_star for r in rungs if r.priced]
    assert priced == sorted(priced, reverse=True)


def test_useful_rung_below_reexec_rung():
    # Pins the 0.16 -> 0.43 inversion as fixed; the pinned rung sits below the
    # useful-data rung it narrows.
    rungs = build_ladder(_mb())
    assert rungs[4].beta_star < rungs[3].beta_star
    assert rungs[5].beta_star < rungs[4].beta_star


def test_monotone_under_wider_gap():
    rungs = build_ladder(_mb(), gap=0.2e-12)
    check_monotone(rungs)


def test_check_monotone_raises_on_violation():
    rungs = build_ladder(_mb())
    doctored = list(rungs)
    doctored[3], doctored[4] = doctored[4], doctored[3]    # force an inversion
    with pytest.raises(LadderMonotonicityError):
        check_monotone(doctored)


@pytest.mark.parametrize("kappa", [1.0, 2.0])
def test_worst_case_monotone_in_band_and_edge(kappa):
    # Property behind the ladder: narrowing the declared band or raising the
    # covert edge never increases max_h beta_phys(h).
    base = dict(r_lo_dec=1.0e-12, dE0=40.0, C_max=312e12, kappa=kappa)
    for r_hi in (1.2e-12, 2.0e-12, 5.0e-12):
        for r_cov in (0.5e-12, 1.0e-12, 2.0e-12):
            _, b0 = beta_phys_worst_case(r_hi_dec=r_hi, r_lo_cov=r_cov, **base)
            _, b_narrow = beta_phys_worst_case(
                r_hi_dec=r_hi - 0.1e-12, r_lo_cov=r_cov, **base)
            _, b_dearer = beta_phys_worst_case(
                r_hi_dec=r_hi, r_lo_cov=r_cov * 1.5, **base)
            assert b_narrow <= b0
            assert b_dearer <= b0


# ---- fail-loudly on incomplete artifacts ------------------------------------


def test_pre_exp7_artifact_fails_loudly():
    with pytest.raises(ValueError, match="r_marg_cov"):
        build_ladder(_mb(r_marg_cov=None))


def test_missing_marginal_useful_band_fails_loudly():
    # Rungs 4-5 price off the qblock-marginal MARGINAL useful edge (B5); an artifact that
    # predates that capture must fail loudly, not fall back to the retired
    # total-quotient useful band.
    with pytest.raises(ValueError, match="r_useful_marg"):
        build_ladder(_mb(r_useful_marg_lo=None, r_useful_marg_hi=None))


# ---- regression pins on the committed artifact ------------------------------


def test_committed_artifact_rung_values():
    # The numbers quoted in tab:ladder / fig:ladder of paper/main.tex.
    rungs = build_ladder(load_measured_bands(ARTIFACT))
    check_monotone(rungs)
    betas = {r.name: r.beta_star for r in rungs if r.priced}
    # Bare rung is the full-envelope power-only floor (B4, third round): the
    # pooled declared band [r_lo, r_hi] = [0.786, 19.5] pJ/op over the int8
    # marginal covert edge, up from the retired fp16-ceiling hybrid's 1.34,
    # under the format-consistent cross-format clip kappa - h (numerical-audit
    # finding 2; the single-format clip gave 1.910). Its maximiser h*~0.047 sits
    # inside the feasible window (below h_max~0.24). De-collapsing the declared
    # floor off the covert edge moved this 1.954 -> 1.953: r_hi dominates the
    # numerator's difference, so estimand purity is nearly free here.
    assert betas["bare meter"] == pytest.approx(1.953, abs=5e-3)
    assert betas["precision declared & checked"] == pytest.approx(1.156, abs=5e-3)
    assert betas["declared work re-executed"] == pytest.approx(0.337, abs=5e-3)
    # Rung 4 prices off the qblock-marginal MARGINAL useful edge (B5, third
    # round): r_useful^marg [2.53, 3.16] pJ/op replaced the total-quotient
    # [3.18, 4.22], so covert work is cheaper and the rung ROSE (0.057->0.071).
    assert betas["covert work on useful data"] == pytest.approx(0.071, abs=5e-3)
    # Rung 5 narrows dE0 from the published 41.2 W to the resident-operand
    # 31.3 W (15 of Exp-3's 25 idle windows), 0.76 of the published width.
    assert betas["overhead band pinned to the window"] == pytest.approx(
        0.059, abs=5e-3)


def test_committed_artifact_branch_values():
    # The branch rows of tab:ladder.
    mb = load_measured_bands(ARTIFACT)
    at_precision, at_reexec = build_covert_format_branch(mb)
    assert at_precision.beta_star == pytest.approx(0.662, abs=5e-3)
    assert at_precision.S1 == pytest.approx(1.620, abs=5e-3)
    assert at_reexec.beta_star == pytest.approx(0.225, abs=5e-3)
    assert at_reexec.S1 == pytest.approx(0.240, abs=5e-3)


def test_committed_artifact_pinned_band_is_the_quoted_one():
    # The manuscript quotes P0 in [69.7, 101.0] W, dP0 = 31.3 W.
    from powertoflops.measured import resident_overhead_band
    lo, hi = resident_overhead_band(load_measured_bands(ARTIFACT))
    assert (lo, hi) == (pytest.approx(69.7, abs=0.05),
                        pytest.approx(101.0, abs=0.05))
    assert hi - lo == pytest.approx(31.3, abs=0.05)


def test_committed_artifact_precision_rung_is_headline():
    mb = load_measured_bands(ARTIFACT)
    assert build_ladder(mb)[1].beta_star == headline(mb)["beta_star"]


def test_power_and_paperwork_stall_pins_the_quoted_value():
    # The manuscript quotes "power plus paperwork stalls at ~1" in four places
    # including a contributions bullet. It lived only in the plot script and was
    # unpinned; pin it here so the claim cannot drift silently.
    mb = load_measured_bands(ARTIFACT)
    assert power_and_paperwork_stall(mb) == pytest.approx(1.03, abs=5e-3)


def test_power_and_paperwork_stall_is_worse_than_re_execution():
    # The point of the stall: paperwork WITHOUT an observed operating point does
    # not buy the re-execution rung's collapse. It must sit above that rung and
    # near one whole machine.
    mb = load_measured_bands(ARTIFACT)
    reexec = next(r for r in build_ladder(mb)
                  if r.name == "declared work re-executed")
    stall = power_and_paperwork_stall(mb)
    assert stall > reexec.beta_star
    assert 0.9 < stall < 1.2


def test_power_and_paperwork_stall_grows_with_the_dvfs_assumption():
    # Sanity on the direction of the only assumption it carries: a wider
    # unobserved DVFS range concedes more, never less.
    mb = load_measured_bands(ARTIFACT)
    import powertoflops.ladder as L
    base = power_and_paperwork_stall(mb)
    original = L.DVFS_FACTOR
    try:
        L.DVFS_FACTOR = original + 0.5
        assert power_and_paperwork_stall(mb) > base
        L.DVFS_FACTOR = 1.0          # clock fully observed -> gap closes
        assert power_and_paperwork_stall(mb) < base
    finally:
        L.DVFS_FACTOR = original
    assert DVFS_FACTOR == 1.5        # the pinned manuscript value
