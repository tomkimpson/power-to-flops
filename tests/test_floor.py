"""Closed-form floor: internal consistency and the reported numerical anchors."""

from pathlib import Path

import numpy as np
import pytest

from powertoflops.config import A100_DENSE_PEAK_OPS, DEFAULT
from powertoflops.floor import (
    beta_bare,
    beta_budget,
    beta_phys,
    beta_phys_worst_case,
    beta_two_band,
    covert_headroom_rate,
    energy_slack,
)
from powertoflops.measured import load_measured_bands

_REPO = Path(__file__).resolve().parents[1]

# The 2026-07-16 pre-R2 bands the reviewer used for the clipped worst case.
# beta is normalised by the declared-format (fp16 dense) capacity, so T cancels
# and (dE0, C_max) can be passed as (dP0, F_max).
_OLD_R_LO = 9.084265191935023e-13
_OLD_R_HI = 2.059471073953942e-11
_OLD_DP0 = 91.51573280060181 - 84.96540837711991
_F_MAX_FP16 = A100_DENSE_PEAK_OPS["fp16"]


def _bare_args(f=DEFAULT.floor):
    return dict(
        hfu_dec=f.hfu_dec,
        r_hi=f.r_hi,
        r_lo=f.r_lo,
        dE0=f.dE0,
        C_max=f.C_max,
    )


def test_two_band_reduces_to_bare():
    """beta_two_band collapses to beta_bare when the bands coincide."""
    f = DEFAULT.floor
    bare = beta_bare(**_bare_args(f))
    two = beta_two_band(
        hfu_dec=f.hfu_dec,
        r_hi_dec=f.r_hi,
        r_lo_dec=f.r_lo,
        r_lo_cov=f.r_lo,  # covert edge == declared cheapest
        dE0=f.dE0,
        C_max=f.C_max,
    )
    assert two == pytest.approx(bare, rel=1e-12)


def test_budget_matches_bare():
    """The budget form reproduces beta_bare for the matching allowance split."""
    f = DEFAULT.floor
    u_dec = f.hfu_dec * (f.r_hi - f.r_lo) * f.C_max  # declared-rate slack energy
    u_0 = f.dE0                                       # overhead band energy
    budget = beta_budget(
        u_dec=u_dec, u_0=u_0, u_m=0.0, u_eta=0.0,
        r_lo_cov=f.r_lo, C_max=f.C_max,
    )
    assert budget == pytest.approx(beta_bare(**_bare_args(f)), rel=1e-12)


def test_vacuity():
    """The headline claim: the bare power meter is vacuous (beta >> 1)."""
    beta = beta_bare(**_bare_args())
    assert beta > 1.0


def test_monotonic_in_ratio():
    """beta grows with the exchange-rate band ratio r_hi / r_lo."""
    f = DEFAULT.floor
    base = _bare_args(f)
    wider = {**base, "r_hi": f.r_hi * 2.0}
    assert beta_bare(**wider) > beta_bare(**base)


def test_covert_headroom_integral_reproduces_beta():
    """The time-domain covert headroom integrates to the closed-form floor beta.

    The time-resolved envelope F_cov^max(t) is the twin of beta_two_band:
    integrating it over the window and dividing by C_max must return exactly the
    same dimensionless floor (paper Sec. 4.2 bridge to Sec. 3). The identity is
    independent of the declared schedule's *shape*, so a bursty declared trace is
    used here only to exercise the non-constant case.
    """
    f = DEFAULT.floor
    T, dt = 120.0, 0.01
    t = np.arange(0.0, T + 0.5 * dt, dt)
    # Bursty declared work at 40 % mean utilisation, f0 = 1 Hz, clipped to [0, F_max].
    F_dec = np.clip(0.4 * f.F_max * (1.0 + 0.8 * np.sin(2.0 * np.pi * 1.0 * t)),
                    0.0, f.F_max)
    F_cov_max = covert_headroom_rate(
        F_dec, f.r_hi, f.r_lo, f.r_lo_cov, f.P0_hi - f.P0_lo,
    )

    C_max = f.F_max * float(t[-1])
    C_dec = float(np.trapezoid(F_dec, t))

    beta_integral = float(np.trapezoid(F_cov_max, t)) / C_max
    beta_closed = beta_two_band(
        hfu_dec=C_dec / C_max,
        r_hi_dec=f.r_hi, r_lo_dec=f.r_lo, r_lo_cov=f.r_lo_cov,
        dE0=(f.P0_hi - f.P0_lo) * float(t[-1]), C_max=C_max,
    )
    assert beta_integral == pytest.approx(beta_closed, rel=1e-12)
    # The bare-meter floor is of order a machine of capacity: loose, not vacuous.
    assert beta_integral > 1.0


def _old_band_args(kappa=1.0):
    """Single-band bare-meter setup at the reviewer's normalisation."""
    return dict(
        r_hi_dec=_OLD_R_HI,
        r_lo_dec=_OLD_R_LO,
        r_lo_cov=_OLD_R_LO,
        dE0=_OLD_DP0,
        C_max=_F_MAX_FP16,
        kappa=kappa,
    )


def test_energy_slack_is_two_band():
    """energy_slack(h, ...) is exactly beta_two_band(hfu_dec=h, ...)."""
    f = DEFAULT.floor
    for h in (0.0, 0.3, 1.0):
        assert energy_slack(
            h, r_hi_dec=f.r_hi, r_lo_dec=f.r_lo, r_lo_cov=f.r_lo_cov,
            dE0=f.dE0, C_max=f.C_max,
        ) == beta_two_band(
            hfu_dec=h, r_hi_dec=f.r_hi, r_lo_dec=f.r_lo, r_lo_cov=f.r_lo_cov,
            dE0=f.dE0, C_max=f.C_max,
        )


def test_beta_phys_reviewer_anchor():
    """The clipped floor reproduces the review's corrected worst case.

    Old (pre-R2) bands, kappa = 1, F_max = 312e12 nominal op/s: the reviewer
    reports energy-slack S(1) = 21.69 clipping to beta* = 0.957 at h = 0.044.
    """
    kwargs = _old_band_args()
    slack_args = {k: v for k, v in kwargs.items() if k != "kappa"}
    h_star, beta_star = beta_phys_worst_case(**kwargs)
    assert energy_slack(1.0, **slack_args) == pytest.approx(21.69, abs=5e-3)
    assert h_star == pytest.approx(0.0431, abs=5e-4)
    assert beta_star == pytest.approx(0.9569, abs=5e-4)


def test_beta_phys_worst_case_matches_grid():
    """Closed-form (h*, beta*) agrees with a dense grid max of beta_phys."""
    for kappa in (1.0, 2.0):
        kwargs = _old_band_args(kappa=kappa)
        h = np.linspace(0.0, 1.0, 20001)
        curve = beta_phys(h, **kwargs)
        h_star, beta_star = beta_phys_worst_case(**kwargs)
        i = int(np.argmax(curve))
        # The closed form is exact; the grid undershoots the kink by at most
        # (grid step / 2) * slope, so bound from above and match to that error.
        assert beta_star >= float(curve[i]) - 1e-12
        assert beta_star == pytest.approx(float(curve[i]), abs=1e-3)
        assert h_star == pytest.approx(float(h[i]), abs=1e-4)


def test_beta_phys_zero_at_full_utilisation():
    """At h = 1 the capacity clip leaves no covert room, for any kappa."""
    for kappa in (0.5, 1.0, 2.0):
        assert beta_phys(1.0, **_old_band_args(kappa=kappa)) == 0.0


def test_beta_phys_clip_vs_slack_regimes():
    """Small h: the energy slack binds; large h: the capacity clip binds."""
    kwargs = _old_band_args()
    slack_args = {k: v for k, v in kwargs.items() if k != "kappa"}
    h_lo, h_hi = 0.01, 0.9
    assert beta_phys(h_lo, **kwargs) == pytest.approx(
        energy_slack(h_lo, **slack_args), rel=1e-12
    )
    assert beta_phys(h_hi, **kwargs) == pytest.approx((1.0 - h_hi) * 1.0, rel=1e-12)


def test_beta_phys_kappa():
    """Worst case grows with kappa, is bounded by kappa, and the clip scales."""
    h_star1, beta_star1 = beta_phys_worst_case(**_old_band_args(kappa=1.0))
    h_star2, beta_star2 = beta_phys_worst_case(**_old_band_args(kappa=2.0))
    assert beta_star2 > beta_star1
    assert beta_star1 <= 1.0 and beta_star2 <= 2.0
    kwargs1, kwargs2 = _old_band_args(kappa=1.0), _old_band_args(kappa=2.0)
    assert beta_phys(0.9, **kwargs2) == pytest.approx(
        2.0 * beta_phys(0.9, **kwargs1), rel=1e-12
    )
    # Degenerate branch: overhead slack alone exceeds the clip at h = 0.
    tiny = dict(_old_band_args(kappa=1e-4))
    h_star, beta_star = beta_phys_worst_case(**tiny)
    assert (h_star, beta_star) == (0.0, pytest.approx(1e-4))

    with pytest.raises(ValueError):
        beta_phys(0.5, **_old_band_args(kappa=0.0))
    with pytest.raises(ValueError):
        beta_phys_worst_case(**_old_band_args(kappa=-1.0))


def test_beta_phys_cross_format_clip():
    """kappa_dec = kappa relaxes the clip to kappa - h (the bare rung's case).

    When the cheating schedule executes the declared ops in the covert format
    (numerical-audit finding 2), they occupy h/kappa of resource-time, so the
    clip is kappa - h rather than (1 - h) kappa; the crossing therefore sits
    at h* = (kappa - b)/(s + 1) with beta* = kappa - h*, and the closed form
    must agree with a dense grid max.
    """
    kappa = 2.0
    kwargs = _old_band_args(kappa=kappa)
    kwargs["kappa_dec"] = kappa
    # Clip regime: (1 - h/kappa) kappa = kappa - h.
    assert beta_phys(0.9, **kwargs) == pytest.approx(kappa - 0.9, rel=1e-12)
    # Closed form matches the grid max, and relaxing the clip cannot shrink
    # the worst case relative to the single-format clip.
    h = np.linspace(0.0, 1.0, 20001)
    curve = beta_phys(h, **kwargs)
    h_star, beta_star = beta_phys_worst_case(**kwargs)
    i = int(np.argmax(curve))
    assert beta_star >= float(curve[i]) - 1e-12
    assert beta_star == pytest.approx(float(curve[i]), abs=1e-3)
    assert h_star == pytest.approx(float(h[i]), abs=1e-4)
    _, beta_single = beta_phys_worst_case(**_old_band_args(kappa=kappa))
    assert beta_star >= beta_single
    with pytest.raises(ValueError):
        beta_phys_worst_case(**_old_band_args(kappa=kappa), kappa_dec=0.0)


def test_beta_phys_worst_case_clamps_crossing_to_unit_interval():
    """A crossing past h = 1 (possible only for kappa_dec > 1) yields S(1).

    With kappa_dec > 1 the clip no longer vanishes at h = 1, so for a narrow
    band (small s) the slack line can stay below the clip on all of [0, 1];
    the unclamped crossing formula would then return an infeasible h* > 1 and
    overstate beta*. The max over [0, 1] is S(1) = s + b, at h = 1.
    """
    r_lo = 1.0e-12                        # 1 pJ/op
    kwargs = dict(
        r_hi_dec=1.2 * r_lo,              # s = 0.2
        r_lo_dec=r_lo,
        r_lo_cov=r_lo,
        dE0=0.1 * r_lo * _F_MAX_FP16,     # b = 0.1
        C_max=_F_MAX_FP16,
        kappa=2.0,
        kappa_dec=2.0,                    # unclamped crossing: h* = 1.9/1.2 > 1
    )
    h_star, beta_star = beta_phys_worst_case(**kwargs)
    assert h_star == 1.0
    assert beta_star == pytest.approx(0.3, rel=1e-12)     # S(1) = s + b
    h = np.linspace(0.0, 1.0, 20001)
    assert beta_star == pytest.approx(float(np.max(beta_phys(h, **kwargs))),
                                      abs=1e-3)


def test_beta_phys_new_headline_regression():
    """Guards the paper's printed numbers from the joint-model recompute.

    Jointly format-consistent fp16-declared case study (referee B1) on the
    2026-07-22 same-device capture, covert edges = Exp-7 marginal LCBs
    (referee B2), via powertoflops.joint_model — the only path to the headline:

        int8 covert (kappa = 2): beta* = 1.156 at h* = 0.422, S(1) = 2.39
        fp16 covert (kappa = 1): beta* = 0.662 at h* = 0.338, S(1) = 1.62

    The retired 1.861 headline paired the pooled fp32 upper edge and int8
    floor with fp16 capacity — a cross-format mix the joint model's
    validator now rejects (tests/test_joint_model.py pins the rejection).
    """
    from powertoflops.joint_model import headline

    mb = load_measured_bands(_REPO / "results" / "bench" / "measured_bands.json")

    res8 = headline(mb, declared="fp16", covert="int8")
    res16 = headline(mb, declared="fp16", covert="fp16")
    assert res8["case"].kappa == pytest.approx(
        A100_DENSE_PEAK_OPS["int8"] / A100_DENSE_PEAK_OPS["fp16"])
    assert res8["beta_star"] == pytest.approx(1.156, rel=1e-3)
    assert res8["h_star"] == pytest.approx(0.422, rel=1e-2)
    assert res8["S1"] == pytest.approx(2.39, rel=1e-2)
    assert res16["beta_star"] == pytest.approx(0.662, rel=1e-3)
    assert res16["h_star"] == pytest.approx(0.338, rel=1e-2)
    assert res16["S1"] == pytest.approx(1.62, rel=1e-2)
    # Energy binds below the crossing — the corrected per-regime statement
    # (the retired "energy never binds" claim was false at S(0) < kappa).
    assert res8["b"] < res8["case"].kappa


def test_beta_phys_feasible_window_bound():
    """Guards the B1 attainability numbers printed in the headline/conclusion.

    The unconstrained peak beta* = 1.156 sits at h* = 0.422, above the
    busy-time ceiling h_max ~ 0.24 at which a single-window witness can be
    constructed (ass:corners). It is therefore a model UPPER bound, not an
    attained width. Confining the maximisation to a feasible window
    (h <= h_max) drops the model worst case to ~0.77 — the number the
    manuscript now quotes beside the measured witness value 0.41 machines.
    """
    from powertoflops.joint_model import headline

    mb = load_measured_bands(_REPO / "results" / "bench" / "measured_bands.json")
    res8 = headline(mb, declared="fp16", covert="int8")
    case = res8["case"]
    args = dict(r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec,
                r_lo_cov=case.r_lo_cov, dE0=case.dP0, C_max=case.C_max,
                kappa=case.kappa)

    h_max = 0.24
    # The unconstrained maximiser is genuinely above the feasible ceiling.
    assert res8["h_star"] > h_max
    # Feasible-window worst case = max_{h <= h_max} beta_phys(h). Energy binds
    # in this regime, so the max is at h_max itself.
    grid = np.linspace(0.0, h_max, 2401)
    beta_feasible = max(float(beta_phys(h, **args)) for h in grid)
    assert beta_feasible == pytest.approx(0.77, abs=0.02)
    assert float(beta_phys(h_max, **args)) == pytest.approx(beta_feasible, rel=1e-6)
