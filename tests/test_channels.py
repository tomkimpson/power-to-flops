"""Channel-residual terms (A1 study #1): limits, scaling, and the bare-meter anchor.

Each verifier channel turns one *irreducible* budget term into a curve. These tests
pin the limiting behaviour (the curve goes to zero as the channel is fully deployed),
the analytic scaling (1/sqrt(n) for re-execution, linear in m for the transfer gap),
the monotone direction of the composed floor, and -- the load-bearing guard -- that
the no-channel limit reproduces the Stage-1 closed form beta(empty) exactly.
"""

import pytest

from powertoflops.config import DEFAULT
from powertoflops.floor import (
    beta_bare,
    beta_composed,
    u_0_metered,
    u_dec_reexec,
    u_m_transfer,
)


def _C_dec(f=DEFAULT.floor) -> float:
    return f.hfu_dec * f.C_max


# ---- u_dec(n): re-execution allowance -------------------------------------------

def test_u_dec_sqrt_n_scaling():
    """Quadrupling n halves the declared allowance (exact 1/sqrt(n))."""
    c, C_dec = DEFAULT.channel, _C_dec()
    u_n = u_dec_reexec(c.sigma_dec, C_dec, n=25, z_gamma=c.z_gamma)
    u_4n = u_dec_reexec(c.sigma_dec, C_dec, n=100, z_gamma=c.z_gamma)
    assert u_n / u_4n == pytest.approx(2.0, rel=1e-12)


def test_u_dec_vanishes_with_samples():
    """u_dec -> 0 as the re-execution sample count grows large."""
    c, C_dec = DEFAULT.channel, _C_dec()
    u1 = u_dec_reexec(c.sigma_dec, C_dec, n=1, z_gamma=c.z_gamma)
    u_big = u_dec_reexec(c.sigma_dec, C_dec, n=10**6, z_gamma=c.z_gamma)
    assert u_big < u1
    assert u_big == pytest.approx(u1 / 1000.0, rel=1e-12)


def test_u_dec_rejects_nonpositive_n():
    c, C_dec = DEFAULT.channel, _C_dec()
    with pytest.raises(ValueError):
        u_dec_reexec(c.sigma_dec, C_dec, n=0, z_gamma=c.z_gamma)


# ---- u_m(m): transfer-gap allowance ---------------------------------------------

def test_u_m_zero_at_matched_hardware():
    """A perfectly matched reference device (m=0) concedes no transfer band."""
    c = DEFAULT.channel
    assert u_m_transfer(0.0, c.r_bar_dec, _C_dec()) == 0.0


def test_u_m_linear_in_gap():
    """The transfer allowance is linear in the device-to-device gap m."""
    c, C_dec = DEFAULT.channel, _C_dec()
    assert u_m_transfer(0.10, c.r_bar_dec, C_dec) == pytest.approx(
        2.0 * u_m_transfer(0.05, c.r_bar_dec, C_dec), rel=1e-12
    )


# ---- u_0(q0): metered overhead band ---------------------------------------------

def test_u_0_limits():
    """q0=0 is the full overhead band; q0=1 is fully metered (zero residual)."""
    f = DEFAULT.floor
    assert u_0_metered(0.0, f.dE0) == pytest.approx(f.dE0, rel=1e-12)
    assert u_0_metered(1.0, f.dE0) == pytest.approx(0.0, abs=1e-12)


# ---- composed floor: monotonicity and the bare-meter anchor ---------------------

def test_beta_composed_decreasing_in_n_and_q0():
    """Deploying more re-execution (n) or overhead metering (q0) lowers the floor."""
    f, c, C_dec = DEFAULT.floor, DEFAULT.channel, _C_dec()
    m = 0.05

    def beta_at(n, q0):
        return beta_composed(
            u_dec=u_dec_reexec(c.sigma_dec, C_dec, n, c.z_gamma),
            u_0=u_0_metered(q0, f.dE0),
            u_m=u_m_transfer(m, c.r_bar_dec, C_dec),
            u_eta=0.0,
            r_lo_cov=f.r_lo_cov,
            m=m,
            C_max=f.C_max,
        )

    assert beta_at(100, 0.5) < beta_at(10, 0.5)   # more re-execution
    assert beta_at(100, 0.9) < beta_at(100, 0.5)  # more overhead metering


def test_beta_composed_increasing_in_transfer_gap():
    """A wider transfer gap m raises the floor (wider numerator and lower divisor)."""
    f, c, C_dec = DEFAULT.floor, DEFAULT.channel, _C_dec()

    def beta_at(m):
        return beta_composed(
            u_dec=u_dec_reexec(c.sigma_dec, C_dec, 100, c.z_gamma),
            u_0=u_0_metered(0.5, f.dE0),
            u_m=u_m_transfer(m, c.r_bar_dec, C_dec),
            u_eta=0.0,
            r_lo_cov=f.r_lo_cov,
            m=m,
            C_max=f.C_max,
        )

    assert beta_at(0.15) > beta_at(0.05)


def test_no_channel_limit_reproduces_bare():
    """Anchor: the no-channel limit equals beta(empty) exactly.

    With the *full* conceded declared band (no re-execution channel), no transfer
    gap (m=0), no overhead metering (q0=0) and no meter noise, beta_composed must
    reproduce the Stage-1 closed form -- the regression guard tying this study back
    to powertoflops.floor.beta_bare. This reuses the same allowance split as
    tests/test_floor.py::test_budget_matches_bare.
    """
    f, C_dec = DEFAULT.floor, _C_dec()
    u_dec_full = f.hfu_dec * (f.r_hi - f.r_lo) * f.C_max  # full band, not sigma_dec
    beta = beta_composed(
        u_dec=u_dec_full,
        u_0=u_0_metered(0.0, f.dE0),
        u_m=u_m_transfer(0.0, DEFAULT.channel.r_bar_dec, C_dec),
        u_eta=0.0,
        r_lo_cov=f.r_lo,  # bare meter: covert edge == declared cheapest
        m=0.0,
        C_max=f.C_max,
    )
    bare = beta_bare(
        hfu_dec=f.hfu_dec, r_hi=f.r_hi, r_lo=f.r_lo,
        dE0=f.dE0, C_max=f.C_max,
    )
    assert beta == pytest.approx(bare, rel=1e-12)


def test_beta_composed_rejects_unit_gap():
    with pytest.raises(ValueError):
        beta_composed(0.0, 0.0, 0.0, 0.0, r_lo_cov=1e-12, m=1.0, C_max=1e18)
