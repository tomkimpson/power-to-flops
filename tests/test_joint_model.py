"""Joint format-consistency guard for the headline floor (referee B1, no GPU).

The first-round headline paired an fp32 declared upper edge with the int8
pooled floor against fp16 capacity and kappa = 2 — extrema from three formats
that cannot coexist in one observation window. These tests pin the repair:
powertoflops.joint_model is the only path to the headline, and it must REJECT any
cross-format combination while reproducing floor.beta_phys_worst_case exactly
on a self-consistent case.
"""

from __future__ import annotations

import numpy as np
import pytest

from powertoflops.config import A100_DENSE_PEAK_OPS
from powertoflops.floor import beta_phys, beta_phys_worst_case, energy_slack
from powertoflops.joint_model import FormatMixError, JointCase, headline, validate_joint_case
from powertoflops.measured import MeasuredBands


def _mb(**over) -> MeasuredBands:
    """Artifact stub shaped like a post-Exp-7 measured_bands.json."""
    base = dict(
        # pooled descriptive span (cross-format by construction — the min is
        # an int8 cell, the max an fp32 cell, as in the committed artifact)
        r_lo=0.8e-12, r_hi=20.0e-12, r_lo_cov=1.2e-12, w0=1.8,
        P0_lo=60.0, P0_hi=100.0, sigma_dec=1.0e-13,
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


def _case(mb: MeasuredBands, declared="fp16", covert="int8", **over) -> JointCase:
    """A self-consistent case built the same way headline() builds it."""
    band = mb.format_bands[declared]
    base = dict(
        declared_format=declared, covert_format=covert,
        r_lo_dec=band["r_lo"], r_hi_dec=band["r_hi"],
        r_lo_cov=mb.r_marg_cov[covert]["r_marg_lcb95"],
        C_max=A100_DENSE_PEAK_OPS[declared],
        kappa=A100_DENSE_PEAK_OPS[covert] / A100_DENSE_PEAK_OPS[declared],
        dP0=mb.P0_hi - mb.P0_lo,
    )
    base.update(over)
    return JointCase(**base)


def test_consistent_case_validates():
    mb = _mb()
    validate_joint_case(_case(mb), mb)                      # must not raise
    validate_joint_case(_case(mb, covert="fp16"), mb)       # covert == declared


def test_headline_matches_direct_composition():
    mb = _mb()
    got = headline(mb, "fp16", "int8")
    case = got["case"]
    args = dict(r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec,
                r_lo_cov=case.r_lo_cov, dE0=case.dP0, C_max=case.C_max)
    h_ref, b_ref = beta_phys_worst_case(**args, kappa=case.kappa)
    assert got["h_star"] == h_ref
    assert got["beta_star"] == b_ref
    assert got["S1"] == pytest.approx(float(energy_slack(1.0, **args)))
    # slack line: S(0) = b, S(1) = s + b
    assert got["S1"] == pytest.approx(got["s"] + got["b"])


def test_headline_inputs_are_single_format():
    # The declared side comes from the declared format's own band, the covert
    # edge from the covert format's marginal LCB — never the pooled extrema.
    mb = _mb()
    case = headline(mb, "fp16", "int8")["case"]
    assert case.r_lo_dec == mb.format_bands["fp16"]["r_lo"]
    assert case.r_hi_dec == mb.format_bands["fp16"]["r_hi"]
    assert case.r_lo_cov == mb.r_marg_cov["int8"]["r_marg_lcb95"]
    assert case.r_hi_dec != mb.r_hi          # pooled max is an fp32 cell
    assert case.r_lo_dec != mb.r_lo          # pooled min is an int8 cell


def test_old_cross_format_headline_inputs_are_rejected():
    # The exact first-round combination (referee B1): pooled fp32 upper edge
    # and pooled int8 floor as the "declared band", fp16 capacity, kappa = 2.
    mb = _mb()
    old = _case(mb, r_lo_dec=mb.r_lo, r_hi_dec=mb.r_hi, r_lo_cov=mb.r_lo)
    with pytest.raises(FormatMixError, match="declared format's own cells"):
        validate_joint_case(old, mb)


def test_total_ec_edge_rejected_as_covert_denominator():
    # Referee B2: the covert denominator must be the Exp-7 marginal LCB, not a
    # total-E/C edge that amortises baseline power over the ops.
    mb = _mb()
    with pytest.raises(FormatMixError, match="marginal LCB"):
        validate_joint_case(_case(mb, r_lo_cov=mb.r_lo_cov), mb)


def test_mismatched_capacity_and_kappa_rejected():
    mb = _mb()
    with pytest.raises(FormatMixError, match="peak"):
        validate_joint_case(
            _case(mb, C_max=A100_DENSE_PEAK_OPS["fp32"]), mb)
    with pytest.raises(FormatMixError, match="kappa"):
        validate_joint_case(_case(mb, covert="fp16", kappa=2.0), mb)


def test_unknown_format_rejected():
    mb = _mb()
    with pytest.raises(FormatMixError, match="unknown"):
        validate_joint_case(_case(mb, declared_format="fp8"), mb)


def test_pre_exp7_artifact_fails_loudly():
    # Referee B2: no silent fallback to a total-E/C covert edge.
    mb = _mb(r_marg_cov=None)
    with pytest.raises(ValueError, match="r_marg_cov"):
        headline(mb)


def test_pre_format_bands_artifact_fails_loudly():
    # Referee B1: no silent fallback to the pooled cross-format band.
    mb = _mb(format_bands=None)
    with pytest.raises(ValueError, match="format_bands"):
        headline(mb)


def test_binding_regimes_around_h_star():
    # "Energy never binds" was false (referee B1): energy is the binding side
    # of the min for h < h*, capacity for h > h*, both at h*.
    mb = _mb()
    got = headline(mb, "fp16", "int8")
    case, h_star = got["case"], got["h_star"]
    assert 0.0 < h_star < 1.0
    args = dict(r_hi_dec=case.r_hi_dec, r_lo_dec=case.r_lo_dec,
                r_lo_cov=case.r_lo_cov, dE0=case.dP0, C_max=case.C_max)
    for h in (0.0, 0.5 * h_star):
        assert beta_phys(h, **args, kappa=case.kappa) == pytest.approx(
            float(energy_slack(h, **args)))                 # energy binds
    for h in (h_star + 0.5 * (1 - h_star), 1.0):
        assert beta_phys(h, **args, kappa=case.kappa) == pytest.approx(
            (1.0 - h) * case.kappa)                          # capacity binds
    assert float(energy_slack(h_star, **args)) == pytest.approx(
        (1.0 - h_star) * case.kappa)                         # both at h*


def test_monotone_in_covert_edge():
    # A cheaper covert edge can only widen the floor: beta* is non-increasing
    # in r_lo_cov (sanity for the LCB direction — using the LCB rather than
    # the point slope must not shrink the reported floor).
    mb_point = _mb()
    lcb = dict(mb_point.r_marg_cov["int8"])
    hi = dict(lcb, r_marg_lcb95=lcb["r_marg"])               # edge at the point slope
    mb_hi = _mb(r_marg_cov={"int8": hi, "fp16": mb_point.r_marg_cov["fp16"]})
    assert (headline(mb_point, "fp16", "int8")["beta_star"]
            >= headline(mb_hi, "fp16", "int8")["beta_star"])


def test_declared_fp32_scenario_is_a_unit_artifact():
    # Documented reason the headline fixes declared = fp16: with declared fp32
    # the same physics reads as a huge beta purely because the yardstick
    # (declared capacity) shrinks 16x — kappa = 624/19.5 = 32.
    mb = _mb()
    got = headline(mb, "fp32", "int8")
    assert got["case"].kappa == pytest.approx(32.0)
    assert got["beta_star"] > headline(mb, "fp16", "int8")["beta_star"]


def test_numpy_scalars_accepted():
    # h grids and artifact JSON round-trips produce numpy floats; the validator
    # must compare values, not types.
    mb = _mb()
    case = _case(mb)
    case = JointCase(**{**case.__dict__,
                        "r_hi_dec": np.float64(case.r_hi_dec)})
    validate_joint_case(case, mb)
