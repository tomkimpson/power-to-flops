"""Tests for wiring measured bands into the config (no GPU)."""

from __future__ import annotations

import dataclasses
import json

from powertoflops.config import DEFAULT
from powertoflops.floor import beta_bare
from powertoflops.measured import MeasuredBands, config_with_measured, load_measured_bands


def _mb(**over) -> MeasuredBands:
    base = dict(
        r_lo=2.0e-12, r_hi=8.0e-12, r_lo_op=2.0e-12, w0=2.5,
        P0_lo=60.0, P0_hi=110.0, sigma_dec=1.0e-13,
        gpu_name="NVIDIA A100", driver_version="580.159.04",
    )
    base.update(over)
    return MeasuredBands(**base)


def test_override_touches_exactly_the_six_fields():
    from powertoflops.measured import covert_edge

    mb = _mb(r_marg_cov=_R_MARG_COV)
    cfg = config_with_measured(mb)
    # the six measured fields are swapped in
    assert cfg.floor.r_lo == mb.r_lo
    assert cfg.floor.r_hi == mb.r_hi
    # the covert denominator is the Exp-7 MARGINAL edge, not an Exp-2 quotient
    assert cfg.floor.r_lo_cov == covert_edge(mb, "int8")
    assert cfg.floor.P0_lo == mb.P0_lo
    assert cfg.floor.P0_hi == mb.P0_hi
    assert cfg.channel.sigma_dec == mb.sigma_dec
    # everything else is carried through unchanged
    assert cfg.floor.hfu_dec == DEFAULT.floor.hfu_dec
    assert cfg.floor.F_max == DEFAULT.floor.F_max
    assert cfg.floor.T == DEFAULT.floor.T
    assert cfg.channel.z_gamma == DEFAULT.channel.z_gamma
    assert cfg.channel.r_bar_dec == DEFAULT.channel.r_bar_dec
    assert cfg.sim == DEFAULT.sim
    assert cfg.bench == DEFAULT.bench


def test_default_is_not_mutated():
    before = dataclasses.asdict(DEFAULT)
    config_with_measured(_mb(r_marg_cov=_R_MARG_COV))
    assert dataclasses.asdict(DEFAULT) == before


def test_measured_flows_through_beta_bare():
    cfg = config_with_measured(_mb(r_marg_cov=_R_MARG_COV))
    f = cfg.floor
    # Mirror the headline path (plot_beta_phys): the bare-meter denominator is
    # the covert edge r_lo_cov = the Exp-7 marginal LCB, 0.55 pJ/op here.
    beta = beta_bare(f.hfu_dec, f.r_hi, f.r_lo_cov, f.dE0, f.C_max)
    assert beta > 0.0
    # with r_hi/r_lo_cov = 8.0/0.55 and hfu_dec = 1, the declared term alone
    # is ~13.5, so the marginal edge concedes strictly more than the Exp-2
    # quotient (2.0 pJ/op) would have: 13.5 vs 3.
    assert beta > 13.0


def test_config_never_prices_covert_work_at_the_exp2_quotient():
    # Regression guard: the Exp-2 operand edge r_lo_op is a total quotient that
    # amortises baseline power the covert work does not draw. Wiring it into the
    # covert DENOMINATOR would understate beta. It must never be used there.
    mb = _mb(r_marg_cov=_R_MARG_COV)
    cfg = config_with_measured(mb)
    assert cfg.floor.r_lo_cov != mb.r_lo_op
    assert cfg.floor.r_lo_cov < mb.r_lo_op      # marginal edge is the cheaper one


def test_config_refuses_an_artifact_without_the_marginal_edge():
    # No Exp-7 capture means no defensible covert denominator: fail loud rather
    # than silently falling back to the Exp-2 quotient.
    import pytest

    with pytest.raises(ValueError, match="r_marg_cov"):
        config_with_measured(_mb())


def test_load_measured_bands_ignores_extra_keys(tmp_path):
    payload = dataclasses.asdict(_mb())
    payload["affine_r2"] = 0.998     # extra diagnostic, must be ignored
    path = tmp_path / "measured_bands.json"
    path.write_text(json.dumps(payload))
    mb = load_measured_bands(path)
    assert mb.r_lo == 2.0e-12
    assert mb.w0 == 2.5
    assert mb.source_path == str(path)


# ---- referee B1/B2: per-format bands + Exp-7 marginal fields ---------------


_FORMAT_BANDS = {"fp16": {"r_lo": 1.1e-12, "r_hi": 2.2e-12, "n_cells": 12}}
_R_MARG_COV = {"int8": {"r_marg": 0.6e-12, "r_marg_lcb95": 0.55e-12,
                        "se_slope": 2e-14, "r2": 0.99, "n_cells": 6,
                        "intercept_j": 900.0, "dose0_energy_j": 905.0,
                        "stratum_mhz": 1395, "cov_dtype": "int8",
                        "alpha": 0.05, "strata": {}}}


def test_new_band_fields_load_from_artifact(tmp_path):
    payload = dataclasses.asdict(
        _mb(format_bands=_FORMAT_BANDS, r_marg_cov=_R_MARG_COV,
            r_marginal=1.61e-12))
    path = tmp_path / "measured_bands.json"
    path.write_text(json.dumps(payload))
    mb = load_measured_bands(path)
    assert mb.format_bands == _FORMAT_BANDS
    assert mb.r_marg_cov == _R_MARG_COV
    assert mb.r_marginal == 1.61e-12
    from powertoflops.measured import covert_edge, format_band
    assert format_band(mb, "fp16") == (1.1e-12, 2.2e-12)
    assert covert_edge(mb, "int8") == 0.55e-12


def test_legacy_artifact_fails_loudly_not_silently(tmp_path):
    # An artifact predating the per-format split / Exp 7 loads (fields None),
    # but the accessors the headline uses must refuse it — never fall back to
    # the pooled cross-format band or a total-E/C covert edge (referee B1/B2).
    import pytest
    from powertoflops.measured import covert_edge, format_band

    payload = dataclasses.asdict(_mb())
    path = tmp_path / "measured_bands.json"
    path.write_text(json.dumps(payload))
    mb = load_measured_bands(path)
    assert mb.format_bands is None and mb.r_marg_cov is None
    with pytest.raises(ValueError, match="format_bands"):
        format_band(mb, "fp16")
    with pytest.raises(ValueError, match="r_marg_cov"):
        covert_edge(mb, "int8")


def test_useful_marginal_edge_reads_fields_and_fails_loud(tmp_path):
    # B5 (third round): the ladder denominator is the Exp-8 MARGINAL useful edge
    # (r_useful_marg); an artifact predating that capture must fail loud, never
    # fall back to the total-quotient useful band.
    import pytest
    from powertoflops.measured import useful_marginal_edge

    mb = _mb(r_useful_marg_lo=2.0e-12, r_useful_marg_hi=2.9e-12)
    assert useful_marginal_edge(mb) == (2.0e-12, 2.9e-12)

    with pytest.raises(ValueError, match="r_useful_marg"):
        useful_marginal_edge(_mb())          # fields default to None


def test_accessors_reject_missing_format():
    import pytest
    from powertoflops.measured import covert_edge, format_band

    mb = _mb(format_bands=_FORMAT_BANDS, r_marg_cov=_R_MARG_COV)
    with pytest.raises(ValueError, match="fp8"):
        format_band(mb, "fp8")
    with pytest.raises(ValueError, match="fp16"):
        covert_edge(mb, "fp16")


def test_new_fields_do_not_leak_into_config():
    # config_with_measured still overrides exactly the six fields.
    mb = _mb(format_bands=_FORMAT_BANDS, r_marg_cov=_R_MARG_COV,
             r_marginal=1.61e-12)
    cfg = config_with_measured(mb)
    assert cfg.floor.r_lo == mb.r_lo
    assert cfg.floor.F_max == DEFAULT.floor.F_max
    assert cfg.bench == DEFAULT.bench
