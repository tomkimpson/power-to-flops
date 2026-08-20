"""Band-extraction math on synthetic records (no GPU).

These validate the load-bearing analysis: feed RunRecords with *known* r and
power values and assert the extracted bands recover the planted numbers.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from powertoflops.bench.bands import (
    IDLE_TOL_FRAC,
    clean_overhead_records,
    clean_records,
    exchange_band,
    extract_bands,
    marginal_rate,
    marginal_rate_compute_bound,
    operand_core,
    overhead_band,
    sigma_dec_from,
)
from powertoflops.bench.records import REFERENCE_DIST, RunRecord


def _rec(**over) -> RunRecord:
    base = dict(
        experiment=1, sweep_id="s", repeat_index=0,
        M=4, N=4, K=4, iters=1, ops_nominal=128,
        dtype="fp16", locality="dram",
        operand_dist_a="dense_random", operand_dist_b="dense_random",
        util_frac=None,
        energy_j=1.0, duration_s=1.0, mean_power_w=1.0,
        meter_method="nvml_energy_counter", r_j_per_op=1.0e-12,
        gpu_name="A100", gpu_uuid="u", driver_version="x", cuda_version="y",
        torch_version="z", persistence_mode=True, sm_clock_mhz=1410,
        mem_clock_mhz=1215, power_limit_w=400.0, temperature_c=50.0,
        throttled=False, throttle_reasons=0, clocks_locked=False,
        timestamp_utc="2026-06-18T00:00:00+00:00",
    )
    base.update(over)
    return RunRecord(**base)


def test_exchange_band_recovers_min_max():
    rs = [1.0e-12, 4.0e-12, 12.0e-12, 7.0e-12]
    recs = [_rec(dtype=f"d{i}", r_j_per_op=r) for i, r in enumerate(rs)]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == 1.0e-12
    assert r_hi == 12.0e-12


def test_exchange_band_medians_over_repeats():
    # two repeats of one cell; median is robust to a single outlier
    recs = [
        _rec(dtype="fp16", repeat_index=0, r_j_per_op=2.0e-12),
        _rec(dtype="fp16", repeat_index=1, r_j_per_op=2.2e-12),
        _rec(dtype="fp16", repeat_index=2, r_j_per_op=9.9e-12),  # outlier
        _rec(dtype="fp32", repeat_index=0, r_j_per_op=8.0e-12),
    ]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == 2.2e-12          # median of the fp16 cell
    assert r_hi == 8.0e-12


def test_exchange_band_drops_harmful_throttle_and_reference():
    recs = [
        _rec(dtype="a", r_j_per_op=3.0e-12),
        _rec(dtype="b", r_j_per_op=99.0e-12, throttle_reasons=0x40),  # HW thermal
        _rec(dtype="c", operand_dist_a=REFERENCE_DIST, r_j_per_op=50.0e-12),
    ]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == r_hi == 3.0e-12


def test_trimmed_rows_excluded_from_bands():
    # The pre-specified high-tail exclusion lives at band extraction now: rows
    # flagged trimmed at capture must not widen any band.
    recs = [
        _rec(dtype="a", r_j_per_op=3.0e-12),
        _rec(dtype="b", r_j_per_op=99.0e-12, trimmed=True),
    ]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == r_hi == 3.0e-12


def test_sw_power_cap_runs_are_kept():
    # SW power cap (0x4) is the intended full-power regime, NOT harmful throttling:
    # the high-energy cell must stay in the band.
    recs = [
        _rec(dtype="a", r_j_per_op=3.0e-12),
        _rec(dtype="b", r_j_per_op=9.0e-12, throttle_reasons=0x4, throttled=True),
    ]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == 3.0e-12
    assert r_hi == 9.0e-12


def test_clean_records_drops_idle_stalled_window():
    # Within a cell, a window drawing < 0.5x the cell's peak power spent its wall
    # time mostly idle (a stretched bracket), so r = E/C is inflated; drop it.
    recs = [
        _rec(repeat_index=0, mean_power_w=500.0, r_j_per_op=1.7e-12),
        _rec(repeat_index=1, mean_power_w=510.0, r_j_per_op=1.8e-12),
        _rec(repeat_index=2, mean_power_w=88.0, r_j_per_op=5.3e-12),  # stalled
    ]
    kept = clean_records(recs)
    assert len(kept) == 2
    assert all(r.mean_power_w >= 0.5 * 510.0 for r in kept)


def test_singleton_cell_is_never_dropped():
    # A lone window is its own peak, so it survives even at low (idle-like) power.
    assert len(clean_records([_rec(dtype="x", mean_power_w=88.0)])) == 1


def test_exchange_band_excludes_stalled_window():
    # The stalled bf16 window must not enter the cell median that sets r_lo.
    recs = [
        _rec(dtype="bf16", repeat_index=0, mean_power_w=500.0, r_j_per_op=1.70e-12),
        _rec(dtype="bf16", repeat_index=1, mean_power_w=510.0, r_j_per_op=1.80e-12),
        _rec(dtype="bf16", repeat_index=2, mean_power_w=490.0, r_j_per_op=1.75e-12),
        _rec(dtype="bf16", repeat_index=3, mean_power_w=88.0, r_j_per_op=5.30e-12),
        _rec(dtype="fp32", mean_power_w=400.0, r_j_per_op=18.0e-12),
    ]
    r_lo, r_hi = exchange_band(recs)
    assert r_hi == 18.0e-12
    assert r_lo == 1.75e-12          # median of the 3 kept bf16 windows, not 1.775


def test_operand_core_returns_descriptive_envelope_and_w0():
    dists = {"zeros": 1.0e-12, "low_entropy": 1.5e-12,
             "dense_random": 2.0e-12, "adversarial_toggle": 3.0e-12}
    recs = [_rec(experiment=2, operand_dist_a=d, operand_dist_b=d, r_j_per_op=r)
            for d, r in dists.items()]
    r_lo_op, r_hi_op, w0 = operand_core(recs)
    assert r_lo_op == 1.0e-12
    assert r_hi_op == 3.0e-12
    assert w0 == 3.0


def test_sigma_dec_from_declared_typical():
    vals = [2.00e-12, 2.10e-12, 1.95e-12, 2.05e-12]
    recs = [_rec(experiment=2, operand_dist_a="declared_typical",
                 operand_dist_b="declared_typical", r_j_per_op=v) for v in vals]
    # plus a non-declared run that must be ignored
    recs.append(_rec(experiment=2, operand_dist_a="zeros", r_j_per_op=9e-12))
    got = sigma_dec_from(recs)
    assert np.isclose(got, np.std(vals, ddof=1))


def test_sigma_dec_zero_when_insufficient():
    recs = [_rec(experiment=2, operand_dist_a="declared_typical", r_j_per_op=2e-12)]
    assert sigma_dec_from(recs) == 0.0


def test_overhead_band_recovers_slope_and_intercept():
    # plant P = r*F + P0 exactly: r = 1e-9 W/(FLOP/s), P0 = 80 W. The idle row is
    # written the way the runner writes it: zero FLOPs, r = NaN.
    r_true, P0_true = 1.0e-9, 80.0
    recs = []
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        F = d * 3.0e11                      # FLOP/s
        duration = 1.0
        P = r_true * F + P0_true
        recs.append(_rec(
            experiment=3, util_frac=d, ops_nominal=int(F * duration),
            duration_s=duration, mean_power_w=P,
            r_j_per_op=(1.0e-12 if F > 0 else float("nan")),
        ))
    ob = overhead_band(recs)
    assert np.isclose(ob.r_marginal, r_true, rtol=1e-6)
    # fit intercept is reported separately from the band ...
    assert np.isclose(ob.fit_intercept_w, P0_true, rtol=1e-6)
    # ... and the band is the observed idle value widened by the stated tolerance.
    assert np.isclose(ob.P0_lo, P0_true * (1.0 - IDLE_TOL_FRAC))
    assert np.isclose(ob.P0_hi, P0_true * (1.0 + IDLE_TOL_FRAC))
    assert ob.affine_r2 > 0.999


def test_overhead_band_idle_spread():
    # two idle runs -> the band is their observed spread widened by the tolerance
    recs = [
        _rec(experiment=3, util_frac=0.0, ops_nominal=0, duration_s=1.0, mean_power_w=70.0,
             r_j_per_op=float("nan")),
        _rec(experiment=3, util_frac=0.0, ops_nominal=0, duration_s=1.0, mean_power_w=90.0,
             r_j_per_op=float("nan")),
        _rec(experiment=3, util_frac=1.0, ops_nominal=int(3e11), duration_s=1.0,
             mean_power_w=380.0),
    ]
    ob = overhead_band(recs)
    assert np.isclose(ob.P0_lo, 70.0 * (1.0 - IDLE_TOL_FRAC))
    assert np.isclose(ob.P0_hi, 90.0 * (1.0 + IDLE_TOL_FRAC))
    assert ob.n_idle == 2


def test_overhead_band_contains_measured_idle_regression():
    # Regression for the committed-band bug (review C3): the real Exp-3 idle rows
    # read 77.65-78.02 W, yet the committed band was [84.97, 91.52] W — the fit
    # intercept ±1SE — because _clean_r dropped every zero-FLOP row (r = NaN)
    # before overhead_band ever saw it. The band MUST contain the observed idle.
    idle_powers = (77.65, 77.72, 77.90, 78.02)
    recs = [
        _rec(experiment=3, util_frac=0.0, repeat_index=i, ops_nominal=0, duration_s=1.0,
             mean_power_w=p, r_j_per_op=float("nan"))
        for i, p in enumerate(idle_powers)
    ]
    # Loaded duty-cycle rows arranged so a fit over them alone extrapolates to an
    # intercept near 88 W (as the committed artifact did).
    for d in (0.25, 0.5, 0.75, 1.0):
        F = d * 3.0e11
        recs.append(_rec(
            experiment=3, util_frac=d, ops_nominal=int(F), duration_s=1.0,
            mean_power_w=88.0 + 1.0e-9 * F,
        ))
    ob = overhead_band(recs)
    assert ob.P0_lo <= min(idle_powers)
    assert ob.P0_hi >= max(idle_powers)
    assert ob.n_idle == len(idle_powers)


def test_overhead_band_falls_back_to_intercept_without_idle():
    # No util==0 rows at all -> explicit fallback to fit intercept ± 1SE.
    recs = []
    for d in (0.25, 0.5, 0.75, 1.0):
        F = d * 3.0e11
        recs.append(_rec(
            experiment=3, util_frac=d, ops_nominal=int(F), duration_s=1.0,
            mean_power_w=80.0 + 1.0e-9 * F + (0.5 if d == 0.5 else 0.0),
        ))
    ob = overhead_band(recs)
    assert ob.n_idle == 0
    assert ob.P0_lo == ob.fit_intercept_w - ob.fit_se_w
    assert ob.P0_hi == ob.fit_intercept_w + ob.fit_se_w


def test_marginal_rate_recovers_planted_line():
    # plant P = r*F + P0 across sustained-rate cells: the slope is the marginal
    # exchange rate dP/dF the theory uses (review C2), NOT total E/C.
    r_true, P0_true = 2.0e-9, 75.0
    recs = []
    for d in (0.2, 0.4, 0.6, 0.8, 1.0):
        F = d * 2.0e11
        recs.append(_rec(experiment=3, util_frac=d, ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true))
    mr = marginal_rate(recs)
    assert np.isclose(mr.r_marginal, r_true, rtol=1e-6)
    assert np.isclose(mr.intercept_w, P0_true, rtol=1e-6)
    assert mr.r2 > 0.999
    assert mr.n_cells == 5


def test_marginal_rate_is_robust_to_a_repeat_outlier():
    # repeats collapse to the cell median before the fit, so one bad window in a
    # cell must not tilt the marginal slope.
    r_true, P0_true = 2.0e-9, 75.0
    recs = []
    for d in (0.25, 0.5, 0.75, 1.0):
        F = d * 2.0e11
        for k in range(3):
            recs.append(_rec(experiment=3, util_frac=d, repeat_index=k,
                             ops_nominal=int(F), duration_s=1.0,
                             mean_power_w=r_true * F + P0_true))
    # outlier repeat in the d=0.5 cell (median of 3 ignores it)
    recs.append(_rec(experiment=3, util_frac=0.5, repeat_index=3,
                     ops_nominal=int(0.5 * 2.0e11), duration_s=1.0, mean_power_w=999.0))
    mr = marginal_rate(recs)
    assert np.isclose(mr.r_marginal, r_true, rtol=1e-6)
    assert np.isclose(mr.intercept_w, P0_true, rtol=1e-6)


def test_marginal_rate_compute_bound_ignores_bent_mid_ladder():
    # The subfit (review C2 sensitivity check) uses only the idle anchors and
    # the compute-bound cubes; launch/bandwidth-bound mid-ladder shapes that
    # bend the pooled fit must not tilt it.
    r_true, P0_true = 2.0e-9, 75.0
    recs = []
    for variant in ("ctx", "no_ctx", "cached", "cooled", "post_load"):
        recs.append(_rec(experiment=3, M=0, N=0, K=0, ops_nominal=0, util_frac=0.0,
                         idle_variant=variant, duration_s=1.0,
                         mean_power_w=P0_true, r_j_per_op=float("nan")))
    for j, m in enumerate((2048, 3072, 4096, 6144, 8192)):
        F = (j + 1) * 5.0e10
        recs.append(_rec(experiment=3, M=m, N=m, K=m, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true))
    # mid-ladder small/flat shapes far off the line (bent by launch overhead)
    for j, (m, n, k) in enumerate(((256, 256, 256), (8192, 8192, 64))):
        F = (j + 1) * 1.0e10
        recs.append(_rec(experiment=3, M=m, N=n, K=k, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true + 120.0))
    sub = marginal_rate_compute_bound(recs)
    # atol=0: the default atol=1e-8 dwarfs slopes of order 1e-9
    assert np.isclose(sub.r_marginal, r_true, rtol=1e-6, atol=0.0)
    assert np.isclose(sub.intercept_w, P0_true, rtol=1e-6, atol=0.0)
    assert sub.n_cells == 10  # 5 idle anchors + 5 cubes
    # the pooled fit is tilted by the bent shapes; the subfit is the contrast
    assert not np.isclose(marginal_rate(clean_overhead_records(recs)).r_marginal,
                          r_true, rtol=1e-3, atol=0.0)


def test_marginal_rate_compute_bound_respects_min_dim():
    r_true, P0_true = 2.0e-9, 75.0
    recs = [_rec(experiment=3, M=0, N=0, K=0, ops_nominal=0, util_frac=0.0,
                 idle_variant=v, duration_s=1.0, mean_power_w=P0_true,
                 r_j_per_op=float("nan")) for v in ("ctx", "cached")]
    for j, m in enumerate((1024, 2048, 4096)):
        F = (j + 1) * 5.0e10
        recs.append(_rec(experiment=3, M=m, N=m, K=m, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true))
    assert marginal_rate_compute_bound(recs).n_cells == 4          # 1024 excluded
    assert marginal_rate_compute_bound(recs, cube_min_dim=1024).n_cells == 5


def test_extract_bands_reports_marginal_r():
    exp1 = [_rec(dtype="fp16", r_j_per_op=1e-12),
            _rec(dtype="fp32", r_j_per_op=10e-12)]
    exp2 = [_rec(experiment=2, operand_dist_a="zeros", r_j_per_op=1e-12),
            _rec(experiment=2, operand_dist_a="adv", r_j_per_op=2e-12)]
    r_true, P0_true = 1.0e-9, 80.0
    exp3 = []
    for d in (0.0, 0.5, 1.0):
        F = d * 3.0e11
        exp3.append(_rec(experiment=3, util_frac=d, ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true,
                         r_j_per_op=(1e-12 if F > 0 else float("nan"))))
    br = extract_bands(exp1, exp2, exp3)
    assert np.isclose(br.r_marginal, r_true, rtol=1e-6)
    assert np.isclose(br.P0_fit_intercept, P0_true, rtol=1e-6)


def test_extract_bands_composes():
    exp1 = [_rec(dtype="fp16", r_j_per_op=1e-12),
            _rec(dtype="fp32", r_j_per_op=10e-12)]
    exp2 = [_rec(experiment=2, operand_dist_a="zeros", r_j_per_op=1e-12),
            _rec(experiment=2, operand_dist_a="adv", r_j_per_op=2e-12)]
    exp3 = [_rec(experiment=3, util_frac=0.0, ops_nominal=1, duration_s=1.0, mean_power_w=80.0),
            _rec(experiment=3, util_frac=1.0, ops_nominal=int(3e11), duration_s=1.0,
                 mean_power_w=380.0)]
    br = extract_bands(exp1, exp2, exp3)
    assert br.r_lo == 1e-12 and br.r_hi == 10e-12
    assert br.w0 == 2.0
    assert br.r_lo_op == 1e-12


# ---- R2: shape-varying ladder, idle variants, operand-crossed Exp 1 ---------


def test_marginal_rate_separates_ladder_shapes():
    # The R2.4 ladder varies sustained F through SHAPE at fixed dtype/locality/
    # dist/util — each rung must be its own cell, not collapse into one.
    r_true, P0_true = 1.5e-9, 70.0
    recs = []
    for size in (256, 512, 1024, 2048):
        F = size * 1.0e8
        recs.append(_rec(experiment=3, M=size, N=size, K=size, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=r_true * F + P0_true))
    mr = marginal_rate(recs)
    assert mr.n_cells == 4
    assert np.isclose(mr.r_marginal, r_true, rtol=1e-6)
    assert np.isclose(mr.intercept_w, P0_true, rtol=1e-6)


def test_overhead_band_spans_idle_variants():
    # Five idle-variation cells (R2.4) bound controllable P0: the band must
    # cover the full observed range across variants, each variant its own cell.
    idles = {"no_ctx": 62.0, "ctx": 66.0, "cached": 70.0,
             "post_load": 90.0, "cooled": 74.0}
    exp3 = [_rec(experiment=3, M=0, N=0, K=0, iters=0, util_frac=0.0,
                 ops_nominal=0, idle_variant=v, mean_power_w=p,
                 r_j_per_op=float("nan"))
            for v, p in idles.items()]
    for size in (2048, 4096):   # two busy rungs so the affine fit has cells
        F = size * 1.0e8
        exp3.append(_rec(experiment=3, M=size, N=size, K=size, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=1.5e-9 * F + 70.0))
    ob = overhead_band(exp3)
    assert ob.n_idle == 5
    assert ob.P0_lo == 62.0 * (1.0 - IDLE_TOL_FRAC)
    assert ob.P0_hi == 90.0 * (1.0 + IDLE_TOL_FRAC)
    # The resident-operand band drops the two cold-card cells (no_ctx, ctx) —
    # states the window the floor prices cannot be in — under the same rule.
    assert ob.n_idle_resident == 3
    assert ob.P0_lo_resident == 70.0 * (1.0 - IDLE_TOL_FRAC)
    assert ob.P0_hi_resident == 90.0 * (1.0 + IDLE_TOL_FRAC)
    assert ob.P0_hi_resident - ob.P0_lo_resident < ob.P0_hi - ob.P0_lo


def test_resident_overhead_band_absent_without_resident_cells():
    # No intercept fallback for the restricted band: a sweep that ran only the
    # cold-card idle cells has no resident band, and the ladder must be told so
    # rather than handed the unrestricted one.
    exp3 = [_rec(experiment=3, M=0, N=0, K=0, iters=0, util_frac=0.0,
                 ops_nominal=0, idle_variant=v, mean_power_w=p,
                 r_j_per_op=float("nan"))
            for v, p in (("no_ctx", 62.0), ("ctx", 66.0))]
    for size in (2048, 4096):
        F = size * 1.0e8
        exp3.append(_rec(experiment=3, M=size, N=size, K=size, util_frac=1.0,
                         ops_nominal=int(F), duration_s=1.0,
                         mean_power_w=1.5e-9 * F + 70.0))
    ob = overhead_band(exp3)
    assert ob.n_idle == 2                    # the unrestricted band still lands
    assert ob.P0_lo_resident is None and ob.P0_hi_resident is None
    assert ob.n_idle_resident == 0


def test_exchange_band_spans_operand_cells():
    # R2.2 crosses operand dists into Exp 1: cells with the same dtype/locality
    # but different dists must NOT pool — the band is the full driver envelope.
    recs = [_rec(dtype="fp16", operand_dist_a="zeros", r_j_per_op=1e-12),
            _rec(dtype="fp16", operand_dist_a="adversarial_toggle",
                 r_j_per_op=5e-12)]
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == 1e-12
    assert r_hi == 5e-12


def test_clock_stratum_prefers_p50_falls_back():
    from powertoflops.bench.bands import CLOCK_BIN_MHZ, clock_stratum

    assert CLOCK_BIN_MHZ == 45
    r_new = _rec(sm_clock_p50_mhz=1395.0, sm_clock_mhz=840)
    assert clock_stratum(r_new) == 1395          # p50 wins over the snapshot
    r_legacy = _rec(sm_clock_mhz=1125)
    assert clock_stratum(r_legacy) == 1125       # legacy fallback
    r_binned = _rec(sm_clock_p50_mhz=1400.0)
    assert clock_stratum(r_binned) == 1395       # 1400 -> nearest 45 MHz bin


def test_exchange_band_strata_partitions_by_achieved_clock():
    from powertoflops.bench.bands import exchange_band_strata

    recs = [
        _rec(dtype="fp16", sm_clock_p50_mhz=1395.0, r_j_per_op=1e-12),
        _rec(dtype="fp32", sm_clock_p50_mhz=1395.0, r_j_per_op=4e-12),
        _rec(dtype="fp16", sm_clock_p50_mhz=1125.0, r_j_per_op=2e-12),
        _rec(dtype="fp32", sm_clock_p50_mhz=1125.0, r_j_per_op=8e-12),
    ]
    strata = exchange_band_strata(recs)
    assert set(strata) == {1395, 1125}
    assert strata[1395]["r_lo"] == 1e-12 and strata[1395]["r_hi"] == 4e-12
    assert strata[1125]["r_lo"] == 2e-12 and strata[1125]["r_hi"] == 8e-12
    assert strata[1395]["n_records"] == 2


def test_operand_core_strata_partitions_by_achieved_clock():
    from powertoflops.bench.bands import operand_core_strata

    recs = [
        _rec(experiment=2, operand_dist_a="zeros",
             sm_clock_p50_mhz=1395.0, r_j_per_op=1e-12),
        _rec(experiment=2, operand_dist_a="adversarial_toggle",
             sm_clock_p50_mhz=1395.0, r_j_per_op=2e-12),
        _rec(experiment=2, operand_dist_a="zeros",
             sm_clock_p50_mhz=1125.0, r_j_per_op=1.5e-12),
        _rec(experiment=2, operand_dist_a="adversarial_toggle",
             sm_clock_p50_mhz=1125.0, r_j_per_op=4.5e-12),
    ]
    strata = operand_core_strata(recs)
    assert set(strata) == {1395, 1125}
    assert strata[1395]["r_lo_op"] == 1e-12
    assert strata[1395]["w0"] == 2.0
    assert np.isclose(strata[1125]["w0"], 3.0)


# ---- referee B1/B2: per-format bands + Exp-7 marginal covert cost ----------


def test_exchange_band_by_format_partitions_by_dtype():
    from powertoflops.bench.bands import exchange_band_by_format

    recs = [
        _rec(dtype="fp16", operand_dist_a="zeros", r_j_per_op=1.1e-12),
        _rec(dtype="fp16", operand_dist_a="declared_typical", r_j_per_op=2.2e-12),
        _rec(dtype="fp32", operand_dist_a="zeros", r_j_per_op=3.0e-12),
        _rec(dtype="fp32", operand_dist_a="declared_typical", r_j_per_op=20.0e-12),
        _rec(dtype="int8", operand_dist_a="zeros", r_j_per_op=0.8e-12),
    ]
    fb = exchange_band_by_format(recs)
    assert set(fb) == {"fp16", "fp32", "int8"}
    assert fb["fp16"] == {"r_lo": 1.1e-12, "r_hi": 2.2e-12, "n_cells": 2}
    assert fb["fp32"] == {"r_lo": 3.0e-12, "r_hi": 20.0e-12, "n_cells": 2}
    assert fb["int8"]["r_lo"] == fb["int8"]["r_hi"] == 0.8e-12
    # the pooled band still spans formats — that is exactly why the headline
    # must not read it (referee B1)
    r_lo, r_hi = exchange_band(recs)
    assert r_lo == 0.8e-12 and r_hi == 20.0e-12


def test_exchange_band_by_format_applies_the_same_cleaning():
    from powertoflops.bench.bands import exchange_band_by_format

    recs = [
        _rec(dtype="fp16", r_j_per_op=2.0e-12),
        _rec(dtype="fp16", operand_dist_a="zeros", r_j_per_op=99e-12,
             throttle_reasons=0x40),                        # harmful throttle
        _rec(dtype="fp16", operand_dist_a="low_entropy", r_j_per_op=50e-12,
             trimmed=True),
    ]
    fb = exchange_band_by_format(recs)
    assert fb["fp16"]["r_lo"] == fb["fp16"]["r_hi"] == 2.0e-12


def _exp7(dose_iters, energy_j, cov_dtype="int8", ops_per_iter=2 * 2048 ** 3,
          **over):
    """One Exp-7 window: fixed fp16 declared burst + a covert dose."""
    base = dict(
        experiment=7, dtype="fp16", operand_dist_a="declared_typical",
        operand_dist_b="declared_typical",
        cov_dtype=cov_dtype, cov_dist="zeros", cov_iters=dose_iters,
        cov_ops_nominal=dose_iters * ops_per_iter,
        energy_j=energy_j, duration_s=10.0, mean_power_w=energy_j / 10.0,
        window_pad_s=3.0,
    )
    base.update(over)
    return _rec(**base)


def test_marginal_covert_cost_recovers_planted_slope():
    from powertoflops.bench.bands import marginal_covert_cost

    r_true, E0 = 0.6e-12, 900.0
    ops = 2 * 2048 ** 3
    recs = [_exp7(d, E0 + r_true * d * ops) for d in (0, 10, 20, 40, 60, 80)]
    mc = marginal_covert_cost(recs, "int8")
    assert np.isclose(mc.r_marg, r_true, rtol=1e-9, atol=0.0)
    assert np.isclose(mc.intercept_j, E0, rtol=1e-9)
    assert np.isclose(mc.dose0_energy_j, E0)
    assert mc.n_cells == 6
    assert mc.r2 > 0.999999
    assert mc.cov_dtype == "int8"
    # exact fit: zero residuals, so the LCB collapses onto the point slope
    assert np.isclose(mc.r_marg_lcb95, mc.r_marg, atol=0.0)


def test_marginal_covert_cost_lcb_is_one_sided_below():
    from powertoflops.bench.bands import marginal_covert_cost

    r_true, E0 = 0.6e-12, 900.0
    ops = 2 * 2048 ** 3
    rng = np.random.default_rng(0)
    recs = [_exp7(d, E0 + r_true * d * ops + rng.normal(0.0, 3.0))
            for d in (0, 10, 20, 40, 60, 80)]
    mc = marginal_covert_cost(recs, "int8")
    assert mc.se_slope > 0.0
    assert mc.r_marg_lcb95 < mc.r_marg          # lower limit, security-safe side


def test_marginal_covert_cost_medians_over_repeats():
    from powertoflops.bench.bands import marginal_covert_cost

    r_true, E0 = 0.6e-12, 900.0
    ops = 2 * 2048 ** 3
    recs = []
    for d in (0, 20, 40, 80):
        for k in range(3):
            recs.append(_exp7(d, E0 + r_true * d * ops, repeat_index=k))
    # outlier repeat in one dose cell: the median ignores it (kept below 2x the
    # cell peak so the within-cell stall guard, which keys off peak power,
    # does not evict the good windows instead)
    recs.append(_exp7(40, 1.5 * E0, repeat_index=3))
    mc = marginal_covert_cost(recs, "int8")
    assert np.isclose(mc.r_marg, r_true, rtol=1e-9, atol=0.0)


def test_marginal_covert_cost_filters_by_covert_format_and_hygiene():
    from powertoflops.bench.bands import marginal_covert_cost

    r_int8, r_fp16, E0 = 0.6e-12, 1.0e-12, 900.0
    ops = 2 * 2048 ** 3
    recs = [_exp7(d, E0 + r_int8 * d * ops) for d in (0, 20, 40, 80)]
    recs += [_exp7(d, E0 + r_fp16 * d * ops, cov_dtype="fp16")
             for d in (0, 20, 40, 80)]
    # contaminated int8 rows that must not tilt the fit
    recs.append(_exp7(20, 5000.0, throttle_reasons=0x40))
    recs.append(_exp7(40, 5000.0, trimmed=True))
    mc8 = marginal_covert_cost(recs, "int8")
    mc16 = marginal_covert_cost(recs, "fp16")
    assert np.isclose(mc8.r_marg, r_int8, rtol=1e-9, atol=0.0)
    assert np.isclose(mc16.r_marg, r_fp16, rtol=1e-9, atol=0.0)


def test_marginal_covert_cost_keeps_low_dose_low_power_windows():
    # Mean window power legitimately GROWS with the dose; a cross-dose stall
    # guard would discard the low doses wholesale. Doses spanning 4x in power
    # must all survive (the per-cell guard still applies within a dose).
    from powertoflops.bench.bands import clean_marginal_records

    recs = [_exp7(d, e) for d, e in ((0, 1000.0), (40, 2500.0), (80, 4000.0))]
    assert len(clean_marginal_records(recs, "int8")) == 3


def test_marginal_covert_cost_requires_three_dose_cells():
    import pytest
    from powertoflops.bench.bands import marginal_covert_cost

    recs = [_exp7(0, 900.0), _exp7(80, 1200.0)]
    with pytest.raises(ValueError, match="3 dose cells"):
        marginal_covert_cost(recs, "int8")


def test_marginal_covert_cost_reports_common_stratum():
    from powertoflops.bench.bands import marginal_covert_cost

    recs = [_exp7(d, 900.0 + d, sm_clock_p50_mhz=1395.0)
            for d in (0, 20, 40, 80)]
    mc = marginal_covert_cost(recs, "int8")
    assert mc.stratum_mhz == 1395
    assert mc.strata == {}


def test_marginal_covert_cost_splits_strata_when_clock_mixed():
    from powertoflops.bench.bands import marginal_covert_cost

    r_hi_clk, r_lo_clk, E0 = 0.5e-12, 0.7e-12, 900.0
    ops = 2 * 2048 ** 3
    recs = [_exp7(d, E0 + r_hi_clk * d * ops, sm_clock_p50_mhz=1395.0)
            for d in (0, 20, 40, 80)]
    recs += [_exp7(d, E0 + r_lo_clk * d * ops, sm_clock_p50_mhz=1125.0)
             for d in (0, 20, 40, 80)]
    mc = marginal_covert_cost(recs, "int8")
    assert mc.stratum_mhz is None            # no common stratum
    assert set(mc.strata) == {1395, 1125}
    assert np.isclose(mc.strata[1395]["r_marg"], r_hi_clk, rtol=1e-9, atol=0.0)
    assert np.isclose(mc.strata[1125]["r_marg"], r_lo_clk, rtol=1e-9, atol=0.0)


# ---- marginal useful band (qblock-marginal dose-response; referee B5, third round) ----


def _qexp7(arm, dose_iters, energy_j, ops_per_iter, **over):
    """One qblock-arm dose window: fixed fp16 declared burst + a qblock dose."""
    return _exp7(dose_iters, energy_j, cov_dtype=arm, ops_per_iter=ops_per_iter,
                 cov_dist="qblock_pythia", **over)


# Realistic qblock op counts per forward (~ops_nominal_qblock): the 2048x8
# cell ~7e12, the 512x8 cell ~1.7e12. Signal dwarfs the metering noise so the
# dose slope is identifiable, as in the real capture.
_OPS_2048x8 = 7_150_000_000_000
_OPS_512x8 = 1_680_000_000_000


def test_useful_marginal_band_recovers_per_arm_slopes():
    from powertoflops.bench.bands import useful_marginal_band

    r_a, r_b, E0 = 2.2e-12, 3.1e-12, 900.0     # 512x8 is the dearer arm
    recs = [_qexp7("qblock_2048x8", d, E0 + r_a * d * _OPS_2048x8, _OPS_2048x8)
            for d in (0, 10, 20, 30, 40)]
    recs += [_qexp7("qblock_512x8", d, E0 + r_b * d * _OPS_512x8, _OPS_512x8)
             for d in (0, 10, 20, 30, 40)]
    ub = useful_marginal_band(recs)
    assert ub.n_cells == 2
    assert np.isclose(ub.r_lo_point, min(r_a, r_b), rtol=1e-9, atol=0.0)
    assert np.isclose(ub.r_hi_point, max(r_a, r_b), rtol=1e-9, atol=0.0)
    # exact fit -> both LCBs collapse onto their point slopes
    assert np.isclose(ub.r_lo, min(r_a, r_b), atol=0.0)
    assert np.isclose(ub.r_hi, max(r_a, r_b), atol=0.0)


def test_useful_marginal_band_endpoints_are_lower_limits():
    from powertoflops.bench.bands import useful_marginal_band

    r_a, r_b, E0 = 2.2e-12, 3.1e-12, 900.0
    rng = np.random.default_rng(1)
    recs = [_qexp7("qblock_2048x8", d,
                   E0 + r_a * d * _OPS_2048x8 + rng.normal(0, 3.0), _OPS_2048x8)
            for d in (0, 10, 20, 30, 40)]
    recs += [_qexp7("qblock_512x8", d,
                    E0 + r_b * d * _OPS_512x8 + rng.normal(0, 3.0), _OPS_512x8)
             for d in (0, 10, 20, 30, 40)]
    ub = useful_marginal_band(recs)
    assert ub.r_lo < ub.r_lo_point           # lower limit on the floor (rung 4)
    assert ub.r_hi < ub.r_hi_point           # lower limit on the top (rung 5)


def test_useful_marginal_band_hi_is_lcb_of_max_slope_arm():
    from powertoflops.bench.bands import marginal_covert_cost, useful_marginal_band

    r_a, r_b, E0 = 2.2e-12, 3.1e-12, 900.0
    rng = np.random.default_rng(2)
    recs = [_qexp7("qblock_2048x8", d,
                   E0 + r_a * d * _OPS_2048x8 + rng.normal(0, 3.0), _OPS_2048x8)
            for d in (0, 10, 20, 30, 40)]
    recs += [_qexp7("qblock_512x8", d,
                    E0 + r_b * d * _OPS_512x8 + rng.normal(0, 3.0), _OPS_512x8)
             for d in (0, 10, 20, 30, 40)]
    ub = useful_marginal_band(recs)
    dearest = marginal_covert_cost(recs, "qblock_512x8")   # largest slope arm
    assert ub.r_hi == dearest.r_marg_lcb95


def test_useful_marginal_band_refuses_no_qblock_arms():
    import pytest

    from powertoflops.bench.bands import useful_marginal_band

    # generic int8 Exp-7 rows carry no "qblock_" arm — refuse to price a band
    with pytest.raises(ValueError, match="qblock covert arms"):
        useful_marginal_band([_exp7(d, 900.0 + d) for d in (0, 20, 40, 80)])


def _qrec(M, N, r, i=0, **over):
    """One validated-qblock record (experiment 6, trained-weight dist tag)."""
    return _rec(experiment=6, M=M, N=N, K=4096, dtype="int8",
                operand_dist_a="qblock_pythia", operand_dist_b="qblock_pythia",
                repeat_index=i, r_j_per_op=r, **over)


def _qcell(M, N, rs):
    return [_qrec(M, N, r, i=i) for i, r in enumerate(rs)]


def test_useful_band_lcb_endpoints():
    # Referee B5: the band endpoints are one-sided 95% lower confidence limits
    # on the per-cell mean r (Student-t, exp7 convention), not raw extrema.
    # r_lo = min over cells of the cell LCB; r_hi = the LCB of the cell with
    # the LARGEST mean (rung 5 prices the covert denominator, so an
    # overstated top would understate beta* — both endpoints must be lower
    # limits). The point estimates ride along for disclosure.
    from scipy import stats

    from powertoflops.bench.bands import useful_band

    cheap = [1.0e-12, 2.0e-12, 3.0e-12, 4.0e-12, 5.0e-12]
    dear = [r + 2.0e-12 for r in cheap]
    ub = useful_band(_qcell(512, 8, cheap) + _qcell(2048, 8, dear))

    mean_c = float(np.mean(cheap))
    sd_c = float(np.std(cheap, ddof=1))
    t95 = float(stats.t.ppf(0.95, len(cheap) - 1))
    lcb_c = mean_c - t95 * sd_c / np.sqrt(len(cheap))
    assert ub.r_lo == pytest.approx(lcb_c, rel=1e-12)
    assert ub.r_hi == pytest.approx(lcb_c + 2.0e-12, rel=1e-9)
    assert ub.r_lo_point == pytest.approx(mean_c, rel=1e-12)
    assert ub.r_hi_point == pytest.approx(mean_c + 2.0e-12, rel=1e-9)
    assert ub.r_lo < ub.r_lo_point and ub.r_hi < ub.r_hi_point
    assert ub.alpha == 0.05
    assert ub.n_cells == 2
    assert set(ub.per_cell) == {"512x8x4096", "2048x8x4096"}
    assert ub.per_cell["512x8x4096"]["n"] == 5
    assert isinstance(ub.strata, dict)


def test_useful_band_sigma_floor_at_nvml_resolution():
    # Near-identical repeats must not collapse the LCB onto the mean: the
    # per-cell sd is floored at 0.5% of the mean (NVML power repeatability,
    # same convention as the matched-pair sigma floor).
    from scipy import stats

    from powertoflops.bench.bands import useful_band

    ub = useful_band(_qcell(512, 8, [2.0e-12] * 5))
    t95 = float(stats.t.ppf(0.95, 4))
    expected = 2.0e-12 - t95 * (0.005 * 2.0e-12) / np.sqrt(5)
    assert ub.r_lo == pytest.approx(expected, rel=1e-12)
    assert ub.r_lo < 2.0e-12


def test_useful_band_requires_five_untrimmed_repeats_per_cell():
    from powertoflops.bench.bands import useful_band

    with pytest.raises(ValueError, match="repeats"):
        useful_band(_qcell(512, 8, [1e-12, 2e-12, 3e-12, 4e-12]))


def test_useful_band_refuses_stale_random_weight_records():
    # Records from the retired random-weight benchmark ("qblock_typical")
    # must fail loudly — they can never re-enter a band (referee B5).
    from powertoflops.bench.bands import useful_band

    stale = [_qrec(512, 8, r, i=i) for i, r in
             enumerate([1e-12, 2e-12, 3e-12, 4e-12, 5e-12])]
    stale = [dataclasses.replace(r, operand_dist_a="qblock_typical")
             for r in stale]
    with pytest.raises(ValueError, match="qblock_pythia"):
        useful_band(stale)


def test_useful_band_lcb_never_exceeds_point():
    # Monotonicity of the convention: using the LCB never widens the floor
    # relative to the point estimate, whatever the spread.
    from powertoflops.bench.bands import useful_band

    for spread in (0.0, 0.01, 0.2):
        rs = [2.0e-12 * (1 + spread * (i - 2)) for i in range(5)]
        ub = useful_band(_qcell(512, 8, rs))
        assert ub.r_lo <= ub.r_lo_point
        assert ub.r_hi <= ub.r_hi_point


# ---- conditional factor widths (paper eq:widthfactor) -----------------------


def _factorial(w_kappa=10.0, w_alpha=2.0, w_C=1.05, base=1.0e-12, clocks=None):
    """A clean 3-factor factorial with planted per-factor widths.

    r = base * kappa * alpha * C_eff exactly, so factor_widths must recover the
    planted widths and their product must equal the pooled span.
    """
    kap = {"int8": 1.0, "fp16": w_kappa ** 0.5, "fp32": w_kappa}
    alp = {"zeros": 1.0, "dense_random": w_alpha}
    ceff = {"resident": 1.0, "streamed": w_C}
    out = []
    for d, k in kap.items():
        for o, a in alp.items():
            for loc, c in ceff.items():
                clk = 1410 if clocks is None else clocks[o]
                out.append(_rec(dtype=d, operand_dist_a=o, locality=loc,
                                sm_clock_mhz=clk, sm_clock_p50_mhz=float(clk),
                                r_j_per_op=base * k * a * c))
    return out


def test_factor_widths_recovers_planted_widths():
    from powertoflops.bench.bands import factor_widths

    got = factor_widths(_factorial(w_kappa=10.0, w_alpha=2.0, w_C=1.05))
    assert got["kappa"]["w"] == pytest.approx(10.0)
    assert got["alpha"]["w"] == pytest.approx(2.0)
    assert got["C_eff"]["w"] == pytest.approx(1.05)


def test_factorisation_is_exact_on_a_separable_factorial():
    # If r factorises exactly, the product of conditional widths IS the pooled
    # span -- the identity paper eq:widthfactor asserts. Any residual on real
    # data is factor interaction, not a defect of the estimator.
    from powertoflops.bench.bands import factorisation_check

    got = factorisation_check(_factorial(w_kappa=8.0, w_alpha=2.0, w_C=1.04),
                              fmt="fp16")
    assert got["rel_err"] == pytest.approx(0.0, abs=1e-12)
    # And pinning kappa leaves exactly w_alpha * w_C.
    assert got["pinned_pred"] == pytest.approx(2.0 * 1.04)
    assert got["pinned_rel_err"] == pytest.approx(0.0, abs=1e-12)


def test_factor_widths_ignores_incomplete_groups():
    # A group missing a level on the axis is not a like-for-like comparison and
    # must be skipped, not averaged in. Dropping one operand cell from a single
    # (dtype, locality) group must leave the width untouched.
    from powertoflops.bench.bands import factor_widths

    full = _factorial(w_alpha=2.0)
    partial = [r for r in full
               if not (r.dtype == "fp32" and r.locality == "streamed"
                       and r.operand_dist_a == "zeros")]
    assert factor_widths(partial)["alpha"]["w"] == pytest.approx(
        factor_widths(full)["alpha"]["w"])
    assert factor_widths(partial)["alpha"]["n_groups"] == \
        factor_widths(full)["alpha"]["n_groups"] - 1


def test_clock_balance_flags_a_confounded_axis():
    # The nuisance check: the achieved clock is NOT in the conditioning key
    # (locking is privilege-denied), so factor_widths reports whether the axis's
    # levels ran at a common clock. Exp 1's operand families do (balance 1.000);
    # Exp 2's slide 1410 -> 1324 MHz against operand cost, and that is why its
    # w0 is descriptive only.
    from powertoflops.bench.bands import factor_widths

    balanced = factor_widths(_factorial())
    assert balanced["alpha"]["clock_balance"] == pytest.approx(1.0)
    confounded = factor_widths(
        _factorial(clocks={"zeros": 1410, "dense_random": 1320}))
    assert confounded["alpha"]["clock_balance"] == pytest.approx(1410 / 1320)


def test_committed_exp1_factor_widths_pin_the_manuscript():
    # Regression pin on the sweep that produced results/bench/measured_bands.json
    # (identified by its exchange_band matching the artifact). These are the
    # numbers Sec. "How big is beta?" and tab:inputs quote, and the two checks
    # that license reading tab:ladder as one factor pinned per rung.
    import pathlib

    from powertoflops.bench.bands import factor_widths, factorisation_check
    from powertoflops.bench.records import read_jsonl

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "results/bench/exp1_exchange_e2958cc757b9.jsonl")
    if not src.exists():                       # artifact not fetched in this checkout
        pytest.skip(f"{src.name} not present")
    recs = read_jsonl(src)

    w = factor_widths(recs)
    assert w["kappa"]["w"] == pytest.approx(13.53, abs=0.05)
    assert w["alpha"]["w"] == pytest.approx(1.88, abs=0.02)
    assert w["C_eff"]["w"] == pytest.approx(1.04, abs=0.01)
    # The operand and locality axes ran at a common clock; precision to 3%.
    assert w["alpha"]["clock_balance"] == pytest.approx(1.0, abs=1e-9)
    assert w["C_eff"]["clock_balance"] == pytest.approx(1.0, abs=1e-9)
    assert w["kappa"]["clock_balance"] < 1.03

    c = factorisation_check(recs, fmt="fp16")
    assert c["product"] == pytest.approx(26.5, abs=0.2)     # vs pooled 24.8
    assert abs(c["rel_err"]) < 0.10                          # ~7% interaction
    assert c["pinned_pred"] == pytest.approx(1.96, abs=0.02)  # w_alpha * w_C
    assert c["pinned_obs"] == pytest.approx(2.00, abs=0.02)   # fp16's own band
    assert abs(c["pinned_rel_err"]) < 0.04                    # ~2%
