"""Matched-energy pair (R2.6 / referee B3): pure planning, tuning, validation.

The GPU harness (scripts/run_matched_pair.py) wraps these with real measured
windows; here a fake linear energy model stands in for measure_b, so the
solver's convergence contract is pinned without a GPU. The B3 layer adds the
fixed-T, same-C_dec construction: plan_pair (pre-flight feasibility),
acceptance_threshold/acceptance_verdicts (the sensor's one-sided no-alarm
decision), validate_pair_payload (infeasible artifacts are a hard analysis
failure, never a publishable witness), and summarize_pair (linkage of the
witness parameters to the theorem's corner assumptions).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from powertoflops.bench.pair import (
    PairCalibration,
    PairInfeasibleError,
    PairValidationError,
    acceptance_threshold,
    acceptance_verdicts,
    capacity_frac_used,
    plan_pair,
    solve_covert_iters,
    summarize_pair,
    tune_covert_iters,
    validate_pair_payload,
)
from powertoflops.measured import MeasuredBands


def _linear_meter(e_declared: float, e_per_covert: float, noise: float = 0.0):
    """measure_b(n_covert) -> E_B for a perfectly affine (optionally noisy) rig."""
    rng = np.random.default_rng(0)

    def measure_b(n_covert: int) -> float:
        eps = rng.normal(0.0, noise) if noise > 0 else 0.0
        return e_declared + n_covert * e_per_covert + eps

    return measure_b


def test_solver_initial_guess():
    # E_A = 900 J, declared share of run B = 300 J, covert 2 J/iter -> 300 iters
    assert solve_covert_iters(900.0, 300.0, 2.0) == 300
    assert solve_covert_iters(100.0, 300.0, 2.0) == 0     # never negative


def test_tune_converges_first_round_on_affine_rig():
    measure = _linear_meter(e_declared=300.0, e_per_covert=2.0)
    res = tune_covert_iters(measure, e_a=900.0, n0=300, e_per_covert=2.0,
                            tol_frac=0.01, max_rounds=8)
    assert res.converged
    assert res.n_covert == 300
    assert abs(res.e_b - 900.0) / 900.0 < 0.01
    assert len(res.history) == 1


def test_tune_corrects_a_biased_calibration():
    # Calibration said 2.0 J/iter but the real rig burns 2.5 J/iter: the loop
    # must walk to the correct count within the round budget.
    measure = _linear_meter(e_declared=300.0, e_per_covert=2.5)
    res = tune_covert_iters(measure, e_a=900.0, n0=300, e_per_covert=2.0,
                            tol_frac=0.01, max_rounds=8)
    assert res.converged
    assert abs(res.e_b - 900.0) / 900.0 < 0.01
    # exact balance is 240 (600 J covert / 2.5 J per iter); convergence may
    # stop a step early once inside the 1% tolerance
    assert abs(res.n_covert - 240) <= 2


def test_tune_reports_non_convergence():
    # A rig whose energy ignores n entirely can never match: converged=False,
    # history exhausted, best effort returned.
    res = tune_covert_iters(lambda n: 500.0, e_a=900.0, n0=100,
                            e_per_covert=2.0, tol_frac=0.01, max_rounds=3)
    assert not res.converged
    assert len(res.history) == 3
    assert res.e_b == 500.0


def test_tune_never_goes_negative():
    # Overshoot from a huge n0 must clamp at 0 covert iters, not go negative.
    measure = _linear_meter(e_declared=890.0, e_per_covert=2.0)
    res = tune_covert_iters(measure, e_a=900.0, n0=500, e_per_covert=2.0,
                            tol_frac=0.01, max_rounds=8)
    assert res.converged
    assert res.n_covert == 5
    assert all(n >= 0 for n, _ in res.history)


def test_capacity_frac_used():
    # covert ops vs what (1-h) of the format peak allows over the window
    frac = capacity_frac_used(covert_ops=1.0e15, h=0.5, wall_s=10.0,
                              peak_ops_per_s=624.0e12)
    assert np.isclose(frac, 1.0e15 / (0.5 * 624.0e12 * 10.0))
    assert frac < 1.0   # this example is feasible


# ---- B3 fixed-T, same-C_dec construction -------------------------------------
#
# Round-number rig: T = 10 s window, declared iters cost 4 J (dear operands) /
# 2 J (cheap operands) marginal, covert iters 1 J; declared peak 1e13 op/s,
# covert peak 2e13 op/s, 1e10 nominal ops per iteration on both sides. At
# h = 0.4: n_dec = 4000, energy gap = 8000 J -> n_cov0 = 8000, run-A busy
# 4.0 s, run-B busy 3.2 + 4.0 = 7.2 s, h_max = 0.9*10/18 = 0.5,
# capacity_frac = 8e13/(0.6 * 2e13 * 10) = 2/3.

PEAK_DEC = 1.0e13
PEAK_COV = 2.0e13


def _calib(**over) -> PairCalibration:
    kw = dict(T_s=10.0, e_idle_j=800.0,
              m_dear_j=4.0, m_cheap_j=2.0, m_cov_j=1.0,
              t_dear_s=1.0e-3, t_cheap_s=8.0e-4, t_cov_s=5.0e-4,
              ops_dec=1.0e10, ops_cov=1.0e10)
    kw.update(over)
    return PairCalibration(**kw)


def _plan(h: float, calib: PairCalibration | None = None, **over):
    return plan_pair(calib or _calib(), h,
                     peak_dec_ops=PEAK_DEC, peak_cov_ops=PEAK_COV, **over)


def test_plan_pair_worked_example():
    plan = _plan(0.4)
    assert plan.n_dec == 4000
    assert plan.n_cov0 == 8000
    assert np.isclose(plan.busy_a_s, 4.0)
    assert np.isclose(plan.busy_b_s, 7.2)
    assert np.isclose(plan.h_max, 0.5)
    assert np.isclose(plan.capacity_frac, 2.0 / 3.0)
    assert np.isclose(plan.e_gap_j, 8000.0)


def test_plan_pair_covert_dose_scales_with_h():
    assert _plan(0.2).n_cov0 * 2 == _plan(0.4).n_cov0


def test_plan_pair_rejects_h_beyond_safety_margin():
    # h_max = 0.5, safety 0.95 -> anything above 0.475 must hard-fail
    with pytest.raises(PairInfeasibleError):
        _plan(0.49)
    with pytest.raises(PairInfeasibleError):
        _plan(0.5)


def test_plan_pair_rejects_empty_energy_gap():
    # cheap operands as dear as the dear ones: no freed energy, no witness
    with pytest.raises(PairInfeasibleError):
        _plan(0.4, _calib(m_cheap_j=4.0))


def test_plan_pair_rejects_capacity_overrun():
    # a tiny covert peak makes the tuned dose exceed (1-h) of covert capacity
    with pytest.raises(PairInfeasibleError):
        plan_pair(_calib(), 0.4, peak_dec_ops=PEAK_DEC, peak_cov_ops=1.0e12)


def test_acceptance_threshold_is_mean_plus_z_sigma():
    # mean 101.5, sample std (ddof=1) 1.5 -> mean + 3*sigma = 106.0
    assert np.isclose(acceptance_threshold([100.0, 103.0, 101.5]), 106.0)
    assert np.isclose(acceptance_threshold([100.0, 103.0, 101.5], z=1.0),
                      103.0)
    with pytest.raises(ValueError):
        acceptance_threshold([288.0])          # a variance needs >= 2 repeats


def test_acceptance_verdicts_one_sided_upper():
    # accepted (no alarm) iff E_obs <= threshold; equality accepted
    assert acceptance_verdicts([102.9, 103.0, 103.1], 103.0) == \
        [True, True, False]


# ---- payload validation: infeasibility is a hard analysis failure ------------


def _valid_payload() -> dict:
    e_a = [10000.0, 10010.0, 10005.0]
    e_b = [9990.0, 10002.0, 9998.0]
    return {
        "pair_id": "deadbeef0000",
        "window_s": 10.0,
        "h": 0.4,
        "h_max": 0.5,
        "declared": {"dtype": "fp16", "dist_dear": "adversarial_toggle",
                     "dist_cheap": "zeros", "shape": [2048, 2048, 2048],
                     "iters_a": 4000, "iters_b": 4000,
                     "ops_nominal_per_iter": 1.0e10},
        "covert": {"dtype": "int8", "dist": "zeros",
                   "shape": [2048, 2048, 2048], "iters": 8000,
                   "ops_nominal_per_iter": 1.0e10, "m_j_per_iter": 1.0},
        "calibration": {"T_s": 10.0, "e_idle_j": 800.0,
                        "m_dear_j": 4.0, "m_cheap_j": 2.0, "m_cov_j": 1.0,
                        "t_dear_s": 1.0e-3, "t_cheap_s": 8.0e-4,
                        "t_cov_s": 5.0e-4,
                        "ops_dec": 1.0e10, "ops_cov": 1.0e10},
        "repeats_a": [{"energy_j": e, "duration_s": 10.0 + 0.05 * i,
                       "pad_s": 5.9} for i, e in enumerate(e_a)],
        "repeats_b": [{"energy_j": e, "duration_s": 10.0, "pad_s": 2.7}
                      for e in e_b],
        "tuning": {"n_covert": 8000, "converged": True,
                   "history": [[8000, 10002.0]]},
        "acceptance": {"rule": "alarm iff E_obs > max(honest repeats)",
                       "threshold_j": 10010.0,
                       "verdicts": [True, True, True],
                       "all_accepted": True},
        "capacity_frac_used": 2.0 / 3.0,
        "capacity_feasible": True,
        "tol_frac": 0.01,
    }


def test_validate_pair_payload_accepts_the_valid_construction():
    validate_pair_payload(_valid_payload())   # must not raise


def test_validate_rejects_cdec_mismatch():
    p = _valid_payload()
    p["declared"]["iters_b"] = 2000
    with pytest.raises(PairValidationError, match="C_dec"):
        validate_pair_payload(p)


def test_validate_rejects_pad_underflow():
    p = _valid_payload()
    p["repeats_b"][1]["pad_s"] = 0.0
    with pytest.raises(PairValidationError, match="pad"):
        validate_pair_payload(p)


def test_validate_rejects_window_overrun():
    p = _valid_payload()
    p["repeats_b"][0]["duration_s"] = 10.5   # > 2% off the 10 s window
    with pytest.raises(PairValidationError, match="window"):
        validate_pair_payload(p)


def test_validate_rejects_capacity_overrun():
    p = _valid_payload()
    p["capacity_frac_used"] = 1.02
    with pytest.raises(PairValidationError, match="capacity"):
        validate_pair_payload(p)


def test_validate_rejects_non_convergence():
    p = _valid_payload()
    p["tuning"]["converged"] = False
    with pytest.raises(PairValidationError, match="converg"):
        validate_pair_payload(p)


def test_validate_rejects_alarmed_repeats():
    # acceptance is recomputed from the repeats, not trusted from the file
    p = _valid_payload()
    p["repeats_b"][2]["energy_j"] = 10500.0
    with pytest.raises(PairValidationError, match="alarm"):
        validate_pair_payload(p)


def test_validate_rejects_missing_repeats():
    p = _valid_payload()
    p["repeats_a"] = []
    with pytest.raises(PairValidationError, match="repeat"):
        validate_pair_payload(p)


def test_validate_rejects_the_retired_first_round_artifact():
    # The c922a1f pair (523 vs 262 declared iters, +4.5% wall) must hard-fail:
    # the referee's checklist makes infeasible artifacts unpublishable.
    path = Path(__file__).resolve().parents[1] / "results" / "bench" / \
        "matched_pair_c922a1f23b27.json"
    payload = json.loads(path.read_text())
    with pytest.raises(PairValidationError):
        validate_pair_payload(payload)


# ---- linkage of the witness to the theorem corner -----------------------------


def _fake_bands() -> MeasuredBands:
    return MeasuredBands(
        r_lo=1.0e-12, r_hi=2.0e-12, r_lo_cov=5.0e-13, w0=2.0,
        P0_lo=60.0, P0_hi=100.0, sigma_dec=6.0e-14,
        format_bands={"fp16": {"r_lo": 1.0e-12, "r_hi": 2.0e-12,
                               "n_cells": 12}},
        r_marg_cov={"int8": {"r_marg": 5.5e-13, "r_marg_lcb95": 5.0e-13}},
    )


def test_summarize_pair_links_witness_to_corner():
    s = summarize_pair(_valid_payload(), _fake_bands())
    # witnessed covert compute in machines of declared (fp16) capacity:
    # 8000 iters * 1e10 ops / (312e12 op/s * 10 s)
    assert np.isclose(s["beta_witness"], 8.0e13 / 3.12e15)
    # S(h=0.4) = 0.4*(2-1)/0.5 + 40/(5e-13*312e12) = 0.8 + 0.2564...
    assert np.isclose(s["S_h"], 0.8 + 40.0 / (5.0e-13 * 312.0e12))
    # clip (1-h)*kappa = 0.6*2 = 1.2 > S(h): energy binds at the witness h
    assert np.isclose(s["beta_phys_h"], s["S_h"])
    assert np.isclose(s["kappa"], 2.0)
    # achieved marginal rates reported against the band edges
    assert np.isclose(s["r_marg_dec_dear"], 4.0e-10)
    assert np.isclose(s["r_marg_dec_cheap"], 2.0e-10)
    assert np.isclose(s["r_marg_cov_achieved"], 1.0e-10)
    assert s["beta_witness"] < s["beta_phys_h"]


# ---- GPU smoke (inert on login nodes) -----------------------------------------


@pytest.mark.gpu
def test_run_matched_pair_smoke(tmp_path):
    import dataclasses

    torch = pytest.importorskip("torch")
    pytest.importorskip("pynvml")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from powertoflops.config import DEFAULT
    from powertoflops.bench.pair import run_matched_pair

    # One low-h point, tiny windows, 2 repeats — a couple of minutes. The
    # 3% tolerance derisks the NVML 10 Hz zero-order hold on 3 s windows;
    # the real capture keeps 1% on 10 s windows.
    cfg = dataclasses.replace(
        DEFAULT.bench,
        pair_window_s=3.0, pair_repeats=2, pair_calib_repeats=2,
        pair_h_values=(0.10,), pair_h_auto_edge=False,
        pair_tol_frac=0.03, pair_max_tune=6,
    )
    paths = run_matched_pair(tmp_path, cfg)
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    validate_pair_payload(payload)          # the artifact must be a witness
    assert payload["declared"]["iters_a"] == payload["declared"]["iters_b"]
    assert payload["covert"]["iters"] > 0
    assert len(payload["repeats_a"]) == 2 and len(payload["repeats_b"]) == 2
    for rep in payload["repeats_a"] + payload["repeats_b"]:
        assert rep["pad_s"] > 0.0
        assert abs(rep["duration_s"] - cfg.pair_window_s) < 0.2
    assert payload["acceptance"]["all_accepted"] is True
    assert (tmp_path / f"matched_pair_{payload['pair_id']}_manifest.json").exists()


def test_tuning_target_undershoots_by_the_tolerance():
    # The cheater aims at the BOTTOM of the honest envelope: tuning centered
    # on a single honest draw converges from above half the time and then
    # alarms against max(honest repeats) — the smoke-test failure of job
    # 14632514. Target (1 - tol) * E_A_ref keeps E_B inside the acceptance
    # region while staying energy-degenerate within tolerance.
    from powertoflops.bench.pair import tuning_target
    assert np.isclose(tuning_target(1000.0, 0.01), 990.0)
    assert np.isclose(tuning_target(288.0, 0.03), 279.36)


def test_validate_rejects_a_wide_energy_mismatch():
    # acceptance alone is not enough: the pair must remain energy-degenerate
    # (mean |dE|/E within 3x the tuning tolerance), else it is just a cheaper
    # accepted run, not a matched-energy witness.
    p = _valid_payload()
    for r in p["repeats_b"]:
        r["energy_j"] = 9000.0            # accepted (below threshold), but -10%
    with pytest.raises(PairValidationError, match="degenera"):
        validate_pair_payload(p)


def test_tune_uses_secant_when_calibration_slope_is_wrong():
    # Rig matches at n=300 (E = 400 + 2n), but calibration claims 1 J/iter.
    # Fixed-slope Newton oscillates 600<->0 forever; a secant update recovers
    # the true slope from the first two points and converges. This is the
    # h=0.100 non-convergence of Slurm 14633255 (calib 0.0087 vs actual
    # ~0.011 J/iter in the interleaved context).
    def measure_b(n):
        return 400.0 + 2.0 * n

    res = tune_covert_iters(measure_b, e_a=1000.0, n0=600, e_per_covert=1.0,
                            tol_frac=0.01, max_rounds=5)
    assert res.converged
    assert res.n_covert == 300


def test_validate_rejects_zero_covert_work():
    # A pair tuned to n_cov=0 is just a cheaper declared run, not a witness:
    # the B3 construction requires an explicit nonzero covert workload
    # (Slurm 14634845 drove the dose to 0 in the noise-fragile h=0.05 smoke).
    p = _valid_payload()
    p["covert"]["iters"] = 0
    with pytest.raises(PairValidationError, match="covert"):
        validate_pair_payload(p)


def test_acceptance_threshold_floors_sigma_at_meter_resolution():
    # Two near-identical honest repeats give a tiny sample sigma (0.07);
    # without the floor the 3-sigma threshold sits 0.2 J above the mean and a
    # perfectly matched run trips it (Slurm 14638999). The 0.5%-of-mean floor
    # keeps the threshold at the meter's resolution: mean 1000.05 + 3*5.0.
    thr = acceptance_threshold([1000.0, 1000.1])
    assert np.isclose(thr, 1000.05 + 3.0 * 0.005 * 1000.05)
    # a genuinely matched run (0.7% high) is then accepted
    assert acceptance_verdicts([1007.0], thr) == [True]
