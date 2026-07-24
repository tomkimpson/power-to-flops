"""Band-artifact assembly: provenance, the mixed-UUID guard, capture manifests.

All pure (no GPU / NVML): synthetic RunRecords through powertoflops.bench.analysis, and
the capture-side manifest builder in powertoflops.bench.records. The review found the
committed measured_bands.json silently pooled two different GPUs (Exp 1 vs
Exp 2/3) picked newest-by-mtime with no record of which files went in (C5);
these tests pin the replacement behaviour.
"""

from __future__ import annotations

import pytest

from powertoflops.bench.analysis import MixedDeviceError, band_payload, gather_uuids
from powertoflops.bench.records import RunRecord, build_manifest


def _rec(**over) -> RunRecord:
    base = dict(
        experiment=1, sweep_id="s1", repeat_index=0,
        M=4, N=4, K=4, iters=1, ops_nominal=128,
        dtype="fp16", locality="dram",
        operand_dist_a="dense_random", operand_dist_b="dense_random",
        util_frac=None,
        energy_j=1.0, duration_s=1.0, mean_power_w=300.0,
        meter_method="nvml_energy_counter", r_j_per_op=1.0e-12,
        gpu_name="A100", gpu_uuid="GPU-aaa", driver_version="x",
        cuda_version="y", torch_version="z", persistence_mode=True,
        sm_clock_mhz=1410, mem_clock_mhz=1215, power_limit_w=400.0,
        temperature_c=50.0, throttled=False, throttle_reasons=0,
        clocks_locked=False, timestamp_utc="2026-07-21T00:00:00+00:00",
    )
    base.update(over)
    return RunRecord(**base)


def _full_experiments(uuid="GPU-aaa", sweep="s1"):
    """A minimal but band-extractable exp1/exp2/exp3 triple on one device."""
    exp1 = [_rec(gpu_uuid=uuid, sweep_id=sweep, dtype="fp16", r_j_per_op=1e-12),
            _rec(gpu_uuid=uuid, sweep_id=sweep, dtype="fp32", r_j_per_op=10e-12)]
    exp2 = [_rec(gpu_uuid=uuid, sweep_id=sweep, experiment=2,
                 operand_dist_a="zeros", r_j_per_op=1e-12),
            _rec(gpu_uuid=uuid, sweep_id=sweep, experiment=2,
                 operand_dist_a="adversarial_toggle", r_j_per_op=2e-12)]
    exp3 = [_rec(gpu_uuid=uuid, sweep_id=sweep, experiment=3, util_frac=0.0,
                 ops_nominal=0, mean_power_w=78.0, r_j_per_op=float("nan")),
            _rec(gpu_uuid=uuid, sweep_id=sweep, experiment=3, util_frac=1.0,
                 ops_nominal=int(3e11), mean_power_w=380.0)]
    return exp1, exp2, exp3


def test_gather_uuids_unions_and_sorts():
    a = [_rec(gpu_uuid="GPU-bbb"), _rec(gpu_uuid="GPU-aaa")]
    b = [_rec(gpu_uuid="GPU-aaa")]
    assert gather_uuids(a, b) == ["GPU-aaa", "GPU-bbb"]


def test_mixed_uuids_refused_by_default():
    # The committed artifact's exact failure mode: Exp 1 measured on one card,
    # Exp 2/3 on another, silently pooled into one band.
    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    exp1_other = [_rec(gpu_uuid="GPU-bbb", dtype="fp16", r_j_per_op=1e-12),
                  _rec(gpu_uuid="GPU-bbb", dtype="fp32", r_j_per_op=10e-12)]
    with pytest.raises(MixedDeviceError) as ei:
        band_payload(exp1_other, exp2, exp3)
    # the error must name the devices so the operator can pick inputs
    assert "GPU-aaa" in str(ei.value) and "GPU-bbb" in str(ei.value)


def test_single_device_payload_carries_provenance():
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(
        exp1, exp2, exp3,
        inputs={"exp1": "results/bench/e1.jsonl", "exp2": "e2.jsonl",
                "exp3": "e3.jsonl"},
    )
    assert payload["gpu_uuids"] == ["GPU-aaa"]
    assert payload["multi_device"] is False
    assert payload["inputs"]["exp1"]["path"] == "results/bench/e1.jsonl"
    assert payload["inputs"]["exp2"]["sweep_ids"] == ["s1"]
    assert payload["inputs"]["exp3"]["n_records"] == 2
    assert "analysis_git_commit" in payload
    # the bands themselves are still there
    assert payload["r_lo"] == 1e-12 and payload["r_hi"] == 10e-12


def test_payload_carries_compute_bound_subfit():
    # R3.3 sensitivity check: idle anchors + compute-bound cubes subfit is
    # reported beside the pooled marginal fit.
    exp1, exp2, _ = _full_experiments()
    exp3 = [_rec(experiment=3, util_frac=0.0, M=0, N=0, K=0, ops_nominal=0,
                 mean_power_w=78.0, r_j_per_op=float("nan"))]
    for j, m in enumerate((2048, 4096, 8192)):
        F = (j + 1) * 1.0e11
        exp3.append(_rec(experiment=3, util_frac=1.0, M=m, N=m, K=m,
                         ops_nominal=int(F), mean_power_w=2.0e-9 * F + 78.0))
    payload = band_payload(exp1, exp2, exp3)
    assert payload["r_marginal_cubes"] == pytest.approx(2.0e-9, rel=1e-6)
    assert payload["r_marginal_cubes_r2"] > 0.999
    assert payload["r_marginal_cubes_n_cells"] == 4


def test_payload_survives_degenerate_subfit():
    # A ladder with no compute-bound cubes cannot fit the subfit; the payload
    # must still build, just without the subfit keys.
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(exp1, exp2, exp3)
    assert "r_marginal_cubes" not in payload
    assert "r_marginal" in payload


def test_allow_multi_device_reports_per_device_and_spread():
    e1a, e2a, e3a = _full_experiments(uuid="GPU-aaa", sweep="sa")
    e1b, e2b, e3b = _full_experiments(uuid="GPU-bbb", sweep="sb")
    # device b runs slightly dearer so the spread is non-degenerate
    e1b = [_rec(gpu_uuid="GPU-bbb", sweep_id="sb", dtype="fp16",
                r_j_per_op=1.2e-12),
           _rec(gpu_uuid="GPU-bbb", sweep_id="sb", dtype="fp32",
                r_j_per_op=12e-12)]
    payload = band_payload(e1a + e1b, e2a + e2b, e3a + e3b,
                           allow_multi_device=True)
    assert payload["multi_device"] is True
    assert set(payload["per_device"]) == {"GPU-aaa", "GPU-bbb"}
    assert payload["per_device"]["GPU-aaa"]["r_lo"] == 1e-12
    assert payload["per_device"]["GPU-bbb"]["r_lo"] == 1.2e-12
    # device-to-device spread: [min, max] across devices per band quantity
    assert payload["device_spread"]["r_lo"] == [1e-12, 1.2e-12]


def test_allow_multi_device_tolerates_partial_coverage():
    # The committed mixed case: device a has ONLY exp1, device b only exp2/3.
    # Per-device entries carry what that device measured; no crash.
    e1a, _, _ = _full_experiments(uuid="GPU-aaa", sweep="sa")
    _, e2b, e3b = _full_experiments(uuid="GPU-bbb", sweep="sb")
    payload = band_payload(e1a, e2b, e3b, allow_multi_device=True)
    assert payload["per_device"]["GPU-aaa"]["r_lo"] == 1e-12
    assert "P0_lo" not in payload["per_device"]["GPU-aaa"]
    assert payload["per_device"]["GPU-bbb"]["w0"] == 2.0
    # nothing is measurable on >= 2 devices here, so no spread entries
    assert payload["device_spread"] == {}


def test_payload_carries_clock_strata():
    exp1, exp2, exp3 = _full_experiments()
    exp1 = [_rec(gpu_uuid="GPU-aaa", dtype="fp16", r_j_per_op=1e-12,
                 sm_clock_p50_mhz=1395.0),
            _rec(gpu_uuid="GPU-aaa", dtype="fp32", r_j_per_op=10e-12,
                 sm_clock_p50_mhz=1125.0)]
    payload = band_payload(exp1, exp2, exp3)
    strata = payload["clock_strata"]
    assert set(strata["exp1"]) == {1395, 1125}
    assert strata["exp1"][1395]["r_lo"] == 1e-12
    assert "exp2" in strata


def _qblock_cell(M, N, rs, uuid="GPU-qqq"):
    """>= 5 repeats of one validated-qblock cell (LCB needs the repeats)."""
    return [
        _rec(experiment=6, gpu_uuid=uuid, dtype="int8", M=M, N=N, K=4096,
             operand_dist_a="qblock_pythia", operand_dist_b="qblock_pythia",
             repeat_index=i, r_j_per_op=r)
        for i, r in enumerate(rs)
    ]


def test_qblock_band_in_payload_without_device_pooling():
    # qblock may run in a separate allocation (different GPU): it must not trip
    # the exp1-3 single-device guard, and its uuid is recorded separately.
    # Since the B5 rebuild the endpoints are one-sided LCBs and ride with a
    # versioned useful_band block (pre-B5 artifacts lack it and fail the
    # ladder guard).
    from powertoflops.bench.analysis import band_payload

    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    cheap = [1.3e-12, 1.4e-12, 1.4e-12, 1.5e-12, 1.4e-12]
    dear = [2.0e-12, 2.1e-12, 1.9e-12, 2.0e-12, 2.0e-12]
    qblock = _qblock_cell(2048, 8, cheap) + _qblock_cell(512, 8, dear)
    payload = band_payload(exp1, exp2, exp3, qblock=qblock,
                           inputs={"qblock": "qblock_x.jsonl"},
                           qblock_provenance={"model": "pythia-6.9b"})
    ub = payload["useful_band"]
    assert ub["version"] == 2
    assert payload["r_useful_lo"] == ub["r_lo"]
    assert payload["r_useful_hi"] == ub["r_hi"]
    # LCB endpoints sit below the cell means, in the verifier-safe direction
    assert ub["r_lo"] < sum(cheap) / len(cheap)
    assert ub["r_hi"] < sum(dear) / len(dear)
    assert ub["r_lo_point"] == pytest.approx(sum(cheap) / len(cheap))
    assert ub["weights_provenance"] == {"model": "pythia-6.9b"}
    assert payload["qblock_uuids"] == ["GPU-qqq"]
    assert payload["gpu_uuids"] == ["GPU-aaa"]   # guard scope unchanged
    assert payload["inputs"]["qblock"]["path"] == "qblock_x.jsonl"


def test_qblock_multi_device_takes_min_lcb_and_records_spread():
    # 3-card capture: the headline endpoints are the min over devices of the
    # device LCBs (the safe direction for lower limits); per-device bands and
    # the spread are recorded for the disclosure.
    from powertoflops.bench.analysis import band_payload

    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    qa = (_qblock_cell(512, 8, [2.0e-12] * 5, uuid="GPU-q1")
          + _qblock_cell(2048, 8, [3.0e-12] * 5, uuid="GPU-q1"))
    qb = (_qblock_cell(512, 8, [2.2e-12] * 5, uuid="GPU-q2")
          + _qblock_cell(2048, 8, [3.3e-12] * 5, uuid="GPU-q2"))
    payload = band_payload(exp1, exp2, exp3, qblock=qa + qb)
    ub = payload["useful_band"]
    assert payload["qblock_uuids"] == ["GPU-q1", "GPU-q2"]
    assert set(ub["per_device"]) == {"GPU-q1", "GPU-q2"}
    # device q1 is cheaper on both cells, so the headline is q1's LCBs
    assert ub["r_lo"] == ub["per_device"]["GPU-q1"]["r_lo"]
    assert ub["r_hi"] == ub["per_device"]["GPU-q1"]["r_hi"]
    assert ub["device_spread"]["r_lo"] == [
        ub["per_device"]["GPU-q1"]["r_lo"], ub["per_device"]["GPU-q2"]["r_lo"]]
    # and the payload keys mirror the headline
    assert payload["r_useful_lo"] == ub["r_lo"]


def _exp7_records(uuid="GPU-aaa", sweep="s7"):
    """Minimal fit-able Exp-7 dose sweep on one device."""
    ops = 2 * 2048 ** 3
    return [
        _rec(experiment=7, gpu_uuid=uuid, sweep_id=sweep,
             operand_dist_a="declared_typical", cov_dtype="int8",
             cov_dist="zeros", cov_iters=d, cov_ops_nominal=d * ops,
             energy_j=900.0 + 0.6e-12 * d * ops, duration_s=10.0,
             mean_power_w=(900.0 + 0.6e-12 * d * ops) / 10.0)
        for d in (0, 20, 40, 80)
    ]


def test_payload_carries_format_bands():
    # Referee B1: per-declared-format bands ride beside the pooled span.
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(exp1, exp2, exp3)
    fb = payload["format_bands"]
    assert set(fb) == {"fp16", "fp32"}
    assert fb["fp16"]["r_lo"] == fb["fp16"]["r_hi"] == 1e-12
    assert fb["fp32"]["r_lo"] == fb["fp32"]["r_hi"] == 10e-12
    # pooled span unchanged
    assert payload["r_lo"] == 1e-12 and payload["r_hi"] == 10e-12


def test_exp7_payload_carries_marginal_costs_and_provenance():
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(exp1, exp2, exp3, exp7=_exp7_records(),
                           inputs={"exp7": "exp7_marginal_s7.jsonl"})
    mc = payload["r_marg_cov"]["int8"]
    assert mc["r_marg"] == pytest.approx(0.6e-12, rel=1e-9)
    assert mc["r_marg_lcb95"] <= mc["r_marg"]
    assert payload["inputs"]["exp7"]["path"] == "exp7_marginal_s7.jsonl"
    assert payload["inputs"]["exp7"]["sweep_ids"] == ["s7"]
    assert payload["n_records"]["exp7"] == 4


def test_exp7_joins_the_single_device_guard():
    # Unlike qblock, exp7 feeds the SAME headline as exp1 (powertoflops.joint_model
    # pairs an exp1 declared band with an exp7 covert edge), so a foreign-UUID
    # exp7 must be refused (referee C5).
    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    with pytest.raises(MixedDeviceError) as ei:
        band_payload(exp1, exp2, exp3, exp7=_exp7_records(uuid="GPU-eee"))
    assert "GPU-eee" in str(ei.value)


def test_payload_without_exp7_has_no_marginal_covert_block():
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(exp1, exp2, exp3)
    assert "r_marg_cov" not in payload


# ---- marginal useful band (Exp 8; referee B5, third round) ------------------

_QM_ARMS = (("qblock_512x8", 3.1e-12, 1_680_000_000_000),      # dearer arm
            ("qblock_2048x8", 2.2e-12, 7_150_000_000_000))     # cheaper arm


def _qblock_marg_records(uuid="GPU-qm", arms=_QM_ARMS):
    """A fit-able Exp-8 dose sweep: per-arm qblock dose at a fixed declared burst."""
    recs = []
    for arm, r, ops in arms:
        for d in (0, 10, 20, 30, 40):
            e = 900.0 + r * d * ops
            recs.append(_rec(experiment=8, gpu_uuid=uuid, sweep_id="s8",
                             operand_dist_a="declared_typical", cov_dtype=arm,
                             cov_dist="qblock_pythia", cov_iters=d,
                             cov_ops_nominal=d * ops, energy_j=e,
                             duration_s=10.0, mean_power_w=e / 10.0))
    return recs


def test_qblock_marg_band_in_payload_without_device_pooling():
    # The MARGINAL useful edge is a covert denominator on a possibly separate
    # device: it must not trip the single-device guard (exp1-3 on GPU-aaa here).
    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    payload = band_payload(
        exp1, exp2, exp3, qblock_marg=_qblock_marg_records(),
        inputs={"qblock_marg": "qblock_marginal_s8.jsonl"},
        qblock_marg_provenance={"model": "pythia-6.9b"})
    umb = payload["useful_marginal_band"]
    assert umb["version"] == 1
    assert payload["r_useful_marg_lo"] == umb["r_lo"]
    assert payload["r_useful_marg_hi"] == umb["r_hi"]
    # exact fit -> LCB collapses onto the point slopes: r_hi = dearest arm's LCB
    assert umb["r_hi"] == pytest.approx(3.1e-12, rel=1e-6)
    assert umb["r_lo"] == pytest.approx(2.2e-12, rel=1e-6)
    assert umb["weights_provenance"] == {"model": "pythia-6.9b"}
    assert payload["qblock_marg_uuids"] == ["GPU-qm"]
    assert payload["inputs"]["qblock_marg"]["path"] == "qblock_marginal_s8.jsonl"


def test_qblock_marg_multi_device_takes_min_lcb_and_records_spread():
    exp1, exp2, exp3 = _full_experiments(uuid="GPU-aaa")
    qa = _qblock_marg_records(uuid="GPU-q1")
    qb = _qblock_marg_records(uuid="GPU-q2", arms=(
        ("qblock_512x8", 3.3e-12, 1_680_000_000_000),
        ("qblock_2048x8", 2.4e-12, 7_150_000_000_000)))
    payload = band_payload(exp1, exp2, exp3, qblock_marg=qa + qb)
    umb = payload["useful_marginal_band"]
    assert payload["qblock_marg_uuids"] == ["GPU-q1", "GPU-q2"]
    assert set(umb["per_device"]) == {"GPU-q1", "GPU-q2"}
    # min over devices — the safe direction for a one-sided lower limit
    assert umb["r_lo"] == pytest.approx(2.2e-12, rel=1e-6)
    assert umb["r_hi"] == pytest.approx(3.1e-12, rel=1e-6)
    assert payload["r_useful_marg_lo"] == umb["r_lo"]


def test_payload_without_qblock_marg_has_no_marginal_useful_block():
    exp1, exp2, exp3 = _full_experiments()
    payload = band_payload(exp1, exp2, exp3)
    assert "r_useful_marg_lo" not in payload
    assert "useful_marginal_band" not in payload


def test_core_payload_per_device_and_spread():
    from powertoflops.bench.analysis import core_payload

    def dev(uuid, scale):
        return [
            _rec(experiment=4, gpu_uuid=uuid, dtype="fp32",
                 r_j_per_op=scale * 10e-12),
            _rec(experiment=4, gpu_uuid=uuid, dtype="int8",
                 r_j_per_op=scale * 1e-12),
            _rec(experiment=4, gpu_uuid=uuid, dtype="fp16",
                 operand_dist_a="zeros", r_j_per_op=scale * 0.5e-12),
        ]

    payload = core_payload(dev("GPU-aaa", 1.0) + dev("GPU-bbb", 1.2),
                           inputs={"core": ["a.jsonl", "b.jsonl"]})
    assert set(payload["per_device"]) == {"GPU-aaa", "GPU-bbb"}
    assert payload["per_device"]["GPU-aaa"]["r_lo"] == 1e-12
    assert payload["per_device"]["GPU-aaa"]["r_hi"] == 10e-12
    assert payload["per_device"]["GPU-aaa"]["r_lo_cov"] == 0.5e-12
    assert payload["device_spread"]["r_lo"] == [1e-12, 1.2e-12]
    assert payload["n_devices"] == 2


def test_build_manifest_fields():
    m = build_manifest(
        sweep_id="abc123", experiment=2, config={"repeats": 5},
        gpu_uuid="GPU-aaa", gpu_name="A100", driver_version="580",
        git_commit="deadbeef", hostname="node01", slurm_job_id="123",
        started_utc="2026-07-21T00:00:00+00:00",
        finished_utc="2026-07-21T00:10:00+00:00",
        n_records=30, outputs=["exp2_operand_abc123.jsonl"],
    )
    for key in ("sweep_id", "experiment", "config", "gpu_uuid", "gpu_name",
                "driver_version", "git_commit", "hostname", "slurm_job_id",
                "started_utc", "finished_utc", "n_records", "outputs"):
        assert key in m
    assert m["config"] == {"repeats": 5}
    assert m["outputs"] == ["exp2_operand_abc123.jsonl"]
