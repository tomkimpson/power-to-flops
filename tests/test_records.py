"""RunRecord round-trip (JSONL/CSV) and schema-completeness tests (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powertoflops.bench.records import (
    RunRecord,
    read_jsonl,
    record_for,
    to_csv,
    to_jsonl,
)


def _rec(**over) -> RunRecord:
    base = dict(
        experiment=2, sweep_id="abc123", repeat_index=0,
        M=8192, N=8192, K=8192, iters=64, ops_nominal=2 * 8192 ** 3 * 64,
        dtype="fp16", locality="dram",
        operand_dist_a="dense_random", operand_dist_b="dense_random",
        util_frac=None,
        energy_j=123.4, duration_s=0.4, mean_power_w=308.5,
        meter_method="nvml_energy_counter", r_j_per_op=1.2e-12,
        gpu_name="NVIDIA A100", gpu_uuid="GPU-xyz", driver_version="580.159.04",
        cuda_version="12040", torch_version="2.5.1+cu124",
        persistence_mode=True, sm_clock_mhz=1410, mem_clock_mhz=1215,
        power_limit_w=400.0, temperature_c=55.0, throttled=False,
        throttle_reasons=0, clocks_locked=False,
        timestamp_utc="2026-06-18T00:00:00+00:00",
    )
    base.update(over)
    return RunRecord(**base)


def test_jsonl_roundtrip(tmp_path):
    recs = [_rec(repeat_index=i, energy_j=100.0 + i) for i in range(3)]
    path = tmp_path / "exp2.jsonl"
    to_jsonl(recs, path)
    back = read_jsonl(path)
    assert back == recs


def test_jsonl_appends(tmp_path):
    path = tmp_path / "exp2.jsonl"
    to_jsonl([_rec(repeat_index=0)], path)
    to_jsonl([_rec(repeat_index=1)], path)
    assert len(read_jsonl(path)) == 2


def test_csv_has_all_fields(tmp_path):
    path = tmp_path / "exp2.csv"
    to_csv([_rec()], path)
    header = path.read_text().splitlines()[0].split(",")
    assert header == RunRecord.field_names()
    # every schema field is represented as a column
    assert len(header) == len(RunRecord.field_names())


def test_trimmed_flag_roundtrips(tmp_path):
    path = tmp_path / "x.jsonl"
    to_jsonl([_rec(repeat_index=0, trimmed=True), _rec(repeat_index=1)], path)
    back = read_jsonl(path)
    assert back[0].trimmed is True
    assert back[1].trimmed is False


def test_legacy_jsonl_without_trimmed_loads(tmp_path):
    # Records written before the trimmed flag existed must load as trimmed=False
    # (their high tail was already discarded at capture time).
    import dataclasses
    import json
    row = dataclasses.asdict(_rec())
    del row["trimmed"]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(row) + "\n")
    back = read_jsonl(path)
    assert len(back) == 1
    assert back[0].trimmed is False


def test_legacy_jsonl_with_flops_field_names_loads(tmp_path):
    # Records written before the nominal-ops rename (review C7) used "flops"
    # and "r_j_per_flop"; they must map onto the new schema on read.
    import dataclasses
    import json
    row = dataclasses.asdict(_rec())
    row["flops"] = row.pop("ops_nominal")
    row["r_j_per_flop"] = row.pop("r_j_per_op")
    del row["trimmed"]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(row) + "\n")
    back = read_jsonl(path)
    assert len(back) == 1
    assert back[0].ops_nominal == 2 * 8192 ** 3 * 64
    assert back[0].r_j_per_op == 1.2e-12


def test_util_frac_none_survives_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    to_jsonl([_rec(util_frac=None), _rec(util_frac=0.5)], path)
    back = read_jsonl(path)
    assert back[0].util_frac is None
    assert back[1].util_frac == 0.5


def test_telemetry_fields_roundtrip(tmp_path):
    # Per-window telemetry (R2.1 clock stratification): median/min/max SM clock,
    # max temperature, and the OR of throttle bits seen DURING the window.
    rec = _rec(sm_clock_p50_mhz=1395.0, sm_clock_min_mhz=1380.0,
               sm_clock_max_mhz=1410.0, temperature_max_c=61.0,
               throttle_union=0x4)
    path = tmp_path / "x.jsonl"
    to_jsonl([rec], path)
    back = read_jsonl(path)
    assert back[0].sm_clock_p50_mhz == 1395.0
    assert back[0].sm_clock_min_mhz == 1380.0
    assert back[0].sm_clock_max_mhz == 1410.0
    assert back[0].temperature_max_c == 61.0
    assert back[0].throttle_union == 0x4


def test_telemetry_fields_default_none(tmp_path):
    # None (not NaN): NaN breaks dataclass equality and strict JSON; None marks
    # "telemetry absent" (legacy records, fake meters) unambiguously.
    rec = _rec()
    assert rec.sm_clock_p50_mhz is None
    assert rec.sm_clock_min_mhz is None
    assert rec.sm_clock_max_mhz is None
    assert rec.temperature_max_c is None
    assert rec.throttle_union == 0
    path = tmp_path / "x.jsonl"
    to_jsonl([rec], path)
    assert read_jsonl(path) == [rec]


def test_legacy_jsonl_without_telemetry_fields_loads(tmp_path):
    # Records written before the R2 telemetry fields must load with defaults.
    import dataclasses
    import json
    row = dataclasses.asdict(_rec())
    for f in ("sm_clock_p50_mhz", "sm_clock_min_mhz", "sm_clock_max_mhz",
              "temperature_max_c", "throttle_union", "idle_variant"):
        del row[f]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(row) + "\n")
    back = read_jsonl(path)
    assert back[0].sm_clock_p50_mhz is None
    assert back[0].throttle_union == 0
    assert back[0].idle_variant == ""


def test_idle_variant_roundtrips(tmp_path):
    # Exp-3 idle-variation cells (R2.4) are labeled by which controllable-P0
    # state they measured; busy cells carry the empty string.
    path = tmp_path / "x.jsonl"
    to_jsonl([_rec(idle_variant="post_load"), _rec()], path)
    back = read_jsonl(path)
    assert back[0].idle_variant == "post_load"
    assert back[1].idle_variant == ""


def test_exp7_covert_dose_fields_roundtrip(tmp_path):
    # Exp-7 windows (referee B1/B2 marginal capture) carry the covert dose
    # beside the declared workload fields.
    rec = _rec(experiment=7, cov_dtype="int8", cov_dist="zeros",
               cov_iters=40, cov_ops_nominal=40 * 2 * 2048 ** 3,
               window_pad_s=3.25)
    path = tmp_path / "x.jsonl"
    to_jsonl([rec], path)
    back = read_jsonl(path)
    assert back[0].cov_dtype == "int8"
    assert back[0].cov_iters == 40
    assert back[0].cov_ops_nominal == 40 * 2 * 2048 ** 3
    assert back[0].window_pad_s == 3.25


def test_legacy_jsonl_without_covert_fields_loads(tmp_path):
    # Every committed pre-Exp-7 record must load with inert covert defaults.
    import dataclasses
    import json
    row = dataclasses.asdict(_rec())
    for f in ("cov_dtype", "cov_dist", "cov_iters", "cov_ops_nominal",
              "window_pad_s"):
        del row[f]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(row) + "\n")
    back = read_jsonl(path)
    assert back[0].cov_dtype == "" and back[0].cov_dist == ""
    assert back[0].cov_iters == 0 and back[0].cov_ops_nominal == 0
    assert back[0].window_pad_s == 0.0


# --- record_for: figures are pinned to the records that priced the bands ------
# Regression guard for the referee-round defect where five plot scripts picked
# their input newest-by-mtime. Git does not preserve mtimes, so a fresh clone
# plotted a retired capture under a caption quoting the current one.

def _bands(tmp_path, **inputs) -> Path:
    p = tmp_path / "measured_bands.json"
    p.write_text(json.dumps({"inputs": inputs}))
    return p


def test_record_for_reads_the_named_record_not_the_newest(tmp_path):
    # The named record is the OLDEST file present; mtime selection would take
    # the other one. record_for must ignore mtime entirely.
    named, decoy = tmp_path / "exp1_named.jsonl", tmp_path / "exp1_decoy.jsonl"
    named.write_text("")
    decoy.write_text("")
    import os
    os.utime(named, (1, 1))          # named is far older
    _bands(tmp_path, exp1={"path": f"results/bench/{named.name}"})
    assert record_for(1, tmp_path) == named


def test_record_for_takes_only_the_basename_so_bench_dir_redirects(tmp_path):
    # The artifact records a repo-relative path; --bench-dir must still be able
    # to point at a scratch copy.
    (tmp_path / "exp3_x.jsonl").write_text("")
    _bands(tmp_path, exp3={"path": "results/bench/exp3_x.jsonl"})
    assert record_for(3, tmp_path).parent == tmp_path


def test_record_for_fails_loudly_when_bands_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="measured_bands.json"):
        record_for(1, tmp_path)


def test_record_for_fails_loudly_when_experiment_unnamed(tmp_path):
    _bands(tmp_path, exp1={"path": "results/bench/exp1_x.jsonl"})
    with pytest.raises(KeyError, match="exp2"):
        record_for(2, tmp_path)


def test_record_for_fails_loudly_when_named_record_missing(tmp_path):
    _bands(tmp_path, exp1={"path": "results/bench/exp1_gone.jsonl"})
    with pytest.raises(FileNotFoundError, match="exp1_gone"):
        record_for(1, tmp_path)


def test_record_for_resolves_the_committed_artifact():
    # End-to-end against the real artifact: each band figure's record must be
    # the one measured_bands.json names, and must exist.
    for experiment in (1, 2, 3):
        path = record_for(experiment)
        assert path.exists() and path.name.startswith(f"exp{experiment}_")
