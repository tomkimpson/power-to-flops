"""measure_workload orchestration against a fake meter (no GPU).

Verifies the warmup/repeat counts and the high-energy-tail trimming without
touching CUDA or NVML. The real sync hygiene is exercised by the @pytest.mark.gpu
smoke test under SLURM.
"""

from __future__ import annotations

from contextlib import contextmanager

from powertoflops.bench.meter import EnergySample, measure_workload


class FakeMeter:
    """Scripts a sequence of EnergySamples; counts work() calls."""

    def __init__(self, energies):
        self._energies = list(energies)
        self._i = 0
        self.work_calls = 0

    @contextmanager
    def measure(self):
        box = {}
        yield lambda: box["s"]
        e = self._energies[self._i]
        self._i += 1
        box["s"] = EnergySample(energy_j=e, duration_s=1.0, mean_power_w=e,
                                method="fake")


def test_warmup_and_repeat_counts():
    meter = FakeMeter([10.0, 11.0, 12.0])
    calls = {"n": 0}

    def work():
        calls["n"] += 1

    samples = measure_workload(meter, work, warmup=2, repeats=3, discard_frac=0.0)
    assert calls["n"] == 2 + 3          # warmup + measured windows
    assert len(samples) == 3


def test_all_repeats_returned_high_tail_flagged():
    # 5 samples, discard_frac 0.2 -> ALL saved; only the highest-energy one is
    # FLAGGED trimmed (the exclusion itself moves downstream to band extraction).
    meter = FakeMeter([10.0, 50.0, 11.0, 12.0, 13.0])
    samples = measure_workload(meter, lambda: None, warmup=0, repeats=5,
                               discard_frac=0.2)
    assert len(samples) == 5
    flagged = [s for s in samples if s.trimmed]
    assert len(flagged) == 1
    assert flagged[0].energy_j == 50.0


def test_no_flag_when_frac_zero():
    meter = FakeMeter([10.0, 50.0, 11.0])
    samples = measure_workload(meter, lambda: None, warmup=0, repeats=3,
                               discard_frac=0.0)
    assert len(samples) == 3
    assert not any(s.trimmed for s in samples)


def test_non_physical_energy_flagged_not_dropped():
    # A zero/negative-energy window (counter miss) is saved but flagged, and it
    # does not consume the high-tail budget of the physical samples.
    meter = FakeMeter([0.0, 10.0, 50.0, 11.0, 12.0, 13.0])
    samples = measure_workload(meter, lambda: None, warmup=0, repeats=6,
                               discard_frac=0.2)
    assert len(samples) == 6
    flagged = sorted(s.energy_j for s in samples if s.trimmed)
    assert flagged == [0.0, 50.0]   # counter miss + top 20% of the 5 physical


def test_measurement_order_preserved():
    # Flagging must not reorder samples: repeat_index k must stay sample k.
    meter = FakeMeter([13.0, 10.0, 50.0, 11.0, 12.0])
    samples = measure_workload(meter, lambda: None, warmup=0, repeats=5,
                               discard_frac=0.2)
    assert [s.energy_j for s in samples] == [13.0, 10.0, 50.0, 11.0, 12.0]


# ---- per-window telemetry (R2.1 clock stratification) ----------------------


def test_energy_sample_telemetry_defaults_none():
    # Fake meters and legacy paths produce samples without telemetry; the
    # fields must exist with None/0 defaults so downstream code can rely on them.
    s = EnergySample(energy_j=1.0, duration_s=1.0, mean_power_w=1.0, method="fake")
    assert s.sm_clock_p50_mhz is None
    assert s.sm_clock_min_mhz is None
    assert s.sm_clock_max_mhz is None
    assert s.temperature_max_c is None
    assert s.throttle_union == 0


def test_telemetry_sampler_stats():
    from powertoflops.bench.meter import _TelemetrySampler

    readings = iter([(1410.0, 55.0, 0x0), (1380.0, 61.0, 0x4), (1395.0, 58.0, 0x1)])
    sampler = _TelemetrySampler(lambda: next(readings))
    for _ in range(3):
        sampler._sample_once()
    stats = sampler.stats()
    assert stats["sm_clock_p50_mhz"] == 1395.0
    assert stats["sm_clock_min_mhz"] == 1380.0
    assert stats["sm_clock_max_mhz"] == 1410.0
    assert stats["temperature_max_c"] == 61.0
    assert stats["throttle_union"] == 0x5


def test_telemetry_sampler_empty_stats_are_none():
    from powertoflops.bench.meter import _TelemetrySampler

    sampler = _TelemetrySampler(lambda: (None, None, None))
    stats = sampler.stats()
    assert stats["sm_clock_p50_mhz"] is None
    assert stats["sm_clock_min_mhz"] is None
    assert stats["sm_clock_max_mhz"] is None
    assert stats["temperature_max_c"] is None
    assert stats["throttle_union"] == 0


def test_telemetry_sampler_skips_none_entries_per_field():
    from powertoflops.bench.meter import _TelemetrySampler

    # A failed NVML sub-query yields None for that field only; the other
    # fields of the same tick must still count.
    readings = iter([(1410.0, None, 0x4), (None, 60.0, None)])
    sampler = _TelemetrySampler(lambda: next(readings))
    for _ in range(2):
        sampler._sample_once()
    stats = sampler.stats()
    assert stats["sm_clock_p50_mhz"] == 1410.0
    assert stats["temperature_max_c"] == 60.0
    assert stats["throttle_union"] == 0x4


def test_telemetry_sampler_thread_collects_and_stops():
    import time

    from powertoflops.bench.meter import _TelemetrySampler

    sampler = _TelemetrySampler(lambda: (1410.0, 55.0, 0), poll_s=0.001)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    n = len(sampler._clocks)
    assert n >= 1
    time.sleep(0.02)
    assert len(sampler._clocks) == n   # no samples after stop()
