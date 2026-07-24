"""GPU + NVML smoke tests (run under SLURM; skipped by `pytest -m 'not gpu'`).

Marked @pytest.mark.gpu and skipped when CUDA is unavailable, so they are inert
in CI and on login nodes but exercise the real meter on an allocated GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pynvml")

pytestmark = pytest.mark.gpu

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"
)


@needs_cuda
def test_meter_measures_positive_energy():
    from powertoflops.bench.meter import Meter, measure_workload
    from powertoflops.bench.workload import MatmulSpec, make_matmul

    from powertoflops.bench.workload import seconds_per_matmul

    meter = Meter(0)
    try:
        gen = torch.Generator(device="cuda").manual_seed(0)
        # Calibrate to ~1 s so the window spans many NVML energy-counter updates
        # (a few-ms window reads stale/zero deltas — probe finding 2026-06-18).
        spec = MatmulSpec(8192, 8192, 8192, "fp16", iters=5)
        t_one = seconds_per_matmul(spec, n_time=5, device="cuda", generator=gen)
        iters = max(1, int(round(1.0 / t_one)))
        work = make_matmul(MatmulSpec(8192, 8192, 8192, "fp16", iters=iters),
                           device="cuda", generator=gen)
        samples = measure_workload(meter, work, warmup=2, repeats=3, discard_frac=0.0)
        assert len(samples) >= 1
        for s in samples:
            assert s.duration_s > 0.0
            assert s.energy_j > 0.0
            assert 50.0 < s.mean_power_w < 600.0   # plausible A100 draw
            assert s.method in ("nvml_energy_counter", "nvml_power_integral")
    finally:
        meter.close()


@needs_cuda
def test_int8_matmul_at_least_as_fast_as_fp16():
    """int8 must achieve more ops/s than fp16 (issue #25 regression).

    A100 int8 IMMA peak is 2x the fp16 HMMA peak, so a healthy int8 path lands
    well above fp16's ops rate. The broken path (per-iteration int32 output
    allocation, or a non-tensor-core fallback kernel) measured ~0.25x fp16.
    The 1.2x bound is loose enough to tolerate clocks/throttling, tight enough
    to catch either failure mode.
    """
    from powertoflops.bench.workload import MatmulSpec, ops_nominal_matmul, seconds_per_matmul

    gen = torch.Generator(device="cuda").manual_seed(0)
    ops_per_s = {}
    for dtype in ("fp16", "int8"):
        spec = MatmulSpec(8192, 8192, 8192, dtype, iters=5)
        t_one = seconds_per_matmul(spec, n_time=5, device="cuda", generator=gen)
        ops_per_s[dtype] = ops_nominal_matmul(8192, 8192, 8192) / t_one
    assert ops_per_s["int8"] >= 1.2 * ops_per_s["fp16"], (
        f"int8 {ops_per_s['int8'] / 1e12:.1f} TOP/s vs "
        f"fp16 {ops_per_s['fp16'] / 1e12:.1f} TFLOP/s — int8 path degraded"
    )


@needs_cuda
def test_query_device_populated():
    import pynvml

    from powertoflops.bench.device import query_device

    pynvml.nvmlInit()
    try:
        info = query_device(0)
        assert info.name
        assert info.sm_clock_mhz > 0
        assert info.power_limit_w > 0
    finally:
        pynvml.nvmlShutdown()
