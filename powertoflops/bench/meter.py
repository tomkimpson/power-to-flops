"""NVML energy/power meter and the measurement loop.

Primary path: ``nvmlDeviceGetTotalEnergyConsumption`` — a cumulative on-device
millijoule counter (A100/Ampere+). Energy over a window is one subtraction
E1 - E0, integrated in firmware at a cadence far finer than NVML's ~50-100 ms
power-query rate, with no host-side integration error. Fallback (older silicon
or a denied counter): integrate ``nvmlDeviceGetPowerUsage`` in a background
sampler thread between the synced fences. ``EnergySample.method`` records which
path produced each number.

Correctness hinge: ``torch.cuda.synchronize()`` is called immediately before
latching E0 and immediately before latching E1, so the energy delta brackets
exactly the kernels of the window — no tail of the previous kernel, no missing
tail of this one. The window must run enough back-to-back matmuls (BenchConfig.
iters) to draw tens of joules so mJ quantisation is negligible (< 0.01 %).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterator

try:
    import pynvml
except ImportError:
    # The orchestration layer (EnergySample, measure_workload, _flag_trimmed)
    # is pure and testable against a fake meter anywhere; only the real Meter
    # needs NVML, and it raises at construction if pynvml is missing.
    pynvml = None


@dataclass(frozen=True)
class EnergySample:
    energy_j: float
    duration_s: float
    mean_power_w: float
    method: str   # "nvml_energy_counter" | "nvml_power_integral"
    # Flagged (not discarded) by the pre-specified exclusion rule: the
    # highest-energy discard_frac tail (throttle/preemption excursions) and
    # non-physical zero/negative-energy counter misses. All samples are saved;
    # band extraction (powertoflops.bench.bands) applies the exclusion downstream.
    trimmed: bool = False
    # Per-window telemetry (R2.1 clock stratification), sampled during the
    # window by _TelemetrySampler. None = telemetry absent (fake meters).
    sm_clock_p50_mhz: float | None = None
    sm_clock_min_mhz: float | None = None
    sm_clock_max_mhz: float | None = None
    temperature_max_c: float | None = None
    throttle_union: int = 0


_POWER_POLL_S = 0.01       # fallback power-sampler cadence [s]
_TELEMETRY_POLL_S = 0.1    # telemetry cadence [s]; NVML clock/temp update ~1 Hz+


class Meter:
    """Energy meter for one device. Manages NVML init/shutdown."""

    def __init__(self, device_index: int = 0, *, sync: Callable[[], None] | None = None):
        if pynvml is None:
            raise RuntimeError("Meter requires nvidia-ml-py (pynvml) — not installed")
        self.device_index = device_index
        pynvml.nvmlInit()
        self._owns_nvml = True
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._sync = sync if sync is not None else _torch_sync
        self._use_counter = self._probe_energy_counter()

    # -- capability probe -------------------------------------------------
    def _probe_energy_counter(self) -> bool:
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return True
        except Exception:
            return False

    def supports_energy_counter(self) -> bool:
        return self._use_counter

    # -- low-level reads --------------------------------------------------
    def _energy_mj(self) -> int:
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def _power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _read_telemetry(self) -> tuple:
        """(sm_clock_mhz, temp_c, throttle_bits), None per failed sub-query."""
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
        except Exception:
            clock = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle,
                                                   pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
        except Exception:
            throttle = None
        return clock, temp, throttle

    # -- measurement context ---------------------------------------------
    @contextmanager
    def measure(self) -> Iterator[Callable[[], EnergySample]]:
        """Bracket a workload and yield ``read()`` -> :class:`EnergySample`.

        Call ``read()`` *after* the ``with`` block (the sample is finalised on
        exit). Syncs and latches on both enter and exit.
        """
        box: dict[str, EnergySample] = {}
        telemetry = _TelemetrySampler(self._read_telemetry)

        if self._use_counter:
            self._sync()
            telemetry.start()
            e0 = self._energy_mj()
            t0 = time.perf_counter()
            try:
                yield lambda: box["sample"]
            finally:
                self._sync()
                t1 = time.perf_counter()
                e1 = self._energy_mj()
                telemetry.stop()
                dur = t1 - t0
                energy = (e1 - e0) / 1000.0
                box["sample"] = EnergySample(
                    energy_j=energy,
                    duration_s=dur,
                    mean_power_w=energy / dur if dur > 0 else float("nan"),
                    method="nvml_energy_counter",
                    **telemetry.stats(),
                )
        else:
            sampler = _PowerSampler(self._power_w)
            self._sync()
            telemetry.start()
            sampler.start()
            t0 = time.perf_counter()
            try:
                yield lambda: box["sample"]
            finally:
                self._sync()
                t1 = time.perf_counter()
                sampler.stop()
                telemetry.stop()
                dur = t1 - t0
                energy = sampler.integral_j()
                box["sample"] = EnergySample(
                    energy_j=energy,
                    duration_s=dur,
                    mean_power_w=energy / dur if dur > 0 else float("nan"),
                    method="nvml_power_integral",
                    **telemetry.stats(),
                )

    def close(self) -> None:
        if getattr(self, "_owns_nvml", False):
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._owns_nvml = False

    def __del__(self):
        self.close()


class _TelemetrySampler:
    """Background thread sampling (SM clock, temperature, throttle) mid-window.

    Clock locking is privilege-denied on shared nodes, so the achieved operating
    state must be RECORDED per measured window for analysis-side stratification
    (review C4). ``read_state`` returns ``(sm_clock_mhz, temp_c, throttle_bits)``
    with None for any sub-query that failed; each field is aggregated over the
    ticks where it was present. Torch's launch loop releases the GIL, so a 10 Hz
    thread schedules reliably alongside the workload.
    """

    def __init__(self, read_state: Callable[[], tuple], poll_s: float = _TELEMETRY_POLL_S):
        self._read = read_state
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._clocks: list[float] = []
        self._temps: list[float] = []
        self._throttle_union = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample_once(self) -> None:
        try:
            clock, temp, throttle = self._read()
        except Exception:
            return
        if clock is not None:
            self._clocks.append(float(clock))
        if temp is not None:
            self._temps.append(float(temp))
        if throttle is not None:
            self._throttle_union |= int(throttle)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._poll_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def stats(self) -> dict:
        """Aggregate to the EnergySample/RunRecord telemetry fields."""
        clocks = self._clocks
        return {
            "sm_clock_p50_mhz": float(_median(clocks)) if clocks else None,
            "sm_clock_min_mhz": min(clocks) if clocks else None,
            "sm_clock_max_mhz": max(clocks) if clocks else None,
            "temperature_max_c": max(self._temps) if self._temps else None,
            "throttle_union": self._throttle_union,
        }


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


class _PowerSampler:
    """Background thread sampling power; trapezoidal energy integral."""

    def __init__(self, read_power: Callable[[], float]):
        self._read = read_power
        self._stop = threading.Event()
        self._ts: list[float] = []
        self._pw: list[float] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._ts.append(time.perf_counter())
            self._pw.append(self._read())
            time.sleep(_POWER_POLL_S)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def integral_j(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(self._ts)):
            dt = self._ts[i] - self._ts[i - 1]
            e += 0.5 * (self._pw[i] + self._pw[i - 1]) * dt
        return e


def _torch_sync() -> None:
    import torch
    # Guarded: syncing would CREATE a CUDA context, and the Exp-3 "no_ctx" idle
    # cell must be measured before one exists. With no context there is nothing
    # of ours to fence, so skipping is exact, not an approximation.
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _flag_trimmed(samples: list[EnergySample],
                  discard_frac: float) -> list[EnergySample]:
    """Flag (never drop) samples the pre-specified exclusion rule covers.

    Two flag sources, order preserved: non-physical zero/negative-energy windows
    (a counter-miss signature: window shorter than the NVML energy-counter update
    interval), and the highest-energy ``discard_frac`` tail of the *physical*
    samples (throttle/preemption excursions). Band extraction excludes flagged
    rows; the raw values all survive on disk.
    """
    physical_idx = [i for i, s in enumerate(samples) if s.energy_j > 0.0]
    k = int(len(physical_idx) * discard_frac) if discard_frac > 0.0 else 0
    tail = set(sorted(physical_idx,
                      key=lambda i: samples[i].energy_j)[len(physical_idx) - k:]
               if k > 0 else [])
    return [
        replace(s, trimmed=True) if (i not in physical_idx or i in tail) else s
        for i, s in enumerate(samples)
    ]


def measure_workload(
    meter: Meter,
    work: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    discard_frac: float = 0.2,
) -> list[EnergySample]:
    """Warm up, then measure ``repeats`` synced windows; flag the high tail.

    Warmup windows absorb JIT/autotune, lazy CUDA-context init, and clock
    ramp/thermal soak. Each measured window syncs-latch-run-sync-latch via
    :meth:`Meter.measure`. Returns ALL samples in measurement order, with the
    exclusion rule applied as a ``trimmed`` flag (see :func:`_flag_trimmed`) —
    ``discard_frac`` records the capture-time choice, but the actual exclusion
    happens at band extraction, so it can be revisited without re-measuring.
    """
    for _ in range(warmup):
        work()
    samples: list[EnergySample] = []
    for _ in range(repeats):
        with meter.measure() as read:
            work()
        samples.append(read())
    return _flag_trimmed(samples, discard_frac)
