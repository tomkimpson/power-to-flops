"""GPU device introspection and (graceful) DVFS control over NVML.

:func:`query_device` snapshots the device state recorded with every run for
provenance and hygiene (clocks, power limit, persistence, temperature, throttle
reasons). The ``try_*`` controls attempt clock-locking / power-capping and return
``False`` on permission denial rather than raising — on a shared SLURM cluster
these usually need elevated privileges, so the harness must degrade gracefully:
the precision and operand axes need no privilege, and achieved clocks are
recorded per run so any passive governor variation is still captured.

Requires ``nvidia-ml-py`` (imported as ``pynvml``) only for the live GPU
controls below; the import is guarded so the pure analysis and weight-extraction
paths (which import this module transitively via qblock but never call these
functions) run in NVML-free environments. Calling any function here without
pynvml installed raises a clear error.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

try:
    import pynvml
except ImportError:
    # NVML-free envs (pure analysis / trained-weight extraction) import this
    # module transitively but never touch the GPU controls; fail only if one
    # is actually called (see _require_nvml), not at import time.
    pynvml = None

from .records import HARMFUL_THROTTLE_MASK


def _require_nvml():
    """Return the pynvml module or raise a clear error if it is not installed."""
    if pynvml is None:
        raise ModuleNotFoundError(
            "nvidia-ml-py (pynvml) is required for GPU device controls but is "
            "not installed in this environment — these functions only run on a "
            "measurement node")
    return pynvml


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    uuid: str
    driver_version: str
    cuda_version: str
    persistence_mode: bool
    sm_clock_mhz: int
    mem_clock_mhz: int
    power_limit_w: float
    max_power_limit_w: float
    temperature_c: float
    throttle_reasons: int
    locked_clocks: tuple[int, int] | None


def _decode(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def query_device(index: int = 0) -> DeviceInfo:
    """Snapshot device ``index`` via NVML. Assumes ``nvmlInit`` already called."""
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        cuda_ver = str(pynvml.nvmlSystemGetCudaDriverVersion_v2())
    except Exception:
        cuda_ver = "unknown"
    try:
        pm = bool(pynvml.nvmlDeviceGetPersistenceMode(h))
    except Exception:
        pm = False
    try:
        throttle = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
    except Exception:
        throttle = 0
    try:
        max_pl = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)[1] / 1000.0
    except Exception:
        max_pl = float("nan")
    return DeviceInfo(
        name=_decode(pynvml.nvmlDeviceGetName(h)),
        uuid=_decode(pynvml.nvmlDeviceGetUUID(h)),
        driver_version=_decode(pynvml.nvmlSystemGetDriverVersion()),
        cuda_version=cuda_ver,
        persistence_mode=pm,
        sm_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
        mem_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)),
        power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
        max_power_limit_w=max_pl,
        temperature_c=float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
        throttle_reasons=throttle,
        locked_clocks=None,
    )


def is_throttled(index: int = 0) -> bool:
    """True if a *harmful* (thermal / HW-slowdown) throttle is asserted.

    Uses HARMFUL_THROTTLE_MASK, so the SW power cap (the intended full-power
    regime the densest workloads sit in) and GpuIdle do NOT count as throttling —
    only clock cuts that make energy-per-FLOP unrepresentative do.
    """
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        reasons = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
    except Exception:
        return False
    return bool(reasons & HARMFUL_THROTTLE_MASK)


def try_lock_clocks(index: int, sm_mhz: int) -> bool:
    """Attempt to lock the SM clock to ``sm_mhz``. Return success; never raise."""
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, sm_mhz, sm_mhz)
        return True
    except Exception:
        return False


def reset_locked_clocks(index: int) -> bool:
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
        return True
    except Exception:
        return False


def try_set_power_limit(index: int, watts: float) -> bool:
    """Attempt to set the power-management limit [W]. Return success; never raise."""
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        pynvml.nvmlDeviceSetPowerManagementLimit(h, int(watts * 1000))
        return True
    except Exception:
        return False


def reset_power_limit(index: int) -> bool:
    """Restore the NVML *default* power limit. Return success; never raise.

    Independent of whatever this process set — persistence mode keeps a lowered
    cap alive across jobs, so callers must invoke this on every exit path.
    """
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h)
        pynvml.nvmlDeviceSetPowerManagementLimit(h, default_mw)
        return True
    except Exception:
        return False


def supported_sm_clocks(index: int = 0) -> list[int]:
    """List of supported SM clocks [MHz] at the max memory clock (best effort)."""
    h = _require_nvml().nvmlDeviceGetHandleByIndex(index)
    try:
        mems = pynvml.nvmlDeviceGetSupportedMemoryClocks(h)
        clocks: set[int] = set()
        for mc in mems:
            clocks.update(int(c) for c in pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mc))
        return sorted(clocks)
    except Exception:
        return []


def nvidia_smi_dump() -> str:
    """Capture `nvidia-smi -q` for the run log (provenance). Empty on failure."""
    try:
        return subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return ""
