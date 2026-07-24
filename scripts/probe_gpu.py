"""Step-0 capability probe for the A2 bench harness (run on an allocated GPU).

Confirms the two methodology unknowns before the full sweep is trusted:
  1. does nvmlDeviceGetTotalEnergyConsumption work on this device (use the energy
     counter) or must we fall back to power-sample integration?
  2. is clock-locking (nvmlDeviceSetGpuLockedClocks) permitted, or is DVFS denied
     on this shared node (so Exp 1's V^2 axis is passive-only)?
  3. is power-capping (nvmlDeviceSetPowerManagementLimit) permitted — gates the
     6.5 limit-cycle sweep. The probe always restores the default limit afterwards
     (persistence mode keeps a lowered cap alive across jobs).

Also runs one tiny measured matmul end-to-end so the meter path is exercised.

Run under SLURM, e.g.:
    srun --account=oz022 --partition=milan-gpu --gres=gpu:a100:1 --time=00:15:00 \
        python scripts/probe_gpu.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pynvml  # noqa: E402
import torch  # noqa: E402

from powertoflops.bench.device import (  # noqa: E402
    query_device,
    reset_locked_clocks,
    reset_power_limit,
    supported_sm_clocks,
    try_lock_clocks,
    try_set_power_limit,
)
from powertoflops.bench.meter import Meter, measure_workload  # noqa: E402
from powertoflops.bench.workload import MatmulSpec, make_matmul  # noqa: E402


def main() -> None:
    print("=== Step-0 A100 probe ===")
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))

    pynvml.nvmlInit()
    info = query_device(0)
    print("\n[device]")
    for k, v in info.__dict__.items():
        print(f"  {k:18s} = {v}")

    print("\n[energy counter]")
    meter = Meter(0)
    print("  supports_energy_counter =", meter.supports_energy_counter())

    print("\n[meter smoke: 2048^3 fp16 x16 iters, 2 warmup, 3 repeats]")
    gen = torch.Generator(device="cuda").manual_seed(0)
    work = make_matmul(MatmulSpec(2048, 2048, 2048, "fp16", iters=16),
                       device="cuda", generator=gen)
    samples = measure_workload(meter, work, warmup=2, repeats=3, discard_frac=0.0)
    for i, s in enumerate(samples):
        print(f"  rep {i}: {s.energy_j:8.3f} J  {s.duration_s*1e3:7.2f} ms  "
              f"{s.mean_power_w:6.1f} W  [{s.method}]")
    meter.close()

    print("\n[DVFS / clock-lock permission]")
    clocks = supported_sm_clocks(0)
    print("  #supported SM clocks =", len(clocks),
          ("(e.g. %s)" % clocks[:: max(len(clocks) // 5, 1)]) if clocks else "")
    target = clocks[len(clocks) // 2] if clocks else 1200
    ok = try_lock_clocks(0, target)
    print(f"  try_lock_clocks({target} MHz) -> {ok}",
          "(DVFS axis available)" if ok else "(DENIED — Exp 1 DVFS is passive-only)")
    if ok:
        reset_locked_clocks(0)

    print("\n[power-limit permission]")
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        lo_mw, hi_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        print(f"  constraints = [{lo_mw / 1000:.0f}, {hi_mw / 1000:.0f}] W")
    except Exception as e:
        print(f"  constraints query failed: {e}")
    try:
        default_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
        print(f"  default limit = {default_w:.0f} W")
    except Exception as e:
        print(f"  default-limit query failed: {e}")
    current_w = query_device(0).power_limit_w
    try:
        ok = try_set_power_limit(0, current_w - 25.0)
        readback = query_device(0).power_limit_w
        print(f"  try_set_power_limit({current_w - 25.0:.0f} W) -> {ok}, "
              f"read back {readback:.0f} W",
              "(6.5 cap sweep available)" if ok
              else "(DENIED — 6.5 blocked on admin, like clock-lock)")
    finally:
        # persistence mode keeps a lowered cap alive across jobs — always restore
        reset_ok = reset_power_limit(0)
        restored = query_device(0).power_limit_w
        print(f"  reset_power_limit -> {reset_ok}, restored limit = {restored:.0f} W")

    pynvml.nvmlShutdown()
    print("\n=== probe complete ===")


if __name__ == "__main__":
    main()
