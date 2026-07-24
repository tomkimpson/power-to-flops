"""Diagnose the int8 matmul path: kernel dispatch + achieved throughput (issue #25).

Experiment 1 measured int8 ~2.5x dearer per op than fp16 on the A100 — the
opposite of the hardware expectation (624 TOPS int8 vs 312 TFLOP/s fp16). The
two suspects were (1) a fresh int32 output allocated every iteration (fixed in
powertoflops.bench.workload.make_matmul by reusing an ``out=`` buffer) and (2)
``torch._int_mm`` not dispatching to the IMMA tensor-core kernels. This script
settles both without ncu/nsys:

  * runs one warmed window per dtype under ``torch.profiler`` and prints the
    CUDA kernel names — IMMA tensor-core GEMMs carry ``imma``/``i8`` in their
    mangled names (e.g. ``sm80_xmma_gemm_i8i8_i8i32_...``);
  * times a warmed window (powertoflops.bench.workload.seconds_per_matmul) and prints
    achieved TFLOP/s (TOP/s for int8) against the A100 dense spec peak.

GPU-only. On OzSTAR submit e.g.:
    sbatch --account=oz022 --partition=milan-gpu --gres=gpu:a100:1 \
           --time=00:10:00 --mem=16G --output=results/bench/diagnose-int8-%j.out \
           --wrap "python -u scripts/diagnose_int8_kernels.py"
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from powertoflops.bench.workload import (  # noqa: E402
    MatmulSpec, ops_nominal_matmul, make_matmul, seconds_per_matmul,
)

# A100 dense (non-sparse) tensor-core spec peaks, TFLOP/s (TOP/s for int8).
# fp32 is the non-tensor-core FMA path (tf32 OFF, matching the bench label).
A100_DENSE_PEAK_TFLOPS = {
    "fp32": 19.5,
    "tf32": 156.0,
    "fp16": 312.0,
    "bf16": 312.0,
    "int8": 624.0,
}


def profiled_kernel_names(spec: MatmulSpec, device: str) -> list[str]:
    """Warm up, then profile one window; return the CUDA kernel names seen."""
    gen = torch.Generator(device=device).manual_seed(0)
    work = make_matmul(spec, device=device, generator=gen)
    work()                                   # warm-up: autotuning + lazy init
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        work()
        torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        # Keep device-side kernels only (runtime API rows have no device time).
        dev_t = getattr(evt, "self_device_time_total",
                        getattr(evt, "self_cuda_time_total", 0))
        if dev_t > 0 and "memcpy" not in evt.key.lower():
            names.append(evt.key)
    return names


def achieved_tflops(spec: MatmulSpec, device: str) -> float:
    gen = torch.Generator(device=device).manual_seed(0)
    t_one = seconds_per_matmul(spec, n_time=spec.iters, device=device,
                               generator=gen)
    return ops_nominal_matmul(spec.M, spec.N, spec.K) / t_one / 1e12


def int8_variants(size: int, iters: int, device: str) -> None:
    """Race candidate int8 GEMM routes: layout and compiler variants.

    cuBLAS only dispatches int8 to the IMMA tensor-core kernels for certain
    operand layouts (classically: B column-major, the TN gemm); row-major x
    row-major falls back to the SIMT ``igemm`` path. torch.compile with
    max-autotune can instead emit a Triton int8 kernel using mma.s8.
    """
    import time

    gen = torch.Generator(device=device).manual_seed(0)
    A = torch.randint(-128, 128, (size, size), dtype=torch.int8,
                      device=device, generator=gen)
    B = torch.randint(-128, 128, (size, size), dtype=torch.int8,
                      device=device, generator=gen)
    B_col = B.t().contiguous().t()          # same values, column-major storage
    out = torch.empty((size, size), dtype=torch.int32, device=device)

    variants: dict[str, callable] = {
        "nn  (A,B row-major — bench baseline)":
            lambda: torch._int_mm(A, B, out=out),
        "nt  (B column-major)":
            lambda: torch._int_mm(A, B_col, out=out),
    }
    try:
        compiled = torch.compile(torch._int_mm, mode="max-autotune",
                                 dynamic=False)
        variants["compile (max-autotune Triton)"] = lambda: compiled(A, B)
    except Exception as e:                              # noqa: BLE001
        print(f"torch.compile variant unavailable: {e}")

    ops = ops_nominal_matmul(size, size, size)
    for label, fn in variants.items():
        print(f"\n--- int8 variant: {label} ---")
        try:
            fn()                                        # warm-up / autotune
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            t_one = (time.perf_counter() - t0) / iters
            print(f"achieved {ops / t_one / 1e12:7.1f} TOP/s   "
                  f"({100 * ops / t_one / 1e12 / 624:.0f}% of 624 dense peak)")
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                fn()
                torch.cuda.synchronize()
            for evt in prof.key_averages():
                dev_t = getattr(evt, "self_device_time_total",
                                getattr(evt, "self_cuda_time_total", 0))
                if dev_t > 0 and "memcpy" not in evt.key.lower():
                    print(f"  {evt.key}")
        except Exception as e:                          # noqa: BLE001
            print(f"FAILED: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtypes", nargs="+", default=["int8", "fp16"],
                    choices=sorted(A100_DENSE_PEAK_TFLOPS))
    ap.add_argument("--size", type=int, default=8192,
                    help="square matmul dimension M=N=K (default: 8192, as exp1)")
    ap.add_argument("--iters", type=int, default=5,
                    help="matmuls per profiled/timed window")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", action="store_true",
                    help="also race int8 layout/compiler variants for IMMA dispatch")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable — run on a GPU node (see module docstring)")

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    for dtype in args.dtypes:
        spec = MatmulSpec(args.size, args.size, args.size, dtype,
                          iters=args.iters)
        tflops = achieved_tflops(spec, args.device)
        peak = A100_DENSE_PEAK_TFLOPS[dtype]
        unit = "TOP/s" if dtype == "int8" else "TFLOP/s"
        print(f"\n=== {dtype}  ({args.size}^3, iters={args.iters}) ===")
        print(f"achieved {tflops:7.1f} {unit}   "
              f"({100 * tflops / peak:.0f}% of {peak:.0f} {unit} A100 dense peak)")
        print("CUDA kernels in one profiled window:")
        names = profiled_kernel_names(spec, args.device)
        for name in names:
            print(f"  {name}")
        if dtype == "int8":
            # cuBLAS IMMA kernels carry imma/i8; the cutlass tensor-core int8
            # GEMMs carry tensorop_i16832gemm_s8 (mma.sync.m16n8k32.s8).
            hit = [n for n in names if any(
                p in n.lower() for p in ("imma", "i8i8", "tensorop"))]
            verdict = ("IMMA tensor-core path CONFIRMED"
                       if hit else "NO imma/i8 kernel — tensor-core path MISSED")
            print(f"int8 dispatch: {verdict}")

    if args.variants:
        int8_variants(args.size, args.iters, args.device)


if __name__ == "__main__":
    main()
