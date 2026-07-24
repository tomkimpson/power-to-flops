"""Matmul workload builder, operand-value generators, and the nominal op count.

The bench drives a dense matmul whose NOMINAL OPERATION count is known
analytically (C = 2*M*N*K per matmul: each multiply-accumulate counted as two
ops, whatever the numeric format — int8 MACs are not FLOPs, review C7) and
reads energy per nominal op r = E/C off the meter. Two axes are controlled
here:

  * precision (the kappa factor) via ``dtype`` — "fp32" (tf32 OFF), "tf32" (fp32
    tensors with the tf32 tensor-core path ON), "fp16", "bf16", "int8";
  * operand *values* (the alpha switching-activity factor) via ``operand_fill``.

:func:`ops_nominal_matmul` is pure (no torch); the per-format capacity these
counts are compared against lives in :data:`powertoflops.config.A100_DENSE_PEAK_OPS`.
:func:`operand_fill` and
:func:`make_matmul` build torch tensors but accept ``device="cpu"`` so the
operand generators are testable without a GPU (tests/test_workload.py).

The measured window runs ``iters`` matmuls back-to-back with *no host
synchronisation inside* (the sync fences live in powertoflops.bench.meter), keeping the
GPU saturated so the energy is compute-bound and the counter resolution is fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Literal

import torch

OperandDist = Literal[
    "zeros", "sparse_struct", "low_entropy",
    "dense_random", "adversarial_toggle", "declared_typical",
]

# Map our dtype labels to a (torch dtype, tf32-allowed) pair. "int8" is handled
# specially in make_matmul (integer matmul path), so it is absent here.
_FLOAT_DTYPES: dict[str, tuple[torch.dtype, bool]] = {
    "fp32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "fp16": (torch.float16, False),
    "bf16": (torch.bfloat16, False),
}

# Integer bit-view dtype for constructing max-toggle bit patterns (same width).
_BITVIEW: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
}

# Complementary bit patterns (A, ~A) per integer width — maximal Hamming distance
# between neighbours = maximal switching activity in the datapath.
_TOGGLE_PATTERNS: dict[torch.dtype, tuple[int, int]] = {
    torch.int16: (0x5555, -0x5556),        # 0x5555 and 0xAAAA (two's-complement)
    torch.int32: (0x55555555, -0x55555556),
    torch.int8: (0x55, -0x56),             # 0x55 and 0xAA
}

SPARSE_FRACTION = 0.5   # structured sparsity: half the operands forced to zero


def ops_nominal_matmul(M: int, N: int, K: int) -> int:
    """Nominal op count of one dense (M,K)*(K,N) matmul: 2*M*N*K (pure).

    "Nominal operations" counts each multiply-accumulate as two ops regardless
    of numeric format (fp32 ... int8) — the format-neutral workload measure the
    bands are expressed in; format-dependent capacity is a separate model
    (:data:`powertoflops.config.A100_DENSE_PEAK_OPS`).
    """
    return 2 * M * N * K


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "int8":
        return torch.int8
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"unknown dtype label {dtype!r}")
    return _FLOAT_DTYPES[dtype][0]


def operand_fill(
    shape: tuple[int, int],
    dtype: str,
    dist: OperandDist,
    *,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build an operand tensor with the requested value distribution.

    The distributions span the activity factor from cheapest to most expensive:
    ``zeros`` (no switching) < ``sparse_struct`` / ``low_entropy`` <
    ``declared_typical`` / ``dense_random`` < ``adversarial_toggle`` (maximal
    bit switching). ``declared_typical`` is a standard-normal stand-in for
    realistic activations/weights; it priors sigma_dec.
    """
    tdt = _torch_dtype(dtype)
    rows, cols = shape

    if dist == "zeros":
        return torch.zeros(shape, dtype=tdt, device=device)

    if dist == "adversarial_toggle":
        return _adversarial(shape, tdt, device)

    # The value distributions below are generated as float and cast (or generated
    # as int directly for the int8 path).
    if tdt == torch.int8:
        base = _randint8(shape, device, generator)
    else:
        if dist in ("dense_random",):
            base = _rand_uniform(shape, device, generator, lo=-1.0, hi=1.0)
        elif dist == "declared_typical":
            base = torch.randn(shape, device=device, generator=generator)
        elif dist == "low_entropy":
            # Few distinct levels -> low operand-bit entropy.
            base = torch.randint(0, 4, shape, device=device, generator=generator).float()
        elif dist == "sparse_struct":
            base = _rand_uniform(shape, device, generator, lo=-1.0, hi=1.0)
        else:
            raise ValueError(f"unknown operand dist {dist!r}")
        base = base.to(tdt)

    if dist == "sparse_struct":
        base = _apply_structured_sparsity(base)
    elif dist == "low_entropy" and tdt == torch.int8:
        base = (base % 4)  # collapse to 4 levels for the int8 path too

    return base.contiguous()


def _rand_uniform(shape, device, generator, *, lo, hi) -> torch.Tensor:
    u = torch.rand(shape, device=device, generator=generator)
    return lo + (hi - lo) * u


def _randint8(shape, device, generator) -> torch.Tensor:
    return torch.randint(-128, 128, shape, device=device, dtype=torch.int8,
                         generator=generator)


def _apply_structured_sparsity(t: torch.Tensor) -> torch.Tensor:
    """Force a fixed fraction of operands to zero in a deterministic 1:2 pattern.

    A simple column-strided mask (every other column zeroed) gives exactly
    SPARSE_FRACTION zeros with a structured (not random) pattern, mirroring the
    2:4 structured sparsity real accelerators exploit.
    """
    out = t.clone()
    stride = int(round(1.0 / (1.0 - SPARSE_FRACTION)))  # = 2 for 0.5
    out[:, ::stride] = 0
    return out


def _adversarial(shape, tdt: torch.dtype, device) -> torch.Tensor:
    """Checkerboard of two complementary bit patterns (maximal switching)."""
    if tdt == torch.int8:
        a, b = _TOGGLE_PATTERNS[torch.int8]
        flat = torch.empty(shape, dtype=torch.int8, device=device)
        _fill_checkerboard(flat, a, b)
        return flat
    iv = _BITVIEW[tdt]
    a, b = _TOGGLE_PATTERNS[iv]
    bits = torch.empty(shape, dtype=iv, device=device)
    _fill_checkerboard(bits, a, b)
    return bits.view(tdt)


def _fill_checkerboard(t: torch.Tensor, a: int, b: int) -> None:
    """In-place checkerboard fill: t[i,j] = a if (i+j) even else b."""
    t[:] = b
    t[::2, ::2] = a
    t[1::2, 1::2] = a


@dataclass(frozen=True)
class MatmulSpec:
    """One matmul configuration (a single sweep cell's workload)."""

    M: int
    N: int
    K: int
    dtype: str                 # fp32 | tf32 | fp16 | bf16 | int8
    dist_a: str = "dense_random"
    dist_b: str = "dense_random"
    iters: int = 64            # back-to-back matmuls per measured window
    # Operand-residency control (R2.2 locality de-confound): number of A/B
    # operand sets used round-robin across iterations, at byte-identical
    # shape/kernel/op count. 1 = one set reused (L2-resident after the first
    # iteration); 8 = between reuses of a set, 7x its footprint flows through
    # L2, guaranteeing eviction (DRAM-streamed).
    n_buffers: int = 1


def make_matmul(
    spec: MatmulSpec,
    *,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> Callable[[], None]:
    """Pre-allocate operands and return a closure running ``spec.iters`` matmuls.

    The closure does no host sync; operand buffers are pre-allocated and
    round-robined per iteration (``spec.n_buffers`` A/B sets — the residency
    axis), so all energy is the datapath doing the MACs plus the operand
    traffic the residency setting implies (not allocation). Each set is an
    independent random draw: identical values would let L2 dedup/compression
    blur the residency contrast. Sets the tf32 backend flag per dtype label.
    """
    A_sets = [operand_fill((spec.M, spec.K), spec.dtype, spec.dist_a,
                           device=device, generator=generator)
              for _ in range(spec.n_buffers)]
    B_sets = [operand_fill((spec.K, spec.N), spec.dtype, spec.dist_b,
                           device=device, generator=generator)
              for _ in range(spec.n_buffers)]
    iters = spec.iters
    n_sets = spec.n_buffers

    if spec.dtype == "int8":
        # Two int8-only constraints (issue #25), both value-preserving:
        # (1) cuBLAS dispatches int8 to the IMMA/cutlass tensor-core kernels
        #     only for the TN layout — B must be column-major, else it falls
        #     back to a SIMT igemm kernel at ~11% of peak (measured 5.8x
        #     slower; see scripts/diagnose_int8_kernels.py --variants);
        # (2) reuse the int32 accumulator like the float path reuses `out`
        #     below, else every iteration allocates a fresh M*N*4-byte output.
        B_sets = [B.t().contiguous().t() for B in B_sets]
        out_i32 = torch.empty((spec.M, spec.N), dtype=torch.int32, device=device)

        def run_int8() -> None:
            for i in range(iters):
                torch._int_mm(A_sets[i % n_sets], B_sets[i % n_sets],
                              out=out_i32)  # int8 x int8 -> int32 (CUDA)
        return run_int8

    torch.backends.cuda.matmul.allow_tf32 = _FLOAT_DTYPES[spec.dtype][1]
    out = torch.empty((spec.M, spec.N), dtype=A_sets[0].dtype, device=device)

    def run_float() -> None:
        for i in range(iters):
            torch.matmul(A_sets[i % n_sets], B_sets[i % n_sets], out=out)
    return run_float


def seconds_per_matmul(
    spec: MatmulSpec,
    *,
    n_time: int = 5,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> float:
    """Time one matmul of ``spec`` (averaged over ``n_time``), with sync fences.

    Used by the runner to calibrate how many back-to-back matmuls fill the target
    measured-window wall time (so the NVML energy counter spans many updates and
    the delta is accurate regardless of dtype throughput).
    """
    work = make_matmul(replace(spec, iters=n_time), device=device,
                       generator=generator)
    # Warm up FIRST: the initial call pays cuBLAS autotuning + lazy-context init,
    # which otherwise inflates the estimate ~100x and starves the window of iters
    # (the sub-ms-window zero-energy bug, 2026-06-18). Time only the warmed run.
    work()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    work()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_time
