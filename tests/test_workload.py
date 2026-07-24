"""Operand-generator and FLOP-count tests (CPU torch; no GPU needed).

The operand generators accept device="cpu" so their value-distribution
invariants are testable without a GPU. Skipped entirely if torch is absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from powertoflops.bench.workload import (  # noqa: E402
    SPARSE_FRACTION,
    MatmulSpec,
    make_matmul,
    operand_fill,
    ops_nominal_matmul,
)


def test_ops_nominal_matmul_is_2mnk():
    assert ops_nominal_matmul(2, 3, 4) == 2 * 2 * 3 * 4
    assert ops_nominal_matmul(8192, 8192, 8192) == 2 * 8192 ** 3


def test_a100_dense_peaks_per_format():
    # Per-format capacity model (review C7): one scalar F_max hides that the
    # A100's int8 dense peak is 2x fp16/bf16 and tf32 is 8x fp32.
    from powertoflops.config import A100_DENSE_PEAK_OPS
    assert set(A100_DENSE_PEAK_OPS) == {"fp32", "tf32", "fp16", "bf16", "int8"}
    assert A100_DENSE_PEAK_OPS["int8"] == 2 * A100_DENSE_PEAK_OPS["fp16"]
    assert A100_DENSE_PEAK_OPS["fp16"] == A100_DENSE_PEAK_OPS["bf16"]
    assert A100_DENSE_PEAK_OPS["tf32"] == 8 * A100_DENSE_PEAK_OPS["fp32"]
    assert A100_DENSE_PEAK_OPS["fp32"] == 19.5e12


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_zeros_are_zero(dtype):
    t = operand_fill((16, 16), dtype, "zeros", device="cpu")
    assert torch.count_nonzero(t).item() == 0


def test_int8_zeros_are_zero():
    t = operand_fill((16, 16), "int8", "zeros", device="cpu")
    assert torch.count_nonzero(t).item() == 0


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_adversarial_has_two_complementary_bit_patterns(dtype):
    t = operand_fill((8, 8), dtype, "adversarial_toggle", device="cpu")
    iv = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
          torch.float32: torch.int32}[t.dtype]
    bits = t.view(iv)
    uniq = torch.unique(bits).tolist()
    assert len(uniq) == 2
    a, b = sorted(uniq)
    # complementary patterns: a == ~b  (two's complement: ~b = -b-1)
    assert a == ~b


def test_adversarial_int8_two_patterns():
    t = operand_fill((8, 8), "int8", "adversarial_toggle", device="cpu")
    uniq = sorted(torch.unique(t).tolist())
    assert uniq == [-86, 85]          # 0xAA, 0x55


def test_sparse_struct_hits_target_sparsity():
    t = operand_fill((32, 32), "fp16", "sparse_struct", device="cpu")
    zero_frac = 1.0 - torch.count_nonzero(t).item() / t.numel()
    assert abs(zero_frac - SPARSE_FRACTION) < 1e-9


def test_low_entropy_has_few_levels():
    t = operand_fill((64, 64), "fp32", "low_entropy", device="cpu")
    assert torch.unique(t).numel() <= 4


def test_dense_random_is_dense():
    t = operand_fill((32, 32), "fp16", "dense_random", device="cpu")
    # essentially no exact zeros from a continuous uniform fill
    assert torch.count_nonzero(t).item() >= t.numel() - 2


# ---- operand-buffer rotation (R2.2 locality de-confound) --------------------
#
# Locality is controlled by WHERE operands are served from, at byte-identical
# shape/kernel/op count: n_buffers=1 reuses one A/B set (L2-resident after the
# first iteration); n_buffers=8 round-robins 8 sets so 7x the footprint flows
# through L2 between reuses (guaranteed eviction). The tests spy on torch.matmul
# to observe which operand tensors each iteration actually receives.


def _spy_matmul(monkeypatch):
    seen = []
    real = torch.matmul

    def spy(A, B, out=None):
        seen.append((A, B))
        return real(A, B, out=out)

    monkeypatch.setattr(torch, "matmul", spy)
    return seen


def test_matmul_spec_defaults_single_buffer():
    spec = MatmulSpec(M=8, N=8, K=8, dtype="fp32")
    assert spec.n_buffers == 1


def test_single_buffer_reuses_same_operands(monkeypatch):
    seen = _spy_matmul(monkeypatch)
    spec = MatmulSpec(M=8, N=8, K=8, dtype="fp32", iters=3, n_buffers=1)
    work = make_matmul(spec, device="cpu",
                       generator=torch.Generator().manual_seed(0))
    work()
    assert len(seen) == 3
    ids = {(id(a), id(b)) for a, b in seen}
    assert len(ids) == 1


def test_buffer_rotation_cycles_sets(monkeypatch):
    seen = _spy_matmul(monkeypatch)
    spec = MatmulSpec(M=8, N=8, K=8, dtype="fp32", iters=4, n_buffers=2)
    work = make_matmul(spec, device="cpu",
                       generator=torch.Generator().manual_seed(0))
    work()
    assert len(seen) == 4
    ids = [(id(a), id(b)) for a, b in seen]
    assert ids[0] == ids[2] and ids[1] == ids[3]   # round-robin period 2
    assert ids[0] != ids[1]                        # genuinely distinct sets


def test_rotated_sets_have_independent_values(monkeypatch):
    # Each rotated set must be a fresh random draw, not a copy: identical
    # values would let L2 dedup/compression blur the residency contrast.
    seen = _spy_matmul(monkeypatch)
    spec = MatmulSpec(M=8, N=8, K=8, dtype="fp32", iters=2, n_buffers=2)
    work = make_matmul(spec, device="cpu",
                       generator=torch.Generator().manual_seed(0))
    work()
    (a0, b0), (a1, b1) = seen
    assert not torch.equal(a0, a1)
    assert not torch.equal(b0, b1)
