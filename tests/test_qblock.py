"""Quantized-transformer-block workload (referee B5 rebuild).

Pins the three B5 fixes: (1) quantization scales are APPLIED at dequant (the
old bare int32->fp16 cast saturated at |x| > 65504 and NaN'd the softmax);
(2) the op count includes every matmul the meter sees (fp16 attention was
metered but uncounted — anti-conservative for a denominator lower bound);
(3) the forward is validated against an fp32 reference, with finiteness a
hard assertion on the measured workload itself.

The quantized graph runs on CPU via the exact-arithmetic _int_matmul fallback,
so everything but the tensor-core dispatch is CPU-testable; the GPU-marked
tests exercise the real torch._int_mm path under SLURM.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from powertoflops.bench.qblock import (  # noqa: E402
    QBLOCK_MARG_EXPERIMENT,
    QBlockSpec,
    _int_matmul,
    apply_rope,
    assert_finite,
    make_qblock,
    ops_elementwise_estimate,
    ops_nominal_qblock,
    qblock_arm_label,
    qblock_reference,
    quantize_sym_int8,
    validate_qblock,
)
from powertoflops.bench.qweights import quantize_weights, synthetic_weights  # noqa: E402
from powertoflops.config import DEFAULT  # noqa: E402


def _small(d=64, n_head=4, ffn_mult=2, seq=8, batch=2, iters=1, seed=0):
    """Synthetic-weight fixture: (spec, weights, quantized weights, x0)."""
    spec = QBlockSpec(d_model=d, n_head=n_head, ffn_mult=ffn_mult,
                      seq=seq, batch=batch, iters=iters)
    g = torch.Generator().manual_seed(seed)
    w = synthetic_weights(d, n_head, ffn_mult, generator=g)
    x0 = torch.randn(seq * batch, d, generator=g)
    return spec, w, quantize_weights(w), x0


# ------------------------------------------------------------- op accounting

def test_ops_nominal_counts_all_metered_matmuls():
    # Six int8 matmuls: 2 * tokens * (4 + 2*ffn_mult) * d^2, PLUS the fp16
    # attention (QK^T and AV, 2*batch*n_head*seq^2*d_head each = together
    # 4*batch*seq^2*d). The old int8-only count metered the attention energy
    # while excluding its ops, inflating r = E/C — the wrong direction for a
    # denominator lower bound (referee B5).
    spec = QBlockSpec(d_model=4096, n_head=32, ffn_mult=4, seq=512, batch=8)
    tokens = 512 * 8
    int8_ops = 2 * tokens * 12 * 4096 ** 2
    attn_ops = 4 * 8 * 512 ** 2 * 4096
    assert ops_nominal_qblock(spec) == int8_ops + attn_ops


def test_ops_nominal_scales_with_seq_squared():
    a = ops_nominal_qblock(QBlockSpec(d_model=64, n_head=4, ffn_mult=2,
                                      seq=16, batch=2))
    b = ops_nominal_qblock(QBlockSpec(d_model=64, n_head=4, ffn_mult=2,
                                      seq=32, batch=2))
    # int8 part doubles with tokens; attention part quadruples with seq
    int8_16 = 2 * 32 * 8 * 64 ** 2
    attn_16 = 4 * 2 * 16 ** 2 * 64
    assert a == int8_16 + attn_16
    assert b == 2 * int8_16 + 4 * attn_16


def test_elementwise_residual_below_two_percent_at_configured_cells():
    # The uncounted elementwise work (softmax, LN, RoPE, GELU, quant/dequant,
    # residual) is bounded explicitly: a generous estimate must stay <= 2% of
    # the counted ops at every configured cell — the "explicitly subtract and
    # validate excluded components" arm of B5.
    cfg = DEFAULT.bench
    for seq, batch in cfg.qblock_cells:
        spec = QBlockSpec(d_model=cfg.qblock_d_model, n_head=cfg.qblock_n_head,
                          ffn_mult=cfg.qblock_ffn_mult, seq=seq, batch=batch)
        ratio = ops_elementwise_estimate(spec) / ops_nominal_qblock(spec)
        assert ratio <= 0.02, (seq, batch, ratio)


# ------------------------------------------------------------- quantization

def test_quantize_sym_int8_range_and_scale():
    x = torch.randn(64, 64) * 3.0
    q, scale = quantize_sym_int8(x)
    assert q.dtype == torch.int8
    assert q.abs().max().item() <= 127
    assert scale > 0
    assert torch.allclose(q.float() * scale, x, atol=scale)


def test_quantize_sym_int8_zeros_input():
    q, scale = quantize_sym_int8(torch.zeros(8, 8))
    assert q.abs().max().item() == 0
    assert scale > 0   # never a zero divisor


# --------------------------------------------------- the B5 overflow defect

def test_unscaled_cast_overflows_where_scaled_dequant_is_finite():
    # The retired forward did torch._int_mm(...).to(float16) with no scales:
    # at d >= ~512 the int32 products exceed fp16's 65504 and saturate to inf.
    # Pin the mechanism and the fix in one place.
    k = 512
    a = torch.full((4, k), 127, dtype=torch.int8)
    b = torch.full((k, 4), 127, dtype=torch.int8)
    prod = _int_matmul(a, b)
    assert prod.dtype == torch.int32
    assert prod[0, 0].item() == k * 127 * 127          # exact arithmetic
    bad = prod.to(torch.float16)                        # the old code path
    assert torch.isinf(bad).all()
    good = (prod.to(torch.float32) * (0.01 * 0.01)).to(torch.float16)
    assert torch.isfinite(good).all()                   # the fixed path


def test_forward_finite_at_overflow_prone_width():
    # d = 512 is wide enough that the unscaled cast would saturate; the scaled
    # forward must stay finite end to end.
    spec, w, qw, x0 = _small(d=512, n_head=8, ffn_mult=2, seq=8, batch=2)
    step, out = make_qblock(spec, qw, x0, device="cpu")
    step()
    assert_finite(out)


# ---------------------------------------------------- reference + validation

def test_quantized_forward_matches_fp32_reference():
    spec, w, qw, x0 = _small()
    step, out = make_qblock(spec, qw, x0, device="cpu")
    step()
    ref = qblock_reference(x0, w, spec)
    metrics = validate_qblock(out, ref)
    assert metrics["all_finite"]
    assert metrics["cosine"] >= DEFAULT.bench.qblock_min_cosine
    assert metrics["rel_l2"] <= DEFAULT.bench.qblock_max_rel_l2


def test_reference_is_parallel_residual():
    # out = x + attn(ln1 x) + mlp(ln2 x): zeroing every weight AND bias leaves
    # exactly the input (the residual stream) — pins the Pythia block shape.
    spec, w, _, x0 = _small()
    zeroed = dataclasses.replace(
        w, **{name: torch.zeros_like(getattr(w, name))
              for name in ("w_q", "w_k", "w_v", "w_o", "w_up", "w_down",
                           "b_q", "b_k", "b_v", "b_o", "b_up", "b_down")})
    ref = qblock_reference(x0, zeroed, spec)
    assert torch.allclose(ref, x0, atol=1e-6)


def test_apply_rope_identity_at_position_zero_and_norm_preserving():
    b, nh, s, dh = 2, 4, 6, 16
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, nh, s, dh, generator=g)
    k = torch.randn(b, nh, s, dh, generator=g)
    q_r, k_r = apply_rope(q, k, rotary_ndims=8)
    # position 0: cos = 1, sin = 0 -> identity
    assert torch.allclose(q_r[:, :, 0], q[:, :, 0], atol=1e-6)
    # rotation preserves the norm; the pass-through dims are untouched
    assert torch.allclose(q_r.norm(), q.norm(), atol=1e-4)
    assert torch.equal(q_r[..., 8:], q[..., 8:])
    assert torch.allclose(k_r.norm(), k.norm(), atol=1e-4)


def test_validate_qblock_flags_nonfinite():
    ref = torch.ones(4, 4)
    bad = torch.ones(4, 4)
    bad[0, 0] = float("inf")
    assert validate_qblock(bad, ref)["all_finite"] is False
    assert validate_qblock(ref, ref)["all_finite"] is True


def test_assert_finite_raises_on_inf_and_nan():
    assert_finite(torch.ones(3))
    for v in (float("inf"), float("nan")):
        t = torch.ones(3)
        t[1] = v
        with pytest.raises(RuntimeError, match="finite"):
            assert_finite(t)


def test_step_writes_measured_output_buffer():
    # The finiteness assertion must see the MEASURED workload's own output,
    # so step() writes into a persistent buffer every iteration.
    spec, w, qw, x0 = _small(iters=2)
    step, out = make_qblock(spec, qw, x0, device="cpu")
    assert torch.all(out == 0)      # untouched before the first step
    step()
    assert out.abs().sum() > 0
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------- config

def test_qblock_config_cells_and_measurement_knobs():
    cfg = DEFAULT.bench
    # a seq axis (attention-share sweep validating the op accounting) and a
    # batch axis, all servable from the 8 x 2048-token activation artifact
    assert cfg.qblock_cells == ((512, 4), (512, 8), (1024, 8),
                                (2048, 4), (2048, 8))
    assert cfg.qblock_d_model % cfg.qblock_n_head == 0
    assert cfg.qblock_repeats >= 8          # one-sided LCB needs the repeats
    assert 0 < cfg.qblock_alpha <= 0.05
    assert cfg.qblock_soak_s > 0
    assert 0.9 <= cfg.qblock_min_cosine < 1.0
    assert 0.0 < cfg.qblock_max_rel_l2 <= 0.5
    assert "qblock" in cfg.qblock_weights_npz
    assert "qblock" in cfg.qblock_acts_npz
    assert cfg.qblock_manifest.endswith(".json")


def test_run_qblock_refuses_to_start_without_weight_artifact(tmp_path):
    # No random fallback: a capture without the sha256-pinned real-weight
    # artifact must abort BEFORE touching the GPU, naming the fetch script.
    from powertoflops.bench.qblock import run_qblock

    cfg = dataclasses.replace(
        DEFAULT.bench,
        qblock_weights_npz=str(tmp_path / "missing.npz"),
        qblock_acts_npz=str(tmp_path / "missing_acts.npz"),
        qblock_manifest=str(tmp_path / "missing_manifest.json"),
    )
    with pytest.raises(FileNotFoundError, match="fetch_qblock_weights"):
        run_qblock(tmp_path, cfg, device_index=0)


# ---------------------------------------- marginal useful dose (qblock-marginal; B5)

def test_qblock_marg_experiment_number_is_eight():
    # 4=core, 5=pair, 6=qblock, 7=exp7 taken; the marginal useful capture is 8.
    assert QBLOCK_MARG_EXPERIMENT == 8


def test_qblock_arm_label_matches_bands_prefix():
    from powertoflops.bench.bands import QBLOCK_MARG_ARM_PREFIX

    label = qblock_arm_label(2048, 8)
    assert label == "qblock_2048x8"
    assert label.startswith(QBLOCK_MARG_ARM_PREFIX)


def test_qblock_marg_config_grid_is_feasible():
    from powertoflops.bench.marginal import MAX_BUSY_FRAC, build_dose_sweep

    cfg = DEFAULT.bench
    assert 0.0 in cfg.qblock_marg_cov_fracs          # intercept anchor present
    assert cfg.marg_declared_frac + max(cfg.qblock_marg_cov_fracs) <= MAX_BUSY_FRAC
    arms = [qblock_arm_label(s, b) for s, b in cfg.qblock_marg_cells]
    pts = build_dose_sweep(arms, cfg.qblock_marg_cov_fracs, cfg.marg_declared_frac)
    assert len(pts) == len(arms) * len(cfg.qblock_marg_cov_fracs)


def test_run_qblock_marginal_refuses_to_start_without_weight_artifact(tmp_path):
    # Same no-random-fallback contract as run_qblock: abort (CPU, pre-GPU) when
    # the sha256-pinned artifact is absent, naming the fetch script.
    from powertoflops.bench.qblock import run_qblock_marginal

    cfg = dataclasses.replace(
        DEFAULT.bench,
        qblock_weights_npz=str(tmp_path / "missing.npz"),
        qblock_acts_npz=str(tmp_path / "missing_acts.npz"),
        qblock_manifest=str(tmp_path / "missing_manifest.json"),
    )
    with pytest.raises(FileNotFoundError, match="fetch_qblock_weights"):
        run_qblock_marginal(tmp_path, cfg, device_index=0)


# ---------------------------------------------------------------------- GPU

@pytest.mark.gpu
def test_qblock_forward_finite_and_validated_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    spec, w, qw, x0 = _small(d=256, n_head=4, ffn_mult=4, seq=32, batch=2,
                             iters=2)
    qw_gpu = dataclasses.replace(
        qw, **{f"w_{n}_i8": getattr(qw, f"w_{n}_i8").cuda()
               for n in ("q", "k", "v", "o", "up", "down")})
    step, out = make_qblock(spec, qw_gpu, x0, device="cuda")
    step()
    torch.cuda.synchronize()
    assert_finite(out)
    ref = qblock_reference(x0, w, spec)
    metrics = validate_qblock(out.cpu(), ref)
    assert metrics["all_finite"]
    assert metrics["cosine"] >= DEFAULT.bench.qblock_min_cosine


@pytest.mark.gpu
def test_run_qblock_marginal_smoke(tmp_path):
    import os

    pytest.importorskip("pynvml")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from powertoflops.bench.qblock import run_qblock_marginal
    from powertoflops.bench.records import read_jsonl

    cfg = DEFAULT.bench
    if not os.path.exists(cfg.qblock_weights_npz):
        pytest.skip("qblock weight artifact absent "
                    "(run scripts/fetch_qblock_weights.py)")

    # Tiny grid: one arm, 3 doses, short windows — minutes, not the full sweep.
    cfg = dataclasses.replace(
        cfg, qblock_marg_cells=((512, 8),),
        qblock_marg_cov_fracs=(0.0, 0.2, 0.4),
        marg_window_s=3.0, repeats=2, warmup=1, qblock_soak_s=1.0,
    )
    path = run_qblock_marginal(tmp_path, cfg)
    recs = [r for r in read_jsonl(path)
            if r.experiment == QBLOCK_MARG_EXPERIMENT]
    assert len(recs) == 3 * 2
    doses = {r.cov_iters for r in recs}
    assert 0 in doses and len(doses) == 3
    spec = QBlockSpec(d_model=cfg.qblock_d_model, n_head=cfg.qblock_n_head,
                      ffn_mult=cfg.qblock_ffn_mult, seq=512, batch=8)
    for r in recs:
        assert r.energy_j > 0.0
        assert r.cov_dtype == "qblock_512x8"
        assert abs(r.duration_s - cfg.marg_window_s) < 1.0
        assert r.ops_nominal > 0                     # declared burst described
        assert (r.cov_ops_nominal > 0) == (r.cov_iters > 0)
        if r.cov_iters > 0:
            assert r.cov_ops_nominal == ops_nominal_qblock(spec) * r.cov_iters
