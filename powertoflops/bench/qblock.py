"""Trained-weight int8 transformer block — the useful-operand workload (B5).

The capability ladder's bottom rungs price covert work at the energy cost of
USEFUL operations, measured from an int8-quantized transformer forward step.
The first version of this benchmark (referee B5) was invalid: uniform-random
weights, quantization scales discarded (the bare ``int32 -> fp16`` cast
saturates at |x| > 65504 and NaNs the softmax), and fp16 attention energy
metered but excluded from the op count. This rebuild fixes all three:

  * Weights are ONE trained Pythia-6.9B layer, loaded sha256-verified via
    :mod:`powertoflops.bench.qweights` (no random fallback); activations are real
    hidden states of text through the model's lower layers.
  * Every dequant multiplies the kept activation x weight scales onto the
    int32 product in fp32 before the fp16 cast (:func:`make_qblock`), and the
    forward is validated against an fp32 reference (:func:`qblock_reference`,
    :func:`validate_qblock`) with finiteness a HARD assertion on the measured
    workload's own output buffer.
  * :func:`ops_nominal_qblock` counts every matmul the meter sees — the six
    int8 matmuls AND the fp16 attention (QK^T, AV). Under-counting metered
    work inflates r = E/C, which is anti-conservative for the covert-cost
    denominator the band feeds. The remaining uncounted elementwise work
    (softmax, LN, RoPE, GELU, quant/dequant, residual) is bounded explicitly
    by :func:`ops_elementwise_estimate` (< 2% of the count, pinned in tests):
    it is obligatory overhead of useful transformer computation that earns no
    counted ops under the same nominal-op convention as C_cov.

The block computes Pythia's actual layer function: parallel residual
``x + attn(ln1 x) + mlp(ln2 x)``, rotary embeddings on the first
``rotary_pct`` of head dims, causal attention, biases everywhere — with the
six projection/MLP matmuls in int8 (``torch._int_mm``, TN layout) and the
attention in fp16, softmax in fp32.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import socket
import time
import uuid as uuid_mod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from ..config import BenchConfig, DEFAULT
from .analysis import analysis_git_commit
from . import device as dev
from .meter import Meter, measure_workload
from .qweights import (
    QBlockWeights,
    QuantizedQBlockWeights,
    load_qblock_activations,
    load_qblock_weights,
    quantize_weights,
)
from .records import RunRecord, build_manifest, to_csv, to_jsonl

QBLOCK_EXPERIMENT = 6
# qblock-marginal (referee B5, third round): the marginal useful-cost dose-response
# (run_qblock_marginal). Distinct experiment id so the total-quotient qblock
# capture (6) and the incremental one (8) never mix.
QBLOCK_MARG_EXPERIMENT = 8
# Distinguishes the validated trained-weight capture from the retired
# random-weight one ("qblock_typical"); band extraction refuses the old tag.
QBLOCK_DIST = "qblock_pythia"


def qblock_arm_label(seq: int, batch: int) -> str:
    """The covert-arm tag for a qblock (seq, batch) cell in the dose sweep.

    Must match :data:`powertoflops.bench.bands.QBLOCK_MARG_ARM_PREFIX` so
    :func:`powertoflops.bench.bands.useful_marginal_band` recognises the arm.
    """
    return f"qblock_{seq}x{batch}"


@dataclass(frozen=True)
class QBlockSpec:
    """One qblock measurement cell: model dims + (seq, batch) + iters."""

    d_model: int
    n_head: int
    ffn_mult: int
    seq: int
    batch: int
    iters: int = 1      # forward steps per measured window


def ops_nominal_qblock(spec: QBlockSpec) -> int:
    """Nominal op count of ONE forward step (pure, no torch).

    Counts every matmul whose energy the meter integrates: the six int8
    matmuls (4 projections d x d + MLP up d x fd + MLP down fd x d over
    tokens = seq*batch rows) plus the fp16 attention (QK^T and AV, each
    2 * batch * n_head * seq^2 * d_head = together 4 * batch * seq^2 * d).
    Elementwise work stays excluded per the project-wide 2*M*N*K nominal-op
    convention and is bounded by :func:`ops_elementwise_estimate`.
    """
    tokens = spec.seq * spec.batch
    int8_ops = 2 * tokens * (4 + 2 * spec.ffn_mult) * spec.d_model ** 2
    attn_ops = 4 * spec.batch * spec.seq ** 2 * spec.d_model
    return int8_ops + attn_ops


def ops_elementwise_estimate(spec: QBlockSpec) -> int:
    """Generous upper estimate of the UNCOUNTED elementwise ops per step.

    Deliberately over-counts (e.g. 10 ops per softmax/LN/GELU element, the
    full head dim treated as rotary) so the pinned <= 2% residual bound is
    conservative. This is the explicit accounting of excluded components the
    referee's B5 checklist requires.
    """
    tokens = spec.seq * spec.batch
    d = spec.d_model
    fd = spec.ffn_mult * d
    scores = spec.batch * spec.n_head * spec.seq ** 2
    return (
        2 * 10 * tokens * d          # two layernorms (fp32)
        + 2 * 6 * tokens * d         # RoPE on q and k (rotary <= head dim)
        + 12 * scores                # scale + mask + fp32 softmax
        + 10 * tokens * fd           # GELU
        + 5 * (3 * tokens * d + tokens * fd)   # 4 dynamic-quant sites
        + 2 * (5 * tokens * d + tokens * fd)   # dequant scale + bias adds
        + 2 * tokens * d             # parallel-residual adds
    )


def quantize_sym_int8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric int8 quantization: (q, scale), x ~= q * scale."""
    amax = float(x.abs().max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def _int_matmul(a_i8: torch.Tensor, b_i8: torch.Tensor) -> torch.Tensor:
    """int8 x int8 -> int32 matmul: tensor cores on CUDA, exact fallback on CPU.

    The CPU path computes the identical int32 arithmetic, so the whole
    quantized graph is unit-testable without a GPU.
    """
    if a_i8.is_cuda:
        return torch._int_mm(a_i8, b_i8)
    return a_i8.to(torch.int32) @ b_i8.to(torch.int32)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, rotary_ndims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT-NeoX rotary embedding on the first ``rotary_ndims`` head dims.

    q, k: [batch, n_head, seq, d_head]. Half-split rotate convention
    (rotate_half(x) = cat(-x2, x1)), frequencies 10000^(-2i/rotary_ndims),
    math in fp32, cast back to the input dtype. rotary_ndims = 0 is a no-op.
    """
    if rotary_ndims <= 0:
        return q, k
    if rotary_ndims % 2:
        raise ValueError("rotary_ndims must be even")
    seq = q.shape[-2]
    half = rotary_ndims // 2
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, rotary_ndims, 2, device=q.device,
                     dtype=torch.float32) / rotary_ndims))
    pos = torch.arange(seq, device=q.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # [seq, half]
    emb = torch.cat((freqs, freqs), dim=-1)         # [seq, rotary_ndims]
    cos, sin = emb.cos(), emb.sin()

    def rot(x: torch.Tensor) -> torch.Tensor:
        x_r = x[..., :rotary_ndims].float()
        x1, x2 = x_r[..., :half], x_r[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        out = x_r * cos + rotated * sin
        return torch.cat((out.to(x.dtype), x[..., rotary_ndims:]), dim=-1)

    return rot(q), rot(k)


def assert_finite(t: torch.Tensor, context: str = "qblock output") -> None:
    """Hard failure on inf/NaN — never publish a window whose workload broke."""
    if not bool(torch.isfinite(t).all()):
        raise RuntimeError(
            f"{context} is not finite (inf/NaN) — the quantized forward is "
            f"numerically invalid; refusing to measure/publish")


def _causal_mask(seq: int, device, dtype=torch.float32) -> torch.Tensor:
    """Additive causal mask [seq, seq]: 0 on/below the diagonal, -inf above."""
    mask = torch.zeros(seq, seq, device=device, dtype=dtype)
    return mask.masked_fill(
        torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool),
                   diagonal=1),
        float("-inf"))


def make_qblock(
    spec: QBlockSpec,
    qw: QuantizedQBlockWeights,
    x0: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> tuple[Callable[[], None], torch.Tensor]:
    """Build the measured workload; returns (step, out_buffer).

    ``x0`` is the fp32 [tokens, d] real-activation input (batch-major rows).
    ``step`` runs ``spec.iters`` forward passes; every pass writes the block
    output into the persistent fp16 ``out_buffer`` so the harness can assert
    finiteness on the MEASURED workload's own output, not a separate run.

    Numerics (the B5 fix): each int8 matmul's int32 product is dequantized in
    fp32 as ``prod * (x_scale * w_scale) + bias`` before the fp16 cast — a
    bare cast saturates fp16 at d_model >= ~512. Activation quantization is
    dynamic (recomputed per step), as in real dynamic-quant int8 inference.
    """
    device = torch.device(device)
    d, nh = spec.d_model, spec.n_head
    dh = d // nh
    tokens = spec.seq * spec.batch
    w = qw.weights
    rotary_ndims = int(dh * w.rotary_pct)
    if rotary_ndims % 2:
        rotary_ndims -= 1

    w_i8 = {n: getattr(qw, f"w_{n}_i8").to(device)
            for n in ("q", "k", "v", "o", "up", "down")}
    scales = {n: getattr(qw, f"s_{n}")
              for n in ("q", "k", "v", "o", "up", "down")}
    bias = {n: getattr(w, f"b_{n}").float().to(device)
            for n in ("q", "k", "v", "o", "up", "down")}
    ln1_w, ln1_b = w.ln1_w.float().to(device), w.ln1_b.float().to(device)
    ln2_w, ln2_b = w.ln2_w.float().to(device), w.ln2_b.float().to(device)

    x32 = x0.detach().to(device=device, dtype=torch.float32)
    x16 = x32.to(torch.float16)
    mask16 = _causal_mask(spec.seq, device, dtype=torch.float16)
    inv_sqrt_dh = 1.0 / math.sqrt(dh)
    iters = spec.iters
    out_buffer = torch.zeros(tokens, d, device=device, dtype=torch.float16)

    def _heads(t: torch.Tensor) -> torch.Tensor:
        # (tokens, d) -> (batch, n_head, seq, d_head)
        return (t.view(spec.batch, spec.seq, nh, dh)
                 .permute(0, 2, 1, 3).contiguous())

    def _mm(x_fp32: torch.Tensor, name: str) -> torch.Tensor:
        """Dynamic-quant int8 matmul with the scales APPLIED at dequant."""
        x_q, x_s = quantize_sym_int8(x_fp32)
        prod = _int_matmul(x_q.contiguous(), w_i8[name])
        return ((prod.to(torch.float32) * (x_s * scales[name]) + bias[name])
                .to(torch.float16))

    def step() -> None:
        for _ in range(iters):
            # attention branch (Pythia parallel residual reads ln1(x))
            h1 = torch.nn.functional.layer_norm(
                x32, (d,), ln1_w, ln1_b, w.ln_eps)
            q = _mm(h1, "q")
            k = _mm(h1, "k")
            v = _mm(h1, "v")
            qh, kh, vh = _heads(q), _heads(k), _heads(v)
            qh, kh = apply_rope(qh, kh, rotary_ndims)
            scores = (qh * inv_sqrt_dh) @ kh.transpose(-1, -2) + mask16
            probs = torch.softmax(scores.float(), dim=-1).to(torch.float16)
            ctx = ((probs @ vh).permute(0, 2, 1, 3).reshape(tokens, d))
            attn_out = _mm(ctx.float(), "o")
            # MLP branch on ln2(x)
            h2 = torch.nn.functional.layer_norm(
                x32, (d,), ln2_w, ln2_b, w.ln_eps)
            up = torch.nn.functional.gelu(_mm(h2, "up"))
            down = _mm(up.float(), "down")
            out_buffer.copy_(x16 + attn_out + down)

    return step, out_buffer


def qblock_reference(
    x0: torch.Tensor, w: QBlockWeights, spec: QBlockSpec,
) -> torch.Tensor:
    """fp32 unquantized reference of the same block function.

    Same graph as :func:`make_qblock` (parallel residual, RoPE, causal mask,
    biases) with plain fp32 matmuls — the validation target. Anchored once to
    HF's GPTNeoXLayer by scripts/fetch_qblock_weights.py --crosscheck.
    """
    d, nh = spec.d_model, spec.n_head
    dh = d // nh
    tokens = spec.seq * spec.batch
    rotary_ndims = int(dh * w.rotary_pct)
    if rotary_ndims % 2:
        rotary_ndims -= 1
    device = x0.device
    x = x0.detach().float()

    def par(t: torch.Tensor) -> torch.Tensor:
        return t.float().to(device)

    def heads(t: torch.Tensor) -> torch.Tensor:
        return (t.view(spec.batch, spec.seq, nh, dh)
                 .permute(0, 2, 1, 3).contiguous())

    h1 = torch.nn.functional.layer_norm(
        x, (d,), par(w.ln1_w), par(w.ln1_b), w.ln_eps)
    q = h1 @ par(w.w_q) + par(w.b_q)
    k = h1 @ par(w.w_k) + par(w.b_k)
    v = h1 @ par(w.w_v) + par(w.b_v)
    qh, kh, vh = heads(q), heads(k), heads(v)
    qh, kh = apply_rope(qh, kh, rotary_ndims)
    scores = (qh / math.sqrt(dh)) @ kh.transpose(-1, -2)
    scores = scores + _causal_mask(spec.seq, device)
    probs = torch.softmax(scores, dim=-1)
    ctx = (probs @ vh).permute(0, 2, 1, 3).reshape(tokens, d)
    attn_out = ctx @ par(w.w_o) + par(w.b_o)

    h2 = torch.nn.functional.layer_norm(
        x, (d,), par(w.ln2_w), par(w.ln2_b), w.ln_eps)
    up = torch.nn.functional.gelu(h2 @ par(w.w_up) + par(w.b_up))
    down = up @ par(w.w_down) + par(w.b_down)
    return x + attn_out + down


def validate_qblock(
    quant_out: torch.Tensor, ref_out: torch.Tensor,
) -> dict:
    """Agreement metrics between the quantized forward and the fp32 reference.

    Returns {cosine, rel_l2, max_abs, all_finite}. Thresholds live in
    BenchConfig (qblock_min_cosine / qblock_max_rel_l2); finiteness is a hard
    requirement regardless of thresholds.
    """
    q = quant_out.detach().float().flatten()
    r = ref_out.detach().float().flatten()
    finite = bool(torch.isfinite(q).all()) and bool(torch.isfinite(r).all())
    if finite:
        cosine = float(torch.nn.functional.cosine_similarity(q, r, dim=0))
        rel_l2 = float((q - r).norm() / r.norm())
    else:
        cosine, rel_l2 = float("nan"), float("nan")
    return {
        "cosine": cosine,
        "rel_l2": rel_l2,
        "max_abs": float(q.abs().max()),
        "all_finite": finite,
    }


def _load_inputs(cfg: BenchConfig, seq: int, batch: int):
    """Weights + one cell's activations, sha256-verified. CPU-only — safe to
    call before any CUDA touch so a missing artifact aborts cheaply."""
    w = load_qblock_weights(cfg.qblock_weights_npz, cfg.qblock_manifest)
    x0 = load_qblock_activations(cfg.qblock_acts_npz, cfg.qblock_manifest,
                                 seq=seq, batch=batch)
    return w, x0


def _qw_to(qw: QuantizedQBlockWeights, device) -> QuantizedQBlockWeights:
    return dataclasses.replace(
        qw, **{f"w_{n}_i8": getattr(qw, f"w_{n}_i8").to(device)
               for n in ("q", "k", "v", "o", "up", "down")})


def _largest_cell(cfg: BenchConfig) -> tuple[int, int]:
    return max(cfg.qblock_cells, key=lambda sb: (sb[0] * sb[1], sb[0]))


def _spec(cfg: BenchConfig, seq: int, batch: int, iters: int = 1) -> QBlockSpec:
    return QBlockSpec(d_model=cfg.qblock_d_model, n_head=cfg.qblock_n_head,
                      ffn_mult=cfg.qblock_ffn_mult, seq=seq, batch=batch,
                      iters=iters)


def _validate_on_device(
    cfg: BenchConfig, spec: QBlockSpec, w: QBlockWeights,
    qw: QuantizedQBlockWeights, x0: torch.Tensor, device,
) -> dict:
    """Run one step + fp32 reference on ``device``; raise below thresholds."""
    step, out = make_qblock(dataclasses.replace(spec, iters=1), qw, x0,
                            device=device)
    step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert_finite(out, "pre-flight qblock output")
    ref = qblock_reference(x0.to(device), w, spec)
    metrics = validate_qblock(out, ref)
    if (not metrics["all_finite"]
            or metrics["cosine"] < cfg.qblock_min_cosine
            or metrics["rel_l2"] > cfg.qblock_max_rel_l2):
        raise RuntimeError(
            f"qblock pre-flight validation failed: {metrics} vs thresholds "
            f"cosine >= {cfg.qblock_min_cosine}, "
            f"rel_l2 <= {cfg.qblock_max_rel_l2} — refusing to measure")
    return metrics


def preflight_qblock(cfg: BenchConfig = DEFAULT.bench,
                     device_index: int = 0) -> dict:
    """Load, quantize, and validate on the GPU without measuring anything.

    The --preflight-only entry point: exercises the full artifact -> device ->
    forward -> reference chain on the largest configured cell and returns the
    validation metrics (raising on any failure).
    """
    seq, batch = _largest_cell(cfg)
    w, x0 = _load_inputs(cfg, seq, batch)
    qw = _qw_to(quantize_weights(w), torch.device(f"cuda:{device_index}"))
    return _validate_on_device(cfg, _spec(cfg, seq, batch), w, qw, x0,
                               torch.device(f"cuda:{device_index}"))


def _calibrate_qblock_iters(spec: QBlockSpec, cfg: BenchConfig,
                            qw: QuantizedQBlockWeights,
                            x0: torch.Tensor, device) -> int:
    """Steps per window to fill ~target_window_s (same rationale as the runner)."""
    if cfg.target_window_s <= 0:
        return 1
    step, _ = make_qblock(dataclasses.replace(spec, iters=1), qw, x0,
                          device=device)
    step()                       # warm: autotune + lazy init
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    t_one = time.perf_counter() - t0
    if t_one <= 0:
        return 1
    return max(1, min(cfg.max_iters, int(round(cfg.target_window_s / t_one))))


def run_qblock(out_dir: str | Path, cfg: BenchConfig = DEFAULT.bench,
               device_index: int = 0) -> Path:
    """Measure every (seq, batch) cell of cfg.qblock_cells; write records.

    Order of operations (each earlier stage aborts the capture):
      1. load + sha256-verify the trained-weight artifacts (CPU, pre-CUDA);
      2. GPU pre-flight: one validated step against the fp32 reference;
      3. thermal soak (~qblock_soak_s of the largest cell) so per-cell
         calibration freezes warm (B3 capture lesson);
      4. per cell: calibrate iters, measure cfg.qblock_repeats windows,
         then assert the measured workload's own output is finite.

    Same on-disk contract as powertoflops.bench.runner.run_experiment: JSONL + CSV +
    a provenance manifest (now carrying the weights provenance and the
    pre-flight validation metrics), records tagged experiment=6 /
    dist "qblock_pythia". Cell identity for band extraction lives in the
    shape fields: M=seq, N=batch, K=d_model.
    """
    out_dir = Path(out_dir)
    seq_max, batch_max = _largest_cell(cfg)
    w, x0_max = _load_inputs(cfg, seq_max, batch_max)   # pre-CUDA, fails loudly
    extraction_manifest = json.loads(
        Path(cfg.qblock_manifest).read_text(encoding="utf-8"))

    sweep_id = uuid_mod.uuid4().hex[:12]
    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"cuda:{device_index}")
    qw = _qw_to(quantize_weights(w), device)

    validation = _validate_on_device(
        cfg, _spec(cfg, seq_max, batch_max), w, qw, x0_max, device)

    # Thermal soak: run the largest cell until qblock_soak_s has elapsed so
    # calibration and the first measured cell see a warm card.
    soak_step, _ = make_qblock(_spec(cfg, seq_max, batch_max, iters=1),
                               qw, x0_max, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < cfg.qblock_soak_s:
        soak_step()
        torch.cuda.synchronize()

    meter = Meter(device_index)
    records: list[RunRecord] = []
    try:
        for seq, batch in cfg.qblock_cells:
            spec = _spec(cfg, seq, batch)
            x0 = load_qblock_activations(
                cfg.qblock_acts_npz, cfg.qblock_manifest, seq=seq, batch=batch)
            iters = _calibrate_qblock_iters(spec, cfg, qw, x0, device)
            step, out_buf = make_qblock(dataclasses.replace(spec, iters=iters),
                                        qw, x0, device=device)
            samples = measure_workload(meter, step, warmup=cfg.warmup,
                                       repeats=cfg.qblock_repeats,
                                       discard_frac=cfg.discard_frac)
            assert_finite(out_buf, f"measured qblock output (seq={seq}, "
                                   f"batch={batch})")
            ops = ops_nominal_qblock(spec) * iters
            info = dev.query_device(device_index)
            throttled = dev.is_throttled(device_index)
            stamp = datetime.now(timezone.utc).isoformat()
            for k, s in enumerate(samples):
                records.append(RunRecord(
                    experiment=QBLOCK_EXPERIMENT, sweep_id=sweep_id,
                    repeat_index=k,
                    M=seq, N=batch, K=cfg.qblock_d_model, iters=iters,
                    ops_nominal=ops, dtype="int8", locality="model",
                    operand_dist_a=QBLOCK_DIST, operand_dist_b=QBLOCK_DIST,
                    util_frac=None,
                    energy_j=s.energy_j, duration_s=s.duration_s,
                    mean_power_w=s.mean_power_w, meter_method=s.method,
                    r_j_per_op=(s.energy_j / ops) if ops > 0 else float("nan"),
                    gpu_name=info.name, gpu_uuid=info.uuid,
                    driver_version=info.driver_version,
                    cuda_version=info.cuda_version,
                    torch_version=torch.__version__,
                    persistence_mode=info.persistence_mode,
                    sm_clock_mhz=info.sm_clock_mhz,
                    mem_clock_mhz=info.mem_clock_mhz,
                    power_limit_w=info.power_limit_w,
                    temperature_c=info.temperature_c,
                    throttled=throttled,
                    throttle_reasons=info.throttle_reasons,
                    clocks_locked=False, timestamp_utc=stamp,
                    trimmed=s.trimmed,
                    sm_clock_p50_mhz=s.sm_clock_p50_mhz,
                    sm_clock_min_mhz=s.sm_clock_min_mhz,
                    sm_clock_max_mhz=s.sm_clock_max_mhz,
                    temperature_max_c=s.temperature_max_c,
                    throttle_union=s.throttle_union,
                ))
    finally:
        meter.close()

    base = out_dir / f"qblock_{sweep_id}"
    to_jsonl(records, base.with_suffix(".jsonl"))
    to_csv(records, base.with_suffix(".csv"))
    manifest = build_manifest(
        sweep_id=sweep_id, experiment=QBLOCK_EXPERIMENT,
        config=dataclasses.asdict(cfg),
        gpu_uuid=records[-1].gpu_uuid if records else "unknown",
        gpu_name=records[-1].gpu_name if records else "unknown",
        driver_version=records[-1].driver_version if records else "unknown",
        git_commit=analysis_git_commit(),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(records),
        outputs=[base.with_suffix(".jsonl").name, base.with_suffix(".csv").name],
        extra={
            "weights_provenance": w.provenance,
            "acts_sha256": extraction_manifest["files"]["acts"]["sha256"],
            "validation": validation,
        },
    )
    (base.parent / f"{base.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return base.with_suffix(".jsonl")


def _seconds_per_qblock(spec: QBlockSpec, qw: QuantizedQBlockWeights,
                        x0: torch.Tensor, device) -> float:
    """Warm-then-time one qblock forward step [s] (dose calibration)."""
    step, _ = make_qblock(dataclasses.replace(spec, iters=1), qw, x0,
                          device=device)
    step()                       # warm: autotune + lazy init
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    return max(time.perf_counter() - t0, 1e-6)


def run_qblock_marginal(out_dir: str | Path, cfg: BenchConfig = DEFAULT.bench,
                        device_index: int = 0) -> Path:
    """qblock-marginal (referee B5, third round): marginal useful-cost dose-response.

    The Exp-7 construction (:mod:`powertoflops.bench.marginal`) with the trained-weight
    qblock forward as the swept covert dose: every measured window holds the
    SAME fixed fp16 declared burst (calibrated once to ``marg_declared_frac`` of
    ``marg_window_s``) plus ``n_forward`` qblock forward passes of one cell,
    then idle-pads to the fixed window. Because the declared burst and the idle
    baseline are identical at every dose, they land in the intercept and cancel
    in the dose slope, so regressing window energy on covert nominal ops
    (:func:`powertoflops.bench.bands.useful_marginal_band`) gives the MARGINAL useful
    cost dE/dC_cov — the incremental estimand, not the total quotient E/C of
    :func:`run_qblock` (which folds in baseline power, anti-conservative as a
    covert denominator). One covert arm per ``qblock_marg_cells`` (seq, batch)
    cell, tagged ``qblock_<seq>x<batch>``; feed the JSONL to
    scripts/analyze_bands.py --qblock-marg.

    Reuses run_qblock's artifact loading / quantization / pre-flight / soak and
    run_marginal's declared burst / windowing / drift-reference splices.
    """
    import random

    from .marginal import (MarginalPoint, build_dose_sweep,
                           capacity_frac_of_dose)
    from .records import REFERENCE_DIST
    from .workload import (MatmulSpec, make_matmul, ops_nominal_matmul,
                           seconds_per_matmul)

    out_dir = Path(out_dir)
    cells = list(cfg.qblock_marg_cells)
    seq_max, batch_max = max(cells, key=lambda sb: (sb[0] * sb[1], sb[0]))
    w, x0_max = _load_inputs(cfg, seq_max, batch_max)   # pre-CUDA, fails loudly
    extraction_manifest = json.loads(
        Path(cfg.qblock_manifest).read_text(encoding="utf-8"))

    sweep_id = uuid_mod.uuid4().hex[:12]
    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"cuda:{device_index}")
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    qw = _qw_to(quantize_weights(w), device)

    validation = _validate_on_device(
        cfg, _spec(cfg, seq_max, batch_max), w, qw, x0_max, device)

    # Thermal soak on the largest cell so calibration freezes warm (B3 lesson).
    soak_step, _ = make_qblock(_spec(cfg, seq_max, batch_max, iters=1),
                               qw, x0_max, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < cfg.qblock_soak_s:
        soak_step()
        torch.cuda.synchronize()

    n_buf = dict(cfg.locality_buffers)[cfg.marg_declared_locality]
    Md, Nd, Kd = cfg.marg_declared_shape

    records: list[RunRecord] = []
    meter = Meter(device_index)
    try:
        # --- fixed fp16 declared burst (calibrated once) --------------------
        spec_d = MatmulSpec(Md, Nd, Kd, cfg.marg_declared_dtype,
                            dist_a=cfg.marg_declared_dist,
                            dist_b=cfg.marg_declared_dist,
                            iters=cfg.calib_matmuls, n_buffers=n_buf)
        t_decl = seconds_per_matmul(spec_d, n_time=cfg.calib_matmuls,
                                    device=device, generator=generator)
        n_decl = max(1, int(round(cfg.marg_declared_frac * cfg.marg_window_s
                                  / t_decl)))
        work_decl = make_matmul(dataclasses.replace(spec_d, iters=n_decl),
                                device=device, generator=generator)
        ops_decl = ops_nominal_matmul(Md, Nd, Kd) * n_decl

        # --- reference cell (campaign drift convention) ---------------------
        spec_ref = MatmulSpec(Md, Nd, Kd, "fp16", iters=cfg.calib_matmuls,
                              n_buffers=n_buf)
        t_ref = seconds_per_matmul(spec_ref, n_time=cfg.calib_matmuls,
                                   device=device, generator=generator)
        n_ref = max(1, min(cfg.max_iters,
                           int(round(cfg.target_window_s / t_ref))))
        work_ref = make_matmul(dataclasses.replace(spec_ref, iters=n_ref),
                               device=device, generator=generator)
        ops_ref = ops_nominal_matmul(Md, Nd, Kd) * n_ref

        # --- per-arm activations + one-forward calibration ------------------
        arm: dict[str, tuple] = {}
        for seq, batch in cells:
            spec = _spec(cfg, seq, batch)
            x0 = load_qblock_activations(
                cfg.qblock_acts_npz, cfg.qblock_manifest, seq=seq, batch=batch)
            t_one = _seconds_per_qblock(spec, qw, x0, device)
            arm[qblock_arm_label(seq, batch)] = (spec, x0, t_one)

        # --- execution order: seeded shuffle + reference splices ------------
        pts = build_dose_sweep(list(arm), cfg.qblock_marg_cov_fracs,
                               cfg.marg_declared_frac)
        rng = random.Random(cfg.seed)
        rng.shuffle(pts)
        ordered: list[MarginalPoint] = []
        for i, p in enumerate(pts):
            if cfg.ref_every > 0 and i % cfg.ref_every == 0:
                ordered.append(MarginalPoint("", 0.0, is_reference=True))
            ordered.append(p)

        def _emit(sp: MarginalPoint, samples, *, iters, ops, cov_iters,
                  cov_ops, pads) -> None:
            info = dev.query_device(device_index)
            throttled = dev.is_throttled(device_index)
            stamp = datetime.now(timezone.utc).isoformat()
            for k, s in enumerate(samples):
                records.append(RunRecord(
                    experiment=0 if sp.is_reference else QBLOCK_MARG_EXPERIMENT,
                    sweep_id=sweep_id, repeat_index=k,
                    M=Md, N=Nd, K=Kd, iters=iters, ops_nominal=ops,
                    dtype=cfg.marg_declared_dtype,
                    locality=cfg.marg_declared_locality,
                    operand_dist_a=(REFERENCE_DIST if sp.is_reference
                                    else cfg.marg_declared_dist),
                    operand_dist_b=(REFERENCE_DIST if sp.is_reference
                                    else cfg.marg_declared_dist),
                    util_frac=None,
                    energy_j=s.energy_j, duration_s=s.duration_s,
                    mean_power_w=s.mean_power_w, meter_method=s.method,
                    r_j_per_op=(s.energy_j / ops) if ops > 0 else float("nan"),
                    sm_clock_p50_mhz=s.sm_clock_p50_mhz,
                    sm_clock_min_mhz=s.sm_clock_min_mhz,
                    sm_clock_max_mhz=s.sm_clock_max_mhz,
                    temperature_max_c=s.temperature_max_c,
                    throttle_union=s.throttle_union,
                    gpu_name=info.name, gpu_uuid=info.uuid,
                    driver_version=info.driver_version,
                    cuda_version=info.cuda_version,
                    torch_version=torch.__version__,
                    persistence_mode=info.persistence_mode,
                    sm_clock_mhz=info.sm_clock_mhz,
                    mem_clock_mhz=info.mem_clock_mhz,
                    power_limit_w=info.power_limit_w,
                    temperature_c=info.temperature_c,
                    throttled=throttled,
                    throttle_reasons=info.throttle_reasons,
                    clocks_locked=False, timestamp_utc=stamp,
                    trimmed=s.trimmed,
                    cov_dtype=sp.cov_dtype,
                    cov_dist=(QBLOCK_DIST if sp.cov_dtype else ""),
                    cov_iters=cov_iters, cov_ops_nominal=cov_ops,
                    window_pad_s=(pads[cfg.warmup + k]
                                  if pads and cfg.warmup + k < len(pads)
                                  else 0.0),
                ))

        for sp in ordered:
            if sp.is_reference:
                samples = measure_workload(meter, work_ref, warmup=1,
                                           repeats=cfg.repeats,
                                           discard_frac=cfg.discard_frac)
                _emit(sp, samples, iters=n_ref, ops=ops_ref, cov_iters=0,
                      cov_ops=0, pads=[])
                continue

            spec, x0, t_one = arm[sp.cov_dtype]
            cap = capacity_frac_of_dose(cov_frac=sp.cov_frac,
                                        declared_frac=cfg.marg_declared_frac)
            assert cap < 1.0, (
                f"dose {sp.cov_frac} infeasible next to declared "
                f"{cfg.marg_declared_frac}")
            n_cov = (0 if sp.cov_frac == 0.0 else
                     max(1, int(round(sp.cov_frac * cfg.marg_window_s
                                      / t_one))))
            cov_ops = ops_nominal_qblock(spec) * n_cov
            step = out_buf = None
            if n_cov:
                step, out_buf = make_qblock(
                    dataclasses.replace(spec, iters=n_cov), qw, x0,
                    device=device)

            pads: list[float] = []

            def window() -> None:
                t_start = time.perf_counter()
                work_decl()
                if step is not None:
                    step()
                torch.cuda.synchronize()
                pad = cfg.marg_window_s - (time.perf_counter() - t_start)
                pads.append(max(0.0, pad))
                if pad <= 0.0:
                    print(f"  WARNING: window underflowed pad by {-pad:.2f}s "
                          f"(arm {sp.cov_dtype} f={sp.cov_frac})", flush=True)
                else:
                    time.sleep(pad)

            print(f"  cell arm={sp.cov_dtype:>16s} f={sp.cov_frac:.2f} "
                  f"n_cov={n_cov}", flush=True)
            samples = measure_workload(meter, window, warmup=cfg.warmup,
                                       repeats=cfg.repeats,
                                       discard_frac=cfg.discard_frac)
            if out_buf is not None:
                assert_finite(out_buf, f"measured qblock-marg output "
                                       f"(arm {sp.cov_dtype}, dose {n_cov})")
            _emit(sp, samples, iters=n_decl, ops=ops_decl, cov_iters=n_cov,
                  cov_ops=cov_ops, pads=pads)
            del step, out_buf
            torch.cuda.empty_cache()
    finally:
        meter.close()

    base = out_dir / f"qblock_marginal_{sweep_id}"
    to_jsonl(records, base.with_suffix(".jsonl"))
    to_csv(records, base.with_suffix(".csv"))
    manifest = build_manifest(
        sweep_id=sweep_id, experiment=QBLOCK_MARG_EXPERIMENT,
        config=dataclasses.asdict(cfg),
        gpu_uuid=records[-1].gpu_uuid if records else "unknown",
        gpu_name=records[-1].gpu_name if records else "unknown",
        driver_version=records[-1].driver_version if records else "unknown",
        git_commit=analysis_git_commit(),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(records),
        outputs=[base.with_suffix(".jsonl").name, base.with_suffix(".csv").name],
        extra={
            "weights_provenance": w.provenance,
            "acts_sha256": extraction_manifest["files"]["acts"]["sha256"],
            "validation": validation,
        },
    )
    (base.parent / f"{base.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return base.with_suffix(".jsonl")
