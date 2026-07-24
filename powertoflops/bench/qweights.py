"""Trained-weight loading for the qblock benchmark (referee B5).

The invalidated benchmark drew uniform-random int8 weights; the manuscript's
"real weight and activation statistics" claim requires actual trained weights.
This module loads one extracted Pythia-6.9B transformer layer (produced by
``scripts/fetch_qblock_weights.py`` on a network-connected node) and prepares
it for the int8 measurement path:

  * :func:`split_neox_qkv` — the HF GPT-NeoX fused ``query_key_value`` tensor
    is PER-HEAD interleaved ([n_head, 3*head_dim, d]), not three stacked
    [d, d] blocks; a naive ``chunk(3)`` silently scrambles heads.
  * :func:`load_qblock_weights` / :func:`load_qblock_activations` — read the
    local ``.npz`` artifacts and hard-fail unless their sha256 matches the
    committed extraction manifest. There is deliberately NO random fallback:
    a measurement must never silently run on synthetic weights again.
  * :func:`quantize_weights` — per-tensor symmetric int8 with the scales KEPT
    (the B5 defect was discarding them), plus the column-major TN layout the
    IMMA tensor-core path requires.
  * :func:`synthetic_weights` — Gaussian stand-ins for unit tests only.

Weight convention: every linear weight is stored ``[in, out]`` so the forward
is ``y = x @ w + b`` (HF stores ``[out, in]``; the fetch script transposes).
Activation convention: rows are batch-major, ``row = b * seq + s``, matching
``make_qblock``'s head reshape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# npz keys of the 16 tensors an extracted layer carries.
_WEIGHT_KEYS = ("w_q", "w_k", "w_v", "w_o", "w_up", "w_down")
_BIAS_KEYS = ("b_q", "b_k", "b_v", "b_o", "b_up", "b_down")
_LN_KEYS = ("ln1_w", "ln1_b", "ln2_w", "ln2_b")


@dataclass(frozen=True)
class QBlockWeights:
    """One transformer layer's parameters (fp32, weights ``[in, out]``)."""

    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    b_q: torch.Tensor
    b_k: torch.Tensor
    b_v: torch.Tensor
    b_o: torch.Tensor
    b_up: torch.Tensor
    b_down: torch.Tensor
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    rotary_pct: float
    ln_eps: float
    provenance: dict = field(default_factory=dict)

    @property
    def d_model(self) -> int:
        return self.w_q.shape[0]


@dataclass(frozen=True)
class QuantizedQBlockWeights:
    """Per-tensor symmetric int8 matmul weights + their scales (B5 fix: the
    scales are carried and APPLIED at dequant, never discarded)."""

    weights: QBlockWeights          # fp params: biases, layernorms, metadata
    w_q_i8: torch.Tensor
    w_k_i8: torch.Tensor
    w_v_i8: torch.Tensor
    w_o_i8: torch.Tensor
    w_up_i8: torch.Tensor
    w_down_i8: torch.Tensor
    s_q: float
    s_k: float
    s_v: float
    s_o: float
    s_up: float
    s_down: float


def split_neox_qkv(
    qkv_w: torch.Tensor, qkv_b: torch.Tensor, n_head: int,
) -> tuple[torch.Tensor, ...]:
    """Split HF GPT-NeoX's fused query_key_value into per-projection tensors.

    ``qkv_w`` is the HF Linear weight [3*d, d] (out, in); HF views the fused
    output as [n_head, 3*head_dim] and slices q/k/v per head, so the rows are
    head-interleaved. Returns (w_q, w_k, w_v, b_q, b_k, b_v) with the weights
    still in HF [out, in] = [d, d] convention.
    """
    three_d, d = qkv_w.shape
    if three_d != 3 * d or d % n_head != 0:
        raise ValueError(
            f"split_neox_qkv: expected [3*d, d] with d divisible by n_head, "
            f"got {tuple(qkv_w.shape)} with n_head={n_head}")
    dh = d // n_head
    w = qkv_w.view(n_head, 3, dh, d)
    b = qkv_b.view(n_head, 3, dh)
    w_q, w_k, w_v = (w[:, j].reshape(d, d) for j in range(3))
    b_q, b_k, b_v = (b[:, j].reshape(d) for j in range(3))
    return w_q, w_k, w_v, b_q, b_k, b_v


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_against_manifest(
    path: Path, manifest_path: Path, file_key: str,
) -> dict:
    """Check ``path`` exists and hash-matches the manifest; return the manifest."""
    path, manifest_path = Path(path), Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — the extracted-layer artifacts are local-only "
            f"(not committed); regenerate with scripts/fetch_qblock_weights.py "
            f"on a network-connected node")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    want = manifest["files"][file_key]["sha256"]
    got = _sha256(path)
    if got != want:
        raise ValueError(
            f"{path} sha256 mismatch: manifest {manifest_path} pins "
            f"{want}, file has {got} — stale or corrupted artifact; re-run "
            f"scripts/fetch_qblock_weights.py")
    return manifest


def load_qblock_weights(
    npz_path: str | Path, manifest_path: str | Path,
) -> QBlockWeights:
    """Load an extracted layer's weights, sha256-verified against the manifest.

    Hard failure on a missing or mismatched artifact — measurements must never
    fall back to random weights (the retired B5 behaviour).
    """
    npz_path = Path(npz_path)
    manifest = _verify_against_manifest(npz_path, Path(manifest_path), "weights")
    data = np.load(npz_path)
    tensors = {k: torch.from_numpy(np.asarray(data[k], dtype=np.float32))
               for k in (*_WEIGHT_KEYS, *_BIAS_KEYS, *_LN_KEYS)}
    provenance = {k: manifest[k] for k in
                  ("model", "revision", "layer") if k in manifest}
    provenance["weights_sha256"] = manifest["files"]["weights"]["sha256"]
    return QBlockWeights(
        **tensors,
        rotary_pct=float(manifest["rotary_pct"]),
        ln_eps=float(manifest["ln_eps"]),
        provenance=provenance,
    )


def load_qblock_activations(
    npz_path: str | Path, manifest_path: str | Path, seq: int, batch: int,
) -> torch.Tensor:
    """Real hidden-state activations for one (seq, batch) cell, fp32.

    The artifact stores ``acts`` of shape [n_seg, seg_len, d] — hidden states
    of disjoint real-text segments at the extracted layer's input. Returns
    [batch*seq, d] in batch-major row order (row = b*seq + s).
    """
    npz_path = Path(npz_path)
    _verify_against_manifest(npz_path, Path(manifest_path), "acts")
    acts = np.asarray(np.load(npz_path)["acts"])
    n_seg, seg_len, _d = acts.shape
    if seq > seg_len:
        raise ValueError(
            f"seq={seq} exceeds the stored segment length {seg_len}")
    if batch > n_seg:
        raise ValueError(
            f"batch={batch} exceeds the {n_seg} stored segments")
    x = acts[:batch, :seq].astype(np.float32).reshape(batch * seq, -1)
    return torch.from_numpy(x)


def _tn(w: torch.Tensor) -> torch.Tensor:
    """Column-major B for IMMA tensor-core dispatch (workload.py issue #25)."""
    return w.t().contiguous().t()


def quantize_weights(w: QBlockWeights) -> QuantizedQBlockWeights:
    """Per-tensor symmetric int8 quantization of the six matmul weights.

    Scales are computed from the fp32 values and RETURNED — the forward path
    multiplies them back at dequant (the B5 fix). Per-tensor is the scheme the
    manuscript describes; if the validation criterion fails on real weights,
    the documented fallback is per-output-channel scales (the dequant then
    multiplies a row vector instead of a scalar).
    """
    from .qblock import quantize_sym_int8   # single quantizer, one definition

    q: dict[str, torch.Tensor] = {}
    s: dict[str, float] = {}
    for name in ("q", "k", "v", "o", "up", "down"):
        w_i8, scale = quantize_sym_int8(getattr(w, f"w_{name}").float())
        q[f"w_{name}_i8"] = _tn(w_i8)
        s[f"s_{name}"] = scale
    return QuantizedQBlockWeights(weights=w, **q, **s)


def synthetic_weights(
    d_model: int, n_head: int, ffn_mult: int,
    generator: torch.Generator | None = None,
    *,
    rotary_pct: float = 0.25,
    ln_eps: float = 1e-5,
) -> QBlockWeights:
    """Gaussian stand-in weights — TESTS ONLY, never a measurement input.

    Zero-mean Gaussian (std 0.02, the usual transformer init scale) so the
    quantized-vs-reference validation is exercised on plausible magnitudes;
    real captures load the Pythia artifact via :func:`load_qblock_weights`.
    """
    if d_model % n_head != 0:
        raise ValueError("d_model must be divisible by n_head")
    fd = ffn_mult * d_model

    def rand(*shape):
        return 0.02 * torch.randn(*shape, generator=generator)

    return QBlockWeights(
        w_q=rand(d_model, d_model), w_k=rand(d_model, d_model),
        w_v=rand(d_model, d_model), w_o=rand(d_model, d_model),
        w_up=rand(d_model, fd), w_down=rand(fd, d_model),
        b_q=rand(d_model), b_k=rand(d_model), b_v=rand(d_model),
        b_o=rand(d_model), b_up=rand(fd), b_down=rand(d_model),
        ln1_w=torch.ones(d_model), ln1_b=torch.zeros(d_model),
        ln2_w=torch.ones(d_model), ln2_b=torch.zeros(d_model),
        rotary_pct=rotary_pct, ln_eps=ln_eps,
        provenance={"source": "synthetic"},
    )
