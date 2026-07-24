"""Real-weight loading for the qblock benchmark (referee B5): NeoX QKV split,
sha256-verified artifact loaders, per-tensor weight quantization.

All CPU (no GPU marker): the loaders and the split are pure tensor plumbing.
The real Pythia-6.9B artifact is exercised by the fetch script's cross-check
and the GPU pre-flight; these tests use hand-built fakes.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powertoflops.bench.qweights import (  # noqa: E402
    QBlockWeights,
    load_qblock_activations,
    load_qblock_weights,
    quantize_weights,
    split_neox_qkv,
    synthetic_weights,
)


# ---------------------------------------------------------------- QKV split

def _interleave_neox(wq, wk, wv, bq, bk, bv, n_head):
    """Build the HF GPT-NeoX fused query_key_value layout from separates.

    HF views the fused [3d, d] weight as [n_head, 3*head_dim, d] and slices
    q/k/v PER HEAD — the rows are head-interleaved, not three [d, d] blocks.
    """
    d = wq.shape[0]
    dh = d // n_head
    rows = []
    brows = []
    for h in range(n_head):
        rows += [wq[h * dh:(h + 1) * dh], wk[h * dh:(h + 1) * dh],
                 wv[h * dh:(h + 1) * dh]]
        brows += [bq[h * dh:(h + 1) * dh], bk[h * dh:(h + 1) * dh],
                  bv[h * dh:(h + 1) * dh]]
    return torch.cat(rows), torch.cat(brows)


def test_split_neox_qkv_recovers_per_head_interleaved_tensors():
    d, n_head = 8, 2
    g = torch.Generator().manual_seed(0)
    wq0 = torch.randn(d, d, generator=g)
    wk0 = torch.randn(d, d, generator=g)
    wv0 = torch.randn(d, d, generator=g)
    bq0 = torch.randn(d, generator=g)
    bk0 = torch.randn(d, generator=g)
    bv0 = torch.randn(d, generator=g)
    qkv_w, qkv_b = _interleave_neox(wq0, wk0, wv0, bq0, bk0, bv0, n_head)
    assert qkv_w.shape == (3 * d, d)

    wq, wk, wv, bq, bk, bv = split_neox_qkv(qkv_w, qkv_b, n_head)
    assert torch.equal(wq, wq0) and torch.equal(wk, wk0)
    assert torch.equal(wv, wv0)
    assert torch.equal(bq, bq0) and torch.equal(bk, bk0)
    assert torch.equal(bv, bv0)


def test_split_neox_qkv_is_not_a_block_chunk():
    # A naive chunk(3, dim=0) split is WRONG for n_head > 1; pin that the
    # per-head split differs from it on a marked tensor.
    d, n_head = 8, 2
    qkv_w = torch.arange(3 * d * d, dtype=torch.float32).reshape(3 * d, d)
    qkv_b = torch.arange(3 * d, dtype=torch.float32)
    wq, _, _, bq, _, _ = split_neox_qkv(qkv_w, qkv_b, n_head)
    chunk_q = qkv_w.chunk(3, dim=0)[0]
    assert not torch.equal(wq, chunk_q)
    # head 0 rows 0..3 are q, head 1's q rows start at 3*dh = 12
    dh = d // n_head
    assert torch.equal(wq[:dh], qkv_w[:dh])
    assert torch.equal(wq[dh:], qkv_w[3 * dh:4 * dh])
    assert torch.equal(bq[dh:], qkv_b[3 * dh:4 * dh])


def test_split_neox_qkv_rejects_bad_shapes():
    with pytest.raises(ValueError):
        split_neox_qkv(torch.zeros(10, 4), torch.zeros(10), n_head=2)


# ---------------------------------------------------------- synthetic weights

def test_synthetic_weights_shapes_and_metadata():
    g = torch.Generator().manual_seed(0)
    w = synthetic_weights(d_model=16, n_head=4, ffn_mult=2, generator=g)
    assert isinstance(w, QBlockWeights)
    assert w.w_q.shape == (16, 16)          # [in, out]
    assert w.w_up.shape == (16, 32)
    assert w.w_down.shape == (32, 16)
    assert w.b_up.shape == (32,)
    assert w.ln1_w.shape == (16,)
    assert w.rotary_pct == 0.25
    assert w.ln_eps > 0
    assert w.provenance.get("source") == "synthetic"


# -------------------------------------------------------- weight quantization

def test_quantize_weights_roundtrip_and_tn_layout():
    g = torch.Generator().manual_seed(1)
    w = synthetic_weights(d_model=16, n_head=4, ffn_mult=2, generator=g)
    qw = quantize_weights(w)
    for name in ("q", "k", "v", "o", "up", "down"):
        w_i8 = getattr(qw, f"w_{name}_i8")
        scale = getattr(qw, f"s_{name}")
        ref = getattr(w, f"w_{name}")
        assert w_i8.dtype == torch.int8
        assert w_i8.shape == ref.shape
        assert scale > 0
        # dequantized values track the originals within one quantization step
        assert torch.allclose(w_i8.float() * scale, ref.float(), atol=scale)
        # column-major TN layout for the IMMA tensor-core path
        assert w_i8.stride() == (1, w_i8.shape[0])
    # fp params ride along for the dequant/bias/LN path
    assert torch.equal(qw.weights.b_q, w.b_q)


# ------------------------------------------------------------------- loaders

def _write_artifact(tmp_path, *, d=16, n_head=4, ffn_mult=2,
                    n_seg=3, seg_len=8, corrupt=False):
    """Fake weights npz + activations npz + committed-style manifest."""
    fd = ffn_mult * d
    rng = np.random.default_rng(0)
    arrays = {
        "w_q": rng.standard_normal((d, d)), "w_k": rng.standard_normal((d, d)),
        "w_v": rng.standard_normal((d, d)), "w_o": rng.standard_normal((d, d)),
        "w_up": rng.standard_normal((d, fd)),
        "w_down": rng.standard_normal((fd, d)),
        "b_q": rng.standard_normal(d), "b_k": rng.standard_normal(d),
        "b_v": rng.standard_normal(d), "b_o": rng.standard_normal(d),
        "b_up": rng.standard_normal(fd), "b_down": rng.standard_normal(d),
        "ln1_w": np.ones(d), "ln1_b": np.zeros(d),
        "ln2_w": np.ones(d), "ln2_b": np.zeros(d),
    }
    weights_npz = tmp_path / "weights.npz"
    np.savez(weights_npz, **{k: v.astype(np.float16) for k, v in arrays.items()})

    acts = rng.standard_normal((n_seg, seg_len, d)).astype(np.float16)
    acts_npz = tmp_path / "acts.npz"
    np.savez(acts_npz, acts=acts)

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "model": "EleutherAI/pythia-6.9b", "revision": "deadbeef", "layer": 15,
        "d_model": d, "n_head": n_head, "ffn_mult": ffn_mult,
        "rotary_pct": 0.25, "ln_eps": 1e-5,
        "files": {
            "weights": {"name": weights_npz.name, "sha256": sha(weights_npz)},
            "acts": {"name": acts_npz.name, "sha256": sha(acts_npz)},
        },
    }
    if corrupt:
        manifest["files"]["weights"]["sha256"] = "0" * 64
        manifest["files"]["acts"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return weights_npz, acts_npz, manifest_path, arrays


def test_load_qblock_weights_verifies_hash_and_shapes(tmp_path):
    weights_npz, _, manifest, arrays = _write_artifact(tmp_path)
    w = load_qblock_weights(weights_npz, manifest)
    assert w.w_q.shape == (16, 16)
    assert w.w_up.shape == (16, 32)
    assert w.w_q.dtype == torch.float32       # upcast for reference math
    assert w.rotary_pct == 0.25
    assert w.ln_eps == 1e-5
    assert w.provenance["model"] == "EleutherAI/pythia-6.9b"
    assert w.provenance["revision"] == "deadbeef"
    # values survive the fp16 round trip
    assert np.allclose(w.w_q.numpy(),
                       arrays["w_q"].astype(np.float16).astype(np.float32))


def test_load_qblock_weights_rejects_hash_mismatch(tmp_path):
    weights_npz, _, manifest, _ = _write_artifact(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="sha256"):
        load_qblock_weights(weights_npz, manifest)


def test_load_qblock_weights_missing_file_names_fetch_script(tmp_path):
    _, _, manifest, _ = _write_artifact(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_qblock_weights"):
        load_qblock_weights(tmp_path / "nope.npz", manifest)


def test_load_qblock_activations_slices_batch_major(tmp_path):
    _, acts_npz, manifest, _ = _write_artifact(tmp_path, n_seg=3, seg_len=8)
    x = load_qblock_activations(acts_npz, manifest, seq=4, batch=2)
    # row order is batch-major (row = b*seq + s), matching make_qblock's view
    assert x.shape == (8, 16)
    assert x.dtype == torch.float32
    raw = np.load(acts_npz)["acts"]
    assert np.allclose(x[:4].numpy(), raw[0, :4].astype(np.float32))
    assert np.allclose(x[4:].numpy(), raw[1, :4].astype(np.float32))


def test_load_qblock_activations_bounds_guards(tmp_path):
    _, acts_npz, manifest, _ = _write_artifact(tmp_path, n_seg=3, seg_len=8)
    with pytest.raises(ValueError, match="seq"):
        load_qblock_activations(acts_npz, manifest, seq=9, batch=1)
    with pytest.raises(ValueError, match="batch"):
        load_qblock_activations(acts_npz, manifest, seq=4, batch=4)


def test_load_qblock_activations_rejects_hash_mismatch(tmp_path):
    _, acts_npz, manifest, _ = _write_artifact(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="sha256"):
        load_qblock_activations(acts_npz, manifest, seq=4, batch=2)
