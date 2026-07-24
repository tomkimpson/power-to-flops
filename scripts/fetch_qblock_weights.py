"""Extract one trained Pythia transformer layer for the qblock benchmark (B5).

The retired qblock benchmark drew uniform-random int8 weights; the referee
(item B5) requires actual trained weights and representative activations. This
script runs ON A NETWORK-CONNECTED NODE (it downloads a Hugging Face model),
extracts one GPT-NeoX layer's parameters and the real hidden states feeding it,
and writes the three sha256-pinned artifacts that powertoflops.bench.qweights loads:

    data/qblock/<model>_layer<L>_weights.npz    # 16 tensors, weights [in, out]
    data/qblock/<model>_layer<L>_acts.npz       # acts [n_seg, seg_len, d]
    data/qblock/<model>_layer<L>_manifest.json  # provenance + sha256 of both

Only the manifest is committed (it pins the artifact hashes); the .npz files
are local-only and regenerated here. The runner and band extraction hard-fail
if a hash does not match, so a measurement can never silently run on stale or
synthetic weights.

Weight convention: HF stores every linear weight [out, in]; we transpose to
[in, out] so the reference forward is ``y = x @ w + b`` (matches qweights and
qblock_reference). The fused GPT-NeoX ``query_key_value`` is split PER HEAD via
:func:`powertoflops.bench.qweights.split_neox_qkv` (a naive chunk scrambles heads).

Activations: real hidden states at the INPUT of layer L for ``n_seg`` disjoint
``seg_len``-token segments of a text corpus, captured from ``output_hidden_states``
(``hidden_states[L]``). The next layer's output (``hidden_states[L+1]``) is the
cross-check target.

Cross-check (``--crosscheck``, on by default): the whole extraction is validated
by confirming that our fp32 reference of the layer, applied to ``hidden_states[L]``,
reproduces the model's own ``hidden_states[L+1]`` within tight tolerance. This
consumes only ``output_hidden_states`` (a stable API surface), so it does not
depend on the version-specific signature of the layer's ``forward``.

Usage (network-connected node, GPU optional):
    python scripts/fetch_qblock_weights.py --out data/qblock
    python scripts/fetch_qblock_weights.py --model EleutherAI/pythia-6.9b \\
        --revision main --layer 15 --n-seg 8 --seg-len 2048 --device cuda
    python scripts/fetch_qblock_weights.py --corpus-file mytext.txt   # own corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from powertoflops.bench.qblock import QBlockSpec, qblock_reference, validate_qblock  # noqa: E402
from powertoflops.bench.qweights import (  # noqa: E402
    load_qblock_weights,
    split_neox_qkv,
)


# --------------------------------------------------------------- extraction

def _first_attr(module, names):
    """Return the first present submodule among ``names`` (API drift guard)."""
    for n in names:
        if hasattr(module, n):
            return getattr(module, n), n
    raise AttributeError(
        f"none of {names} on {type(module).__name__}; the transformers "
        f"GPT-NeoX layout may have changed — inspect and extend the name list")


def extract_layer_weights(layer, n_head: int):
    """Pull one GPT-NeoX layer's parameters as [in, out] fp32 tensors.

    Handles both the fused ``query_key_value`` layout and a split
    q/k/v_proj layout, and the ``dense`` vs ``o_proj`` / ``dense_h_to_4h``
    vs ``up_proj`` naming variants across transformers versions.
    """
    attn = layer.attention
    if hasattr(attn, "query_key_value"):
        qkv = attn.query_key_value
        w_q_oi, w_k_oi, w_v_oi, b_q, b_k, b_v = split_neox_qkv(
            qkv.weight.detach().float(), qkv.bias.detach().float(), n_head)
    else:
        qp, _ = _first_attr(attn, ["q_proj"])
        kp, _ = _first_attr(attn, ["k_proj"])
        vp, _ = _first_attr(attn, ["v_proj"])
        w_q_oi, b_q = qp.weight.detach().float(), qp.bias.detach().float()
        w_k_oi, b_k = kp.weight.detach().float(), kp.bias.detach().float()
        w_v_oi, b_v = vp.weight.detach().float(), vp.bias.detach().float()

    o_proj, _ = _first_attr(attn, ["dense", "o_proj", "out_proj"])
    mlp = layer.mlp
    up, _ = _first_attr(mlp, ["dense_h_to_4h", "up_proj"])
    down, _ = _first_attr(mlp, ["dense_4h_to_h", "down_proj"])
    ln1 = layer.input_layernorm
    ln2 = layer.post_attention_layernorm

    def io(w):  # [out, in] -> [in, out]
        return w.detach().float().t().contiguous()

    return {
        "w_q": io(w_q_oi), "w_k": io(w_k_oi), "w_v": io(w_v_oi),
        "w_o": io(o_proj.weight),
        "w_up": io(up.weight), "w_down": io(down.weight),
        "b_q": b_q, "b_k": b_k, "b_v": b_v,
        "b_o": o_proj.bias.detach().float(),
        "b_up": up.bias.detach().float(), "b_down": down.bias.detach().float(),
        "ln1_w": ln1.weight.detach().float(), "ln1_b": ln1.bias.detach().float(),
        "ln2_w": ln2.weight.detach().float(), "ln2_b": ln2.bias.detach().float(),
    }


# ------------------------------------------------------------------- corpus

def _load_corpus_text(args) -> tuple[str, dict]:
    """Return (text, provenance). File takes priority; else a HF dataset."""
    if args.corpus_file:
        p = pathlib.Path(args.corpus_file)
        text = p.read_text(encoding="utf-8", errors="ignore")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, {"source": "file", "path": str(p), "sha256": sha}
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "no --corpus-file given and the `datasets` package is not "
            "installed — either `pip install datasets` or pass "
            "--corpus-file <text>") from exc
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    return text, {"source": "hf-datasets", "dataset": args.dataset,
                  "config": args.dataset_config, "split": args.dataset_split}


def _segment_tokens(text, tokenizer, n_seg, seg_len):
    """Tokenize ``text`` and cut it into ``n_seg`` disjoint ``seg_len`` blocks."""
    ids = tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    need = n_seg * seg_len
    if ids.shape[0] < need:
        raise SystemExit(
            f"corpus tokenizes to {ids.shape[0]} tokens; need "
            f"{need} (= {n_seg} x {seg_len}) — supply a larger corpus")
    return ids[:need].reshape(n_seg, seg_len)


# ------------------------------------------------------------------- manifest

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-6.9b")
    ap.add_argument("--revision", default="main",
                    help="HF model revision (pin a step for reproducibility)")
    ap.add_argument("--layer", type=int, default=15, help="0-indexed layer")
    ap.add_argument("--out", default="data/qblock", help="output directory")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="model + activation dtype (fp16 halves memory)")
    ap.add_argument("--n-seg", type=int, default=8,
                    help="activation segments (>= max qblock batch)")
    ap.add_argument("--seg-len", type=int, default=2048,
                    help="tokens per segment (>= max qblock seq)")
    ap.add_argument("--corpus-file", default=None,
                    help="UTF-8 text file for activations (else a HF dataset)")
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--no-crosscheck", action="store_true",
                    help="skip the reference-vs-model layer validation")
    ap.add_argument("--crosscheck-seq", type=int, default=256)
    ap.add_argument("--crosscheck-batch", type=int, default=4)
    ap.add_argument("--min-cosine", type=float, default=0.998)
    ap.add_argument("--max-rel-l2", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    base = f"{short}_layer{args.layer}"
    weights_npz = out_dir / f"{base}_weights.npz"
    acts_npz = out_dir / f"{base}_acts.npz"
    manifest_path = out_dir / f"{base}_manifest.json"

    print(f"loading {args.model}@{args.revision} ({args.dtype}) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch_dtype,
        output_hidden_states=True)
    model.eval().to(device)

    cfg = model.config
    n_head = cfg.num_attention_heads
    d_model = cfg.hidden_size
    ffn_mult = cfg.intermediate_size // d_model
    n_layers = cfg.num_hidden_layers
    if not 0 <= args.layer < n_layers:
        raise SystemExit(f"--layer {args.layer} out of range [0, {n_layers})")
    ln_eps = float(getattr(cfg, "layer_norm_eps", 1e-5))
    core = model.gpt_neox if hasattr(model, "gpt_neox") else model.base_model
    layer = core.layers[args.layer]
    # Rotary fraction: read the model's ACTUAL rotary_ndims. transformers moved
    # the fraction into config.rope_parameters["partial_rotary_factor"], so
    # getattr(cfg, "rotary_pct") silently defaults to 1.0 on 5.x and rotates
    # every head dim (32 -> 128 for Pythia) — the reference then disagrees with
    # the model. The attention module is the ground truth.
    head_size = d_model // n_head
    rotary_ndims_model = getattr(layer.attention, "rotary_ndims", None)
    if rotary_ndims_model is None:
        raise SystemExit(
            "cannot read layer.attention.rotary_ndims — inspect the "
            "transformers GPT-NeoX attention module and set rotary_pct by hand")
    rotary_pct = rotary_ndims_model / head_size

    # --- weights
    w = extract_layer_weights(layer, n_head)
    np.savez(weights_npz,
             **{k: v.to(torch.float16).cpu().numpy() for k, v in w.items()})
    print(f"wrote {weights_npz.name}  ({d_model=} {n_head=} {ffn_mult=})")

    # --- activations (hidden_states[L]) + crosscheck target (hidden_states[L+1])
    text, corpus_prov = _load_corpus_text(args)
    seg_ids = _segment_tokens(text, tokenizer, args.n_seg, args.seg_len)
    acts = np.empty((args.n_seg, args.seg_len, d_model), dtype=np.float16)
    hs_next = np.empty_like(acts)
    with torch.no_grad():
        for i in range(args.n_seg):
            ids = torch.from_numpy(seg_ids[i][None, :]).to(device)
            hs = model(input_ids=ids).hidden_states  # tuple, len n_layers+1
            acts[i] = hs[args.layer][0].float().cpu().numpy()
            hs_next[i] = hs[args.layer + 1][0].float().cpu().numpy()
    np.savez(acts_npz, acts=acts)
    print(f"wrote {acts_npz.name}  ({args.n_seg} x {args.seg_len} x {d_model})")

    # --- manifest (pins both hashes; loader verifies against it)
    manifest = {
        "model": args.model, "revision": args.revision, "layer": args.layer,
        "d_model": d_model, "n_head": n_head, "ffn_mult": ffn_mult,
        "rotary_pct": rotary_pct, "ln_eps": ln_eps,
        "corpus": corpus_prov,
        "files": {
            "weights": {"name": weights_npz.name, "sha256": _sha256(weights_npz)},
            "acts": {"name": acts_npz.name, "sha256": _sha256(acts_npz)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path.name}")

    # --- crosscheck: reference of the layer reproduces the model's own output
    if not args.no_crosscheck:
        wl = load_qblock_weights(weights_npz, manifest_path)  # round-trips the loader
        b, s = args.crosscheck_batch, args.crosscheck_seq
        x0 = torch.from_numpy(acts[:b, :s].reshape(b * s, d_model)).float()
        spec = QBlockSpec(d_model=d_model, n_head=n_head, ffn_mult=ffn_mult,
                          seq=s, batch=b, iters=1)
        ref = qblock_reference(x0, wl, spec)
        target = torch.from_numpy(hs_next[:b, :s].reshape(b * s, d_model)).float()
        m = validate_qblock(ref, target)
        print(f"crosscheck vs model hidden_states[L+1]: cosine={m['cosine']:.5f} "
              f"rel_l2={m['rel_l2']:.5f} finite={m['all_finite']}")
        if (not m["all_finite"] or m["cosine"] < args.min_cosine
                or m["rel_l2"] > args.max_rel_l2):
            raise SystemExit(
                f"crosscheck FAILED (cosine >= {args.min_cosine}, "
                f"rel_l2 <= {args.max_rel_l2}) — the extracted layer does not "
                f"reproduce the model; artifacts written but DO NOT commit the "
                f"manifest until this passes")
    print("done.")


if __name__ == "__main__":
    main()
