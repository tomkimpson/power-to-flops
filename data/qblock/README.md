# qblock trained-weight artifacts

The useful-operand band (`r_useful_lo`/`r_useful_hi`, capability-ladder rungs
4–5) is priced from an int8-quantized forward of **one trained Pythia-6.9B
transformer layer**. Uniform-random weights will not do: the energy per operation
depends on operand statistics, so pricing "covert work on useful data" requires
operands that are actually useful. Three artifacts drive it:

| file | tracked? | contents |
|------|----------|----------|
| `pythia-6.9b_layer15_weights.npz`   | no (`*.npz` gitignored) | 16 tensors of layer 15, weights stored `[in, out]` |
| `pythia-6.9b_layer15_acts.npz`      | no (`*.npz` gitignored) | real hidden states `[n_seg, seg_len, d]` feeding that layer |
| `pythia-6.9b_layer15_manifest.json` | **yes** | model/revision/layer + shapes + **sha256 of both `.npz`** |

The `.npz` files are large and local-only; the **manifest is committed** and
pins their sha256. `powertoflops.bench.qweights` refuses to load a `.npz` whose hash
does not match the manifest, and the benchmark hard-fails rather than fall back
to synthetic weights — so a measurement can never silently run on stale or
random data.

## Regenerate (network-connected node)

```bash
python scripts/fetch_qblock_weights.py --out data/qblock \
    --model EleutherAI/pythia-6.9b --revision main --layer 15 \
    --n-seg 8 --seg-len 2048 --dtype float16 --device cuda
```

This downloads the model, extracts layer 15's parameters, captures the real
activations at its input from a `wikitext-103-raw-v1` corpus (override with
`--corpus-file`), writes all three files, and **cross-checks** that the fp32
reference of the layer reproduces the model's own `hidden_states[L+1]` before
it trusts the extraction. Commit the regenerated `manifest.json` if the hashes
change; never commit the `.npz`.

### On OzSTAR (offline compute node)

The GPU/CPU compute nodes have no network, so pre-cache on a login node first:

```bash
export HF_HOME=/fred/oz022/tkimpson/hf_cache
hf download EleutherAI/pythia-6.9b --revision main          # ~14 GB (login)
```

then materialise the corpus deterministically (login node has `datasets`):

```python
# data/qblock/wikitext103_corpus.txt — first ~400k chars of wikitext-103 train
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
buf, N = [], 0
for t in ds["text"]:
    if t and t.strip():
        buf.append(t.strip()); N += len(t)
        if N >= 400_000: break
open("data/qblock/wikitext103_corpus.txt", "w").write("\n\n".join(buf))
```

The corpus `.txt` is gitignored (regenerable); its sha256 is pinned in the
manifest. Submit the extraction offline with `HF_HUB_OFFLINE=1` — see
`scripts/fetch_qblock_weights.slurm` (GPU) or run on a CPU node with
`--dtype float32 --device cpu` (fp16 matmul is unimplemented on CPU).

## Downstream

```bash
# 1. capture on OzSTAR A2 (needs the .npz present, hashes matching):
python scripts/run_qblock.py --preflight-only      # validate load + forward
python scripts/run_qblock.py --out results/bench    # measure the cells
# 2. land the one-sided-LCB useful band in measured_bands.json:
python scripts/analyze_bands.py --exp1 ... --exp2 ... --exp3 ... \
    --qblock results/bench/qblock_<sweep>.jsonl
```
