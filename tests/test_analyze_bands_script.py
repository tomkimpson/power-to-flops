"""CLI-layer helpers in scripts/analyze_bands.py (referee B5).

Only the qblock provenance reader lives in the script (file I/O over the
manifest sidecars is a CLI concern; the band math is tested against
powertoflops.bench.analysis). Loaded via importlib because scripts/ is not a package.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "analyze_bands.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_bands_cli", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_capture(tmp_path, sweep, uuid, weights_sha, *, with_manifest=True):
    """A qblock JSONL + its sibling run manifest (run_qblock's on-disk shape)."""
    jsonl = tmp_path / f"qblock_{sweep}.jsonl"
    jsonl.write_text("")
    if with_manifest:
        manifest = tmp_path / f"qblock_{sweep}_manifest.json"
        manifest.write_text(json.dumps({
            "sweep_id": sweep,
            "gpu_uuid": uuid,
            "extra": {
                "weights_provenance": {
                    "model": "EleutherAI/pythia-6.9b", "revision": "abc",
                    "layer": 15, "weights_sha256": weights_sha,
                },
                "acts_sha256": "cafe",
                "validation": {"cosine": 0.99, "all_finite": True},
            },
        }))
    return jsonl


def test_qblock_provenance_none_when_no_paths():
    mod = _load_script()
    assert mod._qblock_provenance([]) is None


def test_qblock_provenance_none_when_no_manifest(tmp_path):
    mod = _load_script()
    jsonl = _write_capture(tmp_path, "s0", "GPU-0", "sha0", with_manifest=False)
    assert mod._qblock_provenance([str(jsonl)]) is None


def test_qblock_provenance_reads_weights_and_captures(tmp_path):
    mod = _load_script()
    jsonl = _write_capture(tmp_path, "s1", "GPU-1", "shaA")
    prov = mod._qblock_provenance([str(jsonl)])
    assert prov["model"] == "EleutherAI/pythia-6.9b"
    assert prov["weights_sha256"] == "shaA"
    assert prov["acts_sha256"] == "cafe"
    assert len(prov["captures"]) == 1
    cap = prov["captures"][0]
    assert cap["sweep_id"] == "s1"
    assert cap["gpu_uuid"] == "GPU-1"
    assert cap["validation"]["all_finite"] is True


def test_qblock_provenance_merges_multi_card_captures(tmp_path):
    mod = _load_script()
    a = _write_capture(tmp_path, "s2", "GPU-2", "shaSAME")
    b = _write_capture(tmp_path, "s3", "GPU-3", "shaSAME")
    prov = mod._qblock_provenance([str(a), str(b)])
    assert prov["weights_sha256"] == "shaSAME"
    assert {c["gpu_uuid"] for c in prov["captures"]} == {"GPU-2", "GPU-3"}


def test_qblock_provenance_rejects_mismatched_weights(tmp_path):
    mod = _load_script()
    a = _write_capture(tmp_path, "s4", "GPU-4", "shaX")
    b = _write_capture(tmp_path, "s5", "GPU-5", "shaY")
    with pytest.raises(SystemExit, match="one trained-weight artifact"):
        mod._qblock_provenance([str(a), str(b)])
