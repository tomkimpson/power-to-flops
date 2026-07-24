"""R2 sweep-construction tests: build_sweep grids, Exp-3 ordering, idle cells.

Requires torch importable (the runner imports it at module level) but no GPU:
build_sweep and the ordering helpers are declarative, and the idle-cell work
builder accepts device="cpu".
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from powertoflops.bench.runner import (  # noqa: E402
    _build_work,
    build_sweep,
    ordered_points,
)
from powertoflops.config import DEFAULT  # noqa: E402

CFG = DEFAULT.bench


def _small_cfg(**over):
    base = dict(
        exp1_shape=(8, 8, 8),
        locality_buffers=(("resident", 1), ("streamed", 2)),
        exp3_ladder=((8, 8, 8), (16, 16, 16)),
        exp3_idle_window_s=0.01,
        exp3_cooldown_s=0.0,
        target_window_s=0.0,   # disable calibration (no timing on CPU)
    )
    base.update(over)
    return dataclasses.replace(CFG, **base)


# ---- config shape ------------------------------------------------------------


def test_config_carries_sweep_version():
    assert CFG.sweep_version == "r2"


def test_duty_cycle_fields_are_gone():
    # Exp 3 no longer duty-cycle mixes (review C3): the old knobs must not
    # linger as dead config.
    for gone in ("util_fracs", "exp3_period_s", "exp3_n_periods",
                 "shape_cache", "shape_dram", "localities"):
        assert not hasattr(CFG, gone)


# ---- Exp 1: precision x operand x locality factorial ------------------------


def test_exp1_full_factorial():
    pts = build_sweep(1, CFG)
    n_expected = len(CFG.dtypes) * len(CFG.locality_buffers) * len(CFG.operand_dists)
    assert len(pts) == n_expected == 60
    combos = {(p.dtype, p.locality, p.dist_a) for p in pts}
    assert len(combos) == n_expected


def test_exp1_fixed_shape_and_buffer_mapping():
    # Locality is residency (n_buffers), never problem size (review C4).
    pts = build_sweep(1, CFG)
    n_buf = dict(CFG.locality_buffers)
    for p in pts:
        assert (p.M, p.N, p.K) == CFG.exp1_shape
        assert p.n_buffers == n_buf[p.locality]
        assert p.dist_a == p.dist_b
        assert p.util_frac is None
        assert p.sm_clock is None   # dvfs passive (locks denied on OzSTAR)


# ---- Exp 2: operand rounds ---------------------------------------------------


def test_exp2_rounds_repeat_each_dist():
    pts = build_sweep(2, CFG)
    assert len(pts) == CFG.exp2_rounds * len(CFG.operand_dists) == 24
    for dist in CFG.operand_dists:
        assert sum(p.dist_a == dist for p in pts) == CFG.exp2_rounds
    for p in pts:
        assert p.dtype == CFG.exp2_dtype
        assert p.locality == CFG.exp2_locality
        assert (p.M, p.N, p.K) == CFG.exp1_shape


# ---- Exp 3: intrinsic-F ladder + idle variants -------------------------------


def test_exp3_ladder_plus_idle_cells():
    pts = build_sweep(3, CFG)
    ladder = [p for p in pts if not p.idle_variant]
    idles = [p for p in pts if p.idle_variant]
    assert len(ladder) == len(CFG.exp3_ladder) == 10
    assert len(idles) == len(CFG.exp3_idle_variants) == 5
    assert {(p.M, p.N, p.K) for p in ladder} == set(CFG.exp3_ladder)
    for p in ladder:
        assert p.util_frac == 1.0
        assert p.dtype == CFG.exp3_dtype
        assert p.dist_a == "dense_random"
    assert {p.idle_variant for p in idles} == set(CFG.exp3_idle_variants)
    for p in idles:
        assert p.util_frac == 0.0
        assert p.iters == 0


def test_exp3_ordering_pins_idle_variants():
    cfg = _small_cfg()
    ordered = ordered_points(3, build_sweep(3, cfg), cfg)
    variants = [p.idle_variant for p in ordered]
    # no_ctx must precede ANY GPU work (it measures the context-free idle);
    # ctx/cached follow while the card is still cold; post_load/cooled are
    # pinned after the ladder (hot idle, then cooled idle).
    assert variants[0] == "no_ctx"
    assert variants[1] == "ctx"
    assert variants[2] == "cached"
    assert variants[-2:] == ["post_load", "cooled"]
    middle = ordered[3:-2]
    assert all(not p.idle_variant for p in middle)
    assert any(p.is_reference for p in middle)          # drift refs still spliced
    assert sum(not p.is_reference for p in middle) == len(cfg.exp3_ladder)


def test_non_exp3_ordering_unchanged():
    ordered = ordered_points(1, build_sweep(1, CFG), CFG)
    assert ordered[0].is_reference           # ref spliced at the head as before
    assert sum(not p.is_reference for p in ordered) == 60


# ---- Exp 4: cross-device core sweep -------------------------------------------


def test_exp4_core_cells():
    pts = build_sweep(4, CFG)
    dense = [p for p in pts if p.dist_a == "dense_random"]
    zeros = [p for p in pts if p.dist_a == "zeros"]
    assert len(pts) == 8
    assert {(p.dtype, p.locality) for p in dense} == {
        (d, loc) for d in CFG.core_dtypes for loc, _ in CFG.locality_buffers
    }
    assert {(p.dtype, p.locality) for p in zeros} == {
        (CFG.core_zeros_dtype, loc) for loc, _ in CFG.locality_buffers
    }
    for p in pts:
        assert (p.M, p.N, p.K) == CFG.exp1_shape


def test_unknown_experiment_raises():
    with pytest.raises(ValueError):
        build_sweep(7, CFG)


# ---- idle-cell work builder (CPU-safe) ----------------------------------------


@pytest.mark.parametrize("variant", ["no_ctx", "ctx", "cached"])
def test_build_work_idle_cells_do_no_ops(variant):
    cfg = _small_cfg()
    sp = next(p for p in build_sweep(3, cfg) if p.idle_variant == variant)
    work, ops, iters = _build_work(sp, cfg, device="cpu", generator=None)
    assert ops == 0
    assert iters == 0
    work()   # must not raise; sleeps exp3_idle_window_s
