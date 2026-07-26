"""Exp-7 sweep construction and dose feasibility (pure parts; no GPU).

The GPU harness itself (powertoflops.bench.marginal.run_marginal) is exercised by the
gpu-marked smoke test at the bottom, inert on login nodes.
"""

from __future__ import annotations

import dataclasses

import pytest

from powertoflops.bench.marginal import (
    MAX_BUSY_FRAC,
    MARGINAL_EXPERIMENT,
    build_dose_sweep,
    build_marginal_sweep,
    capacity_frac_of_dose,
)
from powertoflops.config import DEFAULT


def test_experiment_number_is_seven():
    # 4 = core, 5 = pair, 6 = qblock are taken; the marginal capture is 7.
    assert MARGINAL_EXPERIMENT == 7


def test_sweep_is_formats_by_doses_with_zero_anchors():
    cfg = DEFAULT.bench
    pts = build_marginal_sweep(cfg)
    assert len(pts) == len(cfg.marg_cov_dtypes) * len(cfg.marg_cov_fracs)
    for dtype in cfg.marg_cov_dtypes:
        doses = sorted(p.cov_frac for p in pts if p.cov_dtype == dtype)
        assert doses == sorted(cfg.marg_cov_fracs)
        # each arm carries its own zero-dose intercept anchor, format-tagged
        assert any(p.cov_frac == 0.0 for p in pts if p.cov_dtype == dtype)
    assert not any(p.is_reference for p in pts)


def test_default_grid_never_underflows_the_pad():
    cfg = DEFAULT.bench
    assert cfg.marg_declared_frac + max(cfg.marg_cov_fracs) <= MAX_BUSY_FRAC
    for p in build_marginal_sweep(cfg):
        cap = capacity_frac_of_dose(cov_frac=p.cov_frac,
                                    declared_frac=cfg.marg_declared_frac)
        assert cap < 1.0          # every dose is capacity-feasible


def test_overfull_grid_is_refused():
    cfg = dataclasses.replace(DEFAULT.bench, marg_cov_fracs=(0.0, 0.65))
    with pytest.raises(ValueError, match="pad would underflow"):
        build_marginal_sweep(cfg)


def test_grid_without_zero_dose_is_refused():
    cfg = dataclasses.replace(DEFAULT.bench, marg_cov_fracs=(0.06, 0.12, 0.24))
    with pytest.raises(ValueError, match="zero dose"):
        build_marginal_sweep(cfg)


def test_unknown_covert_format_is_refused():
    cfg = dataclasses.replace(DEFAULT.bench, marg_cov_dtypes=("int8", "fp4"))
    with pytest.raises(ValueError, match="fp4"):
        build_marginal_sweep(cfg)


def test_build_dose_sweep_is_arms_by_doses_with_zero_anchors():
    # The format-agnostic core (qblock-marginal / qblock uses cell labels as arms).
    arms = ("qblock_512x8", "qblock_2048x8")
    fracs = (0.0, 0.1, 0.2, 0.3, 0.4)
    pts = build_dose_sweep(arms, fracs, declared_frac=0.30)
    assert len(pts) == len(arms) * len(fracs)
    for a in arms:
        doses = sorted(p.cov_frac for p in pts if p.cov_dtype == a)
        assert doses == sorted(fracs)
        assert any(p.cov_frac == 0.0 for p in pts if p.cov_dtype == a)
    assert not any(p.is_reference for p in pts)


def test_build_dose_sweep_refuses_pad_underflow_and_missing_zero():
    with pytest.raises(ValueError, match="pad would underflow"):
        build_dose_sweep(("a",), (0.0, 0.65), declared_frac=0.30)
    with pytest.raises(ValueError, match="zero dose"):
        build_dose_sweep(("a",), (0.1, 0.2), declared_frac=0.30)


def test_marginal_sweep_matches_dose_sweep_core():
    # build_marginal_sweep must delegate to the shared core unchanged.
    cfg = DEFAULT.bench
    assert build_marginal_sweep(cfg) == build_dose_sweep(
        cfg.marg_cov_dtypes, cfg.marg_cov_fracs, cfg.marg_declared_frac)


@pytest.mark.gpu
def test_run_marginal_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pynvml")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from powertoflops.bench.marginal import run_marginal
    from powertoflops.bench.records import read_jsonl

    # Tiny grid: 2 doses x 1 format, short windows — minutes, not half an hour.
    cfg = dataclasses.replace(
        DEFAULT.bench,
        marg_window_s=3.0, marg_cov_dtypes=("int8",),
        marg_cov_fracs=(0.0, 0.2, 0.4), repeats=2, warmup=1,
    )
    path = run_marginal(tmp_path, cfg)
    recs = [r for r in read_jsonl(path) if r.experiment == MARGINAL_EXPERIMENT]
    assert len(recs) == 3 * 2
    doses = {r.cov_iters for r in recs}
    assert 0 in doses and len(doses) == 3
    for r in recs:
        assert r.energy_j > 0.0
        assert r.cov_dtype == "int8"
        assert abs(r.duration_s - cfg.marg_window_s) < 1.0
        assert r.ops_nominal > 0            # declared burst is described
        assert (r.cov_ops_nominal > 0) == (r.cov_iters > 0)
