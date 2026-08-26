"""A2 single-GPU bench harness — measure the exchange-rate bands on real silicon.

The closed form reports the verification floor beta in terms of three band
widths: the declared exchange-rate band r_hi/r_lo, the covert rate r_lo^cov,
and the overhead band dE0. The covert rate is a MARGINAL dose slope (capture
exp7), never an Exp-2 total quotient; the Exp-2 operand core
w0 = r_hi^0/r_lo^0 is descriptive and prices nothing.
This package runs the manuscript's Experiments 1-5 on NVIDIA A100s and extracts
those bands, replacing the Horowitz-2014 placeholders in
:mod:`powertoflops.config` with measured numbers (wired in via :mod:`powertoflops.measured`).

NAMING: throughout this package "Exp-N" is the CAPTURE IDENTIFIER -- the digit in
the record filename -- and NOT the manuscript's experiment number. The two
schemes deliberately disagree: captures are numbered in the order they were
written, manuscript experiments in the order the argument needs them. The map is
    exp1 -> Experiment 1 (exchange band / factor widths)
    exp3 -> Experiment 2 (overhead band)
    exp7 -> Experiment 3 (marginal covert cost)
    qblock_marginal -> Experiment 4 (marginal useful cost)
    exp2 -> Experiment 5 (operand sweep; appendix only, prices nothing)
    exp4_core, qblock, matched_pair -> unnumbered in the manuscript
This map is the authority (the README points here). Do not "fix" a capture id
to match a manuscript number.

NVML is a *proxy* meter: on-chip and prover-owned, which is the untrusted
channel the framework distinguishes from a trusted off-chip meter. For the
Amount query this is largely benign — we integrate to energy — and the paper
states the assumption explicitly; characterising the on-chip/off-chip gap needs
external metering hardware and is out of scope here.

Submodules:
    records   RunRecord schema + JSONL/CSV/manifest (de)serialisation [pure]
    bands     records -> measured bands (r_hi/r_lo, w0, dE0)    [pure, no GPU]
    analysis  bands + provenance payload, mixed-UUID guard      [pure, no GPU]
    workload  matmul builder, operand generators, FLOP count    [torch]
    meter     NVML energy/power meter + measurement loop         [torch + NVML]
    device    device introspection + (graceful) DVFS control     [NVML]
    runner    sweep construction + experiment orchestration      [torch + NVML]

The pure submodules (records, bands, analysis) and `pytest -m "not gpu"` need
neither torch nor NVML, so the analysis layer runs anywhere; only the live
measurement does.
"""

from __future__ import annotations

from .records import RunRecord, build_manifest, read_jsonl, to_csv, to_jsonl
from .bands import BandResult, extract_bands
from .analysis import MixedDeviceError, band_payload

__all__ = [
    "RunRecord",
    "to_jsonl",
    "read_jsonl",
    "to_csv",
    "build_manifest",
    "BandResult",
    "extract_bands",
    "MixedDeviceError",
    "band_payload",
]
