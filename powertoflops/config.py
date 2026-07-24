"""Parameters for the closed form and the measurement harness.

Single source of truth — scripts import :data:`DEFAULT` rather than hardcoding
numbers ("no magic numbers"). Every value carries its units and a provenance
note.

IMPORTANT: the band values in :class:`FloorParams` are order-of-magnitude
placeholders from Horowitz, "Computing's Energy Problem (and What We Can Do
About It)", ISSCC 2014. **No reported result uses them.** Every number in the
paper comes from the *measured* bands in ``results/bench/measured_bands.json``,
loaded via :func:`powertoflops.measured.load_measured_bands`. The placeholders
survive only as the analytic sanity check the unit tests exercise, where what
matters is the band *structure* (ratios driving beta), not the absolute scale.

Notation matches the paper:
    r        exchange rate / energy-per-op            [J/FLOP]
    F        FLOP rate                                [FLOP/s]
    P0       non-compute overhead power               [W]
    E0       overhead *energy* over the window = P0 T [J]
    C_max    capacity ceiling = F_max T               [FLOP]
    dE0      overhead *energy* band width             [J]
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

# A100 (SXM4 80GB) DENSE peak rates per numeric format, in nominal op/s (each
# multiply-accumulate = 2 ops). Source: NVIDIA A100 Tensor Core GPU datasheet
# (fp32 = non-tensor-core CUDA-core peak; tf32/fp16/bf16/int8 = tensor-core
# dense peaks, NO 2:4 structured sparsity — the sparse figures are 2x these).
# This is the format-dependent capacity model the review's C7 asks for: a
# single scalar F_max hides that int8 capacity is 2x fp16 and tf32 is 8x fp32,
# which matters wherever covert and declared work use different formats
# (kappa = F_max^cov / F_max^dec in the capacity-clipped beta).
A100_DENSE_PEAK_OPS = MappingProxyType({
    "fp32": 19.5e12,
    "tf32": 156.0e12,
    "fp16": 312.0e12,
    "bf16": 312.0e12,
    "int8": 624.0e12,
})


@dataclass(frozen=True)
class FloorParams:
    """Bands that set the closed-form verification floor beta.

    The declared-side exchange-rate band [r_lo, r_hi] is the product of the CMOS
    factors r ~ kappa(precision) * alpha(operands) * C_eff(locality) * V^2
    (paper Sec. 5). The headline ratio r_hi / r_lo ~ 10-20x comes from precision
    (fp32 -> int8) compounded by DVFS (V^2, ~1.5-2x) and operand locality.
    """

    # Declared-side exchange-rate band [J/FLOP].
    r_lo: float = 1.0e-12          # cheapest declared op (~A100 fp16 ballpark)
    r_hi: float = 15.0e-12         # ~15x band (mid of the 10-20x range)

    # Cheapest covert edge r_lo^cov (the divisor of beta). For the bare power
    # meter the covert edge coincides with the declared cheapest rate.
    r_lo_cov: float = 1.0e-12

    # Declared hardware FLOP utilisation HFU_dec = C_dec / C_max (dimensionless).
    hfu_dec: float = 1.0

    # Overhead *power* band [W]; idle/cooling/leakage draw the verifier cannot
    # subtract. The overhead energy band is (P0_hi - P0_lo) * T -> see dE0.
    P0_lo: float = 50.0
    P0_hi: float = 150.0

    # Capacity scale.
    F_max: float = 1.0e15          # 1 PFLOP/s ceiling
    T: float = 3600.0              # 1 h monitoring window [s]

    @property
    def dE0(self) -> float:
        """Overhead energy band width Delta E_0 = (P0_hi - P0_lo) * T  [J]."""
        return (self.P0_hi - self.P0_lo) * self.T

    @property
    def C_max(self) -> float:
        """Capacity ceiling C_max = F_max * T  [FLOP]."""
        return self.F_max * self.T


@dataclass(frozen=True)
class ChannelParams:
    """Verifier-channel residual parameters for the A1 channel-residual study.

    These set how far each *irreducible* budget term (u_dec, u_m, u_0) shrinks as
    the corresponding channel is deployed (A1 deliverable #1: "the
    paper states the allowances only schematically; this study makes them curves").
    The channel-residual sweeps over n / m / q0 live in the plotting script; these
    are the fixed companions.

    PLACEHOLDER VALUES, grounded by A2 hardware (A1 study #1).
    """

    # Confidence multiplier z_gamma at level 1 - gamma for the re-execution
    # standard error sigma_dec / sqrt(n). Two-sided 95% normal quantile; edit
    # this to expose gamma's effect.
    z_gamma: float = 1.96

    # Declared-typical per-FLOP rate std [J/FLOP]. PLACEHOLDER: a *narrow* sub-band
    # of the full declared band (r_hi - r_lo) = 1.4e-11 -- only the activity-factor
    # alpha jitter on declared operands survives re-execution, not the full
    # precision x DVFS x locality band. To be primed by A2 Experiment 2 restricted
    # to declared-typical operands.
    sigma_dec: float = 3.0e-13

    # Representative declared rate scaling the transfer band 2m [J/FLOP].
    # Mid-band (r_lo + r_hi)/2 of FloorParams; a strict worst-case variant uses
    # r_hi. Not r_lo^cov (which enters the divisor of beta_composed instead).
    r_bar_dec: float = 8.0e-12


@dataclass(frozen=True)
class SimParams:
    """Synthetic-trace generation knobs."""

    dt: float = 1.0                # sample period [s] (NVML-cadence-like)
    sigma_eta: float = 5.0         # meter-noise std [W] (PLACEHOLDER)
    seed: int = 0                  # RNG seed for reproducibility


@dataclass(frozen=True)
class BenchConfig:
    """Knobs for the A2 single-GPU bench harness (powertoflops/bench/, scripts/run_bench).

    Single source of truth for the R2 measurement campaign
    (see README.md, "Measurement campaign"):

        Exp 1  exchange-rate band r_hi/r_lo : precision x operand x locality
               factorial at ONE fixed shape on ONE GPU (review C4/C5)
        Exp 2  covert edge r_lo^cov, operand core w0 : operand VALUES only,
               repeated rounds so each dist samples several thermal states
        Exp 3  marginal rate dP/dF + overhead band : workload-intrinsic
               sustained-F ladder plus idle-variation cells (no duty cycle, C3)
        Exp 4  compact cross-device core sweep (band corners on >=3 GPUs, C5/C6)

    No magic numbers in the runner — every shape, dtype, repeat count lives here.

    Locality is operand RESIDENCY at byte-identical shape/kernel/op count, not
    problem size (the old cache-vs-dram shapes were a size confound, review C4):
    ``locality_buffers`` maps each locality label to the number of A/B operand
    sets round-robined across iterations (powertoflops.bench.workload.make_matmul).
    Working sets vs the A100's 40 MB L2 at exp1_shape = 2048^3 (A+B per set,
    C reused): fp16/bf16 16 MB, int8 8 MB — "resident" (1 set) fits, "streamed"
    (8 sets) forces eviction between reuses; fp32/tf32 32 MB + 16 MB C spills
    marginally, attenuating (not confounding) the contrast at those formats.
    """

    # Campaign tag, snapshotted into every run manifest.
    sweep_version: str = "r2"

    # Exp 1 fixed shape (M, N, K). Nominal op count per matmul is 2*M*N*K (see
    # powertoflops.bench.workload.ops_nominal_matmul). 2048^3 keeps windows well above
    # the NVML energy-counter update interval at every dtype (1024^3 was
    # launch-bound with sub-ms windows, 2026-06-18) while fp16 operand sets
    # still fit L2 for the resident arm.
    exp1_shape: tuple[int, int, int] = (2048, 2048, 2048)

    # Exp 1 precision axis (the kappa factor; fp32 -> int8 is the precision span).
    # "fp32" = tf32 OFF, "tf32" = fp32 tensors with tf32 allowed (a backend flag,
    # not a dtype). Handled in powertoflops.bench.workload.make_matmul.
    dtypes: tuple[str, ...] = ("fp32", "tf32", "fp16", "bf16", "int8")

    # Exp 1 locality axis (the C_eff factor): label -> n_buffers (operand sets
    # rotated per iteration; see class docstring for the residency arithmetic).
    locality_buffers: tuple[tuple[str, int], ...] = (
        ("resident", 1), ("streamed", 8),
    )

    # Exp 1 DVFS axis (the V^2 factor): SM clocks [MHz] to attempt to lock. None
    # leaves the clock to the governor — the OzSTAR case: locking is privilege-
    # denied by the driver, so the achieved clock is RECORDED per
    # window (meter._TelemetrySampler) and stratified in analysis instead.
    dvfs_sm_clocks: tuple[int, ...] | None = None

    # Operand-value axis (the alpha factor), shared by Exp 1 (factorial) and
    # Exp 2 (values-only sweep). "declared_typical" primes sigma_dec.
    operand_dists: tuple[str, ...] = (
        "zeros", "sparse_struct", "low_entropy",
        "dense_random", "adversarial_toggle", "declared_typical",
    )
    exp2_dtype: str = "fp16"          # fixed precision for the operand sweep
    exp2_locality: str = "streamed"   # fixed residency for the operand sweep
    # Rounds: each dist appears this many times, shuffled, so it samples several
    # well-separated thermal/clock states rather than one (clock stratification
    # then exposes any dist-vs-DVFS confound instead of baking it in).
    exp2_rounds: int = 4

    # Exp 3 sustained-rate ladder: F varies through shape alone, every window
    # 100% busy at its own natural rate (no on/off mixing — the duty-cycle
    # design made P affine in F by construction, review C3). Small cubes are
    # launch-bound (low F), skinny-K shapes bandwidth-bound, big cubes
    # compute-bound (high F): a ~60-90x sustained-F spread at one dtype, so
    # marginal_rate()'s OLS slope stays interpretable as dP/dF.
    exp3_ladder: tuple[tuple[int, int, int], ...] = (
        (256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
        (8192, 8192, 64), (8192, 8192, 256),
        (2048, 2048, 2048), (3072, 3072, 3072), (4096, 4096, 4096),
        (6144, 6144, 6144), (8192, 8192, 8192),
    )
    exp3_dtype: str = "fp16"
    exp3_locality: str = "streamed"
    # Idle-variation cells bound the CONTROLLABLE part of P0 (review C3): the
    # only knobs available without privilege on a shared node. no_ctx = before
    # torch touches the GPU; ctx = CUDA context up; cached = context + operand
    # pool held (HBM occupancy); post_load = right after the hottest ladder
    # cell; cooled = after exp3_cooldown_s of sleep. Ordering is pinned in
    # powertoflops.bench.runner (no_ctx/ctx/cached first, post_load/cooled last).
    exp3_idle_variants: tuple[str, ...] = (
        "no_ctx", "ctx", "cached", "post_load", "cooled",
    )
    exp3_idle_window_s: float = 5.0   # idle windows: longer for a stable delta
    exp3_cooldown_s: float = 60.0     # sleep before the "cooled" idle cell

    # Exp 4 cross-device core sweep: the exchange-band corners ({fp32,fp16,int8}
    # x residency, dense operands) plus the covert edge (zeros at fp16), compact
    # enough for one short allocation per device.
    core_dtypes: tuple[str, ...] = ("fp32", "fp16", "int8")
    core_zeros_dtype: str = "fp16"

    # Matched-energy pair (referee B3): the fixed-T, same-C_dec attainability
    # witness. Run A = n_dec declared iterations on DEAR operands + idle pad
    # to pair_window_s; run B = the SAME n_dec declared iterations on CHEAP
    # operands + a tuned covert fill + pad to the same window. Both runs are
    # fp16-declared / int8-covert — the joint-model headline corner (B1).
    # Scripts: scripts/run_matched_pair.py; analysis: scripts/analyze_pair.py.
    pair_declared_dtype: str = "fp16"        # single declared format (B1)
    pair_declared_shape: tuple[int, int, int] = (2048, 2048, 2048)
    pair_declared_locality: str = "streamed"  # same locality in both runs
    pair_dear_dist: str = "adversarial_toggle"  # ~the fp16 r_hi_dec cell
    pair_cheap_dist: str = "zeros"              # ~the fp16 r_lo_dec cell
    pair_covert_dtype: str = "int8"          # the exp7 covert-edge format
    pair_covert_dist: str = "zeros"
    pair_covert_shape: tuple[int, int, int] = (2048, 2048, 2048)
    # Requested utilisations: infeasible ones are skipped at pre-flight (the
    # smoke of job 14632514 measured h_max ~ 0.21 on a T=3 s rig; 0.25 stays
    # in the grid in case the 10 s capture lands higher).
    pair_h_values: tuple[float, ...] = (0.10, 0.15, 0.25)
    pair_h_auto_edge: bool = True     # add a 0.9*h_max point from calibration
    pair_window_s: float = 10.0       # fixed wall window T (exp7 convention)
    pair_repeats: int = 8             # scored ABAB repeats per h
    pair_tol_frac: float = 0.01       # |E_A - E_B| / E_A convergence target
    pair_max_tune: int = 6            # tuning-iteration cap (then frozen)
    pair_safety_frac: float = 0.95    # reject h above safety * h_max
    pair_calib_busy_frac: float = 0.3  # busy share of calibration windows
    pair_calib_repeats: int = 3       # windows per calibration quantity
    pair_warmup_windows: int = 3      # A+B warm-up windows before freezing dose

    # Exp 7 marginal covert-cost capture (referee B1/B2): fixed-wall-window,
    # idle-padded, randomized-dose intervention. Every measured window holds
    # the SAME fp16 declared burst (calibrated once to marg_declared_frac of
    # the window) plus cov_iters covert matmuls of one covert format, then
    # sleeps to marg_window_s. Regressing window energy on covert nominal ops
    # (powertoflops.bench.bands.marginal_covert_cost) identifies dE/dC_cov — the
    # marginal covert cost with declared work, shape, and window held fixed,
    # unlike Exp 3's between-shape slope. Dose fractions keep total busy time
    # <= ~0.78 of the window so the pad never underflows; the zero dose is
    # captured per covert format as the intercept anchor.
    marg_declared_dtype: str = "fp16"
    marg_declared_dist: str = "declared_typical"
    marg_declared_shape: tuple[int, int, int] = (2048, 2048, 2048)
    marg_declared_locality: str = "streamed"
    marg_declared_frac: float = 0.30      # declared busy fraction of the window
    marg_window_s: float = 10.0           # fixed measured wall window [s]
    marg_cov_dtypes: tuple[str, ...] = ("int8", "fp16")
    marg_cov_dist: str = "zeros"          # adversary-cheapest covert operands
    marg_cov_shape: tuple[int, int, int] = (2048, 2048, 2048)
    marg_cov_fracs: tuple[float, ...] = (0.0, 0.06, 0.12, 0.24, 0.36, 0.48)

    # Quantized transformer block (R2.7 / referee B5): TRAINED-weight int8
    # transformer layer pricing the useful-operand band. Dims match the
    # extracted Pythia-6.9B layer (hidden 4096, 32 heads, ffn 16384); cells
    # are (seq_len, batch) pairs — the seq axis sweeps the attention share of
    # the op count (a mis-accounting flag if r rises with seq^2), all
    # servable from the 8 x 2048-token activation artifact.
    qblock_d_model: int = 4096
    qblock_n_head: int = 32
    qblock_ffn_mult: int = 4
    qblock_cells: tuple[tuple[int, int], ...] = (
        (512, 4), (512, 8), (1024, 8), (2048, 4), (2048, 8))
    # Local-only artifacts (sha256-pinned by the committed manifest); regen
    # with scripts/fetch_qblock_weights.py on a network-connected node.
    qblock_weights_npz: str = "data/qblock/pythia-6.9b_layer15_weights.npz"
    qblock_acts_npz: str = "data/qblock/pythia-6.9b_layer15_acts.npz"
    qblock_manifest: str = "data/qblock/pythia-6.9b_layer15_manifest.json"
    # Repeats for the one-sided 95% LCB band endpoints (up from the global 5;
    # with discard_frac 0.2 this leaves >= 6 untrimmed windows per cell).
    qblock_repeats: int = 8
    qblock_alpha: float = 0.05        # one-sided level of the band LCBs
    qblock_soak_s: float = 10.0       # thermal soak before calibration (B3 lesson)
    # Pre-flight validation thresholds: quantized forward vs fp32 reference
    # on the largest cell. Finiteness is a hard assertion regardless.
    qblock_min_cosine: float = 0.98
    qblock_max_rel_l2: float = 0.25

    # Exp 8 (referee B5, third round): marginal useful-cost dose-response. The
    # Exp-7 construction (fixed fp16 declared burst, idle-padded to a fixed
    # window) with the trained-weight qblock forward as the swept covert dose,
    # so baseline power cancels in the slope and the useful edge becomes a
    # genuine INCREMENTAL lower bound — not the standalone total quotient E/C
    # of run_qblock (which folds in idle baseline, anti-conservative as a
    # covert denominator). Each cell is a covert "arm" (one qblock (seq,batch)
    # shape); its dose is the number of forward passes. The declared burst and
    # window inherit the marg_* fields, so the useful edge and the generic int8
    # covert edge sit on one axis. Two cells bracket the attention-share axis
    # while bounding capture time.
    qblock_marg_cells: tuple[tuple[int, int], ...] = ((512, 8), (2048, 8))
    qblock_marg_cov_fracs: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40)

    # Measurement window. The NVML total-energy counter updates coarsely (tens of
    # ms), so a window must span MANY updates for the delta to be accurate — a few
    # ms reads stale/zero deltas (probe finding 2026-06-18). Since dtypes differ in
    # throughput by ~30x, a fixed matmul count can't equalise window length; the
    # runner instead CALIBRATES iters per cell to ~target_window_s of wall time
    # (powertoflops.bench.runner). ``iters`` is only the fallback when calibration is off.
    target_window_s: float = 1.5      # wall time per measured window [s]
    calib_matmuls: int = 5            # matmuls timed to estimate per-matmul cost
    max_iters: int = 400000           # safety cap on calibrated iters (the 256^3
                                      # ladder cell is launch-bound at ~5 us per
                                      # matmul, so ~300k fill a 1.5 s window)
    iters: int = 64                   # fallback iters (target_window_s <= 0)
    warmup: int = 3                   # discarded warmup windows (JIT/autotune/soak)
    repeats: int = 5                  # measured windows per sweep point
    discard_frac: float = 0.2         # trim fraction (slowest/highest-energy tail)

    # Thermal-drift control (powertoflops.bench.runner): randomise sweep order and splice a
    # fixed reference point every ``ref_every`` runs so slow drift is regressable.
    ref_every: int = 8
    seed: int = 0                     # RNG seed for operand fills + sweep shuffle


@dataclass(frozen=True)
class Config:
    floor: FloorParams = FloorParams()
    channel: ChannelParams = ChannelParams()
    sim: SimParams = SimParams()
    bench: BenchConfig = BenchConfig()


DEFAULT = Config()
