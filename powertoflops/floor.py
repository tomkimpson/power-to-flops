"""Closed-form verification floor beta — the width of the identified set.

Inverting a power trace returns a set, not a number; beta is that set's width,
expressed in machines of declared-format capacity (paper Sec. 3). Against a
bare meter beta is of order one machine, so an energy budget bounds an
operation count only loosely — but it does bound it. We do NOT claim energy
accounting is vacuous; the ladder (Sec. 5) prices what each verifier capability
buys. These are pure functions of the band parameters; they take no trace and
no RNG, so they are trivially testable and serve as the analytic reference the
numeric, trace-based inversion path must reproduce.

Closed forms (paper Sec. 3):

    bare meter:   beta(emptyset) = HFU_dec (r_hi / r_lo - 1)
                                   + dE0 / (r_lo C_max)

    two-band:     beta = HFU_dec (r_hi^dec - r_lo^dec) / r_lo^cov
                         + dE0 / (r_lo^cov C_max)

    budget form:  beta = (u_dec + u_0 + u_m + u_eta) / (r_lo^cov C_max)

``dE0`` is the overhead *energy* band width [J] (= (P0_hi - P0_lo) T).

The meter-noise allowance u_eta is the only *reducible* budget term. The
Floor-A study (A1) validates that the integrated meter
noise grows only as sqrt(T) under correlated, non-stationary noise, so
beta_eta = u_eta / (r_lo^cov C_max) shrinks as 1/sqrt(T) (C_max = F_max T grows
linearly). For stationary AR(1) noise the effective independent-sample count is
N_eff = T / (2 tau_corr) and u_eta = sigma sqrt(2 tau_corr T); see
:func:`std_energy_ar1` for the exact discrete result.
"""

from __future__ import annotations

import numpy as np


def beta_bare(
    hfu_dec: float,
    r_hi: float,
    r_lo: float,
    dE0: float,
    C_max: float,
) -> float:
    """Power-only floor beta(emptyset) = HFU_dec (r_hi/r_lo - 1) + dE0/(r_lo C_max).

    Single band: with power alone the verifier cannot tell declared joules from
    covert joules, so there is one band [r_lo, r_hi] and the divisor is the band
    floor r_lo itself -- the cheapest rate the hardware reaches (operand zeros).
    The general two-band form, where a wider channel separates the covert edge
    r_lo^cov from the declared floor, is :func:`beta_two_band`; do NOT divide a
    declared-band numerator by a different r_lo^cov here (that silently evaluates
    the two-band floor, not beta(emptyset)).
    """
    return hfu_dec * (r_hi / r_lo - 1.0) + dE0 / (r_lo * C_max)


def beta_two_band(
    hfu_dec: float,
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    dE0: float,
    C_max: float,
) -> float:
    """General two-band floor (declared band may differ from the covert edge).

    Reduces exactly to :func:`beta_bare` when the declared band and covert edge
    coincide (r_lo_cov == r_lo_dec, and r_hi_dec is the declared upper edge).
    """
    declared_term = hfu_dec * (r_hi_dec - r_lo_dec) / r_lo_cov
    overhead_term = dE0 / (r_lo_cov * C_max)
    return declared_term + overhead_term


def energy_slack(
    h,
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    dE0: float,
    C_max: float,
):
    """Energy-slack index S(h): the unclipped no-alarm budget width.

    This is :func:`beta_two_band` evaluated at declared utilisation h = HFU_dec,
    renamed (review C1): unbounded in h, it measures how far energy alone is
    from binding, not how much compute can physically hide. Array-friendly in h.
    """
    return beta_two_band(
        hfu_dec=h, r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
        dE0=dE0, C_max=C_max,
    )


def beta_phys(
    h,
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    dE0: float,
    C_max: float,
    kappa: float = 1.0,
    kappa_dec: float = 1.0,
):
    """Capacity-clipped identified-set width beta_phys(h) = min[S(h), clip(h)].

    kappa = F_max^cov / F_max^dec is the format capacity ratio (int8 vs fp16 on
    A100: kappa = 2); beta is in units of the declared-format capacity C_max, so
    values above 1 are legitimate when kappa > 1. Callers derive kappa from
    config.A100_DENSE_PEAK_OPS; this module stays hardware-agnostic.

    kappa_dec is the capacity ratio of the format the declared operations
    EXECUTE in (relative to the declared-format yardstick) inside the cheating
    schedule. In every single-declared-format scenario kappa_dec = 1 and the
    clip is the familiar (1 - h) kappa. The bare cross-format rung is the
    exception: a bare meter pins no format on the declared side either, so the
    cheating schedule executes the declared ops in the cheap covert format
    (kappa_dec = kappa); they then occupy only h/kappa of resource-time and the
    clip relaxes to kappa - h. General clip: (1 - h/kappa_dec) kappa.
    Array-friendly in h.
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if kappa_dec <= 0.0:
        raise ValueError(f"kappa_dec must be positive, got {kappa_dec}")
    slack = energy_slack(
        h, r_hi_dec=r_hi_dec, r_lo_dec=r_lo_dec, r_lo_cov=r_lo_cov,
        dE0=dE0, C_max=C_max,
    )
    return np.minimum(slack, (1.0 - np.asarray(h) / kappa_dec) * kappa)[()]


def beta_phys_worst_case(
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    dE0: float,
    C_max: float,
    kappa: float = 1.0,
    kappa_dec: float = 1.0,
) -> tuple[float, float]:
    """(h*, beta*) maximising beta_phys over h in [0, 1].

    With S(h) = s h + b (s = (r_hi_dec - r_lo_dec)/r_lo_cov rising) and the
    clip (1 - h/kappa_dec) kappa = kappa - c h falling (c = kappa/kappa_dec;
    see :func:`beta_phys` for kappa_dec), the max sits at the crossing:
    h* = (kappa - b)/(s + c), beta* = (kappa s + c b)/(s + c). At
    kappa_dec = 1 this is the single-format form h* = (kappa - b)/(s + kappa),
    beta* = kappa (s + b)/(s + kappa); at kappa_dec = kappa (the bare
    cross-format rung) the clip is kappa - h. If the overhead slack alone
    already exceeds the clip (b >= kappa) the clip binds everywhere and
    (h*, beta*) = (0, kappa). The opposite corner exists only for
    kappa_dec > 1 (the clip no longer vanishes at h = 1): if the crossing
    lands past h = 1 the slack binds on all of [0, 1] and
    (h*, beta*) = (1, S(1)) = (1, s + b).
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if kappa_dec <= 0.0:
        raise ValueError(f"kappa_dec must be positive, got {kappa_dec}")
    s = (r_hi_dec - r_lo_dec) / r_lo_cov
    b = dE0 / (r_lo_cov * C_max)
    if b >= kappa:
        return 0.0, kappa
    c = kappa / kappa_dec
    h_star = (kappa - b) / (s + c)
    if h_star >= 1.0:
        return 1.0, s + b
    beta_star = (kappa * s + c * b) / (s + c)
    return h_star, beta_star


def covert_headroom_rate(
    F_dec: np.ndarray,
    r_hi_dec: float,
    r_lo_dec: float,
    r_lo_cov: float,
    dP0: float,
) -> np.ndarray:
    """Instantaneous covert FLOP-rate headroom F_cov^max(t) [FLOP/s].

    The time-resolved twin of :func:`beta_two_band`. Given a *known* declared
    schedule F_dec(t), the verifier raises no alarm while the meter is explainable
    by honest declared work at the dear rate plus high overhead,
    P(t) <= r_hi_dec F_dec(t) + P0_hi. A cheater runs declared work as cheaply as
    admissible (r_lo_dec), idles overhead to P0_lo, and runs covert work at the
    cheapest covert edge r_lo_cov; the covert rate that still evades is

        F_cov^max(t) = [ (r_hi_dec - r_lo_dec) F_dec(t) + dP0 ] / r_lo_cov,

    with dP0 = P0_hi - P0_lo the overhead *power* band. Integrating over the window
    reproduces the scalar floor exactly: with dE0 = dP0 * T and C_dec = int F_dec dt,

        trapz(F_cov^max, t) / C_max == beta_two_band(hfu_dec=C_dec/C_max,
            r_hi_dec, r_lo_dec, r_lo_cov, dE0, C_max).

    The bare-meter green bound F <= (P - P0_lo)/r_lo^cov of the time-resolved
    tracker is the F_dec = 0 slice of this expression (no declared cover, so the
    only headroom is the overhead term dP0 / r_lo_cov on top of the observed peaks).
    """
    return ((r_hi_dec - r_lo_dec) * F_dec + dP0) / r_lo_cov


def beta_budget(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    C_max: float,
) -> float:
    """Budget form: total energy allowance over r_lo^cov C_max.

    u_eta (meter noise) is the only reducible term — it averages as
    1/sqrt(N_eff) with N_eff = T / (2 tau_corr) for AR(1); see
    :func:`u_eta_floor_a`. The bias terms (u_dec, u_0, u_m) do not average away.
    """
    return (u_dec + u_0 + u_m + u_eta) / (r_lo_cov * C_max)


def u_dec_reexec(sigma_dec: float, C_dec: float, n: int, z_gamma: float) -> float:
    """Declared-rate allowance under energy-logged sampled re-execution [J].

    Channel-residual curve (A1 study #1).
    Energy-logged re-execution of ``n`` declared samples reads the *actual*
    declared operands, so the prover's freedom to declare cheap data cancels and
    the declared-side band collapses *statistically* rather than being conceded at
    full width. The standard error of the mean declared rate is sigma_dec / sqrt(n)
    [J/FLOP]; multiplying by the declared compute C_dec [FLOP] lifts it to an
    energy allowance, paralleling the worst-case declared term (r_hi - r_lo) C_dec:

        u_dec(n) = z_gamma * (sigma_dec / sqrt(n)) * C_dec.

    Here ``sigma_dec`` is the declared-typical per-FLOP rate std [J/FLOP] (the
    activity-factor residual on declared operands, primed by A2 Experiment 2) and
    ``z_gamma`` is the confidence multiplier at level 1 - gamma. As n -> inf the
    allowance vanishes (u_dec ~ 1/sqrt(n)).

    This curve is the *re-execution regime*: it does NOT continuously connect to
    the bare-meter full-band term (r_hi - r_lo) C_dec. The bare meter concedes the
    *entire* operand band because it cannot see operands; re-execution *replaces*
    that with the narrow statistical width sigma_dec. The worst-case band and the
    sqrt(n) width answer two different questions. (The Exp-2 operand core w0 is
    descriptive only and prices neither: the ladder leaves the clock/locality
    rung unpriced rather than reading w0 as a certified edge.)
    """
    if n <= 0:
        raise ValueError("n (re-execution samples) must be >= 1")
    return z_gamma * sigma_dec * C_dec / float(n) ** 0.5


def u_m_transfer(m: float, r_bar_dec: float, C_dec: float) -> float:
    """Hardware-transfer allowance for a reference-vs-audited device gap [J].

    Channel-residual curve (A1 study #1). Re-execution runs on the verifier's
    *reference* silicon while the prover owns *audited-class* silicon; the device-to-device
    spread ``m`` (a fractional gap) enters the declared energy prediction at
    *full width* 2m -- false-alarm control at the top edge, cheating at the
    bottom:

        u_m(m) = 2 m * r_bar_dec * C_dec.

    The gap is a property of the *declared* work being re-executed, so it scales
    with the declared mid-band rate ``r_bar_dec`` = (r_lo + r_hi)/2, not the covert
    edge r_lo^cov (which separately enters the divisor as r_lo^cov (1 - m); see
    :func:`beta_composed`). A strict worst-case variant would use r_hi. As m -> 0
    (matched reference hardware) the allowance vanishes.
    """
    return 2.0 * m * r_bar_dec * C_dec


def u_0_metered(q0: float, dE0: float) -> float:
    """Overhead-band allowance after facility metering pins a fraction q0 [J].

    Channel-residual curve (A1 study #1: "facility metering -> overhead").
    The bare overhead allowance is the full band
    u_0 = dE0 (carried as a bounded nuisance, never known and subtracted). Facility
    metering that pins a fraction ``q0`` of the band leaves the residual

        u_0(q0) = (1 - q0) * dE0.

    q0 -> 1 fully meters the overhead (u_0 -> 0); q0 -> 0 is the bare band (dE0).
    ``q0`` is an A1-introduced residual parameter (same status as ``n`` and ``m``),
    not a spec quantity; the symbol avoids ``rho`` (the thermal pole in the paper).
    """
    return (1.0 - q0) * dE0


def beta_composed(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    m: float,
    C_max: float,
) -> float:
    """Composed floor with the transfer-gap covert-edge divisor [dimensionless].

    The budget form :func:`beta_budget` divided by the lowered covert edge
    r_lo^cov (1 - m): the same transfer
    gap ``m`` that widens the numerator via :func:`u_m_transfer` also lowers the
    cheapest covert rate the cheater can reach.

        beta = (u_dec + u_0 + u_m + u_eta) / (r_lo^cov (1 - m) C_max).

    At the no-channel limit (m = 0, u_m = 0, u_eta = 0, and u_dec set to the
    *full* conceded band (r_hi - r_lo) C_dec, u_0 = dE0) this reduces exactly to
    :func:`beta_bare` -- the regression anchor tying the channel-residual study
    back to the Stage-1 closed form.
    """
    if not 0.0 <= m < 1.0:
        raise ValueError("transfer gap m must satisfy 0 <= m < 1")
    return beta_budget(u_dec, u_0, u_m, u_eta, r_lo_cov * (1.0 - m), C_max)


# ---- additivity breakdown (A1 study #3): the channel-interaction term g_int -----
#
# The closed form assumes covert energy *simply adds*:
#   E = r_dec C_dec + r_cov C_cov + E_0   (additive; the verifier inverts with this).
# The paper (working-notes sec:channel_interaction) refines it with an interaction
# g_int -- declared and covert work share caches, bandwidth, cooling, DVFS -- whose
# *negative* branch (covert work filling declared memory-bound stall cycles) lets
# covert load LOWER observed power. Parametrised as a fraction of covert energy,
#   g_int = -g r_lo^cov C_cov,   g in [0, 1)   (g=0 additive; g>0 stall-filling),
# the world emits E = r_dec C_dec + r_lo^cov (1 - g) C_cov + E_0, so for a given
# slack the cheater fits 1/(1-g) more covert FLOPs than the additive inversion
# concludes. The interaction therefore enters the *divisor* r_lo^cov (1 - g) --
# structurally like the transfer gap m in beta_composed, but a distinct mechanism:
# m lowers the covert *edge* (a per-op rate), g exploits *shared declared work* (a
# cross-term). If both are present the divisor factors multiply, r_lo^cov(1-m)(1-g).


def beta_additive(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    C_max: float,
) -> float:
    """Nominal floor under the additive assumption g_int = 0.

    A named alias of :func:`beta_budget`: the width the verifier's additive
    inversion reports, valuing every slack joule at the bare covert edge
    r_lo^cov. The g=0 baseline the breakdown map reads against.
    """
    return beta_budget(u_dec, u_0, u_m, u_eta, r_lo_cov, C_max)


def beta_interaction(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    C_max: float,
    g: float,
) -> float:
    """True identified-set width when the world carries a negative interaction g_int.

    A stall-filling interaction g_int = -g r_lo^cov C_cov (g >= 0) lowers the
    effective covert edge to r_lo^cov (1 - g): the cheater fits more covert FLOPs per
    slack joule than the additive inversion concludes, so the true floor is the
    nominal one divided by (1 - g):

        beta_true = beta_additive / (1 - g),   i.e. beta_true / beta_nominal = 1/(1-g).

    Reduces to :func:`beta_additive` at g = 0 (the regression anchor tying the study
    back to beta_bare). ``g`` is a workload-coupling fraction, NOT the transfer gap m
    (which lowers the covert edge itself; see :func:`beta_composed`).
    """
    if not 0.0 <= g < 1.0:
        raise ValueError("interaction fraction g must satisfy 0 <= g < 1")
    return beta_budget(u_dec, u_0, u_m, u_eta, r_lo_cov * (1.0 - g), C_max)


def beta_rho(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    C_max: float,
    g_max: float,
) -> float:
    """rho-aware conservative floor carrying the worst admissible interaction.

    Evaluates :func:`beta_interaction` at the bound g = g_max, so beta_rho >=
    beta_true for every true g in [0, g_max] -- conservativeness restored by carrying
    the interaction bound |g_int| <= rho (paper's nuisance constraint). Reduces to
    :func:`beta_additive` at g_max = 0. The paper bounds the *energy* |g_int| <= rho;
    :func:`g_from_rho` bridges rho to the fraction g_max.
    """
    return beta_interaction(u_dec, u_0, u_m, u_eta, r_lo_cov, C_max, g_max)


def g_from_rho(rho: float, r_lo_cov: float, C_cov: float) -> float:
    """Map the paper's energy-interaction bound |g_int| <= rho [J] to the fraction g.

    From g_int = -g r_lo^cov C_cov, the energy bound rho corresponds to

        g = rho / (r_lo^cov C_cov).

    Evaluated at C_cov = C_max this is the worst-case fraction (the most covert
    energy a bounded interaction can hide). Lets the divisor form carry the paper's
    symbol rho without changing the multiplicative "more FLOPs per joule" semantics.
    """
    return rho / (r_lo_cov * C_cov)


def conservativeness_gap(
    u_dec: float,
    u_0: float,
    u_m: float,
    u_eta: float,
    r_lo_cov: float,
    C_max: float,
    g: float,
) -> float:
    """beta_true - beta_nominal: the width the additive floor understates [dimensionless].

    The colormap quantity of the breakdown map. Zero at g = 0; strictly positive and
    growing as g -> 1. Closed form:

        gap = beta_additive * g / (1 - g).
    """
    return beta_interaction(u_dec, u_0, u_m, u_eta, r_lo_cov, C_max, g) - beta_additive(
        u_dec, u_0, u_m, u_eta, r_lo_cov, C_max
    )


def u_eta_floor_a(sigma: float, T: float, tau_corr: float) -> float:
    """Meter-noise energy allowance (Floor A) for stationary AR(1) noise.

    The integrated zero-mean noise U = int eta dt has std sigma sqrt(2 tau T)
    in the large-window limit (Floor-A study, A1), i.e. N_eff = T / (2 tau_corr)
    effective independent samples:

        u_eta = sigma * T / sqrt(N_eff) = sigma * sqrt(2 * tau_corr * T).

    This corrects the Stage-1 stub, which used N_eff = T / tau_corr (a factor of
    2 too large). For the exact discrete result valid at any window, and the
    white-noise branch, see the analytic companions :func:`std_energy_ar1` /
    :func:`std_energy_white` below.
    """
    if tau_corr <= 0.0:
        raise ValueError("tau_corr must be > 0 for a finite N_eff")
    return sigma * (2.0 * tau_corr * T) ** 0.5


def std_energy_ar1(sigma: float, T: float, dt: float, tau_corr: float) -> float:
    """Exact discrete std of U = sum_i eta_i dt for stationary AR(1) noise.

    On a uniform grid of N = round(T/dt)+1 samples with
    Cov(eta_i, eta_j) = sigma^2 phi^|i-j|, phi = exp(-dt/tau_corr),

        Var(U) = dt^2 sigma^2 [ N (1+phi)/(1-phi)
                                - 2 phi (1 - phi^N) / (1-phi)^2 ].

    The large-N / small-(dt/tau) limit is sigma sqrt(2 tau_corr T) (see
    :func:`u_eta_floor_a`); the phi -> 0 limit is the white result
    sigma sqrt(dt T) (see :func:`std_energy_white`). This is the analytic
    reference the Monte-Carlo estimator must reproduce.
    """
    if tau_corr <= 0.0:
        raise ValueError("tau_corr must be > 0; use std_energy_white for white noise")
    n = int(round(T / dt)) + 1
    phi = float(np.exp(-dt / tau_corr))
    one_minus = 1.0 - phi
    var = dt * dt * sigma * sigma * (
        n * (1.0 + phi) / one_minus
        - 2.0 * phi * (1.0 - phi ** n) / (one_minus * one_minus)
    )
    return float(var) ** 0.5


def std_energy_white(sigma: float, T: float, dt: float) -> float:
    """Std of U = sum_i eta_i dt for i.i.d. (white) meter noise.

    Var(U) = dt^2 sigma^2 N with N = T/dt independent samples, so
    std(U) = sigma sqrt(dt T). (Uses the rectangular-sum sample count N = T/dt,
    not the endpoint-inclusive grid size, matching the AR(1) asymptote.)
    """
    return sigma * (dt * T) ** 0.5


def beta_eta(
    sigma: float,
    T: float,
    dt: float,
    tau_corr: float,
    r_lo_cov: float,
    C_max: float,
) -> float:
    """Reducible meter-noise floor beta_eta = u_eta / (r_lo^cov C_max).

    Uses the exact AR(1) allowance :func:`std_energy_ar1`. Since C_max scales
    linearly with the monitoring window while u_eta ~ sqrt(T), beta_eta ~
    1/sqrt(T): the noise floor is reducible by monitoring longer.
    """
    return std_energy_ar1(sigma, T, dt, tau_corr) / (r_lo_cov * C_max)
