# Speaker Notes — Can Power Draw Constrain Covert Compute?

**Talk:** 8 minutes, 10 slides (title + 9). Timing marks are cumulative
targets at the *end* of each slide; the schedule lands at 7:30 to leave ~30 s
of buffer.
**Venue:** MATS (date/venue placeholder on the title slide — update it).

---

## 1. Title — 0:10

One sentence and move: "This is about what a power meter bolted to the wall
can tell you about what an AI data centre actually ran."

## 2. Why Verify Compute? — 0:50

- First bullet: say the examples the slide omits — *agreements* meaning
  compute thresholds, treaties, reporting rules. All of it is an honour
  system until someone can check.
- Walk the diagram — this is the cast for the whole talk: the **prover** is
  the monitored operator; the **verifier** is the inspector. The prover
  makes a declaration — "we used less than X FLOP", "we ran no training" —
  and hands over supporting stuff.
- Unpack the "stuff" in one breath: logs and telemetry; weight checkpoints
  (proof-of-learning transcripts — the verifier can re-execute segments);
  cryptographic proofs (zkPoT — a proof the committed weights really came
  from the declared training).
- Optional beat if time feels good: all that stuff attests the *declared*
  work — nothing handed over can demonstrate the *absence* of work. That's
  the gap this talk lives in.

## 3. Verification Must Survive an Adversary — 1:25

- The prover may cheat: hide compute, run training it denied.
- The killer problem with the evidence: it is produced by the (possibly
  cheating) prover, on silicon the prover controls — counters, clocks,
  firmware, all of it.
- Off-chip analogue sensors dodge this entirely: the measurement never
  passes through the operator's stack. Physically separate, retrofittable,
  can be sealed against tampering.

## 4. This Work: Power Channel — 2:05

- We pick one channel: power — the tightest link to computation, because
  every operation costs energy.
- Same game as slide 2, but now the verifier has a second arrow: P(t) off
  the wall, through a meter the prover cannot forge.
- The two questions the trace can be asked of the declaration:
  - *What kind* of compute ran — training cadence, periodicity in the
    time profile. **The nod:** "that's our companion paper on power
    traces — I won't talk about it today."
  - *How much* compute ran — integrated energy. "Today is this one: can a
    trace bound total FLOP?"

## 5. What The Meter Sees — 3:10

- Walk the equation left to right: power is exchange rate times operation
  rate plus overhead. The verifier wants F; the meter only ever sees the
  product r·F. And r — joules per operation — is the prover's choice, and
  invisible.
- Panel walk: two different workloads — little compute at a high rate, much
  compute at a low rate — drive the *identical* trace. Inverting the trace
  can't return an answer, only a set of consistent answers: panel (c),
  total compute C = ∫F dt on the x-axis.
- Land the goal: bound the width C_hi − C_lo. Since the honest declaration
  is the bottom edge, that width IS the maximum covert compute:
  C_covert^max.
- **Spoken (the equation is not on the slide):** "We derive that bound in
  closed form — roughly, the free joules the ambiguity leaves over, divided
  by the price of the cheapest covert operation. Every term in it is a
  measurable band." That's the licence for the next two slides.

## 6. What Sets the Exchange Rate r? — 4:10

- The slide that earns the whole talk. The intuition to break: "an
  operation costs what an operation costs."
- Zoom out through the four scales: which cells toggle (operand values, α);
  how many gates one result takes (numeric format, κ — one fp32 word or
  four int8s in the same fabric); how far operands travel (memory
  hierarchy, C_eff); what voltage charges every switch (DVFS, V²).
- Spoken, not shown: the clock frequency *cancels* — cost per op doesn't
  depend on how fast you run, only on what each switch costs.
- Punchline (say it — it's no longer printed): every one of these factors
  is the prover's to choose, and none is visible to an off-chip meter. "The
  joules-to-FLOPs conversion is a free parameter of the adversary."

## 7. How Wide Are the Bands? — 5:00

- Lean slide; don't enumerate experiments — "we measured the bands
  directly on an A100."
- The figure, the declared band: sweep a matmul over format, operand
  values, and locality, and the cost per op spans **24.8×**
  (0.79–19.5 pJ/op) — dominated by precision (13.5×), operands worth ~2×,
  locality ~nothing. The V² factor is assumed from device physics, not
  swept — the platform denied clock control.
- The overhead band — say this is a *different experiment*: with no
  declared work running, the idle draw wanders over a 41 W band
  ([60, 101] W) that the verifier must concede.
- Spoken, since it's no longer on the slide: "we also measured the third
  band — the cheapest a *hidden* operation can be — at about half a
  picojoule." That's the denominator the bound needs; one sentence is
  enough.
- If pressed on trust: this is a lab calibration of *bands* on the NVML
  counter, not adversarial telemetry.

## 8. How Much Compute Can Hide? — 6:15

- Define β: the width from slide 5, C_covert^max, in units of the machine's
  capacity. β = 1 means a whole machine of compute could be hiding.
- Walk the ladder left to right; each rung is a capability a protocol could
  stipulate, and each is priced from the measured bands:
  - **1.16** — power plus a checked precision declaration: *more than an
    entire machine* of ambiguity survives.
  - **?** — observing clock and locality: unpriced. The capability doesn't
    exist yet (attested telemetry, locked clocks are prospective), and our
    experiments couldn't sweep DVFS to measure what it would buy.
  - **0.34** — re-execute the declared work on your own hardware to
    measure its true energy per op. A hardware gap persists (energy/op is
    hardware-specific; same-SKU transfers at 2–3%, cross-vendor unknown).
  - **0.07** — restrict covert work to being *useful*: hidden zero-ops
    compute nothing. Measured on a real quantised transformer block —
    useful covert work costs ~5× the silicon floor.
- Two spoken beats (no longer printed):
  - Re-execution only pays at an *observed* operating point — so "power
    plus paperwork" with no on-site sensing stalls at β ≈ 1.
  - No capability reaches the covert side; the last step is a restriction
    on what the adversary wants, not a measurement of the channel.
- Optional colour if ahead of time: "we also built the attack — matched-
  energy cheating runs hiding 0.41 machines of covert int8, and the meter
  accepts every trace."

## 9. Limitations — 6:50

- Flash, don't read. Headline delivery: "the usual caveats — one GPU model,
  DVFS never swept, on-die telemetry rather than a real wall meter, single
  device not a facility, and everything is time-integrated."
- If one deserves a sentence: DVFS — the prover owns the clock and we
  couldn't sweep it, so on that axis our β is if anything *understated*.

## 10. Summary — 7:30

- Four lines, slowly:
  - Verification is important.
  - Off-chip analogue measurements can serve it.
  - Power + declarations + reasonable assumptions constrain covert compute
    to ≲ 7% of machine capacity.
  - Next: more realistic, representative hardware.
- Close on the bookend: "and the other question — *what kind* of compute
  ran — is in the trace's time profile. That's the companion paper; happy
  to talk about it in questions."

---

## Likely questions

- *"Can't you just see the training steps in the trace?"* — Yes, the
  profile carries cadence and periodicity — that's the *what kind*
  question, the companion paper. It doesn't help count totals under our
  threat model: with r free to vary, the trace's shape adds nothing about
  totals beyond its integral.
- *"What's the actual closed form?"* — C_covert^max ≤ [(r_hi − r_lo)·C_dec
  + ΔE₀] / r_lo^cov: the energy ambiguity of the declared work plus the
  overhead slack, divided by the cheapest covert op — then clipped by the
  spare capacity the declared work leaves free. The paper also takes the
  worst case over the declared utilisation; the 1.16 is that worst case.
- *"Why not just subtract the idle baseline?"* — The prover controls fans,
  clocks, idle states; a known-and-subtracted P₀ is a hiding channel.
  Hence a band, not a baseline.
- *"Is 7% good enough? Does this kill power-based verification?"* — It
  prices it. A bare meter as an op-counter is dead (worth knowing before a
  regime proposes it), but power as the *anchor* plus stipulated
  capabilities gets to ~0.07 — with the conditions stated, which is what a
  treaty designer needs.
- *"Has anyone actually cheated it?"* — We did: matched-energy runs hiding
  0.41 machines of covert int8 behind an accepted trace. That's a lower
  witness; the model bound says up to ~1.16 in the worst case.
- *"What about newer chips / FP4?"* — Cheaper datapaths widen the precision
  envelope and speed up the cheapest format, so the numbers get worse; the
  closed form is device-agnostic and can be re-measured per chip.
- *"Why is the clock rung unpriced rather than estimated?"* — Two
  independent reasons: no trusted channel to observe it exists yet, and we
  never measured the width it would remove (the governor pinned the clock
  to ±3%, so the ~1.5× V² factor is literature physics, not our data).
- *"Facility scale?"* — Open. It's a multi-resource allocation problem —
  per-device clips, prover-chosen work placement, shared cooling — not a
  change of units; that's the "next goal" bullet.
