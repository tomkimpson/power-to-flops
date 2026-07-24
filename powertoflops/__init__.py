"""power-to-flops — how much can an off-chip power meter certify about compute?

Reusable library modules. Standalone entry points live in ``scripts/``.

Closed form and bands:
    config       parameter dataclasses (the *measured* band values live in
                 ``results/bench/measured_bands.json``, not here)
    floor        closed-form verification floor beta, in machines of capacity
    measured     load the measured-bands artifact and derive the named edges
    joint_model  the jointly-feasible-scenario headline (one declared format,
                 one covert format, one observation window)
    ladder       the capability ladder: cumulative max_h beta_phys per rung
    plotstyle    house figure style; ``save`` writes timestamp-free PDFs

Measurement:
    bench        NVML power/energy harness for the A100 campaign (Exp 1-8) plus
                 the pure-Python band extraction that consumes its records
"""
