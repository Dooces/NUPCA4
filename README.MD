# NUPCA3 Skeleton (v1.5b)

This repository is a **from-scratch**, axiom-aligned skeleton for the NUPCA3 agent.
It intentionally keeps the control loop explicit and avoids external RL framework semantics.

## What is implemented

- Package and module structure mapping **A0–A17** to concrete files.
- A single authoritative step loop in `nupca3/step_pipeline.py`.
- REST-only structural edits are only executed inside `edits/rest_processor.py` and only when `rest(t)=True`.
- Typed data records in `nupca3/types.py`.

## What is stubbed (by design)

Several components require environment/task definitions or user-provided probe functions. These are stubbed behind stable interfaces:

- A11 probe set details and semantic integrity metrics
- A12 edit scoring specifics (ΔJ, ΔS, ΔC) beyond interface and accounting
- A8 action synthesis / planning policy (commit gating is implemented; action selection is a placeholder)

## Run

```bash
python example_agent.py
```

## Notes

This is a structural scaffold intended to be filled with your precise axiom math; it is not presented as a final learning system.
