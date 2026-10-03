# D1-H1-SAFE-v1 (Stage 4)

Post-hoc exploratory mechanism diagnostic. Additive only; no training.

## Layout

- `code/run_d1.py` — driver: verify | smoke | parity | run | validate | analyze
- `code/analyze_d1.py`, `code/write_report_d1.py` — analysis + report
- `tests/test_d1.py` — 15 focused invariants (unittest)
- `outputs/{unmasked,masked,bf}/...` — 22 cells: episodes.csv (35 frozen
  fields), result.json, telemetry.json
- `traces/*.npz` — full step traces for episodes 0,1 of every regime
  (528 traced episodes); int/float/fork schema in D1_CONFIG.json
- top-level: D1_CONFIG.json, D1_U_PARITY.json, D1_VALIDATION.json,
  D1_SUMMARY.csv, D1_SEED_CAPACITY.csv, D1_REGIME_LEVEL.csv,
  D1_CONFLICT_TELEMETRY.csv, D1_REFILL_TELEMETRY.csv,
  D1_LOCAL_FORK_SUMMARY.csv, statistics.json, D1_DIAGNOSTIC_REPORT.md

## Reproduce

Use the frozen runtime interpreter:

```
/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python \
  artifacts/aamas2027/stage4/d1_h1_safe/code/run_d1.py verify
... smoke --episodes 3
... run --which u     # U cells + mandatory exact-parity gate
... run --which m     # only after U parity PASS
... run --which bf
... validate
... analyze
```

The masked cells are only interpretable after D1_U_PARITY.json is PASS;
the driver halts before running M on any parity failure.

Generated: 2026-10-03T03:43:41Z
