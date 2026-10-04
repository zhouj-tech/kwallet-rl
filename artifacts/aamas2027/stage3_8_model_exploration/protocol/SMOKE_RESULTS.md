# SCFAC Smoke Test Results (ENGINEERING ONLY — NOT model performance)

Date: 2026-10-03
Runner: `smoke/run_smoke.py`
Runtime: `/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python`
(Python 3.10.22, numpy 2.0.2, torch 2.5.1, CPU, torch threads = 4)

## Setup (per task spec section 11)

- SHORT ENGINEERING smoke tests only; these are NOT scientific results.
- One capacity: C = 800. One seed: 8007 (smoke-only seed, not a design seed).
- Very short training: 4 episodes of the historical protocol (MIX12_EQ master pool,
  first rows, fixed order; entropy schedule; exact frozen PPO semantics).
- Tiny validation panel: first 2 VAL-v3 episodes per regime (12 regimes x 2 = 24
  episodes per validation), deterministic argmax evaluation.
- Smoke-only deviation (documented): checkpoint-bank episodes overridden from the
  protocol {50, 100, ..., 1000} to {2, 4} so the full bank + validation + reload
  path executes in seconds. `run_config.json` still records the protocol default
  field `checkpoint_every: 50`; the effective smoke override is recorded here and
  in `smoke/smoke_results_*.json`.

## Per-candidate checks

| Check | SC-M | IA-SC | TP-SC |
|---|---|---|---|
| training loop runs (4 episodes) | PASS | PASS | PASS |
| params finite | PASS | PASS | PASS |
| gradients finite after update | PASS | PASS | PASS |
| no NaN/Inf in rewards/metrics/update losses | PASS | PASS | PASS |
| checkpoint bank works (ckpt_ep0002/0004 + bank_manifest.json) | PASS | PASS | PASS |
| validation works (2 validations, finite macro12) | PASS | PASS | PASS |
| torch RNG unchanged at every validation | PASS | PASS | PASS |
| model reload works (exact state_dict equality) | PASS | PASS | PASS |
| deterministic evaluation works (identical repeat metrics) | PASS | PASS | PASS |
| Money identity (training + deployment eval) | PASS | PASS | PASS |
| parameter count (77907 / 80083 / 77907) | PASS | PASS | PASS |
| TP prior decays as coded | PASS (lambda==0, teacher_loss==0) | PASS (lambda==0, teacher_loss==0) | PASS (lambda(0)=0.01, matches 0.01*max(1-e/250,0) exactly) |
| IA feature pipeline stable (q(x) 9-dim, c(x,a_s) 4-dim, aug 83-dim, all finite) | n/a | PASS | n/a |
| **SMOKE PASS** | **PASS** | **PASS** | **PASS** |

Wall time per candidate: SC-M 4.8 s, IA-SC 4.9 s, TP-SC 4.0 s (includes 2 tiny-panel
validations, checkpoint save/load, reload-identity, deterministic re-evaluation and
one manual PPO update for gradient finiteness).

## Bug found and fixed by smoke

`common/evaluate.py::evaluate_panel` inserted the scalar `macro12_mean_money` key
into the per-regime dict before computing `macro12_value_accept_ratio`, so the
second comprehension iterated over a float (`TypeError`). Unit tests did not cover
this code path; the smoke run exercised it on the first validation. Fixed by
snapshotting the regime names before inserting scalar keys. Full unit suite
re-run afterwards: 38/38 PASS.

## Smoke validation numbers (DO NOT INTERPRET)

Val macro12 Money after 2 and 4 training episodes (engineering signal only,
reported for traceability; absolutely not model performance):

- SC-M: [-3309.58, 30.42]
- IA-SC: [29.58, 16.67]
- TP-SC: [-3309.58, 30.42]

Artifacts: `smoke/smoke_results.json`, per-candidate JSON files
`smoke/smoke_results_SCM.json`, `smoke/smoke_results_IASC.json`,
`smoke/smoke_results_TPSC.json`, run directories `smoke/{SCM,IASC,TPSC}_C800_s8007/`
with `run_config.json`, `training_metrics.json`, `run_summary.json`, `bank/`.
