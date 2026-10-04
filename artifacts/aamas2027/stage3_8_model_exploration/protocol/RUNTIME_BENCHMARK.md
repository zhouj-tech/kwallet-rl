# RUNTIME BENCHMARK — SC-M C=800 seed=323 (full-budget control run)

Date: 2026-10-03. Purpose per task spec section 12: ONE full-budget
scientific-format control training to measure real wall time, validation time,
resource usage, and checkpoint-bank footprint. This is a single-seed
engineering/runtime control.

**DO NOT infer architecture quality from this single run.**

## Configuration

- Model: SC-M (conditional factorized AC, 77,907 params)
- Capacity: C = 800; seed: 323; 1000 training episodes, MIX12_EQ master pool
  first 1000 rows in fixed order (SHA256-verified)
- Protocol: frozen historical PPO (lr 3e-4 Adam, gamma 0.99, GAE lambda 0.95,
  clip 0.2, critic 0.5, 4 epochs, minibatch 256, grad-clip 1.0, entropy
  0.03→0.003 linear over 1000 episodes)
- Checkpoint bank: 20 validations at episodes 50..1000 step 50 on VAL-v3
  (20 episodes x 12 regimes, deterministic argmax, torch-RNG-untouched asserted)
- Runtime: `/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python`
  Python 3.10.22, numpy 2.0.2, torch 2.5.1, CPU-only, torch threads = 4,
  Apple M5 (10 cores)

## Wall time

| Phase | Seconds | Notes |
|---|---|---|
| Training rollout + PPO updates (1e6 env transitions) | 231.0 | 4,330 transitions/sec |
| VAL-v3 validations (20 x 240 episodes = 4.8M argmax steps) | 291.2 | 16,483 steps/sec |
| Checkpoint saves (20) | 0.1 | |
| **Total (in-loop)** | **522.2** | |
| **End-to-end real** (incl. pool load, I/O) | **524.5** | user 1078.7 s, sys 68.7 s |

Training throughput is dominated by the PPO update (64 minibatch backward
passes per episode); validation is 480 argmax-only episodes per checkpoint.

## Resource usage

- Peak memory footprint: 429,605,896 bytes (~410 MiB), `/usr/bin/time -l`
- Device: CPU only; threads pinned to 4 (`torch.set_num_threads(4)`)
- Average core utilization ≈ 2.1 cores (user+sys / real)
- Checkpoint-bank disk footprint: 19 MB (20 x ~0.9 MB `.pth` + manifests)

## Checkpoint bank verification

- 20 checkpoint files present (`ckpt_ep0050.pth` .. `ckpt_ep1000.pth`): VERIFIED
- torch RNG unchanged at every validation: TRUE
- Designations (all tie-broken by the frozen rules; final episode dominates here):
  - best_by_money: episode **1000** (final VAL-v3 macro12 Money 2959.50)
  - best_by_value_accept_ratio: episode **1000**
  - best_by_training_return: episode **1000**
- bank_manifest.json: `outputs/SCM_C800_s323/bank/bank_manifest.json`

## SCREEN-v3 evaluation of the completed checkpoint

**PRE-SCREEN CONTROL / DO NOT INTERPRET ALONE.**

- Checkpoint: best_by_money = ep 1000, shared deployment pipeline
  (reload + deterministic argmax; zero BF calls)
- SCREEN-v3: 480 episodes (40 x 12 regimes), eval seconds: **29.5**
- macro12 mean Money: 2966.39 (value_accept_ratio macro12 in
  `runtime/screen_eval_scm_c800_s323.json`)
- Artifacts: `runtime/eval_screen.py`, `runtime/screen_eval_scm_c800_s323.json`,
  `runtime/scm_c800_s323_train.log`, `outputs/SCM_C800_s323/`

## Overnight SCREEN planning (from this control run)

- Per-run wall estimate: SC-M ≈ 8.8 min; IA-SC/TP-SC slightly higher (larger
  inputs / teacher labels) → budget ≈ 10-12 min per run
- 12-run SCREEN matrix sequential: ≈ 2-2.5 h; with 2 parallel workers: ≈ 1-1.5 h
- Recommended max parallel workers: 2 (each with 4 torch threads on 10 cores)
