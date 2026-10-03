# H1 Multi-Seed Exploratory Expansion — K-Wallet AAMAS 2027, Stage 3.5

**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** This expansion asks one
question: does the seed-123 B0 finding for hybrid **H1** (frozen BF-T0.5
settlement + frozen SC-FAC conditional flush head, conditioned on the BF
settle index) generalize across the other four frozen SC-FAC training seeds?

No training, no tuning, no new method, no frozen artifact modified.

## Matrix

- Hybrid: **H1 only** (BF settle + SC flush conditioned on the BF settle index,
  joint action through the unmodified E0).
- Capacities: C=800, C=1200.
- Seeds: 123, 323, 532, 777, 999 — **seed 123 is reused unchanged** from the
  validated B0 pilot (`../b0_diagnostic/outputs/pilots/H1_C{800,1200}_S123/`,
  sha256-pinned in `outputs/seed123_reference_sha.json`); the other 8 cells are
  newly evaluated here.
- Protocol: NEW12-v1, 12 regimes × 200 episodes, T=1000, deterministic argmax,
  original reward (Money = settled − 10·flushes), frozen streams identical for
  every seed.

## Per-cell provenance gates (all must pass before a cell is written)

1. Checkpoint SHA256 matches the frozen Stage 2 job matrix **and** the frozen
   Stage 3 `seed_level_scores.csv` for that (C, seed).
2. Frozen `adapter.check_config`: C, k=24, F=3, T=1000, `cfg.seed == seed`,
   cpu, original reward, `conditional_factorized_ac`, condition_mode=full,
   settle_embed_dim=32, conditional_hidden_size=256.
3. The loaded checkpoint must reproduce its own frozen SC-FAC NEW12 episodes
   **exactly** on provenance regimes US/TPLS/PLB (200 episodes each) when run
   as pure SC/SC — proving the checkpoint is the one that produced the frozen
   reference for that seed.
4. Per episode: Money identity, accepted+drops=T, finite logits/metrics, valid
   action ranges (asserted inside the shared B0 evaluator).

## Layout

```
h1_multiseed/
  code/run_h1_multi.py        # verify | run | validate | analyze | all
  code/test_h1_stats.py       # offline numerical tests (Student-t, stats)
  outputs/
    C800_S323/ ... C1200_S999/  # 8 new cells: episodes.csv (2400 rows,
                                # frozen 35-field schema) + result.json
    seed123_reference_sha.json  # sha256 pins of the reused B0 outputs
    H1_VALIDATION.json          # section-11 validation record
    run_console.log
  H1_SEED_LEVEL_SCORES.csv      # 10 rows: 5 seeds x 2 C (+ per-regime money)
  H1_METHOD_SUMMARY.csv         # mean/SD/SE/95% t CI/min/max per capacity
  H1_PAIRED_VS_SC.csv           # per-seed deltas + per-C paired t summary
  H1_VS_BF.csv                  # per-seed deltas vs fixed BF + summary
  H1_REGIME_LEVEL.csv           # 120 rows: seed x C x regime detail
  statistics.json               # all computed statistics
  H1_MULTI_SEED_REPORT.md       # sections 1-14 + screening verdict
  README.md
```

## Reproduction

```bash
PY=/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python
cd /Users/zhouzhou/Desktop/kwallet-aamas-b0
M=artifacts/aamas2027/stage3_5/h1_multiseed
$PY $M/code/run_h1_multi.py verify      # all 10 frozen bundles + hashes
$PY $M/code/test_h1_stats.py            # 8 offline stats tests
$PY $M/code/run_h1_multi.py run --all   # 8 new cells (refuses overwrite)
$PY $M/code/run_h1_multi.py validate    # section-11 gate
$PY $M/code/run_h1_multi.py analyze     # CSVs + statistics.json + report
```

`run` writes each cell only if its directory does not yet exist; `analyze`
refuses to run unless `outputs/H1_VALIDATION.json` records PASS.

## Interpretation discipline

n=5 seeds (df=4): confidence intervals are wide and raw two-sided p-values are
descriptive screening statistics, not significance claims; nothing is
multiplicity-adjusted. NEW12-v1 is the same stream family used for Stage 2B/3
method selection, so results are not a fresh confirmatory test. BF-T0.5 is a
single deterministic score per capacity — no BF seed uncertainty exists or is
fabricated. Regime-level patterns are descriptive; regimes are not independent
experimental replicates. Any trainable successor (B1) is out of scope here and
would require a fresh held-out confirmatory stream release.
