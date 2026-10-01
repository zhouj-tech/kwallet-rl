# Upstream integration diff summary

Date: 2026-09-30
Branch: `integration/yingda-icassp2027-handoff`
Base: `zhouj-tech/kwallet-rl:main` (`bc47a4c`)
Merge: `7512a6e` (no-ff merge of `work/icassp2027-reproduce-improve`
`e2732a8`) plus `cf2abfe` (gitignore/source-tracking fix).

Commands used for this summary:

```bash
git diff --stat upstream/main...HEAD      # 110 files, +11568/-3
git diff --name-status upstream/main...HEAD
```

The three-dot diff shows exactly what this branch introduces relative to
the merge base shared with upstream: **109 added files, 1 modified file,
0 deletions, 0 renames**. Every file recovered by upstream since the
common baseline (`src/ideaextra/`, `src/idea5/`, `research/`) is kept
byte-for-byte; the merge touched no upstream-tracked file except
`.gitignore` (rule union, see below).

## CODE — 22 files, `src/kwallet/`

New canonical installable package (pyproject-driven, `src/` layout):

- `envs/kwallet.py` — ported environment (3k+2 state, flush-first,
  cooldown/refill, drop taxonomy, Money).
- `data/` — `regimes.py`, `pools.py`, `switching.py` (12-regime generator
  port, deterministic cached pools).
- `policies/actors.py` — JA-PPO, IFAC, SC-FAC (+ no-cond/shuffled
  ablations); `policies/set_actors.py` — Set-IFAC, Set-SC-FAC.
- `training/ppo.py` — clipped PPO/GAE.
- `baselines/rules.py`, `baselines/rules_strong.py` — FA/FWF, BFP.
- `evaluation/` — rollout, parallel CPU shards, paired stats, switching
  OOD, parameter/latency compute.
- `cli.py` — doctor/gen-pools/train/evaluate/bench.

## SCRIPTS — 12 files, `scripts/`

Experiment drivers and asset pipeline: `run_experiments.py` (smoke/main/
ablation tiers with run manifests), `run_kscale.py`, `run_switching.py`,
`aggregate_results.py`, `aggregate_kscale.py`, `paired_stats.py`,
`make_paper_assets.py`, `make_concept_figures.py`, pool/queue helpers.
The three `run_kwallet_dqn_baseline_*` scripts and the `idea5` shell
drivers predate/continue the upstream recovery line and are untouched.

## EXPERIMENTS — 4 files, `experiments/`

Frozen manifests `manifest_matrix_smoke.csv`, `manifest_matrix_main.csv`,
`manifest_matrix_ablation.csv`, `manifest_kscale.csv` — per-run method,
C, k, F, seed, episode count, DONE/FAILED status, artifact path, wall
time.

## RESULTS — 15 files, `results/tables/`

Compact committed tables only (raw `runs/` remain ignored): long per-seed
CSVs, main/regime tables, paired contrasts with CIs and raw p-values,
tau/post-hoc breakdowns for smoke/main/ablation, the k-scale transfer
table, and `SUBMISSION_ARTIFACT_SHA256.json` artifact hashes.

## PAPER — 36 files, `paper/icassp2027/`

ICASSP 2027 manuscript tree: `main.tex`, compiled 5-page `main.pdf`,
`refs.bib`, frozen figure assets (incl. author-provided conceptual
figures), auto-generated `assets/tables/`, `assets/paper_claims.json`
machine-readable claim ledger, and review/audit documents
(`FINAL_SUBMISSION_AUDIT.md`, `STATISTICAL_AUDIT.md`, gate/checklist
files). The directory matches the repository's long-standing ignore
policy (`paper/icassp2027/` listed in `.gitignore`) but the paper files
are force-tracked deliberately so the submission candidate ships with
the code. The author-provided `paper/old paper.pdf` and the template zip
are intentionally NOT committed.

## TESTS — 5 files, `tests/`

`test_env.py` (transition/drop semantics), `test_data.py` (pool
determinism/hashing), `test_policies.py` (factorized log-probabilities
vs enumerated distributions), `test_set_equivariance.py`,
`test_smoke.py` — 37 tests, all passing after the merge.

## DOCS — 12 files, `docs/`

Research documentation: `PAPER_CODE_MAP.md` (PORT/REIMPLEMENTED/NEW
evidence labels), `REPRODUCTION_REPORT.md`, `DECISIONS.md`,
`ENVIRONMENT.md`, `NOVELTY_AND_OVERLAP.md`, `CHANGELOG_RESEARCH.md`,
`AGENT_PROGRESS.md`, `BLOCKERS.md`, `AGENTS_MERGE_PROPOSAL.md`,
`TRAE_START_HERE.md`, `TRAE_TASK_BOARD.md`,
`TRAE_KWALLET_ICASSP2027_EXECUTION.md`; plus this handoff set
(`YINGDA_HANDOFF_20260930.md`, `UPSTREAM_PR_BODY.md`).

## ROOT — 4 files

- `pyproject.toml` (new) — package metadata/deps; pytest config.
- `AGENTS.md`, `TRAE_MASTER_PLAN.md` (new, fork-side workflow docs) —
  kept alongside upstream's `research/` workflow files; no conflict with
  them.
- `.gitignore` (the only modified file): union of both branches' rules.
  Upstream additions kept verbatim (`paper/`, `*.zip`, `logs/`,
  `.Rhistory`, `paper_planning/**/submission_package/`,
  `src/idea3/fair_benchmark_results/`, `src/idea5/general_model/results_*/`);
  fork additions kept (`runs/`, `*.egg-info/`, `paper/icassp2027/`);
  `.trae/` added for local IDE state. The bare `data/` and `results/`
  patterns were anchored to `/data/` and `/results/` because the bare
  `data/` rule had silently swallowed the source package
  `src/kwallet/data/` — those four source files are newly tracked in
  commit `cf2abfe`. Root-level generated `data/` and `results/`
  directories remain ignored.

## Verification after merge

- `python -m pytest` → 37 passed (on the integration branch, after
  tracking the previously ignored `src/kwallet/data/` package).
- `main.pdf` rebuilds from `main.tex` → 5 pages, 0 errors, 0 overfull.
- No scientific output file was regenerated during integration; the PDF
  in the tree is the frozen candidate rebuilt only for a read-only build
  sanity check and then restored to the committed bytes.
