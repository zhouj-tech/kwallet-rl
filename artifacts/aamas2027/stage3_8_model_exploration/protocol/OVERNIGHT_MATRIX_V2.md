# OVERNIGHT_MATRIX_V2 — frozen before any IA-SC / TP-SC SCREEN-v3 score observation

Frozen: 2026-10-03 22:24 CST (14:24 UTC), autonomous overnight session, Stage 3.8B.
Authoritative task spec: overnight PI instruction (Stage 3.8B multi-seed SC-FAC model screening).
Evidence base commit: `281e02920a32e9878177593d62c7f2d841ed6078` (worktree verified at
`~/Desktop/kwallet-aamas-explore`, branch `aamas2027-explore-sandbox`; verified unchanged this session).

## Historical qualification (mandatory recording)

A full-budget SC-M C800 seed323 runtime-control run was already executed during preflight
and its engineering outputs exist under `outputs/SCM_C800_s323/` (training 522.2 s,
20-checkpoint bank, VAL-v3 designations all at episode 1000, rng untouched at validations).

Seed323 was already part of the original prospectively designed matrix before that runtime
run. The expanded 4-seed matrix below is being fixed now **before ANY IA-SC or TP-SC
SCREEN-v3 candidate score has been observed**. No IA-SC or TP-SC SCREEN-v3 score exists
on disk at freeze time; the only SCREEN-v3 number in existence is the SC-M control
number from the preflight runtime benchmark (clearly labeled PRE-SCREEN CONTROL).

## Frozen matrix

MODELS: SC-M, IA-SC, TP-SC

SCREEN TRAINING SEEDS: 123, 323, 777, 999

HELD-BACK TRAINING SEED: 532

CAPACITIES: 800, 1200

Every seed is fully crossed with both capacities.

Initial full training matrix: 3 models x 4 seeds x 2 capacities = **24 full-budget learned-policy runs**.

| Cell | Model | C | Seed | Status at freeze |
|---|---|---|---|---|
| SCM_C800_S123 | SC-M | 800 | 123 | planned |
| SCM_C800_S323 | SC-M | 800 | 323 | reuse-prefight (protocol-equivalent, verified) |
| SCM_C800_S777 | SC-M | 800 | 777 | planned |
| SCM_C800_S999 | SC-M | 800 | 999 | planned |
| SCM_C1200_S123 | SC-M | 1200 | 123 | planned |
| SCM_C1200_S323 | SC-M | 1200 | 323 | planned |
| SCM_C1200_S777 | SC-M | 1200 | 777 | planned |
| SCM_C1200_S999 | SC-M | 1200 | 999 | planned |
| IASC_C800_S123 | IA-SC | 800 | 123 | planned |
| IASC_C800_S323 | IA-SC | 800 | 323 | planned |
| IASC_C800_S777 | IA-SC | 800 | 777 | planned |
| IASC_C800_S999 | IA-SC | 800 | 999 | planned |
| IASC_C1200_S123 | IA-SC | 1200 | 123 | planned |
| IASC_C1200_S323 | IA-SC | 1200 | 323 | planned |
| IASC_C1200_S777 | IA-SC | 1200 | 777 | planned |
| IASC_C1200_S999 | IA-SC | 1200 | 999 | planned |
| TPSC_C800_S123 | TP-SC | 800 | 123 | planned |
| TPSC_C800_S323 | TP-SC | 800 | 323 | planned |
| TPSC_C800_S777 | TP-SC | 800 | 777 | planned |
| TPSC_C800_S999 | TP-SC | 800 | 999 | planned |
| TPSC_C1200_S123 | TP-SC | 1200 | 123 | planned |
| TPSC_C1200_S323 | TP-SC | 1200 | 323 | planned |
| TPSC_C1200_S777 | TP-SC | 1200 | 777 | planned |
| TPSC_C1200_S999 | TP-SC | 1200 | 999 | planned |

## Freeze rules (no exception tonight)

- No seed replacement. No dropping bad seeds. No adding seed532 to SCREEN.
- No replacing a capacity. No candidate-specific training budget.
- Every run: 1000 episodes, T=1000, k=24, F=3, MIX12_EQ first-1000 fixed order,
  frozen PPO, checkpoint bank {50..1000 step 50}, VAL-v3 selection; retain
  best_by_money / best_by_value_accept_ratio / best_by_training_return + all 20 checkpoints.
- Primary policy for all method comparison: best_by_money. No selector switch after scores seen.
- Held-back seed 532: only after a SCREEN survivor; CHECK-v3 stage.

## Preflight verification at freeze time

- Worktree HEAD = evidence commit (OK); branch `aamas2027-explore-sandbox` (OK); nothing staged (OK).
- SCFAC-DESIGN-v3: SHA256SUMS all OK; OVERLAP_CHECK result PASS, 0 seed / 0 fingerprint collisions.
- Frozen training pool file SHA256 = cached expected = cell provenance (OK).
- Runtime lock: Python 3.10.22 / NumPy 2.0.2 / Torch 2.5.1 / torch threads 4 (OK).
- SC-M C800 s323 cell verified protocol-equivalent for reuse.

## Preflight blockers discovered this session (RESOLVED — PI-approved repairs)

1. **IA-SC paired initialization (RESOLVED).** `common/train.py` now routes IA-SC through
   `common/models.build_paired_ia_model()`: global seed set exactly as matched SC-M ->
   SC-M reference init built first -> torch RNG state captured immediately after ->
   IA-SC built -> `init_ia_from_sc` (overlapping blocks copied verbatim, added columns
   zero) -> captured RNG restored before training. Production-path tests added
   (build exactness incl. RNG restore, matched-SC functional equality, driver wiring
   AST check, and an end-to-end mini IA run through `train_candidate`).
2. **C1200 support (RESOLVED).** Capacity C and wallet size w = C/k are threaded through
   `evaluate.make_eval_env`/`evaluate_agent_on_pool`/`evaluate_panel`, the training
   validation path, IA q(x)/c(x,a_s) features, the agent-level c-feature derivation,
   the TP BF teacher flush threshold (strictly 0.5*C/k via env.wallet_size), the
   run CLIs (`--capacity`) and output naming (`*_C1200_s*`). Formulas unchanged;
   C800 behavior bit-identical (regression-parity test). SCFAC-DESIGN-v3 streams
   untouched.

Validation evidence: full unit suite **49/49 PASS** (was 38/38; +4 paired-init
production-path tests, +7 capacity tests). C1200 engineering smokes **3/3 PASS**
(SC-M, IA-SC, TP-SC through the full production CLI at C=1200, incl. checkpoint bank,
validation, deterministic reload-eval at C1200, Money identity, IA feature spot checks,
TP threshold spot checks) — see `runtime/smoke_c1200.py`,
`smoke/c1200/smoke_c1200_results.json`.

Both blockers confirmed resolved; the prospectively frozen 24-run matrix is unblocked.

## Matrix completion rules

- Full matrix must finish before any candidate promotion decision.
- A partial matrix must be labeled INCOMPLETE. Never declare a winner from partial results.
- No git add / commit / push / merge anywhere in any worktree tonight.