# IMPLEMENTATION_VALIDATION — SCFAC candidate experiments preflight report

Date: 2026-10-03. Scope: implementation of the SCFAC candidate experiments per
`MODEL_DESIGN.md` (this directory) and the assigned task specification.
Per the task spec: the SCREEN matrix was NOT started; nothing was committed or pushed.

## 1. Git/worktree status

- Current branch: `aamas2027-explore-sandbox`; worktree HEAD
  `281e02920a32e9878177593d62c7f2d841ed6078` (unchanged; no commits made).
- All new work confined to `sandbox/aamas2027_explore/scfac_candidates/`
  (untracked; nothing staged/committed/pushed).
- Frozen worktrees verified read-only for this session:
  - `kwallet-rl` HEAD `5573ec642f0f28c218f3e6058478f62ab6db6b2b`
  - `kwallet-aamas-b0` HEAD `4f19ece7680fba17e7d8f0f768b47a3c901fa67a`
  - `kwallet-aamas-explore` HEAD `281e029...` (this repo)
  - `kwallet-aamas-results` HEAD `96d4d1d64d489b65dcec7539e8c179d1505097cb`
  - `kwallet-aamas-stage4-d1` HEAD `281e029...`
  - `kwallet-aamas-artifacts` (data root, not a git worktree)
- Pre-existing dirty entries in frozen trees (`research/PROJECT_STATUS.md`,
  `collab/`, `NEXT_STAGE_EXPERIMENT_DESIGN.md`, cache dirs) predate this session;
  this session only READ from these trees (sandbox-enforced write protection).

## 2. Frozen evidence commit

- Worktree evidence commit: `281e02920a32e9878177593d62c7f2d841ed6078`
- Source SHA256s (frozen adapter `SOURCE_HASHES`, re-verified at vendoring time):
  - E0 env `kwallet_ctx_attn_fair_benchmark.py`:
    `d46246795552f22f0ba143ae38230692ff99997d57c9b0681f456432a9df1921`
  - SC-FAC trainer `run_conditional_factorized_ac_benchmark.py`:
    `c3710b3a8bbf905f2d047b5bf4444830cdc32239e39b49becc149540f6e79b04`
  - 12-regime generator `kwallet_ideaextra_generator.py` (vendored byte-identical):
    `d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8`

## 3. Files created

Under `sandbox/aamas2027_explore/scfac_candidates/`:

- `common/`: `frozen_env.py`, `frozen_kwallet_ideaextra_generator.py`,
  `config.py`, `pools.py`, `seedrule.py`, `features.py`, `teacher.py`,
  `models.py`, `ppo.py`, `evaluate.py`, `bank.py`, `train.py`, `release.py`,
  `validators.py`, `__init__.py`
- `sc_m/run_train.py`, `ia_sc/run_train.py`, `tp_sc/run_train.py`,
  `generate_design_v3.py`
- `configs/frozen_pool_sha256.txt` (training-pool hash cache)
- `data/scfac_design_v3/`: VAL-v3/SCREEN-v3/CHECK-v3/FULL-DEV-v3 `.npz`,
  `MANIFEST.json`, `OVERLAP_CHECK.json`, `SHA256SUMS.txt`
- `tests/`: `test_common.py`, `test_sc_m.py`, `test_ia.py`, `test_tp.py`,
  `test_data.py`, `run_all_tests.py`
- `smoke/run_smoke.py` + smoke JSON artifacts; `runtime/eval_screen.py` +
  benchmark artifacts; `outputs/SCM_C800_s323/`
- `reports/`: `SMOKE_RESULTS.md`, `RUNTIME_BENCHMARK.md`, this report

## 4. SC-M implementation

Conditional factorized actor-critic ("condH256") with exact frozen semantics:
encoder 74→128→128; settle head 128→25; settle embedding 25x32; conditional
flush head (128+32)→256→25; value 128→1. Flush distribution conditioned on the
settlement action (argmax in diagnostics, executed action in updates). Frozen
PPO protocol implemented verbatim (GAE, entropy schedule `min(ep/episodes,1)`,
advantage normalization, unclipped value loss, grad-norm clip). Training pool:
MIX12_EQ master, SHA256-verified, first 1000 rows in fixed order.

## 5. IA implementation (IA-SC)

83-dim state = base 74 + q(x) 9-dim (usable fraction, feasible fraction, usable
balance/C, max usable balance/w, v/w, oversize, p_1..p_3 refill-delay fractions).
Conditional flush input 164-dim = [h(128), emb(32), c(x,a_s) 4-dim] with
c = (u_s, b_s/w, 1[s in V], 1[s in V]*(b_s-v)/w); all-zero when a_s = k
(no-op). Paired initialization from SC-M: overlapping blocks copied verbatim,
new columns (encoder 74..82, flush-head 160..163) zeroed; functional equality
with SC-M at init is unit-tested (with a nonzero dummy c, proving the zeroed
columns contribute nothing). Per-step c(x, executed a_s) threaded through the
rollout batch, minibatch update, act(), argmax evaluation, and value bootstrap.

## 6. TP implementation (TP-SC)

Architecture identical to SC-M. Teacher labels from the frozen BF-T0.5
controller (verified line-by-line against
`kwallet-aamas-b0/tools/aamas_stage2/adapter.py::bf_action`): settlement =
min-(balance,index) over usable AND balance>=tx, no-op k; flush = min-(balance,
index) over usable, excluding the LEARNER's sampled a_s, balance < 0.5*C/k
strict, no-op k. Auxiliary loss = lambda_BF(e) * [-log pi_s(b(x)) - log
pi_f(q_BF(x,a_s)|x,a_s)]; lambda_BF(e) = 0.01*max(1-e/250, 0). Teacher labels
are computed ONLY while lambda > 0 (episodes 0..249); `common/evaluate.py` never
imports the teacher (AST-checked in tests; runtime BF-call counters assert zero
calls during deployment evaluation).

## 7. Parameter-count verification

- SC-M: 77,907 (unit-asserted, and printed by the full run)
- IA-SC: 80,083 = 77,907 + 9*128 (encoder) + 4*256 (flush head) (unit-asserted)
- TP-SC: 77,907 (unit-asserted)

## 8. Development release hashes (SCFAC-DESIGN-v3)

```
9e63d0c73ac11261f1bf563687921afc8a86c9a384d9fa48af420518de62a2c8  VAL-v3.npz
1ee83726bb9acfe93ea6d9775cf018383ffaa161db280e914d75736325f18d24  SCREEN-v3.npz
36a6f0bf61f4880e798d6f59e15dda6c09057649c421f6c57bf426c199ae041f  CHECK-v3.npz
3420582bd7828a24024ee719ac407fe237eb4ce135d1e7c238f2ec820884735e  FULL-DEV-v3.npz
```

Split sizes: VAL-v3 20/regime, SCREEN-v3 40, CHECK-v3 60, FULL-DEV-v3 200
(12 regimes each). Seed rule:
`SHA256("SCFAC-DESIGN-v3|<split>|<regime>|<zero-based-episode>|20261003|<collision-counter>")[:8]`
big-endian; collision counter increments only on (seed or fingerprint) collision
and every increment is logged. Generation output: 0 collisions.

## 9. Overlap validation

`OVERLAP_CHECK.json`: result **PASS**. Seed and episode-fingerprint disjointness
verified against all historical pools (master training 20000532-family, val
220000532-family, static eval 100000532-family, NEW12 710000001-family) and
across all four v3 splits; 8,100 historical fingerprints loaded and cross-checked.
Unit tests re-verify split sizes, seed-rule conformance, and full disjointness.

## 10. Unit-test results

Self-contained runner (`tests/run_all_tests.py`; frozen venv has no pytest):
**38/38 PASS** — frozen env layout/refill/Money identities, frozen-source
equivalence, GAE/entropy exactness, seed rule, pool hashes, BF label contracts,
D1_SEED_CAPACITY.csv unique-key rejection, Money validator, SC-M/IA-SC/TP-SC
architecture and behavior contracts, release data integrity and disjointness.

## 11. Smoke-test results

All three candidates **SMOKE PASS** (see `reports/SMOKE_RESULTS.md`): training
loop, finite gradients, no NaN/Inf, checkpoint bank, validation, model reload,
deterministic evaluation, Money identity, TP prior decay exactness, IA feature
pipeline stability. Smoke found and fixed one real bug
(`evaluate_panel` macro12 scalar-key iteration) — full suite re-passed after.

## 12. Runtime benchmark

Full-budget control run SC-M C800 s323 (see `reports/RUNTIME_BENCHMARK.md`):
training 231.0 s (4,330 transitions/s), 20 VAL-v3 validations 291.2 s
(16,483 steps/s), end-to-end 524.5 s (~8.7 min), peak memory ~410 MiB,
checkpoint bank 19 MB / 20 checkpoints verified, all designations at ep 1000,
final VAL-v3 macro12 Money 2959.50. SCREEN-v3 control evaluation (best_by_money
ep 1000): 29.5 s, macro12 2966.39 — **PRE-SCREEN CONTROL / DO NOT INTERPRET
ALONE**. 20-checkpoint count verified.

## 13. Known limitations

- Single-seed runtime control; no scientific or architectural conclusions may be
  drawn from it or from the SCREEN-v3 control number.
- IA-SC and TP-SC full-budget runs intentionally NOT started (per spec).
- No IA/TP paired-init full-run evidence yet beyond smoke.
- Peak memory measured via `/usr/bin/time -l` (not a profiler).
- Unit tests use a self-contained runner (frozen venv lacks pytest).
- Pre-existing dirty entries exist in some frozen worktrees (documented in
  section 1; unrelated to this work).

## 14. Deviations from MODEL_DESIGN.md

1. **Vendored frozen sources**: env/generator are byte-identical copies under
   `common/` (SHA256-verified) instead of in-place imports, because frozen
   worktrees must remain untouched. Trainer/BF semantics were transcribed and
   unit-tested against the frozen sources.
2. **Smoke-only checkpoint override**: smoke runs used checkpoint episodes {2, 4}
   to exercise the bank/validation/reload path quickly (documented in
   `reports/SMOKE_RESULTS.md`). The full-budget run used the protocol bank
   {50, 100, ..., 1000} with no override.
3. No other deviations; all protocol constants (PPO, entropy schedule, seed
   rules, split sizes, lambda schedule, Money identity) match MODEL_DESIGN.md.

## 15. Whether overnight SCREEN can safely start

YES — all unit tests pass (38/38), all three smoke suites pass, the SCFAC-DESIGN-v3
release is generated with PASS overlap validation, and the full-budget control
run completed cleanly at ~8.7 min/run end-to-end. The 12-run SCREEN matrix
(3 candidates x C800/C1200 x 2 seeds... per assigned matrix) is estimated at
~2-2.5 h sequential or ~1-1.5 h with 2 parallel workers on this 10-core machine.

IMPLEMENTATION_STATUS:
PASS

SC_M_READY:
YES

IA_SC_READY:
YES

TP_SC_READY:
YES

DESIGN_V3_READY:
YES

TESTS_PASS:
YES

SMOKE_PASS:
YES

FULL_SC_M_C800_S323_WALL_MINUTES:
8.74

RECOMMENDED_MAX_PARALLEL_WORKERS:
2

ESTIMATED_12_RUN_WALL_HOURS:
1.0-1.5 (2 workers) / 2.0-2.5 (sequential)

OVERNIGHT_SCREEN_READY:
YES

---

# Session-16 repair addendum (2026-10-03 overnight, Stage 3.8B)

PI-approved repairs to two preflight defects discovered at overnight start.
No scientific protocol, architecture, PPO setting, reward, feature definition,
or SCFAC-DESIGN-v3 stream was changed.

## Repair 1 — IA-SC paired initialization wired into the production path

MODEL_DESIGN C1-F requires IA-SC to start from the paired vanilla-control weight
blocks with added feature columns zero ("initial functions match before learning").
`init_ia_from_sc` existed and was unit-tested, but `train_candidate` built IA-SC
with a fresh seeded init. Fixed:

- `common/models.build_paired_ia_model()`: seed -> build SC-M reference init ->
  capture torch RNG state -> build IA-SC -> `init_ia_from_sc` (overlapping blocks
  copied verbatim, added columns zero) -> restore captured RNG state.
- `common/train.py` routes IA-SC through this helper (state_fn and agent
  wallet_size threaded at the same time; see repair 2).

Tests added (tests/test_ia.py): production-sequence exactness incl. RNG restore,
matched-SC(seed) vs IA-SC(seed) initial-function equality on valid states with
zeroed columns, driver-wiring AST check, and an end-to-end mini IA-SC run through
`train_candidate` (50 episodes, tiny VAL panel, rng-unchanged asserted).

## Repair 2 — C1200 capacity threading (restores capacity-generic MODEL_DESIGN)

The implementation only supported C=800 end-to-end. Threaded capacity C and
wallet size w = C/k through:

- `common/evaluate.py`: new `make_eval_env(capacity)`; `evaluate_agent_on_pool` /
  `evaluate_panel` accept capacity (fresh eval env per episode at the run capacity).
- `common/train.py`: validation uses `capacity=config.C`; IA state_fn uses
  `wallet_size = config.C / config.k`; agent gets wallet_size.
- `common/features.py`: `extract_q_features` / `extract_c_features` / `augment_state`
  accept capacity / wallet_size (formulas unchanged; default = frozen C800).
- `common/ppo.py`: `ConditionalFactorizedPPOAgent(wallet_size=...)` threads w into
  the IA c(x,a_s) derivations (act-path and diagnostic-path).
- `common/teacher.py`: `bf_flush_label(..., wallet_size=...)`; `teacher_labels` uses
  `env.wallet_size` — TP flush threshold is strictly 0.5 * C/k at the run capacity.
- Entry CLIs (`sc_m/ia_sc/tp_sc/run_train.py`): `--capacity` flag; output naming
  `*_C{C}_s{seed}`; `CommonRunConfig(C=...)`.

C800 behavior is bit-identical (regression-parity unit test). SCFAC-DESIGN-v3
streams untouched (checksums re-verified).

## Validation evidence (gates)

- Full unit suite: **49/49 PASS** (38/38 original + 4 paired-init production-path
  tests + 7 capacity tests; runner `tests/run_all_tests.py`).
- C1200 engineering smokes (through the production CLI at C=1200): **SC-M PASS,
  IA-SC PASS, TP-SC PASS** — training loop, checkpoint bank, validation,
  deterministic reload-evaluation at C1200 (identical twice), Money identity,
  IA C1200 feature spot checks, TP threshold = 0.5*(1200/24) spot checks.
  Runner: `runtime/smoke_c1200.py`; results `smoke/c1200/smoke_c1200_results.json`.

## Status after repairs

IMPLEMENTATION_STATUS: PASS
IA_PAIRED_INIT_WIRED: YES
C1200_SUPPORTED: YES
TESTS_PASS: 49/49
C1200_SMOKE_PASS: 3/3
OVERNIGHT_MATRIX_RESUMED: YES
