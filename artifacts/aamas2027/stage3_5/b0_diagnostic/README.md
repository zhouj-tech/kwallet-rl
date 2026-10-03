# B0 Hybrid Diagnostic — K-Wallet AAMAS 2027, Stage 3.5

**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** B0 does not introduce a
new winning method, retrains nothing, and changes no frozen artifact. It takes
the frozen **BF-T0.5** rule components and the frozen **SC-FAC seed=123**
policy and re-combines them on the frozen **NEW12-v1** streams to localize the
Stage 3 SC-FAC → BF-T0.5 Money gap:

- recovery from replacing **settlement choice**, or
- recovery from replacing **flush choice**,
- including their interaction.

## Hybrids

| ID | Settlement | Flush |
|----|------------|-------|
| H1 | BF-T0.5 settle | SC-FAC conditional flush head, **conditioned on the BF settle index** |
| H2 | SC-FAC settle (frozen deterministic decoding) | BF-T0.5 flush, excluding the SC settle wallet |

Frozen BF rules (k = 24 wallets, capacity C, no-op index k):

- settle: `s = argmin_{i usable, b_i >= x} (b_i, i)`; no-op if infeasible.
- flush:  `f = argmin_{j usable, j != s, b_j < 0.5 * C/k} (b_j, j)`; no-op if
  none eligible. Threshold is strictly `<`, ties break by wallet index, the
  settlement wallet is excluded, a no-op settlement does not suppress flushing,
  and an oversized/infeasible transaction never suppresses a valid flush.

The joint action `s*(k+1)+f` is submitted to the **unmodified frozen E0
environment** (which processes the flush before the settlement each step). H1
feeds the BF settle index directly into SC-FAC's
`forward_flush_given_settle(state, settle)` — the SC settle head is never used
to select H1's settlement.

## Pilot matrix (exactly four runs)

NEW12-v1, 12 regimes × 200 episodes, T=1000, deterministic actions, identical
environment/reward/protocol to Stage 2B, SC checkpoint seed=123 only:

1. H1, C=800  2. H1, C=1200  3. H2, C=800  4. H2, C=1200

No additional seeds without PI approval.

## Layout

```
b0_diagnostic/
  code/
    b0_lib.py        # frozen IO/loaders, BF helpers, hybrid action, evaluator
    run_b0.py        # verify | parity | smoke | pilot | analyze | all
  tests/
    test_b0.py       # unittest suite (22 tests), no pytest needed
  outputs/
    parity/parity_ep200.json     # exact-replay gate vs frozen episodes
    smoke/                       # 3-episode smoke results + traces
    pilots/{H1,H2}_C{800,1200}_S123/{episodes.csv,result.json}
    B0_ANALYSIS.json
  B0_PILOT_SUMMARY.csv           # one row per pilot, all metrics + comparisons
  B0_PER_REGIME.csv              # 48 rows: hybrid x C x regime
  B0_DIAGNOSTIC_REPORT.md        # tables, decomposition, screening, caveats
  README.md
```

## Frozen inputs (read-only, hash-pinned)

- Stage 2 adapter/environment imported from the frozen lineage
  (`5573ec6…`) via `tools/aamas_stage2/adapter.py`; its SHA is verified by
  `verify`.
- SC-FAC seed=123 checkpoints for C=800/C=1200 (SHA-verified).
- NEW12-v1 streams (manifest-verified).
- Stage 2 raw episode tarball
  (`KWALLET_AAMAS_STAGE2B_RAW_20261002.tar.gz`,
  SHA256 `576c9baa…89ace3`) and Stage 3 tables
  `artifacts/aamas2027/stage3/{seed_level_scores.csv,regime_level_scores.csv}`;
  reference numbers are **loaded**, never hardcoded, and the Stage 3 tables are
  cross-checked against raw episodes at tolerance 1e-9 during `analyze`.

Paths default to `/Users/zhouzhou/Desktop/kwallet-rl`,
`/Users/zhouzhou/Desktop/kwallet-aamas-artifacts`, and
`/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/data/streams/NEW12-v1`; override
with `B0_FROZEN_REPO`, `B0_FROZEN_ARTIFACTS`, `B0_NEW12_ROOT` if relocated.

## Reproduction

Use the interpreter that contains the frozen stack (Python 3.10, numpy 2.0.2,
torch 2.5.1 CPU):

```bash
PY=/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python
cd /Users/zhouzhou/Desktop/kwallet-aamas-b0
B0=artifacts/aamas2027/stage3_5/b0_diagnostic

$PY $B0/code/run_b0.py verify                 # hash/manifest verification
$PY $B0/tests/test_b0.py                      # 22 focused unit tests
$PY $B0/code/run_b0.py smoke                  # 3-episode smoke, identity checks
$PY $B0/code/run_b0.py parity --save          # exact replay gate (2x12x200)
$PY $B0/code/run_b0.py pilot --all            # the four pilots
$PY $B0/code/run_b0.py analyze                # CSVs + JSON + report
```

`run_b0.py all` runs the whole sequence in order. The pilot command refuses to
start unless the cached parity gate passes, and pilot writers refuse to
overwrite existing pilot directories.

## Guarantees and gates

- Before any pilot, B0 reproduces **every metric of every frozen SC-FAC seed123
  and BF-T0.5 episode exactly** (24 regime blocks, three replays each: SC/SC,
  BF/BF through the joint-action environment, BF/BF through the SC-FAC
  evaluation environment).
- Per episode the code asserts `Money = settled_value − 10·flushes` and
  `accepted + drops = T`, plus finiteness of all logits/metrics/actions.
- No file under frozen Stage 2/Stage 3 paths, checkpoints, streams, or
  historical run_info is ever written.

## Screening heuristics (not significance tests)

A hybrid is flagged *promising for expansion* if **either**:

- A. it improves over SC-FAC seed123 at **both** capacities; or
- B. at either capacity, Δ vs SC123 > +200 Money **or** gap recovery
  `(M_hybrid − M_SC123)/(M_BF − M_SC123)` > 0.30.

After the four pilots the run stops; expansion to multiple seeds, training, and
any commit/push/merge require explicit PI approval.
