# PR body (ready to paste)

Head: `Yingda-Yu:integration/yingda-icassp2027-handoff`
Base: `zhouj-tech:main`

Title:
**Integrate ICASSP 2027 K-Wallet reproduction, structured policies, and evaluation suite**

---

## Background

This PR integrates the complete research line developed on the
`Yingda-Yu/Kwallet-Rl` fork since the shared baseline. The recovered
upstream repository contained the DQN-era code but no PPO implementation
of the methods described in the internal K-Wallet manuscript. We
therefore (1) ported the environment and 12-regime traffic generator
faithfully, (2) reimplemented the learned methods as a clean canonical
`kwallet` package, (3) added a permutation-equivariant policy that
transfers across wallet counts, (4) ran a frozen, fully manifested
multi-seed experiment suite with paired statistics, and (5) built an
ICASSP 2027 manuscript whose every number traces to a committed CSV via
`paper/icassp2027/assets/paper_claims.json`.

The branch is a true merge (no squash, no force rewrite) on top of the
latest upstream `main` (`bc47a4c`), so the upstream recovery baseline
(`src/ideaextra`, `src/idea5`) and the new research-workflow files
(`research/`) are fully preserved.

## What was added

- `src/kwallet/`: canonical package — ported environment
  (`envs/kwallet.py`), data layer (`data/`), PPO (`training/ppo.py`),
  four learned policies (`policies/`): **JA-PPO** (joint (k+1)^2 head),
  **IFAC** (independent factorized heads), **SC-FAC**
  (settle->flush conditioned factorization, with no-conditioning and
  shuffled-conditioning ablations), and **Set-SC-FAC / Set-IFAC**
  (permutation-equivariant, k-independent set encoder); rule baselines
  FA/FWF and a strong validation-selected feedback rule **BFP**;
  evaluation module for rollouts, parallel CPU evaluation, paired
  statistics, and abrupt-regime-switch evaluation.
- `pyproject.toml` installable package (`pip install -e ".[test]"`),
  CLI `python -m kwallet.cli`.
- `scripts/`: matrix/k-scale/switching drivers, aggregation, paired
  statistics, and a paper-asset generator that reads only result CSVs.
- `tests/`: 37 tests covering transition semantics, pool determinism,
  factorized probability math against enumerated distributions, set
  equivariance, and end-to-end smoke runs.
- `docs/`: code-to-claim map (PORT/REIMPLEMENTED/NEW evidence labels),
  reproduction report, decisions log, handoff guide.

## Experimental additions

- Five-seed stationary matrix at `C in {800,900,1000,1200}`, `k=24`,
  `F=3`, deterministic evaluation on fixed held-out pools.
- Capacity sweep; conditioning zero/shuffle ablations (n=3).
- Cross-k zero-shot transfer matrix over `k in {6,12,24}` (train at each
  k, deploy at all k without retraining; flat-MLP reference).
- Six-scenario abrupt regime switching (post-switch Money and avoidable
  drops).
- Efficiency analysis (625 vs 50 output logits, parameter counts,
  per-step CPU latency).
- Seed-paired contrasts, bootstrap 95% CIs, Holm step-down correction
  within pre-specified families; manifests in `experiments/` record
  every run including FAILED ones; compact tables in `results/tables/`
  with artifact SHA-256 manifest.

## Main findings

- Factorized heads reliably beat the quadratic joint head (SC-FAC vs
  JA-PPO significant at all four capacities after Holm; IFAC 3/4 raw).
- The settle-conditioned flush path shows **no** statistically detectable
  Money/additional-value gain over independent factors; the null result
  is reported with exact p-values and ablations rather than hidden.
- The set encoder enables genuine zero-shot deployment at unseen wallet
  counts (a flat k-shaped MLP cannot load at a different k), with
  asymmetric transfer quality; one failed k=24 training run (seed 532)
  is retained in the primary aggregate, so no matched-scale superiority
  is claimed.
- The strong BFP feedback rule is the strongest tested controller on
  stationary streams and after abrupt switches; learned policies beat the
  naive rules post-switch but not BFP — an explicit boundary result.

## Reproducibility

- `python -m pytest` -> **37 passed** after the merge.
- Full command sequence in `docs/YINGDA_HANDOFF_20260930.md`
  (gen-pools -> smoke -> main/ablation/kscale/switching -> aggregate ->
  paired stats -> paper assets -> pdflatex).
- Results are reproducible from committed scripts and manifests; frozen
  summary CSVs are committed for comparison while raw run directories
  stay ignored. PPO is seeded but not bit-reproducible across platforms.
- One integration fix is included: the root `.gitignore` bare `data/`
  pattern had silently hidden the source package `src/kwallet/data/`
  from git; rules are anchored now and the four source files are tracked.

## Paper

`paper/icassp2027/` contains the ICASSP 2027 submission candidate
(official spconf template): `main.tex`, a compiled 5-page `main.pdf`,
25 verified references, frozen figures, and auto-generated tables.
`assets/paper_claims.json` links headline numbers to source CSVs;
`FINAL_SUBMISSION_AUDIT.md` and `STATISTICAL_AUDIT.md` document the
freeze and the evidence ledger. The paper directory is listed in
`.gitignore` by historical convention but its files are deliberately
tracked here. Author metadata, copyright, EDICS, and final AI-disclosure
approval remain author-side actions; nothing has been submitted.

## Compatibility with upstream changes

- Merged from current upstream `main` with `--no-ff`; **no force push,
  no history rewrite, no squashing**.
- Only merge conflict was `.gitignore`, resolved as a rule union; every
  upstream ignore entry is retained.
- Upstream recovery/workflow commits, `research/` control files, and
  `src/ideaextra`/`src/idea5` code are untouched; there are zero file
  deletions in this PR.
- The fork's `AGENTS.md`/`TRAE_MASTER_PLAN.md` coexist with upstream's
  `research/` workflow documents; they do not conflict path-wise.

## Review guide

1. Start with `docs/YINGDA_HANDOFF_20260930.md` and
   `docs/PAPER_CODE_MAP.md` for orientation and provenance labels.
2. Environment semantics: `src/kwallet/envs/kwallet.py` +
   `tests/test_env.py`; data: `src/kwallet/data/` + `tests/test_data.py`.
3. Policy factorization math: `policies/actors.py`,
   `policies/set_actors.py` + `tests/test_policies.py`,
   `tests/test_set_equivariance.py`.
4. Frozen evidence: `experiments/*.csv` (run status) ->
   `results/tables/*_long.csv` (per-seed) ->
   `paper/icassp2027/assets/paper_claims.json` -> Tables 1-4 in
   `main.tex`.
5. Null/negative results: Sec. 4 (conditioning ablations, switching
   contrasts) and the failed-seed disclosure in Table 3/Discussion.
6. Integration mechanics: merge commit `7512a6e`, gitignore fix
   `cf2abfe`, diff summary in `docs/UPSTREAM_INTEGRATION_DIFF.md`.
