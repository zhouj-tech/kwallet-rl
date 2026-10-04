# Stage 3.8 — SC-FAC-Centered Model Exploration (CLOSED)

**Status:** CLOSED — 2026-10-04. No candidate promoted. Branch: post-hoc exploratory, not confirmatory.

## Lineage

- Scientific base commit: `281e02920a32e9878177593d62c7f2d841ed6078`
- Frozen lineage: `5573ec642f0f28c218f3e6058478f62ab6db6b2b`
- Executed in worktree `kwallet-aamas-explore`, branch `aamas2027-explore-sandbox`, under `sandbox/aamas2027_explore/scfac_candidates/` (source sandbox left intact).
- Archival branch: `aamas2027-stage3-8-archive` (this archive). Never merged to main.

## Final verdict

| Verdict | Value |
|---|---|
| IA_SC_SCREEN_PASS | NO |
| TP_SC_SCREEN_PASS | NO |
| SCREEN_SURVIVORS | NONE |
| RECOMMEND_HELDBACK_SEED532_CHECK | NO |

No CHECK-v3 / FULL-DEV-v3 / NEW13 / seed532 artifact was authorized or consumed.

## Final SCREEN-v3 means (macro12 Money)

| Model | C800 | C1200 |
|---|---:|---:|
| SC-M | 3576.7427 | 14379.2766 |
| IA-SC | 3720.0443 | 14102.5219 |
| TP-SC | 3967.0005 | 14480.1307 |
| BF (fixed reference) | 4661.9917 | 14687.3979 |

## Layout

- `STAGE3_8_FINAL_SUMMARY.md` — full closure summary (rationale, audit findings,
  candidates, protocol, per-seed results, gate interpretation, conclusions,
  unexecuted idea bank).
- `design/` — design + pre-sprint audit + overnight plan/handoff documents.
- `protocol/` — SCFAC-DESIGN-v3 manifests, split hashes/overlap validation,
  overnight matrix, run ledger, implementation/smoke/runtime-benchmark receipts.
- `code/` — final executed candidate implementations (`sc_m/`, `ia_sc/`,
  `tp_sc/`, shared `common/` library), the evaluation/analysis runtime
  (`runtime/`), split generator (`generate_design_v3.py`), and
  `provenance_runtime/` (recovery scheduler — infrastructure, not a method).
- `results/` — compact final reports and CSV/JSON aggregates.
- `screen_eval/` — all SCREEN-v3 evaluation JSONs: BF references (C800/C1200),
  16 learned best_by_money records (IA-SC, TP-SC), and 8 SC-M `__all.json`
  records containing all three checkpoint-selector designations
  (best_by_money / best_by_value_accept_ratio / best_by_training_return).
- `cell_summaries/` — 24 training cells (`<CELL_ID>/` with run_config.json,
  run_summary.json, training_metrics.json) plus `CELL_INDEX.csv`.
- `ARCHIVE_MANIFEST_SHA256.txt` — SHA-256 of every archived file (except itself),
  deterministic sorted relative paths.

## Checkpoint banks

The 24 full checkpoint `bank/` directories (20 checkpoints per cell) are NOT in
this Git archive. They are preserved in the external full backup:

`~/Desktop/kwallet-archives/KWallet_Stage3_8_FULL_BANK_20261004.tar.gz`

(plus `.sha256` and `_MANIFEST.txt` sidecars). Source banks in the sandbox were
not deleted.

## Prohibitions carried forward

- Do not reinterpret SCREEN-v3 after the fact; the gate is frozen.
- Do not call this campaign confirmatory — it is post-hoc exploratory.
- Do not consume seed532 / CHECK-v3 / FULL-DEV-v3 / NEW13 based on these results.
