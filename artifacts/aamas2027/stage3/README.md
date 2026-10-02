# Stage 3 — frozen Stage 2B analysis

The immutable source is `../stage2/raw/KWALLET_AAMAS_STAGE2B_RAW_20261002.tar.gz`, SHA256 `576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`. Scientific lineage is `5573ec642f0f28c218f3e6058478f62ab6db6b2b`. This directory contains derived analysis only; no evaluator is imported or executed.

From the repository root, use a **separate analysis environment** (tested on Python 3.12). Never install packages into or change the frozen evaluation runtime:

```sh
python3 -m venv /tmp/kwallet-stage3-analysis
/tmp/kwallet-stage3-analysis/bin/python -m pip install -r artifacts/aamas2027/stage3/code/requirements.txt
/tmp/kwallet-stage3-analysis/bin/python artifacts/aamas2027/stage3/code/run_stage3_analysis.py
/tmp/kwallet-stage3-analysis/bin/python artifacts/aamas2027/stage3/code/test_stage3_analysis.py
/tmp/kwallet-stage3-analysis/bin/python artifacts/aamas2027/stage3/code/run_stage3_analysis.py
```

The last command includes the test receipt in the final deliverable hash inventory. It does not rerun simulation. `--audit-only` performs extraction and acceptance without importing SciPy/Matplotlib. `--raw PATH --out PATH` supports another location. The raw archive must be outside the derived output root.

Each run rechecks the tarball, safely extracts immutable working copies under ignored `work/raw/`, verifies all internal checksums and per-job scientific outputs, then recomputes every table, figure, statistics.json, and report. Existing extracted bytes must match the tarball; they are never silently replaced. Existing derived files are regenerated deterministically. No notebook/manual table edits are used. Original files and extracted raw inputs are rehashed after analysis. Run in one process at a time.

The primary inference unit is the training seed (n=5), not episode or regime. Learned policy means and paired effects use df=4 Student t intervals. A/B/C Holm families have exactly 6/2/2 tests. Family C uses one-sample tests of five SC-minus-fixed-BF deltas. BF has one deterministic score per capacity, n_training_seeds=0 and null uncertainty; its row is retained in seed_level_scores.csv with a blank seed, not five fabricated rows. Full precision is retained in CSV/JSON; report numbers are rounded for presentation. Regime-level and decomposition tables are descriptive only.

Acceptance and provenance qualifications are in STAGE3_RAW_ACCEPTANCE.md. In particular, 38 completed ledger entries plus four approved skipped validation representatives constitute 42 valid scientific jobs. A root-level JA pilot duplicate is excluded. Legacy benchmark labels, null runtime-lock fields, and inconsistent approval-header version strings are preserved and disclosed. Pool/checkpoint bytes and original training receipts are not in this results archive; the analysis does not pretend to independently verify those absent files.

`statistics.json` contains all numeric tables and protocol/runtime metadata. `artifact_inventory.csv` and `SHA256SUMS.txt` identify the deliverables; `raw_file_inventory.csv` covers every extracted source file. Four figure families are exported as vector PDF/SVG and preview PNG. Figure captions state the unit of uncertainty and descriptive status. Tests compare actual contrasts against independent SciPy APIs, check an analytic df=4 p-value identity, family correction, degeneracy, seed pairing, data units, and input guards. The integration test also regenerates all 34 scientific outputs (tables, reports, JSON, and figure exports) in a separate temporary directory directly from the frozen tarball, requiring byte-identical hashes. The test receipt records the compared hashes; inventories and code/test receipts themselves are excluded from that comparison.

No historical ICDM numbers, PR #3 artifacts, new simulations, or revised statistical tests enter these results. This is an analysis handoff, not a paper rewrite or an authorization to push.
