# Stage 3 raw acceptance

Scientific completeness: **PASS — 42/42 jobs, 100,800 episode rows.**
Bookkeeping completeness: **38/42 in the original ledger; reconciled, not repaired.**
Raw SHA256: `576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`. Frozen scientific lineage: `5573ec642f0f28c218f3e6058478f62ab6db6b2b`.

## Acceptance checks

- Every named job has episodes.csv and result.json, 2400 rows, twelve regimes with 200 unique episode indices each, and recorded PASS.
- Frozen matrix matches 40 learned evaluations (four methods × two capacities × five seeds) plus two deterministic rule evaluations.
- Money = settled − 10 × flushes, count identities, finite values, ratios, complete episode coverage, and saved summaries verified independently.
- Maximum Money identity error: 0.0; maximum saved-summary arithmetic difference: 1.13686837722e-13 (absolute tolerance 1e-9).
- All episode/pool fingerprints match the frozen NEW12 manifest; exogenous requested values match across jobs.
- 155 internal SHA256 entries verified; 157 regular files covered by the outer tarball hash. Internal checksum exclusions: README.md, SHA256SUMS.txt.
- Result source hashes, adapter/matrix/artifact-manifest bindings, embedded job definitions and scientific git lineage verified.

## Four ledger omissions

The missing completed entries are exactly the frozen orchestrator's VALIDATION_REPS. Each has valid full scientific outputs and matching Mac/server episode hashes in MAC_RUNTIME_APPROVED.json:

- `A-EVAL-IF-C800-S123` — 2400 rows; PASS; episodes SHA256 `596bbb1df69a882cc5afd3382905962889e15ff26142972c5058ee510cacd79c`.
- `A-EVAL-SC-C800-S123` — 2400 rows; PASS; episodes SHA256 `ffcbb566f9ee3264bbdedd352d0fed55da13f74e7b304002204db0c09785260b`.
- `B-EVAL-ZERO-C800-S123` — 2400 rows; PASS; episodes SHA256 `c2cfe9ced5caceb0fee8e33fa67a751940b203389d1c87cb0e611b5ecf6a6ae7`.
- `C-EVAL-BFT05-C800` — 2400 rows; PASS; episodes SHA256 `820f8ec0d4ee835697adaa883a7251b2512c712ac8ee3e8e3be36be5490ecd78`.

The orchestrator skipped these already-valid outputs rather than adding completed entries. The full log also records a resume skip of JA-PPO C800/123, which already has a ledger entry; it is not a fifth omission. The root-level output/episodes.csv is byte-identical to A-EVAL-JA-C800-S123/episodes.csv and is excluded as a duplicate pilot. Only the 42 job directories enter statistics.

## Provenance qualifications (not concealed)

- Raw freeze contains outputs/manifests, not checkpoint bytes, pool arrays, training receipts, or all original runtime/gate receipts; those cannot be independently rehashed from this archive.
- Legacy local-benchmark labels and null runtime_lock_sha256 are retained, not repaired; acceptance uses the provided final freeze, embedded MAC_RUNTIME_APPROVED chain and five representative episode-hash parity attestations.
- Approval header Python/Torch version strings differ from per-job runtime_local strings; report both rather than asserting literal lock equality.

Approval header: `{"numpy_version": "2.0.2", "python_version": "3.10.12", "torch_version": "2.5.1+cpu"}`.

Actual job runtime groups:

```json
[
  {
    "jobs": 42,
    "runtime": {
      "nproc": 10,
      "numpy_version": "2.0.2",
      "platform": "Darwin 25.5.0 arm64",
      "processor": "arm",
      "python_version": "3.10.22",
      "torch_interop_threads": 10,
      "torch_threads": 4,
      "torch_version": "2.5.1"
    }
  }
]
```

The reported approval attests Mac-versus-server parity for Stage-2 representative jobs. It is not a new claim of universal cross-platform equality or a reopening of historical SC-FAC drift. No raw outputs or ledger entries were modified. Analysis proceeds from the user-designated frozen official bundle, with these provenance limitations visible.

Exact per-job checks and hashes: stage2b_job_summary.csv. Exact raw-file inventory: raw_file_inventory.csv.
