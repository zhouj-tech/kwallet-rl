# AAMAS Stage 2 evaluation adapter

Stage 1 is approved. JA-PPO/IFAC historical replay passed bit-exactly. SC-FAC current-runtime repeats passed exactly at summary and episode levels; its negligible historical cross-platform difference is accepted and is not investigated here. **G3 is not yet accepted.** Local contract tests are not learned-policy replay evidence.

This additive tool imports only the hash-pinned original idea3 environment and idea4 learned evaluators. It has no trainer or generator entry point. Nothing imports collaborator PR #3. New evaluation requires explicit release flags and hash-bound G2/G3 receipts. OLD validation cannot select NEW pools and cannot consume future zero checkpoints.

## Architecture and invariants

- Load all 46 CSV rows and validate the fixed grid: four training rows depend only on G0/G1; all 42 evaluation rows retain G0/G1/G2/G3. The adapter refuses to execute training rows.
- Verify all 236 handed-off files (36 checkpoints, 14 pools, 186 metadata), all five historical source hashes, selected checkpoint/config, and the complete recorded runtime snapshot. Verify again before publishing. Source imports disable bytecode writes; historical cache paths must be external.
- Construct the original agent, strict-load its BEST state on CPU, and call original `evaluate_agent_on_array`. Do not call `.eval()`, change dtype, alter action masks, or replace historical deterministic action selection. JA uses joint argmax; IF uses independent argmax heads; SC uses settle argmax then conditional flush argmax. Full/zero conditioning comes from verified config.
- BF-T0.5 independently selects the smallest `(balance,index)` feasible usable settlement wallet and the smallest eligible *other* usable wallet below `0.5*C/k` to flush. Threshold is strict; oversized requests still permit flushing; no-op index is k. Submit `s*(k+1)+f` to unchanged E0, which owns flush-first transitions, freezing, and full replenishment. No future transactions enter the rule.
- Validate 1000 transactions per episode, count/value/Money identities, finite metrics and exact coverage. Arithmetic checks use absolute `1e-9`, zero relative tolerance; this does not relax exact current-runtime parity.
- Publish `episodes.csv`, `result.json`, and hashed `receipt.json` only after complete validation. Optional SC gate values are preserved in `diagnostics.json` and original regime summaries. OLD outputs also contain `parity.json`. Partial files stay in hidden `.incomplete-*` directories; completed directories are never overwritten.
- OLD parity reloads the same checkpoint into a second original agent and directly calls the original evaluator. All raw metrics and summaries must be exactly equal, and serialized CSV metrics must exactly round-trip. JA/IF archived summary metrics also require exact equality. SC historical comparison is explicitly `NOT_A_GATE`; current-runtime SC parity remains exact. BF has no original policy evaluator, so its acceptance uses rule contract tests plus OLD E0 validity checks.
- A subset validation has `macro12_money=null`; it cannot be mistaken for a fresh 12-regime score. Every output says `OLD12_VALIDATION` or `NEW12-v1` and includes checkpoint, pool, source, adapter, matrix, runtime and git provenance.

Existing server dependencies only: Python (3.10+ syntax), NumPy, PyTorch and Matplotlib from the **approved locked runtime**. Do not install, upgrade or adjust versions to make a check pass. Unit tests use standard `unittest` and NumPy; no pytest needed.

## Trae: OLD-only acceptance commands

These commands are for Trae to run on the already approved server runtime after this engineering package is synchronized. Codex has not run them remotely. They perform no training or NEW12 work. Stop on the first failure and report receipts/logs; do not alter code, environment, inputs or tolerance.

Use exactly the approved Stage 1 launcher and thread/determinism settings. Preserve any existing OMP/MKL/OpenBLAS/PYTHONHASHSEED settings; **do not invent new settings**. If Stage 1 set PyTorch flags inside a launcher, use that same setup before invoking this adapter. The capture below records the process as it actually runs and must be checked against the approved Stage 1 report.

```sh
cd /data/sijia/kwallet-rl
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/data/sijia/aamas2027_stage2/runtime_cache/xdg
export MPLCONFIGDIR=/data/sijia/aamas2027_stage2/runtime_cache/matplotlib
export AAMAS_REPO=/data/sijia/kwallet-rl
export AAMAS_ARTIFACTS=/data/sijia/aamas2027_artifacts
export AAMAS_UNIT_RECEIPT=/data/sijia/aamas2027_stage2/unit_tests/receipt.json
/data/sijia/.venvs/kwallet/bin/python tools/aamas_stage2/test_adapter.py
```

Set `AAMAS_STAGE1_REPORT` to the actual final approved report (its path was not supplied to Codex). The file must exist. No fabricated runtime lock/report is included.

```sh
: "${AAMAS_STAGE1_REPORT:?Set the actual approved Stage 1 report path}"
/data/sijia/.venvs/kwallet/bin/python tools/aamas_stage2/adapter.py capture-runtime \
  --repo /data/sijia/kwallet-rl \
  --artifacts /data/sijia/aamas2027_artifacts \
  --work-root /data/sijia/aamas2027_stage2 \
  --stage1-report "$AAMAS_STAGE1_REPORT"
```

The candidate is `runtime_capture/runtime_lock.json`. It deliberately has `runtime_confirmed=false`. Compare its snapshot to the approved Stage 1 runtime and record confirmation by setting only `runtime_confirmed=true` in this new administrative file. Do not change the snapshot to force a match. This confirms runtime identity; it does not reopen G0 or the accepted SC drift. Compute and retain the SHA after confirmation:

```sh
AAMAS_RUNTIME_LOCK=/data/sijia/aamas2027_stage2/runtime_capture/runtime_lock.json
AAMAS_RUNTIME_SHA=$(sha256sum "$AAMAS_RUNTIME_LOCK" | cut -d ' ' -f 1)
```

Bounded acceptance cohort: seed123 for JA/IF/full SC/zero SC, both capacities, plus both deterministic BF jobs; exactly 10 OLD-US jobs. Each learned job evaluates 200 OLD-US episodes twice (adapter and direct evaluator); BF evaluates 200 once. Total 3.6 million OLD E0 steps. This is adapter validation, not another Stage 1 campaign. Outputs are under `old_validation/<job_id>/`.

```sh
for AAMAS_JOB in \
  A-EVAL-JA-C800-S123 A-EVAL-IF-C800-S123 A-EVAL-SC-C800-S123 B-EVAL-ZERO-C800-S123 C-EVAL-BFT05-C800 \
  A-EVAL-JA-C1200-S123 A-EVAL-IF-C1200-S123 A-EVAL-SC-C1200-S123 B-EVAL-ZERO-C1200-S123 C-EVAL-BFT05-C1200
do
  /data/sijia/.venvs/kwallet/bin/python tools/aamas_stage2/adapter.py old-parity \
    --repo /data/sijia/kwallet-rl \
    --artifacts /data/sijia/aamas2027_artifacts \
    --work-root /data/sijia/aamas2027_stage2 \
    --runtime-lock "$AAMAS_RUNTIME_LOCK" --runtime-lock-sha256 "$AAMAS_RUNTIME_SHA" \
    --job-id "$AAMAS_JOB" --regimes US || break
done
```

Any failed command returns nonzero and publishes no success receipt. Do not rerun completed output directories. The next command rejects missing, stale, failed or incomplete evidence, including an incomplete loop:

```sh
/data/sijia/.venvs/kwallet/bin/python tools/aamas_stage2/adapter.py g3-candidate \
  --repo /data/sijia/kwallet-rl --artifacts /data/sijia/aamas2027_artifacts \
  --work-root /data/sijia/aamas2027_stage2 \
  --runtime-lock "$AAMAS_RUNTIME_LOCK" --runtime-lock-sha256 "$AAMAS_RUNTIME_SHA" \
  --unit-test-receipt "$AAMAS_UNIT_RECEIPT"
```

This creates `g3_candidate/g3_receipt.json` with **approved=false** and hashes of all ten OLD receipts plus unit-test evidence. Authorized G3 acceptance requires reviewing those results and confirming original-source hashes/no historical modifications; then record `approved=true` and freeze its SHA. No NEW12 outcome is needed or permitted for this decision. Any changed adapter or matrix invalidates the old receipts and runtime binding. No G3 PASS is claimed in this commit candidate.

## Future release interface (not authorized to execute now)

`evaluate` additionally requires `--allow-new12 --new12-root PATH --freeze-receipt PATH --freeze-receipt-sha256 SHA --g3-receipt PATH --g3-receipt-sha256 SHA`. These inputs do not exist as a claim of this implementation. There is intentionally no NEW12 runnable command in this handoff.

The G2 freeze receipt must contain `dataset_id="NEW12-v1"`, `approved=true`, and `manifest_sha256`. The manifest uses the plan's schema with explicit keys: `schema_version=1`, `bit_generator="PCG64"`, `calibration_seed=1234567`, `calibration_sample_size=200000`, `no_results_viewed_attestation=true`, regime order, and hash bindings `runtime_lock_sha256`, `job_matrix_sha256`, `OLD_manifest_sha256`, `adapter_sha256`, `plan_sha256`, `generator_sha256`, `overlap_report_sha256`. Every pool has `regime`, `filename`, `episode_count=200`, `T=1000`, `dtype="int32"`, `order="C"`, `episode_seeds`, `episode_hashes`, `size_bytes`, `sha256`, `payload_sha256`, `burst_mask_filename`, `burst_mask_sha256`. Payload/episode hashes are canonical little-endian int32 C-order bytes. `overlap_report.json` requires `status="PASS"`, `collision_count=0`, `old_episode_count=7700`, `new_episode_count=2400`; generation reproducibility and seed disjointness remain G2 sign-off evidence. The adapter independently checks actual value/mask files and content overlap against all original pools and between new episodes, before inference and before publication.

Future zero-training receipts are inputs, not produced by this evaluation adapter. Each requires `status="complete"`, exact `job_id`, `runtime_lock_sha256`, `source_hashes`, `exit_status=0`; `best_checkpoint_path`, `best_checkpoint_sha256`, `run_info_path`, `run_info_sha256`, `config_sha256`; `training_history_path/sha256`, `validation_history_path/sha256`, `result_path/sha256`; `selected_validation_episode`, `selected_validation_score`. Each `_path/sha256` notation denotes two fields, e.g. `training_history_path` and `training_history_sha256`. Paths must remain under the matching new training job directory. The adapter verifies 1000 finite returns, all 20 validations at 50-step cadence, value-accept-ratio selection with earliest strict-max tie rule, config identity, CPU/mode/shape, and referenced hashes before loading. Trae must preserve the trainer log tying the selected BEST bytes to that validation event; metrics alone cannot cryptographically prove a training trajectory. No receipt writer or training execution is part of this task.

## Local validation record

The development machine ran contract tests using NumPy and the recovered immutable local payload. Checks cover source/artifact hashes, all recovered model/config mappings, all OLD pool structures, rule edge cases, output/accounting rejection, runtime drift rejection, and no training entry point. It lacks PyTorch and Matplotlib, so original learned-policy replay was **not run locally**. G3 remains pending the commands above. No NEW12 files or outcomes were generated or inspected.
