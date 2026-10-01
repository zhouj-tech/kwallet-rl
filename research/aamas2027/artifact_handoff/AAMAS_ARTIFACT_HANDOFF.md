# AAMAS recovered-artifact handoff

Prepared 2026-10-01, original-lineage core C800/1200. Repository base verified locally: `02bde16`. This handoff authorizes no experiment and records no replay result. No server was accessed; no transfer, commit, or push was performed. Historical source files are copied without modification; collaborator PR #3 artifacts are excluded.

## Package and canonical source

Local payload root: `/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/`.

Canonical source root: `/Users/zhouzhou/Downloads/KWallet_Organized/`. Principal and zero checkpoints/metadata come from `02_完整研究资料/04_ActorCritic与SCFAC研究/ac/results`; historical pools and pool metadata come from `01_日常研究`. Each checkpoint/pool matches the previously recovered archive fingerprint. Other byte-identical source copies are not included.

The adjacent `AAMAS_ARTIFACT_MANIFEST.csv` is the authoritative per-file inventory, with complete source paths, destination-relative paths, sizes, SHA-256, method/settings/seed/condition mode, timestamp, provenance, and pool fields. Empty fields mean not applicable, not unknown experimental values. `full` is SC-FAC's full-conditioning mode; JA/IF use `not_applicable`. The package itself contains only historical payload files, not extra copies of these generated handoff documents.

## Exact contents

| Type | Files |
|---|---:|
| best_checkpoint | 36 |
| run_info | 36 |
| run_config | 26 |
| result_metadata | 36 |
| training_history | 36 |
| validation_history | 36 |
| pool | 14 |
| pool_summary | 14 |
| pool_manifest | 2 |

**Totals: 36 best checkpoints, 14 pools, 186 metadata files; 236 payload files; 93,859,040 bytes (89.511 MiB).** This is the logical transfer payload size, not filesystem block allocation or compressed size. Metadata adds 52,336,984 bytes to the 41,522,056 checkpoint/pool bytes. Manifest/document files live in Git and are not counted in that payload.

Principal selections: JA-PPO / IFAC / SC-FAC × C800/1200 × seeds123,323,532,777,999 =30. Zero-conditioning selections: SC-FAC `zero_settle_embedding` × C800/1200 × seeds123,323,532 =6. Every selection has run_info, result JSON, training history, and validation history. Standalone run_config exists for 26 of 36; the remaining ten have their complete config in run_info, and no replacement config was generated.

Settings without a standalone run_config: IFAC C800 seed123, IFAC C800 seed323, IFAC C800 seed532, IFAC C800 seed777, IFAC C800 seed999, IFAC C1200 seed123, IFAC C1200 seed323, IFAC C1200 seed532, IFAC C1200 seed777, IFAC C1200 seed999.

Best checkpoints are selected by the original run metadata and completed-run timestamp. Last checkpoints are unnecessary: run_info contains the configuration/selection policy, validation history supplies selection context, and the result JSON supplies expected evaluation. A historical path mentioning a last model inside unchanged metadata is not a requirement to transfer it. No last weight, plots, summary-text duplicate, alternate seed/configuration, or other experiment result is packaged.

The 14 arrays are the 5,000×1,000 MIX12_EQ master, the 300×1,000 validation pool, and twelve 200×1,000 static evaluation pools. The original run uses 1,000 training episodes and 200 validation episodes; preserve these distinctions. SHA-256 hashes complete .npy files; historical MD5 fingerprints hash array payloads. Pool generation summaries and both original seed/fingerprint manifests are included. Do not generate replacement pools.

## Packaging validation

The assembly procedure checks exactly 36 selected best checkpoint identities, 14 pool identities, all required metadata, unique source paths, unique destination paths, unique file SHA-256 values, and exact source/destination byte hashes. It also rejects any unexpected destination file. Post-copy verification must pass before this package is described as ready. Reading/hashing artifacts is not runtime replay; model deserialization and simulator execution remain pending.

## Exact transfer command — PREPARED ONLY, NOT EXECUTED

```sh
rsync -av --checksum --partial --progress /Users/zhouzhou/Desktop/kwallet-aamas-artifacts/ wku-gpu:/data/sijia/aamas2027_artifacts/
```

The trailing source slash transfers the contents into the intended remote root. No `--delete` is used. Obtain explicit transfer authorization before running it. Do not treat rsync's transport checksum as a substitute for the manifest SHA-256 check. The receiver must compare every manifest path, size and SHA-256 before replay and reject missing/extra payload files. These handoff files must reach the receiver through a separately approved Git commit/pull or explicit document handoff; neither is executed here.

## First replay specification — STOP AFTER THIS CHECK

- Method: original **JA-PPO** (`BasicPPOAgent`), never IFAC/SC-FAC or collaborator code.
- C=800, k=24, F=3, T=1000; training seed=123; historical timestamp=`20260508_222529_095916`.
- Best checkpoint relative path: `checkpoints/basic_ppo/basic_ppo_trainMIX12_EQ_C800_k24_T1000_F3_seed123/20260508_222529_095916/best_model.pth`.
- Best checkpoint SHA-256: `bb00559d921e41f589618148e2b8c208faa4ccfbb92d8825b2b41fd3a65ca6d1`.
- Original US pool relative path: `pools/US_static_eval_T1000.npy`.
- US pool SHA-256: `3f9fc877a2a251f29ec2f9a18cc2b363dc1584d4cc8593df0eb4fe9142655455`; historical fingerprint `md5_array_payload:b15a94b5a2ac36da2485a2307d3fba25`.
- Expected result relative path: `metadata/runs/basic_ppo/basic_ppo_trainMIX12_EQ_C800_k24_T1000_F3_seed123/20260508_222529_095916/cross_regime_results.json`; SHA-256 `0666aa725a2ddd18426c40030f079a8c022c045308561c179efe2d3a3ba9cd28`.
- Expected JSON subtree: `test_results.US`, including `num_episodes` and every `summary` statistic below. This is the US-only result, not the macro-regime mean Money.
- Evaluate all 200 US episodes in original stored order, 1,000 steps each, resetting the original environment per episode. No resampling or stream generation. Use deterministic joint-action argmax (`agent.act(..., deterministic=True)`) and original action encoding/transition order.
- Restore the original architecture and selected weights using run_info config. Use CPU for the first replay to match the recorded training/evaluation device; no GPU performance assumption is needed. Use the original code at `src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py` and `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py`. Call the existing evaluation path rather than a default script entry point that might start training. Do not alter checkpoint weights or historical files.
- Rebase historical absolute **file locations** in memory to the handed-off paths only. Do not rewrite archived JSON or change environment/reward/model/evaluation settings. Record software versions, dtype/device and path mappings. A runtime incompatibility is a stop condition, not permission to patch semantics.
- Aggregate using the original `summarize_episode_metrics`: mean, population std (`numpy.std`, ddof=0), min, max and median. Acceptance ratios are averaged per episode, not recomputed as a ratio of pooled totals. Money uses p=1 and tau=10 with executed flush count.

### Exact archived US metrics

200 episodes. Every value below is copied directly from the selected archived result JSON, not recomputed by a new rollout.

| Metric | Mean | Population std | Min | Max | Median |
|---|---:|---:|---:|---:|---:|
| settled | 5282.88 | 260.2901181374353 | 4650.0 | 5977.0 | 5270.5 |
| drops | 802.535 | 9.609306686749049 | 772.0 | 825.0 | 803.0 |
| oversize_drops | 774.545 | 12.19991700791444 | 737.0 | 807.0 | 776.0 |
| insufficient_drops | 27.99 | 5.210556592150209 | 14.0 | 40.0 | 28.0 |
| flushes | 193.9 | 9.54515583948214 | 173.0 | 223.0 | 193.0 |
| utilization | 0.0066036 | 0.000325362647671794 | 0.0058125 | 0.00747125 | 0.006588125 |
| avg_tx_value | 26.75417902144778 | 0.29318689583219726 | 26.017543859649123 | 27.697916666666668 | 26.76 |
| drop_rate | 0.802535 | 0.009609306686749034 | 0.772 | 0.825 | 0.803 |
| value_accept_ratio | 0.1057368704219604 | 0.0060129086552053995 | 0.09069277578405367 | 0.12285966823572941 | 0.10539464009861954 |
| count_accept_ratio | 0.197465 | 0.009609306686749048 | 0.175 | 0.228 | 0.197 |
| total_requested_value | 49985.275 | 536.9457229320296 | 48570.0 | 51344.0 | 50027.0 |
| total_tx_count | 1000.0 | 0.0 | 1000.0 | 1000.0 | 1000.0 |
| accepted_count | 197.465 | 9.609306686749049 | 175.0 | 228.0 | 197.0 |
| eval_money_p | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| eval_money_tau | 10.0 | 0.0 | 10.0 | 10.0 | 10.0 |
| eval_money | 3343.88 | 168.68753836605714 | 2920.0 | 3797.0 | 3343.0 |

Exact nonnumeric summary fields:

- `training_objective` = `original_reward`
- `evaluation_money_formula` = `eval_money = money_p * settled - money_tau * flushes`
- `money_method` = `true_money_via_settled`

### Pass/fail tolerance and required response

1. **Before replay:** every payload size and SHA-256 must match the manifest exactly. Checkpoint identity/configuration and pool dimensions/dtype/order must match. Hash mismatch, missing metadata, wrong checkpoint, or unsupported loading is **FAIL / STOP**.
2. **Replay coverage:** `num_episodes == 200` exactly; each episode has 1,000 decisions. Require all archived numeric fields and all nonnumeric metadata above; a missing field fails the comparison.
3. **Numeric comparison:** for each of the 16 metrics × five summary statistics, require finite values and absolute error **≤1e-9**, with **relative tolerance 0**. Episode counts and metadata strings must match exactly. This is an arithmetic tolerance, not a statistical confidence interval; different actions/returns cannot be excused as seed variation. Save the actual values and absolute differences for all 80 numeric comparisons.
4. **Decision:** PASS only if every check passes. The archived record contains aggregate summaries, not the complete historical per-episode action trace; passing cannot prove bitwise-identical hidden activations or action traces. Preserve new per-episode outputs separately for diagnosis, never overwrite historical JSON.
5. **On any difference:** stop and report the first divergence plus all metric deltas, hashes, checkpoint timestamp, config/path mappings, and runtime versions. **Do not change code, environment, reward, pool, dtype, or tolerance to force agreement. Do not retrain or proceed to the wider campaign.** Ask the PI to assess the discrepancy.
6. A first-replay PASS is a gate report, not permission to launch the remaining experiments. Await explicit continuation authorization.

## Git boundary

Only `research/aamas2027/artifact_handoff/AAMAS_ARTIFACT_MANIFEST.csv` and `research/aamas2027/artifact_handoff/AAMAS_ARTIFACT_HANDOFF.md` are new Git candidates for this task. The package is outside the repository. Do not include PROJECT_STATUS.md, collab/, checkpoints, pools, or generated replay results in a commit. No commit, push, remote connection, transfer, or experiment is performed by preparing this handoff.
