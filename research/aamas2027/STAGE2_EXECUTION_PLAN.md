# Stage 2 execution plan — frozen design, NOT authorization to execute

Stage 1 is scientifically reproduced and approved by the PI. JA-PPO and IFAC passed bit-exact historical replay. Two SC-FAC runs on the locked current server runtime are exactly identical at summary and per-episode levels. The negligible historical/current SC-FAC numerical drift is an accepted cross-platform/runtime reproducibility note, not a failed gate. Do not investigate the historical 0.16 Money difference further.

This revision implements bounded additive engineering and OLD-only validation tools. It does not authorize training, NEW12 generation/evaluation, or a runtime change. No new scientific results are claimed. The locked-server OLD parity commands still require execution by Trae; local contract tests are not a substitute.
## 1. Fixed scope, gates, and job counts

Exactly **46 scientific jobs**: four zero-control training jobs +30 principal fresh evaluations +10 zero-control fresh evaluations +two BF-T0.5 fresh evaluations. Zero principal training jobs. Each fresh evaluation is 12 regimes ×200 episodes ×1000 steps. Full SC-FAC results from family A are reused by families B and C, never rerun or counted twice. The four trainer invocations also automatically evaluate OLD12; these built-in historical diagnostics are not extra fresh jobs and must never be merged into NEW12 results.

STAGE2_JOB_MATRIX.csv enumerates every job separately, including immutable recovered checkpoint paths/hashes and run_info paths. Generated zero-checkpoint paths are bound by a training receipt, not an invented timestamp or “latest file” search. `training_required=true` denotes exactly four training rows; a later zero-evaluation row has `false` and depends on its training job. NA is the rule's seed, not a sixth learned seed.

Runtime roots: repository `/data/sijia/kwallet-rl`; immutable handed-off artifacts `/data/sijia/aamas2027_artifacts`; NEW outputs `/data/sijia/aamas2027_stage2`. Each evaluation writes `<output_path>/result.json` and `episodes.csv`; training writes under its unique output_path and later gets `receipt.json`. No original result directory is a destination.

Required gates (dependency names are used in the CSV):

- **G0_STAGE1_APPROVED — SATISFIED:** explicit PI approval in the current instruction. Preserve the final Stage 1 report and its runtime provenance. The accepted SC-FAC cross-platform drift does not reopen this gate.
- **G1_ARTIFACT_RUNTIME:** verify all handoff sizes/hashes, source hashes below, six zero-config consensus, and record `runtime_lock.json` from the approved Stage 1 environment. Include the literal interpreter path, Python/NumPy/PyTorch versions, installed-package lock, platform/CPU, BLAS/OpenMP libraries, thread counts, device/dtypes, deterministic settings, and approved Stage 1 report hash. Use CPU and the same locked runtime throughout. No package install/upgrade, precision change, compilation, thread change, or automatic fallback after locking.
- **G2_NEW12_FROZEN:** after G0, generate/check/freeze NEW12 using section 2. Sign off the manifest and its SHA-256 before any NEW12 policy result is produced or viewed. Freeze this plan, job CSV, source/handoff manifests, runtime lock, and adapter version hashes alongside it. Hashes of future arrays are intentionally absent now.
- **G3_ADAPTER_ACCEPTED:** validate the additive adapter in section 5, verify it on approved OLD-only fixtures/parity checks, and confirm no historical source diff or PR #3 dependency. No NEW12 outcome is used for adapter acceptance. This is an engineering gate, not a new scientific choice.

Dependencies: the four B-TRAIN-ZERO jobs require only G0 and G1, because they use historical training/validation pools and the unchanged trainer. G2 and G3 are not training dependencies. All NEW12 evaluations retain G0/G1/G2/G3; the four new-zero evaluations also require their corresponding completed training receipt. Suggested evaluation release order is G1 → G3 (OLD-only) → G2. Execution still requires an explicit release; default serial CPU execution avoids oversubscription and runtime drift. No job may choose another model, seed, capacity, threshold, reward, or test stream.

## 2. NEW12-v1 — exact design only

Original source: `/data/sijia/kwallet-rl/src/ideaextra/kwallet_ideaextra_generator.py`. SHA-256: `d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8`. Call **generate_regime_pool(regime_key, num_episodes=200, episode_length=1000, base_seed=..., calibration_sample_size=200000)** directly, not the CLI whose seed offsets differ. Retain its REGIME_SPECS, PCG64 via NumPy default_rng, calibration seed1234567, rounding, int32 values, clipping and burst behavior exactly. Do not import the misleading organized file named original_regime_generator.py, which contains table-generation code.

Regime order and per-regime episode seed intervals:

| i | Regime | base_seed | Inclusive episode seeds | Pool filename |
|---:|---|---:|---|---|
| 0 | US | 710000001 | 710000001–710000200 | `US_NEW12_v1_T1000.npy` |
| 1 | TLS | 710100001 | 710100001–710100200 | `TLS_NEW12_v1_T1000.npy` |
| 2 | LNS | 710200001 | 710200001–710200200 | `LNS_NEW12_v1_T1000.npy` |
| 3 | TLNS | 710300001 | 710300001–710300200 | `TLNS_NEW12_v1_T1000.npy` |
| 4 | TPLS | 710400001 | 710400001–710400200 | `TPLS_NEW12_v1_T1000.npy` |
| 5 | PLS | 710500001 | 710500001–710500200 | `PLS_NEW12_v1_T1000.npy` |
| 6 | UB | 710600001 | 710600001–710600200 | `UB_NEW12_v1_T1000.npy` |
| 7 | TLB | 710700001 | 710700001–710700200 | `TLB_NEW12_v1_T1000.npy` |
| 8 | LNB | 710800001 | 710800001–710800200 | `LNB_NEW12_v1_T1000.npy` |
| 9 | TLNB | 710900001 | 710900001–710900200 | `TLNB_NEW12_v1_T1000.npy` |
| 10 | TPLB | 711000001 | 711000001–711000200 | `TPLB_NEW12_v1_T1000.npy` |
| 11 | PLB | 711100001 | 711100001–711100200 | `PLB_NEW12_v1_T1000.npy` |

Save under `/data/sijia/aamas2027_stage2/streams/NEW12-v1/`. Each array is shape(200,1000), dtype little-endian int32, C-contiguous, generated with episode j using base_seed+j (j=0…199). Save the original returned burst mask as `<regime>_NEW12_v1_T1000_burst_mask.npy`, shape(200,1000), int8. Masks are generator audit material and never policy input. The 12 value arrays are the sole shared test set for all 42 fresh jobs, both capacities and every seed. No per-method pool regeneration.

**Runtime assumption:** use the exact NumPy build/version from G1, the approved Stage 1 interpreter `/data/sijia/.venvs/kwallet/bin/python`, and a fresh generator process with its calibration cache initially empty. That version is an unresolved *recorded runtime fact*, not a free choice for Trae: copy it from the approved Stage 1 runtime lock. It is not available in the supplied brief, so no version number is invented here. No GPU RNG or global seed replacement. Require the bit-generator class to be PCG64; stop if this assumption fails. Repeating the identical generation in a clean process must give identical array-payload hashes before freezing; retain only one canonical copy, not two test sets.

**Seed disjointness audit from historical summaries, not CLI defaults:** master seed20000532 with per-regime offsets100000*i and counts417 for i0…7,416 for i8…11; validation seed220000532 with offsets100000*i and25 episodes each; OLD static eval base100000532+1000000*i with200 episodes each. Historical shuffle RNGs use master/validation base+999999. NEW12's710000001…711100200 namespace is disjoint from all these ranges. Preserve calibration seed1234567: it is an original calibration seed, not an evaluation-episode seed. Also prohibit reuse of any other recorded historical episode seed found in the handed-off manifests. If recovered metadata contradicts these formulas, stop before generation; do not change the namespace silently.

**Content overlap check:** hash each episode's 1000 values as canonical little-endian int32 C-order bytes. Compare every NEW12 episode hash against all 5000 master +300 validation +2400 OLD test episodes (not only consumed training/validation prefixes), and against every other NEW12 episode. Check both seed disjointness and content disjointness; one does not substitute for the other. Any collision, source/parameter mismatch, malformed array, nonfinite value, or incorrect shape/dtype stops the entire release. Do not discard/resample offending episodes, switch seeds, or inspect policy outcomes. PI must approve a documented plan revision before retrying.

**Manifest schema:** `new12_manifest.json` contains schema_version, dataset_id=NEW12-v1, creation_utc, regime_order, generator_path, generator_sha256, git_sha, dirty_source_status, runtime_lock_sha256, Python/NumPy version, bit_generator, calibration_seed/sample_size, exact serialized REGIME_SPECS, plan_sha256, job_matrix_sha256, OLD_manifest_sha256, no_results_viewed_attestation, and overlap_report_sha256. Each pool entry has regime, filename, episode_count, T, dtype/order, episode_seed_start/end, all200 episode seeds/hashes, file bytes, full-file SHA-256, array-payload SHA-256, and burst-mask filename/hash/shape. `overlap_report.json` records checked OLD files/hashes, reconstructed seed sets, counts and collisions. Write the final manifest atomically after checks, record its SHA-256 in `freeze_receipt.json`, make streams read-only, and verify hashes before/after each evaluation. Do not insert outcome fields into the stream manifest.

## 3. BF-T0.5 — frozen mathematical rule

Let b_i be observed current wallet balance, B=C/k, x the current transaction, and U={i: original E0._usable(i)}. Wallet indices are0…k−1; index k is the original no-op for each action component.

Settlement s = argmin_(i in U,b_i≥x) (b_i,i), or k when the set is empty. Flush f = argmin_(j in U,j≠s,b_j<0.5B) (b_j,j), or k when empty. Submit joint action s*(k+1)+f to the unmodified original E0.step. Use exact existing floating comparisons; no epsilon, rounding, tie randomization or ≤ threshold.

- x>B: settlement is no-op; still compute an eligible flush. No oversized-transaction early return.
- No feasible settlement for any other reason: same rule, flush may occur.
- b_j=0.5B: not eligible. Trigger is strictly less than half capacity.
- Exclude s before choosing f, so there is no simultaneous settle/flush conflict. If s=k, it excludes no actual wallet.
- Policy computes s then f from the same pre-transition state. E0 executes flush before settlement; do not manually settle first or alter balances.
- A flush makes the wallet unavailable using original freeze_until=time+F−1; after the original delay E0 restores **full B**, never0.5B.
- Information: current transaction, balances, availability/timers and known constants only. No future stream, generator label, NEW12 seed, lookahead, learned weights or hindsight. Full state visible to the learned policies already contains this information.
- Threshold0.5, minimum-balance order, tie rule and handling of oversized requests are fixed. **Zero tunable parameters after results; no threshold sweep before or after the test.** The threshold is informed by a prior local diagnostic, not retrospectively claimed to be historically preregistered.

This is a new independently implemented AAMAS comparator from this definition. Do not copy/import PR #3 code or numerical results. Prior local diagnostic outputs are not NEW12 evidence. Run BF using the same original E0 observation/control semantics (baseline mode, shaping disabled); only its action-selection logic differs.

## 4. Six-run zero-control audit and four exact future commands

Audit source: handed-off run_info files, verified against AAMAS_ARTIFACT_MANIFEST.csv. Original Downloads paths in that manifest are no longer present at audit time; the already handed-off local payload remains available and matches the recorded hashes. No source metadata was reconstructed. All six complete configuration dictionaries differ **only at seed and env.C** (including identical original output root). The common config hash below removes seed, env.C and the machine-specific output block, serializes JSON with sorted keys and compact separators, and hashes UTF-8:

`837a3c32c95a5cb3373a8cf6210e8909e5fe07f9be81a628683e608a99635374`

| C | Seed | Historical timestamp | run_info SHA-256 |
|---:|---:|---|---|
| 800 | 123 | 20260605_021448_926613 | `b40a92eb7e9105f43a7db61e541789c489ab49d4be7f5eb3f3a924d31afbf3f0` |
| 800 | 323 | 20260605_023723_903461 | `6cb8140e07f38c6a85c8ffaf284d638dafa581d1a10d4c79ff119556e4943a6e` |
| 800 | 532 | 20260605_025933_252988 | `bf1a47cd6068d4eb93aa5551f54ce3d426de40a2d396c3fd28eab54cb301dac6` |
| 1200 | 123 | 20260605_053713_707635 | `f3c08935197718fd59a82a7dbb0292c8f8359b945d4de0ab191e49dc7b86ffde` |
| 1200 | 323 | 20260605_055946_134106 | `d474c68042d898b2ee3f98f8d2581a79c0d3d59ce84fdb0f2875f335a0c894a5` |
| 1200 | 532 | 20260605_062222_028865 | `dd6d105e41534bcf8f5723e318516e2f279157bf97f3c6c243f562d43811b00e` |

| Field | Verified value for all six |
|---|---|
| C / k / F / T | C800 or1200; k24; F3; T1000 |
| Training episodes / actual input | 1000; first1000 in the original stored master order, once. train_use_episodes=3000 caps an available subset; not3000 training episodes. |
| Validation | Every50 episodes, first200 of the300 stored validation episodes, deterministic evaluation. |
| Pool filenames | MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy and same prefix_val_T1000.npy; exact handed-off bytes. |
| PPO | learning_rate3e−4; gamma.98; GAE.95; clip.2; update_epochs4; minibatch256; value_coef.5; entropy .03→.003; max_grad_norm1.0. Preserve source scheduling and optimizer. |
| Reward / objective | original; shapingfalse; alpha_drop.02; beta_flush.01; reward_t=env_reward_t. No new Money reward. |
| Money | p1, tau10; settled_scale50/tau_scaled.2/hybrid_alpha.1 are preserved config fields, not activated reward changes. |
| Widths | shared hidden128; branch_hidden128; settle embedding32; conditional hidden256; risk_feature_dim10. Preserve all inactive/legacy fields too. |
| Condition | top-level and conditional.condition_mode both zero_settle_embedding |
| Checkpoint selection | val_metric=value_accept_ratio; strict score>best, so earliest best is retained on ties; use_best_model_for_final_eval=true |
| Final evaluator | original deterministic argmax,200 episodes/regime, max_steps1000, OLD12 automatically evaluated by original main |
| Device / saving | CPU; save_modefull; debugfalse. Framework seeding is the original set_seed(seed). |

Preserve the complete reference config, not only the summarized table. Model/time/code/runtime differences must be logged. Runtime equivalence to the historical Mac environment is NOT established by equal config: the new four will use the Stage1-approved current CPU runtime. Record cohort=historical_recovered for old controls and cohort=stage2_new_current_runtime for new ones, keep all five paired effects visible, and state this reproducibility limitation. Do not claim new weights reproduce an unrun historical seed or remove the cohort distinction.

The original CLI has no pool-directory flag. The following **shell variable definition and commands are text only**. The in-memory bootstrap selects handed-off OLD pools and a new output root, validates all non-C/non-seed scientific fields against recovered config, then calls the unchanged original main. It does not edit historical source or JSON. Importing the historical module can create its configured cache directories; set XDG_CACHE_HOME=/data/sijia/aamas2027_stage2/runtime_cache/xdg and MPLCONFIGDIR=/data/sijia/aamas2027_stage2/runtime_cache/matplotlib before approved execution; record these fixed values in G1. Do not run these lines now.

```sh
export XDG_CACHE_HOME=/data/sijia/aamas2027_stage2/runtime_cache/xdg
export MPLCONFIGDIR=/data/sijia/aamas2027_stage2/runtime_cache/matplotlib
AAMAS_ZERO_BOOTSTRAP=$(cat <<'PYBOOT'
import copy, hashlib, importlib.util, json, pathlib, sys
repo = pathlib.Path("/data/sijia/kwallet-rl")
artifacts = pathlib.Path("/data/sijia/aamas2027_artifacts")
manifest = repo / "research/aamas2027/artifact_handoff/AAMAS_ARTIFACT_MANIFEST.csv"
import csv
records = list(csv.DictReader(manifest.open()))
ref = next(r for r in records if r["artifact_type"] == "run_info" and r["method"] == "SC-FAC-zero" and r["C"] == "800" and r["training_seed"] == "123")
p = artifacts / ref["transfer_relative_path"]
assert hashlib.sha256(p.read_bytes()).hexdigest() == ref["sha256"]
historical = json.loads(p.read_text())["config"]
source = repo / "src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == "c3710b3a8bbf905f2d047b5bf4444830cdc32239e39b49becc149540f6e79b04"
spec = importlib.util.spec_from_file_location("aamas_original_scfac", source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.DATA_POOL_DIR = artifacts / "pools"
sys.modules["kwallet_ctx_attn_fair_benchmark"].DATA_POOL_DIR = m.DATA_POOL_DIR
actual = m.apply_args_to_config(m.parse_args())
def normalized(c):
    c = copy.deepcopy(c)
    c.pop("seed")
    c["env"].pop("C")
    c.pop("output")
    return c
assert normalized(actual) == normalized(historical), "STOP: training protocol differs"
assert (int(actual["env"]["C"]), actual["seed"]) in [(800,777),(800,999),(1200,777),(1200,999)]
assert not pathlib.Path(actual["output"]["output_dir"]).exists(), "STOP: output already exists"
m.main()
PYBOOT
)
```

Run these four commands exactly once each only after G0/G1 and explicit training release (not authorized by this engineering task):

```sh
/data/sijia/.venvs/kwallet/bin/python -c "$AAMAS_ZERO_BOOTSTRAP" --model_mode conditional_factorized_ac --train_regime MIX12_EQ --C 800 --k 24 --F 3 --T 1000 --seed 777 --episodes 1000 --eval_episodes 200 --device cpu --reward_mode original --money_p 1 --money_tau 10 --val_metric value_accept_ratio --conditional_embed_dim 32 --conditional_hidden_size 256 --condition_mode zero_settle_embedding --save_mode full --output_dir /data/sijia/aamas2027_stage2/training/B-TRAIN-ZERO-C800-S777

/data/sijia/.venvs/kwallet/bin/python -c "$AAMAS_ZERO_BOOTSTRAP" --model_mode conditional_factorized_ac --train_regime MIX12_EQ --C 800 --k 24 --F 3 --T 1000 --seed 999 --episodes 1000 --eval_episodes 200 --device cpu --reward_mode original --money_p 1 --money_tau 10 --val_metric value_accept_ratio --conditional_embed_dim 32 --conditional_hidden_size 256 --condition_mode zero_settle_embedding --save_mode full --output_dir /data/sijia/aamas2027_stage2/training/B-TRAIN-ZERO-C800-S999

/data/sijia/.venvs/kwallet/bin/python -c "$AAMAS_ZERO_BOOTSTRAP" --model_mode conditional_factorized_ac --train_regime MIX12_EQ --C 1200 --k 24 --F 3 --T 1000 --seed 777 --episodes 1000 --eval_episodes 200 --device cpu --reward_mode original --money_p 1 --money_tau 10 --val_metric value_accept_ratio --conditional_embed_dim 32 --conditional_hidden_size 256 --condition_mode zero_settle_embedding --save_mode full --output_dir /data/sijia/aamas2027_stage2/training/B-TRAIN-ZERO-C1200-S777

/data/sijia/.venvs/kwallet/bin/python -c "$AAMAS_ZERO_BOOTSTRAP" --model_mode conditional_factorized_ac --train_regime MIX12_EQ --C 1200 --k 24 --F 3 --T 1000 --seed 999 --episodes 1000 --eval_episodes 200 --device cpu --reward_mode original --money_p 1 --money_tau 10 --val_metric value_accept_ratio --conditional_embed_dim 32 --conditional_hidden_size 256 --condition_mode zero_settle_embedding --save_mode full --output_dir /data/sijia/aamas2027_stage2/training/B-TRAIN-ZERO-C1200-S999
```

Commands are syntax/flag checked statically, not executed. Defaults not exposed by CLI (e.g. val_every, optimizer settings) are protected by complete config equality. Hash/source mismatch is a stop, not permission to edit the assertion. Administrative differences are restricted to output/data paths and recorded runtime; intended scientific differences are C and training seed.

After each trainer finishes, the additive adapter checks a hash-bound, externally supplied completed training receipt and its referenced run_info matching the job, all1000 training entries/20 scheduled validations, and exactly one selected best checkpoint path. Verify strict maximum validation-score selection with the historical tie rule, load-state architecture/mode metadata, and compute checkpoint SHA-256 before any NEW12 evaluation. Write `receipt.json` containing job_id, config/common-config hashes, run_info/result/validation history paths+hashes, best checkpoint path+SHA, selected validation episode/score, runtime/git/source hashes, and exit status. Use the recorded path in the CSV's receipt reference; no newest-file glob or OLD12-score selection. The trainer's automatic OLD12 evaluation and newly generated last checkpoint may be retained under the new job root but neither chooses a model nor enters fresh results.

## 5. Isolated additive evaluation adapter — implementation and OLD acceptance

The historical evaluators already accept a provided array and return per-episode raw_results, but they lack the common Stage2 job/manifest/receipt wrapper and BF rule. The implementation is `tools/aamas_stage2/adapter.py`; contract tests and exact OLD-only server commands are in `tools/aamas_stage2/test_adapter.py` and `tools/aamas_stage2/README.md`. No historical source is edited. G3 remains pending locked-runtime OLD acceptance, not Stage 1 approval. No scientific choice is delegated to the adapter.

| Method | Original module under src/idea4/ac/code | Agent |
|---|---|---|
| JA-PPO | kwallet_basic_ppo_fair_benchmark.py | BasicPPOAgent |
| IFAC | run_factorized_ac_benchmark.py | FactorizedPPOAgent |
| SC-FAC / zero SC | run_conditional_factorized_ac_benchmark.py | ConditionalFactorizedPPOAgent |

Required pipeline: read one CSV job → enforce approval/runtime/freeze receipts → verify source, checkpoint, run_info and array hashes → construct original make_env(config,max_steps=1000) and corresponding agent with JA arguments(config,env.state_size,env.k), IF arguments(config,env.state_size,env.base_state_size,config["attention_context"]["window_size"],env.k), and SC/zero arguments(config,env.state_size,env.base_state_size,env.k) → strict model.load_state_dict of the selected state dictionary on CPU → preserve full/zero condition mode from run_info (zero has both config locations checked) → call original evaluate_agent_on_array separately on the exact twelve NEW12 arrays with num_eval_episodes200 and max_steps1000 → normalize returned raw_results into the common schema without changing values → independently verify Money and totals → atomically publish outputs and their hashes.

Use original model/inference mode and tensor dtype accepted in Stage1. No autocast, quantization, torch.compile, batching across episodes/steps, argmax tie alteration, masking/repair, softmax-temperature adjustment, sampled actions or gradients/optimizer updates. JA uses joint argmax; IF uses its original independent deterministic heads; SC uses settle argmax then conditional flush argmax. These must not be replaced by joint MAP enumeration. Hashes and dimensions alone do not guarantee semantic parity; test adapter versus the original functions on OLD-only approved fixtures before G3.

For BF, construct original E0 with the same C/k/F/T/reward settings and baseline mode. Its thin loop must duplicate the historical evaluator's episode-reset/current-tx accounting, execute the specified rule, call original env.step, and obtain original env.get_metrics. Compute accepted_count/total_requested_value as the original evaluators do, not from inferred balance differences. Unit checks on synthetic *states*, not new evaluation streams, cover oversized request, no feasible wallet, equal balances, exact .5 threshold, and flush/settle exclusion. No collaborator imports. File-import provenance must identify the original idea3/idea4 paths; src/kwallet from PR #3 is forbidden.

**Outputs:** each evaluation job emits2400 episode rows and one result.json, plus a receipt with output hashes. CSV episode schema: schema_version, dataset_id, job_id, family, method, C,k,F,T,training_seed (null/NA for BF), condition_mode, cohort, regime, episode_index, episode_seed, episode_sha256, pool_sha256, checkpoint_sha256 (null for BF), settled, flushes, money, drops, oversize_drops, insufficient_drops, accepted_count, total_tx_count, total_requested_value, value_accept_ratio, count_accept_ratio, utilization. Preserve optional model-specific gate diagnostics separately; do not force them into BF or select on them. Invalid-action diagnostics are optional supplementary fields only if the original path already exposes them; no semantic instrumentation of historical sources is required.

result.json contains the same provenance, git SHA/dirty patch hash/source hashes/adapter hash/runtime_lock hash/freeze-manifest hash, exact checkpoint source and hash, rule_spec_hash for BF, NumPy/PyTorch/device/thread metadata, per-regime200-episode summaries, macro12_money, row counts, start/end UTC/runtime, validation flags, and file hashes. Include source/checkpoint identity even when two policies produce identical decisions. Cross-check drops+accepted_count=1000, oversize+insufficient=drops, flushes within[0,1000], counts nonnegative/integral, requested value positive, ratios within[0,1], and money=settled−10×flushes. Reject NaN/Inf/missing metrics; never drop bad episodes or impute cells. Metadata strings and counters must match exactly; arithmetic identities use abs tolerance1e−9/rtol0, never a statistical tolerance.

A job is complete only when every required file exists, all2400 unique(regime,episode_index) rows are present, hashes still match, and a success receipt is written. Partial outputs stay marked incomplete; do not overwrite a completed job. A technical retry requires a logged failure reason and the same scientific inputs, never a new seed or replacement checkpoint. A config/semantic discrepancy needs PI review, not a retry with altered behavior.

## 6. Primary statistical analysis — frozen before outcomes

Primary Money = settled−10×executed flushes. For method m, capacity c, seed s, regime r, let y be mean Money over200 episodes. Score S_mcs=(1/12) sum_r y. There is exactly one score per training seed per method/C; all12 regimes have equal weight. Do not pool regimes as independent training replicates or substitute reward/value_accept_ratio for Money.

For each learned method/C, report five S values, mean, sample SD(ddof1), SE=SD/sqrt5, and two-sided95% t CI mean±t_(.975,4)SE (critical value2.7764451051977987). For learned-versus-learned contrasts use paired training-seed labels on identical NEW12 arrays: d_s=S_left−S_right. Report five differences, mean delta, SD_delta, SE_delta,95% paired t CI, and primary effect size **absolute delta in Money units**. Also report paired standardized effect d_z=mean_delta/SD_delta; if SD_delta=0 mark d_z undefined, not infinity disguised as evidence. Seed pairing does not imply equal RNG trajectories across architectures. Seed intervals are conditional on this fixed test pool, not uncertainty over all possible worlds or real transaction data.

Contrasts, all two-sided, alpha=.05:

- Family A, **six tests**: SC−JA, IF−JA, SC−IF at each C800/1200.
- Family B, **two tests**: full SC−zero SC at each C800/1200.
- Family C, **two tests**: SC−BF-T0.5 at each C800/1200.

For families A/B use the paired t-test. For family C use a one-sample t-test of the five deltas against zero. Both compute t statistic mean_delta/SE_delta, df4, raw p=2*t.sf(abs(t),df4), and Holm-adjusted p within its predeclared family (sorted raw p, cumulative max of(m−rank+1)*p, clipped to1, restore order; ties deterministic by contrast_id). Publish all10 contrasts and all three correction families together. Unadjusted95% CIs are labeled marginal, not familywise intervals. Do not select a favorable family. If SD_delta=0, record the degenerate interval and mark the t-test/d_z undefined (do not claim significance from a division by zero); deterministic exact-zero differences are reported as such. No additional significance test is substituted after results.

The rule has one deterministic macro score per C, with no training-seed SD/CI. For family C subtract that same fixed score from each of the five SC seed scores: d_s = S_SC,s − S_BF. Test these five deltas against zero with a one-sample t-test; this is not a comparison of two stochastic paired samples. Retain exactly the same t interval, df4, and two-test Holm family. The interval reflects learned-training variation conditional on the shared pool, not five independently trained rules. Use no fake zero-width rule confidence bar. No bootstrap/sign test is added to the primary analysis. Episode/regime breakdowns and accepted-value/flush decomposition are descriptive secondary analyses, all settings retained.

Null/negative effects are reported with identical tables/plots and wording. A crossing-zero CI is inconclusive, not equivalence. If conditioning is mixed, narrow that claim; if the rule wins, state it in the main paper. No new seeds, capacity, model, threshold, metric, one-sided test, favorable checkpoint or post-hoc test family is authorized.

Analysis outputs: seed_scores.csv (method,C,seed,cohort,macro12_money,12 regime means,provenance hashes); method_summary.csv (method,C,n_training_seeds,mean_money,sd_money,se_money,ci_low,ci_high; BF has n_training_seeds=0 and null uncertainty); paired_effects.csv (contrast_id,family,C,seed,left_method,right_method,left_score,right_score,delta); contrast_summary.csv (contrast_id,family,C,n=5,mean_delta,sd_delta,se_delta,ci_low,ci_high,d_z,t,df,raw_p,holm_p,test_status). All inputs are the fixed42 fresh evaluations, never old automatic trainer tests.

## 7. Figure/table data contracts — no fabricated values

- **Figure2:** method_summary columns method,C,n_training_seeds,mean_money,ci_low,ci_high, plus seed_scores method,C,seed,macro12_money for points. Include JA/IF/full SC/zero SC and BF; separate C panels, same colors/units. Learned95% seed intervals; BF fixed-score marker without seed error bar. Label NEW12 and sample sizes. No historical values fill missing fresh cells.
- **Figure3:** paired_effects filtered familyB: C,seed,delta; contrast_summary: mean_delta,ci_low,ci_high,n,holm_p. Show all five differences, mean CI, zero reference, and distinguish historical/new zero-control cohorts without selecting one cohort's results. Optional decomposition comes from paired regime/episode settled and flushes, not another experiment.
- **Table2:** method_summary method,C,mean_money,sd_money,n_training_seeds plus clearly signed SC−method mean_delta/CI and Holm p from contrast_summary for JA/IF/zero/BF. SC self-contrast is NA, not a test. Rule uncertainty cells are NA. Keep IF−JA in the principal contrast report even if not a dedicated table column. Dataset/checkpoint/runtime hashes accompany machine-readable sources.

## 8. Stop conditions and evidence integrity

Stop immediately and mark affected job/campaign blocked on: revoked Stage1 approval (G0 is currently satisfied); missing runtime lock; NEW12 seed/content collision; wrong generator/spec/NumPy lock/dtype/shape or repeat-generation mismatch; changed stream manifest or any policy outcome produced/viewed before freeze; artifact/source/checkpoint/config/hash mismatch; zero training differs beyond C/seed/administrative paths; validation cadence/selection changed; missing/nondeterministic checkpoint behavior; NaN/Inf/invalid arithmetic or incomplete episode coverage; evaluator/adapter semantic mismatch; PR #3 or unapproved import; accidental use of OLD12 results as fresh data; unexpected completed output directory; attempt to tune or extend the grid.

Do not silently modify code, environment, checkpoint, episode, stream seed, training seed, floating tolerance or configuration to recover a favorable result. Preserve logs/quarantine outputs; report the first mismatch and all deltas with provenance. Only an explicit PI-approved plan amendment can change scientific inputs. No architecture sweep or extra seed is automatically authorized.

## 9. Budget and execution-readiness boundary

Four new training runs =4m training steps +16m scheduled validation steps. Unchanged trainer adds9.6m OLD12 final-evaluation steps. Fresh evaluations =40 learned ×2.4m +2 rule ×2.4m =100.8m steps. Total core simulator workload: **130.4m steps** (4+16+9.6+100.8), plus bounded OLD-only adapter checks; generator work is separate. No additional principal training.

Prior completed zero runs took approximately22–23 minutes each on their original runtime; do not claim that predicts the server GPU. Reserve2–4 CPU-hours for the four train/validation/automatic-old-eval jobs and5–10 CPU-hours for fresh evaluations, plus recovery/adapter/generation overhead. Measure the first approved job without changing the science. Above-budget runtime triggers reporting/rescheduling, not silent downsizing or extra tuning.

**Remaining scientific choices:** none in the job matrix, seed namespace, rule, training protocol, metric, contrasts, or stop rules. **Unresolved release/engineering facts:** the captured and confirmed approved-runtime lock, and OLD-only acceptance of the implemented adapter. G0 is satisfied. The accepted SC-FAC historical drift is not a remaining blocker. The current brief does not provide the final approved NumPy/PyTorch versions, and this plan does not invent them. The six-config equality is verified; hardware/runtime equality to the old campaign is not.

After those gates close, Trae can execute the fixed campaign mechanically with no new scientific decisions. **Implementation is not G3 acceptance:** the isolated adapter and tests are implemented, but actual OLD-only learned-policy parity must pass on the approved runtime. Local contract tests do not establish runtime parity. Follow the README acceptance checklist and stop on any mismatch; do not alter historical code or environment. No Stage2 result, stream hash, generated checkpoint timestamp or runtime pass is asserted to exist now.

## Source identity recorded during local audit

| File relative to original repository | SHA-256 |
|---|---|
| `src/ideaextra/kwallet_ideaextra_generator.py` | `d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8` |
| `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py` | `d46246795552f22f0ba143ae38230692ff99997d57c9b0681f456432a9df1921` |
| `src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py` | `e4a669678453a2884ac6834ee599dd85a98e6bac59ac59630513f7c2d022d0ea` |
| `src/idea4/ac/code/run_factorized_ac_benchmark.py` | `365971c27702d7c05f0a77dbc9f94ed80b3fd2fe470c7ce925409effef2a2971` |
| `src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py` | `c3710b3a8bbf905f2d047b5bf4444830cdc32239e39b49becc149540f6e79b04` |
