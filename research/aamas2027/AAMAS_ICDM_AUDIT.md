# AAMAS 2027 — ICDM lineage audit

Initial audit: 2026-09-30; local recovery update: 2026-10-01. Scope: minimum-change revision of the user's rejected ICDM work. This is a planning document, not an experiment report from new runs. No experiment, server access, paper/code edit, commit, or push was performed for this audit. PR #3 is not an evidence source.

Evidence labels: **FACT** means directly inspected source, saved artifact, or an arithmetic recomputation of saved results. **INFERENCE** means scientific interpretation. **PROPOSAL** means work not yet executed. Previously saved replay reports are distinguished from checks performed in this audit.

## 2026-10-01 Phase 0 reconciliation — current decision

**Experiments remain stopped. PATH A — ARTIFACTS RECOVERED is selected, subject to a later model-load/replay gate.** The server absence reported by Trae is compatible with complete local recovery; it is not evidence of permanent loss. This task performed local enumeration, archive reads, checksums, JSON/config inspection, and arithmetic comparisons only. No model was loaded, no simulator was stepped, no new stream was generated, and no GPU/server was accessed. No artifacts were moved or extracted.

**Current AAMAS capacities are C=800/1200.** Both are directly confirmed by recovered original checkpoint/result configurations for every principal method and all ten historical seeds. The local recovered source and results also contain C=900/1000, but there is still no exact rejected PDF/submission receipt proving those were in the rejected submission. They remain separate recovered evidence, not authority to retain C=900 in the revision. The previous C900/1200 plan is superseded; do not globally relabel C900 numerical results as C800.

**Recovery result:** 60/60 principal settings have both best and last checkpoints and a result JSON. Six selected zero-conditioning settings (three seeds at each target C) likewise have best/last checkpoints and raw result JSONs; the raw means match the selected ablation table. All 14 required pools exist. File presence was confirmed independently of aggregate CSVs: checkpoint bytes were read, hashed, and checked against both loose copies and the archive; their internal PyTorch ZIP members passed CRC checks without deserialization. Actual runtime/model compatibility remains untested in this task.

**New training count:** zero principal runs; four zero-control runs to extend the two capacities to seeds 123/323/532/777/999. The exact missing additions are (800,777), (800,999), (1200,777), (1200,999). Across the full ten-seed zero-control grid, fourteen settings are missing; only four belong to the minimum five-seed plan. Six found zero controls are evidence, not an inference from mode availability. This replaces the prior assumption with an explicit inventory.

**Strong-rule provenance correction:** the Git/server historical baseline set is not shown to contain the proposed threshold-feedback rule. Separate local `audit_artifacts` scripts and CSVs implement a mathematically compatible diagnostic; they are outside the original main campaign and are not collaborator PR #3. Their old-pool numbers remain exploratory audit evidence, not submitted-paper baseline results. Recommend independently implementing the mathematical rule as a **new AAMAS baseline**, freezing a remaining-balance trigger of .5 before new scores. A triggered flush refills to the full C/k after the original delay; .5 never means a partial refill. Do not copy PR #3 code or results.

## Local recovery inventory and verification

Search date: 2026-10-01. Enumerated Desktop, Downloads, Documents, and all local `.codex/.chatgpt-projects` trees, including hidden/ignored project artifacts. The final scan excluded Git internals, app bundles, and node_modules; it enumerated 24,896 files without an access error. Relevant archive directories were inspected non-destructively. No unrelated system directory was searched. “MISSING” below means absent from this search, not globally nonexistent.

An ordinary ignore-aware file listing misses some loose historical artifacts. The final no-ignore pass found complete checkpoint copies in BOTH the project mirror and Downloads. The ZIP is a third byte-identical source, not the only source. No recovery transfer or runtime validation has happened on the server.

### Exact roots and path construction

The following roots are absolute. Every checkpoint/result row below is uniquely resolved by method, C, seed, and timestamp; use this manifest rather than the newest matching file.

- **R (primary loose result/checkpoint root):** `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/02_完整研究资料/04_ActorCritic与SCFAC研究/ac/results`
- **D (verified duplicate):** `/Users/zhouzhou/Downloads/KWallet_Organized/02_完整研究资料/04_ActorCritic与SCFAC研究/ac/results`
- **Z (archive):** `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized.zip`
- **H (separate local diagnostic):** `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/audit_artifacts`
- **W (organized daily tree):** `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/01_日常研究`

Method codes: JA-PPO=`basic_ppo`; IFAC=`factorized_ac`; SC-FAC=`conditional_factorized_ac`. Scenario name S:

- JA/IF: `{code}_trainMIX12_EQ_C{C}_k24_T1000_F3_seed{seed}`.
- SC: `conditional_factorized_ac_trainMIX12_EQ_C{C}_k24_T1000_F3_seed{seed}_condE32_condH256`.
- Zero-control: SC name plus `_conditionzero_settle_embedding`.

**Exact best checkpoint:** `R/{code}/checkpoints/{S}/{timestamp}/best_model.pth`; last checkpoint changes only the basename to `last_model.pth`. The same relative paths exist under D. Archive member equals `KWallet_Organized/02_完整研究资料/04_ActorCritic与SCFAC研究/ac/results/{code}/checkpoints/{S}/{timestamp}/best_model.pth`. Exact raw result is `R/{code}/runs/{S}/{timestamp}/cross_regime_results.json`; config/selection/fingerprints are in `run_info.json` in that directory. Principal results also exist at `W/results/raw/kwallet/{code}/runs/{S}/{timestamp}/cross_regime_results.json`.

SHA-256 below is for the complete best-checkpoint file, not its tensor payload. Last checkpoints and duplicate copies were also fully read and validated. This confirms recoverable bytes, not a successful forward pass. All principal rows include training-history records. Pool matching used the original array-payload MD5 convention, separately from the new complete-file SHA-256.

### Principal matrix: 60 requested settings

| Method | C | Seed | Classification | Result JSON | Last checkpoint | Selected timestamp | Best bytes | Best SHA-256 |
|---|---:|---:|---|---|---|---|---:|---|
| JA-PPO | 800 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260508_222529_095916` | 430967 | `bb00559d921e41f589618148e2b8c208faa4ccfbb92d8825b2b41fd3a65ca6d1` |
| JA-PPO | 800 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_221753_427685` | 430967 | `c625537a7a0fca183aa7f9f3e69c54e754a9a4a34014cdc881892c9fd53bb210` |
| JA-PPO | 800 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_224719_801722` | 430967 | `ea2c6d1a4d7c6049069effb5d3d7a5dad09b282cffaa01fb5b3a6767963d2a49` |
| JA-PPO | 800 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_231658_207297` | 430967 | `c1e39cf015bdcd1ca8ec6fa64c703bfff1921a6cf258fa862bf94c97b0cdc662` |
| JA-PPO | 800 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_234622_339334` | 430967 | `bf9bfa40054ca03e6ded8c0a61b495016cdaf55c10c3e76f3521df23256438f1` |
| JA-PPO | 800 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_001532_795939` | 430967 | `3c9f2dc570e068d499820614912b7c1cc53a28db0f53ec36ef207e741a5fb4b2` |
| JA-PPO | 800 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_004432_112478` | 430967 | `cb6e67e6cede8633dca867728b10f6d8b297b7b7119597153e76b5cf049ef8b2` |
| JA-PPO | 800 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_011348_775417` | 430967 | `afd094e57de6624e071f1af5851e1e4922654e6679df2c48d07d8132aa3686ee` |
| JA-PPO | 800 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_014308_829711` | 430967 | `9ce8fd0dd47e660a4a9c839c8b66bfa6f3031ff65c4846f998f5abdeb3a6377d` |
| JA-PPO | 800 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_021232_313222` | 430967 | `ea01022fbb9be504baa8ec49625620d44567c7cea89ee1d8691db0140254ad78` |
| IFAC | 800 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260508_224206_743115` | 134709 | `811d2759fb7112fa086380f3b7c49560402ff34fd0829af788b37192b44fdba7` |
| IFAC | 800 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_223023_020401` | 134709 | `9aea46c05faf40e17669bf404a00302086e39c8c9eec6701922e5d19fbafb6df` |
| IFAC | 800 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_225934_742365` | 134709 | `442ef7cd7177d9f759d3b74833d0113fa3af9a1cb1a06f89267300f3ca6a868e` |
| IFAC | 800 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_232915_556325` | 134709 | `7cd327b95f064872aa7f3db883e9f2b3169cc013d2388133c0973df542d6ab69` |
| IFAC | 800 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260518_235842_316631` | 134709 | `1aa0695126eeebb8a058a42c73c7c0d665be6579023c4630438334cae69f9ccc` |
| IFAC | 800 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_002735_405818` | 134709 | `79d661517191854cc7bb1d2a03314e3a0f9b185a132bc222105ceed367271995` |
| IFAC | 800 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_005654_384438` | 134709 | `7aeab6df917bd975a704d5d76b031ba7ecd8d3c55111e917481dcdb9f6277c64` |
| IFAC | 800 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_012610_536937` | 134709 | `aced1e45d9c7afa4a1f1071065792bd8f2d9b1d8de258bbb6a06ad0e12eb70c5` |
| IFAC | 800 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_015530_129592` | 134709 | `e8f5f412194be622e5bbc4c58bfbe98e30310056875de39a6e721eb1228101fe` |
| IFAC | 800 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_022457_927620` | 134709 | `640dfd157585a77bdd8c00bcfaf5fa7f97abc707175d139132cf6b8923486cfd` |
| SC-FAC | 800 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_180310_098181` | 316469 | `ab0f7102a41b31c97621722ad2906d677c85a0d8e1a3804191680eae5257cfb2` |
| SC-FAC | 800 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_022720_005236` | 316469 | `2ae29e0ccb832c63289ae74b8998c25670d9a4fe4f797b5f6b22b4721698b2b1` |
| SC-FAC | 800 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_024910_030504` | 316469 | `f07b24081fa01442a8547437b27835ff101be6bc269e6d99312914b2d7095b92` |
| SC-FAC | 800 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_031048_321526` | 316469 | `377b4a6ad94ef2ec2f5ffd44c1a83845271d9e0a1ffa7fb29766825f6bc5a8d8` |
| SC-FAC | 800 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_033226_542926` | 316469 | `5fa2cc934a0ae928ccd3f8d95602b939e398b282f86c981ce7f1eb3535e2998a` |
| SC-FAC | 800 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_104603_336683` | 316469 | `b290eb48057f461262d7fd5fa929e04e177758086b6c51d6631f71348d10f7e7` |
| SC-FAC | 800 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_110749_701257` | 316469 | `281ac20f6e28bad000cc29b46bd9a5529fcbc111b77a708b13a33af5f27b37af` |
| SC-FAC | 800 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_112929_005045` | 316469 | `b1f57de6b5ef38d945e53b384284a1af1633ab8c68802bd150ae8c3eccbba70d` |
| SC-FAC | 800 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_115137_045318` | 316469 | `d65f6988c3c5b1987865b8c3b199593329013b8f1cf5bafab9057b756c84bcde` |
| SC-FAC | 800 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_121337_784057` | 316469 | `80e967122a6669fd118d9c07e843e64889d0bf020fe17dd856a5cdd7ab4230a6` |
| JA-PPO | 1200 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_024159_789634` | 430967 | `ccce13c42a33b68f55d05a3bfe17c0751d736c1f9df81d810f9bf019eeff6ce2` |
| JA-PPO | 1200 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_031121_041920` | 430967 | `6eb850d8a02012e4b67baf3352b9ba0f055b253e27b816e0d3f13b9720a44f43` |
| JA-PPO | 1200 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_034047_345279` | 430967 | `c415f1b5e3437abf9bcf96987a527c9891480a977008422e56f50ae065204aca` |
| JA-PPO | 1200 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_041041_286454` | 430967 | `ba91e2e556afdcd4d9cb197db8d20bec4487bd6c0c6bf14072cfecf4b94559e1` |
| JA-PPO | 1200 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_044006_987162` | 430967 | `e6dbaa4acbb1f81e7febe88431be11613b3154d72dc92b88f54f1fb3e6b5f524` |
| JA-PPO | 1200 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_050932_925273` | 430967 | `db0d881905fa3a9817151714a85e82ee14acdf55c2a33d64adaea96f69b4f7ee` |
| JA-PPO | 1200 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_053836_449829` | 430967 | `92a0d7d9ee50bfa77600787a1b67b324aeaaf8c6b81c24df515af8269b4127e7` |
| JA-PPO | 1200 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_060734_679428` | 430967 | `afcc5ec2667f1ed64e9842974e547bcbe95bfa01ba09be536861221436ed3109` |
| JA-PPO | 1200 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_063706_631471` | 430967 | `cd34633e2b4d60e4ee4aee2c0924b23e598dd465b79cd528f00aecda0fc86280` |
| JA-PPO | 1200 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_070609_906283` | 430967 | `30eb2210d6af68fc85a3da943223115dad45550052d3e966dfdfc7b9bf95174a` |
| IFAC | 1200 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_025420_675991` | 134709 | `8d2536f3f4d956062800fb4ca6ce96fd9d1ff1f395403b0ed4d4ad5ba47211c6` |
| IFAC | 1200 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_032345_740594` | 134709 | `da8581dab8c4e34f2890e53acf55a7a40ed1b089836c1df693fb72a2a4ea46a9` |
| IFAC | 1200 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_035315_785947` | 134709 | `05337b6e91fa78df218c5816bc6aecc3ef0d0f9f9868e924892821e9f9f35488` |
| IFAC | 1200 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_042305_814867` | 134709 | `0d9e9ad311b4788d019338ed55bc580c189ce49c2f047dc67838424bbead62a8` |
| IFAC | 1200 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_045231_365773` | 134709 | `5903e275ec7f85e214cef8ccb5747b152e5f40aa4aecba7d0f9d1ffdb292835b` |
| IFAC | 1200 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_052153_364776` | 134709 | `99c8acd3f1614e5902ccea84e3480deb01e842d8e264d91ff1e21a83d36f8a7d` |
| IFAC | 1200 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_055046_925938` | 134709 | `e005c6a3bf5841ae13753ef908af112d30ff2259940f5b048cd8556b23a5dd48` |
| IFAC | 1200 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_062001_556996` | 134709 | `8c246fc6ed3d8fd43af3e2148e5899d892fdac8025963c71e24e3116ffc9ad41` |
| IFAC | 1200 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_064914_982285` | 134709 | `6f3942185a2cf3ebaebfde32ea5f99819d24a83994f057fa18ceac43a462e9d1` |
| IFAC | 1200 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_071828_860484` | 134709 | `b387dc2cb203da5441a26cf2525960bea4c1dc5f8c9b1bce98dee0e115f4d994` |
| SC-FAC | 1200 | 123 | CHECKPOINT FOUND | FOUND | FOUND | `20260519_205531_968350` | 316469 | `1cddad0e7db59a464e106e626eabc138c1b314b845d340d3f27cfafcdf1147de` |
| SC-FAC | 1200 | 323 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_081059_886585` | 316469 | `043d81a604db9f17d0803798a32f23313f98719a185f08303d6a25b44931c004` |
| SC-FAC | 1200 | 532 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_083306_953556` | 316469 | `80f83cbdd1cb7070d84ce3d100d37c4d5386f21f47546118cf1352a75d5a6736` |
| SC-FAC | 1200 | 777 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_085518_721730` | 316469 | `54cc2f0d15938aa26a7b7f92fb133d8f2b0065bb4c366a896b30b63f9a425e5a` |
| SC-FAC | 1200 | 999 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_091731_850008` | 316469 | `4142bb0b92d5e8962c164cf44ca8278e4f3e447208347eabeeb0b49ba21aacdb` |
| SC-FAC | 1200 | 2027 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_123530_739478` | 316469 | `904aa1f991133a3610ef3a487b8bf65a70a32b36121d9b196df80b86123e6958` |
| SC-FAC | 1200 | 3407 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_125722_810653` | 316469 | `94ce149b4545d68fcdddbe1b743d17bd32e745d0f451f187568804cf42b0bf77` |
| SC-FAC | 1200 | 4501 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_131910_993620` | 316469 | `f8505eb1f034e15ebedc244d4a4e702c270a9732f55e463057aa01508fa0d47f` |
| SC-FAC | 1200 | 6101 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_134111_268116` | 316469 | `fa1cc30995375237321f2603dfc2adc8457e09590d6fcb86e8d3fe12706cb800` |
| SC-FAC | 1200 | 8888 | CHECKPOINT FOUND | FOUND | FOUND | `20260520_140351_652840` | 316469 | `76b79745ed2b7e7ba1e59d777f4fecfd547df5f02b2d9f136df8f9edeeab46bb` |

Principal totals: 60 CHECKPOINT FOUND; 60 RESULT JSON FOUND; 0 only-aggregate; 0 missing. Each capacity has JA-PPO 10/10, IFAC 10/10, SC-FAC 10/10. Both best and last weights exist for all 60. This is 120 selected checkpoint files, not 120 independent settings.

### Zero-conditioning matrix: all 20 possible historical seed/settings

The six recovered selections are tied to the June 5 completed all-capacity logs, not selected by performance. C800/seed123 has two additional earlier best-checkpoint entries; the selected timestamp below matches the raw result that reproduces the reported ablation mean. Do not count these duplicates as extra seeds.

| C | Seed | Classification | Result JSON | Selected timestamp | Best SHA-256 | Planned new training? |
|---:|---:|---|---|---|---|---|
| 800 | 123 | CHECKPOINT FOUND | FOUND | `20260605_021448_926613` | `543880353dbd6bbde5a684f182b364ea9367ad6f8e964124fd85b8905ad4f2a7` | NO |
| 800 | 323 | CHECKPOINT FOUND | FOUND | `20260605_023723_903461` | `db65c331367459e550e4862697c0ffbf018f1bd83389c05f85c2842ab1396f0e` | NO |
| 800 | 532 | CHECKPOINT FOUND | FOUND | `20260605_025933_252988` | `c3481ab853b35ab5541724fddf68c7e4166009116d5230039cc388ca52e44cd6` | NO |
| 800 | 777 | MISSING | MISSING | `—` | — | YES |
| 800 | 999 | MISSING | MISSING | `—` | — | YES |
| 800 | 2027 | MISSING | MISSING | `—` | — | NO |
| 800 | 3407 | MISSING | MISSING | `—` | — | NO |
| 800 | 4501 | MISSING | MISSING | `—` | — | NO |
| 800 | 6101 | MISSING | MISSING | `—` | — | NO |
| 800 | 8888 | MISSING | MISSING | `—` | — | NO |
| 1200 | 123 | CHECKPOINT FOUND | FOUND | `20260605_053713_707635` | `0bb31d4704e085a27017b9b3e116197e9dd399a36002408be3601d8f90ccffa3` | NO |
| 1200 | 323 | CHECKPOINT FOUND | FOUND | `20260605_055946_134106` | `69db13f4e49fc0777841917f7cd99eec30e85e7acd49f3c241ba9739743cf8c0` | NO |
| 1200 | 532 | CHECKPOINT FOUND | FOUND | `20260605_062222_028865` | `9d9199646afc1a23a2a088151bf1c89e5d408b907fb102c9a8b3c2de45983902` | NO |
| 1200 | 777 | MISSING | MISSING | `—` | — | YES |
| 1200 | 999 | MISSING | MISSING | `—` | — | YES |
| 1200 | 2027 | MISSING | MISSING | `—` | — | NO |
| 1200 | 3407 | MISSING | MISSING | `—` | — | NO |
| 1200 | 4501 | MISSING | MISSING | `—` | — | NO |
| 1200 | 6101 | MISSING | MISSING | `—` | — | NO |
| 1200 | 8888 | MISSING | MISSING | `—` | — | NO |

The six selected best checkpoint files are each 316,469 bytes; all six last checkpoints also exist. All six run_info configurations explicitly set `condition_mode=zero_settle_embedding` and use C=800/1200, k24, F3, T1000, E32/H256, original reward, 1,000 training episodes, and 200 evaluation episodes/regime. JSON configuration inspection across the archive found twelve zero-mode run records overall: seeds 123/323/532 at each of C800/900/1000/1200. The other capacities are outside the revised two-capacity plan. No zero-control seed 777 or 999 artifact was found at either target C; the other five missing seeds per capacity are deliberately not requested.

### Historical transaction pools: 14/14 found

Exact primary path for every row is `W/data/transaction_pools/{filename}` using the absolute W above. All arrays are int32. The modification time is local file metadata in UTC, not proof of generation time. Each row has a matching `{stem}_summary.json` in the same directory. The validation file has **300 stored episodes**; the historical trainer uses **200** for checkpoint selection. Master contains 5,000; historical training consumes its first 1,000 episodes.

| Filename | Shape | Bytes | SHA-256 of .npy file | Modification time (UTC) |
|---|---|---:|---|---|
| `MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy` | 5000×1000 | 20000128 | `47f283231334cd65605947e3b046100e095a2410013e10792bd5521c85e9bf18` | 2026-09-06T03:25:14.431274+00:00 |
| `MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy` | 300×1000 | 1200128 | `9f86d389b1c745717049c28fa984e83005d363c402d15dfcd9051809c01c64bb` | 2026-09-06T03:25:14.436797+00:00 |
| `US_static_eval_T1000.npy` | 200×1000 | 800128 | `3f9fc877a2a251f29ec2f9a18cc2b363dc1584d4cc8593df0eb4fe9142655455` | 2026-09-06T03:25:14.866140+00:00 |
| `TLS_static_eval_T1000.npy` | 200×1000 | 800128 | `40e4a6d8404ab538d3e58d64614641bfb8ae98c1cbfba128792f00982a8346b4` | 2026-09-06T03:25:14.694974+00:00 |
| `LNS_static_eval_T1000.npy` | 200×1000 | 800128 | `4f6ca995a456200e8264486f7de1d8f1952a563e2a555a954f66b14429325937` | 2026-09-06T03:25:14.359414+00:00 |
| `TLNS_static_eval_T1000.npy` | 200×1000 | 800128 | `96cd1b6faba1e210e4f213d216f5bc7524e2ba4fb491b8e4fe04de5f087b652a` | 2026-09-06T03:25:14.620311+00:00 |
| `TPLS_static_eval_T1000.npy` | 200×1000 | 800128 | `76b7922eb7e8587c02bc583d06db201d801d6b141ff9b54f5fa26340a402ef05` | 2026-09-06T03:25:14.762625+00:00 |
| `PLS_static_eval_T1000.npy` | 200×1000 | 800128 | `4e3aa54546a2ef55aa882332b130bf4b77565fa46665d6b803b6094cf41ddd9f` | 2026-09-06T03:25:14.513887+00:00 |
| `UB_static_eval_T1000.npy` | 200×1000 | 800128 | `9d38be603ebed0700cd6c4af2c8acbd9fc79c04ac05be66cbcba20a16a7b2701` | 2026-09-06T03:25:14.795143+00:00 |
| `TLB_static_eval_T1000.npy` | 200×1000 | 800128 | `31644c51e88e708cf8614870300ac3a1e6eeaee59314c4ab22c5f5c8f01f89ba` | 2026-09-06T03:25:14.548200+00:00 |
| `LNB_static_eval_T1000.npy` | 200×1000 | 800128 | `7f9bd7ece2a0efbd4a5db135df52de8d2c5b251535e8929865756ae0fc1eeb36` | 2026-09-06T03:25:14.319677+00:00 |
| `TLNB_static_eval_T1000.npy` | 200×1000 | 800128 | `891dbe4add63aa0f81f1a1396661a4c7b7a27f270468bb79ac1c990c0ec00811` | 2026-09-06T03:25:14.582634+00:00 |
| `TPLB_static_eval_T1000.npy` | 200×1000 | 800128 | `69fa1007390719a17266dc014f199756c0ff55c36d788b122cc824010c375ae4` | 2026-09-06T03:25:14.729513+00:00 |
| `PLB_static_eval_T1000.npy` | 200×1000 | 800128 | `684fc74167a56128fe48be5e70ee0fc0faeb7a329220449ac380326ca3762418` | 2026-09-06T03:25:14.475011+00:00 |

The primary files have three identical loose duplicates each and two matching members each in Z. Verified alternate roots: Downloads' `KWallet_Organized/01_日常研究/data/transaction_pools`, and the project/Downloads `KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/` copies (master in `train`, validation in `val`, evaluation in `eval`; exact alternate paths are listed next). These are duplicate bytes, not independent datasets. The collaboration directory name here is an old local bundle and does not mean PR #3 evidence.

| Pool | Additional exact loose paths (identical SHA-256) |
|---|---|
| `MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/train/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/train/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy` |
| `MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/val/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/val/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy` |
| `US_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/US_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/US_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/US_static_eval_T1000.npy` |
| `TLS_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLS_static_eval_T1000.npy` |
| `LNS_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/LNS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/LNS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/LNS_static_eval_T1000.npy` |
| `TLNS_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLNS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TLNS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLNS_static_eval_T1000.npy` |
| `TPLS_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TPLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TPLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TPLS_static_eval_T1000.npy` |
| `PLS_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/PLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/PLS_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/PLS_static_eval_T1000.npy` |
| `UB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/UB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/UB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/UB_static_eval_T1000.npy` |
| `TLB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLB_static_eval_T1000.npy` |
| `LNB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/LNB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/LNB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/LNB_static_eval_T1000.npy` |
| `TLNB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLNB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TLNB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TLNB_static_eval_T1000.npy` |
| `TPLB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TPLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/TPLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/TPLB_static_eval_T1000.npy` |
| `PLB_static_eval_T1000.npy` | `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/PLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/01_日常研究/data/transaction_pools/PLB_static_eval_T1000.npy`<br>`/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/collaboration/general_ac_data/eval/PLB_static_eval_T1000.npy` |

Manifests/fingerprints: `W/data/transaction_pools/ideaextra_manifest.json`, `W/reproduce_figures/seeds/transaction_pool_seed_manifest.csv`, and `W/reproduce_figures/seeds/transaction_pool_fingerprints.csv`; corresponding submission-package copies also exist. The fingerprint CSV uses **MD5 of array contents**, not the .npy file bytes. This recovery recomputed that convention and obtained 14/14 manifest matches, 840/840 matches against the 60 principal run_info pool fingerprints, and 84/84 against the six zero-control records. New SHA-256 values above identify complete files for transfer. No pool was moved or regenerated.

### Archives searched

| Exact archive path | Entries | Checkpoint-named entries | Requested C800/1200 k24 checkpoint/result entries |
|---|---:|---:|---:|
| `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized.zip` | 9458 | 1242 | 331 |
| `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/experiment_gap_audit.zip` | 6 | 0 | 0 |
| `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/05_历史备份/原有归档/idea4_ac_results_20260505.zip` | 180 | 18 | 0 |
| `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/06_版本历史与环境/original_python_environment.zip` | 21381 | 1 | 0 |
| `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/06_版本历史与环境/original_git_history.zip` | 6343 | 0 | 0 |
| `/Users/zhouzhou/Downloads/项目论文/PRICAI_paper.zip` | 56 | 0 | 0 |
| `/Users/zhouzhou/Downloads/项目论文/CVAA(EI).zip` | 3 | 0 | 0 |
| `/Users/zhouzhou/Downloads/KWallet_Organized/05_历史备份/原有归档/idea4_ac_results_20260505.zip` | 180 | 18 | 0 |
| `/Users/zhouzhou/Downloads/KWallet_Organized/03_论文与分析记录/experiment_gap_audit.zip` | 6 | 0 | 0 |
| `/Users/zhouzhou/Downloads/KWallet_Organized/06_版本历史与环境/original_python_environment.zip` | 21381 | 1 | 0 |
| `/Users/zhouzhou/Downloads/KWallet_Organized/06_版本历史与环境/original_git_history.zip` | 6343 | 0 | 0 |

Archive counts include duplicates, last models, alternative heads/reward configurations, and unrelated historical families. They are **not** the selected experiment matrix. The May 5 archive is too early and covers other wallet counts. The Python-environment archive's checkpoint suffix is not an experiment weight. The Git archive was inspected as an archive, not unpacked into or substituted for the live repository. No relevant tar/tgz/7z project archive was found in the searched roots.

### Scope of “recovered” and next gate

The matching checkpoint bytes, 66 raw result records, matching run configs, histories, and pools make a historical replay feasible. They do not establish a replay result. The next action is a local-to-server **artifact handoff plan and hash verification**, followed by a separately approved replay. The minimal eventual payload is 30 principal best checkpoints (first five seeds at both C), six zero best checkpoints, and fourteen pools: **41,522,056 bytes (~39.60 MiB)** before small metadata and packaging overhead. No binary belongs in the four-file Git commit. Do not transfer, evaluate, or train as part of this recovery task.

For a future first replay, use JA-PPO C800/seed123, timestamp `20260508_222529_095916`, best checkpoint and the historical US pool; compare all saved US metrics from its exact raw JSON. Use original deterministic argmax, then stop for review before broad evaluation. The selected first-replay artifacts are directly recoverable; no principal retraining is justified by server absence alone.

### Strong baseline: independent new AAMAS implementation

The diagnostic code at `H/heuristic_diagnostic.py` chooses the tightest-fit usable settlement wallet, then the lowest-balance *other* usable wallet below .5×C/k for a flush. It calls the original environment. `H/verify_heuristic_original.py` and `H/heuristic_original_replay.csv` record a prior original-simulator check; this task read them but did not execute them. This is **separate local audit provenance**, not an original campaign baseline, a Trae run, or PR #3. Its historical numbers must remain labeled exploratory and must not be promoted into a new AAMAS result table.

**PROPOSAL:** independently implement from the mathematical specification the **best-fit, balance-threshold full-refill rule (BF-T0.5)** in the original environment. Choose settlement by minimum usable b_i≥x; choose flush by minimum b_j among other usable wallets satisfying b_j<0.5(C/k); use wallet index for ties and no-op if an eligible set is empty. Even if x is oversized, a flush may occur. Submit the pair through the original transition; do not manually alter balances or delay. The environment replenishes a flushed wallet to full C/k after F, never to half capacity.

This is fair because it uses only the same current observation, one settlement/one flush, and the same cost/timing/budget constraints. It requires **no tuning** in this plan. Freeze .5 before AAMAS evaluation, acknowledge that the choice is informed by the prior local diagnostic, and do not describe it as a historically preregistered or optimized threshold. Implement no collaborator code and import no collaborator numbers. The baseline is newly added to the AAMAS campaign; its independent implementation and outputs are still pending. Replace the prior ambiguous “half-refill” label with “balance-threshold full-refill.”

---

## Retained source audit, with evidence scope corrected

The following September 30 analysis is retained for traceability. References to all four recovered capacities describe the broader local archive, **not confirmed contents of the rejected submission**. The current inventory and C800/1200 plan above govern execution. Old local feedback means are exploratory diagnostic evidence only.

## 1. Manuscript identity and confidence

**FACT:** The most complete ICDM-oriented manuscript located is **Settle-Conditioned Policy Learning for Streaming Transaction Collateral Control**, in [main.tex](</Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/overleaf/kwallet_paper_submission_v2/main.tex>). Its main source SHA-256 is `9325ecd064d6cbba15cc9f87ddc82f4f40f40b60c32e49ca7847c6b12680c993`. The active main source and its sections agree with the corresponding `project_review/inspection/paper_planning/overleaf/kwallet_paper_submission_v2` copy and the organized daily manuscript. The accompanying `icdm_framing_revision_report.md` explicitly records ICDM framing.

I read the complete active LaTeX manuscript and its relevant appendix/supplement sources, followed the active figure/table inputs, inspected the TikZ diagrams, and viewed the standalone quantitative figure PDFs. All 16 active input dependencies checked were present. I did not inspect a compiled full-paper PDF: none was located and no local LaTeX compiler was available. Visual judgments about TikZ placement are source-based, not a claim to have checked the final submitted page layout.

**UNRESOLVED:** This cannot yet be certified as the exact rejected submission. No submitted main PDF, submitted supplement PDF, submission receipt, or reviewer reports were located in the searched repository, local project/archive, Desktop, Downloads, and attachments. The package verification note explicitly says the PDFs still need to be added. Actual rejection reasons are unknown; weaknesses below are reviewer-style judgments, not reconstructed reviewer comments.

The nested `submission_package/main/main_source_snapshot` is stale/incomplete: its main source hash is `df5e044fad5276a64164c154f837ebc6df94bb8fd1802ea792d539ce16063c1e`; its nine section files differ from, or are missing relative to, the active version. It lacks the problem-formulation section and regime table referenced by the active paper. Do not silently call that snapshot the rejected submission. Confirm the final submitted version with the PI before rewriting paper source. Planning can proceed provisionally.

## 2. Recoverable code and evidence lineage

Reference roots (read-only in this task):

- **M:** /Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/overleaf/kwallet_paper_submission_v2
- **W:** /Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/01_日常研究
- **Z:** /Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized.zip
- **H:** /Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/audit_artifacts

Paths below use these explicit roots. They identify original local artifacts, not collaborator code. In particular, `W/src/kwallet` is a relocation of the original `src/idea3`/`src/idea4` implementation inside the organized archive; it must not be confused with PR #3's independently rebuilt package of the same name.

The local Git working copy is `/Users/zhouzhou/Desktop/kwallet-rl`, branch `main`, HEAD `bc47a4c695cc7ab2cce2159bb7bd075867bc25af`. Before this task, `research/PROJECT_STATUS.md` was already modified and `collab/` was untracked. Neither is part of this task's changes.

| Paper claim / component | Original code | Experiment and saved evidence | Paper destination / confidence |
|---|---|---|---|
| Sequential collateral-control dynamics | `W/src/kwallet/kwallet_ctx_attn_fair_benchmark.py`; Git counterpart `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py` | Original pool files and `W/results/published/reproducibility/kwallet_environment_regime_audit.md` | Problem formulation; overview. High confidence in implementation; exact submission identity pending. |
| JA-PPO versus IFAC versus SC-FAC | `W/src/kwallet/kwallet_basic_ppo_fair_benchmark.py`, `run_factorized_ac_benchmark.py`, `run_conditional_factorized_ac_benchmark.py`; Git counterparts under `src/idea4/ac/code/` | `W/results/main_experiment_index.csv` maps 120 selected runs; raw `W/results/raw/kwallet/{basic_ppo,factorized_ac,conditional_factorized_ac}/runs/.../cross_regime_results.json` | `M/tables/table_k24_money_compact.tex`; ten-seed summary and paired CIs. Recomputed all 120 Money values from raw regime summaries. |
| Conditioning contribution | Same SC-FAC script, existing `condition_mode=zero_settle_embedding` | `W/results/published/conditioning_ablation/paper_ready_ablation_v1/` and June 5 completion logs; 12 zero-conditioning runs and 12 matched full results | Existing ablation report/table; mixed evidence. Selected zero checkpoints recoverable, not a proposed new architecture. |
| Compact policy outputs | The three original policy definitions | Analytical counts `(k+1)^2` versus `2(k+1)`; `W/results/published/supporting_tables_original/output_size_scaling_plot_data.csv` | Action-parameterization figure and output-size plot; supported as a representation count, not latency proof. |
| Wallet-count diagnostic | Same original families, but SC-FAC uses conditional hidden width 128 in this study rather than 256 in the main study | `W/results/published/wallet_scaling/` and `supporting_tables_original/k_scaling_C1200_table.csv`; 3 seeds | Supplement scaling plot/table. Separate evidence layer; not zero-shot transfer. |
| Penalty sensitivity | Recompute Money from the same saved accepted value / flush counts | `W/results/published/penalty_sensitivity/`; 3-seed summaries | Supplement tau plot/table. Post-hoc scoring sensitivity, not reward-adaptive retraining. |
| Fair FA/FWF comparison | `W/src/kwallet/evaluate_standard_policies_same_pool.py` and fairness notes | `W/results/published/rule_baselines/`, `comparison_fairness/` | Constrained one-flush adapters in main table; native policies have different semantics. |
| Stronger feedback comparator | `H/heuristic_diagnostic.py`, `H/verify_heuristic_original.py` | `H/heuristic_diagnostic.csv`, `H/heuristic_original_replay.csv`; 48 regime rows in original simulator | Later local diagnostic of the original lineage; absent from active main comparison. Not PR #3 BFP. |
| Computation costs | Original final-shape models | `W/results/published/inference_cost/compute_benchmark_summary.csv` | Architecture-only CPU measurements; randomly initialized models, not end-to-end deployment. |
| General/two-pool collateral extension | Different extension scripts and assumptions | `W/results/published/collateral_extensions/` | Active source still includes an extension table despite older removal notes. Remove from this minimum-change AAMAS scope. |

The four main original code files differ from their current Git counterparts only in import/root/output-path relocation in the inspected diffs. That supports implementation continuity, but does **not** establish the exact training-time Git commit. Preserve source hashes, configs, pool fingerprints, and checkpoint paths as the reproducibility record.

SHA-256 of the organized source inspected:

| Component | Hash |
|---|---|
| Environment | `f8fda64010fcd9c00f04943d818c0a740b060e75358b0daa3b46ec0beebcd752` |
| JA-PPO | `2deeee09ec1ccbfe0619f95e889c51b5636d29810b3146c534b072a803b63f37` |
| IFAC | `8c4704adeec9b19f7b3bdd41c33b6070bbaf77af670ddab90f482ea1cd5a481f` |
| SC-FAC | `356a1195b0df16dfb6c3ec54b6b6dedc342d49eba4e24709e4ebde0aa16aee69` |

**September 30 read-only checks on the broader archive (retained):** all 120 indexed raw result records exist; macro-averaging their 12 Money summaries reproduces the selected seed table to at most 3.64e-12. All 120 selected `best_model_path` suffixes have a unique matching archive member in Z. The archive contains 1,242 checkpoint entries overall; do not select weights by newest timestamp. The six zero-conditioning checkpoints needed for the proposed two-capacity study also match the exact June 5 completion-log timestamps uniquely. Presence is verified; loading/replay of every checkpoint has not been performed in this task.

**Prior recorded checks, not rerun here:** `H/checkpoint_replay.csv` records exact replay of five metrics for three C=900, seed=123 models on US; `H/fingerprint_verification.json` reports 1,680 fingerprint checks with zero mismatches. Treat these as saved audit evidence until Trae validates the recovery manifest.

The generator in current Git, the inspection tree, and the original full archive subtree has identical SHA-256 `d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8`. Use that verified generator. The organized file named `W/docs/reference_dependencies/original_regime_generator.py` actually contains a table-regeneration script; its filename is misleading and it must not be used to generate streams. This is a concrete packaging/provenance defect, not evidence that the saved pools are wrong.

## 3. Reconstructed research problem and protocol

**Problem:** A single autonomous online controller allocates arriving transaction value across a fixed budget of homogeneous collateral wallets. It chooses a settlement wallet (or no settlement) and a flush/replenishment wallet (or no flush). Replenishing consumes a fee and temporarily makes the chosen resource unavailable; accepting now changes capacity available later. This is a single-agent sequential resource-control problem, not a multi-agent system.

**Environment:** total collateral C, k wallets initially of capacity C/k, horizon T=1,000 decisions, frozen/replenishment delay F=3. The broader recovered local result grid spans C in {800,900,1000,1200}, k=24. The revised historical core is C={800,1200}; exact rejected-paper membership of the additional capacities remains unconfirmed. The observation has 3k+2 features: balances, flags, timers, current transaction value, and time. At k=24 it has 74 features. The policy samples settlement first, then flush; the simulator executes flush **before** settlement. Flushing the selected settlement wallet can therefore prevent settlement. Explain these two orders separately.

**Objective versus optimization surrogate:** primary evaluation is `Money = accepted transaction value - 10 × executed flush count` (value multiplier p=1). Training uses the original shaped reward: accepted value/1000, rejection penalty -0.02, executed-flush penalty -0.01. Validation chooses the best checkpoint by value-acceptance ratio, not Money. These are different objectives; retain them and disclose them. Correct prose to match implementation: an invalid/non-executable flush request does not incur an executed-flush fee. Also reconcile the paper's usable/frozen flag convention and t/T notation with the code's convention and t/(T-1). These are documentation fixes, not authority to change the simulator.

**Policies:** JA-PPO uses a flat joint categorical head of (k+1)^2 logits; IFAC uses independent settlement and flush heads; SC-FAC uses settlement followed by a flush distribution conditioned on a learned settlement-index embedding. At k=24 the factorized policies emit 50 selected-path logits versus JA-PPO's 625. Main SC-FAC uses embedding size 32 and conditional hidden width 256 with a shared 128-unit representation. IFAC has fewer parameters, so its contrast with SC-FAC does not isolate conditioning alone. The existing zero-embedding control preserves SC-FAC's head dimensions while removing the settlement signal. No conditioning reimplementation is needed.

**Training:** shared PPO family settings include learning rate 3e-4, gamma .98, GAE .95, clipping .2, four PPO epochs, minibatch 256, value coefficient .5, entropy coefficient .03 to .003, gradient cap 1.0. The mixed master contains 5,000 episodes; the configured training subset limit is 3,000, but the 1,000-episode loop consumes the first 1,000 once. Do not describe these runs as training on all 3,000 or 5,000 episodes. Each run uses 1 million training steps and validation every 50 episodes on 200 fixed episodes.

**Data:** MIX12_EQ combines six synthetic amount families with stationary/bursty variants: US, TLS, LNS, TLNS, TPLS, PLS, UB, TLB, LNB, TLNB, TPLB, PLB. Exact family distributions, truncation, calibration, and burst construction are in `/Users/zhouzhou/Desktop/kwallet-rl/src/ideaextra/kwallet_ideaextra_generator.py` and the published generator audit. Evaluation uses 200 fixed episodes per regime, 2,400 per model. These are held-out streams from known families. They are not unseen-regime OOD, online learning at test time, or arrival-rate bursts. The heavy-tail families are capped.

**Seeds:** the principal three-method comparison already has ten seeds per capacity: 123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888. The mechanism control, scaling, and penalty diagnostics use three seeds {123,323,532}; do not assign ten-seed confidence to those results.

**Baselines and history:** retain JA-PPO, IFAC, SC-FAC, and the precisely labeled constrained FA/FWF rules. DQN and other older exploratory models are background/supplement only. Do not replace these with collaborator Set-SC-FAC, cross-k, switching, or rebuilt BFP results. The original paper's problem and SC-FAC implementation remain the core; the revised argument centers on structured action representation because causal conditioning evidence is weaker.

## 4. Main numerical evidence and statistical limits

**FACT: historical macro-average Money, ten learned-policy training seeds.** Rule columns are fixed-policy means on the same old streams, not ten independent trained seeds.

| C | JA-PPO | IFAC | SC-FAC | Constrained FA | Constrained FWF | Later original-environment feedback diagnostic |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 3368.57 | 3607.05 | 3999.25 | 2772.78 | 2288.24 | 4622.56 |
| 900 | 5636.16 | 6347.35 | 6627.98 | 4047.32 | 3512.46 | 7073.82 |
| 1000 | 8332.27 | 8786.44 | 9064.80 | 5402.43 | 4802.56 | 9661.24 |
| 1200 | 14443.01 | 14472.93 | 14687.65 | 8591.65 | 7810.86 | 15639.23 |

The feedback column is exploratory later evidence, displayed separately from the submitted-era comparison. Reaggregation of its 48 original-simulator rows matches the saved vectorized diagnostic. It exceeds SC-FAC's mean by approximately 15.59%, 6.73%, 6.58%, and 6.48%. It accepted every individually feasible transaction in those old streams. That is an accepted-value feasibility ceiling on those streams, **not** a proof of optimal Money.

**Retrospective statistical sensitivity, recomputed from the saved ten-seed table:** two-sided paired t intervals over training-seed differences, df=9. Holm correction below treats the eight SC-FAC comparisons as one family. This was not a historical preregistration. Minor endpoint differences from old tables reflect their rounded t critical value.

| C | Contrast | Mean difference | Unadjusted 95% paired CI | Raw p | Holm p, 8 comparisons |
|---:|---|---:|---|---:|---:|
| 800 | SC-FAC − JA-PPO | 630.68 | [258.58, 1002.78] | .004002 | .020010 |
| 900 | SC-FAC − JA-PPO | 991.82 | [657.98, 1325.67] | .000086 | .000605 |
| 1000 | SC-FAC − JA-PPO | 732.53 | [495.61, 969.46] | .000064 | .000509 |
| 1200 | SC-FAC − JA-PPO | 244.64 | [24.50, 464.77] | .033097 | .099289 |
| 800 | SC-FAC − IFAC | 392.20 | [-58.11, 842.51] | .080312 | .134919 |
| 900 | SC-FAC − IFAC | 280.63 | [133.11, 428.14] | .001981 | .011885 |
| 1000 | SC-FAC − IFAC | 278.35 | [59.44, 497.27] | .018285 | .073139 |
| 1200 | SC-FAC − IFAC | 214.72 | [-19.00, 448.44] | .067460 | .134919 |

**FACT: existing mechanism ablation**, full SC-FAC minus zero settlement embedding, three matched training seeds:

| C | Mean Money difference | Unadjusted 95% paired CI |
|---:|---:|---|
| 800 | +51.19 | [-518.19, 620.57] |
| 900 | +249.32 | [17.42, 481.22] |
| 1000 | -50.10 | [-280.53, 180.34] |
| 1200 | -190.40 | [-433.88, 53.07] |

The ablation summary's “unfavorable” label at C=900 reflects its flush metric; it must not be mistaken for a negative Money difference. At C=1200 the full policy flushes more and its mean Money is lower. Report the primary Money effect first and use accepted value / flush counts to explain it. Do not discard unfavorable capacities or interpret a CI crossing zero as equivalence.

The prior rule is simple: choose the smallest usable balance that can settle the current transaction; among other usable wallets below half of their full C/k balance, flush the smallest balance, otherwise no-op. It can still replenish when the current transaction is oversized. It uses current state only, at most one flush, the original F and execution order. It is not PR #3's rule with an early oversized-transaction return. Its threshold was fixed at .5 without a recorded search, but its design followed inspection of old results; fresh-stream evaluation is warranted.

Main raw files contain regime summaries rather than the full per-episode records needed for a new stream-level uncertainty analysis. Training seeds, episodes, and regimes are not interchangeable replicates. Existing repeated test inspection and architecture/configuration selection further limit confirmatory interpretation. Fresh streams improve confirmation within the same generator; they cannot establish real-world generalization.

## 5. Headline-claim audit

A = strongly supported; B = supported but statistically weak/qualified; C = limited-seed diagnostic; D = unsupported or ambiguous; E = contradicted by comparable later original-lineage evidence. A applies to the precise wording, not an unrestricted generalization.

| Candidate headline | Class | Evidence and permissible wording |
|---|---|---|
| Factorization reduces selected-path policy outputs from quadratic to linear in k | A | Direct architecture count: 625 versus 50 at k=24. No claim that all conditional distributions are materialized in 50 logits. |
| SC-FAC has the highest mean Money among the three learned policies at all four recovered local capacities | A | Verified ten-seed table; retain “mean,” “these policies,” and “historical settings.” |
| SC-FAC beats JA-PPO statistically at every resource level | B | Four unadjusted positive paired intervals; C=1200 does not survive the stated eight-test Holm family. Qualify or remove “every.” |
| Independent factorization alone reliably causes all improvement | D | IFAC means exceed JA-PPO at all four C, but architecture/optimization differences and small high-C gap preclude a blanket causal statement. New A includes the direct paired contrast. |
| Conditioning consistently improves on IFAC / is the source of the gain | D | SC-FAC versus IFAC is capacity-confounded; matched zero-embedding control is mixed. C=900 is promising, not universal. |
| Conditioning yields a positive effect at C=900 | C | Three-seed matched control, unadjusted positive CI. Additional-capacity diagnostic only; C900 is excluded from the current C800/1200 confirmation plan. |
| The learned method outperforms strong domain-aware rules | E | Later original-environment balance-threshold full-refill feedback diagnostic exceeds learned means at every C. This conclusion does not use collaborator results. |
| SC-FAC beats the specific constrained FA/FWF adapters | A | Saved historical comparison, with adapter/action semantics disclosed. This is not “beats heuristics generally.” |
| Robust across unseen regimes / stream switching / zero-shot k | D | No corresponding evidence in this ICDM experiment lineage. Known-family static test streams and separate-k training do not establish these claims. |
| Scales in performance across k | C | Three-seed fixed-total-C diagnostic changes both action dimension and per-wallet feasibility; SC-FAC width also differs. Keep bounded supplement analysis. |
| Robust to changed training reward | D | Existing tau analysis rescored saved actions; it did not retrain agents for changed penalties. |
| Faster online inference than JA-PPO | E | Original architecture-only CPU batch-1 means: JA .0346 ms, IF .0382 ms, SC .0687 ms. Compact logits do not imply faster sequential action selection. |
| Ready for real financial deployment / general payment-network control | D | Synthetic scalar-value simulator, no real trace/network validation. Remove deployment-level conclusions. |

## 6. Frozen components — KEEP AS-IS

| Component | Decision |
|---|---|
| Research problem and single-agent interpretation | **KEEP AS-IS**: online settlement/replenishment with delayed resource availability. |
| Core simulator and action timing | **KEEP AS-IS**: original homogeneous-wallet environment, one settlement/one flush, flush-before-settle dynamics. Correct inconsistent prose only. |
| Original main method | **KEEP AS-IS**: SC-FAC E32/H256, original PPO training and evaluation path; no set model or new architecture. |
| Reward and checkpoint selection | **KEEP AS-IS**: original shaped reward and value-acceptance validation. Disclose mismatch with Money; do not retune now. |
| Principal benchmark | **KEEP AS-IS**: 12 known synthetic families, T=1000, F=3, k=24; use the confirmed C800/1200 core; retain other recovered capacities only as explicitly separate diagnostics. Add fresh streams only after approval. |
| Principal baseline set | **KEEP AS-IS**: JA-PPO, IFAC, constrained FA/FWF remain labeled. Add one fair feedback rule; do not replace baseline implementations. |
| Main evaluation metric | **KEEP AS-IS**: macro-regime mean Money with flush cost 10. Acceptance and flushes remain secondary explanations. |

Only a demonstrated scientific defect should reopen a frozen component, and that would require a separate PI scope decision. No such implementation change is authorized here.

## 7. Exactly three priority weaknesses

### P1 — Weak rule comparisons overstate the practical advantage

**Why it matters:** Beating constrained FA/FWF does not demonstrate that learning is useful relative to a competent controller. The original-environment diagnostic already gives contrary evidence. **Missing:** a frozen, directly comparable confirmation of the best simple feedback rule on fresh streams. **Smallest fix:** retain the old comparisons, independently implement the specified balance-threshold full-refill rule as a new AAMAS baseline on the same two new test settings, publish its favorable result if it persists, and describe the learned-policy contribution narrowly. Do not launch a new method to chase the rule.

### P2 — Conditioning is more central in the story than the mechanism evidence warrants

**Why it matters:** SC-FAC versus IFAC changes more than the settlement signal; a larger conditional head is a confound. The available shape-matched zero control is mixed across C. **Missing:** a five-seed, paired confirmation at the two confirmed core capacities, where the existing conditioning intervals are inconclusive. **Smallest fix:** C=800 and C=1200, reuse three zero-control seeds and all full-model seeds; train only zero-control seeds 777 and 999 at each C. Shift the title/central claim toward structured factorization if the control remains mixed. No new architecture.

### P3 — Evidence presentation is stronger than the documented confirmatory protocol

**Why it matters:** Ten-seed evidence already exists; the issue is not a generally single-seed paper. Multiple contrasts, repeated reuse of the old test pools, incomplete submission identity, and sparse run-level provenance can produce exaggerated certainty. **Missing:** a frozen manifest, explicitly defined inference unit/test family, and untouched evaluation streams. **Smallest fix:** recover weights, preserve all historical settings, freeze the small matrix before new evaluation, report paired seed CIs and multiplicity, retain episode records, and distinguish exact submission uncertainty from verified data. Do not rerun all 120 training jobs.

## 8. Scope decision and acceptance-quality assessment

**INFERENCE:** The strongest coherent contribution is an empirical study of compact structured policies for a delayed-resource online controller, preserving SC-FAC as the main instantiated method. Factorization supplies the clearest structural result; conditioning is a tested refinement with limits. Generic autoregressive factorization is not established here as a new general algorithm. Explain the problem-specific coupling and cite its antecedents when revising related work.

This minimum revision can materially improve soundness and clarity. It cannot guarantee AAMAS acceptance: incremental architectural novelty, synthetic-only validation, and a stronger simple rule remain genuine significance risks. If the PI requires a paper whose headline is learned-policy superiority over strong rules, current evidence does not support that objective within this scope.

**OUT OF SCOPE FOR AAMAS:** PR #3 integration; Set-SC-FAC; broad cross-k/zero-shot or switching/OOD programs; recurrent/Transformer history models; new rewards, simulator, or agent architecture; large hyperparameter sweeps; exhaustive 10+ seed reruns; idea5/two-pool/general-collateral extensions; deployment claims. Do not merge the collaborator branch or import its numbers.

## 9. PI decisions / missing artifacts

1. Confirm whether the identified active source is the rejected ICDM version; provide the submitted PDF/source receipt and reviews if available. Missing reviews must remain missing in the report.
2. Approve the bounded contribution: structured policy evidence with mixed conditioning and transparent strong-rule results; no “RL beats strong heuristics” headline.
3. Approve the separate proposed execution budget: four new training runs plus fixed checkpoint evaluation. This audit has not launched or delegated them.
4. Resolve author accounts, qualified reciprocal reviewer, and potential concurrent-submission overlap with the ICASSP track using the policy checklist in the paper plan.

The companion experiment matrix defines exact execution and stopping conditions. The paper plan defines figures, claims, submission obligations, and the draft Trae handoff.
