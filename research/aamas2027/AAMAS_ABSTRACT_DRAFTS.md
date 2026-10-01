# AAMAS 2027 — Abstract drafts after local recovery

Updated: 2026-10-01. The operative historical core is **C800/1200**, k24, F3, T1000; all 60 principal setting/seed artifacts, six corresponding zero controls, and fourteen pools were physically recovered. Checkpoint/pool bytes and metadata were verified; **no replay or new experiment was executed**. Additional local C900/1000 records do not prove membership in the exact rejected submission and are not used to define these abstracts.

## Recommendation

Version A is the safest current draft because it uses only recovered principal/control evidence. It does not present the separate local feedback diagnostic as an original campaign baseline or a completed new AAMAS comparison. Do not interpret this draft as a finding that learned policies beat strong heuristics. Versions B/C remain internal and contain explicit future-result placeholders.

Recommended title: **Structured Policy Factorization for Autonomous Streaming Collateral Control**. Method and environment remain unchanged. The prospective new baseline is **BF-T0.5**, independently implemented from its mathematical specification in the original environment: .5 is a remaining-balance trigger, and every completed flush restores full C/k. No collaborator implementation or numerical result is used. This baseline is new to the AAMAS campaign, not claimed as a novel heuristic invention.

The experiment hold remains active. A future five-seed confirmation at C800/1200 requires four missing zero-control runs, subject to successful separately approved replay. No abstract was submitted by this task.

## A. Conservative / safest — 171 words

An autonomous controller managing streaming transactions must decide both where to settle each request and which collateral wallet to replenish. These decisions are coupled because replenishment restores resources but temporarily removes a wallet from service. We study structured policies for this sequential decision problem under a fixed collateral budget. We compare a flat joint-action PPO policy, independent settlement and replenishment heads, and a settlement-conditioned factorization that conditions replenishment on the selected settlement action. At 24 wallets, factorization reduces the selected-path policy outputs from 625 to 50. Recovered experiments across twelve synthetic transaction regimes, two collateral capacities, and ten training seeds show that the conditioned policy has the highest mean net accepted value among these three learned policies. A matched three-seed zero-conditioning control, however, does not establish a consistent additional benefit from the settlement signal. These results support a bounded empirical comparison of structured action representations rather than a general claim of conditioning or learned-policy superiority. The setting makes explicit how coupled actions and delayed resource availability interact in autonomous online control.

## B. Balanced — internal until results are complete — 155 words

Streaming collateral control requires an autonomous agent to coordinate immediate settlement with replenishment decisions that constrain future resource availability. We examine structured policies for this problem using the original K-Wallet environment and a common PPO training protocol. Independent and settlement-conditioned factorizations replace a quadratic joint-action head with a linear number of selected-path outputs. The recovered historical core comprises twelve transaction regimes, collateral capacities 800 and 1200, and ten training seeds per learned method. SC-FAC achieves the highest mean net accepted value among the three learned policies, while a matched zero-conditioning control leaves the contribution of the settlement signal unresolved. We distinguish the policy-representation comparison from a practical comparison against a domain-aware rule that uses best-fit settlement and a fixed remaining-balance trigger for full replenishment. [Finalize the paired five-seed fresh-stream results and independently implemented rule comparison after approval and execution.] This analysis separates compact action representation, conditioning, and domain knowledge without changing the original control problem.

## C. More ambitious framing — internal until results are complete — 156 words

Autonomous resource-control agents often make coupled decisions whose effects extend beyond the current request. In streaming collateral control, choosing a settlement wallet and choosing a wallet to flush jointly affect immediate return and future feasibility. We study this structure through joint, independent-factorized, and settlement-conditioned PPO policies. Factorization reduces selected-path policy outputs from 625 to 50 at 24 wallets. Recovered ten-seed experiments at collateral capacities 800 and 1200 favor the conditioned policy in mean net accepted value among the learned alternatives, but the available three-seed zero-input control does not demonstrate a reliable conditioning gain. A compact confirmation study is designed to test this distinction against fresh streams and an independently specified, fixed-threshold full-refill heuristic under identical information and timing constraints. [Replace with completed paired effects and uncertainty; do not imply that the planned campaign has run.] The contribution is an evidence-based account of where structured action representations help and which stronger mechanism or practical-superiority claims remain unestablished.

## Claim safeguards

- “Recovered” means physical files, hashes, raw JSONs, and configurations are available, not that they were rerun on the server.
- The ten-seed claim concerns the two confirmed core capacities and three learned methods. It is descriptive mean ordering, not a simultaneous-significance claim.
- The six recovered controls are seeds123/323/532 at both core C. Do not call the planned five-seed mechanism test complete.
- The separate local original-environment feedback diagnostic remains disclosed in the audit as exploratory evidence. Omitting its unconfirmed-as-submission numbers from this abstract does not permit suppressing a stronger comparator from the eventual evaluation.
- No C900 value may be relabeled C800. No PR #3 result enters any draft.
- Selected-path output count is not inference latency, parameter count, or a runtime guarantee.
- “Online” describes sequential decision-making, not online updates to trained weights. Known-family test streams are not OOD, switching, or real-world validation.
