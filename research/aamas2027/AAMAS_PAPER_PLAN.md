# AAMAS 2027 — Minimum-change paper plan

Updated: 2026-10-01 after Trae Phase 0 and local artifact recovery. Planning only; experiment hold remains in force. The companion audit identifies the best recovered ICDM source and unresolved exact-submission identity. The experiment matrix is proposed, not executed. No collaborator PR #3 evidence is used.

## Current recovery decision — supersedes the previous execution assumption

PATH A is selected. Local physical recovery found 60/60 principal best/last checkpoints with raw JSONs at **C800/1200**, six selected zero-control best/last checkpoints and raw JSONs, and all fourteen historical pools. Checksums, archive CRCs, original configs, and 924 run-level pool fingerprints were checked. No model load or replay was performed. Principal weights also exist as loose files in Downloads and the project mirror; absence from the server checkout does not require retraining.

The exact rejected PDF/source version remains unconfirmed. C900/1000 occur in recovered local source/results, but are not included in the revised core without submission-specific evidence. Do not rewrite their old numerical values as C800 results. The audit preserves that distinction.

**Minimum future training:** four zero-control jobs, C800/1200 × seeds777/999, contingent on a separately approved replay gate. The principal three-method confirmation reuses existing seeds123/323/532/777/999. The next Trae action is manifest preparation/hash verification only. No execution approval is implied by this plan.

**Baseline:** add an independently specified **BF-T0.5** rule in the original environment. .5 triggers a flush when remaining balance is below half capacity; the flush still restores **full C/k** after the original delay. No tuning and no PR #3 code/results. The old local diagnostic is separate, exploratory audit evidence and must not be called a result of the original campaign or new AAMAS experiment.

## One paper story

**Recommended title: Structured Policy Factorization for Autonomous Streaming Collateral Control.**

An autonomous controller must repeatedly decide where to settle a current transaction and which collateral resource to replenish. Settlement and replenishment are coupled because replenishment temporarily removes a wallet from service. A flat joint action head enumerates every pair, growing quadratically with the number of wallets. The original paper studies independent and settlement-conditioned factorizations; the latter preserves a direct dependence between selected action components with a linear number of selected-path outputs.

The empirical contribution is a carefully bounded comparison of these policy representations under one fixed sequential-control protocol. The recovered ten-seed results at C800/1200 favor SC-FAC in mean Money among the learned policies, while the shape-matched control shows that explicit conditioning is not uniformly beneficial. A newly added, independently implemented rule will supply a necessary practical reference. A separate local audit diagnostic was favorable to that rule; it is exploratory context, not an original-campaign or completed AAMAS baseline result. Preserve the problem and SC-FAC method, but make compact representation and measured limitations the central story.

This is single-agent control. “Online” describes decisions from current state during a stream; it does not mean the trained policy updates at deployment. Do not advertise multi-agent coordination, online adaptation of weights, universal robustness, or novel generic autoregressive factorization. Acceptance remains uncertain because novelty and rule superiority are substantive limitations, not presentation issues that a venue change removes.

### Contribution statements to aim for

1. A precise delayed-resource streaming-control formulation with coupled settlement/replenishment decisions, explicit execution order, and reproducible synthetic regimes.
2. An original application and comparison of joint, independent-factorized, and settlement-conditioned policies, with an exact quadratic-to-linear selected-path output count. Preserve SC-FAC as the main instantiated method without presenting generic factorization as new theory.
3. A transparent empirical account of learned-policy gains, conditioning limits, and domain-aware feedback performance, with paired seed uncertainty and a small frozen confirmation study.

## Ranked titles

| Rank | Title | Reason |
|---:|---|---|
| 1 | Structured Policy Factorization for Autonomous Streaming Collateral Control | Faithful to the original task, identifies the representation contribution, avoids promising a conditioning gain. |
| 2 | Learning Structured Actions for Streaming Collateral Control | Short and accessible; less explicit about factorization. |
| 3 | Joint and Factorized Policies for Sequential Collateral Control | Most conservative comparative framing; less distinctive. |
| 4 | Settlement-Conditioned Policies for Autonomous Collateral Control | Closest to ICDM title; use only if conditioning remains a carefully qualified refinement. |
| 5 | Coupled Settlement and Replenishment Decisions in Streaming Resource Control | Highlights the decision problem; risks understating the learning contribution. |

## Headline claim contract

| CLAIM | EVIDENCE REQUIRED | CURRENT STATUS | NEW EXPERIMENT NEEDED? YES/NO |
|---|---|---|---|
| The structured representation reduces selected-path logits from (k+1)^2 to 2(k+1). | Architecture equations and exact counts. | Verified in original code: 625 to 50 at k=24. No runtime guarantee. | **NO** |
| SC-FAC has the highest historical mean among JA-PPO, IFAC, and SC-FAC at the two core capacities. | Complete ten-seed table and raw-result mapping. | Verified 60 core records and matching checkpoints; this precise descriptive claim is supported. | **NO** |
| Factorized policies improve over flat JA-PPO on fresh streams under the fixed protocol. | Paired SC−JA and IF−JA, fixed seeds and pools, intervals/multiplicity. | Historical means support the direction; fresh confirmation pending. | **YES — A** |
| Settlement conditioning adds a reliable benefit beyond head dimensions. | Full−zero matched architecture, at both planned C, five seeds. | Existing three-seed effects are mixed; cannot make a general positive claim. | **YES — B**, then qualify even if positive |
| The learned controller beats a strong domain-aware rule. | Same-environment, same-stream comparison without rule tuning. | Contrary separate local diagnostic at the two core C and two additional recovered C; not a completed new AAMAS baseline. Do not make this claim now. | **YES — C** to assess, not to promise reversal |
| Relative performance is consistent across the two confirmed core resource levels. | All capacities retained, effect sizes, seed uncertainty, interaction caveats. | Mean ordering is descriptive; strength of evidence and conditioning effect vary. | **NO** for bounded historical description; **YES — A/B** for fresh two-C statements |
| The method generalizes to unseen regimes, switching, or zero-shot k. | A different evaluation program. | Missing from the relevant lineage. | **NO for this revision — OMIT CLAIM** |
| Factorization gives faster online inference. | Relevant end-to-end timing under matched implementation/hardware. | Existing batch-1 architecture benchmark does not support SC speed superiority. | **NO for this revision — OMIT CLAIM** |

Replace future-result placeholders only after the fixed matrix is complete. If B is inconclusive, the paper still reports the conditioning control. If C confirms the rule's advantage, retain it in the main paper and discuss the boundary of the learned approach. Do not rescue the headline by moving an unfavorable comparator to a hidden appendix.

## Audit of every figure used by the recovered active source

Source root: `/Users/zhouzhou/.codex/.chatgpt-projects/g-p-6a3ca98152f88191a4854b7ad60fd02d/KWallet_Organized/03_论文与分析记录/overleaf/kwallet_paper_submission_v2`.

The active main source uses five figures; its supplement uses two quantitative figures. The first three are TikZ and were inspected as source; the four standalone quantitative PDFs were viewed. Exact numbering/layout in the rejected PDF remains unverified. Dormant figure files and collaborator visuals are not treated as submitted figures.

| Current figure | Decision | Specific problem | Minimum redesign |
|---|---|---|---|
| `figure1_overview.tex` | **REDRAW** | Many panels repeat the next two diagrams; small text; the general-collateral extension competes with the core story. | One full-width pipeline: current transaction + wallet state → controller → (settle,flush) → original transition → next state. Explicitly distinguish policy sampling order from environment execution order. Remove the extension. Add three small action-head insets. |
| `figure2_action_parameterization.tex` | **KEEP content; REMOVE standalone placement** | Three wide repeated state/encoder panels consume space; overlaps overview and architecture. | Merge its joint/independent/conditioned comparison into Figure 1. Use equations and 625/50 counts, not decorative categorical bars. |
| `figure3_scfac_architecture.tex` | **REDRAW / merge** | Dense layout scaled down; the source draws a critic-to-joint-action arrow, suggesting the value head selects the action. | Show settlement sample → index embedding → conditional flush; show critic as a dashed training-only branch. Keep the shared state input. Merge into Figure 1; optional larger detailed diagram in supplement. |
| `output_size_scaling_plot.pdf` | **REMOVE** as standalone | Plots an elementary count; IFAC and SC-FAC curves coincide and duplicate the legend. It can be misread as runtime scaling. | A compact analytical count row in Table 1 and Figure 1: joint (k+1)^2, factorized 2(k+1), 625→50 at k=24. No statistical error bars for deterministic counts. |
| `k24_C1200_scfac_basicppo_regime_gain_heatmap.pdf` | **REPLACE** in main | Narrow 12-row mean-only heatmap; only JA comparator; no seed uncertainty; color range begins above zero and visually emphasizes uniformly positive gains. | Use main Figure 3 for the matched mechanism effect with a zero reference and paired seed points. If per-regime gains are retained, use a supplement forest plot from exact seed-level records, with zero-centered effects and explicitly defined CIs. |
| Supplement `k_scaling_money_plot.pdf` | **KEEP in supplement; REDRAW labels** | Curves mostly overlap; fixed C changes C/k and feasibility as well as action dimension. Three seeds and different SC width limit conclusions. | State C=1200, each C/k, seeds, SC H128, and uncertainty type; optionally annotate the feasible-value ceiling using existing records. Label separate-k training, not transfer. Do not add runs. |
| Supplement `tau_sensitivity_money_plot.pdf` | **KEEP in supplement; REDRAW style** | Easy to misread as learned adaptation to different penalties; uses three seeds, unlike main ten-seed results. | Caption “post-hoc rescoring of fixed policies,” specify three seeds and intervals, standardize colors. Keep tau=10 reference. Do not imply reward retraining. |

### Final main figures: three, with distinct jobs

**Figure 1 — decision problem and structured policy.** Full-width vector diagram merging the useful content above. Illustrate one current transaction and temporary wallet unavailability; use a small state-transition example whose numbers obey the original dynamics. Use TikZ/vector drawing, not image generation. Include the sampling/execution order distinction and keep the critic outside the action path.

**Figure 2 — main performance with uncertainty.** Panel A: historical C={800,1200}, JA/IF/SC ten-seed means and 95% seed CIs, with the original constrained rules labeled as fixed-rule results. Keep the separate local feedback diagnostic in an explicitly exploratory supplement; do not relabel it as a completed original/AAMAS baseline. Panel B: NEW12 C={800,1200}, five-seed learned-policy means/CIs and the fixed feedback mean. Keep old/new panels separate, name their pools and sample sizes, and never connect them as one learning curve. Keep the prior audit feedback comparison clearly separate from new AAMAS baseline results. Use Money units and identical method colors. Do not fabricate a seed CI for rules.

**Figure 3 — mechanism, not another architecture cartoon.** At C=800 and 1200, paired full−zero Money differences with all five seed points, mean, and 95% paired CI. A zero reference is mandatory. Optionally add accepted-value and fee decomposition as a secondary aligned panel using the same saved records. No new trajectory experiment is necessary.

Use vector PDF/SVG where possible, colorblind-safe consistent method colors, readable two-column sizing, units on every quantitative axis, and captions defining the replicate and CI. Avoid 3D effects, truncated bar axes, excessive legends, and meaning conveyed only by color. Render the final manuscript at actual column size before judging readability. No figures are redrawn in this task.

## Table audit and final set

| Existing table / group | Decision |
|---|---|
| Main `table_k24_money_compact.tex` | **KEEP evidence, REFORMAT.** Its many metric/CI columns are dense. Move the two-C core means to Figure 2A; retain exact numerical historical values and all contrasts in supplement. Do not replace ten-seed values with five-seed subset values. |
| `table_regime_definitions.tex` and generator parameter table | **KEEP compact definitions.** Main setup gives six families × stationary/bursty and key caps/scale; supplement holds exact parameters, calibration, and hashes. Do not expand unofficial mnemonic names as if they were official definitions. |
| `table1_method_complexity.tex` | **KEEP/MERGE** into new Table 1; output counts and parameter sizes are different quantities. Do not sum actor and critic totals that share a trunk. |
| Evidence coverage, method-name mapping, fairness-accounting tables | **KEEP in supplement**, compact. They document train/eval-only status, seed counts, action semantics, and exact source names. |
| Full k24 and k-scaling tables | **KEEP in supplement**, explicitly label ten versus three seeds and H256 versus H128. |
| Conditioning table | **PROMOTE core results** to Figure 3/new Table 2; keep all two-C core historical three-seed effects in supplement, including negative ones. |
| Tau sensitivity / compute benchmark tables | **KEEP as bounded supplement diagnostics**; post-hoc scoring and model-only randomly initialized CPU timing, respectively. |
| General-collateral main/appendix tables | **REMOVE from AAMAS revision**; different assumptions and unnecessary scope. |
| Dormant dense `table2_k24_main_comparison.tex` and duplicate scaling tables | **DO NOT reactivate** as additional evidence. They duplicate existing source layers. |

**Final Table 1 — policy and protocol summary:** JA-PPO / IFAC / SC-FAC / zero-control / feedback rule; action factorization, output count, key dimensions or rule parameter, training status. Common C,k,F,T, reward and validation selection go in a short caption/setup paragraph rather than repeated columns.

**Final Table 2 — fresh confirmation at two capacities:** one row per method and C; Money mean ± training-seed SD for learned methods, fixed-pool mean for the rule, and paired difference from SC-FAC with an explicitly signed 95% CI. Use two small C panels if needed. Keep primary-metric contrasts in the main text; flush/acceptance/drop detail goes to supplement. Figure 3 supplies paired seed detail for the mechanism contrast rather than duplicating all table values.

## Eight-page section-level rewrite

Page allocations are approximate and include figures/tables; references are separate. The total below is 8 pages. This is a rewrite plan, not permission to edit source now.

| Section | Pages | KEEP from ICDM | REWRITE | ADD | REMOVE |
|---|---:|---|---|---|---|
| 1. Introduction | 1.0 | Resource-limited online settlement motivation and coupled actions. | Lead with autonomous sequential decisions and delayed availability; bound the novelty. | Three precise contributions; one sentence acknowledging strong-rule boundary. | Broad network/deployment promises and extension roadmap. |
| 2. Related Work | 0.6 | Relevant collateral-control and RL citations after verification. | Organize by structured/autoregressive actions and sequential resource control. | Clearly distinguish existing factorization ideas from this empirical instantiation. | Unrelated idea5/two-pool material and unsupported novelty assertions. |
| 3. Problem Formulation | 1.0 | Original state, action, C/k, F, T, Money definitions. | Separate policy sampling from simulator execution; correct flag/time/fee prose. | One explicit transition-order example and distinction between shaped reward and evaluation objective. | Any implication of future information or multiple agents. |
| 4. Method | 1.2 | JA/IF/SC factorization equations, SC E32/H256 and original PPO. | Compact comparison around Figure 1; explain what conditional input can represent without claiming necessity. | Existing zero-input control and head-capacity caveat. | Repeated overview/architecture descriptions and new model proposals. |
| 5. Experimental Setup | 1.0 | Original train/validation/test protocol and rule constraints. | Make 1,000 consumed training episodes, checkpoint selection, seeds, and metric aggregation explicit. | OLD12/NEW12 distinction, frozen feedback rule, pairing/multiplicity, recovery manifest. | Mixing three-seed auxiliaries into ten-seed claims. |
| 6. Results and Analysis | 2.3 | Historical two-C core learned-policy comparison. | Three questions: compact-policy performance, conditioning effect, strong-rule comparison. | Figures 2–3, new Table 2, all unfavorable outcomes and effect-size interpretation. | General-collateral extension and broad robustness claims. |
| 7. Discussion / Limitations | 0.7 | Synthetic benchmark and fixed-environment limitations. | Explain rule strength, head/selection confounds, limited seed precision, and known-family scope. | Why compact representation is distinct from latency or practical superiority. | Speculative deployment claims and large future architecture menu. |
| 8. Conclusion | 0.2 | Original problem and main method. | One bounded contribution and result statement. | A clear boundary on conditioning and heuristics. | Any conclusion that exceeds the tested settings. |

## Official submission check — verified 2026-09-30

**Schedule and fit.** OpenReview author registration: September 17, 2026; abstract: October 1 AoE; paper: October 8 AoE. Beijing equivalents are October 2 and October 9 at 19:59:59. The account deadline has passed; account readiness must be checked immediately. LEARN explicitly includes single-agent learning and is the recommended area. Camera-ready: January 25, 2027; conference: May 3–7. Findings consideration is automatic unless opted out. [Official main-track call](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/).

**Submission rules.** Use the official mandatory LaTeX template, English PDF, double-blind review, at most eight content pages plus references, and a 100–300-word plain-text abstract. Do not alter layout settings. Supplement: one anonymous ZIP, ≤25MB; essential evidence stays in the paper. Accepted supplements need an archival public version. Author order cannot change after acceptance. Concurrent substantially similar archival submissions and thin slicing are prohibited. AI-assisted experimental design requires disclosure of prompts, tool and version; generated illustrations are restricted to research about generative AI. [Official instructions and template download](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/).

**Practical obligations.** Nominate a qualified reciprocal reviewer or declare exemption by October 1 AoE; at least one nominee must accept the OpenReview reviewer invitation by October 8 AoE. Assigned reviews and post-rebuttal acknowledgment must be completed. Appendices inside the paper count toward eight pages. For missing historical AI records, disclose the gap rather than reconstructing prompts; retain subsequent records. [Official FAQ](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/).

Exemption applies when no author qualifies, or all qualified authors serve in specified senior/organizing roles; nomination alone is insufficient. [Reciprocal-reviewer policy](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/reciprocal-reviewer-policy/). Eligibility is a relevant PhD, or third-year-or-later PhD status with at least three relevant peer-reviewed publications. Reviews are due November 13, and rebuttal acknowledgment by December 3. The reviewer page has a looser appendix wording; follow the explicit, newer FAQ clarification above. [Reviewer guidelines](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/reviewer-guidelines/).

**Application to this project — INFERENCE, not organizer clearance:** A completed rejection at ICDM is not itself an ongoing parallel submission or an archival publication. Resubmission is therefore not barred on that basis. Remaining checks: confirm no appeal/other active substantially overlapping submission, and agree a non-overlapping or sequential submission plan for the ICASSP track. Separate branches alone do not establish distinct scientific contributions. An unmerged PR is not itself a submitted paper, so do not assert a violation merely because PR #3 exists.

This planning session contributes experimental design: preserve the user brief/conversation and accurately disclose the Codex tool/model information available at submission. Do not invent a precise model build if unavailable. Use data-driven plots and authored vector diagrams, with human verification, rather than generated artwork. This follows the project’s actual methodology use, not a generic disclaimer.

The official template link was identified; template package contents were not installed or compiled in this task. Final formatting must be checked in the actual official template. Accepted-paper registration fees/payment deadlines and presentation obligations were not established from the inspected main-track sources; check the official registration instructions when published. Do not substitute another track's dates.

## Rescheduled work sequence — experiment hold remains active

The September 30 schedule is superseded. October 1 priorities are author/abstract/reviewer obligations and local recovery handoff; do not interpret deadline pressure as experiment authorization. Official policy dates remain those verified September 30 in the checklist above.

| Step | Deliverable | Authorization / gate |
|---|---|---|
| 1 — current | Four planning files plus exact recovery manifest; PI confirms C800/1200 and source identity. | Local document update only; no binaries transferred. |
| 2 — next Trae task | Review the manifest, map intended destination paths, and define checksum/replay checks. | No training, evaluation, pool generation, or transfer command execution. |
| 3 — after explicit release | Copy only approved artifacts and validate hashes; replay JA-PPO C800/seed123 on original US. | Halt on any mismatch; report replay before broad execution. |
| 4 — after replay approval | Complete four zero-control jobs and the fixed 40 learned / two rule evaluations on approved AAMAS streams. | No additional seeds or architecture changes. |
| 5 | Statistics, claim decisions, core figures/tables and revised abstract. | Include negative/inconclusive findings; no threshold selection. |
| 6 | Eight-page source revision, anonymity, reproducibility and rendering checks. | Separate paper-edit authorization; preserve original method. |
| 7 | Author review and submission. | Remove placeholders; verify reviewer acceptance and submission readiness. |

If time no longer permits this sequence, the PI should choose a historical-evidence submission or defer; do not represent planned results as completed. Do not automatically execute a larger campaign because the server lacks the local archive.

## Exact next task for Trae — RECOVERY HANDOFF ONLY, draft not sent

Read the updated AAMAS_ICDM_AUDIT.md inventory and AAMAS_EXPERIMENT_MATRIX.md. Prepare a recovery handoff manifest for the original ICDM core C800/1200: 30 principal **best** checkpoints (JA-PPO/IFAC/SC-FAC × seeds123/323/532/777/999), six zero-conditioning best checkpoints (seeds123/323/532 at both C), their exact run_info/config/result records, and the fourteen historical pools. Resolve each checkpoint by its recorded timestamp, never newest-file selection. Use the recorded SHA-256 values to define source/destination verification. The minimal checkpoint/pool payload is 41,522,056 bytes before metadata.

Identify the destination paths and the precise first replay specification: JA-PPO C800/k24/F3/T1000/seed123, timestamp20260508_222529_095916, deterministic argmax on the original US pool, compared with that run's saved US metrics. Report the ready-to-transfer manifest and any incompatible software/configuration assumptions. **Stop for approval. Do not execute transfer commands, replay, training, evaluation, NEW12 generation, or baseline implementation in this task.**

Do not merge/change PR #3 or use its code/numbers. Do not place checkpoints, pools, or generated results in the planning-file commit. The only candidate Git files are the four `research/aamas2027/AAMAS_*.md` documents. Do not commit or push them without the user's next instruction. Once recovery/replay is explicitly released and passes, the separate future experiment budget is four zero-control jobs at (800,777),(800,999),(1200,777),(1200,999); this handoff does not authorize those jobs.

## PI decisions required before execution/submission

1. Confirm the exact rejected source, or explicitly accept the documented provisional source identity.
2. Accept the representation-focused story and transparent stronger-rule result; conditioning stays a bounded mechanism question.
3. Confirm the recovery handoff first; separately approve replay and then the four-run budget/fixed evaluation matrix. No execution is authorized now.
4. Confirm all-author OpenReview readiness, reciprocal reviewer/exemption, authorship, and ICASSP submission separation. Decide Findings opt-out only if desired; the recommended default is to retain consideration.

No paper rewrite, figure production, experiment/replay execution, artifact transfer, reviewer nomination, abstract submission, or message to Trae was performed by this update.
