# SC-FAC next-generation model design

**Design only · PI review · 2026-10-03**

Target: `sandbox/aamas2027_explore/scfac_candidates/MODEL_DESIGN.md` in `/Users/zhouzhou/Desktop/kwallet-aamas-explore`, branch `aamas2027-explore-sandbox`, evidence commit `281e02920a32e9878177593d62c7f2d841ed6078`.

**Recommendation:** start with an inventory-aware SC-FAC that uses no BF actions or code. Compare it with a separate SC-FAC using a decaying, training-only BF prior. Preserve learned settlement and conditional learned flushing in both. Do not revive the failed masking-led BF-settlement branch or combine several changes into an opaque successor.

The proposed initial screen is **12 full-budget training runs**, including a shared vanilla SC-FAC control selected by Money. None is authorized or executed by this document. No experiment code, existing plan, frozen result, or paper is changed. FACT denotes audited evidence; INFERENCE denotes an interpretation; PROPOSAL denotes an untested design. Existing sandbox handoff instructions are reference material, not execution instructions for this task.

## 1. Current evidence summary

### What is established

**FACT:** Stage 3 uses five training seeds, two capacities and twelve fixed fresh evaluation regimes. SC-FAC's macro12 Money means are 3954.36/14372.98 at C800/C1200, versus deterministic BF-T0.5 at 4661.32/15298.89. Factorized methods have favorable descriptive means against JA-PPO, but no pre-specified structural contrast passes Holm correction. Full SC-FAC has no detected incremental benefit over its zero-conditioning control. BF beats SC-FAC in both corrected comparisons. BF is one fixed controller per capacity, not five seeded observations.

**FACT:** H1 supplies BF settlements to frozen SC conditional flush heads. Its means are 4012.88/13873.67; the gains over SC are +58.52/−499.32, with broad exploratory intervals. This is not robust evidence for composing BF settlement with a pretrained flush head.

**FACT:** D1 masking removes executed same-wallet conflicts, but improves mean H1 Money by only **+1.3505 at C800 and +13.3151 at C1200**, or 0.029%/0.087% of BF. D1's declared mask gate fails. Its own mismatch classification is WEAK, not proof of a particular distribution-shift mechanism. The raw C800 p=.0272 is exploratory and does not make the tiny effect strategically important.

| Controller | C | Flushes/episode | Insufficient drops/episode |
|---|---:|---:|---:|
| Vanilla SC | 800 | 221.919 | 40.090 |
| H1 unmasked | 800 | 218.075 | 39.534 |
| BF | 800 | 257.801 | 0 |
| Vanilla SC | 1200 | 604.180 | 7.671 |
| H1 unmasked | 1200 | 524.038 | 44.486 |
| BF | 1200 | 541.610 | 0 |

The important objective is effective replenishment, **not indiscriminately more flushing**. Vanilla SC already executes about 62.57 more flushes/episode than BF at C1200 and still drops more feasible transactions.

Two H1 examples sharpen this interpretation:

- C800/777: vanilla SC flushes 224.324 with 35.593 insufficient drops; H1 flushes 148.268 with 86.365 insufficient drops. H1 has zero direct conflicts, so the D1 mask cannot repair this failure.
- C1200/323: vanilla SC flushes 608.585 with 6.058 insufficient drops; H1 flushes 343.519 with 170.833 insufficient drops. D1 masking recovers only 3.416 Money.

**Scope correction:** D1's starvation counters describe H1 U/M trajectories with BF settlement, not vanilla SC trajectories. Their non-oversize zero-feasible counts average 94589/104704 per 2.4-million-step cell at C800/C1200; BF has zero on the inspected streams. Vanilla SC's insufficient-drop records support an inventory concern but do not establish its exact pre-action starvation frequency. H2 already excludes its chosen settlement wallet from BF flushing; H2's failure is further evidence that exclusion alone is insufficient.

### Provenance and artifact qualifications

Audited sources are the committed Stage 3 `statistics.json` and regime scores; B0/H1 score CSVs and per-cell outputs; D1 `D1_SUMMARY.csv`, `statistics.json`, `D1_CONFLICT_TELEMETRY.csv`, `D1_VALIDATION.json`, and all 22 cell-level result/episode/telemetry sets. D1 means agree with the cell outputs, and its ALL12 action/conflict counters agree with telemetry JSON. No evaluator was imported or run for this review. Verbose Trae reasoning and uncommitted sandbox interpretations were not used as scientific ground truth.

Three audit cautions must accompany future reuse:

1. `D1_SEED_CAPACITY.csv` has **20 rows for ten capacity/seed keys**. Its first block incorrectly copies U telemetry into M fields; the later block matches the actual M telemetry. Do not aggregate this table or silently deduplicate it by arbitrary row order. Use the reconciled per-cell sources. This is a reporting defect, not evidence of twenty independent runs; frozen artifacts remain untouched.
2. D1's refill tracker retains only the last cycle per wallet when reporting never-reused refills at episode close. That field is **not an all-cycle unnecessary-flush count**. Future telemetry must close each replaced cycle and distinguish terminal censoring before making such claims.
3. A mask changes an action only when the excluded logit would win **at that same visited state**. This does not mean two diverged U/M trajectories take identical actions at every other time step.

Evidence anchors: D1 `statistics.json` SHA256 `f5d6642ef08ae997e14e592b98af7cae671672e5fd9dc03b0b74e619c9ff1ce5`; D1 summary SHA256 `dc576d4592cc15721ea666a2024fc100c565e50493687009e8a9ad5d9aa28960`; Stage 3 `statistics.json` SHA256 `ee3961d54fa4fb55015542ab5f7f5b49eff37d7bebe61195cf1ec5fd91dc80e0`. The frozen raw tarball remains bound to `576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`.

## 2. Diagnosed SC-FAC failure mechanisms

| Hypothesis | Evidence strength and precise limitation | Discriminating experiment |
|---|---|---|
| Inventory information is difficult to extract from a flat vector | Plausible: raw balances/timers exist, but global service capacity and refill pipeline are implicit. No representation bottleneck has been demonstrated | Explicit physical summaries and selected-wallet features, with the same action support and PPO objective |
| Flush frequency and target allocation interfere in optimization | Plausible: under-flushing in H1 and overspending in native SC differ by setting. Does not establish a softmax bottleneck | Reparameterize flush into event probability and conditional target, without extra parameters or decoder changes |
| Training fails to associate choices with future shortages | Plausible, not proven. Return is dense; F=3 is short | A training-only future-availability predictor, judged by control outcomes rather than prediction accuracy alone |
| A competent replenishment behavior is hard to discover consistently | Plausible: BF supplies a strong example, and H1 shows pretrained heads can be brittle off their training distribution | Decaying BF supervision on native SC rollouts, followed by entirely rule-free PPO |
| Checkpoint selection wastes attainable Money | The metric mismatch is real; its size is unknown because only historical selected BEST models survive | Compare Money/value-acceptance/return selections from the same new checkpoint bank on independent development streams |
| Direct same-wallet collision dominates | D1 argues against this as a major explanation of the H1 deficit | Do not repeat the mask experiment or make masking the main successor |
| The conditional policy is inherently too weak | Unsupported: an unrestricted autoregressive factorization can represent an arbitrary joint distribution; finite capacity/optimization are distinct issues | Do not solve an unestablished expressivity problem by increasing network width |

**Credit-assignment qualification:** the frozen settings use gamma=.98, GAE lambda=.95 and F=3. Gamma cubed is .941; `(gamma*lambda)^3` is about .807. Immediate refill consequences are not simply beyond the discount horizon. Longer-run depletion, value approximation and attribution remain plausible, but neither increasing gamma nor adding a long-horizon model is justified as the first change.

**Resource definition:** “usable” means unfrozen, not sufficiently funded. With k24, F3 and at most one flush per step, at most two wallets remain frozen at the next decision; all-wallets-frozen starvation is largely structurally excluded. Measure **covering balance for non-oversize requests**, liquid value, and replenishment timing rather than treating zero usable wallets as the important endpoint. BF's observed zero starvation is not a theorem for every admissible stream.

**Reward qualification:** with shaping disabled, the undiscounted episode sum obeys `Money = 1000*R + 20*drops`. The reward is close in form to Money but includes an additional drop penalty; PPO also discounts the return. This algebra does not prove the objectives have identical optima or quantify reward-Money correlation. Keep reward fixed initially and diagnose selection first.

## 3. Design principles

1. Keep both settlement and flush learned, with `pi(a_s,a_f|x)=pi_s(a_s|x)*pi_f(a_f|x,a_s)`. An inference-time BF settlement rule would change the central method, not merely improve SC-FAC.
2. Test one mechanism per candidate. Do not initially combine inventory features, a gate, an auxiliary loss and imitation.
3. Preserve E0, action ordering, reward, capacity/horizon settings, budgets and data. No forced flushes, flush floors/caps, post-step repair, or extra action masks in the initial candidates.
4. Money selects every new learned checkpoint. Preserve alternative selections from the same bank to isolate metric effects.
5. Prefer physical, causally available information. Do not supply the future transaction sequence, future burst label, regime identity, final-pool statistics, or privileged oracle values to the actor.
6. Treat BF as a behavioral reference, not an optimality certificate. A learned improvement must preserve settlement while saving sufficient costs or improve resource use on fresh streams.
7. Retaining learned decisions preserves the *capacity* to adapt; it does not demonstrate OOD robustness, switching adaptation or superiority to BF. Those claims require separate evidence.

### Common architecture and optimization contract

The audited SC configuration is: state width 74, encoder `74→128→128` with ReLU; settlement head `128→25`; selected-index embedding `25×32`; conditional flush head `160→256→25` with ReLU; critic `128→1`. Total **77907 parameters**, including biases. Use the recovered condH256 configuration, not the source file's lower default width.

All candidates use the same clipped joint-log-probability PPO loss, value coefficient .5, entropy coefficient linearly .03→.003 over training, gamma .98, GAE .95, clip .2, Adam lr3e-4, four update epochs, minibatch 256, max gradient norm 1, 1000 episodes of 1000 steps. No additional environment interactions or BF rollouts are hidden in a candidate's budget. Keep the unshaped original reward. Same run-time/device/thread lock and seed handling apply to all comparisons. Exact optimizer/default initialization details must be serialized from the frozen configuration before implementation; do not opportunistically select different settings per model.

For all models, choose `best_by_money` on a common validation panel at the same scheduled checkpoints; no early stopping. Also retain `best_by_value_accept_ratio`, `best_by_training_return`, the complete checkpoint bank, and final checkpoint. Section 8 fixes how these are computed. None of the historical evidence implies a currently measured selection gain.

## 4. Candidate models

### C1 — Inventory-aware SC-FAC (IA-SC): primary

**A. Hypothesis.** SC-FAC can represent resource maintenance but must learn too much arithmetic and index-to-wallet association from a flat observation. Explicit inventory and selected-wallet semantics should improve learnability without limiting its actions.

**B/C. Relationship to SC/BF.** Both learned heads, their conditional order, PPO and action support remain. **BF relationship: NONE.** The additional quantities come from wallet balances, current demand and deterministic refill timing; they do not call BF, rank wallets by BF, encode its .5 threshold or supply BF actions. The hypothesis was motivated partly by BF's behavior, but the method has no BF dependency.

**D. Exact representation.** Let `w=C/k`, current demand v, balances b_i, usable flags u_i, and feasible set `V={i:u_i=1,b_i>=v}`. Append nine features to the original observation at F3:

`q(x)=(sum u_i/k, |V|/k, sum u_i*b_i/C, max_(i:u_i=1)b_i/w, v/w, 1[v>w], p_1,p_2,p_3)`.

Empty-set maximum is zero. `p_l` is the fraction of wallets already scheduled to refill in l decision steps. Derive it from the observed frozen flag and timer, not the future stream. In this E0, a frozen wallet's remaining delay is `F*observed_normalized_remaining+1`; a frozen zero timer still means one step remains. Validate integer reconstruction and the flag distinction against E0 boundary fixtures. If F changes, this architecture's feature specification changes; do not claim zero-shot variable-F support.

The encoder becomes `83→128→128`. Settlement remains `128→25`. Preserve the selected-index embedding 32 and append four physical selected-wallet features to the flush head:

`c(x,a_s)=(u_s, b_s/w, 1[s in V], 1[s in V]*(b_s-v)/w)`.

For settlement no-op, all four are zero. The fourth feature is hypothetical residual balance **if settlement succeeds**; E0 is still flush-first, and no action is executed while forming it. The flush head becomes `164→256→25`; critic remains `128→1`. No gate, auxiliary head, imitation or mask.

**E. Policy.**

`pi_IA(a_s,a_f|x)=pi_s(a_s|x,q(x))*pi_f(a_f|x,q(x),e(a_s),c(x,a_s))`.

Deterministic evaluation is settlement argmax, then conditional flush argmax, with original index ties. Policy output remains 2(k+1)=50 logits at k24; the extra features do not imply linear total RL complexity.

**F/G. Training and selection.** Common PPO and reward; no extra coefficient. Initialize the existing weight blocks exactly as the paired vanilla control and initialize added feature columns to zero, so initial functions match before learning. Money-based selection and all three retained selection labels apply.

**H. Expected signature.** Better covering-balance distribution and fewer insufficient drops at C800; fewer excess or poorly timed flushes without lost settlement at C1200; less seed variability. More flushes everywhere is not the predicted success signature.

**I. Falsification.** If features change diagnostics but cannot materially improve Money beyond the retrained Money-selected control, or gains depend on a single seed/regime, reject the architectural claim. If only the validation metric explains improvement, use the simpler control.

**J. Overhead.** Added weights: `9*128+4*256=2176`; total **80083**, +2.79%. Preserving the frozen two-encoder-call action path adds about 3328 multiply-accumulates per decision, plus O(k) feature calculations. Caching the encoder once would reduce the increment to2176, but any caching change must be applied equally to all new models. Wall-time overhead is unmeasured; a batched implementation should be small, while Python feature extraction could dominate. Estimate 1–2 engineering days including timer and initial-function tests.

**K. Paper value and ablations.** A small, interpretable SC-FAC extension testing resource-state representation rather than a larger network. It can still look like feature engineering; the defensible contribution is which conditional physical information changes learning, with evidence. After a win, require global-only q and selected-wallet-only c ablations. Do not initially train their full factorial. These summaries are permutation-invariant quantities, but the overall flattened/index-embedding network is **not permutation-equivariant**.

### C2 — Event/target SC-FAC (ET-SC): architectural reserve

**A. Hypothesis.** A single flush softmax entangles “replenish now?” with which wallet receives replenishment, making aggregate timing difficult to optimize.

**B/C. Relationship.** Keep learned settlement, original state/embedding and conditional hidden width 256. Replace only the last flush layer. **BF relationship: NONE.**

**D/E. Architecture and factorization.** One Bernoulli gate logit gives `g(x,a_s)`; 24 target logits give `r(j|x,a_s)`. No new masks or heuristic eligibility thresholds:

`pi(a_s,a_f|x)=pi_s(a_s|x)*[(1-g) if a_f=k; g*r(a_f|x,a_s) otherwise]`.

For sampling, sample whether to flush and, only on a flush event, its target. Gate and target use the same original conditional hidden vector. Keep original support, including choices E0 ignores, so masking is not a hidden treatment.

This has **exactly the expressive power of a categorical flush distribution**, not a richer policy family: `g=1-p_noop` and `r_j=p_j/g`. It tests optimization and parameterization. Gate+target output counts 1+24 equal the original 25; total parameters remain **77907**.

**F/G. Loss/selection.** Common joint PPO. When no flush occurs, target log-probability contributes nothing. Flush entropy is `H(Bernoulli(g))+g*H(r)`, not an unweighted sum. Money selection/retention as above. A naive zero gate bias starts with 50% no-flush, unlike a near-uniform 25-way policy's 4%; control this exploration confound. Use common zero final-output weights, zero target biases and gate bias `log(24)`; include a vanilla zero-output initialization control if this reserve is tested. This is separate from the default primary/secondary initialization contract.

**Deterministic semantics:** reconstruct `[g*r_0,…,g*r_23,1-g]` and take argmax. A .5 gate threshold followed by a target argmax is a different decoder; e.g., g=.6 and largest target probability .1 gives a largest flush probability .06 versus no-flush .4. Do not credit a decoder change as an architectural gain. A later decoder intervention would need a separately labeled matched evaluation.

**H/I. Prediction/falsification.** Better scheduling without extra flush costs or starvation, visible in the learned event probability and target allocation. Reject if it only shifts overall frequency, or improvement disappears under matched initialization/decoder controls. The D1 data do not establish this as the actual bottleneck.

**J/K. Cost/paper value.** Same parameter count, negligible arithmetic increment for probability composition, approximately 1–2 engineering days for probability/entropy/decoder tests. A clean extension of structured factorization, but a reparameterization win needs particularly careful controls. Mandatory controls: matched initialization, correct common decoder, and vanilla 25-way output; no BF-related ablation is relevant. Reserve rather than the first funded comparison.

### C3 — Availability-supervised SC-FAC (AS-SC): learning-signal reserve

**A. Hypothesis.** Shared features do not readily encode the consequences of joint choices for subsequent service capacity; self-supervision on shortages could improve representation.

**B/C. Relationship.** Keep both original actor heads and all inference decisions. **BF relationship: STRUCTURAL INSPIRATION ONLY.** Healthy availability is the diagnostic motivation; labels come solely from the learner's own trajectory, with no BF calls, threshold, actions or target flush count.

**D/E. Architecture.** Add a training-only sigmoid predictor `185→64→1` (ReLU hidden) on `[h(x_t),e(a_s,t),onehot(a_f,t)]`, of widths 128+32+25. Actor factorization is unchanged SC-FAC. No input features from the future, no action mask, no policy override; the head can be removed at deployment.

**F. Objective.** For H=2F=6, target

`y_t=(1/H)*sum_(l=1..H) 1[v_(t+l)<=w AND no usable wallet can cover v_(t+l)]`.

Use realized on-policy states; exclude labels whose six future decision states are not all inside the episode. Stop gradients through targets. Add `lambda_aux*MSE(yhat,y)` to common PPO, nominal `lambda_aux=.05`, fixed before any screen. No coefficient grid is funded. Use an unweighted loss, so no unreported class-balancing hyperparameter is introduced.

This is **representation supervision**, not a causal action-value estimate or a proof of repaired credit assignment. Future actions and requests also affect the label. Prediction improvement alone is not a success. Availability need not be maximized at unlimited cost.

**G/H/I. Selection and expected effect.** Money selection remains primary. Expect fewer shortage bursts at similar/lower flush cost and more stable Money; do not select by auxiliary loss. Reject if only Brier/MSE improves, Money does not materially improve, or the network learns to predict unavoidable/near-terminal shortage without changing useful behavior.

**J/K. Cost and controls.** Extra parameters `185*64+64+64+1=11969`, total **89876 during training** (+15.36%), original **77907 at inference** after removing the head. One short target computation and a small MLP per PPO sample; estimate 5–20% neural-update overhead, not measured whole-run overhead. About 2–3 engineering days. Mandatory lambda=0 control and, if it wins, a detached-actor-feature auxiliary control to distinguish shared-representation benefit from bookkeeping. Existing source includes inactive auxiliary-risk branches in other legacy modes; generic auxiliary prediction is not a new invention, and those branches should not be silently activated or imported as the candidate.

### C4 — SC-FAC with a decaying BF training prior (TP-SC): secondary

**A. Hypothesis.** The principal limitation is discovering and retaining competent resource maintenance, rather than representation or policy expressivity. Early rule-derived supervision may guide the actor toward competent behavior, after which PPO can improve it.

**B/C. Relationship.** The complete native SC-FAC policy is trained and deployed; both decisions remain learned. **BF relationship: TRAINING-TIME TEACHER / PRIOR.** This is an explicit transfer of BF knowledge, not BF-free learning. It is not a runtime hybrid.

**D/E. Architecture and labels.** Same original 77907-parameter architecture and factorization. During native SC rollout at pre-action state x, store:

- `b(x)`: BF minimum-balance feasible settlement, index tie-breaking, no-op when none;
- `q_BF(x,a_s)`: BF's lowest-balance usable wallet below `.5*C/k`, excluding the **learner's sampled settlement a_s**, or no-op.

All executed rollout actions are sampled from native SC, never from BF. The flush teaching target is evaluated under the learner's actual condition, preventing a training loss that sees only BF-selected settlement conditions. This is deliberately a **BF-derived conditional teacher**, not the joint full-BF policy unless a_s=b(x). H2 demonstrates that these conditional targets are not guaranteed optimal under arbitrary learner settlements; treat that as a real risk, not an oracle label.

The deployment policy stays `pi_s(a_s|x)*pi_f(a_f|x,a_s)` with the original deterministic decoder. Remove BF helper dependencies from the deployment entry point; BF may still be evaluated separately as a comparator.

**F. Objective.** Add

`L_teacher=mean[-log pi_s(b(x)|x) - log pi_f(q_BF(x,a_s)|x,a_s)]`

to common PPO with `lambda_BF(e)=.01*max(1-e/250,0)`, for zero-based training episode e. The last 750 episodes are entirely PPO. Store labels with the rollout and reuse them during its four update epochs. No teacher trajectories, BC pretraining stage or extra environment interactions. This loss is cross-entropy/forward teacher-to-policy KL; do not use `KL(pi || delta_BF)`, which is infinite for off-teacher mass. Lambda=0 is the shared vanilla control; no coefficient sweep is part of the initial 12 runs.

**G. Checkpoint selection.** Money selects checkpoints over the full run, with standard three-selection retention. Also retain the episode 250 prior-removal-boundary checkpoint (its last update still used the positive episode 249 coefficient) and the best Money checkpoint from episodes 300–1000 as diagnostics, not substitute primary selections. If the selected model predates prior removal, do not claim late independent refinement. A fresh held-out challenger must pass the development gate with a preselected prior-free-period diagnostic checkpoint too; do not silently swap to it after testing the primary selection.

**H/I. Expected behavior/falsification.** Earlier competence, fewer severe shortage trajectories, and sustained Money improvement after the prior disappears. Disagreement with BF is descriptive, not itself evidence of better adaptation. Reject the beyond-BF claim if it merely clones BF, loses competence once the prior decays, or requires a live rule to prevent collapse. Retaining BF-level behavior can be useful distillation, but does not meet this campaign's intended improvement threshold.

**J. Overhead.** Zero additional trainable parameters; BF-derived labels cost O(k) during training. Both actor distributions are already computed for PPO at the stored settlement; labels change the supervised log-probabilities, so no new deployment network is needed. Extra training wall time is unmeasured; provision 5–20% for labeling/loss work. Inference cost is unchanged. Approximately 1–2 engineering days plus teacher-fidelity and rule-free-deployment tests.

**K. Paper value and mandatory controls.** A legitimate SC-FAC-centered knowledge-transfer hypothesis, with more narrative risk than C1: a reviewer may reasonably ask whether it only learns BF. Require the lambda=0 control, a finite imitation-only control, and an always-on-teacher control if C4 wins. The imitation-only control uses the same finite teacher-loss schedule and 250,000-state teacher-label budget, but no PPO policy, value or entropy losses: its critic is unused/frozen, avoiding indirect value-loss updates to the shared actor encoder. Its actor freezes after episode 250 while the declared interaction budget is still accounted for. The always-on control uses PPO plus teacher weight .01 throughout 1000 episodes; disclose its larger 1M-state label budget. These are later ablations, not extra initial runs. Report label budget, prior schedule, prior-removal-boundary and later performance, and BF agreement. The convincing story would be useful finite supervision followed by better learned control on fresh streams; imitation alone is insufficient.

## 5. Should SC-FAC explicitly learn from BF?

The observations justify testing whether BF provides useful competence, but they do not show that copying BF is necessary. The clean order is **resource representation first; a bounded training prior as an independent comparison; no inference-time BF component**.

| Level of transfer | Legitimacy / interpretability | Can it exceed BF or remain flexible? | Cost and risk to the paper |
|---|---|---|---|
| Physical summaries / refill pipeline | Directly interpretable E0 quantities; no rule assumption | Both learned heads unrestricted; possible but unproved improvement | Low cost; cleanest SC extension; may look like feature engineering without ablations |
| BF-specific features: rankings, threshold margin, BF action IDs | Legitimate domain assistance if disclosed | Actor can disagree; these are still runtime BF-derived computations | Low–medium cost; effectively embedding the rule, easily overstated as BF-free |
| BF-derived candidate support | Removes actions the rule deems implausible | Can improve only within retained support; no general optimality guarantee | Medium; confounds support quality with RL and drifts toward rule engineering |
| BC warm start | Clear training-only transfer | Can exceed teacher after RL; forgetting and occupancy shift possible | Medium; demonstration collection and extra budget must be counted; not first choice |
| Auxiliary prediction of BF decisions | Encourages a BF-relevant representation without necessarily matching actor actions | Flexible, but representation prediction may not produce competent control | Medium; weaker mechanistic link than direct finite supervision; not another initial candidate |
| Decaying BF actor regularization | Explicit bounded prior, as C4 | Policy remains unrestricted and can depart during later PPO | Low–medium; can become teacher mimicry; prior-off and teacher-only controls essential |
| Residual deviation from BF | Transparent combination if honestly named | Some flexibility, but the base action remains the rule | Medium–high; scientifically a BF-based controller, not a small SC-FAC extension |
| BF settlement, emergency override or flush-floor rule at inference | Directly interpretable operational heuristic | Learned subdecision may adapt, but rule constrains behavior and can introduce H1-like incompatibility | Would weaken the desired SC-centered claim; D1 does not support reviving the previous mask-led A+ branch |

**BF_INTEGRATION_RECOMMENDATION: LIGHT.** Here LIGHT means physical insight plus one finite, explicitly labeled training-prior candidate. It does **not** mean that C4 contains only a little BF information: its targets encode the full relevant rule semantics during the first quarter of training. Neither recommended deployed actor calls BF, inherits its fixed settlement, uses its ranked support, or receives its action as an input.

There is a stronger reason to test C4 than to impose BF at inference: it asks whether learned SC can internalize competent maintenance while preserving its own two learned decisions. It remains a hypothesis, not an endorsement of the PI's intuition as an established result. If the BF-free candidate succeeds equally well, prefer it. If the teacher is required to match but not exceed BF, label the result competence transfer rather than a new superior controller.

Relevant primary literature establishes these ingredients as prior art: [Deep Sets](https://arxiv.org/abs/1703.06114) motivates explicit symmetry-aware aggregation, but C1 itself is not a Deep Sets architecture; [Jaderberg et al.](https://arxiv.org/abs/1611.05397) motivates shared auxiliary training signals, not evidence for our proposed shortage target; [Hester et al.](https://arxiv.org/abs/1704.03732) shows demonstration-based learning as a known approach, not a result about PPO or K-Wallet. These are targeted references, not a completed novelty survey. Do not claim novelty merely for adding a gate, an auxiliary loss, or imitation.

## 6. Candidate comparison table

| Candidate | Distinct hypothesis | BF involvement | Parameters: train / deployed | Initial extra knobs | Recommended status |
|---|---|---|---:|---|---|
| IA-SC | Explicit global and selected-wallet resource semantics improve learning | NONE | 80083 / 80083 | None | PRIMARY |
| ET-SC | Event frequency/target parameterization improves optimization | NONE | 77907 / 77907 | Matched output initialization/decoder controls | Reserve; same expressivity |
| AS-SC | Future-availability prediction improves shared representations | STRUCTURAL INSPIRATION ONLY | 89876 / 77907 | One auxiliary weight, H fixed 6 | Reserve; prediction is not causal control |
| TP-SC | Finite competent-behavior supervision improves exploration/stability | TRAINING-TIME TEACHER / PRIOR | 77907 / 77907 | One fixed prior weight/schedule | SECONDARY |
| Vanilla SC-M | Selection-only/common-training control | NONE | 77907 / 77907 | None | Mandatory shared control |

No initial candidate has an inference-time BF component. Parameters are analytic counts for k24/F3/condH256; implementation must verify actual counts. Wall time is not inferred from parameter count alone.

## 7. Recommended top 2 candidates

**Primary: IA-SC.** It is the smallest clean test of the most plausible representational weakness, introduces only 2176 weights, keeps SC's action structure, and needs no new training objective. Its semantic conditioning features specifically preserve the link between the learned settlement choice and learned resource maintenance.

**Secondary: TP-SC.** It tests a meaningfully different hypothesis: whether a competent behavioral prior is needed even with the same original architecture. BF's strength makes this a credible test; the prior's finite duration and removal at deployment prevent it from becoming a disguised runtime BF hybrid. Its paper claim must remain conditional on surpassing BF and on prior-off evidence.

Do not combine IA and TP during this screen. If both improve, choose one using Section 11's predeclared rule. A combined model would require a new factorial design; it is not a free “best of both” next step. ET/AS are serious alternatives but are not automatic rescue experiments if the top two fail.

## 8. Recommended control experiments

### Same-bank checkpoint-selection control

Retrain vanilla SC-FAC under the shared protocol. At episodes 50,100,…,1000 save all 20 checkpoints and their deterministic validation scores. Designate:

- `best_by_money`: highest equal-weight mean of per-regime mean Money on VAL-v3;
- `best_by_value_accept_ratio`: highest equal-weight mean of per-regime mean episode value-accept ratios on the same VAL-v3;
- `best_by_training_return`: checkpoint with highest mean **undiscounted original on-policy episode return over the preceding 50 training episodes**, including the checkpoint episode;
- ties: earliest training episode; retain the final checkpoint regardless.

Training-return selection is a noisy diagnostic, not the preferred criterion. No validation or development output feeds back into optimizer settings or early stopping. Evaluation uses a separate environment and preserves training RNG state. Re-evaluate the three selected vanilla checkpoints on the same independent development panel; the within-run difference isolates the **selection-rule** effect on that checkpoint bank without extra training. It does not reconstruct the inaccessible historical alternatives, and a new bank may differ through runtime realization and data ordering.

Candidate versus vanilla comparisons always use `best_by_money` for both. If Money-selected vanilla closes the BF gap, do not attribute that gain to a new architecture. No candidate may select a different checkpoint on SCREEN/CHECK/FULL-DEV.

### Shared training and data contract

Use the approved historical MIX12_EQ training pool and exact original first 1000 training episodes out of its declared first 3000, in the historical fixed order, for all initial methods. This keeps one million training transitions/run and avoids the prior proposal's unapproved 1200-episode change. Record pool/episode hashes and order; do not reshuffle only one method. All RNG differences stem from the declared training seed and model, not different transaction streams. Lock the runtime prospectively; an inaccessible pool or missing provenance blocks execution rather than inviting substitution.

Generate a fresh, **development-only** release before training: `SCFAC-DESIGN-v3`, using the exact twelve frozen regime specifications and generator semantics. Separate nonoverlapping pools:

| Split | Episodes/regime | Purpose |
|---|---:|---|
| VAL-v3 | 20 | All checkpoint selection, every 50 episodes |
| SCREEN-v3 | 40 | First budget-allocation decision |
| CHECK-v3 | 60 | Independent development check after adding a seed |
| FULL-DEV-v3 | 200 | Five-seed development decision for one finalist |

Use US/TLS/LNS/TLNS/TPLS/PLS/UB/TLB/LNB/TLNB/TPLB/PLB, T1000, original calibration/rounding/clipping and burst parameters; PCG64 with pinned generator/runtime. Derive each episode seed as the first eight SHA256 bytes, big-endian, of `SCFAC-DESIGN-v3|<split>|<regime>|<zero-based-episode>|20261003|<collision-counter>`, starting counter 0. Validate seed and episode-fingerprint disjointness against historical train/validation/evaluation pools, NEW12 and all new splits. Only a collision permits counter increment; log it. Never redraw based on a policy score or a stream mean. Freeze the actual manifests/hashes before training. Both capacities and all methods share identical arrays. No streams are generated in this task.

These panels are separate from the existing sandbox's SCREEN/CHECK subsets of NEW12. Do not reuse those already inspected subsets and relabel them fresh. Development is still development even when independent of NEW12; no formal claim follows from a screening p-value.

### Later controls, only after a finalist exists

Before any formal claim, compare the finalist with JA-PPO, IFAC, SC-M, SC-FAC-zero and BF-T0.5 on common pools, seeds, training budget and Money selection. New matched learned baselines are required to isolate method gains; archived scores remain context. For IA require global-only and selected-wallet-only ablations. For TP require teacher-only and always-on-prior controls, each disclosing their supervision budget; the lambda=0 case is already SC-M. Include deterministic BF as a fixed controller, not replicated seeds. D1 does not justify adding new support masks to these controls.

## 9. Multi-seed screening matrix

Initial models: **SC-M, IA-SC, TP-SC**. No ET/AS training yet.

| Stage | Training seeds × capacities | Incremental new trainings | What may be concluded |
|---|---|---:|---|
| SCREEN | seeds 323 and 777, each at C800 and C1200, for all 3 models | **12**: four/model | Development evidence only; never an inference from one seed/capacity |
| CHECK | Add seed 123 at both capacities for each survivor and the shared control; evaluate all completed 323/777/123 runs on CHECK | +4 if one survivor; +6 if both | Three-seed test of whether SCREEN promise repeats on a new panel |
| FULL-DEV | Select one finalist; add 532/999 at both capacities for finalist **and control**; evaluate all five seeds on FULL-DEV | **+8** | Five-seed development evidence on complete fixed sets |

Maximum planned candidate/control training totals: **24** with one CHECK survivor; **26** with two. Keeping the control on all five seeds is essential; counting only four added finalist runs at FULL would be wrong. Failed runs and eliminated candidates remain in the ledger. No seed replacement.

Seeds 323/777 deliberately cover the known H1 stress cases at opposite capacities; evaluate both seeds at **both** capacities to avoid confounding seed with capacity. They are not demonstrated hard seeds for native SC: C800/777 was actually a strong native SC run. Selection of stress seeds is post-hoc discovery, disclosed as such. Seed 123 anchors the previous pilot history;532/999 broaden the final set. No promotion based on whichever individual seed looks best.

All stages use complete 1000-episode trainings. Screening cuts the number of models/seeds and evaluation panels, not the per-method training budget. A short smoke/throughput check is an engineering test, never a scientific score or selected checkpoint. CHECK/FULL are never consumed early to rescue a weak SCREEN result.

## 10. Compute/runtime estimate

**Measured evaluation anchor:** H1 full 2400-episode receipts are about 191–198 seconds on the recorded local CPU setup; D1's instrumented 22-cell run was 4055 seconds. Neither measures new-model training throughput or promises GPU acceleration. Inspected historical run-info files provide timestamps, not a reliable isolated training-duration measurement; do not promote the sandbox's verbal “1–3 hour” estimate to a fact.

Each training run has 1M training transitions. Validation adds `20 checkpoints*12 regimes*20 episodes*1000 =4.8M` evaluation transitions/run. Across initial 12 runs that is **12M training plus 57.6M validation** transitions. Primary SCREEN scoring adds 5.76M; the two alternative checkpoint selections for the four vanilla cells add 3.84M. Two deterministic BF SCREEN evaluations add another 0.96M transitions. Total initial scientific evaluation workload is about **68.16M transitions**, roughly 1.5 hours at the H1 reference rate, with no guarantee the new implementation matches it.

For planning only, assume .5–2 hours of **training excluding validation** per full run until measured: initial 12 trainings imply 6–24 worker-hours, plus approximately 1.5–2.5 evaluation hours and overhead, giving **about 8–27 worker-hours**. Two genuinely independent workers might reduce wall time to roughly 4–14 hours; do not assume linear scaling on one oversubscribed host. Feature extraction or teacher labels could dominate small-network arithmetic. Report actual throughput on the first approved control run and revise the prospective schedule without shortening other models or changing success thresholds.

Indicative coding/test budget before an overnight launch: shared checkpoint-bank/selection tooling 0.5–1 day; IA 1–2 days; TP 1–2 days, partly shared. These are estimates, not implemented capabilities. Default runtime remains CPU unless all matched methods are prospectively moved to the same approved device/runtime. Codex does not access the GPU server.

If a finalist proceeds to formal testing, five seeds×two capacities for the finalist, four learned baselines and two mechanism ablations means **70 learned training runs total in that formal matrix**, plus two BF evaluation cells. Completed matching finalist/control runs can be reused with receipts; a screened-out candidate adds up to six discarded training runs, so the full discovery+formal path can reach 76 trainings. That expansion needs a separate PI budget decision; it is not an overnight plan. The immediate authorization request would be12 runs, not 70.

## 11. Kill / promotion criteria

These are predeclared **engineering/resource-allocation gates**, not significance tests. Use the same panel and the same available seeds at each stage. For capacity C define:

`B_C = fixed BF macro12 Money`, `V_C = mean SC-M Money`, `G_C=max(B_C−V_C,0)`, and `Delta_C=mean_s(S_candidate,C,s−S_SC-M,C,s)`.

Never divide by a tiny or negative “BF gap.” If control already beats BF, G=0; compare meaningful absolute gains and consider the simpler control as the result. Do not claim “100% gap recovery” from a denominator artifact. If a development BF score were nonpositive, pause to review the relevance thresholds; do not adapt them after final testing.

**A. Minimal promise: SCREEN→CHECK.** At **both capacities**, require `Delta_C >= max(100 Money, .25*G_C)`. Require no seed-level loss versus its matched SC-M worse than `.05*abs(B_C)`. Record whether shortage exposure, settled value and flush cost move consistently with the candidate hypothesis. A +20/+50 gain is insufficient, irrespective of a convenient percentage or p-value. Passing at only one capacity is interesting but does not advance this two-capacity challenger campaign.

**B. Sufficient evidence for five-seed expansion: CHECK→FULL-DEV.** On the independent CHECK panel, require `Delta_C >= max(150, .40*G_C)` at both capacities, at least 2/3 positive paired deltas at each, and the same no-collapse limit. There must be no provenance, budget or seed-completeness failure. Choose at most one finalist. If both qualify, maximize the smaller of their two capacity-level advantages versus BF normalized by `abs(B_C)`. If the difference in this score is less than .01, prefer IA because it has no teacher dependency. This rule is fixed before outcomes; no architecture fusion.

**C. Sufficient evidence to spend fresh NEW13.** On FULL-DEV, require all of:

- mean candidate−BF at each capacity at least `max(100 Money,.01*abs(B_C))`;
- at least 4/5 individual trained policies above BF at each capacity;
- mean paired gain over SC-M at each capacity at least `max(150,.40*G_C)` if an architectural/training-prior advance is to be claimed;
- no individual loss versus matched SC-M worse than 5% of BF; complete accounting and mechanism telemetry; unchanged primary selection rule;
- for TP, the prespecified best prior-free-period checkpoint also meets the candidate−BF mean/4-of-5 conditions on this development panel. If either capacity fails, TP cannot advance to NEW13 in this campaign. Primary `best_by_money` is not silently replaced. Passing establishes retained autonomous competence; claiming further improvement after teacher removal additionally requires comparison against the retained boundary checkpoint.

These thresholds are intentionally more demanding than recovering a fraction of the historical deficit. A candidate reaching halfway to BF is promising discovery evidence, not a final BF challenger. If SC-M itself clearly beats BF and candidates fail to add meaningful gain, stop architecture expansion and report the selection/training-control result honestly; that simpler result would still require its own fresh confirmation.

**D. Failure and stopping.** If both top candidates fail SCREEN/CHECK, stop this design branch. If the one finalist fails FULL-DEV, do not automatically train ET/AS, tune lambda, add features, widen networks or change reward. Such work requires a new PI-reviewed question and must retain this failure record. Nonfinite training is a method failure unless a verified infrastructure cause is demonstrated. Infrastructure resume keeps the same seed/configuration/data. A slower run is not shortened selectively. The complete matched comparison matrix must fit a new budget before final data are generated.

Do not use a significance threshold with two or three seeds to decide these gates. Report all individual scores and uncertainty at five seeds; screening and seed choice are adaptive development, even with gates. Nothing here changes the frozen Stage 3 test families.

## 12. Proposed overnight experiment order

This is a proposed future run order, **not a task to execute now**. Prior to launch, PI approval, isolated implementation, synthetic/OLD-only correctness tests, source/hash receipts, all development manifests, and the fixed configuration table must exist. The D1 summary defect must be documented in the new input validator; it must reject duplicate keys and use reconciled per-cell data, without editing the frozen file. Repair future cycle accounting in new code only.

1. Establish common runtime and throughput with SC-M. Verify train/eval RNG separation, joint PPO likelihood, deterministic checkpoint loading, count/Money identities, initial IA/control function equality, teacher labels excluding the actual learner settlement, teacher decay to zero, and complete removal of teacher calls from TP deployment.
2. Round A: C800/777 and C1200/323, for SC-M then IA then TP (six total runs). Pair capacities across at most two workers if hardware permits; use the same concurrency/thread setting for every model. These are the known H1-stress settings, not a basis for a one-seed verdict.
3. Round B: C800/323 and C1200/777, again all three models (six runs). Complete both rounds before applying SCREEN elimination. No late rescue run selected by which capacity looks favorable.
4. Score the fixed SCREEN panel; evaluate the vanilla checkpoint-selection counterfactuals; write all 12 outcomes and measured compute. If the time window ends first, record an incomplete matrix and resume the declared remaining cells later. Do not announce a winner from Round A.
5. Stop after the SCREEN report. CHECK's additional 4/6 runs require the predeclared gate and PI allocation. Do not roll directly into five seeds, reward tuning, guardrails, ET/AS, NEW13, or a paper rewrite overnight.

For an eight-hour window, all 12 initial runs are feasible only near the favorable end of the throughput range. A finished six-run partial matrix is less informative than a complete screen and must be labeled accordingly; wall time cannot justify a lower scientific bar.

## 13. Risks of overfitting NEW12

NEW12 has selected questions, mechanisms and stress seeds repeatedly. It is discovery evidence. D1's U/M parity and numeric agreement do not restore statistical freshness. Existing sandbox SCREEN0–39/CHECK40–99/FULL0–199 are partitions of that same consumed release and cannot validate the new method confirmatorily.

Use the new development panels in Section 8 for every new checkpoint decision and gate. If manifests reveal overlap with old pools, block release before scores are generated. Do not inspect the CHECK/FULL-DEV panels until their declared stage. Preserve failed models, masks, lambda choices and checkpoint selections; selecting a method is part of the statistical history, even when it is not called hyperparameter tuning.

BF-specific insight is also a form of development information. Training on BF labels is allowed only in the explicitly labeled TP arm and only on training-visited states. Do not generate teacher labels on final pools for optimization, use final starvation diagnostics to redesign features, or select per-regime controllers after inspection. Retained flexibility is an architectural property; superior generalization remains unproved.

## 14. What would need fresh NEW13 evaluation

Any assertion that the new learned policy improves on BF, vanilla Money-selected SC, or the other policy families requires fresh confirmation **after the selected method, all checkpoints, data-generation rules, comparators, masks, decoder, analysis and code/runtime hashes are frozen**.

Proposed final release: NEW13-v1, same twelve named regime specifications, k24/F3/T1000, C800/C1200, 200 independent episodes/regime, paired identically across all methods. Generate only after a PI-held new random seed nonce is released following the freeze. Use the pinned generator/calibration/PCG64 process and mechanical seed/fingerprint overlap checks. The actual nonce and pool hashes cannot be supplied now because the release must not exist before the freeze. No regeneration based on performance; a viewed release is consumed permanently.

Training seeds remain 123/323/532/777/999; all matched learned methods use all five. Primary outcome hierarchy: episode→regime mean→equal macro12 mean→one score per trained seed. BF is fixed per capacity. Report sample SD/SE and marginal two-sided Student-t intervals with df4; pair learned methods by seed; compare against BF with a one-sample t-test of the five learned-minus-fixed-BF deltas.

Prospective confirmatory family P contains four tests: finalist−BF and finalist−SC-M, each at both capacities, Holm across all four. A broad statement of superiority requires positive adjusted evidence in every named setting; it cannot be inferred from an exploratory screen. Secondary family contains finalist−JA, finalist−IFAC, finalist−SC-zero, and finalist−each of its two mandated ablations, each at both capacities: ten tests with a separately declared Holm family. Report raw and adjusted p-values, effects, CIs and every seed. This controls each declared family, not one combined global error rate. No episode-level pseudo-replication, dropping bad seeds, post-hoc one-sided tests or enlarging n after p-values are viewed.

The new three-selector checkpoint comparison is exploratory unless separately included in the frozen final statistical plan; extra selected checkpoints must not multiply the number of primary opportunities to win. Regime patterns and telemetry remain descriptive. Five seeds leave wide uncertainty: do not call a null result equivalence or estimate power from a favorable pilot. [Agarwal et al.](https://arxiv.org/abs/2108.13264) motivates taking few-run RL uncertainty seriously; it does not justify replacing our predeclared Money aggregation with a different favorable statistic.

Fresh episodes within the same families test within-family generalization, **not adaptation to distribution shift**. A claim of superior adaptivity would require a separately predeclared SHIFT13/switching or parameter-shift evaluation with identical resources and information for every controller. No such experiment is in the initial budget, and the model cannot acquire that claim merely by remaining neural or deviating from BF.

## 15. Final recommendation

The failure diagnosis should be **resource-maintenance competence under coupled learned decisions**, not “add a stronger heuristic” or “increase the network.” The data support moving away from same-wallet masking as the main method and away from any universal flush-more prescription. They do not yet identify representation, optimization or exploration as the unique cause.

IA-SC is the best first scientific test because its physical features are small, causal, interpretable, and BF-free. TP-SC is a credible second test because it preserves native SC actions and can ask whether finite behavioral guidance changes what the policy learns. It should be described as teacher-assisted learning, not disguised as a purely architectural method. The strongest outcome would be a coherent behavior change and material Money improvement that survives matched selection and fresh confirmation; matching BF or a tiny gain over historical SC would not be sufficient.

**PRIMARY_CANDIDATE:** Inventory-aware SC-FAC (IA-SC), with nine global inventory/refill features and four selected-wallet features; both heads learned.

**SECONDARY_CANDIDATE:** SC-FAC with a decaying, training-only BF prior (TP-SC), then 750 episodes of native PPO with no teacher loss; no BF component at deployment.

**BEST_NON_BF_CANDIDATE:** IA-SC.

**BF_INTEGRATION_RECOMMENDATION:** LIGHT.

**WHY:** extract physical resource insight first; separately test finite teacher assistance because BF supplies demonstrably competent behavior. Do not freeze settlement to BF, impose its threshold at inference, or claim the prior is necessary before comparing it with the BF-free method.

**NUMBER_OF_INITIAL_TRAINING_RUNS:** 12 =3 models (SC-M, IA, TP) ×2 seeds (323,777) ×2 capacities (800,1200). No initial runs for ET/AS. Maximum candidate/control discovery budget is24 or26 runs through FULL-DEV, conditional on gates.

**ESTIMATED_RUNTIME:** approximately 8–27 worker-hours for the initial 12-run screen, including validation/screen scoring, based on explicit provisional training assumptions and measured evaluation throughput; approximately 4–14 hours with two effective workers, subject to calibration. Implementation/testing time is additional. An overnight completion is not guaranteed.

**READY_FOR_IMPLEMENTATION:** YES — the isolated screen is specified sufficiently for implementation review after PI approval. This is not approval to train. Execution still requires the new development manifests, runtime/receipt lock, correctness tests and provenance guards. The frozen D1 summary defect blocks blind reuse of that CSV, not this design because the underlying evidence has been reconciled. No implementation, training, commit or push has occurred in this task.
