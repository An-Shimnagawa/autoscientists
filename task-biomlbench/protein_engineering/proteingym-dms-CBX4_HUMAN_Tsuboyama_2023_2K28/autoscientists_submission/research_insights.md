---
task: proteingym-dms-CBX4_HUMAN_Tsuboyama_2023_2K28
run_id: biomlb_pg_cbx4_2
started_at: "2026-04-28T23:48:11Z"
champion_at: "2026-04-29T03:04:53Z"
---

# Research Insights for ProteinGym DMS CBX4 Human Fitness Prediction

AutoScientists discovered that a two-component final blend — a large OOF stacking ensemble (`super_stack_018`, weight 0.8274) combined with a mutation-count-stratified ensemble (`mut_strat_023`, weight 0.1726) — achieves mean Spearman ρ = 0.956765 across three fold strategies. The headline finding is that **progressive meta-stacking of protein language model embeddings, guided by greedy forward-backward LOO selection, substantially outperforms any single embedding model**, and that a late-stage mutation-stratified blend provides a small but consistent residual gain by handling single- and double-substitution variants with separate optimal weights.

## Findings

**1. Delta embeddings from ProtT5-XL and ESM2 are complementary; their cross-family ensemble is the strongest single predictor.**
The first cycle established that ESM2-650M delta embeddings with SVR/Ridge reached Spearman 0.9351 (cycle 1, gpu1) and ProtT5-XL SVR/Ridge reached 0.9506 (gpu2). A two-model cross-family OOF blend (ProtT5 + ESM2-650M) immediately jumped to 0.9528, then 0.9535 after a metric correction. Adding ESM2-3B as a third family member yielded 0.9537 (gpu3), and the ProtT5 + ESM2-650M + ESM2-3B triple ensemble reached 0.9541 (gpu3, cycle 4). The gain is attributable to diversity across architectures: ProtT5 is trained with a T5 masked-span objective, while ESM2 uses masked-language modelling, so their representations capture different aspects of the sequence-fitness landscape.

**2. OOF stacking of 8–14 base models with RidgeCV meta-learner outperforms direct model blends.**
Systematic OOF stacking of growing model pools drove the score from 0.9541 (triple ensemble) to 0.9549 (11-model RidgeCV, gpu5 cycle 4). The RidgeCV meta-learner consistently outperformed non-linear alternatives: LightGBM (0.9523), SVR RBF (0.9546), ElasticNet (0.9548), and GP nonlinear meta (0.9543) all fell short of RidgeCV. The optimal number of base models in the pool was 8–14 — expanding to all 20 available degraded performance (0.9546 < 0.9550), confirming that indiscriminate pool growth hurts.

**3. Greedy forward-backward LOO model selection is the most reliable stacking strategy at this scale.**
After simple RidgeCV stacking plateaued near 0.9549–0.9551, greedy selection (exp_emb_greedy_select_011, gpu2) found a 7-model optimal subset scoring 0.9551. Iterative LOO refinement in successive rounds (loo_r3 through loo_r11) incrementally advanced the champion from 0.9551 to 0.9565 over multiple rounds, crossing 0.956 at the 10-model stage (gpu6 loo_expand_10_015: 0.9561, gpu5 loo_r5_016: 0.9562). The final 11–14 model greedy stacks formed the backbone of the super-stack component of the champion.

**4. Mutation-count stratification provides a small residual gain beyond global-weight stacking.**
The CBX4 DMS dataset contains only single (n_muts=1) and double (n_muts=2) amino acid substitution variants. exp_emb_mut_strat_023 (gpu5) hypothesized — and confirmed — that separate SLSQP-optimized weights for each stratum outperform global-weight ensembling. The stratified model contributed with weight 0.1726 in the final blend. This is the only mutation-structure-aware component in the champion.

**5. ESM2-3B provides no systematic advantage over ESM2-650M as a standalone model for this short protein.**
gpu3 tested ESM2-3B SVR+Ridge (0.9455 vs. 0.9352 for 650M in direct comparison cycles). ESM2-3B improved ensemble diversity as a base model but did not replace ESM2-650M in the final stack, suggesting scaling the language model beyond 650M yields diminishing returns for a ~69-residue CBX4 sequence with 2,282 single- and double-substitution variants.

### Insights

**G1 — Cross-family protein language model diversity is the primary lever for DMS fitness prediction.**
*Claim:* Blending predictions from models trained with different pre-training objectives (masked-span T5 vs. masked-LM ESM) yields larger gains than scaling within a single model family.
*Disconfirming evidence:* ESM2-650M → ESM2-3B adding as much as ProtT5 → ESM2-650M cross-family blending.
*Observed:* ProtT5 alone: 0.9506; ProtT5 + ESM2-650M blend: 0.9535 (+0.0029); adding ESM2-3B (same family as 650M): 0.9537 (+0.0002). Cross-family gain dominates within-family scaling.

**G2 — RidgeCV is a superior meta-learner compared to non-linear alternatives when OOF meta-features have high mutual correlation.**
*Claim:* When base models are all from the same class (protein LM delta embeddings + linear/SVR regressors), their OOF outputs are strongly correlated. A linear meta-learner with regularization avoids overfitting the meta-feature space better than non-linear alternatives.
*Disconfirming evidence:* LightGBM or SVR meta-learner outperforming RidgeCV consistently across pool sizes.
*Observed:* LightGBM meta (13 models): 0.9523; SVR meta (11 models): 0.9546; ElasticNet (11 models): 0.9548; RidgeCV (10 selective): 0.9550; all below RidgeCV greedy stacks at 0.9551+.

**G3 — Greedy forward-backward LOO model selection prevents performance degradation from over-inclusion in stacking.**
*Claim:* Beyond ~10–14 models, adding more base models to an OOF RidgeCV stack introduces correlated noise. LOO selection identifies a minimal diverse subset, avoiding the marginal dilution observed with 20-model pools.
*Disconfirming evidence:* Performance monotonically increasing with pool size in multi-agent stacking experiments.
*Observed:* Ridge stacking 20 models: 0.9546 < 10-model selective: 0.9550 < greedy 7-model: 0.9551.

**G4 — Mutation-count stratification of ensemble weights is beneficial when the dataset contains a discrete covariate (n_muts) that correlates with prediction difficulty.**
*Claim:* When single and double mutations have different error profiles across base models, optimizing blend weights per stratum outperforms a single global blend.
*Disconfirming evidence:* Stratified weights not improving over global weights on tasks where all variants have the same mutation count.
*Observed:* Mutation-stratified blend (mut_strat_023) contributed 0.1726 weight in the final 2-component champion, with per-stratum SLSQP optimization used to derive weights.

### Task-Specific Findings

**T1 — ProtT5-XL delta embeddings outperform ESM2-650M and ESM2-3B as standalone models for CBX4.**
In parallel cycle-1/2 experiments: ProtT5-XL SVR+Ridge reached 0.9506 vs. ESM2-650M delta SVR+Ridge at 0.9454 and ESM2-3B at 0.9455. ProtT5's T5-Encoder architecture and per-residue embedding scheme may better represent the stability-relevant context of this short chromobox-domain protein.

**T2 — Mutation-site pooling for ProtT5 embeddings does not improve over mean pooling.**
exp_fast_prott5_mutsite_svr_008 (gpu5): 0.9508 — slightly above ProtT5 mean-pooled SVR+Ridge but below champion. Using only the embeddings at the mutated position loses context from the surrounding protein structure.

**T3 — Ankh-base embeddings are weak standalone (0.849) but contribute diversity to ensembles when included as a stacking feature.**
Ankh-base SVR standalone scored 0.8491 — far below the champion. However, exp_emb_ankh_stack_014 (gpu5) incorporating Ankh predictions as an additional stacking feature yielded 0.9554 (marginal improvement +0.000012 over loo_r3 champion at that point). Ankh embeddings encode qualitatively different representations but carry lower intrinsic signal for this task.

**T4 — Zero-shot LLR scoring from ESM-1v adds negligible signal on top of ESM2-650M embeddings for CBX4.**
exp_zs_esm1v_llr_003 (gpu6): 0.9462 — appending ESM-1v masked-marginal log-likelihood ratios to ESM2-650M embeddings added only +0.0001 over ESM2 alone (0.9461). Zero-shot evolutionary scoring does not provide complementary information beyond what is already captured by supervised delta embeddings for this dataset.

**T5 — Concatenating ProtT5 and ESM2 embeddings with PCA reduction (joint 2304-dim → PCA) is worse than late-fusion OOF blending.**
exp_emb_prott5_esm2_concat_pca_005 (gpu2): 0.9516 and exp_emb_prott5_esm2_concat_007 (gpu2): 0.9380. Early fusion (feature concatenation) consistently underperformed late fusion (ensemble of separately trained models) by 0.015–0.017 Spearman. This likely reflects the mismatch in embedding dimensionalities and the difficulty of training a single regressor on a 2304-dim joint space versus each model's native space.

**T6 — Isotonic post-processing hurts rank correlation.**
exp_zs_isotonic_019 (gpu6): 0.9540 (-0.0023 vs. 0.9564 champion at that point). Isotonic regression calibrates probability/magnitude but disrupts the rank order that Spearman correlation measures — a hard dead end for this metric.

## Dead Ends and Negative Results

**ESM2-650M + one-hot concat Ridge:** val 0.8952. Adding one-hot mutation encoding to 1280-dim ESM2 embeddings is dominated by noise from the sparse positional features. Retired.

**ESM2-3B standalone SVR+Ridge:** 0.9455 — no advantage over ESM2-650M (0.9454) despite 5x parameter count increase. Scaling within ESM2 family yields diminishing returns for short proteins. Retained as ensemble diversity contributor only.

**ProtT5 kernel ridge (cosine + polynomial kernels):** 0.9480 — below ProtT5 SVR+Ridge (0.9506) and ESM2/ProtT5 blends. Retired.

**ESM2 ExtraTrees:** 0.9278 — tree-based models on high-dimensional embeddings substantially underperform linear/SVR models for this regression task. Retired.

**ESM2 SVR large-C:** 0.9443 — tuning SVR cost parameter alone does not close the gap; architectural diversity is more important than single-model hyperparameter tuning. Retired.

**LightGBM meta-learner on 13 OOF models:** 0.9523 < RidgeCV 0.9550 champion at equivalent pool size. Non-linear capacity in the meta-learner does not help when base models are strongly correlated. Retired.

**SVR RBF meta-learner on 11 OOF models:** 0.9546 < RidgeCV. Same conclusion as LightGBM meta.

**ElasticNet meta-learner:** 0.9548 — close to but consistently below RidgeCV. L1 sparsification does not improve over pure L2 regularization for this meta-feature matrix.

**Bootstrap RidgeCV (B=100 random model subsets):** 0.9549 — did not surpass greedy selection champion (0.9550). Bagging the meta-learner adds compute but no gain. Retired.

**Mean rank aggregation (Borda count) of 11 models:** 0.9547 < 0.9550. Nonparametric rank aggregation fails to capture the calibration differences between models. Retired.

**Ridge stacking all 20 models:** 0.9546 — below selective 10-model champion (0.9550). Over-inclusion of weakly correlated models dilutes the meta-signal. Retired.

**Super-meta stacking (stacking outputs of top stacking models):** Two attempts — gpu3 exp_emb_super_meta_009: 0.9551; gpu6 exp_zs_super_meta_013: 0.9551 — tied or just below greedy champion at the same stage. Adding a second stacking level does not reliably improve over single-level greedy LOO selection.

**Corrupt submission (gpu5 exp_fast_greedy_fwd_012):** Agent reported 0.9568 but true score on revalidation was 0.6866. Flagged as corrupt submission; discarded.

**Polynomial Ridge meta (degree=2 features on OOF predictions):** exp_zs_poly_ridge_015: 0.9549 — adding polynomial interactions to 3-model meta-features does not improve over linear RidgeCV on a regression task this smooth.

**ProtT5 + ESM2 early fusion (joint concatenation, 2304-dim):** 0.9380 — early fusion is substantially worse than late fusion. See T5 above.

**Bayesian Ridge meta-learner:** 0.9553 — marginally below greedy LOO champion at that stage. No improvement over standard RidgeCV.

**Rank-transformed OOF meta-features:** 0.9548 — below raw-score RidgeCV. Rank transformation in the meta-feature step discards calibration information that RidgeCV can exploit.

## Coordination and Team Dynamics

**Team structure and division of labor:** Six GPU agents (gpu1–gpu6) ran experiments in parallel across 5+ cycles. Two teams were defined: `team_embeddings` (gpu1–gpu5, focused on protein LM delta embeddings and their combinations) and `team_zeroshot` (gpu6, focused on zero-shot evolutionary scoring hybrids and stacking orchestration). gpu4 operated as a fast-iteration agent focused on greedy selection variants.

**Progressive sharing of submission files:** Agents systematically referenced each other's output CSV files as pool inputs from cycle 3 onward. By cycle 4, gpu5 and gpu6 were loading from 8–11 agents' workspace repos. This cross-agent reuse is the structural reason the meta-stacking approach could incorporate 14–25+ base predictions.

**Convergent discovery of optimal pool composition:** Multiple agents (gpu2, gpu3, gpu4, gpu5, gpu6) independently converged on overlapping sets of base models for their stacking pools. The core 8-model set — `prott5_esm2_ens`, `prott5_esm2_3b`, `prott5_svr`, `esm2_ridge`, `prott5_mutsite`, `prott5_concat_pca`, `gpu4_huber`, `gpu3_gp_meta` — appeared across all greedy selection results from cycle 5 onward. This cross-agent convergence increases confidence that these 8 models represent the diversity-optimal subset of the explored model space.

**Corrupt submission detection:** gpu5's exp_fast_greedy_fwd_012 claimed a score of 0.9568 but was flagged when independent revalidation returned 0.6866. The harness correctly discarded this result and the run continued without it propagating.

**Late-stage fine-grained optimization:** The final champion improvements (0.9556 → 0.9568) were driven almost entirely by gpu2 running iterated meta-blend searches (exp_emb_meta_blend_023 through _027), narrowing in on the optimal 2-component blend weight (0.8274 for super_stack_018, 0.1726 for mut_strat_023) via fine grid search.

**gpu6 coordination role:** gpu6 (`team_zeroshot`) ran most of the 20-model pool experiments and orchestrated several of the LOO expansion rounds. Its zero-shot ESM-1v LLR approach did not contribute to the champion directly, but its quantile ensemble output (exp_zs_quantile_ens_016) became a pool member in several downstream greedy stacks.

## Limitations of These Insights

**No held-out test set:** All reported scores are mean Spearman across the three specified fold strategies (random, modulo, contiguous) on the full labeled dataset. No independent held-out test set was used; the reported 0.9568 may be optimistic due to the large number of meta-ensemble configurations evaluated against the same fold structure.

**Meta-stacking leakage risk:** The final super-stack incorporates predictions from up to 25+ base submissions, some of which were themselves selected after observing the same fold-strategy Spearman scores used to evaluate the champion. Greedy LOO selection partially mitigates this, but does not eliminate it entirely.

**Single run, no independent replication:** All findings come from one 3.3-hour run. The specific blend weight (0.8274/0.1726) and the composition of the greedy model pool may not replicate in a second run with different agent initialization order or random state.

**Limited model diversity beyond protein LMs:** The entire model family explored is delta embeddings from protein language models (ESM2, ProtT5, Ankh) with linear/SVR regressors. Physics-based stability predictors (FoldX, Rosetta ddG), structure-conditioned models, or evolutionary-covariance-based approaches were not explored. The Ankh embedding attempt (0.8491 standalone) suggests that not all language models perform equally on this task, but no fundamentally different feature class was tested.

**CBX4-specific structural context ignored:** The CBX4 protein (human chromobox protein 4, a chromodomain involved in H3K27me3 reading) has known domain structure. No structure-aware features (contact maps, solvent accessibility at mutated residues, conservation profiles) were incorporated. The mutation-stratified component (mut_strat_023) is the only approach that used any protein-level metadata (mutation count).

**Unexplored axes:**
- Structure-conditioned models (ESMFold-based features, inverse folding models)
- Evolutionary co-variation features (EVcouplings, MSA-based)
- Broader protein LM coverage (ProteinBERT, ESM-IF, SaProt)
- Held-out validation with scaffold-based or positional splits to assess generalization beyond the three provided fold strategies
