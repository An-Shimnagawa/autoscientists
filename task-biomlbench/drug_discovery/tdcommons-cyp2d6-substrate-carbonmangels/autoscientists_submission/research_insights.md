---
task: tdcommons-cyp2d6-substrate-carbonmangels
run_id: biomlb_cyp2d6_4
started_at: "2026-04-22T00:09:36Z"
champion_at: "2026-04-22T02:42:22Z"
---

# Research Insights for TDCommons CYP2D6 Substrate Prediction

AutoScientists discovered that per-group SHAP-guided cross-fingerprint feature fusion combined with a calibrated SVM-RBF as a fifth base model in a stacking ensemble substantially outperforms both single-model approaches and transformer-based methods on this small-molecule binary classification task. The headline finding is that **feature diversity across orthogonal chemical representations — each reduced via independent SHAP selection — is the primary driver of AUPRC gains**, explaining the improvement from the cycle-1 baseline (0.676 AUPRC, Mordred + 3-model stack) to the final champion (0.753 AUPRC, 160-dim cross-SHAP + LGB/XGB/CatBoost/ExtraTrees/SVM-RBF + LR meta). A secondary finding is that neural approaches including fine-tuned and frozen ChemBERTa transformers failed conclusively on this dataset (n ≈ 532, ~28% positive), scoring 0.41–0.55 AUPRC vs. 0.68+ for classical ensembles.

## Findings

**1. Cross-fingerprint SHAP fusion is the primary performance driver.**
The largest single-cycle gain occurred in cycle 2 when Team Beta introduced per-group SHAP selection across five orthogonal chemical representations (Mordred-50 + ECFP4-30 + ECFP6-30 + MACCS-20 + AtomPair-15 + TopoTorsion-15 = 130 features, exp_beta_007) rather than operating on any single feature source. This pushed AUPRC from 0.676 (cycle 1, Mordred-only stack) to 0.737 (+0.061). The key mechanism: selecting the most informative bits independently from each group prevents high-dimensional noise from one source corrupting the signal from others.

**2. Adding a calibrated SVM-RBF as a fifth base model yields the final champion gain.**
In cycle 3, Team Alpha's GPU2 agent tested appending a CalibratedClassifierCV(SVC(kernel='rbf'), method='isotonic', cv=3) to the four-tree-model stack (exp_alpha_010). This lifted AUPRC from 0.737 to 0.753 (+0.016). The improvement is attributed to the SVM's kernel-based decision boundary being orthogonal to the tree models' axis-aligned splits, providing OOF predictions that the LR meta-learner could exploit. The same experiment confirmed this by comparing the five-model stack to the four-model stack within the same run; the five-model version was selected only when it outperformed the four-model baseline.

**3. Naively concatenating raw fingerprints hurts performance; per-group SHAP filtering is necessary.**
Early cycle-1 experiments combining raw ECFP6 (2048 bits) with Mordred (1200+) yielded AUPRC 0.625 (exp_alpha_005, gpu2), below the Mordred-only 3-model champion at 0.676. Similarly, Team Beta's exp_beta_005 (SVM with Tanimoto kernel on Morgan FP + Mordred ensemble, 0.661) underperformed. Once independent SHAP selection was applied per group, the same fingerprint types (Morgan, MACCS, AtomPair, TopoTorsion) contributed positively. This confirms that high-dimensional fingerprint bits contain mostly noise for n ≈ 532 and require group-specific dimensionality reduction before fusion.

**4. Adding more feature groups saturates and eventually degrades performance.**
Team Beta tested extending the 130-feature champion (5 groups) to 160 features by adding RDKit 2D descriptors (top-15) and Avalon fingerprints (top-15), yielding 7 total groups (exp_beta_008, 0.736). This was a marginal degradation (−0.001 from champion 0.737), noted in the experiment's result JSON as "Adding more feature groups did not help." Team Alpha's 7-group experiment (exp_alpha_009, cross-SHAP + GBM meta, 0.664) showed a larger drop, partly attributable to the GBM meta-learner overfitting the OOF stack on small data. The feature-diversity lever appears saturated at approximately 5–6 groups and ~130–160 selected features.

**5. Transformer fine-tuning fails conclusively on this dataset size.**
Team Gamma (GPU6) ran ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) in two configurations. Full fine-tuning with focal loss (exp_gamma_001, 15 epochs, lr=2e-5) achieved mean CV AUPRC 0.552 ± 0.077, with extreme fold variance (fold 1: 0.484, fold 3: 0.459). Frozen encoder with learned mean+max pooling head (exp_gamma_002, 30 epochs) performed worse at 0.415 ± 0.092. Both are far below the classical ensemble baseline. The analyst team attributed this to the 44M-parameter ChemBERTa model overfitting on per-fold training sets of ≈390–440 molecules.

### Insights

**G1 — Per-group SHAP selection before cross-fingerprint fusion is essential for small-n molecular classification.**
*Claim:* On datasets with n < 600, independently applying SHAP-guided feature selection within each fingerprint/descriptor group before concatenation provides substantially more benefit than using raw or jointly selected feature matrices, because each group's noise floor is controlled before combining.
*Disconfirming evidence:* A task where cross-group joint feature selection outperforms independent per-group selection at similar dataset sizes.
*Observed:* Raw ECFP6 + Mordred concatenation = 0.625 AUPRC (exp_alpha_005); same feature types with per-group SHAP = 0.737 (exp_beta_007).

**G2 — SVM-RBF provides orthogonal diversity in stacking ensembles where all other base models are tree-based.**
*Claim:* When four tree-based models (LightGBM, XGBoost, CatBoost, ExtraTrees) form the base learner set, a calibrated SVM-RBF adds prediction diversity that a fifth tree model would not, yielding a measurable stacking gain on small imbalanced molecular datasets.
*Disconfirming evidence:* A task where adding SVM to a 4-tree stack shows no improvement or degradation compared to a fifth tree model.
*Observed:* exp_alpha_010 improved from 0.737 (4-tree stack reference) to 0.753 with SVM as fifth base; outcome was KEEP.

**G3 — Feature diversity has diminishing returns and an inflection point near 5–6 groups for n ≈ 530.**
*Claim:* On datasets of n ≈ 530, increasing orthogonal feature groups from 1 to 5 provides large AUPRC gains, but expanding beyond 5–6 groups shows near-zero or negative returns, likely because additional groups introduce correlated noise and the LR meta-learner is over-dimensioned.
*Disconfirming evidence:* A small-molecule dataset of similar size where 7+ feature groups consistently improve over 5-group baselines.
*Observed:* 1 group = 0.676, 3 groups (Mordred+Morgan+MACCS) = 0.720 (exp_beta_006), 5 groups = 0.737 (exp_beta_007), 7 groups = 0.736 (exp_beta_008, marginal degradation).

**G4 — Transformer fine-tuning (even with focal loss) is unreliable for binary molecular property classification at n < 600.**
*Claim:* Pre-trained molecular transformers such as ChemBERTa require substantially more than 500 training examples to converge reliably on binary classification with scaffold-based splits; focal loss does not compensate for the small-data regime.
*Disconfirming evidence:* A dataset of n ≈ 400–600 where ChemBERTa fine-tuning consistently outperforms classical descriptors + ensemble.
*Observed:* exp_gamma_001 (full fine-tuning) = 0.552 ± 0.077; exp_gamma_002 (frozen embeddings) = 0.415 ± 0.092; both far below the classical ML baseline of 0.676+.

### Task-Specific Findings

**T1 — The effective training size per fold (≈390–440 molecules, ~28% positive) creates high AUPRC variance across folds.**
Fold-level AUPRC in the champion experiment (exp_alpha_010) ranged across folds, and the beta_007 champion's per-fold scores showed std ≈ 0.023 (cv_fold_scores: 0.730, 0.734, 0.699, 0.770, 0.744). This variance reflects both scaffold-based split heterogeneity and the small effective positive count per fold (approximately 24–42 positives in validation folds). Results with CV std > 0.025 should be treated cautiously.

**T2 — LightGBM with scale_pos_weight = (n_neg / n_pos) ≈ 2.6 is an effective baseline imbalance strategy; more complex methods add marginal value.**
The champion and all high-performing experiments computed scale_pos_weight per fold as (negative count / positive count), which naturally adjusts for the ~28% positive rate. Experiments attempting SMOTE oversampling (exp_beta_009) and Balanced RandomForest (analyst2 proposal for exp_gamma_004) did not yield improvements beyond the gradient re-weighting approach already in the champion architecture.

**T3 — GBM meta-learners consistently underperform LR meta-learners on OOF stacks of this size.**
Multiple experiments tested replacing the LogisticRegression meta-learner with GradientBoostingClassifier or LightGBM (exp_alpha_007 with GBM meta, exp_alpha_009 with GBM meta). These consistently scored below the LR meta variants with identical features, consistent with the analyst notes stating "nonlinear meta-learner overfit on small OOF sample." With only 5 OOF predictions as meta-features (post cycle 3) and ~532 samples, the LR meta-learner's low capacity is beneficial.

**T4 — L1/LinearSVC feature selection from Mordred (114 selected features) underperforms SHAP-based selection.**
Team Beta's GPU4 agent tested Mordred + L1/LinearSVC feature selection yielding 114 features (exp_beta_gpu4_004, 0.677 AUPRC). This was competitive with the 3-model baseline but substantially below the SHAP-based selection approach at 0.737. The difference suggests that LinearSVC L1 selection within a single descriptor source captures less signal than SHAP-based selection applied independently per feature group.

**T5 — Cycle 4 experiments were still running at the submission deadline.**
All six cycle-4 experiments (multi-seed ensemble, Optuna-tuned SVM, 200-dim cross-SHAP, 6-model stack + RF, tuned LR meta C, SVM calibration comparison) completed their Mordred+SHAP feature-selection phase (~17 minutes each) but had not finished training by the deadline (04:05 UTC). The submission uses the cycle-3 champion (exp_alpha_010, 0.753 AUPRC). No cycle-4 scores are available.

## Dead Ends and Negative Results

**ChemBERTa full fine-tuning with focal loss (exp_gamma_001):** Mean CV AUPRC 0.552 ± 0.077, far below classical baseline (0.676). Fold variance extreme (range 0.459–0.665). Conclusively falsified for this dataset size. Do not retry without substantially more data or strong domain-specific pre-training.

**Frozen ChemBERTa + lightweight classification head (exp_gamma_002):** Mean CV AUPRC 0.415 ± 0.092 — worse than full fine-tuning. The frozen 1,536-dim embeddings (mean + max pooling over 768-dim hidden states) did not capture substrate-relevant patterns with a two-layer MLP head. Falsified; frozen ChemBERTa embeddings alone are insufficient for this task.

**Raw fingerprint concatenation without selection (exp_alpha_003, exp_alpha_005):** Adding Morgan ECFP6 (2048) to Mordred (1200+) and training LightGBM with Optuna yielded 0.625–0.661 AUPRC, both below the descriptor-only baseline. Confirmed: raw fingerprint concatenation is harmful without per-group noise filtering.

**L1/LinearSVC feature selection from Mordred (exp_beta_gpu4_004):** 114 selected features + 4-model stack = 0.677. Competitive with baseline but does not capture the multi-group diversity that SHAP-based fusion does. Closed axis.

**Expanding to 7 feature groups (exp_beta_008 with RDKit2D + Avalon, exp_alpha_009 with 7 groups + GBM meta):** exp_beta_008 at 0.736 (−0.001), exp_alpha_009 at 0.664. Adding RDKit 2D descriptors and Avalon fingerprints as 6th and 7th groups provides marginal or negative returns. Noted in exp_beta_008's result JSON: "Adding more feature groups did not help." Axis appears saturated at 5 groups.

**GBM meta-learner (exp_alpha_009, GBM meta with 7-group features; various others):** Consistently underperformed LR meta-learner variants. Nonlinear meta-learner with 5-column OOF input on ~530 training samples overfits. LR meta-learner is confirmed superior for this task's OOF stack dimension.

**SMOTE oversampling (exp_beta_009):** Tested on champion 130-dim features with SMOTE per fold. Did not beat champion 0.737 (scored as DISCARD in result). Gradient re-weighting via scale_pos_weight already handles the class imbalance adequately.

**Random Subspace Ensemble of 200 LightGBMs (exp_gamma_005, 0.674):** 200 LightGBM models each trained on a random subset of top-100 Mordred SHAP features scored 0.674, below the 4-model stacking champion. Subspace diversification did not match the structural diversification from multi-group feature fusion.

**Tanimoto-kernel SVM on Morgan FP + Mordred ensemble stacking (exp_beta_005):** 0.661 AUPRC in cycle 1 before cross-group SHAP was developed. Superseded by the cross-SHAP approach which uses SVM more effectively as a fifth stacking base model rather than a standalone model.

## Coordination and Team Dynamics

**Three-team structure with clear axis assignments:** Admin formed three teams at startup: Team Alpha (analyst3, gpu1, gpu2) on classical descriptors and feature engineering; Team Beta (analyst1, gpu3, gpu4) on alternative fingerprint fusion strategies; Team Gamma (analyst2, gpu5, gpu6) on neural approaches (ChemBERTa, GNN) and ensemble optimization. This separation prevented redundant experiments and ensured orthogonal coverage. Each analyst prepared strategy documents and proposal scripts that GPU agents executed directly.

**Cross-team knowledge transfer drove the final champion:** The winning architecture (exp_alpha_010) was built directly on Team Beta's exp_beta_007 feature set (160-dim cross-SHAP). Team Alpha's cycle-3 GPU2 agent adopted Beta's feature engineering intact and contributed a new base model (calibrated SVM-RBF), cross-pollinating the two teams' insights into the champion.

**Analyst1 maintained detailed cycle-level memory:** The analyst1 MEMORY.md files contain structured cycle analysis documents that explicitly tracked the champion trajectory, failed hypotheses, and the noise floor estimate (±0.005–0.010 per experiment). Analyst1 identified the "cross-group SHAP" paradigm as the key cycle-2 insight and articulated it as the governing principle for all subsequent proposals.

**Transformer falsification was rapid:** Team Gamma's neural experiments (cycle 1 and cycle 2) were run early and their failure (0.41–0.55) was immediately incorporated into the analyst2 cycle-2 strategy memo, which explicitly stated: "Do NOT circle back to fine-tuning." This pre-empted wasted GPU compute on neural approaches in cycles 3 and 4.

**Cycle 4 ran out of time:** All six cycle-4 experiments were launched but none completed training before the deadline at ~04:06 UTC. The multi-agent system correctly noted this in sessions.jsonl and verified the submission CSV was valid (135 rows) before shutdown. No cycle-4 results were incorporated into the final submission.

**Analyst3 was active, not unassigned:** Unlike the parallel hERG run where analyst3 was idle, analyst3 in this run was active across four sessions preparing detailed proposals and implementation scripts for cycle 3 Team Alpha experiments. However, analyst3's proposals (nested-CV SHAP, isotonic calibration) were not the experiments that ultimately ran on GPU agents in cycle 3; instead, GPU agents ran iterations closer to the beta_007 paradigm.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. All AUPRC values are 5-fold scaffold CV means; the train/test split was not revealed during the run, so the val-to-test gap is unknown. The champion CV score (0.753) may overestimate held-out test performance given the optimization pressure over ~42 experiments.

**Cycle 4 gap:** All six cycle-4 experiments were still running at deadline. Whether multi-seed ensembling, Optuna-tuned SVM parameters, expanded 200-dim feature budgets, or tuned LR meta C would have improved over 0.753 is unknown.

**Feature saturation claim is approximate:** The "5-group saturation" finding rests on a single 7-group experiment (exp_beta_008, −0.001) and one heavily confounded 7-group + GBM-meta experiment (exp_alpha_009, −0.073, where the GBM meta change likely dominates the effect). A clean 7-group experiment with LR meta was not run.

**SVM gain generalizability:** The SVM-RBF gain (+0.016) was measured in one experiment by one agent. The interaction between the calibrated SVM's OOF predictions and the LR meta-learner on this specific 160-dim feature set has not been tested under different random seeds or feature budgets.

**Unexplored axes:**
- Chemprop (D-MPNN GNN): proposed as Team Beta's initial experiment (exp_beta_001) but the cycle-1 result from Team Beta's GPU3 (0.661 for exp_beta_005, a variant) and GPU4 (0.651) suggests GNNs underperformed; however, a well-tuned Chemprop with proper hyperparameter search was not exhaustively tested.
- Broader seed sweeps across the champion architecture.
- SHAP budget ablation: which of the 5 groups contributes most to the gain from beta_006 (3 groups, 0.720) to beta_007 (5 groups, 0.737) was not isolated.
- Larger Mordred SHAP budgets (>50 features) with the LR meta-learner: a 200-dim experiment was planned for cycle 4 but did not complete.
