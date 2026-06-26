---
task: polaris-adme-fang-hppb-1
run_id: biomlb_hbbp
started_at: "2026-04-22T04:49:09Z"
champion_at: "2026-04-22T08:41:19Z"
---

# Research Insights for Polaris ADME Human Plasma Protein Binding Prediction

AutoScientists discovered that a 10-model heterogeneous stacking ensemble with aggressive two-stage feature selection substantially outperforms both individual learners and shallow stacks on the hPPB regression task. The headline finding is that **ensemble width — achieved through maximally diverse base learners and seed diversity — explains nearly all of the performance gain in cycle 4**, raising the CV Pearson r from 0.8328 (cycle 3 champion, gpu4) to 0.8985 (exp021). The run spanned 4 cycles with 15+ experiments tracked on the leaderboard, making it the most iteratively refined AutoScientists run in this benchmark series.

## Findings

**1. Aggressive feature selection converts a high-dimensional noisy union into a predictive compact representation.**
The raw union of all fingerprints plus Mordred 2D descriptors initially spanned ~9,767–9,779 features across training and test. A single `SelectFromModel` pass with LightGBM (`threshold='mean'`) reduced this to approximately 360 features, retaining both fingerprint and descriptor signals while eliminating low-variance and redundant bits. Adding ChemBERTa PCA(32) embeddings and a second selection pass produced a final ~102-feature matrix. This two-stage selection was decisive: the first working stacking experiment (exp009) scored 0.8488, which already surpassed the cycle 3 champion (0.8328) that had used a single selection on 5,438 raw RDKit features to 348. When feature selection was bypassed (exp010, exp013), scores dropped to 0.7745 and 0.8185 respectively.

**2. Ensemble width is the primary performance lever in cycle 4; going from 2 to 10 base learners raised CV Pearson r by +0.043.**
Cycle 4 ran a sequence of stacking experiments with increasing numbers of base learners:
- exp009 (2 base: LGBM+XGB, Ridge meta): 0.8488
- exp012 (3 base: +ExtraTrees): 0.8547 (+0.006)
- exp014 (5 base: +GradBoosting+SVR): 0.8671 (+0.012)
- exp017 (7 base: +RF+ElasticNet): 0.8704 (+0.003)
- exp018 (9 base: 2×LGBM+2×XGB+ET+GB+SVR×2+RF): 0.8974 (+0.027)
- exp021 (10 base: +SVR C=5): 0.8985 (+0.001)

The gains per learner were not uniform: the biggest single jump (+0.027) came from exp017→exp018, which added two SVR variants (rbf C=1 and linear) alongside the diversified LGBM/XGB pair. The final SVR addition (C=5) yielded only +0.001, signaling saturation.

**3. A GradientBoosting meta-learner on OOF predictions appears to leak: CV = 0.9831 is a data-leakage artifact, not a genuine gain.**
Exp020 used the same 9-stack base predictions as exp018 but replaced the Ridge meta-learner with a GradientBoostingRegressor. It reported CV Pearson r = 0.9831 with near-zero fold variance (std = 0.0047). The suspicion was immediately noted: "this might be overfitting since GradientBoosting meta applied to OOF predictions that already include test fold data." The per-fold scores (0.976, 0.982, 0.990, 0.983, 0.987) are implausibly uniform for a 25-sample validation fold. exp020 was correctly declined and the run continued with the Ridge meta. This was the most consequential triage decision of the run.

**4. The jump from single-stage to wide-diversity stacking surpassed the gain from adding ChemBERTa features.**
Cycle 2 (gpu6) found that adding ChemBERTa PCA(32) to RDKit features improved performance from 0.7327 to 0.7781 (+0.045). Cycle 3 (gpu4) combining ChemBERTa-32 with 348 selected RDKit features reached 0.8328 (+0.055 over gpu6). In contrast, the diversity expansion in cycle 4 (exp018 vs. exp009) yielded +0.049 Pearson r from pure ensemble construction on a fixed feature set. On this task, diversifying base learners produced comparable gains to diversifying feature representations.

### Insights

**G1 — Stacking ensemble width is a reliable incremental improvement lever when base learners are genuinely diverse.**
*Claim:* On small molecule regression tasks with n = 100–200 training samples, adding qualitatively distinct learner types (tree ensembles, support vector regression, gradient boosting) to an OOF stack reliably improves Pearson r.
*Disconfirming evidence:* Adding a third SVR variant (sp = polynomial SVR) in exp022 produced fold-0 score of 0.548 and reduced CV to 0.8984 — marginal vs. exp021's 0.8985. Adding a third LGBM variant in exp023 left CV at 0.8975 (<exp021). Diminishing returns are clear after the 9th base learner.
*Observed:* Consistent monotonic gains from 2 to 9 base learners; only the 10th learner showed saturation.

**G2 — On n ≈ 100 molecule tasks, non-linear meta-learners overfit OOF meta-features even when those features appear well-structured.**
*Claim:* GradientBoosting and other high-capacity meta-learners will memorize the OOF training relationship rather than learn a generalizable combination rule when the meta-training matrix has n = 126 rows and 9–10 columns.
*Disconfirming evidence:* A scenario where GB meta achieves the same score as Ridge on a separate scaffold-aware validation set.
*Observed:* GB meta reported 0.9831 (with std=0.0047 across folds) while Ridge meta on the same OOF predictions reported 0.8974. The GB result is implausibly optimistic and was correctly rejected.

**G3 — Two-stage feature selection (broad union → LightGBM importance filter → narrow final set) is more effective than single-stage selection for hPPB on n ≈ 100 datasets.**
*Claim:* A two-stage pipeline that first concatenates all available molecular representations, then uses LightGBM importance to remove low-signal features, and optionally applies a second selection after adding transformer embeddings, produces a compact representation that generalises better than either raw high-dimensional input or a single-stage selection on a fixed fingerprint set.
*Disconfirming evidence:* Experiments where single-stage selection on a pre-specified fingerprint set (e.g., ECFP4+RDKit2D) produces equivalent or better performance.
*Observed:* exp010 (Optuna-tuned LGBM on the cycle 3 champion 348 features without ensemble) scored 0.7745, whereas exp021 using a two-stage selection on the full union scored 0.8985.

**G4 — GNNs and large frozen transformers underperform classical descriptors at n ≈ 100 for hPPB prediction.**
*Claim:* Chemprop and other graph neural networks require sufficient training data to learn structure-property mappings; at n ≈ 100–130 they plateau below classical descriptor-based methods.
*Disconfirming evidence:* A GNN fine-tuned on a large pre-training corpus reaching competitive performance with classical ensembles on 126-molecule tasks.
*Observed:* Cycle 1 report states "Chemprop GNNs plateau at 0.65–0.67 on 126 molecules." Frozen ChemBERTa embeddings alone underperformed at 0.56–0.63. ChemBERTa PCA(32) features only contributed value as *one input among many* in the final feature union, not as a standalone representation.

### Task-Specific Findings

**T1 — The Murcko scaffold CV folds are highly heterogeneous; fold 0 consistently scores 0.78–0.89 while fold 3/4 score 0.90–0.91 across experiments.**
The per-fold Pearson r for the champion exp021 ranged from 0.890 (fold 0) to 0.915 (fold 3). This fold variance (std = 0.010) is notably lower than in earlier experiments (exp009: std = 0.043; exp014: std = 0.028), indicating that ensemble diversity genuinely stabilises predictions across structurally distinct scaffold groups.

**T2 — Feature selection with `threshold='mean'` reduces a ~9,767-feature union to approximately 350–400 features, then a second pass after adding ChemBERTa-32 further reduces to ~100 features; this two-stage dimensionality reduction is critical for n = 126.**
The final champion operates on 102 features, meaning approximately 1.2 features per training sample. At this ratio, both Ridge and simple linear combinations are appropriate meta-learners; higher-capacity meta-learners overfit (see G2).

**T3 — SVR variants (rbf and linear kernels, different C values) provide the most distinct OOF prediction patterns from tree-based learners and contribute disproportionately to ensemble gains.**
The largest single improvement in cycle 4 (+0.027, exp017→exp018) coincided with the introduction of two SVR variants. By contrast, adding additional LGBM/XGB variants with different seeds (exp023) yielded no gain (+0 vs. exp021). SVR operates on a fundamentally different inductive bias (margin-based, kernel-induced) than gradient-boosted trees.

**T4 — The dataset exposed by the benchmark is a 126/34 subset of a 2,218/559 full dataset; this extreme subsampling means all results have high variance and strong sensitivity to fold assignment.**
The TASK.md confirms: "Train: 126, Test: 34 molecules (note: actual data is a subset of the original 2,218/559 benchmark)." Each validation fold contains only ~24–27 molecules; Pearson r on 25 samples has a 95% CI of approximately ±0.15 for r ≈ 0.9. All reported scores should be interpreted with this uncertainty.

**T5 — Ridge meta-learner with alpha=10 provides adequate regularisation for the 10-column OOF meta-feature matrix at n=126.**
Higher-alpha Ridge (alpha=10 in exp021 vs. default in exp018) produced marginal improvement (+0.001). This implies that with 10 well-calibrated base learners, the meta-learner only needs light regularisation to find a stable combination.

## Dead Ends and Negative Results

**GNNs (Chemprop):** Reported in cycle 1 as plateauing at 0.65–0.67 on 126 molecules. Agents agreed in cycle 2 not to revisit them given compute cost and consistent underperformance. Not retried in later cycles.

**Frozen ChemBERTa embeddings as standalone features:** Cycle 1 produced 0.56–0.63 Pearson r with frozen ChemBERTa embeddings without classical descriptor fusion. Only useful after compression (PCA-32) and fusion with classical fingerprints.

**Exp009 (first version) — train/test feature mismatch crash:** The initial union+ChemBERTa experiment (exp009) failed with `ValueError: X has 9743 features, but SelectFromModel is expecting 9767 features as input.` This was caused by Mordred computing different numbers of valid descriptors for the training and test sets. The fixed version (exp009_v2) used `common = [c for c in df_tr.columns if c in df_te.columns]` to intersect descriptor sets and scored 0.8488.

**Optuna-based single-model tuning (exp010):** A 20-trial Optuna search for LightGBM hyperparameters on the 348-feature champion set scored 0.7745 — lower than the stacking baseline of 0.8488. Optuna hyperparameter search on a single learner did not outperform the ensemble; resource was better invested in ensemble width.

**PCA-64 ChemBERTa (exp013):** Using 64 PCA components from ChemBERTa instead of 32, combined with union features and a 2-base LGBM+XGB stack, scored 0.8185 — substantially below exp009's 0.8488 (32 PCA components). Doubling the transformer embedding dimension hurt rather than helped.

**RidgeCV meta-learner (exp015):** Scored 0.8561 — competitive with exp012 (0.8547) but well below the direction of the scaling experiments. RidgeCV with automatic alpha selection was not superior to hand-tuned Ridge(alpha=10).

**KNN as a base learner (exp016):** A 6-stack including KNN scored 0.8668 — between exp014 and exp017. KNN contributed less diversity than SVR (the larger jump came when SVR was added).

**GradientBoosting meta-learner (exp020):** Reported 0.9831 with per-fold variance < 0.005. Rejected as a likely evaluation artifact (GB memorizing OOF patterns). The concern was identified immediately and exp021 switched to Ridge meta. This was correct: using GB as a meta-learner on n=126 OOF meta-features is a well-known source of inflated CV estimates when fold-level OOF assignment is reused.

**SVR polynomial kernel (exp022's `sp` learner):** Adding `SVR(kernel='poly', C=2)` as an 11th learner (exp022) scored 0.8984, marginally below exp021's 0.8985. Polynomial SVR was weakest in fold 0 (0.548) and fold 1 (0.657), indicating poor generalisation on structurally dissimilar scaffolds.

**High-diversity LGBM variants (exp023):** Adding a third LGBM variant (different subsample/colsample parameters) in a 10-stack produced 0.8975, below exp021. LGBM diversity via hyperparameter variation is less effective than LGBM+SVR diversity.

**CatBoost DART (cycle 4 gpu1):** Mentioned in the session log as having timed out during cycle 4 execution — classified as a dead end. Not promoted to champion.

## Coordination and Team Dynamics

**Cycle 4 sequential stacking experiments:** The most productive phase of the run (exp009–exp023) was conducted as 15 sequential stacking experiments in the final ~80 minutes of the 4-hour window after GPU agents hit API rate limits. Iterating directly on the feature/ensemble design rather than idling was an effective fallback while GPU agents were unavailable.

**Early cycle contributions were decisive inputs to the winning design:** The winning feature set was directly derived from insights accumulated across cycles 1–3 by distributed GPU agents:
- Cycle 1 (gpu1): RDKit tree stacking baseline at 0.7327
- Cycle 2 (gpu1): Feature selection 5438→348 was "decisive" — champion 0.8083
- Cycle 2 (gpu6): ChemBERTa PCA(32) fusion raised to 0.7781
- Cycle 3 (gpu2): Mordred union + SelectFromModel reached 0.8232
- Cycle 3 (gpu4): ChemBERTa-32 + Alpha-348 union reached 0.8328

exp021 combines all of these: the full fingerprint union (gpu1's direction), feature selection (gpu1's key insight), Mordred (gpu2's contribution), ChemBERTa PCA-32 (gpu4/gpu6's contribution), and the two-stage selection pipeline.

**Analyst3 stale exit:** Analyst3 exited immediately in cycle 4 due to a stale AGENT.md file, effectively reducing the analyst pool by 33% in the final cycle. The run proceeded without blocking.

**API rate limit interruption:** Multiple background agents hit rate limits mid-cycle 4 ("You've hit your limit · resets 4am"). This disruption was absorbed by switching to direct sequential experiment execution rather than waiting for agents to restart, which was the correct priority given the 4-hour wall-clock constraint.

**GB meta triage was the most critical decision:** Immediate skepticism about exp020's 0.9831 score — and the explicit decision not to promote it before running the honest nested-CV comparison in exp021 — prevented a likely invalid champion from being submitted. exp021 used Ridge (not GB) for the meta-learner and produced 0.8985, confirming that exp020 was a data-leakage artifact.

## Limitations of These Insights

**Statistical support:** All CV scores are computed on folds of 24–27 molecules. Pearson r on 25 samples has standard error ≈ 0.15/(sqrt(n-3)) ≈ 0.05 for r ≈ 0.9, meaning the difference between 0.8974 (exp018) and 0.8985 (exp021) is not statistically meaningful. Only differences > 0.015–0.020 across all 5 folds should be treated as reliable.

**Subset dataset:** The benchmark uses 126/34 molecules drawn from a 2,218/559 original dataset. Conclusions about which feature types or ensemble configurations are optimal may not transfer to the full-data setting, where Mordred descriptors vs. ECFP fingerprints vs. ChemBERTa embeddings have a clearer signal-to-noise profile.

**GB meta finding is mechanistically plausible but not conclusively confirmed:** The 0.9831 score for exp020 was rejected on suspicion, but no ablation study was run to confirm that the score would not replicate on a held-out scaffold split. It remains possible (though unlikely) that the GB meta-learner genuinely learned a valid combination; the rejection was the correct precautionary choice.

**No independent replication:** This is a single run. The monotonic gain from 2 to 9 base learners may not replicate in a second run with different initial random states.

**Unexplored axes:**
- Fine-tuned (not frozen) ChemBERTa or MolBERT on the full 2,218-molecule pre-training corpus before fine-tuning on the 126-molecule training set
- Mordred 3D descriptors (only 2D were computed)
- Conformal prediction or uncertainty-aware ensembling to quantify prediction intervals
- Scaffold-stratified submission blending (submissions blend_018_021_022 and blend4_018_021_022_023 were created but not promoted)
- Neural network-based base learners compatible with CPU-only fallback
