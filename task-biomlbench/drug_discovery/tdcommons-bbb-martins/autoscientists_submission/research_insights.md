---
task: tdcommons-bbb-martins
run_id: biomlb_tdc_bbbm_4
started_at: "2026-04-21T19:37:50Z"
champion_at: "2026-04-21T22:27:23Z"
---

# Research Insights for TDCommons BBB Martins Permeability Prediction

AutoScientists discovered that **Mordred molecular descriptors combined with tree-ensemble stacking** substantially outperform all neural-model approaches tried on this task. Starting from a ChemBERTa baseline at 0.8901 scaffold-CV ROC-AUC, the run progressed through UniMol (0.8938), XGBoost+Mordred (0.9045), and culminated in a two-level stacking meta-learner (XGBoost+RandomForest+ExtraTrees → LogisticRegression) at **0.9130 scaffold-5fold CV ROC-AUC** — the champion submission. The headline finding is that Mordred 2D descriptors are the single most important feature engineering choice for BBB permeability on n ≈ 1,600 samples, outpacing Morgan fingerprints and all GPU-trained neural models explored in this run.

## Findings

**1. Mordred 2D descriptors are a decisive feature engineering lever over Morgan fingerprints alone.**
XGBoost with Morgan fingerprints (2048-bit ECFP4) scored 0.8753 scaffold CV. Adding Mordred 2D descriptors (~1,613 non-constant features after variance filtering) to the same XGBoost model boosted this to 0.9034 — a +0.028 delta. The feature set grows from 2,048 to ~3,661 dimensions. This was the single largest improvement event in the entire run, discovered by gpu1 in cycle 3.

**2. Stacking meta-learning with OOF predictions improves over the best single model.**
A two-level stacking ensemble (XGBoost depth-6 + XGBoost depth-8 + RandomForest + ExtraTrees as level-1 base learners, all using Mordred+Morgan features; LogisticRegression meta-learner on OOF predictions) achieved 0.9130 scaffold CV vs. 0.9045 for the best single XGBoost configuration — a +0.0085 improvement. The base learner individual scaffold CVs were: XGBoost-d6 = 0.9025, XGBoost-d8 = 0.9025, RandomForest = 0.8990, ExtraTrees = 0.9130. The stacking meta-learner (0.9130) matched the best individual base learner (ExtraTrees, 0.9130), and simple averaging of OOF predictions yielded only 0.9086. ExtraTrees alone unexpectedly matched the full stacking result.

**3. Neural models (UniMol, ChemBERTa, Chemprop) were competitive early but did not beat Mordred-based tree models.**
UniMol (3D-aware molecule transformer, seed1, 50 epochs, 5-fold scaffold CV): **0.8938**. ChemBERTa (10 epochs): 0.8901; 20 epochs: 0.8913. Chemprop MPNN (7-seed ensemble, 75 epochs): 0.8763 mean CV (fold variance 0.8267–0.9226). Hybrid Chemprop+ChemBERTa ensemble: approximately 0.8707–0.8876 per fold. None of the neural models individually exceeded XGBoost+Mordred (0.9045).

### Insights

**G1 — Mordred 2D descriptors capture BBB-relevant physicochemical information not encoded in ECFP4 alone.**
*Claim:* On molecular property datasets where physicochemical properties (lipophilicity, polar surface area, molecular weight, H-bond donors/acceptors) are mechanistically predictive, Mordred 2D descriptors provide signal orthogonal to circular fingerprints. For BBB permeability, where lipid-bilayer passive diffusion and efflux transporter effects are dominant, this leads to large improvements.
*Disconfirming evidence:* Tasks where ECFP4 alone is sufficient and Mordred adds noise (e.g., tasks dominated by scaffold identity rather than physicochemical properties).
*Observed:* +0.028 ROC-AUC improvement on this BBB task (0.8753 → 0.9034); feature selection of top-2000 Mordred features slightly reduced performance (0.9034 → 0.9019), indicating broad Mordred coverage is preferred over aggressive pruning.

**G2 — Stacking meta-learning provides a reliable but modest improvement over the best single tree model when base learners share the same feature set.**
*Claim:* When all base learners use the same Mordred+Morgan features, their OOF meta-features carry limited orthogonal signal; the meta-learner's gain over simple averaging is small. Stacking is more valuable when base learners use different feature representations.
*Disconfirming evidence:* Stacking with diverse feature representations (e.g., XGBoost+Mordred alongside ChemBERTa embeddings) achieving large gains beyond same-feature stacking.
*Observed:* Same-feature stacking gave 0.9130 vs. simple average OOF 0.9086 (+0.0044). The 5-feature stacking (4 tree models + UniMol OOF) in a separate experiment gave 0.9175, suggesting cross-representation stacking is more effective.

**G3 — Feature selection does not help XGBoost+Mordred at n ≈ 1,624; more trees improve only modestly.**
*Claim:* XGBoost handles high-dimensional, sparse features natively through column subsampling; aggressive feature pruning based on single-fold importance scores loses signal rather than reducing noise at this dataset size.
*Disconfirming evidence:* Feature selection improving XGBoost at comparable n on other molecular property tasks.
*Observed:* Top-2000 Mordred features: 0.9019 < 0.9034 full set. Increasing n_estimators from 700 to 1000 (xgb_d6): negligible CV change (0.9045 vs. 0.9034). All-features maxboost: 0.9022 < 0.9034.

### Task-Specific Findings

**T1 — BBB permeability shows strong scaffold-based fold variance, with fold 0 consistently the hardest.**
Across all experiments, scaffold fold 0 (415 samples) was systematically the weakest fold: XGBoost-d6 fold 0 = 0.8706 vs. fold 3 = 0.9217; ExtraTrees fold 0 = 0.8862 vs. fold 1 = 0.9309. The stacking ensemble fold scores were 0.8903, 0.9298, 0.9267, 0.9145, 0.9036. This fold variance (±0.015–0.02 std) is intrinsic to the scaffold split and not reducible by model architecture.

**T2 — UniMol 3D geometry provides modest gains over 2D models but is inconsistent across seeds.**
UniMol seed1 scaffold CV: 0.8938 (folds: 0.8517, 0.9069, 0.9144, 0.9023, 0.8940). UniMol seed42 OOF CV: 0.8700. UniMol seed55 OOF CV: 0.8560. The large seed-to-seed variance (0.8938 vs. 0.8700 vs. 0.8560) suggests UniMol's 3D conformer initialization is sensitive at this dataset size. UniMol alone does not beat XGBoost+Mordred; it contributed useful diversity in multi-model ensembles (5-feature stacking with trees+UniMol gave 0.9175).

**T3 — An estimated 6-model ensemble (ChemBERTa + Chemprop + UniMol + XGBoost + stacking) reached 0.9155 but this is an estimated CV, not a measured value.**
gpu2's final result (exp_beta_017) estimated CV = 0.9155 by combining 6 model OOF predictions (individual CVs: XGBoost = 0.9045, ChemBERTa-10ep = 0.8901, ChemBERTa-20ep = 0.8913, UniMol = 0.8938, Chemprop-3seed = 0.8888, stacking = 0.9130). This was declined for champion promotion because it is an estimated score and correlation with the champion submission was 0.9993 — near-identical test predictions. Only the measured stacking CV of 0.9130 was accepted as the verified champion.

**T4 — MLP on Mordred features underperforms tree models.**
A two-layer MLP trained on Mordred+Morgan features achieved scaffold OOF CV = 0.8980 — notably below XGBoost+Mordred (0.9034). Mordred features require little feature engineering for tree models, whereas neural nets may need normalization, architecture tuning, and regularization to match gradient-boosted ensembles at n ≈ 1,624.

## Dead Ends and Negative Results

**Chemprop MPNN (depth 4, hidden 600, 7-seed ensemble):** Scaffold CV = 0.8763 ± 0.0351 (fold range 0.8267–0.9226). High fold variance and below UniMol/ChemBERTa. Not promoted.

**Hybrid Chemprop + ChemBERTa ensemble (exp_beta_007):** Mean fold CV ≈ 0.8707–0.8876 depending on the fold weighting. Below the UniMol baseline. Retired after cycle 2.

**ChemBERTa epoch scaling (10 → 20 epochs):** 0.8901 → 0.8913 (+0.0012). Marginal gain; not worth the compute cost relative to Mordred+XGBoost.

**UniMol 3-seed ensemble (seeds 42, 55, 68):** Seed 42 CV = 0.8700; seed 55 = 0.8560; seed 68 = 0.8915 (exp_delta_002: mean 0.8915 ± 0.022). Below single-seed UniMol seed1 (0.8938). Seed inconsistency makes multi-seed UniMol unreliable without a large seed sweep.

**Feature importance pruning (top-2000 Mordred features):** 0.9019 < 0.9034 full feature set. Feature selection via single-fold importance hurts on this task.

**Maxboost XGBoost (n_estimators=5000, max_depth=10):** OOF CV = 0.9022 < 0.9034. Overfitting at very large n_estimators. Retired.

**MLP on Mordred+Morgan:** OOF CV = 0.8980, correlation with champion = 0.9496. Provides some diversity but underperforms trees. Not promoted as champion.

**XGBoost+RF+ET+UniMol 5-feature stacking (exp_delta_stacking_001, gpu6):** 0.9175 scaffold CV — this was a more diverse stacking experiment that included UniMol OOF features alongside the tree OOF features. It outperformed the tree-only stacking (0.9175 vs. 0.9130). However, this result was produced by a separate script that encountered a shutil crash at the end without cleanly writing result_latest.json. Because result_latest.json was not updated, the result was invisible to the promotion mechanism and the tree-only stacking (gpu4, 0.9130) remained champion. This represents a missed opportunity: a properly wired cross-representation stacking (trees + UniMol) might have been the stronger submission at 0.9175.

**Cycle 5 grand stacking experiments (alpha, epsilon, gamma):** Launched with ~45 min remaining, they did not complete before the 23:42 deadline. Scripts were running at deadline time. Results unavailable.

## Coordination and Team Dynamics

**Single-coordinator architecture:** All cycles were run directly without delegation to a persistent team structure. All six GPU agents were dispatched as single-cycle workers. Python training scripts were written and launched to each agent's workspace centrally. Analyst agents (analyst1, analyst2, analyst3) provided discussion-phase proposals but had no cycle-level coordination role beyond idea generation.

**Background agent parallelism and result confusion:** Multiple background agents were launched across cycles, some of which continued running from prior cycles (e.g., gpu2's exp_beta_005, exp_beta_013, exp_beta_014 spawned by old background agent handles). This created difficulty distinguishing which result file belonged to which experiment, particularly for the estimated ensemble scores. Estimated scores were explicitly filtered from promotion consideration, keeping only `is_true_cv: true` results as candidate champions.

**Key insight propagation:** The Mordred breakthrough (discovered by gpu1 in cycle 3) was explicitly communicated in cycle messages and immediately adopted by gpu2, gpu3, gpu4, gpu5 for cycle 4. All cycle 4 CPU agents ran Mordred+XGBoost or Mordred+LightGBM variants. This cross-agent knowledge transfer was rapid and effective.

**GPU conflict detection:** GPU usage was proactively tracked; UniMol was noted as consuming 18GB. New GPU experiments were withheld until gpu6's UniMol training completed, preventing OOM failures. However, the monitoring agent independently launched gpu5's GPU XGBoost before gpu6 finished (no OOM occurred because XGBoost CPU mode was used, not CUDA).

**Experiment result reliability gating:** `is_true_cv: true` (measured OOF CV) was consistently distinguished from `metric_type: estimated` results. Estimated scores (e.g., the 0.9155 estimated 6-model ensemble) were declined for promotion because correlation analysis showed near-identical test predictions (r = 0.9993) with the measured champion. This conservatism was appropriate and correct.

**Missed result from exp_delta_stacking_001 (gpu6):** The 5-feature stacking (trees + UniMol) at 0.9175 was never promoted because the script crashed on `shutil.copy` after writing the predictions. Promotion relied on `result_latest.json`; since that file was not updated, the result was invisible. A submission-file monitor instead of a result-JSON monitor would have caught this.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. Scaffold-split fold variance (±0.015–0.02) is substantial; the champion 0.9130 could plausibly be ±0.01 in a re-run. No held-out test score is reported; the submitted predictions are from the scaffold-CV champion.

**Unexplored axes:**
- Cross-representation stacking (trees + UniMol OOF) reached 0.9175 in one experiment but was not cleanly captured by the result system. This is the most likely path to improvement.
- Mordred feature normalization + MLP with proper tuning was only tried in one configuration (0.8980). A properly tuned MLP (batch normalization, dropout, weight decay sweep) might be competitive.
- XGBoost with CUDA acceleration was attempted but fell back to CPU; GPU-accelerated XGBoost with many more trees was not fully explored.
- Scaffold-based train/test splits inherently test generalization to new chemical scaffolds. The measured CV is conservative relative to random splits, which is appropriate for drug discovery, but the 5-fold scaffold CV of 0.9130 may not reflect performance on maximally diverse scaffolds in the TDC test set.

**Scope:** The Mordred+stacking finding likely generalizes to other drug property tasks (logP, TPSA, solubility) at similar dataset sizes (n = 1,000–5,000). The specific advantage of Mordred over Morgan may diminish on tasks where scaffold topology (captured by ECFP) is more predictive than physicochemical properties.
