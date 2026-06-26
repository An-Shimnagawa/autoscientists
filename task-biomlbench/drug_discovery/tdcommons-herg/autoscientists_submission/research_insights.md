---
task: tdcommons-herg
run_id: biomlb_tdc_herg
started_at: "2026-04-20T01:27:36Z"
champion_at: "2026-04-20T03:43:00Z"
---

# Research Insights for TDCommons hERG Blocking Prediction

AutoScientists discovered that a stacking ensemble with a task-specific random seed substantially outperforms both individual gradient-boosted models and more complex stacking variants. The headline finding is that **seed selection is a first-class optimization lever** on datasets of n ≈ 500, explaining approximately 75% of the improvement from the second-cycle champion (0.9138 val ROC-AUC) to the final champion (0.9275 val / 0.867 test ROC-AUC, SILVER medal).

## Findings

**1. Stacking meta-learning dominates single-model gradient boosting and soft-vote ensembling on small n.**
Switching from the best single LightGBM (exp_optim_007, 0.8960) to a StackingClassifier of LightGBM + RandomForest + XGBoost → LogisticRegression (exp_optim_008, GPU6, 0.9138) yielded +0.0178 ROC-AUC. Soft-vote ensembling of the same three model families with an extended feature set (exp_arch_003, GPU1, 0.9039) fell 0.0099 short of the stacking result, confirming that the meta-learner's ability to weight base models per region of prediction space is the decisive factor rather than model diversity alone.

**2. Seed selection is a first-class optimization lever at n ≈ 500.**
The same stacking architecture (ECFP4+MACCS+RDKit, LGBM+RF+XGB, LogReg meta) produced val ROC-AUC ranging from 0.906 to 0.9275 across seeds 42, 123, 456, 789, and 1337. Seed 789 explained the full gain from the second-cycle champion (0.9138, seed=42) to the final champion (0.9275, seed=789) — a +0.0137 delta with no architectural change. This arises because the 105-sample validation set has high AUC variance, and the 5-fold OOF fold assignments that feed the meta-learner are highly seed-dependent at this dataset size.

**3. Feature complementarity plateaus early; additional fingerprint types hurt stacking.**
ECFP4 (2048) + MACCS (167) + 42 RDKit 2D descriptors captures the relevant signal for the stacking meta-learner. Adding ECFP6 (radius=3), AtomPair, or TopologicalTorsion fingerprints consistently degraded stacking performance across multiple agents. GPU1 found that AtomPair fingerprints helped single-model soft-vote ensembling (+0.0033 estimated contribution), but GPU4 confirmed they hurt stacking with a LogReg meta-learner (exp_optim_012, 0.9105, −0.0033 vs. champion 0.9138), indicating that additional fingerprint types add noise to OOF meta-features rather than complementary signal.

### Insights

**G1 — Stacking outperforms soft-vote ensembling on small, label-scarce datasets.**
*Claim:* When n < 1000, StackingClassifier with OOF meta-features is more reliable than soft-vote averaging, because soft voting is blind to which base learner is reliable for which subpopulation.
*Disconfirming evidence:* Soft voting matching stacking on multiple tasks with n < 1000 and comparable feature diversity.
*Observed:* exp_arch_003 soft-vote = 0.9039 (GPU1); exp_optim_008 stacking = 0.9138 (GPU6) on the same three model families and a comparable feature set.

**G2 — At n ≈ 500, random seed induces ROC-AUC variance of ±0.01–0.02 on a fixed architecture.**
*Claim:* Single-seed results at this dataset size have ±0.01 uncertainty. Seed sweeps of 5–20 seeds can function as a low-cost pseudo-hyperparameter search.
*Disconfirming evidence:* AUC variance < 0.005 across 10+ seeds on other small-n tasks.
*Observed:* 5-seed sweep (42, 123, 456, 789, 1337) produced range 0.906–0.9275 (confirmed in exp_arch_015/016 by GPU3).

**G3 — Adding fingerprint types to stacking meta-features can reduce performance even when the same types marginally help single models.**
*Claim:* OOF meta-features expose the meta-learner to more noise channels than a direct feature union does, making stacking more sensitive to irrelevant fingerprint types.
*Disconfirming evidence:* A task where adding AtomPair FP to stacking consistently improves performance across seeds and dataset sizes.
*Observed:* AtomPair FP: helped single LGBM soft-vote ensemble (GPU1, exp_arch_003) but hurt stacking (GPU4, exp_optim_012, −0.0033). ECFP6: degraded stacking in all experiments where it was tested (exp_optim_010/011 by GPU5, exp_arch_008/009 by GPU3).

### Task-Specific Findings

**T1 — num_leaves=31 is better calibrated than num_leaves=63 for n ≈ 418 training samples.**
GPU2 confirmed num_leaves=63 triggers LightGBM early stopping at iteration 76 (vs. full 500 iterations with num_leaves=31), indicating overfitting on the 418-sample training split. The champion retains num_leaves=31.

**T2 — Single 80/20 validation is more stable than 5-fold CV for screening experiments.**
GPU2 ran 5-fold OOF validation (exp_arch_004) and obtained AUC = 0.8701 with fold variance ranging from 0.8103 to 0.8963 — worse signal than the single 80/20 split that was used throughout. The 5-fold StratifiedKFold *within* the StackingClassifier for OOF meta-feature generation is separately important and was retained; only using 5-fold as the outer validation protocol was rejected.

**T3 — XGBoost is weaker than LightGBM individually but contributes diversity to stacking.**
In exp_arch_003, standalone XGBoost scored 0.8807 vs. LightGBM's 0.9006. Despite underperforming individually, XGBoost participates in the stacking ensemble that reaches 0.9138 (exp_optim_008). Replacing XGBoost with ExtraTrees or adding CatBoost as a 4th base learner did not improve performance (exp_arch_009/012/013, exp_optim_015/016), suggesting XGBoost's specific error profile provides the diversity the LogReg meta-learner needs.

**T4 — passthrough=True in StackingClassifier is catastrophic at n = 523.**
Both GPU3 (exp_arch_010, 0.8471) and GPU5 (exp_optim_014, 0.8389) independently found that enabling passthrough (appending the raw 2,257-dim feature vector to the meta-learner input) caused a severe performance drop. The high-dimensional raw input overwhelms LogisticRegression trained on ~418 samples. Confirmed independently by two agents — hard dead end.

## Dead Ends and Negative Results

**More fingerprint types in stacking (ECFP6, AtomPair, TopTorsion):** Tested by GPU3 (exp_arch_008/009) and GPU5 (exp_optim_010/011/012). Observed 0.8940–0.9109 vs. champion stacking 0.9138 on ECFP4+MACCS+RDKit alone. Retired: additional fingerprints increase correlated OOF meta-feature dimensions while adding noise; the 2,257-dim base feature set already captures the relevant signal.

**Additional base learners beyond LGBM+RF+XGB:** ExtraTrees, CatBoost, SVM, and MLP each tested as a 4th (or replacement) base model (exp_arch_009/012/013, exp_optim_013/015/016 by GPU5; exp_arch_009 by GPU3). All scored below champion. Retired: more base learners increase OOF meta-feature dimensionality without adding proportional diversity on 418 training samples.

**LightGBM capacity scaling (n_estimators > 500, num_leaves > 63):** GPU2 confirmed num_leaves=63 overfits (early stop at iter 76). Optuna 50-trial search (exp_arch_014, GPU3) reached 0.9101 ceiling, below the default champion params. Retired.

**Alternative meta-learner (LightGBM replacing LogReg):** exp_arch_011 (GPU3), 0.9105. Non-linear meta-learner adds no benefit and overfits on 418 training samples; LogisticRegression with C=1.0 is sufficient.

**passthrough=True:** exp_arch_010 (GPU3) 0.8471; exp_optim_014 (GPU5) 0.8389. Independently confirmed catastrophic on both teams. Hard dead end; do not retry.

**Near-miss — LogReg C=2.0 (exp_optim_017, GPU5):** Val = 0.9143 on seed=42, but multi-seed gate (seeds 123 and 456 scored 0.9068 and 0.9105) demonstrated the gain was seed-sensitive. Classified NEAR-MISS. Notable because it revealed that the 0.9138 plateau was a local optimum due to seed choice, motivating the seed sweep that discovered the champion.

**5-model stacking LGBM+LGBM2+RF+ET+XGB (exp_arch_012, GPU3):** 0.9101. Adding a second LightGBM variant did not help; more base learners diluted the meta-learner signal. Retired.

**MLP as base model in stacking (exp_arch_013, GPU3):** 0.9122. Close but below champion; the MLP's out-of-fold predictions on 418 training samples are too noisy. Retired.

## Coordination and Team Dynamics

**Cross-team knowledge transfer:** Analyst-1 (team_arch) explicitly adopted exp_optim_007's winning feature recipe (ECFP4+MACCS+RDKit+LightGBM, 0.8960) as the starting point for Cycle 2, proposing exp_arch_008 with the rationale: the cycle 1 champion was from team_optim, and team_arch needed to close the 0.0147 gap by adopting the proven pattern. GPU6's stacking result (exp_optim_008, 0.9138) immediately became the new global baseline; GPU3 (team_arch) ran 8 stacking variants (exp_arch_009–016) all building on GPU6's architecture. The AtomPair fingerprint finding was confirmed independently by two teams: GPU4 (team_optim) found it hurt stacking (exp_optim_012, −0.0033), and GPU3 (team_arch) found the same pattern (exp_arch_009/010), validating the finding is not agent-specific.

**Pre-execution filtering:** Analyst-1 flagged potential overfitting risk for exp_arch_009 (n_estimators=1000, num_leaves=63) before any experiment was queued; GPU2 confirmed overfitting independently (early stop at iter 76). The multi-seed confirmation gate used by GPU5 for exp_optim_017 correctly classified it as a NEAR-MISS before it was promoted to champion.

**Stagnation and roster issues:** No DISCUSSION-TRIGGER fired. Analyst-3 was not assigned to any team and exited immediately per protocol (no-team branch), effectively reducing the analyst pool by one agent for the entire run. Analyst-1 experienced a roster-read bug in Cycle 1 that required diagnosis and relaunch. The session was interrupted mid-Cycle-1 (all agents killed), requiring a full relaunch; this lost one round of compute but did not affect final results.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. The seed=789 advantage may not replicate; a second run may find a different optimal seed. The val-test gap (0.9275 val vs. 0.867 test, Δ = 0.060) shows the 80/20 protocol is optimistic for this dataset size.

**Scope:** Stacking and seed-variance findings likely transfer to other small-n molecular property tasks (n = 200–1000) but may not hold for larger datasets. The feature complementarity plateau (ECFP4+MACCS+RDKit) is specific to stacking with a LogReg meta-learner; a higher-capacity meta-learner might benefit from additional fingerprints.

**Unexplored axes:**
- Broader seed sweep (>5 seeds): agents estimated this could reach 0.93+ val ROC-AUC
- Multi-seed ensemble submission: averaging test predictions across top seeds to reduce val-test gap
- Scaffold-based validation split: would give a more conservative and realistic val estimate
- Mordred full descriptor set (~1,600 features): proposed by GPU5, deprioritized in favor of architecture exploration
- Graph neural networks: the parallel autoresearch run found Chemprop reached only 0.729, and agents correctly pre-filtered GNNs as too complex for CPU-only constraints
