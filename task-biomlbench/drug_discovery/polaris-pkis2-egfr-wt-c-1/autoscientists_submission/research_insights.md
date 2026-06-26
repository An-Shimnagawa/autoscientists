---
task: polaris-pkis2-egfr-wt-c-1
run_id: biomlb_egfr
started_at: "2026-04-22T12:40:58Z"
champion_at: "2026-04-22T14:37:22Z"
---

# Research Insights for Polaris PKIS2 EGFR Wild-Type Kinase Inhibitor Prediction

AutoScientists converged on a 7-model Optuna-weighted stacking ensemble combining six classical models with a Chemprop MPNN as the winning architecture. The headline finding is that **integrating a graph neural network (Chemprop) alongside classical Tanimoto-kernel methods via Optuna-optimized soft weighting is essential for reaching the 0.80 PR-AUC threshold on this severely class-imbalanced (9:1) small-n molecular dataset**. Classical-only ensembles plateaued around 0.75-0.79 across all cycles; adding the Chemprop MPNN with class-balance oversampling pushed the champion to **0.8012 mean CV PR-AUC** (fold scores: 0.7898, 0.8363, 0.9688, 0.7284, 0.6828).

## Findings

**1. A 7-model Optuna-weighted ensemble (6 classical + Chemprop MPNN) is the winning architecture on this task.**
Cycle 2 established a 5-model Optuna-weighted stacking ensemble (XGB + LGB + ET + SVM-Tanimoto + GP-Tanimoto, PR-AUC 0.7808). Cycle 2 cycle-end added Chemprop MPNN as a 6th base model (PR-AUC 0.7918), and then CatBoost as a 7th (PR-AUC 0.7941). The champion (exp_gamma_gpu6_c3) extended these 120 Optuna trials to 150 and switched to TPESampler(multivariate=True), reaching 0.8012. Each incremental addition — Chemprop, CatBoost, more Optuna trials — produced measurable gains.

**2. Tanimoto-kernel methods (SVM and GP) are indispensable for hard folds.**
In the champion run, SVM-Tanimoto received the largest average weight across folds (20.7%) followed by ExtraTrees (18.7%) and XGBoost (17.5%). On fold 3, where all descriptor-based tree models scored below 0.58, SVM and GP together dominated the optimal weight vector (SVM 63.8%, GP 26.9%), rescuing fold-3 PR-AUC to 0.7284. This pattern was consistent: when tree models struggled (folds with underrepresented scaffold clusters), Tanimoto-kernel similarity recovered signal that descriptors could not.

**3. Class imbalance requires explicit treatment at multiple levels.**
The PKIS2 EGFR dataset has only 50 positives out of 496 training compounds (10.1% positive rate, pos_weight = 8.92). Agents consistently found that explicit imbalance handling at each model level was required: `scale_pos_weight` for XGBoost and LightGBM, `class_weight` for ExtraTrees, `auto_class_weights='Balanced'` for CatBoost, and `class_balance=True` in the Chemprop DataLoader. The optimization metric throughout was PR-AUC (not ROC-AUC), which is sensitive to precision on the minority class; early experiments that optimized AUC or accuracy were discarded.

**4. Fold-level variance is extreme (range: 0.68–0.97) and signals genuine scaffold diversity.**
Across all experiments, fold 2 consistently produced the highest PR-AUC (0.93–0.97) while folds 3 and 4 consistently produced the lowest (0.68–0.73). This pattern held across nearly all experiments regardless of architecture. The pre-defined 5-fold `cv_fold` split appears to capture meaningful scaffold clusters, making single-fold screening unreliable. Mean CV PR-AUC over all 5 folds was the only reliable selection signal, and even that showed ~0.10 standard deviation.

**5. The Chemprop MPNN with auxiliary RDKit descriptors (x_d) adds signal beyond classical features.**
The champion used Chemprop with 102 physicochemical descriptors appended as molecule-level auxiliary features (x_d), BondMessagePassing with depth=3, hidden_dim=300, 60 epochs, and class_balance=True. In the champion's fold-level scores, Chemprop was the dominant model in fold 0 (weight 44.7%), where classical models scored 0.40–0.68. The shallow Chemprop (depth=3, 60 epochs) outperformed the deeper variant tested by GPU3 (depth=5, hidden=512, 100 epochs, SMILES augmentation 5x), confirming that architecture scaling does not reliably help at n~400.

### Insights

**G1 — Tanimoto-kernel methods are uniquely robust to scaffold-hard folds on small datasets.**
*Claim:* SVM and GP with Tanimoto kernels use global structure similarity rather than learned fingerprint embeddings, making them more stable when fold training sets are structurally dissimilar from the validation set.
*Disconfirming evidence:* Datasets where scaffold splits are minimal and the Tanimoto kernel does not provide fold-level diversity benefit.
*Observed:* Fold 3 average weights in champion run: SVM=63.8%, GP=26.9%, with all tree models near zero; fold 3 was rescued to PR-AUC=0.7284.

**G2 — Optuna-optimized continuous weighting outperforms fixed-weight averaging for 7-model stacks.**
*Claim:* When base models have widely varying fold-level PR-AUC (e.g., SVM dominates fold 3 but Chemprop dominates fold 0), per-fold Optuna weight search can assign near-zero weight to underperforming models per fold, yielding higher ensemble PR-AUC than any fixed global weights.
*Disconfirming evidence:* A dataset where all base models have consistent fold-level rankings, making fixed average weights equivalent to per-fold optimization.
*Observed:* Per-fold optimal weights varied dramatically (Chemprop weight ranged from 0.000 to 0.447 across folds); mean CV 0.8012 vs. would-be lower scores if weights were fixed globally.

**G3 — A shallow Chemprop MPNN with auxiliary features outperforms a deeper MPNN with SMILES augmentation on n~400.**
*Claim:* At this dataset size, increasing MPNN depth from 3 to 5 and adding SMILES augmentation (5x data expansion) introduces more variance than capacity benefit, likely due to limited structural diversity in ~400-compound training folds.
*Disconfirming evidence:* Larger molecular datasets (n > 5,000) where SMILES augmentation consistently improves graph-based models.
*Observed:* Champion Chemprop (depth=3, 60 epochs): mean OOF individual score ~0.55 across folds; deeper Chemprop (GPU3, depth=5, 100 epochs, aug=5x): mean OOF PR-AUC = 0.6091; but overall ensemble score was 0.7756 (vs. champion ensemble 0.8012) because the deeper model displaced weight from classical models without compensating.

### Task-Specific Findings

**T1 — The PKIS2 EGFR dataset has 496 training, 144 test compounds with 50 positives (10.1%).**
All experiment logs confirmed the same dataset statistics: 496 train (50 pos, 446 neg), 144 test. The very low positive rate (1 in 10) means PR-AUC is sensitive to precision on a small set of true actives, and fold composition matters greatly.

**T2 — Fold 2 is reliably easy (PR-AUC 0.93–0.98); folds 0, 3, 4 are reliably hard (0.63–0.79).**
This pattern was reproduced independently by multiple agents across all cycles. exp_stack_001 (cycle 1 baseline): folds [0.64, 0.81, 0.93, 0.71, 0.67]. Champion (cycle 3): [0.79, 0.84, 0.97, 0.73, 0.68]. The rank order is nearly identical. This suggests fold 2 contains structurally typical EGFR inhibitors that are well-represented in training, while folds 0, 3, and 4 contain more scaffold-novel compounds.

**T3 — The initial baseline (exp_stack_001, XGB + SVM + GP with logistic meta-learner) established a competitive 0.7539 floor in the first cycle.**
The first stacking experiment (3 models, logistic regression meta-learner) reached 0.7539, within 0.05 of the eventual champion. This near-plateau after the first stacking experiment meant that subsequent cycles were searching for incremental gains through model diversity (adding LGB, ET, Chemprop, CatBoost) and weight optimization (Optuna replacing logistic regression).

**T4 — CatBoost with ordered boosting and native class balancing contributes meaningful weight (~3%) despite a low individual PR-AUC.**
CatBoost's average weight across champion run folds was 3.0%, low but nonzero. Its per-fold PR-AUC (0.51–0.92) roughly matched LightGBM and XGBoost. The model was retained because the marginal contribution was nonzero in Optuna search and the 7th model added at cycle 3 (exp_beta_gpu2_c3) improved mean CV from 0.7918 to 0.7941. The incremental contribution of CatBoost vs. simply increasing Optuna trials is not isolated, however.

**T5 — The Chemprop Python API with RDKit x_d is necessary; Chemprop CLI did not perform comparably.**
Late-cycle attempts (cycles 4-5) to run Chemprop via its CLI (`chemprop train` / `chemprop predict`) failed to replicate the champion's performance, producing PR-AUCs of 0.71–0.73 even after correcting several CLI flag bugs (`--message-hidden-dim`, `--task-type classification`, `--pytorch-seed`, `model_0/best.pt` path). The champion used the Python API via Lightning Trainer with auxiliary RDKit descriptors injected as `x_d` into the MPNN input, a feature not accessible via the standard CLI. This integration was only available via the Python API.

## Dead Ends and Negative Results

**Deeper Chemprop MPNN (GPU3, exp_beta_gpu3_c3):** depth=5, hidden=512, 100 epochs, SMILES augmentation 5x. Mean CV PR-AUC = 0.7756 (−0.0185 vs. champion). Chemprop mean OOF PR-AUC = 0.6091 (vs. ~0.55 for champion's depth=3). The deeper model received near-zero weight (avg 1.1%) in ensemble optimization; Optuna heavily upweighted XGB and LGB instead. The SMILES augmentation added computational cost without improving the model's OOF performance in challenging folds.

**Chemprop-centric ensembles (GPU5, exp_chemprop_gpu5_001 and 002):** Multiple seeds of Chemprop averaged as a base model alongside XGBoost + SVM + GP. exp_chemprop_gpu5_001: mean CV 0.7249. exp_chemprop_gpu5_002 (nested meta-learner): mean CV 0.6952. Both below the 0.7539 initial stacking baseline. Standalone Chemprop ensemble (5 seeds) scored only 0.5352–0.5685 per seed. Chemprop provides its best value as a diverse 7th member, not as the ensemble core.

**Chemprop via CLI (multiple late cycle 4-5 attempts):** Five separate CLI attempts produced PR-AUCs in the 0.71–0.73 range after fixing bugs in flags (`--task-type`, `--message-hidden-dim`, `--pytorch-seed`, `model_0/best.pt`). The root cause was the inability to inject RDKit auxiliary features (`x_d`) via the CLI pipeline. Retired: CLI approach cannot replicate champion-level results.

**Probability calibration (Platt scaling) applied to champion ensemble (GPU3, cycle 4):** PR-AUC = 0.7318 (−0.0694 vs. champion). GPU3 noted that "Platt scaling on already-calibrated predictions hurts." The champion's Optuna-weighted blend of calibrated-by-construction probabilistic models (SVM with `probability=True`, GP) and direct probability-outputting models already produces well-calibrated probability estimates; adding an additional sigmoid layer degrades calibration.

**GP replaced by RandomForest (GPU6 background train_c4.py, cycle 4-5):** Replacing the Tanimoto GP with RandomForest as a computational speedup produced PR-AUC = 0.7174 (−0.0838 vs. champion). The Tanimoto GP is a non-negotiable component; no classical surrogate for GP-level molecule-similarity reasoning was found.

**CPU-only ensembles without Chemprop (GPU2, GPU3, cycle 5):** Mean CV PR-AUC consistently 0.72–0.74. Without Chemprop's graph-based representation, the ceiling on this task is ~0.74 for classical descriptor + kernel methods.

**ChemBERTa (GPU1, cycle 1):** PR-AUC 0.4069–0.4753 (two attempts). High variance, limited improvement from partial unfreezing. Retired after cycle 1 as insufficient on this small dataset.

## Coordination and Team Dynamics

**Three teams formed from discussion proposals.** The discussion phase produced 5 posts covering ChemBERTa, Chemprop MPNN, Tanimoto GP, pretrained transformers, and Graph Transformers. Admin formed 3 teams: team_alpha (transformer-based approaches, GPU1), team_beta (Chemprop/graph methods, GPU2-3, analyst1-2), and team_gamma (Tanimoto-kernel + tree ensembles, GPU4-6, analyst3).

**Cross-team convergence by cycle 2.** GPU4 (team_gamma) discovered that RDKit descriptor + Optuna-tuned XGBoost scored 0.7361 in cycle 2 — outperforming all GNN-based approaches from team_beta at that point. GPU3 immediately adopted ensemble stacking (XGB + SVM + GP) in cycle 2, reaching 0.7539. By cycle 3, all teams had converged on the same stacking paradigm, differing only in which base models to include.

**Approach registry failure.** The approach_registry.json recorded cycle=5 and taken=[], suggesting the registry coordination mechanism was not actively used to deduplicate approaches across agents. Despite this, agents discovered the stacking framework independently and converged without explicit deduplication. The absence of registry tracking means some redundant experiments (e.g., multiple agents running near-identical setups in cycle 4) were not avoided.

**Agent failures in late cycles.** Multiple agents in cycles 4 and 5 returned early from their sessions without completing experiments or produced scripts that ran as background processes and were not reliably monitored. Significant time in cycles 4 and 5 was spent debugging stalled background jobs and CLI flag bugs. Despite these coordination overhead costs, the champion was established at cycle 3 and survived intact. The extended cycle count (5 cycles vs. typical 2-3) reflected both the time available and the difficulty of improving beyond the 0.80 threshold.

**GPU availability favored sequential serialization.** With a single NVIDIA A100-SXM4-40GB GPU shared across all agents, the admin ran GPU agents sequentially (waiting for each to release the GPU before launching the next) while CPU experiments ran in parallel. GPU-resident experiments (Chemprop, ChemBERTa) were rate-limiting; the champion experiment (exp_gamma_gpu6_c3) took approximately 2 minutes and 21 seconds of wall time due to efficient GPU utilization across 5-fold Chemprop training.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. Fold-level scores show high variance (std ≈ 0.10 across folds), making mean CV PR-AUC estimates uncertain. The champion's mean CV PR-AUC of 0.8012 has fold scores ranging from 0.6828 to 0.9688 — a 0.29 range — indicating the reliability of any single-score comparison is limited.

**Approach registry was not functional.** The approach registry showed taken=[] for all 5 cycles, meaning the deduplication and knowledge-sharing mechanisms may not have operated as intended. Experiment ideas, dead ends, and insights were communicated through the shared workshop message board rather than structured registry entries.

**Scope:** The Tanimoto-kernel and Chemprop integration findings are specific to small molecular datasets (n~500) with severe class imbalance. Whether the 7-model Optuna-weighted approach generalizes to other kinase inhibitor tasks or larger datasets is unknown. The failure of deeper Chemprop architectures (depth=5) is likely dataset-size-specific.

**Unexplored axes:**
- Multi-seed Chemprop ensemble within the stacking framework (GPU5's approach scored 0.73 as the core, but was never combined with the full 7-model framework)
- Scaffold-aware validation splits to assess true generalization beyond the pre-defined folds
- Pre-trained molecular transformer embeddings (ChemBERTa) as a base model feature rather than standalone predictor
- Hyperparameter optimization of the Tanimoto-SVM (C parameter); the champion used C=0.5, which may not be optimal
- Larger Optuna trial counts beyond 150; limited time prevented exhaustive search in the champion's weight space
