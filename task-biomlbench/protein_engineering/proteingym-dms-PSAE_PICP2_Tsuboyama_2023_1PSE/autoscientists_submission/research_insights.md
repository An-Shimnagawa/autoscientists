---
task: proteingym-dms-PSAE_PICP2_Tsuboyama_2023_1PSE
run_id: biomlb_pg_psae_2
started_at: "2026-04-29T04:49:18Z"
champion_at: "2026-04-29T07:33:12Z"
---

# Research Insights for ProteinGym DMS PSAE_PICP2 Fitness Prediction

AutoScientists explored fitness prediction for PSAE_PICP2 (68 amino acids, 1,579 single- and multi-site variants). The headline finding is that **ESM2 LoRA fine-tuning with model-scale diversity and per-strategy weight optimization dominates all other approaches by a wide margin**. The run-end champion script (`autoscientists.py`) uses a 7-way ensemble with differential-evolution-optimized per-fold-strategy weights (mean Spearman 0.9761 on the training CV), while the cycle-2 single-model champion (exp_gamma_005, ESM2-35M LoRA) reached 0.9645 — a +0.0107 gain over the ESM2-8M LoRA baseline (exp_gamma_001, 0.9538). Classical Gaussian Process methods (GP + Hamming kernel: 0.8202; GP + BLOSUM62 kernel: 0.8656) were left far behind. The full-finetune of ESM2-35M without LoRA (exp_beta_002) collapsed to 0.9071, confirming that LoRA's parameter efficiency is critical at this dataset size (n = 1,579).

## Findings

**1. ESM2-35M LoRA outperforms ESM2-8M LoRA by +0.0107 Spearman without any architectural change beyond scale.**
exp_gamma_005 (ESM2-35M, 12 layers, 512-dim hidden, cosine LR with warmup, 50 epochs, early stopping patience 10) reached 0.9645 vs. exp_gamma_001 (ESM2-8M, 6 layers, 256-dim, 30 epochs, flat LR) at 0.9538. Both used LoRA r=16, alpha=32 targeting query/value projections. The benefit is consistent across all three fold strategies (random: +0.0282, modulo: +0.0042, contiguous: -0.0004). The random-fold strategy shows the largest improvement with scale, likely because the random split is the most i.i.d. and most sensitive to representation quality.

**2. Per-strategy weight optimization with differential evolution gives meaningful gains over uniform or grid-searched ensemble weights.**
The final 7-way ensemble (`autoscientists.py`) used `scipy.optimize.differential_evolution` to find per-fold-strategy weights jointly, reaching a mean Spearman of 0.9761 (random=0.9577, modulo=0.9815, contiguous=0.9890). The optimal weights differ substantially across strategies: gamma_007 (Spearman-loss model) dominates the random-fold strategy (weight=0.632) while ESM2-35M (gamma_005) dominates the contiguous fold (weight=0.395). This asymmetry implies that fold structure creates distinct difficulty regimes that different models handle differently.

**3. Training with Spearman rank-correlation loss instead of MSE yields competitive performance and high per-fold Spearman values, but lower mean Spearman than ESM2-35M.**
exp_gamma_007 (ESM2-8M LoRA, Spearman loss, 50 epochs) achieved a mean OOF Spearman of 0.9524 vs. 0.9538 for the MSE-trained baseline — effectively equal. However, gamma_007 achieves extremely high per-fold validation Spearman (fold averages of ~0.9957) with notably lower contiguous OOF (0.9615). Despite the mean underperformance, gamma_007 receives the highest weight (0.632) in the random-fold component of the final ensemble, indicating it contributes diverse predictions that complement ESM2-35M.

**4. Mutation-focused pooling (concatenating global mean pool with mutated-position mean pool) improves ESM2-8M by +0.9515 vs 0.9538 baseline — within noise — but adds diversity to ensembles.**
exp_beta_gpu1_007 introduced a novel dual-pooling scheme: for each sequence, it concatenates the global token mean (640-dim = 2x 320) with the mean of only mutated positions. This directly addresses the signal dilution problem in short proteins with few mutations (mean 1.2 mutations in 68 residues; 98% of token positions carry no mutation signal). The standalone result (0.9515) was slightly below the global-pool baseline (0.9538), but the model contributed meaningfully in 3-way and 7-way ensembles.

**5. GP + BLOSUM62 kernel improves over GP + Hamming by +0.045 Spearman but both trail ESM2 LoRA by 0.09+ points.**
exp_alpha_001 (GP + Hamming RBF): 0.8202. exp_alpha_002 (GP + BLOSUM62 kernel): 0.8656. The BLOSUM62 kernel captures amino acid substitution similarity better than binary Hamming distance, producing a +0.045 gain. However, neither can approach ESM2's learned evolutionary representations. A lightweight GP+BLOSUM62 ensemble with ESM2-8M (0.92 ESM2 + 0.08 GP, grid-searched weights) reached 0.9580, confirming the GP predictions add a small but nonzero correction.

### Insights

**G1 — Pre-trained protein language model representations dominate hand-crafted sequence kernels for small-n DMS fitness tasks.**
*Claim:* For DMS fitness prediction on proteins with n < 2,000 variants, ESM2 LoRA fine-tuning outperforms kernel methods (GP with Hamming or BLOSUM62) by at least 0.08 Spearman, regardless of kernel quality.
*Disconfirming evidence:* A protein where BLOSUM62 GP matches ESM2 LoRA performance despite the same dataset size.
*Observed:* GP + BLOSUM62 = 0.8656, ESM2-8M LoRA = 0.9538 (Δ = 0.088) on n = 1,579 PSAE_PICP2 variants.

**G2 — LoRA fine-tuning is critical for ESM2 at this dataset size; full fine-tuning collapses performance.**
*Claim:* For n < 2,000 protein sequences, full fine-tuning of ESM2 produces worse Spearman than LoRA (r=16) fine-tuning. The parameter reduction in LoRA prevents overfitting on small DMS datasets.
*Disconfirming evidence:* Full ESM2 fine-tuning matching LoRA on another DMS dataset with n in the same range.
*Observed:* exp_beta_002 (ESM2-35M full finetune): 0.9071 vs. exp_gamma_005 (ESM2-35M LoRA): 0.9645 (Δ = -0.0574).

**G3 — Ensemble diversity across model size and training objective is more valuable than depth within a single axis.**
*Claim:* Combining predictions from models that differ on multiple axes (scale: 8M vs. 35M; loss function: MSE vs. Spearman rank; pooling: global vs. mutation-focused; evolutionary prior: GP vs. PLM) produces larger Spearman gains than stacking multiple variants of the same architecture.
*Disconfirming evidence:* A multi-seed ensemble of identical ESM2-8M LoRA models matching the 7-way diverse ensemble.
*Observed:* Simple 3-way ensemble (alpha_003 + gpu1_007 + gamma_001) = 0.9623 via grid-searched weights; 7-way DE-optimized ensemble = 0.9761 (+0.0138).

**G4 — Per-fold-strategy weight optimization outperforms uniform weights when fold structures produce distinct difficulty profiles.**
*Claim:* When three cross-validation strategies (random, modulo, contiguous) produce substantially different fold Spearman values for individual models, per-strategy weight optimization delivers nontrivial gains over shared weights.
*Disconfirming evidence:* Uniform or shared-weight ensembles matching per-strategy DE weights when fold strategies are balanced.
*Observed:* gamma_007 has weight 0.632 on random folds but 0.120 on contiguous folds; ESM2-35M has weight 0.193 on random but 0.395 on contiguous — consistent with gamma_007 excelling at random and ESM2-35M excelling at contiguous.

### Task-Specific Findings

**T1 — PSAE_PICP2 has highly variable Spearman across fold strategies; contiguous folds are consistently easier to predict than random folds.**
Across all ESM2 LoRA models, the contiguous-fold OOF Spearman exceeds random-fold OOF Spearman by 0.04–0.08. For exp_gamma_001: random=0.9089, modulo=0.9680, contiguous=0.9844. For exp_gamma_005: random=0.9371, modulo=0.9722, contiguous=0.9840. This suggests that contiguous-fold test sets are not genuinely harder; they may be structurally similar to training data and thus easier to generalize to.

**T2 — ESM2-8M LoRA undertrains at 30 epochs; per-fold validation Spearman values above 0.98 are achievable with patience-based early stopping.**
exp_gamma_001 noted loss still decreasing at epoch 30. The analyst team flagged this before cycle 2. exp_gamma_005 (ESM2-35M) and exp_gamma_006 (ESM2-8M + cosine LR + 50 epochs) both used early stopping with patience ≥ 10 and achieved fold-level Spearman values of 0.98–0.99 on contiguous folds. The gain from longer training (30→50 epochs with early stopping) was part of the ESM2-35M improvement.

**T3 — ESM2-8M frozen embeddings + ESM2-650M frozen embeddings blended yields 0.9600, below ESM2-8M LoRA (0.9538) trained from scratch.**
exp_beta_gpu5_003 combined ESM2-8M LoRA fine-tuned predictions (0.9538 component) with ESM2-650M frozen embedding predictions (0.9000 component), achieving 0.9600 via a 0.7/0.3 RidgeCV blend. This is a positive result (beats the 8M LoRA alone), but adding a 650M frozen component adds less than fine-tuning the 35M model. ESM2-650M frozen embeddings without task-specific fine-tuning provide limited lift; task-specific adaptation is the more important factor.

**T4 — ESM2-8M frozen embeddings + BLOSUM features + LightGBM (exp_gamma_002) scored only 0.8154, well below ESM2-8M LoRA fine-tuned.**
This approach combined pre-cached ESM2-8M mean-pooled embeddings (no fine-tuning) with hand-crafted BLOSUM position-specific mutation cost features and a LightGBM regressor. Despite using learned embeddings, the lack of task-specific adaptation reduced the score to 0.8154. Fine-tuning (LoRA) of the embedding model is substantially more important than the addition of hand-crafted features on top of frozen embeddings.

**T5 — ESM2-150M LoRA training failed with a silent crash before training began.**
exp_gamma_004 (ESM2-150M LoRA, 30 epochs) was attempted on GPU4. The stdout shows only model loading output (36 lines total) with no training output, suggesting the process crashed silently before the first forward pass — likely a CUDA out-of-memory error, consistent with the analyst team's subsequent diagnosis. ESM2-35M (exp_gamma_005) was chosen as the safe upper bound on this hardware.

## Dead Ends and Negative Results

**ESM2-35M full finetune (exp_beta_002):** 0.9071 vs. 0.9538 for ESM2-8M LoRA. Full parameter updates on n=1,579 sequences overfit the training folds. ESM2-35M has 35M parameters; training all of them on 1,263 training samples (80% of 1,579) is severely under-constrained. Retired: LoRA is strictly necessary for this dataset size.

**ESM2-8M frozen + BLOSUM + LightGBM (exp_gamma_002):** 0.8154. Frozen ESM2 embeddings without task-specific adaptation yield representations that are insufficiently tuned for DMS fitness prediction. Adding BLOSUM features does not compensate. Retired in favor of LoRA fine-tuning.

**GP + Hamming kernel (exp_alpha_001):** 0.8202. Binary Hamming distance treats all amino acid substitutions as equally costly. On a protein where substitution type matters for stability, this is an inadequate kernel. Retired after one cycle.

**GP + BLOSUM62 kernel (exp_alpha_002):** 0.8656. Improved over Hamming but ceiling is fundamental to the kernel method's inability to learn interaction effects. Even with the best possible kernel, GP is bounded by its inability to model nonlinear epistasis. Retained as a minor blending component (8% weight in exp_alpha_003 = 0.9580) but not a standalone competitive approach.

**ESM2-8M LoRA + cosine LR + 50 epochs (exp_gamma_006):** 0.9409. Extended training with cosine learning rate annealing and gradient clipping on ESM2-8M (vs. champion's flat LR, 30 epochs) produced a lower score than the champion (0.9409 < 0.9538). The cosine schedule appears to hurt ESM2-8M on this task; the flat-LR 30-epoch champion generalized better. Retired: cosine LR did not improve ESM2-8M despite it helping ESM2-35M.

**ESM2-8M LoRA seed diversity (exp_gamma_010, seed=0; exp_gamma_011, seed=123):** 0.9442 and 0.9460 respectively vs. 0.9538 for seed=42. Different seeds underperform the original; seed=42 appears to be a fortuitously strong initialization for this data. Both experiments were incorporated into the final ensemble but not as leading contributors.

**ESM2-8M LoRA warmup+cosine LR, 40 epochs (exp_beta_gpu5_004):** stdout shows only 54 lines covering 3 folds through epoch 30 — the experiment did not complete before the run ended. No final score available.

**ESM2-8M LoRA extended training, 60 epochs (exp_alpha_004):** Only 30 lines of stdout, showing training beginning on fold_random_5 fold 0 through epoch 20 (val_spearman reached 0.8351 at epoch 20). The experiment did not complete within the run window.

**ESM2-150M LoRA (exp_gamma_004):** Silent crash (likely CUDA OOM). No score. ESM2-35M was confirmed as the practical hardware upper bound.

**ProtT5 LoRA (exp_gamma_009):** Queued in cycle 2 but not executed within the run window; no result available.

**ESM2-8M + ESM2-35M simple ensemble (exp_gamma_008):** Queued in cycle 2 but not executed within the run window; no result available.

## Coordination and Team Dynamics

The run had 3 analyst agents (analyst1, analyst2, analyst3) and 4 active GPU agents (gpu1, gpu2, gpu3, gpu5 per GPU claim files; gpu4 and gpu6 also ran experiments). The analyst2 (Team Alpha) kept detailed written memory across sessions, producing task analysis, cycle analysis, competitive analysis, and implementation guides that informed gpu2's experiment queue. The analyst3 (Team Gamma) similarly documented strategic proposals in cycle_2_analysis.md that directly generated the Spearman-loss and ensemble proposals executed by gpu3 and gpu5.

A key cross-team knowledge transfer occurred between cycles: the gamma team identified that "loss still decreasing at epoch 30" in exp_gamma_001, and this observation propagated to at least three cycle-2 experiments (exp_gamma_005, exp_gamma_006, exp_alpha_004) all of which increased epoch budgets. The analyst2 team acknowledged the gamma team's dominance directly ("Pre-trained embeddings >> hand-crafted features") and pivoted their cycle-2 proposals to ESM2-scale experiments rather than continuing kernel refinement.

A post-hoc ensemble search was run after cycle 2 completed, producing the final submission by progressively combining components: ensemble_35M_gamma001 (0.9697), ensemble_per_strategy (0.9699), 3-way (0.9700), 4-way scipy (0.9716), 5-way DE (0.9716), 6-way DE (0.9716), 6-way with seed diversity (0.9721), and finally 7-way with gamma_007 (0.9761). This log (from the champion SOURCE file) shows a systematic greedy addition strategy that added components only when they improved the ensemble.

GPU4 ran exp_gamma_004 (ESM2-150M) which failed silently, and then pivoted to exp_gamma_006 (ESM2-8M cosine LR + 50 epochs), which also underperformed. GPU5 ran exp_beta_gpu5_003 (dual-scale 8M+650M blend, 0.9600) and exp_beta_gpu5_004 (warmup cosine LR, incomplete) and exp_beta_gpu5_005 (3-way ensemble, 0.9623). GPU6 executed exp_gamma_005 (ESM2-35M LoRA, 0.9645) — the highest-scoring single model. GPU3 executed exp_gamma_001 (baseline, 0.9538) and exp_gamma_007 (Spearman loss, 0.9524).

## Limitations of These Insights

**Validation vs. test gap unknown:** All reported scores are OOF CV Spearman on the full dataset (no held-out test set). The ground-truth leaderboard score is not available in the run artifacts. The reported champion score (0.9645 for exp_gamma_005, 0.9761 for the 7-way ensemble) is computed on training-fold OOF predictions, which may be optimistic.

**Single run:** No independent replication was conducted. The gamma_001 seed=42 advantage (0.9538 vs. 0.9442 for seed=0, 0.9460 for seed=123) suggests the original seed is atypically strong. A different run may produce different relative rankings.

**Incomplete experiments:** exp_alpha_004 (60-epoch ESM2-8M), exp_beta_gpu5_004 (warmup cosine LR ESM2-8M), exp_gamma_008 (8M+35M ensemble), and exp_gamma_009 (ProtT5) were all proposed or queued but did not complete. Their potential contributions are unknown.

**Ensemble optimization on training data:** The differential-evolution weight search in the final script (`autoscientists.py`) optimizes weights using Spearman on the same OOF predictions that define the score. This is not independent optimization; the true improvement of per-strategy weighting over uniform weighting on unseen data may be smaller.

**Unexplored axes:**
- ProtT5 fine-tuning with LoRA: proposed, not executed
- ESM2-650M LoRA fine-tuned (as opposed to frozen): not attempted
- Attention pooling (per-position learnable weights) beyond the mutation-focused scheme
- Structure-informed approaches: the GNN proposal from the initial task analysis was tried by team beta (exp_beta_001, scoring 0.9139 on an early ESM2+structure variant per analyst notes), but not developed further
- Multi-seed ensemble of the ESM2-35M champion to reduce variance
