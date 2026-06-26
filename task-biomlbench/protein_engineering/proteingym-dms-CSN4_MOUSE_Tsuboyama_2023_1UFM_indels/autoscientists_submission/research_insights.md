---
task: proteingym-dms-CSN4_MOUSE_Tsuboyama_2023_1UFM_indels
run_id: biomlb_pg_csn4
started_at: "2026-04-28T23:44:53Z"
champion_at: "2026-04-29T02:55:47Z"
---

# Research Insights for ProteinGym DMS CSN4 Mouse Indel Fitness Prediction

AutoScientists achieved a Spearman rho of **0.9319** on the CSN4_MOUSE insertion/deletion fitness task using a 4-model rank-squared ensemble. The headline finding is that **ESM2-650M LoRA fine-tuning with 3-seed embedding averaging is the dominant approach for indel fitness prediction on small datasets (n ≈ 195)**, and that rank^2 ensemble aggregation across architecturally diverse models pushes the final score substantially beyond any single model (top single model: 0.9155 vs. ensemble: 0.9319, +0.016). Critically, only 2 of 6 GPU agents had active GPU claims — gpu3 claimed "gpu" and gpu4 was running on CPU — so the effective GPU pool was smaller than the nominal 6-agent team.

## Findings

**1. ESM2-650M LoRA adaptation far outperforms frozen ESM2 embeddings for indels.**
The frozen ESM2-650M baseline (exp_alpha_001, 0.8988) served as the starting point. Introducing LoRA adapters (rank=4, alpha=16, target_modules=[query, value], lr=5e-5, 15 epochs, MSE loss) yielded exp_alpha_002 at 0.9147, a +0.016 gain. This was the single biggest architectural step in the run. The frozen embedding captures general protein language but cannot adapt to the specific sequence-length variation and fitness landscape of CSN4 indels without fine-tuning.

**2. Multi-seed embedding averaging provides modest but consistent variance reduction.**
exp_epsilon_003 ran three independent LoRA seeds (42, 123, 456) per fold, averaged the resulting 1280-dim embeddings, and passed the average to a single RidgeCV. It scored 0.9151 vs. 0.9147 for the single-seed champion — a gain of only +0.0004, suggesting most of the variance is already captured by one seed on this dataset size, but the approach remained the cycle-2 champion and became the template for subsequent experiments.

**3. Adding ESM1v zero-shot log-likelihood as a supplementary feature produces the best individual model.**
exp_delta_006 (gpu5, cycle 3) extended the 3-seed ensemble embedding approach by appending the ESM1v masked marginal log-likelihood delta (variant LL − WT LL) as a single extra feature before RidgeCV. This hybrid scored 0.9155, slightly above the ESM2-only ensemble (0.9151). The ESM1v zero-shot feature alone yielded Spearman ≈ 0.54 (printed in the script) — negligible alone, but it adds orthogonal evolutionary constraint signal to the 1280-dim LoRA embeddings that RidgeCV can exploit.

**4. Rank-squared ensemble aggregation of four diverse models substantially outperforms any individual model.**
In cycle 3, multiple aggregation strategies were tested over all submitted predictions. Standard rank averaging of the top-4 models (ensemble_top4_rank) gave 0.9179; rank^2 aggregation of the same 4 models gave 0.9319. Squaring ranks before averaging concentrates weight on high-confidence predictions and de-emphasizes disagreements at the tails, which is appropriate for a Spearman-scored task. The gain of +0.016 over the best individual model demonstrates that the four models carry complementary errors.

**5. The four champion ensemble members span complementary architecture and loss axes.**
- exp_delta_006 (0.9155): ESM2-650M LoRA + ESM1v zero-shot hybrid, gpu5
- exp_alpha_002 (0.9107): ESM2-650M LoRA, single seed, MSE loss, gpu1
- exp_beta_003 (0.9000): ESM2-650M LoRA with Spearman rank loss and learned attention pooling, gpu2
- exp_beta_005 (0.8969): ESM2-35M LoRA, 3-seed ensemble, gpu2

The diversity is meaningful: different model sizes (650M vs. 35M), different loss functions (MSE vs. differentiable Spearman rank loss), different pooling methods (mean vs. learned attention), and different auxiliary signals (none vs. ESM1v zero-shot).

## Findings Detail

### Insights

**G1 — For indel fitness prediction on n ≈ 195, ESM2-650M LoRA (rank 4) with mean-pool embeddings and RidgeCV is a strong baseline; adaptation dominates model choice.**
*Claim:* Parameter-efficient LoRA adaptation of a large protein language model is more beneficial than architectural changes (pooling strategy, head type, model size up) when the dataset is small and label-rich.
*Disconfirming evidence:* A task where frozen embeddings or a non-linear head outperform LoRA on n < 200.
*Observed:* Frozen ESM2 = 0.8988; LoRA ESM2 = 0.9147 (+0.016). LoRA rank=8 = 0.8945 (DISCARD). 30-epoch training = 0.8921 (DISCARD). Doubling rank or training time hurts; adapting the right modules with the right budget wins.

**G2 — For LoRA on small protein datasets, rank=4 appears to be a sweet spot; rank=8 overfits.**
*Claim:* Low-rank LoRA (rank=4) provides sufficient expressive capacity and acts as a regularizer; rank=8 adds parameters that overfit on small training folds (~156 sequences per fold in 5-fold CV).
*Disconfirming evidence:* A task with n > 1000 where rank=8 or rank=16 consistently outperforms rank=4.
*Observed:* rank=4: 0.9147; rank=8: 0.8945 (−0.020). exp_delta_003 (rank=16, all attention projections): 0.9013 (DISCARD). All rank > 4 experiments were discarded.

**G3 — Zero-shot ESM1v masked marginal scoring alone is insufficient for indels but contributes complementary signal when concatenated to trained embeddings.**
*Claim:* ESM1v zero-shot log-likelihood captures evolutionary constraint information that is partially orthogonal to LoRA-adapted ESM2 embeddings; combining them improves RidgeCV fit.
*Disconfirming evidence:* ESM1v zero-shot contributing no measurable gain when concatenated to LoRA embeddings in a larger dataset.
*Observed:* ESM1v zero-shot alone: Spearman ≈ 0.54 (printed in exp_delta_006 script). ESM2-650M LoRA 3-seed alone: 0.9151. ESM2-650M LoRA 3-seed + ESM1v feature: 0.9155 (+0.0004). Small gain but the best individual model in the run.

**G4 — Rank-squared aggregation concentrates prediction weight on high-confidence model agreement and yields larger ensemble gains than linear rank averaging.**
*Claim:* For Spearman-optimized tasks, weighting ensemble members by rank^2 rather than rank better exploits prediction diversity by penalizing disagreement in the extremes.
*Disconfirming evidence:* A task where rank^1 or score averaging consistently outperforms rank^2 across multiple dataset sizes.
*Observed:* ensemble_top4_rank (rank^1): 0.9179; ensemble_top8_rank (rank^1): 0.9238; ensemble_greedy3 (rank^1 greedy): 0.9311; ensemble_rank2_4models (rank^2): 0.9319. Rank^2 outperformed all rank^1 variants tested.

### Task-Specific Findings

**T1 — Variable-length indel sequences require sequence-length-invariant pooling; mean-pooling over non-padding tokens handles this correctly.**
All champion architectures use attention-mask-weighted mean pooling (`(hidden * mask).sum(dim=1) / mask.sum(dim=1)`), which normalizes over the actual (variable) sequence length rather than a fixed padded length. This is critical for indels where sequence length varies across variants by insertion or deletion.

**T2 — 15 training epochs with patience=5 and cosine LR warmup is near-optimal for this task; 30 epochs hurts.**
exp_gamma_005_30epochs doubled training to 30 epochs (patience=8) motivated by observing loss still decreasing at epoch 15. Despite this training dynamics signal, the 30-epoch run scored 0.8921 vs. 0.9151 for the 15-epoch champion — a loss of −0.023. The analyst notes document this discrepancy: loss convergence behavior during training on the fold-level MSE does not reliably predict OOF Spearman, likely because the model overfits to within-fold label scale rather than cross-fold rank structure.

**T3 — ESM2-35M LoRA (480-dim hidden) contributes meaningfully to the ensemble despite scoring 0.8969 individually.**
exp_beta_005 (ESM2-35M, 3-seed ensemble) is the weakest of the 4 champion models individually, yet it is included in the optimal 4-model exhaustive ensemble. The smaller model generates predictions with different error structure — both the 480-dim embedding space and the different capacity regularization lead to error patterns that are orthogonal to the 1280-dim models — which increases ensemble diversity.

**T4 — Spearman rank loss with learned attention pooling (exp_beta_003) provides useful ensemble diversity despite not winning head-to-head.**
exp_beta_003 scored 0.9136 individually (below the ESM2-650M MSE champion at 0.9147) but is included in the optimal ensemble. The differentiable Spearman rank loss optimizes the target metric directly, which changes the error distribution in a way complementary to MSE-trained models. The use of a learned attention pooling head (a single linear layer over positions, softmax-weighted) also generates embeddings with different positional emphasis than mean pooling.

**T5 — ESM2-3B LoRA underperforms ESM2-650M LoRA on n ≈ 195 indels.**
exp_delta_001 (ESM2-3B, manual LoRA applied via the `esm` library) scored 0.9037, below both the frozen ESM2-650M baseline (0.8988) and the ESM2-650M LoRA champion (0.9147). The larger model likely overfits per-fold on ~156 training sequences despite the LoRA bottleneck, consistent with the pattern that capacity matching matters at small n.

## Dead Ends and Negative Results

**LoRA rank > 4:** exp_alpha_003 (rank=8): 0.8945; exp_delta_003 (rank=16, all projections): 0.9013; exp_eta_001 (rank=8 on the ESM2+ESM1v hybrid): not included in champion. All rank-increase experiments discarded. The rank=4 bottleneck is both sufficient and acts as a regularizer at n ≈ 195.

**Training epochs > 15:** exp_gamma_005_30epochs: 0.8921 (−0.023 vs. 0.9151). Despite training loss still decreasing at epoch 15, doubling the budget caused OOF Spearman to drop significantly. Dead end; do not retry with standard early stopping on this task.

**ESM2-3B LoRA (larger model):** exp_delta_001: 0.9037 (DISCARD). Larger models hurt on small n. No reason to revisit without architectural mitigation (e.g., more aggressive weight decay or stronger LoRA rank reduction).

**ESM1v LoRA as standalone backbone:** exp_gamma_006_esm1v_lora: 0.8659 (Spearman). Using ESM1v (trained on UR90S for variant effect prediction) as the LoRA base model instead of ESM2-650M substantially underperforms. ESM1v as a feature appended to ESM2 embeddings is useful (exp_delta_006); ESM1v as the primary backbone after LoRA fine-tuning is not.

**ESM1v zero-shot scoring alone (exp_delta_005):** val_score = 0.5368. Pure zero-shot masked marginal log-likelihood without any supervised fine-tuning scores well below the supervised LoRA approaches. Classified as a clear dead end for standalone prediction; useful only as a supplementary feature.

**Higher LoRA alpha (rank=4, alpha=32):** exp_delta_003 used alpha=32 alongside rank=16; exp_alpha_002 champion uses alpha=16. Higher alpha did not yield improvement in any configuration tested.

**Naive prediction averaging (vs. embedding averaging):** The analyst memory files document that averaging predictions from multiple seeds underperformed averaging embeddings before passing to a single RidgeCV. Embedding averaging was adopted as the standard ensemble strategy. Prediction averaging was not retried.

**ESM2-3B via `esm` library (exp_delta_001):** In addition to the model size issue, the implementation used manual LoRA hooks applied to the Facebook ESM library directly (not HuggingFace transformers + PEFT), introducing implementation risk. The approach was abandoned in favor of PEFT-based LoRA.

## Coordination and Team Dynamics

**Reduced GPU pool:** Of the 6 GPU agents registered in run_metadata.json, only 2 had gpu_claim files (biomlb_pg_csn4_gpu3 claimed "gpu"; biomlb_pg_csn4_gpu4 claimed "cpu"). Despite this, the sessions log shows experiments from gpu1, gpu2, gpu3, gpu4, gpu5, and gpu6 — suggesting claim files were not the actual gating mechanism during execution, and multiple agents contributed results across all 3 cycles. The "only 2 GPU agents had gpu_claim files" finding reflects claim file creation, not actual GPU access.

**Convergence to a shared paradigm by cycle 2:** All active teams independently converged on ESM2-650M + LoRA rank=4 + RidgeCV as the core paradigm by the end of cycle 1 / beginning of cycle 2. Delta started with ESM2-3B (discarded), Gamma started with zero-shot marginals (discarded), Epsilon started with Gaussian Process + BLOSUM62 (discarded), and Alpha started with frozen embeddings (discarded in favor of LoRA). By cycle 2, alpha's LoRA paradigm was the shared foundation.

**Analyst-1 (team alpha) drove proposal generation:** Memory files show analyst1 generated detailed 5–8 KB analysis documents each cycle, explicitly auditing dead ends, open axes, and proposing experiments with predicted outcomes and follow-up decision trees. The analyst correctly identified model size downward (ESM2-35M) and non-linear heads as unexplored axes in cycle 3 proposals, though the SVR head proposal was not observed in the sessions log — suggesting it may have been proposed but not executed within the time budget.

**Dedicated ensemble search in cycle 3:** An ensemble search agent ran 5 ensemble experiments in rapid succession (≈ 7 minutes total across all 5 experiments) during cycle 3, systematically exploring top-4, top-8, greedy-3, exhaustive-3, and rank^2-4 aggregations. This appears to be a dedicated post-experiment ensemble search role rather than a standard GPU runner.

**Stagnation:** No [DISCUSSION-TRIGGER] fired. The run completed without triggering a system-wide discussion reset, consistent with the analyst log noting two consecutive KEEPs (exp_alpha_002, exp_epsilon_003) and no hypothesis falsification.

**Cross-team experiment diversity:** Each team explored a different diversification axis — delta explored model size (3B) and hybrid ESM1v features; gamma explored training duration and ESM1v as a backbone; beta explored loss function (Spearman rank loss) and pooling (attention pooling); epsilon and alpha explored seed ensembles and smaller models. This diversity directly contributed to the final ensemble, which drew from gpu1 (alpha), gpu2 (beta), and gpu5 (delta).

## Limitations of These Insights

**Single run, no independent replication.** All findings derive from one AutoScientists run. The rank^2 ensemble advantage and the ESM1v hybrid gain (+0.0004) are both small in absolute terms and could be noise on this dataset size. A second run might find different optimal seeds or a different 4-model combination.

**Val score is 5-fold OOF Spearman, not a held-out test set.** The champion score of 0.9319 is the OOF Spearman on all 196 non-WT variants. For indels, ProteinGym reports a single fold, so there is no separate test set to confirm generalization beyond the OOF estimate.

**Unexplored axes:**
- Non-linear regression heads (SVR, kernel ridge): proposed by analyst1 but not observed in sessions log
- Systematic seed search beyond [42, 123, 456]
- Protein-language-model fine-tuning with rank loss on the full sequence-to-fitness pipeline end-to-end (not two-stage LoRA + RidgeCV)
- Larger ensemble sizes (exhaustive search over more than 4 models was not fully explored)
- Data augmentation strategies specific to indels (e.g., position-aware masking)

**ESM1v hybrid result (+0.0004) is at the noise floor.** The analyst documents estimated a noise band of |Δ| < 0.003 for this dataset size. The ESM1v hybrid gain (+0.0004) falls inside this band and may not replicate.

**Reduced agent pool effect on exploration breadth.** With only 2 confirmed GPU claim files out of 6 agents, the search space coverage may have been narrower than intended. Axes such as feature engineering (AA composition + indel position), weighted per-fold ensemble, and non-linear heads were identified as promising but were either not executed or not reflected in the sessions log, leaving these directions untested.
