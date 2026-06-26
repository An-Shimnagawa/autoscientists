---
task: proteingym-dms-SBI_STAAM_Tsuboyama_2023_2JVG
run_id: biomlb_pg_sbi
started_at: "2026-04-28T16:57:17Z"
champion_at: "2026-04-28T20:36:12Z"
---

# Research Insights for ProteinGym DMS SBI_STAAM Fitness Prediction

AutoScientists tackled stability fitness prediction for Immunoglobulin-binding protein Sbi from *S. aureus* (56 AA, 1,025 single-substitution variants, Tsuboyama 2023). The run began with an ESM-1v zero-shot baseline at Spearman 0.276 and climbed to 0.822 via a weighted ensemble of ESM-2 3B mean-pooled embeddings with calibrated stacking (gpu6) and ESM-2 650M supervised embeddings augmented with zero-shot marginal scores (gpu3). The headline finding is that **larger ESM-2 models (3B vs. 650M) and a calibrated meta-learner stacking ensemble substantially outperform both LoRA fine-tuning and vanilla supervised embeddings** on this short, stability-focused protein. A non-posted 5-way optimal ensemble (grid search, step=0.05) reached 0.834 on the internal validation metric but was not recorded as the final submission.

## Findings

**1. ESM-1v zero-shot LLR gives near-random performance on this stability dataset.**
The ESM-1v masked marginal (zero-shot) score yielded Spearman 0.276 averaged across three fold strategies. Per-fold variance was high (range 0.02–0.53 on modulo splits), indicating the ESM-1v evolutionary likelihood signal is poorly aligned with the thermodynamic stability readout measured for SBI_STAAM.

**2. ESM-2 3B masked marginal is dramatically stronger than ESM-1v.**
Switching to the ESM-2 3B masked marginal zero-shot score (same paradigm, larger model) produced Spearman 0.650 — a +0.374 jump using the same zero-shot approach without any supervised training. This suggests SBI_STAAM stability correlates substantially better with ESM-2 3B evolutionary representations than with ESM-1v.

**3. Supervised embeddings plus zero-shot marginal features outperform either alone.**
ESM-2 650M mean-pooled embeddings with a Ridge+SVR ensemble scored 0.685. Adding the pre-computed ESM-2 3B masked marginal score as a single additional feature — alongside five biochemical delta features (BLOSUM62, hydrophobicity, charge, MW, polarity) — and upgrading from fixed-alpha Ridge+SVR to RidgeCV+GradientBoosting pushed the score to 0.756. The zero-shot signal captures evolutionary constraints that the supervised embedding pathway does not redundantly encode.

**4. Larger ESM-2 embeddings (3B, 2560-dim) with calibrated stacking are the single most impactful upgrade.**
ESM-2 3B mean-pooled embeddings (2560-dim) combined with the same marginal and biochem features, but using a calibrated stacking ensemble (RidgeCV + SVR + GradientBoosting base learners; meta-Ridge with positive constraint and inner 5-fold OOF) scored 0.819 — +0.063 over the ESM-2 650M supervised baseline. The meta-learner consistently assigned highest weight to SVR (~0.37–0.47) and RidgeCV (~0.37–0.45), with GradientBoosting contributing a smaller share (~0.16–0.19), suggesting a preference for regularized smooth learners on this 2566-dim feature space.

**5. Weighted ensembling of diverse ESM-2 architectures provides additive gains over any single model.**
A grid-searched weighted ensemble of gpu6 (ESM-2 3B stacked, w=0.66) and gpu3 (ESM-2 650M supervised, w=0.34) achieved Spearman 0.822 — +0.003 over the gpu6 single-model. Weights found by 1D grid search in 0.01 increments. A 5-way ensemble (gpu6=0.65, gpu3=0.20, gpu5=0.10, gpu2=0.05, gpu4=0.0) reached 0.834 on the internal leaderboard but was not posted to the workshop before the run ended.

**6. LoRA fine-tuning of ESM-2 does not outperform frozen embedding approaches on this dataset.**
ESM-2 650M LoRA (last 8 layers, rank=8, 8 epochs, ~0.4M trainable params) scored 0.727. ESM-2 3B LoRA (last 12 layers, rank=16, 6 epochs, ~20% marginal ensemble) scored 0.745. Both fall below the frozen-embedding stacking approach at 0.819, indicating that for this short protein (56 AA) and dataset size (n=1,025), gradient updates to adapter layers do not generalize better than extracting and learning over fixed ESM-2 3B representations.

**7. The contiguous fold split is consistently the hardest across all methods.**
All experiments showed a marked Spearman drop on `fold_contiguous_5` vs. `fold_random_5`. For the champion ensemble: random=0.895, modulo=0.813, contiguous=0.759. For the best single model (gpu6): random=0.892, modulo=0.806, contiguous=0.759. The position-aware XGB approach was most sensitive: random=0.874, modulo=0.563, contiguous=0.446. Contiguous splits place consecutive-sequence variants in the same fold, creating a structural generalization challenge.

### Insights

**G1 — ESM-2 3B zero-shot masked marginals substantially outperform ESM-1v on stability DMS tasks.**
*Claim:* For stability-focused DMS datasets on short proteins, ESM-2 3B masked marginal LLR is a more reliable zero-shot predictor than ESM-1v.
*Disconfirming evidence:* A stability task where ESM-1v and ESM-2 3B zero-shot performance are comparable.
*Observed:* ESM-1v zero-shot = 0.276; ESM-2 3B zero-shot = 0.650 on the same task with identical scoring methodology.

**G2 — Pre-computed zero-shot marginal scores function as effective single-dimensional features in supervised models.**
*Claim:* Adding the ESM-2 masked marginal score as a scalar feature to a supervised embedding model provides complementary signal beyond what the mean-pooled embedding captures.
*Disconfirming evidence:* A task where including marginal scores as a feature consistently hurts or has no effect on a supervised embedding model.
*Observed:* ESM-2 650M embeddings alone + Ridge+SVR = 0.685; same embeddings + ESM-2 3B marginal + biochem + RidgeCV+GBR = 0.756.

**G3 — Calibrated stacking with a positive-constraint meta-Ridge outperforms fixed-weight ensembles.**
*Claim:* Learning blend weights from OOF meta-features via a non-negative Ridge meta-learner is more effective than equal-weight or hand-tuned blends, especially when base learners have heterogeneous biases (Ridge vs. SVR vs. GBR).
*Disconfirming evidence:* A task where OOF stacking matches or underperforms simple averaging on the same base learners.
*Observed:* ESM-2 3B stacked ensemble = 0.819 vs. baseline 650M ensemble at 0.685–0.756 with fixed-weight blends.

**G4 — LoRA fine-tuning on small stability DMS datasets (n~1,000) does not exceed frozen-embedding approaches.**
*Claim:* When training on ~800 sequences per fold split, LoRA adapters (rank 8–16, 6–8 epochs) do not outperform extracting and regression-stacking on frozen ESM-2 3B embeddings.
*Disconfirming evidence:* A DMS task of similar size and protein length where LoRA fine-tuning beats frozen-embedding stacking by more than a few Spearman points.
*Observed:* Best LoRA result (ESM-2 3B, rank=16) = 0.745; best frozen-embedding stacking = 0.819.

### Task-Specific Findings

**T1 — SBI_STAAM's very short sequence (56 AA) makes mean-pooling informative but sequence-context limited.**
With 56 amino acids, the mean-pooled embedding averages over very few tokens. Despite this, the ESM-2 3B 2560-dim representation yielded strong Spearman. The contiguous fold gap (random=0.892 vs. contiguous=0.759 for the best single model) suggests positional context is encoded unevenly and becomes harder to generalize when structurally adjacent positions are split across folds.

**T2 — Position-aware concatenation of mean and positional embeddings with XGBoost/LightGBM hurt performance.**
ESM-2 650M mean-pooled (1280-dim) + positional embeddings (1280-dim) + ESM score + biochem features = 2566-dim total, trained with XGBoost/LightGBM/Ridge (equal weights) scored 0.628 overall. The position-aware approach degraded particularly on modulo (0.563) and contiguous (0.446) folds vs. the simpler mean-pooled approach, suggesting the doubled feature space adds noise that tree-based boosters cannot regularize effectively at n~820 training samples.

**T3 — Contiguous fold is severely hurt by position-aware XGBoost approaches.**
The `esm2_position_xgb_001` contiguous fold Spearman was 0.446, vs. 0.874 on random — a 0.428 gap. By contrast, the frozen ESM-2 3B stacked approach had a 0.133 gap (0.892 random vs. 0.759 contiguous). The position-aware features amplify the train-test distributional shift in contiguous splits, where contiguous sequence segments define fold boundaries.

**T4 — The 5-way optimal ensemble (0.834) was not posted to the workshop.**
GPU3's grid search over all five submission files (step=0.05) found optimal weights {gpu6: 0.65, gpu3: 0.20, gpu5: 0.10, gpu2: 0.05, gpu4: 0.0} for a combined Spearman of 0.834. This result was computed after the workshop posting window closed (result_latest.json shows `posted_to_workshop: false`). The registered champion is the 2-way ensemble at 0.822.

**T5 — Biochemical delta features (5-dim) contribute positively but have low marginal impact at the 2560-dim scale.**
All top experiments combined ESM-2 embeddings with five physicochemical features (BLOSUM62, hydrophobicity, charge, MW, polarity). The absolute contribution of these features is difficult to isolate, but their inclusion was consistent across all experiments achieving >0.75 Spearman. Given that these five features are essentially redundant with information present in the 2560-dim ESM-2 representation, their primary role may be to provide a regularization-friendly low-dimensional anchor for the meta-learner.

## Dead Ends and Negative Results

**ESM-1v zero-shot (0.276):** Evolutionary likelihood from ESM-1v is poorly predictive of thermodynamic stability for SBI_STAAM. Spearman was near-random (0.276 mean across three fold strategies, with per-fold range 0.02–0.53 on modulo). Not explored further after ESM-2 3B zero-shot proved vastly superior.

**Position-aware embedding concatenation with XGBoost (0.628):** Concatenating mean-pooled and residue-level positional embeddings doubled the feature dimension to 2566 and trained XGB/LGB/Ridge. This approach scored below the simpler mean-pooled ESM-2 650M baseline (0.685) and showed severe contiguous-fold degradation (0.446). Retired.

**ESM-2 650M LoRA fine-tuning (0.727):** 8-epoch LoRA (rank=8, last 8 layers, ~0.41M trainable params), including a 0.3-weight marginal blending step, reached 0.727. Despite the model being updated with labeled data, this underperformed the frozen ESM-2 3B embedding approach by 0.09 Spearman. Training loss converged (8 epochs, final ~0.06–0.09) so underfitting is unlikely; the issue appears to be generalization from LoRA adapter updates on n~820 training sequences.

**ESM-2 3B LoRA fine-tuning (0.745):** Scaling LoRA to 3B (rank=16, last 12 layers, 6 epochs, 20% marginal ensemble) improved over 650M LoRA but still underperformed frozen-embedding stacking by 0.074. Classified DISCARD (delta = -0.010 vs. prior champion at the time). Not retried with alternative regularization or epoch counts.

## Coordination and Team Dynamics

The run included two analyst agents (analyst1, analyst2) whose memory logs contained no entries, indicating they either did not execute or ran without generating storable findings. Only three GPU agents (gpu1, gpu3, gpu6) posted results to the workshop; GPU1 additionally ran the two initial zero-shot experiments and the final weighted ensemble. GPU2 (position-aware XGB), GPU4 (LoRA 650M), and GPU5 (LoRA 3B) posted results but those were not selected as the final champion.

GPU6 introduced the decisive architectural improvement — switching from ESM-2 650M to ESM-2 3B embeddings and replacing the fixed-weight ensemble with calibrated stacking. GPU1 then built the weighted ensemble combining GPU6 and GPU3 submissions by grid-searching blend weights. GPU3 subsequently ran a 5-way grid search post-champion that identified a higher-scoring (0.834) combination but could not post it in time.

The approach registry shows three ensemble strategies explored in cycle 2 (`ensemble-avg-gpu3-gpu6`, `weighted-ensemble-gpu6-gpu3-gpu2`, `optimal-5way-ensemble-grid-search`). The optimal 5-way ensemble was the last to complete and was not posted.

## Limitations of These Insights

**Score source:** The task description labels the difficulty as "Easy" and the dataset size as 1,025 sequences. The prompt initially cited the champion score as 0.276 (the first chronological entry in the SOURCE log, corresponding to the ESM-1v baseline). The true final posted champion score is 0.822 (ensemble_weighted_gpu6_gpu3_001), as confirmed by the result_latest.json and autoscientists.py in the submission directory.

**Single run, no replication:** All results are from one AutoScientists run. No independent replication was performed. The meta-learner blend weights and the grid-search optimal weights may not be stable across bootstrap resamples of the dataset.

**LoRA configuration space not exhausted:** Only two LoRA configurations were tested (650M rank=8 last-8, 3B rank=16 last-12). Lower rank, different target layers, longer training, or lower learning rates were not explored. The conclusion that frozen embeddings outperform LoRA is limited to the tested configurations.

**No ablation of biochemical features:** The five physicochemical delta features (BLOSUM62, hydrophobicity, charge, MW, polarity) were included in all mid-to-late experiments but never ablated. Their independent contribution to the final 0.819–0.822 range is unknown.

**Contiguous fold generalization:** The large contiguous-fold deficit (random 0.895, contiguous 0.759 for champion) suggests the model has reduced generalization when adjacent positions are clustered together in the test set. This may indicate position-dependent fitness effects that the mean-pooled embedding cannot fully capture for SBI_STAAM's 56-residue sequence.

**The optimal 5-way ensemble (0.834) was not submitted.** If GPU3's ensemble had been posted, it would have been the registered champion by a margin of +0.012 over the 2-way ensemble. Whether the 5-way result generalizes or is over-fit to the internal CV metric is unknown.
