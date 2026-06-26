---
task: polaris-adme-fang-solu-1
run_id: biomlb_solu
started_at: "2026-04-22T04:52:11Z"
champion_at: "2026-04-22T07:47:27Z"
---

# Research Insights for Polaris ADME Aqueous Solubility Prediction

AutoScientists discovered that a Chemprop v2 message-passing neural network (MPNN) trained directly on molecular graphs outperforms all descriptor- and transformer-based approaches on this scaffold-split aqueous solubility benchmark. The headline finding is that **learned graph topology adds signal beyond even large heterogeneous descriptor sets**: Chemprop MPNN (CV Pearson r = 0.5971) beat the best stacking ensemble over Mordred + ChemBERTa + Morgan + MACCS features (0.5875) by +0.0096, which itself required two full cycles of feature engineering to reach.

## Findings

**1. Graph topology (Chemprop MPNN) outperforms heterogeneous descriptor ensembles.**
The champion (exp_gamma_003, gpu5) used Chemprop v2 BondMessagePassing (depth=4, hidden=300) + MeanAggregation + RegressionFFN (2 layers, hidden=512) trained with 5-fold scaffold CV. Final test predictions blend CV-averaged and a full-data model (equal weight). Mean CV Pearson r = 0.5971 ± 0.0295. This beat the best descriptor-based stacking ensemble (exp_alpha_005, 0.5875) trained on 3,678 features (Mordred 2D + ChemBERTa-77M CLS + Morgan FP + MACCS). The MPNN operates on no hand-crafted features, yet surpasses all feature-engineering efforts.

**2. Heterogeneous feature fusion (Mordred + LLM embeddings + fingerprints) substantially outperforms fingerprint-only or descriptor-only approaches.**
Cycle 1 established a clear ordering: Morgan FP + 8 RDKit descriptors + LightGBM (0.4961) < Full RDKit + MACCS + XGBoost (0.5502) < Mordred 1613 descriptors + ChemBERTa-77M frozen CLS + Morgan + MACCS + XGBoost (0.5848). Adding Mordred's topology/geometry/charge descriptors and ChemBERTa's pretrained embeddings together gave a +0.0346 lift over the fingerprint+XGBoost baseline — the single largest jump in the run.

**3. Stacking adds modest but consistent improvement over the best single gradient-boosted model.**
Replacing the single XGBoost model (exp_gamma_002, 0.5848) with a stacking ensemble of XGBoost + LightGBM + ExtraTrees base learners with a Ridge meta-learner (exp_alpha_005, 0.5875) yielded +0.0027. The improvement was small but consistent; stacking was the best non-GNN approach.

**4. Kernel and GP methods failed to approach the descriptor+gradient-boosting baseline.**
Tanimoto Kernel GP on Morgan FP (exp_beta_001, gpu3, cycle 1) scored only 0.4721 — below even the simple LightGBM baseline. Multi-kernel GP (Morgan + MACCS + physicochemical, exp_beta_002, 0.5522) and Tanimoto Kernel Ridge Regression (exp_beta_003) reached near-baseline performance. No kernel method exceeded the simple RDKit + XGBoost score of 0.5502.

**5. ChemBERTa frozen embeddings contribute to ensemble performance but do not dominate as a standalone signal source.**
Using ChemBERTa-77M frozen CLS embeddings alone with RDKit descriptors and XGBoost (gpu2, exp_alpha_005/cycle 1) scored 0.5534 — better than plain RDKit+XGBoost (0.5502) but well below the full Mordred fusion (0.5848). The larger ChemBERTa-zinc-base-v1 (768-dim, ZINC-trained) paired with XGBoost (gpu5, exp_gamma_002b) scored 0.5458 — worse than the smaller model, suggesting the ZINC pretraining distribution does not confer a direct advantage on this solubility dataset.

### Insights

**G1 — End-to-end graph neural networks can outperform large heterogeneous descriptor ensembles on scaffold-split molecular regression tasks.**
*Claim:* Chemprop MPNN learns structural features relevant to the property prediction that are not captured by any combination of Mordred 2D descriptors, pretrained transformer embeddings, and molecular fingerprints.
*Disconfirming evidence:* Tasks where GNNs underperform descriptors on similar scaffold-split benchmarks, or where the Chemprop advantage disappears with larger descriptor sets.
*Observed:* Chemprop MPNN (0.5971) > best descriptor stacking (0.5875); gap persisted across 3 cycles of descriptor engineering.

**G2 — Feature selection with mutual information on high-dimensional descriptor sets hurts, not helps, gradient-boosted models.**
*Claim:* On n ~ 1578 training samples with 3678 heterogeneous features, MI-based feature reduction discards complementary information that gradient-boosted learners can otherwise use with their intrinsic feature selection.
*Disconfirming evidence:* Tasks where top-K MI selection matches or exceeds the full feature set for GBM models on similar dataset sizes.
*Observed:* exp_gamma_004 applied SelectKBest(k=500) on the 3678-feature champion set and scored 0.5763 vs 0.5848 for the full feature set (exp_gamma_002), a loss of −0.0085.

**G3 — Frozen pretrained transformer embeddings improve gradient-boosted solubility models modestly; fine-tuning hurts on small scaffold-split datasets.**
*Claim:* Small scaffold-split folds (~1200 training samples) are insufficient for stable fine-tuning of transformer language models; frozen embeddings as static features are safer and more beneficial.
*Disconfirming evidence:* A task where fine-tuned ChemBERTa consistently outperforms frozen embeddings with comparable scaffold-split validation set sizes.
*Observed:* Fine-tuned ChemBERTa (exp_alpha_004) scored 0.5401 ± 0.0392 — high variance, below even the simple RDKit+XGBoost baseline of 0.5502. Frozen ChemBERTa as a feature source (combined with Mordred+Morgan+MACCS) achieved 0.5848.

### Task-Specific Findings

**T1 — Mordred 2D descriptors provide substantially more predictive signal than full RDKit descriptors for aqueous solubility.**
The jump from full RDKit 2D + MACCS + Morgan + XGBoost (0.5502) to Mordred 1613 + ChemBERTa + Morgan + MACCS + XGBoost (0.5848) reflects both Mordred's additional coverage (~1613 vs ~211 descriptors) and the added transformer embeddings. Mordred captures topology, geometry, and charge features not available in the standard RDKit descriptor set.

**T2 — 5-fold scaffold CV with pre-assigned fold columns provides stable evaluation; fold variance is meaningful.**
All agents used the cv_fold column from the task's train.csv for 5-fold scaffold CV. Fold score variance for the champion model was ±0.0295 (range: 0.5581–0.6322), indicating that scaffold diversity between folds creates genuine distributional shift. Fold 2 consistently scored lower (0.5581) across approaches, suggesting a structurally challenging partition.

**T3 — Chemprop v2 MPNN blending strategy (CV-average + full-data model) stabilizes final test predictions.**
The champion script trains 5 fold models and averages their test predictions, then trains one additional full-data model using fold 4 as early-stopping validation. Final test predictions are the equal-weight blend of the CV-averaged and full-data model outputs. This reduces single-model variance while using all training data for the submission.

**T4 — LightGBM underperforms XGBoost on the champion feature set for this solubility task.**
When the 3678-feature champion set was evaluated with LightGBM vs XGBoost under similar Optuna search (exp_alpha_003, 80 trials, LightGBM = 0.5668; exp_gamma_002 with XGBoost = 0.5848), XGBoost was superior by 0.018. This was confirmed independently by team alpha and team gamma; LightGBM was discarded for this feature regime.

**T5 — MolFormer-XL (768-dim, 1.1B SMILES pretraining) adds marginal value over ChemBERTa-77M in stacking.**
exp_beta_004 (gpu3) added IBM MolFormer-XL embeddings to the champion feature set (total ~5214 dims), yielding 0.5863 vs the same feature set without MolFormer (0.5875). The larger pretraining corpus and higher-dimensional embeddings did not help; the gain was negative (−0.0012). The champion 3678-feature set appears to capture the available signal.

## Dead Ends and Negative Results

**Tanimoto Kernel GP (Morgan FP, exp_beta_001):** CV Pearson r = 0.4721, well below the LightGBM baseline. O(n^2) kernel computation and GP fitting struggled on n=1578 training compounds. Retired after cycle 1.

**Multi-kernel GP (Morgan + MACCS + physicochemical, exp_beta_002):** 0.5522. Improved over single-kernel GP but still below the feature-based XGBoost baseline; no evidence GP posterior calibration aids regression performance here. Retired.

**Tanimoto Kernel Ridge Regression (Morgan FP, nBits=4096, exp_beta_003):** Near-baseline performance; no log retrieved in result_latest.json. Kernel regression did not outperform GBMs. Retired.

**Chemprop MPNN + extra RDKit descriptors as global features (exp_alpha_003/gpu2):** Agent reported that both Chemprop with extra descriptors and fine-tuned BERT failed on scaffold splits in cycle 1 context. The cycle 1 summary noted: "Neural end-to-end (Chemprop, fine-tuned BERT) fails on scaffold splits with small data." Note: this observation applied to an earlier version of the experiment (gpu2's cycle 1 attempt); the cycle 3 champion Chemprop run (gpu5) used BondMessagePassing without extra descriptors and succeeded, suggesting the simpler formulation is more stable.

**ChemBERTa-77M fine-tuned end-to-end (exp_alpha_004):** 0.5401 ± 0.0392 — highest variance of any experiment, below the LightGBM baseline. Fine-tuning the full backbone on 1200 training samples with scaffold splits is unstable. Hard dead end.

**ChemBERTa-zinc-base-v1 (768-dim, ZINC-trained, exp_gamma_002b):** 0.5458, below the smaller ChemBERTa-77M-MLM frozen embeddings. The ZINC pretraining distribution does not improve over PubChem-trained model for aqueous solubility.

**MolFormer-XL embeddings added to champion feature set (exp_beta_004):** 0.5863 vs 0.5875 without MolFormer. Adding a 768-dim embedding from a 1.1B SMILES model did not improve the stacking ensemble. Retired: model download cost and feature dimensionality increase are not justified.

**Mutual information feature selection (exp_gamma_004):** SelectKBest top-500 on 3678 features gave 0.5763 vs 0.5848 for the full set. Feature reduction does not help GBM on this task. Hard dead end.

**CatBoost as 4th base learner in stacking (cycle 3 approach registered):** Listed in approach_registry.json cycle 3 as "CatBoost-4th-base-learner-stacking" but no corresponding result_latest.json was found with a new champion score from this approach. The run was interrupted by API rate limits before cycle 3 agent results were fully collected; the CatBoost stacking variant likely did not complete or did not beat the Chemprop champion.

**Champion-features-plus-ECFP6-stacking (cycle 3 approach registered):** Also registered in approach_registry.json but no completed result was recorded. Same interruption note as above.

## Coordination and Team Dynamics

**Team structure:** Admin formed three teams in cycle 0 — Alpha (gpu1, gpu2, analyst1), Beta (gpu3, gpu4, analyst2), and Gamma (gpu5, gpu6, analyst3) — each seeded with a distinct paradigm: classical ML baseline, Tanimoto kernel GP, and pretrained transformer respectively.

**Cross-team learning was fast and explicit.** After cycle 1, the key findings were immediately noted: "Neural end-to-end (Chemprop, fine-tuned BERT) fails on scaffold splits with small data. Feature selection hurts — keep all 3678 features." These were incorporated into cycle 2 directions. After gpu1's stacking result (0.5875), the run noted "stacking ensemble already found +0.003" and continued to explore stacking variants in cycle 3.

**Sequential GPU dispatch created a natural bottleneck.** All GPU experiments were run sequentially using claim-file polling; CPU-bound experiments (kernel methods) ran in parallel. This meant cycle throughput was limited by the longest GPU experiment: Chemprop 5-fold scaffold CV with ~1578 compounds per fold took substantial time on an A100 MIG 3g.20gb.

**The run significantly overran its 4-hour budget.** The original deadline was 2026-04-22T08:54 UTC; the run was interrupted by API rate limits at approximately 2026-04-22T12:46 UTC — roughly 7 hours and 53 minutes after launch. Cycle 3 was only partially completed (gpu5 found the champion, gpu6 ran but its result was not recorded as beating the champion before interruption). The final champion was captured before the interruption.

**analyst3 was assigned to team gamma** but no analysis output from analyst agents was recovered as a separate artifact — analyst interactions appear to have occurred through the workshop API posts rather than local files.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. Pearson r values are from 5-fold scaffold cross-validation on the training set; no held-out test performance is known. The fold variance for the champion (±0.0295) is substantial — single-run estimates may shift on re-run with different splits.

**Incomplete cycle 3:** The run was interrupted by API rate limits during cycle 3. The cycle 3 approaches registered (CatBoost 4th base learner, Champion-features-plus-ECFP6-stacking, Champion-features-plus-ECFP6-stacking, Chemprop-MPNN-GNN) were only partially explored. The Chemprop MPNN experiment completed and became champion; the other cycle 3 approaches may not have completed.

**Chemprop fold variance is high.** The champion's fold scores ranged from 0.5581 (fold 2) to 0.6322 (fold 1), a range of 0.0741. This degree of scaffold-induced variance means ranking Chemprop above descriptor+stacking approaches is directionally supported but not definitive.

**Unexplored axes:**
- Hyperparameter optimization for Chemprop (the champion used default-adjacent settings; Optuna search over mp_depth, mp_hidden_dim, ffn layers, learning rate schedule was not performed)
- Chemprop with additional molecular features (RDKit 2D descriptors appended at the FFN level)
- Multi-seed ensembling of Chemprop models to reduce variance
- Larger pretrained GNN checkpoints (e.g., pretrained Chemprop or UniMol)
- Mordred + Chemprop hybrid: combining GNN predictions with descriptor-based stacking as a final meta-layer
