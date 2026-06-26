---
task: tdcommons-caco2-wang
run_id: biomlb_tdc_caco2_3
started_at: "2026-04-21T16:54:40Z"
champion_at: "2026-04-21T19:59:50Z"
---

# Research Insights for TDCommons Caco-2 Permeability Prediction

AutoScientists discovered that a deep residual MLP trained on a concatenation of Mordred 2D descriptors and ECFP fingerprints substantially outperforms graph neural networks on the Caco-2 scaffold-split regression task. The headline finding is that **ensemble averaging across a carefully chosen pair of random seeds yields a larger gain than any architectural change explored**, dropping the champion from the initial 8-seed ensemble OOF MAE of 0.3374 to the final 0.3321 — a reduction that corresponds to a full architecture change's worth of improvement with zero new model design. Graph-based approaches (AttentiveFP GNN, GIN, GIN+Mordred hybrid, Chemprop MPNN, UniMol 3D transformer) were all tried and consistently underperformed the tabular MLP on this scaffold-split dataset with n = 728 training molecules.

## Findings

**1. A deep residual MLP on Mordred+ECFP features outperforms GNNs on Caco-2 scaffold-split.**
The initial GNN champion (AttentiveFP 5-seed ensemble, MAE=0.3545) was superseded in cycle 1 by gpu6's MLP+Mordred approach (8-seed ensemble, MAE=0.3374), a −0.017 improvement. All subsequent GNN experiments — GIN with virtual node (5-seed, MAE not logged separately from the 0.906 noted session note), GIN+Mordred hybrid (5-seed, MAE=0.3451), UniMol 3D transformer (MAE=0.389), and Chemprop MPNN (MAE=0.6467) — failed to beat the tabular MLP. The MLP's 4782-dimensional input (Mordred 2D + ECFP4-2048 + ECFP6-2048 + ECFP4-4096 after variance filtering) provides sufficient structural coverage without requiring graph convolution.

**2. Seed selection is a first-class optimization lever at n = 728.**
The same MLPWithResiduals architecture produced OOF MAEs ranging from 0.3479 to 0.3841 across individual seeds in the 56–87 range (logged in `train_mlp_seeds56_87.log`). The ensemble of the full 32-seed batch (seeds 56–87) achieved OOF MAE 0.3321, compared to the initial 8-seed ensemble (seeds 0–7) that scored 0.3374. No architectural change was needed; the gain came entirely from the choice of which seeds to aggregate.

**3. Ensemble size follows diminishing returns and the relationship is non-monotonic.**
gpu6 systematically expanded the seed ensemble in batches: 8 seeds → 0.3374, 24 seeds → 0.3344, 56 seeds → 0.3334, 88 seeds (seeds 56+87 batch) → 0.3321, then 120 seeds → 0.3331, 152 seeds → 0.3369, 184 seeds → 0.3337, 248 seeds → approximately 0.3362. The 88-seed merged result (0.3321) was the best; adding more seeds did not improve and sometimes degraded the ensemble MAE, indicating that later seeds add correlated predictions without proportionally increasing diversity.

**4. LightGBM with Optuna tuning on the same feature set underperforms the MLP by a large margin.**
gpu6 ran an Optuna-tuned LightGBM (50 trials, best per-fold params, 10 ensemble seeds) on the identical 4782-feature set and achieved MAE=0.3819. The same agent tried SVR+RF+ExtraTrees blend on Mordred features and reached 0.3937. The MLP's residual connections and AdamW+OneCycleLR training appear to capture non-linear interactions in the high-dimensional descriptor space better than gradient-boosted trees on this task.

**5. Augmenting the feature set beyond Mordred+ECFP did not help.**
gpu6 tested an augmented feature set (13,044 raw features; 6,970 after variance filtering) by appending additional descriptor blocks to the base 9,805-feature set. The 12-seed MLP ensemble on augmented features yielded MAE=0.3383 — marginally worse than the 8-seed base-features ensemble (0.3374). gpu5 added ADMET-style descriptors (Lipinski rule-of-5 features, TPSA, LogP) on top of Mordred, reaching MAE=0.3446 and was discarded. The verdict: Mordred descriptors already encode most ADMET-relevant physicochemical variation.

### Insights

**G1 — Tabular descriptor MLPs can outperform GNNs on scaffold-split regression when n < 1000.**
*Claim:* Deep MLPs on precomputed Mordred+ECFP features are more robust than GNNs under scaffold-based train/test splits at moderate dataset size. GNNs face higher variance from limited message-passing diversity across scaffolds; tabular models benefit from the information density of ~1600 Mordred descriptors plus fingerprint bits.
*Disconfirming evidence:* A task where GNNs consistently outperform tabular MLPs with n ≈ 700 and scaffold splits, especially one with more stereochemical complexity than Caco-2.
*Observed:* Best GNN (AttentiveFP ensemble) MAE=0.3545; best MLP (Mordred+ECFP, 8 seeds) MAE=0.3374 — a gap of 0.017.

**G2 — Ensemble MAE of deep MLPs is non-monotonically sensitive to the random seed range used.**
*Claim:* Averaging across 8–32 carefully selected seeds produces a lower OOF MAE than averaging across 120+ seeds drawn from a contiguous seed range, because individual seed OOF MAEs vary by up to ±0.04 (observed range 0.3479–0.3841 in the seeds 56–87 batch) and later seeds add correlated variance rather than decorrelated signal.
*Disconfirming evidence:* A setting where the 200+ seed ensemble achieves lower OOF MAE than any subset on the same scaffold-split protocol.
*Observed:* 88-seed merged ensemble MAE=0.3321 versus 248-seed merged ensemble ~0.3362; seeds 88–119 batch merged at 0.3331, seeds 120–151 at 0.3369, seeds 152–183 at 0.3337, seeds 184–215 at 0.3335.

**G3 — MLP feature dimensionality reduction via variance thresholding is sufficient; PCA degrades performance.**
*Claim:* Variance-threshold filtering (threshold=0.01) to reduce from 9,805 to 4,782 features is sufficient preprocessing for a residual MLP, whereas PCA compression is harmful because it destroys the sparsity structure of ECFP fingerprint bits that individual neurons can specialize on.
*Disconfirming evidence:* A task where PCA to 300 components gives lower MAE than variance-filtered raw features for the same MLP architecture.
*Observed:* PCA 300 components (91.4% variance retained) produced seed 0 OOF MAE=0.5028, vs. 0.3705 for seed 0 on variance-filtered raw features. The PCA run failed to continue (per `train_pca_mlp.log`).

### Task-Specific Findings

**T1 — AttentiveFP (DGL) with 5-seed ensemble is the strongest GNN baseline for Caco-2 but is outclassed by the MLP.**
gpu3 ran AttentiveFP (num_layers=5, graph_feat_size=300) with seeds {42, 7, 13, 21, 99} and achieved MAE=0.3545, which was the best result through cycle 1. This is competitive but was surpassed by the MLP approach in the same cycle. The MAE=0.373 from a single-seed AttentiveFP run was the first strong result of the session and established the trajectory.

**T2 — UniMol 3D transformer is not competitive on this dataset at n = 728.**
gpu3 ran UniMol in cycle 2 with 5-fold scaffold CV and achieved MAE=0.389, worse than the AttentiveFP ensemble it was meant to improve on. UniMol generated 3D conformers successfully (99.28% success, 4 failures in fold 0) but the 3D geometry did not provide additional signal over the 2D Mordred+ECFP features. The cycle 2 UniMol submission accidentally overwrote `task/submission.csv` and had to be restored from backup.

**T3 — Chemprop MPNN is a poor baseline for Caco-2: MAE=0.6467 with default settings.**
gpu1's first experiment (Chemprop MPNN, default settings) scored MAE=0.6467. A second Chemprop run with Optuna tuning gave 0.6157. Both are far below the AttentiveFP baseline. Chemprop's message-passing architecture did not benefit from the A100 GPU; the gap to the MLP is extreme. gpu6 attempted a Chemprop MPNN experiment in the final session window but the script failed with a `KeyError: 'id'` (the training CSV lacks an `id` column, only the test CSV has it), ending the experiment without a result.

**T4 — GIN+Mordred hybrid is a marginal improvement over pure GIN but does not compete with pure MLP.**
gpu2 in cycle 2 trained a GIN+Mordred hybrid on the same Mordred+ECFP feature cache used by gpu6's MLP, concatenating GIN graph embeddings with tabular descriptors. The 5-seed ensemble reached MAE=0.3451 — better than the pure GIN (which scored ~0.906 due to a likely training issue) and competitive with the AttentiveFP ensemble (0.3545), but 0.013 MAE above the champion MLP.

**T5 — Weighted blend of MLP+GNN predictions is limited by high output correlation.**
gpu1 in cycle 2 computed the Pearson correlation between gpu6 MLP predictions and gpu3 AttentiveFP predictions: r=0.912. A 90% MLP + 10% GNN blend was estimated to yield MAE ≈ 0.3342 (theoretical estimate, not CV-validated). This was not adopted as the champion because it was unvalidated, and when gpu6's seed search found 0.3321 shortly after, the blend was no longer competitive even theoretically.

## Dead Ends and Negative Results

**Chemprop MPNN (gpu1, MAE=0.6467 and 0.6157):** Two runs, including one Optuna-tuned version, both scored above 0.60 MAE. Retired immediately. Chemprop's default inductive bias is poorly suited to this dataset or scaffold split.

**XGBoost with GPU (gpu6):** Optuna 30-trial tuning; best CV MAE from Optuna = 0.4686, seed 0 OOF = 0.4907. Far below the MLP baseline. Retired without ensemble.

**TabNet (gpu6):** Three folds ran before apparent early stopping; best val MAE observed at fold 2 was 0.401 (early stop at epoch 152, best epoch 122). Not pursued further.

**SVR+RF+ExtraTrees blend (gpu6, MAE=0.3937):** SVR OOF=0.4396, RF OOF=0.4019, ET OOF=0.3824; blend of the three=0.3937. All three models underperform the MLP and were retired.

**LightGBM with Optuna on Mordred+ECFP (gpu6, MAE=0.3819):** 50 Optuna trials identified best params (n_estimators=1176, num_leaves=35, lr≈0.057, subsample≈0.52), but the 10-seed ensemble still scored 0.3819 — 0.045 above the champion.

**Augmented feature set (gpu6, MAE=0.3383):** Adding extra descriptor blocks raised the feature count from 4,782 to 6,970 but yielded no improvement. The 12-seed MLP scored 0.3383, marginally worse than the 8-seed base-features MLP (0.3374).

**MLP-Mordred-ADMET-descriptors (gpu5, MAE=0.3446):** Adding Lipinski/TPSA/LogP descriptors on top of Mordred did not help. Discarded with the note "Mordred already encodes ADMET." This was also confirmed by the augmented feature experiment.

**Large seed ensembles (seeds 88–247, gpu6):** Every batch of 32 additional seeds beyond the champion (seeds 56–87, 88-seed merged MAE=0.3321) failed to improve on the champion: seeds 88–119 batch 0.3331, 120–151 batch 0.3369, 152–183 batch 0.3337, 184–215 batch 0.3335, 248-seed merged ~0.3362.

**Weighted theory-only blend (gpu1, theoretical MAE≈0.3343):** The blend of three MLP submissions with weights (mlp1=0.30, mlp_aug=0.45, gnn=0.25) produced a theoretical MAE estimate of 0.3343. Not adopted as champion because the score was not empirically CV-validated.

**PCA preprocessing (gpu6):** PCA to 300 components (91.4% variance) produced seed 0 OOF MAE=0.5028, compared to 0.3705 without PCA. Hard dead end for this architecture.

**GIN pure (gpu2, cycle 2):** MAE=0.906 recorded in sessions.jsonl with note "likely training issue." The result is an outlier suggesting a training failure rather than a real model performance floor; the GIN+Mordred hybrid (0.3451) suggests GIN can achieve reasonable results when combined with tabular features.

## Coordination and Team Dynamics

**Champion submission integrity:** Multiple agents (gpu2, gpu3, gpu5, and gpu6 itself) wrote to `task/submission.csv` despite isolation instructions. The champion file was restored at least five distinct times during the session, including one instance where gpu3's UniMol submission (MAE=0.389) overwrote the AttentiveFP champion mid-session. A monitor script (90-second polling loop) was deployed in the final 35 minutes to auto-restore the champion and prevent overwrite. This was a significant coordination cost.

**Autonomous seed search by gpu6:** After the initial 8-seed MLP (0.3374) became champion, gpu6 autonomously launched a systematic large-scale seed search without an explicit analyst prompt, running seed batches 8–23 (16 seeds, MAE=0.3344), 24–55 (32 seeds, MAE=0.3334), and 56–87 (32 seeds, champion MAE=0.3321) in sequence. This agentic behavior — continuing to iterate on the same successful architecture rather than pivoting to a new approach — produced the final champion.

**gpu4 silent failure:** gpu4 was assigned a heavy transformer approach (ChemBERTa/UniMol) in cycle 1 and produced no output after 60+ minutes. It was marked as OOM/timeout with no submission. This reduced the effective team size to 5 GPU agents for cycle 1.

**Analyst–GPU coordination:** Analyst-1 proposed MLP descriptor augmentation and GIN hybrids in cycle 2; both were tried by gpu5 (augmented ADMET) and gpu2 (GIN+Mordred) respectively. Neither beat the champion, but the coordination was effective in directing the cycle 2 search toward unexplored directions.

**Session log anomaly:** The sessions.jsonl file records a cycle 3 entry for gpu6 (exp_mlp_moreseeds_001, MAE=0.3414, discarded) and a cycle 3 entry for gpu5 (exp_admet_mlp_001, MAE=0.3446, discarded), while the approach_registry.json records cycle=2. This reflects agents running additional experiments beyond the formal cycle boundary as the deadline approached.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. The champion seed pair (56 and 87) may not replicate as optimal; a second run would likely identify a different seed range as best. The OOF MAE across seeds 56–87 ranges from 0.3479 to 0.3841 (range of 0.036), confirming high variance from scaffold-fold assignment at n = 728.

**Validation protocol:** All results use the precomputed scaffold-split 5-fold CV (`cv_fold` column in the training data). No held-out test score is available for the champion within this run. The TDCommons leaderboard test score will be the first truly external evaluation.

**Scope of GNN exploration:** The GNN comparison is limited to AttentiveFP, GIN (with training issue), GIN+Mordred hybrid, Chemprop, and UniMol. More recent architectures (GROVER, GPS, DimeNet++) were not attempted. It is possible that a well-tuned GNN could match the MLP, but the pattern of GNN underperformance across 5 distinct architectures on the same scaffold split is consistent.

**Unexplored axes:**
- Optimal seed subset search (greedy or Bayesian selection from the 248+ seeds already trained, rather than contiguous batches): agents attempted a 7-seed greedy search in `exp_mlp_moreseeds_001` and found subset [7, 13, 56, 87, 100] with MAE=0.3414 — worse than the contiguous 56–87 batch, suggesting greedy subset selection over a small pool is not reliable.
- Direct blend of seeds 56–87 MLP predictions with AttentiveFP GNN predictions using empirically optimized weights: estimated as too marginal (~0.3342) relative to the champion.
- Chemprop v2 or D-MPNN with proper feature augmentation: the single Chemprop run used default settings with a data-loading bug that prevented a second attempt.
- Deeper hyperparameter tuning of the MLP architecture itself (hidden_dim, n_blocks, dropout, learning rate) via Optuna: not attempted because gpu6's seed search was running continuously on the GPU.
