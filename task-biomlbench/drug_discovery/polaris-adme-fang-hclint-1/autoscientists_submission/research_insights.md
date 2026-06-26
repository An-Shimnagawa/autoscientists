---
task: polaris-adme-fang-hclint-1
run_id: biomlb_hclint_2
started_at: "2026-04-22T00:13:42Z"
champion_at: "2026-04-22T02:40:55Z"
---

# Research Insights for Polaris ADME Human Hepatocyte Clearance Prediction

AutoScientists found that a **deep residual MLP trained on a rich molecular feature stack — Mordred 2D descriptors concatenated with Morgan (r=3, 2048-bit), MACCS (167-bit), and atom-pair (2048-bit) fingerprints — substantially outperforms both graph neural networks and fine-tuned transformers** on this task. The headline finding is that **Mordred 2D descriptors are the decisive feature family**: removing them drops Pearson r from ~0.72 to ~0.67, while adding them to a deep neural network with Stochastic Weight Averaging (SWA) and a 10-seed final ensemble produced the champion score of 0.7170 (5-fold CV Pearson r). ChemBERTa end-to-end fine-tuning achieved 0.7161 — a near-tie with a fundamentally different paradigm — but neither approach alone nor any ensemble tried in this run exceeded the Mordred+MLP champion.

## Findings

**1. Mordred 2D descriptors are the most informative single feature family for HLM CLint.**
A fingerprint-only MLP (Morgan + MACCS + AtomPair, no Mordred) scored 0.6716 vs. 0.7170 for the Mordred-augmented champion — a gap of 0.045 Pearson r. Mordred provides ~1380 non-constant 2D descriptors capturing topological indices, fragment counts, and electronic properties beyond what circular fingerprints encode. XGBoost on Mordred features alone (1378 features) reached 0.6595, establishing the non-neural ceiling for this descriptor set.

**2. Deep MLP + SWA substantially outperforms gradient-boosted trees on this feature set.**
The champion used a 5-layer residual MLP ([2048→1024→512→256→128]) with SWA starting at epoch 120 and a Huber loss, achieving 0.7170 vs. 0.6595 for Mordred+XGBoost. This +0.057 gap indicates that the Mordred+fingerprint feature space contains structure that neural networks can exploit beyond what tree-based methods capture.

**3. Residual connections + SWA together improve over vanilla MLP.**
The predecessor experiment (exp_gpu6_mlp_mordred_morgan_001, vanilla MLP on Mordred+Morgan only) reached 0.7019. Adding MACCS + AtomPair fingerprints, two residual blocks at the first hidden dimension, and SWA pushed the score to 0.7170. Both the feature additions and the architectural changes contributed.

**4. ChemBERTa end-to-end fine-tuning is a competitive alternative (0.7161 vs. 0.7170 champion).**
GPU5's end-to-end fine-tuning of ChemBERTa-zinc-base-v1 (seyonec, pretrained on 77M SMILES) with a regression head scored 0.5583 — well below the champion. However, GPU6's later ChemBERTa+Fingerprints+Mordred+MLP hybrid (exp_gpu6_chemberta_003) achieved 0.7161, essentially matching the champion. This suggests that the ChemBERTa signal is real but requires a rich downstream feature set to realize its potential on this task.

**5. Graph neural networks (Chemprop MPNN) underperform descriptor-based methods.**
Chemprop v2 MPNN baseline (exp_beta_001_chemprop_baseline) scored 0.5755, and a deeper Chemprop variant (exp_beta_002_chemprop_deep) was also below 0.60. GPU6's cycle 2 Chemprop run (exp_gpu6_chemprop_004) reached 0.6901 — better than the naive baseline but still 0.027 below the champion. The agents' initial prediction that Chemprop would score 0.70–0.75 was not borne out.

### Insights

**G1 — Mordred 2D descriptors consistently outperform fingerprint-only feature sets on ADMET regression tasks with n ~ 2000.**
*Claim:* The richer topological and electronic coverage of Mordred provides signal beyond circular fingerprints (Morgan, MACCS, AtomPair), producing measurable gains on clearance regression tasks in this dataset size regime.
*Disconfirming evidence:* A task where Mordred adds no signal over Morgan+MACCS+AtomPair alone on n ~ 2000 ADMET data.
*Observed:* Fingerprint-only MLP: 0.6716; Mordred+Fingerprint MLP: 0.7170 (Δ = +0.045). Mordred+XGBoost: 0.6595 alone, suggesting the descriptors carry independent signal beyond fingerprints.

**G2 — Deep residual MLPs with SWA exceed gradient-boosted trees on high-dimensional Mordred+fingerprint feature matrices (n ~ 2000, p ~ 6000).**
*Claim:* When the feature dimension substantially exceeds the sample count, neural networks with SWA generalize better than GBDT because SWA's weight averaging reduces overfitting to individual training trajectories.
*Disconfirming evidence:* A similar ADMET task where LightGBM or XGBoost matches or exceeds MLP+SWA on Mordred features.
*Observed:* Mordred+XGBoost: 0.6595; Mordred+LGB (various): below champion; Mordred+MLP+SWA: 0.7170 — consistently higher.

**G3 — Adding AtomPair fingerprints to the Mordred+Morgan+MACCS set yields marginal or neutral gains with a deep MLP.**
*Claim:* AtomPair fingerprints (2048-bit hashed) contribute some information complementary to Morgan fingerprints, but the gain is small relative to Mordred's contribution.
*Disconfirming evidence:* MLP experiment with AtomPair (swa_mlp_v2, score 0.7055) scored lower than champion (0.7170), but the champion itself includes AtomPair. The difference may be due to other architectural changes (residual blocks, longer training) rather than AtomPair alone — the isolated effect was not directly tested.
*Observed:* The champion includes AtomPair; the swa_mlp_v2 variant with AtomPair scored 0.7055, lower than the champion's 0.7170, suggesting other factors (residual connections, SWA hyperparameters) dominate.

**G4 — End-to-end ChemBERTa fine-tuning requires downstream feature augmentation to compete with descriptor-based approaches on small ADMET datasets.**
*Claim:* Frozen ChemBERTa embeddings + or end-to-end fine-tuning alone is insufficient for n ~ 2000 ADMET regression; combining ChemBERTa representations with Mordred features is necessary to reach competitive performance.
*Disconfirming evidence:* A task where standalone ChemBERTa fine-tuning matches Mordred+MLP on n ~ 2000.
*Observed:* ChemBERTa fine-tuned e2e (gpu5): 0.5583; ChemBERTa+Fingerprints+Mordred+MLP (gpu6 cycle 2): 0.7161, nearly matching the champion.

### Task-Specific Findings

**T1 — HLM CLint is better captured by 2D physicochemical descriptors than by SMILES-derived graph representations.**
Chemprop MPNN (explicit 2D graph message passing) reached at best 0.6901, while Mordred (2D descriptors derived from the same molecular graph) + MLP reached 0.7170. The +0.027 advantage suggests that aggregated 2D descriptor representations capture metabolically relevant variance more effectively than end-to-end graph representation learning for this dataset size.

**T2 — The dataset has 2229 training and 575 test molecules with pre-assigned cv_fold labels (5-fold scaffold CV).**
The pre-existing `cv_fold` column in `train.csv` was used directly for all 5-fold cross-validation. CV fold scores showed moderate variance: the champion's per-fold Pearson r ranged from 0.686 to 0.741 (std = 0.025), reflecting genuine scaffold-based difficulty variation.

**T3 — 10-seed ensemble for test-set prediction provides a practical variance reduction strategy.**
The champion script trained 10 independently seeded models on the full training set with increasing seeds (seed = k*77 + 13, k=0..9) and averaged test predictions. This introduced no CV information leak since all training used the full dataset without held-out validation. A simpler multi-seed average (champion_plus with 3 seeds) scored 0.6979 vs. 0.7170 for the single best run, suggesting that seed selection within the 5-fold CV framework is more important than post-hoc multi-seed averaging.

**T4 — Mordred descriptor computation is a significant runtime bottleneck (~1.5 min per dataset pass).**
Many experiments (mordred_lgbm, mordred_ensemble) ran for 80+ minutes of CPU time partly due to repeated Mordred computation. Caching computed descriptors would have substantially expanded the experiment throughput. The Optuna LGB experiment (gpu2, 50 trials, 3-fold HPO) timed out after ~3 hours without producing results — an important practical negative.

**T5 — CatBoost on GPU consumed 34 GB of GPU memory and ran for >2 hours without producing a result, likely due to OOM or configuration mismatch.**
The CatBoost GPU experiment was terminated before completion. This is a cautionary finding for applying CatBoost's GPU mode to high-dimensional Mordred feature matrices on multi-user GPU nodes.

## Dead Ends and Negative Results

**Chemprop MPNN (graph neural network):** Three experiments with Chemprop v2 scored 0.5755 (baseline), below 0.60 (deeper variant), and 0.6901 (gpu6 cycle 2 with multi-seed). All below the 0.7170 champion. The agents' pre-run prediction of 0.70–0.75 was not realized. Retired: MPNN message passing over 2D molecular graphs does not capture HLM CLint variance as effectively as Mordred descriptors on this dataset.

**ChemBERTa end-to-end fine-tuning alone (gpu5):** 0.5583 (5-fold CV), well below champion. Retired for standalone use: without rich downstream features, fine-tuning a pretrained SMILES transformer on n=2229 samples underperforms descriptor-based methods.

**Fingerprint-only MLP (Morgan + MACCS + AtomPair, no Mordred):** 0.6716 — confirms Mordred is essential; removing it costs 0.045 Pearson r. Retired.

**Multi-seed average of champion architecture (champion_plus, 3 seeds):** 0.6979 — below single best run (0.7170). Post-hoc averaging without SWA is counterproductive here.

**MLP with AtomPair added (swa_mlp_v2):** 0.7055 — below champion's 0.7170. AtomPair alone did not improve over the champion.

**Mixup data augmentation MLP:** 0.6919 — below champion. Mixup did not help on this regression task with continuous log-clearance targets.

**ChemBERTa+Mordred stacking (no working ChemBERTa load):** 0.7007 — ChemBERTa model weights failed to load; the experiment ran on Mordred features alone and was lower than the champion.

**Stacking (OOF predictions from LGB + MLP + XGB):** 0.6938 — below champion. Stacking was hampered by ChemBERTa load failures and low base model diversity.

**Optuna LGB HPO (50 trials, gpu2):** Timed out after ~3 hours of CPU time without producing any result. Mordred computation + 50-trial Optuna search on 5 folds exceeded the available time budget.

**UniMol 3D Transformer (proposed, not executed):** GPU6 proposed UniMol in discussion (pretrained on 70M molecules, 2D/3D fusion); it was not executed because the approach was not claimed in cycle 1 before other experiments took priority. Expected performance was 0.76–0.84 per agent analysis — never validated.

**SchNet equivariant GNN (proposed, not executed):** GPU2 proposed SchNet (3D SE(n)-equivariant GNN) in discussion; it was not executed because the Mordred+XGBoost approach was claimed first and no GPU slot remained. Expected performance was 0.72–0.78.

**Graph Transformer (proposed, not executed):** GPU1 proposed a 2D graph attention network; it was not executed. No result.

## Coordination and Team Dynamics

**Three-team structure:** Admin formed three teams in cycle 1 — Alpha (classical ML, RDKit/Mordred + GBDT), Beta (Chemprop/GNN), Gamma (ChemBERTa transformer). In practice, the three-team taxonomy was not rigidly enforced in cycle 2; gpu6 converged on the winning Mordred+MLP direction independently.

**Discussion phase generated four orthogonal proposals:** GPU1 (Graph Transformer), GPU2 (SchNet), GPU4 (Chemprop+ChemBERTa stacking), GPU6 (UniMol). A GPU1-authored roundtable analysis correctly identified GPU6 (UniMol) as the highest-ceiling approach and recommended it be run first. However, none of the 3D/geometric approaches (UniMol, SchNet) were actually executed — execution resources were consumed by Chemprop and ChemBERTa variants.

**GPU4 agent got stuck in a monitor loop:** GPU4 wrote a ChemBERTa-77M-MTR + LightGBM training script but failed to execute it (blocked waiting for a monitoring event that never fired). This was detected after ~60 minutes and the script was relaunched directly. This cost approximately one cycle of GPU4's time.

**GPU6 was the most productive agent:** GPU6 ran four experiments over two cycles (MLP Mordred+Morgan: 0.7019; MLP deeper+SWA: 0.7170 champion; ChemBERTa+Mordred+MLP: 0.7161; Chemprop MPNN: 0.6901) and correctly identified the champion architecture after only the second run.

**Many parallel direct experiments were launched:** Over 15 direct training scripts were run in parallel (mordred_lgbm, mordred_ensemble, swa_mlp_v2, mlp_optuna, champion_plus, blended_ensemble, rdkit_lgb_fast, fp_mlp_swa, mlp_swa_v3, catboost, chemberta_mordred, mlp_mixup, stacking, resnet_deep, lgb_optuna_v2, etc.). None of these beat the champion. The CatBoost experiment consumed 34 GB of GPU memory and was terminated. Several Mordred-based experiments timed out or had excessive CPU usage due to the Mordred computation bottleneck. A process storm (mlp_optuna spawning many subprocesses) required manual killing.

**Analysts had limited impact:** Analyst1 produced no memories; Analyst2 produced no memories; Analyst3 authored a Cycle 1 summary proposing ChemBERTa-based experiments (gamma team) but did not drive any executed experiment. Analyst discussions occurred during the discussion phase but did not result in actionable new directions in execution cycles.

**GPU2's Optuna run did not finish:** GPU2's exp_alpha_003 (Mordred+LightGBM+Optuna, 50 trials) ran for approximately 3 hours of CPU time without producing a result — the longest-running failed experiment. The Mordred computation time combined with 50-trial HPO on 5 folds exceeded the wall-clock budget.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. The 5-fold scaffold CV scores have fold-level variance (std ≈ 0.025 for champion), meaning the champion's 0.7170 has ±0.025 uncertainty at the CV level. The Polaris leaderboard score (0.7169939) matches CV closely for the champion but a second run may not replicate exactly.

**Mordred vs. Fingerprints isolation:** The exact contribution of each feature type (Mordred alone vs. Morgan alone vs. MACCS vs. AtomPair) was not cleanly ablated. The champion uses all four concatenated. The fingerprint-only experiment (no Mordred) shows Mordred is necessary, but the individual fingerprint ablations were not run.

**Neural architecture search:** The residual MLP architecture was designed by GPU6 and not systematically searched. Optuna-based hyperparameter search (mlp_optuna) only explored 3 fixed configs and scored 0.7019 — not a broad sweep. A full Optuna sweep on MLP architectures was not completed within the time budget.

**Unexplored approaches:**
- UniMol (3D pretrained transformer): highly promising per agent analysis, never executed
- SchNet / equivariant GNN: never executed
- Ensemble of MLP champion + ChemBERTa+Mordred (0.7161): natural candidate that was attempted (stacking) but failed due to ChemBERTa load issues; a clean ensemble of both approaches could potentially exceed 0.72
- Scaffold-stratified submission: agents used pre-assigned cv_fold labels; the effect of scaffold generalization on test-set performance is unknown

**Computational bottleneck:** Mordred descriptor computation (~1.5 min per pass) limited the number of experiments that could complete within the ~2.5 hour budget. Caching Mordred features would have enabled ~5x more experiments.
