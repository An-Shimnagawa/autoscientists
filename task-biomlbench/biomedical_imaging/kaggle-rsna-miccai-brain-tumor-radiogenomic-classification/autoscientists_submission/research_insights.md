---
task: kaggle-rsna-miccai-brain-tumor-radiogenomic-classification
run_id: biomlb_rsna_2
started_at: "2026-04-28T03:29:35Z"
champion_at: "2026-04-28T14:28:47Z"
---

# Research Insights for Kaggle RSNA-MICCAI Brain Tumor Radiogenomic Classification

AutoScientists explored 21 approaches across 4 cycles to predict MGMT promoter methylation from multi-sequence brain MRI. The headline finding is that a **pretrained EfficientNet-B0 with per-sequence slice-attention pooling followed by a 2-layer cross-sequence Transformer encoder** is a stable foundation for this task, and that **cosine annealing warm restarts with a moderate initial cycle length (T_0=10) is the most impactful single optimization lever**. The champion (exp_gamma_011) reached 5-fold CV ROC-AUC = 0.6311 after iterating from a LightGBM classical-ML baseline of 0.5743 — progress consistent with the intrinsic difficulty of the task (the original Kaggle competition winning score was ~0.59–0.60).

## Findings

**1. Cross-sequence Transformer fusion is the core architectural win.**
Moving from a purely slice-level MLP-based architecture to a 2-layer Transformer encoder over per-sequence embeddings (exp_beta_003) produced the first major jump: AUC 0.5924 (soft-stacking ensemble baseline) → 0.6209. The Transformer's CLS-token mechanism learns non-linear correlations across the four MRI sequences (FLAIR, T1w, T1wCE, T2w), which a simple concatenation or per-sequence scalar weighting does not capture.

**2. CosineAnnealingWarmRestarts is a decisive training-schedule improvement; cycle length matters.**
Switching from fixed cosine-warmup training (exp_beta_003, 0.6209) to AdamW + CosineAnnealingWarmRestarts (T_0=5) yielded +0.0056 AUC (exp_gamma_008, 0.6265). A further increase to T_0=10 — allowing the optimizer a longer initial descent before the first LR reset — added another +0.0046 (exp_gamma_011, 0.6311). T_0=20 was tested and degraded performance (0.6039), establishing a non-monotone relationship between initial cycle length and final AUC.

**3. Backbone scaling hurts; EfficientNet-B0 outperforms B2 and larger models.**
Multiple experiments replacing the B0 backbone with B2 (exp_alpha_006, 0.5940; exp_alpha_009) consistently scored below the B0 champion. With only ~526 training patients, larger backbones overfit despite pretrained initialization. ViT-B/16 also underperformed (exp_beta_004, 0.6079), with agents explicitly attributing the gap to insufficient CNN-style inductive bias at this sample size.

**4. Label smoothing (0.05) is a reliable regularizer at this sample size.**
All experiments incorporating label smoothing (LabelSmoothingBCE, ε=0.05) in the CosineWarmRestart framework scored consistently above 0.62; removing or replacing it with FocalLoss uniformly degraded performance. The combination FocalLoss + LabelSmoothing was especially harmful (exp_gamma_009, 0.6015), causing heavy overfitting (train AUC > 0.9, rising val loss in later epochs).

**5. Ensemble methods failed to improve over single models on this task.**
A soft-stacking ensemble of 5 diverse architectures (champion Transformer, ViT-B/16, 3D ResNet, PyRadiomics+XGBoost, ResNet18) trained with a logistic regression meta-learner (exp_alpha_004) scored only 0.5924 — below the established champion of 0.6209. Analyst-1 concluded that base model predictions were highly correlated despite architectural diversity, providing no exploitable diversity signal for the meta-learner.

### Insights

**G1 — Cross-sequence fusion outperforms slice-level aggregation alone for multi-sequence MRI classification.**
*Claim:* A Transformer operating over sequence-level embeddings (one embedding per MRI sequence) captures sequence-interaction features (e.g., FLAIR edema + T1wCE enhancement co-patterns) that a per-slice or per-sequence independent head cannot.
*Disconfirming evidence:* A task where sequences are fully redundant, in which case concatenation or pooling would match Transformer.
*Observed:* exp_beta_001 (per-sequence multi-head slice attention, no cross-sequence fusion) was staged as a prior step that underperformed; exp_beta_003 (cross-sequence Transformer) reached 0.6209 from a 0.592 baseline.

**G2 — On small medical imaging datasets (n ~ 500), backbone capacity should be kept minimal.**
*Claim:* Pretrained EfficientNet-B0 provides sufficient representational power; larger backbones (B2, ViT-B/16) overfit due to excess parameters relative to training set size.
*Disconfirming evidence:* Experiments with aggressive dropout or heavy augmentation rescuing larger backbone performance.
*Observed:* B2 variants (exp_alpha_006: 0.5940, B2+Transformer cycle 2) consistently underperformed B0. ViT-B/16 (exp_beta_004: 0.6079) underperformed B0+Transformer (0.6209). Agents explicitly noted memory and overfitting as primary failure modes.

**G3 — CosineAnnealingWarmRestarts benefits from a moderate T_0; too short or too long degrades performance.**
*Claim:* An initial cosine cycle that is too short (T_0=5) causes premature LR resets before the model settles; one that is too long (T_0=20) prevents the beneficial multi-cycle exploration within the epoch budget.
*Disconfirming evidence:* Monotone improvement with increasing T_0, or tasks where T_0=5 consistently equals or beats T_0=10.
*Observed:* T_0=5: 0.6265; T_0=10: 0.6311 (+0.0046); T_0=20: 0.6039 (−0.0226 vs T_0=10). Non-monotone, with T_0=10 as peak in this experimental range.

**G4 — Ensemble meta-learning provides minimal benefit when base models share the same failure mode.**
*Claim:* For radiogenomics tasks where signal-to-noise is inherently low (weak imaging-genomics correlation), diverse architectures make similar errors because the fundamental limit is label quality rather than inductive bias. Meta-learners cannot exploit error diversity that does not exist.
*Disconfirming evidence:* A radiogenomics task with strong imaging-genomic signal where diverse architectures reach divergent per-sample predictions.
*Observed:* Analyst-1 confirmed high prediction correlation across 5 architectures post-training; exp_alpha_004 soft-stacking ensemble (0.5924) underperformed every individual deep learning model.

### Task-Specific Findings

**T1 — Slice selection from the middle 30–70% of the stack is a stable preprocessing choice.**
Sampling 8 evenly-spaced slices from the middle 30–70% of each DICOM sequence (capturing the tumor-bearing region for most cases) was established in the first cycle and never changed. Increasing to 16 slices (exp_beta_005, exp_alpha_009) did not improve performance and introduced GPU memory pressure; the 8-slice fixed-region protocol remained stable across all 21 approaches.

**T2 — 5-fold CV is necessary but exhibits high fold variance (std ~ 0.03–0.04), making single-run rankings unreliable.**
Across all experiments, per-fold AUC ranged from ~0.55 to ~0.69 within the same model. The champion (exp_gamma_011) had fold scores [0.6568, 0.6504, 0.6086, 0.5866, 0.6531] — a spread of 0.070. This variance arises from the small per-fold validation set (~105 cases) and the inherently noisy MGMT labels. Single-fold or 3-fold validation would be unreliable for this dataset.

**T3 — Vertical flip and rotation augmentations degrade performance; horizontal flip is the only safe TTA augmentation.**
exp_beta_005 (16 slices + hflip + vflip + rotation augmentation) scored 0.6086, below the champion. Agents confirmed that vflip and rotation are anatomically invalid for axial brain MRI. A separate TTA test (exp_gamma_007: 4-view TTA including vflip and rotation) confirmed the finding: with-TTA AUC dropped to 0.5316 vs. no-TTA 0.5845 from the same model. Only horizontal flip is anatomically valid and safe for brain MRI.

**T4 — SWA (Stochastic Weight Averaging) with swa_lr=1e-4 degrades performance; SWA won 0 of 5 folds.**
exp_alpha_008 tested SWA over the last 5 epochs with swa_lr=1e-4, scoring 0.6033 vs. the champion's 0.6265. Fold-level comparison showed SWA underperformed the base model on all 5 folds. The agent noted swa_lr=1e-4 may be too high for this architecture/dataset combination; lower values were not explored.

**T5 — 3D volumetric modeling (3D ResNet-18) exhibits extreme fold variance and underperforms 2D slice-based approaches.**
exp_gamma_004 (3D ResNet-18 volumetric) mean AUC = 0.5792 with per-fold scores [0.6469, 0.5948, 0.5340, 0.5351, 0.5851] — a range of 0.111. Agents attributed this to (a) variable slice counts requiring lossy padding to a fixed 3D volume, and (b) CPU-intensive preprocessing for large DICOM volumes. 3D modeling was abandoned after Cycle 1.

**T6 — Classical ML (LightGBM + hand-crafted intensity/texture features) establishes a competitive floor near 0.57.**
The first experiment (exp_alpha_001: LightGBM + statistical + texture features from middle-third slices) achieved 5-fold CV AUC = 0.5743. This matches the difficulty floor observed in the Kaggle competition. The gap between this baseline and the deep learning champion (0.6311) is real but modest (~0.057), consistent with the known weak imaging-genomic signal for MGMT status.

## Dead Ends and Negative Results

**Backbone scaling (EfficientNet-B2, multiple experiments):** Tested in at least 3 independent experiments (exp_alpha_006: 0.5940, one cycle-2 variant: 0.6036). Consistent degradation vs. B0 baseline on all runs. Retired: insufficient training data (n~418 after CV split) to support B2's additional parameters.

**ViT-B/16:** exp_beta_004, AUC = 0.6079. Agents noted CNN inductive bias is beneficial for small-n medical imaging; ViT requires large-scale pretraining on domain-specific data to compete. Retired after Cycle 1.

**3D ResNet-18 volumetric:** exp_gamma_004, AUC = 0.5792, fold variance 0.111. High compute overhead, high variance, below champion. Hard dead end.

**DenseNet121 as backbone (MIL variants):** exp_gamma_005 (DenseNet121 MIL, 0.6118) and exp_gamma_006 (DenseNet121 StrongAug, 0.6204, NEAR-MISS). DenseNet121 approaches showed high fold variance and consistently underperformed the B0+Transformer architecture. Not pursued after Cycle 2.

**Multi-view TTA (vflip + rotation):** exp_gamma_007: with-TTA AUC = 0.5316 vs. no-TTA = 0.5845 from the same model. Hard dead end: vflip and rotation are anatomically invalid for axial brain MRI.

**Input-level MixUp (alpha=0.2):** exp_beta_008, AUC = 0.6133 vs. champion 0.6265. MixUp doubled GPU memory usage (~6310 MiB vs. ~3602 MiB) and was harmful in mid-folds. Input-level MixUp is not beneficial for this architecture/dataset combination.

**Focal Loss (gamma=0.5 and gamma=2.0):** exp_alpha_010 (FocalLoss gamma=2: 0.6166), exp_beta_007 (FocalLoss gamma=0.5 + LabelSmoothing: 0.6152), exp_gamma_009 (FocalLoss gamma=2 + LabelSmoothing + CosineWarmRestart: 0.6015). FocalLoss at any tested gamma consistently underperformed label-smoothing BCE. The combination of FocalLoss + LabelSmoothing was particularly destabilizing, producing severe overfitting (train AUC > 0.9, escalating val loss).

**Soft-stacking ensemble of 5 diverse architectures:** exp_alpha_004, AUC = 0.5924. Meta-learning across highly correlated base model predictions provided no benefit; the result was below the cycle-1 champion. Analyst-1 formally retired ensemble-based approaches after this failure and pivoted to single-model parameter tuning.

**Stochastic Weight Averaging (SWA):** exp_alpha_008, AUC = 0.6033. SWA won 0 of 5 folds at the tested swa_lr. Retired.

**CosineWarmRestart T_0=20:** exp_gamma_010, AUC = 0.6039. Longer initial cycle than T_0=10 is not beneficial; over-long cycles prevent multi-cycle LR oscillation within the 20-epoch budget. Retired.

**HFlipTTA at inference:** exp_alpha_011, AUC = 0.6102 (below champion 0.6265). The final retraining step also diverged (loss=NaN from epoch 15), likely exacerbated by a symlink path bug in the script's `shutil.copy(__file__)` call. The HFlipTTA contribution could not be cleanly isolated from the training instability; classified as inconclusive/negative.

**16-slice sampling:** exp_beta_005 (16 slices, RichAug: 0.6086) and exp_alpha_009 (16 slices + CrossTransformer: 0.6036). Both below champion (8 slices: 0.6209). Additional slices did not improve information content and added GPU memory pressure. 8 slices from the middle 30–70% region is sufficient.

## Coordination and Team Dynamics

**Three-team structure (Alpha/Beta/Gamma) ensured broad architectural coverage in Cycle 1.** Team Alpha focused on classical ML and lightweight CNNs; Team Beta on 2D CNN + attention/Transformer; Team Gamma on ViT, 3D CNN, and MIL approaches. This division produced diverse Cycle 1 baselines without redundant experiments.

**Analyst-1 (team-alpha) executed an effective ensemble analysis and strategic pivot.** After exp_alpha_004 (soft stacking) failed in Cycle 3, Analyst-1 diagnosed correlated base model predictions as the root cause and formally documented the finding in cycle_3_analysis.md, then redirected team-alpha GPU agents toward parameter tuning (CosineWarmRestart, FocalLoss, SWA). This pivot was rapid and well-documented.

**The cross-sequence Transformer architecture (exp_beta_003) was adopted globally.** Established by gpu3/team-beta in Cycle 1, it became the universal base architecture for all subsequent cycles across all teams. Agents explicitly built on it by changing only training schedules, augmentation, or loss functions, enabling clean A/B comparisons.

**Analyst-2 (team-beta) was active only in Cycle 1.** Analyst-2's session count was 1 and last_seen was at the start of the run (2026-04-28T03:44Z), with memory containing no later entries. Analyst-2 did not participate in Cycle 2 or later. The initial analysis by Analyst-2 (identifying cross-sequence fusion as the key axis and noting 3D volumetric weaknesses) was accurate and informed team strategy, but no further analyst guidance from this agent was available.

**Analyst-3 (team-gamma) was active through Cycle 2** but did not update memory beyond that cycle, limiting strategic guidance to the gamma team in Cycles 3–4. GPU agents on team-gamma (gpu5, gpu6) operated with more autonomy in later cycles.

**A recurring `shutil.copy(__file__)` bug (SameFileError) affected multiple experiments** (exp_alpha_010, exp_gamma_009, exp_gamma_010, exp_alpha_011). This was a script-infrastructure issue where the training script's post-run file copy failed when source and destination resolved to the same path under symlinked filesystem mounts (`/n/holylabs` vs. `/n/holylabs/LABS`). Results were manually recovered in all cases; no experiments were lost, but the bug consumed agent time for diagnosis and repair across 4 experiments.

## Limitations of These Insights

**Task label noise is the primary ceiling.** MGMT promoter methylation status has a weak and indirect imaging correlate. The original Kaggle competition's winning public score was ~0.60, and even state-of-the-art radiogenomics literature rarely exceeds 0.65 AUC on held-out data for this task. The 5-fold CV score (0.6311) is optimistic relative to the expected generalization to a truly independent test set. High fold-level variance (std ~ 0.03–0.04) means that small observed differences between experiments (< 0.01 AUC) are within measurement uncertainty.

**Single run, no independent replication.** All findings come from one AutoScientists run. The T_0=10 cosine cycle advantage (+0.0046 over T_0=5) is modest and may not replicate; a second run might find a different optimal T_0. The approach registry exhausted 21 slots across 4 cycles but did not explore some potentially impactful axes (see below).

**Unexplored axes:**
- Differential learning rates for backbone vs. Transformer head (a common transfer learning technique not tested here)
- Pretraining on medical imaging datasets (e.g., CheXpert, NIH ChestX-ray) before fine-tuning
- Multi-seed ensemble of the champion architecture (5 seeds of exp_gamma_011 were not tested as a submission-time ensemble)
- Sequence dropout / stochastic sequence masking during training as a regularizer
- Larger 5-fold CV ensembles (train-on-all-data final model averaged across multiple folds)
- Attention map visualization to validate that the cross-sequence Transformer attends to radiologically meaningful regions

**Scope of generalizability.** The cross-sequence Transformer finding likely transfers to other multi-sequence MRI classification tasks with small patient cohorts. The T_0 sweep finding is more task-specific; other architectures or loss functions may interact differently with cosine cycle length. The ensemble-failure finding (G4) is expected to generalize primarily to tasks with low imaging-genomic signal, not to imaging tasks with strong ground-truth correspondence.
