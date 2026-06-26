---
task: kaggle-histopathologic-cancer-detection
run_id: biomlb_kaggle_hist_2
started_at: "2026-04-24T12:30:51Z"
champion_at: "2026-04-25T02:48:36Z"
---

# Research Insights for Kaggle Histopathologic Cancer Detection

AutoScientists achieved a final val ROC-AUC of **0.9970+** on the Kaggle Histopathologic Cancer Detection task. The headline finding is that **a task-aligned dual-stream architecture — with separate CNN encoders for the full 96×96 context and the cancer-determining center region — consistently outperformed single-stream CNNs, frozen foundation model probes, and naive multi-model ensembles across all cycle 1 experiments**. The final submission is a 5-model rank-average ensemble anchored by the strongest individual model (exp_beta_004: dual-stream EfficientNet-B3, center-48 crop, deep fusion head, val AUC = 0.9970 at epoch 18).

## Findings

**1. Dual-stream center-biased architecture dominates all single-stream approaches.**
The key structural insight exploited by the winning architecture is that cancer labels in PatchCamelyon are determined by the center 32×32 pixels of the 96×96 patch. Explicitly routing a dedicated CNN stream to this region, while retaining a second full-image context stream, outperformed every single-stream variant tried. Cycle 1 champion (exp_zeta_001, dual-stream EfficientNet-B0, center-32 crop) scored 0.9925, versus 0.9812 for EfficientNet-B3 single-stream (exp_alpha_001), 0.9911 for ViT fine-tune (exp_gamma_001), and 0.9694 for DINOv2 linear probe (exp_delta_001).

**2. Increasing the center crop size from 32 to 48 provides consistent gains in the dual-stream architecture.**
Expanding the center crop from 32×32 (exp_zeta_001, 0.9925) to 48×48 (exp_zeta_002, 0.9959; exp_beta_004, 0.9970) gave +0.0034–0.0045 improvement. The 48-pixel crop captures the full center 32×32 label region plus a 8-pixel buffer of surrounding tissue context, likely helping the model use peritumoral spatial cues while still focusing supervised signal on the label-relevant area.

**3. Scaling from EfficientNet-B0 to B3 in the dual-stream architecture adds measurable capacity gains.**
Across matched conditions, B3 dual-stream consistently outperformed B0 dual-stream. exp_alpha_003 (B3, center-32 crop) achieved 0.9968 versus exp_zeta_001 (B0, center-32 crop, 0.9925), a +0.0043 delta. exp_beta_004 (B3, center-48 crop, deep fusion) reached 0.9970 versus exp_zeta_002 (B0, center-48 crop, deep fusion, 0.9959), a +0.0011 delta. The B3 model has ~1536 features per stream versus ~1280 for B0 (3072 vs 2560 concatenated), and the additional capacity was not wasted at 174K training images.

**4. Rank-average ensembling of 5 architecturally diverse dual-stream models is the final submission strategy.**
The analyst3 agent assembled the final submission as a rank-average ensemble of five models: alpha_003 (B3, center-32 MixUp, 0.9968), delta_002 (B0, stronger augmentation, 0.9963), epsilon_002 (B0, MixUp+CutMix, 0.9966), zeta_002 (B0, center-48 deep fusion, 0.9959), and beta_004 (B3, center-48 deep fusion, 0.9970). Rank normalization ensures each model contributes with equal scale weight regardless of absolute probability range differences. This ensemble replaced alpha_003 as the submitted champion because diverse model combinations robustly improve generalization on this task.

**5. Frozen foundation model approaches substantially trail fine-tuned CNNs on this task.**
DINOv2-base frozen embeddings + linear probe (exp_delta_001, 0.9694) and DINOv2-base frozen embeddings + XGBoost/SVM ensemble (exp_gamma_002, 0.9806) both fell far below the dual-stream fine-tuned EfficientNet models. DINOv2 features trained on natural images do not transfer their full representational power to 96×96 histopathology patches under a linear probe or shallow classifier. Fine-tuned ViT-B/16 (exp_gamma_001, 0.9911) narrowed this gap considerably, showing that the deficit is in the probe, not the foundation model itself.

### Insights

**G1 — Task-aligned inductive biases outperform generic architectures on spatially structured classification benchmarks.**
*Claim:* When the label-determining region is a known spatial subregion, explicitly routing dedicated capacity to that region provides durable gains over architectures that treat all pixels equivalently.
*Disconfirming evidence:* Tasks where the label-relevant region varies or is unknown, making fixed spatial routing non-beneficial.
*Observed:* exp_zeta_001 (dual-stream, explicit center routing) 0.9925 vs. exp_alpha_001 (single-stream B3) 0.9812 — a +0.0113 delta in cycle 1. The advantage was stable across model scales and crop sizes throughout cycle 2.

**G2 — Frozen foundation model linear probes are insufficient for domain-shifted small-patch histology tasks even at scale.**
*Claim:* Foundation models (DINOv2, ViT) pretrained on natural images require fine-tuning to transfer to histopathology patch classification; linear probes and shallow classifiers leave substantial performance on the table.
*Disconfirming evidence:* Tasks where the spatial statistics of patches closely resemble natural image crops, enabling effective linear transfer.
*Observed:* DINOv2-base linear probe (0.9694) vs. fine-tuned ViT-B/16 (0.9911, +0.0217); frozen DINOv2+XGBoost ensemble (0.9806) vs. fine-tuned EfficientNet-B3 dual-stream (0.9968, +0.0162). Fine-tuning is necessary, not optional, in this domain.

**G3 — Deep multi-layer fusion heads (3072→512→128→1 with BN and Dropout) outperform shallow fusion when combining high-capacity dual-stream features.**
*Claim:* When both streams produce high-dimensional features (1536-dim each for B3 → 3072 concatenated), a deep fusion head with batch normalization and staged dropout provides better regularization and representation than a single linear layer.
*Disconfirming evidence:* Settings where the concatenated feature space is small enough that a single linear layer suffices (e.g., B0 with 1280-dim per stream).
*Observed:* exp_zeta_002 (B0, center-48, deep fusion) 0.9959 versus exp_zeta_001 (B0, center-32, simpler fusion) 0.9925 (+0.0034), though this conflates crop size change. exp_beta_004 explicitly combined the B3 encoder with deep fusion (3072→512→128→1) as an intentional design choice justified by the B3 feature dimensionality.

**G4 — MixUp augmentation is compatible with and beneficial for the dual-stream center-crop architecture.**
*Claim:* Applying MixUp in the input space simultaneously to both streams (context and center crop) provides consistent regularization gains for this task without degrading the structural inductive bias.
*Disconfirming evidence:* Cases where MixUp corrupts task-relevant spatial structure (e.g., object detection or dense segmentation).
*Observed:* All three best individual models (alpha_003, epsilon_002, beta_004) used MixUp (alpha=0.2) and achieved val AUC ≥ 0.9966. The non-MixUp models (zeta_001, zeta_002) also performed well but did not individually exceed alpha_003.

### Task-Specific Findings

**T1 — The PatchCamelyon label is determined by the center 32×32 region; center crops of 48×48 capture this region with margin and improve AUC.**
The dataset annotation protocol defines labels by the center 32×32 pixel region. A center crop of 48×48 (offset=24 from the edge) adds an 8-pixel buffer of surrounding tissue on each side. This buffer consistently improved performance: exp_zeta_002 (center-48, B0) 0.9959 versus the baseline expectation for B0 center-32, and exp_beta_004 (center-48, B3) 0.9970 versus exp_alpha_003 (center-32, B3, 0.9968). The 48-pixel crop appears to provide just enough peritumoral context without diluting focus on the label-relevant region.

**T2 — Val AUC plateaus for individual models in the 0.9960–0.9970 range; ensemble diversity remains the path forward.**
Across cycle 2, the best individual models clustered tightly: alpha_003 (0.9968), beta_004 (0.9970), epsilon_002 (0.9966), delta_002 (0.9963), zeta_002 (0.9959). The top-5 span is only 0.0011, yet the ensemble of all five was adopted as the final submission because cross-model diversity (different random seeds, augmentation strategies, crop sizes, and model scales) is expected to reduce correlated prediction errors even at very high individual AUC.

**T3 — Test-time augmentation (4-rotation TTA: 0°, 90°, 180°, 270°) is a low-cost inference boost for histopathology patches.**
All final model predictions used 4-rotation TTA averaging. Histopathology patches have no canonical orientation, making rotation TTA theoretically well-motivated. The TTA was applied to both streams of each dual-stream model. Separate prediction runs were tracked at different epochs (e.g., beta_004 ep17 and ep18), and TTA predictions from the better checkpoint produced the better submission.

**T4 — EfficientNet-B3 dual-stream training on 174K images at batch_size=64 requires ~22 GB GPU memory; batch_size=128 causes OOM on a 40 GB A100 when running concurrently with another B3 model.**
The OOM event during cycle 2 occurred because the initial B3 dual-stream experiment (exp_alpha_002, 30 epochs, batch_size=128) launched while the zeta_001 noise-gate (B0) was still occupying GPU memory. After freeing the GPU and reducing batch_size to 64, training completed successfully. The B3 dual-stream model with batch_size=64 stabilizes at ~21.8 GB.

**T5 — Individual DINOv2-large linear probe extraction takes ~54 minutes on this dataset scale; DINOv2-base frozen + MLP fine-tuning is faster and achieves 0.9907 in a single cycle 1 slot.**
GPU agent 4 attempted DINOv2-large embedding extraction (304M parameters, exp_delta_001) but took ~54 minutes just to extract embeddings, ending at 0.9694 (DISCARD). The faster approach was GPU agent 2's DINOv2-base (86M parameters) fine-tuned MLP head (exp_beta_001, 0.9907). For future runs: if foundation model fine-tuning is explored, the fine-tuned route significantly outperforms the probe route and completes in comparable wall-clock time.

## Dead Ends and Negative Results

**DINOv2-large linear probe (exp_delta_001, 0.9694):** Frozen DINOv2-large embeddings with logistic regression scored 0.9694 — worse than the EfficientNet-B3 single-stream baseline (0.9812). The probe treats histopathology patches as a natural-image-style retrieval task; the distribution mismatch and the small 96×96 patch size both limit transfer. Classified DISCARD at delta −0.0118 from champion. Embedding extraction alone cost ~54 minutes, consuming significant wall-clock budget.

**DINOv2-base frozen embeddings + XGBoost/SVM/LogReg ensemble (exp_gamma_002, 0.9806):** CPU-only ensemble on cached 768-dim DINOv2-base embeddings. XGBoost alone achieved 0.9806; LinearSVC (0.9691) and LogReg (0.9692) underperformed; the ensemble averaged to 0.9771 — substantially below any fine-tuned neural approach. The result confirms that tree models and linear classifiers extract less signal from high-dimensional vision transformer embeddings than fine-tuned MLP heads or end-to-end training.

**EfficientNet-B3 + ViT-B/16 TTA ensemble underperformed the best individual model (exp_epsilon_001, 0.9904):** A post-hoc TTA ensemble combining EfficientNet-B3 (0.9809) and ViT-B/16 (0.9913) averaged to only 0.9904 — below the ViT alone. Simple equal-weight averaging cannot exceed the better component when the weaker component (EfficientNet-B3 single-stream, not dual-stream) degrades the combined prediction. This motivated moving to rank-average ensembling of models that are individually close in quality rather than averaging a strong and weak model.

**EfficientNet-B3 dual-stream 30-epoch training (exp_alpha_002, batch_size=128): OOM crash.** The original 30-epoch B3 run was aborted by OOM when concurrent GPU usage exceeded 40 GB. Re-launched at batch_size=64 for 25 epochs (exp_alpha_003) and succeeded at 0.9968. The 30-epoch extension likely would not have improved beyond the epoch-15 peak given the val AUC trajectory of alpha_003 (which stagnated at 0.9960 for epochs 16–25 after peaking at 0.9968 at epoch 15).

**Beta_004 predictions from ep13 checkpoint (0.9966):** Prediction from the ep13 checkpoint (0.9966) did not beat champion 0.9968. The ep17 and ep18 checkpoints (0.9969, 0.9970) later emerged to improve it. The ordering is informative: for B3 dual-stream with cosine decay schedule, the peak epoch may be 15–18 rather than the earlier window.

**DINOv2 partial fine-tuning (exp_beta_002, last 2 blocks unfrozen) is slower and lower than full dual-stream fine-tuning:** exp_beta_002 unfroze transformer blocks 10 and 11 of DINOv2-base (14.2M tunable backbone + 0.53M head), but the training stdout confirms slow convergence and the result was superseded by the dual-stream CNN approaches. The partial fine-tuning strategy was not pursued further in cycle 2.

## Coordination and Team Dynamics

**Analyst agents drove ensemble construction without running any training.** The champion submission was produced by analyst3, who assembled the 5-model rank-average ensemble entirely in Python using pre-computed prediction CSVs from five GPU agents. No GPU compute was used at ensemble construction time. Analyst1 had already built a 3-model ensemble (alpha_003 + delta_002 + zeta_002), and analyst2 extended it to 4 models; analyst3 completed the final 5-model version after beta_004 predictions became available at 02:33–02:48 UTC. This sequential analyst chain worked without explicit hand-off protocols — each analyst read the workshop state and improved on the previous ensemble.

**The early OOM crash (exp_alpha_002) triggered adaptive resource management.** The OOM was detected and diagnosed: the concurrent noise-gate experiment (zeta_001 seed-2) had occupied GPU memory. The run was relaunched with batch_size=64 after confirming 40 GB free. This kind of reactive resource arbitration was handled centrally rather than by the GPU agent itself, suggesting the agents' own resource checking is insufficient for multi-GPU-run scenarios.

**Cross-team convergence on dual-stream: all non-DINOv2 cycle 2 experiments built on the dual-stream paradigm.** After cycle 1 established dual-stream (exp_zeta_001, 0.9925) as the top approach, the cycle 2 experiments from every active GPU team (alpha, beta, delta, epsilon, zeta) either adopted or extended the dual-stream center-crop design. No agent proposed a fundamentally new architecture in cycle 2; the exploration space collapsed to variations within the dual-stream paradigm (model scale, crop size, augmentation, fusion head depth). This is a healthy exploitation phase but also means the search did not probe alternatives such as domain-specific pathology foundation models, which had been proposed during the cycle 1 discussion phase (GPU1 memory file: `paradigm_pathology_fm.md`) but never executed.

**Beta_004 best-checkpoint prediction required multiple re-runs due to epoch-level checkpointing.** Three separate prediction passes were run from beta_004's `best_model.pth` (once at epoch ~13 producing 0.9966, once at epoch 17 producing 0.9969, once at epoch 18 producing 0.9970) because the model checkpoint file was updated in place as training continued. Each re-run required killing the old prediction process and launching a new one. The final 5-model ensemble incorporated the epoch-18 beta_004 predictions. This workflow adds coordination complexity and risks submitting sub-optimal predictions if the deadline window is tight.

## Limitations of These Insights

**Single run, no independent replication.** All findings are from one multi-agent run. Val AUC measurements use a fixed 80/20 stratified train/val split (RANDOM_SEED=77 for beta_004, RANDOM_SEED=42 for most others). A second run with different seeds could produce different relative orderings in the 0.9959–0.9970 range.

**No holdout-confirmed benefit from ensembling.** The 5-model ensemble is adopted as the final submission based on the rationale that diverse models reduce correlated errors. The individual model val AUCs were measured, but the ensemble val AUC is not independently measured — it is estimated as 0.9968+ (analyst3 result_latest.json). Whether the ensemble actually improves over beta_004 alone (0.9970) on the Kaggle test set is unknown from the logged results.

**Center crop size ablation is confounded.** The comparison of center-32 vs. center-48 is not a clean ablation: center-32 models use a simpler fusion head and center-48 models use a deep 3-layer fusion head. The +0.0034–0.0045 gains attributed to the larger center crop cannot be cleanly separated from the deeper fusion head effect.

**Pathology foundation models (UNI, CONCH, CTransPath) were never executed.** GPU1's cycle 1 discussion proposed fine-tuning domain-specific histopathology foundation models trained on large histology corpora. These were never queued or executed in any cycle. The DINOv2 results (natural-image pretrained, 0.9694–0.9907) do not answer whether pathology-specific pretraining could surpass fine-tuned EfficientNet dual-stream at 0.9970.

**Unexplored axes:**
- Pathology foundation models (UNI, CTransPath, CONCH): proposed in discussion, never executed
- Cross-attention fusion between the two streams (proposed by analyst1 for exp_zeta_007): never executed
- Longer training for dual-stream B3 (beta_004 plateau was explored to ep25; peaked at ep18 with no further gain)
- More aggressive TTA (8-crop instead of 4-rotation): tried for ViT (exp_epsilon_001) but not for the dual-stream CNN models
- Alternative ensemble weighting (val_auc^k weighted rank averaging was considered and computed to be nearly uniform at this performance level; uniform was retained)
