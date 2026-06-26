---
task: kaggle-uw-madison-gi-tract-image-segmentation
run_id: biomlb_uw_2
started_at: "2026-04-28T16:13:22Z"
champion_at: "2026-04-29T02:16:42Z"
---

# Research Insights for Kaggle UW-Madison GI Tract Image Segmentation

AutoScientists discovered that **expanding the adjacent-slice context window from k=5 to k=7 is the single highest-value axis in 2.5D MRI segmentation**, delivering a 32% reduction in normalized Hausdorff distance (0.1382 → 0.1071) and lifting the combined score from 0.5495 to 0.5690. The headline finding is that **Hausdorff distance dominates this metric** (60% weight), making boundary-spatial consistency — controlled primarily by how many adjacent slices are stacked — the primary lever for improvement, ahead of backbone capacity, training duration, or alternative architectures.

## Findings

**1. Increasing k from 5 to 7 adjacent slices cuts Hausdorff distance by 32% on the same backbone.**
With EfficientNet-B4 encoder and identical hyperparameters, raising the slice-stack from k=5 to k=7 dropped normalized Hausdorff from 0.1382 to 0.1071 (−0.031) while Dice moved modestly from 0.0810 to 0.0832 (+0.002). The combined metric jumped from 0.5495 to 0.5690 (+0.0195). Because the Hausdorff component carries 60% weight, this single structural change dominates all other interventions explored.

**2. Extending training from 40 to 60 epochs on the same k=5 architecture does not help.**
EfficientNet-B4 at k=5 for 60 epochs scored 0.5439, below the 40-epoch k=5 result (0.5495). Hausdorff rose from 0.1382 to 0.1474 despite lower training loss (0.5014 → 0.4949), suggesting that longer training on the k=5 architecture sharpens per-slice Dice-aligned predictions at the cost of 3D spatial consistency. The k=5 model's best val-Dice epoch was epoch 23 out of 40; adding 20 more epochs did not improve the full metric.

**3. 2.5D U-Net strongly outperforms SwinUNETR 3D on this small-n dataset (50 patients).**
SwinUNETR 3D with Dice + weighted-BCE loss scored 0.4789 (Dice=0.0188, Hausdorff=0.2143), −0.071 below the cycle-1 champion. The 3D model produced 15,696 missing prediction rows (empty masks), filling 77% of test predictions with blanks. This failure indicates that native 3D volumetric training is severely data-starved at ~50 cases; the model could not learn reliable organ representations without substantially more 3D training data.

**4. EfficientNet-B4 outperforms ResNet34 at matched context widths, with most of the gain coming from Hausdorff.**
ResNet34 at k=3 (20 epochs) scored 0.5395 (Dice=0.0859, Hausdorff=0.1581). EfficientNet-B4 at k=5 (40 epochs) scored 0.5495 (Dice=0.0810, Hausdorff=0.1382). The Dice actually decreased with EfficientNet-B4, but Hausdorff dropped substantially (−0.020), driving the overall improvement. The ResNet34 run was also undertrained (best val-Dice at epoch 19 of 20, still improving at end), so the encoder comparison is confounded by epochs.

**5. Increasing k beyond 7 to k=9 does not improve the combined metric.**
EfficientNet-B4 at k=9, 40 epochs scored 0.5502 (Dice=0.0806, Hausdorff=0.1367), a marginal −0.0188 below the k=7 champion (0.5690). Despite a slightly lower Hausdorff (0.1367 vs 0.1071 is still worse), the k=9 score underperforms k=7, suggesting that at this image resolution (256×256) and training budget, the benefit of additional context saturates between k=7 and k=9.

### Insights

**G1 — In metrics with heavy Hausdorff weighting, spatial context width (k) dominates backbone choice.**
*Claim:* For 2.5D MRI segmentation tasks where normalized Hausdorff distance accounts for ≥50% of the metric, the number of adjacent slices stacked as channels is a more impactful hyperparameter than encoder architecture or training duration.
*Disconfirming evidence:* A dataset where Hausdorff is insensitive to k (e.g., single-organ tasks with large, convex structures where any reasonable model produces spatially coherent predictions).
*Observed:* Moving k from 3→5→7 with the same encoder yielded combined-score deltas of approximately +0.010 and +0.019 respectively, while backbone changes (ResNet34→EfficientNet-B4) and epoch increases (40→60) produced smaller or negative deltas.

**G2 — 3D volumetric architectures require substantially more than 50 training cases to outperform 2.5D.**
*Claim:* Pure 3D segmentation models (e.g., SwinUNETR) trained on datasets with fewer than ~100 volumetric cases will systematically underperform 2.5D approaches that share ImageNet-pretrained encoder weights, because the 3D model cannot leverage 2D pretraining and has too few 3D examples to learn reliable volumetric representations.
*Disconfirming evidence:* A 3D model that benefits from strong domain-specific 3D pretraining (e.g., Medical Segment Anything) and matches or exceeds 2.5D on small n.
*Observed:* SwinUNETR 3D scored 0.4789 vs. 2.5D EfficientNet-B4 at 0.5495 in cycle 1; the 3D model failed to predict 77% of test rows.

**G3 — The optimal k for 2.5D MRI segmentation appears to be in the range k=5 to k=9, with k=7 performing best in this run.**
*Claim:* There is a unimodal relationship between context-slice count (k) and the combined Dice+Hausdorff metric, with a peak around k=7 at 256×256 resolution with 40 training epochs.
*Disconfirming evidence:* A run where k=9 or k=11 consistently outperforms k=7 under matched conditions.
*Observed:* k=3 scored 0.5395, k=5 scored 0.5495, k=7 scored 0.5690, k=9 scored 0.5502. The peak at k=7 may be resolution- or training-budget-specific.

**G4 — Loss alignment with metric weights accelerates Hausdorff learning.**
*Claim:* Using a training loss with the same coefficient structure as the evaluation metric (0.4×BCE + 0.6×Dice, matching the metric's 0.4×Dice + 0.6×(1−Hausdorff) spirit) is a practical heuristic that helps the model prioritize boundary-consistent predictions.
*Disconfirming evidence:* A task where matching loss weights to metric weights does not improve the Hausdorff component relative to Dice-only loss.
*Observed:* The champion and all successful experiments used 0.4×BCE + 0.6×Dice loss. The SwinUNETR experiment (Dice + weighted-BCE, pos_weight=10) produced the worst Hausdorff of all completed experiments.

### Task-Specific Findings

**T1 — Hausdorff contribution dominates the combined metric in all completed experiments.**
The metric formula is 0.4×Dice + 0.6×(1−normalized Hausdorff). In the ResNet34 baseline: Dice contributes 0.034 and Hausdorff contributes 0.505 (93.6% of total 0.5395). In the champion: Dice contributes 0.033 and Hausdorff contributes 0.536 (94.4% of 0.5690). Optimizing Dice in isolation is therefore a poor proxy for the actual metric.

**T2 — Per-epoch val-Dice is a misleading training signal; full Hausdorff evaluation at the end of training is necessary.**
For the k=7 champion, per-epoch val-Dice peaked at epoch 5 (0.0842) but the final full-metric evaluation using best weights (epoch 5 model) produced a combined score of 0.5690 with Hausdorff=0.1071. The per-epoch Dice curves showed noisy, plateau-like behavior from epoch 10 onward (range 0.078–0.083), giving no clear signal of which checkpoint was best for the combined metric. The best-epoch checkpoint selection was based on per-epoch val-Dice, which peaked early; the checkpoint may not be the best for the combined metric.

**T3 — Training loss plateaus before 40 epochs but the Hausdorff component continues improving.**
For both k=5 and k=7 experiments, training loss showed a steady but decelerating decrease: from ~0.65 (epoch 1) to ~0.50 (epoch 40). The "still improving" dynamics signal from the training script was triggered in all 2.5D experiments, but extending to 60 epochs (k=5) did not improve the combined metric. This suggests the training loss and the Hausdorff component of the metric decouple beyond epoch 23–40: lower loss does not guarantee better Hausdorff.

**T4 — SAM ViT-B with LoRA adapters showed slow learning trajectory and did not complete within the run budget.**
The SAM adapter experiment (k=3, 512×512, 516K trainable params via LoRA rank=4, 20 epochs) was launched on gpu5 in cycle 1. The experiment produced val-Dice=0.055 at epoch 6 at a rate of ~224 seconds per epoch, implying ~75 minutes for 20 epochs. However, no final score appears in the session log or result file; the experiment was apparently preempted or the agent moved on to the 60-epoch EfficientNet-B4 run. With only 516K trainable parameters and a 3-class output head adapting a vision foundation model, early training performance lagged behind the 2.5D CNN baselines.

**T5 — 16-bit PNG MRI images require per-image min-max normalization before ImageNet-style standardization.**
Raw pixel values in the training images are uint16 with typical maxima around 125 (far below the 65535 uint16 ceiling), due to narrow MRI window settings. The successful approach normalizes each image to [0, 255] via per-image min-max scaling before applying ImageNet channel statistics (mean/std cycled across k channels). This normalization is needed because pretrained ImageNet weights expect natural-image statistics; without it, the encoder's learned features are misaligned with the input range.

## Dead Ends and Negative Results

**SwinUNETR 3D with Dice + weighted-BCE loss (exp_beta_001, gpu3):** Scored 0.4789, −0.071 below the cycle-1 champion. Generated 15,696 empty predictions (77% of test rows). The model trained for 20 epochs with val-Dice peaking at 0.0976 at epoch 18, but the full-metric evaluation revealed essentially no Dice signal (0.0188) and poor Hausdorff (0.2143). Root cause: insufficient 3D training cases (~50 patients) for a ViT-based 3D model with no 3D pretraining. Retired: 3D native architectures without 3D domain pretraining are not viable at this dataset scale.

**Extended training to 60 epochs at k=5 (exp_gamma_002, gpu5):** Scored 0.5439, −0.006 below the 40-epoch k=5 result (0.5495). Training loss continued declining to 0.4949 (vs. 0.5014 at 40 epochs) but Hausdorff worsened (0.1474 vs. 0.1382). The model's best val-Dice was epoch 23 for both the 40-epoch and 60-epoch runs, indicating that extending CosineAnnealingLR with T_max=40 into the 41–60 epoch range drives the scheduler into the very-low-LR regime too early, potentially overfitting the Dice component at the expense of 3D consistency. Retired for k=5: more epochs do not help without increasing T_max and k simultaneously.

**SAM ViT-B adapter (exp_gamma_001, gpu5):** The SAM adapter experiment with LoRA rank=4 was launched but did not complete or post a result. Six epochs of training were logged before the agent pivoted to the 60-epoch EfficientNet-B4 experiment. With val-Dice=0.055 at epoch 6 (below the 2.5D baseline of 0.088 at epoch 6), and ~224s per epoch, the approach was slower and underperforming. No final score is available to compare against; the approach is a partial non-result.

**k=9 adjacent slices (exp_alpha_003, gpu2):** Scored 0.5502, below k=7 (0.5690). Hausdorff was 0.1367, only marginally better than k=5 (0.1382) and substantially worse than k=7 (0.1071). This suggests the relationship between k and Hausdorff is non-monotonic and that k=7 is near an optimum for this image size and training budget. Note: the gpu2 result_latest.json shows status "running" (the champion k=7 was already posted before this result was filed), so the k=9 result was not formally logged as KEEP or DISCARD in the session log; the 0.5502 score comes from the stdout file.

**Post-processing size filter (PostProcessing_SizeFilter_CPU, gpu6):** The gpu6 agent attempted CPU-side post-processing with connected-component size filtering on the k=5 champion submission. The result file records val_score=0.5495 — identical to the input champion score — and was not posted to the workshop (posted_to_workshop: false). No improvement was achieved. The post-processing code path was registered in the approach registry as "PostProcessing_SizeFilter_CPU" but produced no measurable gain.

## Coordination and Team Dynamics

**Team structure:** The run used 10 agents — 1 admin, 1 analyst (analyst1 was active; analyst2 and analyst3 showed no memory activity), and 6 GPU experiment agents. Only 3 agents (gpu1, gpu2, gpu5) had gpu_claim files by run end, suggesting that gpu3, gpu4, and gpu6 completed cycle-1 experiments before claims expired or were not reclaimed. The effective experiment throughput was lower than a full 6-GPU run.

**Analyst-1 drove all strategic planning.** Analyst-1 wrote the cycle-2 plan and queue, synthesizing cycle-1 results into a coherent experimental agenda (stronger encoders, loss functions, MRI augmentation). The plan correctly diagnosed the 2.5D superiority and the Hausdorff-focus imperative, but none of the 6 planned cycle-2 experiments were executed — the champion emerged from cycle-2 experiments proposed outside the analyst-1 queue (k=7 and k=9 slice-count variants run by gpu1 and gpu2). The cycle-2 queue experiments (EfficientNet-B5, DenseNet121, Lovász loss, etc.) remained unexecuted at run end.

**The k-slice axis was discovered and exploited by gpu1 without analyst direction.** GPU1 generated its own memory note (`finding_k_slices_hausdorff.md`) documenting the k→Hausdorff relationship after its baseline experiment and then directly implemented and ran the k=7 champion, posting the +0.0195 improvement. GPU2 followed with k=9. This self-directed axis exploration, not the analyst-1 proposal queue, produced the champion.

**Cross-agent knowledge flow was limited to the workshop posts.** There is no evidence in memory files that analyst-2, analyst-3, or GPU agents other than gpu1 read or acted on gpu1's memory finding about k-slices. The session log shows only 6 completed experiment events; agents did not explicitly reference each other's findings in the log beyond the workshop channel.

**Two effective teams operated in practice.** Cycle 1 had a "team_beta" running EfficientNet-B4 (gpu4) and SwinUNETR (gpu3), and a "team_alpha" running ResNet34 baseline (gpu1). Cycle 2 saw team_alpha's gpu1 run k=7, team_alpha's gpu2 run k=9, and team_gamma's gpu5 run SAM adapter and 60-epoch variants. The analyst-1 "team_alpha cycle-2" queue was never claimed.

## Limitations of These Insights

**Statistical support:** Single run, no independent replication. The k=7 finding is based on one experiment per k value; the Hausdorff advantage at k=7 over k=9 (0.1071 vs 0.1367) may not replicate across different random seeds, data splits, or training durations.

**Metric decomposition caution:** The per-epoch val-Dice used for checkpoint selection does not match the full metric (which requires running full inference and computing 3D Hausdorff). The champion model's "best epoch" was chosen by val-Dice (epoch 5), not by the combined metric. A different checkpoint might score higher on the combined metric.

**Incomplete experiments:** The SAM adapter experiment (exp_gamma_001) did not complete and the k=9 experiment (exp_alpha_003) result was not formally posted. Insights about these approaches are based on partial stdout logs only.

**Unexplored axes (identified by analyst-1 but unexecuted):**
- Stronger encoders: EfficientNet-B5, DenseNet121 (expected +2–4% by analyst-1)
- Boundary-aware loss: Lovász, surface loss (expected Hausdorff improvement)
- MRI-specific augmentation: ElasticTransform, CLAHE, GaussianBlur
- Ensemble of k=5, k=7, k=9 models (not attempted)
- TTA (test-time augmentation) on final predictions
- Higher image resolution (512×512 used only in the incomplete SAM experiment)
- More training epochs with properly adjusted scheduler T_max

**Scope:** The k-slice finding is specific to 2.5D CNN architectures with ImageNet-pretrained encoders. It may not transfer to fully 3D architectures with 3D pretraining, or to tasks where the Hausdorff metric is not Hausdorff-dominant (i.e., tasks where Dice weight exceeds 50%).
