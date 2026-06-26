---
task: proteingym-dms-Q8EG35_SHEON_Campbell_2022_indels
run_id: biomlb_pg_q8e
started_at: "2026-04-28T17:03:36Z"
champion_at: "2026-04-28T19:52:32Z"
---

# Research Insights for ProteinGym DMS Q8EG35_SHEON Indel Fitness Prediction

AutoScientists converged on the finding that **insertion position is the dominant fitness signal** for this library, with 90.6% of 331 variants sharing a single octamer (SGRPGSLS) inserted at different positions in the 333-residue MtrA protein from *Shewanella oneidensis*. The headline result is Spearman ρ = 0.8105, achieved by differential evolution (DE) optimization of weighted rank ensembles across 15 diverse model submissions. Insertion-site-focused ESM2 features emerged as the strongest single predictors (ρ ≈ 0.806–0.809), outperforming global sequence embeddings (ρ ≈ 0.748–0.779) and classical sequence-only features (ρ ≈ 0.31–0.76). The final champion applied DE to all 15 available cycle-1 and cycle-2 submissions simultaneously, achieving a modest additional gain (+0.0005 over the 4-model DE ensemble).

## Findings

**1. Insertion position, not octamer identity, determines fitness.**
All 331 variants are 8-mer insertions into a 333-residue WT; 300 of 331 (90.6%) share the identical octamer SGRPGSLS, 29 (8.8%) are SSGRPGSL, and 2 (0.6%) are LSSGRPGS — circular shifts of the same sequence. Variants are distributed across 302 unique insertion positions spanning positions 2–333. This means the combinatorial problem collapses to: which positions in MtrA can tolerate the insertion of a disordered linker? Models that exploit positional features outperform those relying on sequence composition or whole-sequence embeddings.

**2. ESM2-based insertion-site features drove the cycle-1-to-cycle-2 jump.**
The biggest single gain in the run came from refocusing ESM2 embeddings onto the insertion site rather than mean-pooling across the full sequence. The first cycle established a baseline of ρ = 0.748 (ESM2-650M last-layer mean-pool + Ridge). Once agents extracted residue-level embeddings at the insertion site and its ±16 flanking residues (exp_alpha_004_esm2_insertion_site_focus, gpu1), the score rose to ρ = 0.806, a +0.058 improvement over the baseline.

**3. Tranception/ESM2-PLL zero-shot scored ρ = 0.796 without supervision.**
GPU5's zero-shot ESM2 pseudo-log-likelihood scorer (labeled "Tranception" in scripts; executed as ESM2-650M masked-LM PLL) achieved ρ = 0.7959 using no training labels. This competed closely with supervised methods in cycle 1 and provided a complementary signal for ensembling (it was consistently included in high-performing ensembles with weight ~1.55 relative to the insertion-site model).

**4. WT per-position PLL (evolutionary conservation) provides orthogonal signal.**
GPU4's exp_beta_010 computed the ESM2 masked-marginal log-likelihood of each position in the WT sequence, then used the PLL at the insertion site and a ±10 window as features for predicting insertion tolerance. This model reached ρ = 0.8093, nearly matching the best ensemble at that point, and provided a signal distinct from supervised ESM2 embeddings — when combined with the rank ensemble, it pushed to ρ = 0.8095 (exp_alpha_005_wt_pos_ensemble).

**5. Weighted rank ensemble with DE-optimized weights outperforms uniform or grid-search weights.**
The progression across ensembling approaches was: unweighted rank average (ρ = 0.8057) → grid-search weighted rank (ρ = 0.8077–0.8092) → DE-optimized weights on 4 models (ρ = 0.8100) → DE-optimized weights on all 15 models (ρ = 0.8105). DE consistently found better weight combinations than exhaustive grid search, and running with 5–7 seeds and taking the best ensured robustness.

### Insights

**G1 — For protein indel fitness, position-specific features dominate sequence composition features when the inserted peptide is nearly invariant.**
*Claim:* When a DMS library holds the inserted sequence nearly constant and varies only insertion position, models that capture positional context (local ESM2 embeddings, positional PLL, context one-hot) will substantially outperform models based on global composition (k-mer TF-IDF, amino acid counts, mean-pooled embeddings).
*Disconfirming evidence:* A DMS library where insertion position is fixed and peptide identity varies — in that case, composition and sequence-based models would be expected to dominate.
*Observed:* The gap between global mean-pool ESM2 (ρ = 0.748) and insertion-site ESM2 (ρ = 0.806) is +0.058; classical k-mer/composition models scored 0.31–0.65 and were excluded from all competitive ensembles.

**G2 — ESM2 per-position PLL of the wild-type sequence is a transferable proxy for site-level insertion tolerance.**
*Claim:* Positions with high ESM2 PLL (well-predicted from context — evolutionarily constrained or structurally critical) are less tolerant of insertions. This signal is computable from the WT sequence alone, without variant fitness data.
*Disconfirming evidence:* A case where WT PLL does not correlate with insertion tolerance, e.g., due to protein disorder or compensatory mutations.
*Observed:* exp_beta_010 reached ρ = 0.8093 using only WT-PLL features at the insertion site plus ensemble weighting; it scored competitively with the supervised champion (0.8092) from the same cycle.

**G3 — DE-optimized rank ensemble weights improve over grid search particularly as the model pool grows.**
*Claim:* As the number of candidate models grows beyond 4–5, the combinatorial space of weights is too large for grid search; DE finds better solutions more efficiently by treating weight optimization as a continuous global optimization problem.
*Disconfirming evidence:* A case where DE finds the same or worse weights than grid search due to overfitting the training signal.
*Observed:* 4-model DE (ρ = 0.8100) exceeded 3-model grid-search ensemble (ρ = 0.8092); 15-model DE (ρ = 0.8105) further improved, though marginally. Running DE with multiple seeds (5–7) and selecting the best guards against local minima.

**G4 — Early analyst finding that simple voting degrades when base models are highly correlated applies here.**
*Claim:* When candidate models use similar feature families (e.g., multiple ESM2 embedding variants), naive averaging can hurt performance. Careful weight optimization or diversity-aware selection is needed.
*Disconfirming evidence:* Cases where uniform averaging outperforms weighted methods due to having truly diverse, equally strong base models.
*Observed:* Analyst-1 found that naive voting of Tranception (ρ = 0.796) and ESM2-layers (ρ = 0.777) degraded to ρ = 0.788 (r = 0.961 correlation between the two), which correctly directed agents toward explicit weight optimization in cycle 2.

### Task-Specific Findings

**T1 — The Q8EG35_SHEON library is dominated by positional variation over sequence variation (302 positions, 3 octamers).**
Out of 331 variants, 300 (90.6%) share octamer SGRPGSLS inserted at one of 302 unique positions. The task is therefore effectively a "position tolerance scan" of a disordered linker along the MtrA protein. Models that encode insertion position explicitly (relative position, one-hot context window, ESM2 attention at position) are the strongest predictors.

**T2 — Flanking-region ESM2 features outperform global mean-pool by +0.058 Spearman.**
exp_alpha_001 (global mean-pool + Ridge) scored ρ = 0.748. exp_alpha_004 (insertion-site + flanking ±16 residues + PLL of inserted residues + SVR/Ridge) scored ρ = 0.806 on the same ESM2-650M model. The +0.058 gain reflects that the relevant structural context is local to the insertion site, and diluting it across 341 positions degrades the signal.

**T3 — Classical k-mer and composition features fail on this task (ρ = 0.31–0.65).**
exp_beta_001 (amino acid composition + Atchley factors + XGBoost) scored ρ = 0.308. exp_beta_002 (k-mer TF-IDF 2–4 + LightGBM) scored ρ = 0.648. exp_beta_003 (k-mer TF-IDF + Ridge tuned) scored ρ = 0.383. These models fail because the sequences are nearly identical (differing only at 8 positions) and global k-mer statistics cannot resolve the positional information.

**T4 — Character-level n-gram TF-IDF (2–6 gram) reaches ρ = 0.760, providing useful diversity in ensembles.**
exp_beta_007_chargram captured sequence context via 2-to-6 character n-grams. While individually weak relative to ESM2-based methods, it contributed positively to the cycle-1 rank ensemble (gpu1 delta_001), providing a lightweight signal that improved the ensemble score from 0.796 (Tranception alone) to 0.806.

**T5 — The ESM2 attention-entropy approach (exp_zeta_001) did not improve over the simpler positional Ridge context method.**
GPU6's exp_zeta_001 computed ESM2 attention-entropy and attention-weight features at each insertion site, hypothesizing that low-entropy (focused) attention regions correspond to structurally critical positions. The resulting ensemble scored ρ = 0.8077, the same as the simpler grid-search weighted ensemble (exp_beta_008), suggesting that the attention-entropy features did not add signal beyond what insertion position and context already captured.

## Dead Ends and Negative Results

**Classical k-mer and composition models (ρ = 0.31–0.65):** exp_beta_001 (XGBoost + Atchley), exp_beta_002 (k-mer LightGBM), exp_beta_003 (k-mer Ridge) all scored below 0.65. The root cause is that k-mer features treat the 8-mer insertion as a global sequence signal, which is nearly identical across variants. These models carry no positional information. Retired after cycle 1; none were included in competitive ensembles.

**Multi-layer ESM2 PCA + Ridge without insertion-site focus (ρ = 0.759):** exp_alpha_003 (gpu2) extracted layers 15/25/30/33, mean-pooled, applied PCA-256, and fit RidgeCV. Score = 0.759. This is better than the single-layer baseline (0.748) but still far below insertion-site focus. Multi-layer features from global mean-pooling add marginal information compared to the simple last-layer baseline.

**Naive ensemble of high-correlated models degrades (ρ = 0.788):** Analyst-1 confirmed that voting between Tranception (0.796) and ESM2-layers (0.777) gave 0.788 — worse than either model alone in the best case. Pair correlation was r = 0.961. Abandoned immediately; informed the shift to weighted ensembling in cycle 2.

**ESM2 attention-entropy features (exp_zeta_001, ρ = 0.8077):** No improvement over simpler weighted ensemble using the same base models without attention features. The attention-entropy signal did not add beyond insertion position + context one-hot.

**15-model DE ensemble gives marginal return (ρ = 0.8105 vs. 4-model ρ = 0.8100):** Extending DE from 4 to 15 models yielded only +0.0005 Spearman. The additional models (older ESM2 mean-pool variants, k-mer models, lower-performing base models) did not provide enough novel signal to improve substantially. The DE correctly down-weighted low-quality models toward zero, effectively recovering the 4-model ensemble.

**exp_alpha_004 (gpu2) scored lower (0.789) than the same experiment on gpu1 (0.806):** Two agents independently ran insertion-site-focused ESM2 experiments. The gpu2 run scored 0.789 vs. gpu1's 0.806, suggesting implementation differences in how the flanking context was encoded or the SVR/Ridge balance was handled. This gap was not diagnosed in detail; the gpu1 version was selected for downstream ensembling.

## Coordination and Team Dynamics

**Discovery of the octamer-invariance finding was rapid and shared across agents.** By cycle 2, multiple agents (gpu4, gpu6, gpu3) independently noted in their experiment docstrings that 90.6% of variants share SGRPGSLS, and that insertion position — not peptide identity — is the operative variable. This finding was embedded in experiment descriptions rather than a shared analysis thread, but it drove convergent design choices (positional features, WT-PLL at insertion site, context one-hot).

**Analyst-1 performed explicit cycle-1 model correlation analysis.** The analyst found that Tranception and ESM2-layers embeddings were correlated at r = 0.961, explaining why simple voting degraded performance. This finding was recorded in the analyst's result_latest.json notes and redirected agents toward diversity-aware ensembling with explicit weight optimization.

**The approach registry was largely unused; coordination happened through shared submission files.** The registry contained only one entry at cycle 3 ("DE-global-ensemble-all-models-cycle2"). Actual experiment tracking happened through agents reading each other's submission CSV files directly from shared workspace paths, and through experiment docstrings that cited prior scores. This decentralized coordination was effective — all cycle-2 and cycle-3 agents had access to all prior predictions.

**Analysts 2 and 3 worked independently on classical features without impact on final champion.** Analyst-2's workspace contains context-feature experiments (Ridge/SVR on positional + physicochemical features) with multiple submission files. These did not appear in the final ensemble. Analyst-3's workspace has no experiments. The analyst pool did not contribute to the champion pipeline in this run.

**Diminishing returns after cycle 2.** The jump from cycle 1 to cycle 2 was +0.028 Spearman (from 0.776 to 0.809). The jump from cycle 2 to the final champion was +0.001 (0.8092 to 0.8105). The ensemble frontier plateaued early, and later cycle-3 experiments were incremental refinements of weighting strategies rather than new modeling approaches.

## Limitations of These Insights

**Single run, no replication.** All results are from a single AutoScientists run. The Spearman scores computed on the training fold labels may be optimistic due to weight overfitting in the DE ensemble (weights were optimized on the full labeled set, not held-out data). No independent test set evaluation is reported.

**Scope is specific to position-scan indel DMS.** The findings about insertion-site ESM2 features and WT-PLL being dominant apply specifically to libraries where insertion position varies and the inserted sequence is nearly invariant. Libraries with diverse insertion sequences or substitution DMS would require different feature engineering.

**Unexplored approaches:**
- True Tranception model (the autoregressive indel LM from Notin et al.) was named in experiment scripts but not actually executed; agents fell back to ESM2-PLL as a substitute.
- ESM-IF1 or ProteinMPNN inverse folding scores, which may capture positional tolerance via structural context rather than sequence context.
- Structural features from AlphaFold2 predictions (e.g., per-residue pLDDT, solvent accessibility at insertion site), which could directly encode rigidity/flexibility and thus insertion tolerance.
- Larger ESM2 variants (3B, 15B parameters) were not tested.

**Weight overfitting risk in DE ensemble.** The final champion (exp_gamma_004_de_ensemble_all) optimized weights for 15 models on the same labeled data used for evaluation. With 331 data points and 15 weight parameters, there is non-trivial risk that the +0.0005 gain over the 4-model DE ensemble reflects overfitting to the training label set rather than a genuine generalization improvement.
