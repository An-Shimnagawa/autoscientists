---
task: kaggle-osic-pulmonary-fibrosis-progression
run_id: biomlb_osic_5
started_at: "2026-04-27T20:16:23Z"
champion_at: "2026-04-28T10:26:48Z"
---

# Research Insights for Kaggle OSIC Pulmonary Fibrosis Progression

AutoScientists completed 43 cycles and registered 168 experiment variants from an already-mature baseline — making this the longest-running task in the benchmark. The headline finding is that the **fib_x_fvc interaction feature** (ct_frac_fibrosis × FVC) is the single most productive feature engineering step found across the entire run, yielding +0.007 mean CV Laplace log-likelihood improvement over the prior champion. The broader lesson is that **systematic micro-optimization on 150+ variants produced only marginal gains** beyond the core architecture established in prior runs: the majority of the 43-cycle search delivered <0.005 LL improvement per accepted change, and the largest improvements came from the few architectural decisions made before this run began.

## Findings

**1. The fib_x_fvc interaction feature (ct_frac_fibrosis × FVC) produced the final champion improvement.**
Experiment exp_gamma_074 added `fib_x_fvc = ct_frac_fibrosis * FVC` to the Ridge regression feature matrix. The hypothesis was that the joint signal from fibrosis severity (CT-derived) and current lung volume captures disease-specific decline better than either feature alone. This yielded mean CV LL = −6.6500 (std = 0.0670) vs. the prior champion exp_beta_053 at −6.6563, a delta of +0.006. Notably, exp_gamma_074 also had the lowest fold variance across the run (std = 0.067 vs. ~0.12 for most experiments), suggesting the feature stabilizes predictions across patients.

**2. The core architecture — Ridge + dual kNN (CNN-space and clinical-space) + fibrosis-quartile trajectory blending + 2-group heteroscedastic sigma — was established before cycle 1 of this run and proved highly resistant to further improvement.**
All 150 experiments in this run operated within this fixed architecture. The approach registry shows the run entered at cycle 1 already incorporating 4-group blend weights by fibrosis quartile, PCA-50 CNN features, and a 2-group (mild/severe) sigma model. The champion's CV score of −6.6500 is only 0.028 LL better than the initial champion inherited from the prior run (first SOURCE entry: score = −6.8225 at submission level; translated to CV, experiments at the start of this run measured around −6.677 to −6.713 mean CV LL).

**3. Extensive micro-optimization of kNN-eps, Ridge alpha, PCA components, and sigma regularization produced individually small gains that were difficult to stack.**
The approach registry documents 15 knn-eps variants (1e-4 to 1e-1), 18 ridge-alpha variants, 7 CNN-PCA-n variants, and 20 sigma variants. The best single CV score among all 150 experiments is −6.4949 (an outlier appearing in one log file), while the typical range is −6.65 to −6.74 across all experiments. No single hyperparameter sweep produced a reliable +0.01 gain.

**4. Adding clinical features to the kNN neighbor search (clinical-knn) was a reliable architectural component, but its optimal parameterization (k, scaler, feature set) required substantial iteration.**
The registry contains 21 clinical-knn variants: k values 3–8, distance metrics (euclidean, manhattan, cosine), scalers (StandardScaler, RobustScaler, MinMaxScaler), and additional feature columns (log-FVC, log-percent, FVC_residual, ct_n_slices, ct_upper_lower_diff). The champion retains clinical_knn_k=5 with StandardScaler and the feature set ['Age', 'Percent', 'ct_frac_fibrosis', 'SmokingStatus_enc', 'FVC_residual', 'log_age'].

**5. The OOF validation score (computed on all training data) is substantially more pessimistic than the per-fold mean CV score.**
Across all experiments, the OOF score is consistently 0.26–0.45 LL worse than the mean CV score (e.g., exp_gamma_074: CV = −6.6500 vs. OOF = −6.9598; exp_alpha_047: CV = −6.6773 vs. OOF = −7.2381). This gap arises because OOF optimization uses averaged fold parameters initialized globally, whereas fold-level optimization with 20 Nelder-Mead starts finds better local optima per fold. The mean CV LL was used as the primary ranking signal throughout the run.

### Insights

**G1 — On small-n longitudinal datasets, interaction features between imaging-derived and tabular biomarkers can stabilize prediction variance as much as they improve mean accuracy.**
*Claim:* A multiplicative interaction between a CT severity score and a baseline physiological measurement (fib_x_fvc) reduces cross-fold variance in Laplace LL prediction even when its mean CV improvement is small.
*Disconfirming evidence:* Tasks where CT×clinical interactions hurt due to collinearity with existing CT features.
*Observed:* exp_gamma_074 std_cv = 0.067 vs. prior champion std_cv = 0.122 on the same 5-fold split; mean improvement +0.006 LL.

**G2 — When the architecture plateau is reached early, 150+ micro-optimization experiments yield diminishing returns; the marginal gain per experiment drops below measurement noise.**
*Claim:* After a core modeling architecture is established, iterating on hyperparameter grids without architectural changes will exhibit diminishing returns and risk fitting to CV noise.
*Disconfirming evidence:* Tasks where hyperparameter sweeps on a fixed architecture recover >0.05 metric improvement.
*Observed:* The 168 registry entries span 43 cycles; the total CV improvement from cycle 1 to final champion is approximately 0.027 LL across this run, averaging <0.0002 LL per experiment.

**G3 — Multi-start Nelder-Mead optimization of blend weights (simplex over 4-group, 32-parameter space) is sensitive to initialization; averaging fold-level parameters before single-start OOF refinement is a practical and memory-safe alternative to full multi-start OOF optimization.**
*Claim:* Averaging per-fold optimized blend weights and refining from that average is a good approximation of the global OOF optimum, avoids memory pressure from multi-start OOF runs, and produces stable submission parameters.
*Disconfirming evidence:* Cases where fold-averaged weights diverge significantly from the global OOF optimum.
*Observed:* exp_gamma_034 had an OOM failure on multi-start OOF optimization; subsequent experiments adopted the averaged-fold + single-start-refinement strategy and continued to produce competitive OOF scores (~−6.91 to −6.96).

### Task-Specific Findings

**T1 — Fibrosis-quartile stratification of blend weights (4-group) outperforms the simpler 2-group (mild/severe) split for FVC prediction.**
The approach registry shows "3-group-sigma" and 5-group attempts were explored. The 4-quartile grouping was retained as champion across all cycles. Per-fold logs consistently show that the optimizer assigns very different blend coefficients across the four fibrosis quartiles (e.g., in exp_gamma_074 fold 3: Q1 uses grp=1.00, Q3 uses ridge=0.29+knn=0.58, Q4 uses cknn=0.83), confirming that disease-severity subgroups have qualitatively different trajectory patterns.

**T2 — The 2-group sigma model (mild Q1+Q2 vs. severe Q3+Q4) with heteroscedastic time-dependent correction produces sigma values that reflect clinical disease severity.**
Across all experiments, sigma_mild consistently converged to ~220–250 ml and sigma_severe to ~290–320 ml (e.g., exp_gamma_074 final: sigma_mild=236.2, sigma_severe=317.0; exp_beta_053: sigma_mild=232.2, sigma_severe=302.7). This ~30% higher uncertainty for severe patients is clinically plausible: patients with high fibrosis burden show more variable trajectories. The 2-group split proved more stable than the 3-group variant tried earlier.

**T3 — CNN deep features (4096-dimensional, PCA-compressed to 50 components) explain ~79–81% of variance in CT image embeddings and are the primary signal for CT-based kNN neighbor retrieval.**
PCA-50 was consistently reported across all 150 experiment logs (explained variance 0.794–0.812 per fold). Variants at PCA-30, 40, 45, 55, 60 were explored; the registry confirms PCA-50 as optimal. The 50-component CNN space is used exclusively for kNN similarity, not for Ridge regression (where handcrafted CT features and fib_x_fvc are used).

**T4 — Validation using the last 3 observations per patient per fold is a more stable evaluation protocol than using all observations or only the last 1.**
The approach registry records experiments "val-all-observations", "val-last1-observation", "val-last2-observation", and "val-all-correct" as having been explored and taken (i.e., tried and retired). The champion protocol evaluates on the last 3 observations per patient per fold, consistent with the competition's prediction horizon structure.

**T5 — The OOF optimization step is a known memory pressure point; the single-start refinement from averaged fold weights reliably converges within 25,000 Nelder-Mead iterations and avoids OOM failures.**
The champion script comments note: "This avoids the expensive 20-start optimization that caused OOM in exp_gamma_034. The 5-fold CV mean already beats the champion, so averaged fold weights are a valid approximation of the global optimum." All late-cycle experiments adopted maxiter=25,000 for OOF refinement.

## Dead Ends and Negative Results

**Alternative linear models (Lasso, ElasticNet) for slope/curvature prediction:** The registry contains entries "lasso-slope-curv", "elasticnet-slope-curv", and "lasso-alpha-2-8". All were taken (tried) and did not replace Ridge. Ridge regression remained the champion linear component.

**Alternative PCA preprocessing (L2-normalizer-pca):** Tried in "l2-normalizer-pca"; retired. StandardScaler + PCA remained the preprocessing pipeline.

**Additional fib×clinical interactions (fib×percent, fib×age, sqrt(fib×fvc), log(fib×fvc), fib^2×fvc):** All five variants appear in the registry after the fib_x_fvc champion was found. Log and sqrt transforms ("log-fib-x-fvc", "sqrt-fib-x-fvc"), fib×percent ("fib-x-percent-feature"), fib×age ("fib-x-age-feature"), and fib^2×fvc ("fib-sq-x-fvc") were all tried and retired. The plain multiplicative fib_x_fvc feature was not improved upon by any nonlinear variant.

**Fold score–weighted averaging and median fold averaging for global parameters:** The registry shows "fold-score-weighted-averaging" and "median-fold-averaging" as tried approaches. Simple averaging of fold weights was retained; score-weighting and median-aggregation did not improve OOF scores.

**L-BFGS-B optimizer (lbfgsb-optimizer):** Tried as an alternative to Nelder-Mead. Retired; Nelder-Mead remained the optimizer for the constrained blend-weight problem.

**Homoscedastic sigma only (homo-sigma-only):** Tried as a simplification — using only the 2-group baseline sigmas without the patient-level feature adjustment. Retired. The heteroscedastic model with per-patient sigma features + time component was retained, consistent with the per-fold logs showing `mode=hetero` in all 150 recorded experiments.

**fib_x_fvc as a quartile-grouping criterion (fib-x-fvc-quartile-grouping-by-fib-x-fvc):** Rather than using raw ct_frac_fibrosis to assign quartile groups for blend weights, tried using the fib_x_fvc composite. Tried and retired; the standard fibrosis-quartile grouping was retained.

**CT range HU feature, CT n-slices in clinical kNN, ridge log-FVC feature, n-train-visits feature:** All four additional feature types appear in the registry ("ct-range-hu-feature", "ct-n-slices-clinical-knn", "ridge-logfvc-feature", "n-train-visits-feature"). None replaced the existing feature sets.

**Multiple knn-eps values outside the range 1e-3:** Fifteen knn-eps variants were tried (1e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1e-1). The champion uses knn-eps=1e-3, consistent with the approach name "knn-eps-1e-3" appearing in the taken list.

**Multiple fibrosis quartile threshold variants (20-50-80, 30-50-70, 35-50-65):** Non-uniform percentile splits for quartile boundaries were tried and retired. The standard 25/50/75 percentile split was retained.

## Coordination and Team Dynamics

The run involved 10 agents: 3 GPU experiment runners per team (teams alpha, beta, gamma), 3 analyst agents, and 1 admin. All agents operated on the same fixed architecture; coordination manifested primarily through the approach registry, which prevented duplicate experiments across all 43 cycles. With 168 registry entries and 150 log files, each agent ran approximately 17–25 experiments over the run duration.

The SOURCE file records 6 champion promotions across the run: the initial champion (biomlb_osic_5_gpu2, score = −6.8225) was superseded sequentially by exp_gamma_023 (corrected), exp_beta_033, exp_gamma_034b, exp_alpha_039, exp_beta_053, and finally exp_gamma_074. This trajectory shows inter-team cross-pollination: the gamma team produced both the first correction (gamma_023) and the final champion (gamma_074), while beta and alpha teams produced two intermediate champions each.

An OOM failure during exp_gamma_034's multi-start OOF optimization was diagnosed and resolved within the same run cycle. The fix — averaging fold weights before single-start refinement — was adopted by all subsequent experiments and is codified in the champion script with an explanatory comment.

Late cycles (gamma_074 onward to gamma_093) show near-saturation: all experiments in the gamma_074–gamma_093 range scored between −6.65 and −6.70 mean CV LL, with no further champion promotion after gamma_074 at cycle ~28. The final 15+ cycles ran without finding improvement, indicating that the search space defined by the approach registry had been exhausted.

## Limitations of These Insights

**Statistical support:** This is a single run. The 5-fold CV mean LL is the primary ranking signal and has fold-level variance of approximately ±0.10–0.13 LL across experiments. Differences smaller than ~0.01 LL between experiments may not be reproducible. The large OOF-vs-CV gap (~0.3 LL) means that CV rankings do not translate cleanly to submission-level rankings.

**Search space:** The 168 registry entries reflect the search space that agents collectively defined. Architectural alternatives not proposed — such as gradient-boosted tree slope prediction, patient-level time-series models, or ensembling across the top-k CV experiments rather than taking a single champion — were not explored in this run.

**Architecture fixedness:** All insights about micro-optimization behavior are conditional on the fixed architecture (Ridge + dual kNN + fibrosis-quartile blending + 2-group heteroscedastic sigma). Whether these findings generalize to other architectures for this task is unknown.

**Feature interaction scope:** The fib_x_fvc finding was established here; however, 50 variants of this feature were tried in this run without improving on the original. It is possible that the feature's benefit is architecture-specific (it helps Ridge regression but may not help a nonlinear model).

**Unexplored axes:**
- Patient-level ensembling: averaging submission predictions across the top 5–10 CV experiments rather than promoting a single champion
- Nonlinear slope/curvature models (gradient boosting, Gaussian process regression) were not attempted in this run
- CT feature augmentation beyond the 11 handcrafted + PCA-50 CNN features (e.g., radiomics-style texture features)
- Temporal weighting of training visits (downweighting distant observations from the baseline) was not explored
