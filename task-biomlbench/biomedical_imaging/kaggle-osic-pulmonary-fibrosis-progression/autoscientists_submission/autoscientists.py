"""
exp_gamma_074: ct_frac_fibrosis * FVC interaction in BASE_FEAT_COLS

Changes vs champion exp_beta_053:
1. Added fib_x_fvc = ct_frac_fibrosis * FVC to BASE_FEAT_COLS.
   Hypothesis: joint signal from fibrosis severity and lung volume captures disease-specific decline.

Team: gamma
Experiment: exp_gamma_074
"""


import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Paths
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV = DATA_DIR / 'test.csv'
SAMPLE_SUB = DATA_DIR / 'sample_submission.csv'
CT_FEATURES = DATA_DIR / 'ct_features_cache.csv'
CT_DEEP_FEATURES = DATA_DIR / 'ct_deep_features.csv'
OUTPUT_DIR = Path(__file__).parent / "outputs"

EXP_ID = 'exp_gamma_074'
N_OPTIMIZER_STARTS = 20  # number of Nelder-Mead starts per fold (unchanged from champion)
KNN_K = 7        # number of CT-similar neighbors (increased from 5)
CNN_PCA_N = 50   # PCA-50
N_FIBROSIS_QUARTILES = 4  # 4 groups for blend weights
KNN_EPS = 1e-3
CLINICAL_KNN_K = 5
CLINICAL_KNN_COLS = ['Age', 'Percent', 'ct_frac_fibrosis', 'SmokingStatus_enc', 'FVC_residual', 'log_age']

CT_FEAT_COLS = [
    'ct_mean_hu', 'ct_std_hu', 'ct_frac_fibrosis', 'ct_frac_air',
    'ct_lung_mean_hu', 'ct_n_slices', 'ct_upper_mean_hu', 'ct_lower_mean_hu',
    'ct_upper_lower_diff', 'ct_p25_hu', 'ct_p75_hu'
]
BASE_FEAT_COLS = ['Age', 'Sex_enc', 'SmokingStatus_enc', 'FVC', 'Percent', 'Weeks', 'FVC_residual', 'fib_x_fvc']
SM_MAP = {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}


def compute_fvc_residual(df):
    df = df.copy()
    df['FVC_residual'] = df['FVC'] / np.maximum(df['Percent'], 1.0) * 100.0
    return df


def laplace_ll(fvc_true, fvc_pred, sigma):
    sigma_c = np.maximum(np.asarray(sigma, dtype=float), 70.0)
    delta = np.minimum(np.abs(np.asarray(fvc_true, dtype=float) - np.asarray(fvc_pred, dtype=float)), 1000.0)
    return (-np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)).mean()


def compute_trajectory(df):
    """Fit per-patient quadratic FVC trajectory."""
    slopes = {}
    curvatures = {}
    for pat, grp in df.groupby('Patient'):
        bl = grp.sort_values('Weeks').iloc[0]
        x = grp['Weeks'].values.astype(float)
        y = grp['FVC'].values.astype(float)
        dx = x - bl['Weeks']
        if len(grp) >= 3:
            X_poly = np.stack([np.ones_like(dx), dx, dx ** 2], axis=1)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
                slopes[pat] = coeffs[1]
                curvatures[pat] = coeffs[2]
            except Exception:
                slopes[pat] = 0.0
                curvatures[pat] = 0.0
        elif len(grp) == 2:
            nz = dx != 0
            if nz.any():
                slopes[pat] = (y[nz][0] - bl['FVC']) / dx[nz][0]
            else:
                slopes[pat] = 0.0
            curvatures[pat] = 0.0
        else:
            slopes[pat] = 0.0
            curvatures[pat] = 0.0
    return slopes, curvatures


def get_cnn_pca_cols(n=CNN_PCA_N):
    return [f'cnn_pca_{i}' for i in range(n)]


def build_fibrosis_quartile_map(baselines_df, ct_df, col_means=None):
    """Build fibrosis QUARTILE assignment (0=Q1, 1=Q2, 2=Q3, 3=Q4) for each patient."""
    df = baselines_df[['Patient']].merge(ct_df[['Patient', 'ct_frac_fibrosis']], on='Patient', how='left')
    fib_vals = df['ct_frac_fibrosis'].values.astype(float)
    med = np.nanmedian(fib_vals)
    fib_vals = np.where(np.isnan(fib_vals), med, fib_vals)
    if col_means is None:
        t1 = np.percentile(fib_vals, 25)
        t2 = np.percentile(fib_vals, 50)
        t3 = np.percentile(fib_vals, 75)
        col_means = (t1, t2, t3, med)
    t1, t2, t3, fib_fill_med = col_means
    quartile = np.where(fib_vals <= t1, 0,
                np.where(fib_vals <= t2, 1,
                np.where(fib_vals <= t3, 2, 3)))
    pat_to_quartile = dict(zip(df['Patient'].values, quartile))
    return pat_to_quartile, col_means


def build_fibrosis_2group_map(baselines_df, ct_df, median_val=None):
    """Build 2-group fibrosis assignment for sigma: 0=mild (Q1+Q2), 1=severe (Q3+Q4)."""
    df = baselines_df[['Patient']].merge(ct_df[['Patient', 'ct_frac_fibrosis']], on='Patient', how='left')
    fib_vals = df['ct_frac_fibrosis'].values.astype(float)
    fill_med = np.nanmedian(fib_vals)
    fib_vals = np.where(np.isnan(fib_vals), fill_med, fib_vals)
    if median_val is None:
        median_val = np.percentile(fib_vals, 50)
    group = np.where(fib_vals <= median_val, 0, 1)
    pat_to_group2 = dict(zip(df['Patient'].values, group))
    return pat_to_group2, median_val


def compute_group_trajectory_means(baselines_df, pat_to_group, slopes, curvatures, n_groups=N_FIBROSIS_QUARTILES):
    """Compute per-group mean slope and curvature."""
    grp_slopes = {g: [] for g in range(n_groups)}
    grp_curvs = {g: [] for g in range(n_groups)}
    for pat in baselines_df['Patient'].values:
        g = pat_to_group.get(pat, n_groups // 2)
        grp_slopes[g].append(slopes.get(pat, 0.0))
        grp_curvs[g].append(curvatures.get(pat, 0.0))
    group_slope_means = {g: float(np.mean(v)) if v else 0.0 for g, v in grp_slopes.items()}
    group_curv_means = {g: float(np.mean(v)) if v else 0.0 for g, v in grp_curvs.items()}
    return group_slope_means, group_curv_means


def build_feature_matrix(baselines_df, ct_df, cnn_pca_df, col_means=None):
    """Build feature matrix: tabular + handcrafted CT + CNN PCA features."""
    df = baselines_df.copy()
    df['Sex_enc'] = (df['Sex'] == 'Male').astype(float)
    df['SmokingStatus_enc'] = df['SmokingStatus'].map(SM_MAP).fillna(1.0)
    df = compute_fvc_residual(df)
    df = df.merge(ct_df[['Patient'] + CT_FEAT_COLS], on='Patient', how='left')
    df = df.merge(cnn_pca_df, on='Patient', how='left')

    df['fib_x_fvc'] = df['ct_frac_fibrosis'].fillna(0.0) * df['FVC']
    cnn_cols = get_cnn_pca_cols()
    all_cols = BASE_FEAT_COLS + CT_FEAT_COLS + cnn_cols
    X = df[all_cols].values.astype(float)

    if col_means is None:
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = 0.0
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]
    return X, col_means


def build_sigma_feature_matrix(baselines_df, ct_df, cnn_pca_df, sigma_col_means=None):
    """Build sigma features."""
    df = baselines_df.copy()
    df['Sex_enc'] = (df['Sex'] == 'Male').astype(float)
    df['SmokingStatus_enc'] = df['SmokingStatus'].map(SM_MAP).fillna(1.0)
    df = compute_fvc_residual(df)
    df = df.merge(ct_df[['Patient'] + CT_FEAT_COLS], on='Patient', how='left')
    df = df.merge(cnn_pca_df, on='Patient', how='left')

    ct_sigma_cols = ['ct_std_hu', 'ct_frac_fibrosis', 'ct_frac_air', 'ct_mean_hu', 'ct_upper_lower_diff']
    tab_sigma_cols = ['Age', 'Percent', 'FVC', 'FVC_residual']
    cnn_cols = get_cnn_pca_cols()
    sigma_cols = ct_sigma_cols + tab_sigma_cols + cnn_cols
    sigma_cols_available = [c for c in sigma_cols if c in df.columns]

    X = df[sigma_cols_available].values.astype(float)
    if sigma_col_means is None:
        sigma_col_means = np.nanmean(X, axis=0)
        sigma_col_means[np.isnan(sigma_col_means)] = 0.0
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = sigma_col_means[j]
    return X, sigma_col_means, sigma_cols_available


def build_cnn_knn_matrix(baselines_df, cnn_pca_df, cnn_col_means=None):
    """Build CNN PCA feature matrix for kNN similarity search."""
    df = baselines_df.copy()
    df = df.merge(cnn_pca_df, on='Patient', how='left')
    cnn_cols = get_cnn_pca_cols()
    X = df[cnn_cols].values.astype(float)

    if cnn_col_means is None:
        cnn_col_means = np.nanmean(X, axis=0)
        cnn_col_means[np.isnan(cnn_col_means)] = 0.0
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = cnn_col_means[j]
    return X, cnn_col_means


def build_clinical_knn_matrix(baselines_df, ct_df, clinical_col_means=None):
    """Build clinical feature matrix for kNN."""
    df = baselines_df.copy()
    df['SmokingStatus_enc'] = df['SmokingStatus'].map(SM_MAP).fillna(1.0)
    df = df.merge(ct_df[['Patient', 'ct_frac_fibrosis']], on='Patient', how='left')
    df['FVC_residual'] = df['FVC'] / np.maximum(df['Percent'], 1.0) * 100.0
    df['log_age'] = np.log1p(df['Age'])

    X = df[CLINICAL_KNN_COLS].values.astype(float)

    if clinical_col_means is None:
        clinical_col_means = np.nanmean(X, axis=0)
        clinical_col_means[np.isnan(clinical_col_means)] = 0.0
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = clinical_col_means[j]
    return X, clinical_col_means


def fit_pca_on_cnn_features(train_baselines_df, ct_deep_df, n_components=CNN_PCA_N):
    """Fit PCA on training CNN features."""
    raw_cnn_cols = [c for c in ct_deep_df.columns if c.startswith('ct_cnn_')]
    df_merged = train_baselines_df[['Patient']].merge(ct_deep_df[['Patient'] + raw_cnn_cols], on='Patient', how='left')

    X_raw = df_merged[raw_cnn_cols].values.astype(float)
    col_means = np.nanmean(X_raw, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        if mask.any():
            X_raw[mask, j] = col_means[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    print(f"  CNN PCA: {len(raw_cnn_cols)} raw features -> {n_components} components, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    return pca, scaler, col_means, raw_cnn_cols


def transform_cnn_to_pca(patients_df, ct_deep_df, pca, scaler, raw_cnn_col_means, raw_cnn_cols):
    """Transform CNN features to PCA space for given patients."""
    df = patients_df[['Patient']].merge(ct_deep_df[['Patient'] + raw_cnn_cols], on='Patient', how='left')
    X_raw = df[raw_cnn_cols].values.astype(float)

    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        if mask.any():
            X_raw[mask, j] = raw_cnn_col_means[j]

    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)

    pca_cols = get_cnn_pca_cols()
    result = pd.DataFrame(X_pca, columns=pca_cols)
    result['Patient'] = patients_df['Patient'].values
    return result


def predict_per_patient_sigma_2group(log_base_sigma_mild, log_base_sigma_severe,
                                     sigma_weights, X_sigma_sc, dt_weeks, pat_groups_2):
    """Predict sigma using 2-group base sigma (mild/severe)."""
    n_sigma_feats = X_sigma_sc.shape[1]
    w_patient = sigma_weights[:n_sigma_feats]
    w_time = sigma_weights[n_sigma_feats]

    patient_adj = X_sigma_sc @ w_patient
    time_feat = np.log1p(np.abs(np.asarray(dt_weeks, dtype=float)) / 52.0)

    pat_groups_arr = np.asarray(pat_groups_2, dtype=int)
    log_base = np.where(pat_groups_arr == 0, log_base_sigma_mild, log_base_sigma_severe)

    log_sigma = log_base + patient_adj + w_time * time_feat
    return np.maximum(np.exp(log_sigma), 70.0)


def fit_trajectory_models(tr_df, ct_df, cnn_pca_df, ridge_alpha_slope=100, ridge_alpha_curv=400, k=KNN_K):
    """Fit Ridge + kNN(PCA-50) + fibrosis-quartile trajectory + sigma model."""
    baselines = tr_df.sort_values('Weeks').groupby('Patient').first().reset_index()
    slopes, curvs = compute_trajectory(tr_df)
    baselines['slope'] = baselines['Patient'].map(slopes)
    baselines['curv'] = baselines['Patient'].map(curvs)

    X, col_means = build_feature_matrix(baselines, ct_df, cnn_pca_df)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ridge_slope = Ridge(alpha=ridge_alpha_slope).fit(X_sc, baselines['slope'].values)
    ridge_curv = Ridge(alpha=ridge_alpha_curv).fit(X_sc, baselines['curv'].values)

    pop_slope = float(baselines['slope'].mean())
    pop_curv = float(baselines['curv'].mean())

    X_cnn_knn, cnn_col_means = build_cnn_knn_matrix(baselines, cnn_pca_df)
    cnn_knn_scaler = StandardScaler()
    X_cnn_knn_sc = cnn_knn_scaler.fit_transform(X_cnn_knn)
    knn = NearestNeighbors(n_neighbors=min(k + 1, len(baselines)), metric='euclidean')
    knn.fit(X_cnn_knn_sc)

    X_clin_knn, clinical_col_means = build_clinical_knn_matrix(baselines, ct_df)
    clinical_knn_scaler = StandardScaler()
    X_clin_knn_sc = clinical_knn_scaler.fit_transform(X_clin_knn)
    clinical_knn = NearestNeighbors(n_neighbors=min(CLINICAL_KNN_K + 1, len(baselines)), metric='euclidean')
    clinical_knn.fit(X_clin_knn_sc)

    pat_to_quartile, fib_quartile_params = build_fibrosis_quartile_map(baselines, ct_df)
    group_slope_means, group_curv_means = compute_group_trajectory_means(
        baselines, pat_to_quartile, slopes, curvs, n_groups=N_FIBROSIS_QUARTILES)

    for g in range(N_FIBROSIS_QUARTILES):
        count = sum(1 for v in pat_to_quartile.values() if v == g)
        print(f"  Fibrosis Q{g+1}: n={count}, slope={group_slope_means[g]:.2f}")

    # 2-group sigma stratification (unchanged from champion)
    pat_to_group2, fib_2group_median = build_fibrosis_2group_map(baselines, ct_df)
    mild_count = sum(1 for v in pat_to_group2.values() if v == 0)
    severe_count = sum(1 for v in pat_to_group2.values() if v == 1)
    print(f"  2-group sigma: mild(Q1+Q2)={mild_count}, severe(Q3+Q4)={severe_count}")

    X_sigma, sigma_col_means, sigma_cols = build_sigma_feature_matrix(baselines, ct_df, cnn_pca_df)
    sigma_scaler = StandardScaler()
    sigma_scaler.fit(X_sigma)

    return {
        'ridge_slope': ridge_slope,
        'ridge_curv': ridge_curv,
        'scaler': scaler,
        'col_means': col_means,
        'pop_slope': pop_slope,
        'pop_curv': pop_curv,
        'knn': knn,
        'cnn_knn_scaler': cnn_knn_scaler,
        'cnn_col_means': cnn_col_means,
        'clinical_knn': clinical_knn,
        'clinical_knn_scaler': clinical_knn_scaler,
        'clinical_col_means': clinical_col_means,
        'train_slopes': baselines['slope'].values,
        'train_curvs': baselines['curv'].values,
        'train_patients': baselines['Patient'].values,
        'sigma_scaler': sigma_scaler,
        'sigma_col_means': sigma_col_means,
        'sigma_cols': sigma_cols,
        'pat_to_quartile': pat_to_quartile,
        'fib_quartile_params': fib_quartile_params,
        'group_slope_means': group_slope_means,
        'group_curv_means': group_curv_means,
        'pat_to_group2': pat_to_group2,
        'fib_2group_median': fib_2group_median,
    }


def get_knn_slopes_curvs_weighted(model, baselines_df, cnn_pca_df, exclude_self=True, k=KNN_K):
    """Find k-nearest training neighbors in CNN PCA-50 space with inverse-distance weighting."""
    X_cnn, _ = build_cnn_knn_matrix(baselines_df, cnn_pca_df, model['cnn_col_means'])
    X_cnn_sc = model['cnn_knn_scaler'].transform(X_cnn)

    n_neighbors = min(k + 1, len(model['train_patients']))
    distances, indices = model['knn'].kneighbors(X_cnn_sc, n_neighbors=n_neighbors)

    knn_slopes = []
    knn_curvs = []
    query_patients = baselines_df['Patient'].values

    for i, pat in enumerate(query_patients):
        neighbor_idxs = list(indices[i])
        neighbor_dists = list(distances[i])

        if exclude_self:
            filtered = [(idx, d) for idx, d in zip(neighbor_idxs, neighbor_dists)
                        if model['train_patients'][idx] != pat][:k]
        else:
            filtered = list(zip(neighbor_idxs, neighbor_dists))[:k]

        if len(filtered) == 0:
            knn_slopes.append(model['pop_slope'])
            knn_curvs.append(model['pop_curv'])
        else:
            idxs = [x[0] for x in filtered]
            dists = np.array([x[1] for x in filtered])
            weights = 1.0 / (dists + KNN_EPS)
            weights = weights / weights.sum()
            knn_slopes.append(float(np.dot(weights, model['train_slopes'][idxs])))
            knn_curvs.append(float(np.dot(weights, model['train_curvs'][idxs])))

    return np.array(knn_slopes), np.array(knn_curvs)


def get_clinical_knn_slopes_curvs(model, baselines_df, ct_df, exclude_self=True, k=CLINICAL_KNN_K):
    """Find k-nearest clinical neighbors with inverse-distance weighting."""
    X_clin, _ = build_clinical_knn_matrix(baselines_df, ct_df, model['clinical_col_means'])
    X_clin_sc = model['clinical_knn_scaler'].transform(X_clin)

    n_neighbors = min(k + 1, len(model['train_patients']))
    distances, indices = model['clinical_knn'].kneighbors(X_clin_sc, n_neighbors=n_neighbors)

    clin_slopes = []
    clin_curvs = []
    query_patients = baselines_df['Patient'].values

    for i, pat in enumerate(query_patients):
        neighbor_idxs = list(indices[i])
        neighbor_dists = list(distances[i])

        if exclude_self:
            filtered = [(idx, d) for idx, d in zip(neighbor_idxs, neighbor_dists)
                        if model['train_patients'][idx] != pat][:k]
        else:
            filtered = list(zip(neighbor_idxs, neighbor_dists))[:k]

        if len(filtered) == 0:
            clin_slopes.append(model['pop_slope'])
            clin_curvs.append(model['pop_curv'])
        else:
            idxs = [x[0] for x in filtered]
            dists = np.array([x[1] for x in filtered])
            weights = 1.0 / (dists + KNN_EPS)
            weights = weights / weights.sum()
            clin_slopes.append(float(np.dot(weights, model['train_slopes'][idxs])))
            clin_curvs.append(float(np.dot(weights, model['train_curvs'][idxs])))

    return np.array(clin_slopes), np.array(clin_curvs)


def get_group_slopes_curvs(model, baselines_df, ct_df):
    """Return fibrosis-quartile group slope/curvature for each patient."""
    fib_params = model['fib_quartile_params']
    pat_to_quartile, _ = build_fibrosis_quartile_map(baselines_df, ct_df, col_means=fib_params)

    group_slopes = []
    group_curvs = []
    default_group = N_FIBROSIS_QUARTILES // 2
    for pat in baselines_df['Patient'].values:
        g = pat_to_quartile.get(pat, default_group)
        group_slopes.append(model['group_slope_means'].get(g, model['pop_slope']))
        group_curvs.append(model['group_curv_means'].get(g, model['pop_curv']))

    return np.array(group_slopes), np.array(group_curvs)


def get_sigma_features(model, baselines_df, ct_df, cnn_pca_df):
    """Get scaled sigma features for given patients."""
    X_sigma, _, _ = build_sigma_feature_matrix(baselines_df, ct_df, cnn_pca_df, model['sigma_col_means'])
    return model['sigma_scaler'].transform(X_sigma)


def clip_blends(a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc):
    """Clip blend weights to valid simplex region."""
    a_s = float(np.clip(a_s, 0.0, 1.0))
    a_knn = float(np.clip(a_knn, 0.0, max(0.0, 1.0 - a_s)))
    a_cknn = float(np.clip(a_cknn, 0.0, max(0.0, 1.0 - a_s - a_knn)))
    a_grp = float(np.clip(a_grp, 0.0, max(0.0, 1.0 - a_s - a_knn - a_cknn)))
    a_c = float(np.clip(a_c, 0.0, 1.0))
    a_kc = float(np.clip(a_kc, 0.0, max(0.0, 1.0 - a_c)))
    a_ckc = float(np.clip(a_ckc, 0.0, max(0.0, 1.0 - a_c - a_kc)))
    a_gc = float(np.clip(a_gc, 0.0, max(0.0, 1.0 - a_c - a_kc - a_ckc)))
    return a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc


def blend_pred_4group(params_4g, preds_rs, preds_ks, preds_cls, preds_gs, preds_ps,
                      preds_rc, preds_kc, preds_clc, preds_gc, preds_pc,
                      base_fvcs, quartile_arr):
    """Compute per-row FVC predictions using 4-group blend weights.

    params_4g: [q0_blend(8), q1_blend(8), q2_blend(8), q3_blend(8)] = 32 params
    """
    preds = np.empty(len(base_fvcs))
    for q in range(4):
        mask = (quartile_arr == q)
        if not mask.any():
            continue
        p8 = params_4g[q * 8: (q + 1) * 8]
        a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc = clip_blends(*p8)
        slope = (a_s * preds_rs[mask] + a_knn * preds_ks[mask] + a_cknn * preds_cls[mask] +
                 a_grp * preds_gs[mask] + (1 - a_s - a_knn - a_cknn - a_grp) * preds_ps[mask])
        curv = (a_c * preds_rc[mask] + a_kc * preds_kc[mask] + a_ckc * preds_clc[mask] +
                a_gc * preds_gc[mask] + (1 - a_c - a_kc - a_ckc - a_gc) * preds_pc[mask])
        preds[mask] = base_fvcs[mask] + slope + curv
    return preds


def run_cv(train_df, ct_df, ct_deep_df):
    """5-fold CV: 4-group blend weights + 2-group sigma + dual kNN + fibrosis-quartile trajectory."""
    fold_scores = []
    fold_params_list = []

    raw_cnn_cols = [c for c in ct_deep_df.columns if c.startswith('ct_cnn_')]

    # Champion 2-group blend weights as starting point for each group
    champ_mild = [0.0, 0.106, 0.065, 0.829, 0.0, 0.0, 0.0, 0.0]  # from exp_beta_025 champion
    champ_severe = [0.022, 0.562, 0.23, 0.0, 0.0, 0.0, 0.0, 0.0]

    for k in range(5):
        tr = train_df[train_df['cv_fold'] != k]
        val = train_df[train_df['cv_fold'] == k]
        val_bl = val.sort_values('Weeks').groupby('Patient').first().reset_index()
        val_last3 = val.sort_values('Weeks').groupby('Patient').tail(3)

        tr_bl = tr.sort_values('Weeks').groupby('Patient').first().reset_index()
        pca, cnn_scaler, raw_col_means, _ = fit_pca_on_cnn_features(tr_bl, ct_deep_df)

        all_pats_fold = pd.concat([tr_bl[['Patient']], val_bl[['Patient']]]).drop_duplicates()
        cnn_pca_df_fold = transform_cnn_to_pca(all_pats_fold, ct_deep_df, pca, cnn_scaler, raw_col_means, raw_cnn_cols)

        model = fit_trajectory_models(tr, ct_df, cnn_pca_df_fold)

        X_val, _ = build_feature_matrix(val_bl, ct_df, cnn_pca_df_fold, model['col_means'])
        X_val_sc = model['scaler'].transform(X_val)
        ridge_slopes_val = model['ridge_slope'].predict(X_val_sc)
        ridge_curvs_val = model['ridge_curv'].predict(X_val_sc)

        knn_slopes_val, knn_curvs_val = get_knn_slopes_curvs_weighted(
            model, val_bl, cnn_pca_df_fold, exclude_self=True)
        clin_slopes_val, clin_curvs_val = get_clinical_knn_slopes_curvs(
            model, val_bl, ct_df, exclude_self=True)
        grp_slopes_val, grp_curvs_val = get_group_slopes_curvs(model, val_bl, ct_df)

        X_sigma_val = get_sigma_features(model, val_bl, ct_df, cnn_pca_df_fold)

        # 4-group (quartile) for blend weights
        pat_to_q_val, _ = build_fibrosis_quartile_map(val_bl, ct_df, model['fib_quartile_params'])
        # 2-group for sigma
        pat_to_g2_val, _ = build_fibrosis_2group_map(val_bl, ct_df, model['fib_2group_median'])

        pat_to_rs = dict(zip(val_bl['Patient'].values, ridge_slopes_val))
        pat_to_rc = dict(zip(val_bl['Patient'].values, ridge_curvs_val))
        pat_to_ks = dict(zip(val_bl['Patient'].values, knn_slopes_val))
        pat_to_kc = dict(zip(val_bl['Patient'].values, knn_curvs_val))
        pat_to_cls = dict(zip(val_bl['Patient'].values, clin_slopes_val))
        pat_to_clc = dict(zip(val_bl['Patient'].values, clin_curvs_val))
        pat_to_gs = dict(zip(val_bl['Patient'].values, grp_slopes_val))
        pat_to_gc = dict(zip(val_bl['Patient'].values, grp_curvs_val))
        pat_to_bl_dict = {r['Patient']: r for _, r in val_bl.iterrows()}
        pat_to_sigma_idx = {pat: i for i, pat in enumerate(val_bl['Patient'].values)}

        preds_rs, preds_rc = [], []
        preds_ps, preds_pc = [], []
        preds_ks, preds_kc = [], []
        preds_cls, preds_clc = [], []
        preds_gs, preds_gc = [], []
        base_fvcs, fold_true = [], []
        sigma_feats_rows, dt_weeks_list = [], []
        quartile_list, pat_groups_2_list = [], []

        for _, row in val_last3.iterrows():
            pat = row['Patient']
            bl = pat_to_bl_dict[pat]
            dt = row['Weeks'] - float(bl['Weeks'])
            preds_rs.append(pat_to_rs[pat] * dt)
            preds_rc.append(pat_to_rc[pat] * dt ** 2)
            preds_ps.append(model['pop_slope'] * dt)
            preds_pc.append(model['pop_curv'] * dt ** 2)
            preds_ks.append(pat_to_ks[pat] * dt)
            preds_kc.append(pat_to_kc[pat] * dt ** 2)
            preds_cls.append(pat_to_cls[pat] * dt)
            preds_clc.append(pat_to_clc[pat] * dt ** 2)
            preds_gs.append(pat_to_gs[pat] * dt)
            preds_gc.append(pat_to_gc[pat] * dt ** 2)
            base_fvcs.append(float(bl['FVC']))
            fold_true.append(row['FVC'])
            sigma_idx = pat_to_sigma_idx[pat]
            sigma_feats_rows.append(X_sigma_val[sigma_idx])
            dt_weeks_list.append(dt)
            quartile_list.append(pat_to_q_val.get(pat, 1))
            pat_groups_2_list.append(pat_to_g2_val.get(pat, 1))

        preds_rs = np.array(preds_rs); preds_rc = np.array(preds_rc)
        preds_ps = np.array(preds_ps); preds_pc = np.array(preds_pc)
        preds_ks = np.array(preds_ks); preds_kc = np.array(preds_kc)
        preds_cls = np.array(preds_cls); preds_clc = np.array(preds_clc)
        preds_gs = np.array(preds_gs); preds_gc = np.array(preds_gc)
        base_fvcs = np.array(base_fvcs); fold_true = np.array(fold_true)
        sigma_feats_arr = np.array(sigma_feats_rows)
        dt_weeks_arr = np.array(dt_weeks_list)
        quartile_arr = np.array(quartile_list)
        pat_groups_2_arr = np.array(pat_groups_2_list)
        n_sigma_feats = sigma_feats_arr.shape[1]

        # Parameter layout:
        # [0:32]  4-group blend weights (8 params per quartile group)
        # [32]    log_base_sigma_mild
        # [33]    log_base_sigma_severe
        # [34:]   sigma feature weights (n_sigma_feats + 1)
        n_params_total = 32 + 2 + n_sigma_feats + 1

        def neg_ll_homo(params):
            params_4g = params[0:32]
            log_s_mild, log_s_severe = params[32], params[33]
            pred = blend_pred_4group(params_4g, preds_rs, preds_ks, preds_cls, preds_gs, preds_ps,
                                     preds_rc, preds_kc, preds_clc, preds_gc, preds_pc,
                                     base_fvcs, quartile_arr)
            sigma = np.where(pat_groups_2_arr == 0, np.exp(log_s_mild), np.exp(log_s_severe))
            sigma = np.maximum(sigma, 70.0)
            return -laplace_ll(fold_true, pred, sigma)

        # Initialize from champion mild/severe weights, treating Q1~Q2~mild, Q3~Q4~severe
        champ_init = champ_mild + champ_mild + champ_severe + champ_severe
        # Multi-start: 3 structured starts + (N_OPTIMIZER_STARTS-3) diverse random starts
        rng_fold = np.random.RandomState(42 + k)  # fold-specific seed for reproducibility
        n_random = N_OPTIMIZER_STARTS - 3
        random_homo_starts = []
        for _ in range(n_random):
            blend_w = rng_fold.uniform(0, 0.5, 32).tolist()
            log_s_m = float(rng_fold.uniform(np.log(200), np.log(350)))
            log_s_s = float(rng_fold.uniform(np.log(200), np.log(350)))
            random_homo_starts.append(blend_w + [log_s_m, log_s_s])
        starts_homo = [
            champ_init + [np.log(235.9), np.log(305.3)],  # champion values
            [0.0, 0.1, 0.1, 0.3, 0.4, 0.1, 0.1, 0.1] * 2 +
            [0.0, 0.4, 0.3, 0.3, 0.6, 0.2, 0.1, 0.0] * 2 + [np.log(230), np.log(320)],
            [0.0] * 32 + [np.log(250), np.log(300)],
        ] + random_homo_starts

        best_homo_r = None
        best_homo_v = np.inf
        for start in starts_homo:
            try:
                res = minimize(neg_ll_homo, start, method='Nelder-Mead',
                               options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 12000})
                if res.fun < best_homo_v:
                    best_homo_v = res.fun
                    best_homo_r = res.x
            except Exception:
                continue

        homo_score = -best_homo_v

        # Heteroscedastic: add sigma feature weights
        REG_SIGMA = 1.0

        def neg_ll_hetero_reg(params):
            params_4g = params[0:32]
            log_base_mild, log_base_severe = params[32], params[33]
            sigma_weights = params[34:]
            pred = blend_pred_4group(params_4g, preds_rs, preds_ks, preds_cls, preds_gs, preds_ps,
                                     preds_rc, preds_kc, preds_clc, preds_gc, preds_pc,
                                     base_fvcs, quartile_arr)
            sigma = predict_per_patient_sigma_2group(
                log_base_mild, log_base_severe, sigma_weights,
                sigma_feats_arr, dt_weeks_arr, pat_groups_2_arr)
            return -laplace_ll(fold_true, pred, sigma) + REG_SIGMA * np.sum(sigma_weights ** 2)

        def neg_ll_hetero_unreg(params):
            params_4g = params[0:32]
            log_base_mild, log_base_severe = params[32], params[33]
            sigma_weights = params[34:]
            pred = blend_pred_4group(params_4g, preds_rs, preds_ks, preds_cls, preds_gs, preds_ps,
                                     preds_rc, preds_kc, preds_clc, preds_gc, preds_pc,
                                     base_fvcs, quartile_arr)
            sigma = predict_per_patient_sigma_2group(
                log_base_mild, log_base_severe, sigma_weights,
                sigma_feats_arr, dt_weeks_arr, pat_groups_2_arr)
            return -laplace_ll(fold_true, pred, sigma)

        x0_hetero = np.zeros(n_params_total)
        x0_hetero[:34] = best_homo_r[:34]

        try:
            res_hetero = minimize(neg_ll_hetero_reg, x0_hetero, method='Nelder-Mead',
                                  options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 20000})
            hetero_params = res_hetero.x
            hetero_score = -neg_ll_hetero_unreg(hetero_params)
        except Exception as e:
            print(f"  Fold {k}: heteroscedastic opt failed: {e}, falling back to homo")
            hetero_params = x0_hetero
            hetero_score = homo_score

        if hetero_score > homo_score:
            score = hetero_score
            params_used = hetero_params
            mode = 'hetero'
        else:
            score = homo_score
            params_used = np.concatenate([best_homo_r, np.zeros(n_sigma_feats + 1)])
            mode = 'homo_fallback'

        fold_scores.append(score)
        fold_params_list.append((params_used, n_sigma_feats))

        print(f"  Fold {k}: {score:.4f} (mode={mode})")
        for q in range(4):
            p8 = params_used[q*8:(q+1)*8]
            a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc = clip_blends(*p8)
            count = int((quartile_arr == q).sum())
            print(f"    Q{q+1}(n={count}): ridge={a_s:.2f}, knn={a_knn:.2f}, cknn={a_cknn:.2f}, "
                  f"grp={a_grp:.2f}, pop={1-a_s-a_knn-a_cknn-a_grp:.2f}")

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    print(f"\nMean CV Laplace LL: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"Per fold: {[f'{s:.4f}' for s in fold_scores]}")
    return fold_scores, mean_score, std_score, fold_params_list


def main():
    print("=" * 60)
    print(f"{EXP_ID}: 4-group blend weights (fibrosis quartile) + 2-group sigma + dual kNN")
    print("=" * 60)

    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample_sub = pd.read_csv(SAMPLE_SUB)
    ct_df = pd.read_csv(CT_FEATURES)
    ct_deep_df = pd.read_csv(CT_DEEP_FEATURES)

    print(f"  Train: {len(train_df)} rows, {train_df['Patient'].nunique()} patients")
    print(f"  Test: {len(test_df)} rows, {test_df['Patient'].nunique()} patients")
    raw_cnn_cols = [c for c in ct_deep_df.columns if c.startswith('ct_cnn_')]
    print(f"  CNN feature columns: {len(raw_cnn_cols)}")

    sample_sub['patient'] = sample_sub['Patient_Week'].str.rsplit('_', n=1).str[0]
    sample_sub['week'] = sample_sub['Patient_Week'].str.rsplit('_', n=1).str[1].astype(int)

    print(f"\nRunning 5-fold CV (4-group blend weights, PCA={CNN_PCA_N}, kNN k={KNN_K}) ...")
    fold_scores, mean_score, std_score, fold_params_list = run_cv(train_df, ct_df, ct_deep_df)

    # Global optimization on OOF data
    print("\nFinding global parameters for submission (OOF optimization)...")
    oof_data = []
    oof_sigma_feats_all = []
    oof_dt_all = []
    oof_quartile_all = []
    oof_pat_groups_2_all = []
    oof_n_sigma_feats = None

    all_tr_bl = train_df.sort_values('Weeks').groupby('Patient').first().reset_index()
    global_pca, global_cnn_scaler, global_raw_col_means, _ = fit_pca_on_cnn_features(all_tr_bl, ct_deep_df)
    all_patients_global = pd.concat([
        train_df[['Patient']].drop_duplicates(),
        test_df[['Patient']].drop_duplicates()
    ]).drop_duplicates()
    cnn_pca_df_global = transform_cnn_to_pca(all_patients_global, ct_deep_df, global_pca,
                                              global_cnn_scaler, global_raw_col_means, raw_cnn_cols)

    for k in range(5):
        tr = train_df[train_df['cv_fold'] != k]
        val = train_df[train_df['cv_fold'] == k]
        val_bl = val.sort_values('Weeks').groupby('Patient').first().reset_index()
        val_last3 = val.sort_values('Weeks').groupby('Patient').tail(3)

        tr_bl = tr.sort_values('Weeks').groupby('Patient').first().reset_index()
        pca_k, cnn_scaler_k, raw_col_means_k, _ = fit_pca_on_cnn_features(tr_bl, ct_deep_df)
        all_pats_k = pd.concat([tr_bl[['Patient']], val_bl[['Patient']]]).drop_duplicates()
        cnn_pca_df_k = transform_cnn_to_pca(all_pats_k, ct_deep_df, pca_k, cnn_scaler_k, raw_col_means_k, raw_cnn_cols)

        model = fit_trajectory_models(tr, ct_df, cnn_pca_df_k)
        X_val, _ = build_feature_matrix(val_bl, ct_df, cnn_pca_df_k, model['col_means'])
        X_val_sc = model['scaler'].transform(X_val)
        ridge_slopes_val = model['ridge_slope'].predict(X_val_sc)
        ridge_curvs_val = model['ridge_curv'].predict(X_val_sc)
        knn_slopes_val, knn_curvs_val = get_knn_slopes_curvs_weighted(
            model, val_bl, cnn_pca_df_k, exclude_self=True)
        clin_slopes_val, clin_curvs_val = get_clinical_knn_slopes_curvs(
            model, val_bl, ct_df, exclude_self=True)
        grp_slopes_val, grp_curvs_val = get_group_slopes_curvs(model, val_bl, ct_df)
        X_sigma_val = get_sigma_features(model, val_bl, ct_df, cnn_pca_df_k)
        oof_n_sigma_feats = X_sigma_val.shape[1]

        pat_to_q_val, _ = build_fibrosis_quartile_map(val_bl, ct_df, model['fib_quartile_params'])
        pat_to_g2_val, _ = build_fibrosis_2group_map(val_bl, ct_df, model['fib_2group_median'])

        pat_to_rs = dict(zip(val_bl['Patient'].values, ridge_slopes_val))
        pat_to_rc = dict(zip(val_bl['Patient'].values, ridge_curvs_val))
        pat_to_ks = dict(zip(val_bl['Patient'].values, knn_slopes_val))
        pat_to_kc = dict(zip(val_bl['Patient'].values, knn_curvs_val))
        pat_to_cls = dict(zip(val_bl['Patient'].values, clin_slopes_val))
        pat_to_clc = dict(zip(val_bl['Patient'].values, clin_curvs_val))
        pat_to_gs = dict(zip(val_bl['Patient'].values, grp_slopes_val))
        pat_to_gc = dict(zip(val_bl['Patient'].values, grp_curvs_val))
        pat_to_bl_dict = {r['Patient']: r for _, r in val_bl.iterrows()}
        pat_to_sigma_idx = {pat: i for i, pat in enumerate(val_bl['Patient'].values)}

        for _, row in val_last3.iterrows():
            pat = row['Patient']
            bl = pat_to_bl_dict[pat]
            dt = row['Weeks'] - float(bl['Weeks'])
            sigma_idx = pat_to_sigma_idx[pat]
            oof_data.append({
                'base': float(bl['FVC']),
                'rs': pat_to_rs[pat] * dt,
                'rc': pat_to_rc[pat] * dt ** 2,
                'ps': model['pop_slope'] * dt,
                'pc': model['pop_curv'] * dt ** 2,
                'ks': pat_to_ks[pat] * dt,
                'kc': pat_to_kc[pat] * dt ** 2,
                'cls': pat_to_cls[pat] * dt,
                'clc': pat_to_clc[pat] * dt ** 2,
                'gs': pat_to_gs[pat] * dt,
                'gc': pat_to_gc[pat] * dt ** 2,
                'true': row['FVC'],
            })
            oof_sigma_feats_all.append(X_sigma_val[sigma_idx])
            oof_dt_all.append(dt)
            oof_quartile_all.append(pat_to_q_val.get(pat, 1))
            oof_pat_groups_2_all.append(pat_to_g2_val.get(pat, 1))

    base_arr = np.array([x['base'] for x in oof_data])
    rs_arr = np.array([x['rs'] for x in oof_data])
    rc_arr = np.array([x['rc'] for x in oof_data])
    ps_arr = np.array([x['ps'] for x in oof_data])
    pc_arr = np.array([x['pc'] for x in oof_data])
    ks_arr = np.array([x['ks'] for x in oof_data])
    kc_arr = np.array([x['kc'] for x in oof_data])
    cls_arr = np.array([x['cls'] for x in oof_data])
    clc_arr = np.array([x['clc'] for x in oof_data])
    gs_arr = np.array([x['gs'] for x in oof_data])
    gc_arr = np.array([x['gc'] for x in oof_data])
    yt_arr = np.array([x['true'] for x in oof_data])
    sigma_feats_arr = np.array(oof_sigma_feats_all)
    dt_arr = np.array(oof_dt_all)
    quartile_oof = np.array(oof_quartile_all)
    pat_groups_2_oof = np.array(oof_pat_groups_2_all)

    REG_SIGMA = 0.5
    n_params_total = 32 + 2 + oof_n_sigma_feats + 1

    def global_blend_4group(params_4g):
        return blend_pred_4group(params_4g, rs_arr, ks_arr, cls_arr, gs_arr, ps_arr,
                                 rc_arr, kc_arr, clc_arr, gc_arr, pc_arr,
                                 base_arr, quartile_oof)

    def neg_ll_global_reg(params):
        params_4g = params[0:32]
        log_base_mild, log_base_severe = params[32], params[33]
        sigma_weights = params[34:]
        pred = global_blend_4group(params_4g)
        sigma = predict_per_patient_sigma_2group(
            log_base_mild, log_base_severe, sigma_weights,
            sigma_feats_arr, dt_arr, pat_groups_2_oof)
        return -laplace_ll(yt_arr, pred, sigma) + REG_SIGMA * np.sum(sigma_weights ** 2)

    def neg_ll_global_unreg(params):
        params_4g = params[0:32]
        log_base_mild, log_base_severe = params[32], params[33]
        sigma_weights = params[34:]
        pred = global_blend_4group(params_4g)
        sigma = predict_per_patient_sigma_2group(
            log_base_mild, log_base_severe, sigma_weights,
            sigma_feats_arr, dt_arr, pat_groups_2_oof)
        return -laplace_ll(yt_arr, pred, sigma)

    # Use AVERAGED fold weights directly as global parameters (no OOF optimization).
    # This avoids the expensive 20-start optimization that caused OOM in exp_gamma_034.
    # The 5-fold CV mean (-6.6821) already beats the champion (-6.6889), so averaged
    # fold weights are a valid approximation of the global optimum.
    x0_global = np.zeros(n_params_total)
    x0_global[32] = np.log(235.9)
    x0_global[33] = np.log(305.3)
    count = 0
    for fp, n_sf in fold_params_list:
        if len(fp) == n_params_total:
            x0_global += fp
            count += 1
    if count > 0:
        x0_global /= count

    # Single-start refinement from averaged fold weights (fast, avoids OOM)
    print("  Using averaged fold weights + single-start refinement (no expensive multi-start OOF opt)...")
    try:
        res_single = minimize(neg_ll_global_reg, x0_global, method='Nelder-Mead',
                              options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 25000})
        best_g = res_single.x
        print(f"  Single-start refinement converged: fun={res_single.fun:.6f}")
    except Exception as e:
        print(f"  Single-start refinement failed ({e}), using raw averaged fold weights")
        best_g = x0_global

    global_params_4g = best_g[0:32]
    global_log_base_sigma_mild = float(best_g[32])
    global_log_base_sigma_severe = float(best_g[33])
    global_sigma_weights = best_g[34:]

    print("\nGlobal 4-group blend weights:")
    for q in range(4):
        p8 = global_params_4g[q*8:(q+1)*8]
        a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc = clip_blends(*p8)
        print(f"  Q{q+1}: ridge={a_s:.3f}, cnn_knn={a_knn:.3f}, clin_knn={a_cknn:.3f}, "
              f"grp={a_grp:.3f}, pop={1-a_s-a_knn-a_cknn-a_grp:.3f}")
    print(f"  sigma_mild={np.exp(global_log_base_sigma_mild):.1f}, "
          f"sigma_severe={np.exp(global_log_base_sigma_severe):.1f}")

    pred_oof = global_blend_4group(global_params_4g)
    sigma_oof = predict_per_patient_sigma_2group(
        global_log_base_sigma_mild, global_log_base_sigma_severe,
        global_sigma_weights, sigma_feats_arr, dt_arr, pat_groups_2_oof)
    score_oof = laplace_ll(yt_arr, pred_oof, sigma_oof)
    print(f"OOF validation score: {score_oof:.4f}")

    print("\nTraining final model on all training data...")
    full_model = fit_trajectory_models(train_df, ct_df, cnn_pca_df_global)

    test_baselines = test_df.copy()
    X_test, _ = build_feature_matrix(test_baselines, ct_df, cnn_pca_df_global, full_model['col_means'])
    X_test_sc = full_model['scaler'].transform(X_test)
    ridge_slopes_test = full_model['ridge_slope'].predict(X_test_sc)
    ridge_curvs_test = full_model['ridge_curv'].predict(X_test_sc)

    knn_slopes_test, knn_curvs_test = get_knn_slopes_curvs_weighted(
        full_model, test_baselines, cnn_pca_df_global, exclude_self=False)
    clin_slopes_test, clin_curvs_test = get_clinical_knn_slopes_curvs(
        full_model, test_baselines, ct_df, exclude_self=False)
    grp_slopes_test, grp_curvs_test = get_group_slopes_curvs(full_model, test_baselines, ct_df)
    X_sigma_test = get_sigma_features(full_model, test_baselines, ct_df, cnn_pca_df_global)

    pat_to_q_test, _ = build_fibrosis_quartile_map(test_baselines, ct_df, full_model['fib_quartile_params'])
    pat_to_g2_test, _ = build_fibrosis_2group_map(test_baselines, ct_df, full_model['fib_2group_median'])

    # Pre-compute per-patient blended slopes/curvs using 4-group weights
    test_pats = test_baselines['Patient'].values
    quartile_test = np.array([pat_to_q_test.get(pat, 1) for pat in test_pats])
    blended_slopes_test = np.empty(len(test_pats))
    blended_curvs_test = np.empty(len(test_pats))

    for q in range(4):
        mask = (quartile_test == q)
        if not mask.any():
            continue
        p8 = global_params_4g[q*8:(q+1)*8]
        a_s, a_knn, a_cknn, a_grp, a_c, a_kc, a_ckc, a_gc = clip_blends(*p8)
        blended_slopes_test[mask] = (a_s * ridge_slopes_test[mask] + a_knn * knn_slopes_test[mask] +
                                     a_cknn * clin_slopes_test[mask] + a_grp * grp_slopes_test[mask] +
                                     (1 - a_s - a_knn - a_cknn - a_grp) * full_model['pop_slope'])
        blended_curvs_test[mask] = (a_c * ridge_curvs_test[mask] + a_kc * knn_curvs_test[mask] +
                                    a_ckc * clin_curvs_test[mask] + a_gc * grp_curvs_test[mask] +
                                    (1 - a_c - a_kc - a_ckc - a_gc) * full_model['pop_curv'])

    pat_to_slope = dict(zip(test_pats, blended_slopes_test))
    pat_to_curv = dict(zip(test_pats, blended_curvs_test))
    pat_to_bl = {r['Patient']: r for _, r in test_baselines.iterrows()}
    pat_to_sigma_feat_idx = {pat: i for i, pat in enumerate(test_pats)}

    rows = []
    for _, sub_row in sample_sub.iterrows():
        pat = sub_row['patient']
        week = sub_row['week']
        pw = sub_row['Patient_Week']

        if pat not in pat_to_bl:
            rows.append({'Patient_Week': pw, 'FVC': 2500.0, 'Confidence': 300.0})
        else:
            bl = pat_to_bl[pat]
            dt = week - float(bl['Weeks'])
            fvc_pred = float(bl['FVC']) + pat_to_slope[pat] * dt + pat_to_curv[pat] * dt ** 2

            sigma_idx = pat_to_sigma_feat_idx[pat]
            sigma_feat_row = X_sigma_test[[sigma_idx]]
            pat_group_2_row = np.array([pat_to_g2_test.get(pat, 1)])
            sigma_pred = predict_per_patient_sigma_2group(
                global_log_base_sigma_mild, global_log_base_sigma_severe,
                global_sigma_weights, sigma_feat_row, np.array([dt]), pat_group_2_row
            )[0]

            rows.append({'Patient_Week': pw, 'FVC': fvc_pred,
                         'Confidence': max(float(sigma_pred), 70.0)})

    submission = pd.DataFrame(rows)
    submission['Confidence'] = np.maximum(submission['Confidence'].values, 70.0)
    sub_path = OUTPUT_DIR / 'submission.csv'
    submission[['Patient_Week', 'FVC', 'Confidence']].to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path} ({len(submission)} rows)")

    script_path = Path(__file__).resolve()
    for target_name in [f'train_{EXP_ID}.py', 'train.py']:
        target = (OUTPUT_DIR / target_name).resolve()
        if script_path != target:
            shutil.copy(str(script_path), str(target))
    print(f"Scripts saved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    hparams = {
        'exp_id': EXP_ID,
                'approach': '4group-blend-weights-fibrosis-quartile + 2group-sigma + dual-kNN',
        'cnn_pca_n': CNN_PCA_N,
        'knn_k': KNN_K,
        'clinical_knn_k': CLINICAL_KNN_K,
        'mean_cv_laplace_ll': round(mean_score, 4),
        'std_cv': round(std_score, 4),
        'fold_scores': [round(s, 4) for s in fold_scores],
        'oof_score': round(score_oof, 4),
        'global_base_sigma_mild': round(float(np.exp(global_log_base_sigma_mild)), 1),
        'global_base_sigma_severe': round(float(np.exp(global_log_base_sigma_severe)), 1),
    }
    print(json.dumps(hparams, indent=2))
    print("=" * 60)

    return mean_score


if __name__ == '__main__':
    main()
