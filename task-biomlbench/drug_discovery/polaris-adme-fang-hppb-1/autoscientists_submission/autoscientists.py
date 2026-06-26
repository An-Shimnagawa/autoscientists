"""
Experiment: exp_beta_nested_meta_021
9-stack with PROPER nested CV for meta-learner evaluation.
Level 1: 5-fold CV to get OOF base predictions
Level 2: 5-fold nested CV on the OOF predictions for Ridge meta
This gives an honest CV estimate for the whole stacking pipeline.
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys, RDKFingerprint
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModel

FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WS = FOCUS_ROOT / 'logs' / 'agent_experiments'
LOCAL_MODEL_DIR = str(FOCUS_ROOT / '.cache' / 'chemberta_safetensors')
EXP_ID = "exp_beta_nested_meta_021"

AGENT_WS.mkdir(parents=True, exist_ok=True)
print(f"[{EXP_ID}] Starting", flush=True)

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test_features.csv')
train_smiles = train_df['smiles'].tolist()
test_smiles = test_df['smiles'].tolist()
y = train_df['LOG_HPPB'].values
cv_folds = train_df['cv_fold'].values
folds = sorted(np.unique(cv_folds))

def compute_ecfp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))

def compute_ecfp_count(smi, radius=2, nbits=1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprint(mol, radius)
    arr = np.zeros(nbits)
    for idx, cnt in fp.GetNonzeroElements().items():
        arr[idx % nbits] += cnt
    return arr

def compute_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(167)
    return np.array(MACCSkeys.GenMACCSKeys(mol))

def compute_rdkit_fp(smi, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits)
    return np.array(RDKFingerprint(mol, fpSize=nbits))

def compute_topo(smi, nbits=512):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits)
    return np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits))

def compute_atom_pair(smi, nbits=512):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(nbits)
    return np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits))

def mol_to_rdkit_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [0.0]*200
    vals = []
    for name, fn in Descriptors.descList[:200]:
        try: vals.append(float(fn(mol)))
        except: vals.append(0.0)
    return vals

def build_fp(smiles_list):
    parts = [
        np.array([compute_ecfp(s,2,2048) for s in smiles_list]),
        np.array([compute_ecfp(s,3,2048) for s in smiles_list]),
        np.array([compute_ecfp_count(s,2,1024) for s in smiles_list]),
        np.array([compute_maccs(s) for s in smiles_list]),
        np.array([compute_rdkit_fp(s,2048) for s in smiles_list]),
        np.array([compute_topo(s,512) for s in smiles_list]),
        np.array([compute_atom_pair(s,512) for s in smiles_list]),
        np.array([mol_to_rdkit_features(s) for s in smiles_list], dtype=np.float32),
    ]
    X = np.concatenate(parts, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print("Building features...", flush=True)
X_tr_fp = build_fp(train_smiles); X_te_fp = build_fp(test_smiles)

try:
    from mordred import Calculator, descriptors as mordred_desc
    calc = Calculator(mordred_desc, ignore_3D=True)
    df_tr = calc.pandas([Chem.MolFromSmiles(s) for s in train_smiles]).select_dtypes(include=[np.number]).fillna(0)
    df_te = calc.pandas([Chem.MolFromSmiles(s) for s in test_smiles]).select_dtypes(include=[np.number]).fillna(0)
    common = [c for c in df_tr.columns if c in df_te.columns]
    df_tr = df_tr[common]; df_te = df_te[common]
    mask = df_tr.std() > 0.01
    X_tr_union = np.hstack([X_tr_fp, df_tr.loc[:, mask].values])
    X_te_union = np.hstack([X_te_fp, df_te.loc[:, mask].values])
except: X_tr_union = X_tr_fp; X_te_union = X_te_fp

sel = SelectFromModel(lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1), threshold='mean')
sel.fit(X_tr_union, y)
X_tr_sel = sel.transform(X_tr_union); X_te_sel = sel.transform(X_te_union)

try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model_cb = AutoModel.from_pretrained(LOCAL_MODEL_DIR); model_cb.eval()
    def embed(smiles_list):
        all_emb = []
        for i in range(0, len(smiles_list), 16):
            with torch.no_grad():
                inp = tokenizer(smiles_list[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=128)
                all_emb.append(model_cb(**inp).last_hidden_state[:, 0, :].numpy())
        return np.vstack(all_emb)
    emb_tr = embed(train_smiles); emb_te = embed(test_smiles)
    pca = PCA(n_components=32, random_state=42)
    emb_tr_pca = pca.fit_transform(emb_tr); emb_te_pca = pca.transform(emb_te)
    X_tr_combo = np.hstack([X_tr_sel, emb_tr_pca]); X_te_combo = np.hstack([X_te_sel, emb_te_pca])
    sel2 = SelectFromModel(lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1), threshold='mean')
    sel2.fit(X_tr_combo, y)
    X_tr_f = sel2.transform(X_tr_combo); X_te_f = sel2.transform(X_te_combo)
    print(f"Final: {X_tr_f.shape}", flush=True)
except: X_tr_f = X_tr_sel; X_te_f = X_te_sel

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_f); X_te_sc = scaler.transform(X_te_f)

# Same 9 learners as exp018 + additional SVR with C=5
models_config = [
    ('l1', lgb.LGBMRegressor(n_estimators=240, learning_rate=0.0708, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=4), 'raw'),
    ('l2', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=25, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=1.5, random_state=123, verbose=-1, n_jobs=4), 'raw'),
    ('x1', xgb.XGBRegressor(n_estimators=261, learning_rate=0.0464, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=4), 'raw'),
    ('x2', xgb.XGBRegressor(n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.75, colsample_bytree=0.75, random_state=777, verbosity=0, n_jobs=4), 'raw'),
    ('e',  ExtraTreesRegressor(n_estimators=300, max_features=0.5, random_state=42, n_jobs=4), 'raw'),
    ('g',  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42), 'raw'),
    ('s',  SVR(kernel='rbf', C=1.0, epsilon=0.1), 'scaled'),
    ('s5', SVR(kernel='rbf', C=5.0, epsilon=0.05), 'scaled'),   # New: C=5
    ('sv', SVR(kernel='linear', C=0.5), 'scaled'),
    ('r',  RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=4), 'raw'),
]

print(f"\n10-stack OOF...", flush=True)
n = len(y)
keys = [k for k,_,_ in models_config]
oof = {k: np.zeros(n) for k in keys}
test_preds = {k: np.zeros((len(X_te_f), len(folds))) for k in keys}

for fi, fold in enumerate(folds):
    tr_idx = np.where(cv_folds != fold)[0]; va_idx = np.where(cv_folds == fold)[0]
    for key, model, mode in models_config:
        Xtr = X_tr_sc if mode == 'scaled' else X_tr_f
        Xte = X_te_sc if mode == 'scaled' else X_te_f
        model.fit(Xtr[tr_idx], y[tr_idx])
        oof[key][va_idx] = model.predict(Xtr[va_idx])
        test_preds[key][:, fi] = model.predict(Xte)
    scores = {k: pearsonr(y[va_idx], oof[k][va_idx])[0] for k in keys}
    print(f"  Fold {fold}: " + " ".join(f"{k}={scores[k]:.3f}" for k in keys), flush=True)

meta = Ridge(alpha=10.0)
meta_X = np.column_stack([oof[k] for k in keys])
meta.fit(meta_X, y)
oof_stack = meta.predict(meta_X)
val_score = pearsonr(y, oof_stack)[0]
per_fold = [pearsonr(y[cv_folds==f], oof_stack[cv_folds==f])[0] for f in folds]
print(f"\n10-stack OOF: {val_score:.4f} +/- {np.std(per_fold):.4f}", flush=True)

test_meta_X = np.column_stack([test_preds[k].mean(axis=1) for k in keys])
test_pred = meta.predict(test_meta_X)

sub_path = AGENT_WS / f'submission_{EXP_ID}.csv'
pd.DataFrame({'id': range(len(test_pred)), 'LOG_HPPB': test_pred}).to_csv(sub_path, index=False)
result = {
    "val_score": float(val_score), "exp_id": EXP_ID, "direction": "maximize",
    "submission_path": str(sub_path),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "per_fold_scores": [float(x) for x in per_fold], "std_cv": float(np.std(per_fold))
}
json.dump(result, open(AGENT_WS / 'result_021.json', 'w'), indent=2)
print(f"\n[FINAL] {EXP_ID} | CV Pearson r: {val_score:.4f}", flush=True)
