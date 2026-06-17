"""
Meta-learner egitimi.
- Sample her gruptan 80 ornek = 29 x 80 = ~2320 meta-sample
- 14 binary olasılık (1 gate + 3 base + 5 single + 5 multi) cikar
- 14 derived feature uret
- 777 standardized tsfresh feature
- Toplam meta_X: ~808 boyut
- Modeller:
  * base_meta (5-class XGB+LGB)
  * single_anom_meta (5 binary XGB+LGB)
  * multi_anom_meta (5 binary XGB+LGB)
  * router (XGB+LGB binary): single vs combo
"""
import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import entropy as sp_entropy

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GROUPS, GROUP_NAMES, GROUP_LABELS, ANOM_LABELS, BASE_LABELS, BASE_MODEL_NAMES,
    DATA_GEN_DIR, GATE_MODELS_DIR, BASE_MODELS_DIR, SINGLE_ANOM_MODELS_DIR,
    MULTI_ANOM_MODELS_DIR, META_MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
    RANDOM_STATE, TEST_SIZE,
)
from lib import read_series, extract_batch, list_group_csvs, load_model

META_N_PER_GROUP = 80


def compute_derived(P_gate, raw_base, raw_single, raw_multi):
    """14 derived feature uret."""
    feats = []
    feats.append(float(P_gate))                                           # 1
    feats.append(float(np.max(raw_base)))                                  # 2 max base
    feats.append(float(np.argmax(raw_base)))                               # 3 argmax base
    feats.append(float(np.max(raw_single)))                                # 4 max single
    feats.append(float(np.argmax(raw_single)))                             # 5 argmax single
    feats.append(float(np.max(raw_multi)))                                 # 6 max multi
    feats.append(float(np.argmax(raw_multi)))                              # 7 argmax multi
    feats.append(float(np.sum(np.array(raw_single) > 0.5)))                # 8 n_single_above
    feats.append(float(np.sum(np.array(raw_multi) > 0.5)))                 # 9 n_multi_above
    sorted_base = sorted(raw_base, reverse=True)
    feats.append(float(sorted_base[0] - sorted_base[1]) if len(sorted_base)>=2 else 0.0)  # 10 base gap
    # single vs multi agreement
    diff = np.abs(np.array(raw_single) - np.array(raw_multi))
    feats.append(float(np.mean(diff)))                                     # 11 mean diff
    feats.append(float(np.max(diff)))                                       # 12 max diff
    # anom entropy
    s_clip = np.clip(raw_single, 1e-6, 1-1e-6)
    feats.append(float(np.mean([sp_entropy([1-x, x]) for x in s_clip])))  # 13 single entropy
    m_clip = np.clip(raw_multi, 1e-6, 1-1e-6)
    feats.append(float(np.mean([sp_entropy([1-x, x]) for x in m_clip])))  # 14 multi entropy
    return np.array(feats)


def main():
    random.seed(RANDOM_STATE)
    print("="*60)
    print("META-LEARNER EGITIMI")
    print("="*60)

    # 1. Sample: 29 grup x 80 ornek
    all_csvs = []
    all_gids = []
    for gid, name, _, _, _ in GROUPS:
        csvs = list_group_csvs(DATA_GEN_DIR, name)
        n = min(META_N_PER_GROUP, len(csvs))
        chosen = random.sample(csvs, n)
        for c in chosen:
            all_csvs.append(c)
            all_gids.append(gid)
    print(f"Toplam meta sample: {len(all_csvs)} (29 grup x ~{META_N_PER_GROUP})")

    # 2. Seri oku
    series = []
    valid_gids = []
    for csv_path, gid in zip(all_csvs, all_gids):
        s = read_series(csv_path)
        if len(s) >= 50:
            series.append(s)
            valid_gids.append(gid)
    print(f"Basarili: {len(series)}")

    # 3. tsfresh extraction
    print("\ntsfresh extraction...")
    X_tsfresh = extract_batch(series, n_jobs=4)
    print(f"tsfresh feature matrix: {X_tsfresh.shape}")
    X_clean = np.nan_to_num(X_tsfresh, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. 14 binary modelden olasilik cikar
    print("\nModeller yukleniyor...")
    gate_bundle = load_model(GATE_MODELS_DIR / "stationarity_gate.pkl")
    base_bundles = {n: load_model(BASE_MODELS_DIR / f"{n}.pkl") for n in BASE_MODEL_NAMES}
    single_bundles = {n: load_model(SINGLE_ANOM_MODELS_DIR / f"{n}.pkl") for n in ANOM_LABELS}
    multi_bundles = {n: load_model(MULTI_ANOM_MODELS_DIR / f"{n}.pkl") for n in ANOM_LABELS}

    n = len(series)
    print(f"\nEnsemble inference ({n} ornek)...")
    P_gate = np.zeros(n)
    raw_base = np.zeros((n, 3))    # 3 base
    raw_single = np.zeros((n, 5))  # 5 single-anom
    raw_multi = np.zeros((n, 5))   # 5 multi-anom

    def predict_prob(bundle, X):
        Xs = bundle["scaler"].transform(X)
        return bundle["model"].predict_proba(Xs)[:, 1]

    P_gate = predict_prob(gate_bundle, X_clean)
    for j, name in enumerate(BASE_MODEL_NAMES):
        raw_base[:, j] = predict_prob(base_bundles[name], X_clean)
    for j, name in enumerate(ANOM_LABELS):
        raw_single[:, j] = predict_prob(single_bundles[name], X_clean)
        raw_multi[:, j] = predict_prob(multi_bundles[name], X_clean)

    # 5. Derived feature
    print("\nDerived features hesaplaniyor...")
    derived = np.zeros((n, 14))
    for i in range(n):
        derived[i] = compute_derived(P_gate[i], raw_base[i], raw_single[i], raw_multi[i])

    # 6. tsfresh standardize
    tsfresh_scaler = StandardScaler()
    X_ts_scaled = tsfresh_scaler.fit_transform(X_clean)
    joblib.dump(tsfresh_scaler, PROCESSED_DIR / "tsfresh_scaler.pkl")

    # 7. Meta_X olusturma
    # Boyut: 1 (gate) + 3 (base) + 5 (single) + 5 (multi) + 14 (derived) + 777 (tsfresh) = 805
    meta_X = np.hstack([
        P_gate.reshape(-1, 1), raw_base, raw_single, raw_multi, derived, X_ts_scaled,
    ])
    print(f"meta_X boyutu: {meta_X.shape}")
    np.save(PROCESSED_DIR / "meta_X.npy", meta_X)

    # 8. Ground truth etiketler
    BASE_MAP = {b: i for i, b in enumerate(BASE_LABELS)}  # 5-class
    y_base = np.zeros(n, dtype=int)
    y_anom = np.zeros((n, 5), dtype=int)
    y_router = np.zeros(n, dtype=int)  # 0=single(saf base), 1=combo
    SAF_BASE_GIDS = {2, 3, 4}  # G2, G3, G4 saf base

    for i, gid in enumerate(valid_gids):
        base_lbl, anoms = GROUP_LABELS[gid]
        y_base[i] = BASE_MAP[base_lbl]
        for j, a in enumerate(ANOM_LABELS):
            if a in anoms:
                y_anom[i, j] = 1
        # Router: stationary ve saf base → 0 (single), digerleri → 1 (combo)
        if gid == 1 or gid in SAF_BASE_GIDS:
            y_router[i] = 0
        else:
            y_router[i] = 1

    np.save(PROCESSED_DIR / "meta_y_base.npy", y_base)
    np.save(PROCESSED_DIR / "meta_y_anom.npy", y_anom)
    np.save(PROCESSED_DIR / "meta_y_router.npy", y_router)
    print(f"y_base dagilimi: {dict(zip(*np.unique(y_base, return_counts=True)))}")
    print(f"y_anom dagilimi: {y_anom.sum(axis=0).tolist()}")
    print(f"y_router dagilimi: {dict(zip(*np.unique(y_router, return_counts=True)))}")

    # 9. base_meta egit (5-class XGB+LGB)
    print("\n" + "-"*60); print("BASE META-LEARNER (5-class)"); print("-"*60)
    X_tr, X_te, yb_tr, yb_te = train_test_split(
        meta_X, y_base, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_base)
    classes, counts = np.unique(yb_tr, return_counts=True)
    total = len(yb_tr)
    weights = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sw = np.array([weights[int(y)] for y in yb_tr])
    xgb_b = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, num_class=5, objective="multi:softprob",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_b.fit(X_tr, yb_tr, sample_weight=sw)
    lgb_b = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.7, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_b.fit(X_tr, yb_tr)
    proba = 0.5 * xgb_b.predict_proba(X_te) + 0.5 * lgb_b.predict_proba(X_te)
    pred = np.argmax(proba, axis=1)
    print(f"Ensemble: Acc={accuracy_score(yb_te, pred):.4f} F1={f1_score(yb_te, pred, average='weighted'):.4f}")
    print(classification_report(yb_te, pred, target_names=BASE_LABELS, digits=4, zero_division=0))
    joblib.dump({"xgb": xgb_b, "lgb": lgb_b}, META_MODELS_DIR / "base_meta.pkl")

    # 10. anom meta-learners (5+5 binary)
    print("\n" + "-"*60); print("ANOM META-LEARNERS (5 single + 5 multi)"); print("-"*60)
    anom_results = {}
    for j, anom_name in enumerate(ANOM_LABELS):
        y = y_anom[:, j]
        if y.sum() < 5:
            print(f"  [skip] {anom_name}: pos<5")
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            meta_X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        pos_w = max((y_tr == 0).sum() / max(y_tr.sum(), 1), 1.0)
        # Single meta = anom_meta (tek metada hem single hem multi egitilebilir, ayrildi)
        xgb_a = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_w, random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", verbosity=0)
        xgb_a.fit(X_tr, y_tr)
        lgb_a = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=63, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1)
        lgb_a.fit(X_tr, y_tr)
        p = 0.5 * xgb_a.predict_proba(X_te)[:, 1] + 0.5 * lgb_a.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        f1 = f1_score(y_te, pred, average='binary', zero_division=0)
        acc = accuracy_score(y_te, pred)
        print(f"  {anom_name:<22} F1={f1:.4f} Acc={acc:.4f}")
        joblib.dump({"xgb": xgb_a, "lgb": lgb_a}, META_MODELS_DIR / f"anom_{anom_name}.pkl")
        anom_results[anom_name] = {"f1": round(float(f1), 4), "acc": round(float(acc), 4)}

    # 11. Router (binary: single vs combo)
    print("\n" + "-"*60); print("ROUTER (single vs combo)"); print("-"*60)
    X_tr, X_te, yr_tr, yr_te = train_test_split(
        meta_X, y_router, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_router)
    pos_w = max((yr_tr == 0).sum() / max(yr_tr.sum(), 1), 1.0)
    xgb_r = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=pos_w, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="logloss", verbosity=0)
    xgb_r.fit(X_tr, yr_tr)
    lgb_r = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1)
    lgb_r.fit(X_tr, yr_tr)
    p = 0.5 * xgb_r.predict_proba(X_te)[:, 1] + 0.5 * lgb_r.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"Router F1={f1_score(yr_te, pred, average='binary'):.4f}")
    print(classification_report(yr_te, pred, target_names=["single", "combo"], digits=4, zero_division=0))
    joblib.dump({"xgb": xgb_r, "lgb": lgb_r}, META_MODELS_DIR / "router.pkl")

    # 12. Default blend params (sweep ile guncellenecek)
    from config import DEFAULT_ALPHA, DEFAULT_THR_ANOM_ONLY, DEFAULT_THR_COMBO
    blend = {
        "alpha": DEFAULT_ALPHA, "thr_anom_only": DEFAULT_THR_ANOM_ONLY,
        "thr_combo": DEFAULT_THR_COMBO,
        "thr_stat_hard": 0.80, "thr_router": 0.40,
    }
    joblib.dump(blend, META_MODELS_DIR / "blend_weights.pkl")

    summary = {"anom_results": anom_results, "n_meta_samples": n}
    with open(RESULTS_DIR / "meta_training.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMeta training tamamlandi.")


if __name__ == "__main__":
    main()
