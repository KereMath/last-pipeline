"""
newplanlast — Ortak yardimcilar
"""
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore")


def read_series(csv_path: Path) -> np.ndarray:
    """CSV'den seri oku."""
    try:
        df = pd.read_csv(csv_path)
        for col in ("data", "value", "values", "y"):
            if col in df.columns:
                return df[col].dropna().values.astype(float)
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) > 0:
            return df[num_cols[0]].dropna().values.astype(float)
    except Exception:
        pass
    return np.array([])


def extract_batch(series_list: List[np.ndarray], n_jobs: int = 4) -> np.ndarray:
    """Tek tsfresh extraction batch'i."""
    dfs = []
    for i, s in enumerate(series_list):
        dfs.append(pd.DataFrame({"id": i, "time": np.arange(len(s)), "value": s.astype(float)}))
    combined = pd.concat(dfs, ignore_index=True)
    X_df = tsfresh_extract(
        combined, column_id="id", column_sort="time", column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True, n_jobs=n_jobs,
    )
    impute(X_df)
    return X_df.values


def list_group_csvs(data_gen_dir: Path, group_name: str) -> List[Path]:
    """Bir grup klasorunden tum CSV'leri listele."""
    root = data_gen_dir / group_name
    return sorted([f for f in root.glob("*.csv") if f.name != "metadata.csv"])


def sample_balanced(
    pos_groups: set,
    neg_groups: set,
    group_names: Dict[int, str],
    data_gen_dir: Path,
    n_per_side: int = 1000,
    seed: int = 42,
) -> Tuple[List[Path], List[int]]:
    """
    Balanced sample: n_per_side pozitif + n_per_side negatif.
    Pozitif/negatif gruplar arasinda esit dagilim.
    """
    import random
    random.seed(seed)

    # Pozitif: pos_groups'tan total n_per_side topla
    n_pos_per_group = max(1, n_per_side // len(pos_groups))
    pos_csvs = []
    for gid in sorted(pos_groups):
        csvs = list_group_csvs(data_gen_dir, group_names[gid])
        if csvs:
            chosen = random.sample(csvs, min(n_pos_per_group, len(csvs)))
            pos_csvs.extend(chosen)
    if len(pos_csvs) > n_per_side:
        pos_csvs = random.sample(pos_csvs, n_per_side)

    # Negatif: neg_groups'tan total n_per_side topla
    n_neg_per_group = max(1, n_per_side // len(neg_groups))
    neg_csvs = []
    for gid in sorted(neg_groups):
        csvs = list_group_csvs(data_gen_dir, group_names[gid])
        if csvs:
            chosen = random.sample(csvs, min(n_neg_per_group, len(csvs)))
            neg_csvs.extend(chosen)
    if len(neg_csvs) > n_per_side:
        neg_csvs = random.sample(neg_csvs, n_per_side)

    all_csvs = pos_csvs + neg_csvs
    labels = [1] * len(pos_csvs) + [0] * len(neg_csvs)
    # Karistir
    combined = list(zip(all_csvs, labels))
    random.shuffle(combined)
    return [c for c, _ in combined], [l for _, l in combined]


def load_series_batch(csv_paths: List[Path], min_len: int = 50) -> Tuple[List[np.ndarray], List[int]]:
    """CSV path listesini seri listesine donustur; min_len'in altindaki seriler atilir."""
    series = []
    keep_indices = []
    for i, p in enumerate(csv_paths):
        s = read_series(p)
        if len(s) >= min_len:
            series.append(s)
            keep_indices.append(i)
    return series, keep_indices


def train_binary(
    X: np.ndarray, y: np.ndarray, model_name: str,
    test_size: float = 0.2, random_state: int = 42,
) -> Dict:
    """
    Bir binary model egit (LGBM + XGB + MLP, en iyi val F1).
    Donus: {best_clf, model, scaler, val_f1, test_f1, test_acc}
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    import xgboost as xgb

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    pos = int(y.sum()); neg = int((y == 0).sum())
    print(f"  [{model_name}] Veri: {X.shape}, pos={pos}, neg={neg}")

    val_ratio = 0.1 / (1 - test_size)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=random_state, stratify=y_tmp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    pos_w = neg / pos if pos > 0 else 1.0

    classifiers = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_w,
            random_state=random_state, n_jobs=-1, eval_metric="logloss", verbosity=0,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=random_state,
        ),
    }

    val_scores = {}
    for cn, clf in classifiers.items():
        clf.fit(X_tr_s, y_tr)
        f1 = f1_score(y_val, clf.predict(X_val_s), average="binary", zero_division=0)
        val_scores[cn] = (clf, f1)
        print(f"     {cn:<10} Val F1={f1:.4f}")

    best_name = max(val_scores, key=lambda k: val_scores[k][1])
    best_clf = val_scores[best_name][0]
    pred_te = best_clf.predict(X_te_s)
    test_f1 = f1_score(y_te, pred_te, average="binary", zero_division=0)
    test_acc = accuracy_score(y_te, pred_te)
    print(f"  >>> En iyi: {best_name} | Test F1={test_f1:.4f} Acc={test_acc:.4f}")

    return {
        "model_name": model_name,
        "best_clf": best_name,
        "model": best_clf,
        "scaler": scaler,
        "val_scores": {k: v[1] for k, v in val_scores.items()},
        "test_f1": round(float(test_f1), 4),
        "test_acc": round(float(test_acc), 4),
        "pos": pos, "neg": neg, "n_features": int(X.shape[1]),
    }


def save_model(out_path: Path, result: Dict):
    """Model + scaler joblib ile kaydet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": result["model"],
        "scaler": result["scaler"],
        "best_clf": result["best_clf"],
        "test_f1": result["test_f1"],
        "test_acc": result["test_acc"],
    }, out_path)
    print(f"  Kaydedildi: {out_path}")


def load_model(path: Path) -> Dict:
    return joblib.load(path)
