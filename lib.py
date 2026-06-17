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


# ---- Klasik ekonometrik test feature'lari (hibrit: tsfresh + klasik) ----
# Validasyon (experiments/classical_signal_check.py): ruptures kacirdigimiz 5/5
# mean_shift'i yakaliyor; ADF+KPSS stationarity %87 baseline. ZA NP'de zayif (3/11)
# ama teorik olarak dogru arac -> dahil edildi.
CLASSICAL_FEATURE_NAMES = [
    "adf_stat", "adf_p", "kpss_stat", "kpss_p",
    "za_stat", "za_p", "za_break_loc",
    "rpt_n_breaks", "rpt_max_mean_jump", "rpt_max_var_ratio", "rpt_break_loc",
    "var_2nd_over_1st",
]


def _znorm(s: np.ndarray) -> np.ndarray:
    x = np.asarray(s, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x - x.mean()


def classical_features(s: np.ndarray) -> np.ndarray:
    """ADF + KPSS + Zivot-Andrews + ruptures -> 12 skaler (olcek-bagimsiz, z-norm iceride).
    Sira CLASSICAL_FEATURE_NAMES ile ayni. Hata durumunda notr default."""
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
    import ruptures as rpt

    x = _znorm(s); n = len(x)
    f = {k: 0.0 for k in CLASSICAL_FEATURE_NAMES}
    f["adf_p"] = 1.0; f["kpss_p"] = 0.1; f["za_p"] = 1.0
    f["za_break_loc"] = 0.5; f["rpt_max_var_ratio"] = 1.0
    f["rpt_break_loc"] = 0.5; f["var_2nd_over_1st"] = 1.0
    if n < 12:
        return np.array([f[k] for k in CLASSICAL_FEATURE_NAMES])

    try:
        a = adfuller(x, autolag="AIC"); f["adf_stat"], f["adf_p"] = float(a[0]), float(a[1])
    except Exception: pass
    try:
        k = kpss(x, regression="ct", nlags="auto"); f["kpss_stat"], f["kpss_p"] = float(k[0]), float(k[1])
    except Exception: pass
    try:
        za = zivot_andrews(x, regression="ct", autolag=None, maxlag=2)  # hizli mod
        f["za_stat"], f["za_p"], f["za_break_loc"] = float(za[0]), float(za[1]), float(za[4]) / n
    except Exception: pass
    try:
        bks = rpt.Pelt(model="rbf", min_size=max(5, n // 20)).fit(x).predict(pen=2 * np.log(n))
        cps = [b for b in bks if 0 < b < n]
        f["rpt_n_breaks"] = float(len(cps))
        bounds = [0] + cps + [n]
        segs = [x[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        means = [seg.mean() for seg in segs if len(seg) > 0]
        vars = [seg.var() + 1e-9 for seg in segs if len(seg) > 0]
        if len(means) >= 2:
            f["rpt_max_mean_jump"] = float(max(abs(means[i + 1] - means[i]) for i in range(len(means) - 1)))
            f["rpt_max_var_ratio"] = float(max(vars) / min(vars))
        if cps:
            # en buyuk mean-jump break'inin konumu
            jumps = [abs(means[i + 1] - means[i]) for i in range(len(means) - 1)]
            f["rpt_break_loc"] = float(cps[int(np.argmax(jumps))]) / n if jumps else cps[0] / n
    except Exception: pass
    try:
        h = n // 2
        f["var_2nd_over_1st"] = float((x[h:].var() + 1e-9) / (x[:h].var() + 1e-9))
    except Exception: pass

    return np.array([f[k] for k in CLASSICAL_FEATURE_NAMES])


def extract_batch(series_list: List[np.ndarray], n_jobs: int = 4) -> np.ndarray:
    """Raw single-view tsfresh (777). FINAL pipeline bunu kullanir.

    NOT — denenen 2 feature genislemesi de realdata'yi iyilestirmedi, geri alindi:
    1. dual-view (raw + detrend/deseason residual): base ayrimini DUSURDU
       (LDA 0.57->0.47, realdata base_ok 32->29). Bkz. experiments/diag_views.py.
    2. hibrit (raw + classical_features ADF/KPSS/ZA/ruptures): NOTR/biraz kotu
       (base_ok 33->32); validasyonda ruptures sinyali vardi ama sentetik->gercek
       domain gap nedeniyle uctan uca transfer olmadi. classical_features() asagida
       baseline/teshis icin korunuyor ama extract_batch'e dahil DEGIL."""
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
