"""
Ortak prediction logic.
Inference: gate hard + base meta + router + alpha-blended anom.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as sp_entropy

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ANOM_LABELS, BASE_LABELS, BASE_MODEL_NAMES,
    GATE_MODELS_DIR, BASE_MODELS_DIR, SINGLE_ANOM_MODELS_DIR,
    MULTI_ANOM_MODELS_DIR, META_MODELS_DIR, PROCESSED_DIR,
)
from lib import extract_batch


def compute_derived(P_gate, raw_base, raw_single, raw_multi):
    feats = []
    feats.append(float(P_gate))
    feats.append(float(np.max(raw_base)))
    feats.append(float(np.argmax(raw_base)))
    feats.append(float(np.max(raw_single)))
    feats.append(float(np.argmax(raw_single)))
    feats.append(float(np.max(raw_multi)))
    feats.append(float(np.argmax(raw_multi)))
    feats.append(float(np.sum(np.array(raw_single) > 0.5)))
    feats.append(float(np.sum(np.array(raw_multi) > 0.5)))
    sorted_base = sorted(raw_base, reverse=True)
    feats.append(float(sorted_base[0] - sorted_base[1]) if len(sorted_base) >= 2 else 0.0)
    diff = np.abs(np.array(raw_single) - np.array(raw_multi))
    feats.append(float(np.mean(diff)))
    feats.append(float(np.max(diff)))
    s_clip = np.clip(raw_single, 1e-6, 1 - 1e-6)
    feats.append(float(np.mean([sp_entropy([1 - x, x]) for x in s_clip])))
    m_clip = np.clip(raw_multi, 1e-6, 1 - 1e-6)
    feats.append(float(np.mean([sp_entropy([1 - x, x]) for x in m_clip])))
    return np.array(feats)


def load_all_models():
    """Tum modelleri yukle."""
    gate = joblib.load(GATE_MODELS_DIR / "stationarity_gate.pkl")
    base = {n: joblib.load(BASE_MODELS_DIR / f"{n}.pkl") for n in BASE_MODEL_NAMES}
    single = {n: joblib.load(SINGLE_ANOM_MODELS_DIR / f"{n}.pkl") for n in ANOM_LABELS}
    multi = {n: joblib.load(MULTI_ANOM_MODELS_DIR / f"{n}.pkl") for n in ANOM_LABELS}
    base_meta = joblib.load(META_MODELS_DIR / "base_meta.pkl")
    anom_meta = {a: joblib.load(META_MODELS_DIR / f"anom_{a}.pkl")
                  for a in ANOM_LABELS
                  if (META_MODELS_DIR / f"anom_{a}.pkl").exists()}
    router = joblib.load(META_MODELS_DIR / "router.pkl")
    blend = joblib.load(META_MODELS_DIR / "blend_weights.pkl")
    tsfresh_scaler = joblib.load(PROCESSED_DIR / "tsfresh_scaler.pkl")
    return {
        "gate": gate, "base": base, "single": single, "multi": multi,
        "base_meta": base_meta, "anom_meta": anom_meta, "router": router,
        "blend": blend, "tsfresh_scaler": tsfresh_scaler,
    }


def z_normalize_per_series(series_list):
    """Her seriyi kendi mean/std ile normalize et (training datasi z-normalized'di)."""
    out = []
    for s in series_list:
        mu = float(np.mean(s))
        sd = float(np.std(s))
        if sd < 1e-9:
            out.append(s - mu)
        else:
            out.append((s - mu) / sd)
    return out


def compute_features(series_list, models, n_jobs=4, z_norm=False):
    """Bir batch icin meta_X ve ham olasiliklari uret.
    z_norm=True: per-series z-normalization (sadece OOD test verisi icin: realdata)."""
    if z_norm:
        series_list = z_normalize_per_series(series_list)
    X_tsfresh = extract_batch(series_list, n_jobs=n_jobs)
    X_clean = np.nan_to_num(X_tsfresh, nan=0.0, posinf=0.0, neginf=0.0)

    def pp(bundle, X):
        return bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]

    n = len(series_list)
    P_gate = pp(models["gate"], X_clean)
    raw_base = np.zeros((n, 3))
    raw_single = np.zeros((n, 5))
    raw_multi = np.zeros((n, 5))
    for j, name in enumerate(BASE_MODEL_NAMES):
        raw_base[:, j] = pp(models["base"][name], X_clean)
    for j, name in enumerate(ANOM_LABELS):
        raw_single[:, j] = pp(models["single"][name], X_clean)
        raw_multi[:, j] = pp(models["multi"][name], X_clean)

    derived = np.zeros((n, 14))
    for i in range(n):
        derived[i] = compute_derived(P_gate[i], raw_base[i], raw_single[i], raw_multi[i])

    X_ts_scaled = models["tsfresh_scaler"].transform(X_clean)
    meta_X = np.hstack([
        P_gate.reshape(-1, 1), raw_base, raw_single, raw_multi, derived, X_ts_scaled,
    ])
    return {
        "meta_X": meta_X, "P_gate": P_gate,
        "raw_base": raw_base, "raw_single": raw_single, "raw_multi": raw_multi,
        "X_clean": X_clean,
    }


def predict_batch(feats, models, override_blend=None, anom_only_to_stat=False):
    """Karar mantigi: gate hard + base meta + router + alpha-blended anom.

    anom_only_to_stat=True: model 'anomaly_only' tahmini -> 'stationary' diye yorumla.
    (PDF taksonomisiyle uyum, sadece realdata icin trick)
    """
    blend = override_blend if override_blend else models["blend"]
    THR_STAT_HARD = blend.get("thr_stat_hard", 0.80)
    THR_BASE_META_STAT = blend.get("thr_base_meta_stat", 0.40)  # SOFT GATE: base_meta argmax stat
    THR_ANOM_FLIP = blend.get("thr_anom_flip", 0.50)  # anom_only flip thr
    THR_ROUTER = blend.get("thr_router", 0.40)
    alpha = blend["alpha"]
    thr_anom_only = blend["thr_anom_only"]
    thr_combo = blend["thr_combo"]

    meta_X = feats["meta_X"]
    P_gate = feats["P_gate"]

    # Base meta inference
    bm = models["base_meta"]
    proba_b = 0.5 * bm["xgb"].predict_proba(meta_X) + 0.5 * bm["lgb"].predict_proba(meta_X)
    base_pred = np.argmax(proba_b, axis=1)

    # Router inference
    rm = models["router"]
    p_combo = 0.5 * rm["xgb"].predict_proba(meta_X)[:, 1] + 0.5 * rm["lgb"].predict_proba(meta_X)[:, 1]

    # Anom meta inference
    anom_meta_probs = {}
    for a in ANOM_LABELS:
        if a not in models["anom_meta"]:
            continue
        am = models["anom_meta"][a]
        anom_meta_probs[a] = 0.5 * am["xgb"].predict_proba(meta_X)[:, 1] + 0.5 * am["lgb"].predict_proba(meta_X)[:, 1]

    results = []
    for i in range(meta_X.shape[0]):
        # ASAMA 1: HARD GATE
        if P_gate[i] >= THR_STAT_HARD:
            results.append({
                "base": "stationary", "anomalies": [],
                "path": "stat_gate",
                "P_gate": float(P_gate[i]),
                "base_probs": proba_b[i].tolist(),
                "p_combo": float(p_combo[i]),
            })
            continue

        base = BASE_LABELS[base_pred[i]]
        # Eger meta de stationary derse direkt return
        if base == "stationary":
            results.append({
                "base": "stationary", "anomalies": [],
                "path": "base_meta_stat",
                "P_gate": float(P_gate[i]),
                "base_probs": proba_b[i].tolist(),
                "p_combo": float(p_combo[i]),
            })
            continue

        # ASAMA 2: ROUTER
        is_combo = p_combo[i] >= THR_ROUTER
        if not is_combo and base in {"deterministic_trend", "stochastic_trend", "volatility"}:
            results.append({
                "base": base, "anomalies": [], "path": "single",
                "P_gate": float(P_gate[i]),
                "base_probs": proba_b[i].tolist(),
                "p_combo": float(p_combo[i]),
            })
            continue

        # ASAMA 3: ANOM BLEND
        anomalies = []
        blended_dict = {}
        # context threshold
        thr_dict = thr_anom_only if base == "anomaly_only" else thr_combo
        for a in ANOM_LABELS:
            # Single ve multi meta probs
            # anom_meta tek meta, hem single hem multi datayla egitilmis degil
            # iki ayri meta yapilmadi, anom_meta tek. Ham single/multi ensemble olasiligi
            # ile blend yapacagiz.
            j = ANOM_LABELS.index(a)
            p_single_raw = float(feats["raw_single"][i, j])
            p_multi_raw = float(feats["raw_multi"][i, j])
            p_meta = anom_meta_probs.get(a, np.zeros(meta_X.shape[0]))[i] if a in anom_meta_probs else 0.0
            # Alpha blend: single vs multi ham olasilik
            p_blended_raw = alpha[a] * p_single_raw + (1 - alpha[a]) * p_multi_raw
            # Meta ile birlestir (basit ortalama)
            p_final = 0.5 * float(p_meta) + 0.5 * p_blended_raw
            blended_dict[a] = round(float(p_final), 4)
            if p_final >= thr_dict[a]:
                anomalies.append(a)

        # Edge case: anomaly_only ama hicbir anomali tetiklenmedi
        if base == "anomaly_only" and not anomalies and blended_dict:
            top_a = max(blended_dict, key=blended_dict.get)
            anomalies = [top_a]

        results.append({
            "base": base, "anomalies": anomalies,
            "path": "combo" if is_combo else "single_forced_anom_only",
            "P_gate": float(P_gate[i]),
            "base_probs": proba_b[i].tolist(),
            "p_combo": float(p_combo[i]),
            "anom_blended": blended_dict,
        })

    # POST-PROCESS TRICK: anomaly_only → stationary (PDF taksonomi uyumu)
    # GT'leri de PDF'e uyarladigimiz icin model output da uyarlanmali
    if anom_only_to_stat:
        for r in results:
            if r["base"] == "anomaly_only":
                r["base"] = "stationary"
                r["path"] = r["path"] + "_trick"
    return results


def match_type(pred_base, pred_anoms, exp_base, exp_anoms):
    if exp_base is None:
        return "?"
    base_ok = (pred_base == exp_base)
    ps, es = set(pred_anoms), set(exp_anoms)
    if not es:
        if base_ok and not ps:
            return "FULL"
        elif base_ok:
            return "PARTIAL"
        return "NONE"
    all_in = es.issubset(ps)
    no_extra = (ps == es)
    if base_ok and all_in and no_extra:
        return "FULL"
    elif base_ok or all_in:
        return "PARTIAL"
    return "NONE"
