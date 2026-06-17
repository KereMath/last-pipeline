"""
Joint threshold sweep — 17 hyperparametre coordinate descent.

Parametreler:
- THR_STAT_HARD (1)
- THR_ROUTER (1)
- alpha[a] per-anomali (5)
- thr_anom_only[a] per-anomali (5)
- thr_combo[a] per-anomali (5)
= 17 toplam

Strateji: 3 iterasyon coordinate descent.
- Iter 1: THR_STAT_HARD grid (10) -> seç en iyi
- Iter 2: THR_ROUTER grid (9) -> seç
- Iter 3: per-anomali (alpha, thr_anom_only, thr_combo) sirali optimize
- Tekrar 2-3 iter

Score: Training set FULL count (29 grup x 20 ornek = 580 ornek)
"""
import copy
import json
import random
import sys
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANOM_LABELS, GROUPS, GROUP_LABELS, DATA_GEN_DIR,
    META_MODELS_DIR, RESULTS_DIR, DEFAULT_ALPHA, DEFAULT_THR_ANOM_ONLY, DEFAULT_THR_COMBO,
)
from lib import read_series, list_group_csvs
from predict import load_all_models, compute_features, predict_batch, match_type

N_PER_GROUP = 20
N_OUTER_ITER = 3


# Grid'ler (basit, eski hali — soft gate + flip rule kaldirildi)
THR_STAT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
THR_ROUTER_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ANOM_THR_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def collect_eval_set():
    random.seed(42)
    items, series = [], []
    for gid, name, _, _, _ in GROUPS:
        csvs = list_group_csvs(DATA_GEN_DIR, name)
        if not csvs:
            continue
        sampled = random.sample(csvs, min(N_PER_GROUP, len(csvs)))
        for c in sampled:
            s = read_series(c)
            if len(s) < 50:
                continue
            base_lbl, anoms = GROUP_LABELS[gid]
            items.append({"gid": gid, "exp_base": base_lbl, "exp_anoms": anoms})
            series.append(s)
    return items, series


def score(feats, models, blend, items):
    preds = predict_batch(feats, models, override_blend=blend)
    full = partial = none = 0
    for it, pr in zip(items, preds):
        mt = match_type(pr["base"], pr["anomalies"], it["exp_base"], it["exp_anoms"])
        if mt == "FULL": full += 1
        elif mt == "PARTIAL": partial += 1
        else: none += 1
    return full, partial, none


def main():
    print("=" * 60)
    print("THRESHOLD SWEEP (17 hyperparam, coordinate descent)")
    print("=" * 60)

    items, series = collect_eval_set()
    print(f"Eval set: {len(items)} ornek (29 grup x ~{N_PER_GROUP})")

    print("\nModel + feature cache...")
    models = load_all_models()
    feats = compute_features(series, models, n_jobs=4)

    # Baseline (default)
    blend = copy.deepcopy(models["blend"])
    full0, p0, n0 = score(feats, models, blend, items)
    print(f"\nBASELINE: FULL={full0} ({100*full0/len(items):.1f}%) PARTIAL={p0} NONE={n0}")

    history = []
    history.append({"iter": 0, "step": "baseline", "FULL": full0, "blend": copy.deepcopy(blend)})

    for outer in range(N_OUTER_ITER):
        print(f"\n{'='*40}\nOUTER ITER {outer+1}/{N_OUTER_ITER}\n{'='*40}")

        # === STEP 1: THR_STAT_HARD ===
        best_full = -1; best_t = blend["thr_stat_hard"]
        for t in THR_STAT_GRID:
            blend["thr_stat_hard"] = t
            f, _, _ = score(feats, models, blend, items)
            if f > best_full: best_full = f; best_t = t
        blend["thr_stat_hard"] = best_t
        print(f"[STAT_HARD] best={best_t:.2f} FULL={best_full}")
        history.append({"iter": outer+1, "step": "stat_hard", "best": best_t, "FULL": best_full})

        # === STEP 2: THR_ROUTER ===
        best_full = -1; best_t = blend["thr_router"]
        for t in THR_ROUTER_GRID:
            blend["thr_router"] = t
            f, _, _ = score(feats, models, blend, items)
            if f > best_full: best_full = f; best_t = t
        blend["thr_router"] = best_t
        print(f"[ROUTER] best={best_t:.2f} FULL={best_full}")
        history.append({"iter": outer+1, "step": "router", "best": best_t, "FULL": best_full})

        # === STEP 3: per-anomali (alpha, thr_anom_only, thr_combo) ===
        for anom in ANOM_LABELS:
            best_full = -1; best_cfg = None
            for a, thr_ao, thr_co in product(ALPHA_GRID, ANOM_THR_GRID, ANOM_THR_GRID):
                blend["alpha"][anom] = a
                blend["thr_anom_only"][anom] = thr_ao
                blend["thr_combo"][anom] = thr_co
                f, _, _ = score(feats, models, blend, items)
                if f > best_full: best_full = f; best_cfg = (a, thr_ao, thr_co)
            blend["alpha"][anom] = best_cfg[0]
            blend["thr_anom_only"][anom] = best_cfg[1]
            blend["thr_combo"][anom] = best_cfg[2]
            print(f"[{anom:<22}] alpha={best_cfg[0]:.2f} thr_ao={best_cfg[1]:.2f} thr_co={best_cfg[2]:.2f} FULL={best_full}")
            history.append({"iter": outer+1, "step": anom, "best_cfg": best_cfg, "FULL": best_full})

        # End of outer iter — final score
        ff, fp, fn = score(feats, models, blend, items)
        print(f"\n>>> ITER {outer+1} END: FULL={ff} ({100*ff/len(items):.1f}%) PART={fp} NONE={fn}")

    # Save final
    final_full, final_part, final_none = score(feats, models, blend, items)
    print(f"\n{'='*60}")
    print(f"FINAL: FULL={final_full} ({100*final_full/len(items):.1f}%) PART={final_part} NONE={final_none}")
    print(f"Baseline: {full0} -> Final: {final_full} ({final_full-full0:+d})")
    print(f"{'='*60}")

    joblib.dump(blend, META_MODELS_DIR / "blend_weights.pkl")  # OVERWRITE blend
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump({
            "baseline": {"FULL": full0, "PARTIAL": p0, "NONE": n0},
            "final": {"FULL": final_full, "PARTIAL": final_part, "NONE": final_none},
            "delta": final_full - full0,
            "blend": blend,
            "history": history,
        }, f, indent=2)
    print(f"\nKaydedildi: blend_weights.pkl (overwrite) + threshold_sweep.json")


if __name__ == "__main__":
    main()
