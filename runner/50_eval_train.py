"""
29-grup egitim seti uzerinde eval (samples_per_leaf=20).
"""
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GROUPS, GROUP_LABELS, GROUP_NAMES, DATA_GEN_DIR, RESULTS_DIR
from lib import read_series, list_group_csvs
from predict import load_all_models, compute_features, predict_batch, match_type

N_PER_GROUP = 20


def main():
    random.seed(42)
    print("=" * 60)
    print("29-GRUP EGITIM SETI EVAL")
    print("=" * 60)

    # Sampling
    items, series_list = [], []
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
            items.append({
                "gid": gid, "group": name, "csv": c.name,
                "exp_base": base_lbl, "exp_anoms": anoms,
            })
            series_list.append(s)
    print(f"Toplam: {len(items)} seri")

    print("\nModeller yukleniyor + inference...")
    models = load_all_models()
    feats = compute_features(series_list, models, n_jobs=4)
    preds = predict_batch(feats, models)
    print(f"Inference tamam.")

    # Match
    per_group = {}
    full = partial = none = 0
    for it, pr in zip(items, preds):
        mt = match_type(pr["base"], pr["anomalies"], it["exp_base"], it["exp_anoms"])
        g = it["gid"]
        per_group.setdefault(g, {"name": it["group"], "FULL": 0, "PARTIAL": 0, "NONE": 0, "total": 0})
        per_group[g][mt] += 1
        per_group[g]["total"] += 1
        if mt == "FULL": full += 1
        elif mt == "PARTIAL": partial += 1
        else: none += 1

    print(f"\n{'='*60}")
    print(f"GENEL: FULL={full} ({100*full/len(items):.1f}%)  PARTIAL={partial}  NONE={none}")
    print(f"{'='*60}")
    for gid in sorted(per_group.keys()):
        p = per_group[gid]
        pct = 100 * p["FULL"] / max(p["total"], 1)
        print(f"  G{gid:02d} ({p['name'][:32]:<32}) n={p['total']:>3} FULL={p['FULL']:>3} ({pct:5.1f}%) P={p['PARTIAL']:>2} N={p['NONE']:>2}")

    summary = {
        "total": len(items), "FULL": full, "PARTIAL": partial, "NONE": none,
        "full_pct": round(100 * full / len(items), 2),
        "per_group": {str(k): v for k, v in per_group.items()},
    }
    with open(RESULTS_DIR / "eval_train.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON: {RESULTS_DIR / 'eval_train.json'}")


if __name__ == "__main__":
    main()
