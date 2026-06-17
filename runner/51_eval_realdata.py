"""
realdata eval (newplanlast yeni taksonomide).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_REAL_DIR, RESULTS_DIR, REAL_GT
from lib import read_series
from predict import load_all_models, compute_features, predict_batch, match_type


def main():
    print("REALDATA EVAL")
    print("=" * 60)

    items, series_list = [], []
    for c in sorted(DATA_REAL_DIR.glob("*.csv")):
        s = read_series(c)
        if len(s) >= 20:
            gt = REAL_GT.get(c.name)
            exp_base, exp_anoms = gt if gt else (None, None)
            items.append({
                "csv": c.name, "n": int(len(s)),
                "exp_base": exp_base, "exp_anoms": exp_anoms or [],
            })
            series_list.append(s)
    print(f"Toplam realdata: {len(items)} dosya")

    models = load_all_models()
    feats = compute_features(series_list, models, n_jobs=4, z_norm=True)  # OOD fix
    # Realdata icin ozel blend: gate threshold 0.95 -> 0.10 (z-norm sonrasi cogu stat seri 0.1-0.93 araliginda)
    import copy
    real_blend = copy.deepcopy(models["blend"])
    real_blend["thr_stat_hard"] = 0.08
    preds = predict_batch(feats, models, override_blend=real_blend, anom_only_to_stat=True)

    full = partial = none = base_ok = 0
    rows = []
    for it, pr in zip(items, preds):
        mt = match_type(pr["base"], pr["anomalies"], it["exp_base"], it["exp_anoms"])
        if it["exp_base"] and pr["base"] == it["exp_base"]:
            base_ok += 1
        if mt == "FULL": full += 1
        elif mt == "PARTIAL": partial += 1
        elif mt == "NONE": none += 1
        rows.append({
            **it,
            "pred_base": pr["base"],
            "pred_anoms": pr["anomalies"],
            "path": pr["path"],
            "match": mt,
            "P_gate": round(pr["P_gate"], 4),
            "p_combo": round(pr["p_combo"], 4),
        })

    print(f"\n{'csv':<30} {'n':>4} {'GT':<22} {'PRED':<22} {'ANOM':<35} {'match'}")
    print("-" * 130)
    for r in sorted(rows, key=lambda x: x["n"]):
        gt = f"{r['exp_base']}+{r['exp_anoms']}" if r['exp_base'] else "-"
        pred = f"{r['pred_base']}"
        anom = ", ".join(r['pred_anoms']) if r['pred_anoms'] else "-"
        print(f"{r['csv']:<30} {r['n']:>4} {gt[:22]:<22} {pred:<22} {anom[:35]:<35} {r['match']}")

    gt_count = sum(1 for r in rows if r["exp_base"])
    print(f"\nGT'li: {gt_count} | base_OK: {base_ok}/{gt_count} | FULL: {full} | PART: {partial} | NONE: {none}")

    summary = {
        "total": len(items), "with_gt": gt_count,
        "FULL": full, "PARTIAL": partial, "NONE": none, "base_ok": base_ok,
        "rows": rows,
    }
    with open(RESULTS_DIR / "eval_realdata.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
