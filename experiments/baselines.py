"""
Tez baseline'lari: klasik yontemler tek basina vs bizim pipeline (39 realdata GT).
- Stationarity ekseni: ADF+KPSS -> stationary mi?  (gate/base baseline)
- Anomali varligi ekseni: ruptures change-point -> anomali var mi? (anom baseline)
Sonuc -> results/baselines.json
"""
import sys, warnings, json
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_REAL_DIR, REAL_GT, RESULTS_DIR
from lib import read_series, _znorm
from statsmodels.tsa.stattools import adfuller, kpss
import ruptures as rpt

rows = []
for c in sorted(DATA_REAL_DIR.glob("*.csv")):
    if c.name not in REAL_GT:
        continue
    s = read_series(c)
    if len(s) < 20:
        continue
    x = _znorm(s); n = len(x)
    gt_b, gt_a = REAL_GT[c.name]
    try: adf_p = adfuller(x, autolag="AIC")[1]
    except Exception: adf_p = 1.0
    try: kpss_p = kpss(x, regression="ct", nlags="auto")[1]
    except Exception: kpss_p = 0.1
    try:
        bks = rpt.Pelt(model="rbf", min_size=max(5, n // 20)).fit(x).predict(pen=2 * np.log(n))
        n_bk = len([b for b in bks if 0 < b < n])
    except Exception:
        n_bk = 0
    rows.append({
        "csv": c.name, "gt_base": gt_b, "gt_has_anom": bool(gt_a),
        "adf_p": adf_p, "kpss_p": kpss_p, "n_breaks": n_bk,
        "pred_stationary": bool(adf_p < 0.05 and kpss_p > 0.05),
        "pred_has_anom": bool(n_bk >= 1),
    })

# Stationarity baseline
stat_ok = sum(1 for r in rows if r["pred_stationary"] == (r["gt_base"] == "stationary"))
# Anomali varligi baseline (precision/recall/F1)
tp = sum(1 for r in rows if r["pred_has_anom"] and r["gt_has_anom"])
fp = sum(1 for r in rows if r["pred_has_anom"] and not r["gt_has_anom"])
fn = sum(1 for r in rows if not r["pred_has_anom"] and r["gt_has_anom"])
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

summary = {
    "n": len(rows),
    "adf_kpss_stationarity_acc": round(stat_ok / len(rows), 3),
    "ruptures_anom_presence": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                                "tp": tp, "fp": fp, "fn": fn},
    "rows": rows,
}
print(f"ADF+KPSS stationarity accuracy : {stat_ok}/{len(rows)} = {100*stat_ok/len(rows):.0f}%")
print(f"  (pipeline base_accuracy referans: 33/39 = 85%)")
print(f"ruptures anomali-varligi       : P={prec:.2f} R={rec:.2f} F1={f1:.2f} (tp={tp} fp={fp} fn={fn})")
with open(RESULTS_DIR / "baselines.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nKaydedildi: {RESULTS_DIR/'baselines.json'}")
