"""
Validasyon: klasik test sinyalleri kacirdigimiz realdata vakalarinda VAR MI?
- ADF, KPSS (stationarity), Zivot-Andrews (yapisal kirilmali birim kok -> trend_shift),
  ruptures (change-point -> mean/variance/trend shift)
- Soru: ZA, kacirdigimiz NP trend_shift'leri yakaliyor mu? ruptures kacan mean/var'i?
Retrain YOK — sadece sinyal var mi diye bakar (+ bedava baseline).
"""
import sys, warnings, json
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_REAL_DIR, REAL_GT
from lib import read_series
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
import ruptures as rpt

def znorm(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean())/s if s > 1e-9 else x - x.mean()

def classical(x):
    x = np.asarray(x, float); n = len(x)
    out = {}
    try: out["adf_p"] = adfuller(x, autolag="AIC")[1]
    except Exception: out["adf_p"] = np.nan
    try: out["kpss_p"] = kpss(x, regression="ct", nlags="auto")[1]
    except Exception: out["kpss_p"] = np.nan
    try:
        za = zivot_andrews(x, regression="ct", autolag="AIC")
        out["za_p"] = float(za[1]); out["za_break"] = int(za[4])/n
    except Exception:
        out["za_p"] = np.nan; out["za_break"] = np.nan
    try:
        z = znorm(x)
        bks = rpt.Pelt(model="rbf", min_size=max(5, n//20)).fit(z).predict(pen=2*np.log(n))
        out["rpt_n"] = len(bks) - 1  # son nokta endpoint
    except Exception:
        out["rpt_n"] = -1
    return out

d = json.load(open(Path(__file__).resolve().parent.parent/"results"/"eval_realdata.json"))
match = {r["csv"]: (r["match"], r["pred_anoms"]) for r in d["rows"]}

rows = []
for c in sorted(DATA_REAL_DIR.glob("*.csv")):
    if c.name not in REAL_GT: continue
    s = read_series(c)
    if len(s) < 20: continue
    gt_b, gt_a = REAL_GT[c.name]
    cl = classical(s)
    m, pa = match.get(c.name, ("?", []))
    rows.append((c.name, len(s), gt_b, gt_a, m, pa, cl))

print(f"{'file':<24}{'n':>4} {'ADFp':>6}{'KPSSp':>6}{'ZAp':>6}{'ZAbrk':>6}{'rptN':>5}  {'match':<8}GT_anom")
for nm,n,gb,ga,m,pa,cl in rows:
    print(f"{nm:<24}{n:>4} {cl['adf_p']:>6.2f}{cl['kpss_p']:>6.2f}{cl['za_p']:>6.2f}{cl['za_break']:>6.2f}{cl['rpt_n']:>5}  {m:<8}{ga}")

# === GO/NO-GO: kacirdigimiz vakalarda sinyal var mi ===
print("\n=== ZA, kacirdigimiz NP trend_shift'leri yakaliyor mu? ===")
np_trend = [(nm,cl,m,pa) for nm,n,gb,ga,m,pa,cl in rows if nm.startswith("np_") and "trend_shift" in ga]
za_hit = sum(1 for nm,cl,m,pa in np_trend if cl["za_p"]<0.05)
print(f"  NP trend_shift dosyasi: {len(np_trend)} | ZA significant (p<0.05): {za_hit} | bizim yakaladigimiz: {sum(1 for _,_,_,pa in np_trend if 'trend_shift' in pa)}")

print("\n=== ruptures, kacirdigimiz mean/variance shift'leri yakaliyor mu? ===")
for anom in ["mean_shift","variance_shift"]:
    miss = [(nm,cl) for nm,n,gb,ga,m,pa,cl in rows if anom in ga and anom not in pa]
    hit = sum(1 for nm,cl in miss if cl["rpt_n"]>=1)
    print(f"  {anom}: kacirdigimiz {len(miss)} dosya | ruptures break buluyor: {hit}  ({[nm for nm,cl in miss]})")

# === Bedava baseline: ADF+KPSS stationarity ===
print("\n=== BASELINE: ADF+KPSS ile stationary/degil (gate baseline) ===")
ok=tot=0
for nm,n,gb,ga,m,pa,cl in rows:
    pred_stat = (cl["adf_p"]<0.05) and (cl["kpss_p"]>0.05)
    true_stat = (gb=="stationary")
    tot+=1; ok+= (pred_stat==true_stat)
print(f"  ADF+KPSS stationary accuracy: {ok}/{tot} = {100*ok/tot:.0f}%  (bizim gate akisi ~karsilastirma)")
