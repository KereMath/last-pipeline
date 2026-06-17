# last-pipeline

Zaman serisi sınıflandırma pipeline'ı: **14 binary model + 7 meta-learner**.
Hocaların verdiği `datasets.md` kaynaklarındaki **39 ground-truth dosya** üzerinde test edildi.

---

## 1. Pipeline Mimarisi

```
Raw time series (z-normalize edilmiş)
        │
        ▼
   tsfresh EfficientFC (777 feature)
        │
        ├─► [1] Stationarity Gate (1 binary)        → P(stat)
        │       └── Hard gate: P_stat ≥ THR_STAT
        │
        ├─► [2] Base Ensemble (3 binary, one-vs-rest)
        │       det_trend / stoch_trend / volatility
        │
        ├─► [3] Single-anom Ensemble (5 binary)
        │       collective / mean_shift / point / trend_shift / variance
        │       (sadece tek-anomali datasıyla eğitildi)
        │
        ├─► [4] Multi-anom Ensemble (5 binary)
        │       (multi-anomali datasıyla eğitildi)
        │
        ▼
   META-VEKTÖR (805 boyut):
       • 1 P_gate + 3 base + 5 single + 5 multi = 14 binary olasılık
       • 14 derived feature (agreement, entropy, gap, max/argmax)
       • 777 standardized tsfresh
        │
        ├─► [5] Base meta-learner (XGB+LGB, 5-class)
        │       stationary / det_trend / stoch_trend / volatility / anomaly_only
        │
        ├─► [6] Anom meta-learner (XGB+LGB, 5 binary)
        │       her anomali için cross-feature aware probability
        │
        ├─► [7] Router (XGB+LGB binary)
        │       single (saf base) vs combo (base + anomali)
        │
        ▼
   KARAR LOGIC:
       if P_gate >= THR_STAT:               → ("stationary", [])
       elif base_meta_argmax = stationary:  → ("stationary", [])
       elif router(P_combo) < THR_ROUTER:   → (base, [])
       else:                                → (base, blended anomalies)

   POST-PROCESS:
       if base == "anomaly_only":           → base'i "stationary" diye yorumla
                                              (PDF taksonomi uyumu)

   Çıktı: (base, anomalies)
```

---

## 2. Veri Taksonomisi (29 grup × 1000 = 29,000 sentetik seri)

**Base sınıfları (5):** `stationary`, `deterministic_trend`, `stochastic_trend`, `volatility`, `anomaly_only`
**Anomali sınıfları (5):** `collective_anomaly`, `mean_shift`, `point_anomaly`, `trend_shift`, `variance_shift`

### Grup yapısı

| # | Grup | n | Etiket |
|---|---|---|---|
| G01 | stationary | 1000 | (stationary, []) |
| G02-G04 | deterministic_trend / stochastic_trend / volatility | 3000 | (base, []) |
| G05-G09 | anomaly_only + tek anomali (col/mean/point/trend/var) | 5000 | (anomaly_only, [anom]) |
| G10-G14 | det_trend + 5 anomali | 5000 | (det_trend, [anom]) |
| G15-G19 | stoch_trend + 5 anomali | 5000 | (stoch_trend, [anom]) |
| G20-G24 | volatility + 5 anomali | 5000 | (volatility, [anom]) |
| G25-G29 | multi-anomali (col+mean, col+point, mean+var, point+var, point+trend) | 5000 | (anomaly_only, [anom1, anom2]) |
| **Toplam** | **29 grup** | **29,000** | |

**Üretim:** Betise library ile, her seri uzunluğu [80, 150] arası rastgele.

---

### ⚠️ Multi-anomali grupları — açık eksiklik

**Şu an:** Sadece 5 multi grup var (G25-G29). Halbuki 5 anomali için **C(5,2) = 10 ikili kombinasyon** mümkün.

**Bizdeki 5 (G25-G29):**
- col+mean, col+point, mean+variance, point+variance, point+trend_shift

**Eksik 5 ikili:**
- col+trend_shift, col+variance, mean+point, mean+trend_shift, trend_shift+variance

Bu eksiklik **bilerek değil** — başlangıç tasarımında 5 yaygın kombi seçtim ama 10'un hepsi de eğitime girmeliydi. Bunu ileride genişletmek için: `10_generate_data.py`'a G30-G34 grupları ekleyip yeniden eğitim. Ortalama tahmini ek iş: ~2 saat.

Önceki versiyonda (`contextual-off-10x` ts-proje'de) da aynı eksiklik vardı — bu da multi-anom modellerinin neden bazı kombinasyonlarda zayıf olduğunu (örn: trend_shift'in tek başına geçtiği serileri yakalayamıyor) açıklıyor.

---

## 3. Model Eğitim Sonuçları (14 Binary) — DETAYLI

Her binary model: 1000 pozitif + 1000 negatif balanced sample, LightGBM / XGBoost / MLP karşılaştırması, en yüksek validation F1 kazanır.

### Stationarity Gate
| Model | Best clf | Val F1 | Test F1 | Test Acc |
|---|---|---|---|---|
| **stationarity_gate** | XGBoost | 0.957 | **0.9677** | 0.9672 |

### Base Ensemble (3 binary, one-vs-rest)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **deterministic_trend** | LightGBM | **0.8950** | 0.8942 |
| **stochastic_trend** | XGBoost | **0.8519** | 0.8589 |
| **volatility** | XGBoost | **0.8241** | 0.8086 |

### Single-Anom Ensemble (5 binary, sadece tek-anomali datası)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **collective_anomaly** | LightGBM | **0.9900** | 0.9900 |
| **mean_shift** | LightGBM | **0.9212** | 0.9200 |
| **point_anomaly** | LightGBM | **1.0000** | 1.0000 |
| **trend_shift** | LightGBM | **0.9545** | 0.9550 |
| **variance_shift** | LightGBM | **0.9529** | 0.9525 |

### Multi-Anom Ensemble (5 binary, multi-anomali datası)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **collective_anomaly** | LightGBM | **0.9950** | 0.9950 |
| **mean_shift** | LightGBM | **0.9975** | 0.9975 |
| **point_anomaly** | XGBoost | **0.9751** | 0.9750 |
| **trend_shift** | LightGBM | **0.9900** | 0.9900 |
| **variance_shift** | LightGBM | **1.0000** | 1.0000 |

### Ortalama F1
- Gate: 0.97
- Base: 0.86
- Single-anom: 0.96
- Multi-anom: 0.99
- **14 binary ortalama: 0.94**

### Her modelin detaylı parametreleri

#### Ortak hyperparameter (tüm 14 binary model)

```python
# LightGBM
LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
               num_leaves=63, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced")

# XGBoost
XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
              subsample=0.8, colsample_bytree=0.8,
              scale_pos_weight=neg_count/pos_count,  # imbalance fix
              eval_metric="logloss")

# MLP
MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
              early_stopping=True, validation_fraction=0.1)
```

- Train/Val/Test split: 70%/10%/20% (stratified)
- Feature: tsfresh `EfficientFCParameters` → 777-boyutlu vektör
- Feature scaling: StandardScaler (per-model)
- Model selection: validation F1'i max veren clf seçilir
- Class weight: balanced (auto)

#### 1) Stationarity Gate (binary)
- **Pozitif:** G01 (saf stationary, 1000 örnek)
- **Negatif:** G02-G29 (28 grup, her birinden ~36 örnek = 1000 toplam balanced)
- **Total:** 1000 pos + 1000 neg = 2000 sample
- **Test F1:** 0.9677

#### 2-4) Base Ensemble (3 binary, one-vs-rest)

| Model | Pozitif gruplar | Negatif | Test F1 |
|---|---|---|---|
| `deterministic_trend` | G02 + G10-G14 (6 grup) | G01, G03-G09, G15-G29 (23 grup) | 0.895 |
| `stochastic_trend` | G03 + G15-G19 (6 grup) | diğerleri | 0.852 |
| `volatility` | G04 + G20-G24 (6 grup) | diğerleri | 0.824 |

#### 5-9) Single-Anom Ensemble (5 binary, **sadece tek-anomali datası**)

| Model | Pozitif gruplar | Test F1 |
|---|---|---|
| `collective_anomaly` | G05, G10, G15, G20 | 0.99 |
| `mean_shift` | G06, G11, G16, G21 | 0.92 |
| `point_anomaly` | G07, G12, G17, G22 | 1.00 |
| `trend_shift` | G08, G13, G18, G23 | 0.95 |
| `variance_shift` | G09, G14, G19, G24 | 0.95 |

- **Önemli:** Multi-anom (G25-G29) gruplarındaki örnekler bu modellerin eğitimine girmiyor.
- Bu sayede single-anom modelleri "saf tek-anomali" pattern'larını öğrenir.

#### 10-14) Multi-Anom Ensemble (5 binary, **sadece multi-anomali datası**)

| Model | Pozitif gruplar | Test F1 |
|---|---|---|
| `collective_anomaly` (multi) | G25, G26 | 0.995 |
| `mean_shift` (multi) | G25, G27 | 0.998 |
| `point_anomaly` (multi) | G26, G28, G29 | 0.975 |
| `trend_shift` (multi) | G29 | 0.99 |
| `variance_shift` (multi) | G27, G28 | 1.00 |

- **Negatif:** Multi-anom gruplarında o anomali geçmeyenler + diğer single-anom grupları (balance için)
- **Önemli:** Multi-anom modelleri "bir seri birden çok anomali içerdiğinde X anomalisi var mı?" sorusunu cevaplar.
- Single ve multi modelleri inference'da **alpha-weighted blend** edilir:
  `P_final[a] = α[a] × P_single[a] + (1-α[a]) × P_multi[a]`
- α[a] joint sweep ile bulunur (60_threshold_sweep.py)

---

## 4. Meta-Learner Sonuçları — DETAYLI

Meta veri: **29 grup × 80 örnek = 2,320 meta_X** (805-boyutlu)

### Base Meta-Learner (5-class XGB+LGB)
- Test Accuracy: **0.873**
- Test F1 (weighted): **0.873**

### Anomali Meta-Learners (5 binary XGB+LGB)
| Anomali | Test F1 | Test Acc |
|---|---|---|
| collective_anomaly | **0.9794** | 0.9914 |
| mean_shift | **0.8298** | 0.9310 |
| point_anomaly | **1.0000** | 1.0000 |
| trend_shift | **0.9494** | 0.9828 |
| variance_shift | **0.9184** | 0.9655 |

### Router (XGB+LGB binary, single vs combo)
- Test F1: **0.9616**, Accuracy: **0.9332**

### Meta-Vektör Yapısı (805 boyut)

```python
meta_X[i] = concat([
    P_gate[i],                # 1   (stationarity gate prob)
    raw_base[i],              # 3   (det_trend, stoch_trend, volatility)
    raw_single[i],            # 5   (5 single-anom prob)
    raw_multi[i],             # 5   (5 multi-anom prob)
    derived[i],               # 14  (türetilmiş feature)
    X_tsfresh_scaled[i]       # 777 (standardized tsfresh)
])  # toplam = 1 + 3 + 5 + 5 + 14 + 777 = 805
```

### 14 türetilmiş feature

1-4: `max_old_base, argmax_old_base, max_old_anom, n_old_anom_above_05`
5-8: `max_new_base, argmax_new_base, max_new_anom, n_new_anom_above_05`
9: `base_agreement` (old/new base argmax aynı mı? 0/1)
10: `base_confidence_gap` (new max - new 2nd max)
11: `anom_entropy` (5 anom prob ortalama binary entropy)
12: `anom_correlation` (old vs new anom corr)
13-14: `total_new_anom_signal, total_old_anom_signal`

Bu feature'lar meta-learner'a "ensemble'lar uyuşuyor mu, ne kadar güvende, hangisi baskın" bilgisini verir.

### Hyperparameter (base meta + anom meta + router)

```python
# Base meta (5-class XGB)
XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
              min_child_weight=3, gamma=0.1,
              subsample=0.8, colsample_bytree=0.7,
              reg_alpha=0.1, reg_lambda=1.0,
              num_class=5, objective="multi:softprob")

# Anom meta (binary XGB)
XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
              scale_pos_weight=neg/pos)

# Router (binary XGB)
XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
              scale_pos_weight=neg/pos)
```

LGB analog hyperparameter ile, ensemble = `0.5 × XGB + 0.5 × LGB`.

### Decision Logic (Inference)

```python
def predict(series):
    # 1. tsfresh extraction (z-norm sonra)
    X_ts = tsfresh_extract(znorm(series))           # 777-d

    # 2. 14 binary inference
    P_gate     = gate(X_ts)                          # 1
    raw_base   = [base_det(X_ts), base_stoch, base_vol]   # 3
    raw_single = [single_a(X_ts) for a in 5_anoms]   # 5
    raw_multi  = [multi_a(X_ts)  for a in 5_anoms]   # 5

    # 3. Meta vektör
    derived = compute_derived(...)                    # 14
    meta_X  = concat([P_gate, raw_base, raw_single, raw_multi, derived, X_ts_scaled])

    # 4. Karar
    if P_gate >= THR_STAT:                            # default 0.08 (realdata için)
        return ("stationary", [])

    base_probs = base_meta(meta_X)                    # 5-class
    base = argmax(base_probs)
    if base == "stationary":
        return ("stationary", [])

    p_combo = router(meta_X)
    if p_combo < THR_ROUTER:                          # default 0.3
        return (base, [])

    anomalies = []
    for a in 5_anoms:
        p_meta  = anom_meta_a(meta_X)
        p_blend_raw = α[a] × raw_single[a] + (1-α[a]) × raw_multi[a]
        p_final = 0.5 × p_meta + 0.5 × p_blend_raw    # meta + raw blend
        thr = THR_ANOM_ONLY[a] if base=="anomaly_only" else THR_COMBO[a]
        if p_final >= thr:
            anomalies.append(a)

    if base == "anomaly_only":
        base = "stationary"                           # post-process trick

    return (base, anomalies)
```

---

## 5. 29-Grup Eğitim Seti Eval (samples_per_group=20 × 29 = 580)

**Sonuç:** FULL = **428/580 (%73.79)**, PARTIAL = 127, NONE = 25

### Grup başına FULL accuracy

| GID | Grup | FULL/total | % |
|---|---|---|---|
| G01 | stationary | 20/20 | **100%** |
| G02 | deterministic_trend | 16/20 | 80% |
| G03 | stochastic_trend | 9/20 | 45% |
| G04 | volatility | 13/20 | 65% |
| G05 | anom_only + collective | 0/20 | 0% |
| G06 | anom_only + mean | 16/20 | 80% |
| G07 | anom_only + point | 8/20 | 40% |
| G08 | anom_only + trend_shift | 3/20 | 15% |
| G09 | anom_only + variance | 15/20 | 75% |
| G10 | det + collective | 0/20 | 0% |
| G11 | det + mean | 13/20 | 65% |
| G12 | det + point | 0/20 | 0% |
| G13 | det + trend_shift | 8/20 | 40% |
| G14 | det + variance | 18/20 | 90% |
| G15 | stoch + collective | 0/20 | 0% |
| G16 | stoch + mean | 12/20 | 60% |
| G17 | stoch + point | 1/20 | 5% |
| G18 | stoch + trend_shift | 5/20 | 25% |
| G19 | stoch + variance | 17/20 | 85% |
| G20 | vol + collective | 12/20 | 60% |
| G21 | vol + mean | 16/20 | 80% |
| G22 | vol + point | 14/20 | 70% |
| G23 | vol + trend_shift | 9/20 | 45% |
| G24 | vol + variance | 17/20 | 85% |
| G25 | multi col+mean | 16/20 | 80% |
| G26 | multi col+point | 20/20 | **100%** |
| G27 | multi mean+variance | 16/20 | 80% |
| G28 | multi point+variance | 16/20 | 80% |
| G29 | multi point+trend | 19/20 | **95%** |

---

## 6. Realdata Eval (Datasets.md kaynakları, 39 GT'li dosya)

**Final sonuç:** **FULL = 11/39, PARTIAL = 22, NONE = 6, base_accuracy = 32/39 (%82)**

### FULL match alan 11 dosya

| # | Dosya | Kaynak | GT | Pipeline tahmini |
|---|---|---|---|---|
| 1 | **strikes.csv** | Brockwell | stationary | stationary ✓ |
| 2 | **sunspots.csv** | Brockwell | stationary | stationary ✓ |
| 3 | **soi_dataframe.csv** | Shumway | stationary | stationary ✓ |
| 4 | **rec_dataframe.csv** | Shumway | stationary | stationary ✓ |
| 5 | **US_investment.csv** | JMulTi | stationary | stationary ✓ |
| 6 | **np_unemploy.csv** | Z&A (Nelson-Plosser) | stationary | stationary ✓ |
| 7 | **INDPRO.csv** | Shumway | stochastic_trend | stochastic_trend ✓ |
| 8 | **UNRATE.csv** | Shumway | stochastic_trend | stochastic_trend ✓ |
| 9 | **airpass.csv** | Brockwell | stochastic_trend | stochastic_trend ✓ |
| 10 | **German_consumption.csv** | JMulTi | stochastic_trend | stochastic_trend ✓ |
| 11 | **W6.csv** | Wei | det_trend + variance_shift | det_trend + variance_shift ✓ |

### PARTIAL kalan 22 dosya (base ✓ ama anomali yanlış / kaçırılmış)

| Dosya | Kaynak | GT | Pred | Eksiklik |
|---|---|---|---|---|
| W1.csv | Wei | stationary + [point] | stationary + [] | point sinyali yok (eğitimde çoklu point yok) |
| W2.csv | Wei | stationary + [variance] | stationary + [mean] | mean tetiklendi, variance kaçırıldı |
| W3.csv | Wei | stationary + [variance] | stationary + [] | variance sinyali yetersiz |
| GermanGNP.csv | JMulTi | det_trend + [trend_shift] | det_trend + [variance] | trend yerine variance |
| Polish_productivity.csv | JMulTi | stoch_trend + [trend_shift] | stoch_trend + [] | trend_shift kaçırıldı |
| np_velocity.csv | Z&A | stoch_trend + [] | stoch_trend + [mean] | yanlış anomali ekledi |
| np_interest.csv | Z&A | stoch_trend + [] | stoch_trend + [variance] | yanlış anomali |
| np_cpi.csv, np_employmt, np_gnpdefl, np_gnpperca, np_indprod, np_M, np_nomgnp, np_realgnp, np_realwag, np_sp500, np_wages | Z&A | stoch_trend + [trend_shift] | stoch_trend + [] | trend_shift kaçırıldı (10 dosya) |
| tcpd_nile.csv | TCPD | stationary + [mean_shift] | stationary + [] | mean_shift kaçırıldı |
| tcpd_ozone.csv | TCPD | stoch_trend + [trend_shift] | stoch_trend + [mean] | trend yerine mean |
| tcpd_robocalls.csv | TCPD | stoch_trend + [mean_shift] | stoch_trend + [] | mean_shift kaçırıldı |
| tcpd_seatbelts.csv | TCPD | stoch_trend + [mean_shift] | stationary + [mean_shift] | base yanlış, anom doğru |

### NONE 6 dosya (base/anom çoğunlukla yanlış)

| Dosya | Kaynak | GT | Pred | Sebep |
|---|---|---|---|---|
| deaths.csv | Brockwell | stoch_trend + [] | stationary + [] | gate false positive |
| uspop.csv | Brockwell | det_trend + [] | stoch_trend + [var] | base + anom yanlış |
| W5.csv | Wei | det_trend + [mean] | stoch_trend + [] | base + anom yanlış |
| RealInt_dataframe.csv | Bai-Perron | stationary + [mean] | stoch_trend + [var] | base + anom yanlış |
| tcpd_debt_ireland.csv | TCPD | det_trend + [trend_shift] | stationary + [] | base yanlış (n=21 kısa) |
| tcpd_lga_passengers.csv | TCPD | stoch_trend + [mean_shift] | det_trend + [] | base + anom yanlış |

### Asıl sınır

**12 Nelson-Plosser serisinde** model base'i (stochastic_trend) doğru tahmin ediyor AMA `trend_shift` anomalisini kaçırıyor. Sebep: Nelson-Plosser serileri Great Depression (1929) gibi uzun-vadeli ekonomik break içeriyor. Bizim eğitim setimizdeki `trend_shift` örnekleri kısa-vadeli `direction_change` pattern'ları — NP'nin gradual macroeconomic shift'i farklı bir dağılım.

W1, W2, W3 için benzer durum: **W1 4 outlier** (eğitimimizde tek-spike pattern var), **W2/W3 variance heterogeneity** (eğitimde scale_factor=1.5 ile sınırlı).

Yeni veri eklenmeden bu sınır aşılamaz.

---

## 7. Realdata Dosya Listesi (39 dosya, datasets.md kaynakları)

### Wei (Time Series Analysis 4ed) — 5 dosya (W4, W7, W10 kaynakta yok)
- W1, W2, W3, W5, W6

### JMulTi (Lütkepohl, Applied Time Series Econometrics) — 4 dosya
- US_investment, German_consumption, Polish_productivity, GermanGNP

### Shumway & Stoffer — 4 dosya
- rec_dataframe (Recruitment), INDPRO (FRB Production), UNRATE (Unemployment), soi_dataframe

### Brockwell & Davis — 5 dosya
- uspop, strikes, sunspots, deaths, airpass

### Bai-Perron — 1 dosya
- RealInt_dataframe (US ex-post real interest rate)

### TCPD (Turing Change Point Dataset) — 6 dosya
- tcpd_nile, tcpd_seatbelts, tcpd_lga_passengers, tcpd_debt_ireland, tcpd_ozone, tcpd_robocalls

### Nelson-Plosser (Zivot-Andrews 1992) — 14 dosya
- np_cpi, np_employmt, np_gnpdefl, np_nomgnp, np_interest, np_indprod, np_gnpperca, np_realgnp, np_wages, np_realwag, np_sp500, np_unemploy, np_velocity, np_M

**Toplam: 39 GT'li + 1 GT'siz (W10) = 40 dosya**

---

## 8. Reproducibility

```bash
# Bağımlılıklar
pip install -r requirements.txt

# 1. 29,000 sentetik veri (38 grup × 1000)
python runner/10_generate_data.py

# 2. Stationarity gate (1 binary)
python runner/20_train_gate.py

# 3. 3 base binary
python runner/30_train_base.py

# 4. 5 single-anom binary
python runner/31_train_single_anom.py

# 5. 5 multi-anom binary
python runner/32_train_multi_anom.py

# 6. Meta-learners (base + anom + router)
python runner/40_train_meta.py

# 7. 29-grup eğitim seti eval
python runner/50_eval_train.py

# 8. Realdata eval (39 GT'li)
python runner/51_eval_realdata.py

# 9. (İsteğe bağlı) joint threshold sweep
python runner/60_threshold_sweep.py
```

Toplam süre CPU'da: ~1.5-2 saat (en uzun kısım: veri üretim + tsfresh extraction).

---

## 9. Klasör Yapısı

```
last-pipeline/
├── README.md                       (bu dosya)
├── PLAN.md                         (mimari plan)
├── datasets.md                     (hocanın verdiği dataset listesi)
├── config.py                       (29 grup taksonomi + threshold defaults + REAL_GT)
├── lib.py                          (ortak fonksiyonlar)
├── predict.py                      (inference logic + karar mantığı)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── realdata/                   (40 ground-truth dosya, datasets.md uyumlu)
│   └── generated/                  (29,000 sentetik — gitignored, script ile üretilir)
│
├── models/
│   ├── gate/stationarity_gate.pkl
│   ├── base/                       (3 binary)
│   ├── single_anom/                (5 binary)
│   └── multi_anom/                 (5 binary)
│
├── meta/
│   ├── meta_models/
│   │   ├── base_meta.pkl           (5-class softmax)
│   │   ├── anom_<5>.pkl            (5 binary)
│   │   ├── router.pkl              (binary)
│   │   └── blend_weights.pkl       (threshold + alpha config)
│   └── processed_data/             (meta_X.npy — gitignored)
│
├── results/
│   ├── gate_training.json
│   ├── base_training.json
│   ├── single_anom_training.json
│   ├── multi_anom_training.json
│   ├── meta_training.json
│   ├── eval_train.json             (580 örnek, %73.79 FULL)
│   ├── eval_realdata.json          (39 GT'li, 11 FULL)
│   └── threshold_sweep.json
│
└── runner/                         (9 final script)
    ├── 10_generate_data.py
    ├── 20_train_gate.py
    ├── 30_train_base.py
    ├── 31_train_single_anom.py
    ├── 32_train_multi_anom.py
    ├── 40_train_meta.py
    ├── 50_eval_train.py
    ├── 51_eval_realdata.py
    └── 60_threshold_sweep.py
```

---

## 10. Özet Tablosu — En Önemli Metrikler

| Metric | Değer |
|---|---|
| Toplam binary model | **14** (1 gate + 3 base + 5 single + 5 multi) |
| Meta-learner | **7** (1 base 5-class + 5 anom + 1 router) |
| Eğitim verisi | 29,000 sentetik seri |
| Test verisi (realdata) | 39 GT'li dosya (datasets.md kaynakları) |
| Ortalama binary F1 | **0.94** |
| Gate F1 | **0.97** |
| Base meta accuracy | **0.873** |
| 29-grup eğitim FULL | **73.79%** |
| **Realdata FULL** | **11/39 (28%)** |
| **Realdata base accuracy** | **32/39 (82%)** |
| Realdata PARTIAL | 22/39 (base doğru ama anomali yanlış) |
| Realdata NONE | 6/39 |
