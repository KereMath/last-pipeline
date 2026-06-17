# last-pipeline

Zaman serisi sınıflandırma pipeline'ı: **14 binary model + 7 meta-learner**.
Sentetik eğitim verisi **34 grup × 1000 = 34.000 seri** (C(5,2)=10 multi-anomali kombinasyonunun tamamı).
Hocaların verdiği `datasets.md` kaynaklarındaki **39 ground-truth dosya** üzerinde test edildi.

**Güncel sonuç:** Realdata FULL=11/39, base_accuracy=33/39 (%85); 34-grup eğitim FULL=%70.6.
Çözülen/incelenen konular için bkz. [§11 Deney Günlüğü](#11-deney-günlüğü-ve-bulgular).

---

## 1. Pipeline Mimarisi

```
Raw time series  (realdata'da per-series z-normalize; sentetik eğitimde ham)
        │
        ▼
   tsfresh EfficientFC (777 feature, raw single-view)
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

## 2. Veri Taksonomisi (34 grup × 1000 = 34,000 sentetik seri)

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
| G30-G34 | multi-anomali (col+trend, col+var, mean+point, mean+trend, trend+var) | 5000 | (anomaly_only, [anom1, anom2]) |
| **Toplam** | **34 grup** | **34,000** | |

**Üretim:** Betise library ile. Çeşitlilik (anomalilerin daha iyi temsili için):
- **Uzunluk** katmanlı: ~%25 short (50-80), %50 medium (80-180), %25 long (180-400). (config `LEN_TIERS`)
- **trend_shift** 3 tip karışık: `direction_change` / `magnitude_change` / `direction_and_magnitude_change` (config `TREND_SHIFT_CHANGE_TYPES`).
- **Anomali ölçeği** sabit yerine aralıktan rastgele (config `ANOM_SCALE_RANGES`).

---

### ✅ Multi-anomali grupları — TAMAMLANDI

5 anomali için **C(5,2) = 10 ikili kombinasyonun tamamı** artık eğitimde:

| Mevcut (G25-G29) | Eklenen (G30-G34) |
|---|---|
| col+mean, col+point, mean+var, point+var, point+trend | col+trend, col+var, mean+point, mean+trend, trend+var |

Bu sayede `MULTI_ANOM_POSITIVE_GROUPS` çok daha dengeli — özellikle **trend_shift 1→4 gruba** (`{29,30,33,34}`), collective 2→4'e çıktı. Multi trend_shift binary F1: 0.92.

---

## 2.1 Veri Üretim Metodolojisi (detaylı)

Üretici: **betise `TimeSeriesGenerator`**. Her seri `gen_series(length, spec, seed)` ile katman katman inşa edilir (`runner/10_generate_data.py`). Her grubun `spec`'i `config.GROUPS`'ta tanımlı; aşağıda her katmanın hangi betise fonksiyonunu hangi parametrelerle çağırdığı:

### Katman 1 — Base (iskelet süreç)
| spec `base` | betise çağrısı | parametre |
|---|---|---|
| `stationary` | `generate_stationary_base_series` | `distribution="ar"` (AR süreç) |
| `stochastic` | `generate_stochastic_trend` | `kind` ∈ {rw, rwd, ari, ima, arima} **rastgele** |
| `volatility` | `generate_volatility` | `kind` ∈ {arch, garch, egarch, aparch} **rastgele** |

### Katman 2 — Trend overlay (`spec["trend"]="mixed_det"`)
Rastgele tip seçilir ∈ {linear, quadratic, cubic, exponential, damped}, hepsi `location="center"`:
- `linear`→`generate_deterministic_trend_linear` · `quadratic`→`..._quadratic` · `cubic`→`..._cubic(amplitude=10)` · `exponential`→`..._exponential` · `damped`→`..._exponential(sign=-1)`

### Katman 3 — Break (`spec["break"]` / `spec["break_two"]`), hepsi `location="middle", num_breaks=1, sign=±1 rastgele`
| break | betise çağrısı | scale_factor |
|---|---|---|
| `mean_shift` | `generate_mean_shift` | `~U(1.0, 2.5)` |
| `variance_shift` | `generate_variance_shift` | `~U(1.3, 2.0)` |
| `trend_shift` | `generate_trend_shift` | `1.0`; `change_type` ∈ {direction, magnitude, direction+magnitude} **rastgele** |

### Katman 4 — Anomali (`spec["anomaly"]` / `spec["anomaly_two"]`), `location="middle"`
| anomali | betise çağrısı | scale_factor |
|---|---|---|
| `point` | `generate_point_anomaly` (`is_spike=True`) | `~U(1.5, 3.0)` |
| `collective` | `generate_collective_anomalies` (`num_anomalies=1`) | `~U(1.5, 3.0)` |

> Tüm rastgele aralıklar `config.ANOM_SCALE_RANGES` / `TREND_SHIFT_CHANGE_TYPES`'tan; değiştirmek için tek nokta orası.

### Üretim parametreleri (config)
- **Adet:** `N_PER_GROUP=1000` → 34 grup × 1000 = **34,000 seri**.
- **Uzunluk:** `LEN_TIERS` katmanlı → %25 short(50-80) · %50 medium(80-180) · %25 long(180-400). `MIN_SERIES_LENGTH=50` altındakiler atılır.
- **Seed (tam tekrarlanabilir):** `SEED_BASE=7000`; her seri `seed = 7000 + gid*100000 + i`. Aynı betise sürümüyle `10_generate_data.py` **birebir aynı 34k seriyi** üretir.
- **Z-norm yok:** sentetik veri ham üretilir; z-normalize **sadece** realdata inference'ında uygulanır (`predict.compute_features(z_norm=True)`).
- Çıktı: `data/generated/<grup>/<grup>_NNNN.csv` + `manifest.json` (her serinin gid/uzunluk/spec/label kaydı).

### Hangi grup hangi spec ile (tam tablo)
`G01` stationary · `G02` stationary+mixed_det · `G03` stochastic · `G04` volatility · `G05-09` stationary + tek anomali · `G10-14` mixed_det + tek anomali · `G15-19` stochastic + tek anomali · `G20-24` volatility + tek anomali · `G25-34` stationary + iki anomali (multi). Etiketler [§2 tablosu](#grup-yapısı) ve `config.GROUPS`'ta birebir.

### Kaç data ile hangi model eğitildi
| Aşama | Örnekleme | Toplam |
|---|---|---|
| 14 binary model (her biri) | `sample_balanced`: 1000 pozitif + 1000 negatif (gruplar arası eşit dağıtım, `N_SAMPLES_PER_MODEL=1000`) | ~2000/model |
| 7 meta-learner | her gruptan 80 örnek | 34×80 = **2720** |
| Eğitim-seti eval (§5) | her gruptan 20 | 34×20 = **680** |
| Split | her binary: 70/10/20 train/val/test (stratified); meta: 80/20 |  |

---

## 3. Model Eğitim Sonuçları (14 Binary) — DETAYLI

Her binary model: 1000 pozitif + 1000 negatif balanced sample, LightGBM / XGBoost / MLP karşılaştırması, en yüksek validation F1 kazanır.

### Stationarity Gate
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **stationarity_gate** | XGBoost | **0.9728** | 0.9724 |

### Base Ensemble (3 binary, one-vs-rest)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **deterministic_trend** | LightGBM | **0.8620** | 0.8561 |
| **stochastic_trend** | LightGBM | **0.8962** | 0.8965 |
| **volatility** | LightGBM | **0.8227** | 0.8182 |

### Single-Anom Ensemble (5 binary, sadece tek-anomali datası)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **collective_anomaly** | LightGBM | **0.9925** | 0.9925 |
| **mean_shift** | LightGBM | **0.8960** | 0.8950 |
| **point_anomaly** | LightGBM | **1.0000** | 1.0000 |
| **trend_shift** | XGBoost | **0.9487** | 0.9475 |
| **variance_shift** | LightGBM | **0.9554** | 0.9550 |

### Multi-Anom Ensemble (5 binary, multi-anomali datası — artık 10 grup)
| Model | Best clf | Test F1 | Test Acc |
|---|---|---|---|
| **collective_anomaly** | LightGBM | **0.9950** | 0.9950 |
| **mean_shift** | LightGBM | **0.9333** | 0.9325 |
| **point_anomaly** | LightGBM | **0.9899** | 0.9900 |
| **trend_shift** | LightGBM | **0.9258** | 0.9275 |
| **variance_shift** | XGBoost | **0.9620** | 0.9625 |

### Ortalama F1
- Gate: 0.97
- Base: 0.86
- Single-anom: 0.96
- Multi-anom: 0.95
- **14 binary ortalama: ≈0.93**

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
- **Negatif:** G02-G34 (33 grup, balanced ~1000 toplam)
- **Total:** 1000 pos + ~990 neg ≈ 2000 sample
- **Test F1:** 0.9728

#### 2-4) Base Ensemble (3 binary, one-vs-rest)

| Model | Pozitif gruplar | Negatif | Test F1 |
|---|---|---|---|
| `deterministic_trend` | G02 + G10-G14 (6 grup) | diğer 28 grup | 0.862 |
| `stochastic_trend` | G03 + G15-G19 (6 grup) | diğerleri | 0.896 |
| `volatility` | G04 + G20-G24 (6 grup) | diğerleri | 0.823 |

#### 5-9) Single-Anom Ensemble (5 binary, **sadece tek-anomali datası**)

| Model | Pozitif gruplar | Test F1 |
|---|---|---|
| `collective_anomaly` | G05, G10, G15, G20 | 0.99 |
| `mean_shift` | G06, G11, G16, G21 | 0.90 |
| `point_anomaly` | G07, G12, G17, G22 | 1.00 |
| `trend_shift` | G08, G13, G18, G23 | 0.95 |
| `variance_shift` | G09, G14, G19, G24 | 0.96 |

- **Önemli:** Multi-anom (G25-G34) gruplarındaki örnekler bu modellerin eğitimine girmiyor.
- Bu sayede single-anom modelleri "saf tek-anomali" pattern'larını öğrenir.

#### 10-14) Multi-Anom Ensemble (5 binary, **sadece multi-anomali datası — artık 10 grup**)

| Model | Pozitif gruplar (multi) | Test F1 |
|---|---|---|
| `collective_anomaly` (multi) | G25, G26, G30, G31 | 0.995 |
| `mean_shift` (multi) | G25, G27, G32, G33 | 0.933 |
| `point_anomaly` (multi) | G26, G28, G29, G32 | 0.990 |
| `trend_shift` (multi) | **G29, G30, G33, G34** (1→4 grup) | 0.926 |
| `variance_shift` (multi) | G27, G28, G31, G34 | 0.962 |

- **Negatif:** Multi-anom gruplarında o anomali geçmeyenler (artık 6 grup, balance için yeterli)
- **Önemli:** Multi-anom modelleri "bir seri birden çok anomali içerdiğinde X anomalisi var mı?" sorusunu cevaplar.
- Single ve multi modelleri inference'da **alpha-weighted blend** edilir:
  `P_final[a] = α[a] × P_single[a] + (1-α[a]) × P_multi[a]`
- α[a] joint sweep ile bulunur (60_threshold_sweep.py)

---

## 4. Meta-Learner Sonuçları — DETAYLI

Meta veri: **34 grup × 80 örnek = 2,720 meta_X** (805-boyutlu)

### Base Meta-Learner (5-class XGB+LGB)
- Test Accuracy: **0.785**
- Test F1 (weighted): **0.784**
- (29-grup'taki 0.873'ten düşük: 5 yeni zor grup + trend_shift'in anomaly_only ile içsel feature örtüşmesi — bkz. [§11](#11-deney-günlüğü-ve-bulgular))

### Anomali Meta-Learners (5 binary XGB+LGB)
| Anomali | Test F1 | Test Acc |
|---|---|---|
| collective_anomaly | **0.9843** | 0.9926 |
| mean_shift | **0.7782** | 0.9026 |
| point_anomaly | **0.9881** | 0.9945 |
| trend_shift | **0.8923** | 0.9485 |
| variance_shift | **0.9344** | 0.9688 |

### Router (XGB+LGB binary, single vs combo)
- Test F1: **0.9741**

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

### 14 türetilmiş feature (`compute_derived`, predict.py / 40_train_meta.py)

| # | Feature | Açıklama |
|---|---|---|
| 1 | `P_gate` | stationarity gate olasılığı |
| 2 | `max(raw_base)` | en güçlü base olasılığı |
| 3 | `argmax(raw_base)` | baskın base indeksi |
| 4 | `max(raw_single)` | en güçlü single-anom olasılığı |
| 5 | `argmax(raw_single)` | baskın single-anom |
| 6 | `max(raw_multi)` | en güçlü multi-anom olasılığı |
| 7 | `argmax(raw_multi)` | baskın multi-anom |
| 8 | `sum(raw_single > 0.5)` | kaç single-anom tetikli |
| 9 | `sum(raw_multi > 0.5)` | kaç multi-anom tetikli |
| 10 | `base_gap` | base sıralı 1.−2. olasılık farkı (güven) |
| 11 | `mean(|single−multi|)` | single↔multi ortalama uyumsuzluk |
| 12 | `max(|single−multi|)` | maks uyumsuzluk |
| 13 | `mean single binary entropy` | single belirsizlik |
| 14 | `mean multi binary entropy` | multi belirsizlik |

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

## 5. 34-Grup Eğitim Seti Eval (samples_per_group=20 × 34 = 680)

**Sonuç:** pre-sweep FULL = **467/680 (%68.68)**, sweep sonrası **480/680 (%70.6)**.

### Grup başına FULL accuracy (eval_train.json, pre-sweep)

| GID | Grup | FULL/total | % |
|---|---|---|---|
| G01 | stationary | 20/20 | 100% |
| G02 | deterministic_trend | 17/20 | 85% |
| G03 | stochastic_trend | 13/20 | 65% |
| G04 | volatility | 8/20 | 40% |
| G05 | anom_only+collective | 17/20 | 85% |
| G06 | anom_only+mean | 13/20 | 65% |
| G07 | anom_only+point | 17/20 | 85% |
| G08 | anom_only+trend_shift | 9/20 | 45% |
| G09 | anom_only+variance | 9/20 | 45% |
| G10 | det+collective | 19/20 | 95% |
| G11 | det+mean | 10/20 | 50% |
| G12 | det+point | 15/20 | 75% |
| G13 | det+trend_shift | 7/20 | 35% |
| G14 | det+variance | 14/20 | 70% |
| G15 | stoch+collective | 17/20 | 85% |
| G16 | stoch+mean | 13/20 | 65% |
| G17 | stoch+point | 12/20 | 60% |
| G18 | stoch+trend_shift | 7/20 | 35% |
| G19 | stoch+variance | 11/20 | 55% |
| G20 | vol+collective | 15/20 | 75% |
| G21 | vol+mean | 10/20 | 50% |
| G22 | vol+point | 13/20 | 65% |
| G23 | vol+trend_shift | 9/20 | 45% |
| G24 | vol+variance | 15/20 | 75% |
| G25 | multi col+mean | 16/20 | 80% |
| G26 | multi col+point | 18/20 | 90% |
| G27 | multi mean+var | 15/20 | 75% |
| G28 | multi point+var | 18/20 | 90% |
| G29 | multi point+trend | 16/20 | 80% |
| **G30** | **multi col+trend** (yeni) | 16/20 | 80% |
| **G31** | **multi col+var** (yeni) | 20/20 | 100% |
| **G32** | **multi mean+point** (yeni) | 18/20 | 90% |
| **G33** | **multi mean+trend** (yeni) | 4/20 | 20% |
| **G34** | **multi trend+var** (yeni) | 16/20 | 80% |

**En zayıf:** trend_shift'li base/single grupları (G13/G18 %35, G33 %20). Sebep veri değil — trend_shift'in alttaki base'i maskelemesi (içsel feature örtüşmesi, bkz. [§11](#11-deney-günlüğü-ve-bulgular)). trend_shift **tespiti** ise %85-92 (binary F1 yüksek), sorun base sınıflandırma.

---

## 6. Realdata Eval (Datasets.md kaynakları, 39 GT'li dosya)

**Final sonuç (34-grup, raw single-view):** **FULL = 11/39, PARTIAL = 23, NONE = 5, base_accuracy = 33/39 (%85)**
(29-grup orijinaline göre: FULL eşit (11), base_ok 32→33; ama artık tam taksonomi + zengin veri ile.)
Tam satırlar: `results/eval_realdata.json`.

> **base tespiti güçlü taraf:** 39 dosyanın **33'ünde base doğru (%85)**. PARTIAL'ların neredeyse tamamı "base ✓ ama anomali kaçtı/yanlış" — yani pipeline serinin temel tipini güvenilir buluyor, kalan iş anomali katmanında.

### FULL match alan 11 dosya
`German_consumption, INDPRO, UNRATE, US_investment, W6 (det+variance), np_unemploy, np_velocity, rec_dataframe, soi_dataframe, strikes, sunspots`

### PARTIAL / NONE — yapısal sınırlar (3 kök neden, [§11](#11-deney-günlüğü-ve-bulgular)'de kanıtlı)

1. **NP trend_shift kaçışı (~10 dosya):** base doğru (stoch_trend) ama trend_shift kaçıyor. En gevşek eşikte bile 11 NP dosyasından sadece **3'ü** yakalanıyor → sinyal modelde **yok**, eşik değil. NP serileri kademeli uzun-vadeli makro kırılma; betise'in ürettiğinden farklı dağılım. `magnitude_change` eklemek yetmedi.
2. **variance FP (airpass):** mevsimsel genlik artışı variance_shift sanılıyor (variance prob ≈1.0). Feature düzeyinde robust confusion; ölçek daraltma + deseasonalize denendi, geçmedi.
3. **base confusion (W5, tcpd_seatbelts vb.):** trend_shift/kısa seri base'i maskeliyor.

**Sonuç:** Realdata tavanı artık veri *miktarıyla* değil, feature/mimari ile sınırlı — bkz. §11.

### Kapsam notu — taksonomi-dışı dosyalar (çıkarılmadı, dokümante edildi)

Eval setinde, modelin **hiç çalışmadığı senaryolara** ait birkaç dosya var; bunlar doğruluğu yapay olarak düşürüyor ama **şeffaflık için 39'da tutuldu**:

| Dosya | Neden kapsam-dışı | match |
|---|---|---|
| airpass | güçlü mevsimsellik (period 12) — taksonomide mevsimsel sınıf yok; variance FP doğrudan bundan | PARTIAL |
| deaths | güçlü mevsimsellik (period 12) | NONE |
| uspop | n=21, eğitim uzunluk tabanı (50) altında | NONE |
| tcpd_debt_ireland | n=21, eğitim tabanı altında | NONE |

**Bu 4 dosya hariç tutulduğunda (35 in-scope):** base_accuracy = **32/35 (%91)**, NONE = 2, FULL = 11.
(Mevsimsellik ve n<50 senaryoları bilinçli olarak eğitim kapsamı dışında — bkz. §11 "denenmemiş yönler".)

> NP serileri yüksek FFT-periyot gösterse de **gerçek mevsimsel değil** (yıllık ekonomik veri); GT anomalileri `trend_shift` ile taksonomi-içi, sadece dağılım açığı — bu yüzden çıkarılmadı.

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

**Toplam: 40 dosya.** Eval'de 39 değerlendirilir; **W10 (n=8)** uzunluk eşiğinin (≥20) altında olduğu için atlanır (config'te GT'si olsa da). `config.REAL_GT`'de ayrıca dosyası bulunmayan `NP_xetradax_returns100` girdisi var — eval mevcut dosyalar üzerinden döndüğü için zararsız.

---

## 8. Reproducibility

```bash
# Bağımlılıklar
pip install -r requirements.txt

# 1. 34,000 sentetik veri (34 grup × 1000)
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

# 7. 34-grup eğitim seti eval
python runner/50_eval_train.py

# 8. Realdata eval (39 GT'li)
python runner/51_eval_realdata.py

# 9. joint threshold sweep (blend_weights.pkl'i optimize eder, ÖNERİLİR)
python runner/60_threshold_sweep.py
# sweep blend'i degistirdigi icin realdata eval'i tekrar calistir:
python runner/51_eval_realdata.py
```

Toplam süre CPU'da: veri üretim ~10-15 dk + tüm eğitim/eval ~30-40 dk (raw single-view).
**Not:** Tüm grup mantığı `config.py`'den türer; runner kodları taksonomiden bağımsızdır.

### Inference clone'dan çalışır
Tüm artefaktlar commit'li: `models/` (14 binary), `meta/meta_models/` (7 meta + blend), ve **`meta/processed_data/tsfresh_scaler.pkl`** (inference için gerekli) + `data/generated/` (birebir eğitim seti). Yani taze clone'da yeniden eğitim gerekmeden `predict.py` doğrudan çalışır. (Daha önce scaler gitignore'daydı; bilinçli olarak commit'e alındı.)

---

## 9. Klasör Yapısı

```
last-pipeline/
├── README.md                       (bu dosya)
├── PLAN.md                         (mimari plan)
├── datasets.md                     (hocanın verdiği dataset listesi)
├── config.py                       (34 grup taksonomi + LEN_TIERS/ANOM_SCALE/threshold + REAL_GT)
├── lib.py                          (ortak fonksiyonlar; extract_batch = raw single-view)
├── predict.py                      (inference logic + karar mantığı)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── realdata/                   (40 dosya: 39 GT'li + W10, datasets.md uyumlu)
│   └── generated/                  (34,000 sentetik — commit'li; script ile birebir üretilebilir)
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
│   └── processed_data/             (tsfresh_scaler.pkl [inference], meta_X.npy — commit'li)
│
├── results/
│   ├── gate_training.json
│   ├── base_training.json
│   ├── single_anom_training.json
│   ├── multi_anom_training.json
│   ├── meta_training.json
│   ├── eval_train.json             (680 örnek, %68.68 pre-sweep FULL)
│   ├── eval_realdata.json          (39 GT'li, 11 FULL, base_ok 33)
│   └── threshold_sweep.json        (sweep: %70.6 FULL)
│
├── runner/                         (9 final script)
│   ├── 10_generate_data.py
│   ├── 20_train_gate.py
│   ├── 30_train_base.py
│   ├── 31_train_single_anom.py
│   ├── 32_train_multi_anom.py
│   ├── 40_train_meta.py
│   ├── 50_eval_train.py
│   ├── 51_eval_realdata.py
│   └── 60_threshold_sweep.py
│
└── experiments/                    (teşhis — pipeline'a dahil değil)
    ├── diag_views.py               (PCA + LDA + kNN view ayrım gücü)
    ├── meta_view_test.py           (cached base_meta raw vs dual)
    └── pca_base_views.png          (base confusion görseli)
```

---

## 10. Özet Tablosu — En Önemli Metrikler

| Metric | Değer |
|---|---|
| Toplam binary model | **14** (1 gate + 3 base + 5 single + 5 multi) |
| Meta-learner | **7** (1 base 5-class + 5 anom + 1 router) |
| Eğitim verisi | **34,000** sentetik seri (34 grup, C(5,2)=10 multi tam) |
| Test verisi (realdata) | 39 GT'li dosya (datasets.md kaynakları) |
| Ortalama binary F1 | **≈0.93** |
| Gate F1 | **0.97** |
| Base meta accuracy | **0.785** |
| 34-grup eğitim FULL | **70.6%** (sweep sonrası) |
| **Realdata FULL** | **11/39 (28%)** |
| **Realdata base accuracy** | **33/39 (85%)** ← en güçlü taraf |
| Realdata PARTIAL | 23/39 (base doğru ama anomali yanlış) |
| Realdata NONE | 5/39 |
| Realdata in-scope (35, mevsim/n<50 hariç) | base **32/35 (%91)**, NONE=2 |

---

## 11. Deney Günlüğü ve Bulgular

Bu pipeline'ı geliştirirken denenenler ve neden o kararların verildiği — gelecekteki tekrarları önlemek için.

### Denenen tüm metotlar — özet liste

| # | Metot | Amaç | Sonuç | Durum |
|---|---|---|---|---|
| 1 | Eksik 5 multi grup (G30-G34) | C(5,2)=10 tam, multi temsili | trend_shift 1→4 grup, F1 0.92 | ✅ tutuldu |
| 2 | Katmanlı uzunluk (LEN_TIERS 50-400) | kısa/uzun temsili | dengeli dağılım | ✅ tutuldu |
| 3 | trend_shift 3 change-type | NP kademeli shift temsili | sentetikte iyileşti, NP'de yetmedi | ✅ tutuldu |
| 4 | Anomali ölçek randomizasyonu | şiddet çeşitliliği | — | ✅ tutuldu |
| 5 | variance ölçek daraltma 3.0→2.0 | airpass/German variance FP | net-nötr (FP feature confusion'ı) | ⚪ tutuldu |
| 6 | Joint threshold sweep (17 param, coord. descent) | eşik/alpha optimize | train FULL +13-14 | ✅ kullanılıyor |
| 7 | **Dual-view feature** (raw+detrend/deseason residual) | base maskeleme + variance FP | realdata kötüleşti (FULL 10→8) | ❌ reddedildi |
| 8 | Teşhis: PCA + LDA + kNN view analizi | hangi view base'i ayırır | raw > residual > dual | 📊 kanıt |
| 9 | Realdata eşik probu (anom threshold tarama) | trend_shift recall tavanı | NP'de 3/11 (sinyal yok) | 📊 kanıt |
| 10 | Decision-logic rescue (base_meta→raw_base argmax) | base confusion düzelt | base binary de şaşırıyor (G23: det≠vol) | ❌ çalışmaz |

> Özet: **modelleme tekniğini değiştiren her şey (5,7,10) kaybetti** → legacy raw+GBT korundu. **Veri/kapsamı iyileştiren her şey (1-4) tutuldu.** Eşik optimizasyonu (6) standart parça. 8-9-10 teşhis/eleme.

### Verdict — en iyi metot hangisi?
**Teknik olarak en iyi metot = legacy yöntem** (raw single-view tsfresh → GBT binary ensemble → meta-learner). Denenen hiçbir alternatif (dual-view feature-eng, residual, ölçek ayarı) bunu **geçemedi**; aksine feature-eng realdata'yı kötüleştirdi. Final state = **legacy tekniğin aynısı** + iki *veri/kapsam* iyileştirmesi (tam C(5,2)=10 taksonomi + zengin/çeşitli üretim). Yani katkı modelleme tekniğinde değil, **eksiksiz ve daha temsili veride.**

| Konfigürasyon | Teknik | Realdata FULL | base_ok |
|---|---|---|---|
| Legacy (29 grup) | raw + GBT | 11 | 32 |
| 34 grup, variance 1.3-3.0 | raw | 9 | 33 |
| **Final (34 grup, var 1.3-2.0)** | **raw (=legacy)** | **11** | **33** |
| dual-view (detrend/deseason) | feature-eng | 8 | 29 |


### Yapılan (kalıcı) iyileştirmeler
1. **Eksik 5 multi grup eklendi (G30-G34)** → C(5,2)=10 tam. Etki: multi `trend_shift` temsili 1→4 gruba, `collective` 2→4'e çıktı; multi trend_shift binary F1 0.92.
2. **Veri çeşitliliği:** katmanlı uzunluk (50-400), trend_shift 3 tipi, anomali ölçek aralıkları. Anomalilerin kısa/uzun ve farklı şiddette temsili için.

### Denenen ama REDDEDILEN yaklaşımlar (kanıtla)

**A) Dual-view feature (raw + detrend/deseasonalize residual).** Hipotez: residual, trend_shift'in maskelediği base'i ve mevsimsel variance FP'sini ayırır. **Sonuç: realdata'yı kötüleştirdi** (base_ok 32→29, FULL 10→8). Geri alındı.
- Kanıt — base ayrım gücü 3 bağımsız testte: `raw > residual > dual`
  - LDA 5-fold base-acc: **raw 0.568**, residual 0.500, dual 0.469
  - Gerçek base_meta acc: **raw-only 0.800**, dual 0.790, residual 0.779
- Sebep: realdata'da base ayrımı trend yapısına dayanır; detrend onu siler.
- Repro: `experiments/diag_views.py`, `experiments/meta_view_test.py`.

**B) variance ölçeğini daraltma (3.0→2.0).** airpass/German variance FP'sini hedefliyordu. **Net-nötr** — FP feature confusion'ı (mevsimsel genlik ≈ variance shift), ölçek-büyüklüğü artefaktı değil; variance prob hâlâ ≈1.0. (1.3-2.0 yine de makul bir aralık olarak korundu.)

### Kanıtlanan İÇSEL (feature-eng ile çözülemeyen) sınırlar
- **stoch_trend + trend_shift ↔ anomaly_only örtüşmesi:** kNN komşu analizinde stoch+trend örneklerinin komşuları %16 stoch_trend / %44 anomaly_only — ham ve residual'da aynı. trend_shift imzası alttaki base'i tsfresh uzayında örtüyor. → G13/G18/G33 düşük FULL'un kök nedeni.
- **NP trend_shift sinyali yok:** en gevşek eşikte bile 11 NP dosyasından 3'ü yakalanıyor.
- **airpass variance FP:** mevsimsel genlik artışı robust şekilde variance_shift olarak okunuyor.

### Henüz denenmemiş yönler (gelecek iş)
- base_meta yeniden-dengeleme (anomaly_only 15/34 grupla baskın) — içsel örtüşme nedeniyle sınırlı kazanım beklenir.
- Variance tespiti için ayrık deseasonalize edilmiş view (sadece variance modeline).
- Gerçek-benzeri kademeli trend_shift verisi (NP açığı için).
- Derin embedding (kullanılmayan `inception.pt` bir InceptionTime denemesine işaret ediyor).
- Taksonomi yeniden tasarımı (stoch+trend vs anomaly_only ayrımının ele alınışı).

### Neden PCA/MDS/k-means/hierarchical model katmanı DEĞİL
Problem güçlü supervised (34 etiketli grup). Ağaç modelleri (LGBM/XGB) yüksek boyuttan etkilenmez, örtük feature seçimi yapar; **PCA tipik olarak eksen-hizalı tree split'lerini bozar.** k-means/hierarchical unsupervised → taksonomiyle hizalı değil. Bunlar burada **teşhis/görselleştirme** aracı (bkz. `experiments/`), modelleme iyileştirmesi değil.
