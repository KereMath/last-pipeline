# newplanlast — Yeni Pipeline Mimarisi (PLAN)

> **Amac:** Mevcut pipeline'lardaki (contextual-off, contextual-off-10x) tum karmasiklik
> ve OOD bias kaynaklarini temizleyip sifirdan, daha temiz bir mimari kurmak.

---

## ⚠️ KRİTİK KURAL: HERŞEY SIFIRDAN EĞİTİLECEK ⚠️

**Hiçbir pre-trained model KULLANILMAYACAK.** Aşağıdaki repolardaki TÜM modeller görmezden gelinecek:

- ❌ `tsfresh-ensemble-stationary/trained_models/` (9 detector) — KULLANILMAYACAK
- ❌ `stationary-detection-ml/models/` (21-feature GradientBoosting) — KULLANILMAYACAK
- ❌ `ensemble-alldata/trained_models/` (10 binary, eski) — KULLANILMAYACAK
- ❌ `contextual-off/ensemble-alldata/trained_models/` — KULLANILMAYACAK
- ❌ `contextual-off-10x/ensemble-alldata/trained_models/` — KULLANILMAYACAK
- ❌ `contextual-off-10x/ens-final/meta_models/` — KULLANILMAYACAK

**Sıfırdan eğitilecek modellerin TAM listesi (14 binary + meta):**

| # | Model | Tip | Eğitim datası |
|---|---|---|---|
| 1 | **Stationarity Gate** | Binary | Pos: G1 (saf stat) / Neg: balanced 28 grup |
| 2 | **Base: deterministic_trend** | Binary | Pos: G2 + G10-14 / Neg: diğerleri balanced |
| 3 | **Base: stochastic_trend** | Binary | Pos: G3 + G15-19 / Neg: balanced |
| 4 | **Base: volatility** | Binary | Pos: G4 + G20-24 / Neg: balanced |
| 5 | **Single-anom: collective** | Binary | Pos: TEK-anomali datası (G5, G10, G15, G20) |
| 6 | **Single-anom: mean_shift** | Binary | Pos: TEK-anomali (G6, G11, G16, G21) |
| 7 | **Single-anom: point** | Binary | Pos: TEK-anomali (G7, G12, G17, G22) |
| 8 | **Single-anom: trend_shift** | Binary | Pos: TEK-anomali (G8, G13, G18, G23) |
| 9 | **Single-anom: variance** | Binary | Pos: TEK-anomali (G9, G14, G19, G24) |
| 10 | **Multi-anom: collective** | Binary | Pos: MULTI datası içinde collective (G25, G26) |
| 11 | **Multi-anom: mean_shift** | Binary | Pos: MULTI içinde mean (G25, G27) |
| 12 | **Multi-anom: point** | Binary | Pos: MULTI içinde point (G26, G28, G29) |
| 13 | **Multi-anom: trend_shift** | Binary | Pos: MULTI içinde trend (G29) |
| 14 | **Multi-anom: variance** | Binary | Pos: MULTI içinde variance (G27, G28) |
| M1 | **base_meta** (XGB+LGB) | 5-class | meta_X ⇒ {stat/det/stoch/vol/anom_only} |
| M2 | **single_anom_meta** (×5) | Binary | meta_X ⇒ P(anom_a) tek datalı |
| M3 | **multi_anom_meta** (×5) | Binary | meta_X ⇒ P(anom_a) multi datalı |
| M4 | **router** (XGB+LGB) | Binary | meta_X ⇒ P(single vs combo) |

**Toplam: 14 binary base model + 12 meta-model = 26 model — HEPSI SIFIRDAN.**

---

---

## 1. Sınıf Taksonomisi

### 1 stationary (HARD GATE)
- Tek başına çalışan binary detector
- Pozitif çıkış → cikis = `("stationary", [])`, ileri gitme

### 3 base (non-stationary base tipleri)
- `deterministic_trend` (linear/quadratic/cubic/exponential/damped trendler)
- `stochastic_trend` (rw/rwd/ari/ima/arima)
- `volatility` (arch/garch/egarch/aparch)

### 5 anomali (contextual YOK)
- `collective_anomaly`
- `mean_shift`
- `point_anomaly`
- `trend_shift`
- `variance_shift`

### Çıkış formatı
**`(base, anomalies)`** burada:
- `base ∈ {stationary, deterministic_trend, stochastic_trend, volatility, anomaly_only}` (5 değer)
- `anomalies ⊆ {collective, mean_shift, point, trend_shift, variance}` (0-5 değer)

### Sıkı kurallar
1. `base = "stationary"` ⇒ `anomalies = ∅` (zorla bos)
2. `base ≠ "stationary"` ⇒ `anomalies` herhangi alt-küme olabilir
3. "Stationary + anomali" KOMBINASYONU YOK. Tek anomali olan seriler `base="anomaly_only"` olarak sınıflandırılır

---

## 2. Veri Üretim Planı (newplanlast/data/generated/)

Toplam **20 veri grubu**, her biri **1000 seri** = **20,000 seri**.
Uzunluk [80, 150] (eskiyle aynı).

### Grup 1: Saf stationary (1000 seri)
- Base: AR(p), MA(q), ARMA(p,q) karışık
- Hiçbir overlay yok

### Grup 2-4: Saf base (3000 seri)
- 2: `deterministic_trend` — 5 trend tipi karışık (linear, cubic, quad, exp, damped)
- 3: `stochastic_trend` — 5 stoch tipi karışık (rw, rwd, ari, ima, arima)
- 4: `volatility` — 4 vol tipi karışık (arch, garch, egarch, aparch)

### Grup 5-9: Saf anomali (5 grup × 1000 = 5000 seri)
> Eski "stationary + tek anomali" gruplarına denk gelir. Yeni etiket: `(anomaly_only, [tek_anom])`.
- 5: `(anomaly_only, [collective])` — AR base + collective overlay
- 6: `(anomaly_only, [mean_shift])`
- 7: `(anomaly_only, [point])`
- 8: `(anomaly_only, [trend_shift])`
- 9: `(anomaly_only, [variance])`

### Grup 10-24: Base + tek anomali (15 grup × 1000 = 15000 seri)
> 3 base × 5 anomali = 15 kombinasyon
- 10-14: `det_trend + {collective, mean, point, trend_shift, variance}`
- 15-19: `stoch_trend + {collective, mean, point, trend_shift, variance}`
- 20-24: `volatility + {collective, mean, point, trend_shift, variance}`

### Grup 25-29: Multi-anomali (5 grup × 1000 = 5000 seri)
> 5 yaygın 2-anomali kombinasyonu. Base = anomaly_only (yani stat üstüne 2 anomali).
- 25: `(anomaly_only, [collective, mean_shift])`
- 26: `(anomaly_only, [collective, point])`
- 27: `(anomaly_only, [mean_shift, variance_shift])`
- 28: `(anomaly_only, [point, variance_shift])`
- 29: `(anomaly_only, [point, trend_shift])`

> Multi-anomali base+anom (det+col+mean gibi) ÜRETMEYECEĞIZ — fazla karmaşıklık. Pragmatik: saf multi-anomali yeterli sinyal verir.

**TOPLAM: 29 grup × 1000 = ~29,000 seri** (1 stat + 3 saf base + 5 saf anom + 15 base+anom + 5 multi-anom)

---

## 3. Model Eğitim Planı

Toplam **14 binary model** + 1 meta-learner.

### Model 1: Stationarity Gate (1 binary)
**Hedef:** stationary mi non-stationary mi?

- **Pozitif (label=1):** Grup 1 (stationary) → 1000 seri
- **Negatif (label=0):** Tüm diğer 28 grup'tan eşit pay:
  - Her gruptan ~35-40 seri → ~1000 negatif toplam (balanced)
  - Veya 1000 seri = 28 grup × ~36 her birinden

> Sweep ile P(stat) threshold belirlenir (~0.5-0.85 arası)

### Model 2-4: Base Ensemble (3 binary, one-vs-rest)
**Hedef:** Hangi base type? (sadece non-stationary için anlamlı)

#### Model 2: `det_trend` binary
- **Pozitif:** Grup 2 (saf det) + Grup 10-14 (det+anom) + 0 (det içerikli multi yok) = ~6000 seri, sample **1000**
- **Negatif:** stoch + vol + onların kombo + saf-anom + multi-anom + stationary = sample **1000**

#### Model 3: `stoch_trend` binary
- **Pozitif:** Grup 3 + Grup 15-19 = ~6000, sample 1000
- **Negatif:** det/vol/anom_only/stat = 1000

#### Model 4: `volatility` binary
- **Pozitif:** Grup 4 + Grup 20-24 = ~6000, sample 1000
- **Negatif:** = 1000

### Model 5-9: Single-Anomaly Ensemble (5 binary)
**Hedef:** Bu seride X anomalisi var mı? (TEK-anomali datasıyla eğitilmiş)

#### Model 5: `collective` (single)
- **Pozitif:** Grup 5 (saf collective) + Grup 10/15/20 (base+collective) = ~4000, sample 1000
- **Negatif:** Diğer 4 anomali (saf + base+) + saf base + stationary + multi-anom = sample 1000

> **NOT:** Multi-anomali datası bu modele DAHIL DEĞIL (kullanıcı net dedi: "tek anomalisi olan datalarla eğitilmiş")

#### Model 6-9: aynı yapı (mean, point, trend_shift, variance)

### Model 10-14: Multi-Anomaly Ensemble (5 binary)
**Hedef:** Bu seri çoklu-anomali bağlamında X anomalisini içeriyor mu?

#### Model 10: `collective` (multi)
- **Pozitif:** Multi-anom gruplarında collective içeren = Grup 25 (col+mean) + Grup 26 (col+point) = ~2000 seri, sample 1000
- **Negatif:** Multi-anom'da collective içermeyen (G27, G28, G29) = ~3000, sample 1000

> Bu modeller multi-anomali serisinde hangi anomalilerin hep birlikte olabileceğini öğreniyor

#### Model 11-14: aynı yapı (mean, point, trend_shift, variance)

### Meta-Learner (Stacking) — DETAYLI

**Input meta-vektör (~810 boyut):**
- 14 binary olasılık (1 gate + 3 base + 5 single-anom + 5 multi-anom)
- 14-18 türetilmiş feature:
  - max/argmax (base probs), max/argmax (single-anom), max/argmax (multi-anom)
  - **single_vs_multi_agreement** (her 5 anom için: |single - multi| diff)
  - **single_vs_multi_disagreement_count** (kaç anom'da farkli karar)
  - confidence_gap (base 1st - 2nd)
  - anomaly_entropy
  - **gate_consistency** (P_gate yüksek ve base argmax stationary ise 1)
- 777 standardized tsfresh feature

**Output (4 meta-model):**

#### Meta-1: `base_meta` (5-class XGB+LGB)
- Sınıflar: stationary / det_trend / stoch_trend / volatility / anomaly_only
- Sample weighting: balanced (class_weight='balanced')
- Eğitim: 29 grup × 80 örnek = ~2300 sample meta_X

#### Meta-2: `single_anom_meta` (5 XGB+LGB binary)
- Her anomali için meta-learner: o anomalinin gerçekten varlığını gösterir
- Eğitim datası: tüm 29 grup (her anomali içeren tüm gruplar pozitif)

#### Meta-3: `multi_anom_meta` (5 XGB+LGB binary)
- Sadece multi-anomali grupları için ek meta-learner
- Eğitim: 5 multi-anom grup + diğer gruplar negatif

#### Meta-4: `router` (XGB+LGB binary)
- `single` (saf base) vs `combo` (base+anom veya anom_only)
- Eğitim: G2-G4 (saf base) → 0, diğer her şey → 1
- Output: P(combo)

### BLEND PARAMETRELERI — 2 SEVIYE (önceki kullanıcı noktası)

#### Seviye 1: Single vs Multi Ensemble Blend (per-anomaly alpha)
```python
P_final[a] = α[a] * P_single_anom_meta[a] + (1 - α[a]) * P_multi_anom_meta[a]
```
- **α[a] = 1**: tamamen single ensemble'a güven
- **α[a] = 0**: tamamen multi ensemble'a güven
- **α[a] sweep ile bulunur**: per-anomaly 5 değer, grid [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Mantik: Eğer multi-anom grup deseni daha güvenilirse (örneğin point hep variance ile birlikte), alpha düşürülür.

#### Seviye 2: Per-anomali threshold (final karar)
```python
if P_final[a] >= threshold[a]:
    anomalies.add(a)
```
- threshold[a] grid: [0.30, 0.35, ..., 0.90]

### Router blend (alternatif kullanım)
Yine eski mimarideki gibi:
```python
p_router_final = β * P_router_meta + (1 - β) * P_router_direct_ensemble
```
Burada β şimdilik 1.0 (sabit) — sweep'e dahil etmeyelim, sadece meta-router kullanılsın.

### Inference Karar Mantığı (DETAYLI, alpha+router dahil)

```python
def predict(x):
    # ====== ASAMA 1: HARD STATIONARITY GATE ======
    P_stat = gate.predict_proba(x)
    if P_stat >= THR_STAT_HARD:                  # default ~0.85
        return ("stationary", [])
    
    # ====== ASAMA 2: META-VEKTÖR HAZIRLA ======
    # 14 binary + 14-18 türetilmis + 777 tsfresh
    raw_base   = [base_model[i](x) for i in 3]          # 3 binary
    raw_single = [single_anom[i](x) for i in 5]         # 5 binary
    raw_multi  = [multi_anom[i](x) for i in 5]          # 5 binary
    derived    = compute_derived(P_stat, raw_base, raw_single, raw_multi)
    meta_X     = concat(P_stat, raw_base, raw_single, raw_multi, derived, tsfresh)
    
    # ====== ASAMA 3: BASE META-LEARNER (5-CLASS) ======
    base_probs = base_meta.predict_proba(meta_X)        # 5-class softmax
    base = BASE_LABELS[argmax(base_probs)]
    # base ∈ {stationary, det_trend, stoch_trend, volatility, anomaly_only}
    
    if base == "stationary":
        # Meta-learner stationary derse (soft gate'le örtüşmüyor ama yine de güven)
        return ("stationary", [])
    
    # ====== ASAMA 4: ROUTER (single mi combo mu) ======
    p_combo = router_meta.predict_proba(meta_X)
    is_combo = (p_combo >= THR_ROUTER)              # default 0.40
    
    if not is_combo and base in {det_trend, stoch_trend, volatility}:
        # Single yolu: sadece base, anomali listesi bos
        return (base, [])
    
    # ====== ASAMA 5: ANOMALI BLEND (alpha) ======
    # Her anomali için single ve multi meta-learner çıktısını blend et
    anomalies = []
    for a in 5_anomalies:
        p_single = single_anom_meta[a].predict_proba(meta_X)
        p_multi  = multi_anom_meta[a].predict_proba(meta_X)
        
        # ALPHA BLEND (per-anomaly):
        p_blended = α[a] * p_single + (1 - α[a]) * p_multi
        
        # Context threshold (base-dependent)
        if base == "anomaly_only":
            t = THR[a]                     # standart threshold
        elif base in {det_trend, stoch_trend, volatility}:
            t = THR_combo[a]               # combo için ayrı threshold
        
        if p_blended >= t:
            anomalies.append(a)
    
    # ====== ASAMA 6: EDGE CASE ======
    if base == "anomaly_only" and not anomalies:
        # Fallback: en yüksek blended olasılığı seç
        anomalies = [argmax(P_blended)]
    
    return (base, anomalies)
```

### Sweep Edilen Parametreler (toplam ~18-20 parametre)
1. `THR_STAT_HARD` (1)              — gate threshold
2. `THR_ROUTER` (1)                  — router single/combo eşiği
3. `α[a]` × 5 anomali (5)           — single vs multi blend weight
4. `THR[a]` × 5 anomali (5)         — anomaly_only base'de threshold
5. `THR_combo[a]` × 5 anomali (5)   — det/stoch/vol base'de threshold

Toplam **17 hyperparametre**. Joint coordinate descent (3 iter) ile ~30-45 dk.

---

## 4. Threshold + Alpha Sweep Stratejisi (joint coordinate descent)

**17 hyperparametre joint optimize edilecek** — basitten karmaşığa SOFISTIKE pipeline:

### Sweep aşamaları (3 iterasyon coordinate descent)
**Init:** Trainer'ın öğrendiği değerler (eski mimari önceki sweeplerden)

**Her iter:**
1. Önce `THR_STAT_HARD` grid'i (10 değer): diğer hepsi sabit, 10 eval
2. Sonra `THR_ROUTER` grid'i (9 değer): 9 eval
3. Sonra **per-anomali joint (alpha, threshold) grid**:
   - α[a]: 6 değer × THR[a]: 13 değer × 2 base context (anomaly_only vs combo) = 156 kombo per anom
   - 5 anomali sıralı: 5 × 156 = 780 eval
4. Toplam per iter: ~800 eval × 3 iter = **2400 evaluation**
5. Her eval ~0.3sn (cache'li meta_X) → **~12 dakika sweep**

### Karar metriği
**Training set FULL count** maximize (29 grup × 20 örnek = 580 örnek)

### Alpha[a] grid
`[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (6 değer)

### Threshold grid (per-anomali)
- `THR[a]` (anomaly_only base'de): `[0.30, 0.35, ..., 0.90]` (13 değer)
- `THR_combo[a]` (det/stoch/vol base'de): aynı grid (13 değer)

### Sweep sonrası sonuç
- Final `(THR_STAT_HARD, THR_ROUTER, α[5], THR[5], THR_combo[5])` dict'i kaydedilir
- Sonra realdata + sentetik test bu config'le çalıştırılır

---

## 5. Klasör Yapısı

```
newplanlast/
├── PLAN.md                       (bu dosya)
├── README.md                     (kurulum sonrası dolacak)
├── config.py                     (29 grup, model isimleri, paths, sabit threshold'lar)
│
├── data/
│   ├── generated/                (29 grup × 1000 = ~29000 seri, GITIGNORED)
│   │   ├── G01_stationary/
│   │   ├── G02_det_trend/
│   │   ├── G03_stoch_trend/
│   │   ├── G04_volatility/
│   │   ├── G05_anom_only_collective/
│   │   ├── G06_anom_only_mean/
│   │   ├── G07_anom_only_point/
│   │   ├── G08_anom_only_trend_shift/
│   │   ├── G09_anom_only_variance/
│   │   ├── G10_det_collective/  ... G24_vol_variance
│   │   └── G25_multi_*/  (5 multi-anom grup)
│   ├── synthetic_test/           (sentetik test verisi, 500-1000 stoch_trend)
│   └── realdata/                 (ana ts-proje'den symlink, ortak)
│
├── models/
│   ├── gate/                     (1 .pkl: stationarity_gate)
│   ├── base/                     (3 .pkl: det_trend, stoch_trend, volatility)
│   ├── single_anom/              (5 .pkl)
│   └── multi_anom/               (5 .pkl)
│
├── meta/
│   ├── meta_models/              (base_meta + anom_meta + router + blend_weights)
│   └── processed_data/           (meta_X.npy, meta_y_base.npy, vs.) (GITIGNORED)
│
├── results/
│   ├── evaluation.json
│   ├── threshold_sweep.json
│   ├── realdata.json
│   └── synthetic.json
│
└── runner/
    ├── 10_generate_data.py       (29 grup veri üretimi)
    ├── 20_train_gate.py          (stationarity gate)
    ├── 30_train_base.py          (3 base binary)
    ├── 31_train_single_anom.py   (5 single-anom)
    ├── 32_train_multi_anom.py    (5 multi-anom)
    ├── 40_train_meta.py          (meta-learner stacking)
    ├── 50_eval_train.py          (29 grup eval)
    ├── 51_eval_realdata.py       (realdata)
    ├── 52_eval_synthetic.py      (sentetik test)
    └── 60_threshold_sweep.py     (joint sweep)
```

---

## 6. Eğitim Sırası (script execution order)

```bash
# 1. Veri üret (~5-10 dk)
python newplanlast/runner/10_generate_data.py     # 29000 seri

# 2. Stationarity gate eğit (~3-5 dk)
python newplanlast/runner/20_train_gate.py        # 1 binary model

# 3. Base ensemble eğit (~10-15 dk)
python newplanlast/runner/30_train_base.py        # 3 binary

# 4. Single-anomaly ensemble (~15-20 dk)
python newplanlast/runner/31_train_single_anom.py # 5 binary

# 5. Multi-anomaly ensemble (~10-15 dk)
python newplanlast/runner/32_train_multi_anom.py  # 5 binary

# 6. Meta-learner eğit (~10-15 dk)
python newplanlast/runner/40_train_meta.py        # base_meta + anom_meta + router

# 7. Eval (eğitim seti) (~5 dk)
python newplanlast/runner/50_eval_train.py

# 8. Threshold sweep (~20-30 dk)
python newplanlast/runner/60_threshold_sweep.py

# 9. realdata + sentetik test (~5-10 dk)
python newplanlast/runner/51_eval_realdata.py
python newplanlast/runner/52_eval_synthetic.py
```

**Toplam tahmini süre: 1.5 - 2 saat**

---

## 7. Bekleyen Sorular / Onaylar

Aşağıdaki kararları onayla:

### Soru 1: Multi-anomali grupları yeterli mi?
5 grup öneriyorum (toplam 5000 seri):
- collective + mean_shift
- collective + point
- mean_shift + variance_shift
- point + variance_shift
- point + trend_shift

**Alternatif:** Daha fazla kombi (örn C(5,2) = 10 grup, hepsi 2-anomali) veya 3-anomali grupları ekle.

### Soru 2: Multi-anomali base+anom da olsun mu?
Şu an saf multi-anom (anomaly_only base) öneriyorum. Alternatif: `det_trend + collective + mean_shift` gibi kombi de üretmek (ama 3 base × 10 kombi = 30 yeni grup, çok karmaşık).

### Soru 3: Eski 38000 veriyi tutalım mı?
contextual-off-10x'teki 38000 veriden tekrar üretebilirim ama yeni etiketlerle. Eski veri yeni tax'ına uyumsuz olduğu için **yeniden üretmek daha temiz**.

### Soru 4: Meta-learner kompleksitesi
Önerdiğim: base_meta (5-class XGB+LGB) + 5 single-anom meta + 5 multi-anom meta + router (XGB+LGB) = **12 meta-model**.
**Alternatif basitleştirme:** Single+multi anom binary'lerini birleştir (5 değil 10 değil 5 birleşik meta).

### Soru 5: Threshold sweep stratejisi
Joint coordinate descent (önerdiğim) vs tam Bayesian/Optuna optimization. Joint daha hızlı (~30dk), Bayesian daha optimal ama daha yavaş (1-2 saat).

### Soru 6: realdata için ground truth eşleme
- W1 (stationary + outliers) → eski mantıkta (stat, [point]); yeni mantıkta **(anomaly_only, [point])** ✓
- airpass (seasonal stoch_trend) → (stoch_trend, []) — seasonal hâlâ ayrı kategori değil

---

## 8. Tahmini Sayısal Sonuç Beklentileri

Önceki pipeline'larla karşılaştırma:

| Metric | ts-proje (1x) | contextual-off-10x | **newplanlast (beklenti)** |
|---|---|---|---|
| Toplam model | 9+10+9 (ensembles) + meta = ~28 | 8+9 + meta = ~17 | **14 binary + meta = ~14** |
| Eğitim setinde FULL | %88.21 | %84.21 | **>%90 hedef** |
| Realdata FULL | 3 | 0 | **>3 hedef** |
| Sentetik base acc | %72 | %79 | **>%85 hedef** |
| Bias kaynağı | eski ensemble contextual | yok ama az veri | **yok, dengeli veri** |

---

## ONAYLAYACAĞIM:
1. **Mimari B (5 base + 5 anom + multi-anom ensemble) doğru mu?**
2. **20 saf grup + 5 multi-anom grup = 29 grup, 1000 seri/grup yeterli mi?**
3. **Stationarity gate'in negatif tarafı tüm 28 grup'tan eşit pay (balanced) doğru mu?**
4. **Multi-anomali sadece saf (anomaly_only base) olsun mu?**
5. **Threshold 0'dan joint coordinate descent sweep ile?**
6. **Eski tsfresh-ensemble-stationary & stationary-detection-ml KULLANILMAYACAK doğrulamak.**

Hazır mıyım: **Evet.** Eski kod altyapısının %70'i (sample_group, extract_batch, predict_with_thresholds vs.) reuse edilecek. Yeni şey: veri etiketleri, model sayısı (14 binary), yeni decision logic, joint sweep.

Toplam tahmini iş: **1.5-2 saat çalışma süresi** (veri + 14 eğitim + meta + sweep + eval).

**Onay verdiğinde adım adım başlayacağım.**
