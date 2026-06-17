"""
newplanlast — Konfigurasyon
29 grup veri taksonomisi + 14 binary base model + meta-learner.
HERSEY SIFIRDAN EGITILECEK — eski model KULLANILMIYOR.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---- Dizinler (hepsi REPO icinde, ts-proje bagimsiz) ----
DATA_GEN_DIR = BASE_DIR / "data" / "generated"
DATA_REAL_DIR = BASE_DIR / "data" / "realdata"

GATE_MODELS_DIR = BASE_DIR / "models" / "gate"
BASE_MODELS_DIR = BASE_DIR / "models" / "base"
SINGLE_ANOM_MODELS_DIR = BASE_DIR / "models" / "single_anom"
MULTI_ANOM_MODELS_DIR = BASE_DIR / "models" / "multi_anom"
META_MODELS_DIR = BASE_DIR / "meta" / "meta_models"
PROCESSED_DIR = BASE_DIR / "meta" / "processed_data"
RESULTS_DIR = BASE_DIR / "results"

for d in [GATE_MODELS_DIR, BASE_MODELS_DIR, SINGLE_ANOM_MODELS_DIR,
          MULTI_ANOM_MODELS_DIR, META_MODELS_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Genel ayarlar ----
N_PER_GROUP = 1000
MIN_SERIES_LENGTH = 50
LEN_RANGE = (80, 150)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Sample size her binary model icin (balanced)
N_SAMPLES_PER_MODEL = 1000  # pos + 1000 neg

# ---- Sinif listeleri ----
BASE_LABELS = ["stationary", "deterministic_trend", "stochastic_trend",
                "volatility", "anomaly_only"]   # 5-class
ANOM_LABELS = ["collective_anomaly", "mean_shift", "point_anomaly",
                "trend_shift", "variance_shift"]  # 5 anom

BASE_MODEL_NAMES = ["deterministic_trend", "stochastic_trend", "volatility"]  # 3 binary

# ---- 29 grup taksonomisi ----
# Format: (gid, name, subpath, spec, label)
# label = (base, [anomalies])
GROUPS = [
    # === SAF (4) ===
    (1, "G01_stationary", "G01_stationary",
        {"base": "stationary"},
        ("stationary", [])),
    (2, "G02_det_trend", "G02_det_trend",
        {"base": "stationary", "trend": "mixed_det"},  # 5 trend tip karisik
        ("deterministic_trend", [])),
    (3, "G03_stoch_trend", "G03_stoch_trend",
        {"base": "stochastic"},
        ("stochastic_trend", [])),
    (4, "G04_volatility", "G04_volatility",
        {"base": "volatility"},
        ("volatility", [])),

    # === SAF ANOMALI (5) — base=anomaly_only ===
    (5, "G05_anom_only_collective", "G05_anom_only_collective",
        {"base": "stationary", "anomaly": "collective"},
        ("anomaly_only", ["collective_anomaly"])),
    (6, "G06_anom_only_mean", "G06_anom_only_mean",
        {"base": "stationary", "break": "mean_shift"},
        ("anomaly_only", ["mean_shift"])),
    (7, "G07_anom_only_point", "G07_anom_only_point",
        {"base": "stationary", "anomaly": "point"},
        ("anomaly_only", ["point_anomaly"])),
    (8, "G08_anom_only_trend_shift", "G08_anom_only_trend_shift",
        {"base": "stationary", "break": "trend_shift"},
        ("anomaly_only", ["trend_shift"])),
    (9, "G09_anom_only_variance", "G09_anom_only_variance",
        {"base": "stationary", "break": "variance_shift"},
        ("anomaly_only", ["variance_shift"])),

    # === DET_TREND + 5 ANOMALI (5) ===
    (10, "G10_det_collective", "G10_det_collective",
        {"base": "stationary", "trend": "mixed_det", "anomaly": "collective"},
        ("deterministic_trend", ["collective_anomaly"])),
    (11, "G11_det_mean", "G11_det_mean",
        {"base": "stationary", "trend": "mixed_det", "break": "mean_shift"},
        ("deterministic_trend", ["mean_shift"])),
    (12, "G12_det_point", "G12_det_point",
        {"base": "stationary", "trend": "mixed_det", "anomaly": "point"},
        ("deterministic_trend", ["point_anomaly"])),
    (13, "G13_det_trend_shift", "G13_det_trend_shift",
        {"base": "stationary", "trend": "mixed_det", "break": "trend_shift"},
        ("deterministic_trend", ["trend_shift"])),
    (14, "G14_det_variance", "G14_det_variance",
        {"base": "stationary", "trend": "mixed_det", "break": "variance_shift"},
        ("deterministic_trend", ["variance_shift"])),

    # === STOCH_TREND + 5 ANOMALI (5) ===
    (15, "G15_stoch_collective", "G15_stoch_collective",
        {"base": "stochastic", "anomaly": "collective"},
        ("stochastic_trend", ["collective_anomaly"])),
    (16, "G16_stoch_mean", "G16_stoch_mean",
        {"base": "stochastic", "break": "mean_shift"},
        ("stochastic_trend", ["mean_shift"])),
    (17, "G17_stoch_point", "G17_stoch_point",
        {"base": "stochastic", "anomaly": "point"},
        ("stochastic_trend", ["point_anomaly"])),
    (18, "G18_stoch_trend_shift", "G18_stoch_trend_shift",
        {"base": "stochastic", "break": "trend_shift"},
        ("stochastic_trend", ["trend_shift"])),
    (19, "G19_stoch_variance", "G19_stoch_variance",
        {"base": "stochastic", "break": "variance_shift"},
        ("stochastic_trend", ["variance_shift"])),

    # === VOLATILITY + 5 ANOMALI (5) ===
    (20, "G20_vol_collective", "G20_vol_collective",
        {"base": "volatility", "anomaly": "collective"},
        ("volatility", ["collective_anomaly"])),
    (21, "G21_vol_mean", "G21_vol_mean",
        {"base": "volatility", "break": "mean_shift"},
        ("volatility", ["mean_shift"])),
    (22, "G22_vol_point", "G22_vol_point",
        {"base": "volatility", "anomaly": "point"},
        ("volatility", ["point_anomaly"])),
    (23, "G23_vol_trend_shift", "G23_vol_trend_shift",
        {"base": "volatility", "break": "trend_shift"},
        ("volatility", ["trend_shift"])),
    (24, "G24_vol_variance", "G24_vol_variance",
        {"base": "volatility", "break": "variance_shift"},
        ("volatility", ["variance_shift"])),

    # === MULTI-ANOMALI (5) — saf, base=anomaly_only ===
    (25, "G25_multi_col_mean", "G25_multi_col_mean",
        {"base": "stationary", "anomaly": "collective", "break": "mean_shift"},
        ("anomaly_only", ["collective_anomaly", "mean_shift"])),
    (26, "G26_multi_col_point", "G26_multi_col_point",
        {"base": "stationary", "anomaly_two": ["collective", "point"]},
        ("anomaly_only", ["collective_anomaly", "point_anomaly"])),
    (27, "G27_multi_mean_variance", "G27_multi_mean_variance",
        {"base": "stationary", "break_two": ["mean_shift", "variance_shift"]},
        ("anomaly_only", ["mean_shift", "variance_shift"])),
    (28, "G28_multi_point_variance", "G28_multi_point_variance",
        {"base": "stationary", "anomaly": "point", "break": "variance_shift"},
        ("anomaly_only", ["point_anomaly", "variance_shift"])),
    (29, "G29_multi_point_trend", "G29_multi_point_trend",
        {"base": "stationary", "anomaly": "point", "break": "trend_shift"},
        ("anomaly_only", ["point_anomaly", "trend_shift"])),
]

# Hizli erisim
GROUP_LABELS = {gid: lbl for gid, _, _, _, lbl in GROUPS}
GROUP_NAMES = {gid: name for gid, name, _, _, _ in GROUPS}
ALL_GIDS = set(GROUP_LABELS.keys())

# ---- Pozitif gruplar her model icin ----
# Stationarity gate
GATE_POSITIVE = {1}  # sadece G1 stationary

# Base binary models (pozitif gruplar)
BASE_POSITIVE_GROUPS = {
    "deterministic_trend": {2} | set(range(10, 15)),     # G2 + G10-14
    "stochastic_trend":    {3} | set(range(15, 20)),     # G3 + G15-19
    "volatility":          {4} | set(range(20, 25)),     # G4 + G20-24
}

# Single-anom binary models (pozitif = o anomaliyi iceren TEK-anom gruplari)
# Tek anom: saf-anom (G5-9) + base+anom (G10-24)
SINGLE_ANOM_POSITIVE_GROUPS = {
    # collective: G5 (saf), G10 (det), G15 (stoch), G20 (vol)
    "collective_anomaly": {5, 10, 15, 20},
    "mean_shift":         {6, 11, 16, 21},
    "point_anomaly":      {7, 12, 17, 22},
    "trend_shift":        {8, 13, 18, 23},
    "variance_shift":     {9, 14, 19, 24},
}

# Multi-anom binary models (pozitif = MULTI-anom datasinda o anomalinin gectigi)
# G25: col+mean, G26: col+point, G27: mean+var, G28: pt+var, G29: pt+trend
MULTI_ANOM_POSITIVE_GROUPS = {
    "collective_anomaly": {25, 26},
    "mean_shift":         {25, 27},
    "point_anomaly":      {26, 28, 29},
    "trend_shift":        {29},
    "variance_shift":     {27, 28},
}

# Multi gruplari (tum 5)
MULTI_GROUPS = {25, 26, 27, 28, 29}

# ---- Decision thresholds (sweep ile bulunacak, defaultlar) ----
DEFAULT_THR_STAT_HARD = 0.80
DEFAULT_THR_ROUTER = 0.40
DEFAULT_ALPHA = {a: 0.5 for a in ANOM_LABELS}  # single vs multi blend
DEFAULT_THR_ANOM_ONLY = {a: 0.5 for a in ANOM_LABELS}  # anom_only base
DEFAULT_THR_COMBO = {a: 0.55 for a in ANOM_LABELS}  # det/stoch/vol + anom

# ---- Realdata ground truth (TRICK: PDF'e uyumlu, "anomaly_only" KALDI) ----
# Trick: model "anomaly_only" derse "stationary" diye yorumlanir (post-process)
# GT'leri PDF'e uyarladik: stat+anom = stationary base + anom listesi
REAL_GT = {
    "W1.csv": ("stationary", ["point_anomaly"]),     # Wei: stat + AO/IO outliers
    "W2.csv": ("stationary", ["variance_shift"]),    # stat in mean, var shift
    "W3.csv": ("stationary", ["variance_shift"]),    # stat in mean, var shift
    "W5.csv": ("deterministic_trend", ["mean_shift"]),
    "W6.csv": ("deterministic_trend", ["variance_shift"]),
    "W10.csv": ("stochastic_trend", ["point_anomaly"]),
    "uspop.csv": ("deterministic_trend", []),
    "strikes.csv": ("stationary", []),
    "sunspots.csv": ("stationary", []),
    "airpass.csv": ("stochastic_trend", []),
    "deaths.csv": ("stochastic_trend", []),
    "INDPRO.csv": ("stochastic_trend", []),
    "UNRATE.csv": ("stochastic_trend", []),
    "soi_dataframe.csv": ("stationary", []),
    "rec_dataframe.csv": ("stationary", []),
    "GermanGNP.csv": ("deterministic_trend", ["trend_shift"]),
    "US_investment.csv": ("stationary", []),
    "German_consumption.csv": ("stochastic_trend", []),
    "Polish_productivity.csv": ("stochastic_trend", ["trend_shift"]),
    "RealInt_dataframe.csv": ("stationary", ["mean_shift"]),  # Bai-Perron break
    "NP_xetradax_returns100.csv": ("stationary", []),
    # === TCPD datasets (yeni indirildi) ===
    "tcpd_nile.csv": ("stationary", ["mean_shift"]),         # Aswan Dam 1898
    "tcpd_seatbelts.csv": ("stochastic_trend", ["mean_shift"]),  # UK law 1983
    "tcpd_lga_passengers.csv": ("stochastic_trend", ["mean_shift"]),  # 9/11
    "tcpd_debt_ireland.csv": ("deterministic_trend", ["trend_shift"]),  # 2008 crisis
    "tcpd_ozone.csv": ("stochastic_trend", ["trend_shift"]),     # Montreal Protocol
    "tcpd_robocalls.csv": ("stochastic_trend", ["mean_shift"]),  # FCC 2018
    # === Nelson-Plosser 14 series (Z&A 1992) ===
    "np_cpi.csv": ("stochastic_trend", ["trend_shift"]),     # Consumer prices, break 1873
    "np_employmt.csv": ("stochastic_trend", ["trend_shift"]),  # 1929
    "np_gnpdefl.csv": ("stochastic_trend", ["trend_shift"]),  # 1929
    "np_nomgnp.csv": ("stochastic_trend", ["trend_shift"]),   # 1929
    "np_interest.csv": ("stochastic_trend", []),             # yok rejection
    "np_indprod.csv": ("stochastic_trend", ["trend_shift"]),  # 1929
    "np_gnpperca.csv": ("stochastic_trend", ["trend_shift"]),
    "np_realgnp.csv": ("stochastic_trend", ["trend_shift"]),
    "np_wages.csv": ("stochastic_trend", ["trend_shift"]),
    "np_realwag.csv": ("stochastic_trend", ["trend_shift"]),  # 1940
    "np_sp500.csv": ("stochastic_trend", ["trend_shift"]),    # 1936
    "np_unemploy.csv": ("stationary", []),                   # debated, mostly stat
    "np_velocity.csv": ("stochastic_trend", []),             # 1949 borderline
    "np_M.csv": ("stochastic_trend", ["trend_shift"]),       # 1929
}
