"""
Stationarity Gate egitimi.
Pozitif: G1 (stationary) — 1000 seri
Negatif: G2-G29'dan balanced (28 grup'tan esit pay) — 1000 seri
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GROUP_NAMES, GATE_POSITIVE, ALL_GIDS, DATA_GEN_DIR,
    GATE_MODELS_DIR, N_SAMPLES_PER_MODEL, RESULTS_DIR,
)
from lib import sample_balanced, load_series_batch, extract_batch, train_binary, save_model


def main():
    print("="*60)
    print("STATIONARITY GATE EGITIMI")
    print("="*60)

    pos_groups = GATE_POSITIVE  # {1}
    neg_groups = ALL_GIDS - pos_groups  # G2-G29

    print(f"Pozitif gruplar: {sorted(pos_groups)}")
    print(f"Negatif gruplar: {len(neg_groups)} grup (G2-G29)")

    csvs, labels = sample_balanced(
        pos_groups, neg_groups, GROUP_NAMES, DATA_GEN_DIR,
        n_per_side=N_SAMPLES_PER_MODEL, seed=42,
    )
    print(f"Toplam sample: {len(csvs)} (pos={sum(labels)}, neg={len(labels)-sum(labels)})")

    print("\nSeriler okunuyor...")
    series, keep = load_series_batch(csvs)
    y = np.array([labels[i] for i in keep], dtype=int)
    print(f"Basarili: {len(series)}, atlanan: {len(csvs)-len(series)}")

    print("\ntsfresh extraction...")
    X = extract_batch(series, n_jobs=4)
    print(f"Feature matrix: {X.shape}")

    result = train_binary(X, y, "stationarity_gate")
    save_model(GATE_MODELS_DIR / "stationarity_gate.pkl", result)

    summary = {k: v for k, v in result.items() if k not in {"model", "scaler"}}
    with open(RESULTS_DIR / "gate_training.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOzet: {RESULTS_DIR / 'gate_training.json'}")


if __name__ == "__main__":
    main()
