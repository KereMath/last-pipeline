"""
5 multi-anomaly binary modeller.
Pozitif: multi-anom datasinda (G25-G29) o anomalinin gectigi seriler.
Negatif: multi-anom dis kalanlar (+saf gruplari kullanabiliriz balanced icin).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GROUP_NAMES, MULTI_ANOM_POSITIVE_GROUPS, MULTI_GROUPS,
    DATA_GEN_DIR, MULTI_ANOM_MODELS_DIR, N_SAMPLES_PER_MODEL, RESULTS_DIR,
)
from lib import sample_balanced, load_series_batch, extract_batch, train_binary, save_model


def main():
    results = {}
    for anom_name, pos_groups in MULTI_ANOM_POSITIVE_GROUPS.items():
        print("\n" + "="*60)
        print(f"MULTI-ANOM: {anom_name}")
        print("="*60)
        # Negatif: multi-anom gruplari arasinda o anomaliyi icermeyenler
        # + diger saf gruplar (sinyali daha kuvvetlendirmek icin)
        neg_groups = MULTI_GROUPS - pos_groups  # multi'de o anom yok
        # Eger negatif cok az ise, saf gruplari ekle (anom_only + base)
        if len(neg_groups) < 2:
            # Sadece 1-2 grup, sample size yetersiz olabilir
            # ek olarak saf anom_only diger gruplari ekle (o anomali yok orada da)
            from config import SINGLE_ANOM_POSITIVE_GROUPS
            # Other single anom groups (o anom haric)
            for k, v in SINGLE_ANOM_POSITIVE_GROUPS.items():
                if k != anom_name:
                    neg_groups = neg_groups | v
        print(f"Pos (multi'de {anom_name}): {sorted(pos_groups)}")
        print(f"Neg ({len(neg_groups)} grup)")

        csvs, labels = sample_balanced(
            pos_groups, neg_groups, GROUP_NAMES, DATA_GEN_DIR,
            n_per_side=N_SAMPLES_PER_MODEL, seed=200 + hash(anom_name) % 1000,
        )
        series, keep = load_series_batch(csvs)
        y = np.array([labels[i] for i in keep], dtype=int)
        X = extract_batch(series, n_jobs=4)

        result = train_binary(X, y, f"multi_{anom_name}")
        save_model(MULTI_ANOM_MODELS_DIR / f"{anom_name}.pkl", result)
        results[anom_name] = {k: v for k, v in result.items() if k not in {"model", "scaler"}}

    with open(RESULTS_DIR / "multi_anom_training.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
