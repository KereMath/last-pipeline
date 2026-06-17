"""
5 single-anomaly binary modeller.
Pozitif: tek-anomali datasi (saf anom + base+anom).
Negatif: diger her sey (multi-anom HARIC).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GROUP_NAMES, SINGLE_ANOM_POSITIVE_GROUPS, ALL_GIDS, MULTI_GROUPS,
    DATA_GEN_DIR, SINGLE_ANOM_MODELS_DIR, N_SAMPLES_PER_MODEL, RESULTS_DIR,
)
from lib import sample_balanced, load_series_batch, extract_batch, train_binary, save_model


def main():
    results = {}
    for anom_name, pos_groups in SINGLE_ANOM_POSITIVE_GROUPS.items():
        print("\n" + "="*60)
        print(f"SINGLE-ANOM: {anom_name}")
        print("="*60)
        # Negatif = ALL_GIDS - pos - multi (multi excluded per plan)
        neg_groups = ALL_GIDS - pos_groups - MULTI_GROUPS
        print(f"Pos (tek-anom): {sorted(pos_groups)}")
        print(f"Neg ({len(neg_groups)} grup, multi haric)")

        csvs, labels = sample_balanced(
            pos_groups, neg_groups, GROUP_NAMES, DATA_GEN_DIR,
            n_per_side=N_SAMPLES_PER_MODEL, seed=100 + hash(anom_name) % 1000,
        )
        series, keep = load_series_batch(csvs)
        y = np.array([labels[i] for i in keep], dtype=int)
        X = extract_batch(series, n_jobs=4)

        result = train_binary(X, y, f"single_{anom_name}")
        save_model(SINGLE_ANOM_MODELS_DIR / f"{anom_name}.pkl", result)
        results[anom_name] = {k: v for k, v in result.items() if k not in {"model", "scaler"}}

    with open(RESULTS_DIR / "single_anom_training.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
