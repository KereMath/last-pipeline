"""
3 base binary model: deterministic_trend, stochastic_trend, volatility.
Her biri one-vs-rest balanced.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GROUP_NAMES, BASE_POSITIVE_GROUPS, ALL_GIDS,
    DATA_GEN_DIR, BASE_MODELS_DIR, N_SAMPLES_PER_MODEL, RESULTS_DIR,
)
from lib import sample_balanced, load_series_batch, extract_batch, train_binary, save_model


def main():
    results = {}
    for base_name, pos_groups in BASE_POSITIVE_GROUPS.items():
        print("\n" + "="*60)
        print(f"BASE BINARY: {base_name}")
        print("="*60)
        neg_groups = ALL_GIDS - pos_groups
        print(f"Pos: {sorted(pos_groups)} | Neg: {len(neg_groups)} grup")

        csvs, labels = sample_balanced(
            pos_groups, neg_groups, GROUP_NAMES, DATA_GEN_DIR,
            n_per_side=N_SAMPLES_PER_MODEL, seed=42 + hash(base_name) % 1000,
        )
        series, keep = load_series_batch(csvs)
        y = np.array([labels[i] for i in keep], dtype=int)
        X = extract_batch(series, n_jobs=4)

        result = train_binary(X, y, f"base_{base_name}")
        save_model(BASE_MODELS_DIR / f"{base_name}.pkl", result)
        results[base_name] = {k: v for k, v in result.items() if k not in {"model", "scaler"}}

    with open(RESULTS_DIR / "base_training.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOzet: {RESULTS_DIR / 'base_training.json'}")


if __name__ == "__main__":
    main()
