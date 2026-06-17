"""
29 grup x 1000 = 29000 seri uret.
Multi-anomali destegi (G25-G29).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # ceylanhoca/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betise.core.generator import TimeSeriesGenerator
from config import (
    GROUPS, N_PER_GROUP, LEN_RANGE, DATA_GEN_DIR,
)

SEED_BASE = 7000  # newplanlast seed


def gen_series(length, spec, seed):
    """spec'ten bir seri uret. Multi-anomali destekler."""
    random.seed(seed)
    np.random.seed(seed)
    ts = TimeSeriesGenerator(length=length)

    # 1. Base
    base = spec.get("base", "stationary")
    if base == "stationary":
        df, _ = ts.generate_stationary_base_series(distribution="ar")
    elif base == "stochastic":
        kinds = ["rw", "rwd", "ari", "ima", "arima"]
        df, _ = ts.generate_stochastic_trend(kind=random.choice(kinds))
    elif base == "volatility":
        kinds = ["arch", "garch", "egarch", "aparch"]
        df, _ = ts.generate_volatility(kind=random.choice(kinds))

    # 2. Trend overlay (mixed_det: 5 trend tip karisik)
    if "trend" in spec:
        kind = spec["trend"]
        sign = spec.get("trend_sign", 1)
        if kind == "mixed_det":
            # 5 trend tip arasinda secim
            trend_kinds = ["linear", "quadratic", "cubic", "exponential", "damped"]
            kind = random.choice(trend_kinds)
        if kind == "linear":
            df, _ = ts.generate_deterministic_trend_linear(df, sign=sign)
        elif kind == "quadratic":
            df, _ = ts.generate_deterministic_trend_quadratic(df, sign=sign, location="center")
        elif kind == "cubic":
            df, _ = ts.generate_deterministic_trend_cubic(df, sign=sign, amplitude=10.0, location="center")
        elif kind == "exponential":
            df, _ = ts.generate_deterministic_trend_exponential(df, sign=sign)
        elif kind == "damped":
            df, _ = ts.generate_deterministic_trend_exponential(df, sign=-1)  # damped = exp neg

    # 3. Break (mean_shift / variance_shift / trend_shift)
    breaks = []
    if "break" in spec:
        breaks.append(spec["break"])
    if "break_two" in spec:
        breaks.extend(spec["break_two"])
    for b_kind in breaks:
        sign = random.choice([-1, 1])
        if b_kind == "mean_shift":
            df, _ = ts.generate_mean_shift(df, signs=[sign], location="middle", num_breaks=1, scale_factor=1.5)
        elif b_kind == "variance_shift":
            df, _ = ts.generate_variance_shift(df, signs=[sign], location="middle", num_breaks=1, scale_factor=1.5)
        elif b_kind == "trend_shift":
            df, _ = ts.generate_trend_shift(df, sign=sign, location="middle", num_breaks=1,
                                              change_types=["direction_change"], scale_factor=1.0)

    # 4. Anomaly (point / collective)
    anomalies = []
    if "anomaly" in spec:
        anomalies.append(spec["anomaly"])
    if "anomaly_two" in spec:
        anomalies.extend(spec["anomaly_two"])
    for a_kind in anomalies:
        if a_kind == "point":
            df, _ = ts.generate_point_anomaly(df, location="middle", scale_factor=2.0, is_spike=True)
        elif a_kind == "collective":
            df, _ = ts.generate_collective_anomalies(df, num_anomalies=1, location="middle", scale_factor=2.0)

    return df["data"].values


def main():
    random.seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    manifest = []
    fail_total = 0
    print(f"Toplam grup: {len(GROUPS)}, hedef her grup: {N_PER_GROUP}")

    for gid, name, subpath, spec, label in GROUPS:
        out_dir = DATA_GEN_DIR / subpath
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        fail = 0
        for i in range(N_PER_GROUP):
            length = random.randint(LEN_RANGE[0], LEN_RANGE[1])
            seed = SEED_BASE + gid * 100000 + i
            try:
                s = gen_series(length, spec, seed)
                if len(s) < 50:
                    fail += 1
                    continue
                fname = f"{name}_{i:04d}.csv"
                pd.DataFrame({"value": s}).to_csv(out_dir / fname, index=False)
                manifest.append({
                    "gid": gid, "name": name, "subpath": subpath,
                    "file": fname, "length": int(len(s)),
                    "spec": spec, "label": list(label),
                })
                ok += 1
            except Exception:
                fail += 1
        fail_total += fail
        print(f"  Grup {gid:2d} ({name:<30}) {ok}/{N_PER_GROUP}{'  HATA: '+str(fail) if fail else ''}")

    with open(DATA_GEN_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nToplam: {len(manifest)} seri, {fail_total} hata")
    print(f"Manifest: {DATA_GEN_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
