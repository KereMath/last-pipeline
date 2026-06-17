"""
Teshis: cached meta_X uzerinde ham vs residual view'in base ayrim gucu.
- PCA 2D gorseller (base'e gore renk)
- LDA 5-fold CV accuracy: ham / residual / dual (hangi view base'i ayiriyor)
- trend_shift'li stoch ornekleri anomaly_only ile ortusuyor mu (kNN saflik)
Extraction YOK — cache kullanir.
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_LABELS, ANOM_LABELS

PD = Path(__file__).resolve().parent.parent / "meta" / "processed_data"
OUT = Path(__file__).resolve().parent
X = np.load(PD / "meta_X.npy")
yb = np.load(PD / "meta_y_base.npy")
ya = np.load(PD / "meta_y_anom.npy")

# meta_X = [probs+derived(28) | raw_tsfresh(777) | residual_tsfresh(777)]
RAW = X[:, 28:28+777]
RES = X[:, 28+777:28+777+777]
DUAL = X[:, 28:]
print("RAW", RAW.shape, "RES", RES.shape, "DUAL", DUAL.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def lda_cv(M, y, name):
    M = np.nan_to_num(M)
    acc = cross_val_score(LinearDiscriminantAnalysis(), M, y, cv=5, scoring="accuracy").mean()
    print(f"  LDA 5-fold base-acc [{name:<8}] = {acc:.4f}")
    return acc

print("\n=== Base ayrim gucu (LDA CV) ===")
lda_cv(RAW, yb, "raw")
lda_cv(RES, yb, "residual")
lda_cv(DUAL, yb, "dual")

# trend_shift index
ti = ANOM_LABELS.index("trend_shift")
SI = BASE_LABELS.index("stochastic_trend")
AO = BASE_LABELS.index("anomaly_only")
# stoch + trend_shift ornekleri
mask_stoch_trend = (yb == SI) & (ya[:, ti] == 1)
print(f"\nstoch_trend + trend_shift ornek sayisi: {mask_stoch_trend.sum()}")

# kNN saflik: stoch+trend ornekleri en yakin komsular hangi base?
from sklearn.neighbors import NearestNeighbors
def knn_neighbor_base(M, name):
    M = StandardScaler().fit_transform(np.nan_to_num(M))
    nn = NearestNeighbors(n_neighbors=11).fit(M)
    idx = nn.kneighbors(M[mask_stoch_trend], return_distance=False)[:, 1:]  # self haric
    neigh_base = yb[idx]
    frac_stoch = (neigh_base == SI).mean()
    frac_ao = (neigh_base == AO).mean()
    print(f"  [{name:<8}] stoch+trend komsulari: %{100*frac_stoch:.0f} stoch_trend, %{100*frac_ao:.0f} anomaly_only")
print("\n=== stoch+trend confusion (kNN komsu base dagilimi) ===")
knn_neighbor_base(RAW, "raw")
knn_neighbor_base(RES, "residual")

# PCA gorseller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
colors = ["#888","#1f77b4","#d62728","#2ca02c","#9467bd"]
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, M, title in [(axes[0], RAW, "RAW tsfresh"), (axes[1], RES, "RESIDUAL tsfresh")]:
    P = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(np.nan_to_num(M)))
    for c in range(5):
        m = yb == c
        ax.scatter(P[m,0], P[m,1], s=6, alpha=0.4, c=colors[c], label=BASE_LABELS[c])
    # stoch+trend ornekleri ekstra isaretle
    ax.scatter(P[mask_stoch_trend,0], P[mask_stoch_trend,1], s=40, facecolors="none",
               edgecolors="black", linewidths=1.2, label="stoch+trend_shift")
    ax.set_title(f"PCA — {title} (base'e gore)"); ax.legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "pca_base_views.png", dpi=110)
print(f"\nKaydedildi: {OUT/'pca_base_views.png'}")
