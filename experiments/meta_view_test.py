"""Cached base_meta deneyi: raw-only vs dual tsfresh kolonlari ile base accuracy.
Extraction YOK. meta_X cached kolonlarini keser."""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_LABELS, RANDOM_STATE, TEST_SIZE
import xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

PD = Path(__file__).resolve().parent.parent / "meta" / "processed_data"
X = np.load(PD / "meta_X.npy"); yb = np.load(PD / "meta_y_base.npy")
SI = BASE_LABELS.index("stochastic_trend"); VI = BASE_LABELS.index("volatility")

VIEWS = {
    "dual (mevcut)": np.arange(X.shape[1]),
    "raw-only":      np.r_[0:805],            # probs+derived+raw
    "residual-only": np.r_[0:28, 805:1582],   # probs+derived+residual
}
for name, cols in VIEWS.items():
    M = X[:, cols]
    Xtr, Xte, ytr, yte = train_test_split(M, yb, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yb)
    cl, cnt = np.unique(ytr, return_counts=True); tot=len(ytr)
    w={int(c):tot/(len(cl)*n) for c,n in zip(cl,cnt)}; sw=np.array([w[int(y)] for y in ytr])
    xb=xgb.XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=6,min_child_weight=3,gamma=0.1,
        subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,num_class=5,
        objective="multi:softprob",random_state=RANDOM_STATE,n_jobs=-1,verbosity=0)
    xb.fit(Xtr,ytr,sample_weight=sw)
    lb=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=7,num_leaves=63,
        subsample=0.8,colsample_bytree=0.7,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
    lb.fit(Xtr,ytr)
    pred=np.argmax(0.5*xb.predict_proba(Xte)+0.5*lb.predict_proba(Xte),axis=1)
    acc=accuracy_score(yte,pred); f1=f1_score(yte,pred,average="weighted")
    rec_s=recall_score(yte,pred,labels=[SI],average="macro",zero_division=0)
    rec_v=recall_score(yte,pred,labels=[VI],average="macro",zero_division=0)
    print(f"[{name:<14}] base_acc={acc:.4f} F1={f1:.4f} | stoch_recall={rec_s:.3f} vol_recall={rec_v:.3f}")
