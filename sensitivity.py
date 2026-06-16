import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score

# ---- folder where injected_*.csv live (CHANGE if needed) ----
FOLDER = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'

DEVICES = ['smart_plug','thermostat','motion_sensor']

def load(d):
    df = pd.read_csv(os.path.join(FOLDER, f'injected_{d}.csv'))
    return df['power_injected'].values, df['is_fault'].values

# ---- Sweep 1: LOF n_neighbors (contamination fixed at 0.027) ----
print("=== LOF n_neighbors sweep (contamination=0.027) ===")
neighbors = [5,10,15,20,25,30,35,40,50,60,75]
for d in DEVICES:
    x,y = load(d)
    xs = StandardScaler().fit_transform(x.reshape(-1,1))
    f1s = [f1_score(y,(LocalOutlierFactor(n_neighbors=k,contamination=0.027).fit_predict(xs)==-1).astype(int),zero_division=0) for k in neighbors]
    print(f"  {d:14s}", [f"{v:.2f}" for v in f1s])
print("  n_neighbors:", neighbors)

# ---- Sweep 2: contamination ----
print("\n=== contamination sweep (Isolation Forest) ===")
contams = [0.010,0.015,0.020,0.025,0.027,0.030,0.035,0.040,0.050]
for d in DEVICES:
    x,y = load(d)
    xs = StandardScaler().fit_transform(x.reshape(-1,1))
    f1s = [f1_score(y,(IsolationForest(contamination=c,random_state=42).fit_predict(xs)==-1).astype(int),zero_division=0) for c in contams]
    print(f"  {d:14s}", [f"{v:.2f}" for v in f1s])
print("  contamination:", contams)
print("\nDone.")